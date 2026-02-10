"""Priority queue with dead-letter queue, retry semantics, and backpressure.

Queue invariants
~~~~~~~~~~~~~~~~
* Jobs are dequeued in strict ``(priority, created_at)`` order.
* A job that fails ≤ ``max_retries`` times is re-enqueued with RETRYING
  status; after that it moves to the dead-letter queue.
* Backpressure: ``enqueue()`` raises ``QueueFullError`` when the pending
  count reaches ``max_size``.
* All operations are O(log n) via ``heapq``.

Thread safety: the queue is protected by a ``threading.Lock``.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass
from typing import Any

from umcp.fleet.models import (
    Job,
    JobResult,
    JobStatus,
    QueueStats,
    ValidationVerdict,
)


class QueueFullError(Exception):
    """Raised when the queue has reached its backpressure limit."""


class QueueEmptyError(Exception):
    """Raised on non-blocking dequeue from an empty queue."""


@dataclass
class _QueueItem:
    """Heap item wrapping a Job for priority ordering."""

    priority: int  # JobPriority.value (lower = higher priority)
    created_at: float  # tie-breaker (FIFO within same priority)
    job: Job

    def __lt__(self, other: _QueueItem) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class PriorityQueue:
    """Thread-safe, priority-ordered job queue with DLQ and retry logic.

    Parameters
    ----------
    max_size
        Maximum number of pending jobs.  0 = unbounded.
    default_max_retries
        Default retry limit for jobs that don't specify their own.
    retry_delay_s
        Base delay (seconds) before a retried job becomes eligible again.
        Actual delay = ``retry_delay_s * attempt`` (linear backoff).
    """

    def __init__(
        self,
        *,
        max_size: int = 0,
        default_max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        self._max_size = max_size
        self._default_max_retries = default_max_retries
        self._retry_delay_s = retry_delay_s

        self._heap: list[_QueueItem] = []
        self._jobs: dict[str, Job] = {}  # job_id → Job
        self._dlq: list[Job] = []  # dead-letter queue
        self._lock = threading.Lock()

        # Stats accumulators
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._wait_times: list[float] = []
        self._run_times: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, job: Job) -> Job:
        """Add a job to the queue.

        Raises ``QueueFullError`` if backpressure limit reached.
        """
        with self._lock:
            pending = sum(
                1 for j in self._jobs.values() if j.status in {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RETRYING}
            )
            if self._max_size and pending >= self._max_size:
                raise QueueFullError(f"Queue full: {pending}/{self._max_size} pending jobs")

            if job.max_retries <= 0:
                job.max_retries = self._default_max_retries

            job.status = JobStatus.QUEUED
            job.queued_at = time.time()
            self._jobs[job.job_id] = job
            self._total_submitted += 1
            heapq.heappush(
                self._heap,
                _QueueItem(
                    priority=job.priority.value,
                    created_at=job.created_at,
                    job=job,
                ),
            )
            return job

    def dequeue(self) -> Job | None:
        """Pop the highest-priority eligible job, or ``None`` if empty.

        Skipped items: jobs that have been cancelled or already moved
        to a terminal state are silently discarded.
        """
        with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                job = item.job
                if job.is_terminal:
                    continue  # stale entry
                if job.status == JobStatus.CANCELLED:
                    continue
                job.status = JobStatus.ASSIGNED
                job.assigned_at = time.time()
                if job.queued_at:
                    self._wait_times.append(job.assigned_at - job.queued_at)
                return job
            return None

    def complete(self, job_id: str, result: JobResult) -> Job | None:
        """Mark a job as completed with the given result."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            self._total_completed += 1
            if job.started_at:
                self._run_times.append(job.completed_at - job.started_at)
            return job

    def fail(self, job_id: str, error: str) -> Job | None:
        """Mark a job as failed.

        If the job has retries remaining it is re-enqueued with
        ``RETRYING`` status; otherwise it moves to the dead-letter queue.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            job.attempt += 1
            if job.attempt < job.max_retries:
                # Re-enqueue with linear backoff
                job.status = JobStatus.RETRYING
                job.worker_id = None
                job.assigned_at = None
                job.started_at = None
                delay = self._retry_delay_s * job.attempt
                job.queued_at = time.time() + delay
                heapq.heappush(
                    self._heap,
                    _QueueItem(
                        priority=job.priority.value,
                        created_at=job.queued_at,
                        job=job,
                    ),
                )
            else:
                # Exhausted retries → dead-letter queue
                job.status = JobStatus.DEAD_LETTERED
                job.completed_at = time.time()
                job.result = JobResult(
                    verdict=ValidationVerdict.NON_EVALUABLE,
                    exit_code=1,
                    errors=[f"Exhausted {job.max_retries} retries. Last: {error}"],
                )
                self._dlq.append(job)
                self._total_failed += 1
            return job

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a pending/queued job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and not job.is_terminal:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
            return job

    def get_job(self, job_id: str) -> Job | None:
        """Retrieve a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_dlq(self) -> list[Job]:
        """Return a snapshot of the dead-letter queue."""
        with self._lock:
            return list(self._dlq)

    def replay_dlq(self, job_id: str) -> Job | None:
        """Move a dead-lettered job back into the queue for another attempt."""
        with self._lock:
            for i, job in enumerate(self._dlq):
                if job.job_id == job_id:
                    self._dlq.pop(i)
                    job.status = JobStatus.QUEUED
                    job.attempt = 0
                    job.queued_at = time.time()
                    job.result = None
                    heapq.heappush(
                        self._heap,
                        _QueueItem(
                            priority=job.priority.value,
                            created_at=job.queued_at,
                            job=job,
                        ),
                    )
                    return job
            return None

    def stats(self) -> QueueStats:
        """Return a snapshot of queue statistics."""
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> QueueStats:
        """Return queue stats — caller must hold ``_lock``."""
        counts: dict[str, int] = {s.value: 0 for s in JobStatus}
        for job in self._jobs.values():
            counts[job.status.value] += 1

        avg_wait = sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0
        avg_run = sum(self._run_times) / len(self._run_times) if self._run_times else 0.0

        return QueueStats(
            pending=counts.get("queued", 0) + counts.get("pending", 0),
            running=counts.get("running", 0) + counts.get("assigned", 0),
            completed=self._total_completed,
            failed=self._total_failed,
            dead_lettered=len(self._dlq),
            total_submitted=self._total_submitted,
            avg_wait_s=avg_wait,
            avg_run_s=avg_run,
        )

    def pending_count(self) -> int:
        """Number of jobs waiting to be dequeued."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values() if j.status in {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RETRYING}
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise full queue state for diagnostics."""
        with self._lock:
            return {
                "stats": self._stats_unlocked().to_dict(),
                "pending": [j.to_dict() for j in self._jobs.values() if not j.is_terminal],
                "dead_letter_queue": [j.to_dict() for j in self._dlq],
            }
