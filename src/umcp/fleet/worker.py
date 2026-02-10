"""Worker — task execution, heartbeat, and pool management.

Worker lifecycle
~~~~~~~~~~~~~~~~
1. ``Worker`` registers with a ``Scheduler`` via ``start()``.
2. The worker polls the scheduler for jobs on a configurable interval.
3. On receiving a job, the worker runs ``umcp validate <target>`` in a
   subprocess and reports the result back.
4. Heartbeats are sent on each poll cycle.  The scheduler marks workers
   as OFFLINE if heartbeats stop.
5. ``drain()`` stops accepting new work; ``stop()`` terminates the loop.

``WorkerPool`` manages a fleet of workers — starting, stopping, and
scaling them as a group.

Thread safety: each worker runs in its own ``threading.Thread``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from umcp.fleet.models import (
    Job,
    JobResult,
    JobStatus,
    ValidationVerdict,
    WorkerInfo,
    WorkerStatus,
)


@dataclass
class WorkerConfig:
    """Configuration for a Worker."""

    poll_interval_s: float = 1.0
    heartbeat_interval_s: float = 5.0
    validate_timeout_s: float = 300.0
    capacity: int = 1
    tags: dict[str, str] = field(default_factory=dict)


class Worker:
    """A single validation worker that polls the scheduler for jobs.

    Parameters
    ----------
    worker_id
        Unique identifier.
    scheduler
        Reference to the central scheduler (duck-typed; must have
        ``poll()``, ``report_result()``, ``heartbeat()``, ``register_worker()``).
    config
        Worker configuration.
    """

    def __init__(
        self,
        worker_id: str,
        *,
        scheduler: Any = None,
        config: WorkerConfig | None = None,
    ) -> None:
        self._config = config or WorkerConfig()
        self._info = WorkerInfo(
            worker_id=worker_id,
            hostname=os.uname().nodename,
            pid=os.getpid(),
            capacity=self._config.capacity,
            tags=self._config.tags,
        )
        self._scheduler = scheduler
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._drain = False

    @property
    def worker_id(self) -> str:
        return self._info.worker_id

    @property
    def info(self) -> WorkerInfo:
        return self._info

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._drain = False
        self._info.status = WorkerStatus.IDLE

        if self._scheduler:
            self._scheduler.register_worker(self._info)

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"worker-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker (waits up to ``timeout`` seconds)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._info.status = WorkerStatus.OFFLINE

    def drain(self) -> None:
        """Stop accepting new work; finish current jobs then stop."""
        self._drain = True
        self._info.status = WorkerStatus.DRAINING

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main worker loop — poll, execute, report."""
        last_heartbeat = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # Heartbeat
            if now - last_heartbeat >= self._config.heartbeat_interval_s:
                self._send_heartbeat()
                last_heartbeat = now

            # Drain check
            if self._drain and self._info.active_jobs == 0:
                break

            # Poll for work
            if not self._drain and self._info.active_jobs < self._config.capacity and self._scheduler:
                job = self._scheduler.poll(self.worker_id)
                if job:
                    self._execute(job)

            self._stop_event.wait(self._config.poll_interval_s)

        self._info.status = WorkerStatus.OFFLINE

    def _send_heartbeat(self) -> None:
        """Send heartbeat to the scheduler."""
        self._info.last_heartbeat = time.time()
        if self._scheduler:
            self._scheduler.heartbeat(self.worker_id)

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _execute(self, job: Job) -> None:
        """Execute a single validation job."""
        self._info.status = WorkerStatus.BUSY
        self._info.active_jobs += 1
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.worker_id = self.worker_id

        try:
            result = self._run_validation(job)
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            self._info.completed_count += 1

            if self._scheduler:
                self._scheduler.report_result(job.job_id, result)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._info.failed_count += 1
            if self._scheduler:
                self._scheduler.report_failure(job.job_id, error_msg)

        finally:
            self._info.active_jobs -= 1
            if not self._drain and self._info.active_jobs == 0:
                self._info.status = WorkerStatus.IDLE

    def _run_validation(self, job: Job) -> JobResult:
        """Run ``umcp validate <target>`` as a subprocess."""
        cmd = ["umcp", "validate", job.target]
        if job.strict:
            cmd.append("--strict")

        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.validate_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return JobResult(
                verdict=ValidationVerdict.NON_EVALUABLE,
                exit_code=124,
                errors=[f"Validation timed out after {self._config.validate_timeout_s}s"],
                duration_s=time.time() - started,
            )

        duration = time.time() - started

        # Parse verdict from exit code  (0 = CONFORMANT, 1 = NONCONFORMANT)
        if proc.returncode == 0:
            verdict = ValidationVerdict.CONFORMANT
        elif proc.returncode == 1:
            verdict = ValidationVerdict.NONCONFORMANT
        else:
            verdict = ValidationVerdict.NON_EVALUABLE

        # Try to parse JSON report from stdout
        report: dict[str, Any] = {}
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            report = json.loads(proc.stdout)

        errors: list[str] = []
        if proc.returncode != 0 and proc.stderr:
            errors = [line for line in proc.stderr.strip().splitlines() if line.strip()]

        return JobResult(
            verdict=verdict,
            exit_code=proc.returncode,
            report=report,
            errors=errors,
            duration_s=round(duration, 4),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "info": self._info.to_dict(),
            "config": {
                "poll_interval_s": self._config.poll_interval_s,
                "heartbeat_interval_s": self._config.heartbeat_interval_s,
                "validate_timeout_s": self._config.validate_timeout_s,
                "capacity": self._config.capacity,
            },
            "is_running": self.is_running,
            "drain": self._drain,
        }


# ---------------------------------------------------------------------------
# WorkerPool — manages a fleet of workers
# ---------------------------------------------------------------------------


class WorkerPool:
    """Manages a pool of workers for horizontal scaling.

    Parameters
    ----------
    scheduler
        Central scheduler that workers will poll.
    pool_size
        Number of workers to create.
    config
        Shared config for all workers in the pool.
    """

    def __init__(
        self,
        *,
        scheduler: Any = None,
        pool_size: int = 4,
        config: WorkerConfig | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._config = config or WorkerConfig()
        self._workers: list[Worker] = []

        for i in range(pool_size):
            w = Worker(
                f"pool-{i}",
                scheduler=scheduler,
                config=self._config,
            )
            self._workers.append(w)

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def active(self) -> int:
        return sum(1 for w in self._workers if w.is_running)

    def start(self) -> None:
        """Start all workers."""
        for w in self._workers:
            w.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop all workers."""
        for w in self._workers:
            w.stop(timeout=timeout / max(len(self._workers), 1))

    def drain(self) -> None:
        """Drain all workers (finish current work, stop accepting new)."""
        for w in self._workers:
            w.drain()

    def scale_up(self, count: int = 1) -> list[Worker]:
        """Add workers to the pool."""
        new_workers: list[Worker] = []
        base = len(self._workers)
        for i in range(count):
            w = Worker(
                f"pool-{base + i}",
                scheduler=self._scheduler,
                config=self._config,
            )
            w.start()
            self._workers.append(w)
            new_workers.append(w)
        return new_workers

    def scale_down(self, count: int = 1) -> int:
        """Remove idle workers from the pool.  Returns actual removed count."""
        removed = 0
        for _ in range(count):
            # Find an idle worker to remove
            for i, w in enumerate(self._workers):
                if w.info.status == WorkerStatus.IDLE:
                    w.stop(timeout=5.0)
                    self._workers.pop(i)
                    removed += 1
                    break
        return removed

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.size,
            "active": self.active,
            "workers": [w.to_dict() for w in self._workers],
        }
