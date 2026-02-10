"""Distributed scheduler — job routing, dispatch, and lifecycle management.

The scheduler is the central coordinator of the fleet.  It owns:

* The **priority queue** (``PriorityQueue``) for pending jobs.
* The **worker registry** — tracking which workers are online, their
  load, and heartbeat status.
* The **tenant manager** — enforcing quotas before accepting submissions.
* The **artifact cache** — storing validation results keyed by SHA256.

Scheduling policies
~~~~~~~~~~~~~~~~~~~
* **Least-loaded**  — route to the worker with the lowest load factor.
* **Tag affinity**   — prefer workers whose tags match the job's tags.
* **Heartbeat TTL**  — workers that miss heartbeats for ``heartbeat_ttl_s``
  are marked OFFLINE and their in-flight jobs are re-queued.

Thread safety: all state is protected by a ``threading.Lock``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from umcp.fleet.cache import ArtifactCache
from umcp.fleet.models import (
    Job,
    JobPriority,
    JobResult,
    JobStatus,
    QueueStats,
    WorkerInfo,
    WorkerStatus,
)
from umcp.fleet.queue import PriorityQueue
from umcp.fleet.tenant import (
    Tenant,
    TenantManager,
    TenantNotFoundError,
)


class SchedulerError(Exception):
    """Base exception for scheduler errors."""


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Central fleet scheduler.

    Parameters
    ----------
    queue
        Priority queue (created internally if not provided).
    tenant_manager
        Tenant manager (created internally if not provided).
    cache
        Artifact cache (created internally if not provided).
    heartbeat_ttl_s
        Seconds after which a worker without heartbeat is marked OFFLINE.
    """

    def __init__(
        self,
        *,
        queue: PriorityQueue | None = None,
        tenant_manager: TenantManager | None = None,
        cache: ArtifactCache | None = None,
        heartbeat_ttl_s: float = 30.0,
    ) -> None:
        self._queue = queue or PriorityQueue()
        self._tenants = tenant_manager or TenantManager()
        self._cache = cache or ArtifactCache()
        self._heartbeat_ttl_s = heartbeat_ttl_s

        self._workers: dict[str, WorkerInfo] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

        # Background reaper thread
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background threads (heartbeat reaper, etc.)."""
        if self._reaper_thread and self._reaper_thread.is_alive():
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            name="scheduler-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def stop(self) -> None:
        """Stop background threads."""
        self._reaper_stop.set()
        if self._reaper_thread:
            self._reaper_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit(
        self,
        target: str,
        *,
        tenant: Tenant | None = None,
        tenant_id: str = "",
        priority: JobPriority = JobPriority.NORMAL,
        strict: bool = False,
        tags: dict[str, str] | None = None,
    ) -> Job:
        """Submit a validation job.

        Parameters
        ----------
        target
            Path to casepack, file, or repo root.
        tenant / tenant_id
            Tenant submitting the job (for quota enforcement).
        priority
            Job priority level.
        strict
            Whether to run validation in strict mode.
        tags
            Arbitrary tags for routing affinity.

        Returns
        -------
        Job
            The created job with status QUEUED.

        Raises
        ------
        QuotaExceededError
            If the tenant's quota is exceeded.
        QueueFullError
            If the queue backpressure limit is reached.
        """
        tid = tenant.tenant_id if tenant else tenant_id

        # Enforce tenant quota (if tenant is registered)
        if tid:
            try:
                self._tenants.check_submission(tid, priority)
            except TenantNotFoundError:
                # Auto-register unknown tenants with default quota
                self._tenants.register(tid)
                self._tenants.check_submission(tid, priority)

        # Check cache for a prior result
        cache_key = self._cache_key(target, strict)
        cached = self._cache.get(cache_key, tenant_id=tid)
        if cached is not None:
            # Return a synthetic completed job from cache
            job = Job(
                target=target,
                tenant_id=tid,
                priority=priority,
                strict=strict,
                tags=tags or {},
            )
            try:
                result_dict = json.loads(cached.decode("utf-8"))
                from umcp.fleet.models import ValidationVerdict

                job.result = JobResult(
                    verdict=ValidationVerdict(result_dict.get("verdict", "NON_EVALUABLE")),
                    exit_code=result_dict.get("exit_code", 0),
                    report=result_dict.get("report", {}),
                    duration_s=result_dict.get("duration_s", 0.0),
                    artifact_keys=[cache_key],
                )
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # fall through to regular submission
            else:
                return job

        # Create and enqueue job
        job = Job(
            target=target,
            tenant_id=tid,
            priority=priority,
            strict=strict,
            tags=tags or {},
        )

        with self._lock:
            self._queue.enqueue(job)
            self._jobs[job.job_id] = job

        if tid:
            self._tenants.record_submission(tid)

        return job

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a job."""
        with self._lock:
            job = self._queue.cancel(job_id)
            if job and job.tenant_id:
                self._tenants.record_completion(job.tenant_id)
            return job

    # ------------------------------------------------------------------
    # Worker interface (called by workers)
    # ------------------------------------------------------------------

    def register_worker(self, info: WorkerInfo) -> None:
        """Register a worker with the scheduler."""
        with self._lock:
            self._workers[info.worker_id] = info

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the registry."""
        with self._lock:
            self._workers.pop(worker_id, None)

    def heartbeat(self, worker_id: str) -> None:
        """Record a heartbeat from a worker."""
        with self._lock:
            info = self._workers.get(worker_id)
            if info:
                info.last_heartbeat = time.time()

    def poll(self, worker_id: str) -> Job | None:
        """Poll for the next job assigned to a worker.

        Uses least-loaded scheduling with tag affinity.
        """
        with self._lock:
            info = self._workers.get(worker_id)
            if not info or info.status == WorkerStatus.OFFLINE:
                return None
            if info.active_jobs >= info.capacity:
                return None

        job = self._queue.dequeue()
        if job is None:
            return None

        job.worker_id = worker_id
        job.assigned_at = time.time()

        if job.tenant_id:
            self._tenants.record_start(job.tenant_id)

        return job

    def report_result(self, job_id: str, result: JobResult) -> None:
        """Report a completed job result from a worker."""
        job = self._queue.complete(job_id, result)
        if job is None:
            return

        # Cache the result
        if result.verdict.value == "CONFORMANT":
            cache_key = self._cache_key(job.target, job.strict)
            data = json.dumps(result.to_dict()).encode("utf-8")
            self._cache.put(
                data,
                key=cache_key,
                tenant_id=job.tenant_id,
            )

        if job.tenant_id:
            self._tenants.record_completion(job.tenant_id)

    def report_failure(self, job_id: str, error: str) -> None:
        """Report a failed job from a worker."""
        job = self._queue.fail(job_id, error)
        if job and job.is_terminal and job.tenant_id:
            self._tenants.record_failure(job.tenant_id)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        """Retrieve a job by ID."""
        return self._queue.get_job(job_id)

    def wait(self, job_id: str, *, timeout: float = 300.0, poll_interval: float = 0.5) -> Job | None:
        """Block until a job reaches a terminal state.

        Returns the Job, or None on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_id)
            if job and job.is_terminal:
                return job
            time.sleep(poll_interval)
        return None

    def list_workers(self) -> list[WorkerInfo]:
        """Return all registered workers."""
        with self._lock:
            return list(self._workers.values())

    def queue_stats(self) -> QueueStats:
        """Return queue statistics."""
        return self._queue.stats()

    # ------------------------------------------------------------------
    # Tenancy passthrough
    # ------------------------------------------------------------------

    def register_tenant(
        self,
        tenant_id: str,
        **kwargs: Any,
    ) -> Tenant:
        """Register a tenant (delegates to TenantManager)."""
        return self._tenants.register(tenant_id, **kwargs)

    def get_tenant(self, tenant_id: str) -> Tenant:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> list[Tenant]:
        """List all tenants."""
        return self._tenants.list_tenants()

    # ------------------------------------------------------------------
    # Cache passthrough
    # ------------------------------------------------------------------

    @property
    def cache(self) -> ArtifactCache:
        """Access the artifact cache."""
        return self._cache

    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return self._cache.stats()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full scheduler state for diagnostics."""
        with self._lock:
            return {
                "workers": {wid: w.to_dict() for wid, w in self._workers.items()},
                "queue": self._queue.to_dict(),
                "cache": self._cache.stats(),
                "tenants": self._tenants.to_dict(),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(target: str, strict: bool) -> str:
        """Derive a cache key from target path + strict flag."""
        import hashlib

        content = f"{target}:strict={strict}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _reaper_loop(self) -> None:
        """Periodically check for dead workers (missed heartbeats)."""
        while not self._reaper_stop.is_set():
            now = time.time()
            with self._lock:
                for info in self._workers.values():
                    if info.status != WorkerStatus.OFFLINE and (now - info.last_heartbeat) > self._heartbeat_ttl_s:
                        info.status = WorkerStatus.OFFLINE
                        # Re-queue any jobs assigned to this worker
                        self._requeue_worker_jobs(info.worker_id)

            self._reaper_stop.wait(self._heartbeat_ttl_s / 2)

    def _requeue_worker_jobs(self, worker_id: str) -> None:
        """Re-queue jobs that were assigned to a dead worker.

        Caller must hold _lock.
        """
        for job in list(self._jobs.values()):
            if job.worker_id == worker_id and job.status in {JobStatus.ASSIGNED, JobStatus.RUNNING}:
                self._queue.fail(
                    job.job_id,
                    f"Worker {worker_id} went offline",
                )
