"""UMCP Fleet — Distributed Scheduler, Worker, Queue, Cache, and Tenancy.

This subpackage provides the "run-at-fleet-scale" substrate for UMCP
validation.  Every component follows UMCP conventions:

  * ``from __future__ import annotations`` in every file.
  * Dataclass-first (Pydantic only at the API boundary).
  * Three-valued status: CONFORMANT / NONCONFORMANT / NON_EVALUABLE.
  * Optional-dependency guarding (redis, celery, etc.).
  * SHA256 content-addressable artifact identity.

Public surface:

  Scheduler  — submit, route, and track validation jobs
  Worker     — register workers, heartbeat, task lifecycle
  Queue      — priority queue with DLQ, retry, backpressure
  Cache      — content-addressable multi-node artifact cache
  Tenant     — multi-tenant isolation, quotas, namespaces

Quick start (in-process, no external dependencies)::

    from umcp.fleet import Scheduler, Worker, Cache, Tenant

    tenant = Tenant(tenant_id="acme")
    scheduler = Scheduler()
    worker = Worker(worker_id="local-0", scheduler=scheduler)
    worker.start()
    job = scheduler.submit("casepacks/hello_world", tenant=tenant)
    result = scheduler.wait(job.job_id)
"""

from __future__ import annotations

from umcp.fleet.cache import ArtifactCache
from umcp.fleet.models import (
    Job,
    JobPriority,
    JobResult,
    JobStatus,
    QueueStats,
    TenantQuota,
    WorkerInfo,
    WorkerStatus,
)
from umcp.fleet.queue import PriorityQueue
from umcp.fleet.scheduler import Scheduler
from umcp.fleet.tenant import Tenant, TenantManager
from umcp.fleet.worker import Worker, WorkerPool

__all__ = [
    "ArtifactCache",
    "Job",
    "JobPriority",
    "JobResult",
    "JobStatus",
    "PriorityQueue",
    "QueueStats",
    "Scheduler",
    "Tenant",
    "TenantManager",
    "TenantQuota",
    "Worker",
    "WorkerInfo",
    "WorkerPool",
    "WorkerStatus",
]
