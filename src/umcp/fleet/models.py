"""Fleet data models — shared dataclasses for scheduler, worker, queue, cache, and tenant.

All models are frozen dataclasses unless mutation is structurally required
(e.g. ``Job`` tracks mutable status).  Serialisation uses explicit
``.to_dict()`` methods — never ``dataclasses.asdict()``.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(Enum):
    """Lifecycle states of a validation job.

    Terminal states: COMPLETED, FAILED, CANCELLED, DEAD_LETTERED.
    """

    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"


class JobPriority(Enum):
    """Job priority levels — lower numeric value = higher priority."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class WorkerStatus(Enum):
    """Worker lifecycle states."""

    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    OFFLINE = "offline"


class ValidationVerdict(Enum):
    """Three-valued validation verdict (mirrors UMCP convention)."""

    CONFORMANT = "CONFORMANT"
    NONCONFORMANT = "NONCONFORMANT"
    NON_EVALUABLE = "NON_EVALUABLE"


# ---------------------------------------------------------------------------
# Job models
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """A single validation job submitted to the scheduler.

    Jobs are mutable during their lifecycle (status, worker assignment,
    retry count, timestamps).  The ``job_id`` is stable and unique.
    """

    target: str  # casepack/file/repo path
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    tenant_id: str = ""
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    strict: bool = False

    # Scheduling metadata
    created_at: float = field(default_factory=time.time)
    queued_at: float | None = None
    assigned_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    attempt: int = 0
    max_retries: int = 0  # 0 = use queue default
    tags: dict[str, str] = field(default_factory=dict)

    # Result (populated on completion)
    result: JobResult | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.DEAD_LETTERED,
        }

    @property
    def elapsed(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        if self.started_at:
            return time.time() - self.started_at
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "target": self.target,
            "tenant_id": self.tenant_id,
            "priority": self.priority.name,
            "status": self.status.value,
            "strict": self.strict,
            "created_at": self.created_at,
            "queued_at": self.queued_at,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "worker_id": self.worker_id,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "tags": self.tags,
            "elapsed": self.elapsed,
        }
        if self.result:
            d["result"] = self.result.to_dict()
        return d


@dataclass(frozen=True)
class JobResult:
    """Immutable result of a completed validation job."""

    verdict: ValidationVerdict
    exit_code: int
    report: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    artifact_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "report": self.report,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 4),
            "artifact_keys": self.artifact_keys,
        }


# ---------------------------------------------------------------------------
# Worker models
# ---------------------------------------------------------------------------


@dataclass
class WorkerInfo:
    """Runtime information about a registered worker."""

    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    hostname: str = ""
    pid: int = 0
    capacity: int = 1  # max concurrent jobs
    active_jobs: int = 0
    completed_count: int = 0
    failed_count: int = 0
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.status == WorkerStatus.IDLE and self.active_jobs < self.capacity

    @property
    def load(self) -> float:
        """Load factor in [0, 1]."""
        return self.active_jobs / self.capacity if self.capacity else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "hostname": self.hostname,
            "pid": self.pid,
            "capacity": self.capacity,
            "active_jobs": self.active_jobs,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "load": round(self.load, 3),
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Queue models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueStats:
    """Snapshot of queue state."""

    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    dead_lettered: int = 0
    total_submitted: int = 0
    avg_wait_s: float = 0.0
    avg_run_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "dead_lettered": self.dead_lettered,
            "total_submitted": self.total_submitted,
            "avg_wait_s": round(self.avg_wait_s, 4),
            "avg_run_s": round(self.avg_run_s, 4),
        }


# ---------------------------------------------------------------------------
# Tenant models
# ---------------------------------------------------------------------------


@dataclass
class TenantQuota:
    """Resource quotas for a tenant."""

    max_concurrent_jobs: int = 4
    max_queued_jobs: int = 100
    max_retries: int = 3
    max_artifact_bytes: int = 500 * 1024 * 1024  # 500 MB
    rate_limit_per_minute: int = 60
    allowed_priorities: list[JobPriority] = field(
        default_factory=lambda: list(JobPriority),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_queued_jobs": self.max_queued_jobs,
            "max_retries": self.max_retries,
            "max_artifact_bytes": self.max_artifact_bytes,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "allowed_priorities": [p.name for p in self.allowed_priorities],
        }


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """An entry in the artifact cache."""

    key: str  # SHA256 content-address
    tenant_id: str = ""
    size_bytes: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def content_key(data: bytes) -> str:
        """SHA256-based content-addressable key."""
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tenant_id": self.tenant_id,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "hit_count": self.hit_count,
            "metadata": self.metadata,
        }
