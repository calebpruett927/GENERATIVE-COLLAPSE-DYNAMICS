"""Multi-tenant isolation, quotas, namespaces, and rate limiting.

Tenancy model
~~~~~~~~~~~~~
* Each ``Tenant`` has a unique ``tenant_id`` and a ``TenantQuota``.
* The ``TenantManager`` enforces quotas at submission time:
  concurrency limits, queue depth, artifact storage, and rate limiting.
* Tenants are namespace-scoped: a tenant's jobs, artifacts, and metrics
  are isolated from other tenants.  Cross-tenant access is denied.
* Rate limiting uses a sliding-window counter per minute.

Thread safety: the manager is protected by a ``threading.Lock``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from umcp.fleet.models import JobPriority, TenantQuota

# ---------------------------------------------------------------------------
# Quota enforcement errors
# ---------------------------------------------------------------------------


class QuotaExceededError(Exception):
    """Raised when a tenant operation violates its quota."""


class TenantNotFoundError(Exception):
    """Raised when an operation references an unregistered tenant."""


class RateLimitExceededError(QuotaExceededError):
    """Raised when a tenant exceeds its per-minute rate limit."""


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------


@dataclass
class Tenant:
    """A registered tenant with quota and runtime counters."""

    tenant_id: str
    name: str = ""
    quota: TenantQuota = field(default_factory=TenantQuota)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    # Runtime counters (mutable)
    active_jobs: int = 0
    queued_jobs: int = 0
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    artifact_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name or self.tenant_id,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "quota": self.quota.to_dict(),
            "active_jobs": self.active_jobs,
            "queued_jobs": self.queued_jobs,
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "artifact_bytes": self.artifact_bytes,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Rate limiter (sliding window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Per-tenant sliding-window rate limiter."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, tenant_id: str, limit: int) -> bool:
        """Return True if the tenant is within rate limit."""
        now = time.time()
        window = self._windows[tenant_id]
        # Purge entries older than 60s
        cutoff = now - 60.0
        self._windows[tenant_id] = [t for t in window if t > cutoff]
        return len(self._windows[tenant_id]) < limit

    def record(self, tenant_id: str) -> None:
        """Record a submission event for the tenant."""
        self._windows[tenant_id].append(time.time())

    def current_count(self, tenant_id: str) -> int:
        """Current count in the sliding window."""
        now = time.time()
        cutoff = now - 60.0
        return sum(1 for t in self._windows.get(tenant_id, []) if t > cutoff)


# ---------------------------------------------------------------------------
# Tenant Manager
# ---------------------------------------------------------------------------


class TenantManager:
    """Central registry for tenants with quota enforcement.

    The manager is the single authority for tenant CRUD and all quota
    checks.  The scheduler delegates to the manager before accepting
    any job submission.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._rate_limiter = _RateLimiter()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Tenant CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        tenant_id: str,
        *,
        name: str = "",
        quota: TenantQuota | None = None,
        tags: dict[str, str] | None = None,
    ) -> Tenant:
        """Register a new tenant.  Raises if ID already exists."""
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"Tenant '{tenant_id}' already registered")
            tenant = Tenant(
                tenant_id=tenant_id,
                name=name or tenant_id,
                quota=quota or TenantQuota(),
                tags=tags or {},
            )
            self._tenants[tenant_id] = tenant
            return tenant

    def get(self, tenant_id: str) -> Tenant:
        """Retrieve a tenant.  Raises ``TenantNotFoundError`` if missing."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            return tenant

    def list_tenants(self) -> list[Tenant]:
        """Return all registered tenants."""
        with self._lock:
            return list(self._tenants.values())

    def update_quota(self, tenant_id: str, quota: TenantQuota) -> Tenant:
        """Update quotas for an existing tenant."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            tenant.quota = quota
            return tenant

    def disable(self, tenant_id: str) -> None:
        """Disable a tenant (rejects new submissions)."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            tenant.enabled = False

    def enable(self, tenant_id: str) -> None:
        """Re-enable a tenant."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            tenant.enabled = True

    def remove(self, tenant_id: str) -> bool:
        """Remove a tenant from the registry."""
        with self._lock:
            return self._tenants.pop(tenant_id, None) is not None

    # ------------------------------------------------------------------
    # Quota enforcement (called by Scheduler before enqueuing)
    # ------------------------------------------------------------------

    def check_submission(
        self,
        tenant_id: str,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> None:
        """Validate that a tenant may submit a new job.

        Raises ``QuotaExceededError`` or subclass if any limit is violated.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            if not tenant.enabled:
                raise QuotaExceededError(f"Tenant '{tenant_id}' is disabled")

            q = tenant.quota

            # Concurrency check
            if tenant.active_jobs >= q.max_concurrent_jobs:
                raise QuotaExceededError(
                    f"Tenant '{tenant_id}' concurrency limit: {tenant.active_jobs}/{q.max_concurrent_jobs}"
                )

            # Queue depth check
            if tenant.queued_jobs >= q.max_queued_jobs:
                raise QuotaExceededError(
                    f"Tenant '{tenant_id}' queue depth limit: {tenant.queued_jobs}/{q.max_queued_jobs}"
                )

            # Priority check
            if priority not in q.allowed_priorities:
                raise QuotaExceededError(f"Tenant '{tenant_id}' not allowed priority {priority.name}")

            # Rate limit check
            if not self._rate_limiter.check(tenant_id, q.rate_limit_per_minute):
                raise RateLimitExceededError(f"Tenant '{tenant_id}' rate limit exceeded: {q.rate_limit_per_minute}/min")

    def record_submission(self, tenant_id: str) -> None:
        """Record a successful submission (increment counters)."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.queued_jobs += 1
                tenant.total_submitted += 1
                self._rate_limiter.record(tenant_id)

    def record_start(self, tenant_id: str) -> None:
        """Record a job starting execution."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.active_jobs += 1
                tenant.queued_jobs = max(0, tenant.queued_jobs - 1)

    def record_completion(self, tenant_id: str) -> None:
        """Record a job completing (success or failure)."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.active_jobs = max(0, tenant.active_jobs - 1)
                tenant.total_completed += 1

    def record_failure(self, tenant_id: str) -> None:
        """Record a job failure."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.active_jobs = max(0, tenant.active_jobs - 1)
                tenant.total_failed += 1

    def check_artifact_budget(self, tenant_id: str, size_bytes: int) -> None:
        """Check if a tenant has artifact storage budget remaining."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            if tenant.artifact_bytes + size_bytes > tenant.quota.max_artifact_bytes:
                raise QuotaExceededError(
                    f"Tenant '{tenant_id}' artifact budget exceeded: "
                    f"{tenant.artifact_bytes + size_bytes} > {tenant.quota.max_artifact_bytes}"
                )

    def record_artifact(self, tenant_id: str, size_bytes: int) -> None:
        """Record artifact storage usage."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.artifact_bytes += size_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialise full tenant registry for diagnostics."""
        with self._lock:
            return {
                "tenants": {tid: t.to_dict() for tid, t in self._tenants.items()},
                "rate_limiter": {tid: self._rate_limiter.current_count(tid) for tid in self._tenants},
            }
