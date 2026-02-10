"""Tests for umcp.fleet — distributed scheduler, worker, queue, cache, tenant.

Test groups:
  test_models_*         — data model serialisation and invariants
  test_queue_*          — priority ordering, DLQ, retry, backpressure
  test_cache_*          — content-addressable storage, LRU, TTL, tenancy
  test_tenant_*         — quota enforcement, rate limiting, CRUD
  test_worker_*         — worker lifecycle, pool scaling
  test_scheduler_*      — end-to-end: submit → route → execute → cache
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from umcp.fleet.cache import ArtifactCache, FilesystemBackend
from umcp.fleet.models import (
    CacheEntry,
    Job,
    JobPriority,
    JobResult,
    JobStatus,
    QueueStats,
    TenantQuota,
    ValidationVerdict,
    WorkerInfo,
    WorkerStatus,
)
from umcp.fleet.queue import PriorityQueue, QueueFullError
from umcp.fleet.scheduler import Scheduler
from umcp.fleet.tenant import (
    QuotaExceededError,
    RateLimitExceededError,
    TenantManager,
    TenantNotFoundError,
)
from umcp.fleet.worker import Worker, WorkerPool

# =====================================================================
# Models
# =====================================================================


class TestJobModel:
    def test_job_default_id(self) -> None:
        job = Job(target="casepacks/hello_world")
        assert len(job.job_id) == 16
        assert job.status == JobStatus.PENDING
        assert not job.is_terminal

    def test_job_terminal_states(self) -> None:
        for status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DEAD_LETTERED):
            job = Job(target="x", status=status)
            assert job.is_terminal

    def test_job_non_terminal_states(self) -> None:
        for status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.RETRYING):
            job = Job(target="x", status=status)
            assert not job.is_terminal

    def test_job_elapsed(self) -> None:
        job = Job(target="x", started_at=100.0, completed_at=105.0)
        assert job.elapsed == pytest.approx(5.0)

    def test_job_serialisation(self) -> None:
        job = Job(target="casepacks/test", tenant_id="acme", priority=JobPriority.HIGH)
        d = job.to_dict()
        assert d["target"] == "casepacks/test"
        assert d["tenant_id"] == "acme"
        assert d["priority"] == "HIGH"
        assert d["status"] == "pending"

    def test_job_result_serialisation(self) -> None:
        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
            duration_s=1.2345,
        )
        d = result.to_dict()
        assert d["verdict"] == "CONFORMANT"
        assert d["exit_code"] == 0
        assert d["duration_s"] == 1.2345


class TestWorkerInfoModel:
    def test_worker_availability(self) -> None:
        info = WorkerInfo(worker_id="w1", capacity=2, active_jobs=0)
        assert info.is_available
        info.active_jobs = 2
        assert not info.is_available

    def test_worker_load(self) -> None:
        info = WorkerInfo(worker_id="w1", capacity=4, active_jobs=2)
        assert info.load == pytest.approx(0.5)

    def test_worker_serialisation(self) -> None:
        info = WorkerInfo(worker_id="w1", hostname="node-0")
        d = info.to_dict()
        assert d["worker_id"] == "w1"
        assert d["hostname"] == "node-0"


class TestQueueStatsModel:
    def test_queue_stats_serialisation(self) -> None:
        stats = QueueStats(pending=5, running=2, completed=10)
        d = stats.to_dict()
        assert d["pending"] == 5
        assert d["running"] == 2
        assert d["completed"] == 10


class TestCacheEntryModel:
    def test_content_key(self) -> None:
        key = CacheEntry.content_key(b"hello world")
        assert len(key) == 64  # SHA256 hex


class TestTenantQuotaModel:
    def test_default_quota(self) -> None:
        q = TenantQuota()
        assert q.max_concurrent_jobs == 4
        assert q.max_queued_jobs == 100
        assert q.max_retries == 3


# =====================================================================
# Priority Queue
# =====================================================================


class TestPriorityQueue:
    def test_enqueue_dequeue_fifo(self) -> None:
        q = PriorityQueue()
        q.enqueue(Job(target="a"))
        q.enqueue(Job(target="b"))
        out1 = q.dequeue()
        out2 = q.dequeue()
        assert out1 is not None and out1.target == "a"
        assert out2 is not None and out2.target == "b"

    def test_priority_ordering(self) -> None:
        q = PriorityQueue()
        q.enqueue(Job(target="low", priority=JobPriority.LOW))
        q.enqueue(Job(target="critical", priority=JobPriority.CRITICAL))
        q.enqueue(Job(target="normal", priority=JobPriority.NORMAL))

        out = q.dequeue()
        assert out is not None and out.target == "critical"
        out = q.dequeue()
        assert out is not None and out.target == "normal"
        out = q.dequeue()
        assert out is not None and out.target == "low"

    def test_dequeue_empty(self) -> None:
        q = PriorityQueue()
        assert q.dequeue() is None

    def test_backpressure(self) -> None:
        q = PriorityQueue(max_size=2)
        q.enqueue(Job(target="a"))
        q.enqueue(Job(target="b"))
        with pytest.raises(QueueFullError):
            q.enqueue(Job(target="c"))

    def test_complete(self) -> None:
        q = PriorityQueue()
        q.enqueue(Job(target="x"))
        job = q.dequeue()
        assert job is not None
        result = JobResult(verdict=ValidationVerdict.CONFORMANT, exit_code=0)
        completed = q.complete(job.job_id, result)
        assert completed is not None
        assert completed.status == JobStatus.COMPLETED
        assert completed.result is not None
        assert completed.result.verdict == ValidationVerdict.CONFORMANT

    def test_fail_with_retry(self) -> None:
        q = PriorityQueue(default_max_retries=3, retry_delay_s=0.0)
        q.enqueue(Job(target="x"))
        job = q.dequeue()
        assert job is not None

        failed = q.fail(job.job_id, "timeout")
        assert failed is not None
        assert failed.status == JobStatus.RETRYING
        assert failed.attempt == 1

        # Should be re-enqueued
        retried = q.dequeue()
        assert retried is not None
        assert retried.job_id == job.job_id

    def test_fail_exhausted_retries_dlq(self) -> None:
        q = PriorityQueue(default_max_retries=1, retry_delay_s=0.0)
        q.enqueue(Job(target="x"))
        job = q.dequeue()
        assert job is not None

        q.fail(job.job_id, "error 1")
        assert job.status == JobStatus.DEAD_LETTERED
        assert len(q.get_dlq()) == 1

    def test_cancel(self) -> None:
        q = PriorityQueue()
        job = q.enqueue(Job(target="x"))
        q.cancel(job.job_id)
        assert job.status == JobStatus.CANCELLED

    def test_replay_dlq(self) -> None:
        q = PriorityQueue(default_max_retries=1, retry_delay_s=0.0)
        q.enqueue(Job(target="x"))
        job = q.dequeue()
        assert job is not None
        q.fail(job.job_id, "oops")
        assert len(q.get_dlq()) == 1

        replayed = q.replay_dlq(job.job_id)
        assert replayed is not None
        assert replayed.status == JobStatus.QUEUED
        assert len(q.get_dlq()) == 0

    def test_stats(self) -> None:
        q = PriorityQueue()
        q.enqueue(Job(target="a"))
        q.enqueue(Job(target="b"))
        stats = q.stats()
        assert stats.pending >= 2
        assert stats.total_submitted == 2

    def test_pending_count(self) -> None:
        q = PriorityQueue()
        q.enqueue(Job(target="a"))
        q.enqueue(Job(target="b"))
        assert q.pending_count() == 2
        q.dequeue()
        assert q.pending_count() == 1


# =====================================================================
# Artifact Cache
# =====================================================================


class TestArtifactCacheMemory:
    def test_put_get(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"hello world")
        assert len(key) == 64
        data = cache.get(key)
        assert data == b"hello world"

    def test_get_missing(self) -> None:
        cache = ArtifactCache()
        assert cache.get("nonexistent") is None

    def test_has(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"test data")
        assert cache.has(key)
        assert not cache.has("missing")

    def test_delete(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"data")
        assert cache.delete(key)
        assert cache.get(key) is None
        assert not cache.delete("missing")

    def test_tenant_isolation(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"acme-data", tenant_id="acme")
        # Same key, different tenant → miss
        assert cache.get(key, tenant_id="other") is None
        # Correct tenant → hit
        assert cache.get(key, tenant_id="acme") == b"acme-data"

    def test_lru_eviction(self) -> None:
        cache = ArtifactCache(max_bytes=100)
        # Put 60 bytes
        k1 = cache.put(b"A" * 60)
        # Put 60 more → should evict k1
        k2 = cache.put(b"B" * 60)
        assert cache.get(k1) is None
        assert cache.get(k2) == b"B" * 60

    def test_ttl_eviction(self) -> None:
        cache = ArtifactCache(ttl_s=0.1)
        key = cache.put(b"ephemeral")
        assert cache.get(key) == b"ephemeral"
        time.sleep(0.15)
        assert cache.get(key) is None

    def test_stats(self) -> None:
        cache = ArtifactCache()
        cache.put(b"data")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["total_bytes"] == 4

    def test_clear(self) -> None:
        cache = ArtifactCache()
        cache.put(b"a", tenant_id="t1")
        cache.put(b"b", tenant_id="t2")
        count = cache.clear(tenant_id="t1")
        assert count == 1
        assert cache.stats()["total_entries"] == 1

    def test_clear_all(self) -> None:
        cache = ArtifactCache()
        cache.put(b"a")
        cache.put(b"b")
        count = cache.clear()
        assert count == 2
        assert cache.stats()["total_entries"] == 0

    def test_content_key_deterministic(self) -> None:
        k1 = ArtifactCache.content_key(b"hello")
        k2 = ArtifactCache.content_key(b"hello")
        assert k1 == k2

    def test_tenant_usage(self) -> None:
        cache = ArtifactCache()
        cache.put(b"AAAA", tenant_id="acme")
        cache.put(b"BB", tenant_id="acme")
        cache.put(b"CCC", tenant_id="other")
        assert cache.tenant_usage("acme") == 6
        assert cache.tenant_usage("other") == 3

    def test_hit_rate(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"data")
        cache.get(key)  # hit
        cache.get(key)  # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        # 2 hits / 3 total = 0.6667
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.001)


class TestArtifactCacheFilesystem:
    def test_filesystem_backend(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(tmp_path / "cache")
        cache = ArtifactCache(backend=backend)
        key = cache.put(b"persistent data")
        assert cache.get(key) == b"persistent data"

        # Verify file exists on disk
        files = list((tmp_path / "cache").rglob("*.bin"))
        assert len(files) == 1

    def test_filesystem_clear(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(tmp_path / "cache")
        cache = ArtifactCache(backend=backend)
        cache.put(b"a")
        cache.put(b"b")
        cache.clear()
        assert cache.stats()["total_entries"] == 0


# =====================================================================
# Tenant Manager
# =====================================================================


class TestTenantManager:
    def test_register_get(self) -> None:
        mgr = TenantManager()
        t = mgr.register("acme", name="Acme Corp")
        assert t.tenant_id == "acme"
        assert t.name == "Acme Corp"
        assert mgr.get("acme") is t

    def test_register_duplicate(self) -> None:
        mgr = TenantManager()
        mgr.register("acme")
        with pytest.raises(ValueError, match="already registered"):
            mgr.register("acme")

    def test_get_not_found(self) -> None:
        mgr = TenantManager()
        with pytest.raises(TenantNotFoundError):
            mgr.get("missing")

    def test_list_tenants(self) -> None:
        mgr = TenantManager()
        mgr.register("a")
        mgr.register("b")
        assert len(mgr.list_tenants()) == 2

    def test_remove(self) -> None:
        mgr = TenantManager()
        mgr.register("acme")
        assert mgr.remove("acme")
        assert not mgr.remove("acme")

    def test_disable_enable(self) -> None:
        mgr = TenantManager()
        mgr.register("acme")
        mgr.disable("acme")
        with pytest.raises(QuotaExceededError, match="disabled"):
            mgr.check_submission("acme")
        mgr.enable("acme")
        mgr.check_submission("acme")  # should not raise

    def test_concurrency_quota(self) -> None:
        mgr = TenantManager()
        mgr.register("acme", quota=TenantQuota(max_concurrent_jobs=1))
        mgr.record_submission("acme")
        mgr.record_start("acme")
        with pytest.raises(QuotaExceededError, match="concurrency"):
            mgr.check_submission("acme")

    def test_queue_depth_quota(self) -> None:
        mgr = TenantManager()
        mgr.register("acme", quota=TenantQuota(max_queued_jobs=1))
        mgr.record_submission("acme")
        with pytest.raises(QuotaExceededError, match="queue depth"):
            mgr.check_submission("acme")

    def test_priority_quota(self) -> None:
        mgr = TenantManager()
        mgr.register("acme", quota=TenantQuota(allowed_priorities=[JobPriority.NORMAL]))
        with pytest.raises(QuotaExceededError, match="priority"):
            mgr.check_submission("acme", priority=JobPriority.CRITICAL)

    def test_rate_limit(self) -> None:
        mgr = TenantManager()
        mgr.register("acme", quota=TenantQuota(rate_limit_per_minute=2))
        mgr.check_submission("acme")
        mgr.record_submission("acme")
        mgr.check_submission("acme")
        mgr.record_submission("acme")
        with pytest.raises(RateLimitExceededError):
            mgr.check_submission("acme")

    def test_artifact_budget(self) -> None:
        mgr = TenantManager()
        mgr.register("acme", quota=TenantQuota(max_artifact_bytes=100))
        mgr.check_artifact_budget("acme", 50)
        mgr.record_artifact("acme", 50)
        mgr.check_artifact_budget("acme", 50)
        mgr.record_artifact("acme", 50)
        with pytest.raises(QuotaExceededError, match="artifact budget"):
            mgr.check_artifact_budget("acme", 10)

    def test_record_completion_failure(self) -> None:
        mgr = TenantManager()
        t = mgr.register("acme")
        mgr.record_submission("acme")
        mgr.record_start("acme")
        assert t.active_jobs == 1
        mgr.record_completion("acme")
        assert t.active_jobs == 0
        assert t.total_completed == 1
        mgr.record_submission("acme")
        mgr.record_start("acme")
        mgr.record_failure("acme")
        assert t.total_failed == 1

    def test_serialisation(self) -> None:
        mgr = TenantManager()
        mgr.register("acme")
        d = mgr.to_dict()
        assert "tenants" in d
        assert "acme" in d["tenants"]


# =====================================================================
# Worker
# =====================================================================


class TestWorker:
    def test_worker_info(self) -> None:
        w = Worker("w1")
        assert w.worker_id == "w1"
        assert w.info.status == WorkerStatus.IDLE

    def test_worker_not_running_before_start(self) -> None:
        w = Worker("w1")
        assert not w.is_running

    def test_worker_serialisation(self) -> None:
        w = Worker("w1")
        d = w.to_dict()
        assert d["worker_id"] == "w1"
        assert not d["is_running"]


class TestWorkerPool:
    def test_pool_size(self) -> None:
        pool = WorkerPool(pool_size=3)
        assert pool.size == 3
        assert pool.active == 0

    def test_pool_serialisation(self) -> None:
        pool = WorkerPool(pool_size=2)
        d = pool.to_dict()
        assert d["pool_size"] == 2
        assert len(d["workers"]) == 2


# =====================================================================
# Scheduler
# =====================================================================


class TestScheduler:
    def test_submit_creates_job(self) -> None:
        s = Scheduler()
        job = s.submit("casepacks/hello_world")
        assert job.status in {JobStatus.QUEUED, JobStatus.COMPLETED}
        assert job.target == "casepacks/hello_world"

    def test_submit_with_tenant(self) -> None:
        s = Scheduler()
        s.register_tenant("acme")
        job = s.submit("casepacks/test", tenant_id="acme")
        assert job.tenant_id == "acme"

    def test_submit_auto_registers_tenant(self) -> None:
        s = Scheduler()
        job = s.submit("casepacks/test", tenant_id="new-tenant")
        assert job.tenant_id == "new-tenant"
        tenant = s.get_tenant("new-tenant")
        assert tenant.tenant_id == "new-tenant"

    def test_cancel_job(self) -> None:
        s = Scheduler()
        job = s.submit("casepacks/test")
        cancelled = s.cancel(job.job_id)
        assert cancelled is not None
        assert cancelled.status == JobStatus.CANCELLED

    def test_get_job(self) -> None:
        s = Scheduler()
        job = s.submit("casepacks/test")
        found = s.get_job(job.job_id)
        assert found is not None
        assert found.job_id == job.job_id

    def test_cache_hit(self) -> None:
        s = Scheduler()
        # Manually populate cache
        import json

        cache_key = s._cache_key("casepacks/cached", False)
        data = json.dumps(
            {
                "verdict": "CONFORMANT",
                "exit_code": 0,
                "report": {},
                "duration_s": 0.5,
            }
        ).encode()
        s.cache.put(data, key=cache_key)

        job = s.submit("casepacks/cached")
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.result.verdict == ValidationVerdict.CONFORMANT

    def test_queue_stats(self) -> None:
        s = Scheduler()
        s.submit("a")
        s.submit("b")
        stats = s.queue_stats()
        assert stats.total_submitted >= 2

    def test_register_worker(self) -> None:
        s = Scheduler()
        info = WorkerInfo(worker_id="w1")
        s.register_worker(info)
        workers = s.list_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "w1"

    def test_poll_no_workers(self) -> None:
        s = Scheduler()
        s.submit("test")
        assert s.poll("unregistered") is None

    def test_poll_assigns_job(self) -> None:
        s = Scheduler()
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)
        s.submit("casepacks/test")
        job = s.poll("w1")
        assert job is not None
        assert job.worker_id == "w1"
        assert job.status == JobStatus.ASSIGNED

    def test_report_result(self) -> None:
        s = Scheduler()
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)
        s.submit("casepacks/test")
        job = s.poll("w1")
        assert job is not None

        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
            duration_s=1.0,
        )
        s.report_result(job.job_id, result)
        completed = s.get_job(job.job_id)
        assert completed is not None
        assert completed.status == JobStatus.COMPLETED

    def test_report_failure(self) -> None:
        s = Scheduler(queue=PriorityQueue(default_max_retries=1, retry_delay_s=0.0))
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)
        s.submit("casepacks/test")
        job = s.poll("w1")
        assert job is not None

        s.report_failure(job.job_id, "crashed")
        found = s.get_job(job.job_id)
        assert found is not None
        assert found.status == JobStatus.DEAD_LETTERED

    def test_heartbeat(self) -> None:
        s = Scheduler()
        info = WorkerInfo(worker_id="w1")
        s.register_worker(info)
        s.heartbeat("w1")
        workers = s.list_workers()
        assert workers[0].last_heartbeat > 0

    def test_cache_stats(self) -> None:
        s = Scheduler()
        stats = s.cache_stats()
        assert "total_entries" in stats

    def test_list_tenants(self) -> None:
        s = Scheduler()
        s.register_tenant("a")
        s.register_tenant("b")
        assert len(s.list_tenants()) == 2

    def test_serialisation(self) -> None:
        s = Scheduler()
        s.register_tenant("acme")
        s.submit("test", tenant_id="acme")
        d = s.to_dict()
        assert "workers" in d
        assert "queue" in d
        assert "cache" in d
        assert "tenants" in d


# =====================================================================
# Integration: Scheduler + Worker end-to-end (in-process)
# =====================================================================


class TestSchedulerWorkerIntegration:
    """End-to-end tests using a mock validation subprocess."""

    def test_submit_poll_complete_cycle(self) -> None:
        """Simulate the full job lifecycle without subprocess."""
        s = Scheduler()
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)

        # Submit
        job = s.submit("casepacks/hello_world")
        assert job.status == JobStatus.QUEUED

        # Poll
        polled = s.poll("w1")
        assert polled is not None
        assert polled.status == JobStatus.ASSIGNED

        # Report result
        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
            duration_s=0.5,
        )
        s.report_result(polled.job_id, result)

        # Verify final state
        done = s.get_job(polled.job_id)
        assert done is not None
        assert done.status == JobStatus.COMPLETED
        assert done.result is not None
        assert done.result.verdict == ValidationVerdict.CONFORMANT

    def test_multi_tenant_isolation(self) -> None:
        """Two tenants should have independent quotas."""
        s = Scheduler()
        s.register_tenant("alpha", quota=TenantQuota(max_queued_jobs=2))
        s.register_tenant("beta", quota=TenantQuota(max_queued_jobs=2))

        s.submit("a1", tenant_id="alpha")
        s.submit("a2", tenant_id="alpha")
        # Alpha is full
        with pytest.raises(QuotaExceededError):
            s.submit("a3", tenant_id="alpha")

        # Beta still works
        s.submit("b1", tenant_id="beta")
        s.submit("b2", tenant_id="beta")

    def test_priority_dispatch_order(self) -> None:
        """Higher-priority jobs should be polled first."""
        s = Scheduler()
        info = WorkerInfo(worker_id="w1", capacity=10)
        s.register_worker(info)

        s.submit("low", priority=JobPriority.LOW)
        s.submit("critical", priority=JobPriority.CRITICAL)
        s.submit("normal", priority=JobPriority.NORMAL)

        j1 = s.poll("w1")
        j2 = s.poll("w1")
        j3 = s.poll("w1")
        assert j1 is not None and j1.target == "critical"
        assert j2 is not None and j2.target == "normal"
        assert j3 is not None and j3.target == "low"

    def test_cache_hit_on_second_submit(self) -> None:
        """After completing a CONFORMANT job, re-submitting should hit cache."""
        s = Scheduler()
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)

        # First run
        s.submit("casepacks/test")
        polled = s.poll("w1")
        assert polled is not None
        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
        )
        s.report_result(polled.job_id, result)

        # Second submit — should hit cache
        j2 = s.submit("casepacks/test")
        assert j2.status == JobStatus.COMPLETED
        assert j2.result is not None
        assert j2.result.verdict == ValidationVerdict.CONFORMANT

    def test_retry_then_succeed(self) -> None:
        """Jobs should be retried on failure before dead-lettering."""
        s = Scheduler(queue=PriorityQueue(default_max_retries=2, retry_delay_s=0.0))
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)

        s.submit("casepacks/flaky")

        # First attempt fails
        j1 = s.poll("w1")
        assert j1 is not None
        s.report_failure(j1.job_id, "timeout")

        # Should be re-queued
        j2 = s.poll("w1")
        assert j2 is not None
        assert j2.job_id == j1.job_id

        # Second attempt succeeds
        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
        )
        s.report_result(j2.job_id, result)
        done = s.get_job(j2.job_id)
        assert done is not None
        assert done.status == JobStatus.COMPLETED
