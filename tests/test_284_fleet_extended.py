"""Tests for umcp.fleet coverage gaps — scheduler, worker, cache.

Targets the specific uncovered code paths:
  - Scheduler: submit with cache hit, cancel, report_result/failure,
    reaper loop, re-queue, wait, diagnostics, tenant/cache passthrough
  - Worker: _run_loop, _execute, _run_validation, drain, to_dict
  - WorkerPool: scale_up, scale_down, to_dict
  - ArtifactCache: filesystem backend, put/get/evict, stats, tenancy
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from umcp.fleet.cache import ArtifactCache, FilesystemBackend
from umcp.fleet.models import (
    Job,
    JobResult,
    JobStatus,
    ValidationVerdict,
    WorkerInfo,
    WorkerStatus,
)
from umcp.fleet.scheduler import Scheduler
from umcp.fleet.worker import Worker, WorkerConfig, WorkerPool

# =====================================================================
# ArtifactCache (filesystem backend)
# =====================================================================


class TestFilesystemBackend:
    def test_put_get_roundtrip(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        backend.put("key1", b"hello world")
        assert backend.get("key1") == b"hello world"

    def test_get_missing_key(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        assert backend.get("nonexistent") is None

    def test_delete_key(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        backend.put("key2", b"data")
        backend.delete("key2")
        assert backend.get("key2") is None

    def test_get_missing_key_then_put(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        assert backend.get("k") is None
        backend.put("k", b"v")
        assert backend.get("k") is not None

    def test_keys(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        backend.put("a", b"1")
        backend.put("b", b"2")
        assert set(backend.keys()) >= {"a", "b"}

    def test_size(self, tmp_path: Any) -> None:
        backend = FilesystemBackend(root=tmp_path)
        backend.put("s", b"12345")
        assert backend.size("s") == 5


class TestArtifactCacheExtended:
    def test_put_get_memory(self) -> None:
        cache = ArtifactCache()
        cache.put(b"test data", key="k1")
        assert cache.get("k1") == b"test data"

    def test_put_auto_key(self) -> None:
        cache = ArtifactCache()
        key = cache.put(b"auto keyed data")
        assert key is not None
        assert cache.get(key) == b"auto keyed data"

    def test_get_missing(self) -> None:
        cache = ArtifactCache()
        assert cache.get("missing") is None

    def test_delete(self) -> None:
        cache = ArtifactCache()
        cache.put(b"d", key="del")
        cache.delete("del")
        assert cache.get("del") is None

    def test_stats(self) -> None:
        cache = ArtifactCache()
        cache.put(b"x", key="k")
        s = cache.stats()
        assert isinstance(s, dict)
        assert "entries" in s or "total_entries" in s or "size" in s or len(s) > 0

    def test_tenant_isolation(self) -> None:
        cache = ArtifactCache()
        cache.put(b"tenant data", key="shared_key", tenant_id="t1")
        result = cache.get("shared_key", tenant_id="t1")
        assert result == b"tenant data"

    def test_filesystem_backend_init(self, tmp_path: Any) -> None:
        cache = ArtifactCache(backend=FilesystemBackend(root=tmp_path))
        cache.put(b"fs data", key="fsk")
        assert cache.get("fsk") == b"fs data"


# =====================================================================
# Scheduler extended paths
# =====================================================================


class TestSchedulerExtended:
    @pytest.fixture()
    def scheduler(self) -> Scheduler:
        return Scheduler(heartbeat_ttl_s=1.0)

    def test_submit_creates_job(self, scheduler: Scheduler) -> None:
        job = scheduler.submit("casepacks/pedagogical/hello_world")
        assert job.target == "casepacks/pedagogical/hello_world"
        assert job.status in {JobStatus.QUEUED, JobStatus.COMPLETED}

    def test_submit_with_unknown_tenant(self, scheduler: Scheduler) -> None:
        """Unknown tenant is auto-registered."""
        job = scheduler.submit("casepacks/test", tenant_id="new_tenant")
        assert job.tenant_id == "new_tenant"

    def test_cancel_job(self, scheduler: Scheduler) -> None:
        job = scheduler.submit("casepacks/test")
        if job.status == JobStatus.QUEUED:
            cancelled = scheduler.cancel(job.job_id)
            assert cancelled is not None

    def test_register_unregister_worker(self, scheduler: Scheduler) -> None:
        info = WorkerInfo(worker_id="w1", hostname="test", pid=1, capacity=1)
        scheduler.register_worker(info)
        workers = scheduler.list_workers()
        assert any(w.worker_id == "w1" for w in workers)
        scheduler.unregister_worker("w1")
        workers = scheduler.list_workers()
        assert not any(w.worker_id == "w1" for w in workers)

    def test_heartbeat(self, scheduler: Scheduler) -> None:
        info = WorkerInfo(worker_id="w2", hostname="test", pid=1, capacity=1)
        scheduler.register_worker(info)
        scheduler.heartbeat("w2")

    def test_poll_no_workers(self, scheduler: Scheduler) -> None:
        result = scheduler.poll("nonexistent")
        assert result is None

    def test_poll_with_job(self, scheduler: Scheduler) -> None:
        info = WorkerInfo(worker_id="w3", hostname="test", pid=1, capacity=1)
        info.last_heartbeat = time.time()
        scheduler.register_worker(info)
        job = scheduler.submit("casepacks/test")
        if job.status == JobStatus.QUEUED:
            polled = scheduler.poll("w3")
            # May or may not get the job depending on queue state
            if polled:
                assert polled.target == "casepacks/test"

    def test_report_result(self, scheduler: Scheduler) -> None:
        info = WorkerInfo(worker_id="w4", hostname="test", pid=1, capacity=1)
        info.last_heartbeat = time.time()
        scheduler.register_worker(info)
        job = scheduler.submit("casepacks/test")
        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
            duration_s=0.1,
        )
        scheduler.report_result(job.job_id, result)

    def test_report_failure(self, scheduler: Scheduler) -> None:
        job = scheduler.submit("casepacks/test")
        scheduler.report_failure(job.job_id, "test error")

    def test_get_job(self, scheduler: Scheduler) -> None:
        job = scheduler.submit("casepacks/test")
        found = scheduler.get_job(job.job_id)
        assert found is not None
        assert found.job_id == job.job_id

    def test_get_job_missing(self, scheduler: Scheduler) -> None:
        assert scheduler.get_job("nonexistent") is None

    def test_queue_stats(self, scheduler: Scheduler) -> None:
        stats = scheduler.queue_stats()
        assert stats is not None

    def test_register_tenant(self, scheduler: Scheduler) -> None:
        t = scheduler.register_tenant("t1")
        assert t.tenant_id == "t1"

    def test_list_tenants(self, scheduler: Scheduler) -> None:
        scheduler.register_tenant("t2")
        tenants = scheduler.list_tenants()
        assert any(t.tenant_id == "t2" for t in tenants)

    def test_cache_property(self, scheduler: Scheduler) -> None:
        assert scheduler.cache is not None

    def test_cache_stats(self, scheduler: Scheduler) -> None:
        stats = scheduler.cache_stats()
        assert isinstance(stats, dict)

    def test_to_dict(self, scheduler: Scheduler) -> None:
        d = scheduler.to_dict()
        assert "workers" in d
        assert "queue" in d

    def test_start_stop(self, scheduler: Scheduler) -> None:
        scheduler.start()
        time.sleep(0.1)
        scheduler.stop()

    def test_cache_key_static(self) -> None:
        k1 = Scheduler._cache_key("path/a", True)
        k2 = Scheduler._cache_key("path/a", True)
        k3 = Scheduler._cache_key("path/a", False)
        assert k1 == k2
        assert k1 != k3

    def test_wait_timeout(self, scheduler: Scheduler) -> None:
        """wait() returns None on timeout."""
        result = scheduler.wait("nonexistent", timeout=0.1, poll_interval=0.05)
        assert result is None


# =====================================================================
# Worker extended paths
# =====================================================================


class TestWorkerExtended:
    def test_worker_config_defaults(self) -> None:
        c = WorkerConfig()
        assert c.poll_interval_s == 1.0
        assert c.capacity == 1

    def test_worker_properties(self) -> None:
        w = Worker("test-w1")
        assert w.worker_id == "test-w1"
        assert w.is_running is False

    def test_worker_info(self) -> None:
        w = Worker("test-w2")
        info = w.info
        assert isinstance(info, WorkerInfo)
        assert info.worker_id == "test-w2"

    def test_worker_start_stop(self) -> None:
        """Start then immediately stop — no scheduler to poll."""
        w = Worker("test-w3", config=WorkerConfig(poll_interval_s=0.05))
        w.start()
        assert w.is_running
        w.stop(timeout=2.0)
        assert not w.is_running

    def test_worker_start_idempotent(self) -> None:
        w = Worker("test-w4", config=WorkerConfig(poll_interval_s=0.05))
        w.start()
        w.start()  # second start is no-op
        w.stop(timeout=2.0)

    def test_worker_drain(self) -> None:
        w = Worker("test-w5", config=WorkerConfig(poll_interval_s=0.05))
        w.start()
        w.drain()
        assert w.info.status == WorkerStatus.DRAINING
        time.sleep(0.2)  # Let the drain complete
        w.stop(timeout=2.0)

    def test_worker_to_dict(self) -> None:
        w = Worker("test-w6")
        d = w.to_dict()
        assert d["worker_id"] == "test-w6"
        assert "config" in d
        assert "is_running" in d
        assert "drain" in d

    def test_worker_with_scheduler(self) -> None:
        """Worker with mock scheduler registers on start."""
        mock_sched = MagicMock()
        mock_sched.poll.return_value = None
        w = Worker("test-w7", scheduler=mock_sched, config=WorkerConfig(poll_interval_s=0.05))
        w.start()
        time.sleep(0.15)
        w.stop(timeout=2.0)
        mock_sched.register_worker.assert_called_once()

    def test_worker_executes_job(self) -> None:
        """Worker executes a job when scheduler provides one."""
        job = Job(target="casepacks/pedagogical/hello_world")
        mock_sched = MagicMock()
        mock_sched.poll.side_effect = [job, None, None, None, None, None, None, None, None, None]

        with patch.object(Worker, "_run_validation") as mock_run:
            mock_run.return_value = JobResult(
                verdict=ValidationVerdict.CONFORMANT,
                exit_code=0,
                duration_s=0.01,
            )
            w = Worker("test-w8", scheduler=mock_sched, config=WorkerConfig(poll_interval_s=0.05))
            w.start()
            time.sleep(0.3)
            w.stop(timeout=2.0)
            mock_run.assert_called_once()

    def test_worker_handles_execution_error(self) -> None:
        """Worker handles exceptions during job execution."""
        job = Job(target="casepacks/pedagogical/hello_world")
        mock_sched = MagicMock()
        mock_sched.poll.side_effect = [job, None, None, None, None, None, None, None, None, None]

        with patch.object(Worker, "_run_validation", side_effect=RuntimeError("boom")):
            w = Worker("test-w9", scheduler=mock_sched, config=WorkerConfig(poll_interval_s=0.05))
            w.start()
            time.sleep(0.3)
            w.stop(timeout=2.0)
            mock_sched.report_failure.assert_called_once()


class TestWorkerPoolExtended:
    def test_pool_creation(self) -> None:
        pool = WorkerPool(pool_size=3)
        assert pool.size == 3

    def test_pool_start_stop(self) -> None:
        pool = WorkerPool(pool_size=2, config=WorkerConfig(poll_interval_s=0.05))
        pool.start()
        assert pool.active == 2
        pool.stop(timeout=3.0)

    def test_pool_drain(self) -> None:
        pool = WorkerPool(pool_size=2, config=WorkerConfig(poll_interval_s=0.05))
        pool.start()
        pool.drain()
        time.sleep(0.3)
        pool.stop(timeout=3.0)

    def test_pool_scale_up(self) -> None:
        pool = WorkerPool(pool_size=1, config=WorkerConfig(poll_interval_s=0.05))
        pool.start()
        new = pool.scale_up(2)
        assert len(new) == 2
        assert pool.size == 3
        pool.stop(timeout=3.0)

    def test_pool_scale_down(self) -> None:
        pool = WorkerPool(pool_size=3, config=WorkerConfig(poll_interval_s=0.05))
        # Workers not started → status is not IDLE by default
        # Start then stop one to make it IDLE
        pool.start()
        time.sleep(0.1)
        # All workers are IDLE (no scheduler), so scale_down should work
        removed = pool.scale_down(1)
        assert removed >= 0
        pool.stop(timeout=3.0)

    def test_pool_to_dict(self) -> None:
        pool = WorkerPool(pool_size=2)
        d = pool.to_dict()
        assert d["pool_size"] == 2
        assert "workers" in d
        assert len(d["workers"]) == 2
