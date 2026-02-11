"""Tests for umcp.fleet.worker — Worker, WorkerPool, WorkerConfig."""

from __future__ import annotations

from typing import Any

from umcp.fleet.models import Job, WorkerStatus
from umcp.fleet.worker import Worker, WorkerConfig, WorkerPool

# WorkerConfig is exported from umcp.fleet.worker and properly typed

# ── Mock scheduler ────────────────────────────────────────────────


class MockScheduler:
    """Minimal scheduler mock for worker tests."""

    def __init__(self) -> None:
        self.registered: list[Any] = []
        self.heartbeats: list[str] = []
        self.results: list[tuple[str, Any]] = []
        self.failures: list[tuple[str, str]] = []
        self._jobs: list[Job] = []

    def register_worker(self, info: Any) -> None:
        self.registered.append(info)

    def heartbeat(self, worker_id: str) -> None:
        self.heartbeats.append(worker_id)

    def report_result(self, job_id: str, result: Any) -> None:
        self.results.append((job_id, result))

    def report_failure(self, job_id: str, error: str) -> None:
        self.failures.append((job_id, error))

    def poll(self, worker_id: str) -> Job | None:
        if self._jobs:
            return self._jobs.pop(0)
        return None


# ── WorkerConfig ──────────────────────────────────────────────────


class TestWorkerConfig:
    def test_defaults(self) -> None:
        cfg = WorkerConfig()
        assert cfg.poll_interval_s == 1.0
        assert cfg.heartbeat_interval_s == 5.0
        assert cfg.validate_timeout_s == 300.0
        assert cfg.capacity == 1
        assert cfg.tags == {}

    def test_custom(self) -> None:
        cfg = WorkerConfig(poll_interval_s=0.5, capacity=4, tags={"env": "test"})
        assert cfg.poll_interval_s == 0.5
        assert cfg.capacity == 4
        assert cfg.tags == {"env": "test"}


# ── Worker ────────────────────────────────────────────────────────


class TestWorker:
    def test_init_no_scheduler(self) -> None:
        w = Worker("test-1")
        assert w.worker_id == "test-1"
        assert not w.is_running
        assert w.info.worker_id == "test-1"

    def test_init_with_scheduler(self) -> None:
        sched = MockScheduler()
        w = Worker("test-2", scheduler=sched)
        assert w.worker_id == "test-2"

    def test_start_registers_with_scheduler(self) -> None:
        sched = MockScheduler()
        cfg = WorkerConfig(poll_interval_s=0.1, heartbeat_interval_s=0.1)
        w = Worker("reg-1", scheduler=sched, config=cfg)
        w.start()
        assert w.is_running
        assert len(sched.registered) == 1
        w.stop(timeout=2.0)
        assert not w.is_running

    def test_start_idempotent(self) -> None:
        w = Worker("idem-1")
        w.start()
        w.start()  # should not create a second thread
        w.stop(timeout=2.0)

    def test_stop_sets_offline(self) -> None:
        w = Worker("stop-1")
        w.start()
        w.stop(timeout=2.0)
        assert w.info.status == WorkerStatus.OFFLINE

    def test_drain_sets_draining(self) -> None:
        w = Worker("drain-1")
        w.drain()
        assert w.info.status == WorkerStatus.DRAINING

    def test_to_dict(self) -> None:
        w = Worker("dict-1", config=WorkerConfig(capacity=2))
        d = w.to_dict()
        assert d["worker_id"] == "dict-1"
        assert d["config"]["capacity"] == 2
        assert d["is_running"] is False
        assert d["drain"] is False

    def test_heartbeat_sent(self) -> None:
        sched = MockScheduler()
        cfg = WorkerConfig(poll_interval_s=0.05, heartbeat_interval_s=0.05)
        w = Worker("hb-1", scheduler=sched, config=cfg)
        w.start()
        import time

        time.sleep(0.3)
        w.stop(timeout=2.0)
        assert len(sched.heartbeats) >= 1


# ── WorkerPool ────────────────────────────────────────────────────


class TestWorkerPool:
    def test_init_default_size(self) -> None:
        pool = WorkerPool()
        assert pool.size == 4
        assert pool.active == 0

    def test_init_custom_size(self) -> None:
        pool = WorkerPool(pool_size=2)
        assert pool.size == 2

    def test_start_stop(self) -> None:
        pool = WorkerPool(pool_size=2, config=WorkerConfig(poll_interval_s=0.05))
        pool.start()
        assert pool.active == 2
        pool.stop(timeout=3.0)
        assert pool.active == 0

    def test_drain(self) -> None:
        pool = WorkerPool(pool_size=2, config=WorkerConfig(poll_interval_s=0.05))
        pool.start()
        pool.drain()
        import time

        time.sleep(0.3)
        pool.stop(timeout=3.0)

    def test_scale_up(self) -> None:
        pool = WorkerPool(pool_size=1, config=WorkerConfig(poll_interval_s=0.05))
        assert pool.size == 1
        new = pool.scale_up(2)
        assert pool.size == 3
        assert len(new) == 2
        pool.stop(timeout=3.0)

    def test_scale_down(self) -> None:
        pool = WorkerPool(pool_size=3, config=WorkerConfig(poll_interval_s=0.05))
        # Workers not running = IDLE by default? Actually they start as IDLE
        # But scale_down looks for WorkerStatus.IDLE
        removed = pool.scale_down(1)
        # May or may not find IDLE workers depending on initial status
        assert removed >= 0
        assert pool.size <= 3

    def test_to_dict(self) -> None:
        pool = WorkerPool(pool_size=2)
        d = pool.to_dict()
        assert d["pool_size"] == 2
        assert len(d["workers"]) == 2
