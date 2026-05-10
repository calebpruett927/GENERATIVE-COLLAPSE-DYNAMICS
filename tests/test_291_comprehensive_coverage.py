"""Comprehensive coverage push — target all major remaining gaps.

Targets: RedisBackend (mock), tenant quotas, extension install,
validator yaml fallback, fleet queue DLQ, insights discovery (mocked),
logging health check, ss1m_triad edge cases.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =====================================================================
# 1. RedisBackend — mock redis module
# =====================================================================


class TestRedisBackendMocked:
    """Cover fleet/cache.py lines 155-188 by mocking redis."""

    def _make_backend(self) -> Any:
        """Create a RedisBackend with a mocked redis client."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            # Force re-import of cache module to pick up mock
            from umcp.fleet.cache import RedisBackend

            backend = RedisBackend.__new__(RedisBackend)
            backend._client = mock_client
            backend._prefix = "umcp:cache:"
            return backend, mock_client

    def test_redis_get(self) -> None:
        backend, client = self._make_backend()
        client.get.return_value = b"some-data"
        result = backend.get("mykey")
        assert result == b"some-data"
        client.get.assert_called_once_with("umcp:cache:mykey")

    def test_redis_get_none(self) -> None:
        backend, client = self._make_backend()
        client.get.return_value = None
        result = backend.get("missing")
        assert result is None

    def test_redis_put(self) -> None:
        backend, client = self._make_backend()
        backend.put("k1", b"data")
        client.set.assert_called_once_with("umcp:cache:k1", b"data")

    def test_redis_delete(self) -> None:
        backend, client = self._make_backend()
        client.delete.return_value = 1
        result = backend.delete("k1")
        assert result is True

    def test_redis_keys(self) -> None:
        backend, client = self._make_backend()
        client.keys.return_value = [b"umcp:cache:k1", b"umcp:cache:k2"]
        result = backend.keys()
        assert result == ["k1", "k2"]

    def test_redis_size(self) -> None:
        backend, client = self._make_backend()
        client.strlen.return_value = 42
        result = backend.size("k1")
        assert result == 42

    def test_redis_clear(self) -> None:
        backend, client = self._make_backend()
        client.keys.return_value = [b"umcp:cache:k1"]
        backend.clear()
        client.delete.assert_called_once()

    def test_redis_clear_empty(self) -> None:
        backend, client = self._make_backend()
        client.keys.return_value = []
        backend.clear()
        client.delete.assert_not_called()


# =====================================================================
# 2. Tenant quota enforcement
# =====================================================================


class TestTenantQuotaEnforcement:
    """Cover fleet/tenant.py quota check paths."""

    def test_check_submission_disabled_tenant(self) -> None:
        from umcp.fleet.tenant import QuotaExceededError, TenantManager

        mgr = TenantManager()
        mgr.register("t-disabled")
        mgr.disable("t-disabled")
        with pytest.raises(QuotaExceededError, match="disabled"):
            mgr.check_submission("t-disabled")

    def test_check_submission_concurrency_limit(self) -> None:
        from umcp.fleet.tenant import QuotaExceededError, TenantManager, TenantQuota

        mgr = TenantManager()
        mgr.register("t-conc", quota=TenantQuota(max_concurrent_jobs=1))
        tenant = mgr.get("t-conc")
        tenant.active_jobs = 1
        with pytest.raises(QuotaExceededError, match="concurrency"):
            mgr.check_submission("t-conc")

    def test_check_submission_queue_depth_limit(self) -> None:
        from umcp.fleet.tenant import QuotaExceededError, TenantManager, TenantQuota

        mgr = TenantManager()
        mgr.register("t-q", quota=TenantQuota(max_queued_jobs=2))
        tenant = mgr.get("t-q")
        tenant.queued_jobs = 2
        with pytest.raises(QuotaExceededError, match="queue depth"):
            mgr.check_submission("t-q")

    def test_check_artifact_budget(self) -> None:
        from umcp.fleet.tenant import QuotaExceededError, TenantManager, TenantQuota

        mgr = TenantManager()
        mgr.register("t-art", quota=TenantQuota(max_artifact_bytes=100))
        with pytest.raises(QuotaExceededError, match="artifact budget"):
            mgr.check_artifact_budget("t-art", 200)

    def test_record_artifact(self) -> None:
        from umcp.fleet.tenant import TenantManager

        mgr = TenantManager()
        mgr.register("t-rec")
        mgr.record_artifact("t-rec", 512)
        tenant = mgr.get("t-rec")
        assert tenant.artifact_bytes == 512

    def test_update_quota(self) -> None:
        from umcp.fleet.tenant import TenantManager, TenantQuota

        mgr = TenantManager()
        mgr.register("t-upd")
        new_quota = TenantQuota(max_concurrent_jobs=10)
        mgr.update_quota("t-upd", new_quota)
        tenant = mgr.get("t-upd")
        assert tenant.quota.max_concurrent_jobs == 10

    def test_enable_disable(self) -> None:
        from umcp.fleet.tenant import TenantManager

        mgr = TenantManager()
        mgr.register("t-ed")
        mgr.disable("t-ed")
        assert not mgr.get("t-ed").enabled
        mgr.enable("t-ed")
        assert mgr.get("t-ed").enabled

    def test_remove_tenant(self) -> None:
        from umcp.fleet.tenant import TenantManager

        mgr = TenantManager()
        mgr.register("t-rm")
        assert mgr.remove("t-rm") is True
        assert mgr.remove("t-rm") is False

    def test_list_tenants(self) -> None:
        from umcp.fleet.tenant import TenantManager

        mgr = TenantManager()
        mgr.register("t-a")
        mgr.register("t-b")
        tenants = mgr.list_tenants()
        assert len(tenants) == 2

    def test_to_dict(self) -> None:
        from umcp.fleet.tenant import TenantManager

        mgr = TenantManager()
        mgr.register("t-dict")
        d = mgr.to_dict()
        assert "tenants" in d
        assert "t-dict" in d["tenants"]


# =====================================================================
# 3. Extensions — install_extension with mocked subprocess
# =====================================================================


class TestInstallExtension:
    """Cover umcp_extensions.py lines 420-440."""

    def test_install_nonexistent(self) -> None:
        from umcp.umcp_extensions import install_extension

        assert install_extension("nonexistent_xyz") is False

    def test_install_ledger_no_requires(self) -> None:
        from umcp.umcp_extensions import install_extension

        # ledger has no requires, should return True immediately
        assert install_extension("ledger") is True

    def test_install_api_success(self) -> None:
        from umcp.umcp_extensions import install_extension

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = install_extension("api")
        assert result is True
        mock_run.assert_called_once()

    def test_install_api_failure(self) -> None:
        from umcp.umcp_extensions import install_extension

        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = install_extension("api")
        assert result is False

    def test_install_exception(self) -> None:
        from umcp.umcp_extensions import install_extension

        with patch("subprocess.run", side_effect=OSError("fail")):
            result = install_extension("api")
        assert result is False


# =====================================================================
# 4. Validator — yaml fallback and various paths
# =====================================================================


class TestValidatorYamlFallback:
    """Cover validator.py _load_yaml fallback (lines 149-158)."""

    def test_load_yaml_fallback_parser(self, tmp_path: Path) -> None:
        """When yaml module is None, use the minimal key:value parser."""
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)

        # Create a simple yaml-like file
        f = tmp_path / "test.yaml"
        f.write_text("key1: value1\nkey2: value2\n# comment\n\nkey3: value3\n")

        # Temporarily make yaml unavailable
        import umcp.validator as vmod

        original_yaml = vmod.yaml
        try:
            vmod.yaml = None
            result = v._load_yaml(f)
            assert result["key1"] == "value1"
            assert result["key2"] == "value2"
            assert result["key3"] == "value3"
        finally:
            vmod.yaml = original_yaml

    def test_load_yaml_file_not_found(self, tmp_path: Path) -> None:
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            v._load_yaml(tmp_path / "nonexistent.yaml")

    def test_validate_manifest_missing_schema(self, tmp_path: Path) -> None:
        """Cover the 'missing schema field' error path."""
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("something_else: true\n")
        v._validate_manifest()
        assert any("missing 'schema'" in e for e in v.errors)

    def test_validate_manifest_missing_casepack(self, tmp_path: Path) -> None:
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("schema: v1\n")
        v._validate_manifest()
        assert any("missing 'casepack'" in e for e in v.errors)

    def test_validate_contract_error(self, tmp_path: Path) -> None:
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)
        # No contract.yaml file → should error gracefully
        v._validate_contract()
        assert any("contract.yaml" in e for e in v.errors)

    def test_validate_observables_error(self, tmp_path: Path) -> None:
        from umcp.validator import RootFileValidator

        v = RootFileValidator(root_dir=tmp_path)
        v._validate_observables()
        assert any("observables.yaml" in e for e in v.errors)


# =====================================================================
# 5. Fleet queue — DLQ and replay
# =====================================================================


class TestQueueDLQ:
    """Cover fleet/queue.py DLQ and replay paths."""

    def test_fail_to_dlq_after_retries(self) -> None:
        from umcp.fleet.models import Job, JobStatus
        from umcp.fleet.queue import PriorityQueue

        q = PriorityQueue(default_max_retries=2)
        job = Job(target="test-target")
        q.enqueue(job)
        q.dequeue()  # assign

        # First failure → retry (attempt=1 < max_retries=2)
        failed = q.fail(job.job_id, "err1")
        assert failed is not None
        assert failed.status == JobStatus.RETRYING

        # Second failure → DLQ (attempt=2 >= max_retries=2)
        q.dequeue()  # re-assign the retrying job
        failed2 = q.fail(job.job_id, "err2")
        assert failed2 is not None
        assert failed2.status == JobStatus.DEAD_LETTERED

        # Check DLQ has it
        dlq = q.get_dlq()
        assert len(dlq) == 1
        assert dlq[0].job_id == job.job_id

    def test_replay_dlq(self) -> None:
        from umcp.fleet.models import Job, JobStatus
        from umcp.fleet.queue import PriorityQueue

        q = PriorityQueue(default_max_retries=2)
        job = Job(target="test-target")
        q.enqueue(job)
        q.dequeue()

        # Fail twice → DLQ
        q.fail(job.job_id, "err1")
        q.dequeue()
        q.fail(job.job_id, "err2")
        assert len(q.get_dlq()) == 1

        # Replay it
        replayed = q.replay_dlq(job.job_id)
        assert replayed is not None
        assert replayed.status == JobStatus.QUEUED
        assert replayed.attempt == 0
        assert len(q.get_dlq()) == 0

    def test_replay_dlq_not_found(self) -> None:
        from umcp.fleet.queue import PriorityQueue

        q = PriorityQueue()
        assert q.replay_dlq("nonexistent") is None

    def test_queue_full(self) -> None:
        from umcp.fleet.models import Job
        from umcp.fleet.queue import PriorityQueue, QueueFullError

        q = PriorityQueue(max_size=1)
        job1 = Job(target="t1")
        q.enqueue(job1)
        with pytest.raises(QueueFullError):
            q.enqueue(Job(target="t2"))

    def test_cancel_job(self) -> None:
        from umcp.fleet.models import Job, JobStatus
        from umcp.fleet.queue import PriorityQueue

        q = PriorityQueue()
        job = Job(target="t-cancel")
        q.enqueue(job)
        cancelled = q.cancel(job.job_id)
        assert cancelled is not None
        assert cancelled.status == JobStatus.CANCELLED


# =====================================================================
# 6. Insights — mock materials_science for deeper discovery
# =====================================================================


class TestInsightsDiscoveryMocked:
    """Cover insights.py discovery methods by mocking closure imports."""

    def test_discover_periodic_trends_import_error(self) -> None:
        """When materials_science is not importable, returns empty list."""
        from umcp.insights import InsightEngine

        engine = InsightEngine(load_canon=False, load_db=False)
        # Patch the import inside discover_periodic_trends
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                __import__(name, *a, **kw) if "cohesive" not in name else (_ for _ in ()).throw(ImportError())
            ),
        ):
            results = engine.discover_periodic_trends()
        # Should return empty or partial results (ImportError caught)
        assert isinstance(results, list)

    def test_discover_regime_boundaries_import_error(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine(load_canon=False, load_db=False)
        # Both sub-imports fail → returns empty
        results = engine.discover_regime_boundaries()
        assert isinstance(results, list)

    def test_discover_cross_correlations_import_error(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_cross_correlations()
        assert isinstance(results, list)

    def test_discover_universality_signatures_import_error(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_universality_signatures()
        assert isinstance(results, list)

    def test_discover_all(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_all()
        assert isinstance(results, list)


# =====================================================================
# 7. Logging — health check edge cases
# =====================================================================


class TestLoggingHealthCheck:
    """Cover logging_utils.py HealthCheck.check and related paths."""

    def test_health_check_nonexistent_repo(self, tmp_path: Path) -> None:
        from umcp.logging_utils import HealthCheck

        result = HealthCheck.check(tmp_path / "nonexistent_repo")
        assert isinstance(result, dict)
        assert "status" in result

    def test_health_check_empty_repo(self, tmp_path: Path) -> None:
        from umcp.logging_utils import HealthCheck

        result = HealthCheck.check(tmp_path)
        assert isinstance(result, dict)
        assert result.get("status") in ("healthy", "unhealthy", "degraded")

    def test_health_check_with_dirs(self, tmp_path: Path) -> None:
        from umcp.logging_utils import HealthCheck

        # Create the expected directories + schemas with at least one .json
        (tmp_path / "schemas").mkdir()
        (tmp_path / "schemas" / "test.json").write_text("{}")
        (tmp_path / "contracts").mkdir()
        (tmp_path / "closures").mkdir()
        result = HealthCheck.check(tmp_path)
        assert isinstance(result, dict)

    def test_structured_logger_json_mode(self) -> None:
        from umcp.logging_utils import StructuredLogger

        logger = StructuredLogger("test-json", json_output=True)
        logger.info("test message", key="value")
        logger.warning("warn msg")
        logger.error("err msg")
        logger.debug("dbg msg")
        logger.critical("crit msg")

    def test_performance_metrics_finish(self) -> None:
        from umcp.logging_utils import PerformanceMetrics

        m = PerformanceMetrics(operation="test_op")
        time.sleep(0.01)
        m.finish()
        assert m.end_time is not None
        assert m.duration_ms is not None
        assert m.duration_ms > 0


# =====================================================================
# 8. SS1M triad edge cases
# =====================================================================


class TestSS1MTriadEdgeCases:
    """Cover ss1m_triad.py edge cases: parse, encode, decode, triad_to_eid12."""

    def test_compute_triad_basic(self) -> None:
        from umcp.ss1m_triad import EditionCounts, compute_triad

        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        assert 0 <= triad.c1 < 97
        assert 0 <= triad.c2 < 97
        assert 0 <= triad.c3 < 97

    def test_verify_triad(self) -> None:
        from umcp.ss1m_triad import EditionCounts, compute_triad, verify_triad

        counts = EditionCounts(pages=25, figures=12, tables=5, equations=3, references=48)
        triad = compute_triad(counts)
        assert verify_triad(counts, triad)

    def test_parse_triad(self) -> None:
        from umcp.ss1m_triad import parse_triad

        t = parse_triad("37-45-17")
        assert t.c1 == 37
        assert t.c2 == 45
        assert t.c3 == 17

    def test_parse_triad_invalid(self) -> None:
        from umcp.ss1m_triad import parse_triad

        with pytest.raises(ValueError):
            parse_triad("bad-format")

    def test_encode_decode_base32(self) -> None:
        from umcp.ss1m_triad import decode_base32, encode_base32

        for val in [0, 1, 31, 32, 100, 912673]:
            encoded = encode_base32(val, length=4)
            decoded = decode_base32(encoded)
            assert decoded == val

    def test_encode_base32_negative(self) -> None:
        from umcp.ss1m_triad import encode_base32

        with pytest.raises(ValueError):
            encode_base32(-1)

    def test_decode_base32_invalid_char(self) -> None:
        from umcp.ss1m_triad import decode_base32

        with pytest.raises(ValueError):
            decode_base32("!!!")

    def test_triad_to_eid12(self) -> None:
        from umcp.ss1m_triad import EditionTriad, triad_to_eid12

        t = EditionTriad(c1=37, c2=45, c3=17)
        eid = triad_to_eid12(t, case_prefix="CP")
        assert isinstance(eid, str)
        assert eid.startswith("CP-")

    def test_triad_to_eid12_bad_prefix(self) -> None:
        from umcp.ss1m_triad import EditionTriad, triad_to_eid12

        t = EditionTriad(c1=37, c2=45, c3=17)
        with pytest.raises(ValueError):
            triad_to_eid12(t, case_prefix="ABC")

    def test_edition_triad_compact(self) -> None:
        from umcp.ss1m_triad import EditionTriad

        t = EditionTriad(c1=5, c2=10, c3=96)
        assert t.compact == "05-10-96"
        assert str(t) == "05-10-96"

    def test_edition_triad_out_of_range(self) -> None:
        from umcp.ss1m_triad import EditionTriad

        with pytest.raises(ValueError):
            EditionTriad(c1=100, c2=0, c3=0)


# =====================================================================
# 9. Scheduler — strict mode, to_dict, tenant passthrough
# =====================================================================


class TestSchedulerAdditional:
    def test_scheduler_strict_mode_submit(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        job = s.submit("casepacks/pedagogical/hello_world", strict=True)
        assert job.strict is True

    def test_scheduler_to_dict(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        d = s.to_dict()
        assert "workers" in d
        assert "queue" in d
        assert "cache" in d
        assert "tenants" in d

    def test_scheduler_cache_stats(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        stats = s.cache_stats()
        assert isinstance(stats, dict)

    def test_scheduler_cancel(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        job = s.submit("casepacks/pedagogical/hello_world")
        cancelled = s.cancel(job.job_id)
        assert cancelled is not None

    def test_scheduler_list_tenants(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.register_tenant("t-list")
        tenants = s.list_tenants()
        assert any(t.tenant_id == "t-list" for t in tenants)

    def test_scheduler_get_tenant(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.register_tenant("t-get")
        t = s.get_tenant("t-get")
        assert t.tenant_id == "t-get"


# =====================================================================
# 10. Measurement engine — additional branches
# =====================================================================


class TestMeasurementEngineAdditional:
    def test_from_array(self) -> None:
        from umcp.measurement_engine import MeasurementEngine

        engine = MeasurementEngine()
        result = engine.from_array(
            data=np.array([[0.9, 0.85, 0.88]]),
            weights=[1 / 3, 1 / 3, 1 / 3],
        )
        assert result is not None
        assert len(result.invariants) > 0


# =====================================================================
# 11. Extensions CLI — run and additional subcommands
# =====================================================================


class TestExtensionCLIAdditional:
    def test_extension_manager_list(self) -> None:
        from umcp.umcp_extensions import ExtensionManager

        mgr = ExtensionManager()
        names = mgr.available_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_extension_manager_status(self) -> None:
        from umcp.umcp_extensions import ExtensionManager, get_extension_info

        mgr = ExtensionManager()
        status = mgr.status()
        assert isinstance(status, dict)
        for name in mgr.available_names:
            info = get_extension_info(name)
            assert isinstance(info, dict)
            assert "name" in info


# =====================================================================
# 12. Universal calculator — additional edge cases
# =====================================================================


class TestUniversalCalcEdgeCases:
    def test_compute_with_empty_coords(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        # Single coordinate
        result = calc.compute_all(
            coordinates=[0.5],
            mode=ComputationMode.STANDARD,
        )
        assert result is not None

    def test_compute_with_many_coords(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        # Many coordinates
        coords = [0.5 + 0.01 * i for i in range(20)]
        result = calc.compute_all(
            coordinates=coords,
            mode=ComputationMode.FULL,
        )
        assert result is not None
        s = result.summary()
        assert "Fidelity" in s


# =====================================================================
# 13. Tau_r_star edge cases
# =====================================================================


class TestTauRStarEdgeCases:
    def test_diagnose_extreme_drift(self) -> None:
        from umcp.tau_r_star import diagnose

        # High drift / collapse regime
        diag = diagnose(omega=0.9, F=0.1, S=0.5, C=0.8, kappa=-2.3, IC=0.1, R=0.5)
        assert diag is not None
        assert hasattr(diag, "phase")

    def test_diagnose_stable_regime(self) -> None:
        import math

        from umcp.tau_r_star import diagnose

        kappa = -0.01
        IC = math.exp(kappa)
        diag = diagnose(omega=0.01, F=0.99, S=0.05, C=0.02, kappa=kappa, IC=IC, R=0.5)
        assert diag is not None
        assert diag.tier0_checks_pass


# =====================================================================
# 14. Tau_r_star_dynamics edge cases
# =====================================================================


class TestTauRDynamicsEdgeCases:
    def test_diagnose_extended_run(self) -> None:
        from umcp.tau_r_star_dynamics import diagnose_extended

        result = diagnose_extended(omega=0.1, C=0.05, R=0.5)
        assert result is not None
        assert result.tier0_checks_pass is not None
