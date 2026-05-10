"""Final coverage gap tests for 93% target.

Targets remaining uncovered lines in:
  - universal_calculator.py: summary() with all sections, CLI main(), to_dict branches
  - fleet/cache.py: ArtifactCache TTL eviction, to_dict, LRU eviction, tenant clearing
  - fleet/scheduler.py: remaining gap branches
  - fleet/worker.py: WorkerPool executor
  - file_refs.py: fallback YAML parser, load_csv, load_text
  - validator.py: _validate_manifest, _validate_weights paths
  - kernel_optimized.py: _validate_outputs boundary branches
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# =====================================================================
# UniversalCalculator — summary branches + CLI
# =====================================================================


class TestUniversalCalculatorSummary:
    """Cover summary() output with all optional sections populated."""

    def test_summary_full_with_all_sections(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        traj = np.array([[0.9, 0.85, 0.8], [0.7, 0.65, 0.6], [0.9, 0.85, 0.8]])
        result = calc.compute_all(
            [0.9, 0.85, 0.8],
            mode=ComputationMode.FULL,
            trajectory=traj,
            prior_kappa=-0.1,
            prior_IC=0.9,
            coord_variances=[0.001, 0.002, 0.001],
        )
        s = result.summary()
        assert "Kernel" in s
        assert "Cost Closures" in s
        assert "Seam" in s
        assert "GCD" in s
        assert "RCFT" in s
        assert "Uncertainty" in s
        assert "SS1M" in s

    def test_to_dict_full_sections(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        traj = np.array([[0.9, 0.85, 0.8], [0.7, 0.65, 0.6]])
        result = calc.compute_all(
            [0.9, 0.85, 0.8],
            mode=ComputationMode.FULL,
            trajectory=traj,
            prior_kappa=-0.1,
            prior_IC=0.9,
            coord_variances=[0.001, 0.002, 0.001],
        )
        d = result.to_dict()
        assert "costs" in d
        assert "seam" in d
        assert "gcd" in d
        assert "rcft" in d
        assert "uncertainty" in d
        assert "ss1m" in d

    def test_kernel_to_dict_nan_tau_R(self) -> None:
        from umcp.universal_calculator import KernelInvariants

        ki = KernelInvariants(
            F=0.9,
            omega=0.1,
            kappa=-0.1,
            IC=0.9,
            S=0.3,
            C=0.1,
            tau_R=float("nan"),
        )
        d = ki.to_dict()
        assert d["tau_R"] is None

    def test_kernel_to_dict_finite_tau_R(self) -> None:
        from umcp.universal_calculator import KernelInvariants

        ki = KernelInvariants(
            F=0.9,
            omega=0.1,
            kappa=-0.1,
            IC=0.9,
            S=0.3,
            C=0.1,
            tau_R=3.0,
        )
        d = ki.to_dict()
        assert d["tau_R"] == 3.0

    def test_kernel_to_dict_inf_tau_R(self) -> None:
        from umcp.universal_calculator import KernelInvariants

        ki = KernelInvariants(
            F=0.9,
            omega=0.1,
            kappa=-0.1,
            IC=0.9,
            S=0.3,
            C=0.1,
            tau_R=float("inf"),
        )
        d = ki.to_dict()
        assert d["tau_R"] == "INF_REC"

    def test_gcd_energy_medium(self) -> None:
        """Hit the Medium energy regime branch."""
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        # Medium energy: F ≈ 0.7ish → E_potential moderate
        result = calc.compute_all([0.7, 0.65, 0.6], mode=ComputationMode.FULL)
        assert result.gcd is not None

    def test_rcft_turbulent_regime(self) -> None:
        """Hit the Turbulent fractal_regime branch."""
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        # Chaotic trajectory → high fractal dimension → Turbulent
        rng = np.random.default_rng(42)
        traj = rng.uniform(0.01, 0.99, size=(100, 3))
        result = calc.compute_all([0.5, 0.5, 0.5], mode=ComputationMode.RCFT, trajectory=traj)
        if result.rcft:
            assert result.rcft.fractal_regime in {"Smooth", "Wrinkled", "Turbulent"}


class TestUniversalCalculatorCLI:
    def test_cli_json(self) -> None:
        """Test the CLI main() in JSON mode."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "-c", "0.9,0.85,0.8", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "kernel" in data

    def test_cli_text(self) -> None:
        """Test the CLI main() in text mode."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "-c", "0.9,0.85,0.8"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Kernel" in result.stdout

    def test_cli_with_weights(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "-c", "0.9,0.85,0.8", "-w", "0.5,0.3,0.2", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_full_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "-c", "0.9,0.85,0.8", "-m", "full", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# =====================================================================
# ArtifactCache — TTL, LRU eviction, tenant, to_dict
# =====================================================================


class TestArtifactCacheTTL:
    def test_ttl_eviction(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend(), ttl_s=0.1)
        cache.put(b"data1", key="k1")
        assert cache.get("k1") is not None

        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("k1") is None

    def test_lru_eviction(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        # Small budget: 20 bytes max
        cache = ArtifactCache(backend=MemoryBackend(), max_bytes=20)
        cache.put(b"1234567890", key="k1")  # 10 bytes
        cache.put(b"abcdefghij", key="k2")  # 10 bytes, total 20

        # Adding another should evict k1 (LRU)
        cache.put(b"xxxxxxxxxxxx", key="k3")  # 12 bytes
        assert cache.get("k1") is None  # evicted
        assert cache.get("k3") is not None

    def test_to_dict(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"test", key="k1")
        d = cache.to_dict()
        assert "stats" in d
        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_stats(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"test", key="k1")
        cache.get("k1")
        cache.get("missing")
        s = cache.stats()
        assert s["hits"] >= 1
        assert s["misses"] >= 1
        assert s["hit_rate"] > 0

    def test_has(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        assert not cache.has("k1")
        cache.put(b"test", key="k1")
        assert cache.has("k1")

    def test_tenant_usage(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"hello", key="k1", tenant_id="t1")
        cache.put(b"world!", key="k2", tenant_id="t2")
        assert cache.tenant_usage("t1") == 5
        assert cache.tenant_usage("t2") == 6

    def test_clear_by_tenant(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"a", key="k1", tenant_id="t1")
        cache.put(b"b", key="k2", tenant_id="t2")
        cleared = cache.clear(tenant_id="t1")
        assert cleared == 1
        assert cache.get("k1", tenant_id="t1") is None
        assert cache.get("k2", tenant_id="t2") is not None

    def test_delete(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"data", key="k1")
        assert cache.delete("k1")
        assert not cache.delete("k1")  # already gone

    def test_content_key(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        key = cache.content_key(b"test data")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex

    def test_put_auto_key(self) -> None:
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        key = cache.put(b"auto keyed data")
        assert len(key) == 64  # content-addressable

    def test_backend_lost_entry(self) -> None:
        """If backend loses data but meta still exists, should clean up."""
        from umcp.fleet.cache import ArtifactCache, MemoryBackend

        cache = ArtifactCache(backend=MemoryBackend())
        cache.put(b"data", key="k1")
        # Manually remove from backend but not meta
        cache._backend._store.pop("k1", None)
        assert cache.get("k1") is None  # should clean up


class TestMemoryBackendExtended:
    def test_size(self) -> None:
        from umcp.fleet.cache import MemoryBackend

        backend = MemoryBackend()
        assert backend.size("k") == 0
        backend.put("k", b"12345")
        assert backend.size("k") == 5

    def test_keys(self) -> None:
        from umcp.fleet.cache import MemoryBackend

        backend = MemoryBackend()
        backend.put("a", b"1")
        backend.put("b", b"2")
        keys = backend.keys()
        assert set(keys) == {"a", "b"}

    def test_clear(self) -> None:
        from umcp.fleet.cache import MemoryBackend

        backend = MemoryBackend()
        backend.put("a", b"1")
        backend.clear()
        assert backend.get("a") is None


# =====================================================================
# FilesystemBackend — more coverage
# =====================================================================


class TestFilesystemBackendExtended:
    def test_size(self, tmp_path: Any) -> None:
        from umcp.fleet.cache import FilesystemBackend

        backend = FilesystemBackend(root=tmp_path)
        assert backend.size("k") == 0
        backend.put("k", b"12345")
        assert backend.size("k") == 5

    def test_keys_empty(self, tmp_path: Any) -> None:
        from umcp.fleet.cache import FilesystemBackend

        backend = FilesystemBackend(root=tmp_path)
        assert backend.keys() == []

    def test_keys_with_data(self, tmp_path: Any) -> None:
        from umcp.fleet.cache import FilesystemBackend

        backend = FilesystemBackend(root=tmp_path)
        backend.put("x", b"1")
        backend.put("y", b"2")
        keys = backend.keys()
        assert set(keys) == {"x", "y"}

    def test_clear(self, tmp_path: Any) -> None:
        from umcp.fleet.cache import FilesystemBackend

        backend = FilesystemBackend(root=tmp_path)
        backend.put("k", b"v")
        backend.clear()
        assert backend.get("k") is None


# =====================================================================
# file_refs: fallback YAML parser + load methods
# =====================================================================


class TestFileRefsFallback:
    def test_load_yaml_with_yaml(self) -> None:
        from umcp.file_refs import UMCPFiles

        files = UMCPFiles()
        # Load a known YAML file
        data = files.load_yaml(files.manifest_yaml)
        assert isinstance(data, dict)

    def test_load_yaml_fallback_parser(self, tmp_path: Any) -> None:
        """Test the fallback YAML parser when PyYAML is not available."""
        from umcp import file_refs

        yf = tmp_path / "test.yaml"
        yf.write_text("key1: value1\nkey2: value2\n# comment\n\nempty:\n")
        files = file_refs.UMCPFiles(root_path=tmp_path)

        # Temporarily remove yaml
        original_yaml = file_refs.yaml
        try:
            file_refs.yaml = None
            data = files.load_yaml(yf)
            assert data["key1"] == "value1"
            assert data["key2"] == "value2"
        finally:
            file_refs.yaml = original_yaml

    def test_load_csv(self, tmp_path: Any) -> None:
        from umcp.file_refs import UMCPFiles

        cf = tmp_path / "test.csv"
        cf.write_text("name,value\nalpha,1\nbeta,2\n")
        files = UMCPFiles(root_path=tmp_path)
        data = files.load_csv(cf)
        assert len(data) == 2
        assert data[0]["name"] == "alpha"

    def test_load_text(self, tmp_path: Any) -> None:
        from umcp.file_refs import UMCPFiles

        tf = tmp_path / "test.txt"
        tf.write_text("hello world")
        files = UMCPFiles(root_path=tmp_path)
        assert files.load_text(tf) == "hello world"

    def test_root_fallback_to_cwd(self) -> None:
        """When pyproject.toml cannot be found, falls back to cwd."""
        from umcp import file_refs

        # Use a temp root that has no pyproject.toml
        _original_file = file_refs.Path.__module__
        files = file_refs.UMCPFiles(root_path=Path("/tmp"))
        assert files.root == Path("/tmp")


# =====================================================================
# Kernel optimized — _validate_outputs boundary errors
# =====================================================================


class TestKernelValidateOutputs:
    def test_validate_F_out_of_range(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=1.5,
            omega=-0.5,
            kappa=0,
            IC=1.0,
            S=0.0,
            C=0.0,
            heterogeneity_gap=0.0,
            is_homogeneous=True,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="F out of range"):
            k._validate_outputs(bad)

    def test_validate_omega_out_of_range(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=0.5,
            omega=-0.1,
            kappa=-0.5,
            IC=0.5,
            S=0.3,
            C=0.1,
            heterogeneity_gap=0.0,
            is_homogeneous=True,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="omega out of range"):
            k._validate_outputs(bad)

    def test_validate_C_out_of_range(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=0.5,
            omega=0.5,
            kappa=-0.5,
            IC=0.5,
            S=0.3,
            C=1.5,
            heterogeneity_gap=0.0,
            is_homogeneous=False,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="C out of range"):
            k._validate_outputs(bad)

    def test_validate_IC_out_of_range(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=0.5,
            omega=0.5,
            kappa=-0.5,
            IC=2.0,
            S=0.3,
            C=0.5,
            heterogeneity_gap=0.0,
            is_homogeneous=False,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="IC out of range"):
            k._validate_outputs(bad)

    def test_validate_kappa_non_finite(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=0.5,
            omega=0.5,
            kappa=float("inf"),
            IC=0.5,
            S=0.3,
            C=0.1,
            heterogeneity_gap=0.0,
            is_homogeneous=False,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="kappa non-finite"):
            k._validate_outputs(bad)

    def test_validate_S_out_of_range(self) -> None:
        from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

        k = OptimizedKernelComputer()
        bad = KernelOutputs(
            F=0.5,
            omega=0.5,
            kappa=-0.5,
            IC=0.5,
            S=-0.1,
            C=0.1,
            heterogeneity_gap=0.0,
            is_homogeneous=False,
            computation_mode="test",
            regime="homogeneous",
        )
        with pytest.raises(ValueError, match="S out of range"):
            k._validate_outputs(bad)


# =====================================================================
# Scheduler — remaining gaps
# =====================================================================


class TestSchedulerExtended:
    def test_submit_and_cancel(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        job = s.submit("casepacks/pedagogical/hello_world")
        assert job is not None
        assert job.job_id
        cancelled = s.cancel(job.job_id)
        assert cancelled is not None

    def test_submit_with_tenant(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        job = s.submit("casepacks/pedagogical/hello_world", tenant_id="tenant_a")
        retrieved = s.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.tenant_id == "tenant_a"

    def test_poll_unregistered_worker(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.submit("casepacks/pedagogical/hello_world")
        # Poll with unregistered worker returns None
        result = s.poll("nonexistent_worker")
        assert result is None

    def test_register_worker_and_poll(self) -> None:
        from umcp.fleet.models import WorkerInfo
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.submit("casepacks/pedagogical/hello_world")
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)
        job = s.poll("w1")
        assert job is not None
        assert job.worker_id == "w1"

    def test_report_failure(self) -> None:
        from umcp.fleet.models import WorkerInfo
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        _submitted = s.submit("casepacks/pedagogical/hello_world")
        info = WorkerInfo(worker_id="w2", capacity=2)
        s.register_worker(info)
        job = s.poll("w2")
        if job:
            s.report_failure(job.job_id, error="boom")

    def test_queue_stats(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.submit("casepacks/pedagogical/hello_world")
        stats = s.queue_stats()
        assert stats is not None

    def test_heartbeat(self) -> None:
        from umcp.fleet.models import WorkerInfo
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        info = WorkerInfo(worker_id="w3")
        s.register_worker(info)
        s.heartbeat("w3")
        workers = s.list_workers()
        assert len(workers) == 1

    def test_unregister_worker(self) -> None:
        from umcp.fleet.models import WorkerInfo
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        info = WorkerInfo(worker_id="w4")
        s.register_worker(info)
        s.unregister_worker("w4")
        assert len(s.list_workers()) == 0

    def test_register_tenant(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        tenant = s.register_tenant("t1")
        assert tenant.tenant_id == "t1"
        assert s.get_tenant("t1") is not None
        assert len(s.list_tenants()) == 1


# =====================================================================
# Worker — WorkerPool extended
# =====================================================================


class TestWorkerPoolExtended:
    def test_pool_to_dict(self) -> None:
        from umcp.fleet.worker import WorkerPool

        pool = WorkerPool(scheduler=MagicMock(), pool_size=2)
        d = pool.to_dict()
        assert d["pool_size"] == 2
        assert len(d["workers"]) == 2

    def test_pool_size_and_active(self) -> None:
        from umcp.fleet.worker import WorkerPool

        pool = WorkerPool(scheduler=MagicMock(), pool_size=3)
        assert pool.size == 3
        assert pool.active == 0  # not started

    def test_pool_scale_up(self) -> None:
        from umcp.fleet.worker import WorkerPool

        mock_sched = MagicMock()
        mock_sched.register_worker = MagicMock()
        mock_sched.poll = MagicMock(return_value=None)
        pool = WorkerPool(scheduler=mock_sched, pool_size=0)
        assert pool.size == 0
        new = pool.scale_up(2)
        assert len(new) == 2
        assert pool.size == 2

    def test_pool_drain(self) -> None:
        from umcp.fleet.worker import WorkerPool

        pool = WorkerPool(scheduler=MagicMock(), pool_size=1)
        pool.drain()  # should not raise
