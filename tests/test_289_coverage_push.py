"""Coverage push tests — target remaining uncovered lines to reach 93%.

Focuses on:
- universal_calculator.py: summary(), to_json(), FULL mode, RCFT mode, main() CLI
- umcp_extensions.py: run_extension, _do_load, main CLI paths
- fleet/scheduler.py: start/stop reaper, report_result, cache hit path, tenant ops
- fleet/worker.py: _run_validation, to_dict
- measurement_engine.py: from_dataframe
- insights.py: discover methods
- validator.py: uncovered paths
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =====================================================================
# UniversalCalculator — summary, to_json, FULL, RCFT modes
# =====================================================================


class TestUniversalCalculatorSummary:
    """Cover the summary() text rendering branches."""

    def test_full_mode_generates_gcd_section(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.88],
            mode=ComputationMode.FULL,
        )
        s = result.summary()
        assert "GCD Metrics" in s
        assert "E_potential" in s

    def test_full_mode_to_json(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.88],
            mode=ComputationMode.FULL,
        )
        j = result.to_json()
        data = json.loads(j)
        assert "kernel" in data
        assert "gcd" in data

    def test_full_mode_with_coord_variances(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.88],
            coord_variances=[0.001, 0.002, 0.001],
            mode=ComputationMode.FULL,
        )
        s = result.summary()
        assert "Uncertainty" in s
        assert "σ_F" in s

    def test_rcft_mode_generates_rcft_section(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        traj = np.column_stack([np.linspace(0.8, 0.3, 20), np.linspace(0.9, 0.4, 20), np.linspace(0.85, 0.35, 20)])
        result = calc.compute_all(
            coordinates=[0.5, 0.6, 0.55],
            trajectory=traj,
            mode=ComputationMode.RCFT,
        )
        s = result.summary()
        assert "RCFT Metrics" in s
        assert "D_fractal" in s

    def test_summary_costs_and_seam(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.88],
            prior_kappa=-0.15,
            prior_IC=0.86,
            mode=ComputationMode.STANDARD,
        )
        s = result.summary()
        assert "Cost Closures" in s
        assert "Seam Accounting" in s

    def test_full_mode_to_dict_has_all_sections(self) -> None:
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.88],
            coord_variances=[0.001, 0.002, 0.001],
            prior_kappa=-0.15,
            prior_IC=0.86,
            mode=ComputationMode.FULL,
        )
        d = result.to_dict()
        assert "costs" in d
        assert "seam" in d
        assert "gcd" in d
        assert "uncertainty" in d

    def test_main_cli_standard(self) -> None:
        """Cover the main() CLI entry point."""
        from umcp.universal_calculator import main

        with patch("sys.argv", ["calc", "-c", "0.9,0.85,0.88"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
            output = captured.getvalue()
            assert "Fidelity" in output

    def test_main_cli_json(self) -> None:
        from umcp.universal_calculator import main

        with patch("sys.argv", ["calc", "-c", "0.9,0.85,0.88", "--json"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
            data = json.loads(captured.getvalue())
            assert "kernel" in data

    def test_full_mode_high_energy_regime(self) -> None:
        """High curvature triggers 'High' energy regime."""
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.1, 0.95, 0.1, 0.95],
            mode=ComputationMode.FULL,
        )
        assert result.gcd is not None

    def test_rcft_mode_single_point(self) -> None:
        """RCFT mode without trajectory uses single-point estimate."""
        from umcp.universal_calculator import ComputationMode, UniversalCalculator

        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.5, 0.6, 0.55],
            mode=ComputationMode.RCFT,
        )
        assert result.rcft is not None
        assert result.rcft.D_fractal >= 1.0


# =====================================================================
# Extensions — run_extension, _do_load, reset
# =====================================================================


class TestExtensionsRemainingPaths:
    def test_run_extension_not_found(self) -> None:
        from umcp.umcp_extensions import run_extension

        assert run_extension("nonexistent_ext_99") == 1

    def test_do_load_nonexistent(self) -> None:
        from umcp.umcp_extensions import manager

        result = manager._do_load("nonexistent_ext_99")
        assert result is None

    def test_manager_reset(self) -> None:
        from umcp.umcp_extensions import manager

        manager.reset()
        assert not manager._started

    def test_run_extension_no_command(self) -> None:
        """Cover the 'no run command' path."""
        from umcp.umcp_extensions import EXTENSIONS, run_extension

        # Temporarily add an extension with no command
        original = dict(EXTENSIONS)
        try:
            from umcp.umcp_extensions import ExtensionInfo

            EXTENSIONS["_test_nocommand"] = ExtensionInfo(
                name="_test_nocommand",
                module="umcp",
                description="test",
                type="test",
                command="",
                requires=[],
            )
            result = run_extension("_test_nocommand")
            assert result == 1
        finally:
            EXTENSIONS.clear()
            EXTENSIONS.update(original)

    def test_print_status_table(self) -> None:
        from umcp.umcp_extensions import _print_status_table

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _print_status_table()
        output = captured.getvalue()
        assert "Extension" in output or "extension" in output.lower()


# =====================================================================
# Scheduler — start/stop reaper, register_tenant paths
# =====================================================================


class TestSchedulerAdvanced:
    def test_start_and_stop_reaper(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.start()
        assert s._reaper_thread is not None
        assert s._reaper_thread.is_alive()
        s.stop()
        # Should not raise on double stop
        s.stop()

    def test_start_idempotent(self) -> None:
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        s.start()
        thread1 = s._reaper_thread
        s.start()  # Should not create new thread
        assert s._reaper_thread is thread1
        s.stop()

    def test_report_result_conformant(self) -> None:
        """Cover report_result with a CONFORMANT result → cache store path."""
        from umcp.fleet.models import JobResult, ValidationVerdict, WorkerInfo
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        _job = s.submit("casepacks/pedagogical/hello_world")
        info = WorkerInfo(worker_id="w1", capacity=2)
        s.register_worker(info)
        polled = s.poll("w1")
        assert polled is not None

        result = JobResult(
            verdict=ValidationVerdict.CONFORMANT,
            exit_code=0,
            report={"status": "ok"},
            duration_s=1.0,
        )
        s.report_result(polled.job_id, result)

    def test_submit_auto_registers_tenant(self) -> None:
        """Cover the auto-register path for unknown tenants."""
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        # Submit with a tenant_id that isn't registered yet
        job = s.submit("casepacks/pedagogical/hello_world", tenant_id="auto_tenant")
        assert job.tenant_id == "auto_tenant"

    def test_wait_timeout(self) -> None:
        """Cover the wait() timeout path."""
        from umcp.fleet.scheduler import Scheduler

        s = Scheduler()
        job = s.submit("casepacks/pedagogical/hello_world")
        result = s.wait(job.job_id, timeout=0.1, poll_interval=0.05)
        # Should timeout since no worker is processing
        assert result is None


# =====================================================================
# Worker — _run_validation, to_dict
# =====================================================================


class TestWorkerInternals:
    def test_worker_to_dict(self) -> None:
        from umcp.fleet.worker import Worker

        w = Worker("test-w1", scheduler=MagicMock())
        d = w.to_dict()
        assert d["worker_id"] == "test-w1"
        assert "config" in d
        assert "info" in d
        assert d["is_running"] is False

    def test_run_validation_timeout(self) -> None:
        """Cover the subprocess timeout path in _run_validation."""
        import subprocess

        from umcp.fleet.models import Job, ValidationVerdict
        from umcp.fleet.worker import Worker, WorkerConfig

        config = WorkerConfig(validate_timeout_s=0.001)
        w = Worker("test-w2", scheduler=MagicMock(), config=config)

        job = Job(target="nonexistent_casepack_xyz")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 0.001)):
            result = w._run_validation(job)
        assert result.verdict == ValidationVerdict.NON_EVALUABLE
        assert result.exit_code == 124

    def test_run_validation_success(self) -> None:
        """Cover the subprocess success path."""
        from umcp.fleet.models import Job, ValidationVerdict
        from umcp.fleet.worker import Worker

        w = Worker("test-w3", scheduler=MagicMock())
        job = Job(target="casepacks/pedagogical/hello_world")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"status": "ok"}'
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = w._run_validation(job)
        assert result.verdict == ValidationVerdict.CONFORMANT

    def test_run_validation_nonconformant(self) -> None:
        """Cover returncode == 1 path."""
        from umcp.fleet.models import Job, ValidationVerdict
        from umcp.fleet.worker import Worker

        w = Worker("test-w4", scheduler=MagicMock())
        job = Job(target="casepacks/pedagogical/hello_world")

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "{}"
        mock_proc.stderr = "Error: validation failed\n"
        with patch("subprocess.run", return_value=mock_proc):
            result = w._run_validation(job)
        assert result.verdict == ValidationVerdict.NONCONFORMANT
        assert len(result.errors) > 0

    def test_run_validation_unknown_exit(self) -> None:
        """Cover returncode > 1 (NON_EVALUABLE) path."""
        from umcp.fleet.models import Job, ValidationVerdict
        from umcp.fleet.worker import Worker

        w = Worker("test-w5", scheduler=MagicMock())
        job = Job(target="casepacks/pedagogical/hello_world")

        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_proc.stdout = "not json"
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = w._run_validation(job)
        assert result.verdict == ValidationVerdict.NON_EVALUABLE


# =====================================================================
# MeasurementEngine — from_dataframe
# =====================================================================


class TestMeasurementEngineDataframe:
    def test_from_dataframe_with_t_column(self) -> None:
        """Cover the from_dataframe path with time column."""
        pd = pytest.importorskip("pandas")
        from umcp.measurement_engine import MeasurementEngine

        df = pd.DataFrame(
            {
                "t": [0, 1, 2, 3],
                "x": [0.9, 0.85, 0.80, 0.75],
                "y": [0.8, 0.82, 0.84, 0.86],
            }
        )
        engine = MeasurementEngine()
        result = engine.from_dataframe(df)
        assert len(result.invariants) > 0

    def test_from_dataframe_without_t_column(self) -> None:
        """Cover the from_dataframe path without time column."""
        pd = pytest.importorskip("pandas")
        from umcp.measurement_engine import MeasurementEngine

        df = pd.DataFrame(
            {
                "x": [0.9, 0.85, 0.80, 0.75],
                "y": [0.8, 0.82, 0.84, 0.86],
            }
        )
        engine = MeasurementEngine()
        result = engine.from_dataframe(df)
        assert len(result.invariants) > 0

    def test_from_dataframe_type_error(self) -> None:
        """Cover the TypeError path."""
        pytest.importorskip("pandas")
        from umcp.measurement_engine import MeasurementEngine

        engine = MeasurementEngine()
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            engine.from_dataframe("not a dataframe")  # type: ignore[arg-type]


# =====================================================================
# Validator — uncovered minor paths
# =====================================================================


class TestValidatorPaths:
    def test_validator_on_real_repo(self) -> None:
        """Trigger the RootFileValidator.validate_all on the real repo root."""
        from umcp.validator import RootFileValidator

        v = RootFileValidator(Path("/workspaces/GENERATIVE-COLLAPSE-DYNAMICS"))
        result = v.validate_all()
        assert isinstance(result, dict)
        assert "status" in result

    def test_validator_on_nonexistent(self) -> None:
        """Trigger the RootFileValidator on a nonexistent path."""
        from umcp.validator import RootFileValidator

        v = RootFileValidator(Path("/tmp/nonexistent_repo_xyz"))
        result = v.validate_all()
        assert isinstance(result, dict)


# =====================================================================
# Insights — startup insight, pattern discovery
# =====================================================================


class TestInsightsDiscovery:
    def test_startup_insight(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine()
        startup = engine.show_startup_insight()
        assert isinstance(startup, str)

    def test_discover_cross_correlations(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine()
        results = engine.discover_cross_correlations()
        assert isinstance(results, list)

    def test_discover_universality(self) -> None:
        from umcp.insights import InsightEngine

        engine = InsightEngine()
        results = engine.discover_universality_signatures()
        assert isinstance(results, list)
