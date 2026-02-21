"""
Tests for top-level closure modules.

Covers: F_from_omega, hello_world, stiffness_check, tau_R_compute, tau_R_optimized
Previously untested — these closures had zero dedicated unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/F_from_omega.py
# ═══════════════════════════════════════════════════════════════════


class TestFFromOmega:
    """Unit tests for F_from_omega.compute (centripetal force)."""

    def test_basic_computation(self):
        from closures.F_from_omega import compute

        result = compute(10.0, 0.5, 1.0)
        assert "F" in result
        assert result["F"] == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "omega,r,m,expected_F",
        [
            (1.0, 1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0, 4.0),
            (0.0, 1.0, 1.0, 0.0),
            (10.0, 2.0, 3.0, 600.0),
            (5.0, 0.1, 0.5, 1.25),
        ],
    )
    def test_parametrized_values(self, omega, r, m, expected_F):
        from closures.F_from_omega import compute

        result = compute(omega, r, m)
        assert result["F"] == pytest.approx(expected_F)

    def test_returns_dict(self):
        from closures.F_from_omega import compute

        result = compute(1.0, 1.0, 1.0)
        assert isinstance(result, dict)

    def test_zero_mass(self):
        from closures.F_from_omega import compute

        result = compute(10.0, 1.0, 0.0)
        assert result["F"] == 0.0

    def test_zero_radius(self):
        from closures.F_from_omega import compute

        result = compute(10.0, 0.0, 1.0)
        assert result["F"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# closures/hello_world.py
# ═══════════════════════════════════════════════════════════════════


class TestHelloWorld:
    """Unit tests for hello_world.compute."""

    def test_basic_computation(self):
        from closures.hello_world import compute

        result = compute(10.0)
        assert "F" in result
        assert result["F"] == 10.0

    def test_returns_dict(self):
        from closures.hello_world import compute

        assert isinstance(compute(1.0), dict)

    @pytest.mark.parametrize("omega", [0.0, 0.5, 1.0, 100.0])
    def test_identity_mapping(self, omega):
        from closures.hello_world import compute

        assert compute(omega)["F"] == omega


# ═══════════════════════════════════════════════════════════════════
# closures/stiffness_check.py
# ═══════════════════════════════════════════════════════════════════


class TestStiffnessCheck:
    """Unit tests for stiffness_check.compute."""

    def test_valid_stiffness(self):
        from closures.stiffness_check import compute

        result = compute(1000.0)
        assert result["valid"] is True

    def test_zero_stiffness_invalid(self):
        from closures.stiffness_check import compute

        result = compute(0.0)
        assert result["valid"] is False

    def test_negative_stiffness_invalid(self):
        from closures.stiffness_check import compute

        result = compute(-100.0)
        assert result["valid"] is False

    def test_huge_stiffness_invalid(self):
        from closures.stiffness_check import compute

        result = compute(1e13)
        assert result["valid"] is False

    @pytest.mark.parametrize(
        "kappa,expected",
        [
            (0.001, True),
            (1.0, True),
            (1e6, True),
            (1e11, True),
            (1e12, False),  # boundary: not < 1e12
            (0.0, False),
            (-1.0, False),
        ],
    )
    def test_boundary_values(self, kappa, expected):
        from closures.stiffness_check import compute

        assert compute(kappa)["valid"] is expected

    def test_returns_dict(self):
        from closures.stiffness_check import compute

        assert isinstance(compute(1.0), dict)


# ═══════════════════════════════════════════════════════════════════
# closures/tau_R_compute.py
# ═══════════════════════════════════════════════════════════════════


class TestTauRCompute:
    """Unit tests for tau_R_compute.compute."""

    def test_basic_computation(self):
        from closures.tau_R_compute import compute

        result = compute(10.0, 0.1)
        assert result["tau_R"] == pytest.approx(1.0)

    def test_raises_on_zero_omega(self):
        from closures.tau_R_compute import compute

        with pytest.raises(ValueError, match="positive"):
            compute(0.0, 0.1)

    def test_raises_on_zero_damping(self):
        from closures.tau_R_compute import compute

        with pytest.raises(ValueError, match="positive"):
            compute(10.0, 0.0)

    def test_raises_on_negative_omega(self):
        from closures.tau_R_compute import compute

        with pytest.raises(ValueError, match="positive"):
            compute(-1.0, 0.1)

    @pytest.mark.parametrize(
        "omega,damping,expected",
        [
            (1.0, 1.0, 1.0),
            (2.0, 0.5, 1.0),
            (100.0, 0.01, 1.0),
            (5.0, 0.2, 1.0),
        ],
    )
    def test_inverse_relationship(self, omega, damping, expected):
        from closures.tau_R_compute import compute

        assert compute(omega, damping)["tau_R"] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════
# closures/tau_R_optimized.py
# ═══════════════════════════════════════════════════════════════════


class TestTauROptimized:
    """Unit tests for tau_R_optimized: ReturnResult, OptimizedReturnComputer."""

    def test_return_result_dataclass(self):
        from closures.tau_R_optimized import ReturnResult

        r = ReturnResult(
            tau_R=5.0,
            reference_index=3,
            distance=0.01,
            margin=0.09,
            is_stable=True,
            computation_mode="full_search",
            candidates_checked=10,
        )
        assert r.tau_R == 5.0
        assert r.is_stable is True
        assert r.candidates_checked == 10

    def test_optimized_return_computer_init(self):
        from closures.tau_R_optimized import OptimizedReturnComputer

        comp = OptimizedReturnComputer(eta=0.1, H_rec=64)
        assert comp.eta == 0.1
        assert comp.H_rec == 64
        assert comp.norm_type == "l2"

    def test_compute_tau_R_empty_domain(self):
        from closures.tau_R_optimized import INF_REC, OptimizedReturnComputer

        comp = OptimizedReturnComputer(eta=0.1, H_rec=64)
        psi_t = np.array([0.5, 0.5])
        trace = np.array([[0.5, 0.5]])
        result = comp.compute_tau_R(psi_t, trace, t=0)
        assert result.tau_R == INF_REC
        assert result.reference_index is None

    def test_compute_tau_R_with_return(self):
        from closures.tau_R_optimized import OptimizedReturnComputer

        comp = OptimizedReturnComputer(eta=0.2, H_rec=64)
        # Trace where t=5 returns to t=0
        trace = np.zeros((6, 2))
        trace[0] = [0.5, 0.5]
        trace[1] = [0.8, 0.2]
        trace[2] = [0.9, 0.1]
        trace[3] = [0.7, 0.3]
        trace[4] = [0.6, 0.4]
        trace[5] = [0.5, 0.5]  # returns to t=0
        result = comp.compute_tau_R(trace[5], trace, t=5)
        assert result.tau_R < float("inf")
        assert result.reference_index is not None

    def test_compute_tau_R_no_return(self):
        from closures.tau_R_optimized import INF_REC, OptimizedReturnComputer

        comp = OptimizedReturnComputer(eta=0.001, H_rec=64)
        # Trace diverges monotonically
        trace = np.array([[i * 0.1, 1 - i * 0.1] for i in range(10)])
        result = comp.compute_tau_R(trace[9], trace, t=9)
        assert result.tau_R == INF_REC

    def test_norm_types(self):
        from closures.tau_R_optimized import OptimizedReturnComputer

        for norm in ["l2", "l1", "linf"]:
            comp = OptimizedReturnComputer(eta=0.1, H_rec=64, norm_type=norm)  # type: ignore[arg-type]
            assert comp.norm_type == norm

    def test_compute_function_exists(self):
        from closures.tau_R_optimized import compute_tau_R_optimized

        assert callable(compute_tau_R_optimized)
