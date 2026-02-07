"""
Tests for kernel_optimized.py - Optimized kernel computation.

These tests verify the optimized kernel implementation against
the formal specifications in KERNEL_SPECIFICATION.md.
"""

import numpy as np
import pytest
from umcp.kernel_optimized import (
    ErrorBounds,
    KernelOutputs,
    OptimizedKernelComputer,
    compute_kernel_outputs,
)


class TestOptimizedKernelComputer:
    """Tests for the OptimizedKernelComputer class."""

    def test_compute_basic(self):
        """Basic kernel computation works."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert isinstance(result, KernelOutputs)
        assert 0 <= result.F <= 1
        assert 0 <= result.omega <= 1
        assert result.IC > 0

    def test_fidelity_uniform_weights(self):
        """F = arithmetic mean with uniform weights."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert abs(result.F - 0.6) < 1e-10

    def test_drift_identity(self):
        """omega = 1 - F per Lemma 6."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert abs(result.F + result.omega - 1.0) < 1e-10

    def test_am_gm_inequality(self):
        """F >= IC always (AM-GM inequality)."""
        computer = OptimizedKernelComputer()
        for _ in range(50):
            c = np.random.uniform(0.01, 0.99, size=5)
            w = np.random.uniform(0.1, 1.0, size=5)
            w = w / w.sum()
            result = computer.compute(c, w)
            assert result.F >= result.IC - 1e-10

    def test_kappa_negative(self):
        """kappa <= 0 since all c in (0, 1]."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert result.kappa <= 0

    def test_homogeneous_detection(self):
        """Homogeneous coordinates detected correctly."""
        computer = OptimizedKernelComputer()
        c = np.array([0.75, 0.75, 0.75, 0.75])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        result = computer.compute(c, w)
        assert result.is_homogeneous
        assert result.C == 0  # No dispersion

    def test_homogeneous_am_gm_equality(self):
        """F = IC for homogeneous state (Lemma 4)."""
        computer = OptimizedKernelComputer()
        c = np.array([0.75, 0.75, 0.75])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert result.is_homogeneous
        assert abs(result.F - result.IC) < 1e-10

    def test_heterogeneous_detection(self):
        """Heterogeneous coordinates not marked homogeneous."""
        computer = OptimizedKernelComputer()
        c = np.array([0.9, 0.3, 0.7, 0.1])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        result = computer.compute(c, w)
        assert not result.is_homogeneous
        assert result.C > 0  # Dispersion exists

    def test_amgm_gap_computed(self):
        """AM-GM gap is F - IC."""
        computer = OptimizedKernelComputer()
        c = np.array([0.9, 0.3, 0.7, 0.1])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        result = computer.compute(c, w)
        assert abs(result.amgm_gap - (result.F - result.IC)) < 1e-10

    def test_entropy_non_negative(self):
        """S >= 0 always."""
        computer = OptimizedKernelComputer()
        for _ in range(50):
            c = np.random.uniform(0.01, 0.99, size=5)
            w = np.random.uniform(0.1, 1.0, size=5)
            w = w / w.sum()
            result = computer.compute(c, w)
            assert result.S >= 0

    def test_weight_sum_validation(self):
        """Weights that don't sum to 1 raise error."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([0.1, 0.2, 0.3])  # Sum = 0.6
        with pytest.raises(ValueError):
            computer.compute(c, w)

    def test_regime_classification(self):
        """Regime is classified."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = computer.compute(c, w)
        assert isinstance(result.regime, str)
        assert len(result.regime) > 0

    def test_computation_mode_tracked(self):
        """Computation mode is tracked."""
        computer = OptimizedKernelComputer()
        # Homogeneous case
        c_homo = np.array([0.75, 0.75, 0.75])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result_homo = computer.compute(c_homo, w)
        assert result_homo.computation_mode == "fast_homogeneous"

        # Heterogeneous case
        c_hetero = np.array([0.9, 0.3, 0.5])
        result_hetero = computer.compute(c_hetero, w)
        assert result_hetero.computation_mode == "full_heterogeneous"


class TestComputeKernelOutputs:
    """Tests for the compute_kernel_outputs function."""

    def test_returns_dict(self):
        """Function returns a dict with all invariants."""
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = compute_kernel_outputs(c, w)
        assert isinstance(result, dict)
        assert "F" in result
        assert "omega" in result
        assert "IC" in result
        assert "kappa" in result

    def test_matches_class(self):
        """Function returns same values as class."""
        computer = OptimizedKernelComputer()
        c = np.array([0.8, 0.6, 0.4])
        w = np.array([1 / 3, 1 / 3, 1 / 3])

        class_result = computer.compute(c, w)
        func_result = compute_kernel_outputs(c, w)

        assert abs(class_result.F - func_result["F"]) < 1e-10
        assert abs(class_result.IC - func_result["IC"]) < 1e-10


class TestKernelOutputs:
    """Tests for the KernelOutputs dataclass."""

    def test_outputs_accessible(self):
        """KernelOutputs fields are accessible."""
        outputs = KernelOutputs(
            F=0.6,
            omega=0.4,
            S=0.5,
            C=0.1,
            kappa=-0.5,
            IC=0.55,
            amgm_gap=0.05,
            regime="normal",
            is_homogeneous=False,
            computation_mode="full_heterogeneous",
        )
        assert outputs.F == 0.6
        assert outputs.omega == 0.4
        assert outputs.kappa == -0.5


class TestErrorBounds:
    """Tests for the ErrorBounds dataclass."""

    def test_bounds_accessible(self):
        """ErrorBounds fields are accessible."""
        bounds = ErrorBounds(F=0.01, omega=0.01, kappa=0.001, S=0.01)
        assert bounds.F == 0.01
        assert bounds.omega == 0.01
