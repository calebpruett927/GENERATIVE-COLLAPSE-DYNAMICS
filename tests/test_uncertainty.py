"""Tests for uncertainty propagation module.

Tests delta-method uncertainty propagation through kernel statistics.
"""

import numpy as np

from umcp.uncertainty import (
    KernelGradients,
    UncertaintyBounds,
    compute_kernel_gradients,
    kappa_sensitivity_bound,
    ledger_change_sensitivity,
    propagate_independent_uncertainty,
    propagate_uncertainty,
)


class TestKernelGradients:
    """Test gradient computation."""

    def test_compute_gradients_basic(self) -> None:
        """Basic gradient computation."""
        c = np.array([0.8, 0.9, 0.7])
        w = np.array([0.5, 0.3, 0.2])

        grads = compute_kernel_gradients(c, w)

        assert isinstance(grads, KernelGradients)
        assert grads.grad_F.shape == (3,)
        assert grads.grad_omega.shape == (3,)
        assert grads.grad_S.shape == (3,)
        assert grads.grad_kappa.shape == (3,)
        assert grads.grad_C.shape == (3,)

    def test_gradient_F_equals_weights(self) -> None:
        """∂F/∂c_i = w_i."""
        c = np.array([0.5, 0.6, 0.7])
        w = np.array([0.4, 0.35, 0.25])

        grads = compute_kernel_gradients(c, w)

        np.testing.assert_allclose(grads.grad_F, w)

    def test_gradient_omega_negative_weights(self) -> None:
        """∂ω/∂c_i = -w_i (since ω = 1 - F)."""
        c = np.array([0.5, 0.6, 0.7])
        w = np.array([0.4, 0.35, 0.25])

        grads = compute_kernel_gradients(c, w)

        np.testing.assert_allclose(grads.grad_omega, -w)

    def test_gradient_kappa_formula(self) -> None:
        """∂κ/∂c_i = w_i/c_i."""
        c = np.array([0.5, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])

        grads = compute_kernel_gradients(c, w)

        expected = w / c
        np.testing.assert_allclose(grads.grad_kappa, expected)

    def test_gradient_S_finite(self) -> None:
        """∂S/∂c_i is finite for valid c."""
        c = np.array([0.5, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])

        grads = compute_kernel_gradients(c, w)

        assert np.all(np.isfinite(grads.grad_S))

    def test_gradient_C_finite(self) -> None:
        """∂C/∂c_i is finite for valid c."""
        c = np.array([0.5, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])

        grads = compute_kernel_gradients(c, w)

        assert np.all(np.isfinite(grads.grad_C))


class TestUncertaintyPropagation:
    """Test uncertainty propagation."""

    def test_propagate_diagonal_covariance(self) -> None:
        """Propagation with diagonal covariance."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.diag([0.01, 0.01, 0.01])

        bounds = propagate_uncertainty(c, w, V)

        assert isinstance(bounds, UncertaintyBounds)
        assert bounds.var_F > 0
        assert bounds.var_omega > 0
        assert bounds.var_S > 0
        assert bounds.var_kappa > 0
        assert bounds.var_C >= 0

    def test_propagate_zero_covariance(self) -> None:
        """Zero covariance gives zero variance."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.zeros((3, 3))

        bounds = propagate_uncertainty(c, w, V)

        assert bounds.var_F == 0
        assert bounds.var_omega == 0
        assert bounds.var_S == 0
        assert bounds.var_kappa == 0
        assert bounds.var_C == 0

    def test_propagate_formula_var_F(self) -> None:
        """Var(F) = w^T V w."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.diag([0.01, 0.02, 0.03])

        bounds = propagate_uncertainty(c, w, V)

        # Manual calculation
        expected_var_F = w @ V @ w
        np.testing.assert_allclose(bounds.var_F, expected_var_F)

    def test_propagate_var_omega_equals_var_F(self) -> None:
        """Var(ω) = Var(F) since ω = 1 - F."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.diag([0.01, 0.02, 0.03])

        bounds = propagate_uncertainty(c, w, V)

        np.testing.assert_allclose(bounds.var_F, bounds.var_omega)

    def test_std_values(self) -> None:
        """Standard deviations are sqrt of variances."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.diag([0.01, 0.02, 0.03])

        bounds = propagate_uncertainty(c, w, V)

        np.testing.assert_allclose(bounds.std_F, np.sqrt(bounds.var_F))
        np.testing.assert_allclose(bounds.std_omega, np.sqrt(bounds.var_omega))
        np.testing.assert_allclose(bounds.std_S, np.sqrt(bounds.var_S))
        np.testing.assert_allclose(bounds.std_kappa, np.sqrt(bounds.var_kappa))
        np.testing.assert_allclose(bounds.std_C, np.sqrt(bounds.var_C))

    def test_propagate_independent(self) -> None:
        """Independent propagation is equivalent to diagonal covariance."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        var_c = np.array([0.01, 0.02, 0.03])

        bounds_ind = propagate_independent_uncertainty(c, w, var_c)
        bounds_diag = propagate_uncertainty(c, w, np.diag(var_c))

        np.testing.assert_allclose(bounds_ind.var_F, bounds_diag.var_F)
        np.testing.assert_allclose(bounds_ind.var_kappa, bounds_diag.var_kappa)


class TestSensitivityBounds:
    """Test sensitivity bound functions."""

    def test_kappa_sensitivity_bound_basic(self) -> None:
        """kappa sensitivity bound is max(w)/epsilon."""
        w = np.array([0.5, 0.3, 0.2])
        epsilon = 0.01

        bound = kappa_sensitivity_bound(w, epsilon)

        expected = 0.5 / 0.01  # max(w) / epsilon
        assert bound == expected

    def test_kappa_sensitivity_increases_with_smaller_epsilon(self) -> None:
        """Smaller epsilon means higher sensitivity."""
        w = np.array([0.5, 0.3, 0.2])

        bound_small_eps = kappa_sensitivity_bound(w, 0.001)
        bound_large_eps = kappa_sensitivity_bound(w, 0.1)

        assert bound_small_eps > bound_large_eps

    def test_ledger_change_sensitivity(self) -> None:
        """Ledger change sensitivity bound."""
        w = np.array([0.5, 0.3, 0.2])
        epsilon = 0.01
        delta_c = np.array([0.01, 0.02, 0.01])

        bound = ledger_change_sensitivity(w, epsilon, delta_c)

        expected = np.dot(w, np.abs(delta_c)) / epsilon
        np.testing.assert_allclose(bound, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_channel(self) -> None:
        """Works with single channel."""
        c = np.array([0.8])
        w = np.array([1.0])
        V = np.array([[0.01]])

        grads = compute_kernel_gradients(c, w)
        bounds = propagate_uncertainty(c, w, V)

        assert grads.grad_F.shape == (1,)
        assert bounds.var_F > 0

    def test_many_channels(self) -> None:
        """Works with many channels."""
        n = 100
        c = np.random.uniform(0.5, 1.0, n)
        w = np.ones(n) / n
        V = np.eye(n) * 0.01

        grads = compute_kernel_gradients(c, w)
        bounds = propagate_uncertainty(c, w, V)

        assert grads.grad_F.shape == (n,)
        assert bounds.var_F > 0

    def test_small_c_values(self) -> None:
        """Handles small but positive c values."""
        c = np.array([0.01, 0.02, 0.03])
        w = np.array([0.5, 0.3, 0.2])
        V = np.eye(3) * 0.0001

        # Should not raise
        grads = compute_kernel_gradients(c, w)
        bounds = propagate_uncertainty(c, w, V)

        assert np.all(np.isfinite(grads.grad_kappa))
        assert np.isfinite(bounds.var_kappa)

    def test_extreme_weight_concentration(self) -> None:
        """Handles concentrated weights."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.98, 0.01, 0.01])
        V = np.eye(3) * 0.01

        grads = compute_kernel_gradients(c, w)
        bounds = propagate_uncertainty(c, w, V)

        assert np.all(np.isfinite(grads.grad_F))
        assert np.isfinite(bounds.var_F)


class TestMathematicalProperties:
    """Test mathematical properties."""

    def test_gradient_sum_F_equals_one(self) -> None:
        """Sum of F gradients equals 1 when weights sum to 1."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])

        grads = compute_kernel_gradients(c, w)

        np.testing.assert_allclose(np.sum(grads.grad_F), 1.0)

    def test_variance_non_negative(self) -> None:
        """All variances are non-negative."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])
        V = np.eye(3) * 0.01

        bounds = propagate_uncertainty(c, w, V)

        assert bounds.var_F >= 0
        assert bounds.var_omega >= 0
        assert bounds.var_S >= 0
        assert bounds.var_kappa >= 0
        assert bounds.var_C >= 0

    def test_higher_input_variance_higher_output(self) -> None:
        """Higher input variance leads to higher output variance."""
        c = np.array([0.7, 0.8, 0.9])
        w = np.array([0.5, 0.3, 0.2])

        V_low = np.eye(3) * 0.001
        V_high = np.eye(3) * 0.01

        bounds_low = propagate_uncertainty(c, w, V_low)
        bounds_high = propagate_uncertainty(c, w, V_high)

        assert bounds_high.var_F > bounds_low.var_F
        assert bounds_high.var_kappa > bounds_low.var_kappa

    def test_correlated_channels_affect_variance(self) -> None:
        """Positive correlation increases variance."""
        c = np.array([0.7, 0.8])
        w = np.array([0.5, 0.5])

        # Independent
        V_ind = np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float64)
        # Positively correlated
        V_corr = np.array([[0.01, 0.008], [0.008, 0.01]], dtype=np.float64)

        bounds_ind = propagate_uncertainty(c, w, V_ind)
        bounds_corr = propagate_uncertainty(c, w, V_corr)

        # Positive correlation increases variance of weighted sum
        assert bounds_corr.var_F > bounds_ind.var_F
