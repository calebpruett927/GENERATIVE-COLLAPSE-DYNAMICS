"""Tests for frozen contract constants and functions.

Tests the canonical values and functions from The Physics of Coherence.
"""

import numpy as np
import pytest

from umcp.frozen_contract import (
    ALPHA,
    DEFAULT_CONTRACT,
    DEFAULT_THRESHOLDS,
    DOMAIN_MAX,
    DOMAIN_MIN,
    EPSILON,
    FACE_POLICY,
    LAMBDA,
    P_EXPONENT,
    TIMEZONE,
    TOL_SEAM,
    NonconformanceType,
    Regime,
    check_seam_pass,
    classify_regime,
    compute_budget_delta_kappa,
    compute_kernel,
    compute_seam_residual,
    compute_tau_R,
    cost_curvature,
    equator_phi,
    gamma_omega,
)


class TestCanonicalConstants:
    """Test canonical constant values."""

    def test_domain_bounds(self) -> None:
        """Normalization domain is [0, 1]."""
        assert DOMAIN_MIN == 0.0
        assert DOMAIN_MAX == 1.0

    def test_epsilon(self) -> None:
        """Guard band is 10^-8."""
        assert EPSILON == 1e-8

    def test_face_policy(self) -> None:
        """Face policy is pre_clip."""
        assert FACE_POLICY == "pre_clip"

    def test_closure_constants(self) -> None:
        """Closure constants match canon."""
        assert P_EXPONENT == 3
        assert ALPHA == 1.0
        assert LAMBDA == 0.2

    def test_seam_tolerance(self) -> None:
        """Seam tolerance is 0.005."""
        assert TOL_SEAM == 0.005

    def test_timezone(self) -> None:
        """Timezone is America/Chicago."""
        assert TIMEZONE == "America/Chicago"


class TestRegimeClassification:
    """Test regime classification."""

    def test_stable_regime(self) -> None:
        """Stable regime when all conditions met."""
        regime = classify_regime(omega=0.02, F=0.95, S=0.10, C=0.10, integrity=0.80)
        assert regime == Regime.STABLE

    def test_watch_regime_omega(self) -> None:
        """Watch regime when ω in watch range."""
        regime = classify_regime(omega=0.15, F=0.85, S=0.10, C=0.10, integrity=0.80)
        assert regime == Regime.WATCH

    def test_collapse_regime(self) -> None:
        """Collapse regime when ω >= 0.30."""
        regime = classify_regime(omega=0.35, F=0.65, S=0.20, C=0.20, integrity=0.50)
        assert regime == Regime.COLLAPSE

    def test_critical_overlay(self) -> None:
        """Critical overlay when I < 0.30."""
        regime = classify_regime(omega=0.02, F=0.95, S=0.10, C=0.10, integrity=0.20)
        assert regime == Regime.CRITICAL

    def test_critical_takes_precedence(self) -> None:
        """Critical overlay takes precedence over other regimes."""
        # Even with stable omega, critical I triggers CRITICAL
        regime = classify_regime(omega=0.01, F=0.99, S=0.05, C=0.05, integrity=0.25)
        assert regime == Regime.CRITICAL

    def test_threshold_values(self) -> None:
        """Default thresholds match canon."""
        assert DEFAULT_THRESHOLDS.omega_stable_max == 0.038
        assert DEFAULT_THRESHOLDS.F_stable_min == 0.90
        assert DEFAULT_THRESHOLDS.S_stable_max == 0.15
        assert DEFAULT_THRESHOLDS.C_stable_max == 0.14
        assert DEFAULT_THRESHOLDS.omega_watch_max == 0.30
        assert DEFAULT_THRESHOLDS.I_critical_max == 0.30


class TestCostClosures:
    """Test cost closure functions."""

    def test_gamma_omega_formula(self) -> None:
        """Γ(ω) = ω^p / (1 - ω + ε)."""
        omega = 0.1
        p = 3
        eps = 1e-8
        expected = omega**p / (1 - omega + eps)
        np.testing.assert_allclose(gamma_omega(omega), expected)

    def test_gamma_omega_zero(self) -> None:
        """Γ(0) = 0."""
        assert gamma_omega(0.0) == 0.0

    def test_gamma_omega_approaches_infinity(self) -> None:
        """Γ(ω) grows as ω → 1."""
        assert gamma_omega(0.99) > gamma_omega(0.5)
        assert gamma_omega(0.999) > gamma_omega(0.99)

    def test_gamma_omega_cubic(self) -> None:
        """Γ uses cubic exponent by default."""
        # With p=3, doubling omega increases numerator by 8x
        ratio = gamma_omega(0.4) / gamma_omega(0.2)
        # (0.4/0.2)^3 = 8, but denominator also changes
        assert ratio > 4  # Should be significantly larger

    def test_cost_curvature(self) -> None:
        """D_C = α·C."""
        C = 0.15
        assert cost_curvature(C) == ALPHA * C
        assert cost_curvature(C, alpha=2.0) == 2.0 * C


class TestBudgetCalculations:
    """Test budget identity calculations."""

    def test_budget_delta_kappa(self) -> None:
        """Δκ_budget = R·τ_R - (D_ω + D_C)."""
        R = 0.5
        tau_R = 10.0
        D_omega = 0.02
        D_C = 0.05
        expected = R * tau_R - (D_omega + D_C)
        result = compute_budget_delta_kappa(R, tau_R, D_omega, D_C)
        np.testing.assert_allclose(result, expected)

    def test_seam_residual(self) -> None:
        """s = Δκ_budget - Δκ_ledger."""
        delta_budget = 0.5
        delta_ledger = 0.48
        expected = delta_budget - delta_ledger
        result = compute_seam_residual(delta_budget, delta_ledger)
        np.testing.assert_allclose(result, expected)


class TestSeamPassConditions:
    """Test PASS condition checking."""

    def test_pass_all_conditions_met(self) -> None:
        """PASS when all conditions satisfied."""
        delta_kappa = 0.1
        I_ratio = np.exp(delta_kappa)
        passed, failures = check_seam_pass(
            residual=0.001,  # |s| < 0.005
            tau_R=5.0,  # finite
            I_ratio=I_ratio,  # matches exp(Δκ)
            delta_kappa=delta_kappa,
        )
        assert passed is True
        assert len(failures) == 0

    def test_fail_residual_too_large(self) -> None:
        """FAIL when |s| > tol_seam."""
        delta_kappa = 0.1
        I_ratio = np.exp(delta_kappa)
        passed, failures = check_seam_pass(
            residual=0.01,  # > 0.005
            tau_R=5.0,
            I_ratio=I_ratio,
            delta_kappa=delta_kappa,
        )
        assert passed is False
        assert any("tol_seam" in f for f in failures)

    def test_fail_tau_R_infinite(self) -> None:
        """FAIL when τ_R is infinite (INF_REC)."""
        passed, failures = check_seam_pass(
            residual=0.001,
            tau_R=float("inf"),
            I_ratio=1.0,
            delta_kappa=0.0,
        )
        assert passed is False
        assert any("INF_REC" in f for f in failures)

    def test_fail_identity_mismatch(self) -> None:
        """FAIL when I_ratio doesn't match exp(Δκ)."""
        passed, failures = check_seam_pass(
            residual=0.001,
            tau_R=5.0,
            I_ratio=1.5,  # Wrong!
            delta_kappa=0.1,  # exp(0.1) ≈ 1.105
        )
        assert passed is False
        assert any("exp(Δκ)" in f for f in failures)


class TestEquatorClosure:
    """Test equator diagnostic closure."""

    def test_equator_on_equator(self) -> None:
        """Φ = 0 on equator."""
        # F = 1.00 - 0.75ω - 0.55C
        omega = 0.1
        C = 0.1
        F = 1.00 - 0.75 * omega - 0.55 * C
        phi = equator_phi(omega, F, C)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_equator_above(self) -> None:
        """Φ > 0 above equator (high fidelity)."""
        phi = equator_phi(omega=0.1, F=0.95, C=0.1)
        assert phi > 0

    def test_equator_below(self) -> None:
        """Φ < 0 below equator (low fidelity)."""
        phi = equator_phi(omega=0.1, F=0.80, C=0.1)
        assert phi < 0


class TestReturnMetric:
    """Test return metric computation."""

    def test_tau_R_immediate_return(self) -> None:
        """τ_R = 1 when immediate return."""
        # Create trace where Ψ(t) ≈ Ψ(t-1)
        trace = np.array(
            [
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],  # Same as t=0
                [0.5, 0.5, 0.5],  # Different
            ]
        )
        tau_R = compute_tau_R(trace, t=1, eta=0.01, H_rec=10)
        assert tau_R == 1.0

    def test_tau_R_no_return(self) -> None:
        """τ_R = inf when no return within horizon."""
        # Create diverging trace
        trace = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.3, 0.3, 0.3],
                [0.5, 0.5, 0.5],
                [0.7, 0.7, 0.7],
                [0.9, 0.9, 0.9],
            ]
        )
        tau_R = compute_tau_R(trace, t=4, eta=0.01, H_rec=3)
        assert tau_R == float("inf")

    def test_tau_R_delayed_return(self) -> None:
        """τ_R reflects delay to return."""
        trace = np.array(
            [
                [0.8, 0.7, 0.6],  # t=0
                [0.5, 0.5, 0.5],  # t=1, different
                [0.6, 0.6, 0.6],  # t=2, different
                [0.79, 0.69, 0.59],  # t=3, close to t=0
            ]
        )
        tau_R = compute_tau_R(trace, t=3, eta=0.1, H_rec=10)
        assert tau_R == 3.0  # Returns to t=0


class TestKernelComputation:
    """Test Tier-1 kernel invariant computation."""

    def test_kernel_fidelity(self) -> None:
        """F = Σ w_i c_i."""
        c = np.array([0.9, 0.8, 0.7])
        w = np.array([0.5, 0.3, 0.2])
        kernel = compute_kernel(c, w, tau_R=1.0)
        expected_F = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        np.testing.assert_allclose(kernel.F, expected_F)

    def test_kernel_drift(self) -> None:
        """ω = 1 - F."""
        c = np.array([0.9, 0.8, 0.7])
        w = np.array([0.5, 0.3, 0.2])
        kernel = compute_kernel(c, w, tau_R=1.0)
        np.testing.assert_allclose(kernel.omega, 1 - kernel.F)

    def test_kernel_integrity(self) -> None:
        """IC = exp(κ)."""
        c = np.array([0.9, 0.8, 0.7])
        w = np.array([0.5, 0.3, 0.2])
        kernel = compute_kernel(c, w, tau_R=1.0)
        np.testing.assert_allclose(kernel.IC, np.exp(kernel.kappa))

    def test_kernel_entropy_bounds(self) -> None:
        """0 ≤ S ≤ ln(2) for binary entropy."""
        c = np.array([0.5, 0.5, 0.5])  # Maximum entropy
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        kernel = compute_kernel(c, w, tau_R=1.0)
        assert 0 <= kernel.S <= np.log(2) + 0.01


class TestFrozenContract:
    """Test frozen contract dataclass."""

    def test_default_contract_values(self) -> None:
        """Default contract matches canon."""
        assert DEFAULT_CONTRACT.domain_min == 0.0
        assert DEFAULT_CONTRACT.domain_max == 1.0
        assert DEFAULT_CONTRACT.epsilon == 1e-8
        assert DEFAULT_CONTRACT.p == 3
        assert DEFAULT_CONTRACT.alpha == 1.0
        assert DEFAULT_CONTRACT.lambda_ == 0.2
        assert DEFAULT_CONTRACT.tol_seam == 0.005
        assert DEFAULT_CONTRACT.timezone == "America/Chicago"

    def test_frozen_contract_immutable(self) -> None:
        """Frozen contract is immutable."""
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_CONTRACT.epsilon = 1e-6  # type: ignore


class TestNonconformanceTypes:
    """Test nonconformance enumeration."""

    def test_all_types_defined(self) -> None:
        """All canonical nonconformance types exist."""
        assert NonconformanceType.SEAM_FAILURE.value == "seam_failure"
        assert NonconformanceType.NO_RETURN.value == "no_return"
        assert NonconformanceType.TIER0_FAILURE.value == "tier0_failure"
        assert NonconformanceType.CLOSURE_FAILURE.value == "closure_failure"
        assert NonconformanceType.SYMBOL_FAILURE.value == "symbol_failure"
        assert NonconformanceType.DIAGNOSTIC_MISUSE.value == "diagnostic_misuse"
