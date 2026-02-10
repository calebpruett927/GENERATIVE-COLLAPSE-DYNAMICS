"""Tests for UMA.INTSTACK.v1 / Contract-1.0 frozen claims.

Twelve claim groups derived from the January 12, 2026 contract snapshot.
Each test is a direct falsification of a stated computational invariant.

Every claim is welded to a seam: the claim runs forward through collapse
and must demonstrate return under frozen rules.  "Frozen" means consistent
across the seam — the same ε, the same tol_seam, the same closure forms on
both sides of the collapse-return boundary.  Constants are not constant in
the arbitrary sense; they are consistent in the seam sense.  Reality is
declared by showing closure after collapse, which is why each claim's
falsification tests verify seam-level consistency, not just numerical output.

Reference: UMA.INTSTACK.v1.yaml, frozen_contract.py, KERNEL_SPECIFICATION.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from umcp.compute_utils import clip_coordinates
from umcp.frozen_contract import (
    ALPHA,
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    Regime,
    check_seam_pass,
    classify_regime,
    compute_budget_delta_kappa,
    compute_kernel,
    compute_seam_residual,
    compute_tau_R,
    cost_curvature,
    gamma_omega,
)
from umcp.kernel_optimized import OptimizedKernelComputer, validate_kernel_bounds
from umcp.measurement_engine import safe_tau_R, tau_R_display
from umcp.ss1m_triad import EditionCounts, EditionTriad, compute_triad, verify_triad

# ============================================================================
# Helpers
# ============================================================================

RNG = np.random.default_rng(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
HELLO_INVARIANTS = REPO_ROOT / "casepacks" / "hello_world" / "expected" / "invariants.json"
HELLO_RECEIPT = REPO_ROOT / "casepacks" / "hello_world" / "expected" / "ss1m_receipt.json"


def _random_psi(n: int = 5, *, rng: np.random.Generator = RNG) -> np.ndarray:
    """Random coordinates in (0, 1)."""
    return rng.uniform(0.05, 0.95, size=n)


def _uniform_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


# ============================================================================
# Claim 1 — Boundedness + log-safety is enforced, not optional
# ============================================================================


class TestClaim1_Boundedness:
    """oor_policy = clip_and_flag with ε = 1e-8.

    Boundedness is a return guarantee, not a numerical safety net.
    The ε-clamp ensures no closure can fully die — if cᵢ = 0, that
    component has no path back through collapse.  The clamp at ε = 1e-8
    guarantees even the most degraded closure retains enough structure
    to return.  This is seam-critical: without it, ln(0) = −∞ makes
    κ = −∞, IC = 0, and the entire identity stack collapses.
    """

    @pytest.mark.parametrize("raw", [-0.1, 0.0, 1.0, 1.2])
    def test_clip_forces_into_epsilon_band(self, raw: float) -> None:
        """Values outside [ε, 1-ε] are clipped."""
        c = np.array([raw])
        result = clip_coordinates(c, epsilon=EPSILON)
        clipped = result.c_clipped[0]
        assert EPSILON <= clipped <= 1 - EPSILON

    def test_clip_flags_emitted_for_oor(self) -> None:
        """Out-of-range coordinates emit flags."""
        c = np.array([-0.1, 0.5, 1.2])
        result = clip_coordinates(c, epsilon=EPSILON)
        # Indices 0 and 2 should be flagged
        assert 0 in result.oor_indices
        assert 2 in result.oor_indices
        assert result.clip_count >= 2

    def test_kappa_IC_never_nan_inf_after_clip(self) -> None:
        """κ and IC are finite for any input after clipping."""
        raw_values = np.array([-0.1, 0.0, 0.5, 1.0, 1.2])
        result = clip_coordinates(raw_values, epsilon=EPSILON)
        c = result.c_clipped
        w = _uniform_weights(len(c))
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert math.isfinite(ko.kappa), f"κ is not finite: {ko.kappa}"
        assert math.isfinite(ko.IC), f"IC is not finite: {ko.IC}"
        assert ko.IC > 0, f"IC must be positive, got {ko.IC}"

    def test_clip_preserves_interior_values(self) -> None:
        """Values already in [ε, 1-ε] are unchanged."""
        c = np.array([0.3, 0.5, 0.7])
        result = clip_coordinates(c, epsilon=EPSILON)
        np.testing.assert_array_equal(c, result.c_clipped)
        assert result.clip_count == 0


# ============================================================================
# Claim 2 — Fidelity–drift identity is exact: ω = 1 − F
# ============================================================================


class TestClaim2_FidelityDrift:
    """F is the weighted mean; ω ≡ 1 − F by definition."""

    def test_identity_exact_random(self) -> None:
        """ω = 1 − F to float precision for random Ψ and weights."""
        for _ in range(50):
            c = _random_psi(n=8, rng=RNG)
            w = RNG.dirichlet(np.ones(8))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert (
                abs(ko.omega - (1 - ko.F)) < 1e-15
            ), f"ω={ko.omega}, 1-F={1 - ko.F}, diff={abs(ko.omega - (1 - ko.F))}"

    def test_identity_homogeneous(self) -> None:
        """Homogeneous case: all c_i equal."""
        c_val = 0.75
        c = np.full(4, c_val)
        w = _uniform_weights(4)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert abs(ko.F - c_val) < 1e-12
        assert abs(ko.omega - (1 - c_val)) < 1e-15

    def test_identity_extreme_low_fidelity(self) -> None:
        """Near-zero fidelity: ω ≈ 1."""
        c = np.full(3, EPSILON)
        w = _uniform_weights(3)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert abs(ko.omega - (1 - ko.F)) < 1e-15
        assert ko.omega > 0.99


# ============================================================================
# Claim 3 — Entropy S uses Bernoulli channel form
# ============================================================================


class TestClaim3_Entropy:
    """S(t) = Σ wᵢ h(c̃ᵢ(t)) with h(c) = −(c ln c + (1−c) ln(1−c))."""

    @staticmethod
    def _bernoulli_h(c: float) -> float:
        if c <= 0 or c >= 1:
            return 0.0
        return -(c * math.log(c) + (1 - c) * math.log(1 - c))

    def test_entropy_manual_computation(self) -> None:
        """Spot-check against manual Bernoulli entropy."""
        c = np.array([0.3, 0.7, 0.5])
        w = np.array([0.4, 0.3, 0.3])
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        expected_S = sum(wi * self._bernoulli_h(ci) for ci, wi in zip(c, w, strict=True))
        assert abs(ko.S - expected_S) < 1e-12, f"S={ko.S}, expected={expected_S}"

    def test_entropy_finite_for_all_rows(self) -> None:
        """S is always finite for any bounded Ψ."""
        for _ in range(50):
            c = _random_psi(n=6, rng=RNG)
            w = RNG.dirichlet(np.ones(6))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert math.isfinite(ko.S), f"S is not finite: {ko.S}"

    def test_entropy_near_zero_at_boundaries(self) -> None:
        """S ≈ 0 when all cᵢ ≈ 0 or ≈ 1 (after ε-clipping)."""
        for boundary in [EPSILON, 1 - EPSILON]:
            c = np.full(4, boundary)
            w = _uniform_weights(4)
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            # Bernoulli entropy at ε or 1-ε is very small
            assert ko.S < 0.001, f"S should be ≈0 at boundary, got {ko.S}"

    def test_entropy_maximized_at_half(self) -> None:
        """S is maximized when all cᵢ = 0.5 (Bernoulli entropy maximum)."""
        c = np.full(5, 0.5)
        w = _uniform_weights(5)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert abs(ko.S - math.log(2)) < 1e-12


# ============================================================================
# Claim 4 — Curvature C is population-std normalized by 0.5
# ============================================================================


class TestClaim4_Curvature:
    """C = σ_pop(c) / 0.5 with ddof=0."""

    def test_curvature_manual(self) -> None:
        """Reproduce C from population std."""
        c = np.array([0.2, 0.4, 0.6, 0.8])
        w = _uniform_weights(4)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        expected_C = float(np.std(c, ddof=0) / 0.5)
        assert abs(ko.C - expected_C) < 1e-12, f"C={ko.C}, expected={expected_C}"

    def test_curvature_bounded_01(self) -> None:
        """0 ≤ C ≤ 1 for any Ψ ∈ [0,1]^n."""
        for _ in range(100):
            c = _random_psi(n=7, rng=RNG)
            w = RNG.dirichlet(np.ones(7))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert 0.0 <= ko.C <= 1.0 + 1e-9, f"C out of range: {ko.C}"

    def test_curvature_uses_population_std(self) -> None:
        """Verify ddof=0 (population), not ddof=1 (sample)."""
        c = np.array([0.3, 0.7])
        w = _uniform_weights(2)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        pop_std = float(np.std(c, ddof=0))
        sample_std = float(np.std(c, ddof=1))
        assert abs(ko.C - pop_std / 0.5) < 1e-12
        # Sample std differs
        assert abs(ko.C - sample_std / 0.5) > 1e-6

    def test_curvature_zero_homogeneous(self) -> None:
        """C = 0 when all coordinates are equal."""
        c = np.full(5, 0.6)
        w = _uniform_weights(5)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert abs(ko.C) < 1e-15


# ============================================================================
# Claim 5 — Integrity is log-additive: κ = ln(IC)
# ============================================================================


class TestClaim5_LogIntegrity:
    """κ(t) = Σ ln(c̃ᵢ(t)), IC(t) = exp(κ), so κ = ln(IC)."""

    def test_kappa_ln_IC_identity(self) -> None:
        """For every random row, |κ − ln(IC)| < 1e-12."""
        for _ in range(100):
            c = _random_psi(n=6, rng=RNG)
            w = RNG.dirichlet(np.ones(6))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert abs(ko.kappa - math.log(ko.IC)) < 1e-12, f"κ={ko.kappa}, ln(IC)={math.log(ko.IC)}"

    def test_dominance_monotonicity(self) -> None:
        """If row A dominates row B (all cᵢ higher), IC_A > IC_B and ω_A < ω_B."""
        c_low = np.array([0.3, 0.4, 0.2, 0.5])
        c_high = c_low + 0.1  # Coordinate-wise domination
        w = _uniform_weights(4)
        ko_low = compute_kernel(c_low, w, tau_R=1.0, epsilon=EPSILON)
        ko_high = compute_kernel(c_high, w, tau_R=1.0, epsilon=EPSILON)
        assert ko_high.IC > ko_low.IC, "IC should increase with higher coordinates"
        assert ko_high.omega < ko_low.omega, "ω should decrease with higher coordinates"

    def test_IC_is_geometric_mean_unweighted(self) -> None:
        """IC = exp(Σ ln cᵢ) = ∏ cᵢ (unweighted product)."""
        c = np.array([0.5, 0.8, 0.6])
        w = _uniform_weights(3)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        product = float(np.prod(c))
        assert abs(ko.IC - product) < 1e-12, f"IC={ko.IC}, ∏cᵢ={product}"

    @pytest.mark.bounded_identity
    def test_amgm_inequality_holds(self) -> None:
        """IC ≤ F always (AM-GM inequality)."""
        for _ in range(100):
            c = _random_psi(n=5, rng=RNG)
            w = _uniform_weights(5)
            # Use optimized computer for weighted comparison (AM-GM on weighted means)
            computer = OptimizedKernelComputer(epsilon=EPSILON)
            out = computer.compute(np.clip(c, EPSILON, 1 - EPSILON), w)
            assert out.IC <= out.F + 1e-9, f"AM-GM violated: IC={out.IC}, F={out.F}"


# ============================================================================
# Claim 6 — Return time τ_R is a hitting-time with typed non-return boundary
# ============================================================================


class TestClaim6_TauR:
    """τ_R = min{Δt > 0 : ‖Ψ(t)−Ψ(t−Δt)‖ < η}; INF_REC if no return."""

    def test_exact_return(self) -> None:
        """Construct trace where first return lag is exactly 2."""
        # t=0: [0.5, 0.5], t=1: [0.9, 0.9], t=2: [0.5, 0.5]  ← returns to t=0
        trace = np.array(
            [
                [0.5, 0.5],
                [0.9, 0.9],
                [0.5, 0.5],
            ]
        )
        tau = compute_tau_R(trace, t=2, eta=0.1, H_rec=10, norm="L2")
        assert tau == 2.0, f"Expected τ_R=2, got {tau}"

    def test_return_at_lag_1(self) -> None:
        """Adjacent identical points → τ_R = 1."""
        trace = np.array(
            [
                [0.3, 0.4],
                [0.3, 0.4],
            ]
        )
        tau = compute_tau_R(trace, t=1, eta=0.1, H_rec=10, norm="L2")
        assert tau == 1.0

    def test_no_return_is_inf(self) -> None:
        """Monotonically diverging trace → τ_R = inf (INF_REC)."""
        trace = np.array(
            [
                [0.1, 0.1],
                [0.3, 0.3],
                [0.5, 0.5],
                [0.7, 0.7],
                [0.9, 0.9],
            ]
        )
        tau = compute_tau_R(trace, t=4, eta=0.05, H_rec=10, norm="L2")
        assert math.isinf(tau), f"Expected inf, got {tau}"

    def test_inf_rec_typed_boundary(self) -> None:
        """INF_REC is a typed sentinel, not a numeric value."""
        assert tau_R_display(float("inf")) == "INF_REC"
        assert safe_tau_R("INF_REC") == float("inf")
        assert math.isinf(safe_tau_R("INF_REC"))

    def test_no_return_budget_zero(self) -> None:
        """When τ_R = INF_REC, the contract forces R·τ_R = 0 for budgeting.

        If you never observed return, you have zero budget for the seam,
        because the seam does not exist for you.  This is the anti-cheat
        condition: continuity cannot be synthesized from structure alone —
        it must be measured.  In practice: seam pass check fails when
        τ_R = inf (condition 2).
        """
        _, failures = check_seam_pass(
            residual=0.0,
            tau_R=float("inf"),
            I_ratio=1.0,
            delta_kappa=0.0,
        )
        assert len(failures) > 0
        assert any("not finite" in f for f in failures)


# ============================================================================
# Claim 7 — Weld budget + residual identity is mechanical and must close
# ============================================================================


class TestClaim7_WeldBudget:
    """Δκ_budget = R·τ_R − (D_ω + D_C); residual s must close to zero.

    The weld budget is the load-bearing claim of the protocol.  It prices
    the cost of crossing the seam: if the budget is positive, you can afford
    the return; if zero or negative, you cannot.  The residual s measures
    the gap between modeled and measured change in log-integrity.  Every
    term is computed under frozen (seam-consistent) closures — same Γ, same
    D_C, same R on both sides.
    """

    def test_budget_formula(self) -> None:
        """Verify the budget formula directly."""
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        budget = compute_budget_delta_kappa(R, tau, D_w, D_c)
        expected = R * tau - (D_w + D_c)
        assert abs(budget - expected) < 1e-15

    def test_residual_zero_when_budget_equals_ledger(self) -> None:
        """s = 0 when Δκ_budget = Δκ_ledger."""
        budget = 0.94  # R·τ_R - (D_ω + D_C)
        ledger = 0.94  # κ(t1) - κ(t0)
        s = compute_seam_residual(budget, ledger)
        assert abs(s) < 1e-15

    def test_seam_pass_all_conditions(self) -> None:
        """Construct a seam where all PASS conditions hold."""
        # Define seam parameters
        kappa_t0 = -0.5
        kappa_t1 = -0.4
        delta_kappa = kappa_t1 - kappa_t0  # 0.1

        IC_t0 = math.exp(kappa_t0)
        IC_t1 = math.exp(kappa_t1)
        I_ratio = IC_t1 / IC_t0

        # Choose R, tau_R, D_ω, D_C so budget matches ledger
        R = 0.2
        D_omega = 0.01
        D_C = 0.02
        # Solve: R·τ = delta_kappa + D_ω + D_C → τ = (0.1 + 0.03) / 0.2 = 0.65
        tau_R = (delta_kappa + D_omega + D_C) / R

        budget = compute_budget_delta_kappa(R, tau_R, D_omega, D_C)
        residual = compute_seam_residual(budget, delta_kappa)

        passes, failures = check_seam_pass(
            residual=residual,
            tau_R=tau_R,
            I_ratio=I_ratio,
            delta_kappa=delta_kappa,
        )
        assert passes, f"Seam should PASS but failed: {failures}"
        assert abs(residual) < TOL_SEAM

    def test_exponential_identity(self) -> None:
        """IC₁/IC₀ ≈ exp(Δκ_ledger)."""
        kappa_t0, kappa_t1 = -1.2, -0.8
        IC_t0 = math.exp(kappa_t0)
        IC_t1 = math.exp(kappa_t1)
        delta_kappa = kappa_t1 - kappa_t0
        ratio = IC_t1 / IC_t0
        assert abs(ratio - math.exp(delta_kappa)) < 1e-12

    def test_seam_fail_on_large_residual(self) -> None:
        """Seam must FAIL if |s| > tol_seam."""
        passes, failures = check_seam_pass(
            residual=0.1,  # >> 0.005 tolerance
            tau_R=5.0,
            I_ratio=1.0,
            delta_kappa=0.0,
        )
        assert not passes
        assert any("tol_seam" in f for f in failures)


# ============================================================================
# Claim 8 — Seam-model closures are frozen and testable: Γ, D_C, R
# ============================================================================


class TestClaim8_FrozenClosures:
    """Γ(ω)=ω^p/(1−ω+ε) with p=3; D_C=α·C with α=1; R=λ with λ=0.2.

    These values are not arbitrary constants — they are consistent across
    the seam.  The same Γ form, the same α, the same λ must govern both
    the outbound computation and the return verification.  If any of these
    changed between sides, the seam would be undefined and the weld could
    not be evaluated.  "Frozen" means "does not change within a single
    collapse-return cycle."
    """

    def test_gamma_formula(self) -> None:
        """Γ(ω) = ω³ / (1 − ω + ε) with p=3."""
        omega = 0.25
        expected = omega**3 / (1 - omega + EPSILON)
        actual = gamma_omega(omega, p=P_EXPONENT, epsilon=EPSILON)
        assert abs(actual - expected) < 1e-15

    def test_gamma_monotone_increasing(self) -> None:
        """Γ(ω) is monotonically increasing on [0, 1)."""
        omegas = np.linspace(0.01, 0.99, 50)
        gammas = [gamma_omega(o) for o in omegas]
        for i in range(1, len(gammas)):
            assert gammas[i] >= gammas[i - 1], f"Γ not monotone at ω={omegas[i]}: {gammas[i]} < {gammas[i - 1]}"

    def test_gamma_nonnegative(self) -> None:
        """Γ(ω) ≥ 0 for all ω ∈ [0, 1)."""
        for omega in np.linspace(0.0, 0.99, 100):
            assert gamma_omega(omega) >= 0.0

    def test_gamma_stable_near_one(self) -> None:
        """Γ(ω) is finite near ω → 1 because of +ε guard band."""
        gamma_near_1 = gamma_omega(0.9999)
        assert math.isfinite(gamma_near_1)
        gamma_at_1 = gamma_omega(1.0)
        assert math.isfinite(gamma_at_1)

    def test_cost_curvature_alpha_one(self) -> None:
        """D_C = α·C with α=1.0."""
        C = 0.42
        assert abs(cost_curvature(C, alpha=ALPHA) - C) < 1e-15

    def test_frozen_constants(self) -> None:
        """Verify frozen constants match contract.

        These values are consistent across the seam — same on both sides
        of any collapse-return boundary.  They are not "chosen" constants;
        they are the rules under which return is evaluated.
        """
        assert P_EXPONENT == 3
        assert abs(ALPHA - 1.0) < 1e-15
        assert abs(EPSILON - 1e-8) < 1e-20
        assert abs(TOL_SEAM - 0.005) < 1e-15


# ============================================================================
# Claim 9 — Regime gates are threshold rules + worst-of join
# ============================================================================


class TestClaim9_RegimeGates:
    """Stable/Watch/Collapse/Critical from declared thresholds; worst-of join.

    The regime classification is a phase diagram, not a status bar.  The
    gates partition the invariant space into regions where different dynamical
    behaviors dominate. In Stable, small perturbations decay. In Watch, they
    persist. In Collapse, they grow.  The thresholds are frozen (consistent
    across the seam) so that regime labels have the same meaning on both
    sides of collapse-return.
    """

    @pytest.mark.parametrize(
        "omega, F, S, C, IC, expected_regime",
        [
            # Well inside Stable
            (0.01, 0.99, 0.05, 0.05, 0.95, Regime.STABLE),
            # Just below stable omega threshold
            (0.037, 0.963, 0.10, 0.10, 0.80, Regime.STABLE),
            # At watch boundary (ω = 0.038)
            (0.038, 0.962, 0.10, 0.10, 0.80, Regime.WATCH),
            # Mid-watch
            (0.15, 0.85, 0.20, 0.20, 0.60, Regime.WATCH),
            # At collapse boundary (ω = 0.30)
            (0.30, 0.70, 0.30, 0.30, 0.50, Regime.COLLAPSE),
            # Deep collapse
            (0.80, 0.20, 0.50, 0.60, 0.40, Regime.COLLAPSE),
            # Critical overlay overrides everything when IC < 0.30
            (0.01, 0.99, 0.05, 0.05, 0.25, Regime.CRITICAL),
            # Critical even with collapse-level omega
            (0.50, 0.50, 0.40, 0.40, 0.10, Regime.CRITICAL),
            # Stable omega but S too high → Watch fallback
            (0.02, 0.98, 0.20, 0.05, 0.80, Regime.WATCH),
            # Stable omega but C too high → Watch fallback
            (0.02, 0.98, 0.05, 0.20, 0.80, Regime.WATCH),
        ],
    )
    def test_regime_thresholds(
        self,
        omega: float,
        F: float,
        S: float,
        C: float,
        IC: float,
        expected_regime: Regime,
    ) -> None:
        result = classify_regime(omega, F, S, C, IC)
        assert (
            result == expected_regime
        ), f"ω={omega}, F={F}, S={S}, C={C}, IC={IC}: expected {expected_regime}, got {result}"

    def test_critical_overlay_overrides_stable(self) -> None:
        """Critical overlay takes precedence over stable conditions."""
        # All conditions scream "Stable" except IC < 0.30
        result = classify_regime(omega=0.01, F=0.99, S=0.01, C=0.01, integrity=0.20)
        assert result == Regime.CRITICAL

    def test_worst_of_join_multi_channel(self) -> None:
        """In multi-channel, joint regime = worst-of per-channel regimes.

        We simulate by classifying each channel independently and taking max severity.
        """
        severity = {Regime.STABLE: 0, Regime.WATCH: 1, Regime.COLLAPSE: 2, Regime.CRITICAL: 3}
        channels = [
            (0.01, 0.99, 0.05, 0.05, 0.95),  # Stable
            (0.15, 0.85, 0.20, 0.20, 0.60),  # Watch
            (0.01, 0.99, 0.03, 0.03, 0.90),  # Stable
        ]
        per_channel = [classify_regime(*ch) for ch in channels]
        joint = max(per_channel, key=lambda r: severity[r])
        assert joint == Regime.WATCH  # Worst of (Stable, Watch, Stable)

    def test_boundary_flip_exact(self) -> None:
        """Labels flip exactly at declared thresholds."""
        # Just below collapse
        below = classify_regime(0.299, 0.701, 0.20, 0.20, 0.50)
        assert below == Regime.WATCH
        # At collapse
        at = classify_regime(0.30, 0.70, 0.20, 0.20, 0.50)
        assert at == Regime.COLLAPSE


# ============================================================================
# Claim 10 — SS1m receipt triad is a deterministic prime-mod checksum
# ============================================================================


class TestClaim10_SS1mTriad:
    """C1=(P+F+T+E+R) mod 97, C2=(P+2F+3T+5E+7R) mod 97,
    C3=(P·F + T·E + R) mod 97."""

    def test_published_example(self) -> None:
        """Reproduce the Episteme's published C=[11,33,6] from (64,2,12,92,35)."""
        counts = EditionCounts(pages=64, figures=2, tables=12, equations=92, references=35)
        triad = compute_triad(counts)
        assert triad.c1 == 11, f"C1 expected 11, got {triad.c1}"
        assert triad.c2 == 33, f"C2 expected 33, got {triad.c2}"
        assert triad.c3 == 6, f"C3 expected 6, got {triad.c3}"

    def test_triad_manual_computation(self) -> None:
        """Manual recomputation from formula."""
        P, F, T, E, R = 64, 2, 12, 92, 35
        c1 = (P + F + T + E + R) % 97
        c2 = (1 * P + 2 * F + 3 * T + 5 * E + 7 * R) % 97
        c3 = (P * F + T * E + R) % 97
        counts = EditionCounts(P, F, T, E, R)
        triad = compute_triad(counts)
        assert triad.c1 == c1
        assert triad.c2 == c2
        assert triad.c3 == c3

    def test_verify_triad(self) -> None:
        """verify_triad returns True for matching counts/triad."""
        counts = EditionCounts(pages=64, figures=2, tables=12, equations=92, references=35)
        expected = EditionTriad(c1=11, c2=33, c3=6)
        assert verify_triad(counts, expected)

    def test_verify_triad_fails_on_mismatch(self) -> None:
        """verify_triad returns False for wrong triad."""
        counts = EditionCounts(pages=64, figures=2, tables=12, equations=92, references=35)
        wrong = EditionTriad(c1=12, c2=33, c3=6)  # C1 off by 1
        assert not verify_triad(counts, wrong)

    def test_triad_is_deterministic(self) -> None:
        """Same inputs always produce same outputs."""
        counts = EditionCounts(10, 20, 30, 40, 50)
        t1 = compute_triad(counts)
        t2 = compute_triad(counts)
        assert t1 == t2

    def test_modulus_is_97(self) -> None:
        """All components are mod 97 (a prime)."""
        # Large values to test modular arithmetic
        counts = EditionCounts(1000, 2000, 3000, 4000, 5000)
        triad = compute_triad(counts)
        assert 0 <= triad.c1 < 97
        assert 0 <= triad.c2 < 97
        assert 0 <= triad.c3 < 97


# ============================================================================
# Claim 11 — Sensitivity: budget is affine in primary variables
# ============================================================================


class TestClaim11_Sensitivity:
    """∂Δκ/∂τ_R = R, ∂Δκ/∂R = τ_R, ∂Δκ/∂D_ω = −1, ∂Δκ/∂D_C = −1."""

    DELTA = 1e-7  # Finite difference step

    def test_dkappa_d_tau_R(self) -> None:
        """∂Δκ/∂τ_R = R."""
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        f0 = compute_budget_delta_kappa(R, tau, D_w, D_c)
        f1 = compute_budget_delta_kappa(R, tau + self.DELTA, D_w, D_c)
        deriv = (f1 - f0) / self.DELTA
        assert abs(deriv - R) < 1e-5, f"∂Δκ/∂τ_R={deriv}, expected R={R}"

    def test_dkappa_d_R(self) -> None:
        """∂Δκ/∂R = τ_R."""
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        f0 = compute_budget_delta_kappa(R, tau, D_w, D_c)
        f1 = compute_budget_delta_kappa(R + self.DELTA, tau, D_w, D_c)
        deriv = (f1 - f0) / self.DELTA
        assert abs(deriv - tau) < 1e-5, f"∂Δκ/∂R={deriv}, expected τ_R={tau}"

    def test_dkappa_d_D_omega(self) -> None:
        """∂Δκ/∂D_ω = −1."""
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        f0 = compute_budget_delta_kappa(R, tau, D_w, D_c)
        f1 = compute_budget_delta_kappa(R, tau, D_w + self.DELTA, D_c)
        deriv = (f1 - f0) / self.DELTA
        assert abs(deriv - (-1.0)) < 1e-5, f"∂Δκ/∂D_ω={deriv}, expected -1"

    def test_dkappa_d_D_C(self) -> None:
        """∂Δκ/∂D_C = −1."""
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        f0 = compute_budget_delta_kappa(R, tau, D_w, D_c)
        f1 = compute_budget_delta_kappa(R, tau, D_w, D_c + self.DELTA)
        deriv = (f1 - f0) / self.DELTA
        assert abs(deriv - (-1.0)) < 1e-5, f"∂Δκ/∂D_C={deriv}, expected -1"

    def test_budget_linearity_random(self) -> None:
        """Budget Δκ = R·τ − D_ω − D_C is affine (linear in each variable)."""
        for _ in range(20):
            R = RNG.uniform(0.01, 1.0)
            tau = RNG.uniform(1.0, 100.0)
            D_w = RNG.uniform(0.0, 0.5)
            D_c = RNG.uniform(0.0, 0.5)
            alpha = RNG.uniform(0.1, 5.0)

            # Scale tau by alpha: budget should change by alpha * R * (original tau part)
            b1 = compute_budget_delta_kappa(R, tau, D_w, D_c)
            b2 = compute_budget_delta_kappa(R, alpha * tau, D_w, D_c)
            assert abs(b2 - b1 - R * (alpha - 1) * tau) < 1e-10


# ============================================================================
# Claim 12 — Integration golden casepack matches expected
# ============================================================================


class TestClaim12_GoldenCasepack:
    """hello_world casepack expected invariants match byte-for-byte."""

    @pytest.fixture()
    def invariants(self) -> dict[str, Any]:
        assert HELLO_INVARIANTS.exists(), f"Missing {HELLO_INVARIANTS}"
        return json.loads(HELLO_INVARIANTS.read_text(encoding="utf-8"))

    @pytest.fixture()
    def receipt(self) -> dict[str, Any]:
        assert HELLO_RECEIPT.exists(), f"Missing {HELLO_RECEIPT}"
        return json.loads(HELLO_RECEIPT.read_text(encoding="utf-8"))

    def test_invariants_schema_reference(self, invariants: dict[str, Any]) -> None:
        assert invariants["schema"] == "schemas/invariants.schema.json"

    def test_invariants_contract_id(self, invariants: dict[str, Any]) -> None:
        assert invariants["contract_id"] == "UMA.INTSTACK.v1"

    def test_row_t0_fidelity_drift(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: ω=0.01, F=0.99, identity ω=1−F."""
        row = invariants["rows"][0]
        assert row["t"] == 0
        assert abs(row["omega"] - 0.01) < 1e-9
        assert abs(row["F"] - 0.99) < 1e-9
        assert abs(row["omega"] - (1 - row["F"])) < 1e-15

    def test_row_t0_entropy(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: S matches expected value."""
        row = invariants["rows"][0]
        expected_S = 0.056001534354847386
        assert abs(row["S"] - expected_S) < 1e-12

    def test_row_t0_curvature(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: C = 0 (single coordinate)."""
        row = invariants["rows"][0]
        assert abs(row["C"] - 0.0) < 1e-15

    def test_row_t0_tau_R_typed(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: τ_R = INF_REC (typed sentinel string)."""
        row = invariants["rows"][0]
        assert row["tau_R"] == "INF_REC"

    def test_row_t0_kappa_IC_identity(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: κ and IC satisfy κ = ln(IC)."""
        row = invariants["rows"][0]
        assert abs(row["kappa"] - math.log(row["IC"])) < 1e-12

    def test_row_t0_regime(self, invariants: dict[str, Any]) -> None:
        """Golden row t=0: Stable regime, no critical overlay."""
        row = invariants["rows"][0]
        assert row["regime"]["label"] == "Stable"
        assert row["regime"]["critical_overlay"] is False

    def test_receipt_kernel_matches_invariants(self, invariants: dict[str, Any], receipt: dict[str, Any]) -> None:
        """SS1m receipt kernel values must match invariants exactly."""
        row = invariants["rows"][0]
        kernel = receipt["receipt"]["kernel"]
        for key in ("omega", "F", "S", "C", "kappa", "IC"):
            assert abs(row[key] - kernel[key]) < 1e-15, f"{key}: invariant={row[key]}, receipt={kernel[key]}"
        assert row["tau_R"] == kernel["tau_R"]

    def test_receipt_contract_reference(self, receipt: dict[str, Any]) -> None:
        assert receipt["receipt"]["contract"]["id"] == "UMA.INTSTACK.v1"

    def test_receipt_tolerances(self, receipt: dict[str, Any]) -> None:
        """Receipt declares frozen tolerances."""
        tols = receipt["receipt"]["tolerances"]
        assert abs(tols["epsilon"] - 1e-8) < 1e-20
        assert abs(tols["tol_seam"] - 0.005) < 1e-15
        assert abs(tols["tol_id"] - 1e-9) < 1e-20


# ============================================================================
# Cross-cutting: kernel_optimized agrees with frozen_contract
# ============================================================================


class TestCrossCutting:
    """Verify OptimizedKernelComputer produces consistent results."""

    def test_optimized_vs_frozen_fidelity_drift(self) -> None:
        """Both kernel implementations agree on F and ω."""
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        c_safe = np.clip(c, EPSILON, 1 - EPSILON)

        frozen = compute_kernel(c_safe, w, tau_R=1.0, epsilon=EPSILON)
        optimized = OptimizedKernelComputer(epsilon=EPSILON).compute(c_safe, w)

        assert abs(frozen.F - optimized.F) < 1e-12
        assert abs(frozen.omega - optimized.omega) < 1e-12

    def test_optimized_vs_frozen_entropy(self) -> None:
        """Both implementations agree on S."""
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        c_safe = np.clip(c, EPSILON, 1 - EPSILON)

        frozen = compute_kernel(c_safe, w, tau_R=1.0, epsilon=EPSILON)
        optimized = OptimizedKernelComputer(epsilon=EPSILON).compute(c_safe, w)

        assert abs(frozen.S - optimized.S) < 1e-12

    def test_optimized_vs_frozen_curvature(self) -> None:
        """Both implementations agree on C."""
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        c_safe = np.clip(c, EPSILON, 1 - EPSILON)

        frozen = compute_kernel(c_safe, w, tau_R=1.0, epsilon=EPSILON)
        optimized = OptimizedKernelComputer(epsilon=EPSILON).compute(c_safe, w)

        assert abs(frozen.C - optimized.C) < 1e-12

    def test_validate_kernel_bounds_accepts_valid(self) -> None:
        """validate_kernel_bounds accepts well-formed outputs."""
        c = _random_psi(n=4)
        w = _uniform_weights(4)
        c_safe = np.clip(c, EPSILON, 1 - EPSILON)
        ko = compute_kernel(c_safe, w, tau_R=1.0, epsilon=EPSILON)
        assert validate_kernel_bounds(ko.F, ko.omega, ko.C, ko.IC, ko.kappa, epsilon=EPSILON)
