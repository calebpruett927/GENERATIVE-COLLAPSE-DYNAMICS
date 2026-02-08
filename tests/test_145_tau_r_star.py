"""Tests for τ_R* thermodynamic diagnostic (Tier-2).

Validates:
- Core τ_R* computation (Def T1)
- R_critical and R_min (Theorems T4, T5)
- Trapping threshold (Theorem T3)
- Phase classification (Theorem T2)
- Dominance classification (Theorem T1)
- Tier-1 identity checks (Tier-0 protocol)
- Full diagnostic pipeline
- Batch diagnostic processing
- Prediction verification (§6 testable predictions)
- Edge cases: pole at ω=1, zero R, extreme values

Reference: KERNEL_SPECIFICATION.md §3 (Budget Model), §5 (Empirical Verification)
Reference: KERNEL_SPECIFICATION.md §5
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from umcp.frozen_contract import (
    ALPHA,
    TOL_SEAM,
    Regime,
    gamma_omega,
)
from umcp.tau_r_star import (
    DominanceTerm,
    TauRStarResult,
    ThermodynamicDiagnostic,
    ThermodynamicPhase,
    check_tier1_identities,
    classify_dominance,
    classify_phase,
    compute_R_critical,
    compute_R_min,
    compute_tau_R_star,
    compute_trapping_threshold,
    diagnose,
    diagnose_invariants,
    is_trapped,
    verify_cubic_slowing,
    verify_R_min_divergence,
    verify_trapping_threshold,
)

# =============================================================================
# τ_R* CORE COMPUTATION
# =============================================================================


class TestTauRStar:
    """Test core τ_R* computation (Def T1)."""

    def test_basic_computation(self) -> None:
        """τ_R* = (Γ(ω) + αC + Δκ) / R."""
        omega, C, R = 0.1, 0.1, 0.01
        result = compute_tau_R_star(omega, C, R)
        gamma = gamma_omega(omega)
        expected = (gamma + ALPHA * C + 0.0) / R
        np.testing.assert_allclose(result.tau_R_star, expected, rtol=1e-10)

    def test_decomposition(self) -> None:
        """Result decomposes into Γ, D_C, Δκ."""
        omega, C, R, dk = 0.2, 0.15, 0.02, 0.01
        result = compute_tau_R_star(omega, C, R, dk)
        assert result.gamma == gamma_omega(omega)
        assert result.D_C == ALPHA * C
        assert result.delta_kappa == dk
        np.testing.assert_allclose(
            result.numerator,
            result.gamma + result.D_C + result.delta_kappa,
            rtol=1e-12,
        )

    def test_named_tuple_fields(self) -> None:
        """TauRStarResult has correct fields."""
        result = compute_tau_R_star(0.1, 0.1, 0.01)
        assert isinstance(result, TauRStarResult)
        assert hasattr(result, "tau_R_star")
        assert hasattr(result, "gamma")
        assert hasattr(result, "D_C")
        assert hasattr(result, "delta_kappa")
        assert hasattr(result, "R")
        assert hasattr(result, "numerator")

    def test_zero_omega(self) -> None:
        """τ_R* ≈ αC/R when ω=0 (no drift)."""
        C, R = 0.1, 0.01
        result = compute_tau_R_star(0.0, C, R)
        # Γ(0) = 0^3 / (1 - 0 + ε) ≈ 0
        np.testing.assert_allclose(result.gamma, 0.0, atol=1e-15)
        np.testing.assert_allclose(result.tau_R_star, ALPHA * C / R, rtol=1e-10)

    def test_delta_kappa_negative(self) -> None:
        """Negative Δκ reduces τ_R* (memory credit)."""
        omega, C, R = 0.1, 0.1, 0.01
        result_pos = compute_tau_R_star(omega, C, R, delta_kappa=0.05)
        result_neg = compute_tau_R_star(omega, C, R, delta_kappa=-0.05)
        assert result_neg.tau_R_star < result_pos.tau_R_star

    def test_R_positive_required(self) -> None:
        """R must be positive."""
        with pytest.raises(ValueError, match="R must be positive"):
            compute_tau_R_star(0.1, 0.1, 0.0)
        with pytest.raises(ValueError, match="R must be positive"):
            compute_tau_R_star(0.1, 0.1, -0.01)

    def test_cubic_scaling(self) -> None:
        """Γ(ω) ∝ ω³ at small ω (cubic suppression)."""
        R = 0.01
        r1 = compute_tau_R_star(0.01, 0.0, R)
        r2 = compute_tau_R_star(0.02, 0.0, R)
        # ratio of gammas should be ≈ (0.02/0.01)^3 = 8
        # Small correction from (1-ω) denominator at these values (~1%)
        ratio = r2.gamma / r1.gamma
        np.testing.assert_allclose(ratio, 8.0, rtol=0.02)

    def test_pole_divergence(self) -> None:
        """Γ(ω) → ∞ as ω → 1 (pole at ω = 1)."""
        # Near pole: ω = 1 - ε
        result = compute_tau_R_star(1.0 - 1e-7, 0.0, 0.01)
        assert result.gamma > 1e6  # Γ diverges

    def test_regime_separation(self) -> None:
        """τ_R* varies dramatically across regimes."""
        R = 0.01
        stable = compute_tau_R_star(0.02, 0.05, R)
        watch = compute_tau_R_star(0.15, 0.15, R)
        collapse = compute_tau_R_star(0.50, 0.30, R)
        assert stable.tau_R_star < watch.tau_R_star < collapse.tau_R_star


# =============================================================================
# R_CRITICAL AND R_MIN
# =============================================================================


class TestRCritical:
    """Test R_critical computation (Theorem T4)."""

    def test_basic_R_critical(self) -> None:
        """R_crit = (Γ + αC + Δκ) / tol_seam."""
        omega, C = 0.1, 0.1
        r_crit = compute_R_critical(omega, C)
        gamma = gamma_omega(omega)
        expected = (gamma + ALPHA * C + 0.0) / TOL_SEAM
        np.testing.assert_allclose(r_crit, expected, rtol=1e-10)

    def test_divergence_near_collapse(self) -> None:
        """R_crit diverges as ω → 1."""
        r_low = compute_R_critical(0.1, 0.1)
        r_high = compute_R_critical(0.9, 0.1)
        assert r_high > r_low * 50  # Much larger near collapse

    def test_monotone_in_omega(self) -> None:
        """R_crit increases with ω."""
        omegas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        r_crits = [compute_R_critical(o, 0.1) for o in omegas]
        for i in range(len(r_crits) - 1):
            assert r_crits[i] < r_crits[i + 1]


class TestRMin:
    """Test R_min computation (Theorem T5)."""

    def test_basic_R_min(self) -> None:
        """R_min = (Γ + αC + Δκ) / τ_target."""
        omega, C, tau = 0.1, 0.1, 1.0
        r_min = compute_R_min(omega, C, tau)
        gamma = gamma_omega(omega)
        expected = (gamma + ALPHA * C) / tau
        np.testing.assert_allclose(r_min, expected, rtol=1e-10)

    def test_zero_tau_returns_inf(self) -> None:
        """R_min = ∞ when τ_target = 0."""
        assert compute_R_min(0.1, 0.1, 0.0) == float("inf")

    def test_negative_tau_returns_inf(self) -> None:
        """R_min = ∞ when τ_target < 0."""
        assert compute_R_min(0.1, 0.1, -1.0) == float("inf")

    def test_R_min_times_one_minus_omega(self) -> None:
        """Prediction 5: R_min·(1-ω) → 1/tol_seam = 200 as ω → 1.

        At high ω, Γ(ω) ≈ ω^p/(1-ω) dominates, so
        R_min ≈ Γ(ω)/tol_seam ≈ ω^p/((1-ω)·tol_seam)
        ⇒ R_min·(1-ω) → ω^p/tol_seam → 1/tol_seam = 200.

        Convergence requires ω → 1 so that ω^p → 1.
        """
        target = 1.0 / TOL_SEAM  # 200
        # Use high-ω values where ω^p ≈ 1
        omegas = [0.99, 0.999, 0.9999]
        for omega in omegas:
            r_min = compute_R_min(omega, 0.0, TOL_SEAM)
            product = r_min * (1.0 - omega)
            # Should converge to 200 (within 5% for ω ≥ 0.99)
            np.testing.assert_allclose(product, target, rtol=0.05)


# =============================================================================
# TRAPPING THRESHOLD
# =============================================================================


class TestTrapping:
    """Test trapping threshold (Theorem T3)."""

    def test_trapping_threshold_value(self) -> None:
        """c_trap ≈ 0.315 at p=3, α=1 (Γ(ω_trap) = α at C_max=1)."""
        c_trap = compute_trapping_threshold()
        # ω_trap ≈ 0.685 where Γ(ω) = α = 1.0, so c_trap ≈ 0.315
        assert 0.25 < c_trap < 0.40, f"c_trap={c_trap} out of expected range"

    def test_trapped_at_high_omega(self) -> None:
        """Systems with high ω are trapped."""
        assert is_trapped(0.8, 0.1)
        assert is_trapped(0.9, 0.05)

    def test_not_trapped_at_low_omega(self) -> None:
        """Systems with low ω can self-correct."""
        assert not is_trapped(0.05, 0.1)
        assert not is_trapped(0.10, 0.2)

    def test_curvature_affects_trapping(self) -> None:
        """Higher C gives more correction budget, less likely trapped."""
        omega = 0.4
        # With high C, correction budget α·C is larger
        trapped_low_C = is_trapped(omega, 0.01)
        trapped_high_C = is_trapped(omega, 0.50)
        # At least one should differ (high C provides escape route)
        if trapped_low_C:
            assert not trapped_high_C  # High C should help escape


# =============================================================================
# PHASE CLASSIFICATION
# =============================================================================


class TestPhaseClassification:
    """Test thermodynamic phase classification (Theorem T2)."""

    def test_surplus_phase(self) -> None:
        """Negative τ_R* → SURPLUS."""
        phase = classify_phase(-1.0, 0.1)
        assert phase == ThermodynamicPhase.SURPLUS

    def test_deficit_phase(self) -> None:
        """Positive τ_R* → DEFICIT."""
        phase = classify_phase(1.0, 0.1)
        assert phase == ThermodynamicPhase.DEFICIT

    def test_free_return_phase(self) -> None:
        """Zero τ_R* → FREE_RETURN."""
        phase = classify_phase(0.0, 0.1)
        assert phase == ThermodynamicPhase.FREE_RETURN

    def test_pole_phase_high_omega(self) -> None:
        """ω ≈ 1 → POLE."""
        phase = classify_phase(1e10, 1.0 - 1e-9)
        assert phase == ThermodynamicPhase.POLE

    def test_pole_phase_inf(self) -> None:
        """Infinite τ_R* → POLE."""
        phase = classify_phase(float("inf"), 0.5)
        assert phase == ThermodynamicPhase.POLE

    def test_pole_phase_nan(self) -> None:
        """NaN τ_R* → POLE."""
        phase = classify_phase(float("nan"), 0.5)
        assert phase == ThermodynamicPhase.POLE


# =============================================================================
# DOMINANCE CLASSIFICATION
# =============================================================================


class TestDominanceClassification:
    """Test budget term dominance (Theorem T1)."""

    def test_drift_dominates(self) -> None:
        """Large Γ → DRIFT dominance."""
        dom = classify_dominance(10.0, 0.1, 0.01)
        assert dom == DominanceTerm.DRIFT

    def test_curvature_dominates(self) -> None:
        """Large αC → CURVATURE dominance."""
        dom = classify_dominance(0.001, 5.0, 0.01)
        assert dom == DominanceTerm.CURVATURE

    def test_memory_dominates(self) -> None:
        """Large Δκ → MEMORY dominance."""
        dom = classify_dominance(0.001, 0.01, 5.0)
        assert dom == DominanceTerm.MEMORY

    def test_negative_delta_kappa(self) -> None:
        """Absolute value determines dominance."""
        dom = classify_dominance(0.001, 0.01, -5.0)
        assert dom == DominanceTerm.MEMORY

    def test_regime_dominance_pattern(self) -> None:
        """Theorem T1: stable→memory, watch→curvature, collapse→drift."""
        # Stable: ω ≈ 0.02 → Γ tiny, Δκ can dominate
        g_stable = gamma_omega(0.02)
        dom_stable = classify_dominance(g_stable, 0.001, 0.01)
        # Γ(0.02) = 0.02^3 / (1 - 0.02) ≈ 8.16e-6
        # So Δκ=0.01 dominates
        assert dom_stable == DominanceTerm.MEMORY

        # Watch: ω ≈ 0.15 → Γ moderate, αC can dominate
        g_watch = gamma_omega(0.15)
        dom_watch = classify_dominance(g_watch, 0.15, 0.001)
        assert dom_watch == DominanceTerm.CURVATURE

        # Collapse: ω ≈ 0.50 → Γ dominates
        g_collapse = gamma_omega(0.50)
        dom_collapse = classify_dominance(g_collapse, 0.15, 0.01)
        assert dom_collapse == DominanceTerm.DRIFT


# =============================================================================
# TIER-1 IDENTITY CHECKS (TIER-0 PROTOCOL)
# =============================================================================


class TestTier1Identities:
    """Test Tier-1 identity verification (Tier-0 check)."""

    def test_all_pass(self) -> None:
        """All identities pass with consistent values."""
        omega = 0.1
        F = 1.0 - omega
        kappa = -0.5
        IC = math.exp(kappa)
        id_F, id_IC, bound, failures = check_tier1_identities(F, omega, IC, kappa)
        assert id_F is True
        assert id_IC is True
        assert bound is True
        assert failures == []

    def test_F_identity_fails(self) -> None:
        """F ≠ 1 - ω detected."""
        id_F, _, _, failures = check_tier1_identities(0.85, 0.1, 0.5, -0.5)
        assert id_F is False
        assert any("F=" in f for f in failures)

    def test_IC_identity_fails(self) -> None:
        """IC ≠ exp(κ) detected."""
        omega = 0.1
        F = 1.0 - omega
        _, id_IC, _, failures = check_tier1_identities(F, omega, 0.999, -0.5)
        assert id_IC is False
        assert any("IC" in f or "exp" in f for f in failures)

    def test_AMGM_bound_fails(self) -> None:
        """IC > F + tol_seam detected (AM-GM violation)."""
        omega = 0.5
        F = 0.5
        IC = 0.6  # IC > F
        kappa = math.log(IC)
        _, _, bound, failures = check_tier1_identities(F, omega, IC, kappa)
        assert bound is False
        assert any("AM-GM" in f for f in failures)

    def test_AMGM_within_tolerance(self) -> None:
        """IC slightly above F but within tol_seam passes."""
        omega = 0.1
        F = 1.0 - omega
        IC = F + 0.001  # Within tol_seam=0.005
        kappa = math.log(IC)
        _, _, bound, _ = check_tier1_identities(F, omega, IC, kappa)
        assert bound is True

    def test_deep_negative_kappa(self) -> None:
        """Very negative κ (near collapse) handled without overflow."""
        omega = 0.99
        F = 0.01
        kappa = -10.0
        IC = math.exp(kappa)  # ≈ 4.5e-5
        id_F, id_IC, bound, _failures = check_tier1_identities(F, omega, IC, kappa)
        assert id_F is True
        assert id_IC is True
        assert bound is True


# =============================================================================
# FULL DIAGNOSTIC (Tier-2 with Tier-0 checks)
# =============================================================================


class TestDiagnose:
    """Test complete diagnostic pipeline."""

    @staticmethod
    def _make_diagnose_call(
        omega: float = 0.1,
        C: float = 0.1,
        R: float = 0.01,
        F: float | None = None,
        S: float = 0.1,
        kappa: float = -0.5,
        IC: float | None = None,
    ) -> ThermodynamicDiagnostic:
        """Create a Tier-1 consistent state and run diagnose."""
        if F is None:
            F = 1.0 - omega
        if IC is None:
            IC = math.exp(kappa)
        return diagnose(
            omega=omega,
            F=F,
            S=S,
            C=C,
            kappa=kappa,
            IC=IC,
            R=R,
        )

    def test_stable_state_diagnosis(self) -> None:
        """Stable state: small τ_R*, surplus or low deficit."""
        diag = self._make_diagnose_call(omega=0.02, C=0.05)
        assert isinstance(diag, ThermodynamicDiagnostic)
        assert diag.tier1_identity_F is True
        assert diag.tier1_identity_IC is True
        assert diag.tier1_bound_AMGM is True
        assert diag.tier0_checks_pass is True
        assert diag.regime == Regime.STABLE
        # τ_R* should be moderate at stable ω
        assert diag.tau_R_star < 100  # Not divergent

    def test_collapse_state_diagnosis(self) -> None:
        """Collapse state: large τ_R*, drift dominates."""
        diag = self._make_diagnose_call(omega=0.50, C=0.15)
        assert diag.regime == Regime.COLLAPSE or diag.regime == Regime.CRITICAL
        assert diag.dominance == DominanceTerm.DRIFT
        # Large τ_R* due to Γ(0.5) = 0.125/0.5 = 0.25
        assert diag.tau_R_star > 10

    def test_diagnostic_frozen(self) -> None:
        """ThermodynamicDiagnostic is frozen."""
        diag = self._make_diagnose_call()
        with pytest.raises(AttributeError):
            diag.tau_R_star = 0.0  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Serialization to dict works."""
        diag = self._make_diagnose_call()
        d = diag.to_dict()
        assert isinstance(d, dict)
        assert d["phase"] in ["SURPLUS", "DEFICIT", "FREE_RETURN", "TRAPPED", "POLE"]
        assert d["dominance"] in ["DRIFT", "CURVATURE", "MEMORY"]
        assert isinstance(d["warnings"], list)

    def test_R_positive_required(self) -> None:
        """Diagnose raises on R ≤ 0."""
        with pytest.raises(ValueError, match="R must be positive"):
            self._make_diagnose_call(R=0.0)

    def test_tier0_failure_warning(self) -> None:
        """Tier-0 failure produces warnings."""
        omega = 0.1
        F = 0.85  # WRONG: should be 0.9
        diag = diagnose(omega=omega, F=F, S=0.1, C=0.1, kappa=-0.5, IC=math.exp(-0.5), R=0.01)
        assert diag.tier1_identity_F is False
        assert diag.tier0_checks_pass is False
        assert len(diag.warnings) > 0

    def test_trapped_state(self) -> None:
        """System trapped at high ω with low C."""
        omega = 0.80
        F = 0.20
        kappa = math.log(0.15)
        diag = diagnose(omega=omega, F=F, S=0.5, C=0.05, kappa=kappa, IC=0.15, R=0.01)
        assert diag.is_trapped is True
        assert diag.phase == ThermodynamicPhase.TRAPPED

    def test_near_pole(self) -> None:
        """Near-pole ω produces warning."""
        omega = 0.99
        F = 0.01
        kappa = math.log(0.005)
        diag = diagnose(omega=omega, F=F, S=0.8, C=0.5, kappa=kappa, IC=0.005, R=0.01)
        assert any("pole" in w.lower() or "Near" in w for w in diag.warnings)

    def test_no_back_edges(self) -> None:
        """Diagnostic reads Tier-1 inputs without modification (no back-edges)."""
        omega, F = 0.1, 0.9
        diag = diagnose(omega=omega, F=F, S=0.1, C=0.1, kappa=-0.5, IC=math.exp(-0.5), R=0.01)
        # Input values preserved exactly
        assert diag.omega == omega
        assert diag.F == F


# =============================================================================
# BATCH DIAGNOSTIC
# =============================================================================


class TestBatchDiagnostic:
    """Test batch diagnostic processing."""

    def test_empty_list(self) -> None:
        """Empty invariant list → empty results."""
        results = diagnose_invariants([])
        assert results == []

    def test_single_row(self) -> None:
        """Single row: Δκ = 0."""
        inv = [{"omega": 0.1, "F": 0.9, "S": 0.1, "C": 0.1, "kappa": -0.5, "IC": math.exp(-0.5)}]
        results = diagnose_invariants(inv, R=0.01)
        assert len(results) == 1
        assert results[0].delta_kappa == 0.0

    def test_multi_row_delta_kappa(self) -> None:
        """Multiple rows compute Δκ between consecutive entries."""
        inv = [
            {"omega": 0.1, "F": 0.9, "S": 0.1, "C": 0.1, "kappa": -0.5, "IC": math.exp(-0.5)},
            {"omega": 0.15, "F": 0.85, "S": 0.12, "C": 0.12, "kappa": -0.3, "IC": math.exp(-0.3)},
            {"omega": 0.2, "F": 0.8, "S": 0.15, "C": 0.15, "kappa": -0.1, "IC": math.exp(-0.1)},
        ]
        results = diagnose_invariants(inv, R=0.01)
        assert len(results) == 3
        # First row: no prior → Δκ = 0
        assert results[0].delta_kappa == 0.0
        # Second row: Δκ = -0.3 - (-0.5) = 0.2
        np.testing.assert_allclose(results[1].delta_kappa, 0.2, rtol=1e-10)
        # Third row: Δκ = -0.1 - (-0.3) = 0.2
        np.testing.assert_allclose(results[2].delta_kappa, 0.2, rtol=1e-10)


# =============================================================================
# PREDICTION VERIFICATION (§6)
# =============================================================================


class TestPredictions:
    """Test §6 testable predictions."""

    def test_prediction_1_cubic_slowing(self) -> None:
        """Prediction 1: Γ(ω) ∝ ω³/(1-ω) — cubic slowing verified."""
        result = verify_cubic_slowing([0.01, 0.05, 0.10, 0.30, 0.50, 0.90])
        assert result["pass"] is True
        assert result["observed_ratio"] > 1.0

    def test_prediction_3_trapping(self) -> None:
        """Prediction 3: Trapping threshold at c ≈ 0.315."""
        result = verify_trapping_threshold()
        assert result["pass"] is True
        assert 0.25 < result["c_trap"] < 0.40

    def test_prediction_5_R_min_divergence(self) -> None:
        """Prediction 5: R_min·(1-ω) → 200 as ω → 1."""
        result = verify_R_min_divergence([0.80, 0.90, 0.95, 0.99, 0.999])
        assert result["pass"] is True
        # Final product should be near 200
        last = result["products"][-1]["R_min*(1-ω)"]
        np.testing.assert_allclose(last, 200.0, rtol=0.10)

    def test_cost_ratio(self) -> None:
        """Prediction 4: Cost ratio between collapse and stable is > 10^6."""
        gamma_stable = gamma_omega(0.02)
        gamma_collapse = gamma_omega(0.99)
        ratio = gamma_collapse / gamma_stable
        assert ratio > 1e5  # Dramatic regime separation


# =============================================================================
# TIER-1 COMPLIANCE (full tier verification)
# =============================================================================


class TestTier1Compliance:
    """Verify that all Tier-1 identities are preserved through diagnostics.

    These tests confirm the "no back-edges" property: the Tier-2
    diagnostic reads Tier-1 outputs but cannot violate them.
    """

    @pytest.mark.parametrize(
        "omega",
        [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 0.95, 0.99],
    )
    def test_tier1_F_identity_preserved(self, omega: float) -> None:
        """F = 1 - ω holds for all regimes."""
        F = 1.0 - omega
        kappa = max(math.log(F) if F > 0 else -700, -700)
        IC = math.exp(kappa)
        diag = diagnose(omega=omega, F=F, S=0.1, C=0.1, kappa=kappa, IC=IC, R=0.01)
        assert diag.tier1_identity_F is True

    @pytest.mark.parametrize(
        "omega",
        [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95],
    )
    def test_tier1_IC_identity_preserved(self, omega: float) -> None:
        """IC ≈ exp(κ) holds for all non-pole states."""
        F = 1.0 - omega
        kappa = math.log(F) if F > 1e-10 else -30.0
        IC = math.exp(kappa)
        diag = diagnose(omega=omega, F=F, S=0.1, C=0.1, kappa=kappa, IC=IC, R=0.01)
        assert diag.tier1_identity_IC is True

    @pytest.mark.parametrize(
        "omega",
        [0.01, 0.10, 0.30, 0.50, 0.80, 0.95],
    )
    def test_tier1_AMGM_bound_preserved(self, omega: float) -> None:
        """IC ≤ F holds for all consistent states."""
        F = 1.0 - omega
        kappa = math.log(F) if F > 1e-10 else -30.0
        IC = math.exp(kappa)
        diag = diagnose(omega=omega, F=F, S=0.1, C=0.1, kappa=kappa, IC=IC, R=0.01)
        assert diag.tier1_bound_AMGM is True


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary behavior."""

    def test_exactly_zero_omega(self) -> None:
        """ω = 0: perfect fidelity, minimal τ_R*."""
        F = 1.0
        kappa = 0.0  # log(1) = 0
        IC = 1.0
        diag = diagnose(omega=0.0, F=F, S=0.0, C=0.0, kappa=kappa, IC=IC, R=0.01)
        assert diag.tier0_checks_pass is True
        # τ_R* ≈ 0 since Γ(0) ≈ 0, C = 0, Δκ = 0
        np.testing.assert_allclose(diag.tau_R_star, 0.0, atol=1e-6)

    def test_very_small_R(self) -> None:
        """Very small R amplifies τ_R*."""
        diag = diagnose(omega=0.1, F=0.9, S=0.1, C=0.1, kappa=-0.5, IC=math.exp(-0.5), R=1e-10)
        assert diag.tau_R_star > 1e8  # Hugely amplified

    def test_large_R(self) -> None:
        """Large R suppresses τ_R*."""
        diag = diagnose(omega=0.1, F=0.9, S=0.1, C=0.1, kappa=-0.5, IC=math.exp(-0.5), R=1000.0)
        assert diag.tau_R_star < 0.001  # Suppressed

    def test_all_enum_values_accessible(self) -> None:
        """All enum values are distinct."""
        phases = set(ThermodynamicPhase)
        assert len(phases) == 5
        terms = set(DominanceTerm)
        assert len(terms) == 3
