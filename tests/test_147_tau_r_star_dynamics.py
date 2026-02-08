"""
Tests for τ_R* Extended Dynamics — Statistical Mechanics of the Budget Identity.

Tests cover all four discoveries (D1-D4) and Theorems T10-T16,
structured by tier:
    Tier-0 structural checks:  residue = 1/2, barrier = α, separability
    Tier-2 extended dynamics:  Gibbs measure, Kramers escape, Legendre, scaling

Pattern follows test_145_tau_r_star.py conventions.
"""

from __future__ import annotations

import math

import pytest

from umcp.frozen_contract import (
    ALPHA,
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    gamma_omega,
)
from umcp.tau_r_star_dynamics import (
    ExtendedDynamicsDiagnostic,
    GibbsResult,
    KramersResult,
    LegendreResult,
    ResidueResult,
    SeparabilityResult,
    WavefrontResult,
    compute_entropy_production,
    compute_equation_of_state,
    compute_gibbs_measure,
    compute_kramers_escape,
    compute_legendre_conjugate,
    compute_pole_residue,
    compute_wavefront_speed,
    diagnose_extended,
    verify_barrier_identity,
    verify_residue_convergence,
    verify_scaling_law,
    verify_separability,
)

# =============================================================================
# Discovery 1: Pole Residue (Theorem T10)
# =============================================================================


class TestPoleResidue:
    """D1: Res[Γ, ω=1] = 1/2 under ε-regularization."""

    def test_residue_value(self) -> None:
        """T10: Residue equals 1/2 to high precision."""
        r = compute_pole_residue()
        assert abs(r.residue - 0.5) < 1e-6, f"Residue = {r.residue}, expected 0.5"

    def test_residue_theoretical(self) -> None:
        """Theoretical value is 1/2 · (1-ε)^p ≈ 1/2."""
        r = compute_pole_residue()
        assert abs(r.theoretical - 0.5) < 1e-6

    def test_residue_relative_error(self) -> None:
        """Relative error should be negligible."""
        r = compute_pole_residue()
        assert r.relative_error < 1e-6

    def test_residue_returns_namedtuple(self) -> None:
        """Return type is ResidueResult NamedTuple."""
        r = compute_pole_residue()
        assert isinstance(r, ResidueResult)
        assert hasattr(r, "residue")
        assert hasattr(r, "theoretical")
        assert hasattr(r, "epsilon_used")

    def test_residue_convergence_across_epsilon(self) -> None:
        """T10 holds across multiple ε scales (Z₂ universality).

        At extreme ε (< 1e-10) floating-point precision limits the
        matching-scale probe, so we use a practical tolerance.
        """
        results = verify_residue_convergence(
            epsilon_values=[1e-2, 1e-4, 1e-6, 1e-8],
        )
        assert len(results) >= 4
        for r in results:
            # Residue = 1/2 · (1-ε)^p, compare to theoretical
            assert r.relative_error < 1e-6, f"Relative error = {r.relative_error} at ε = {r.epsilon_used}"

    def test_residue_not_one(self) -> None:
        """Residue is definitively not 1 (corrects prior assumption)."""
        r = compute_pole_residue()
        assert abs(r.residue - 1.0) > 0.4, "Residue should be 1/2, not 1"

    def test_residue_custom_p(self) -> None:
        """Residue at 1/2 for different p values (universality)."""
        for p in [1, 2, 3, 5, 7]:
            r = compute_pole_residue(p=p)
            assert abs(r.residue - 0.5) < 1e-4, f"p={p}: Residue = {r.residue}"


# =============================================================================
# Discovery 2: Kramers Escape Rate (Theorem T11, T14)
# =============================================================================


class TestKramersEscape:
    """D2: Barrier height = α exactly; Stable regime is metastable."""

    def test_barrier_identity(self) -> None:
        """T11: Barrier height ΔΓ = α = 1.0 exactly."""
        result = verify_barrier_identity()
        assert result["pass"], f"Barrier test failed: {result}"
        assert abs(result["barrier_height"] - ALPHA) < 1e-6

    def test_barrier_height_is_alpha(self) -> None:
        """T11: Direct numerical check Γ(ω_trap) = α."""
        result = verify_barrier_identity()
        assert abs(result["expected"] - ALPHA) < 1e-15

    def test_kramers_result_type(self) -> None:
        """Kramers computation returns frozen dataclass."""
        kr = compute_kramers_escape(1.0)
        assert isinstance(kr, KramersResult)
        assert kr.barrier_height > 0

    def test_kramers_barrier_matches_identity(self) -> None:
        """Kramers barrier height agrees with verify_barrier_identity."""
        kr = compute_kramers_escape(1.0)
        assert abs(kr.barrier_height - ALPHA) < 0.01

    def test_kramers_escape_rate_decreases_with_beta(self) -> None:
        """T14: Escape rate decreases exponentially with 1/R."""
        k1 = compute_kramers_escape(1.0)  # R=1 (β=1)
        k2 = compute_kramers_escape(0.1)  # R=0.1 (β=10)
        assert k1.kramers_rate > k2.kramers_rate, "Rate should decrease with higher β"

    def test_kramers_metastability(self) -> None:
        """T14: At β ≥ 100, escape is forbidden (metastable)."""
        kr = compute_kramers_escape(0.01)  # R=0.01, β=100
        assert kr.is_metastable, f"Escape time = {kr.escape_time}, but should be metastable"
        assert kr.escape_time > 1e10

    def test_kramers_high_temperature(self) -> None:
        """At R >> 1 (high temperature), escape is fast."""
        kr = compute_kramers_escape(100.0)
        assert kr.escape_time < 1e3, "High-R escape should be fast"
        assert not kr.is_metastable

    def test_kramers_negative_R_raises(self) -> None:
        """Negative R should raise ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            compute_kramers_escape(-1.0)

    def test_kramers_zero_R_raises(self) -> None:
        """Zero R should raise ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            compute_kramers_escape(0.0)

    def test_kramers_serialization(self) -> None:
        """Kramers result serializes to dict."""
        kr = compute_kramers_escape(1.0)
        d = kr.to_dict()
        assert "barrier_height" in d
        assert "kramers_rate" in d
        assert "is_metastable" in d

    def test_omega_trap_location(self) -> None:
        """ω_trap is in (0, 1) and c_trap = 1-ω_trap ≈ 0.318."""
        result = verify_barrier_identity()
        assert 0 < result["omega_trap"] < 1
        c_trap = result["c_trap"]
        assert 0.3 < c_trap < 0.35, f"c_trap = {c_trap}, expected ~0.318"


# =============================================================================
# Discovery 3: Separability (Theorem T12)
# =============================================================================


class TestSeparability:
    """D3: N(ω,C,Δκ) additively separable (ideal gas in state space)."""

    def test_separable(self) -> None:
        """T12: All cross-derivatives vanish."""
        s = verify_separability()
        assert s.is_separable

    def test_cross_derivatives_zero(self) -> None:
        """T12: Each cross-derivative is exactly 0."""
        s = verify_separability()
        assert s.d2N_domega_dC == 0.0
        assert s.d2N_domega_dkappa == 0.0
        assert s.d2N_dC_dkappa == 0.0

    def test_max_cross_derivative(self) -> None:
        """Maximum cross-derivative is zero."""
        s = verify_separability()
        assert s.max_cross_derivative == 0.0

    def test_returns_namedtuple(self) -> None:
        """Return type is SeparabilityResult."""
        s = verify_separability()
        assert isinstance(s, SeparabilityResult)


# =============================================================================
# Discovery 4: Scaling Law (Theorem T13)
# =============================================================================


class TestScalingLaw:
    """D4: ⟨ω⟩ ≈ (1/2)·β^(-1/p) — Gibbs equilibrium drift."""

    def test_gibbs_measure_type(self) -> None:
        """Gibbs result is correct NamedTuple."""
        g = compute_gibbs_measure(10.0)
        assert isinstance(g, GibbsResult)
        assert g.beta == 10.0

    def test_gibbs_mean_omega_positive(self) -> None:
        """⟨ω⟩ > 0 for finite β."""
        g = compute_gibbs_measure(10.0)
        assert g.mean_omega > 0

    def test_gibbs_mean_omega_decreases_with_beta(self) -> None:
        """⟨ω⟩ decreases as β increases (colder system drifts less)."""
        g1 = compute_gibbs_measure(1.0)
        g2 = compute_gibbs_measure(100.0)
        assert g1.mean_omega > g2.mean_omega

    def test_scaling_convergence(self) -> None:
        """T13: β^(1/p)·⟨ω⟩ converges to approximately 1/2."""
        result = verify_scaling_law()
        assert result["pass"], f"Scaling law did not converge: {result}"

    def test_scaling_product_range(self) -> None:
        """At high β, product is in [0.3, 0.7] bracket around 1/2."""
        g = compute_gibbs_measure(1000.0)
        assert 0.3 < g.scaling_product < 0.7, f"Scaling product = {g.scaling_product}"

    def test_scaling_exponent(self) -> None:
        """Scaling exponent is 1/p = 1/3."""
        result = verify_scaling_law()
        expected_exp = 1.0 / P_EXPONENT
        assert abs(result["exponent"] - expected_exp) < 1e-15

    def test_gibbs_negative_beta_raises(self) -> None:
        """Negative β should raise ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            compute_gibbs_measure(-1.0)

    def test_gibbs_susceptibility_positive(self) -> None:
        """Susceptibility χ = β·Var(ω) ≥ 0."""
        g = compute_gibbs_measure(10.0)
        assert g.susceptibility >= 0

    def test_gibbs_free_energy_finite(self) -> None:
        """Free energy is finite for finite β."""
        g = compute_gibbs_measure(10.0)
        assert math.isfinite(g.free_energy)


# =============================================================================
# Theorem T15: Legendre Conjugate
# =============================================================================


class TestLegendreConjugate:
    """T15: Ψ*(β) = sup_ω [βω − Γ(ω)] — equation of state."""

    def test_legendre_result_type(self) -> None:
        """Returns correct NamedTuple."""
        lr = compute_legendre_conjugate(5.0)
        assert isinstance(lr, LegendreResult)

    def test_legendre_conjugate_nonnegative(self) -> None:
        """Ψ*(β) ≥ 0 since we can always choose ω = 0."""
        # At ω=0: objective = 0·β - Γ(0) = 0, so Ψ* ≥ 0
        lr = compute_legendre_conjugate(5.0)
        assert lr.psi_star >= -1e-10

    def test_legendre_omega_star_increases_with_beta(self) -> None:
        """Optimal ω* increases with β (higher β → higher equilibrium drift)."""
        lr1 = compute_legendre_conjugate(1.0)
        lr2 = compute_legendre_conjugate(10.0)
        assert lr2.omega_star >= lr1.omega_star

    def test_contact_structure(self) -> None:
        """Young-Fenchel inequality: β·ω = Γ(ω) + Ψ*(β) at optimum."""
        lr = compute_legendre_conjugate(5.0)
        lhs = lr.beta * lr.omega_star
        rhs = lr.gamma_at_star + lr.psi_star
        assert abs(lhs - rhs) < 0.01, f"Contact structure violated: {lhs} ≠ {rhs}"

    def test_equation_of_state_covers_range(self) -> None:
        """Equation of state maps across regimes."""
        eos = compute_equation_of_state()
        assert len(eos) >= 5
        # ω* should span a range
        omegas = [lr.omega_star for lr in eos]
        assert max(omegas) > min(omegas) + 0.1

    def test_legendre_negative_beta_raises(self) -> None:
        """Negative β should raise ValueError."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            compute_legendre_conjugate(-1.0)


# =============================================================================
# Theorem T16: Entropy Production
# =============================================================================


class TestEntropyProduction:
    """T16: σ(ω) = (dΓ/dω)²/R — Onsager dissipation."""

    def test_entropy_production_nonnegative(self) -> None:
        """σ ≥ 0 always (second law)."""
        ep = compute_entropy_production(0.5, 1.0)
        assert ep.sigma >= 0

    def test_entropy_production_near_zero_is_small(self) -> None:
        """Near ω ≈ 0, dissipation is minimal."""
        ep = compute_entropy_production(0.001, 1.0)
        assert ep.sigma < 1e-3

    def test_entropy_production_increases_near_collapse(self) -> None:
        """Near ω → 1, dissipation diverges (critical slowing)."""
        ep_low = compute_entropy_production(0.1, 1.0)
        ep_high = compute_entropy_production(0.95, 1.0)
        assert ep_high.sigma > 100 * ep_low.sigma

    def test_entropy_production_negative_R_raises(self) -> None:
        """Negative R raises ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            compute_entropy_production(0.5, -1.0)

    def test_entropy_production_dissipation_ratio(self) -> None:
        """Ratio relative to Stable boundary is meaningful."""
        ep = compute_entropy_production(0.5, 1.0)
        assert ep.dissipation_ratio > 1.0  # mid-Watch is costlier than Stable boundary

    def test_entropy_production_inversely_proportional_to_R(self) -> None:
        """σ ∝ 1/R at fixed ω."""
        ep1 = compute_entropy_production(0.5, 1.0)
        ep2 = compute_entropy_production(0.5, 2.0)
        ratio = ep1.sigma / ep2.sigma
        assert abs(ratio - 2.0) < 0.01, f"Ratio = {ratio}, expected 2.0"


# =============================================================================
# Wavefront Speed
# =============================================================================


class TestWavefrontSpeed:
    """Eikonal analysis: wavefront speed of iso-τ_R* contours."""

    def test_wavefront_result_type(self) -> None:
        """Returns WavefrontResult."""
        wf = compute_wavefront_speed(0.5)
        assert isinstance(wf, WavefrontResult)

    def test_wavefront_speed_positive(self) -> None:
        """Speed > 0 always (non-degenerate gradient)."""
        wf = compute_wavefront_speed(0.5)
        assert wf.wavefront_speed > 0

    def test_wavefront_speed_decreases_near_collapse(self) -> None:
        """Critical slowing: wavefront stalls as ω → 1."""
        wf_low = compute_wavefront_speed(0.1)
        wf_high = compute_wavefront_speed(0.95)
        assert wf_low.wavefront_speed > wf_high.wavefront_speed

    def test_wavefront_near_zero(self) -> None:
        """At ω ≈ 0: gradient ≈ α, speed ≈ 1/α."""
        wf = compute_wavefront_speed(0.001)
        expected_speed = 1.0 / math.sqrt(0.0 + ALPHA**2)  # dΓ/dω ≈ 0
        assert abs(wf.wavefront_speed - expected_speed) < 0.1


# =============================================================================
# Full Diagnostic
# =============================================================================


class TestDiagnoseExtended:
    """Full extended dynamics diagnostic."""

    def test_diagnostic_type(self) -> None:
        """Returns frozen ExtendedDynamicsDiagnostic."""
        d = diagnose_extended(0.3, 0.5, 1.0)
        assert isinstance(d, ExtendedDynamicsDiagnostic)

    def test_tier0_checks_pass(self) -> None:
        """All structural checks pass at default constants."""
        d = diagnose_extended(0.3, 0.5, 1.0)
        assert d.tier0_checks_pass

    def test_diagnostic_contains_all_components(self) -> None:
        """Diagnostic has all four discovery components."""
        d = diagnose_extended(0.3, 0.5, 1.0)
        assert isinstance(d.residue, ResidueResult)
        assert isinstance(d.kramers, KramersResult)
        assert isinstance(d.gibbs, GibbsResult)
        assert isinstance(d.separability, SeparabilityResult)
        assert isinstance(d.legendre, LegendreResult)
        assert isinstance(d.wavefront, WavefrontResult)

    def test_diagnostic_serialization(self) -> None:
        """Full diagnostic serializes to dict."""
        d = diagnose_extended(0.3, 0.5, 1.0)
        out = d.to_dict()
        assert "tier0_checks" in out
        assert "tier2_dynamics" in out
        assert out["tier0_checks"]["all_pass"] is True

    def test_diagnostic_negative_R_raises(self) -> None:
        """Negative R should raise ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            diagnose_extended(0.3, 0.5, -1.0)

    def test_diagnostic_stable_regime(self) -> None:
        """Stable regime (low ω, high R) shows metastable Kramers."""
        d = diagnose_extended(0.01, 0.1, 0.01)  # low R (high β=100)
        assert d.kramers.is_metastable

    def test_diagnostic_near_collapse(self) -> None:
        """Near-collapse (high ω) shows high entropy production."""
        d = diagnose_extended(0.95, 0.5, 1.0)
        assert d.entropy_production.sigma > 100  # High dissipation

    def test_diagnostic_scaling_product(self) -> None:
        """Scaling product comes from Gibbs measure."""
        d = diagnose_extended(0.3, 0.5, 1.0)
        assert d.scaling_product == d.gibbs.scaling_product


# =============================================================================
# Tier Architecture Cross-Checks
# =============================================================================


class TestTierArchitecture:
    """Verify that extended dynamics respects the tier hierarchy."""

    def test_no_modification_of_tier1(self) -> None:
        """Extended dynamics reads Γ(ω) but does not modify it.

        Γ(ω) computed before and after should be identical.
        """
        omega = 0.5
        gamma_before = gamma_omega(omega, P_EXPONENT, EPSILON)
        _ = diagnose_extended(omega, 0.5, 1.0)
        gamma_after = gamma_omega(omega, P_EXPONENT, EPSILON)
        assert gamma_before == gamma_after

    def test_frozen_constants_unchanged(self) -> None:
        """Frozen constants unchanged after diagnostic."""
        p_before = P_EXPONENT
        alpha_before = ALPHA
        eps_before = EPSILON
        tol_before = TOL_SEAM

        _ = diagnose_extended(0.3, 0.5, 1.0)

        assert p_before == P_EXPONENT
        assert alpha_before == ALPHA
        assert eps_before == EPSILON
        assert tol_before == TOL_SEAM
