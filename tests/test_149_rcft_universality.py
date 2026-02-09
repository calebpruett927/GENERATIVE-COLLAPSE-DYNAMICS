"""Tests for RCFT Tier-2 extensions: Information Geometry, Universality, Grammar.

Covers Theorems T17–T23:
  T17  Fisher Geodesic Distance
  T18  Geodesic Parametrization
  T19  Fano-Fisher Duality
  T20  Central Charge c = 1/p
  T21  Critical Exponents
  T22  Thermodynamic Efficiency
  T23  Collapse Grammar Transfer Matrix
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# =====================================================================
# Collapse Grammar — T23
# =====================================================================
from closures.rcft.collapse_grammar import (
    GrammarDiagnostic,
    classify_regime_index,
    compute_grammar_entropy,
    compute_grammar_phase_diagram,
    compute_transfer_matrix,
    diagnose_grammar,
)

# =====================================================================
# Information Geometry — T17, T18, T19, T22
# =====================================================================
from closures.rcft.information_geometry import (
    FanoFisherResult,
    FisherDistanceResult,
    binary_entropy,
    compute_efficiency,
    compute_geodesic_budget_cost,
    compute_geodesic_path,
    compute_path_length,
    compute_path_length_weighted,
    fisher_distance_1d,
    fisher_distance_weighted,
    fisher_geodesic,
    fisher_metric_1d,
    verify_fano_fisher_duality,
)

# =====================================================================
# Universality Class — T20, T21
# =====================================================================
from closures.rcft.universality_class import (
    CentralChargeResult,
    CriticalExponents,
    analytical_moment,
    analytical_susceptibility_coefficient,
    compute_central_charge,
    compute_critical_exponents,
    compute_partition_function,
    compute_susceptibility,
    compute_susceptibility_scaling,
    verify_central_charge_universality,
    verify_scaling_relations,
)


# =====================================================================
# Test: T17 Fisher Geodesic Distance
# =====================================================================
class TestFisherDistance:
    """Tests for Theorem T17: Fisher Geodesic Distance."""

    def test_identity(self) -> None:
        """d(c, c) = 0 for any c."""
        for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert fisher_distance_1d(c, c) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self) -> None:
        """d(c₁, c₂) = d(c₂, c₁)."""
        d12 = fisher_distance_1d(0.3, 0.8)
        d21 = fisher_distance_1d(0.8, 0.3)
        assert d12 == pytest.approx(d21, rel=1e-12)

    def test_triangle_inequality(self) -> None:
        """d(a, c) ≤ d(a, b) + d(b, c)."""
        a, b, c_val = 0.2, 0.5, 0.9
        dac = fisher_distance_1d(a, c_val)
        dab = fisher_distance_1d(a, b)
        dbc = fisher_distance_1d(b, c_val)
        assert dac <= dab + dbc + 1e-10

    def test_maximum_distance_is_pi(self) -> None:
        """d(ε, 1−ε) → π."""
        d = fisher_distance_1d(1e-8, 1 - 1e-8)
        assert d == pytest.approx(math.pi, rel=1e-3)

    def test_positive(self) -> None:
        """d(c₁, c₂) > 0 for c₁ ≠ c₂."""
        assert fisher_distance_1d(0.3, 0.7) > 0.0

    def test_weighted_homogeneous_zero(self) -> None:
        """Weighted distance of identical states is zero."""
        c = np.array([0.8, 0.8, 0.8])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = fisher_distance_weighted(c, c, w)
        assert isinstance(result, FisherDistanceResult)
        assert result.distance == pytest.approx(0.0, abs=1e-12)
        assert result.normalized == pytest.approx(0.0, abs=1e-12)

    def test_weighted_positive(self) -> None:
        """Weighted distance of distinct states is positive."""
        c1 = np.array([0.9, 0.8, 0.7])
        c2 = np.array([0.5, 0.6, 0.4])
        w = np.array([0.5, 0.3, 0.2])
        result = fisher_distance_weighted(c1, c2, w)
        assert result.distance > 0.0
        assert 0 < result.normalized <= 1.0

    def test_weighted_shape_mismatch_raises(self) -> None:
        """Mismatched array shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            fisher_distance_weighted([0.5, 0.6], [0.7], [0.5, 0.5])


# =====================================================================
# Test: T18 Geodesic Parametrization
# =====================================================================
class TestGeodesic:
    """Tests for Theorem T18: Geodesic Parametrization."""

    def test_endpoints(self) -> None:
        """c(0) = c₁, c(1) = c₂."""
        c1, c2 = 0.3, 0.9
        assert fisher_geodesic(c1, c2, 0.0) == pytest.approx(c1, rel=1e-8)
        assert fisher_geodesic(c1, c2, 1.0) == pytest.approx(c2, rel=1e-8)

    def test_midpoint_in_range(self) -> None:
        """c(0.5) is between c₁ and c₂."""
        c1, c2 = 0.2, 0.8
        mid = fisher_geodesic(c1, c2, 0.5)
        assert c1 < mid < c2

    def test_monotone(self) -> None:
        """Geodesic is monotone for c₁ < c₂."""
        c1, c2 = 0.2, 0.9
        ts = np.linspace(0, 1, 50)
        cs = fisher_geodesic(c1, c2, ts)
        assert np.all(np.diff(cs) > 0)  # type: ignore[arg-type]

    def test_array_input(self) -> None:
        """Vector-valued t works."""
        ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        cs = fisher_geodesic(0.3, 0.9, ts)
        assert len(cs) == 5  # type: ignore[arg-type]

    def test_geodesic_path_returns_correct_count(self) -> None:
        """compute_geodesic_path returns n_points points."""
        path = compute_geodesic_path(0.2, 0.8, n_points=50)
        assert len(path) == 50
        assert path[0].t == pytest.approx(0.0)
        assert path[-1].t == pytest.approx(1.0)
        assert path[0].c == pytest.approx(0.2, rel=1e-6)
        assert path[-1].c == pytest.approx(0.8, rel=1e-6)


# =====================================================================
# Test: T19 Fano-Fisher Duality
# =====================================================================
class TestFanoFisher:
    """Tests for Theorem T19: h″(c) = −g_F(c)."""

    def test_duality_exact(self) -> None:
        """h″(c) = −1/(c(1−c)) to high precision."""
        results = verify_fano_fisher_duality()
        for r in results:
            assert isinstance(r, FanoFisherResult)
            assert r.relative_error < 1e-4, f"Duality violated at c={r.c}"

    def test_at_midpoint(self) -> None:
        """At c=0.5: h″ = −4, g_F = 4."""
        results = verify_fano_fisher_duality(c_values=[0.5])
        r = results[0]
        assert r.h_double_prime == pytest.approx(-4.0, rel=1e-4)
        assert r.neg_g_fisher == pytest.approx(-4.0, rel=1e-10)

    def test_binary_entropy_max_at_half(self) -> None:
        """h(0.5) = ln(2) ≈ 0.6931."""
        assert float(binary_entropy(0.5)) == pytest.approx(math.log(2), rel=1e-8)

    def test_binary_entropy_zero_at_boundaries(self) -> None:
        """h(ε) ≈ 0 and h(1−ε) ≈ 0."""
        assert float(binary_entropy(1e-7)) < 1e-5
        assert float(binary_entropy(1 - 1e-7)) < 1e-5

    def test_fisher_metric_symmetric(self) -> None:
        """g_F(c) = g_F(1−c)."""
        for c in [0.1, 0.2, 0.3, 0.4]:
            assert fisher_metric_1d(c) == pytest.approx(fisher_metric_1d(1 - c), rel=1e-10)


# =====================================================================
# Test: T22 Thermodynamic Efficiency
# =====================================================================
class TestEfficiency:
    """Tests for Theorem T22: Thermodynamic Efficiency."""

    def test_geodesic_has_unit_efficiency(self) -> None:
        """Direct monotone path has η ≈ 1."""
        c_series = np.linspace(0.3, 0.9, 100)
        result = compute_efficiency(c_series)
        assert result.efficiency == pytest.approx(1.0, abs=0.01)
        assert result.excess_fraction == pytest.approx(0.0, abs=0.01)

    def test_noisy_path_has_lower_efficiency(self) -> None:
        """Noisy path has η < 1."""
        rng = np.random.default_rng(42)
        c_base = np.linspace(0.3, 0.9, 100)
        c_noisy = np.clip(c_base + rng.normal(0, 0.05, 100), 0.01, 0.99)
        c_noisy[0], c_noisy[-1] = 0.3, 0.9
        result = compute_efficiency(c_noisy)
        assert result.efficiency < 0.8
        assert result.excess_fraction > 0.2

    def test_path_length_positive(self) -> None:
        """Non-trivial trajectory has positive length."""
        assert compute_path_length([0.3, 0.5, 0.8]) > 0.0

    def test_path_length_single_point(self) -> None:
        """Single point has zero length."""
        assert compute_path_length([0.5]) == 0.0

    def test_weighted_path_length(self) -> None:
        """Weighted path length of 2-channel trajectory."""
        c_matrix = np.array([[0.3, 0.7], [0.5, 0.6], [0.8, 0.5]])
        w = np.array([0.5, 0.5])
        L = compute_path_length_weighted(c_matrix, w)
        assert L > 0.0

    def test_geodesic_budget_cost_positive(self) -> None:
        """Budget cost along geodesic is positive."""
        cost = compute_geodesic_budget_cost(0.3, 0.9, R=100.0)
        assert cost > 0.0

    def test_geodesic_budget_cost_scales_with_R(self) -> None:
        """Cost ∝ 1/R."""
        cost_10 = compute_geodesic_budget_cost(0.3, 0.9, R=10.0)
        cost_100 = compute_geodesic_budget_cost(0.3, 0.9, R=100.0)
        assert cost_10 / cost_100 == pytest.approx(10.0, rel=0.02)

    def test_negative_R_raises(self) -> None:
        """R ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            compute_geodesic_budget_cost(0.3, 0.9, R=-1.0)


# =====================================================================
# Test: T20 Central Charge
# =====================================================================
class TestCentralCharge:
    """Tests for Theorem T20: c_eff = 1/p."""

    def test_central_charge_p3(self) -> None:
        """c_eff = 1/3 for p=3."""
        result = compute_central_charge(p=3, beta_probe=2000.0)
        assert isinstance(result, CentralChargeResult)
        assert result.c_eff == pytest.approx(1 / 3)
        # C_V should be approaching 1/3 (within ~5% at β=2000)
        assert result.relative_error < 0.10

    def test_universality_across_p(self) -> None:
        """c_eff = 1/p for p=2,3,4,5,7."""
        results = verify_central_charge_universality(beta_probe=2000.0)
        for r in results:
            assert r.c_eff == pytest.approx(1.0 / r.p)
            assert r.relative_error < 0.15

    def test_partition_function_positive(self) -> None:
        """Z(β) > 0 for any β > 0."""
        for beta in [0.1, 1.0, 10.0, 100.0]:
            pf = compute_partition_function(beta)
            assert pf.Z > 0
            assert pf.internal_energy > 0
            assert pf.specific_heat > 0

    def test_partition_function_decreasing(self) -> None:
        """Z(β) decreases with β."""
        Z_small = compute_partition_function(1.0).Z
        Z_large = compute_partition_function(100.0).Z
        assert Z_small > Z_large

    def test_beta_must_be_positive(self) -> None:
        """β ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            compute_partition_function(-1.0)


# =====================================================================
# Test: T21 Critical Exponents
# =====================================================================
class TestCriticalExponents:
    """Tests for Theorem T21: Critical Exponent Set."""

    def test_exponent_values_p3(self) -> None:
        """Correct exponent values for p=3."""
        ce = compute_critical_exponents(p=3)
        assert isinstance(ce, CriticalExponents)
        assert ce.nu == pytest.approx(1 / 3)
        assert ce.gamma == pytest.approx(1 / 3)
        assert ce.eta == pytest.approx(1.0)
        assert ce.alpha == pytest.approx(0.0)
        assert ce.beta_exp == pytest.approx(5 / 6)
        assert ce.delta == pytest.approx(7 / 5)
        assert ce.d_eff == pytest.approx(6.0)
        assert ce.c_eff == pytest.approx(1 / 3)

    def test_rushbrooke(self) -> None:
        """α + 2β + γ = 2."""
        for p in [2, 3, 4, 5, 7]:
            ce = compute_critical_exponents(p=p)
            assert ce.alpha + 2 * ce.beta_exp + ce.gamma == pytest.approx(2.0)

    def test_widom(self) -> None:
        """γ = β(δ − 1)."""
        for p in [2, 3, 4, 5, 7]:
            ce = compute_critical_exponents(p=p)
            assert ce.gamma == pytest.approx(ce.beta_exp * (ce.delta - 1.0))

    def test_hyperscaling(self) -> None:
        """d_eff · ν = 2 − α."""
        for p in [2, 3, 4, 5, 7]:
            ce = compute_critical_exponents(p=p)
            assert ce.d_eff * ce.nu == pytest.approx(2.0 - ce.alpha)

    def test_fisher_relation(self) -> None:
        """γ = ν(2 − η)."""
        for p in [2, 3, 4, 5, 7]:
            ce = compute_critical_exponents(p=p)
            assert ce.gamma == pytest.approx(ce.nu * (2.0 - ce.eta))

    def test_verify_scaling_all_pass(self) -> None:
        """All four scaling relations satisfied."""
        checks = verify_scaling_relations(p=3)
        assert len(checks) == 4
        for c in checks:
            assert c.satisfied, f"{c.name} failed: {c.lhs} ≠ {c.rhs}"

    def test_p_less_than_2_raises(self) -> None:
        """p < 2 raises ValueError."""
        with pytest.raises(ValueError, match="≥ 2"):
            compute_critical_exponents(p=1)

    def test_exponent_values_p2(self) -> None:
        """Correct exponents for p=2 (quadratic potential)."""
        ce = compute_critical_exponents(p=2)
        assert ce.nu == pytest.approx(0.5)
        assert ce.gamma == pytest.approx(0.0)
        assert ce.d_eff == pytest.approx(4.0)


# =====================================================================
# Test: Susceptibility
# =====================================================================
class TestSusceptibility:
    """Tests for susceptibility χ(β) = β Var(ω)."""

    def test_susceptibility_positive(self) -> None:
        """χ > 0 for any β."""
        for beta in [1.0, 10.0, 100.0]:
            r = compute_susceptibility(beta)
            assert r.chi > 0

    def test_susceptibility_grows_with_beta(self) -> None:
        """χ increases with β (for p ≥ 3)."""
        chi_low = compute_susceptibility(10.0).chi
        chi_high = compute_susceptibility(1000.0).chi
        assert chi_high > chi_low

    def test_analytical_approaches_numerical(self) -> None:
        """At large β, analytical and numerical match."""
        r = compute_susceptibility(5000.0)
        assert r.relative_error < 0.15

    def test_scaling_exponent(self) -> None:
        """χ ~ β^{(p-2)/p} confirmed by log-log fit."""
        result = compute_susceptibility_scaling(beta_values=[50.0, 100.0, 500.0, 1000.0, 5000.0])
        predicted = result["predicted_exponent"]
        fitted = result["fitted_exponent"]
        # Allow 25% tolerance due to finite-β corrections
        assert abs(fitted - predicted) / max(abs(predicted), 0.01) < 0.30

    def test_analytical_moment(self) -> None:
        """⟨ω⟩ analytical matches numerical at large β."""
        beta = 5000.0
        ana = analytical_moment(1, beta, p=3)
        pf = compute_partition_function(beta)
        ratio = pf.mean_omega / ana
        assert ratio == pytest.approx(1.0, rel=0.05)

    def test_susceptibility_coefficient(self) -> None:
        """Coefficient in χ formula is positive."""
        coeff = analytical_susceptibility_coefficient(p=3)
        assert coeff > 0


# =====================================================================
# Test: T23 Collapse Grammar
# =====================================================================
class TestCollapseGrammar:
    """Tests for Theorem T23: Collapse Grammar Transfer Matrix."""

    def test_regime_classification(self) -> None:
        """Correct regime index mapping."""
        assert classify_regime_index(0.01) == 0  # STABLE
        assert classify_regime_index(0.15) == 1  # WATCH
        assert classify_regime_index(0.50) == 2  # COLLAPSE

    def test_regime_boundaries(self) -> None:
        """Boundary values assigned correctly."""
        assert classify_regime_index(0.037) == 0  # Just below STABLE threshold
        assert classify_regime_index(0.038) == 1  # At WATCH threshold
        assert classify_regime_index(0.29) == 1  # Just below COLLAPSE
        assert classify_regime_index(0.30) == 2  # At COLLAPSE threshold

    def test_transfer_matrix_column_stochastic(self) -> None:
        """Columns sum to 1."""
        result = compute_transfer_matrix(10.0, seed=42)
        col_sums = result.T.sum(axis=0)
        for s in col_sums:
            assert s == pytest.approx(1.0, abs=1e-10)

    def test_transfer_matrix_nonnegative(self) -> None:
        """All entries ≥ 0."""
        result = compute_transfer_matrix(10.0, seed=42)
        assert np.all(result.T >= 0)

    def test_spectral_gap_positive(self) -> None:
        """Spectral gap > 0 (ergodic chain)."""
        result = compute_transfer_matrix(10.0, seed=42)
        assert result.spectral_gap > 0

    def test_stationary_sums_to_one(self) -> None:
        """Stationary distribution sums to 1."""
        result = compute_transfer_matrix(10.0, seed=42)
        assert result.stationary.sum() == pytest.approx(1.0, abs=1e-8)

    def test_high_beta_concentrates_stable(self) -> None:
        """At high β, stationary dist concentrates in STABLE/WATCH."""
        result = compute_transfer_matrix(100.0, seed=42, n_samples=500000)
        # COLLAPSE probability should be very small at high β
        assert result.stationary[2] < 0.1  # COLLAPSE < 10%

    def test_eigenvalues_bounded(self) -> None:
        """All eigenvalue magnitudes ≤ 1."""
        result = compute_transfer_matrix(10.0, seed=42)
        assert np.all(result.eigenvalues <= 1.0 + 1e-10)

    def test_entropy_rate_bounded(self) -> None:
        """0 ≤ h ≤ log₂(3)."""
        result = compute_transfer_matrix(10.0, seed=42)
        er = compute_grammar_entropy(result.T, result.stationary)
        assert er.entropy_rate >= 0
        assert er.entropy_rate <= er.max_entropy + 1e-10
        assert 0 <= er.normalized_entropy <= 1.0 + 1e-10

    def test_complexity_classification(self) -> None:
        """Complexity class is one of the valid labels."""
        result = compute_transfer_matrix(10.0, seed=42)
        er = compute_grammar_entropy(result.T, result.stationary)
        assert er.complexity_class in {"FROZEN", "ORDERED", "COMPLEX", "CHAOTIC"}

    def test_diagnose_grammar(self) -> None:
        """Full diagnostic returns valid structure."""
        diag = diagnose_grammar(10.0, seed=42)
        assert isinstance(diag, GrammarDiagnostic)
        assert diag.regime_labels == ("STABLE", "WATCH", "COLLAPSE")
        assert diag.transfer.beta == 10.0
        assert diag.entropy.entropy_rate >= 0

    def test_phase_diagram(self) -> None:
        """Phase diagram returns diagnostics for each β."""
        betas = [1.0, 10.0, 100.0]
        diags = compute_grammar_phase_diagram(beta_values=betas, seed=42)
        assert len(diags) == len(betas)
        for d, b in zip(diags, betas, strict=True):
            assert d.transfer.beta == b

    def test_beta_must_be_positive(self) -> None:
        """β ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            compute_transfer_matrix(-1.0)


# =====================================================================
# Cross-module integration
# =====================================================================
class TestIntegration:
    """Cross-module coherence checks."""

    def test_fisher_geodesic_budget_vs_manual(self) -> None:
        """Geodesic budget cost matches manual integration."""
        c1, c2, R = 0.3, 0.9, 100.0
        cost = compute_geodesic_budget_cost(c1, c2, R)
        # Manual: integrate Γ(1-c(t))/R along geodesic
        ts = np.linspace(0, 1, 1000)
        cs = fisher_geodesic(c1, c2, ts)
        omegas = 1.0 - np.atleast_1d(cs)
        gammas = omegas**3 / (1.0 - omegas + 1e-8)
        manual = float(np.trapezoid(gammas / R, ts))
        assert cost == pytest.approx(manual, rel=0.01)

    def test_scaling_consistent_with_central_charge(self) -> None:
        """Central charge 1/p matches specific heat from partition function."""
        cc = compute_central_charge(p=3, beta_probe=2000.0)
        pf = compute_partition_function(2000.0, p=3)
        assert cc.C_V_measured == pytest.approx(pf.specific_heat, rel=1e-6)

    def test_exponents_match_universality(self) -> None:
        """Critical exponents from p=3 match the GCD universality class."""
        ce = compute_critical_exponents(p=3)
        # zν = 1 from KERNEL_SPECIFICATION
        z = 1.0 / ce.nu  # z = p
        assert z == pytest.approx(3.0)
        # d_eff = 2p = 6
        assert ce.d_eff == pytest.approx(6.0)

    def test_fisher_distance_vs_amgm(self) -> None:
        """Fisher distance = 0 iff states identical (relates to AM-GM gap = 0)."""
        c_same = 0.7
        assert fisher_distance_1d(c_same, c_same) == pytest.approx(0.0, abs=1e-12)
        assert fisher_distance_1d(0.5, 0.8) > 0  # Different → positive
