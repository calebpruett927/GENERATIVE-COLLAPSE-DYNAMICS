"""Tests for closures/quantum_mechanics/ters_near_field.py.

TERS Near-Field Rederivation — 7 theorems + 2 validation checks
+ data-driven analysis (pixel kernel maps, MC uncertainty, second-system).

Test coverage:
    - All 7 core theorems (T-TERS-1 through T-TERS-7)
    - Cross-theorem consistency validation
    - Uncertainty propagation analysis
    - Trace vector construction
    - Channel value bounds
    - Fisher distance ordering
    - Kernel identity preservation
    - Data ingestion from Zenodo .npz files
    - Pixel-level kernel maps (MgP, TCNE, MoS₂)
    - Full Monte Carlo uncertainty (10 000 Gaussian samples)
    - Data-driven parameterisation (replaces estimates)
    - Second-system validation (MoS₂, TCNE full-rigour)
"""

from __future__ import annotations

import numpy as np
import pytest

from closures.quantum_mechanics.ters_near_field import (
    CHANNEL_LABELS,
    CHANNEL_UNCERTAINTIES,
    EPSILON,
    N_CHANNELS,
    TheoremResult,
    build_ters_trace,
    compute_all_fisher_distances,
    compute_fisher_distance_gas_to_surface,
    compute_pixel_kernel_map,
    mgp_data_driven,
    monte_carlo_uncertainty,
    mos2_data_driven,
    mos2_data_driven_analysis,
    run_all_ters_theorems,
    run_all_with_validation,
    tcne_data_driven,
    tcne_data_driven_analysis,
    theorem_T_TERS_1_amgm_decomposition,
    theorem_T_TERS_2_screening_sign_reversal,
    theorem_T_TERS_3_linear_regime,
    theorem_T_TERS_4_positional_illusion,
    theorem_T_TERS_5_periodicity_consistency,
    theorem_T_TERS_6_binding_sensitivity,
    theorem_T_TERS_7_channel_projection,
    uncertainty_propagation_analysis,
    validate_cross_theorem_consistency,
)

# ─── Trace Vector Construction ───


class TestBuildTersTrace:
    """Test trace vector construction."""

    def test_shape(self) -> None:
        c, w, labels = build_ters_trace(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert c.shape == (8,)
        assert w.shape == (8,)
        assert len(labels) == 8

    def test_weights_sum_to_one(self) -> None:
        _, w, _ = build_ters_trace(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert abs(float(np.sum(w)) - 1.0) < 1e-12

    def test_clipping_lower(self) -> None:
        c, _, _ = build_ters_trace(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert float(np.min(c)) >= EPSILON

    def test_clipping_upper(self) -> None:
        c, _, _ = build_ters_trace(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert float(np.max(c)) <= 1.0 - EPSILON

    def test_labels_match(self) -> None:
        _, _, labels = build_ters_trace(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert labels == CHANNEL_LABELS

    def test_channel_count(self) -> None:
        assert N_CHANNELS == 8
        assert len(CHANNEL_LABELS) == 8


# ─── Individual Theorem Tests ───


class TestTTERS1:
    """T-TERS-1: Self/Cross Decomposition as AM-GM Gap."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_1_amgm_decomposition()
        assert result.verdict == "PROVEN"

    def test_all_subtests_pass(self) -> None:
        result = theorem_T_TERS_1_amgm_decomposition()
        assert result.n_failed == 0
        assert result.n_passed == result.n_tests

    def test_surface_gap_larger(self) -> None:
        result = theorem_T_TERS_1_amgm_decomposition()
        assert result.details["gap_surf"] > result.details["gap_gas"]

    def test_fisher_identity_tightened(self) -> None:
        """Fisher identity tolerance was tightened from 50% to 25%."""
        result = theorem_T_TERS_1_amgm_decomposition()
        assert result.details["fisher_relative_error"] < 0.25


class TestTTERS2:
    """T-TERS-2: Screening-Induced Sign Reversal as Seam Event."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_2_screening_sign_reversal()
        assert result.verdict == "PROVEN"

    def test_kappa_decreases_on_surface(self) -> None:
        result = theorem_T_TERS_2_screening_sign_reversal()
        assert result.details["delta_kappa_A2u"] < 0

    def test_a2u_dominates(self) -> None:
        result = theorem_T_TERS_2_screening_sign_reversal()
        assert result.details["A2u_dominates"]


class TestTTERS3:
    """T-TERS-3: Linear Regime as ε-Controlled Sensitivity."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_3_linear_regime()
        assert result.verdict == "PROVEN"

    def test_linearity_ratio(self) -> None:
        result = theorem_T_TERS_3_linear_regime()
        assert 1.8 < result.details["linearity_ratio_2x"] < 2.2


class TestTTERS4:
    """T-TERS-4: Ground-State Neglect as Positional Illusion Bound."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_4_positional_illusion()
        assert result.verdict == "PROVEN"

    def test_speedup_approximately_2x(self) -> None:
        result = theorem_T_TERS_4_positional_illusion()
        assert 1.8 < result.details["speedup_ratio"] < 2.1


class TestTTERS5:
    """T-TERS-5: Periodicity Requirement as Frozen Contract Consistency."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_5_periodicity_consistency()
        assert result.verdict == "PROVEN"

    def test_periodic_higher_ic(self) -> None:
        result = theorem_T_TERS_5_periodicity_consistency()
        assert result.details["IC_periodic"] > result.details["IC_cluster"]


class TestTTERS6:
    """T-TERS-6: Binding Distance Sensitivity as κ Finite-Change Bound."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_6_binding_sensitivity()
        assert result.verdict == "PROVEN"

    def test_lemma7_bound(self) -> None:
        result = theorem_T_TERS_6_binding_sensitivity()
        assert result.details["bound_satisfied"]

    def test_screening_dominates(self) -> None:
        result = theorem_T_TERS_6_binding_sensitivity()
        assert result.details["dominant_channel"] in (
            "screening_factor",
            "binding_sensitivity",
            "polarizability_zz",
        )


class TestTTERS7:
    """T-TERS-7: Mode-Dependent Screening as Channel Projection Theorem."""

    def test_proven(self) -> None:
        result = theorem_T_TERS_7_channel_projection()
        assert result.verdict == "PROVEN"

    def test_out_of_plane_dominates(self) -> None:
        result = theorem_T_TERS_7_channel_projection()
        assert result.details["dk_out_over_in_ratio"] > 1.0


# ─── Validation Tests ───


class TestCrossTheoremConsistency:
    """Cross-theorem consistency validation."""

    def test_proven(self) -> None:
        result = validate_cross_theorem_consistency()
        assert result.verdict == "PROVEN"

    def test_all_subtests(self) -> None:
        result = validate_cross_theorem_consistency()
        assert result.n_failed == 0


class TestUncertaintyPropagation:
    """Uncertainty propagation analysis."""

    def test_proven(self) -> None:
        result = uncertainty_propagation_analysis()
        assert result.verdict == "PROVEN"

    def test_uncertainty_estimates_defined(self) -> None:
        for label in CHANNEL_LABELS:
            assert label in CHANNEL_UNCERTAINTIES
            assert 0 < CHANNEL_UNCERTAINTIES[label] < 0.15


# ─── Master Function Tests ───


class TestRunAll:
    """Master run functions."""

    def test_run_all_ters_theorems_count(self) -> None:
        results = run_all_ters_theorems()
        assert len(results) == 7

    def test_run_all_with_validation_count(self) -> None:
        results = run_all_with_validation()
        assert len(results) == 9

    def test_all_proven(self) -> None:
        results = run_all_with_validation()
        for r in results:
            assert r.verdict == "PROVEN", f"{r.name} is {r.verdict}"

    def test_total_tests(self) -> None:
        results = run_all_with_validation()
        total = sum(r.n_tests for r in results)
        assert total == 42  # 32 core + 5 cross + 5 uncertainty

    def test_total_passed(self) -> None:
        results = run_all_with_validation()
        passed = sum(r.n_passed for r in results)
        total = sum(r.n_tests for r in results)
        assert passed == total


# ─── Fisher Distance Tests ───


class TestFisherDistances:
    """Fisher geodesic distance computation."""

    def test_a2u_distance(self) -> None:
        result = compute_fisher_distance_gas_to_surface("A2u")
        assert result["fisher_distance"] > 0.5

    def test_ordering(self) -> None:
        results = compute_all_fisher_distances()
        assert results["A2u"]["fisher_distance"] > results["B1g"]["fisher_distance"]
        assert results["B1g"]["fisher_distance"] > results["A2g"]["fisher_distance"]

    def test_normalized_in_range(self) -> None:
        results = compute_all_fisher_distances()
        for _mode, data in results.items():
            assert 0.0 <= data["fisher_normalized"] <= 1.0

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_fisher_distance_gas_to_surface("X_invalid")


# ─── TheoremResult Dataclass ───


class TestTheoremResult:
    """TheoremResult data structure."""

    def test_pass_rate(self) -> None:
        tr = TheoremResult("test", "stmt", 10, 8, 2, {}, "FALSIFIED")
        assert tr.pass_rate == 0.8

    def test_pass_rate_zero_tests(self) -> None:
        tr = TheoremResult("test", "stmt", 0, 0, 0, {}, "PROVEN")
        assert tr.pass_rate == 0.0


# ─── Pixel Kernel Map Tests ───


class TestPixelKernelMap:
    """Data-driven pixel-level kernel maps from Zenodo data."""

    def test_mgp_grid_shape(self) -> None:
        pmap = compute_pixel_kernel_map("mgp")
        assert pmap.grid_shape == (12, 12)
        assert pmap.F.shape == (12, 12)

    def test_tcne_grid_shape(self) -> None:
        pmap = compute_pixel_kernel_map("tcne")
        assert pmap.grid_shape == (10, 10)

    def test_mos2_grid_shape(self) -> None:
        pmap = compute_pixel_kernel_map("mos2")
        assert pmap.grid_shape == (11, 11)

    def test_amgm_holds_everywhere_mgp(self) -> None:
        """IC ≤ F at every pixel (AM-GM inequality)."""
        pmap = compute_pixel_kernel_map("mgp")
        assert np.all(pmap.IC <= pmap.F + 1e-12)

    def test_amgm_holds_everywhere_mos2(self) -> None:
        pmap = compute_pixel_kernel_map("mos2")
        assert np.all(pmap.IC <= pmap.F + 1e-12)

    def test_amgm_holds_everywhere_tcne(self) -> None:
        pmap = compute_pixel_kernel_map("tcne")
        assert np.all(pmap.IC <= pmap.F + 1e-12)

    def test_budget_identity_mgp(self) -> None:
        """F + ω = 1 at every pixel."""
        pmap = compute_pixel_kernel_map("mgp")
        assert np.allclose(pmap.F + pmap.omega, 1.0, atol=1e-10)

    def test_delta_nonnegative(self) -> None:
        """AM-GM gap Δ ≥ 0 everywhere."""
        pmap = compute_pixel_kernel_map("mgp")
        assert np.all(pmap.delta >= -1e-12)

    def test_channels_shape(self) -> None:
        pmap = compute_pixel_kernel_map("mgp")
        assert pmap.channels.shape == (12, 12, 8)

    def test_channels_in_bounds(self) -> None:
        pmap = compute_pixel_kernel_map("mgp")
        assert float(np.min(pmap.channels)) >= EPSILON - 1e-15
        assert float(np.max(pmap.channels)) <= 1.0 - EPSILON + 1e-15

    def test_summary_keys(self) -> None:
        pmap = compute_pixel_kernel_map("mgp")
        s = pmap.summary()
        assert "F_mean" in s
        assert "regime_counts" in s

    def test_invalid_system_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown system"):
            compute_pixel_kernel_map("nonexistent")


# ─── Data-Driven Parameterisation Tests ───


class TestDataDrivenTraces:
    """Data-driven trace vectors from Zenodo .npz files."""

    def test_mgp_shape(self) -> None:
        c, w = mgp_data_driven()
        assert c.shape == (8,)
        assert w.shape == (8,)

    def test_tcne_shape(self) -> None:
        c, _w = tcne_data_driven()
        assert c.shape == (8,)

    def test_mos2_shape(self) -> None:
        c, _w = mos2_data_driven()
        assert c.shape == (8,)

    def test_weights_sum_to_one(self) -> None:
        _, w = mgp_data_driven()
        assert abs(float(np.sum(w)) - 1.0) < 1e-12

    def test_channels_in_bounds(self) -> None:
        for fn in (mgp_data_driven, tcne_data_driven, mos2_data_driven):
            c, _ = fn()
            assert float(np.min(c)) >= EPSILON - 1e-15
            assert float(np.max(c)) <= 1.0 - EPSILON + 1e-15

    def test_mos2_near_uniform(self) -> None:
        """MoS₂ pristine should have near-uniform channels."""
        c, _ = mos2_data_driven()
        # polarizability_zz and field_gradient should be close to 1.0
        assert c[0] > 0.95  # α_zz / α_max ≈ 1


# ─── Monte Carlo Uncertainty Tests ───


class TestMonteCarlo:
    """Full Gaussian Monte Carlo uncertainty (10 000 samples)."""

    def test_mgp_no_amgm_violations(self) -> None:
        mc = monte_carlo_uncertainty("mgp", n_samples=10_000, seed=42)
        assert mc.amgm_violation_rate == 0.0

    def test_mgp_no_budget_violations(self) -> None:
        mc = monte_carlo_uncertainty("mgp", n_samples=10_000, seed=42)
        assert mc.budget_identity_violation_rate == 0.0

    def test_tcne_no_amgm_violations(self) -> None:
        mc = monte_carlo_uncertainty("tcne", n_samples=10_000, seed=42)
        assert mc.amgm_violation_rate == 0.0

    def test_mos2_no_amgm_violations(self) -> None:
        mc = monte_carlo_uncertainty("mos2", n_samples=10_000, seed=42)
        assert mc.amgm_violation_rate == 0.0

    def test_ci95_contains_mean(self) -> None:
        mc = monte_carlo_uncertainty("mgp", n_samples=10_000, seed=42)
        assert mc.F_ci95[0] <= mc.F_mean <= mc.F_ci95[1]
        assert mc.IC_ci95[0] <= mc.IC_mean <= mc.IC_ci95[1]

    def test_n_samples_recorded(self) -> None:
        mc = monte_carlo_uncertainty("mgp", n_samples=5_000, seed=0)
        assert mc.n_samples == 5_000

    def test_positive_std(self) -> None:
        mc = monte_carlo_uncertainty("mgp", n_samples=10_000, seed=42)
        assert mc.F_std > 0
        assert mc.IC_std > 0

    def test_delta_positive(self) -> None:
        """AM-GM gap should be positive on average."""
        mc = monte_carlo_uncertainty("mgp", n_samples=10_000, seed=42)
        assert mc.delta_mean > 0


# ─── Second-System Validation Tests ───


class TestMoS2Analysis:
    """MoS₂ A'₁ pristine full-rigour analysis."""

    def test_all_dadq_positive(self) -> None:
        result = mos2_data_driven_analysis()
        assert result["all_dadq_positive"] is True

    def test_intensity_cv_low(self) -> None:
        """Pristine MoS₂ has near-uniform intensity."""
        result = mos2_data_driven_analysis()
        assert result["intensity_cv"] < 0.01

    def test_mc_no_amgm_violations(self) -> None:
        result = mos2_data_driven_analysis()
        assert result["mc_amgm_violation_rate"] == 0.0


class TestTCNEAnalysis:
    """TCNE B₁ isolated full-rigour analysis."""

    def test_mixed_sign_dadq(self) -> None:
        result = tcne_data_driven_analysis()
        assert result["n_positive_dadq"] > 0
        assert result["n_negative_dadq"] > 0

    def test_high_intensity_cv(self) -> None:
        """TCNE should have high spatial variation (localised hot spots)."""
        result = tcne_data_driven_analysis()
        assert result["intensity_cv"] > 1.0

    def test_periodic_ic_higher(self) -> None:
        """Periodic MgP should have higher IC than non-periodic TCNE."""
        result = tcne_data_driven_analysis()
        assert result["periodicity_test"] is True

    def test_mc_no_amgm_violations(self) -> None:
        result = tcne_data_driven_analysis()
        assert result["mc_amgm_violation_rate"] == 0.0
