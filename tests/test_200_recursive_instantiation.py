"""Tests for closures/atomic_physics/recursive_instantiation.py.

Validates the Recursive Instantiation Theory — six theorems (T11–T16)
proving that elements are recursive collapse returns with cumulative
drift as the dominant stability predictor.

Test hierarchy:
    test_00_*  — smoke tests (fast, no data dependency)
    test_10_*  — dataclass / structure tests
    test_20_*  — computation tests (step distances, cumulative drift)
    test_30_*  — theorem tests (T11–T16)
    test_40_*  — census / classification tests
    test_50_*  — integration tests (all theorems together)
"""

from __future__ import annotations

import pytest

from closures.atomic_physics.recursive_instantiation import (
    MAGIC_Z,
    PREDICTED_MAGIC_Z,
    TAU_R_THRESHOLD,
    UNSTABLE_LIGHT_Z,
    RecursiveAnalysis,
    RecursiveProfile,
    TheoremResult,
    _classify_recursive_category,
    _is_stable,
    compute_cumulative_drift,
    compute_recursive_analysis,
    compute_step_distances,
    display_census,
    display_summary,
    display_theorem,
    run_all_theorems,
    theorem_T11_cumulative_drift_dominance,
    theorem_T12_recursive_collapse_budget,
    theorem_T13_non_returnable_states,
    theorem_T14_magic_number_drift_absorption,
    theorem_T15_period_efficiency_exhaustion,
    theorem_T16_constant_heterogeneity_rate,
)

# ═══════════════════════════════════════════════════════════════════
# FIXTURES — shared analysis (session-scoped for speed)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def analysis() -> RecursiveAnalysis:
    """Pre-compute recursive analysis once for all tests."""
    return compute_recursive_analysis()


@pytest.fixture(scope="module")
def all_theorems(analysis: RecursiveAnalysis) -> list[TheoremResult]:
    """Run all theorems once for all tests."""
    return run_all_theorems(analysis)


# ═══════════════════════════════════════════════════════════════════
# 00 — SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════


class Test00Smoke:
    """Basic sanity checks — imports, constants, trivial calls."""

    def test_00_constants_exist(self) -> None:
        assert len(MAGIC_Z) == 6
        assert len(PREDICTED_MAGIC_Z) == 3
        assert len(UNSTABLE_LIGHT_Z) == 2
        assert 43 in UNSTABLE_LIGHT_Z
        assert 61 in UNSTABLE_LIGHT_Z

    def test_01_tau_r_threshold_positive(self) -> None:
        assert TAU_R_THRESHOLD > 0
        assert 50 < TAU_R_THRESHOLD < 200

    def test_02_is_stable_basic(self) -> None:
        # H, He, C, Fe are stable
        assert _is_stable(1) is True
        assert _is_stable(2) is True
        assert _is_stable(6) is True
        assert _is_stable(26) is True
        # Tc, Pm are unstable
        assert _is_stable(43) is False
        assert _is_stable(61) is False
        # Bi is stable (longest half-life)
        assert _is_stable(83) is True
        # Po onward is radioactive
        assert _is_stable(84) is False
        assert _is_stable(92) is False
        assert _is_stable(118) is False


# ═══════════════════════════════════════════════════════════════════
# 10 — DATACLASS / STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════


class Test10Structure:
    """Verify dataclass shapes and field types."""

    def test_10_recursive_profile_fields(self, analysis: RecursiveAnalysis) -> None:
        p = analysis.profiles[0]
        assert isinstance(p, RecursiveProfile)
        assert p.Z == 1
        assert isinstance(p.symbol, str)
        assert isinstance(p.F, float)
        assert isinstance(p.IC, float)
        assert isinstance(p.cumulative_drift, float)
        assert isinstance(p.tau_R, float)
        assert isinstance(p.stable, bool)
        assert isinstance(p.category, str)

    def test_11_analysis_has_118_profiles(self, analysis: RecursiveAnalysis) -> None:
        assert len(analysis.profiles) == 118

    def test_12_analysis_has_117_step_distances(self, analysis: RecursiveAnalysis) -> None:
        assert len(analysis.step_distances) == 117

    def test_13_analysis_has_118_cum_drifts(self, analysis: RecursiveAnalysis) -> None:
        assert len(analysis.cumulative_drifts) == 118

    def test_14_profiles_sorted_by_Z(self, analysis: RecursiveAnalysis) -> None:
        zs = [p.Z for p in analysis.profiles]
        assert zs == list(range(1, 119))

    def test_15_theorem_result_has_pass_rate(self) -> None:
        tr = TheoremResult("test", "s", 10, 7, 3, {}, "FALSIFIED")
        assert tr.pass_rate == pytest.approx(0.7)

    def test_16_stable_radio_counts(self, analysis: RecursiveAnalysis) -> None:
        assert analysis.n_stable == 81
        assert analysis.n_radioactive == 37
        assert analysis.n_stable + analysis.n_radioactive == 118


# ═══════════════════════════════════════════════════════════════════
# 20 — COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class Test20Computation:
    """Verify core computation correctness."""

    def test_20_step_distances_positive(self, analysis: RecursiveAnalysis) -> None:
        for d in analysis.step_distances:
            assert d >= 0, f"Negative step distance: {d}"

    def test_21_cumulative_drift_monotonic(self, analysis: RecursiveAnalysis) -> None:
        for i in range(1, len(analysis.cumulative_drifts)):
            assert analysis.cumulative_drifts[i] >= analysis.cumulative_drifts[i - 1]

    def test_22_cumulative_drift_starts_at_zero(self, analysis: RecursiveAnalysis) -> None:
        assert analysis.cumulative_drifts[0] == 0.0

    def test_23_first_element_is_hydrogen(self, analysis: RecursiveAnalysis) -> None:
        assert analysis.profiles[0].symbol == "H"
        assert analysis.profiles[0].Z == 1

    def test_24_cumulative_drift_from_steps(self, analysis: RecursiveAnalysis) -> None:
        """Verify cumulative drift = sum of step distances."""
        for i in range(1, 10):
            expected = sum(analysis.step_distances[:i])
            actual = analysis.cumulative_drifts[i]
            assert actual == pytest.approx(expected, abs=1e-10)

    def test_25_tau_r_equals_drift_over_ic(self, analysis: RecursiveAnalysis) -> None:
        """Verify τ_R = cumulative_drift / IC for all elements."""
        for p in analysis.profiles:
            if p.IC > 1e-10:
                expected = p.cumulative_drift / p.IC
                assert p.tau_R == pytest.approx(expected, rel=1e-6)

    def test_26_h_to_he_largest_step(self, analysis: RecursiveAnalysis) -> None:
        """H→He should be the largest step distance (genesis gap)."""
        assert analysis.step_distances[0] == max(analysis.step_distances)

    def test_27_drift_per_z_decreasing_trend(self, analysis: RecursiveAnalysis) -> None:
        """drift/Z should decrease from Z=10 to Z=118."""
        by_z = {p.Z: p for p in analysis.profiles}
        assert by_z[10].drift_per_Z > by_z[50].drift_per_Z > by_z[118].drift_per_Z

    def test_28_total_drift_reasonable(self, analysis: RecursiveAnalysis) -> None:
        """Total cumulative drift should be between 30 and 100."""
        total = analysis.cumulative_drifts[-1]
        assert 30 < total < 100, f"Unexpected total drift: {total}"

    def test_29_returnability_positive(self, analysis: RecursiveAnalysis) -> None:
        """Returnability = IC / gap should be positive for all elements."""
        for p in analysis.profiles:
            assert p.returnability > 0 or p.returnability == float("inf")


# ═══════════════════════════════════════════════════════════════════
# 30 — THEOREM TESTS (T11–T16)
# ═══════════════════════════════════════════════════════════════════


class Test30Theorems:
    """Verify each theorem proves independently."""

    def test_30_T11_cumulative_drift_dominance(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T11_cumulative_drift_dominance(analysis)
        assert result.verdict == "PROVEN", f"T11 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        assert result.n_failed == 0
        # ρ should be strongly negative
        assert result.details["rho_drift"] < -0.5

    def test_31_T12_recursive_collapse_budget(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T12_recursive_collapse_budget(analysis)
        assert result.verdict == "PROVEN", f"T12 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        # Accuracy > 80%
        assert result.details["accuracy"] > 0.80
        # Mean stable τ_R < mean radio τ_R
        assert result.details["mean_tau_stable"] < result.details["mean_tau_radio"]

    def test_32_T13_non_returnable_states(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T13_non_returnable_states(analysis)
        assert result.verdict == "PROVEN", f"T13 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        # At least 15 tears
        assert result.details["n_tears"] >= 15
        # Quantum percentage ≥ 80%
        assert result.details["pct_quantum"] >= 80

    def test_33_T14_magic_number_drift_absorption(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T14_magic_number_drift_absorption(analysis)
        assert result.verdict == "PROVEN", f"T14 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        # All magic Z≥20 stable
        assert result.details["magic_all_stable"] is True

    def test_34_T15_period_efficiency_exhaustion(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T15_period_efficiency_exhaustion(analysis)
        assert result.verdict == "PROVEN", f"T15 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        # Period 7 has 0% stable
        ps = result.details["period_stats"]
        assert ps[7]["n_stable"] == 0

    def test_35_T16_constant_heterogeneity_rate(self, analysis: RecursiveAnalysis) -> None:
        result = theorem_T16_constant_heterogeneity_rate(analysis)
        assert result.verdict == "PROVEN", f"T16 FALSIFIED: {result.details}"
        assert result.n_passed == result.n_tests
        # Mean gap/Z in expected range
        assert 0.10 <= result.details["mean_gap_per_Z"] <= 0.16


# ═══════════════════════════════════════════════════════════════════
# 40 — CENSUS / CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════


class Test40Census:
    """Verify the recursive instantiation census and classification."""

    def test_40_all_categories_present(self, analysis: RecursiveAnalysis) -> None:
        cats = {p.category for p in analysis.profiles}
        expected = {"PRISTINE", "ROBUST", "STRESSED", "MARGINAL", "DECAYING", "FLEETING", "EPHEMERAL"}
        assert cats == expected

    def test_41_stable_in_first_three_cats(self, analysis: RecursiveAnalysis) -> None:
        """PRISTINE, ROBUST, STRESSED should all be stable (except MARGINAL)."""
        for p in analysis.profiles:
            if p.category in ("PRISTINE", "ROBUST", "STRESSED"):
                assert p.stable is True, f"{p.symbol} (Z={p.Z}) is {p.category} but not stable"

    def test_42_marginal_is_tc_pm(self, analysis: RecursiveAnalysis) -> None:
        marginal = [p for p in analysis.profiles if p.category == "MARGINAL"]
        symbols = {p.symbol for p in marginal}
        assert symbols == {"Tc", "Pm"}

    def test_43_decaying_are_radioactive(self, analysis: RecursiveAnalysis) -> None:
        for p in analysis.profiles:
            if p.category in ("DECAYING", "FLEETING", "EPHEMERAL"):
                assert p.stable is False

    def test_44_pristine_has_low_gap(self, analysis: RecursiveAnalysis) -> None:
        """PRISTINE elements should have gap < 0.10."""
        for p in analysis.profiles:
            if p.category == "PRISTINE":
                assert p.gap < 0.10, f"{p.symbol} (Z={p.Z}) gap={p.gap}"

    def test_45_ephemeral_are_superheavy(self, analysis: RecursiveAnalysis) -> None:
        """EPHEMERAL should be Z ≥ 104 (superheavy)."""
        for p in analysis.profiles:
            if p.category == "EPHEMERAL":
                assert p.Z >= 104, f"{p.symbol} (Z={p.Z}) is EPHEMERAL but Z < 104"

    def test_46_category_counts_sum_to_118(self, analysis: RecursiveAnalysis) -> None:
        from collections import Counter

        counts = Counter(p.category for p in analysis.profiles)
        assert sum(counts.values()) == 118

    def test_47_classify_stable_low_gap(self) -> None:
        assert _classify_recursive_category(26, 0.05, 0.50, True) == "PRISTINE"

    def test_48_classify_unstable_light(self) -> None:
        assert _classify_recursive_category(43, 0.10, 0.40, False) == "MARGINAL"

    def test_49_classify_superheavy(self) -> None:
        assert _classify_recursive_category(110, 0.20, 0.30, False) == "EPHEMERAL"


# ═══════════════════════════════════════════════════════════════════
# 50 — INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class Test50Integration:
    """End-to-end integration tests."""

    def test_50_all_theorems_proven(self, all_theorems: list[TheoremResult]) -> None:
        """All six theorems must prove."""
        for r in all_theorems:
            assert r.verdict == "PROVEN", f"{r.name} FALSIFIED: {r.details}"

    def test_51_grand_total_26_of_26(self, all_theorems: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in all_theorems)
        passed = sum(r.n_passed for r in all_theorems)
        assert total == 26
        assert passed == 26

    def test_52_six_theorems_returned(self, all_theorems: list[TheoremResult]) -> None:
        assert len(all_theorems) == 6

    def test_53_theorem_names_sequential(self, all_theorems: list[TheoremResult]) -> None:
        names = [r.name for r in all_theorems]
        for i, name in enumerate(names):
            assert f"T{11 + i}" in name

    def test_54_display_summary_runs(
        self, all_theorems: list[TheoremResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        display_summary(all_theorems)
        out = capsys.readouterr().out
        assert "GRAND TOTAL" in out
        assert "26/26" in out

    def test_55_display_census_runs(self, analysis: RecursiveAnalysis, capsys: pytest.CaptureFixture[str]) -> None:
        display_census(analysis)
        out = capsys.readouterr().out
        assert "PRISTINE" in out
        assert "EPHEMERAL" in out

    def test_56_display_theorem_runs(
        self, all_theorems: list[TheoremResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        display_theorem(all_theorems[0])
        out = capsys.readouterr().out
        assert "T11" in out

    def test_57_display_theorem_verbose(
        self, all_theorems: list[TheoremResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        display_theorem(all_theorems[0], verbose=True)
        out = capsys.readouterr().out
        assert "rho_drift" in out

    def test_58_analysis_correlations(self, analysis: RecursiveAnalysis) -> None:
        """Key quantitative results must match established values."""
        # ρ ≈ -0.77
        assert -0.85 < analysis.drift_stability_rho < -0.65
        # p-value extremely small
        assert analysis.drift_stability_pval < 1e-15
        # Accuracy ≈ 86%
        assert 0.80 < analysis.tau_R_accuracy < 0.95
        # 21 tears
        assert 15 <= analysis.n_tears <= 30

    def test_59_compute_step_distances_empty(self) -> None:
        """Edge case: empty input."""
        assert compute_step_distances([]) == []

    def test_60_compute_cumulative_drift_empty(self) -> None:
        """Edge case: empty step distances."""
        assert compute_cumulative_drift([]) == [0.0]
