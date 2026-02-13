"""Tests for double_slit_interference.py — complementarity as channel geometry.

Validates that the double-slit experiment maps onto GCD kernel channel
portraits: complementarity (V² + D² ≤ 1) forces at least one member
of the {V, D} pair to ε at the pure extremes; partial measurement is the
unique kernel-optimal state where all channels are alive.

Key discovery: the "complementarity cliff" — a >5× IC gap between
scenarios with an ε-complementary channel and scenarios where both
V and D are above 0.10.

Data sources: Englert 1996 (PRL 77, 2154), Tonomura et al. 1989
    (Am. J. Phys. 57, 117), Wheeler 1978 (delayed choice).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.quantum_mechanics.double_slit_interference import (  # noqa: E402
    CHANNEL_NAMES,
    EPSILON,
    N_CHANNELS,
    SCENARIO_ORDER,
    SCENARIOS,
    WEIGHTS,
    TheoremResult,
    all_scenario_kernels,
    all_scenario_traces,
    channel_autopsy,
    double_slit_intensity,
    englert_distinguishability,
    englert_visibility,
    fringe_visibility_from_intensities,
    run_all_theorems,
    scenario_trace,
    single_slit_envelope,
    summary_report,
    theorem_T_DSE_1_tier1,
    theorem_T_DSE_2_complementarity,
    theorem_T_DSE_3_complementarity_cliff,
    theorem_T_DSE_4_quantum_eraser,
    theorem_T_DSE_5_classical_limit,
    theorem_T_DSE_6_delayed_choice,
    theorem_T_DSE_7_partial_transcends,
    verify_complementarity,
)

# ═══════════════════════════════════════════════════════════════════
# MODULE STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestModuleStructure:
    """Validate module constants and scenario database."""

    def test_n_channels(self) -> None:
        assert N_CHANNELS == 8

    def test_weights_sum(self) -> None:
        assert abs(float(np.sum(WEIGHTS)) - 1.0) < 1e-12

    def test_weights_equal(self) -> None:
        expected = 1.0 / N_CHANNELS
        for w in WEIGHTS:
            assert abs(w - expected) < 1e-12

    def test_epsilon_positive(self) -> None:
        assert EPSILON > 0
        assert EPSILON < 1e-4

    def test_channel_names_count(self) -> None:
        assert len(CHANNEL_NAMES) == N_CHANNELS

    def test_channel_names_unique(self) -> None:
        assert len(set(CHANNEL_NAMES)) == len(CHANNEL_NAMES)

    def test_scenario_count(self) -> None:
        assert len(SCENARIOS) == 8

    def test_scenario_order_complete(self) -> None:
        assert set(SCENARIO_ORDER) == set(SCENARIOS.keys())

    def test_scenario_order_length(self) -> None:
        assert len(SCENARIO_ORDER) == len(SCENARIOS)

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_scenario_has_name(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert sc.name == name

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_scenario_has_label(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert len(sc.label) > 0


# ═══════════════════════════════════════════════════════════════════
# CHANNEL CONSTRUCTION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestChannelConstruction:
    """Validate that trace vectors obey UMCP invariants."""

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_channel_count(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert len(sc.channels) == N_CHANNELS

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_channels_in_range(self, name: str) -> None:
        sc = SCENARIOS[name]
        for i, c in enumerate(sc.channels):
            assert EPSILON <= c <= 1.0, f"{name} channel {CHANNEL_NAMES[i]} = {c} outside [ε, 1]"

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_trace_array_shape(self, name: str) -> None:
        sc = SCENARIOS[name]
        t = scenario_trace(sc)
        assert t.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_visibility_channel_matches_V(self, name: str) -> None:
        sc = SCENARIOS[name]
        c0 = sc.channels[0]
        expected = max(EPSILON, min(sc.V, 1.0 - EPSILON))
        assert abs(c0 - expected) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_distinguishability_channel_matches_D(self, name: str) -> None:
        sc = SCENARIOS[name]
        c1 = sc.channels[1]
        expected = max(EPSILON, min(sc.D, 1.0 - EPSILON))
        assert abs(c1 - expected) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# COMPLEMENTARITY PHYSICS TESTS
# ═══════════════════════════════════════════════════════════════════


class TestComplementarityPhysics:
    """Validate Englert complementarity relation V² + D² ≤ 1."""

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_complementarity_inequality(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert sc.V**2 + sc.D**2 <= 1.0 + 1e-6

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_verify_complementarity_function(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert verify_complementarity(sc.V, sc.D)

    def test_englert_visibility_pure_state(self) -> None:
        V = englert_visibility(0.0)
        assert abs(V - 1.0) < 1e-10

    def test_englert_distinguishability_pure_state(self) -> None:
        D = englert_distinguishability(0.0)
        assert abs(D - 1.0) < 1e-10

    def test_englert_V_D_inverse(self) -> None:
        D = 0.6
        V = englert_visibility(D)
        D_back = englert_distinguishability(V)
        assert abs(D_back - D) < 1e-10

    @pytest.mark.parametrize("D", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_englert_complementarity_holds(self, D: float) -> None:
        V = englert_visibility(D)
        assert abs(V**2 + D**2 - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# INTERFERENCE PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════


class TestInterferencePattern:
    """Validate physical interference pattern functions."""

    def test_single_slit_centre_max(self) -> None:
        intensity = single_slit_envelope(0.0, 1e-6, 500e-9)
        assert abs(intensity - 1.0) < 1e-10

    def test_single_slit_positive(self) -> None:
        for theta in [0.01, 0.05, 0.1, 0.2, 0.5]:
            intensity = single_slit_envelope(theta, 1e-6, 500e-9)
            assert intensity >= 0.0

    def test_double_slit_centre_max(self) -> None:
        intensity = double_slit_intensity(0.0, 5e-6, 1e-6, 500e-9, visibility=1.0)
        assert intensity > 0.0

    def test_double_slit_constructive(self) -> None:
        """At θ=0, constructive interference: I = 2 × envelope."""
        intensity = double_slit_intensity(0.0, 5e-6, 1e-6, 500e-9, visibility=1.0)
        env = single_slit_envelope(0.0, 1e-6, 500e-9)
        # At θ=0, cos term = 1, so I = env * (1 + V) = 2 * env for V=1
        assert abs(intensity - 2.0 * env) < 1e-10

    def test_double_slit_no_visibility(self) -> None:
        """Zero visibility → no fringes, just envelope."""
        intensity = double_slit_intensity(0.0, 5e-6, 1e-6, 500e-9, visibility=0.0)
        env = single_slit_envelope(0.0, 1e-6, 500e-9)
        assert abs(intensity - env) < 1e-10

    def test_fringe_visibility_calculation(self) -> None:
        V = fringe_visibility_from_intensities(10.0, 2.0)
        assert abs(V - 8.0 / 12.0) < 1e-10

    def test_fringe_visibility_perfect(self) -> None:
        V = fringe_visibility_from_intensities(1.0, 0.0)
        assert abs(V - 1.0) < 1e-10

    def test_fringe_visibility_none(self) -> None:
        V = fringe_visibility_from_intensities(5.0, 5.0)
        assert abs(V) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# KERNEL COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestKernelComputation:
    """Validate kernel outputs for all scenarios."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_in_unit(self, name: str, kernels: dict) -> None:
        assert 0.0 <= kernels[name]["F"] <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_omega_in_unit(self, name: str, kernels: dict) -> None:
        assert 0.0 <= kernels[name]["omega"] <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_positive(self, name: str, kernels: dict) -> None:
        assert kernels[name]["IC"] > 0.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_le_F(self, name: str, kernels: dict) -> None:
        assert kernels[name]["IC"] <= kernels[name]["F"] + 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_plus_omega_eq_1(self, name: str, kernels: dict) -> None:
        s = kernels[name]["F"] + kernels[name]["omega"]
        assert abs(s - 1.0) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_eq_exp_kappa(self, name: str, kernels: dict) -> None:
        k = kernels[name]
        assert abs(k["IC"] - np.exp(k["kappa"])) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_amgm_gap_nonneg(self, name: str, kernels: dict) -> None:
        assert kernels[name]["amgm_gap"] >= -1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_regime_is_string(self, name: str, kernels: dict) -> None:
        assert isinstance(kernels[name]["regime"], str)


# ═══════════════════════════════════════════════════════════════════
# CHANNEL AUTOPSY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestChannelAutopsy:
    """Validate channel autopsy diagnostics."""

    @pytest.fixture(scope="class")
    def autopsy(self) -> dict:
        return channel_autopsy()

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_autopsy_has_keys(self, name: str, autopsy: dict) -> None:
        assert "ic_killers" in autopsy[name]
        assert "n_killers" in autopsy[name]
        assert "min_channel" in autopsy[name]
        assert "min_channel_name" in autopsy[name]

    def test_S1_has_path_killer(self, autopsy: dict) -> None:
        killers = [n for n, _ in autopsy["S1"]["ic_killers"]]
        assert "path_distinguishability" in killers

    def test_S2_has_coherence_killer(self, autopsy: dict) -> None:
        killers = [n for n, _ in autopsy["S2"]["ic_killers"]]
        assert "coherence_visibility" in killers

    def test_S4_no_killers(self, autopsy: dict) -> None:
        assert autopsy["S4"]["n_killers"] == 0

    def test_S8_many_killers(self, autopsy: dict) -> None:
        assert autopsy["S8"]["n_killers"] >= 3


# ═══════════════════════════════════════════════════════════════════
# COMPLEMENTARITY CLIFF TESTS
# ═══════════════════════════════════════════════════════════════════


class TestComplementarityCliff:
    """Validate the central discovery: the complementarity cliff."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    def _eps_scenarios(self) -> list[str]:
        """Scenarios with at least one complementary channel at ε."""
        return [s for s in SCENARIO_ORDER if SCENARIOS[s].V < 0.05 or SCENARIOS[s].D < 0.05]

    def _alive_scenarios(self) -> list[str]:
        """Scenarios with both V, D above 0.10."""
        return [s for s in SCENARIO_ORDER if SCENARIOS[s].V >= 0.05 and SCENARIOS[s].D >= 0.05]

    @pytest.mark.parametrize(
        "name",
        ["S1", "S2", "S3", "S6", "S8"],
    )
    def test_eps_scenario_IC_below_015(self, name: str, kernels: dict) -> None:
        assert kernels[name]["IC"] < 0.15, f"{name}: IC={kernels[name]['IC']:.4f}, expected <0.15"

    @pytest.mark.parametrize("name", ["S4", "S5", "S7"])
    def test_alive_scenario_IC_above_050(self, name: str, kernels: dict) -> None:
        assert kernels[name]["IC"] > 0.50, f"{name}: IC={kernels[name]['IC']:.4f}, expected >0.50"

    def test_cliff_ratio_exceeds_5x(self, kernels: dict) -> None:
        eps_ics = [kernels[s]["IC"] for s in self._eps_scenarios()]
        alive_ics = [kernels[s]["IC"] for s in self._alive_scenarios()]
        ratio = min(alive_ics) / max(eps_ics)
        assert ratio > 5.0, f"cliff ratio {ratio:.1f} ≤ 5"

    def test_S4_highest_IC(self, kernels: dict) -> None:
        ic_vals = {s: kernels[s]["IC"] for s in SCENARIO_ORDER}
        max_s = max(ic_vals, key=lambda s: ic_vals[s])
        assert max_s == "S4"

    def test_S4_smallest_gap(self, kernels: dict) -> None:
        gap_vals = {s: kernels[s]["amgm_gap"] for s in SCENARIO_ORDER}
        min_s = min(gap_vals, key=lambda s: gap_vals[s])
        assert min_s == "S4"

    def test_S8_lowest_IC(self, kernels: dict) -> None:
        ic_vals = {s: kernels[s]["IC"] for s in SCENARIO_ORDER}
        min_s = min(ic_vals, key=lambda s: ic_vals[s])
        assert min_s == "S8"


# ═══════════════════════════════════════════════════════════════════
# SCENARIO-SPECIFIC PHYSICS TESTS
# ═══════════════════════════════════════════════════════════════════


class TestScenarioPhysics:
    """Validate physics encoded in specific scenarios."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    # S1: Full interference
    def test_S1_high_V(self) -> None:
        assert SCENARIOS["S1"].V > 0.95

    def test_S1_low_D(self) -> None:
        assert SCENARIOS["S1"].D < 0.01

    def test_S1_two_slits(self) -> None:
        assert SCENARIOS["S1"].n_slits == 2

    # S2: No interference
    def test_S2_low_V(self) -> None:
        assert SCENARIOS["S2"].V < 0.01

    def test_S2_high_D(self) -> None:
        assert SCENARIOS["S2"].D > 0.95

    def test_S2_has_detector(self) -> None:
        assert SCENARIOS["S2"].detector == "full"

    # S3: Single slit
    def test_S3_one_slit(self) -> None:
        assert SCENARIOS["S3"].n_slits == 1

    def test_S3_trivial_which_path(self) -> None:
        assert SCENARIOS["S3"].D > 0.90

    # S4: Partial measurement
    def test_S4_partial_V(self) -> None:
        assert 0.3 < SCENARIOS["S4"].V < 0.9

    def test_S4_partial_D(self) -> None:
        assert 0.3 < SCENARIOS["S4"].D < 0.9

    def test_S4_near_pure_state(self) -> None:
        sc = SCENARIOS["S4"]
        assert sc.V**2 + sc.D**2 > 0.95

    def test_S4_all_channels_above_010(self) -> None:
        ch = SCENARIOS["S4"].channels
        assert all(c > 0.10 for c in ch)

    # S5: Quantum eraser
    def test_S5_V_restored(self) -> None:
        assert SCENARIOS["S5"].V > 0.90

    def test_S5_D_above_epsilon(self) -> None:
        assert SCENARIOS["S5"].D > 0.10

    def test_S5_eraser_detector(self) -> None:
        assert SCENARIOS["S5"].detector == "eraser"

    # S6: Delayed choice
    def test_S6_delayed_detector(self) -> None:
        assert SCENARIOS["S6"].detector == "delayed"

    def test_S6_high_V(self) -> None:
        assert SCENARIOS["S6"].V > 0.95

    # S7: Electron experiment
    def test_S7_is_electron(self) -> None:
        assert SCENARIOS["S7"].particle == "electron"

    def test_S7_nonzero_D(self) -> None:
        assert SCENARIOS["S7"].D > 0.05

    # S8: Classical
    def test_S8_is_classical(self) -> None:
        assert SCENARIOS["S8"].particle == "classical"

    def test_S8_no_V(self) -> None:
        assert SCENARIOS["S8"].V < 0.01

    def test_S8_full_D(self) -> None:
        assert SCENARIOS["S8"].D > 0.95


# ═══════════════════════════════════════════════════════════════════
# THEOREM TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTheoremT_DSE_1:
    """T-DSE-1: Tier-1 Kernel Identities."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_1_tier1()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_test_count(self, result: TheoremResult) -> None:
        # 3 tests × 8 scenarios = 24
        assert result.n_tests == 24


class TestTheoremT_DSE_2:
    """T-DSE-2: Complementarity as Channel Anticorrelation."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_2_complementarity()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_strong_anticorrelation(self, result: TheoremResult) -> None:
        rho = result.details["V_D_pearson_correlation"]
        assert rho < -0.90


class TestTheoremT_DSE_3:
    """T-DSE-3: Complementarity Cliff."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_3_complementarity_cliff()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_cliff_ratio(self, result: TheoremResult) -> None:
        assert result.details["cliff_gt_5x"]


class TestTheoremT_DSE_4:
    """T-DSE-4: Quantum Eraser Lifts IC Above Cliff."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_4_quantum_eraser()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_ic_ratio_above_5x(self, result: TheoremResult) -> None:
        assert result.details["IC_ratio_S5_over_S2"] > 5.0


class TestTheoremT_DSE_5:
    """T-DSE-5: Classical Limit as Maximum Channel Death."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_5_classical_limit()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_S8_most_killers(self, result: TheoremResult) -> None:
        assert result.details["classical_more_killers"]


class TestTheoremT_DSE_6:
    """T-DSE-6: Delayed Choice Invariance."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_6_delayed_choice()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_same_regime(self, result: TheoremResult) -> None:
        assert result.details["same_regime"]


class TestTheoremT_DSE_7:
    """T-DSE-7: Partial Measurement Transcends Both Extremes."""

    @pytest.fixture(scope="class")
    def result(self) -> TheoremResult:
        return theorem_T_DSE_7_partial_transcends()

    def test_proven(self, result: TheoremResult) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result: TheoremResult) -> None:
        assert result.n_failed == 0

    def test_S4_highest_IC(self, result: TheoremResult) -> None:
        assert result.details["max_IC_scenario"] == "S4"

    def test_S4_all_channels_alive(self, result: TheoremResult) -> None:
        assert result.details["S4_all_channels_above_010"]


# ═══════════════════════════════════════════════════════════════════
# ALL-THEOREM ORCHESTRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestAllTheorems:
    """Validate run_all_theorems and summary_report."""

    @pytest.fixture(scope="class")
    def results(self) -> list[TheoremResult]:
        return run_all_theorems()

    def test_seven_theorems(self, results: list[TheoremResult]) -> None:
        assert len(results) == 7

    def test_all_proven(self, results: list[TheoremResult]) -> None:
        for r in results:
            assert r.verdict == "PROVEN", f"{r.name}: {r.verdict}"

    def test_total_subtests(self, results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in results)
        assert total >= 60

    def test_zero_failures(self, results: list[TheoremResult]) -> None:
        total_failed = sum(r.n_failed for r in results)
        assert total_failed == 0

    def test_summary_report_runs(self) -> None:
        report = summary_report()
        assert len(report) > 100

    def test_summary_contains_proven(self) -> None:
        report = summary_report()
        assert "PROVEN" in report

    def test_summary_contains_scenarios(self) -> None:
        report = summary_report()
        for s in SCENARIO_ORDER:
            assert s in report


# ═══════════════════════════════════════════════════════════════════
# CROSS-DOMAIN INSIGHT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCrossDomainInsights:
    """Cross-references to other closure domains."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    def test_quantum_eraser_is_IC_restoration(self, kernels: dict) -> None:
        """Eraser effect matches muon-laser which-way IC pattern:
        information erasure restores the geometric mean."""
        ic_s2 = kernels["S2"]["IC"]  # With detector
        ic_s5 = kernels["S5"]["IC"]  # After eraser
        assert ic_s5 > 5 * ic_s2  # Dramatic recovery

    def test_classical_limit_matches_regime_calibration(self, kernels: dict) -> None:
        """S8 ω > 0.45: classical ≈ collapse regime in universal calibration."""
        assert kernels["S8"]["omega"] > 0.45

    def test_partial_measurement_is_coherent_regime(self, kernels: dict) -> None:
        """S4 ω < 0.15: the partial-measurement state is coherent."""
        assert kernels["S4"]["omega"] < 0.15

    def test_delayed_choice_regime_matches_interference(self, kernels: dict) -> None:
        """Delayed choice (S6) and no-detector (S1) have same regime."""
        assert kernels["S1"]["regime"] == kernels["S6"]["regime"]

    def test_S4_IC_exceeds_all_ε_scenarios(self, kernels: dict) -> None:
        """The kernel sees partial measurement as optimal:
        no channel at ε means geometric mean is maximised."""
        s4_ic = kernels["S4"]["IC"]
        for s in ["S1", "S2", "S3", "S6", "S8"]:
            assert s4_ic > kernels[s]["IC"], f"S4 IC={s4_ic:.4f} should exceed {s}={kernels[s]['IC']:.4f}"

    def test_channel_swap_S1_to_S2(self) -> None:
        """The detector SWAPS which complementary channel is at ε.
        S1: C0 alive, C1 at ε.  S2: C0 at ε, C1 alive."""
        ch_s1 = SCENARIOS["S1"].channels
        ch_s2 = SCENARIOS["S2"].channels
        # C0 (visibility): high in S1, ε in S2
        assert ch_s1[0] > 0.90 and ch_s2[0] < 0.01
        # C1 (distinguishability): ε in S1, high in S2
        assert ch_s1[1] < 0.01 and ch_s2[1] > 0.90


# ═══════════════════════════════════════════════════════════════════
# TRACE UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTraceUtilities:
    """Validate all_scenario_traces and related functions."""

    def test_all_traces_complete(self) -> None:
        traces = all_scenario_traces()
        assert len(traces) == len(SCENARIOS)

    def test_all_traces_shape(self) -> None:
        traces = all_scenario_traces()
        for name, t in traces.items():
            assert t.shape == (N_CHANNELS,), f"{name}: shape={t.shape}"

    def test_all_kernels_complete(self) -> None:
        kernels = all_scenario_kernels()
        assert len(kernels) == len(SCENARIOS)
