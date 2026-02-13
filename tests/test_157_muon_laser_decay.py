"""Tests for muon-laser decay interference closure.

Validates the 7 theorems from the GCD kernel analysis of King & Liu
(2025) "Vacuum muon decay and interaction with laser pulses"
Phys. Rev. Lett. 135, 251802 (arXiv:2507.16891).

A finite-extent laser pulse creates quantum which-way interference
between muon decay pathways, suppressing the vacuum decay rate by
up to 50%.  The master parameter Ω = ξ_μ²Φ⟨g²⟩/(2η_μ) is entirely
classical but controls a purely quantum effect.

Test coverage:
  - Scenario database integrity (8 scenarios, parameters, regimes)
  - 8-channel trace construction (bounds, independence, orientation)
  - Tier-1 kernel identities for all scenarios
  - All 7 theorems (57 individual tests)
  - R[Ω] computation (numerical, perturbative, asymptotic)
  - Channel autopsy (IC killer identification)
  - Physics consistency (constants, scaling relations)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from closures.quantum_mechanics.muon_laser_decay import (
    _R_PRECOMPUTED,
    ALPHA_EM,
    CHANNEL_NAMES,
    DELTA,
    EPSILON,
    M_E_MEV,
    M_MU_MEV,
    N_CHANNELS,
    SCENARIO_ORDER,
    SCENARIOS,
    TAU_MU_US,
    WEIGHTS,
    MuonLaserScenario,
    R_Omega_asymptotic,
    R_Omega_numerical,
    R_Omega_perturbative,
    TheoremResult,
    all_scenario_kernels,
    all_scenario_traces,
    channel_autopsy,
    run_all_theorems,
    scenario_trace,
    summary_report,
    verify_channel_independence,
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: SCENARIO DATABASE INTEGRITY
# ═══════════════════════════════════════════════════════════════════


class TestScenarioDatabase:
    """Verify the scenario database matches King & Liu (2025)."""

    def test_eight_scenarios(self) -> None:
        assert len(SCENARIOS) == 8
        assert set(SCENARIOS.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}

    def test_scenario_order(self) -> None:
        assert SCENARIO_ORDER == ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_scenario_is_frozen_dataclass(self, name: str) -> None:
        sc = SCENARIOS[name]
        assert isinstance(sc, MuonLaserScenario)
        with pytest.raises(AttributeError):
            sc.xi_e = 999.0  # type: ignore[misc]

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_xi_e_positive(self, name: str) -> None:
        assert SCENARIOS[name].xi_e > 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_N_cycles_positive(self, name: str) -> None:
        assert SCENARIOS[name].N_cycles > 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_eta_mu_positive(self, name: str) -> None:
        assert SCENARIOS[name].eta_mu > 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_wavelength_positive(self, name: str) -> None:
        assert SCENARIOS[name].wavelength_nm > 0

    def test_xi_e_spans_range(self) -> None:
        """ξ_e should span at least 1 OOM."""
        vals = [SCENARIOS[s].xi_e for s in SCENARIO_ORDER]
        assert max(vals) / min(vals) >= 10

    def test_eta_mu_spans_range(self) -> None:
        """η_μ should span at least 1 OOM."""
        vals = [SCENARIOS[s].eta_mu for s in SCENARIO_ORDER]
        assert max(vals) / min(vals) >= 10

    def test_regime_classification(self) -> None:
        regimes = {s: SCENARIOS[s].regime for s in SCENARIO_ORDER}
        valid = {"perturbative", "transition", "asymptotic"}
        for s, r in regimes.items():
            assert r in valid, f"{s}: unknown regime '{r}'"

    def test_at_least_two_regimes(self) -> None:
        regimes = {SCENARIOS[s].regime for s in SCENARIO_ORDER}
        assert len(regimes) >= 2

    def test_all_scenarios_in_precomputed(self) -> None:
        for s in SCENARIO_ORDER:
            assert s in _R_PRECOMPUTED

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_R_in_valid_range(self, name: str) -> None:
        """R[Ω] must be in (0, 1]."""
        R = _R_PRECOMPUTED[name]
        assert 0 < R <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_xi_mu_is_delta_times_xi_e(self, name: str) -> None:
        """ξ_μ = δ × ξ_e."""
        sc = SCENARIOS[name]
        assert abs(sc.xi_mu - DELTA * sc.xi_e) < 1e-15

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_Omega_from_scaling_relation(self, name: str) -> None:
        """Ω = 0.8 × (N/10) × (ξ_e/0.02)² / (η_μ/1e-8)."""
        sc = SCENARIOS[name]
        expected = 0.8 * (sc.N_cycles / 10.0) * (sc.xi_e / 0.02) ** 2 / (sc.eta_mu / 1e-8)
        assert abs(sc.Omega - expected) < 1e-10

    def test_omega_spans_range(self) -> None:
        """Ω should span at least 3 OOM."""
        omegas = [SCENARIOS[s].Omega for s in SCENARIO_ORDER]
        assert max(omegas) / min(omegas) >= 1000


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: R[Ω] COMPUTATION
# ═══════════════════════════════════════════════════════════════════


class TestROmega:
    """Verify R[Ω] computation against analytic limits."""

    def test_R_vacuum_equals_one(self) -> None:
        """R(0) = 1 — no laser means no modification."""
        assert abs(R_Omega_numerical(0.0) - 1.0) < 1e-10

    def test_R_tiny_near_one(self) -> None:
        """R(ε) ≈ 1 for negligible Ω."""
        R = R_Omega_numerical(1e-6)
        assert abs(R - 1.0) < 1e-4

    def test_R_large_near_half(self) -> None:
        """R(Ω≫1) → 1/2 (50% floor)."""
        R = R_Omega_numerical(100.0)
        assert abs(R - 0.5) < 0.01

    def test_R_monotone_small_omega(self) -> None:
        """R decreases monotonically for small Ω."""
        omegas = [0.01, 0.05, 0.1, 0.5, 1.0]
        R_vals = [R_Omega_numerical(om) for om in omegas]
        for i in range(len(R_vals) - 1):
            assert R_vals[i] > R_vals[i + 1], (
                f"R not decreasing: R({omegas[i]})={R_vals[i]:.6f} vs R({omegas[i + 1]})={R_vals[i + 1]:.6f}"
            )

    def test_R_never_below_0_47(self) -> None:
        """R[Ω] is always > 0.47 (never exceeds ~53% suppression)."""
        for om in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            R = R_Omega_numerical(om)
            assert R > 0.47, f"R({om}) = {R:.6f} < 0.47"

    def test_perturbative_leading_order(self) -> None:
        """For Ω = 0.01: R ≈ 1 − 5πΩ/12."""
        Om = 0.01
        R_num = R_Omega_numerical(Om)
        R_lo = 1.0 - 5.0 * math.pi * Om / 12.0
        assert abs(R_num - R_lo) / R_num < 0.01

    def test_perturbative_function_small(self) -> None:
        """R_perturbative matches R_numerical for small Ω to < 0.5%."""
        for Om in [0.001, 0.005, 0.01, 0.05]:
            R_num = R_Omega_numerical(Om)
            R_pert = R_Omega_perturbative(Om)
            rel_err = abs(R_num - R_pert) / R_num
            assert rel_err < 0.005, f"Ω={Om}: perturbative rel error {rel_err:.4f}"

    def test_asymptotic_function_large(self) -> None:
        """R_asymptotic matches R_numerical for large Ω to < 5%."""
        for Om in [20.0, 50.0, 100.0]:
            R_num = R_Omega_numerical(Om)
            R_asym = R_Omega_asymptotic(Om)
            rel_err = abs(R_num - R_asym) / max(abs(R_num), 1e-15)
            assert rel_err < 0.05, f"Ω={Om}: asymptotic rel error {rel_err:.4f}"

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_precomputed_matches_property(self, name: str) -> None:
        """R_Omega property returns the precomputed value."""
        assert SCENARIOS[name].R_Omega == _R_PRECOMPUTED[name]


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: TRACE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestTraceConstruction:
    """Verify 8-channel trace vector properties."""

    def test_channel_names_count(self) -> None:
        assert len(CHANNEL_NAMES) == N_CHANNELS

    def test_weights_sum_to_one(self) -> None:
        assert abs(np.sum(WEIGHTS) - 1.0) < 1e-12

    def test_weights_equal(self) -> None:
        assert all(abs(w - 1.0 / N_CHANNELS) < 1e-15 for w in WEIGHTS)

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_trace_shape(self, name: str) -> None:
        c = scenario_trace(SCENARIOS[name])
        assert c.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_trace_bounds(self, name: str) -> None:
        """All channels in [ε, 1−ε]."""
        c = scenario_trace(SCENARIOS[name])
        assert np.all(c >= EPSILON), f"{name}: channel below ε"
        assert np.all(c <= 1.0 - EPSILON), f"{name}: channel above 1−ε"

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_rate_modification_channel(self, name: str) -> None:
        """Channel 0 is R[Ω]."""
        c = scenario_trace(SCENARIOS[name])
        assert abs(c[0] - max(EPSILON, _R_PRECOMPUTED[name])) < 1e-10

    def test_S1_has_epsilon_channels(self) -> None:
        """S1 (ultra-weak) should have multiple channels near ε."""
        c = scenario_trace(SCENARIOS["S1"])
        n_low = int(np.sum(c < 0.01))
        assert n_low >= 3, f"S1 only has {n_low} channels below 0.01"

    def test_all_traces_dict(self) -> None:
        traces = all_scenario_traces()
        assert len(traces) == 8
        for name in SCENARIO_ORDER:
            assert name in traces
            assert traces[name].shape == (N_CHANNELS,)

    def test_traces_differ_across_scenarios(self) -> None:
        """No two scenarios should produce identical traces."""
        traces = all_scenario_traces()
        names = list(traces.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not np.allclose(traces[names[i]], traces[names[j]]), (
                    f"{names[i]} and {names[j]} have identical traces"
                )


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: CHANNEL INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════


class TestChannelIndependence:
    """Verify channel matrix properties."""

    @pytest.fixture()
    def indep(self) -> dict[str, Any]:
        return verify_channel_independence()

    def test_rank_at_least_7(self, indep: dict[str, Any]) -> None:
        """Channel matrix should have rank ≥ 7."""
        assert indep["rank"] >= 7

    def test_no_perfect_degeneracy(self, indep: dict[str, Any]) -> None:
        """No |ρ| = 1.0 between any pair."""
        assert indep["max_offdiag_correlation"] < 0.999

    def test_worst_pair_identified(self, indep: dict[str, Any]) -> None:
        assert "worst_pair" in indep
        assert len(indep["worst_pair"]) == 2


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: TIER-1 KERNEL IDENTITIES
# ═══════════════════════════════════════════════════════════════════


class TestTier1:
    """Verify Tier-1 identities for all scenarios."""

    @pytest.fixture()
    def kernels(self) -> dict[str, dict[str, float]]:
        return all_scenario_kernels()

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_plus_omega_equals_one(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert abs(k["F"] + k["omega"] - 1.0) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_leq_F(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["IC"] <= k["F"] + 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_equals_exp_kappa(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert abs(k["IC"] - math.exp(k["kappa"])) < 1e-6

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_in_valid_range(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert EPSILON < k["F"] < 1.0 - EPSILON

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_delta_nonnegative(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["delta"] >= -1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_kappa_negative(self, kernels: dict, name: str) -> None:
        """κ < 0 since IC < 1."""
        k = kernels[name]
        assert k["kappa"] < 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_S_nonnegative(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["S"] >= 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_C_nonnegative(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["C"] >= 0


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: THEOREM VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestTheorems:
    """Validate all 7 theorems from the muon-laser decay closure."""

    @pytest.fixture()
    def results(self) -> list[TheoremResult]:
        return run_all_theorems()

    def test_seven_theorems(self, results: list[TheoremResult]) -> None:
        assert len(results) == 7

    def test_all_proven(self, results: list[TheoremResult]) -> None:
        for r in results:
            assert r.verdict == "PROVEN", f"{r.name}: {r.verdict} ({r.n_passed}/{r.n_tests})\n{r.details}"

    def test_total_test_count(self, results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in results)
        assert total >= 50, f"Only {total} subtests"

    def test_zero_failures(self, results: list[TheoremResult]) -> None:
        total_failed = sum(r.n_failed for r in results)
        assert total_failed == 0

    @pytest.mark.parametrize(
        ("idx", "name"),
        [
            (0, "T-MLD-1"),
            (1, "T-MLD-2"),
            (2, "T-MLD-3"),
            (3, "T-MLD-4"),
            (4, "T-MLD-5"),
            (5, "T-MLD-6"),
            (6, "T-MLD-7"),
        ],
    )
    def test_theorem_name(self, results: list[TheoremResult], idx: int, name: str) -> None:
        assert results[idx].name == name

    def test_T1_tier1_count(self, results: list[TheoremResult]) -> None:
        """T-MLD-1 should have 24 tests (3 identities × 8 scenarios)."""
        assert results[0].n_tests == 24

    def test_T2_monotonicity(self, results: list[TheoremResult]) -> None:
        """T-MLD-2: R decreasing across perturbative sequence."""
        r = results[1]
        assert r.verdict == "PROVEN"
        assert r.n_passed >= 5

    def test_T3_floor(self, results: list[TheoremResult]) -> None:
        """T-MLD-3: 50% floor for Ω > 2."""
        r = results[2]
        assert r.verdict == "PROVEN"

    def test_T4_F_ordering(self, results: list[TheoremResult]) -> None:
        """T-MLD-4: S7 has highest F, S1 has lowest F."""
        r = results[3]
        assert r.verdict == "PROVEN"
        assert r.details["max_F_scenario"] == "S7"
        assert r.details["min_F_scenario"] == "S1"

    def test_T5_IC_killed(self, results: list[TheoremResult]) -> None:
        """T-MLD-5: S1 has lowest IC due to ε channels."""
        r = results[4]
        assert r.verdict == "PROVEN"
        assert r.details["min_IC_scenario"] == "S1"

    def test_T6_perturbative(self, results: list[TheoremResult]) -> None:
        """T-MLD-6: Perturbative limit verified."""
        r = results[5]
        assert r.verdict == "PROVEN"

    def test_T7_anticorrelation(self, results: list[TheoremResult]) -> None:
        """T-MLD-7: Suppression ↑ while Δ/F ↓ in perturbative sequence."""
        r = results[6]
        assert r.verdict == "PROVEN"
        assert r.details["spearman_supp_deltaF"] < 0


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CHANNEL AUTOPSY
# ═══════════════════════════════════════════════════════════════════


class TestChannelAutopsy:
    """Verify channel autopsy diagnostics."""

    @pytest.fixture()
    def autopsy(self) -> dict[str, Any]:
        return channel_autopsy()

    def test_all_scenarios_present(self, autopsy: dict[str, Any]) -> None:
        for s in SCENARIO_ORDER:
            assert s in autopsy

    def test_S1_has_killers(self, autopsy: dict[str, Any]) -> None:
        """S1 should have IC-killer channels (near ε)."""
        killers = autopsy["S1"]["IC_killers"]
        assert len(killers) >= 3, f"S1 only has {len(killers)} killers"

    def test_S3_has_no_killers(self, autopsy: dict[str, Any]) -> None:
        """S3 (reference, balanced) should have no IC killers."""
        killers = autopsy["S3"]["IC_killers"]
        assert len(killers) == 0, f"S3 has unexpected killers: {killers}"

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_autopsy_has_IC_killers(self, autopsy: dict[str, Any], name: str) -> None:
        assert "IC_killers" in autopsy[name]

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_autopsy_has_n_killers(self, autopsy: dict[str, Any], name: str) -> None:
        assert "n_killers" in autopsy[name]
        assert autopsy[name]["n_killers"] == len(autopsy[name]["IC_killers"])


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: PHYSICS CONSISTENCY
# ═══════════════════════════════════════════════════════════════════


class TestPhysicsConsistency:
    """Verify physical constants and relations."""

    def test_muon_mass(self) -> None:
        """PDG 2024 muon mass."""
        assert abs(M_MU_MEV - 105.66) < 0.01

    def test_electron_mass(self) -> None:
        assert abs(M_E_MEV - 0.511) < 0.001

    def test_delta_mass_ratio(self) -> None:
        """δ = m_e/m_μ ≈ 1/207."""
        assert abs(DELTA - M_E_MEV / M_MU_MEV) < 1e-15
        assert abs(1.0 / DELTA - 206.77) < 0.1

    def test_alpha_em(self) -> None:
        assert abs(ALPHA_EM - 1.0 / 137.036) < 1e-6

    def test_muon_lifetime(self) -> None:
        assert abs(TAU_MU_US - 2.197) < 0.001

    def test_P_Compton_positive(self) -> None:
        for s in SCENARIO_ORDER:
            assert SCENARIOS[s].P_Compton > 0

    def test_P_decay_positive(self) -> None:
        for s in SCENARIO_ORDER:
            assert SCENARIOS[s].P_decay > 0

    def test_P_Compton_scaling(self) -> None:
        """P_Compton ∝ ξ_e² × N."""
        s3 = SCENARIOS["S3"]
        s5 = SCENARIOS["S5"]
        # Both have N=10, so ratio should be (ξ_e5/ξ_e3)²
        expected_ratio = (s5.xi_e / s3.xi_e) ** 2
        actual_ratio = s5.P_Compton / s3.P_Compton
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10

    def test_signal_purity_S1_highest(self) -> None:
        """S1 (weak laser) has the purest decay signal."""
        purities = {s: SCENARIOS[s].P_decay / (SCENARIOS[s].P_decay + SCENARIOS[s].P_Compton) for s in SCENARIO_ORDER}
        assert max(purities, key=lambda s: purities[s]) == "S1"

    def test_signal_purity_S7_lowest(self) -> None:
        """S7 (strong, long pulse) has the most Compton background."""
        purities = {s: SCENARIOS[s].P_decay / (SCENARIOS[s].P_decay + SCENARIOS[s].P_Compton) for s in SCENARIO_ORDER}
        assert min(purities, key=lambda s: purities[s]) == "S7"


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: KERNEL STRUCTURE
# ═══════════════════════════════════════════════════════════════════


class TestKernelStructure:
    """Verify kernel ordering and structure across scenarios."""

    @pytest.fixture()
    def kernels(self) -> dict[str, dict[str, float]]:
        return all_scenario_kernels()

    def test_S1_lowest_F(self, kernels: dict) -> None:
        F_vals = {s: kernels[s]["F"] for s in SCENARIO_ORDER}
        assert min(F_vals, key=lambda s: F_vals[s]) == "S1"

    def test_S7_highest_F(self, kernels: dict) -> None:
        F_vals = {s: kernels[s]["F"] for s in SCENARIO_ORDER}
        assert max(F_vals, key=lambda s: F_vals[s]) == "S7"

    def test_S1_lowest_IC(self, kernels: dict) -> None:
        IC_vals = {s: kernels[s]["IC"] for s in SCENARIO_ORDER}
        assert min(IC_vals, key=lambda s: IC_vals[s]) == "S1"

    def test_S1_highest_delta_over_F(self, kernels: dict) -> None:
        """S1 has the highest Δ/F (most heterogeneous)."""
        dF = {s: kernels[s]["delta"] / kernels[s]["F"] for s in SCENARIO_ORDER}
        assert max(dF, key=lambda s: dF[s]) == "S1"

    def test_asymptotic_F_higher_than_S1(self, kernels: dict) -> None:
        """All asymptotic scenarios have F > S1.F."""
        F_S1 = kernels["S1"]["F"]
        for s in SCENARIO_ORDER:
            if SCENARIOS[s].regime == "asymptotic":
                assert kernels[s]["F"] > F_S1

    def test_F_range_meaningful(self, kernels: dict) -> None:
        """F should span at least 0.2 across scenarios."""
        F_vals = [kernels[s]["F"] for s in SCENARIO_ORDER]
        assert max(F_vals) - min(F_vals) > 0.2

    def test_IC_spans_multiple_OOM(self, kernels: dict) -> None:
        """IC range should exceed 3 OOM."""
        IC_vals = [kernels[s]["IC"] for s in SCENARIO_ORDER]
        ratio = max(IC_vals) / max(min(IC_vals), 1e-15)
        assert ratio > 1000


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════


class TestSummaryReport:
    """Verify the summary_report function."""

    def test_summary_returns_string(self) -> None:
        report = summary_report()
        assert isinstance(report, str)

    def test_summary_contains_proven(self) -> None:
        report = summary_report()
        assert "PROVEN" in report

    def test_summary_contains_7_of_7(self) -> None:
        report = summary_report()
        assert "7/7" in report

    def test_summary_contains_title(self) -> None:
        report = summary_report()
        assert "MUON-LASER DECAY" in report
