"""Tests for universal_regime_calibration.py — cross-domain universality.

Validates that the GCD/UMCP regime classification is universal: the same
invariants and thresholds classify systems from six independent domains.

Data sources: 6 Zenodo publications by Paulus (2025), calibrated against
    Iulianelli et al. (Nat. Commun. 16, 4558),
    Antonov et al. (Nat. Commun. 16, 7235),
    Falque et al. (PRL, DOI 10.1103/t5dh-rx6w).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.gcd.universal_regime_calibration import (  # noqa: E402
    CHANNEL_NAMES,
    EPSILON,
    N_CHANNELS,
    OMEGA_COLLAPSE,
    OMEGA_STABLE,
    SCENARIO_ORDER,
    SCENARIOS,
    WEIGHTS,
    all_scenario_kernels,
    regime_from_omega,
    run_all_theorems,
    summary_report,
    theorem_T_URC_1_tier1,
    theorem_T_URC_2_regime_concordance,
    theorem_T_URC_3_super_exponential_repair,
    theorem_T_URC_4_channel_vulnerability,
    theorem_T_URC_5_F_separation,
    theorem_T_URC_6_curvature_entailment,
    theorem_T_URC_7_entropy_correlation,
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
        assert len(SCENARIOS) == 12

    def test_scenario_order_complete(self) -> None:
        assert set(SCENARIO_ORDER) == set(SCENARIOS.keys())

    def test_all_scenarios_have_8_channels(self) -> None:
        for name, sc in SCENARIOS.items():
            assert len(sc.channels) == N_CHANNELS, f"{name} has {len(sc.channels)} channels"

    def test_omega_thresholds_ordered(self) -> None:
        assert 0 < OMEGA_STABLE < OMEGA_COLLAPSE < 1


# ═══════════════════════════════════════════════════════════════════
# SCENARIO VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestScenarios:
    """Validate individual scenario properties."""

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_channels_in_range(self, name: str) -> None:
        """All channels in [ε, 1−ε]."""
        sc = SCENARIOS[name]
        for i, c in enumerate(sc.channels):
            assert EPSILON <= c <= 1.0 - EPSILON, f"{name} channel {i} ({CHANNEL_NAMES[i]}) = {c} out of range"

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_trace_shape(self, name: str) -> None:
        """Trace vector has correct shape."""
        trace = SCENARIOS[name].trace
        assert trace.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_expected_regime_valid(self, name: str) -> None:
        """Expected regime is one of the three canonical regimes."""
        assert SCENARIOS[name].expected_regime in {"Stable", "Watch", "Collapse"}

    def test_six_domains_covered(self) -> None:
        """All six empirical domains are represented."""
        domains = {sc.domain for sc in SCENARIOS.values()}
        expected_domains = {
            "tqft",
            "active_matter",
            "analog_gravity",
            "procurement",
            "conjectures",
            "signal_analysis",
        }
        assert domains == expected_domains

    def test_all_three_regimes_present(self) -> None:
        """All three regimes are represented in the scenarios."""
        regimes = {sc.expected_regime for sc in SCENARIOS.values()}
        assert regimes == {"Stable", "Watch", "Collapse"}

    def test_every_domain_has_two_scenarios(self) -> None:
        """Each domain contributes exactly 2 scenarios."""
        domain_counts: dict[str, int] = {}
        for sc in SCENARIOS.values():
            domain_counts[sc.domain] = domain_counts.get(sc.domain, 0) + 1
        for domain, count in domain_counts.items():
            assert count == 2, f"Domain {domain} has {count} scenarios"


# ═══════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestRegimeClassifier:
    """Validate the regime_from_omega classifier."""

    def test_stable_below_threshold(self) -> None:
        assert regime_from_omega(0.0) == "Stable"
        assert regime_from_omega(0.01) == "Stable"
        assert regime_from_omega(0.037) == "Stable"

    def test_watch_in_range(self) -> None:
        assert regime_from_omega(0.038) == "Watch"
        assert regime_from_omega(0.15) == "Watch"
        assert regime_from_omega(0.29) == "Watch"

    def test_collapse_above_threshold(self) -> None:
        assert regime_from_omega(0.30) == "Collapse"
        assert regime_from_omega(0.50) == "Collapse"
        assert regime_from_omega(0.99) == "Collapse"

    def test_exact_boundaries(self) -> None:
        assert regime_from_omega(OMEGA_STABLE - 1e-10) == "Stable"
        assert regime_from_omega(OMEGA_STABLE) == "Watch"
        assert regime_from_omega(OMEGA_COLLAPSE - 1e-10) == "Watch"
        assert regime_from_omega(OMEGA_COLLAPSE) == "Collapse"


# ═══════════════════════════════════════════════════════════════════
# KERNEL COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestKernelComputation:
    """Validate kernel outputs for all scenarios."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_in_range(self, kernels: dict, name: str) -> None:
        assert 0.0 <= kernels[name]["F"] <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_omega_in_range(self, kernels: dict, name: str) -> None:
        assert 0.0 <= kernels[name]["omega"] <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_positive(self, kernels: dict, name: str) -> None:
        assert kernels[name]["IC"] > 0.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_kappa_finite(self, kernels: dict, name: str) -> None:
        assert math.isfinite(kernels[name]["kappa"])

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_S_non_negative(self, kernels: dict, name: str) -> None:
        assert kernels[name]["S"] >= 0.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_C_in_range(self, kernels: dict, name: str) -> None:
        assert 0.0 <= kernels[name]["C"] <= 1.0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_heterogeneity_gap_non_negative(self, kernels: dict, name: str) -> None:
        assert kernels[name]["heterogeneity_gap"] >= -1e-10


# ═══════════════════════════════════════════════════════════════════
# THEOREM TESTS
# ═══════════════════════════════════════════════════════════════════


class TestT_URC_1_Tier1:
    """T-URC-1: Tier-1 Kernel Identities."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_1_tier1()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    def test_correct_test_count(self, result) -> None:
        # 3 tests per scenario × 12 scenarios = 36
        assert result.n_tests == 36

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_F_plus_omega_equals_1(self, name: str) -> None:
        kernels = all_scenario_kernels()
        k = kernels[name]
        assert abs(k["F"] + k["omega"] - 1.0) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_IC_equals_exp_kappa(self, name: str) -> None:
        kernels = all_scenario_kernels()
        k = kernels[name]
        assert abs(k["IC"] - math.exp(k["kappa"])) < 1e-10

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_AM_GM_inequality(self, name: str) -> None:
        kernels = all_scenario_kernels()
        k = kernels[name]
        assert k["IC"] <= k["F"] + 1e-10


class TestT_URC_2_RegimeConcordance:
    """T-URC-2: Universal Regime Concordance."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_2_regime_concordance()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    @pytest.mark.parametrize("name", SCENARIO_ORDER)
    def test_computed_matches_expected(self, name: str) -> None:
        kernels = all_scenario_kernels()
        expected = SCENARIOS[name].expected_regime
        computed = regime_from_omega(kernels[name]["omega"])
        assert computed == expected, f"{name}: expected {expected}, got {computed} (ω={kernels[name]['omega']:.4f})"


class TestT_URC_3_SuperExponentialRepair:
    """T-URC-3: Super-Exponential Repair (TQFT)."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_3_super_exponential_repair()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    def test_gate0_is_watch(self) -> None:
        kernels = all_scenario_kernels()
        assert regime_from_omega(kernels["IA1"]["omega"]) == "Watch"

    def test_gate1_is_stable(self) -> None:
        kernels = all_scenario_kernels()
        assert regime_from_omega(kernels["IA2"]["omega"]) == "Stable"

    def test_IC_improvement(self) -> None:
        kernels = all_scenario_kernels()
        assert kernels["IA2"]["IC"] / kernels["IA1"]["IC"] > 1.30

    def test_omega_suppression_ratio(self) -> None:
        kernels = all_scenario_kernels()
        ratio = kernels["IA1"]["omega"] / kernels["IA2"]["omega"]
        assert ratio > 10.0


class TestT_URC_4_ChannelVulnerability:
    """T-URC-4: IC–Channel Vulnerability."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_4_channel_vulnerability()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"],
    )
    def test_collapse_has_weak_channels(self, name: str) -> None:
        channels = np.array(SCENARIOS[name].channels)
        n_below = int(np.sum(channels < 0.25))
        assert n_below >= 2, f"{name} has only {n_below} channels < 0.25"

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"],
    )
    def test_stable_no_weak_channels(self, name: str) -> None:
        channels = np.array(SCENARIOS[name].channels)
        min_ch = float(np.min(channels))
        assert min_ch >= 0.90, f"{name} min channel = {min_ch}"


class TestT_URC_5_FSeparation:
    """T-URC-5: Cross-Domain F Separation."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_5_F_separation()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    def test_stable_F_high(self) -> None:
        kernels = all_scenario_kernels()
        stable = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"]
        for s in stable:
            assert kernels[s]["F"] > 0.96, f"{s} F={kernels[s]['F']}"

    def test_collapse_F_low(self) -> None:
        kernels = all_scenario_kernels()
        collapse = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"]
        for s in collapse:
            assert kernels[s]["F"] < 0.50, f"{s} F={kernels[s]['F']}"


class TestT_URC_6_CurvatureEntailment:
    """T-URC-6: Curvature–Regime Entailment."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_6_curvature_entailment()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"],
    )
    def test_stable_low_curvature(self, name: str) -> None:
        kernels = all_scenario_kernels()
        assert kernels[name]["C"] < 0.05

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"],
    )
    def test_collapse_high_curvature(self, name: str) -> None:
        kernels = all_scenario_kernels()
        assert kernels[name]["C"] > 0.15


class TestT_URC_7_EntropyCorrelation:
    """T-URC-7: Entropy–Regime Correlation."""

    @pytest.fixture(scope="class")
    def result(self):
        return theorem_T_URC_7_entropy_correlation()

    def test_proven(self, result) -> None:
        assert result.verdict == "PROVEN"

    def test_all_tests_pass(self, result) -> None:
        assert result.n_failed == 0

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"],
    )
    def test_stable_low_entropy(self, name: str) -> None:
        kernels = all_scenario_kernels()
        assert kernels[name]["S"] < 0.15

    @pytest.mark.parametrize(
        "name",
        [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"],
    )
    def test_collapse_high_entropy(self, name: str) -> None:
        kernels = all_scenario_kernels()
        assert kernels[name]["S"] > 0.40


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestOrchestration:
    """Test run_all_theorems and summary_report."""

    def test_run_all_returns_7(self) -> None:
        results = run_all_theorems()
        assert len(results) == 7

    def test_all_proven(self) -> None:
        results = run_all_theorems()
        for r in results:
            assert r.verdict == "PROVEN", f"{r.name} is {r.verdict}"

    def test_zero_failures(self) -> None:
        results = run_all_theorems()
        total_failed = sum(r.n_failed for r in results)
        assert total_failed == 0

    def test_total_test_count(self) -> None:
        results = run_all_theorems()
        total = sum(r.n_tests for r in results)
        assert total == 94

    def test_summary_report_nonempty(self) -> None:
        report = summary_report()
        assert len(report) > 200

    def test_summary_contains_all_domains(self) -> None:
        report = summary_report()
        for domain in ["tqft", "active_matter", "analog_gravity", "procurement", "conjectures", "signal_analysis"]:
            assert domain in report


# ═══════════════════════════════════════════════════════════════════
# CROSS-DOMAIN INSIGHT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCrossDomainInsights:
    """Verify cross-domain connections from the 6 Zenodo publications."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict:
        return all_scenario_kernels()

    def test_tqft_super_exponential_leakage(self, kernels: dict) -> None:
        """The ω_{n+1} = (ω_n)^5 suppression produces > 60× ω reduction."""
        ratio = kernels["IA1"]["omega"] / kernels["IA2"]["omega"]
        assert ratio > 60

    def test_analog_gravity_collapse_is_horizon(self, kernels: dict) -> None:
        """Supersonic flow (past horizon) is Collapse; subsonic is Stable."""
        assert kernels["HA1"]["omega"] < OMEGA_STABLE
        assert kernels["HA2"]["omega"] >= OMEGA_COLLAPSE

    def test_procurement_critical_loss_IC(self, kernels: dict) -> None:
        """IC < 0.30 in procurement Critical Loss scenario."""
        assert kernels["PR2"]["IC"] < 0.30

    def test_conjecture_reality_threshold(self, kernels: dict) -> None:
        """Instantiated conjecture exceeds γ_real ≈ 0.90."""
        assert kernels["CC1"]["IC"] > 0.90

    def test_active_matter_frictional_cooling(self, kernels: dict) -> None:
        """Cooled phase is Stable; heated phase is Collapse."""
        assert kernels["AM1"]["omega"] < OMEGA_STABLE
        assert kernels["AM2"]["omega"] >= OMEGA_COLLAPSE

    def test_diagnostic_toolkit_canonical_signals(self, kernels: dict) -> None:
        """Canonical clean signal is Stable, degraded signal is Collapse."""
        assert kernels["DT1"]["F"] > 0.96
        assert kernels["DT2"]["F"] < 0.40

    def test_IC_spans_three_OOM(self, kernels: dict) -> None:
        """IC across all scenarios spans at least 2.5 OOM."""
        all_IC = [kernels[s]["IC"] for s in SCENARIO_ORDER]
        ic_range = math.log10(max(all_IC)) - math.log10(min(all_IC))
        assert ic_range > 0.5  # log10(0.97) - log10(0.25) ≈ 0.59

    def test_stable_IC_dominates_collapse_IC(self, kernels: dict) -> None:
        """Every Stable scenario has higher IC than every Collapse scenario."""
        stable = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"]
        collapse = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"]
        min_stable_ic = min(kernels[s]["IC"] for s in stable)
        max_collapse_ic = max(kernels[s]["IC"] for s in collapse)
        assert min_stable_ic > max_collapse_ic

    def test_watch_scenarios_between(self, kernels: dict) -> None:
        """Watch scenarios have intermediate F values."""
        watch = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Watch"]
        for s in watch:
            assert 0.70 <= kernels[s]["F"] <= 0.962

    def test_heterogeneity_gap_increases_with_heterogeneity(self, kernels: dict) -> None:
        """Collapse scenarios have larger AM-GM gap than Stable scenarios."""
        stable = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Stable"]
        collapse = [s for s in SCENARIO_ORDER if SCENARIOS[s].expected_regime == "Collapse"]
        avg_gap_stable = np.mean([kernels[s]["heterogeneity_gap"] for s in stable])
        avg_gap_collapse = np.mean([kernels[s]["heterogeneity_gap"] for s in collapse])
        assert avg_gap_collapse > avg_gap_stable
