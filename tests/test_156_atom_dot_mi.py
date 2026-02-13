"""Tests for atom dot MI transition closure.

Validates the 7 theorems from the GCD kernel analysis of Donnelly et al.
(2026) "Large-scale analogue quantum simulation using atom dot arrays"
(Nature, DOI: 10.1038/s41586-025-10053-7).

Zenodo data: DOI 10.5281/zenodo.17782840

Test coverage:
  - Device database integrity (6 devices, parameters, regimes)
  - 8-channel trace construction (bounds, independence, orientation)
  - Tier-1 kernel identities for all devices
  - All 7 theorems (30 individual tests)
  - Temperature sweep kernel maps
  - Monte Carlo uncertainty propagation
  - Channel autopsy (IC killer identification)
  - Data file loading (.npz)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from closures.quantum_mechanics import atom_dot_mi_transition as atom_dot
from closures.quantum_mechanics.atom_dot_mi_transition import (
    DEVICE_ORDER,
    DEVICES,
    EPSILON,
    HALL_DEVICE_A,
    N_CHANNELS,
    AtomDotDevice,
    TheoremResult,
    all_device_kernels,
    all_device_traces,
    channel_autopsy,
    device_trace,
    load_device_data,
    load_summary_data,
    mc_kernel_uncertainty,
    run_all_theorems,
    summary_report,
    temperature_sweep_kernels,
    verify_channel_independence,
)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DATABASE INTEGRITY
# ═══════════════════════════════════════════════════════════════════


class TestDeviceDatabase:
    """Verify the device database matches published data."""

    def test_six_devices(self) -> None:
        assert len(DEVICES) == 6
        assert set(DEVICES.keys()) == {"A", "B", "C", "D", "E", "F"}

    def test_device_order(self) -> None:
        assert DEVICE_ORDER == ["A", "B", "C", "D", "E", "F"]

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_device_is_frozen_dataclass(self, name: str) -> None:
        dev = DEVICES[name]
        assert isinstance(dev, AtomDotDevice)
        with pytest.raises(AttributeError):
            dev.t_hop = 999.0  # type: ignore[misc]

    def test_hubbard_parameters_positive(self) -> None:
        for d in DEVICES.values():
            assert d.t_hop > 0, f"{d.name}: t_hop must be positive"
            assert d.U_int > 0, f"{d.name}: U_int must be positive"
            assert d.V_nn > 0, f"{d.name}: V_nn must be positive"

    def test_ut_ratio_ordering(self) -> None:
        """Primary MI sequence: U/t increases A → B → C → D."""
        ut = [DEVICES[d].U_over_t for d in ["A", "B", "C", "D"]]
        assert all(ut[i] < ut[i + 1] for i in range(len(ut) - 1))

    def test_device_E_highest_ut(self) -> None:
        """Device E has the highest U/t (most insulating)."""
        ut = {d: DEVICES[d].U_over_t for d in DEVICE_ORDER}
        assert max(ut, key=ut.get) == "E"  # type: ignore[arg-type]

    def test_regime_classification(self) -> None:
        assert DEVICES["A"].regime == "metallic"
        assert DEVICES["B"].regime == "metallic"
        assert DEVICES["C"].regime == "crossover"
        assert DEVICES["D"].regime == "insulating"
        assert DEVICES["E"].regime == "insulating"
        assert DEVICES["F"].regime == "insulating"

    def test_g_max_ranges(self) -> None:
        """g_max should span ~3 orders of magnitude."""
        g_vals = [DEVICES[d].g_max for d in DEVICE_ORDER]
        ratio = max(g_vals) / min(g_vals)
        assert ratio > 100, f"g_max ratio {ratio:.1f} < 100"

    def test_mott_gap_positive(self) -> None:
        for d in DEVICES.values():
            assert d.mott_gap_meV > 0

    def test_V_over_U_less_than_one(self) -> None:
        for d in DEVICES.values():
            assert 0 < d.V_over_U < 1

    def test_hall_device_A(self) -> None:
        assert len(HALL_DEVICE_A["T_K"]) == 8
        assert len(HALL_DEVICE_A["R_H_Ohm_per_T"]) == 8
        # Hall coefficient should be negative (n-type silicon)
        assert all(r < 0 for r in HALL_DEVICE_A["R_H_Ohm_per_T"])


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: TRACE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestTraceConstruction:
    """Verify 8-channel trace vector properties."""

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_trace_shape(self, name: str) -> None:
        c = device_trace(DEVICES[name])
        assert c.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_trace_bounds(self, name: str) -> None:
        """All channels in [ε, 1−ε]."""
        c = device_trace(DEVICES[name])
        assert np.all(c >= EPSILON), f"{name}: channel below ε"
        assert np.all(c <= 1.0 - EPSILON), f"{name}: channel above 1−ε"

    def test_metallic_positive_orientation(self) -> None:
        """All channels should be higher for metallic devices (on average)."""
        traces = all_device_traces()
        mean_A = np.mean(traces["A"])
        mean_E = np.mean(traces["E"])
        assert mean_A > mean_E, f"Channel orientation wrong: ⟨c_A⟩={mean_A:.4f} should > ⟨c_E⟩={mean_E:.4f}"

    def test_channel_independence_full_rank(self) -> None:
        indep = verify_channel_independence()
        assert indep["full_rank"], f"Rank {indep['rank']} < {indep['max_rank']}"

    def test_channel_independence_rank_six(self) -> None:
        indep = verify_channel_independence()
        assert indep["rank"] == 6, f"Expected rank 6, got {indep['rank']}"

    def test_worst_correlation_below_threshold(self) -> None:
        """No perfect (ρ=1.0) degeneracy."""
        indep = verify_channel_independence()
        assert indep["max_offdiag_correlation"] < 0.999

    def test_all_traces_dict(self) -> None:
        traces = all_device_traces()
        assert len(traces) == 6
        for name in DEVICE_ORDER:
            assert name in traces
            assert traces[name].shape == (N_CHANNELS,)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: TIER-1 KERNEL IDENTITIES
# ═══════════════════════════════════════════════════════════════════


class TestTier1:
    """Verify Tier-1 identities for all devices."""

    @pytest.fixture()
    def kernels(self) -> dict[str, dict[str, float]]:
        return all_device_kernels()

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_F_plus_omega_equals_one(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert abs(k["F"] + k["omega"] - 1.0) < 1e-10

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_IC_leq_F(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["IC"] <= k["F"] + 1e-10

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_IC_equals_exp_kappa(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert abs(k["IC"] - math.exp(k["kappa"])) < 1e-6

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_F_in_valid_range(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert EPSILON < k["F"] < 1.0 - EPSILON

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_delta_nonnegative(self, kernels: dict, name: str) -> None:
        k = kernels[name]
        assert k["delta"] >= -1e-10


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: THEOREMS
# ═══════════════════════════════════════════════════════════════════


class TestTheorems:
    """Run all 7 theorems and verify PROVEN."""

    @pytest.fixture(scope="class")
    def results(self) -> list[TheoremResult]:
        return run_all_theorems()

    @pytest.mark.parametrize("idx", range(7))
    def test_theorem_proven(self, results: list[TheoremResult], idx: int) -> None:
        r = results[idx]
        assert r.verdict == "PROVEN", f"{r.name}: {r.n_passed}/{r.n_tests} passed, details={r.details}"

    @pytest.mark.parametrize("idx", range(7))
    def test_theorem_no_failures(self, results: list[TheoremResult], idx: int) -> None:
        r = results[idx]
        assert r.n_failed == 0

    def test_all_seven_theorems(self, results: list[TheoremResult]) -> None:
        assert len(results) == 7

    def test_total_individual_tests(self, results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in results)
        assert total == 30

    def test_total_passed(self, results: list[TheoremResult]) -> None:
        total = sum(r.n_passed for r in results)
        assert total == 30


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: TEMPERATURE SWEEP KERNEL MAPS
# ═══════════════════════════════════════════════════════════════════


class TestTemperatureSweep:
    """Test G(T) → F(T) kernel trajectory mapping."""

    def test_metallic_sweep_stable(self) -> None:
        sw = temperature_sweep_kernels("A", n_points=10)
        F_range = max(sw["F"]) - min(sw["F"])
        assert F_range < 0.05, f"Device A F range {F_range:.4f} too large"

    def test_insulating_sweep_return(self) -> None:
        sw = temperature_sweep_kernels("E", n_points=10)
        sort_idx = np.argsort(sw["temperature"])
        F_low = sw["F"][sort_idx[0]]
        F_high = sw["F"][sort_idx[-1]]
        assert F_high > F_low, "Device E should show return trajectory"

    def test_sweep_returns_all_keys(self) -> None:
        sw = temperature_sweep_kernels("C", n_points=5)
        for key in ["device", "temperature", "F", "IC", "delta", "kappa", "omega"]:
            assert key in sw

    def test_sweep_array_lengths(self) -> None:
        sw = temperature_sweep_kernels("B", n_points=8)
        n = len(sw["temperature"])
        assert n <= 8
        for key in ["F", "IC", "delta", "kappa", "omega"]:
            assert len(sw[key]) == n

    def test_sweep_tier1_at_each_point(self) -> None:
        sw = temperature_sweep_kernels("D", n_points=5)
        for i in range(len(sw["F"])):
            assert abs(sw["F"][i] + sw["omega"][i] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: MONTE CARLO UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════


class TestMonteCarlo:
    """Test MC uncertainty propagation."""

    @pytest.fixture(scope="class")
    def mc_C(self) -> dict:
        return mc_kernel_uncertainty("C", n_samples=200, rng_seed=42)

    def test_mc_F_uncertainty_small(self, mc_C: dict) -> None:
        assert mc_C["F_std"] < 0.05, f"F_std={mc_C['F_std']:.4f} too large"

    def test_mc_F_mean_consistent(self, mc_C: dict) -> None:
        """MC mean should be close to deterministic value."""
        kernels = all_device_kernels()
        F_det = kernels["C"]["F"]
        assert abs(mc_C["F_mean"] - F_det) < 3 * mc_C["F_std"]

    def test_mc_IC_uncertainty_small(self, mc_C: dict) -> None:
        assert mc_C["IC_std"] < 0.05

    def test_mc_delta_positive(self, mc_C: dict) -> None:
        assert mc_C["delta_mean"] > 0

    def test_mc_sample_count(self, mc_C: dict) -> None:
        assert mc_C["n_samples"] == 200

    def test_mc_insulating_device(self) -> None:
        """MC on insulating device should show larger relative uncertainty."""
        mc_D = mc_kernel_uncertainty("D", n_samples=100, rng_seed=123)
        assert mc_D["F_std"] > 0.001


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CHANNEL AUTOPSY
# ═══════════════════════════════════════════════════════════════════


class TestChannelAutopsy:
    """Verify IC killer identification."""

    @pytest.fixture(scope="class")
    def autopsy(self) -> dict:
        return channel_autopsy()

    def test_metallic_no_killers(self, autopsy: dict) -> None:
        """Metallic devices should have no IC killers (no channels near ε)."""
        assert len(autopsy["A"]["IC_killers"]) == 0
        assert len(autopsy["B"]["IC_killers"]) == 0

    def test_device_E_most_killers(self, autopsy: dict) -> None:
        """Device E (most insulating) should have the most IC killers."""
        n_killers = {d: autopsy[d]["n_low_channels"] for d in DEVICE_ORDER}
        assert n_killers["E"] >= max(n_killers[d] for d in ["A", "B", "C"])

    def test_insulating_has_killers(self, autopsy: dict) -> None:
        for d in ["D", "E", "F"]:
            assert len(autopsy[d]["IC_killers"]) > 0

    def test_autopsy_has_all_devices(self, autopsy: dict) -> None:
        for d in DEVICE_ORDER:
            assert d in autopsy


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: DATA FILES
# ═══════════════════════════════════════════════════════════════════


class TestDataFiles:
    """Verify saved .npz data files."""

    _workspace = Path(__file__).resolve().parents[1]

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_device_npz_exists(self, name: str) -> None:
        fpath = self._workspace / "data" / f"atom_dot_device_{name}.npz"
        assert fpath.exists(), f"Missing {fpath}"

    def test_summary_npz_exists(self) -> None:
        fpath = self._workspace / "data" / "atom_dot_summary.npz"
        assert fpath.exists()

    def test_load_device_data(self) -> None:
        data = load_device_data("A")
        assert "temperature" in data
        assert "conductance" in data
        assert len(data["temperature"]) > 10

    def test_load_summary_data(self) -> None:
        data = load_summary_data()
        assert "devices" in data
        assert "U_over_t" in data
        assert len(data["devices"]) == 6

    @pytest.mark.parametrize("name", DEVICE_ORDER)
    def test_device_npz_has_keys(self, name: str) -> None:
        data = load_device_data(name)
        required = {"temperature", "conductance", "t_hop", "U_int", "U_over_t"}
        for key in required:
            assert key in data, f"Missing key '{key}' in device {name}"


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════


class TestSummaryReport:
    """Verify summary report generation."""

    def test_report_contains_all_theorems(self) -> None:
        report = summary_report()
        for i in range(1, 8):
            assert f"T-ADOT-{i}" in report

    def test_report_contains_device_table(self) -> None:
        report = summary_report()
        for d in DEVICE_ORDER:
            assert d in report

    def test_report_shows_proven(self) -> None:
        report = summary_report()
        assert "7/7 PROVEN" in report


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: PHYSICS CONSISTENCY CHECKS
# ═══════════════════════════════════════════════════════════════════


class TestPhysicsConsistency:
    """Cross-check kernel results against known physics."""

    @pytest.fixture(scope="class")
    def kernels(self) -> dict[str, dict[str, float]]:
        return all_device_kernels()

    def test_F_decreases_through_MI_transition(self, kernels: dict) -> None:
        """F should decrease A → B → C → D (primary MI sequence)."""
        F_vals = [kernels[d]["F"] for d in ["A", "B", "C", "D"]]
        assert all(F_vals[i] > F_vals[i + 1] for i in range(len(F_vals) - 1))

    def test_IC_metallic_much_higher_than_insulating(self, kernels: dict) -> None:
        ic_A = kernels["A"]["IC"]
        ic_E = kernels["E"]["IC"]
        assert ic_A / ic_E > 100

    def test_kappa_more_negative_for_insulators(self, kernels: dict) -> None:
        assert kernels["E"]["kappa"] < kernels["A"]["kappa"]
        assert kernels["D"]["kappa"] < kernels["C"]["kappa"]

    def test_device_A_most_metallic(self, kernels: dict) -> None:
        """Device A has highest F (most metallic)."""
        assert kernels["A"]["F"] == max(kernels[d]["F"] for d in DEVICE_ORDER)

    def test_device_E_most_collapsed(self, kernels: dict) -> None:
        """Device E has lowest F (most collapsed)."""
        assert kernels["E"]["F"] == min(kernels[d]["F"] for d in DEVICE_ORDER)

    def test_AM_GM_gap_positive_everywhere(self, kernels: dict) -> None:
        for d in DEVICE_ORDER:
            assert kernels[d]["delta"] > 0

    def test_crossover_between_metallic_and_insulating(self, kernels: dict) -> None:
        """Device C (crossover) has F between metallic and insulating means."""
        F_C = kernels["C"]["F"]
        F_metal_mean = np.mean([kernels[d]["F"] for d in ["A", "B"]])
        F_insul_mean = np.mean([kernels[d]["F"] for d in ["D", "E", "F"]])
        assert F_insul_mean < F_C < F_metal_mean


# ═══════════════════════════════════════════════════════════════════
# 11. Cross-Domain Insights
# ═══════════════════════════════════════════════════════════════════


class TestUniversalCollapseClassifier:
    """The κ < −2 ↔ IC < 0.15 classifier has perfect accuracy."""

    def test_accuracy_is_perfect(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert result["accuracy"] == 1.0

    def test_precision_is_perfect(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert result["precision"] == 1.0

    def test_recall_is_perfect(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert result["recall"] == 1.0

    def test_no_false_positives(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert result["fp"] == 0

    def test_no_false_negatives(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert result["fn"] == 0

    def test_details_cover_all_devices(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        assert len(result["details"]) == 6
        devices_seen = {d["device"] for d in result["details"]}
        assert devices_seen == set(DEVICE_ORDER)

    def test_all_classified_correct(self) -> None:
        result = atom_dot.universal_collapse_classifier()
        for detail in result["details"]:
            assert detail["correct"], f"Device {detail['device']} misclassified"


class TestCrossoverEntropy:
    """Device C (crossover) is closest to maximum channel entropy."""

    def test_crossover_closest_to_max_entropy(self) -> None:
        ent = atom_dot.crossover_entropy()
        max_S = math.log(atom_dot.N_CHANNELS)
        closest = min(DEVICE_ORDER, key=lambda d: abs(ent[d] - max_S))
        assert closest == "C"

    def test_device_C_entropy_above_1_9(self) -> None:
        ent = atom_dot.crossover_entropy()
        assert ent["C"] > 1.9

    def test_insulating_devices_lower_entropy(self) -> None:
        ent = atom_dot.crossover_entropy()
        for d in ["D", "E"]:
            assert ent[d] < ent["C"], f"Device {d} should have lower S than C"

    def test_all_entropies_positive(self) -> None:
        ent = atom_dot.crossover_entropy()
        for d in DEVICE_ORDER:
            assert ent[d] > 0

    def test_entropy_below_max(self) -> None:
        ent = atom_dot.crossover_entropy()
        max_S = math.log(atom_dot.N_CHANNELS)
        for d in DEVICE_ORDER:
            assert ent[d] <= max_S + 1e-10


class TestKappaCliffSlopes:
    """The steepest κ-cliff identifies the sharpest collapse boundary."""

    def test_returns_five_slopes(self) -> None:
        slopes = atom_dot.kappa_cliff_slopes()
        assert len(slopes) == 5  # 6 devices → 5 intervals

    def test_steepest_slope_is_negative(self) -> None:
        slopes = atom_dot.kappa_cliff_slopes()
        steepest = min(slopes, key=lambda s: s["slope"])
        assert steepest["slope"] < -10

    def test_all_slopes_have_required_keys(self) -> None:
        slopes = atom_dot.kappa_cliff_slopes()
        for s in slopes:
            assert "d0" in s
            assert "d1" in s
            assert "slope" in s
            assert "delta_kappa" in s
            assert "ut_range" in s


class TestDeviceFAnomaly:
    """Device F lifts F above D despite higher U/t."""

    def test_F_F_greater_than_F_D(self) -> None:
        anom = atom_dot.device_f_anomaly()
        assert anom["F_F"] > anom["F_D"]

    def test_F_lift_positive(self) -> None:
        anom = atom_dot.device_f_anomaly()
        assert anom["F_lift"] > 0

    def test_dominant_channel_is_dot_area_or_gap(self) -> None:
        anom = atom_dot.device_f_anomaly()
        # The dominant lift channel is from geometry (dot_area_norm or gap_closure)
        assert anom["dominant_channel"] in ("dot_area_norm", "gap_closure")

    def test_dominant_diff_positive(self) -> None:
        anom = atom_dot.device_f_anomaly()
        assert anom["dominant_diff"] > 0.3

    def test_explanation_nonempty(self) -> None:
        anom = atom_dot.device_f_anomaly()
        assert len(anom["explanation"]) > 50


class TestDeltaOverF:
    """Δ/F measures collapse proximity; E approaches 1.0."""

    def test_device_E_nearest_total_collapse(self) -> None:
        dof = atom_dot.delta_over_F_analysis()
        assert dof["E"]["delta_over_F"] > 0.99

    def test_device_A_furthest_from_collapse(self) -> None:
        dof = atom_dot.delta_over_F_analysis()
        assert dof["A"]["delta_over_F"] < 0.15

    def test_monotone_with_regime(self) -> None:
        """More insulating → higher Δ/F, on average."""
        dof = atom_dot.delta_over_F_analysis()
        metal = np.mean([dof[d]["delta_over_F"] for d in ["A", "B"]])
        insul = np.mean([dof[d]["delta_over_F"] for d in ["D", "E"]])
        assert metal < insul

    def test_all_between_zero_and_one(self) -> None:
        dof = atom_dot.delta_over_F_analysis()
        for d in DEVICE_ORDER:
            assert 0 <= dof[d]["delta_over_F"] <= 1.0 + 1e-10


class TestCriticalExponent:
    """Effective critical exponent from F ~ |U/t − U/t_c|^β."""

    def test_beta_metallic_exists(self) -> None:
        crit = atom_dot.critical_exponent_analog()
        assert "beta_B" in crit

    def test_beta_insulating_exists(self) -> None:
        crit = atom_dot.critical_exponent_analog()
        assert "beta_D" in crit

    def test_beta_metallic_is_negative(self) -> None:
        """F drops as we move away from crossover, so β < 0."""
        crit = atom_dot.critical_exponent_analog()
        assert crit["beta_B"] < 0

    def test_beta_insulating_is_negative(self) -> None:
        crit = atom_dot.critical_exponent_analog()
        assert crit["beta_D"] < 0

    def test_ut_c_is_device_C(self) -> None:
        crit = atom_dot.critical_exponent_analog()
        assert abs(crit["ut_c"] - atom_dot.DEVICES["C"].U_over_t) < 1e-10


class TestInsightsReport:
    """Smoke test for the full insights report."""

    def test_report_contains_all_sections(self) -> None:
        report = atom_dot.insights_report()
        assert "UNIVERSAL COLLAPSE CLASSIFIER" in report
        assert "CROSSOVER ENTROPY" in report
        assert "κ-CLIFF" in report
        assert "Δ/F COLLAPSE PROXIMITY" in report
        assert "DEVICE F ANOMALY" in report
        assert "CRITICAL EXPONENT" in report

    def test_report_nonempty(self) -> None:
        report = atom_dot.insights_report()
        assert len(report) > 500
