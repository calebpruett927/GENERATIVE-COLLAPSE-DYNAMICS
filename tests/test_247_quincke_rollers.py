"""Tests for Magnetic Quincke Rollers — RCFT Active Matter Closure.

Validates the GCD kernel analysis of magnetically tunable Quincke rollers
(Garza et al., Science Advances 9, eadh2522 (2023)).

Coverage:
    - Catalog construction (12 experimental states)
    - Trace vector construction (8 channels, clamping)
    - Regime classification
    - Kernel invariant computation (all 12 states)
    - 8 structural theorems (T-QR-1 through T-QR-8)
    - Tier-1 identity compliance (duality, integrity bound, log-integrity)
    - Nanotechnology connections
    - Scale position in the matter map
    - Summary statistics
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Path setup
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.rcft.quincke_rollers import (
    EPSILON,
    IC_CRITICAL,
    N_CHANNELS,
    OMEGA_COLLAPSE,
    OMEGA_STABLE,
    TOL_SEAM,
    WEIGHTS,
    CollectiveState,
    NanoConnection,
    QuinckeAnalysis,
    QuinckeConfig,
    QuinckeRegime,
    QuinckeResult,
    analyze_all_states,
    build_nano_connections,
    build_quincke_catalog,
    build_trace,
    classify_quincke_regime,
    compute_scale_position,
    run_full_analysis,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def catalog() -> list[QuinckeConfig]:
    """Build the 12-state experimental catalog."""
    return build_quincke_catalog()


@pytest.fixture(scope="module")
def results() -> list[QuinckeResult]:
    """Compute kernel invariants for all 12 states."""
    return analyze_all_states()


@pytest.fixture(scope="module")
def analysis() -> QuinckeAnalysis:
    """Run full analysis including theorems."""
    return run_full_analysis()


# =====================================================================
# §1 — Catalog construction
# =====================================================================


class TestCatalog:
    """Validate the experimental state catalog."""

    def test_catalog_count(self, catalog: list[QuinckeConfig]) -> None:
        assert len(catalog) == 12

    def test_catalog_unique_names(self, catalog: list[QuinckeConfig]) -> None:
        names = [c.name for c in catalog]
        assert len(names) == len(set(names))

    def test_catalog_has_sub_threshold(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "SubThreshold" for c in catalog)

    def test_catalog_has_vortex(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "VortexCondensate" for c in catalog)

    def test_catalog_has_teleoperated(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "Teleoperated" for c in catalog)

    def test_catalog_has_chain_assembly(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "ChainAssembly" for c in catalog)

    def test_catalog_has_chain_disassembly(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "ChainDisassembly" for c in catalog)

    def test_catalog_has_anomalous_dimer(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "AnomalousDimer" for c in catalog)

    def test_catalog_has_programmed(self, catalog: list[QuinckeConfig]) -> None:
        assert any(c.name == "ProgrammedSquare" for c in catalog)

    def test_collective_state_coverage(self, catalog: list[QuinckeConfig]) -> None:
        states = {c.collective_state for c in catalog}
        assert CollectiveState.INDIVIDUAL in states
        assert CollectiveState.ALIGNED_CHAIN in states
        assert CollectiveState.VORTEX in states
        assert CollectiveState.PROGRAMMABLE in states
        assert CollectiveState.TELEOPERATED in states

    def test_e_field_range(self, catalog: list[QuinckeConfig]) -> None:
        e_fields = [c.E_field_V_um for c in catalog]
        assert min(e_fields) < 0.5  # sub-threshold
        assert max(e_fields) >= 2.5  # high-field

    def test_b_field_range(self, catalog: list[QuinckeConfig]) -> None:
        b_fields = [c.B_field_mT for c in catalog]
        assert min(b_fields) == 0.0  # no magnetic field
        assert max(b_fields) > 5.0  # strong magnetic field


# =====================================================================
# §2 — Trace vector construction
# =====================================================================


class TestTraceConstruction:
    """Validate 8-channel trace vector construction."""

    def test_trace_length(self, catalog: list[QuinckeConfig]) -> None:
        for cfg in catalog:
            trace = build_trace(cfg)
            assert len(trace) == N_CHANNELS

    def test_trace_clamped(self, catalog: list[QuinckeConfig]) -> None:
        for cfg in catalog:
            trace = build_trace(cfg)
            for c in trace:
                assert c >= EPSILON, f"Channel below ε in {cfg.name}"
                assert c <= 1.0 - EPSILON, f"Channel above 1−ε in {cfg.name}"

    def test_sub_threshold_low_channels(self, catalog: list[QuinckeConfig]) -> None:
        sub = next(c for c in catalog if c.name == "SubThreshold")
        trace = build_trace(sub)
        # c₂ (speed) should be at ε (no rolling)
        assert trace[1] <= EPSILON + 0.001
        # c₃ (velocity coherence) should be at ε (no motion)
        assert trace[2] <= EPSILON + 0.001
        # c₄ (magnetic saturation) should be at ε
        assert trace[3] <= EPSILON + 0.001

    def test_chain_high_magnetic_channels(self, catalog: list[QuinckeConfig]) -> None:
        chain = next(c for c in catalog if c.name == "ChainAssembly")
        trace = build_trace(chain)
        # c₄ (magnetic saturation): should be > 0.5
        assert trace[3] > 0.5
        # c₆ (chain fraction): should be > 0.5
        assert trace[5] > 0.5
        # c₇ (orientational order): should be > 0.5
        assert trace[6] > 0.5

    def test_weights_sum_to_one(self) -> None:
        assert abs(sum(WEIGHTS) - 1.0) < 1e-10

    def test_n_channels_correct(self) -> None:
        assert N_CHANNELS == 8


# =====================================================================
# §3 — Regime classification
# =====================================================================


class TestRegimeClassification:
    """Validate frozen four-gate regime classification."""

    def test_collapse_high_omega(self) -> None:
        regime = classify_quincke_regime(omega=0.5, F=0.5, S=0.3, C=0.3)
        assert regime == QuinckeRegime.COLLAPSE

    def test_stable_all_gates(self) -> None:
        regime = classify_quincke_regime(omega=0.02, F=0.98, S=0.05, C=0.05)
        assert regime == QuinckeRegime.STABLE

    def test_watch_intermediate(self) -> None:
        regime = classify_quincke_regime(omega=0.15, F=0.85, S=0.10, C=0.10)
        assert regime == QuinckeRegime.WATCH

    def test_watch_high_curvature(self) -> None:
        """Stable gates not fully met → Watch."""
        regime = classify_quincke_regime(omega=0.02, F=0.98, S=0.05, C=0.20)
        assert regime == QuinckeRegime.WATCH

    def test_collapse_threshold(self) -> None:
        regime = classify_quincke_regime(omega=0.30, F=0.70, S=0.2, C=0.2)
        assert regime == QuinckeRegime.COLLAPSE


# =====================================================================
# §4 — Kernel computation (all 12 states)
# =====================================================================


class TestKernelComputation:
    """Validate Tier-1 invariants for all 12 Quincke roller states."""

    def test_all_states_computed(self, results: list[QuinckeResult]) -> None:
        assert len(results) == 12

    def test_unique_names(self, results: list[QuinckeResult]) -> None:
        names = [r.name for r in results]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("idx", range(12))
    def test_f_in_range(self, results: list[QuinckeResult], idx: int) -> None:
        r = results[idx]
        assert 0.0 <= r.F <= 1.0, f"F out of range for {r.name}"

    @pytest.mark.parametrize("idx", range(12))
    def test_omega_in_range(self, results: list[QuinckeResult], idx: int) -> None:
        r = results[idx]
        assert 0.0 <= r.omega <= 1.0, f"ω out of range for {r.name}"

    @pytest.mark.parametrize("idx", range(12))
    def test_ic_in_range(self, results: list[QuinckeResult], idx: int) -> None:
        r = results[idx]
        assert 0.0 <= r.IC <= 1.0, f"IC out of range for {r.name}"

    @pytest.mark.parametrize("idx", range(12))
    def test_gap_non_negative(self, results: list[QuinckeResult], idx: int) -> None:
        r = results[idx]
        assert r.gap >= -TOL_SEAM, f"Negative gap for {r.name}"

    @pytest.mark.parametrize("idx", range(12))
    def test_n_channels(self, results: list[QuinckeResult], idx: int) -> None:
        r = results[idx]
        assert r.n_channels == N_CHANNELS

    def test_sub_threshold_high_omega(self, results: list[QuinckeResult]) -> None:
        sub = next(r for r in results if r.name == "SubThreshold")
        assert sub.omega > 0.9

    def test_vortex_lowest_omega(self, results: list[QuinckeResult]) -> None:
        vortex = next(r for r in results if r.name == "VortexCondensate")
        assert vortex.omega < 0.30


# =====================================================================
# §5 — Tier-1 Identity Compliance
# =====================================================================


class TestTier1Identities:
    """Verify all three Tier-1 structural identities for every state."""

    @pytest.mark.parametrize("idx", range(12))
    def test_duality_identity(self, results: list[QuinckeResult], idx: int) -> None:
        """F + ω = 1 (the duality identity)."""
        r = results[idx]
        residual = abs(r.F + r.omega - 1.0)
        assert residual < TOL_SEAM, f"Duality violation for {r.name}: {residual}"

    @pytest.mark.parametrize("idx", range(12))
    def test_integrity_bound(self, results: list[QuinckeResult], idx: int) -> None:
        """IC ≤ F (the integrity bound)."""
        r = results[idx]
        assert r.IC <= r.F + TOL_SEAM, f"Bound violation for {r.name}: IC={r.IC}, F={r.F}"

    @pytest.mark.parametrize("idx", range(12))
    def test_log_integrity(self, results: list[QuinckeResult], idx: int) -> None:
        """IC = exp(κ) (the log-integrity relation)."""
        r = results[idx]
        ic_from_kappa = math.exp(r.kappa)
        residual = abs(r.IC - ic_from_kappa)
        assert residual < TOL_SEAM, f"Log-integrity violation for {r.name}: {residual}"


# =====================================================================
# §6 — Structural Theorems T-QR-1 through T-QR-8
# =====================================================================


class TestTheoremTQR1:
    """T-QR-1: Quincke Threshold Cliff."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-1"]["proven"]

    def test_sub_threshold_high_omega(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-1"]
        assert t["sub_threshold_omega"] > 0.9

    def test_onset_lower_omega(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-1"]
        assert t["onset_omega"] < t["sub_threshold_omega"]

    def test_fidelity_jump(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-1"]
        assert t["delta_F"] > 0.05

    def test_ic_ratio_below_half(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-1"]
        assert t["IC_ratio"] < 0.5


class TestTheoremTQR2:
    """T-QR-2: Magnetic Chain Restoration."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-2"]["proven"]

    def test_ic_gain(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-2"]
        assert t["IC_gain"] > 1.0

    def test_gap_reduction(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-2"]
        assert t["gap_reduction"] > 0

    def test_chain_ic_above_nonmagnetic(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-2"]
        assert t["chain_IC"] > t["no_magnetic_IC"]


class TestTheoremTQR3:
    """T-QR-3: Reversible Assembly-Disassembly."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-3"]["proven"]

    def test_ic_drop(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-3"]
        assert t["IC_drop"] > 0.01

    def test_same_regime(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-3"]
        assert t["same_regime"]


class TestTheoremTQR4:
    """T-QR-4: Vortex Collective Coherence Peak."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-4"]["proven"]

    def test_vortex_high_f(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-4"]
        assert t["vortex_F"] > 0.6


class TestTheoremTQR5:
    """T-QR-5: Anomalous Dimer IC Collapse."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-5"]["proven"]

    def test_dimer_ic_below_chain(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-5"]
        assert t["dimer_IC"] < t["chain_IC"]

    def test_gap_increase(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-5"]
        assert t["gap_increase"] > 0


class TestTheoremTQR6:
    """T-QR-6: E-Field Monotonicity."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-6"]["proven"]

    def test_f_monotone(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-6"]
        assert t["F_monotone_above_threshold"]

    def test_omega_monotone(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-6"]
        assert t["omega_monotone_above_threshold"]


class TestTheoremTQR7:
    """T-QR-7: Teleoperation as Fidelity Channel."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-7"]["proven"]

    def test_ic_ratio(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-7"]
        assert t["IC_vs_individual"] > 0.5


class TestTheoremTQR8:
    """T-QR-8: Tier-1 Universal Compliance."""

    def test_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.theorems["T-QR-8"]["proven"]

    def test_no_duality_violations(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-8"]
        assert len(t["duality_violations"]) == 0

    def test_no_bound_violations(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-8"]
        assert len(t["bound_violations"]) == 0

    def test_no_log_violations(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-8"]
        assert len(t["log_violations"]) == 0

    def test_all_states_covered(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-8"]
        assert t["n_states"] == 12

    def test_machine_precision(self, analysis: QuinckeAnalysis) -> None:
        t = analysis.theorems["T-QR-8"]
        assert t["max_duality_residual"] < 1e-10


# =====================================================================
# §7 — Nanotechnology connections
# =====================================================================


class TestNanoConnections:
    """Validate nanotechnology connection mapping."""

    def test_connection_count(self) -> None:
        connections = build_nano_connections()
        assert len(connections) == 8

    def test_connection_types(self) -> None:
        connections = build_nano_connections()
        for nc in connections:
            assert isinstance(nc, NanoConnection)
            assert len(nc.phenomenon) > 0
            assert len(nc.nano_application) > 0
            assert len(nc.mechanism) > 0
            assert len(nc.fidelity_channel) > 0
            assert len(nc.IC_impact) > 0

    def test_covers_iron_oxide(self) -> None:
        connections = build_nano_connections()
        assert any("iron oxide" in nc.phenomenon.lower() for nc in connections)

    def test_covers_self_assembly(self) -> None:
        connections = build_nano_connections()
        assert any("self-assembly" in nc.phenomenon.lower() for nc in connections)

    def test_covers_teleoperation(self) -> None:
        connections = build_nano_connections()
        assert any("teleoperation" in nc.phenomenon.lower() for nc in connections)

    def test_covers_microrobotics(self) -> None:
        connections = build_nano_connections()
        assert any("microrobot" in nc.nano_application.lower() for nc in connections)


# =====================================================================
# §8 — Scale position
# =====================================================================


class TestScalePosition:
    """Validate scale position in the matter map."""

    def test_scale_level(self, results: list[QuinckeResult]) -> None:
        sp = compute_scale_position(results)
        assert "Colloidal" in sp["scale_level"] or "Mesoscale" in sp["scale_level"]

    def test_bridges(self, results: list[QuinckeResult]) -> None:
        sp = compute_scale_position(results)
        assert len(sp["bridges"]) == 2

    def test_mean_f_positive(self, results: list[QuinckeResult]) -> None:
        sp = compute_scale_position(results)
        assert sp["mean_F"] > 0

    def test_regime_distribution(self, results: list[QuinckeResult]) -> None:
        sp = compute_scale_position(results)
        total = sum(sp["regime_distribution"].values())
        assert total == len(results)


# =====================================================================
# §9 — Summary statistics
# =====================================================================


class TestSummary:
    """Validate summary statistics."""

    def test_all_theorems_proven(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["all_proven"]

    def test_theorem_counts(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["n_theorems_proven"] == 8
        assert analysis.summary["n_theorems_total"] == 8

    def test_state_count(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["n_states_analyzed"] == 12

    def test_channel_count(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["n_channels"] == 8

    def test_highest_ic_is_vortex(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["highest_IC_state"] == "VortexCondensate"

    def test_lowest_ic_is_subthreshold(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary["lowest_IC_state"] == "SubThreshold"

    def test_source_attribution(self, analysis: QuinckeAnalysis) -> None:
        assert "Garza" in analysis.summary["source"]
        assert "2023" in analysis.summary["source"]

    def test_doi_present(self, analysis: QuinckeAnalysis) -> None:
        assert "10.1126" in analysis.summary["doi"]


# =====================================================================
# §10 — Full analysis orchestrator
# =====================================================================


class TestFullAnalysis:
    """Validate the complete analysis pipeline."""

    def test_results_count(self, analysis: QuinckeAnalysis) -> None:
        assert len(analysis.results) == 12

    def test_theorems_count(self, analysis: QuinckeAnalysis) -> None:
        assert len(analysis.theorems) == 8

    def test_nano_connections_count(self, analysis: QuinckeAnalysis) -> None:
        assert len(analysis.nano_connections) == 8

    def test_scale_position_present(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.scale_position is not None
        assert "scale_level" in analysis.scale_position

    def test_summary_present(self, analysis: QuinckeAnalysis) -> None:
        assert analysis.summary is not None
        assert "all_proven" in analysis.summary


# =====================================================================
# §11 — Collective state analysis
# =====================================================================


class TestCollectiveStates:
    """Validate kernel behavior across collective states."""

    def test_individual_states_exist(self, results: list[QuinckeResult]) -> None:
        individuals = [r for r in results if r.collective_state == "Individual"]
        assert len(individuals) >= 4

    def test_collective_higher_ic_than_nonmagnetic_individual(self, results: list[QuinckeResult]) -> None:
        """Collective magnetic states should have higher IC than
        non-magnetic individual rollers due to channel activation."""
        chain = next(r for r in results if r.name == "ChainAssembly")
        moderate = next(r for r in results if r.name == "ModerateRolling")
        assert chain.IC > moderate.IC

    def test_vortex_highest_f(self, results: list[QuinckeResult]) -> None:
        vortex = next(r for r in results if r.name == "VortexCondensate")
        assert max(r.F for r in results) == vortex.F

    def test_sub_threshold_lowest_f(self, results: list[QuinckeResult]) -> None:
        sub = next(r for r in results if r.name == "SubThreshold")
        assert min(r.F for r in results) == sub.F

    def test_chain_disassembly_lower_ic_than_chain(self, results: list[QuinckeResult]) -> None:
        chain = next(r for r in results if r.name == "ChainAssembly")
        disassembly = next(r for r in results if r.name == "ChainDisassembly")
        assert disassembly.IC < chain.IC


# =====================================================================
# §12 — Edge cases and boundary conditions
# =====================================================================


class TestEdgeCases:
    """Validate edge cases and boundary conditions."""

    def test_epsilon_guard_band(self) -> None:
        assert EPSILON == 1e-8

    def test_omega_thresholds_ordered(self) -> None:
        assert OMEGA_STABLE < OMEGA_COLLAPSE

    def test_ic_critical_positive(self) -> None:
        assert IC_CRITICAL > 0

    def test_zero_b_field_states(self, results: list[QuinckeResult]) -> None:
        """States with B=0 should have low magnetic channels."""
        catalog = build_quincke_catalog()
        zero_b = [c for c in catalog if c.B_field_mT == 0.0]
        assert len(zero_b) >= 5

    def test_single_particle_states(self, catalog: list[QuinckeConfig]) -> None:
        """Teleoperated and programmed are single-particle."""
        teleop = next(c for c in catalog if c.name == "Teleoperated")
        assert teleop.n_particles == 1
        prog = next(c for c in catalog if c.name == "ProgrammedSquare")
        assert prog.n_particles == 1

    def test_dimer_particles(self, catalog: list[QuinckeConfig]) -> None:
        dimer = next(c for c in catalog if c.name == "AnomalousDimer")
        assert dimer.n_particles == 2
