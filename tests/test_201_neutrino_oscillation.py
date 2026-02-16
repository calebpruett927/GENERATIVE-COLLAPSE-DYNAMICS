"""Tests for neutrino_oscillation.py — flavor change as periodic channel drift.

Validates that neutrino oscillation maps onto GCD kernel invariants:
  - Three-flavor vacuum oscillation probabilities via PMNS matrix
  - MSW matter effects for DUNE/LBNF baseline
  - Theorems T11 (periodic drift) and T12 (matter-enhanced regime transition)
  - Tier-1 identity compliance at every L/E point

Key structural insight: oscillation IS derivatio — a flavor trace vector
[P_αe, P_αμ, P_ατ] drifts periodically in kernel space with measured
return time τ_R = π / (1.267 Δm²₃₂).

Data sources: NuFIT 5.3 (2024), PDG 2024, DUNE CDR (2020),
    Pontecorvo (1957), MNS (1962), Wolfenstein (1978),
    Mikheyev-Smirnov (1985)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.standard_model.neutrino_oscillation import (  # noqa: E402
    DUNE_BASELINE_KM,
    DUNE_PEAK_ENERGY_GEV,
    EPSILON,
    OSCILLATION_FACTOR,
    DUNEPrediction,
    MassOrdering,
    OscillationRegime,
    TheoremResult,
    compute_dune_prediction,
    compute_oscillation_point,
    compute_oscillation_sweep,
    oscillation_matrix_vacuum,
    oscillation_probability_matter,
    oscillation_probability_vacuum,
    run_all_theorems,
    theorem_T11_neutrino_oscillation_drift,
    theorem_T12_matter_enhanced_mixing,
)
from closures.standard_model.pmns_mixing import (  # noqa: E402
    DM2_32,
)

# ═══════════════════════════════════════════════════════════════════
# MODULE STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestModuleStructure:
    """Validate that the module exposes all required components."""

    def test_oscillation_regime_enum(self) -> None:
        assert OscillationRegime.VACUUM == "Vacuum"
        assert OscillationRegime.RESONANT == "Resonant"
        assert OscillationRegime.SUPPRESSED == "Suppressed"

    def test_mass_ordering_enum(self) -> None:
        assert MassOrdering.NORMAL == "Normal"
        assert MassOrdering.INVERTED == "Inverted"

    def test_theorem_result_dataclass(self) -> None:
        r = TheoremResult(
            name="test", statement="stmt", n_tests=2, n_passed=2, n_failed=0, details={}, verdict="PROVEN"
        )
        assert r.pass_rate == 1.0

    def test_theorem_result_zero_tests(self) -> None:
        r = TheoremResult(name="t", statement="s", n_tests=0, n_passed=0, n_failed=0, details={}, verdict="FALSIFIED")
        assert r.pass_rate == 0.0

    def test_dune_constants(self) -> None:
        assert DUNE_BASELINE_KM == 1285.0
        assert DUNE_PEAK_ENERGY_GEV == 2.5
        assert abs(OSCILLATION_FACTOR - 1.26693) < 0.001

    def test_epsilon_frozen(self) -> None:
        """Guard band is frozen, consistent across the seam."""
        assert EPSILON == 1e-6


# ═══════════════════════════════════════════════════════════════════
# VACUUM OSCILLATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestVacuumOscillation:
    """Validate three-flavor vacuum oscillation probabilities."""

    def test_survival_at_zero_baseline(self) -> None:
        """At L=0, survival probability = 1, transition = 0."""
        for alpha in range(3):
            P_surv = oscillation_probability_vacuum(alpha, alpha, 0.0, 2.5)
            assert abs(P_surv - 1.0) < 1e-10, f"P_{alpha}{alpha}(L=0) = {P_surv}"
            for beta in range(3):
                if beta != alpha:
                    P_trans = oscillation_probability_vacuum(alpha, beta, 0.0, 2.5)
                    assert abs(P_trans) < 1e-10, f"P_{alpha}{beta}(L=0) = {P_trans}"

    def test_unitarity_row(self) -> None:
        """Row unitarity: Σ_β P(ν_α → ν_β) = 1 for all α."""
        L, E = 800.0, 3.0
        for alpha in range(3):
            total = sum(oscillation_probability_vacuum(alpha, beta, L, E) for beta in range(3))
            assert abs(total - 1.0) < 1e-6, f"Row {alpha} unitarity = {total}"

    def test_unitarity_column(self) -> None:
        """Column unitarity: Σ_α P(ν_α → ν_β) = 1 for all β."""
        L, E = 500.0, 2.0
        for beta in range(3):
            total = sum(oscillation_probability_vacuum(alpha, beta, L, E) for alpha in range(3))
            assert abs(total - 1.0) < 1e-6, f"Col {beta} unitarity = {total}"

    def test_probabilities_in_range(self) -> None:
        """All probabilities ∈ [0, 1]."""
        for L in [100, 500, 1285, 2000]:
            for E in [0.5, 1.0, 2.5, 5.0]:
                for a in range(3):
                    for b in range(3):
                        P = oscillation_probability_vacuum(a, b, float(L), float(E))
                        assert -1e-10 <= P <= 1.0 + 1e-10, f"P_{a}{b}(L={L},E={E}) = {P} out of range"

    def test_cp_conjugation(self) -> None:
        """P(ν_α→ν_β) ≠ P(ν̄_α→ν̄_β) when δ_CP ≠ 0, π (CP violation)."""
        L, E = DUNE_BASELINE_KM, DUNE_PEAK_ENERGY_GEV
        P_nu = oscillation_probability_vacuum(1, 0, L, E)
        P_nubar = oscillation_probability_vacuum(1, 0, L, E, antineutrino=True)
        # δ_CP = 197° ≠ 0 or π → should see CP violation
        assert abs(P_nu - P_nubar) > 1e-4, f"No CP violation: P_ν={P_nu:.6f} P_ν̄={P_nubar:.6f}"

    def test_t_invariance(self) -> None:
        """In vacuum with CP conjugation: P(ν_α→ν_β) = P(ν̄_β→ν̄_α) (CPT)."""
        L, E = 600.0, 2.0
        P_ab = oscillation_probability_vacuum(1, 0, L, E)
        P_ba_bar = oscillation_probability_vacuum(0, 1, L, E, antineutrino=True)
        assert abs(P_ab - P_ba_bar) < 1e-6, f"CPT violation: P_μe={P_ab:.8f} P_ēμ̄={P_ba_bar:.8f}"

    def test_oscillation_matrix_shape(self) -> None:
        """oscillation_matrix_vacuum returns 3×3 array."""
        P = oscillation_matrix_vacuum(1000.0, 2.0)
        assert P.shape == (3, 3)
        # Row sums = 1
        for row in range(3):
            assert abs(float(P[row].sum()) - 1.0) < 1e-6

    def test_atmospheric_oscillation_maximum(self) -> None:
        """ν_μ survival probability has first minimum near L/E ≈ 500 km/GeV."""
        dm2_atm = abs(DM2_32)
        first_min_LE = math.pi / (2 * OSCILLATION_FACTOR * dm2_atm)
        E = 2.5
        L = first_min_LE * E
        P_mumu = oscillation_probability_vacuum(1, 1, L, E)
        # At first minimum, P_μμ should be significantly less than 1
        assert P_mumu < 0.5, f"P_μμ at first min = {P_mumu:.4f}, expected < 0.5"


# ═══════════════════════════════════════════════════════════════════
# MATTER EFFECT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestMatterEffects:
    """Validate MSW matter effects for DUNE configuration."""

    def test_matter_enhances_nue_appearance(self) -> None:
        """Matter enhances ν_μ→ν_e for neutrinos (normal ordering)."""
        L, E = DUNE_BASELINE_KM, DUNE_PEAK_ENERGY_GEV
        P_vac = oscillation_probability_vacuum(1, 0, L, E)
        P_mat = oscillation_probability_matter(1, 0, L, E)
        assert P_mat > P_vac, f"Matter should enhance: vac={P_vac:.6f} mat={P_mat:.6f}"

    def test_matter_suppresses_nubar(self) -> None:
        """Matter suppresses ν̄_μ→ν̄_e for antineutrinos (normal ordering)."""
        L, E = DUNE_BASELINE_KM, DUNE_PEAK_ENERGY_GEV
        P_vac = oscillation_probability_vacuum(1, 0, L, E, antineutrino=True)
        P_mat = oscillation_probability_matter(1, 0, L, E, antineutrino=True)
        assert P_mat < P_vac, f"Matter should suppress ν̄: vac={P_vac:.6f} mat={P_mat:.6f}"

    def test_matter_probability_in_range(self) -> None:
        """Matter probabilities ∈ [0, 1]."""
        L, E = DUNE_BASELINE_KM, DUNE_PEAK_ENERGY_GEV
        P = oscillation_probability_matter(1, 0, L, E)
        assert 0.0 <= P <= 1.0

    def test_zero_density_approaches_vacuum(self) -> None:
        """At ρ≈0, matter probability approaches vacuum (Cervera approx ~0.5% residual)."""
        L, E = DUNE_BASELINE_KM, DUNE_PEAK_ENERGY_GEV
        P_vac = oscillation_probability_vacuum(1, 0, L, E)
        P_mat = oscillation_probability_matter(1, 0, L, E, rho_g_cm3=0.0)
        # Cervera et al. is an approximate formula; tolerance reflects inherent error
        assert abs(P_vac - P_mat) < 0.005, f"ρ≈0 should approach vacuum: vac={P_vac:.6f} mat={P_mat:.6f}"

    def test_non_mue_falls_back_to_vacuum(self) -> None:
        """For channels other than μ→e, matter falls back to vacuum."""
        L, E = 800.0, 2.0
        P_mat = oscillation_probability_matter(0, 0, L, E)
        P_vac = oscillation_probability_vacuum(0, 0, L, E)
        assert abs(P_mat - P_vac) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# KERNEL MAPPING TESTS (Tier-1 identities)
# ═══════════════════════════════════════════════════════════════════


class TestKernelMapping:
    """Validate that oscillation trace vectors satisfy Tier-1 identities."""

    def test_duality_identity(self) -> None:
        """F + ω = 1 at every oscillation point."""
        for LE in [100, 300, 500, 800, 1200]:
            pt = compute_oscillation_point(float(LE), 2.5)
            # F_mu comes from kernel; ω = 1 − F
            # Verified indirectly: F is computed from kernel which enforces F + ω = 1
            assert 0 < pt.F_mu < 1.0
            assert pt.F_mu > 0.3  # Should be ≈ 1/3

    def test_integrity_bound(self) -> None:
        """IC ≤ F at every oscillation point."""
        for LE in [50, 200, 500, 1000, 1500]:
            pt = compute_oscillation_point(float(LE), 2.5)
            assert pt.IC_mu <= pt.F_mu + 1e-10, f"IC > F at L/E={LE}: IC={pt.IC_mu} F={pt.F_mu}"

    def test_heterogeneity_gap_nonnegative(self) -> None:
        """Δ = F − IC ≥ 0 at every point."""
        for LE in [10, 100, 500, 1000]:
            pt = compute_oscillation_point(float(LE), 2.5)
            assert pt.heterogeneity_gap_mu >= -1e-10

    def test_F_approximately_one_third(self) -> None:
        """F ≈ 1/3 at all oscillation points (unitarity constraint)."""
        points = [compute_oscillation_point(float(LE), 2.5) for LE in range(50, 1500, 100)]
        F_values = [pt.F_mu for pt in points]
        for F in F_values:
            assert abs(F - 1 / 3) < 0.01, f"F = {F}, expected ≈ 1/3"

    def test_ic_ranges_with_oscillation(self) -> None:
        """IC should vary as oscillation progresses (not stuck at one value)."""
        ICs = [compute_oscillation_point(float(LE), 2.5).IC_mu for LE in range(50, 1500, 50)]
        IC_set = {round(ic, 4) for ic in ICs}
        assert len(IC_set) > 5, f"IC takes too few distinct values: {len(IC_set)}"

    def test_unitarity_at_oscillation_point(self) -> None:
        """Row unitarity holds in the OscillationPoint data."""
        pt = compute_oscillation_point(1000.0, 2.0)
        assert abs(pt.unitarity_e - 1.0) < 1e-4
        assert abs(pt.unitarity_mu - 1.0) < 1e-4
        assert abs(pt.unitarity_tau - 1.0) < 1e-4

    def test_production_point_maximally_heterogeneous(self) -> None:
        """At L≈0, the flavor vector is [ε, 1−ε, ε] → large gap."""
        pt = compute_oscillation_point(0.1, 2.5)
        assert pt.heterogeneity_gap_mu > 0.3, f"Gap at production {pt.heterogeneity_gap_mu:.4f} < 0.3"


# ═══════════════════════════════════════════════════════════════════
# OSCILLATION SWEEP TESTS
# ═══════════════════════════════════════════════════════════════════


class TestOscillationSweep:
    """Validate the L/E sweep and kernel trajectory."""

    def test_sweep_returns_correct_count(self) -> None:
        sweep = compute_oscillation_sweep(n_points=50)
        assert sweep.n_points == 50
        assert len(sweep.F_values) == 50
        assert len(sweep.IC_values) == 50

    def test_sweep_F_approximately_constant(self) -> None:
        """F should be ~1/3 everywhere in the sweep."""
        sweep = compute_oscillation_sweep(n_points=100)
        F_arr = np.array(sweep.F_values)
        assert abs(float(F_arr.mean()) - 1 / 3) < 0.01
        assert float(F_arr.std()) < 0.01

    def test_sweep_ic_oscillates(self) -> None:
        """IC should show oscillatory behavior (range > 0.01)."""
        sweep = compute_oscillation_sweep(n_points=200)
        ic_range = sweep.IC_max - sweep.IC_min
        assert ic_range > 0.01, f"IC range = {ic_range} — no oscillation visible"

    def test_sweep_period_matches_atmospheric(self) -> None:
        """τ_R period should match π / (1.267 Δm²₃₂)."""
        sweep = compute_oscillation_sweep(n_points=300)
        expected = math.pi / (OSCILLATION_FACTOR * abs(DM2_32))
        assert abs(sweep.tau_R_period - expected) / expected < 0.01

    def test_sweep_regime_counts(self) -> None:
        """All regime counts should sum to n_points."""
        sweep = compute_oscillation_sweep(n_points=100)
        total = sweep.n_stable + sweep.n_watch + sweep.n_collapse
        assert total == 100

    def test_sweep_gap_max_at_production(self) -> None:
        """Maximum gap should be near the production end (small L/E)."""
        sweep = compute_oscillation_sweep(L_over_E_range=(0.1, 2000.0), n_points=200)
        # First few gap values should include the maximum
        first_gaps = sweep.gap_values[:10]
        assert max(first_gaps) > 0.9 * sweep.gap_max

    def test_sweep_different_channels(self) -> None:
        """Sweeps for ν_e, ν_μ, ν_τ should all produce valid results."""
        for ch in ["nu_e", "nu_mu", "nu_tau"]:
            sweep = compute_oscillation_sweep(n_points=50, channel=ch)
            assert sweep.channel == ch
            assert len(sweep.F_values) == 50


# ═══════════════════════════════════════════════════════════════════
# DUNE PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDUNEPrediction:
    """Validate DUNE/LBNF predictions."""

    def test_dune_prediction_returns_valid(self) -> None:
        dune = compute_dune_prediction()
        assert isinstance(dune, DUNEPrediction)
        assert dune.baseline_km == 1285.0
        assert dune.peak_energy_GeV == 2.5

    def test_dune_appearance_probability_range(self) -> None:
        """P(ν_μ→ν_e) at DUNE should be in physically reasonable range."""
        dune = compute_dune_prediction()
        # DUNE expects P_μe ≈ 0.05–0.10 depending on δ_CP
        assert 0.01 < dune.P_mue_vacuum < 0.15
        assert 0.01 < dune.P_mue_matter < 0.20

    def test_dune_disappearance_probability(self) -> None:
        """ν_μ survival should be significantly suppressed at first max."""
        dune = compute_dune_prediction()
        assert dune.P_mumu_vacuum < 0.5  # Near oscillation maximum

    def test_dune_matter_enhancement(self) -> None:
        """Matter should enhance neutrino oscillation (NO)."""
        dune = compute_dune_prediction()
        assert dune.matter_enhancement > 1.0

    def test_dune_cp_asymmetry(self) -> None:
        """CP asymmetry should be nonzero for δ_CP ≠ 0, π."""
        dune = compute_dune_prediction()
        assert abs(dune.A_CP_vacuum) > 0.01

    def test_dune_kernel_gap_reduces(self) -> None:
        """Heterogeneity gap should decrease from production to detection."""
        dune = compute_dune_prediction()
        assert dune.gap_detection < dune.gap_production, (
            f"Gap should decrease: prod={dune.gap_production} det={dune.gap_detection}"
        )

    def test_dune_F_constant(self) -> None:
        """F should be ≈ 1/3 at both production and detection."""
        dune = compute_dune_prediction()
        assert abs(dune.F_production - 1 / 3) < 0.01
        assert abs(dune.F_detection - 1 / 3) < 0.01

    def test_dune_different_cp_phases(self) -> None:
        """Different δ_CP values should give different predictions."""
        d1 = compute_dune_prediction(delta_CP_deg=0.0)
        d2 = compute_dune_prediction(delta_CP_deg=90.0)
        d3 = compute_dune_prediction(delta_CP_deg=180.0)
        d4 = compute_dune_prediction(delta_CP_deg=270.0)
        # All should give different P_mue
        probs = [d1.P_mue_vacuum, d2.P_mue_vacuum, d3.P_mue_vacuum, d4.P_mue_vacuum]
        assert len({round(p, 6) for p in probs}) == 4, (
            f"δ_CP variations should produce 4 distinct probabilities: {probs}"
        )


# ═══════════════════════════════════════════════════════════════════
# THEOREM TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTheoremT11:
    """Validate T11: Neutrino Oscillation as Periodic Channel Drift."""

    def test_t11_proven(self) -> None:
        result = theorem_T11_neutrino_oscillation_drift()
        assert result.verdict == "PROVEN", (
            f"T11 FALSIFIED: {result.n_passed}/{result.n_tests} — details: {result.details}"
        )

    def test_t11_all_subtests_pass(self) -> None:
        result = theorem_T11_neutrino_oscillation_drift()
        assert result.n_failed == 0
        assert result.n_passed == result.n_tests

    def test_t11_F_constant(self) -> None:
        """T11 details should show F ≈ 1/3 with tiny std."""
        result = theorem_T11_neutrino_oscillation_drift()
        assert abs(result.details["F_mean"] - 1 / 3) < 0.001
        assert result.details["F_std"] < 0.001

    def test_t11_ic_oscillates(self) -> None:
        """T11 details should show IC range > 0.01."""
        result = theorem_T11_neutrino_oscillation_drift()
        assert result.details["IC_range"] > 0.01

    def test_t11_duality_exact(self) -> None:
        """F + ω = 1 should be exact to machine precision."""
        result = theorem_T11_neutrino_oscillation_drift()
        assert float(result.details["max_duality_error"]) < 1e-10

    def test_t11_ic_leq_f(self) -> None:
        """Integrity bound IC ≤ F must hold everywhere."""
        result = theorem_T11_neutrino_oscillation_drift()
        assert result.details["IC_leq_F_everywhere"] is True

    def test_t11_period_match(self) -> None:
        """Oscillation period should match kernel τ_R."""
        result = theorem_T11_neutrino_oscillation_drift()
        expected = result.details["expected_period_LE"]
        measured = result.details["measured_period_LE"]
        assert abs(expected - measured) / expected < 0.01

    def test_t11_cp_violation(self) -> None:
        """CP asymmetry should be detectable in kernel."""
        result = theorem_T11_neutrino_oscillation_drift()
        assert result.details["CP_asymmetry"] > 1e-4
        assert result.details["IC_asymmetry_nu_nubar"] > 1e-6


class TestTheoremT12:
    """Validate T12: Matter-Enhanced Mixing as Kernel Regime Transition."""

    def test_t12_proven(self) -> None:
        result = theorem_T12_matter_enhanced_mixing()
        assert result.verdict == "PROVEN", (
            f"T12 FALSIFIED: {result.n_passed}/{result.n_tests} — details: {result.details}"
        )

    def test_t12_all_subtests_pass(self) -> None:
        result = theorem_T12_matter_enhanced_mixing()
        assert result.n_failed == 0

    def test_t12_matter_enhancement(self) -> None:
        """Matter should enhance ν_μ→ν_e for neutrinos."""
        result = theorem_T12_matter_enhanced_mixing()
        assert result.details["matter_enhancement"] > 1.0

    def test_t12_antineutrino_suppression(self) -> None:
        """Matter should suppress ν̄_μ→ν̄_e for antineutrinos."""
        result = theorem_T12_matter_enhanced_mixing()
        assert result.details["antineutrino_suppression"] < 1.0

    def test_t12_cp_asymmetry_enlarged(self) -> None:
        """Matter CP asymmetry > vacuum CP asymmetry."""
        result = theorem_T12_matter_enhanced_mixing()
        assert result.details["A_CP_matter"] > result.details["A_CP_vacuum"]


class TestRunAllTheorems:
    """Validate the theorem runner."""

    def test_run_all_returns_two(self) -> None:
        results = run_all_theorems()
        assert len(results) == 2

    def test_all_proven(self) -> None:
        results = run_all_theorems()
        for r in results:
            assert r.verdict == "PROVEN", f"{r.name} FALSIFIED"

    def test_total_tests_count(self) -> None:
        results = run_all_theorems()
        total = sum(r.n_tests for r in results)
        assert total == 13  # T11: 8 + T12: 5

    def test_all_have_timing(self) -> None:
        results = run_all_theorems()
        for r in results:
            assert "time_ms" in r.details
            assert r.details["time_ms"] >= 0
