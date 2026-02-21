"""Cross-Domain Interconnective Tests — Bridges That Deepen Understanding.

This test suite validates structural connections BETWEEN domains that
were previously tested only in isolation. Each test class proves that
data flowing from Domain A through the kernel produces results that are
structurally consistent with Domain B's analysis — and that Tier-1
identities hold across every boundary.

The tests are organized by the structural connection being verified:

    §1  Three-Scale Bridge: Standard Model → Nuclear → Atomic
    §2  Element Database Consistency: Materials ↔ Atomic Physics
    §3  SM Formalism Theorems (T1–T10): Previously untested in pytest
    §4  Cross-Scale Theorems (T17–T23): Unified Minimal Structure
    §5  Tier-1 Proof Harness: Exercising the 10,162-test proof
    §6  Security Domain: Kernel identity preservation
    §7  Seam Composition Across Domains: Real domain data through seam chains
    §8  Nuclear ↔ Atomic: Magic Number Consistency
    §9  RCFT ↔ QM: Fisher Distance Bridge
    §10 Rosetta Invariance: Same data, multiple lenses, same Tier-1

*Connexio inter campos est probatio quae auget intelligentiam.*
(The connection between domains is the test that grows understanding.)

Cross-references:
    closures/standard_model/particle_physics_formalism.py (T1–T10)
    closures/standard_model/subatomic_kernel.py (31 particles)
    closures/atomic_physics/cross_scale_kernel.py (12-channel bridge)
    closures/atomic_physics/tier1_proof.py (10,162 identity tests)
    closures/materials_science/element_database.py (118 elements)
    closures/unified_minimal_structure.py (T17–T23)
    closures/security/trust_fidelity.py (trust kernel)
    closures/security/trust_integrity.py (trust IC)
    src/umcp/kernel_optimized.py (Tier-1 kernel)
    src/umcp/seam_optimized.py (seam chain)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.atomic_physics.cross_scale_kernel import (  # noqa: E402
    MAGIC_NUMBERS as ATOMIC_MAGIC_NUMBERS,
)
from closures.atomic_physics.cross_scale_kernel import (  # noqa: E402
    binding_energy_per_nucleon,
    compute_all_enhanced,
    compute_enhanced_kernel,
    magic_proximity,
    normalize_element_enhanced,
)
from closures.atomic_physics.periodic_kernel import (  # noqa: E402
    batch_compute_all as compute_periodic_all,
)

# Domain imports — each represents a domain boundary being crossed
from closures.materials_science.element_database import ELEMENTS  # noqa: E402
from closures.security.trust_fidelity import compute_trust_fidelity  # noqa: E402
from closures.security.trust_integrity import compute_trust_integrity  # noqa: E402
from closures.standard_model.particle_physics_formalism import (  # noqa: E402
    theorem_T1_spin_statistics,
    theorem_T2_generation_monotonicity,
    theorem_T3_confinement_IC_collapse,
    theorem_T4_mass_kernel_log_mapping,
    theorem_T5_charge_quantization,
    theorem_T6_cross_scale_universality,
    theorem_T7_symmetry_breaking,
    theorem_T8_ckm_unitarity,
    theorem_T9_running_coupling_flow,
    theorem_T10_nuclear_binding_curve,
)
from closures.standard_model.subatomic_kernel import (  # noqa: E402
    compute_all_composite,
    compute_all_fundamental,
)
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402
from umcp.seam_optimized import SeamChainAccumulator  # noqa: E402

# Frozen constants
EPSILON = 1e-8
TOL_TIER1 = 1e-6  # Tolerance for Tier-1 identity checks


# ═══════════════════════════════════════════════════════════════════
# §1  THREE-SCALE BRIDGE: Standard Model → Nuclear → Atomic
# ═══════════════════════════════════════════════════════════════════


class TestThreeScaleBridge:
    """Verify that kernel signatures are structurally consistent across
    three physical scales: fundamental particles → composite hadrons → atoms.

    This is the deepest cross-domain test: data originates in the Standard
    Model closure, passes through the nuclear bridge (binding energy,
    shell structure), and lands in the atomic closure. Tier-1 must hold
    at every handoff.
    """

    @pytest.fixture(scope="class")
    def fundamental(self) -> list:
        return compute_all_fundamental()

    @pytest.fixture(scope="class")
    def composite(self) -> list:
        return compute_all_composite()

    @pytest.fixture(scope="class")
    def atomic(self) -> list:
        return compute_all_enhanced()

    def test_fundamental_count(self, fundamental: list) -> None:
        """17 fundamental particles from the Standard Model."""
        assert len(fundamental) == 17

    def test_composite_count(self, composite: list) -> None:
        """14 composite hadrons."""
        assert len(composite) == 14

    def test_atomic_count(self, atomic: list) -> None:
        """118 elements from the periodic table."""
        assert len(atomic) == 118

    def test_tier1_holds_at_fundamental_scale(self, fundamental: list) -> None:
        """F + ω = 1, IC ≤ F, IC = exp(κ) for all 17 fundamental particles."""
        for p in fundamental:
            assert abs(p.F + p.omega - 1.0) < TOL_TIER1, f"Duality fails: {p.symbol}"
            assert p.IC <= p.F + TOL_TIER1, f"Integrity bound fails: {p.symbol}"
            assert abs(p.IC - math.exp(p.kappa)) < TOL_TIER1, f"Exp bridge fails: {p.symbol}"

    def test_tier1_holds_at_composite_scale(self, composite: list) -> None:
        """F + ω = 1, IC ≤ F, IC = exp(κ) for all 14 composite hadrons."""
        for p in composite:
            assert abs(p.F + p.omega - 1.0) < TOL_TIER1, f"Duality fails: {p.name}"
            assert p.IC <= p.F + TOL_TIER1, f"Integrity bound fails: {p.name}"
            assert abs(p.IC - math.exp(p.kappa)) < TOL_TIER1, f"Exp bridge fails: {p.name}"

    def test_tier1_holds_at_atomic_scale(self, atomic: list) -> None:
        """F + ω = 1, IC ≤ F, IC = exp(κ) for all 118 elements."""
        for el in atomic:
            assert abs(el.F + el.omega - 1.0) < TOL_TIER1, f"Duality fails: {el.symbol}"
            assert el.IC <= el.F + TOL_TIER1, f"Integrity bound fails: {el.symbol}"
            assert abs(el.IC - math.exp(el.kappa)) < 1e-4, f"Exp bridge fails: {el.symbol}"

    def test_cross_scale_F_ordering(self, fundamental: list, composite: list, atomic: list) -> None:
        """T6 (Cross-Scale Universality): composite ⟨F⟩ < atomic ⟨F⟩ < fundamental ⟨F⟩.

        Confinement compresses fidelity: quarks lose structure when bound
        into hadrons. Atoms recover some structure through shell organization.
        """
        F_fund = sum(p.F for p in fundamental) / len(fundamental)
        F_comp = sum(p.F for p in composite) / len(composite)
        F_atom = sum(el.F for el in atomic) / len(atomic)

        assert F_comp < F_atom < F_fund, (
            f"Cross-scale ordering violated: composite={F_comp:.4f}, atomic={F_atom:.4f}, fundamental={F_fund:.4f}"
        )

    def test_confinement_cliff_IC(self, fundamental: list, composite: list) -> None:
        """T3 verification: IC drops dramatically from quarks to hadrons.

        The quarks that make up a hadron have much higher IC individually
        than the hadron they compose. This is the confinement cliff.
        """
        quarks = [p for p in fundamental if p.category == "Quark"]
        baryons = [p for p in composite if p.category == "Baryon"]

        mean_IC_quarks = sum(p.IC for p in quarks) / len(quarks)
        mean_IC_baryons = sum(p.IC for p in baryons) / len(baryons)

        # IC must drop by at least an order of magnitude
        assert mean_IC_baryons < mean_IC_quarks * 0.1, (
            f"Confinement cliff too shallow: quarks IC={mean_IC_quarks:.6f}, baryons IC={mean_IC_baryons:.6f}"
        )

    def test_heterogeneity_gap_increases_with_scale(self, fundamental: list, composite: list, atomic: list) -> None:
        """The heterogeneity gap Δ = F − IC should be non-trivial at all scales."""
        gaps = {
            "fundamental": sum(p.heterogeneity_gap for p in fundamental) / len(fundamental),
            "composite": sum(p.heterogeneity_gap for p in composite) / len(composite),
            "atomic": sum(el.heterogeneity_gap for el in atomic) / len(atomic),
        }
        for scale, gap in gaps.items():
            assert gap > 0.001, f"Trivial gap at {scale}: Δ={gap:.6f}"


# ═══════════════════════════════════════════════════════════════════
# §2  ELEMENT DATABASE CONSISTENCY: Materials ↔ Atomic Physics
# ═══════════════════════════════════════════════════════════════════


class TestElementDatabaseBridge:
    """Verify that the shared element database produces consistent results
    when consumed by both the atomic physics and materials science domains.

    The element_database.py (materials_science) is the canonical source.
    periodic_kernel.py and cross_scale_kernel.py both import from it.
    These tests verify the data pipeline is consistent.
    """

    def test_element_count_matches(self) -> None:
        """Both consumers see 118 elements."""
        assert len(ELEMENTS) == 118

    def test_all_elements_have_required_fields(self) -> None:
        """Every element has Z, symbol, name, period, block, ionization_energy."""
        for el in ELEMENTS:
            assert el.Z > 0, f"Missing Z for {el.symbol}"
            assert len(el.symbol) <= 3, f"Symbol too long: {el.symbol}"
            assert el.period in range(1, 8), f"Invalid period for {el.symbol}: {el.period}"
            assert el.block in ("s", "p", "d", "f"), f"Invalid block for {el.symbol}: {el.block}"
            assert el.ionization_energy_eV > 0, f"Missing IE for {el.symbol}"

    def test_periodic_kernel_and_cross_scale_agree_on_count(self) -> None:
        """periodic_kernel and cross_scale_kernel both process all 118 elements."""
        periodic_results = compute_periodic_all()
        enhanced_results = compute_all_enhanced()
        assert len(periodic_results) == 118
        assert len(enhanced_results) == 118

    def test_same_element_identity_across_kernels(self) -> None:
        """Element Z, symbol, name must match between both representations."""
        periodic_results = compute_periodic_all()
        enhanced_results = compute_all_enhanced()
        for p, e in zip(periodic_results, enhanced_results, strict=False):
            assert p.Z == e.Z, f"Z mismatch: periodic={p.Z}, enhanced={e.Z}"
            assert p.symbol == e.symbol, f"Symbol mismatch at Z={p.Z}"

    def test_enhanced_kernel_includes_nuclear_channels(self) -> None:
        """The 12-channel cross-scale kernel must include nuclear channels
        that the 8-channel periodic kernel doesn't have."""
        result = compute_all_enhanced()[25]  # Iron (Z=26)
        assert "BE_per_A" in result.channel_labels, "Missing nuclear binding channel"
        assert "magic_prox" in result.channel_labels, "Missing magic proximity channel"
        assert "N_over_Z" in result.channel_labels, "Missing neutron excess channel"

    def test_tier1_preserved_across_kernel_variants(self) -> None:
        """Tier-1 holds regardless of which kernel variant processes the data.

        Both 8-channel periodic and 12-channel enhanced must satisfy
        F + ω = 1, IC ≤ F at every element.
        """
        periodic = compute_periodic_all()
        enhanced = compute_all_enhanced()

        for p in periodic:
            assert abs(p.F + p.omega - 1.0) < TOL_TIER1, f"Periodic Tier-1 fail at {p.symbol}: F+ω={p.F + p.omega}"
            assert p.IC <= p.F + TOL_TIER1, f"Periodic integrity bound fail at {p.symbol}: IC={p.IC} > F={p.F}"

        for e in enhanced:
            assert abs(e.F + e.omega - 1.0) < TOL_TIER1, f"Enhanced Tier-1 fail at {e.symbol}: F+ω={e.F + e.omega}"
            assert e.IC <= e.F + TOL_TIER1, f"Enhanced integrity bound fail at {e.symbol}: IC={e.IC} > F={e.F}"


# ═══════════════════════════════════════════════════════════════════
# §3  SM FORMALISM THEOREMS (T1–T10): Exercised by pytest
# ═══════════════════════════════════════════════════════════════════


class TestSMFormalism:
    """Exercise all 10 Standard Model formalism theorems via pytest.

    These theorems live in particle_physics_formalism.py but previously
    had NO pytest coverage. Each theorem returns a TheoremResult with
    .verdict (str "PROVEN"/"FALSIFIED") and .n_tests (int). This class
    makes them first-class pytest citizens.
    """

    def test_T1_spin_statistics(self) -> None:
        """T1: ⟨F⟩_fermion > ⟨F⟩_boson (fermions are more faithful)."""
        result = theorem_T1_spin_statistics()
        assert result.verdict == "PROVEN", f"T1 failed: {result.statement}"
        assert result.n_tests >= 10, f"T1 too few tests: {result.n_tests}"

    def test_T2_generation_monotonicity(self) -> None:
        """T2: ⟨F⟩_Gen1 < ⟨F⟩_Gen2 < ⟨F⟩_Gen3."""
        result = theorem_T2_generation_monotonicity()
        assert result.verdict == "PROVEN", f"T2 failed: {result.statement}"
        assert result.n_tests >= 5, f"T2 too few tests: {result.n_tests}"

    def test_T3_confinement_IC_collapse(self) -> None:
        """T3: IC drops >98% from quarks to hadrons."""
        result = theorem_T3_confinement_IC_collapse()
        assert result.verdict == "PROVEN", f"T3 failed: {result.statement}"
        assert result.n_tests >= 15, f"T3 too few tests: {result.n_tests}"

    def test_T4_mass_kernel_log_mapping(self) -> None:
        """T4: 13 OOM mass hierarchy → bounded F ∈ [0.37, 0.73]."""
        result = theorem_T4_mass_kernel_log_mapping()
        assert result.verdict == "PROVEN", f"T4 failed: {result.statement}"
        assert result.n_tests >= 5, f"T4 too few tests: {result.n_tests}"

    def test_T5_charge_quantization(self) -> None:
        """T5: IC_neutral / IC_charged ≪ 1 (charge quantization signature)."""
        result = theorem_T5_charge_quantization()
        assert result.verdict == "PROVEN", f"T5 failed: {result.statement}"
        assert result.n_tests >= 5, f"T5 too few tests: {result.n_tests}"

    def test_T6_cross_scale_universality(self) -> None:
        """T6: composite < atomic < fundamental F ordering holds."""
        result = theorem_T6_cross_scale_universality()
        assert result.verdict == "PROVEN", f"T6 failed: {result.statement}"
        assert result.n_tests >= 5, f"T6 too few tests: {result.n_tests}"

    def test_T7_symmetry_breaking(self) -> None:
        """T7: EWSB amplifies generation spread in kernel traces."""
        result = theorem_T7_symmetry_breaking()
        assert result.verdict == "PROVEN", f"T7 failed: {result.statement}"
        assert result.n_tests >= 5, f"T7 too few tests: {result.n_tests}"

    def test_T8_ckm_unitarity(self) -> None:
        """T8: CKM rows pass Tier-1 kernel identity checks."""
        result = theorem_T8_ckm_unitarity()
        assert result.verdict == "PROVEN", f"T8 failed: {result.statement}"
        assert result.n_tests >= 5, f"T8 too few tests: {result.n_tests}"

    def test_T9_running_coupling_flow(self) -> None:
        """T9: α_s(Q²) drift maps to monotone ω(Q) for Q ≥ 10 GeV."""
        result = theorem_T9_running_coupling_flow()
        assert result.verdict == "PROVEN", f"T9 failed: {result.statement}"
        assert result.n_tests >= 5, f"T9 too few tests: {result.n_tests}"

    def test_T10_nuclear_binding_curve(self) -> None:
        """T10: BE/A anti-correlates with the heterogeneity gap."""
        result = theorem_T10_nuclear_binding_curve()
        assert result.verdict == "PROVEN", f"T10 failed: {result.statement}"
        assert result.n_tests >= 5, f"T10 too few tests: {result.n_tests}"

    def test_total_subtests_at_least_74(self) -> None:
        """All 10 theorems should produce at least 74 cumulative subtests."""
        theorems = [
            theorem_T1_spin_statistics,
            theorem_T2_generation_monotonicity,
            theorem_T3_confinement_IC_collapse,
            theorem_T4_mass_kernel_log_mapping,
            theorem_T5_charge_quantization,
            theorem_T6_cross_scale_universality,
            theorem_T7_symmetry_breaking,
            theorem_T8_ckm_unitarity,
            theorem_T9_running_coupling_flow,
            theorem_T10_nuclear_binding_curve,
        ]
        total = sum(t().n_tests for t in theorems)
        assert total >= 74, f"Only {total} tests, expected ≥ 74"

    def test_all_ten_proven(self) -> None:
        """Every single theorem must be PROVEN."""
        theorems = [
            ("T1", theorem_T1_spin_statistics),
            ("T2", theorem_T2_generation_monotonicity),
            ("T3", theorem_T3_confinement_IC_collapse),
            ("T4", theorem_T4_mass_kernel_log_mapping),
            ("T5", theorem_T5_charge_quantization),
            ("T6", theorem_T6_cross_scale_universality),
            ("T7", theorem_T7_symmetry_breaking),
            ("T8", theorem_T8_ckm_unitarity),
            ("T9", theorem_T9_running_coupling_flow),
            ("T10", theorem_T10_nuclear_binding_curve),
        ]
        failures = []
        for name, thm in theorems:
            result = thm()
            if result.verdict != "PROVEN":
                failures.append(f"{name}: {result.statement}")
        assert not failures, f"Theorems failed: {failures}"


# ═══════════════════════════════════════════════════════════════════
# §4  CROSS-SCALE THEOREMS (T17–T23): Unified Minimal Structure
# ═══════════════════════════════════════════════════════════════════


class TestUnifiedMinimalStructure:
    """Exercise theorems T17–T23 from unified_minimal_structure.py.

    These theorems are the cross-scale bridge — they prove that the
    kernel's structural properties hold from subatomic to cosmological
    scales. Previously had zero pytest coverage.
    """

    @pytest.fixture(scope="class")
    def theorems(self) -> dict:
        """Run all 7 theorems once, cache results."""
        from closures.unified_minimal_structure import (
            theorem_T17_mixing_complementarity,
            theorem_T18_scale_invariance,
            theorem_T19_gap_monotonicity,
            theorem_T20_entropy_scale,
            theorem_T21_fidelity_compression,
            theorem_T22_universal_regime,
            theorem_T23_return_universality,
        )

        return {
            "T17": theorem_T17_mixing_complementarity(),
            "T18": theorem_T18_scale_invariance(),
            "T19": theorem_T19_gap_monotonicity(),
            "T20": theorem_T20_entropy_scale(),
            "T21": theorem_T21_fidelity_compression(),
            "T22": theorem_T22_universal_regime(),
            "T23": theorem_T23_return_universality(),
        }

    def test_T17_mixing_complementarity(self, theorems: dict) -> None:
        """T17: CKM + PMNS are complementary mixing matrices."""
        assert theorems["T17"].verdict == "PROVEN", f"T17 failed: {theorems['T17'].statement}"

    def test_T18_scale_invariance(self, theorems: dict) -> None:
        """T18: Tier-1 identities hold across >6 physical scales."""
        assert theorems["T18"].verdict == "PROVEN", f"T18 failed: {theorems['T18'].statement}"

    def test_T19_gap_monotonicity(self, theorems: dict) -> None:
        """T19: Heterogeneity gap is positive and non-trivial at every scale."""
        assert theorems["T19"].verdict == "PROVEN", f"T19 failed: {theorems['T19'].statement}"

    def test_T20_entropy_scale(self, theorems: dict) -> None:
        """T20: Bernoulli field entropy varies across scales."""
        assert theorems["T20"].verdict == "PROVEN", f"T20 failed: {theorems['T20'].statement}"

    def test_T21_fidelity_compression(self, theorems: dict) -> None:
        """T21: F stays bounded [0.35, 0.75] across 44 OOM dynamic range."""
        assert theorems["T21"].verdict == "PROVEN", f"T21 failed: {theorems['T21'].statement}"

    def test_T22_universal_regime(self, theorems: dict) -> None:
        """T22: Stable/Watch/Collapse appear at every scale with frozen thresholds."""
        assert theorems["T22"].verdict == "PROVEN", f"T22 failed: {theorems['T22'].statement}"

    def test_T23_return_universality(self, theorems: dict) -> None:
        """T23: τ_R is finite at every tested scale — systems return."""
        assert theorems["T23"].verdict == "PROVEN", f"T23 failed: {theorems['T23'].statement}"

    def test_all_seven_proven(self, theorems: dict) -> None:
        """Every T17–T23 theorem must be PROVEN."""
        failures = [f"{name}: {result.statement}" for name, result in theorems.items() if result.verdict != "PROVEN"]
        assert not failures, f"Unified theorems failed: {failures}"

    def test_cumulative_subtests(self, theorems: dict) -> None:
        """T17–T23 should produce substantial subtest count."""
        total = sum(r.n_tests for r in theorems.values())
        assert total >= 20, f"Only {total} tests across T17–T23"


# ═══════════════════════════════════════════════════════════════════
# §5  TIER-1 PROOF HARNESS: The 10,162-test proof under pytest
# ═══════════════════════════════════════════════════════════════════


class TestTier1ProofHarness:
    """Exercise the exhaustive Tier-1 proof from tier1_proof.py.

    This proof exists in the closure code but was never invoked by pytest.
    It tests Tier-1 identities against:
      - All 118 elements
      - Random Monte Carlo vectors
      - Adversarial edge cases
      - Varying dimensions and weight distributions
    """

    def test_algebraic_proofs_run(self) -> None:
        """The algebraic pen-and-paper proofs execute without error."""
        from closures.atomic_physics.tier1_proof import (
            prove_identity_1_algebraic,
            prove_identity_2_algebraic,
            prove_identity_3_algebraic,
        )

        prove_identity_1_algebraic()
        prove_identity_2_algebraic()
        prove_identity_3_algebraic()

    def test_periodic_table_full(self) -> None:
        """All 118 elements pass Tier-1 identity checks."""
        from closures.atomic_physics.tier1_proof import test_periodic_table

        passed, failed = test_periodic_table()
        assert passed == 118, f"Only {passed}/118 elements passed"
        assert failed == 0, f"{failed} elements failed Tier-1"

    def test_random_vectors_pass(self) -> None:
        """1000 random trace vectors all pass Tier-1."""
        from closures.atomic_physics.tier1_proof import test_random_vectors

        passed, failed = test_random_vectors(n_trials=1000, max_dim=50)
        assert failed == 0, f"{failed}/1000 random vectors failed Tier-1"
        assert passed == 1000

    def test_adversarial_edge_cases(self) -> None:
        """Edge cases: near-zero, near-one, maximally heterogeneous."""
        from closures.atomic_physics.tier1_proof import test_adversarial_cases

        results = test_adversarial_cases()
        failures = [r for r in results if not r["all_pass"]]
        assert len(failures) == 0, f"{len(failures)} adversarial cases failed"

    def test_compound_molecules(self) -> None:
        """Tier-1 holds for compound molecules (H2O, CO2, etc.)."""
        from closures.atomic_physics.tier1_proof import test_compound_molecules

        passed, failed = test_compound_molecules()
        assert failed == 0, f"{failed} compound molecules failed Tier-1"
        assert passed >= 5, f"Only {passed} molecules tested"


# ═══════════════════════════════════════════════════════════════════
# §6  SECURITY DOMAIN: Kernel identity preservation
# ═══════════════════════════════════════════════════════════════════


class TestSecurityKernelBridge:
    """Verify that security closures produce outputs consistent with
    Tier-1 kernel identities.

    The security domain maps trust signals to the same mathematical
    structure: T + θ = 1 (analogous to F + ω = 1), TIC ≤ T (analogous
    to IC ≤ F). These tests verify the analogy is structurally exact.
    """

    @pytest.fixture(scope="class")
    def scenarios(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate diverse security signal scenarios."""
        rng = np.random.default_rng(42)
        scenarios = []

        # High-trust scenario (all signals near 1)
        scenarios.append(
            (
                np.array([0.95, 0.90, 0.88, 0.92, 0.97]),
                np.ones(5) / 5,
            )
        )

        # Low-trust scenario (one compromised channel)
        scenarios.append(
            (
                np.array([0.95, 0.90, 0.05, 0.92, 0.97]),
                np.ones(5) / 5,
            )
        )

        # Mixed scenario
        scenarios.append(
            (
                np.array([0.70, 0.40, 0.65, 0.80, 0.55, 0.30, 0.90, 0.15]),
                np.ones(8) / 8,
            )
        )

        # Non-uniform weights (concentrated trust)
        scenarios.append(
            (
                np.array([0.95, 0.50, 0.50, 0.50]),
                np.array([0.7, 0.1, 0.1, 0.1]),
            )
        )

        # 100 random scenarios
        for _ in range(100):
            n = rng.integers(3, 20)
            signals = rng.uniform(0.01, 0.99, n)
            weights = rng.dirichlet(np.ones(n))
            scenarios.append((signals, weights))

        return scenarios

    def test_trust_duality_identity(self, scenarios: list) -> None:
        """T + θ = 1 for all security scenarios (analogous to F + ω = 1)."""
        for signals, weights in scenarios:
            result = compute_trust_fidelity(signals, weights)
            T = result["T"]
            theta = result["theta"]
            assert abs(T + theta - 1.0) < 1e-10, f"Trust duality fails: T={T}, θ={theta}, sum={T + theta}"

    def test_trust_integrity_bound(self, scenarios: list) -> None:
        """TIC ≤ T for all security scenarios (analogous to IC ≤ F)."""
        for signals, weights in scenarios:
            tf = compute_trust_fidelity(signals, weights)
            ti = compute_trust_integrity(signals, weights)
            T = tf["T"]
            TIC = ti["TIC"]
            assert TIC <= T + 1e-6, f"Trust integrity bound fails: TIC={TIC} > T={T}"

    def test_trust_exp_bridge(self, scenarios: list) -> None:
        """TIC ≈ exp(σ) for all security scenarios (analogous to IC = exp(κ))."""
        for signals, weights in scenarios:
            ti = compute_trust_integrity(signals, weights)
            sigma = ti["sigma"]
            TIC = ti["TIC"]
            expected = math.exp(sigma)
            # TIC is clipped to [0, 1], so check within that range
            if expected <= 1.0:
                assert abs(TIC - expected) < 1e-6, f"Trust exp bridge fails: TIC={TIC}, exp(σ)={expected}"

    def test_security_kernel_isomorphic_to_core(self, scenarios: list) -> None:
        """Security kernel outputs must be structurally identical to core kernel.

        Feed the same signals through both the security domain and the
        core kernel — the Tier-1 identities must match.
        """
        for signals, weights in scenarios[:10]:
            # Security domain
            tf = compute_trust_fidelity(signals, weights)
            _ti = compute_trust_integrity(signals, weights)

            # Core kernel with same data
            c = np.clip(signals, EPSILON, 1.0 - EPSILON)
            core = compute_kernel_outputs(c, weights, EPSILON)

            # F ≈ T (both are weighted arithmetic means)
            assert abs(core["F"] - tf["T"]) < 0.01, f"F≠T: core F={core['F']:.6f}, trust T={tf['T']:.6f}"


# ═══════════════════════════════════════════════════════════════════
# §7  SEAM COMPOSITION ACROSS DOMAINS: Real data through seam chains
# ═══════════════════════════════════════════════════════════════════


class TestCrossDomainSeamChain:
    """Build seam chains from real domain data and verify that seam
    composition properties hold across domain boundaries.

    The seam chain accumulator (Lemma 20) should compose correctly
    whether the data comes from atoms, particles, or security signals.
    """

    def test_atomic_seam_chain(self) -> None:
        """Build a seam chain from first 10 elements' kernel outputs.

        Each element is a 'time step' — the seam tracks how κ evolves
        as we walk through the periodic table.
        """
        enhanced = compute_all_enhanced()[:10]
        chain = SeamChainAccumulator()

        for i in range(1, len(enhanced)):
            chain.add_seam(
                t0=i - 1,
                t1=i,
                kappa_t0=enhanced[i - 1].kappa,
                kappa_t1=enhanced[i].kappa,
                tau_R=1.0,
                R=0.01,
            )

        metrics = chain.get_metrics()
        assert metrics.total_seams == 9
        # Residual should be bounded
        assert metrics.cumulative_abs_residual < 100.0
        # Total Δκ should equal direct difference
        direct = enhanced[-1].kappa - enhanced[0].kappa
        assert abs(metrics.total_delta_kappa - direct) < 1e-10, "Lemma 20 (additive composition) fails for atomic chain"

    def test_particle_seam_chain(self) -> None:
        """Build a seam chain from fundamental particle kernel outputs."""
        fund = compute_all_fundamental()
        chain = SeamChainAccumulator()

        for i in range(1, len(fund)):
            chain.add_seam(
                t0=i - 1,
                t1=i,
                kappa_t0=fund[i - 1].kappa,
                kappa_t1=fund[i].kappa,
                tau_R=1.0,
                R=0.01,
            )

        metrics = chain.get_metrics()
        assert metrics.total_seams == 16  # 17 particles, 16 transitions
        direct = fund[-1].kappa - fund[0].kappa
        assert abs(metrics.total_delta_kappa - direct) < 1e-10

    def test_cross_domain_composite_chain(self) -> None:
        """Build a seam chain that crosses domain boundaries:
        fundamental (SM) → composite (SM) → atomic (atomic_physics).

        Lemma 20 (additive composition) must hold even across domains.
        """
        fund = compute_all_fundamental()
        comp = compute_all_composite()
        atoms_sample = compute_all_enhanced()[:5]

        # Concatenate kappas across three scales
        kappas = (
            [p.kappa for p in fund[-3:]]  # last 3 fundamental
            + [p.kappa for p in comp[:3]]  # first 3 composite
            + [el.kappa for el in atoms_sample[:3]]  # first 3 atoms
        )

        chain = SeamChainAccumulator()
        for i in range(1, len(kappas)):
            chain.add_seam(
                t0=i - 1,
                t1=i,
                kappa_t0=kappas[i - 1],
                kappa_t1=kappas[i],
                tau_R=1.0,
                R=0.01,
            )

        metrics = chain.get_metrics()
        direct = kappas[-1] - kappas[0]
        assert abs(metrics.total_delta_kappa - direct) < 1e-10, (
            f"Lemma 20 fails across domain boundaries: chain={metrics.total_delta_kappa:.10f}, direct={direct:.10f}"
        )


# ═══════════════════════════════════════════════════════════════════
# §8  NUCLEAR ↔ ATOMIC: Magic Number Consistency
# ═══════════════════════════════════════════════════════════════════


class TestNuclearAtomicBridge:
    """Verify that nuclear physics concepts are correctly reflected
    in atomic kernel signatures.

    The cross_scale_kernel imports nuclear functions (binding energy,
    magic proximity) and feeds them as channels into the atomic kernel.
    These tests verify the bridge is physically meaningful.
    """

    def test_magic_numbers_consistent(self) -> None:
        """MAGIC_NUMBERS in cross_scale_kernel matches nuclear physics."""
        expected = (2, 8, 20, 28, 50, 82, 126)
        assert expected == ATOMIC_MAGIC_NUMBERS

    def test_doubly_magic_nuclei_have_high_magic_proximity(self) -> None:
        """Doubly magic nuclei (Z and N both magic) should have proximity ~1.0.

        He-4 (Z=2, N=2), O-16 (Z=8, N=8), Ca-40 (Z=20, N=20),
        Pb-208 (Z=82, N=126) are doubly magic.
        """
        doubly_magic = [
            (2, 4),  # He-4
            (8, 16),  # O-16
            (20, 40),  # Ca-40
            (82, 208),  # Pb-208
        ]
        for Z, A in doubly_magic:
            mp = magic_proximity(Z, A)
            assert mp >= 0.9, f"Doubly magic Z={Z}, A={A} has low proximity: {mp:.4f}"

    def test_binding_energy_peaks_near_iron(self) -> None:
        """BE/A should peak near Fe-56 (Z=26, A=56)."""
        peak_bea = 0.0
        peak_Z = 0
        for el in ELEMENTS:
            A = round(el.atomic_mass)
            bea = binding_energy_per_nucleon(el.Z, A)
            if bea > peak_bea:
                peak_bea = bea
                peak_Z = el.Z

        # Peak should be near iron (Z=24-30 range per Bethe-Weizsäcker)
        assert 20 <= peak_Z <= 35, f"BE/A peak at Z={peak_Z}, expected near Fe"
        assert peak_bea > 8.0, f"Peak BE/A={peak_bea:.2f}, expected >8 MeV"

    def test_magic_proximity_affects_kernel(self) -> None:
        """Elements near magic numbers should show measurably different
        kernel signatures than elements far from magic numbers.

        Specifically: magic-proximal elements should have higher
        magic_prox channel values in their trace vectors.
        """
        enhanced = compute_all_enhanced()

        # He (Z=2, magic), O (Z=8, magic), Ca (Z=20, magic)
        magic_elements = [e for e in enhanced if e.Z in (2, 8, 20)]
        # Na (Z=11), Al (Z=13), Mn (Z=25) — far from magic
        non_magic = [e for e in enhanced if e.Z in (11, 13, 25)]

        avg_mp_magic = sum(e.magic_proximity for e in magic_elements) / len(magic_elements)
        avg_mp_non = sum(e.magic_proximity for e in non_magic) / len(non_magic)

        assert avg_mp_magic > avg_mp_non, (
            f"Magic proximity not higher for magic elements: magic={avg_mp_magic:.4f}, non-magic={avg_mp_non:.4f}"
        )

    def test_hydrogen_no_binding(self) -> None:
        """Hydrogen (A=1) has zero binding energy — can't bind a single nucleon."""
        bea = binding_energy_per_nucleon(1, 1)
        assert bea == 0.0, f"Hydrogen should have zero binding: {bea}"

    def test_nuclear_channels_in_atomic_kernel(self) -> None:
        """Iron's kernel should show BE/A near maximum in its trace vector."""
        iron = compute_enhanced_kernel(ELEMENTS[25])  # Fe, Z=26
        assert "BE_per_A" in iron.channel_labels
        bea_idx = iron.channel_labels.index("BE_per_A")
        # Iron's normalized BE/A should be very high (near 1.0)
        assert iron.trace_vector[bea_idx] > 0.9, f"Iron BE/A channel too low: {iron.trace_vector[bea_idx]:.4f}"


# ═══════════════════════════════════════════════════════════════════
# §9  RCFT ↔ QM: Fisher Distance Bridge
# ═══════════════════════════════════════════════════════════════════


class TestRCFTQuantumBridge:
    """Verify the RCFT information geometry functions produce valid
    results when called from quantum mechanics closures.

    The TERS near-field closure imports fisher_distance_weighted from
    RCFT. This test verifies the bridge produces mathematically valid
    output — non-negative distances, triangle inequality, etc.
    """

    @pytest.fixture(scope="class")
    def fisher_funcs(self):
        """Import the RCFT Fisher distance functions."""
        from closures.rcft.information_geometry import (
            fisher_distance_weighted,
        )

        return {"weighted": fisher_distance_weighted}

    def test_fisher_distance_non_negative(self, fisher_funcs) -> None:
        """Fisher distance must be ≥ 0 for any two probability vectors."""
        fd = fisher_funcs["weighted"]
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.5, 0.2, 0.3])
        w = np.ones(3) / 3

        result = fd(p, q, w)
        assert result.distance >= 0, f"Negative Fisher distance: {result.distance}"

    def test_fisher_distance_zero_for_identical(self, fisher_funcs) -> None:
        """Fisher distance between identical distributions is 0."""
        fd = fisher_funcs["weighted"]
        p = np.array([0.3, 0.4, 0.3])
        w = np.ones(3) / 3

        result = fd(p, p, w)
        assert abs(result.distance) < 1e-10, f"Non-zero self-distance: {result.distance}"

    def test_fisher_distance_symmetric(self, fisher_funcs) -> None:
        """Fisher distance is symmetric: d(p, q) = d(q, p)."""
        fd = fisher_funcs["weighted"]
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.5, 0.2, 0.3])
        w = np.ones(3) / 3

        d_pq = fd(p, q, w).distance
        d_qp = fd(q, p, w).distance
        assert abs(d_pq - d_qp) < 1e-10, f"Not symmetric: d(p,q)={d_pq}, d(q,p)={d_qp}"

    def test_fisher_distance_triangle_inequality(self, fisher_funcs) -> None:
        """Fisher distance satisfies triangle inequality: d(p,r) ≤ d(p,q) + d(q,r)."""
        fd = fisher_funcs["weighted"]
        p = np.array([0.6, 0.2, 0.2])
        q = np.array([0.3, 0.4, 0.3])
        r = np.array([0.1, 0.1, 0.8])
        w = np.ones(3) / 3

        d_pr = fd(p, r, w).distance
        d_pq = fd(p, q, w).distance
        d_qr = fd(q, r, w).distance

        assert d_pr <= d_pq + d_qr + 1e-10, f"Triangle inequality violated: d(p,r)={d_pr}, d(p,q)+d(q,r)={d_pq + d_qr}"

    def test_fisher_with_kernel_trace_vectors(self, fisher_funcs) -> None:
        """Fisher distance should work on kernel-derived trace vectors.

        Take two elements' trace vectors and compute Fisher distance —
        this is the actual cross-domain usage pattern (physics data → RCFT geometry).
        """
        fd = fisher_funcs["weighted"]

        # Get normalized trace data from two elements
        c1, _w1, _ = normalize_element_enhanced(ELEMENTS[0])  # Hydrogen
        c2, _w2, _ = normalize_element_enhanced(ELEMENTS[25])  # Iron

        # Both should have same number of channels for distance calc
        n = min(len(c1), len(c2))
        result = fd(c1[:n], c2[:n], np.ones(n) / n)
        assert result.distance >= 0, f"Negative Fisher distance for H vs Fe: {result.distance}"
        assert result.distance > 0.01, f"H and Fe should be far apart: {result.distance}"


# ═══════════════════════════════════════════════════════════════════
# §10 ROSETTA INVARIANCE: Same data, multiple lenses, same Tier-1
# ═══════════════════════════════════════════════════════════════════


class TestRosettaInvariance:
    """The Rosetta principle: the same data processed through different
    domain lenses must satisfy the same Tier-1 identities.

    This is the deepest structural test — it proves that Tier-1
    is truly universal, not domain-dependent.
    """

    @pytest.fixture(scope="class")
    def shared_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """A shared 8-channel trace vector used across all lenses."""
        c = np.array([0.85, 0.72, 0.45, 0.91, 0.33, 0.67, 0.78, 0.55])
        w = np.ones(8) / 8
        return c, w

    def test_core_kernel_tier1(self, shared_trace: tuple) -> None:
        """Core kernel satisfies Tier-1 on shared trace."""
        c, w = shared_trace
        k = compute_kernel_outputs(c, w, EPSILON)
        assert abs(k["F"] + k["omega"] - 1.0) < TOL_TIER1
        assert k["IC"] <= k["F"] + TOL_TIER1
        assert abs(k["IC"] - math.exp(k["kappa"])) < TOL_TIER1

    def test_security_lens_tier1(self, shared_trace: tuple) -> None:
        """Security lens (same data interpreted as trust signals) satisfies Tier-1."""
        c, w = shared_trace
        tf = compute_trust_fidelity(c, w)
        ti = compute_trust_integrity(c, w)
        assert abs(tf["T"] + tf["theta"] - 1.0) < TOL_TIER1
        assert ti["TIC"] <= tf["T"] + TOL_TIER1

    def test_same_F_across_lenses(self, shared_trace: tuple) -> None:
        """F (core) ≈ T (security) when computed on identical data."""
        c, w = shared_trace
        core_F = compute_kernel_outputs(c, w, EPSILON)["F"]
        security_T = compute_trust_fidelity(c, w)["T"]
        # Should be nearly identical (both are weighted arithmetic mean)
        assert abs(core_F - security_T) < 0.01, f"F ≠ T on same data: F={core_F:.6f}, T={security_T:.6f}"

    def test_same_IC_across_lenses(self, shared_trace: tuple) -> None:
        """IC (core) ≈ TIC (security) on identical data."""
        c, w = shared_trace
        core_IC = compute_kernel_outputs(c, w, EPSILON)["IC"]
        security_TIC = compute_trust_integrity(c, w)["TIC"]
        assert abs(core_IC - security_TIC) < 0.01, f"IC ≠ TIC on same data: IC={core_IC:.6f}, TIC={security_TIC:.6f}"

    def test_tier1_invariant_across_domains_parametric(self) -> None:
        """Parametric test: 100 random vectors through both core and security.

        Tier-1 must hold in BOTH domain interpretations of the same data.
        """
        rng = np.random.default_rng(12345)
        for _ in range(100):
            n = rng.integers(3, 15)
            c = rng.uniform(0.01, 0.99, n)
            w = rng.dirichlet(np.ones(n))

            # Core
            k = compute_kernel_outputs(c, w, EPSILON)
            assert abs(k["F"] + k["omega"] - 1.0) < TOL_TIER1
            assert k["IC"] <= k["F"] + TOL_TIER1

            # Security
            tf = compute_trust_fidelity(c, w)
            ti = compute_trust_integrity(c, w)
            assert abs(tf["T"] + tf["theta"] - 1.0) < TOL_TIER1
            assert ti["TIC"] <= tf["T"] + TOL_TIER1

    def test_regime_classification_consistent_across_domains(self) -> None:
        """A high-fidelity vector should classify as Stable/Watch in both
        the core kernel and any domain lens."""
        c_stable = np.array([0.95, 0.92, 0.97, 0.94, 0.96, 0.93, 0.98, 0.91])
        w = np.ones(8) / 8

        k = compute_kernel_outputs(c_stable, w, EPSILON)
        tf = compute_trust_fidelity(c_stable, w)

        # Both should see high fidelity
        assert k["F"] > 0.90
        assert tf["T"] > 0.90

        # Both should see low drift
        assert k["omega"] < 0.10
        assert tf["theta"] < 0.10
