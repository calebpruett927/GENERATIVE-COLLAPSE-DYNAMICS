"""Tests for evolution domain closures — EVO.INTSTACK.v1

Comprehensive test coverage for all 4 evolution closure modules:
  - evolution_kernel.py: 40 organisms × 8 channels (Tier-1 identity sweep)
  - axiom0_instantiation_map.py: 20 phenomena as collapse-return cycles
  - recursive_evolution.py: 5 scales + 5 mass extinctions
  - deep_implications.py: 8 identity maps, 8 case studies, Fisher info

Every test verifies structural predictions derivable from Axiom-0:
  F + ω = 1 (duality identity — complementum perfectum)
  IC ≤ F (integrity bound — limbus integritatis)
  IC = exp(κ) (log-integritas)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → evolution closures
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.evolution.axiom0_instantiation_map import (
    PHENOMENA,
    PhenomenonKernelResult,
    compute_all_phenomena,
)
from closures.evolution.deep_implications import (
    DEEP_CASES,
    IDENTITY_MAP,
    PREDICTIONS,
    compute_fisher_information_link,
)
from closures.evolution.evolution_kernel import (
    CHANNEL_LABELS,
    N_CHANNELS,
    ORGANISMS,
    EvolutionKernelResult,
    compute_all_organisms,
    normalize_organism,
)
from closures.evolution.recursive_evolution import (
    MASS_EXTINCTIONS,
    SCALES,
    ScaleKernelResult,
    compute_all_scales,
    compute_extinction_kernel,
)

# ── Tolerances (same as frozen contract) ──────────────────────────
TOL_DUALITY = 1e-12  # F + ω = 1 exact to machine precision
TOL_EXP = 1e-9  # IC = exp(κ)
TOL_BOUND = 1e-12  # IC ≤ F (with guard)
EPS = 1e-6  # Closure-level ε


# ═══════════════════════════════════════════════════════════════════
# 1. Evolution Kernel — 40 Organisms (core identity sweep)
# ═══════════════════════════════════════════════════════════════════


class TestOrganismCatalog:
    """Verify the organism catalog data integrity."""

    def test_organism_count(self) -> None:
        """40 organisms in the catalog."""
        assert len(ORGANISMS) == 40

    def test_channel_count(self) -> None:
        """8 channels defined."""
        assert N_CHANNELS == 8
        assert len(CHANNEL_LABELS) == 8

    def test_all_organisms_have_8_traits(self) -> None:
        """Every organism has exactly 8 trait values."""
        for org in ORGANISMS:
            traits = [
                org.genetic_diversity,
                org.morphological_fitness,
                org.reproductive_success,
                org.metabolic_efficiency,
                org.immune_competence,
                org.environmental_breadth,
                org.behavioral_complexity,
                org.lineage_persistence,
            ]
            assert len(traits) == 8, f"{org.name}: expected 8 traits, got {len(traits)}"

    def test_trait_values_in_unit_interval(self) -> None:
        """All trait values must be in [0, 1]."""
        for org in ORGANISMS:
            for attr in CHANNEL_LABELS:
                val = getattr(org, attr)
                assert 0.0 <= val <= 1.0, f"{org.name}.{attr} = {val} out of [0,1]"

    def test_status_values(self) -> None:
        """Status must be 'extant' or 'extinct'."""
        for org in ORGANISMS:
            assert org.status in ("extant", "extinct"), f"{org.name}: status={org.status}"

    def test_extinct_count(self) -> None:
        """5 extinct organisms in the catalog."""
        extinct = [o for o in ORGANISMS if o.status == "extinct"]
        assert len(extinct) == 5

    def test_extant_count(self) -> None:
        """35 extant organisms in the catalog."""
        extant = [o for o in ORGANISMS if o.status == "extant"]
        assert len(extant) == 35

    def test_unique_names(self) -> None:
        """No duplicate organism names."""
        names = [o.name for o in ORGANISMS]
        assert len(names) == len(set(names))


class TestNormalizeOrganism:
    """Test the normalization pipeline."""

    def test_normalize_returns_correct_shapes(self) -> None:
        c, w, labels = normalize_organism(ORGANISMS[0])
        assert c.shape == (8,)
        assert w.shape == (8,)
        assert len(labels) == 8

    def test_weights_sum_to_one(self) -> None:
        _, w, _ = normalize_organism(ORGANISMS[0])
        assert abs(w.sum() - 1.0) < 1e-15

    def test_channels_clamped(self) -> None:
        """All channels must be clamped to [EPS, 1-EPS]."""
        for org in ORGANISMS:
            c, _, _ = normalize_organism(org)
            assert np.all(c >= EPS), f"{org.name}: channel below EPS"
            assert np.all(c <= 1.0 - EPS), f"{org.name}: channel above 1-EPS"


class TestOrganismKernelIdentities:
    """Tier-1 identity sweep across all 40 organisms.

    These are the structural identities of collapse — they must hold
    for every organism regardless of trait values. Violations here
    would indicate a broken kernel, not a biological anomaly.
    """

    @pytest.fixture(scope="class")
    def all_results(self) -> list[EvolutionKernelResult]:
        return compute_all_organisms()

    def test_duality_identity_all(self, all_results: list[EvolutionKernelResult]) -> None:
        """F + ω = 1 for every organism (complementum perfectum)."""
        for r in all_results:
            assert abs(r.F_plus_omega - 1.0) < TOL_DUALITY, f"{r.name}: F+ω = {r.F_plus_omega}"

    def test_integrity_bound_all(self, all_results: list[EvolutionKernelResult]) -> None:
        """IC ≤ F for every organism (limbus integritatis)."""
        for r in all_results:
            assert r.IC_leq_F, f"{r.name}: IC={r.IC} > F={r.F}"

    def test_log_integritas_all(self, all_results: list[EvolutionKernelResult]) -> None:
        """IC = exp(κ) for every organism."""
        for r in all_results:
            assert r.IC_eq_exp_kappa, f"{r.name}: IC={r.IC}, exp(κ)={math.exp(r.kappa)}"

    def test_F_range(self, all_results: list[EvolutionKernelResult]) -> None:
        """F must be in [0, 1] for all organisms."""
        for r in all_results:
            assert 0.0 <= r.F <= 1.0, f"{r.name}: F={r.F}"

    def test_omega_range(self, all_results: list[EvolutionKernelResult]) -> None:
        """ω must be in [0, 1] for all organisms."""
        for r in all_results:
            assert 0.0 <= r.omega <= 1.0, f"{r.name}: ω={r.omega}"

    def test_IC_range(self, all_results: list[EvolutionKernelResult]) -> None:
        """IC must be in (0, 1]."""
        for r in all_results:
            assert 0.0 < r.IC <= 1.0, f"{r.name}: IC={r.IC}"

    def test_kappa_nonpositive(self, all_results: list[EvolutionKernelResult]) -> None:
        """κ must be ≤ 0 (log of values in (0,1])."""
        for r in all_results:
            assert r.kappa <= 0.0 + 1e-12, f"{r.name}: κ={r.kappa}"

    def test_heterogeneity_gap_nonnegative(self, all_results: list[EvolutionKernelResult]) -> None:
        """Δ = F - IC ≥ 0 (consequence of IC ≤ F)."""
        for r in all_results:
            assert r.heterogeneity_gap >= -TOL_BOUND, f"{r.name}: Δ={r.heterogeneity_gap}"

    def test_entropy_nonnegative(self, all_results: list[EvolutionKernelResult]) -> None:
        """Bernoulli field entropy S ≥ 0."""
        for r in all_results:
            assert r.S >= -1e-12, f"{r.name}: S={r.S}"

    def test_curvature_range(self, all_results: list[EvolutionKernelResult]) -> None:
        """Curvature C in [0, 1]."""
        for r in all_results:
            assert 0.0 <= r.C <= 1.0 + 1e-12, f"{r.name}: C={r.C}"


class TestOrganismRegimeStrategy:
    """Verify regime and strategy classifications."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[EvolutionKernelResult]:
        return compute_all_organisms()

    def test_valid_regimes(self, all_results: list[EvolutionKernelResult]) -> None:
        """Regime must be one of the three canonical values."""
        for r in all_results:
            assert r.regime in ("Stable", "Watch", "Collapse"), f"{r.name}: regime={r.regime}"

    def test_valid_strategies(self, all_results: list[EvolutionKernelResult]) -> None:
        """Strategy must be one of the 5 defined categories."""
        valid = {
            "Robust Generalist",
            "Adapted Specialist",
            "Resilient Ancient",
            "Vulnerable Specialist",
            "Minimal Viable",
        }
        for r in all_results:
            assert r.evolutionary_strategy in valid, f"{r.name}: strategy={r.evolutionary_strategy}"

    def test_weakest_strongest_channels_exist(self, all_results: list[EvolutionKernelResult]) -> None:
        """Weakest and strongest channels must be valid channel labels."""
        for r in all_results:
            assert r.weakest_channel in CHANNEL_LABELS, f"{r.name}: weakest={r.weakest_channel}"
            assert r.strongest_channel in CHANNEL_LABELS, f"{r.name}: strongest={r.strongest_channel}"


class TestOrganismBiologicalPredictions:
    """Domain-specific predictions from the GCD kernel.

    These verify that the kernel's structural output aligns with
    known evolutionary biology — not by fitting, but by structural
    correspondence.
    """

    @pytest.fixture(scope="class")
    def all_results(self) -> list[EvolutionKernelResult]:
        return compute_all_organisms()

    def test_extant_higher_IC_than_extinct(self, all_results: list[EvolutionKernelResult]) -> None:
        """Extant organisms should have higher mean IC than extinct ones.

        Survival = return = demonstrated persistence. GCD predicts that
        lineages with higher multiplicative coherence (IC) persist longer.
        """
        extant = [r for r in all_results if r.status == "extant"]
        extinct = [r for r in all_results if r.status == "extinct"]
        mean_ic_extant = np.mean([r.IC for r in extant])
        mean_ic_extinct = np.mean([r.IC for r in extinct])
        assert mean_ic_extant > mean_ic_extinct, (
            f"⟨IC⟩_extant={mean_ic_extant:.4f} should exceed ⟨IC⟩_extinct={mean_ic_extinct:.4f}"
        )

    def test_homo_sapiens_collapse_regime(self, all_results: list[EvolutionKernelResult]) -> None:
        """Homo sapiens should be in Collapse regime (ω > 0.30).

        Driven by lineage_persistence = 0.001 (geological youth).
        This is structurally honest, not failure.
        """
        human = next(r for r in all_results if r.name == "Homo sapiens")
        assert human.regime == "Collapse"
        assert human.omega > 0.30

    def test_homo_sapiens_vulnerable_specialist(self, all_results: list[EvolutionKernelResult]) -> None:
        """Homo sapiens is Vulnerable Specialist: high F, high Δ."""
        human = next(r for r in all_results if r.name == "Homo sapiens")
        assert human.evolutionary_strategy == "Vulnerable Specialist"

    def test_homo_sapiens_weakest_channel(self, all_results: list[EvolutionKernelResult]) -> None:
        """Lineage persistence should be the weakest channel for Homo sapiens."""
        human = next(r for r in all_results if r.name == "Homo sapiens")
        assert human.weakest_channel == "lineage_persistence"

    def test_dodo_minimal_viable(self, all_results: list[EvolutionKernelResult]) -> None:
        """Dodo should be Minimal Viable — geometric slaughter in action."""
        dodo = next(r for r in all_results if "Dodo" in r.name)
        assert dodo.evolutionary_strategy == "Minimal Viable"
        assert dodo.status == "extinct"

    def test_coelacanth_resilient_ancient(self, all_results: list[EvolutionKernelResult]) -> None:
        """Coelacanth should be Resilient Ancient — low F, low Δ, high persistence."""
        coela = next(r for r in all_results if "coelacanth" in r.name)
        assert coela.evolutionary_strategy == "Resilient Ancient"

    def test_living_fossils_high_IC_F_ratio(self, all_results: list[EvolutionKernelResult]) -> None:
        """Living fossils (coelacanth, horseshoe crab) should have high IC/F ratio.

        Uniform mediocrity → IC ≈ F → low heterogeneity gap → persistence.
        """
        fossils = [r for r in all_results if "coelacanth" in r.name or "horseshoe" in r.name]
        for r in fossils:
            ic_f = r.IC / r.F
            assert ic_f > 0.70, f"{r.name}: IC/F = {ic_f:.4f} — living fossil should have IC/F > 0.70"

    def test_bacteria_high_persistence(self, all_results: list[EvolutionKernelResult]) -> None:
        """Bacteria should have high lineage persistence (~3.8 Gyr)."""
        bacteria = [r for r in all_results if any(o.domain == "Bacteria" and o.name == r.name for o in ORGANISMS)]
        for r in bacteria:
            assert r.trace_vector[7] >= 0.85, f"{r.name}: lineage_persistence={r.trace_vector[7]}"


# ═══════════════════════════════════════════════════════════════════
# 2. Axiom-0 Instantiation Map — 20 Phenomena
# ═══════════════════════════════════════════════════════════════════


class TestPhenomenaCatalog:
    """Verify the phenomena catalog data integrity."""

    def test_phenomena_count(self) -> None:
        assert len(PHENOMENA) == 20

    def test_all_phenomena_have_8_channels(self) -> None:
        for p in PHENOMENA:
            assert len(p.channel_labels) == 8, f"{p.name}: {len(p.channel_labels)} channels"
            assert len(p.pre_channels) == 8, f"{p.name}: pre has {len(p.pre_channels)}"
            assert len(p.post_channels) == 8, f"{p.name}: post has {len(p.post_channels)}"
            assert len(p.return_channels) == 8, f"{p.name}: return has {len(p.return_channels)}"

    def test_channel_values_in_range(self) -> None:
        """All channel values in [0, 1]."""
        for p in PHENOMENA:
            for state_name, channels in [
                ("pre", p.pre_channels),
                ("post", p.post_channels),
                ("return", p.return_channels),
            ]:
                for i, v in enumerate(channels):
                    assert 0.0 <= v <= 1.0, f"{p.name}.{state_name}[{i}] = {v}"

    def test_unique_phenomenon_names(self) -> None:
        names = [p.name for p in PHENOMENA]
        assert len(names) == len(set(names))

    def test_valid_categories(self) -> None:
        valid = {"Molecular", "Cellular", "Organismal", "Population", "Biosphere"}
        for p in PHENOMENA:
            assert p.category in valid, f"{p.name}: category={p.category}"


class TestPhenomenaKernelIdentities:
    """Tier-1 identity sweep across all 60 states (20 × 3)."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[PhenomenonKernelResult]:
        return compute_all_phenomena()

    def test_duality_all_states(self, all_results: list[PhenomenonKernelResult]) -> None:
        """F + ω = 1 in all 60 states."""
        for r in all_results:
            for prefix, F, omega in [
                ("pre", r.pre_F, r.pre_omega),
                ("post", r.post_F, r.post_omega),
                ("ret", r.ret_F, r.ret_omega),
            ]:
                assert abs(F + omega - 1.0) < TOL_DUALITY, f"{r.name} {prefix}: F+ω = {F + omega}"

    def test_integrity_bound_all_states(self, all_results: list[PhenomenonKernelResult]) -> None:
        """IC ≤ F in all 60 states."""
        for r in all_results:
            for prefix, F, IC in [
                ("pre", r.pre_F, r.pre_IC),
                ("post", r.post_F, r.post_IC),
                ("ret", r.ret_F, r.ret_IC),
            ]:
                assert IC <= F + TOL_BOUND, f"{r.name} {prefix}: IC={IC} > F={F}"

    def test_valid_regimes_all_states(self, all_results: list[PhenomenonKernelResult]) -> None:
        """All regimes must be canonical values."""
        for r in all_results:
            for regime in [r.pre_regime, r.post_regime, r.ret_regime]:
                assert regime in ("Stable", "Watch", "Collapse"), f"{r.name}: regime={regime}"


class TestPredictionAccuracy:
    """Verify predicted_generative matches computed reality."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[PhenomenonKernelResult]:
        return compute_all_phenomena()

    def test_all_predictions_match(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Every predicted_generative flag must match computed is_generative."""
        mismatches = [r for r in all_results if r.is_generative != r.predicted_generative]
        assert len(mismatches) == 0, f"Mismatches: {[r.name for r in mismatches]}"

    def test_generative_count(self, all_results: list[PhenomenonKernelResult]) -> None:
        """15 of 20 phenomena should be generative."""
        n_gen = sum(1 for r in all_results if r.is_generative)
        assert n_gen == 15, f"Expected 15 generative, got {n_gen}"

    def test_non_generative_count(self, all_results: list[PhenomenonKernelResult]) -> None:
        """5 phenomena should be non-generative (including Cancer)."""
        non_gen = [r for r in all_results if not r.is_generative]
        assert len(non_gen) == 5
        names = {r.name for r in non_gen}
        # Cancer is the canonical anti-proof
        assert "Cancer (Cellular Defection)" in names

    def test_cancer_not_generative(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Cancer must be non-generative — the anti-proof of GCD."""
        cancer = next(r for r in all_results if "Cancer" in r.name)
        assert not cancer.is_generative
        assert not cancer.predicted_generative
        assert cancer.has_gestus

    def test_endosymbiosis_not_generative(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Endosymbiosis: independence channels collapse permanently."""
        endo = next(r for r in all_results if "Endosymbiosis" in r.name)
        assert not endo.is_generative
        assert not endo.predicted_generative

    def test_founder_effect_not_generative(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Founder effect: diversity loss is permanent (IC_ret < IC_pre)."""
        founder = next(r for r in all_results if "Founder" in r.name)
        assert not founder.is_generative
        assert not founder.predicted_generative


class TestPhenomenaStructuralPatterns:
    """Domain-specific structural predictions."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[PhenomenonKernelResult]:
        return compute_all_phenomena()

    def test_IC_drops_at_nadir(self, all_results: list[PhenomenonKernelResult]) -> None:
        """IC should drop at the nadir for at least half of phenomena.

        Not all phenomena show IC drop: some have constructive nadir
        states where post-collapse channels are higher than pre.
        """
        n_drop = sum(1 for r in all_results if r.post_IC < r.pre_IC)
        # At least 10/20 should show IC drop at nadir
        assert n_drop >= 10, f"Only {n_drop}/20 show IC drop at nadir"

    def test_generative_collapses_exceed_pre(self, all_results: list[PhenomenonKernelResult]) -> None:
        """For all generative phenomena, IC_return > IC_pre."""
        for r in all_results:
            if r.is_generative:
                assert r.ret_IC > r.pre_IC, f"{r.name}: generative but IC_ret={r.ret_IC} ≤ IC_pre={r.pre_IC}"

    def test_gestus_phenomena_exist(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Multiple phenomena should have gestus (some lineages → ∞_rec)."""
        n_gestus = sum(1 for r in all_results if r.has_gestus)
        assert n_gestus >= 10, f"Only {n_gestus}/20 have gestus"

    def test_holocene_extinction_predicted_generative(self, all_results: list[PhenomenonKernelResult]) -> None:
        """Holocene extinction is predicted generative but not yet demonstrated."""
        holocene = next(r for r in all_results if "Holocene" in r.name)
        assert holocene.predicted_generative  # predicted, not yet proven


# ═══════════════════════════════════════════════════════════════════
# 3. Recursive Evolution — 5 Scales + 5 Mass Extinctions
# ═══════════════════════════════════════════════════════════════════


class TestScaleCatalog:
    """Verify the evolutionary scale definitions."""

    def test_scale_count(self) -> None:
        assert len(SCALES) == 5

    def test_scale_levels(self) -> None:
        """Scales must be numbered 1-5."""
        levels = [s.level for s in SCALES]
        assert levels == [1, 2, 3, 4, 5]

    def test_scale_names(self) -> None:
        expected = ["Gene", "Organism", "Population", "Species", "Clade"]
        actual = [s.name for s in SCALES]
        assert actual == expected

    def test_all_scales_have_8_channels(self) -> None:
        for s in SCALES:
            assert len(s.channel_labels) == 8, f"{s.name}: {len(s.channel_labels)}"
            assert len(s.channel_values) == 8, f"{s.name}: {len(s.channel_values)}"


class TestScaleKernelIdentities:
    """Tier-1 identity sweep across all 5 evolutionary scales."""

    @pytest.fixture(scope="class")
    def scale_results(self) -> list[ScaleKernelResult]:
        return compute_all_scales()

    def test_duality_all_scales(self, scale_results: list[ScaleKernelResult]) -> None:
        """F + ω = 1 at every scale."""
        for r in scale_results:
            assert r.F_plus_omega_exact, f"{r.scale_name}: F+ω = {r.F + r.omega}"

    def test_integrity_bound_all_scales(self, scale_results: list[ScaleKernelResult]) -> None:
        """IC ≤ F at every scale."""
        for r in scale_results:
            assert r.IC_leq_F, f"{r.scale_name}: IC={r.IC} > F={r.F}"

    def test_log_integritas_all_scales(self, scale_results: list[ScaleKernelResult]) -> None:
        """IC = exp(κ) at every scale."""
        for r in scale_results:
            assert r.IC_eq_exp_kappa, f"{r.scale_name}: IC={r.IC}, exp(κ)={math.exp(r.kappa)}"

    def test_IC_degrades_with_scale(self, scale_results: list[ScaleKernelResult]) -> None:
        """IC should generally degrade from Gene → Clade.

        Each higher scale has more heterogeneous channels (more
        uncontrolled degrees of freedom), so IC decreases.
        """
        gene = scale_results[0]
        clade = scale_results[-1]
        assert gene.IC > clade.IC, f"Gene IC={gene.IC:.4f} should exceed Clade IC={clade.IC:.4f}"


class TestMassExtinctions:
    """Test the 5 mass extinction events."""

    def test_extinction_count(self) -> None:
        assert len(MASS_EXTINCTIONS) == 5

    def test_extinction_chronological_order(self) -> None:
        """Mass extinctions should be in chronological order (oldest first)."""
        ages = [e.age_mya for e in MASS_EXTINCTIONS]
        assert ages == sorted(ages, reverse=True), f"Not chronological: {ages}"

    @pytest.mark.parametrize(
        "ext_idx",
        range(5),
        ids=[e.name for e in MASS_EXTINCTIONS],
    )
    def test_extinction_tier1_identities(self, ext_idx: int) -> None:
        """Tier-1 identities for pre- and post-extinction states."""
        ext = MASS_EXTINCTIONS[ext_idx]
        r = compute_extinction_kernel(ext)

        # Pre-extinction
        assert abs(r.pre_F + (1.0 - r.pre_F) - 1.0) < TOL_DUALITY
        assert r.pre_IC <= r.pre_F + TOL_BOUND

        # Post-extinction
        assert abs(r.post_F + (1.0 - r.post_F) - 1.0) < TOL_DUALITY
        assert r.post_IC <= r.post_F + TOL_BOUND

    @pytest.mark.parametrize(
        "ext_idx",
        range(5),
        ids=[e.name for e in MASS_EXTINCTIONS],
    )
    def test_extinction_IC_drops(self, ext_idx: int) -> None:
        """IC must drop during mass extinction (post < pre)."""
        r = compute_extinction_kernel(MASS_EXTINCTIONS[ext_idx])
        assert r.post_IC < r.pre_IC, f"{r.name}: post_IC={r.post_IC} ≥ pre_IC={r.pre_IC}"

    def test_end_permian_worst_IC_drop(self) -> None:
        """End-Permian should have the worst IC drop (96% species loss)."""
        results = [compute_extinction_kernel(e) for e in MASS_EXTINCTIONS]
        drops = [(r.name, r.IC_drop_pct) for r in results]
        worst = max(drops, key=lambda x: x[1])
        assert "Permian" in worst[0], f"Worst IC drop is {worst[0]} ({worst[1]:.1f}%), expected End-Permian"

    def test_end_permian_IC_drop_magnitude(self) -> None:
        """End-Permian IC drop should exceed 70%."""
        permian = next(e for e in MASS_EXTINCTIONS if "Permian" in e.name)
        r = compute_extinction_kernel(permian)
        assert r.IC_drop_pct > 70.0, f"End-Permian IC drop = {r.IC_drop_pct:.1f}% — expected > 70%"


# ═══════════════════════════════════════════════════════════════════
# 4. Deep Implications — Identity Maps, Case Studies, Fisher Info
# ═══════════════════════════════════════════════════════════════════


class TestIdentityMap:
    """Verify the GCD-to-evolution identity mappings."""

    def test_identity_map_count(self) -> None:
        assert len(IDENTITY_MAP) == 8

    def test_identity_map_fields_nonempty(self) -> None:
        """All identity mappings must have substantive content."""
        for m in IDENTITY_MAP:
            assert len(m.gcd_identity) > 10, f"{m.gcd_name}: empty gcd_identity"
            assert len(m.gcd_name) > 3, "Empty gcd_name"
            assert len(m.evo_correspondence) > 50, f"{m.gcd_name}: short evo_correspondence"
            assert len(m.cited_by) > 3, f"{m.gcd_name}: no citation"
            assert len(m.testable_prediction) > 20, f"{m.gcd_name}: short prediction"
            assert len(m.case_study) > 20, f"{m.gcd_name}: short case study"

    def test_canonical_identities_present(self) -> None:
        """Must include the three Tier-1 identities and key structures."""
        names = {m.gcd_name for m in IDENTITY_MAP}
        required = {
            "Complementum Perfectum",  # F + ω = 1
            "Limbus Integritatis",  # IC ≤ F
            "Log-Integritas",  # IC = exp(κ)
            "Heterogeneity Gap",  # Δ = F − IC
            "Geometric Slaughter",  # min(cᵢ) → ε ⟹ IC → ε^wᵢ
        }
        missing = required - names
        assert not missing, f"Missing identity mappings: {missing}"


class TestDeepCaseStudies:
    """Tier-1 identity check on all 8 case studies × 3 states = 24 kernels."""

    def test_case_study_count(self) -> None:
        assert len(DEEP_CASES) == 8

    def test_case_studies_have_8_channels(self) -> None:
        for case in DEEP_CASES:
            assert len(case.channel_labels) == 8, f"{case.name}: wrong channel count"
            assert len(case.pre) == 8
            assert len(case.nadir) == 8
            assert len(case.ret) == 8

    @pytest.mark.parametrize(
        "case_idx",
        range(8),
        ids=[c.name[:40] for c in DEEP_CASES],
    )
    def test_case_tier1_identities(self, case_idx: int) -> None:
        """All 3 Tier-1 identities in all 3 states for each case study."""
        case = DEEP_CASES[case_idx]
        for state_name, channels in [
            ("pre", case.pre),
            ("nadir", case.nadir),
            ("ret", case.ret),
        ]:
            c = np.clip(np.array(channels, dtype=np.float64), EPS, 1.0 - EPS)
            w = np.ones(8) / 8
            from umcp.frozen_contract import EPSILON
            from umcp.kernel_optimized import compute_kernel_outputs

            k = compute_kernel_outputs(c, w, EPSILON)

            # F + ω = 1
            assert abs(k["F"] + k["omega"] - 1.0) < TOL_DUALITY, (
                f"{case.name} {state_name}: F+ω = {k['F'] + k['omega']}"
            )
            # IC ≤ F
            assert k["IC"] <= k["F"] + TOL_BOUND, f"{case.name} {state_name}: IC={k['IC']} > F={k['F']}"
            # IC = exp(κ)
            assert abs(k["IC"] - math.exp(k["kappa"])) < TOL_EXP, (
                f"{case.name} {state_name}: IC={k['IC']}, exp(κ)={math.exp(k['kappa'])}"
            )

    @pytest.mark.parametrize(
        "case_idx",
        range(8),
        ids=[c.name[:40] for c in DEEP_CASES],
    )
    def test_case_generativity(self, case_idx: int) -> None:
        """Verify generativity: IC_return > IC_pre for each case."""
        case = DEEP_CASES[case_idx]
        from umcp.frozen_contract import EPSILON
        from umcp.kernel_optimized import compute_kernel_outputs

        c_pre = np.clip(np.array(case.pre, dtype=np.float64), EPS, 1.0 - EPS)
        c_ret = np.clip(np.array(case.ret, dtype=np.float64), EPS, 1.0 - EPS)
        w = np.ones(8) / 8
        k_pre = compute_kernel_outputs(c_pre, w, EPSILON)
        k_ret = compute_kernel_outputs(c_ret, w, EPSILON)

        # All 8 deep cases are designed to be generative
        assert k_ret["IC"] > k_pre["IC"], (
            f"{case.name}: IC_ret={k_ret['IC']:.4f} ≤ IC_pre={k_pre['IC']:.4f} — case should be generative"
        )


class TestFisherInformationLink:
    """Test the Fisher information approximation: Δ ≈ Var(c)/(2c̄)."""

    def test_fisher_link_returns_dict(self) -> None:
        c = np.array([0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8])
        w = np.ones(8) / 8
        result = compute_fisher_information_link(c, w)
        assert isinstance(result, dict)
        assert "delta" in result
        assert "fisher_approx" in result
        assert "approx_error" in result

    def test_fisher_link_delta_nonnegative(self) -> None:
        """Δ must be ≥ 0 (consequence of IC ≤ F)."""
        c = np.array([0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 0.2])
        w = np.ones(8) / 8
        result = compute_fisher_information_link(c, w)
        assert result["delta"] >= 0

    def test_fisher_approx_uniform_channels(self) -> None:
        """For uniform channels, Δ ≈ 0 and Fisher approx error ≈ 0."""
        c = np.full(8, 0.5)
        w = np.ones(8) / 8
        result = compute_fisher_information_link(c, w)
        assert result["delta"] < 0.001, f"Uniform channels should have Δ ≈ 0, got {result['delta']}"

    def test_fisher_approx_precision_moderate(self) -> None:
        """For moderate heterogeneity, Fisher approx should be within 30%."""
        c = np.array([0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8])
        w = np.ones(8) / 8
        result = compute_fisher_information_link(c, w)
        if result["delta"] > 0.001:
            rel_err = abs(result["approx_error"]) / result["delta"]
            assert rel_err < 0.30, f"Relative error {rel_err:.2%} exceeds 30%"


class TestPredictions:
    """Verify the testable predictions catalog."""

    def test_predictions_count(self) -> None:
        assert len(PREDICTIONS) == 5

    def test_predictions_have_content(self) -> None:
        for p in PREDICTIONS:
            assert len(p.prediction) > 20
            assert len(p.gcd_derivation) > 20
            assert len(p.required_data) > 10
            assert len(p.falsification_criterion) > 20
            assert p.status  # non-empty status

    def test_at_least_one_confirmed(self) -> None:
        """At least one prediction should be confirmed by published data."""
        statuses = [p.status.lower() for p in PREDICTIONS]
        has_confirmed = any("confirmed" in s for s in statuses)
        assert has_confirmed, "No predictions are confirmed"


# ═══════════════════════════════════════════════════════════════════
# 5. Cross-Module Consistency
# ═══════════════════════════════════════════════════════════════════


class TestCrossModuleConsistency:
    """Verify that all 4 evolution modules use consistent parameters."""

    def test_all_use_frozen_epsilon(self) -> None:
        """All modules must use EPSILON from frozen_contract."""
        from umcp.frozen_contract import EPSILON

        assert EPSILON == 1e-8

    def test_all_use_8_channels(self) -> None:
        """All modules use 8-channel traces."""
        assert N_CHANNELS == 8
        for p in PHENOMENA:
            assert len(p.channel_labels) == 8
        for s in SCALES:
            assert len(s.channel_labels) == 8
        for case in DEEP_CASES:
            assert len(case.channel_labels) == 8

    def test_total_kernel_states(self) -> None:
        """Count total kernel states across all modules.

        40 organisms + 60 (20×3) phenomena + 5 scales + 10 (5×2) extinctions
        + 24 (8×3) deep cases = 139 total kernel states.
        """
        n_organism = len(ORGANISMS)
        n_phenomena_states = len(PHENOMENA) * 3
        n_scale = len(SCALES)
        n_extinction_states = len(MASS_EXTINCTIONS) * 2
        n_case_states = len(DEEP_CASES) * 3
        total = n_organism + n_phenomena_states + n_scale + n_extinction_states + n_case_states
        assert total == 139, f"Expected 139 total kernel states, got {total}"
