"""Tests for particle_matter_map.py — Unified Cross-Scale Kernel Analysis.

Verifies:
  - All 140 entities at 6 scales compute correctly
  - Tier-1 identities hold universally (F+ω=1, IC≤F, IC=exp(κ))
  - 8 structural theorems (T-PM-1 through T-PM-8)
  - Phase boundary transitions show correct channel flow
  - Scale summaries are internally consistent
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.standard_model.particle_matter_map import (
    EPSILON,
    MatterEntity,
    MatterMap,
    PhaseBoundary,
    ScaleLevel,
    build_atomic,
    build_bulk,
    build_composite,
    build_fundamental,
    build_matter_map,
    build_molecular,
    build_nuclear,
)

# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def matter_map() -> MatterMap:
    """Build the complete matter map once for all tests."""
    return build_matter_map()


@pytest.fixture(scope="module")
def fundamental() -> list[MatterEntity]:
    return build_fundamental()


@pytest.fixture(scope="module")
def composite() -> list[MatterEntity]:
    return build_composite()


@pytest.fixture(scope="module")
def nuclear() -> list[MatterEntity]:
    return build_nuclear()


@pytest.fixture(scope="module")
def atomic() -> list[MatterEntity]:
    return build_atomic()


@pytest.fixture(scope="module")
def molecular() -> list[MatterEntity]:
    return build_molecular()


@pytest.fixture(scope="module")
def bulk() -> list[MatterEntity]:
    return build_bulk()


# ─── Scale builder tests ──────────────────────────────────────────


class TestScaleBuilders:
    """Tests that each scale builder produces the correct number of entities."""

    def test_fundamental_count(self, fundamental: list[MatterEntity]) -> None:
        assert len(fundamental) == 17

    def test_composite_count(self, composite: list[MatterEntity]) -> None:
        assert len(composite) == 14

    def test_nuclear_count(self, nuclear: list[MatterEntity]) -> None:
        assert len(nuclear) == 30

    def test_atomic_count(self, atomic: list[MatterEntity]) -> None:
        assert len(atomic) == 43

    def test_molecular_count(self, molecular: list[MatterEntity]) -> None:
        assert len(molecular) == 20

    def test_bulk_count(self, bulk: list[MatterEntity]) -> None:
        assert len(bulk) == 16

    def test_total_entities(self, matter_map: MatterMap) -> None:
        assert len(matter_map.entities) == 140


# ─── Tier-1 identity tests ───────────────────────────────────────


class TestTier1Identities:
    """Verify the three structural identities across ALL entities."""

    def test_duality_identity(self, matter_map: MatterMap) -> None:
        """F + ω = 1 for every entity."""
        for e in matter_map.entities:
            assert abs(e.F + e.omega - 1.0) < 1e-10, (
                f"{e.name}: F + ω = {e.F + e.omega}, residual = {e.duality_residual}"
            )

    def test_integrity_bound(self, matter_map: MatterMap) -> None:
        """IC ≤ F for every entity."""
        for e in matter_map.entities:
            assert e.IC <= e.F + 1e-12, f"{e.name}: IC={e.IC} > F={e.F}"

    def test_exp_bridge(self, matter_map: MatterMap) -> None:
        """IC = exp(κ) for every entity (tolerance matches 8-place rounding)."""
        for e in matter_map.entities:
            expected_ic = math.exp(e.kappa)
            assert abs(e.IC - expected_ic) < 1e-7, f"{e.name}: IC={e.IC}, exp(κ)={expected_ic}"

    def test_zero_tier1_violations(self, matter_map: MatterMap) -> None:
        """No Tier-1 violations at any scale."""
        assert matter_map.tier1_total_violations == 0


class TestTier1PerScale:
    """Verify Tier-1 identities hold per scale (parametrized)."""

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_duality_per_scale(self, matter_map: MatterMap, scale: str) -> None:
        entities = [e for e in matter_map.entities if e.scale == scale]
        for e in entities:
            assert abs(e.F + e.omega - 1.0) < 1e-10

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_integrity_bound_per_scale(self, matter_map: MatterMap, scale: str) -> None:
        entities = [e for e in matter_map.entities if e.scale == scale]
        for e in entities:
            assert e.IC <= e.F + 1e-12

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_exp_bridge_per_scale(self, matter_map: MatterMap, scale: str) -> None:
        entities = [e for e in matter_map.entities if e.scale == scale]
        for e in entities:
            assert abs(e.IC - math.exp(e.kappa)) < 1e-7


# ─── Theorem tests ───────────────────────────────────────────────


class TestTheoremTPM1:
    """T-PM-1: Confinement Cliff — IC drops >90% at quark→hadron boundary."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-1"]["passed"]

    def test_ic_drop_exceeds_90(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-1"]["IC_drop_pct"] > 90

    def test_quarks_above_hadrons(self, matter_map: MatterMap) -> None:
        t = matter_map.theorem_results["T-PM-1"]
        assert t["mean_IC_quarks"] > t["mean_IC_hadrons"]

    def test_hadron_ic_below_threshold(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-1"]["max_hadron_IC"] < 0.1


class TestTheoremTPM2:
    """T-PM-2: Nuclear Restoration — IC recovers in nuclear regime."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-2"]["passed"]

    def test_recovery_ratio(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-2"]["recovery_ratio"] > 2.0

    def test_nuclear_ic_above_hadron(self, matter_map: MatterMap) -> None:
        t = matter_map.theorem_results["T-PM-2"]
        assert t["mean_IC_nuclear"] > t["mean_IC_hadrons"]


class TestTheoremTPM3:
    """T-PM-3: Shell Amplification — Doubly-magic nuclides have high IC/F."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-3"]["passed"]

    def test_amplification_factor(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-3"]["amplification_factor"] > 1.0


class TestTheoremTPM4:
    """T-PM-4: Periodic Modulation — IC follows block structure."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-4"]["passed"]

    def test_d_block_highest(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-4"]["d_block_highest"]

    def test_noble_gases_low_ic(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-4"]["noble_gases_low_IC"]


class TestTheoremTPM5:
    """T-PM-5: Molecular Emergence — Bond channels produce gap reduction."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-5"]["passed"]

    def test_gap_reduction(self, matter_map: MatterMap) -> None:
        t = matter_map.theorem_results["T-PM-5"]
        assert t["mean_gap_molecules"] < t["mean_gap_atoms"]

    def test_symmetry_polarity_competition(self, matter_map: MatterMap) -> None:
        """High-symmetry molecules have LOWER IC than low-symmetry ones
        due to zero dipole killing the polarity channel."""
        t = matter_map.theorem_results["T-PM-5"]
        assert t["symmetry_polarity_competition"]


class TestTheoremTPM6:
    """T-PM-6: Bulk Averaging — Metals converge in IC/F space."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-6"]["passed"]


class TestTheoremTPM7:
    """T-PM-7: Scale Non-Monotonicity — IC trajectory has local minima."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-7"]["passed"]

    def test_confinement_drop(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-7"]["confinement_drop"]

    def test_at_least_one_local_minimum(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-7"]["n_local_minima"] >= 1

    def test_trajectory_covers_all_scales(self, matter_map: MatterMap) -> None:
        assert len(matter_map.theorem_results["T-PM-7"]["IC_trajectory"]) == 6


class TestTheoremTPM8:
    """T-PM-8: Tier-1 Universal — Zero violations across all 140 entities."""

    def test_theorem_proven(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-8"]["passed"]

    def test_zero_duality_violations(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-8"]["duality_violations"] == 0

    def test_zero_bound_violations(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-8"]["integrity_bound_violations"] == 0

    def test_zero_exp_violations(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-8"]["exp_bridge_violations"] == 0

    def test_all_entities_counted(self, matter_map: MatterMap) -> None:
        assert matter_map.theorem_results["T-PM-8"]["n_entities"] == 140


# ─── Phase boundary transition tests ─────────────────────────────


class TestPhaseBoundaries:
    """Verify phase boundary transitions."""

    def test_five_transitions(self, matter_map: MatterMap) -> None:
        assert len(matter_map.transitions) == 5

    def test_confinement_ic_ratio(self, matter_map: MatterMap) -> None:
        """Confinement should show massive IC drop (ratio << 1)."""
        confinement = matter_map.transitions[0]
        assert confinement.boundary == PhaseBoundary.CONFINEMENT
        assert confinement.IC_ratio < 0.05

    def test_nuclear_binding_recovery(self, matter_map: MatterMap) -> None:
        """Nuclear binding should show IC recovery (ratio >> 1)."""
        nuclear = matter_map.transitions[1]
        assert nuclear.boundary == PhaseBoundary.NUCLEAR_BINDING
        assert nuclear.IC_ratio > 10

    def test_each_boundary_has_die_survive_emerge(self, matter_map: MatterMap) -> None:
        """Every boundary must have at least one channel in each category."""
        for t in matter_map.transitions:
            assert len(t.channels_that_die) > 0, f"{t.boundary}: no channels die"
            assert len(t.channels_that_survive) > 0, f"{t.boundary}: no channels survive"
            assert len(t.channels_that_emerge) > 0, f"{t.boundary}: no channels emerge"


# ─── Scale summary tests ─────────────────────────────────────────


class TestScaleSummaries:
    """Verify scale summaries are consistent."""

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_summary_exists(self, matter_map: MatterMap, scale: str) -> None:
        assert scale in matter_map.summaries

    def test_nuclear_highest_mean_ic(self, matter_map: MatterMap) -> None:
        """Nuclear scale should have the highest mean IC."""
        ics = {s: mm.mean_IC for s, mm in matter_map.summaries.items()}
        best = max(ics, key=lambda k: ics[k])
        assert best == ScaleLevel.NUCLEAR

    def test_composite_lowest_mean_ic(self, matter_map: MatterMap) -> None:
        """Composite scale should have the lowest mean IC (confinement)."""
        ics = {s: mm.mean_IC for s, mm in matter_map.summaries.items()}
        worst = min(ics, key=lambda k: ics[k])
        assert worst == ScaleLevel.COMPOSITE

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_mean_f_plus_omega_equals_one(self, matter_map: MatterMap, scale: str) -> None:
        s = matter_map.summaries[scale]
        assert abs(s.mean_F + s.mean_omega - 1.0) < 0.01  # mean-level consistency

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_mean_ic_le_mean_f(self, matter_map: MatterMap, scale: str) -> None:
        s = matter_map.summaries[scale]
        assert s.mean_IC <= s.mean_F + 0.01  # mean-level consistency


# ─── Entity level tests ──────────────────────────────────────────


class TestEntityProperties:
    """Verify individual entity properties."""

    def test_all_entities_have_channels(self, matter_map: MatterMap) -> None:
        for e in matter_map.entities:
            assert e.n_channels > 0
            assert len(e.channel_names) == e.n_channels
            assert len(e.trace) == e.n_channels

    def test_all_traces_in_range(self, matter_map: MatterMap) -> None:
        for e in matter_map.entities:
            for i, c in enumerate(e.trace):
                assert EPSILON <= c <= 1.0 - EPSILON + 1e-12, f"{e.name} channel {e.channel_names[i]}: {c}"

    def test_all_entities_have_regime(self, matter_map: MatterMap) -> None:
        for e in matter_map.entities:
            assert e.regime in ("Stable", "Watch", "Collapse")

    def test_proton_is_composite(self, matter_map: MatterMap) -> None:
        proton = [e for e in matter_map.entities if "proton" in e.name.lower()]
        assert len(proton) == 1
        assert proton[0].scale == ScaleLevel.COMPOSITE

    def test_iron_is_atomic(self, matter_map: MatterMap) -> None:
        iron = [e for e in matter_map.entities if e.name.startswith("Fe")]
        assert len(iron) == 1
        assert iron[0].scale == ScaleLevel.ATOMIC

    def test_water_is_molecular(self, matter_map: MatterMap) -> None:
        water = [e for e in matter_map.entities if "Water" in e.name and e.scale == ScaleLevel.MOLECULAR]
        assert len(water) == 1

    def test_copper_is_bulk(self, matter_map: MatterMap) -> None:
        cu = [e for e in matter_map.entities if "Copper" in e.name]
        assert len(cu) == 1
        assert cu[0].scale == ScaleLevel.BULK


# ─── Physical consistency tests ───────────────────────────────────


class TestPhysicalConsistency:
    """Verify physically expected relationships."""

    def test_quarks_have_color(self, fundamental: list[MatterEntity]) -> None:
        quarks = [e for e in fundamental if e.category == "Quark"]
        for q in quarks:
            # color_dof channel (index 3) should be non-trivial
            assert q.trace[3] > 0.5, f"{q.name}: color channel = {q.trace[3]}"

    def test_leptons_colorless(self, fundamental: list[MatterEntity]) -> None:
        leptons = [e for e in fundamental if e.category == "Lepton"]
        for lep in leptons:
            # color_dof channel should be at floor (log2(2)/log2(9) ≈ 0.315)
            assert lep.trace[3] < 0.5, f"{lep.name}: color channel = {lep.trace[3]}"

    def test_pions_lighter_than_nucleons(self, composite: list[MatterEntity]) -> None:
        pions = [e for e in composite if "pion" in e.name.lower()]
        nucleons = [e for e in composite if "proton" in e.name.lower() or "neutron" in e.name.lower()]
        for p in pions:
            for n in nucleons:
                assert p.mass_GeV < n.mass_GeV

    def test_iron_peak_binding(self, nuclear: list[MatterEntity]) -> None:
        """Iron-56 / Nickel-62 region should have among the highest BE/A channels."""
        fe56 = next(e for e in nuclear if "Iron-56" in e.name)
        ni62 = next(e for e in nuclear if "Nickel-62" in e.name)
        h1 = next(e for e in nuclear if "Hydrogen-1" in e.name)
        # BE/A channel (index 2) should be high for iron peak and low for H-1
        assert fe56.trace[2] > h1.trace[2]
        assert ni62.trace[2] > h1.trace[2]

    def test_helium4_doubly_magic(self, nuclear: list[MatterEntity]) -> None:
        he4 = next(e for e in nuclear if "Helium-4" in e.name)
        # magic_Z_prox (index 3) and magic_N_prox (index 4) should be 1.0
        assert he4.trace[3] > 0.99
        assert he4.trace[4] > 0.99

    def test_lead208_doubly_magic(self, nuclear: list[MatterEntity]) -> None:
        pb208 = next(e for e in nuclear if "Lead-208" in e.name)
        assert pb208.trace[3] > 0.99  # Z=82 is magic
        assert pb208.trace[4] > 0.99  # N=126 is magic

    def test_noble_gases_zero_en(self, atomic: list[MatterEntity]) -> None:
        """Noble gases have EN ≈ 0 (no electronegativity)."""
        nobles = [e for e in atomic if e.name.split()[0] in ("He", "Ne", "Ar", "Kr", "Xe")]
        for ng in nobles:
            # EN channel (index 6) should be near ε
            assert ng.trace[6] < 0.05, f"{ng.name}: EN channel = {ng.trace[6]}"

    def test_diamond_highest_thermal_conductivity(self, bulk: list[MatterEntity]) -> None:
        diamond = next(e for e in bulk if "Diamond" in e.name)
        others = [e for e in bulk if "Diamond" not in e.name]
        # k_thermal channel (index 1)
        for other in others:
            assert diamond.trace[1] >= other.trace[1]


# ─── Heterogeneity gap tests ─────────────────────────────────────


class TestHeterogeneityGap:
    """Test heterogeneity gap Δ = F − IC properties."""

    def test_gap_always_non_negative(self, matter_map: MatterMap) -> None:
        for e in matter_map.entities:
            assert e.gap >= -1e-12, f"{e.name}: gap = {e.gap}"

    def test_composite_largest_mean_gap(self, matter_map: MatterMap) -> None:
        """Composites should have the largest gap (confinement effect)."""
        gaps = {s: mm.mean_gap for s, mm in matter_map.summaries.items()}
        worst = max(gaps, key=lambda k: gaps[k])
        assert worst == ScaleLevel.COMPOSITE

    @pytest.mark.parametrize("scale", ScaleLevel.ALL)
    def test_gap_bounded_by_f(self, matter_map: MatterMap, scale: str) -> None:
        entities = [e for e in matter_map.entities if e.scale == scale]
        for e in entities:
            assert e.gap <= e.F + 1e-12


# ─── to_dict serialization ───────────────────────────────────────


class TestSerialization:
    """Verify entities can serialize."""

    def test_to_dict(self, matter_map: MatterMap) -> None:
        for e in matter_map.entities[:5]:
            d = e.to_dict()
            assert isinstance(d, dict)
            assert "F" in d
            assert "omega" in d
            assert "IC" in d
            assert "kappa" in d
            assert "regime" in d
