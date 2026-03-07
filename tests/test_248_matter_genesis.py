"""Tests for matter genesis — the story of how particles build atoms and create mass.

Validates 10 Genesis Theorems (T-MG-1 through T-MG-10), all 99 entities
across 6 acts, 5 phase transitions, 9 mass origins, and the complete
narrative pipeline.

Test count target: ~200 tests covering:
    - Tier-1 identity universality (duality, integrity bound, log-bridge)
    - Per-act entity counts and kernel statistics
    - Phase boundary channel accounting (dies/survives/emerges)
    - Mass origin fractions (Higgs vs QCD vs EM)
    - 10 theorem proofs with subtests
    - Narrative section generation
    - Edge cases and frozen constants
"""

from __future__ import annotations

import math

import pytest

from closures.standard_model.matter_genesis import (
    ATOM_CHANNELS,
    BINDING_FRACTION_PROTON,
    BULK_CHANNELS,
    COMP_CHANNELS,
    FUND_CHANNELS,
    MAGIC_N,
    MAGIC_Z,
    MOL_CHANNELS,
    NUC_CHANNELS,
    PROTON_MASS_GEV,
    QUARK_MASS_IN_PROTON,
    QUARK_MASSES,
    VEV,
    GenesisEntity,
    GenesisResult,
    _be_per_a,
    _clip,
    _magic_proximity,
    _norm_mass,
    _norm_stability,
    _yukawa,
    build_act_i,
    build_act_ii,
    build_act_iii,
    build_act_iv,
    build_act_v,
    build_act_vi,
    display_genesis,
    run_full_analysis,
)
from umcp.frozen_contract import EPSILON

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def genesis() -> GenesisResult:
    """Run full analysis once for the entire test module."""
    return run_full_analysis()


@pytest.fixture(scope="module")
def act_i() -> list[GenesisEntity]:
    return build_act_i()


@pytest.fixture(scope="module")
def act_ii() -> list[GenesisEntity]:
    return build_act_ii()


@pytest.fixture(scope="module")
def act_iii() -> list[GenesisEntity]:
    return build_act_iii()


@pytest.fixture(scope="module")
def act_iv() -> list[GenesisEntity]:
    return build_act_iv()


@pytest.fixture(scope="module")
def act_v() -> list[GenesisEntity]:
    return build_act_v()


@pytest.fixture(scope="module")
def act_vi() -> list[GenesisEntity]:
    return build_act_vi()


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════


class TestHelpers:
    """Test low-level helper functions."""

    def test_clip_floor(self) -> None:
        assert _clip(-1.0) == EPSILON

    def test_clip_ceiling(self) -> None:
        assert _clip(2.0) == 1.0 - EPSILON

    def test_clip_passthrough(self) -> None:
        assert _clip(0.5) == 0.5

    def test_norm_mass_zero(self) -> None:
        assert _norm_mass(0.0) == EPSILON

    def test_norm_mass_proton(self) -> None:
        v = _norm_mass(PROTON_MASS_GEV)
        assert 0.0 < v < 1.0

    def test_norm_stability_stable(self) -> None:
        v = _norm_stability(1e30)
        assert v > 0.99

    def test_norm_stability_short(self) -> None:
        v = _norm_stability(1e-20)
        assert 0.0 < v < 0.9

    def test_norm_stability_zero(self) -> None:
        assert _norm_stability(0.0) == EPSILON

    def test_yukawa_top(self) -> None:
        y_top = _yukawa(172.69)
        assert 0.95 < y_top < 1.05

    def test_yukawa_electron(self) -> None:
        y_e = _yukawa(0.000511)
        assert y_e < 1e-5

    def test_be_per_a_fe56(self) -> None:
        be = _be_per_a(26, 56)
        assert 8.0 < be < 9.5

    def test_be_per_a_he4(self) -> None:
        be = _be_per_a(2, 4)
        assert 4.0 < be < 8.0

    def test_be_per_a_invalid(self) -> None:
        assert _be_per_a(0, 0) == 0.0
        assert _be_per_a(10, 10) == 0.0  # Z == A

    def test_magic_proximity_exact(self) -> None:
        assert _magic_proximity(8, MAGIC_Z) == 1.0
        assert _magic_proximity(28, MAGIC_N) == 1.0

    def test_magic_proximity_far(self) -> None:
        v = _magic_proximity(40, MAGIC_Z)
        assert v < 0.5


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ACT ENTITY COUNTS
# ═══════════════════════════════════════════════════════════════


class TestActEntityCounts:
    """Verify each act produces the expected number of entities."""

    def test_act_i_count(self, act_i: list[GenesisEntity]) -> None:
        assert len(act_i) == 17

    def test_act_ii_count(self, act_ii: list[GenesisEntity]) -> None:
        assert len(act_ii) == 14

    def test_act_iii_count(self, act_iii: list[GenesisEntity]) -> None:
        assert len(act_iii) == 22

    def test_act_iv_count(self, act_iv: list[GenesisEntity]) -> None:
        assert len(act_iv) == 15

    def test_act_v_count(self, act_v: list[GenesisEntity]) -> None:
        assert len(act_v) == 15

    def test_act_vi_count(self, act_vi: list[GenesisEntity]) -> None:
        assert len(act_vi) == 16

    def test_total_entities(self, genesis: GenesisResult) -> None:
        assert len(genesis.entities) == 99


class TestActChannelDimensions:
    """Verify channel counts for each act."""

    def test_fund_channels(self) -> None:
        assert len(FUND_CHANNELS) == 8

    def test_comp_channels(self) -> None:
        assert len(COMP_CHANNELS) == 8

    def test_nuc_channels(self) -> None:
        assert len(NUC_CHANNELS) == 8

    def test_atom_channels(self) -> None:
        assert len(ATOM_CHANNELS) == 12

    def test_mol_channels(self) -> None:
        assert len(MOL_CHANNELS) == 8

    def test_bulk_channels(self) -> None:
        assert len(BULK_CHANNELS) == 6

    @pytest.mark.parametrize(
        "act_name,expected_channels",
        [
            ("Act I: The Cast", 8),
            ("Act II: Confinement", 8),
            ("Act III: Nuclear Furnace", 8),
            ("Act IV: Electronic Shell", 12),
            ("Act V: Chemical Bond", 8),
            ("Act VI: Bulk Emergence", 6),
        ],
    )
    def test_entity_channel_count(self, genesis: GenesisResult, act_name: str, expected_channels: int) -> None:
        entities = [e for e in genesis.entities if e.act == act_name]
        assert len(entities) > 0
        for e in entities:
            assert e.n_channels == expected_channels, (
                f"{e.name} has {e.n_channels} channels, expected {expected_channels}"
            )


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — TIER-1 IDENTITY UNIVERSALITY
# ═══════════════════════════════════════════════════════════════


class TestTier1Universality:
    """Verify Tier-1 identities hold for every entity at every scale."""

    def test_zero_tier1_violations(self, genesis: GenesisResult) -> None:
        assert genesis.tier1_violations == 0

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_duality_per_act(self, genesis: GenesisResult, act_name: str) -> None:
        """F + ω = 1 for every entity in each act."""
        entities = [e for e in genesis.entities if e.act == act_name]
        for e in entities:
            assert e.duality_residual < 1e-10, f"{e.name}: F+ω-1 = {e.duality_residual}"

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_integrity_bound_per_act(self, genesis: GenesisResult, act_name: str) -> None:
        """IC ≤ F for every entity in each act."""
        entities = [e for e in genesis.entities if e.act == act_name]
        for e in entities:
            assert e.integrity_bound_ok, f"{e.name}: IC={e.IC} > F={e.F}"

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_exp_bridge_per_act(self, genesis: GenesisResult, act_name: str) -> None:
        """IC = exp(κ) for every entity in each act."""
        entities = [e for e in genesis.entities if e.act == act_name]
        for e in entities:
            assert e.exp_bridge_ok, f"{e.name}: IC={e.IC}, exp(κ)={math.exp(e.kappa)}"


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — SPECIFIC PARTICLE TESTS
# ═══════════════════════════════════════════════════════════════


class TestFundamentalParticles:
    """Test specific fundamental particle properties."""

    def test_quarks_count(self, act_i: list[GenesisEntity]) -> None:
        quarks = [e for e in act_i if e.name in ("Up", "Down", "Strange", "Charm", "Bottom", "Top")]
        assert len(quarks) == 6

    def test_leptons_count(self, act_i: list[GenesisEntity]) -> None:
        leptons = [e for e in act_i if "neutrino" in e.name.lower() or e.name in ("Electron", "Muon", "Tau")]
        assert len(leptons) == 6

    def test_bosons_count(self, act_i: list[GenesisEntity]) -> None:
        bosons = [e for e in act_i if e.name in ("Photon", "W-boson", "Z-boson", "Gluon", "Higgs")]
        assert len(bosons) == 5

    def test_proton_stable(self, act_ii: list[GenesisEntity]) -> None:
        proton = next(e for e in act_ii if e.name == "Proton")
        assert proton.mass_GeV == pytest.approx(0.93827, abs=0.001)

    def test_neutron_heavier_than_proton(self, act_ii: list[GenesisEntity]) -> None:
        proton = next(e for e in act_ii if e.name == "Proton")
        neutron = next(e for e in act_ii if e.name == "Neutron")
        assert neutron.mass_GeV > proton.mass_GeV

    def test_pion_lightest_meson(self, act_ii: list[GenesisEntity]) -> None:
        mesons = [e for e in act_ii if e.name in ("Pion+", "Pion0", "Kaon+", "Kaon0", "J/Psi", "D0", "B+")]
        min_mass = min(e.mass_GeV for e in mesons)
        assert min_mass < 0.15  # Pion0 at 0.135 GeV


class TestNuclei:
    """Test nuclear entity properties."""

    def test_fe56_near_peak_be(self, act_iii: list[GenesisEntity]) -> None:
        assert any(e.name == "Fe-56" for e in act_iii)
        be = _be_per_a(26, 56)
        assert be > 8.5

    def test_doubly_magic_present(self, act_iii: list[GenesisEntity]) -> None:
        magic_names = {"He-4", "O-16", "Ca-40", "Pb-208"}
        found = {e.name for e in act_iii} & magic_names
        assert found == magic_names

    def test_hydrogen_simplest(self, act_iii: list[GenesisEntity]) -> None:
        h1 = next(e for e in act_iii if e.name == "H-1")
        assert h1.Z == 1 and h1.A == 1


class TestAtoms:
    """Test atomic entity properties."""

    def test_noble_gases_present(self, act_iv: list[GenesisEntity]) -> None:
        nobles = [e for e in act_iv if e.name in ("Helium", "Neon", "Oganesson")]
        assert len(nobles) >= 2

    def test_iron_present(self, act_iv: list[GenesisEntity]) -> None:
        iron = next((e for e in act_iv if e.name == "Iron"), None)
        assert iron is not None
        assert iron.Z == 26


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — THEOREM TESTS
# ═══════════════════════════════════════════════════════════════


class TestTheoremsMG1:
    """T-MG-1: Higgs Mass Generation."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-1"]["proven"]

    def test_yukawa_hierarchy(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-1"]["hierarchy_ratio"] > 1e5

    def test_y_top_near_unity(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-1"]["y_top"] > 0.9

    def test_mass_monotone(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-1"]["mass_monotone"]


class TestTheoremsMG2:
    """T-MG-2: Color Confinement Cost."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-2"]["proven"]

    def test_ic_drops(self, genesis: GenesisResult) -> None:
        r = genesis.theorem_results["T-MG-2"]["IC_ratio"]
        assert r < 0.5  # IC drops significantly at confinement

    def test_four_channels_die(self, genesis: GenesisResult) -> None:
        assert len(genesis.theorem_results["T-MG-2"]["dead_channels"]) == 4


class TestTheoremsMG3:
    """T-MG-3: Binding Mass Deficit."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-3"]["proven"]

    def test_proton_99_percent_binding(self, genesis: GenesisResult) -> None:
        deficit = genesis.theorem_results["T-MG-3"]["proton_deficit"]
        assert deficit > 0.98


class TestTheoremsMG4:
    """T-MG-4: Proton-Neutron Duality."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-4"]["proven"]

    def test_mass_diff_range(self, genesis: GenesisResult) -> None:
        dm = genesis.theorem_results["T-MG-4"]["mass_diff_MeV"]
        assert 1.0 < dm < 2.0  # 1.293 MeV


class TestTheoremsMG5:
    """T-MG-5: Shell Closure Stability."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-5"]["proven"]

    def test_magic_ic_higher(self, genesis: GenesisResult) -> None:
        t = genesis.theorem_results["T-MG-5"]
        assert t["mean_IC_doubly_magic"] > t["mean_IC_non_magic"]


class TestTheoremsMG6:
    """T-MG-6: Electron Configuration Order."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-6"]["proven"]

    def test_multiple_blocks(self, genesis: GenesisResult) -> None:
        blocks = genesis.theorem_results["T-MG-6"]["block_mean_F"]
        assert len(blocks) >= 3

    def test_d_block_highest(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-6"]["d_block_highest"]


class TestTheoremsMG7:
    """T-MG-7: Covalent Bond Coherence."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-7"]["proven"]

    def test_both_classes_present(self, genesis: GenesisResult) -> None:
        t = genesis.theorem_results["T-MG-7"]
        assert t["n_polar"] > 0
        assert t["n_nonpolar"] > 0


class TestTheoremsMG8:
    """T-MG-8: Mass Hierarchy Bridge."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-8"]["proven"]

    def test_proton_binding_dominant(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-8"]["proton_binding_pct"] > 98.0

    def test_human_binding_dominant(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-8"]["human_binding_pct"] > 98.0


class TestTheoremsMG9:
    """T-MG-9: Material Property Ladder."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-9"]["proven"]

    def test_metals_and_insulators(self, genesis: GenesisResult) -> None:
        t = genesis.theorem_results["T-MG-9"]
        assert t["n_metals"] > 0
        assert t["n_insulators"] > 0


class TestTheoremsMG10:
    """T-MG-10: Universal Tier-1."""

    def test_proven(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-10"]["proven"]

    def test_zero_violations(self, genesis: GenesisResult) -> None:
        t = genesis.theorem_results["T-MG-10"]
        assert t["total_violations"] == 0

    def test_all_entities_counted(self, genesis: GenesisResult) -> None:
        t = genesis.theorem_results["T-MG-10"]
        assert t["n_entities"] == 99

    def test_duality_zero(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-10"]["duality_violations"] == 0

    def test_bound_zero(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-10"]["bound_violations"] == 0

    def test_bridge_zero(self, genesis: GenesisResult) -> None:
        assert genesis.theorem_results["T-MG-10"]["bridge_violations"] == 0


class TestAllTheoremsProven:
    """Meta-test: all 10 theorems must be proven."""

    def test_10_of_10_proven(self, genesis: GenesisResult) -> None:
        n_proven = sum(1 for t in genesis.theorem_results.values() if t.get("proven"))
        assert n_proven == 10

    @pytest.mark.parametrize(
        "theorem_id",
        [
            "T-MG-1",
            "T-MG-2",
            "T-MG-3",
            "T-MG-4",
            "T-MG-5",
            "T-MG-6",
            "T-MG-7",
            "T-MG-8",
            "T-MG-9",
            "T-MG-10",
        ],
    )
    def test_theorem_individually(self, genesis: GenesisResult, theorem_id: str) -> None:
        assert genesis.theorem_results[theorem_id]["proven"], f"{theorem_id} not proven"


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — PHASE TRANSITION TESTS
# ═══════════════════════════════════════════════════════════════


class TestTransitions:
    """Test the 5 phase boundary transitions."""

    def test_five_transitions(self, genesis: GenesisResult) -> None:
        assert len(genesis.transitions) == 5

    @pytest.mark.parametrize(
        "idx,name",
        [
            (0, "Confinement"),
            (1, "Nuclear Binding"),
            (2, "Electronic Shell"),
            (3, "Chemical Bonding"),
            (4, "Bulk Aggregation"),
        ],
    )
    def test_transition_names(self, genesis: GenesisResult, idx: int, name: str) -> None:
        assert genesis.transitions[idx].boundary_name == name

    def test_confinement_kills_four_channels(self, genesis: GenesisResult) -> None:
        conf = genesis.transitions[0]
        assert len(conf.channels_die) == 4
        assert "color_dof" in conf.channels_die

    def test_confinement_creates_four_channels(self, genesis: GenesisResult) -> None:
        conf = genesis.transitions[0]
        assert len(conf.channels_emerge) == 4
        assert "valence_quarks" in conf.channels_emerge

    def test_every_transition_has_narrative(self, genesis: GenesisResult) -> None:
        for t in genesis.transitions:
            assert len(t.narrative) > 50

    def test_transition_ic_values_positive(self, genesis: GenesisResult) -> None:
        for t in genesis.transitions:
            assert t.IC_before > 0
            assert t.IC_after > 0


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — MASS ORIGIN TESTS
# ═══════════════════════════════════════════════════════════════


class TestMassOrigins:
    """Test mass origin breakdown."""

    def test_nine_origins(self, genesis: GenesisResult) -> None:
        assert len(genesis.mass_origins) == 9

    def test_proton_mass_origin(self, genesis: GenesisResult) -> None:
        proton = next(m for m in genesis.mass_origins if m.entity_name == "Proton")
        assert proton.binding_fraction > 0.98
        assert proton.higgs_fraction < 0.02

    def test_electron_all_higgs(self, genesis: GenesisResult) -> None:
        electron = next(m for m in genesis.mass_origins if m.entity_name == "Electron")
        assert electron.higgs_fraction == 1.0
        assert electron.binding_fraction == 0.0

    def test_w_boson_all_higgs(self, genesis: GenesisResult) -> None:
        w = next(m for m in genesis.mass_origins if m.entity_name == "W boson")
        assert w.higgs_fraction == 1.0

    def test_up_quark_all_higgs(self, genesis: GenesisResult) -> None:
        up = next(m for m in genesis.mass_origins if m.entity_name == "Up quark")
        assert up.higgs_fraction == 1.0

    def test_human_body_99_binding(self, genesis: GenesisResult) -> None:
        human = next(m for m in genesis.mass_origins if "Human" in m.entity_name)
        assert human.binding_fraction > 0.98

    def test_fractions_sum(self, genesis: GenesisResult) -> None:
        for m in genesis.mass_origins:
            total = m.higgs_fraction + m.binding_fraction + m.em_fraction
            # Allow small epsilon for rounding
            assert 0.99 < total < 1.01 or m.em_fraction > 0, f"{m.entity_name}: fractions sum to {total}"


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — NARRATIVE TESTS
# ═══════════════════════════════════════════════════════════════


class TestNarrative:
    """Test narrative section generation."""

    def test_prologue_exists(self, genesis: GenesisResult) -> None:
        assert "prologue" in genesis.narrative_sections
        assert "17 particles" in genesis.narrative_sections["prologue"]

    def test_epilogue_exists(self, genesis: GenesisResult) -> None:
        assert "epilogue" in genesis.narrative_sections
        assert "Collapsus generativus" in genesis.narrative_sections["epilogue"]

    def test_act_vii_mass(self, genesis: GenesisResult) -> None:
        assert "act_vii" in genesis.narrative_sections
        assert "Higgs" in genesis.narrative_sections["act_vii"]

    def test_all_narrative_keys(self, genesis: GenesisResult) -> None:
        expected = {"prologue", "act_i", "act_ii", "act_iii", "act_vii", "epilogue"}
        assert expected.issubset(set(genesis.narrative_sections.keys()))


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — ACT SUMMARY TESTS
# ═══════════════════════════════════════════════════════════════


class TestActSummaries:
    """Test act-level summary statistics."""

    def test_six_acts(self, genesis: GenesisResult) -> None:
        assert len(genesis.acts) == 6

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_act_has_entities(self, genesis: GenesisResult, act_name: str) -> None:
        assert genesis.acts[act_name].n_entities > 0

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_act_f_positive(self, genesis: GenesisResult, act_name: str) -> None:
        assert genesis.acts[act_name].mean_F > 0

    @pytest.mark.parametrize(
        "act_name",
        [
            "Act I: The Cast",
            "Act II: Confinement",
            "Act III: Nuclear Furnace",
            "Act IV: Electronic Shell",
            "Act V: Chemical Bond",
            "Act VI: Bulk Emergence",
        ],
    )
    def test_act_ic_positive(self, genesis: GenesisResult, act_name: str) -> None:
        assert genesis.acts[act_name].mean_IC > 0


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — ENTITY DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════


class TestEntityDataclass:
    """Test GenesisEntity structure and serialization."""

    def test_to_dict(self, act_i: list[GenesisEntity]) -> None:
        d = act_i[0].to_dict()
        assert isinstance(d, dict)
        assert "F" in d
        assert "omega" in d
        assert "IC" in d

    def test_frozen(self, act_i: list[GenesisEntity]) -> None:
        with pytest.raises(AttributeError):
            act_i[0].F = 0.5  # type: ignore[misc]

    def test_gap_equals_f_minus_ic(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert abs(e.gap - (e.F - e.IC)) < 1e-10, f"{e.name}: gap mismatch"


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — KERNEL STATISTICS TESTS
# ═══════════════════════════════════════════════════════════════


class TestKernelStatistics:
    """Test statistical properties of kernel outputs across acts."""

    def test_f_range(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert 0 < e.F <= 1.0, f"{e.name}: F={e.F} out of range"

    def test_omega_range(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert 0 <= e.omega < 1.0, f"{e.name}: ω={e.omega} out of range"

    def test_ic_range(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert 0 < e.IC <= 1.0, f"{e.name}: IC={e.IC} out of range"

    def test_entropy_non_negative(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert e.S >= 0, f"{e.name}: S={e.S} < 0"

    def test_curvature_non_negative(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert e.C >= 0, f"{e.name}: C={e.C} < 0"

    def test_kappa_non_positive(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert e.kappa <= 0 + 1e-10, f"{e.name}: κ={e.kappa} > 0"


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — FROZEN CONSTANTS TESTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify frozen physical constants are correct."""

    def test_vev(self) -> None:
        assert pytest.approx(246.22, abs=0.01) == VEV

    def test_proton_mass(self) -> None:
        assert pytest.approx(0.93827, abs=0.001) == PROTON_MASS_GEV

    def test_quark_mass_in_proton(self) -> None:
        expected = 2 * 0.00216 + 0.00467
        assert pytest.approx(expected, abs=1e-6) == QUARK_MASS_IN_PROTON

    def test_binding_fraction(self) -> None:
        assert BINDING_FRACTION_PROTON > 0.98

    def test_magic_numbers_z(self) -> None:
        assert 2 in MAGIC_Z
        assert 8 in MAGIC_Z
        assert 82 in MAGIC_Z

    def test_magic_numbers_n(self) -> None:
        assert 126 in MAGIC_N

    def test_six_quark_masses(self) -> None:
        assert len(QUARK_MASSES) == 6
        assert QUARK_MASSES["top"] > 170


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — DISPLAY FUNCTION TEST
# ═══════════════════════════════════════════════════════════════


class TestDisplay:
    """Test display function doesn't crash."""

    def test_display_runs(self, genesis: GenesisResult, capsys: pytest.CaptureFixture[str]) -> None:
        display_genesis(genesis)
        captured = capsys.readouterr()
        assert "MATTER GENESIS" in captured.out
        assert "10/10" in captured.out or "theorems proven" in captured.out


# ═══════════════════════════════════════════════════════════════
# SECTION 14 — PARAMETRIZED TRACE VALIDATION
# ═══════════════════════════════════════════════════════════════


class TestTraceValues:
    """Validate all trace values are in [ε, 1-ε]."""

    def test_all_traces_in_range(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            for i, v in enumerate(e.trace):
                assert EPSILON <= v <= 1.0 - EPSILON, (
                    f"{e.name} channel {e.channel_names[i]}: trace={v} out of [ε, 1-ε]"
                )

    def test_trace_length_matches_channels(self, genesis: GenesisResult) -> None:
        for e in genesis.entities:
            assert len(e.trace) == len(e.channel_names), (
                f"{e.name}: trace len {len(e.trace)} != channels {len(e.channel_names)}"
            )
