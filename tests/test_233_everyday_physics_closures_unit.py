"""
Tests for everyday_physics closure modules.

Covers 5 files: epistemic_coherence (7 theorems), thermodynamics,
electromagnetism, optics, wave_phenomena.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/everyday_physics/epistemic_coherence.py — 7 theorems
# ═══════════════════════════════════════════════════════════════════


class TestEpistemicCoherence:
    """Unit tests for epistemic_coherence compute functions."""

    def test_compute_established_science(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_system,
        )

        result = compute_epistemic_system("Established Scientific Consensus")
        assert result.F + result.omega == pytest.approx(1.0, abs=1e-5)
        assert result.IC <= result.F + 1e-5
        assert result.regime == "Stable"

    def test_compute_astrology(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_system,
        )

        result = compute_epistemic_system("Astrology")
        assert result.regime in ("Watch", "Tension", "Collapse")
        assert result.dead_channels >= 1

    def test_compute_all_systems(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_all_epistemic_systems,
        )

        results = compute_all_epistemic_systems()
        assert len(results) == 14
        for r in results:
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)

    def test_unknown_system_raises(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_system,
        )

        with pytest.raises(KeyError):
            compute_epistemic_system("Nonexistent System")

    def test_custom_channels(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_from_channels,
        )

        channels = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        result = compute_epistemic_from_channels("Test", "Test", channels)
        assert result.F + result.omega == pytest.approx(1.0, abs=1e-5)

    def test_custom_channels_wrong_length(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_from_channels,
        )

        with pytest.raises(ValueError):
            compute_epistemic_from_channels("Test", "Test", [0.5] * 5)

    @pytest.mark.parametrize(
        "system_name",
        [
            "Established Scientific Consensus",
            "Frontier Science (Pre-Consensus)",
            "Astrology",
            "Homeopathy",
            "Flat Earth Theory",
            "Numerology",
            "Traditional Herbal Medicine",
            "Generic Conspiracy Theory",
            "Technical Analysis (Finance)",
            "Newtonian Mechanics (Post-Principia, 1687)",
            "Copernican Model (Early, 1543)",
            "Ptolemaic Astronomy (Late)",
            "Political Ideology (Generic)",
        ],
    )
    def test_all_named_systems(self, system_name):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_system,
        )

        result = compute_epistemic_system(system_name)
        assert result.name == system_name
        assert 0.0 <= result.F <= 1.0
        assert 0.0 <= result.omega <= 1.0
        assert result.heterogeneity_gap >= -1e-5

    def test_gap_ratio_property(self):
        from closures.everyday_physics.epistemic_coherence import (
            compute_epistemic_system,
        )

        result = compute_epistemic_system("Established Scientific Consensus")
        assert result.gap_ratio >= 0.0


class TestEpistemicTheorems:
    """Test all 7 epistemic coherence theorems."""

    def test_theorem_EC1_tier1_identities(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC1_tier1_identities,
        )

        result = theorem_EC1_tier1_identities()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC2_persistence_integrity_decoupling(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC2_persistence_integrity_decoupling,
        )

        result = theorem_EC2_persistence_integrity_decoupling()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC3_channel_death_dominance(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC3_channel_death_dominance,
        )

        result = theorem_EC3_channel_death_dominance()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC4_evidence_type_hierarchy(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC4_evidence_type_hierarchy,
        )

        result = theorem_EC4_evidence_type_hierarchy()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC5_paradigm_shift_gap_event(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC5_paradigm_shift_gap_event,
        )

        result = theorem_EC5_paradigm_shift_gap_event()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC6_folk_knowledge_region(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC6_folk_knowledge_region,
        )

        result = theorem_EC6_folk_knowledge_region()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_theorem_EC7_institutional_amplification(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC7_institutional_amplification,
        )

        result = theorem_EC7_institutional_amplification()
        assert result.verdict == "PROVEN"
        assert result.n_failed == 0

    def test_run_all_theorems(self):
        from closures.everyday_physics.epistemic_coherence import run_all_theorems

        results = run_all_theorems()
        assert len(results) == 7
        for r in results:
            assert r.verdict == "PROVEN"
            assert r.pass_rate == 1.0

    def test_theorem_result_properties(self):
        from closures.everyday_physics.epistemic_coherence import (
            theorem_EC1_tier1_identities,
        )

        result = theorem_EC1_tier1_identities()
        assert result.n_tests > 0
        assert result.n_passed == result.n_tests
        assert result.pass_rate == 1.0
        assert isinstance(result.details, dict)


# ═══════════════════════════════════════════════════════════════════
# closures/everyday_physics/thermodynamics.py
# ═══════════════════════════════════════════════════════════════════


class TestThermodynamics:
    """Unit tests for thermodynamics closures."""

    def test_compute_all(self):
        from closures.everyday_physics.thermodynamics import (
            compute_all_thermal_materials,
        )

        results = compute_all_thermal_materials()
        assert len(results) == 20

    def test_tier1_identities(self):
        from closures.everyday_physics.thermodynamics import (
            compute_all_thermal_materials,
        )

        for r in compute_all_thermal_materials():
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
            assert r.IC <= r.F + 1e-5

    def test_copper(self):
        from closures.everyday_physics.thermodynamics import compute_thermal_material

        result = compute_thermal_material(name="Copper", Cp=0.385, k_th=401.0, rho=8960.0, T_m=1358.0, T_b=2835.0)
        assert result.material == "Copper"
        assert result.regime in ("Stable", "Watch", "Collapse")

    @pytest.mark.parametrize("idx", range(20))
    def test_each_material_tier1(self, idx):
        from closures.everyday_physics.thermodynamics import (
            compute_all_thermal_materials,
        )

        r = compute_all_thermal_materials()[idx]
        assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
        assert r.IC <= r.F + 1e-5

    def test_regime_labels(self):
        from closures.everyday_physics.thermodynamics import (
            compute_all_thermal_materials,
        )

        valid = {"Stable", "Watch", "Collapse"}
        for r in compute_all_thermal_materials():
            assert r.regime in valid


# ═══════════════════════════════════════════════════════════════════
# closures/everyday_physics/electromagnetism.py
# ═══════════════════════════════════════════════════════════════════


class TestElectromagnetism:
    """Unit tests for electromagnetism closures."""

    def test_compute_all(self):
        from closures.everyday_physics.electromagnetism import (
            compute_all_em_materials,
        )

        results = compute_all_em_materials()
        assert len(results) == 20

    def test_tier1_identities(self):
        from closures.everyday_physics.electromagnetism import (
            compute_all_em_materials,
        )

        for r in compute_all_em_materials():
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
            assert r.IC <= r.F + 1e-5

    def test_copper_conductor(self):
        from closures.everyday_physics.electromagnetism import (
            compute_electromagnetic_material,
        )

        result = compute_electromagnetic_material(
            name="Copper",
            category="conductor",
            sigma=59.6,
            eps_r=1.0,
            work_fn=4.65,
            band_gap=0.0,
            mu_r=0.999994,
            resistivity=1.68e-8,
        )
        assert result.material == "Copper"
        assert result.category == "conductor"

    @pytest.mark.parametrize("idx", range(20))
    def test_each_material_tier1(self, idx):
        from closures.everyday_physics.electromagnetism import (
            compute_all_em_materials,
        )

        r = compute_all_em_materials()[idx]
        assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)

    def test_regime_labels(self):
        from closures.everyday_physics.electromagnetism import (
            compute_all_em_materials,
        )

        valid = {"Stable", "Watch", "Collapse"}
        for r in compute_all_em_materials():
            assert r.regime in valid


# ═══════════════════════════════════════════════════════════════════
# closures/everyday_physics/optics.py
# ═══════════════════════════════════════════════════════════════════


class TestOptics:
    """Unit tests for optics closures."""

    def test_compute_all(self):
        from closures.everyday_physics.optics import compute_all_optical_materials

        results = compute_all_optical_materials()
        assert len(results) == 20

    def test_tier1_identities(self):
        from closures.everyday_physics.optics import compute_all_optical_materials

        for r in compute_all_optical_materials():
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
            assert r.IC <= r.F + 1e-5

    def test_diamond(self):
        from closures.everyday_physics.optics import compute_optical_material

        result = compute_optical_material(
            name="Diamond",
            n_d=2.417,
            V_d=55.3,
            T_vis=0.71,
            R_vis=0.17,
            E_gap=5.47,
            n_group=2.46,
        )
        assert result.material == "Diamond"

    @pytest.mark.parametrize("idx", range(20))
    def test_each_material_tier1(self, idx):
        from closures.everyday_physics.optics import compute_all_optical_materials

        r = compute_all_optical_materials()[idx]
        assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)

    def test_regime_labels(self):
        from closures.everyday_physics.optics import compute_all_optical_materials

        valid = {"Stable", "Watch", "Collapse"}
        for r in compute_all_optical_materials():
            assert r.regime in valid


# ═══════════════════════════════════════════════════════════════════
# closures/everyday_physics/wave_phenomena.py
# ═══════════════════════════════════════════════════════════════════


class TestWavePhenomena:
    """Unit tests for wave_phenomena closures."""

    def test_compute_all(self):
        from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

        results = compute_all_wave_systems()
        assert len(results) == 24

    def test_tier1_identities(self):
        from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

        for r in compute_all_wave_systems():
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
            assert r.IC <= r.F + 1e-5

    def test_concert_A(self):
        from closures.everyday_physics.wave_phenomena import compute_wave_system

        result = compute_wave_system(
            name="Concert A",
            wave_type="sound",
            frequency=440.0,
            wavelength=0.78,
            phase_velocity=343.0,
            Q_factor=100.0,
            coherence_lengths=50.0,
            amplitude_norm=0.5,
        )
        assert result.system == "Concert A"
        assert result.wave_type == "sound"

    @pytest.mark.parametrize("idx", range(24))
    def test_each_system_tier1(self, idx):
        from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

        r = compute_all_wave_systems()[idx]
        assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)

    def test_regime_labels(self):
        from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

        valid = {"Stable", "Watch", "Collapse"}
        for r in compute_all_wave_systems():
            assert r.regime in valid

    def test_wave_types_present(self):
        from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

        types = {r.wave_type for r in compute_all_wave_systems()}
        assert "sound" in types
        assert "electromagnetic" in types
