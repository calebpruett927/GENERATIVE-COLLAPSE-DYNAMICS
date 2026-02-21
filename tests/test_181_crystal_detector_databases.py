"""Tests for crystal morphology and particle detector databases.

Covers two new database modules in closures/materials_science/:
  - crystal_morphology_database.py   OpenCrystalData compounds + kernel
  - particle_detector_database.py    Scintillators, SLD, shielding + kernel

Test structure follows the manifold layering:
  Layer 0 — Import & construction (modules loadable, data well-formed)
  Layer 1 — Kernel identities (F + ω = 1, IC ≤ F for all entries)
  Layer 2 — Database consistency (lookups, filters, polymorphs)
  Layer 3 — Physics consistency (quenching, shielding, morphology)
  Layer 4 — Edge cases & validation

References:
  Barhate et al. (2024) DOI: 10.1016/j.dche.2024.100150
  Rodrigues & Felizardo (2026) DOI: 10.3390/particles9010020
  Dimiccoli et al. (2025) DOI: 10.3390/particles8040082
"""

from __future__ import annotations

import math

import pytest

from closures.materials_science.crystal_morphology_database import (
    COMPOUNDS,
    ApplicationDomain,
    CrystalCompound,
    CrystalHabit,
    CrystalKernelResult,
    CrystalSystem,
    analyze_polymorphs,
    build_trace,
    compute_all_crystal_kernels,
    compute_crystal_kernel,
    crystal_system_statistics,
    get_compound,
    get_compounds_by_domain,
    get_compounds_by_formula,
    get_compounds_by_habit,
    get_compounds_by_system,
    get_opencrystaldata_compounds,
)
from closures.materials_science.crystal_morphology_database import (
    validate_database as validate_crystal_db,
)
from closures.materials_science.particle_detector_database import (
    SCINTILLATORS,
    SHIELDING_MATERIALS,
    SUPERHEATED_DETECTORS,
    DetectorKernelResult,
    DetectorType,
    QuenchingModel,
    ScintillatorMaterial,
    ShieldingCategory,
    ShieldingMaterial,
    SuperheatedDetector,
    birks_quenching_factor,
    build_scintillator_trace,
    compare_scintillators,
    compute_all_scintillator_kernels,
    compute_scintillator_kernel,
    get_scintillator,
    get_scintillators_by_type,
    get_shielding_by_category,
    get_shielding_material,
    logistic_quenching_factor,
    rank_shielding_materials,
    shielding_effectiveness,
)
from closures.materials_science.particle_detector_database import (
    validate_database as validate_detector_db,
)

# ═══════════════════════════════════════════════════════════════════
#  Layer 0 — Import & Construction (Crystal Morphology)
# ═══════════════════════════════════════════════════════════════════


class TestCrystalImports:
    """Verify crystal morphology database imports and basic structure."""

    def test_compounds_tuple_nonempty(self) -> None:
        assert len(COMPOUNDS) >= 15

    def test_compound_is_frozen_dataclass(self) -> None:
        c = COMPOUNDS[0]
        assert isinstance(c, CrystalCompound)
        with pytest.raises(AttributeError):
            c.name = "modified"  # type: ignore[misc]

    def test_all_compounds_have_required_fields(self) -> None:
        for c in COMPOUNDS:
            assert c.name, "Compound missing name"
            assert c.formula, "Compound missing formula"
            assert c.molecular_weight > 0, f"{c.name}: MW must be > 0"
            assert c.density_g_cm3 > 0, f"{c.name}: density must be > 0"
            assert c.melting_point_K > 0, f"{c.name}: MP must be > 0"
            assert isinstance(c.crystal_system, CrystalSystem)
            assert isinstance(c.habit, CrystalHabit)

    def test_crystal_kernel_result_is_namedtuple(self) -> None:
        kr = compute_crystal_kernel(COMPOUNDS[0])
        assert isinstance(kr, CrystalKernelResult)
        assert hasattr(kr, "F")
        assert hasattr(kr, "omega")
        assert hasattr(kr, "IC")

    def test_build_trace_returns_8_channels(self) -> None:
        trace = build_trace(COMPOUNDS[0])
        assert len(trace) == 8

    def test_trace_values_in_unit_interval(self) -> None:
        for c in COMPOUNDS:
            trace = build_trace(c)
            for i, v in enumerate(trace):
                assert 0.0 <= v <= 1.0, f"{c.name} channel {i}: {v} out of [0,1]"

    def test_to_dict_roundtrip(self) -> None:
        c = COMPOUNDS[0]
        d = c.to_dict()
        assert d["name"] == c.name
        assert d["formula"] == c.formula
        assert d["molecular_weight"] == c.molecular_weight
        assert d["crystal_system"] == c.crystal_system.value


# ═══════════════════════════════════════════════════════════════════
#  Layer 0 — Import & Construction (Particle Detector)
# ═══════════════════════════════════════════════════════════════════


class TestDetectorImports:
    """Verify particle detector database imports and basic structure."""

    def test_scintillators_tuple_nonempty(self) -> None:
        assert len(SCINTILLATORS) >= 8

    def test_shielding_tuple_nonempty(self) -> None:
        assert len(SHIELDING_MATERIALS) >= 5

    def test_superheated_tuple_nonempty(self) -> None:
        assert len(SUPERHEATED_DETECTORS) >= 1

    def test_scintillator_is_frozen_dataclass(self) -> None:
        s = SCINTILLATORS[0]
        assert isinstance(s, ScintillatorMaterial)
        with pytest.raises(AttributeError):
            s.name = "modified"  # type: ignore[misc]

    def test_shielding_is_frozen_dataclass(self) -> None:
        sh = SHIELDING_MATERIALS[0]
        assert isinstance(sh, ShieldingMaterial)

    def test_superheated_is_frozen_dataclass(self) -> None:
        det = SUPERHEATED_DETECTORS[0]
        assert isinstance(det, SuperheatedDetector)

    def test_detector_kernel_result_is_namedtuple(self) -> None:
        kr = compute_scintillator_kernel(SCINTILLATORS[0])
        assert isinstance(kr, DetectorKernelResult)

    def test_scintillator_trace_8_channels(self) -> None:
        trace = build_scintillator_trace(SCINTILLATORS[0])
        assert len(trace) == 8

    def test_scintillator_trace_in_unit_interval(self) -> None:
        for s in SCINTILLATORS:
            trace = build_scintillator_trace(s)
            for i, v in enumerate(trace):
                assert 0.0 <= v <= 1.0, f"{s.name} channel {i}: {v} out of [0,1]"

    def test_to_dict_scintillator(self) -> None:
        s = SCINTILLATORS[0]
        d = s.to_dict()
        assert d["name"] == s.name
        assert d["detector_type"] == s.detector_type.value

    def test_to_dict_shielding(self) -> None:
        sh = SHIELDING_MATERIALS[0]
        d = sh.to_dict()
        assert d["name"] == sh.name
        assert d["neutron_transmission"] == sh.neutron_transmission

    def test_to_dict_superheated(self) -> None:
        det = SUPERHEATED_DETECTORS[0]
        d = det.to_dict()
        assert d["name"] == det.name
        assert d["gamma_sensitive"] == det.gamma_sensitive


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Kernel Identities (Crystal Morphology)
# ═══════════════════════════════════════════════════════════════════


class TestCrystalKernelIdentities:
    """Verify Tier-1 identities hold for all crystal compounds."""

    @pytest.fixture(scope="class")
    def all_kernels(self) -> list[CrystalKernelResult]:
        return compute_all_crystal_kernels()

    def test_duality_identity(self, all_kernels: list[CrystalKernelResult]) -> None:
        """F + ω = 1 for all compounds."""
        for kr in all_kernels:
            residual = abs(kr.F + kr.omega - 1.0)
            assert residual < 1e-12, f"{kr.name}: |F + ω − 1| = {residual:.2e}"

    def test_integrity_bound(self, all_kernels: list[CrystalKernelResult]) -> None:
        """IC ≤ F for all compounds (integrity bound)."""
        for kr in all_kernels:
            assert kr.IC <= kr.F + 1e-12, f"{kr.name}: IC ({kr.IC:.6f}) > F ({kr.F:.6f})"

    def test_F_in_unit_interval(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.F <= 1.0, f"{kr.name}: F = {kr.F}"

    def test_omega_in_unit_interval(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.omega <= 1.0, f"{kr.name}: ω = {kr.omega}"

    def test_S_nonnegative(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.S >= 0.0, f"{kr.name}: S = {kr.S}"

    def test_C_in_unit_interval(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.C <= 1.0, f"{kr.name}: C = {kr.C}"

    def test_kappa_nonpositive(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.kappa <= 0.0 + 1e-12, f"{kr.name}: κ = {kr.kappa}"

    def test_IC_equals_exp_kappa(self, all_kernels: list[CrystalKernelResult]) -> None:
        """IC ≈ exp(κ) — the log-integrity relation."""
        for kr in all_kernels:
            expected = math.exp(kr.kappa)
            assert abs(kr.IC - expected) < 1e-12, f"{kr.name}: IC ({kr.IC:.8f}) ≠ exp(κ) ({expected:.8f})"

    def test_no_nan_or_inf(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            for field in ("F", "omega", "S", "C", "kappa", "IC"):
                v = getattr(kr, field)
                assert not math.isnan(v) and not math.isinf(v), f"{kr.name}: {field} is NaN or Inf"

    def test_regime_valid(self, all_kernels: list[CrystalKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.regime in ("Stable", "Watch", "Collapse"), f"{kr.name}: invalid regime '{kr.regime}'"


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Kernel Identities (Particle Detector)
# ═══════════════════════════════════════════════════════════════════


class TestDetectorKernelIdentities:
    """Verify Tier-1 identities hold for all scintillator materials."""

    @pytest.fixture(scope="class")
    def all_kernels(self) -> list[DetectorKernelResult]:
        return compute_all_scintillator_kernels()

    def test_duality_identity(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            residual = abs(kr.F + kr.omega - 1.0)
            assert residual < 1e-12, f"{kr.name}: |F + ω − 1| = {residual:.2e}"

    def test_integrity_bound(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.IC <= kr.F + 1e-12, f"{kr.name}: IC ({kr.IC:.6f}) > F ({kr.F:.6f})"

    def test_F_in_unit_interval(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.F <= 1.0, f"{kr.name}: F = {kr.F}"

    def test_omega_in_unit_interval(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.omega <= 1.0, f"{kr.name}: ω = {kr.omega}"

    def test_S_nonnegative(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.S >= 0.0, f"{kr.name}: S = {kr.S}"

    def test_C_in_unit_interval(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert 0.0 <= kr.C <= 1.0, f"{kr.name}: C = {kr.C}"

    def test_IC_equals_exp_kappa(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            expected = math.exp(kr.kappa)
            assert abs(kr.IC - expected) < 1e-12, f"{kr.name}: IC ({kr.IC:.8f}) ≠ exp(κ) ({expected:.8f})"

    def test_no_nan_or_inf(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            for field in ("F", "omega", "S", "C", "kappa", "IC"):
                v = getattr(kr, field)
                assert not math.isnan(v) and not math.isinf(v), f"{kr.name}: {field} is NaN or Inf"

    def test_regime_valid(self, all_kernels: list[DetectorKernelResult]) -> None:
        for kr in all_kernels:
            assert kr.regime in ("Stable", "Watch", "Collapse")


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Database Consistency (Crystal)
# ═══════════════════════════════════════════════════════════════════


class TestCrystalLookups:
    """Verify crystal database lookup and filter functions."""

    def test_get_compound_by_name(self) -> None:
        c = get_compound("Cephalexin monohydrate")
        assert c is not None
        assert c.formula == "C16H17N3O4S·H2O"

    def test_get_compound_not_found(self) -> None:
        assert get_compound("Nonexistent") is None

    def test_get_compounds_by_formula_polymorphs(self) -> None:
        """Paracetamol has two polymorphs in the database."""
        variants = get_compounds_by_formula("C8H9NO2")
        assert len(variants) >= 2

    def test_get_compounds_by_system(self) -> None:
        cubics = get_compounds_by_system(CrystalSystem.CUBIC)
        assert len(cubics) >= 3  # Ag dendritic, Ag compact, NaCl, microspheres

    def test_get_compounds_by_habit(self) -> None:
        needles = get_compounds_by_habit(CrystalHabit.NEEDLE)
        assert len(needles) >= 2

    def test_get_compounds_by_domain(self) -> None:
        pharma = get_compounds_by_domain(ApplicationDomain.PHARMACEUTICAL)
        assert len(pharma) >= 3

    def test_opencrystaldata_compounds(self) -> None:
        ocd = get_opencrystaldata_compounds()
        assert len(ocd) >= 6  # All 7 OpenCrystalData entries
        for c in ocd:
            assert c.source_dataset.startswith("OpenCrystalData")


class TestCrystalAnalysis:
    """Test crystal analysis functions."""

    def test_polymorph_analysis(self) -> None:
        result = analyze_polymorphs("C8H9NO2")
        assert result["polymorph_count"] >= 2
        assert "polymorphs" in result
        for p in result["polymorphs"]:
            assert "F" in p
            assert "heterogeneity_gap" in p
            assert p["heterogeneity_gap"] >= 0

    def test_polymorph_single_form(self) -> None:
        result = analyze_polymorphs("NaCl")
        assert result["polymorph_count"] == 1
        assert "note" in result

    def test_crystal_system_statistics(self) -> None:
        stats = crystal_system_statistics()
        assert len(stats) >= 3  # At least cubic, monoclinic, orthorhombic
        for _system, data in stats.items():
            assert data["count"] > 0
            assert 0 < data["mean_F"] < 1
            assert data["mean_heterogeneity_gap"] >= 0


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Database Consistency (Detector)
# ═══════════════════════════════════════════════════════════════════


class TestDetectorLookups:
    """Verify detector database lookup and filter functions."""

    def test_get_scintillator_by_name(self) -> None:
        s = get_scintillator("EJ-200")
        assert s is not None
        assert s.detector_type == DetectorType.PLASTIC_SCINTILLATOR

    def test_get_scintillator_not_found(self) -> None:
        assert get_scintillator("Nonexistent") is None

    def test_get_scintillators_by_type_plastic(self) -> None:
        plastics = get_scintillators_by_type(DetectorType.PLASTIC_SCINTILLATOR)
        assert len(plastics) >= 2  # EJ-200, EJ-208, BC-408

    def test_get_scintillators_by_type_inorganic(self) -> None:
        inorganics = get_scintillators_by_type(DetectorType.INORGANIC_SCINTILLATOR)
        assert len(inorganics) >= 4  # LYSO, NaI, CsI, BGO, PWO

    def test_get_shielding_material(self) -> None:
        sh = get_shielding_material("Paraffin wax")
        assert sh is not None
        assert sh.neutron_transmission == pytest.approx(0.0498, abs=0.001)

    def test_get_shielding_by_category(self) -> None:
        hydro = get_shielding_by_category(ShieldingCategory.HYDROGENOUS)
        assert len(hydro) >= 2  # Polyethylene, paraffin

    def test_compare_scintillators_all(self) -> None:
        result = compare_scintillators()
        assert result["materials_compared"] == len(SCINTILLATORS)
        assert "highest_F" in result
        assert "smallest_gap" in result

    def test_compare_scintillators_subset(self) -> None:
        result = compare_scintillators(["EJ-200", "LYSO:Ce"])
        assert result["materials_compared"] == 2


# ═══════════════════════════════════════════════════════════════════
#  Layer 3 — Physics Consistency
# ═══════════════════════════════════════════════════════════════════


class TestCrystalPhysics:
    """Physical consistency checks for crystal morphology data."""

    def test_microsphere_highest_circularity(self) -> None:
        """Polystyrene microspheres should have the highest circularity."""
        for c in COMPOUNDS:
            if c.habit == CrystalHabit.MICROSPHERE and c.circularity is not None:
                assert c.circularity > 0.95

    def test_needle_low_circularity(self) -> None:
        """Needle crystals should have low circularity."""
        for c in COMPOUNDS:
            if c.habit == CrystalHabit.NEEDLE and c.circularity is not None:
                assert c.circularity < 0.40

    def test_needle_high_aspect_ratio(self) -> None:
        """Needle crystals should have high aspect ratio."""
        for c in COMPOUNDS:
            if c.habit == CrystalHabit.NEEDLE and c.aspect_ratio is not None:
                assert c.aspect_ratio >= 4.0

    def test_blocky_low_aspect_ratio(self) -> None:
        """Blocky crystals should have aspect ratio near 1."""
        for c in COMPOUNDS:
            if c.habit == CrystalHabit.BLOCKY and c.aspect_ratio is not None:
                assert c.aspect_ratio <= 1.5

    def test_silver_density_correct(self) -> None:
        """Silver density should be ~10.49 g/cm³."""
        ag = get_compounds_by_formula("Ag")
        assert len(ag) >= 2
        for c in ag:
            assert c.density_g_cm3 == pytest.approx(10.49, abs=0.1)


class TestDetectorPhysics:
    """Physical consistency checks for detector data."""

    def test_inorganic_higher_density_than_plastic(self) -> None:
        """Inorganic scintillators should have higher density than plastics."""
        plastics = get_scintillators_by_type(DetectorType.PLASTIC_SCINTILLATOR)
        inorganics = get_scintillators_by_type(DetectorType.INORGANIC_SCINTILLATOR)
        max_plastic = max(s.density_g_cm3 for s in plastics)
        min_inorganic = min(s.density_g_cm3 for s in inorganics)
        assert min_inorganic > max_plastic

    def test_lyso_higher_zeff_than_plastic(self) -> None:
        """LYSO:Ce should have much higher Z_eff than plastic scintillator."""
        lyso = get_scintillator("LYSO:Ce")
        ej200 = get_scintillator("EJ-200")
        assert lyso is not None and ej200 is not None
        assert lyso.Z_eff is not None and ej200.Z_eff is not None
        assert lyso.Z_eff > 10 * ej200.Z_eff

    def test_ej200_birks_onsager_parameters(self) -> None:
        """EJ-200 Birks-Onsager parameters from Dimiccoli et al. (2025)."""
        ej200 = get_scintillator("EJ-200")
        assert ej200 is not None
        assert ej200.quenching_model == QuenchingModel.BIRKS_ONSAGER
        assert ej200.birks_inv_MeV_cm == pytest.approx(12.0, abs=0.1)
        assert ej200.eta_e_h == pytest.approx(0.853, abs=0.001)

    def test_lyso_birks_onsager_parameters(self) -> None:
        """LYSO:Ce Birks-Onsager parameters from Dimiccoli et al. (2025)."""
        lyso = get_scintillator("LYSO:Ce")
        assert lyso is not None
        assert lyso.birks_inv_MeV_cm == pytest.approx(361.0, abs=1.0)
        assert lyso.eta_H == pytest.approx(0.053, abs=0.001)

    def test_sld_gamma_insensitive(self) -> None:
        """SLD C3F8 must be gamma-insensitive (Cs-137 test)."""
        sld = SUPERHEATED_DETECTORS[0]
        assert sld.gamma_sensitive is False
        assert sld.neutron_sensitive is True

    def test_sld_droplet_diameter(self) -> None:
        """SLD droplet diameter ~20.4 μm per Rodrigues & Felizardo (2026)."""
        sld = SUPERHEATED_DETECTORS[0]
        assert sld.droplet_diameter_um == pytest.approx(20.4, abs=0.5)

    def test_paraffin_best_neutron_shield(self) -> None:
        """Paraffin should have the lowest neutron transmission."""
        ranked = rank_shielding_materials()
        assert ranked[0][0] == "Paraffin wax"
        assert ranked[0][1] > 0.95  # >95% effectiveness

    def test_aluminum_worst_neutron_shield(self) -> None:
        """Aluminum should have the highest transmission (worst shield)."""
        ranked = rank_shielding_materials()
        assert ranked[-1][0] == "Aluminum"
        assert ranked[-1][2] > 0.99  # Nearly transparent

    def test_shielding_transmission_ordered(self) -> None:
        """Hydrogenous materials should shield better than metal."""
        paraffin = get_shielding_material("Paraffin wax")
        pe = get_shielding_material("Polyethylene (HDPE)")
        al = get_shielding_material("Aluminum")
        assert paraffin is not None and pe is not None and al is not None
        assert paraffin.neutron_transmission < pe.neutron_transmission
        assert pe.neutron_transmission < al.neutron_transmission


class TestQuenchingPhysics:
    """Test quenching factor computations."""

    def test_birks_zero_dEdx(self) -> None:
        """At zero dE/dx, quenching factor should be ~1 (no quenching)."""
        Q = birks_quenching_factor(0.0, 12.0)
        assert pytest.approx(1.0, abs=1e-6) == Q

    def test_birks_high_dEdx(self) -> None:
        """At very high dE/dx, quenching factor should approach 0."""
        Q = birks_quenching_factor(10000.0, 12.0)
        assert Q < 0.02

    def test_birks_onsager_ej200(self) -> None:
        """Birks-Onsager for EJ-200 with measured parameters."""
        Q = birks_quenching_factor(
            dEdx=50.0,
            kB_inv=12.0,
            eta_e_h=0.853,
            eta_H=0.0,
            dEdx_0=134.0,
        )
        assert 0.0 < Q < 1.0

    def test_logistic_zero_dEdx(self) -> None:
        """At zero dE/dx, logistic quenching should give 1."""
        Q = logistic_quenching_factor(0.0, 65.0, 0.75)
        assert pytest.approx(1.0, abs=1e-6) == Q

    def test_logistic_high_dEdx(self) -> None:
        """At very high dE/dx, logistic should approach L_inf."""
        Q = logistic_quenching_factor(100000.0, 65.0, 0.75, L_inf=0.0)
        assert Q < 0.01

    def test_logistic_with_L_inf(self) -> None:
        """Logistic with L_inf > 0 should plateau above L_inf."""
        Q = logistic_quenching_factor(100000.0, 380.0, 1.0, L_inf=0.05)
        assert Q >= 0.05

    def test_birks_invalid_kB(self) -> None:
        """kB_inv <= 0 should return 1.0 (no quenching)."""
        assert birks_quenching_factor(100.0, 0.0) == 1.0
        assert birks_quenching_factor(100.0, -5.0) == 1.0


# ═══════════════════════════════════════════════════════════════════
#  Layer 4 — Validation & Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestCrystalValidation:
    """Test crystal database validation."""

    def test_validate_returns_conformant(self) -> None:
        result = validate_crystal_db()
        assert result["status"] == "CONFORMANT"
        assert len(result["errors"]) == 0

    def test_validate_total_count(self) -> None:
        result = validate_crystal_db()
        assert result["total_compounds"] == len(COMPOUNDS)


class TestDetectorValidation:
    """Test detector database validation."""

    def test_validate_returns_conformant(self) -> None:
        result = validate_detector_db()
        assert result["status"] == "CONFORMANT"
        assert len(result["errors"]) == 0

    def test_validate_counts(self) -> None:
        result = validate_detector_db()
        assert result["total_scintillators"] == len(SCINTILLATORS)
        assert result["total_shielding"] == len(SHIELDING_MATERIALS)
        assert result["total_superheated"] == len(SUPERHEATED_DETECTORS)

    def test_shielding_effectiveness_bounds(self) -> None:
        for sh in SHIELDING_MATERIALS:
            eff = shielding_effectiveness(sh)
            assert 0.0 <= eff <= 1.0, f"{sh.name}: effectiveness {eff}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_crystal_insoluble_compound(self) -> None:
        """Compounds with solubility=None should not fail kernel."""
        ag = get_compound("Silver (dendritic)")
        assert ag is not None
        assert ag.solubility_g_L is None
        kr = compute_crystal_kernel(ag)
        assert 0.0 < kr.F < 1.0

    def test_crystal_high_solubility(self) -> None:
        """Compounds with very high solubility should have valid kernel."""
        sucrose = get_compound("Sucrose")
        assert sucrose is not None
        assert sucrose.solubility_g_L is not None
        assert sucrose.solubility_g_L > 1000
        kr = compute_crystal_kernel(sucrose)
        assert 0.0 < kr.F < 1.0

    def test_detector_very_low_light_yield(self) -> None:
        """PWO has very low light yield — kernel should still work."""
        pwo = get_scintillator("PbWO4 (PWO)")
        assert pwo is not None
        assert pwo.light_yield_ph_per_keV < 1.0
        kr = compute_scintillator_kernel(pwo)
        assert 0.0 < kr.F < 1.0

    def test_detector_very_high_density(self) -> None:
        """High-density materials like PWO should not break kernel."""
        pwo = get_scintillator("PbWO4 (PWO)")
        assert pwo is not None
        assert pwo.density_g_cm3 > 8.0
        kr = compute_scintillator_kernel(pwo)
        assert kr.IC <= kr.F + 1e-12


class TestCrossModuleImports:
    """Verify the __init__.py re-exports work."""

    def test_crystal_kernel_from_init(self) -> None:
        from closures.materials_science import compute_crystal_kernel as fn

        assert callable(fn)

    def test_scintillator_kernel_from_init(self) -> None:
        from closures.materials_science import compute_scintillator_kernel as fn

        assert callable(fn)
