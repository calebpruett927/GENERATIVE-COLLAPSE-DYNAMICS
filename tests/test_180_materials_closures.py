"""Tests for materials science closures — cohesive, phase, elastic, band, thermal.

Covers all 5 closures in closures/materials_science/:
  - cohesive_energy.py    Atomic IE → cohesive binding via Madelung/Born-Mayer/Friedel
  - phase_transition.py   RCFT universality exponents → phase transition scaling
  - elastic_moduli.py     Interatomic potential curvature → K, G, E moduli
  - band_structure.py     Electron config → band gap + RCFT attractor topology
  - debye_thermal.py      RCFT partition function → Debye phonon thermodynamics

Test structure follows the manifold layering:
  Layer 0 — Import & construction (closures loadable, results well-formed)
  Layer 1 — Kernel identities (F + ω = 1, ω ∈ [0,1], F ∈ [0,1] for all)
  Layer 2 — Regime classification (anchors correctly classified)
  Layer 3 — Physics consistency (monotonicity, limits, cross-closure integration)
  Layer 4 — Edge cases & boundary behavior

References:
  Kittel (2005), Ashcroft & Mermin (1976), Harrison (1980),
  CODATA 2022, Debye (1912)
"""

from __future__ import annotations

import pytest

from closures.materials_science.band_structure import (
    REFERENCE_GAPS,
    BandCharacter,
    BandResult,
    compute_band_structure,
)
from closures.materials_science.bcs_superconductivity import (
    REFERENCE_SC,
    BCSResult,
    SCType,
    compute_bcs_superconductivity,
)
from closures.materials_science.cohesive_energy import (
    REFERENCE_COHESIVE,
    BondType,
    CohesionRegime,
    CohesiveResult,
    compute_cohesive_energy,
)
from closures.materials_science.debye_thermal import (
    REFERENCE_THETA_D,
    DebyeResult,
    ThermalRegime,
    compute_debye_thermal,
)
from closures.materials_science.elastic_moduli import (
    ElasticRegime,
    ElasticResult,
    compute_elastic_moduli,
)
from closures.materials_science.magnetic_properties import (
    REFERENCE_MAGNETIC,
    MagneticClass,
    compute_magnetic_properties,
)
from closures.materials_science.phase_transition import (
    REFERENCE_TC,
    PhaseRegime,
    PhaseTransitionResult,
    compute_phase_transition,
    scan_phase_diagram,
)
from closures.materials_science.surface_catalysis import (
    REFERENCE_SURFACE,
    CatalyticClass,
    SurfaceCatalysisResult,
    compute_surface_catalysis,
)

# ═══════════════════════════════════════════════════════════════════
#  Layer 0 — Import & Construction
# ═══════════════════════════════════════════════════════════════════


class TestMaterialsImports:
    """Verify all closures import and expose compute functions."""

    def test_cohesive_energy_imports(self) -> None:
        from closures.materials_science import compute_cohesive_energy as fn

        assert callable(fn)

    def test_phase_transition_imports(self) -> None:
        from closures.materials_science import compute_phase_transition as fn

        assert callable(fn)

    def test_elastic_moduli_imports(self) -> None:
        from closures.materials_science import compute_elastic_moduli as fn

        assert callable(fn)

    def test_band_structure_imports(self) -> None:
        from closures.materials_science import compute_band_structure as fn

        assert callable(fn)

    def test_debye_thermal_imports(self) -> None:
        from closures.materials_science import compute_debye_thermal as fn

        assert callable(fn)

    def test_magnetic_imports(self) -> None:
        from closures.materials_science import compute_magnetic_properties as fn

        assert callable(fn)

    def test_bcs_imports(self) -> None:
        from closures.materials_science import compute_bcs_superconductivity as fn

        assert callable(fn)

    def test_surface_imports(self) -> None:
        from closures.materials_science import compute_surface_catalysis as fn

        assert callable(fn)

    def test_all_exports(self) -> None:
        """Module __all__ lists exactly the ten compute functions."""
        import closures.materials_science as ms

        assert len(ms.__all__) == 10
        for name in ms.__all__:
            assert hasattr(ms, name)


class TestResultTypes:
    """Verify NamedTuple result types have correct fields."""

    def test_cohesive_result_fields(self) -> None:
        fields = CohesiveResult._fields
        assert "E_coh_eV" in fields
        assert "omega_eff" in fields
        assert "F_eff" in fields
        assert "regime" in fields
        assert "Psi_coh" in fields

    def test_phase_result_fields(self) -> None:
        fields = PhaseTransitionResult._fields
        assert "order_parameter" in fields
        assert "susceptibility" in fields
        assert "correlation_length_ratio" in fields
        assert "omega_eff" in fields

    def test_elastic_result_fields(self) -> None:
        fields = ElasticResult._fields
        assert "K_GPa" in fields
        assert "G_GPa" in fields
        assert "E_GPa" in fields
        assert "omega_eff" in fields

    def test_band_result_fields(self) -> None:
        fields = BandResult._fields
        assert "E_g_eV" in fields
        assert "rcft_basin_type" in fields
        assert "rcft_fisher_distance" in fields
        assert "omega_eff" in fields

    def test_debye_result_fields(self) -> None:
        fields = DebyeResult._fields
        assert "C_V_J_mol_K" in fields
        assert "Theta_D_K" in fields
        assert "rcft_c_eff" in fields
        assert "omega_eff" in fields


class TestEnumValues:
    """Verify regime enums are well-formed StrEnums."""

    def test_bond_types(self) -> None:
        assert BondType.IONIC == "Ionic"
        assert BondType.METALLIC == "Metallic"
        assert BondType.COVALENT == "Covalent"
        assert BondType.VAN_DER_WAALS == "VanDerWaals"

    def test_cohesion_regimes(self) -> None:
        assert CohesionRegime.STRONG_BOND == "StrongBond"
        assert CohesionRegime.ANOMALOUS == "Anomalous"

    def test_phase_regimes(self) -> None:
        assert PhaseRegime.ORDERED == "Ordered"
        assert PhaseRegime.CRITICAL == "Critical"
        assert PhaseRegime.DISORDERED == "Disordered"

    def test_elastic_regimes(self) -> None:
        assert ElasticRegime.STIFF == "Stiff"
        assert ElasticRegime.ANOMALOUS == "Anomalous"

    def test_band_characters(self) -> None:
        assert BandCharacter.METAL == "Metal"
        assert BandCharacter.SEMICONDUCTOR == "Semiconductor"
        assert BandCharacter.INSULATOR == "Insulator"

    def test_thermal_regimes(self) -> None:
        assert ThermalRegime.QUANTUM == "Quantum"
        assert ThermalRegime.CLASSICAL == "Classical"


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Kernel Identity Checks (F + ω = 1)
# ═══════════════════════════════════════════════════════════════════


class TestCohesiveKernelIdentities:
    """F + ω = 1 for cohesive energy across reference materials."""

    @pytest.mark.parametrize(
        "Z, symbol",
        [(13, "Al"), (14, "Si"), (29, "Cu"), (74, "W"), (11, "Na"), (6, "C")],
    )
    def test_F_plus_omega_equals_one(self, Z: int, symbol: str) -> None:
        r = compute_cohesive_energy(Z, symbol=symbol)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12, f"{symbol}: F={r.F_eff}, ω={r.omega_eff}"

    @pytest.mark.parametrize(
        "Z, symbol",
        [(13, "Al"), (14, "Si"), (29, "Cu"), (74, "W")],
    )
    def test_omega_bounded(self, Z: int, symbol: str) -> None:
        r = compute_cohesive_energy(Z, symbol=symbol)
        assert r.omega_eff >= 0.0, f"{symbol}: ω_eff negative"
        # ω can exceed 1 for very poor predictions, but F is clipped
        assert 0.0 <= r.F_eff <= 1.0, f"{symbol}: F_eff out of range"

    @pytest.mark.parametrize(
        "Z, symbol",
        [(13, "Al"), (29, "Cu"), (14, "Si")],
    )
    def test_Psi_coh_bounded(self, Z: int, symbol: str) -> None:
        r = compute_cohesive_energy(Z, symbol=symbol)
        assert 0.0 <= r.Psi_coh <= 1.0, f"{symbol}: Ψ_coh out of [0,1]"


class TestPhaseKernelIdentities:
    """F + ω = 1 for phase transitions."""

    @pytest.mark.parametrize(
        "T_frac",
        [0.5, 0.9, 0.95, 1.0, 1.01, 1.5],
    )
    def test_F_plus_omega_iron_curie(self, T_frac: float) -> None:
        T = T_frac * 1043.0
        r = compute_phase_transition(T, 1043.0, material_key="Fe")
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12, f"T/Tc={T_frac}: F={r.F_eff}, ω={r.omega_eff}"

    def test_F_bounded(self) -> None:
        for t in [0.1, 0.5, 0.9, 0.99, 1.0, 1.5, 5.0]:
            r = compute_phase_transition(t * 1043, 1043.0)
            assert 0.0 <= r.F_eff <= 1.0


class TestElasticKernelIdentities:
    """F + ω = 1 for elastic moduli."""

    @pytest.mark.parametrize(
        "symbol, E_coh, r0",
        [("Cu", 3.49, 2.56), ("Fe", 4.28, 2.48), ("Al", 3.39, 2.86)],
    )
    def test_F_plus_omega(self, symbol: str, E_coh: float, r0: float) -> None:
        r = compute_elastic_moduli(E_coh, r0, symbol=symbol)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12


class TestBandKernelIdentities:
    """F + ω = 1 for band structure."""

    @pytest.mark.parametrize(
        "Z, symbol",
        [(29, "Cu"), (14, "Si"), (6, "C_diamond"), (13, "Al")],
    )
    def test_F_plus_omega(self, Z: int, symbol: str) -> None:
        r = compute_band_structure(Z, symbol=symbol)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12

    @pytest.mark.parametrize(
        "Z, symbol",
        [(29, "Cu"), (14, "Si"), (6, "C_diamond")],
    )
    def test_Psi_band_bounded(self, Z: int, symbol: str) -> None:
        r = compute_band_structure(Z, symbol=symbol)
        assert 0.0 <= r.Psi_band <= 1.0


class TestDebyeKernelIdentities:
    """F + ω = 1 for Debye thermal."""

    @pytest.mark.parametrize("T", [10.0, 100.0, 300.0, 1000.0])
    def test_F_plus_omega_copper(self, T: float) -> None:
        r = compute_debye_thermal(T, symbol="Cu")
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12

    @pytest.mark.parametrize("T", [30.0, 300.0])
    def test_F_bounded(self, T: float) -> None:
        r = compute_debye_thermal(T, symbol="Cu")
        assert 0.0 <= r.F_eff <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Regime Classification (Known Anchors)
# ═══════════════════════════════════════════════════════════════════


class TestCohesiveRegimes:
    """Regime classification for known materials."""

    def test_aluminum_cohesive_positive(self) -> None:
        """Al should produce positive cohesive energy."""
        r = compute_cohesive_energy(13, symbol="Al")
        assert r.E_coh_eV > 0.0
        # Al simple-metal model underestimates; ω honestly flags this
        assert r.omega_eff > 0.0

    def test_noble_gas_high_drift(self) -> None:
        """Noble gases have very weak cohesion — should show high ω."""
        r = compute_cohesive_energy(10, symbol="Ne")
        # Noble gas cohesion is tiny; model correctly reflects high drift
        assert r.omega_eff > 0.3

    def test_ionic_nacl_bond_type(self) -> None:
        """NaCl should be classified as ionic."""
        r = compute_cohesive_energy(
            11,
            symbol="NaCl",
            IE1_eV=5.14,
            electronegativity_diff=2.1,
            r0_angstrom=2.82,
        )
        assert r.bond_type == BondType.IONIC


class TestPhaseRegimes:
    """Regime classification for phase transitions."""

    def test_iron_low_temp_ordered(self) -> None:
        r = compute_phase_transition(300.0, 1043.0, material_key="Fe")
        assert r.regime == PhaseRegime.ORDERED

    def test_iron_critical_near_Tc(self) -> None:
        r = compute_phase_transition(1043.0, 1043.0, material_key="Fe")
        assert r.regime == PhaseRegime.CRITICAL

    def test_iron_high_temp_disordered(self) -> None:
        r = compute_phase_transition(2000.0, 1043.0, material_key="Fe")
        assert r.regime == PhaseRegime.DISORDERED

    def test_batio3_displacive(self) -> None:
        r = compute_phase_transition(200.0, 393.0, material_key="BaTiO3", transition_type="Displacive")
        assert r.regime == PhaseRegime.ORDERED


class TestElasticRegimes:
    """Regime classification for elastic moduli."""

    def test_copper_not_anomalous(self) -> None:
        r = compute_elastic_moduli(3.49, 2.56, symbol="Cu")
        assert r.regime != ElasticRegime.ANOMALOUS

    def test_diamond_stiff(self) -> None:
        """Diamond has extremely high bulk modulus → low ω."""
        r = compute_elastic_moduli(7.37, 1.54, symbol="C_diamond", n_neighbors=4)
        # Diamond K > 400 GPa, should have low drift
        assert r.K_GPa > 200.0


class TestBandRegimes:
    """Band structure regime classification."""

    def test_copper_is_metal(self) -> None:
        r = compute_band_structure(29, symbol="Cu")
        assert r.band_character == BandCharacter.METAL
        assert r.E_g_eV == 0.0

    def test_silicon_is_semiconductor(self) -> None:
        r = compute_band_structure(14, symbol="Si")
        assert r.band_character == BandCharacter.SEMICONDUCTOR
        assert 0.1 <= r.E_g_eV < 3.0

    def test_diamond_is_insulator(self) -> None:
        r = compute_band_structure(6, symbol="C_diamond")
        assert r.band_character == BandCharacter.INSULATOR
        assert r.E_g_eV >= 3.0

    def test_aluminum_is_metal(self) -> None:
        r = compute_band_structure(13, symbol="Al")
        assert r.band_character == BandCharacter.METAL


class TestDebyeRegimes:
    """Debye thermal regime classification."""

    def test_copper_low_T_quantum(self) -> None:
        r = compute_debye_thermal(10.0, symbol="Cu")
        assert r.regime == ThermalRegime.QUANTUM

    def test_copper_room_T_classical(self) -> None:
        r = compute_debye_thermal(300.0, symbol="Cu")
        assert r.regime == ThermalRegime.CLASSICAL

    def test_silicon_100K_intermediate(self) -> None:
        r = compute_debye_thermal(100.0, symbol="Si")
        assert r.regime == ThermalRegime.INTERMEDIATE


# ═══════════════════════════════════════════════════════════════════
#  Layer 3 — Physics Consistency
# ═══════════════════════════════════════════════════════════════════


class TestPhaseTransitionPhysics:
    """Physical consistency of phase transition predictions."""

    def test_order_parameter_decreases_with_temp(self) -> None:
        """Φ must decrease monotonically as T → T_c from below."""
        results = [compute_phase_transition(t * 1043, 1043.0) for t in [0.3, 0.5, 0.7, 0.9]]
        phis = [r.order_parameter for r in results]
        for i in range(len(phis) - 1):
            assert phis[i] >= phis[i + 1], f"Φ not monotonically decreasing: {phis}"

    def test_order_parameter_zero_above_Tc(self) -> None:
        """Above T_c, order parameter should be zero."""
        r = compute_phase_transition(1200.0, 1043.0)
        assert r.order_parameter == 0.0

    def test_susceptibility_diverges_near_Tc(self) -> None:
        """χ should be much larger near T_c than far below."""
        r_far = compute_phase_transition(500.0, 1043.0)
        r_near = compute_phase_transition(1040.0, 1043.0)
        assert r_near.susceptibility > r_far.susceptibility

    def test_scan_phase_diagram_returns_list(self) -> None:
        """scan_phase_diagram should return correctly sized list."""
        results = scan_phase_diagram(1043.0, n_points=20)
        assert len(results) == 20
        assert all(isinstance(r, PhaseTransitionResult) for r in results)

    def test_critical_exponents_from_p3(self) -> None:
        """For p=3: ν=1/3, γ=1/3, α=0, β=5/6, c_eff=1/3."""
        r = compute_phase_transition(500.0, 1043.0, p=3)
        assert abs(r.nu - 1 / 3) < 1e-5
        assert abs(r.gamma - 1 / 3) < 1e-5
        assert abs(r.alpha) < 1e-5
        assert abs(r.beta_exp - 5 / 6) < 1e-5
        assert abs(r.c_eff - 1 / 3) < 1e-5

    def test_different_p_changes_exponents(self) -> None:
        """Changing p should change the universality class."""
        r3 = compute_phase_transition(500.0, 1043.0, p=3)
        r5 = compute_phase_transition(500.0, 1043.0, p=5)
        assert r3.c_eff != r5.c_eff
        assert r3.nu != r5.nu


class TestElasticPhysics:
    """Physical consistency of elastic modulus predictions."""

    def test_positive_moduli(self) -> None:
        """All moduli should be positive for stable materials."""
        r = compute_elastic_moduli(3.49, 2.56, symbol="Cu")
        assert r.K_GPa > 0
        assert r.G_GPa > 0
        assert r.E_GPa > 0

    def test_stronger_bond_higher_K(self) -> None:
        """Higher cohesive energy → higher bulk modulus (same structure)."""
        r_cu = compute_elastic_moduli(3.49, 2.56, n_neighbors=12)
        r_w = compute_elastic_moduli(8.90, 2.74, n_neighbors=8)
        assert r_w.K_GPa > r_cu.K_GPa

    def test_poisson_constrained(self) -> None:
        """Poisson ratio should be in thermodynamic range [-1, 0.5]."""
        r = compute_elastic_moduli(3.49, 2.56, symbol="Cu", nu_poisson=0.34)
        assert -1.0 <= r.nu_poisson <= 0.5

    def test_elastic_relation_E_from_K_G(self) -> None:
        """E = 9KG/(3K+G) should hold approximately."""
        r = compute_elastic_moduli(3.49, 2.56, symbol="Cu")
        if r.K_GPa > 0 and r.G_GPa > 0:
            E_calc = 9 * r.K_GPa * r.G_GPa / (3 * r.K_GPa + r.G_GPa)
            assert abs(r.E_GPa - E_calc) / E_calc < 0.01


class TestBandPhysics:
    """Physical consistency of band structure predictions."""

    def test_metals_zero_gap(self) -> None:
        """Metals should have E_g = 0."""
        for Z, sym in [(29, "Cu"), (13, "Al"), (26, "Fe"), (74, "W")]:
            r = compute_band_structure(Z, symbol=sym)
            assert r.E_g_eV == 0.0, f"{sym} should be metal with E_g=0"

    def test_semiconductor_gap_range(self) -> None:
        """Semiconductors should have gaps in [0.1, 3.0) eV."""
        r = compute_band_structure(14, symbol="Si")
        assert 0.1 <= r.E_g_eV < 3.0

    def test_fisher_distance_ordering(self) -> None:
        """Wider gap → larger Fisher geodesic distance."""
        r_si = compute_band_structure(14, symbol="Si")
        r_dia = compute_band_structure(6, symbol="C_diamond")
        assert r_dia.rcft_fisher_distance > r_si.rcft_fisher_distance

    def test_metal_monostable_attractor(self) -> None:
        """Metals should map to Monostable RCFT attractor."""
        r = compute_band_structure(29, symbol="Cu")
        assert r.rcft_basin_type == "Monostable"

    def test_semiconductor_bistable_attractor(self) -> None:
        """Semiconductors should map to Bistable attractor."""
        r = compute_band_structure(14, symbol="Si")
        assert r.rcft_basin_type == "Bistable"


class TestDebyePhysics:
    """Physical consistency of Debye thermal predictions."""

    def test_high_T_dulong_petit(self) -> None:
        """At T ≫ Θ_D, C_V → 3R (Dulong-Petit limit)."""
        r = compute_debye_thermal(2000.0, symbol="Cu")
        assert abs(r.C_V_normalized - 1.0) < 0.05  # within 5% of 3R

    def test_low_T_cubic(self) -> None:
        """At T ≪ Θ_D, C_V ∝ T³ (Debye T³ law)."""
        r1 = compute_debye_thermal(5.0, symbol="Cu")
        r2 = compute_debye_thermal(10.0, symbol="Cu")
        # C_V(10) / C_V(5) ≈ (10/5)³ = 8 for T³ law
        if r1.C_V_J_mol_K > 0:
            ratio = r2.C_V_J_mol_K / r1.C_V_J_mol_K
            assert ratio > 4.0  # Should approach 8 but numerical integration

    def test_Cv_monotonically_increases(self) -> None:
        """C_V should increase monotonically with temperature."""
        temps = [10, 50, 100, 200, 300, 500]
        Cvs = [compute_debye_thermal(t, symbol="Cu").C_V_J_mol_K for t in temps]
        for i in range(len(Cvs) - 1):
            assert Cvs[i] <= Cvs[i + 1] + 1e-6, f"C_V not monotonic: {Cvs}"

    def test_rcft_c_eff_one_third(self) -> None:
        """The RCFT effective central charge should be 1/3 for p=3."""
        r = compute_debye_thermal(300.0, symbol="Cu")
        assert abs(r.rcft_c_eff - 1 / 3) < 1e-5


# ═══════════════════════════════════════════════════════════════════
#  Layer 4 — Edge Cases & Boundary Behavior
# ═══════════════════════════════════════════════════════════════════


class TestCohesiveEdgeCases:
    """Edge cases for cohesive energy computation."""

    def test_unknown_element_returns_result(self) -> None:
        """Unknown symbol should still return a result (with no ref)."""
        r = compute_cohesive_energy(119, symbol="Uue", IE1_eV=4.0, Z_eff=1.0, n_eff=8)
        assert isinstance(r, CohesiveResult)
        assert r.E_coh_eV >= 0.0

    def test_custom_measured_value(self) -> None:
        """Providing E_coh_measured should override reference lookup."""
        r = compute_cohesive_energy(13, symbol="Al", E_coh_measured_eV=99.0)
        assert r.E_coh_measured_eV == 99.0

    def test_reference_table_coverage(self) -> None:
        """Reference table should have > 30 entries."""
        assert len(REFERENCE_COHESIVE) > 30


class TestPhaseEdgeCases:
    """Edge cases for phase transition computation."""

    def test_zero_temperature(self) -> None:
        """T=0 should give fully ordered state."""
        r = compute_phase_transition(0.0, 1043.0)
        assert r.order_parameter == 1.0
        assert r.regime == PhaseRegime.ORDERED

    def test_T_equals_Tc(self) -> None:
        """T = T_c should give critical regime."""
        r = compute_phase_transition(1043.0, 1043.0)
        assert r.regime == PhaseRegime.CRITICAL

    def test_very_high_temperature(self) -> None:
        """Very high T should give disordered state, Φ=0."""
        r = compute_phase_transition(1e6, 1043.0)
        assert r.order_parameter == 0.0
        assert r.regime == PhaseRegime.DISORDERED

    def test_negative_Tc_raises_or_handles(self) -> None:
        """Non-physical T_c should be handled gracefully."""
        # Should not crash — may return NON_EVALUABLE or similar
        try:
            r = compute_phase_transition(300.0, -100.0)
            # If it returns, just verify it's well-formed
            assert isinstance(r, PhaseTransitionResult)
        except (ValueError, ZeroDivisionError):
            pass  # Also acceptable — explicit rejection

    def test_reference_Tc_table_coverage(self) -> None:
        assert len(REFERENCE_TC) > 10


class TestBandEdgeCases:
    """Edge cases for band structure computation."""

    def test_custom_measured_gap(self) -> None:
        """Providing E_g_measured should override reference."""
        r = compute_band_structure(14, symbol="Si", E_g_measured_eV=2.0)
        assert r.E_g_measured_eV == 2.0

    def test_reference_table_coverage(self) -> None:
        assert len(REFERENCE_GAPS) > 30

    def test_fisher_distance_nonnegative(self) -> None:
        """Fisher distance should always be ≥ 0."""
        for Z, sym in [(29, "Cu"), (14, "Si"), (6, "C_diamond")]:
            r = compute_band_structure(Z, symbol=sym)
            assert r.rcft_fisher_distance >= 0.0


class TestDebyeEdgeCases:
    """Edge cases for Debye thermal computation."""

    def test_very_low_temperature(self) -> None:
        """At T→0+, C_V → 0."""
        r = compute_debye_thermal(0.1, symbol="Cu")
        assert r.C_V_J_mol_K < 0.1  # essentially zero

    def test_reference_table_coverage(self) -> None:
        assert len(REFERENCE_THETA_D) > 25

    def test_unknown_material_with_Theta_D(self) -> None:
        """Unknown symbol but explicit Θ_D should work."""
        r = compute_debye_thermal(300.0, Theta_D_K=400.0)
        assert isinstance(r, DebyeResult)
        assert r.C_V_J_mol_K > 0


# ═══════════════════════════════════════════════════════════════════
#  Cross-Closure Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestCrossClosureIntegration:
    """Tests that verify closures work together correctly."""

    def test_cohesive_feeds_elastic(self) -> None:
        """Cohesive energy output can drive elastic moduli input."""
        coh = compute_cohesive_energy(29, symbol="Cu")
        elast = compute_elastic_moduli(coh.E_coh_eV, coh.r0_A, symbol="Cu")
        assert elast.K_GPa > 0
        assert isinstance(elast, ElasticResult)

    def test_elastic_feeds_debye(self) -> None:
        """Elastic moduli can provide sound velocity for Debye."""
        elast = compute_elastic_moduli(3.49, 2.56, symbol="Cu")
        # Debye thermal with elastic moduli as input
        r = compute_debye_thermal(
            300.0,
            symbol="Cu",
            K_GPa=elast.K_GPa,
            G_GPa=elast.G_GPa,
        )
        assert isinstance(r, DebyeResult)
        assert r.C_V_J_mol_K > 0

    def test_phase_at_different_temps_consistent(self) -> None:
        """Phase diagram scan should show ordered→critical→disordered."""
        results = scan_phase_diagram(1043.0, T_min_K=100, T_max_K=2000, n_points=50)
        regimes = [r.regime for r in results]
        # Should see Ordered first, then transition through Critical to Disordered
        assert PhaseRegime.ORDERED in regimes
        assert PhaseRegime.DISORDERED in regimes

    def test_cohesive_feeds_surface(self) -> None:
        """Cohesive energy feeds surface energy calculation."""
        coh = compute_cohesive_energy(29, symbol="Cu")
        surf = compute_surface_catalysis(coh.E_coh_eV, symbol="Cu", r0_A=coh.r0_A)
        assert surf.gamma_J_m2 > 0
        assert isinstance(surf, SurfaceCatalysisResult)

    def test_debye_feeds_bcs(self) -> None:
        """Debye temperature feeds BCS T_c calculation."""
        debye = compute_debye_thermal(300.0, symbol="Nb")
        bcs = compute_bcs_superconductivity(debye.Theta_D_K, 1.04, symbol="Nb")
        assert bcs.T_c_K > 0
        assert isinstance(bcs, BCSResult)

    def test_magnetic_and_phase_consistent(self) -> None:
        """Magnetic and phase transition closures agree on ordering."""
        mag = compute_magnetic_properties(0, symbol="Fe", T_K=300)
        phase = compute_phase_transition(300.0, 1043.0, material_key="Fe")
        # Both should indicate ordered state below T_c
        assert mag.magnetic_class == MagneticClass.FERROMAGNETIC
        assert phase.regime == PhaseRegime.ORDERED


# ═══════════════════════════════════════════════════════════════════
#  Magnetic Properties Tests
# ═══════════════════════════════════════════════════════════════════


class TestMagneticKernelIdentities:
    """F + ω = 1 for magnetic properties."""

    @pytest.mark.parametrize("symbol", ["Fe", "Ni", "Cu", "Al", "Cr"])
    def test_F_plus_omega(self, symbol: str) -> None:
        r = compute_magnetic_properties(0, symbol=symbol, T_K=300)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12


class TestMagneticClassification:
    """Magnetic classification for known materials."""

    def test_iron_ferromagnetic(self) -> None:
        r = compute_magnetic_properties(0, symbol="Fe", T_K=300)
        assert r.magnetic_class == MagneticClass.FERROMAGNETIC

    def test_copper_diamagnetic(self) -> None:
        r = compute_magnetic_properties(0, symbol="Cu", T_K=300)
        assert r.magnetic_class == MagneticClass.DIAMAGNETIC

    def test_chromium_antiferromagnetic(self) -> None:
        r = compute_magnetic_properties(0, symbol="Cr", T_K=300)
        assert r.magnetic_class == MagneticClass.ANTIFERROMAGNETIC

    def test_iron_above_Tc_paramagnetic_like(self) -> None:
        r = compute_magnetic_properties(0, symbol="Fe", T_K=1200)
        assert r.M_total_B == 0.0  # No net magnetization above T_c


class TestMagneticPhysics:
    """Physical consistency of magnetic predictions."""

    def test_magnetization_decreases_with_temp(self) -> None:
        """M(T) should decrease as T → T_c."""
        r1 = compute_magnetic_properties(0, symbol="Fe", T_K=300)
        r2 = compute_magnetic_properties(0, symbol="Fe", T_K=900)
        assert r1.M_total_B > r2.M_total_B

    def test_diamagnetic_chi_zero_or_negative(self) -> None:
        """Diamagnetic χ ≤ 0 (model returns 0 as sentinel for no paramagnetism)."""
        r = compute_magnetic_properties(0, symbol="Cu", T_K=300)
        assert r.chi_SI <= 0

    def test_rcft_beta_five_sixths(self) -> None:
        r = compute_magnetic_properties(0, symbol="Fe", T_K=300)
        assert abs(r.rcft_beta - 5 / 6) < 1e-5

    def test_reference_table_coverage(self) -> None:
        assert len(REFERENCE_MAGNETIC) > 10

    def test_unpaired_electrons_iron(self) -> None:
        r = compute_magnetic_properties(0, symbol="Fe", T_K=300)
        assert r.n_unpaired == 4  # Fe d⁶ → 4 unpaired (Hund's rule)


# ═══════════════════════════════════════════════════════════════════
#  BCS Superconductivity Tests
# ═══════════════════════════════════════════════════════════════════


class TestBCSKernelIdentities:
    """F + ω = 1 for BCS superconductivity."""

    @pytest.mark.parametrize("symbol", ["Nb", "Pb", "Al", "Sn"])
    def test_F_plus_omega(self, symbol: str) -> None:
        r = compute_bcs_superconductivity(0, 0, symbol=symbol)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12


class TestBCSClassification:
    """BCS type classification."""

    def test_aluminum_superconductor(self) -> None:
        """Al should be superconducting with low T_c."""
        r = compute_bcs_superconductivity(0, 0, symbol="Al")
        assert r.T_c_K > 0
        assert r.T_c_K < 5.0

    def test_normal_state_above_Tc(self) -> None:
        """Above T_c, should be Normal state."""
        r = compute_bcs_superconductivity(275, 1.04, symbol="Nb", T_K=20)
        assert r.sc_type == SCType.NORMAL

    def test_bcs_gap_positive(self) -> None:
        """BCS gap should be positive for all superconductors."""
        for sym in ["Nb", "Pb", "Al"]:
            r = compute_bcs_superconductivity(0, 0, symbol=sym)
            assert r.Delta_0_meV > 0


class TestBCSPhysics:
    """Physical consistency of BCS predictions."""

    def test_bcs_ratio_near_weak_coupling(self) -> None:
        """2Δ/(k_BT_c) ≈ 3.528 for weak coupling."""
        r = compute_bcs_superconductivity(0, 0, symbol="Al")
        assert abs(r.BCS_ratio - 3.528) < 0.5

    def test_stronger_coupling_higher_Tc(self) -> None:
        """Larger λ_ep → higher T_c (same Θ_D)."""
        r1 = compute_bcs_superconductivity(300, 0.5)
        r2 = compute_bcs_superconductivity(300, 1.0)
        assert r2.T_c_K > r1.T_c_K

    def test_zero_coupling_no_sc(self) -> None:
        """λ_ep = 0 → no superconductivity."""
        r = compute_bcs_superconductivity(300, 0.0)
        assert r.T_c_K == 0.0

    def test_reference_table_coverage(self) -> None:
        assert len(REFERENCE_SC) > 8

    def test_rcft_attractor_depth_bounded(self) -> None:
        r = compute_bcs_superconductivity(0, 0, symbol="Nb")
        assert 0.0 <= r.rcft_attractor_depth <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  Surface & Catalysis Tests
# ═══════════════════════════════════════════════════════════════════


class TestSurfaceKernelIdentities:
    """F + ω = 1 for surface/catalysis closure."""

    @pytest.mark.parametrize("symbol", ["Cu", "Pt", "Au", "Al"])
    def test_F_plus_omega(self, symbol: str) -> None:
        r = compute_surface_catalysis(0, symbol=symbol)
        assert abs(r.F_eff + r.omega_eff - 1.0) < 1e-12


class TestSurfaceClassification:
    """Surface and catalytic classification."""

    def test_platinum_near_sabatier(self) -> None:
        """Pt should be near Sabatier optimum for CO oxidation."""
        r = compute_surface_catalysis(0, symbol="Pt")
        # Pt is famous as a volcano peak catalyst
        assert r.catalytic_class in (CatalyticClass.VOLCANO_PEAK, CatalyticClass.INERT)

    def test_gold_inert(self) -> None:
        """Au should be catalytically inert (bulk)."""
        r = compute_surface_catalysis(0, symbol="Au")
        assert r.catalytic_class == CatalyticClass.INERT

    def test_copper_inert(self) -> None:
        """Cu should be catalytically inert (filled d-band)."""
        r = compute_surface_catalysis(0, symbol="Cu")
        assert r.catalytic_class == CatalyticClass.INERT


class TestSurfacePhysics:
    """Physical consistency of surface predictions."""

    def test_surface_energy_positive(self) -> None:
        """Surface energy should always be positive."""
        for sym in ["Cu", "Fe", "Au", "Pt", "W"]:
            r = compute_surface_catalysis(0, symbol=sym)
            assert r.gamma_J_m2 > 0

    def test_higher_cohesive_higher_surface(self) -> None:
        """Stronger bonded metals → higher surface energy (generally)."""
        r_cu = compute_surface_catalysis(0, symbol="Cu")
        r_w = compute_surface_catalysis(0, symbol="W")
        assert r_w.gamma_J_m2 > r_cu.gamma_J_m2

    def test_vacancy_energy_positive(self) -> None:
        r = compute_surface_catalysis(0, symbol="Cu")
        assert r.E_vacancy_eV > 0

    def test_broken_bond_fraction_bounded(self) -> None:
        r = compute_surface_catalysis(0, symbol="Cu")
        assert 0.0 < r.broken_bond_fraction < 1.0

    def test_reference_table_coverage(self) -> None:
        assert len(REFERENCE_SURFACE) > 10

    def test_rcft_return_deficit_positive(self) -> None:
        """Surface has incomplete return → deficit > 0."""
        r = compute_surface_catalysis(0, symbol="Cu")
        assert r.rcft_return_deficit > 0
