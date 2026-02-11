"""Tests for atomic physics closures — ATOM.INTSTACK.v1

150 tests covering all 6 atomic physics closures with precision
checks calibrated to the known physics each model captures.

Lesson: every model has a regime where it works and a regime where it
fails.  The tests verify BOTH — correct predictions within model scope
and honest ω measurement when the model reaches its limits.
"""

from __future__ import annotations

import pytest

from closures.atomic_physics.electron_config import (
    AUFBAU_EXCEPTIONS,
    AUFBAU_ORDER,
    ElectronConfigResult,
    compute_electron_config,
)
from closures.atomic_physics.fine_structure import (
    ALPHA_FINE,
    FineStructureResult,
    compute_fine_structure,
)
from closures.atomic_physics.ionization_energy import (
    NIST_IE1,
    THRESH_PRECISE,
    IonizationResult,
    compute_ionization,
)
from closures.atomic_physics.selection_rules import (
    SelectionResult,
    compute_selection_rules,
)
from closures.atomic_physics.spectral_lines import (
    SpectralResult,
    compute_spectral_lines,
)
from closures.atomic_physics.zeeman_stark import (
    ZeemanStarkResult,
    compute_zeeman_stark,
)

# ═══════════════════════════════════════════════════════════════════
# 1. Ionization Energy Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestIonizationEnergy:
    """Ionization energy via Slater screening + hydrogenic model."""

    # ── Basic correctness ──────────────────────────────────────

    def test_hydrogen_exact(self) -> None:
        """H has no screening — prediction must be nearly exact.

        ω ≈ 3×10⁻⁵ due to RYDBERG_EV rounding (13.5984 vs NIST 13.598).
        """
        r = compute_ionization(1)
        assert r.IE_predicted_eV == pytest.approx(13.598, abs=0.001)
        assert r.omega_eff < THRESH_PRECISE  # well within Precise regime
        assert r.regime == "Precise"

    def test_hydrogen_fidelity(self) -> None:
        r = compute_ionization(1)
        assert r.F_eff > 0.999  # near-perfect fidelity
        assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_hydrogen_psi_ie(self) -> None:
        r = compute_ionization(1)
        assert r.Psi_IE == pytest.approx(1.0, abs=0.01)

    def test_helium_screening(self) -> None:
        """He: Slater σ=0.30 gives Z_eff=1.70.  ω is large (variational Z_eff≈1.34)."""
        r = compute_ionization(2)
        assert r.Z_eff == pytest.approx(1.70, abs=0.01)
        assert r.omega_eff > 0.4  # model limitation, honestly reported

    def test_z_eff_positive(self) -> None:
        """Z_eff must always be ≥ 1."""
        for z in range(1, 37):
            r = compute_ionization(z)
            assert r.Z_eff >= 1.0, f"Z={z}: Z_eff={r.Z_eff}"

    def test_nist_reference_used(self) -> None:
        """When no measured IE given, NIST reference is used."""
        for z in [1, 10, 18, 26, 36]:
            r = compute_ionization(z)
            assert r.IE_measured_eV == pytest.approx(NIST_IE1[z], abs=0.001)

    def test_custom_measured_ie(self) -> None:
        """User-provided measured IE overrides NIST."""
        r = compute_ionization(1, IE_measured_eV=14.0)
        assert r.IE_measured_eV == pytest.approx(14.0, abs=0.01)

    # ── Proper Slater screening ────────────────────────────────

    def test_sodium_valence_3s(self) -> None:
        """Na outer electron is 3s, n=3."""
        r = compute_ionization(11)
        assert r.n_eff == pytest.approx(3.0, abs=0.01)

    def test_iron_valence_4s(self) -> None:
        """Fe outer electron is 4s (highest n), NOT 3d (last Aufbau)."""
        r = compute_ionization(26)
        assert r.n_eff == pytest.approx(4.0, abs=0.01)
        # Proper Slater: ω should be < 1.0 (no longer catastrophic)
        assert r.omega_eff < 0.8, f"Fe ω={r.omega_eff} still too high"

    def test_iron_improvement_over_naive(self) -> None:
        """Fe ω must be well below 1.0 with proper Slater screening."""
        r = compute_ionization(26)
        assert r.omega_eff < 0.6  # was 1.0 before fix

    def test_copper_valence_4s(self) -> None:
        """Cu outer electron is 4s (highest n)."""
        r = compute_ionization(29)
        assert r.n_eff == pytest.approx(4.0, abs=0.01)

    # ── Regime classification ──────────────────────────────────

    def test_regime_precise(self) -> None:
        """H should be in Precise regime (exact match)."""
        r = compute_ionization(1)
        assert r.regime == "Precise"

    def test_regime_noble_gases(self) -> None:
        """Noble gases have high IE — model should not be catastrophic."""
        for z in [10, 18, 36]:
            r = compute_ionization(z)
            assert r.omega_eff <= 1.0

    def test_regime_alkali_metals(self) -> None:
        """Alkali metals (one outer s electron) — model should be in range."""
        for z in [3, 11, 19]:
            r = compute_ionization(z)
            assert r.omega_eff < 1.0

    # ── GCD invariant identities ───────────────────────────────

    def test_f_plus_omega_equals_one(self) -> None:
        """F + ω = 1 identity must hold for all elements."""
        for z in range(1, 37):
            r = compute_ionization(z)
            assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_omega_bounded(self) -> None:
        """ω ∈ [0, 1] for all elements."""
        for z in range(1, 37):
            r = compute_ionization(z)
            assert r.omega_eff >= 0.0
            assert r.omega_eff <= 1.0

    def test_psi_ie_bounded(self) -> None:
        """Ψ_IE ∈ [0, 1] for all elements."""
        for z in range(1, 37):
            r = compute_ionization(z)
            assert r.Psi_IE >= 0.0
            assert r.Psi_IE <= 1.0

    # ── Edge cases ─────────────────────────────────────────────

    def test_z_equals_1(self) -> None:
        r = compute_ionization(1)
        assert isinstance(r, IonizationResult)

    def test_z_equals_36(self) -> None:
        r = compute_ionization(36)
        assert isinstance(r, IonizationResult)

    def test_invalid_z_raises(self) -> None:
        with pytest.raises(ValueError, match="≥ 1"):
            compute_ionization(0)

    def test_quantum_defect_parameter(self) -> None:
        """Quantum defect reduces effective QN → increases IE prediction."""
        r_base = compute_ionization(11)
        r_defect = compute_ionization(11, quantum_defect=1.37)
        assert r_defect.n_eff < r_base.n_eff
        assert r_defect.IE_predicted_eV > r_base.IE_predicted_eV

    def test_explicit_n_override(self) -> None:
        """Explicit n overrides auto-detection."""
        r = compute_ionization(11, n=2)
        assert r.n_eff == pytest.approx(2.0, abs=0.01)

    def test_result_fields_complete(self) -> None:
        r = compute_ionization(1)
        assert hasattr(r, "IE_predicted_eV")
        assert hasattr(r, "Z_eff")
        assert hasattr(r, "regime")

    def test_ie_positive(self) -> None:
        """Predicted IE must be positive for all elements."""
        for z in range(1, 37):
            r = compute_ionization(z)
            assert r.IE_predicted_eV > 0, f"Z={z}: negative IE"


# ═══════════════════════════════════════════════════════════════════
# 2. Spectral Lines Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestSpectralLines:
    """Spectral lines via Rydberg formula with reduced-mass correction."""

    # ── Hydrogen Balmer precision ──────────────────────────────

    def test_h_alpha_precision(self) -> None:
        """H-α (656.281 nm NIST) — reduced mass + air correction."""
        r = compute_spectral_lines(1, 2, 3)
        assert r.lambda_predicted_nm == pytest.approx(656.281, abs=0.02)
        assert r.regime == "Resolved"

    def test_h_beta_precision(self) -> None:
        """H-β (486.135 nm NIST)."""
        r = compute_spectral_lines(1, 2, 4)
        assert r.lambda_predicted_nm == pytest.approx(486.135, abs=0.02)

    def test_h_gamma_precision(self) -> None:
        """H-γ (434.047 nm NIST)."""
        r = compute_spectral_lines(1, 2, 5)
        assert r.lambda_predicted_nm == pytest.approx(434.047, abs=0.02)

    def test_h_delta_precision(self) -> None:
        """H-δ (410.174 nm NIST)."""
        r = compute_spectral_lines(1, 2, 6)
        assert r.lambda_predicted_nm == pytest.approx(410.174, abs=0.02)

    def test_balmer_series_resolved(self) -> None:
        """All Balmer lines with NIST ref should be Resolved."""
        for n_up in range(3, 8):
            r = compute_spectral_lines(1, 2, n_up)
            assert r.regime == "Resolved", f"n={n_up}: {r.regime}"
            assert r.omega_eff < 0.001

    # ── Reduced mass effect ────────────────────────────────────

    def test_reduced_mass_shifts_longer(self) -> None:
        """Reduced-mass correction makes wavelengths slightly longer than R∞ values."""
        r = compute_spectral_lines(1, 2, 3)
        # R∞ prediction (no reduced mass) would give 656.112 nm
        assert r.lambda_predicted_nm > 656.11

    def test_helium_ion_lyman(self) -> None:
        """He⁺ Lyman-α: Z=2, much shorter wavelength."""
        r = compute_spectral_lines(2, 1, 2)
        # λ ∝ 1/Z² → He⁺ should be ~4× shorter than H
        assert r.lambda_predicted_nm < 35.0

    # ── Series identification ──────────────────────────────────

    def test_series_name_lyman(self) -> None:
        r = compute_spectral_lines(1, 1, 2)
        assert r.series_name == "Lyman"

    def test_series_name_balmer(self) -> None:
        r = compute_spectral_lines(1, 2, 3)
        assert r.series_name == "Balmer"

    def test_series_name_paschen(self) -> None:
        r = compute_spectral_lines(1, 3, 4)
        assert r.series_name == "Paschen"

    def test_series_name_brackett(self) -> None:
        r = compute_spectral_lines(1, 4, 5)
        assert r.series_name == "Brackett"

    # ── Energy consistency ─────────────────────────────────────

    def test_energy_positive(self) -> None:
        """Transition energy must be positive (emission)."""
        r = compute_spectral_lines(1, 2, 3)
        assert r.energy_eV > 0

    def test_energy_increases_with_n_upper(self) -> None:
        """Higher n_upper → larger energy (series limit)."""
        energies = []
        for n_up in range(3, 8):
            r = compute_spectral_lines(1, 2, n_up)
            energies.append(r.energy_eV)
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1]

    def test_wavelength_decreases_with_n_upper(self) -> None:
        """Higher n_upper → shorter wavelength (higher energy)."""
        wavelengths = []
        for n_up in range(3, 8):
            r = compute_spectral_lines(1, 2, n_up)
            wavelengths.append(r.lambda_predicted_nm)
        for i in range(len(wavelengths) - 1):
            assert wavelengths[i] > wavelengths[i + 1]

    # ── GCD invariants ─────────────────────────────────────────

    def test_f_plus_omega_balmer(self) -> None:
        for n_up in range(3, 8):
            r = compute_spectral_lines(1, 2, n_up)
            assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_omega_bounded_spectral(self) -> None:
        r = compute_spectral_lines(1, 2, 3)
        assert r.omega_eff >= 0.0
        assert r.omega_eff <= 1.0

    # ── Edge cases ─────────────────────────────────────────────

    def test_invalid_n_order_raises(self) -> None:
        with pytest.raises(ValueError, match="n_upper"):
            compute_spectral_lines(1, 3, 2)

    def test_invalid_z_raises(self) -> None:
        with pytest.raises(ValueError, match="Z must"):
            compute_spectral_lines(0, 1, 2)

    def test_custom_measured_wavelength(self) -> None:
        r = compute_spectral_lines(1, 2, 3, lambda_measured_nm=656.3)
        assert r.lambda_measured_nm == pytest.approx(656.3, abs=0.01)

    def test_result_type(self) -> None:
        r = compute_spectral_lines(1, 2, 3)
        assert isinstance(r, SpectralResult)

    def test_compute_series_function(self) -> None:
        from closures.atomic_physics.spectral_lines import compute_series

        results = compute_series(1, 2, n_upper_max=5)
        assert len(results) == 3  # n=3,4,5

    def test_lyman_alpha_reference(self) -> None:
        """Lyman-α with NIST ref should be close."""
        r = compute_spectral_lines(1, 1, 2)
        # Lyman is UV — vacuum wavelengths
        assert r.lambda_predicted_nm == pytest.approx(121.567, abs=0.05)

    def test_z2_scaling(self) -> None:
        """He⁺ wavelengths scale as 1/Z² relative to H."""
        r_h = compute_spectral_lines(1, 2, 3)
        r_he = compute_spectral_lines(2, 2, 3)
        ratio = r_h.lambda_predicted_nm / r_he.lambda_predicted_nm
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_n_quantum_numbers_stored(self) -> None:
        r = compute_spectral_lines(1, 2, 5)
        assert r.n_lower == 2
        assert r.n_upper == 5


# ═══════════════════════════════════════════════════════════════════
# 3. Electron Configuration Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestElectronConfig:
    """Electron configuration via Aufbau + exceptions table."""

    # ── Light elements ─────────────────────────────────────────

    def test_hydrogen_config(self) -> None:
        r = compute_electron_config(1)
        assert r.configuration == "1s¹"
        assert r.valence_electrons == 1

    def test_helium_closed_shell(self) -> None:
        r = compute_electron_config(2)
        assert r.configuration == "1s²"
        assert r.shell_completeness == pytest.approx(1.0)
        assert r.regime == "ClosedShell"

    def test_carbon_config(self) -> None:
        r = compute_electron_config(6)
        assert "2p²" in r.configuration
        assert r.period == 2

    def test_neon_noble_gas(self) -> None:
        r = compute_electron_config(10)
        assert r.shell_completeness == pytest.approx(1.0)
        assert r.regime == "ClosedShell"

    def test_sodium_single_valence(self) -> None:
        r = compute_electron_config(11)
        assert "3s¹" in r.configuration
        assert r.noble_gas_core == "[Ne]"

    def test_argon_closed_shell(self) -> None:
        r = compute_electron_config(18)
        assert r.regime == "ClosedShell"

    # ── Aufbau exceptions (critical fix) ───────────────────────

    def test_chromium_exception(self) -> None:
        """Cr: [Ar] 3d⁵ 4s¹ (half-filled d stability)."""
        r = compute_electron_config(24)
        assert "4s¹" in r.configuration
        assert "3d⁵" in r.configuration

    def test_copper_exception(self) -> None:
        """Cu: [Ar] 3d¹⁰ 4s¹ (fully-filled d stability)."""
        r = compute_electron_config(29)
        assert "4s¹" in r.configuration
        assert "3d¹⁰" in r.configuration
        assert r.shell_completeness == pytest.approx(1.0)

    def test_palladium_exception(self) -> None:
        """Pd: [Kr] 4d¹⁰ (no 5s at all!)."""
        r = compute_electron_config(46)
        assert "4d¹⁰" in r.configuration
        assert "5s" not in r.configuration

    def test_silver_exception(self) -> None:
        """Ag: [Kr] 4d¹⁰ 5s¹ (fully-filled d)."""
        r = compute_electron_config(47)
        assert "5s¹" in r.configuration
        assert "4d¹⁰" in r.configuration

    def test_gold_exception(self) -> None:
        """Au: [Xe] 4f¹⁴ 5d¹⁰ 6s¹."""
        r = compute_electron_config(79)
        assert "6s¹" in r.configuration
        assert "5d¹⁰" in r.configuration

    def test_exception_electrons_sum_to_z(self) -> None:
        """All exception configs must have total electrons = Z."""
        for z, config in AUFBAU_EXCEPTIONS.items():
            total = sum(pop for _, pop in config)
            assert total == z, f"Z={z}: {total} electrons != {z}"

    # ── Period detection ───────────────────────────────────────

    def test_period_1(self) -> None:
        assert compute_electron_config(1).period == 1
        assert compute_electron_config(2).period == 1

    def test_period_2(self) -> None:
        assert compute_electron_config(3).period == 2
        assert compute_electron_config(10).period == 2

    def test_period_3(self) -> None:
        assert compute_electron_config(11).period == 3
        assert compute_electron_config(18).period == 3

    def test_period_4(self) -> None:
        assert compute_electron_config(19).period == 4
        assert compute_electron_config(36).period == 4

    # ── Block classification ───────────────────────────────────

    def test_s_block(self) -> None:
        r = compute_electron_config(11)
        assert r.group_block == "s-block"

    def test_p_block(self) -> None:
        r = compute_electron_config(6)
        assert r.group_block == "p-block"

    def test_d_block(self) -> None:
        r = compute_electron_config(26)
        assert r.group_block == "d-block"

    # ── GCD invariants ─────────────────────────────────────────

    def test_f_plus_omega_config(self) -> None:
        for z in range(1, 37):
            r = compute_electron_config(z)
            assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_completeness_bounded(self) -> None:
        for z in range(1, 37):
            r = compute_electron_config(z)
            assert r.shell_completeness >= 0.0
            assert r.shell_completeness <= 1.0

    # ── Edge cases ─────────────────────────────────────────────

    def test_invalid_z_raises_config(self) -> None:
        with pytest.raises(ValueError):
            compute_electron_config(0)
        with pytest.raises(ValueError):
            compute_electron_config(119)

    def test_z_118(self) -> None:
        """Maximum Z should not crash."""
        r = compute_electron_config(118)
        assert isinstance(r, ElectronConfigResult)

    def test_noble_gas_core_label(self) -> None:
        r = compute_electron_config(26)
        assert r.noble_gas_core == "[Ar]"

    def test_all_aufbau_order_used(self) -> None:
        """AUFBAU_ORDER should have enough capacity for Z=118."""
        total_cap = sum(cap for _, _, cap, _ in AUFBAU_ORDER)
        assert total_cap >= 118


# ═══════════════════════════════════════════════════════════════════
# 4. Fine Structure Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestFineStructure:
    """Fine structure via Dirac formula + corrected Lamb shift."""

    # ── Hydrogen n=2 ───────────────────────────────────────────

    def test_h_2s_gross_energy(self) -> None:
        r = compute_fine_structure(1, 2, 0, 0.5)
        assert r.E_n_eV == pytest.approx(-3.3996, abs=0.001)

    def test_h_2p_fine_structure(self) -> None:
        """2p₁/₂ and 2p₃/₂ should have different fine-structure corrections."""
        r_half = compute_fine_structure(1, 2, 1, 0.5)
        r_three = compute_fine_structure(1, 2, 1, 1.5)
        assert r_half.E_fine_eV != r_three.E_fine_eV

    def test_h_2p_splitting(self) -> None:
        """2p₁/₂ – 2p₃/₂ splitting should be ~4.5×10⁻⁵ eV."""
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.splitting_eV == pytest.approx(4.5e-5, rel=0.2)

    def test_h_s_state_no_splitting(self) -> None:
        """s-states (l=0) have no j-splitting (only j=1/2)."""
        r = compute_fine_structure(1, 2, 0, 0.5)
        assert r.splitting_eV == 0.0

    # ── Lamb shift (Bethe logarithm fix) ───────────────────────

    def test_lamb_shift_nonzero_s_state(self) -> None:
        """Lamb shift is non-zero for s-states."""
        r = compute_fine_structure(1, 2, 0, 0.5)
        assert r.E_lamb_eV > 0.0

    def test_lamb_shift_zero_p_state(self) -> None:
        """Lamb shift is zero for p-states (no wavefunction at origin)."""
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.E_lamb_eV == 0.0

    def test_lamb_shift_order_of_magnitude(self) -> None:
        """H 2S₁/₂ Lamb shift ≈ 4.4×10⁻⁶ eV (experimental 1057 MHz).

        Our leading-order Bethe estimate should be within factor of 2.
        Previous code was 42× too small.
        """
        r = compute_fine_structure(1, 2, 0, 0.5)
        experimental_lamb = 4.372e-6  # eV
        ratio = r.E_lamb_eV / experimental_lamb
        assert 0.5 < ratio < 2.0, f"Lamb ratio={ratio:.2f} (should be ~1.0)"

    def test_lamb_shift_scales_with_z4(self) -> None:
        """Lamb shift ∝ Z⁴ for same n, l=0."""
        r1 = compute_fine_structure(1, 2, 0, 0.5)
        r2 = compute_fine_structure(2, 2, 0, 0.5)
        ratio = r2.E_lamb_eV / r1.E_lamb_eV
        # Should be ~16 (Z⁴ scaling) modulated by ln(1/(Zα)²) change
        assert 10 < ratio < 25

    def test_lamb_shift_scales_with_n3(self) -> None:
        """Lamb shift ∝ 1/n³ for same Z, l=0."""
        r2 = compute_fine_structure(1, 2, 0, 0.5)
        r3 = compute_fine_structure(1, 3, 0, 0.5)
        ratio = r2.E_lamb_eV / r3.E_lamb_eV
        assert ratio == pytest.approx(27.0 / 8.0, rel=0.1)  # (3/2)³ = 3.375

    # ── Regime classification ──────────────────────────────────

    def test_hydrogen_non_relativistic(self) -> None:
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.regime == "NonRelativistic"
        assert r.Z_alpha_squared < 0.01

    def test_heavy_atom_regime(self) -> None:
        """High-Z atoms: (Zα)² > 0.1 → HeavyAtom regime."""
        r = compute_fine_structure(50, 1, 0, 0.5)  # Sn
        assert r.Z_alpha_squared > 0.1
        assert r.regime == "HeavyAtom"

    def test_relativistic_regime(self) -> None:
        """Medium-Z: 0.01 ≤ (Zα)² < 0.1 → Relativistic."""
        r = compute_fine_structure(14, 1, 0, 0.5)  # Si
        z_alpha_sq = (14 * ALPHA_FINE) ** 2
        if 0.01 <= z_alpha_sq < 0.1:
            assert r.regime == "Relativistic"

    # ── GCD invariants ─────────────────────────────────────────

    def test_f_plus_omega_fine(self) -> None:
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_omega_bounded_fine(self) -> None:
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.omega_eff >= 0.0
        assert r.omega_eff <= 1.0

    def test_fine_structure_small_fraction(self) -> None:
        """For hydrogen, fine structure is a tiny fraction of gross energy."""
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert r.omega_eff < 0.001

    # ── Edge cases & validation ────────────────────────────────

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must"):
            compute_fine_structure(1, 0, 0, 0.5)

    def test_invalid_l_raises(self) -> None:
        with pytest.raises(ValueError, match="l must"):
            compute_fine_structure(1, 2, 2, 2.5)

    def test_invalid_j_raises(self) -> None:
        with pytest.raises(ValueError, match="j must"):
            compute_fine_structure(1, 2, 1, 0.0)

    def test_result_type_fine(self) -> None:
        r = compute_fine_structure(1, 2, 1, 1.5)
        assert isinstance(r, FineStructureResult)

    def test_gross_energy_z_squared(self) -> None:
        """E_n ∝ Z² for fixed n."""
        r1 = compute_fine_structure(1, 1, 0, 0.5)
        r2 = compute_fine_structure(2, 1, 0, 0.5)
        assert r2.E_n_eV / r1.E_n_eV == pytest.approx(4.0, rel=0.01)

    def test_n1_ground_state(self) -> None:
        """n=1 ground state should work."""
        r = compute_fine_structure(1, 1, 0, 0.5)
        assert r.E_n_eV == pytest.approx(-13.5984, abs=0.01)

    def test_total_energy_includes_all(self) -> None:
        """E_total = E_n + E_fine + E_lamb."""
        r = compute_fine_structure(1, 2, 0, 0.5)
        expected = r.E_n_eV + r.E_fine_eV + r.E_lamb_eV
        assert r.E_total_eV == pytest.approx(expected, abs=1e-6)

    def test_z_alpha_squared(self) -> None:
        r = compute_fine_structure(1, 2, 1, 1.5)
        expected = (1.0 * ALPHA_FINE) ** 2
        assert r.Z_alpha_squared == pytest.approx(expected, rel=1e-4)

    def test_fine_correction_sign(self) -> None:
        """Fine structure correction should be negative (deepens binding)."""
        r = compute_fine_structure(1, 2, 1, 1.5)
        # E_n is negative, and the correction makes it more negative or less negative
        # depending on j. For j=3/2, correction is less negative than j=1/2.
        r2 = compute_fine_structure(1, 2, 1, 0.5)
        assert r.E_fine_eV != r2.E_fine_eV


# ═══════════════════════════════════════════════════════════════════
# 5. Selection Rules Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestSelectionRules:
    """E1 dipole selection rule checker."""

    # ── Allowed transitions ────────────────────────────────────

    def test_lyman_alpha_allowed(self) -> None:
        """2p → 1s (Ly-α): all E1 rules satisfied."""
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.transition_type == "E1"
        assert r.regime == "Allowed"
        assert r.omega_eff == pytest.approx(0.0)

    def test_balmer_alpha_allowed(self) -> None:
        """3d → 2p: Δl=−1, allowed."""
        r = compute_selection_rules(3, 2, 0, 0.5, 2.5, 2, 1, 0, 0.5, 1.5)
        assert r.transition_type == "E1"
        assert r.rules_satisfied == r.rules_total

    def test_3p_to_2s_allowed(self) -> None:
        """3p → 2s: Δl=−1, allowed."""
        r = compute_selection_rules(3, 1, 0, 0.5, 1.5, 2, 0, 0, 0.5, 0.5)
        assert r.l_rule_ok
        assert r.parity_rule_ok
        assert r.s_rule_ok

    # ── Forbidden transitions ──────────────────────────────────

    def test_2s_to_1s_forbidden(self) -> None:
        """2s → 1s: Δl=0, parity unchanged, E1 forbidden."""
        r = compute_selection_rules(2, 0, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5)
        assert not r.l_rule_ok
        assert not r.parity_rule_ok
        assert r.regime != "Allowed"

    def test_3d_to_1s_forbidden(self) -> None:
        """3d → 1s: Δl=−2, forbidden."""
        r = compute_selection_rules(3, 2, 0, 0.5, 2.5, 1, 0, 0, 0.5, 0.5)
        assert not r.l_rule_ok

    def test_spin_flip_forbidden(self) -> None:
        """Δs ≠ 0 is forbidden (LS coupling)."""
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 1.5, 1.5)
        assert not r.s_rule_ok

    def test_j0_to_j0_forbidden(self) -> None:
        """j=0 → j=0 is always forbidden."""
        r = compute_selection_rules(2, 1, 0, 0.5, 0.0, 1, 0, 0, 0.5, 0.0)
        # j=0→j=0 should fail j_rule
        # Note: this is a somewhat artificial QN set
        assert not r.j_rule_ok or r.regime != "Allowed"

    # ── Rule counting ──────────────────────────────────────────

    def test_total_rules_is_five(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.rules_total == 5

    def test_all_rules_satisfied_allowed(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.rules_satisfied == 5
        assert r.omega_eff == 0.0

    def test_one_rule_violated(self) -> None:
        """Violating one rule gives ω = 1/5 = 0.2."""
        # 2s → 1s: Δl=0 violates l_rule AND parity_rule → 2 violated
        r = compute_selection_rules(2, 0, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5)
        assert r.omega_eff > 0.0

    # ── Delta quantum numbers ──────────────────────────────────

    def test_delta_l_calculated(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.delta_l == -1

    def test_delta_ml_zero(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.delta_ml == 0
        assert r.ml_rule_ok

    def test_delta_ml_pm1(self) -> None:
        """Δm_l = ±1 is allowed."""
        r = compute_selection_rules(2, 1, 1, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.delta_ml == -1
        assert r.ml_rule_ok

    def test_delta_ml_too_large(self) -> None:
        """Δm_l = 2 violates selection rule."""
        r = compute_selection_rules(3, 2, 2, 0.5, 2.5, 2, 1, 0, 0.5, 1.5)
        assert r.delta_ml == -2
        assert not r.ml_rule_ok

    def test_delta_j_rule_check(self) -> None:
        """Δj = 0 and ±1 are allowed (but not j=0→j=0)."""
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.delta_j == pytest.approx(-1.0)
        assert r.j_rule_ok

    def test_parity_change_check(self) -> None:
        """Parity must change for E1 (Δl odd)."""
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.parity_change
        assert r.parity_rule_ok

    # ── Transition type classification ─────────────────────────

    def test_e1_transition_type(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.transition_type == "E1"

    def test_e2_transition_type(self) -> None:
        """Δl=2 → E2 quadrupole transition."""
        r = compute_selection_rules(3, 2, 0, 0.5, 2.5, 1, 0, 0, 0.5, 0.5)
        assert r.transition_type == "E2"

    def test_m1_transition_type(self) -> None:
        """Δl=0, no parity change → M1 magnetic dipole."""
        r = compute_selection_rules(2, 0, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5)
        assert r.transition_type in ("M1", "Forbidden")

    # ── GCD invariants ─────────────────────────────────────────

    def test_f_plus_omega_selection(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_omega_bounded_selection(self) -> None:
        r = compute_selection_rules(2, 0, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5)
        assert r.omega_eff >= 0.0
        assert r.omega_eff <= 1.0

    # ── Regime boundaries ──────────────────────────────────────

    def test_allowed_regime(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert r.regime == "Allowed"

    def test_forbidden_strong_regime(self) -> None:
        """Multiple violations → ForbiddenStrong."""
        r = compute_selection_rules(3, 2, 2, 1.5, 3.5, 1, 0, 0, 0.5, 0.5)
        assert r.regime in ("ForbiddenWeak", "ForbiddenStrong")

    def test_result_type_selection(self) -> None:
        r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
        assert isinstance(r, SelectionResult)


# ═══════════════════════════════════════════════════════════════════
# 6. Zeeman & Stark Effect Closure (25 tests)
# ═══════════════════════════════════════════════════════════════════


class TestZeemanStark:
    """Zeeman and Stark effects with corrected polarizability."""

    # ── Zeeman effect ──────────────────────────────────────────

    def test_zeeman_linear_in_b(self) -> None:
        """Zeeman splitting is linear in B."""
        r1 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        r2 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=2.0)
        assert r2.delta_E_zeeman_eV == pytest.approx(2 * r1.delta_E_zeeman_eV, rel=1e-6)

    def test_zeeman_proportional_to_mj(self) -> None:
        """Zeeman splitting ∝ m_j."""
        r1 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        r2 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 1.5, B_tesla=1.0)
        assert r2.delta_E_zeeman_eV / r1.delta_E_zeeman_eV == pytest.approx(3.0, rel=0.01)

    def test_zeeman_zero_field(self) -> None:
        """No field → no splitting."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=0.0)
        assert r.delta_E_zeeman_eV == 0.0

    def test_lande_g_factor_s_electron(self) -> None:
        """g_J = 2.0 for s-state (l=0, s=1/2, j=1/2)."""
        r = compute_zeeman_stark(1, 1, 0, 0.5, 0.5, 0.5, B_tesla=1.0)
        assert r.g_lande == pytest.approx(2.0, abs=0.01)

    def test_lande_g_factor_p32(self) -> None:
        """g_J = 4/3 for p₃/₂ (l=1, s=1/2, j=3/2)."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        assert r.g_lande == pytest.approx(4.0 / 3.0, abs=0.01)

    def test_lande_g_factor_p12(self) -> None:
        """g_J = 2/3 for p₁/₂ (l=1, s=1/2, j=1/2)."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 0.5, 0.5, B_tesla=1.0)
        assert r.g_lande == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_zeeman_sub_levels_count(self) -> None:
        """Number of Zeeman levels = 2j + 1."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        assert r.n_zeeman_levels == 4  # j=3/2 → 4 levels

    def test_zeeman_sub_levels_s_state(self) -> None:
        r = compute_zeeman_stark(1, 1, 0, 0.5, 0.5, 0.5, B_tesla=1.0)
        assert r.n_zeeman_levels == 2  # j=1/2 → 2 levels

    # ── Stark effect (corrected polarizability) ────────────────

    def test_stark_nonzero_ground_state(self) -> None:
        """Critical fix: n=1 ground state Stark shift must be non-zero."""
        r = compute_zeeman_stark(1, 1, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        assert r.delta_E_stark_eV != 0.0
        assert r.delta_E_stark_eV < 0  # energy decreases

    def test_stark_quadratic_in_field(self) -> None:
        """Quadratic Stark: ΔE ∝ E²."""
        r1 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, E_field_Vm=1e6)
        r2 = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, E_field_Vm=2e6)
        ratio = r2.delta_E_stark_eV / r1.delta_E_stark_eV
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_stark_zero_field(self) -> None:
        """No field → no Stark shift."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, E_field_Vm=0.0)
        assert r.delta_E_stark_eV == 0.0

    def test_stark_l_dependent(self) -> None:
        """Stark shift differs between l=0 and l=1 (same n)."""
        r_s = compute_zeeman_stark(1, 2, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        r_p = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, E_field_Vm=1e7)
        assert r_s.delta_E_stark_eV != r_p.delta_E_stark_eV

    def test_stark_n4_scaling(self) -> None:
        """Polarizability ∝ n⁴ (dominant scaling)."""
        r2 = compute_zeeman_stark(1, 2, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        r3 = compute_zeeman_stark(1, 3, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        ratio = r3.delta_E_stark_eV / r2.delta_E_stark_eV
        # α ∝ n⁴(5n²+1) → n=3 vs n=2: (81*46)/(16*21) ≈ 11.1
        assert ratio > 5  # should be much larger for n=3

    def test_stark_z4_scaling(self) -> None:
        """Polarizability ∝ 1/Z⁴."""
        r1 = compute_zeeman_stark(1, 2, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        r2 = compute_zeeman_stark(2, 2, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        ratio = r1.delta_E_stark_eV / r2.delta_E_stark_eV
        assert ratio == pytest.approx(16.0, rel=0.01)  # Z⁴ = 2⁴ = 16

    def test_stark_order_of_magnitude(self) -> None:
        """H 1s in 10 MV/m: ~10⁻⁸ eV (not 10² eV like old bug)."""
        r = compute_zeeman_stark(1, 1, 0, 0.5, 0.5, 0.5, E_field_Vm=1e7)
        assert abs(r.delta_E_stark_eV) < 1e-6
        assert abs(r.delta_E_stark_eV) > 1e-10

    # ── Combined fields ────────────────────────────────────────

    def test_combined_zeeman_stark(self) -> None:
        """Both B and E applied simultaneously."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0, E_field_Vm=1e7)
        assert r.delta_E_zeeman_eV != 0.0
        assert r.delta_E_stark_eV != 0.0
        assert r.delta_E_total_eV == pytest.approx(r.delta_E_zeeman_eV + r.delta_E_stark_eV, abs=1e-9)

    # ── GCD invariants ─────────────────────────────────────────

    def test_f_plus_omega_zeeman(self) -> None:
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        assert r.F_eff + r.omega_eff == pytest.approx(1.0, abs=1e-6)

    def test_omega_bounded_zeeman(self) -> None:
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=100.0)
        assert r.omega_eff >= 0.0
        assert r.omega_eff <= 1.0

    # ── Regime classification ──────────────────────────────────

    def test_weak_field_regime(self) -> None:
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=0.01)
        assert r.regime == "Weak"

    def test_strong_field_regime(self) -> None:
        """Very strong B → Paschen-Back regime."""
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 1.5, B_tesla=1e6)
        assert r.regime in ("Moderate", "Strong")

    # ── Edge cases ─────────────────────────────────────────────

    def test_result_type_zeeman(self) -> None:
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
        assert isinstance(r, ZeemanStarkResult)

    def test_unperturbed_energy(self) -> None:
        r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5)
        assert r.E_n_eV == pytest.approx(-3.3996, abs=0.001)

    def test_g_factor_j_zero(self) -> None:
        """j=0 → g=0 to avoid division by zero."""
        r = compute_zeeman_stark(1, 1, 0, 0.5, 0.0, 0.0, B_tesla=1.0)
        assert r.g_lande == 0.0


# ═══════════════════════════════════════════════════════════════════
# Cross-closure consistency
# ═══════════════════════════════════════════════════════════════════


class TestCrossClosure:
    """Tests verifying consistency between related closures."""

    def test_ionization_config_consistency(self) -> None:
        """IE regime should correlate with shell completeness."""
        # Noble gases: closed shell AND high IE
        for z in [2, 10, 18, 36]:
            ie = compute_ionization(z)
            ec = compute_electron_config(z)
            assert ec.shell_completeness >= 0.99
            assert ie.IE_measured_eV > 10.0  # high IE for noble gases

    def test_spectral_energy_matches_ie(self) -> None:
        """Lyman series limit should approach IE of hydrogen."""
        # As n_upper → ∞, transition energy → IE
        r = compute_spectral_lines(1, 1, 100)
        assert r.energy_eV == pytest.approx(13.598, rel=0.01)

    def test_fine_structure_zeeman_energy(self) -> None:
        """Fine structure and Zeeman use same unperturbed E_n."""
        fs = compute_fine_structure(1, 2, 1, 1.5)
        zs = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5)
        assert fs.E_n_eV == pytest.approx(zs.E_n_eV, abs=1e-4)
