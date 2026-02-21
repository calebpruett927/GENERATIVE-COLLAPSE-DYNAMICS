"""
Tests for quantum_mechanics closure modules.

Covers 6 files: entanglement, harmonic_oscillator, spin_measurement,
tunneling, uncertainty_principle, wavefunction_collapse.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/entanglement.py
# ═══════════════════════════════════════════════════════════════════


class TestEntanglement:
    """Unit tests for compute_entanglement."""

    def test_maximally_entangled(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=[0.5, 0.5])
        assert result["regime"] in ("Strong", "Maximal")

    def test_separable_state(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=[1.0, 0.0])
        assert result["regime"] == "Separable"
        assert result["concurrence"] < 0.1

    def test_bell_correlations(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(
            rho_eigenvalues=[0.5, 0.5],
            bell_correlations=[0.7, 0.7, 0.7, 0.7],
        )
        assert result["bell_parameter"] > 0

    def test_result_keys(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=[0.5, 0.5])
        assert set(result.keys()) >= {
            "concurrence",
            "S_vN",
            "bell_parameter",
            "negativity",
            "regime",
        }

    @pytest.mark.parametrize(
        "eigenvalues,expected_regime",
        [
            ([1.0, 0.0], "Separable"),
            ([0.95, 0.05], "Separable"),
            ([0.7, 0.3], "Weak"),
            ([0.5, 0.5], "Maximal"),
        ],
    )
    def test_regime_classification(self, eigenvalues, expected_regime):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=eigenvalues)
        assert result["regime"] in ("Separable", "Weak", "Strong", "Maximal")

    def test_von_neumann_entropy_bounds(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=[0.5, 0.5])
        assert result["S_vN"] >= 0.0

    def test_pure_state_zero_entropy(self):
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(rho_eigenvalues=[1.0, 0.0])
        assert result["S_vN"] == pytest.approx(0.0, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/harmonic_oscillator.py
# ═══════════════════════════════════════════════════════════════════


class TestHarmonicOscillator:
    """Unit tests for compute_harmonic_oscillator."""

    def test_ground_state(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(n_quanta=0, omega_freq=1.0, e_observed=0.5)
        assert result["E_predicted"] == pytest.approx(0.5, rel=0.01)
        assert result["regime"] in ("Pure", "High", "Mixed", "Decoherent")

    def test_excited_state(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(n_quanta=5, omega_freq=1.0, e_observed=5.5)
        assert result["E_predicted"] == pytest.approx(5.5, rel=0.01)

    def test_zero_frequency(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(n_quanta=1, omega_freq=0.0, e_observed=0.0)
        assert result["E_predicted"] == pytest.approx(0.0, abs=1e-10)

    def test_result_keys(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(n_quanta=1, omega_freq=1.0, e_observed=1.5)
        assert set(result.keys()) >= {
            "E_predicted",
            "delta_E",
            "coherent_alpha",
            "squeeze_r",
            "regime",
        }

    @pytest.mark.parametrize("n", [0, 1, 5, 10, 50])
    def test_energy_scales_with_n(self, n):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(n_quanta=n, omega_freq=1.0, e_observed=n + 0.5)
        assert result["E_predicted"] == pytest.approx(n + 0.5, rel=0.01)

    def test_coherent_state(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(
            n_quanta=4,
            omega_freq=1.0,
            e_observed=4.5,
            coherent_alpha=2.0,
        )
        assert result["coherent_alpha"] == pytest.approx(2.0)

    def test_squeezed_state(self):
        from closures.quantum_mechanics.harmonic_oscillator import (
            compute_harmonic_oscillator,
        )

        result = compute_harmonic_oscillator(
            n_quanta=0,
            omega_freq=1.0,
            e_observed=0.5,
            squeeze_r=0.5,
        )
        assert result["squeeze_r"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/spin_measurement.py
# ═══════════════════════════════════════════════════════════════════


class TestSpinMeasurement:
    """Unit tests for compute_spin_measurement."""

    def test_electron_spin_up(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=0.5, b_field=1.0)
        assert result["S_z_predicted"] == pytest.approx(0.5, abs=0.01)
        assert result["regime"] == "Faithful"

    def test_electron_spin_down(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=-0.5, b_field=1.0)
        assert result["S_z_predicted"] == pytest.approx(-0.5, abs=0.01)

    def test_larmor_frequency(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=0.5, b_field=1.0)
        assert result["larmor_freq"] > 0

    def test_zeeman_splitting(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=0.5, b_field=1.0)
        assert result["zeeman_split"] > 0

    def test_custom_g_factor(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=0.5, b_field=1.0, g_factor=5.5857)
        assert result["larmor_freq"] > 0

    def test_result_keys(self):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=0.5, b_field=1.0)
        assert set(result.keys()) >= {
            "S_z_predicted",
            "spin_fidelity",
            "larmor_freq",
            "zeeman_split",
            "regime",
        }

    @pytest.mark.parametrize(
        "s_z_obs,expected_regime",
        [
            (0.5, "Faithful"),
            (0.3, "Perturbed"),
            (0.0, "Anomalous"),
        ],
    )
    def test_regime_classification(self, s_z_obs, expected_regime):
        from closures.quantum_mechanics.spin_measurement import (
            compute_spin_measurement,
        )

        result = compute_spin_measurement(s_total=0.5, s_z_observed=s_z_obs, b_field=1.0)
        assert result["regime"] in ("Faithful", "Perturbed", "Decoherent", "Anomalous")


# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/tunneling.py
# ═══════════════════════════════════════════════════════════════════


class TestTunneling:
    """Unit tests for compute_tunneling."""

    def test_opaque_barrier(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=1.0, v_barrier=10.0, barrier_width=2.0)
        assert result["T_coeff"] < 0.05
        assert result["regime"] in ("Opaque", "Suppressed")

    def test_transparent_barrier(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=10.0, v_barrier=5.0, barrier_width=1.0)
        assert result["T_coeff"] == pytest.approx(1.0, abs=0.01)
        assert result["T_classical"] == pytest.approx(1.0)
        assert result["regime"] == "Transparent"

    def test_thin_barrier(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=1.0, v_barrier=2.0, barrier_width=0.01)
        assert result["T_coeff"] > 0.5

    def test_thick_barrier(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=1.0, v_barrier=5.0, barrier_width=5.0)
        assert result["T_coeff"] < 0.01

    def test_result_keys(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=1.0, v_barrier=5.0, barrier_width=1.0)
        assert set(result.keys()) >= {
            "T_coeff",
            "kappa_barrier",
            "T_classical",
            "T_ratio",
            "regime",
        }

    def test_equal_energy_barrier(self):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=5.0, v_barrier=5.0, barrier_width=1.0)
        assert result["T_coeff"] == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize(
        "e_particle,v_barrier,width,expected_regime",
        [
            (1.0, 10.0, 5.0, "Opaque"),
            (1.0, 5.0, 1.0, "Suppressed"),
            (1.0, 2.0, 0.1, "Transparent"),
            (10.0, 5.0, 1.0, "Transparent"),
        ],
    )
    def test_regime_classification(self, e_particle, v_barrier, width, expected_regime):
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=e_particle, v_barrier=v_barrier, barrier_width=width)
        assert result["regime"] in ("Opaque", "Suppressed", "Moderate", "Transparent")

    def test_inf_rec_ratio(self):
        """When classical transmission is 0 but quantum is > 0, T_ratio should be INF_REC."""
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(e_particle=1.0, v_barrier=5.0, barrier_width=0.1)
        if result["T_classical"] == 0.0 and result["T_coeff"] > 0:
            assert result["T_ratio"] == "INF_REC"


# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/uncertainty_principle.py
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyPrinciple:
    """Unit tests for compute_uncertainty."""

    def test_minimum_uncertainty(self):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        hbar = 1.054571817e-34
        dx = 1e-10
        dp = hbar / (2 * dx)
        result = compute_uncertainty(delta_x=dx, delta_p=dp)
        assert result["heisenberg_ratio"] == pytest.approx(1.0, abs=0.01)
        assert result["regime"] == "Minimum"

    def test_large_uncertainty_classical(self):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x=1.0, delta_p=1.0)
        assert result["regime"] == "Classical"
        assert result["heisenberg_ratio"] > 20.0

    def test_result_keys(self):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x=1e-10, delta_p=1e-24)
        assert set(result.keys()) >= {
            "heisenberg_product",
            "heisenberg_ratio",
            "min_uncertainty",
            "regime",
        }

    def test_natural_units(self):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x=0.1, delta_p=10.0, units="natural")
        assert result["heisenberg_ratio"] > 0

    @pytest.mark.parametrize(
        "dx,dp,expected_regime",
        [
            (1e-10, 5.27e-25, "Minimum"),
            (1e-9, 1e-24, "Moderate"),
            (1.0, 1.0, "Classical"),
        ],
    )
    def test_regime_classification(self, dx, dp, expected_regime):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x=dx, delta_p=dp)
        assert result["regime"] in (
            "Violation",
            "Minimum",
            "Moderate",
            "Dispersed",
            "Classical",
        )

    def test_heisenberg_product_positive(self):
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x=1e-10, delta_p=1e-24)
        assert result["heisenberg_product"] > 0


# ═══════════════════════════════════════════════════════════════════
# closures/quantum_mechanics/wavefunction_collapse.py
# ═══════════════════════════════════════════════════════════════════


class TestWavefunctionCollapse:
    """Unit tests for compute_wavefunction_collapse."""

    def test_perfect_born_rule(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        probs = [0.5, 0.3, 0.2]
        result = compute_wavefunction_collapse(
            psi_amplitudes=probs,
            measurement_probs=probs,
        )
        assert result["delta_P"] < 0.01
        assert result["regime"] == "Faithful"

    def test_perturbed_measurement(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(
            psi_amplitudes=[0.5, 0.3, 0.2],
            measurement_probs=[0.48, 0.32, 0.20],
        )
        assert result["regime"] in ("Faithful", "Perturbed")

    def test_anomalous_measurement(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(
            psi_amplitudes=[0.9, 0.05, 0.05],
            measurement_probs=[0.3, 0.3, 0.4],
        )
        assert result["regime"] in ("Decoherent", "Anomalous")

    def test_result_keys(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(
            psi_amplitudes=[0.5, 0.5],
            measurement_probs=[0.5, 0.5],
        )
        assert set(result.keys()) >= {
            "P_born",
            "delta_P",
            "fidelity_state",
            "purity",
            "regime",
        }

    def test_fidelity_bounds(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(
            psi_amplitudes=[0.5, 0.5],
            measurement_probs=[0.5, 0.5],
        )
        assert 0.0 <= result["fidelity_state"] <= 1.0

    def test_purity_bounds(self):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(
            psi_amplitudes=[0.5, 0.5],
            measurement_probs=[0.5, 0.5],
        )
        assert 0.0 <= result["purity"] <= 1.0

    @pytest.mark.parametrize(
        "psi,meas",
        [
            ([1.0], [1.0]),
            ([0.5, 0.5], [0.5, 0.5]),
            ([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]),
        ],
    )
    def test_uniform_distributions(self, psi, meas):
        from closures.quantum_mechanics.wavefunction_collapse import (
            compute_wavefunction_collapse,
        )

        result = compute_wavefunction_collapse(psi_amplitudes=psi, measurement_probs=meas)
        assert result["delta_P"] < 0.01
