"""
Tests for GCD closure modules: energy_potential, entropic_collapse, field_resonance, generative_flux.

All four files were previously completely untested.
"""

from __future__ import annotations

import math

import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/gcd/energy_potential.py
# ═══════════════════════════════════════════════════════════════════


class TestEnergyPotential:
    """Unit tests for compute_energy_potential."""

    def test_basic_computation(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.1, S=0.05, C=0.1)
        assert "E_potential" in result
        assert "E_collapse" in result
        assert "E_entropy" in result
        assert "E_curvature" in result
        assert "regime" in result

    def test_zero_inputs(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.0, S=0.0, C=0.0)
        assert result["E_potential"] == 0.0
        assert result["regime"] == "Low"

    def test_collapse_component(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.5, S=0.0, C=0.0)
        assert result["E_collapse"] == pytest.approx(0.25)
        assert result["E_entropy"] == 0.0
        assert result["E_curvature"] == 0.0

    def test_entropy_component(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.0, S=0.5, C=0.0, alpha=2.0)
        assert result["E_entropy"] == pytest.approx(1.0)

    def test_curvature_component(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.0, S=0.0, C=0.5, beta=2.0)
        assert result["E_curvature"] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "omega,S,C,expected_regime",
        [
            (0.0, 0.0, 0.0, "Low"),
            (0.1, 0.0, 0.0, "Medium"),
            (0.5, 0.5, 0.5, "High"),
        ],
    )
    def test_regime_classification(self, omega, S, C, expected_regime):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=omega, S=S, C=C)
        assert result["regime"] == expected_regime

    def test_invalid_omega(self):
        from closures.gcd.energy_potential import compute_energy_potential

        with pytest.raises(ValueError):
            compute_energy_potential(omega=1.5, S=0.1, C=0.1)

    def test_invalid_S(self):
        from closures.gcd.energy_potential import compute_energy_potential

        with pytest.raises(ValueError):
            compute_energy_potential(omega=0.1, S=-0.1, C=0.1)

    def test_invalid_C(self):
        from closures.gcd.energy_potential import compute_energy_potential

        with pytest.raises(ValueError):
            compute_energy_potential(omega=0.1, S=0.1, C=1.5)

    def test_additivity(self):
        from closures.gcd.energy_potential import compute_energy_potential

        result = compute_energy_potential(omega=0.3, S=0.2, C=0.4)
        total = result["E_collapse"] + result["E_entropy"] + result["E_curvature"]
        assert result["E_potential"] == pytest.approx(total)


# ═══════════════════════════════════════════════════════════════════
# closures/gcd/entropic_collapse.py
# ═══════════════════════════════════════════════════════════════════


class TestEntropicCollapse:
    """Unit tests for compute_entropic_collapse."""

    def test_basic_computation(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        result = compute_entropic_collapse(S=0.5, F=0.8, tau_R=5.0)
        assert "phi_collapse" in result
        assert "S_contribution" in result
        assert "F_contribution" in result
        assert "tau_damping" in result
        assert "regime" in result

    def test_formula_correctness(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        S, F, tau_R, tau_0 = 0.6, 0.7, 3.0, 10.0
        result = compute_entropic_collapse(S=S, F=F, tau_R=tau_R, tau_0=tau_0)
        expected = S * (1.0 - F) * math.exp(-tau_R / tau_0)
        assert result["phi_collapse"] == pytest.approx(expected)

    def test_high_fidelity_low_collapse(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        result = compute_entropic_collapse(S=0.5, F=1.0, tau_R=1.0)
        assert result["phi_collapse"] == pytest.approx(0.0)

    def test_zero_entropy_no_collapse(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        result = compute_entropic_collapse(S=0.0, F=0.5, tau_R=1.0)
        assert result["phi_collapse"] == pytest.approx(0.0)

    def test_large_tau_R_dampens(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        result = compute_entropic_collapse(S=0.8, F=0.2, tau_R=1000.0)
        assert result["phi_collapse"] < 1e-10

    @pytest.mark.parametrize("regime_label", ["Minimal", "Active", "Critical"])
    def test_regime_labels_exist(self, regime_label):
        """At least one input should produce each regime label."""
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        configs = {
            "Minimal": (0.1, 0.9, 50.0),
            "Active": (0.2, 0.8, 10.0),
            "Critical": (0.9, 0.1, 0.1),
        }
        S, F, tau_R = configs[regime_label]
        result = compute_entropic_collapse(S=S, F=F, tau_R=tau_R)
        assert result["regime"] == regime_label

    def test_invalid_S_raises(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        with pytest.raises(ValueError):
            compute_entropic_collapse(S=-0.1, F=0.5, tau_R=1.0)

    def test_invalid_F_raises(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        with pytest.raises(ValueError):
            compute_entropic_collapse(S=0.5, F=1.5, tau_R=1.0)

    def test_invalid_tau_R_raises(self):
        from closures.gcd.entropic_collapse import compute_entropic_collapse

        with pytest.raises(ValueError):
            compute_entropic_collapse(S=0.5, F=0.5, tau_R=-1.0)


# ═══════════════════════════════════════════════════════════════════
# closures/gcd/field_resonance.py
# ═══════════════════════════════════════════════════════════════════


class TestFieldResonance:
    """Unit tests for compute_field_resonance."""

    def test_basic_computation(self):
        from closures.gcd.field_resonance import compute_field_resonance

        result = compute_field_resonance(omega=0.1, S=0.2, C=0.1)
        assert "resonance" in result
        assert "coherence_factor" in result
        assert "order_factor" in result
        assert "curvature_damping" in result
        assert "regime" in result

    def test_perfect_resonance(self):
        from closures.gcd.field_resonance import compute_field_resonance

        result = compute_field_resonance(omega=0.0, S=0.0, C=0.0)
        assert result["resonance"] == pytest.approx(1.0)
        assert result["regime"] == "Coherent"

    def test_formula_correctness(self):
        from closures.gcd.field_resonance import compute_field_resonance

        omega, S, C, C_crit = 0.3, 0.4, 0.1, 0.2
        result = compute_field_resonance(omega=omega, S=S, C=C, C_crit=C_crit)
        expected = (1 - abs(omega)) * (1 - S) * math.exp(-C / C_crit)
        assert result["resonance"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "omega,S,C,expected_regime",
        [
            (0.0, 0.0, 0.0, "Coherent"),
            (0.5, 0.5, 0.1, "Decoupled"),
            (0.9, 0.9, 1.0, "Decoupled"),
        ],
    )
    def test_regime_classification(self, omega, S, C, expected_regime):
        from closures.gcd.field_resonance import compute_field_resonance

        result = compute_field_resonance(omega=omega, S=S, C=C)
        assert result["regime"] == expected_regime

    def test_high_curvature_dampens(self):
        from closures.gcd.field_resonance import compute_field_resonance

        result = compute_field_resonance(omega=0.0, S=0.0, C=10.0)
        assert result["resonance"] < 0.01

    def test_invalid_omega_raises(self):
        from closures.gcd.field_resonance import compute_field_resonance

        with pytest.raises(ValueError):
            compute_field_resonance(omega=1.5, S=0.1, C=0.1)

    def test_invalid_S_raises(self):
        from closures.gcd.field_resonance import compute_field_resonance

        with pytest.raises(ValueError):
            compute_field_resonance(omega=0.1, S=-0.1, C=0.1)


# ═══════════════════════════════════════════════════════════════════
# closures/gcd/generative_flux.py
# ═══════════════════════════════════════════════════════════════════


class TestGenerativeFlux:
    """Unit tests for compute_generative_flux."""

    def test_basic_computation(self):
        from closures.gcd.generative_flux import compute_generative_flux

        result = compute_generative_flux(kappa=-0.5, IC=0.6, C=0.2)
        assert "phi_gen" in result or "Phi_gen" in result or "flux" in result

    def test_returns_dict(self):
        from closures.gcd.generative_flux import compute_generative_flux

        result = compute_generative_flux(kappa=-1.0, IC=0.5, C=0.3)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "kappa,IC,C",
        [
            (-0.01, 0.99, 0.01),
            (-0.5, 0.6, 0.2),
            (-2.0, 0.13, 0.8),
            (-5.0, 0.007, 0.95),
        ],
    )
    def test_multiple_inputs(self, kappa, IC, C):
        from closures.gcd.generative_flux import compute_generative_flux

        result = compute_generative_flux(kappa=kappa, IC=IC, C=C)
        assert isinstance(result, dict)
        assert len(result) > 0
