"""
Tests for new GCD and RCFT closures added in performance/extension update.
"""

import numpy as np
import pytest


class TestMomentumFlux:
    """Tests for GCD momentum_flux closure."""

    def test_imports(self):
        """Module should import cleanly."""
        from closures.gcd import momentum_flux

        assert hasattr(momentum_flux, "compute_momentum_flux")
        assert hasattr(momentum_flux, "compute_scalar_momentum_flux")

    def test_constant_integrity_neutral(self):
        """Constant integrity should produce neutral regime with zero flux."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        kappa = np.ones(50) * -2.0
        C = np.ones(50) * 0.05
        omega = np.ones(50) * 0.01

        result = compute_momentum_flux(kappa, C, omega)

        assert result["regime"] == "Neutral"
        assert abs(result["mean_flux"]) < 0.01
        assert abs(result["net_integrity_change"]) < 1e-10
        assert result["flux_variance"] < 1e-10

    def test_degrading_integrity(self):
        """Declining integrity should produce negative flux (degrading)."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        kappa = np.linspace(-1.0, -5.0, 50)  # Declining
        C = np.linspace(0.02, 0.15, 50)
        omega = np.linspace(0.01, 0.25, 50)

        result = compute_momentum_flux(kappa, C, omega, dt=0.1)

        assert result["mean_flux"] < 0  # Negative flux (degrading direction)
        assert result["net_integrity_change"] < 0  # Net loss of integrity
        assert result["flux_variance"] > 0  # Non-zero variance

    def test_recovering_integrity(self):
        """Recovering integrity should produce positive flux."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        kappa = np.linspace(-5.0, -1.0, 50)  # Recovering
        C = np.linspace(0.15, 0.02, 50)
        omega = np.linspace(0.25, 0.01, 50)

        result = compute_momentum_flux(kappa, C, omega, dt=0.1)

        assert result["mean_flux"] > 0  # Positive flux (recovery)
        assert result["net_integrity_change"] > 0  # Net gain of integrity

    def test_scalar_version(self):
        """Scalar version should work for single time step."""
        from closures.gcd.momentum_flux import compute_scalar_momentum_flux

        result = compute_scalar_momentum_flux(kappa=-2.5, kappa_prev=-2.0, C=0.08, omega=0.05, dt=1.0)

        assert "phi_momentum" in result
        assert "dkappa_dt" in result
        assert "regime" in result
        assert result["dkappa_dt"] == -0.5  # (−2.5 − (−2.0)) / 1.0

    def test_input_validation(self):
        """Should raise ValueError for invalid inputs."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        kappa = np.ones(50) * -2.0
        C = np.ones(50) * 0.05
        omega = np.ones(50) * 0.01

        # Mismatched shapes
        with pytest.raises(ValueError, match="same shape"):
            compute_momentum_flux(kappa[:30], C, omega)

        # Too few points
        with pytest.raises(ValueError, match="at least 2 points"):
            compute_momentum_flux(kappa[:1], C[:1], omega[:1])

        # Invalid omega
        omega_bad = np.ones(50) * 1.5
        with pytest.raises(ValueError, match="must be in"):
            compute_momentum_flux(kappa, C, omega_bad)

        # Negative dt
        with pytest.raises(ValueError, match="must be positive"):
            compute_momentum_flux(kappa, C, omega, dt=-1.0)

    def test_vectorized_output_shapes(self):
        """Output arrays should match input length."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        n_points = 100
        kappa = np.linspace(-1.0, -3.0, n_points)
        C = np.ones(n_points) * 0.05
        omega = np.ones(n_points) * 0.01

        result = compute_momentum_flux(kappa, C, omega)

        assert len(result["phi_momentum"]) == n_points
        assert len(result["dkappa_dt"]) == n_points
        assert len(result["curvature_amplification"]) == n_points
        assert len(result["drift_weighting"]) == n_points


class TestAttractorBasin:
    """Tests for RCFT attractor_basin closure."""

    def test_imports(self):
        """Module should import cleanly."""
        from closures.rcft import attractor_basin

        assert hasattr(attractor_basin, "compute_attractor_basin")

    def test_monostable_system(self):
        """Converging trajectory should be classified as monostable."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        t = np.linspace(0, 10, 100)
        omega = 0.05 + 0.03 * np.exp(-t / 2)
        S = 0.10 + 0.05 * np.exp(-t / 2)
        C = 0.03 + 0.02 * np.exp(-t / 2)

        result = compute_attractor_basin(omega, S, C)

        assert result["regime"] in ["Monostable", "Bistable"]  # Should converge
        assert result["n_attractors_found"] >= 1
        assert result["max_basin_strength"] > 0.5  # Dominant basin

    def test_bistable_oscillation(self):
        """Oscillating system should show multiple attractors."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        t = np.linspace(0, 10, 100)
        omega = 0.15 + 0.10 * np.sin(2 * np.pi * t / 5)
        S = 0.20 + 0.08 * np.sin(2 * np.pi * t / 5 + np.pi / 2)
        C = 0.10 + 0.05 * np.sin(2 * np.pi * t / 5)

        result = compute_attractor_basin(omega, S, C, n_attractors=5)

        assert result["regime"] in ["Bistable", "Multistable"]
        assert result["n_attractors_found"] >= 2
        assert len(result["attractor_locations"]) == result["n_attractors_found"]
        assert len(result["basin_strengths"]) == result["n_attractors_found"]
        assert len(result["basin_volumes"]) == result["n_attractors_found"]

    def test_basin_properties(self):
        """Basin strengths and volumes should be properly normalized."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        t = np.linspace(0, 10, 50)
        omega = 0.10 + 0.05 * np.sin(t)
        S = 0.15 + 0.05 * np.cos(t)
        C = 0.08 * np.ones_like(t)

        result = compute_attractor_basin(omega, S, C)

        # Basin strengths should be normalized (allow small numerical error)
        basin_strengths = np.array(result["basin_strengths"])
        assert np.isclose(np.sum(basin_strengths), 1.0, atol=1e-4)
        assert np.all(basin_strengths >= 0)
        assert np.all(basin_strengths <= 1.0)

        # Basin volumes should sum to 1
        basin_volumes = np.array(result["basin_volumes"])
        assert np.isclose(np.sum(basin_volumes), 1.0, atol=1e-6)
        assert np.all(basin_volumes >= 0)
        assert np.all(basin_volumes <= 1.0)

    def test_attractor_locations_format(self):
        """Attractor locations should be in (ω, S, C) format."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        omega = np.linspace(0.05, 0.15, 50)
        S = np.linspace(0.10, 0.20, 50)
        C = np.linspace(0.02, 0.08, 50)

        result = compute_attractor_basin(omega, S, C)

        attractor_locs = result["attractor_locations"]
        assert isinstance(attractor_locs, list)
        assert len(attractor_locs) == result["n_attractors_found"]

        for loc in attractor_locs:
            assert len(loc) == 3  # (ω, S, C)
            assert 0 <= loc[0] <= 1  # ω ∈ [0,1]
            assert loc[1] >= 0  # S ≥ 0
            assert loc[2] >= 0  # C ≥ 0

    def test_input_validation(self):
        """Should raise ValueError for invalid inputs."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        omega = np.ones(50) * 0.10
        S = np.ones(50) * 0.15
        C = np.ones(50) * 0.05

        # Mismatched shapes
        with pytest.raises(ValueError, match="same shape"):
            compute_attractor_basin(omega[:30], S, C)

        # Too few points
        with pytest.raises(ValueError, match="at least 10 points"):
            compute_attractor_basin(omega[:5], S[:5], C[:5])

        # Invalid omega
        omega_bad = np.ones(50) * 1.5
        with pytest.raises(ValueError, match="must be in"):
            compute_attractor_basin(omega_bad, S, C)

        # Negative S
        S_bad = -np.ones(50) * 0.1
        with pytest.raises(ValueError, match="non-negative"):
            compute_attractor_basin(omega, S_bad, C)

    def test_convergence_rates(self):
        """Convergence rates should be computed for all attractors."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        t = np.linspace(0, 5, 100)
        omega = 0.10 * np.exp(-t / 2)
        S = 0.15 * np.exp(-t / 2)
        C = 0.05 * np.exp(-t / 2)

        result = compute_attractor_basin(omega, S, C)

        conv_rates = result["convergence_rates"]
        assert len(conv_rates) == result["n_attractors_found"]
        assert all(isinstance(r, (int, float)) for r in conv_rates)

    def test_trajectory_classification(self):
        """Trajectory classification should assign each point to an attractor."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        n_points = 80
        omega = np.linspace(0.05, 0.15, n_points)
        S = np.linspace(0.10, 0.20, n_points)
        C = np.linspace(0.02, 0.08, n_points)

        result = compute_attractor_basin(omega, S, C)

        traj_class = result["trajectory_classification"]
        assert len(traj_class) == n_points
        assert all(0 <= idx < result["n_attractors_found"] for idx in traj_class)


class TestNewClosuresIntegration:
    """Integration tests for new closures with existing framework."""

    def test_momentum_flux_with_gcd_invariants(self):
        """Momentum flux should work with realistic GCD invariant series."""
        from closures.gcd.momentum_flux import compute_momentum_flux

        # Simulate GCD invariant evolution
        n = 100
        t = np.linspace(0, 10, n)
        omega = 0.05 + 0.10 * (1 - np.exp(-t / 3))
        F = 1 - omega
        C = 0.05 + 0.03 * omega

        # Compute kappa from F (using log-safe approximation)
        eps = 1e-10
        kappa = np.log(F + eps)

        result = compute_momentum_flux(kappa, C, omega, dt=t[1] - t[0])

        assert "regime" in result
        assert result["net_integrity_change"] < 0  # System degrading
        assert len(result["phi_momentum"]) == n

    def test_attractor_basin_with_rcft_metrics(self):
        """Attractor basin should work with RCFT-augmented trajectories."""
        from closures.rcft.attractor_basin import compute_attractor_basin

        # Simulate trajectory with RCFT characteristics
        n = 100
        t = np.linspace(0, 10, n)

        # Damped oscillation (bistable behavior)
        omega = 0.15 + 0.08 * np.exp(-t / 5) * np.sin(2 * t)
        S = 0.20 + 0.05 * np.exp(-t / 5) * np.cos(2 * t)
        C = 0.10 + 0.03 * np.abs(np.sin(t))

        result = compute_attractor_basin(omega, S, C)

        assert result["n_attractors_found"] >= 1
        assert result["regime"] in ["Monostable", "Bistable", "Multistable"]
        assert "dominant_attractor" in result

    def test_closure_registry_format(self):
        """New closures should be properly registered."""
        from pathlib import Path

        import yaml

        registry_path = Path("closures/registry.yaml")
        assert registry_path.exists()

        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        # Check GCD momentum_flux
        gcd_closures = registry["registry"]["extensions"]["gcd"]
        momentum_found = any(c["name"] == "momentum_flux" for c in gcd_closures)
        assert momentum_found, "momentum_flux not found in GCD registry"

        # Check RCFT attractor_basin
        rcft_closures = registry["registry"]["extensions"]["rcft"]
        attractor_found = any(c["name"] == "attractor_basin" for c in rcft_closures)
        assert attractor_found, "attractor_basin not found in RCFT registry"
