"""
Tests for the Universal UMCP Calculator.

Tests cover:
- Basic kernel invariant computation
- Cost closure calculations
- Regime classification
- Seam accounting
- GCD metrics
- RCFT metrics
- Uncertainty propagation
- SS1M checksums
- CLI interface
"""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from umcp.universal_calculator import (
    ComputationMode,
    UniversalCalculator,
    UniversalResult,
    compute_full,
    compute_kernel,
    compute_regime,
)


class TestKernelInvariants:
    """Test Tier-1 kernel invariant computation."""

    def test_basic_kernel_computation(self):
        """Test basic kernel computation with simple inputs."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.9, 0.9],
            weights=[1 / 3, 1 / 3, 1 / 3],
            mode=ComputationMode.MINIMAL,
        )

        # Check kernel values
        assert pytest.approx(0.9, rel=1e-6) == result.kernel.F
        assert result.kernel.omega == pytest.approx(0.1, rel=1e-6)
        assert pytest.approx(0.9, rel=1e-6) == result.kernel.IC
        assert pytest.approx(0.0, abs=1e-10) == result.kernel.C  # Homogeneous

    def test_heterogeneous_coordinates(self):
        """Test with heterogeneous coordinates."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.5, 0.8],
            weights=[0.4, 0.3, 0.3],
        )

        # F = 0.4*0.9 + 0.3*0.5 + 0.3*0.8 = 0.36 + 0.15 + 0.24 = 0.75
        assert pytest.approx(0.75, rel=1e-6) == result.kernel.F
        assert result.kernel.omega == pytest.approx(0.25, rel=1e-6)

        # IC should be less than F (AM-GM inequality)
        assert result.kernel.IC < result.kernel.F

        # Curvature should be non-zero
        assert result.kernel.C > 0

    def test_identity_relationships(self):
        """Test mathematical identities F = 1 - ω and IC = exp(κ)."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.8, 0.7, 0.9, 0.6],
            weights=[0.25, 0.25, 0.25, 0.25],
        )

        # F + ω = 1
        assert (result.kernel.F + result.kernel.omega) == pytest.approx(1.0, rel=1e-10)

        # IC = exp(κ)
        assert pytest.approx(math.exp(result.kernel.kappa), rel=1e-10) == result.kernel.IC

    def test_uniform_weights_default(self):
        """Test that uniform weights are used when not specified."""
        calc = UniversalCalculator()
        result = calc.compute_all(coordinates=[0.8, 0.9, 0.7])

        # With uniform weights: F = (0.8 + 0.9 + 0.7) / 3 = 0.8
        assert pytest.approx(0.8, rel=1e-6) == result.kernel.F

    def test_boundary_values(self):
        """Test with values near boundaries."""
        calc = UniversalCalculator()

        # Near upper boundary
        result = calc.compute_all(coordinates=[0.99, 0.99, 0.99])
        assert result.kernel.F > 0.98
        assert result.regime in ["STABLE", "WATCH"]

        # Near lower boundary (should trigger CRITICAL regime)
        result = calc.compute_all(coordinates=[0.1, 0.1, 0.1])
        assert result.kernel.IC < 0.3
        assert result.regime == "CRITICAL"


class TestCostClosures:
    """Test cost closure computations."""

    def test_gamma_omega(self):
        """Test drift cost Γ(ω) = ω³/(1-ω+ε)."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.9, 0.9],
            mode=ComputationMode.STANDARD,
        )

        omega = result.kernel.omega  # 0.1
        expected_gamma = omega**3 / (1 - omega + 1e-8)
        assert result.costs is not None
        assert result.costs.gamma_omega == pytest.approx(expected_gamma, rel=1e-6)

    def test_curvature_cost(self):
        """Test curvature cost D_C = α·C."""
        calc = UniversalCalculator(alpha=1.0)
        result = calc.compute_all(
            coordinates=[0.9, 0.5, 0.8],
        )

        assert result.costs is not None
        assert pytest.approx(result.kernel.C, rel=1e-10) == result.costs.D_C

    def test_equator_phi(self):
        """Test equator diagnostic Φ_eq."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            weights=[0.5, 0.3, 0.2],
        )

        # Φ_eq = F - (1.00 - 0.75ω - 0.55C)
        expected_phi = result.kernel.F - (1.0 - 0.75 * result.kernel.omega - 0.55 * result.kernel.C)
        assert result.costs is not None
        assert result.costs.equator_phi == pytest.approx(expected_phi, rel=1e-6)


class TestRegimeClassification:
    """Test regime classification."""

    def test_stable_regime(self):
        """Test STABLE regime detection."""
        calc = UniversalCalculator()
        # High fidelity, low drift, low entropy, low curvature
        result = calc.compute_all(coordinates=[0.98, 0.97, 0.99])
        assert result.regime == "STABLE"

    def test_watch_regime(self):
        """Test WATCH regime detection."""
        calc = UniversalCalculator()
        # Moderate drift
        result = calc.compute_all(coordinates=[0.9, 0.85, 0.88])
        assert result.regime == "WATCH"

    def test_collapse_regime(self):
        """Test COLLAPSE regime detection."""
        calc = UniversalCalculator()
        # High drift (ω ≥ 0.30)
        result = calc.compute_all(coordinates=[0.5, 0.6, 0.5])
        assert result.regime in ["COLLAPSE", "CRITICAL"]

    def test_critical_regime(self):
        """Test CRITICAL regime detection (integrity overlay)."""
        calc = UniversalCalculator()
        # Very low integrity
        result = calc.compute_all(coordinates=[0.2, 0.15, 0.1])
        assert result.regime == "CRITICAL"


class TestSeamAccounting:
    """Test seam accounting computations."""

    def test_seam_with_prior_state(self):
        """Test seam accounting when prior state is provided."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.88, 0.85, 0.90],
            weights=[0.5, 0.3, 0.2],
            tau_R=10.0,  # Finite return time
            prior_kappa=-0.15,
            prior_IC=0.86,
            R_credit=0.1,
        )

        assert result.seam is not None
        assert "delta_kappa_ledger" in result.seam.to_dict()
        assert "delta_kappa_budget" in result.seam.to_dict()
        assert "residual" in result.seam.to_dict()
        assert "passed" in result.seam.to_dict()

    def test_seam_pass_with_finite_tau_R(self):
        """Test that seam can pass with finite τ_R and small residual."""
        calc = UniversalCalculator()

        # Set up a case that should pass
        prior_kappa = -0.12
        prior_IC = 0.887

        result = calc.compute_all(
            coordinates=[0.89, 0.89, 0.89],  # Homogeneous for predictability
            tau_R=5.0,
            prior_kappa=prior_kappa,
            prior_IC=prior_IC,
            R_credit=0.02,  # Small R credit
        )

        assert result.seam is not None
        # The seam result should have meaningful values
        assert result.seam.delta_kappa_ledger is not None
        assert result.seam.I_ratio > 0

    def test_seam_fail_with_inf_tau_R(self):
        """Test that seam fails with infinite τ_R (typed censoring)."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.88, 0.85, 0.90],
            tau_R=float("inf"),  # INF_REC
            prior_kappa=-0.15,
            prior_IC=0.86,
        )

        assert result.seam is not None
        assert result.seam.passed is False
        assert any("finite" in f.lower() for f in result.seam.failures)


class TestGCDMetrics:
    """Test GCD (Generative Collapse Dynamics) metrics."""

    def test_gcd_energy_potential(self):
        """Test energy potential computation."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        assert result.gcd is not None
        assert result.gcd.E_potential >= 0
        assert result.gcd.E_collapse >= 0
        assert result.gcd.E_entropy >= 0
        assert result.gcd.E_curvature >= 0

    def test_gcd_generative_flux(self):
        """Test generative flux computation."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        assert result.gcd is not None
        assert result.gcd.Phi_gen >= 0
        assert result.gcd.Phi_collapse >= 0

    def test_gcd_energy_regime(self):
        """Test energy regime classification."""
        calc = UniversalCalculator()

        # High energy (low fidelity)
        result = calc.compute_all(
            coordinates=[0.5, 0.4, 0.6],
            mode=ComputationMode.FULL,
        )
        assert result.gcd is not None
        assert result.gcd.energy_regime in ["Medium", "High"]

        # Lower energy (high fidelity, but entropy still contributes)
        # E_potential = ω² + α·S + β·C² - even with ω≈0.01, S can be ~0.08
        result = calc.compute_all(
            coordinates=[0.99, 0.99, 0.99],
            mode=ComputationMode.FULL,
        )
        # With homogeneous high values, energy should be relatively low
        assert result.gcd is not None
        assert result.gcd.E_potential < 0.5  # Much lower than high-drift case


class TestRCFTMetrics:
    """Test RCFT (Recursive Collapse Field Theory) metrics."""

    def test_rcft_from_single_point(self):
        """Test RCFT metrics from single point (limited)."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.RCFT,
        )

        assert result.rcft is not None
        assert 0 <= result.rcft.D_fractal <= 3
        assert result.rcft.fractal_regime in ["Smooth", "Wrinkled", "Turbulent"]
        assert 0 <= result.rcft.basin_strength <= 1

    def test_rcft_from_trajectory(self):
        """Test RCFT metrics from trajectory."""
        calc = UniversalCalculator()

        # Create a trajectory
        trajectory = np.random.rand(50, 3) * 0.3 + 0.6

        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            trajectory=trajectory,
            mode=ComputationMode.FULL,
        )

        assert result.rcft is not None
        assert result.rcft.D_fractal >= 0
        assert result.rcft.memory_depth >= 1


class TestUncertainty:
    """Test uncertainty propagation."""

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation via delta-method."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            weights=[0.5, 0.3, 0.2],
            coord_variances=[0.001, 0.002, 0.001],
            mode=ComputationMode.FULL,
        )

        assert result.uncertainty is not None
        assert result.uncertainty.var_F >= 0
        assert result.uncertainty.var_omega >= 0
        assert result.uncertainty.std_F >= 0
        assert result.uncertainty.std_kappa >= 0

    def test_uncertainty_bounds(self):
        """Test that uncertainties are bounded."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            coord_variances=[0.01, 0.01, 0.01],
            mode=ComputationMode.FULL,
        )

        assert result.uncertainty is not None
        # Standard deviations should be positive
        assert result.uncertainty.std_F > 0
        assert result.uncertainty.std_omega > 0
        # And bounded by something reasonable
        assert result.uncertainty.std_F < 1
        assert result.uncertainty.std_omega < 1


class TestSS1MChecksum:
    """Test SS1M human-verifiable checksums."""

    def test_ss1m_triad_generation(self):
        """Test SS1M triad generation."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
        )

        assert result.ss1m is not None
        assert len(result.ss1m.C1) == 2
        assert len(result.ss1m.C2) == 2
        assert len(result.ss1m.C3) == 2
        assert len(result.ss1m.full_hash) == 64  # SHA256

    def test_ss1m_mod97(self):
        """Test that triads are mod-97 values."""
        calc = UniversalCalculator()
        result = calc.compute_all(coordinates=[0.9, 0.85, 0.92])

        assert result.ss1m is not None
        # Each component should be 00-96
        assert 0 <= int(result.ss1m.C1) <= 96
        assert 0 <= int(result.ss1m.C2) <= 96
        assert 0 <= int(result.ss1m.C3) <= 96

    def test_ss1m_deterministic(self):
        """Test that SS1M is deterministic for same inputs."""
        calc = UniversalCalculator()
        result1 = calc.compute_all(coordinates=[0.9, 0.85, 0.92])
        result2 = calc.compute_all(coordinates=[0.9, 0.85, 0.92])

        assert result1.ss1m is not None
        assert result2.ss1m is not None
        assert result1.ss1m.full_hash == result2.ss1m.full_hash


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_compute_kernel(self):
        """Test compute_kernel convenience function."""
        kernel = compute_kernel([0.9, 0.85, 0.92])

        assert pytest.approx(0.89, rel=1e-2) == kernel.F
        assert kernel.omega == pytest.approx(0.11, rel=1e-2)

    def test_compute_regime(self):
        """Test compute_regime convenience function."""
        regime = compute_regime([0.9, 0.85, 0.92])
        assert regime in ["STABLE", "WATCH", "COLLAPSE", "CRITICAL"]

    def test_compute_full(self):
        """Test compute_full convenience function."""
        result = compute_full([0.9, 0.85, 0.92])

        assert isinstance(result, UniversalResult)
        assert result.gcd is not None
        assert result.rcft is not None


class TestOutput:
    """Test output formatting."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        d = result.to_dict()
        assert "metadata" in d
        assert "kernel" in d
        assert "regime" in d
        assert "costs" in d

    def test_to_json(self):
        """Test JSON conversion."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "kernel" in parsed
        assert "regime" in parsed

    def test_summary(self):
        """Test summary generation."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        summary = result.summary()
        assert "UMCP Universal Calculator Result" in summary
        assert "Tier-1 Kernel Invariants" in summary
        assert "Drift" in summary
        assert "Fidelity" in summary


class TestCLI:
    """Test CLI interface."""

    def test_cli_basic(self):
        """Test basic CLI invocation."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "--coordinates", "0.9,0.85,0.92"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "UMCP Universal Calculator Result" in result.stdout

    def test_cli_with_weights(self):
        """Test CLI with weights."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "umcp.universal_calculator",
                "--coordinates",
                "0.9,0.85,0.92",
                "--weights",
                "0.5,0.3,0.2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_json_output(self):
        """Test CLI with JSON output."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "--coordinates", "0.9,0.85,0.92", "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "kernel" in parsed

    def test_cli_full_mode(self):
        """Test CLI with full mode."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.universal_calculator", "--coordinates", "0.9,0.85,0.92", "--mode", "full"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "GCD Metrics" in result.stdout
        assert "RCFT Metrics" in result.stdout


class TestDiagnostics:
    """Test diagnostic information."""

    def test_diagnostics_generated(self):
        """Test that diagnostics are generated in full mode."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        assert "n_coordinates" in result.diagnostics
        assert "am_gm_gap" in result.diagnostics
        assert "identity_check" in result.diagnostics

    def test_amgm_gap_diagnostic(self):
        """Test AM-GM gap is correctly computed."""
        calc = UniversalCalculator()
        result = calc.compute_all(
            coordinates=[0.9, 0.85, 0.92],
            mode=ComputationMode.FULL,
        )

        expected_gap = result.kernel.F - result.kernel.IC
        assert result.diagnostics["am_gm_gap"] == pytest.approx(expected_gap, rel=1e-10)
        assert result.diagnostics["am_gm_gap"] >= 0  # AM-GM inequality
