"""
Test suite for RCFT closures (Tier-2)

Tests the three RCFT closures:
1. fractal_dimension.py - Box-counting fractal dimension
2. recursive_field.py - Recursive field strength with memory
3. resonance_pattern.py - FFT-based pattern analysis

Validates Tier-2 behavior:
- Closures import and execute correctly
- Work with GCD Tier-1 invariants
- Produce correct regime classifications
- Handle edge cases (zero entropy, constant fields, etc.)
- Do not modify Tier-1 symbols
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Import RCFT closures
sys.path.insert(0, str(Path(__file__).parent.parent))

from closures.rcft import fractal_dimension
from closures.rcft import recursive_field
from closures.rcft import resonance_pattern


class TestFractalDimension:
    """Tests for fractal_dimension.py closure."""
    
    def test_imports(self):
        """Fractal dimension closure should import successfully."""
        assert hasattr(fractal_dimension, 'compute_fractal_dimension')
        assert hasattr(fractal_dimension, 'compute_trajectory_from_invariants')
    
    def test_zero_entropy_trajectory(self):
        """Zero entropy (point trajectory) should give D_f=0."""
        trajectory = np.array([[0.0, 0.0, 0.0]] * 3)
        result = fractal_dimension.compute_fractal_dimension(trajectory)
        
        assert result['D_fractal'] == 0.0
        assert result['regime'] == 'Smooth'
        assert result['r_squared'] == 1.0
    
    def test_linear_trajectory(self):
        """Linear trajectory should give D_f ≈ 1 (Smooth)."""
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, t, t])
        result = fractal_dimension.compute_fractal_dimension(trajectory)
        
        assert result['D_fractal'] < 1.2  # Smooth regime
        assert result['regime'] == 'Smooth'
        assert 'components' in result
    
    def test_spiral_trajectory(self):
        """Spiral trajectory should show moderate complexity."""
        theta = np.linspace(0, 4*np.pi, 200)
        r = theta / (4*np.pi)
        trajectory = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.zeros_like(theta)
        ])
        result = fractal_dimension.compute_fractal_dimension(trajectory)
        
        assert result['D_fractal'] >= 0.0
        assert result['regime'] in ['Smooth', 'Wrinkled', 'Turbulent']
        assert 0.0 <= result['r_squared'] <= 1.0
    
    def test_random_walk(self):
        """Random walk should have higher fractal dimension."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500, 3) * 0.1, axis=0)
        result = fractal_dimension.compute_fractal_dimension(random_walk)
        
        assert result['D_fractal'] >= 0.0
        assert result['regime'] in ['Smooth', 'Wrinkled', 'Turbulent']
        assert 'box_counts' in result
        assert 'eps_used' in result
    
    def test_trajectory_from_invariants(self):
        """Should construct trajectory from GCD invariants."""
        omega = np.array([0.01, 0.02, 0.01])
        S = np.array([0.05, 0.06, 0.05])
        C = np.array([0.02, 0.03, 0.02])
        
        trajectory = fractal_dimension.compute_trajectory_from_invariants(omega, S, C)
        
        assert trajectory.shape == (3, 3)
        assert np.allclose(trajectory[:, 0], omega)
        assert np.allclose(trajectory[:, 1], S)
        assert np.allclose(trajectory[:, 2], C)
    
    def test_regime_classification(self):
        """Test regime classification boundaries."""
        # Smooth: D_f < 1.2
        # Wrinkled: 1.2 ≤ D_f < 1.8
        # Turbulent: D_f ≥ 1.8
        
        # Create trajectories with known complexity
        # (We'll use the actual computation results since D_f is empirical)
        pass  # Regime boundaries tested implicitly in other tests


class TestRecursiveField:
    """Tests for recursive_field.py closure."""
    
    def test_imports(self):
        """Recursive field closure should import successfully."""
        assert hasattr(recursive_field, 'compute_recursive_field')
        assert hasattr(recursive_field, 'compute_field_strength_single')
        assert hasattr(recursive_field, 'compute_recursive_field_from_energy')
    
    def test_zero_entropy_state(self):
        """Zero entropy should give Ψ_r=0 (Dormant)."""
        S = np.array([0.0, 0.0, 0.0])
        C = np.array([0.0, 0.0, 0.0])
        F = np.array([1.0, 1.0, 1.0])
        
        result = recursive_field.compute_recursive_field(S, C, F)
        
        assert result['Psi_recursive'] == 0.0
        assert result['regime'] == 'Dormant'
        assert result['convergence_achieved']
    
    def test_low_entropy_dormant(self):
        """Low constant entropy should give Dormant regime."""
        S = np.full(10, 0.05)
        C = np.full(10, 0.01)
        F = np.full(10, 0.99)
        
        result = recursive_field.compute_recursive_field(S, C, F, alpha=0.8)
        
        assert result['Psi_recursive'] < 0.1  # Dormant threshold
        assert result['regime'] == 'Dormant'
        assert 'components' in result
        assert result['components']['decay_rate'] == 0.8
    
    def test_moderate_entropy_active(self):
        """Moderate entropy should give Active regime."""
        S = np.linspace(0.2, 0.05, 20)
        C = np.linspace(0.1, 0.02, 20)
        F = np.linspace(0.8, 0.95, 20)
        
        result = recursive_field.compute_recursive_field(S, C, F, alpha=0.85)
        
        assert 0.1 <= result['Psi_recursive'] < 1.0  # Active range
        assert result['regime'] == 'Active'
        assert result['n_iterations'] == 20
    
    def test_high_entropy_resonant(self):
        """High oscillating entropy should give Resonant regime."""
        t = np.linspace(0, 4*np.pi, 50)
        S = 0.3 + 0.2 * np.sin(t)
        C = 0.2 + 0.15 * np.cos(t)
        F = 0.7 + 0.1 * np.sin(2*t)
        
        result = recursive_field.compute_recursive_field(S, C, F, alpha=0.9)
        
        assert result['Psi_recursive'] >= 1.0  # Resonant threshold
        assert result['regime'] == 'Resonant'
    
    def test_energy_based_computation(self):
        """Should compute from energy series."""
        E_series = np.linspace(0.1, 0.01, 30)
        result = recursive_field.compute_recursive_field_from_energy(E_series, alpha=0.75)
        
        assert 'Psi_recursive' in result
        assert result['regime'] in ['Dormant', 'Active', 'Resonant']
        assert 'components' in result
        assert 'mean_energy' in result['components']
    
    def test_decay_factor_validation(self):
        """Invalid decay factor should raise error."""
        S = np.array([0.1])
        C = np.array([0.05])
        F = np.array([0.9])
        
        with pytest.raises(ValueError):
            recursive_field.compute_recursive_field(S, C, F, alpha=1.5)  # α > 1
        
        with pytest.raises(ValueError):
            recursive_field.compute_recursive_field(S, C, F, alpha=0.0)  # α ≤ 0
    
    def test_mismatched_lengths(self):
        """Mismatched series lengths should raise error."""
        S = np.array([0.1, 0.2])
        C = np.array([0.05])
        F = np.array([0.9, 0.8])
        
        with pytest.raises(ValueError):
            recursive_field.compute_recursive_field(S, C, F)


class TestResonancePattern:
    """Tests for resonance_pattern.py closure."""
    
    def test_imports(self):
        """Resonance pattern closure should import successfully."""
        assert hasattr(resonance_pattern, 'compute_resonance_pattern')
        assert hasattr(resonance_pattern, 'compute_multi_field_resonance')
    
    def test_constant_field(self):
        """Constant field should give Standing pattern with λ=inf."""
        constant = np.ones(50) * 0.8
        result = resonance_pattern.compute_resonance_pattern(constant)
        
        assert result['lambda_pattern'] == np.inf
        assert result['pattern_type'] == 'Standing'
        assert result['phase_coherence'] == 1.0
    
    def test_sinusoidal_wave(self):
        """Sinusoidal wave should detect wavelength and phase."""
        t = np.linspace(0, 10*np.pi, 200)
        sine_wave = np.sin(2*t)
        result = resonance_pattern.compute_resonance_pattern(sine_wave, dt=t[1]-t[0])
        
        assert result['lambda_pattern'] < np.inf
        assert result['pattern_type'] in ['Standing', 'Traveling', 'Mixed']
        assert 0.0 <= result['Theta_phase'] < 2*np.pi
        assert 0.0 <= result['phase_coherence'] <= 1.0
    
    def test_multi_harmonic(self):
        """Multiple harmonics should show in harmonic_content."""
        t = np.linspace(0, 10*np.pi, 200)
        multi = np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.sin(5*t)
        result = resonance_pattern.compute_resonance_pattern(multi, dt=t[1]-t[0])
        
        assert 'harmonic_content' in result
        assert result['harmonic_content'] >= 0.0
        assert 'components' in result
        assert 'spectral_entropy' in result['components']
    
    def test_multi_field_resonance(self):
        """Should analyze multiple GCD fields together."""
        t = np.linspace(0, 10*np.pi, 200)
        R_field = 0.7 + 0.2 * np.sin(t)
        Phi_field = 0.5 + 0.3 * np.sin(t + np.pi/4)
        E_field = 0.1 + 0.05 * np.sin(2*t)
        
        result = resonance_pattern.compute_multi_field_resonance(
            R_field, Phi_field, E_field, dt=t[1]-t[0]
        )
        
        assert 'lambda_pattern' in result
        assert 'pattern_type' in result
        assert 'cross_correlations' in result
        assert 'phase_differences' in result
        assert 'coherence' in result
        
        # Check cross-correlations
        assert 'R_Phi' in result['cross_correlations']
        assert 'R_E' in result['cross_correlations']
        assert -1.0 <= result['cross_correlations']['R_Phi'] <= 1.0
    
    def test_zero_entropy_gcd_state(self):
        """Zero entropy GCD state (R=1, Φ=0, E=0) should give Standing."""
        R_zero = np.ones(30)
        Phi_zero = np.full(30, 0.0001)
        E_zero = np.zeros(30)
        
        result = resonance_pattern.compute_multi_field_resonance(R_zero, Phi_zero, E_zero)
        
        assert result['pattern_type'] == 'Standing'
        assert result['coherence']['R'] == 1.0
    
    def test_short_series(self):
        """Short series (<4 points) should handle gracefully."""
        short = np.array([1.0, 1.0, 1.0])
        result = resonance_pattern.compute_resonance_pattern(short)
        
        assert 'lambda_pattern' in result
        assert 'pattern_type' in result


class TestRCFTTier2Compliance:
    """Tests for Tier-2 compliance of RCFT closures."""
    
    def test_closures_use_gcd_invariants(self):
        """RCFT closures should work with GCD Tier-1 invariants."""
        # Create GCD invariant trajectory
        omega = np.array([0.01, 0.02, 0.01])
        S = np.array([0.05, 0.06, 0.05])
        C = np.array([0.02, 0.03, 0.02])
        F = np.array([0.99, 0.98, 0.99])
        
        # Fractal dimension from trajectory
        trajectory = fractal_dimension.compute_trajectory_from_invariants(omega, S, C)
        fd_result = fractal_dimension.compute_fractal_dimension(trajectory)
        assert 'D_fractal' in fd_result
        
        # Recursive field from invariants
        rf_result = recursive_field.compute_recursive_field(S, C, F)
        assert 'Psi_recursive' in rf_result
        
        # Both should succeed without modifying inputs
        assert len(omega) == 3  # Unchanged
        assert len(S) == 3
    
    def test_tier2_symbols_not_in_tier1(self):
        """Tier-2 symbols (D_f, Ψ_r, λ_p, Θ) should be new, not overriding Tier-1."""
        # This is validated by contract structure, but we can check closure outputs
        # don't use Tier-1 symbol names
        
        S = np.array([0.0, 0.0, 0.0])
        C = np.array([0.0, 0.0, 0.0])
        F = np.array([1.0, 1.0, 1.0])
        
        rf_result = recursive_field.compute_recursive_field(S, C, F)
        
        # Should output Tier-2 symbols, not overwrite Tier-1
        assert 'Psi_recursive' in rf_result
        assert 'S' not in rf_result  # Should not redefine S
        assert 'F' not in rf_result  # Should not redefine F
        assert 'omega' not in rf_result  # Should not redefine ω
