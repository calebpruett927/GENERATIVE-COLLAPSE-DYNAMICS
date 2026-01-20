#!/usr/bin/env python3
"""
Test suite for GCD closures.

Validates:
- All 4 GCD closure implementations
- Input validation
- Output structure
- Regime classification
- Mathematical correctness
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


def load_closure(repo_root, closure_name):
    """Dynamically load a GCD closure module."""
    closure_path = repo_root / "closures" / "gcd" / f"{closure_name}.py"
    assert closure_path.exists(), f"Closure not found: {closure_path}"
    
    spec = importlib.util.spec_from_file_location(closure_name, closure_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Energy Potential Tests
# ============================================================================

def test_energy_potential_exists(repo_root):
    """Test energy_potential.py exists."""
    path = repo_root / "closures" / "gcd" / "energy_potential.py"
    assert path.exists()


def test_energy_potential_zero_entropy(repo_root):
    """Test energy potential at zero entropy state."""
    module = load_closure(repo_root, "energy_potential")
    result = module.compute_energy_potential(omega=0.0, S=0.0, C=0.0)
    
    assert "E_potential" in result
    assert result["E_potential"] == 0.0
    assert result["regime"] == "Low"


def test_energy_potential_stable_regime(repo_root):
    """Test energy potential in stable regime."""
    module = load_closure(repo_root, "energy_potential")
    result = module.compute_energy_potential(omega=0.01, S=0.056, C=0.05)
    
    assert result["E_potential"] > 0
    assert result["regime"] in ["Low", "Medium", "High"]


def test_energy_potential_decomposition(repo_root):
    """Test energy decomposition identity."""
    module = load_closure(repo_root, "energy_potential")
    result = module.compute_energy_potential(omega=0.1, S=0.2, C=0.1)
    
    # E = E_collapse + E_entropy + E_curvature
    E_sum = (result["E_collapse"] + result["E_entropy"] + 
             result["E_curvature"])
    assert abs(E_sum - result["E_potential"]) < 1e-9


def test_energy_potential_invalid_input(repo_root):
    """Test energy potential with invalid inputs."""
    module = load_closure(repo_root, "energy_potential")
    
    with pytest.raises(ValueError):
        module.compute_energy_potential(omega=2.0, S=0.0, C=0.0)


# ============================================================================
# Entropic Collapse Tests
# ============================================================================

def test_entropic_collapse_exists(repo_root):
    """Test entropic_collapse.py exists."""
    path = repo_root / "closures" / "gcd" / "entropic_collapse.py"
    assert path.exists()


def test_entropic_collapse_zero_entropy(repo_root):
    """Test entropic collapse at zero entropy."""
    module = load_closure(repo_root, "entropic_collapse")
    result = module.compute_entropic_collapse(S=0.0, F=1.0, tau_R=1.0)
    
    assert "phi_collapse" in result
    assert result["phi_collapse"] == 0.0
    assert result["regime"] == "Minimal"


def test_entropic_collapse_stable_regime(repo_root):
    """Test entropic collapse in stable regime."""
    module = load_closure(repo_root, "entropic_collapse")
    result = module.compute_entropic_collapse(S=0.056, F=0.99, tau_R=5.0)
    
    assert result["phi_collapse"] >= 0
    assert result["regime"] in ["Minimal", "Active", "Critical"]


def test_entropic_collapse_components(repo_root):
    """Test entropic collapse component structure."""
    module = load_closure(repo_root, "entropic_collapse")
    result = module.compute_entropic_collapse(S=0.2, F=0.8, tau_R=10.0)
    
    assert "S_contribution" in result
    assert "F_contribution" in result
    assert "tau_damping" in result
    assert result["tau_damping"] > 0


def test_entropic_collapse_invalid_input(repo_root):
    """Test entropic collapse with invalid inputs."""
    module = load_closure(repo_root, "entropic_collapse")
    
    with pytest.raises(ValueError):
        module.compute_entropic_collapse(S=-0.1, F=1.0, tau_R=1.0)


# ============================================================================
# Generative Flux Tests
# ============================================================================

def test_generative_flux_exists(repo_root):
    """Test generative_flux.py exists."""
    path = repo_root / "closures" / "gcd" / "generative_flux.py"
    assert path.exists()


def test_generative_flux_zero_entropy(repo_root):
    """Test generative flux at zero entropy state."""
    module = load_closure(repo_root, "generative_flux")
    result = module.compute_generative_flux(kappa=-18.420681, IC=0.0, C=0.0)
    
    assert "phi_gen" in result
    assert result["phi_gen"] >= 0
    assert result["regime"] == "Dormant"


def test_generative_flux_stable_regime(repo_root):
    """Test generative flux in stable regime."""
    module = load_closure(repo_root, "generative_flux")
    result = module.compute_generative_flux(kappa=-2.0, IC=0.135, C=0.05)
    
    assert result["phi_gen"] >= 0
    assert result["regime"] in ["Dormant", "Emerging", "Explosive"]


def test_generative_flux_components(repo_root):
    """Test generative flux component structure."""
    module = load_closure(repo_root, "generative_flux")
    result = module.compute_generative_flux(kappa=-5.0, IC=0.01, C=0.1)
    
    assert "kappa_component" in result
    assert "IC_amplification" in result
    assert "curvature_modulation" in result
    assert result["curvature_modulation"] >= 1.0


def test_generative_flux_invalid_input(repo_root):
    """Test generative flux with invalid inputs."""
    module = load_closure(repo_root, "generative_flux")
    
    with pytest.raises(ValueError):
        module.compute_generative_flux(kappa=1.0, IC=0.1, C=0.1)


# ============================================================================
# Field Resonance Tests
# ============================================================================

def test_field_resonance_exists(repo_root):
    """Test field_resonance.py exists."""
    path = repo_root / "closures" / "gcd" / "field_resonance.py"
    assert path.exists()


def test_field_resonance_zero_entropy(repo_root):
    """Test field resonance at zero entropy (perfect resonance)."""
    module = load_closure(repo_root, "field_resonance")
    result = module.compute_field_resonance(omega=0.0, S=0.0, C=0.0)
    
    assert "resonance" in result
    assert result["resonance"] == 1.0
    assert result["regime"] == "Coherent"


def test_field_resonance_stable_regime(repo_root):
    """Test field resonance in stable regime."""
    module = load_closure(repo_root, "field_resonance")
    result = module.compute_field_resonance(omega=0.01, S=0.056, C=0.05)
    
    assert 0 <= result["resonance"] <= 1
    assert result["regime"] in ["Decoupled", "Partial", "Coherent"]


def test_field_resonance_factorization(repo_root):
    """Test resonance factorization identity."""
    module = load_closure(repo_root, "field_resonance")
    result = module.compute_field_resonance(omega=0.1, S=0.2, C=0.1)
    
    # R = (1-|ω|) · (1-S) · exp(-C/C_crit)
    R_computed = (result["coherence_factor"] * 
                  result["order_factor"] * 
                  result["curvature_damping"])
    assert abs(R_computed - result["resonance"]) < 1e-9


def test_field_resonance_invalid_input(repo_root):
    """Test field resonance with invalid inputs."""
    module = load_closure(repo_root, "field_resonance")
    
    with pytest.raises(ValueError):
        module.compute_field_resonance(omega=2.0, S=0.0, C=0.0)


# ============================================================================
# Integration Tests
# ============================================================================

def test_all_gcd_closures_compatible(repo_root):
    """Test all GCD closures work with same invariant set."""
    # Load all closures
    energy_mod = load_closure(repo_root, "energy_potential")
    collapse_mod = load_closure(repo_root, "entropic_collapse")
    flux_mod = load_closure(repo_root, "generative_flux")
    resonance_mod = load_closure(repo_root, "field_resonance")
    
    # Test with zero entropy invariants
    omega, F, S, C = 0.0, 1.0, 0.0, 0.0
    tau_R, kappa, IC = 1.0, -18.420681, 0.0
    
    # All should execute without errors
    energy = energy_mod.compute_energy_potential(omega, S, C)
    collapse = collapse_mod.compute_entropic_collapse(S, F, tau_R)
    flux = flux_mod.compute_generative_flux(kappa, IC, C)
    resonance = resonance_mod.compute_field_resonance(omega, S, C)
    
    # Verify expected results at zero entropy
    assert energy["E_potential"] == 0.0
    assert collapse["phi_collapse"] == 0.0
    assert flux["phi_gen"] >= 0  # Near zero but not exactly
    assert resonance["resonance"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
