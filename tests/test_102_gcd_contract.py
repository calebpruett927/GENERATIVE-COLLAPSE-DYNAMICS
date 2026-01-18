#!/usr/bin/env python3
"""
Test suite for GCD contract.

Validates:
- GCD contract schema conformance
- Parent contract inheritance (UMA.INTSTACK.v1)
- Tier-1 frozen symbol definitions
- GCD-specific parameters
- Axiom specifications
- Regime classifications
- Mathematical identity specifications
- Closure references
"""

import pytest
import yaml
from pathlib import Path


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def gcd_contract(repo_root):
    """Load GCD contract."""
    contract_path = repo_root / "contracts" / "GCD.INTSTACK.v1.yaml"
    assert contract_path.exists(), f"GCD contract not found: {contract_path}"
    
    with open(contract_path, 'r') as f:
        return yaml.safe_load(f)


def test_gcd_contract_exists(repo_root):
    """Test that GCD contract file exists."""
    contract_path = repo_root / "contracts" / "GCD.INTSTACK.v1.yaml"
    assert contract_path.exists(), "GCD contract must exist"


def test_gcd_contract_schema(gcd_contract):
    """Test GCD contract has correct schema reference."""
    assert "schema" in gcd_contract
    assert gcd_contract["schema"] == "schemas/contract.schema.json"


def test_gcd_contract_id(gcd_contract):
    """Test GCD contract has correct ID."""
    assert "contract" in gcd_contract
    assert "id" in gcd_contract["contract"]
    assert gcd_contract["contract"]["id"] == "GCD.INTSTACK.v1"


def test_gcd_contract_version(gcd_contract):
    """Test GCD contract has version."""
    assert "version" in gcd_contract["contract"]
    assert gcd_contract["contract"]["version"] == "1.0.0"


def test_gcd_contract_parent(gcd_contract):
    """Test GCD contract inherits from UMA.INTSTACK.v1."""
    assert "parent_contract" in gcd_contract["contract"]
    assert gcd_contract["contract"]["parent_contract"] == "UMA.INTSTACK.v1"


def test_gcd_contract_tier_level(gcd_contract):
    """Test GCD contract is Tier-1."""
    assert "tier_level" in gcd_contract["contract"]
    assert gcd_contract["contract"]["tier_level"] == 1


def test_gcd_reserved_symbols(gcd_contract):
    """Test GCD contract defines Tier-1 reserved symbols."""
    tier1 = gcd_contract["contract"]["tier_1_kernel"]
    assert "reserved_symbols" in tier1
    
    symbols = tier1["reserved_symbols"]
    
    # Inherited from UMA.INTSTACK.v1
    required_umcp = ["ω", "F", "S", "C", "τ_R", "κ", "IC", "IC_min"]
    for sym in required_umcp:
        assert sym in symbols, f"Missing UMCP symbol: {sym}"
    
    # GCD extensions
    required_gcd = ["E_potential", "Φ_collapse", "Φ_gen", "R", "I"]
    for sym in required_gcd:
        assert sym in symbols, f"Missing GCD symbol: {sym}"


def test_gcd_frozen_parameters(gcd_contract):
    """Test GCD contract defines frozen parameters."""
    tier1 = gcd_contract["contract"]["tier_1_kernel"]
    assert "frozen_parameters" in tier1
    
    params = tier1["frozen_parameters"]
    
    # UMCP parameters
    assert params["p"] == 3
    assert params["alpha"] == 1.0
    assert params["lambda"] == 0.2
    
    # GCD-specific parameters
    assert "alpha_energy" in params
    assert "beta_energy" in params
    assert "tau_0" in params
    assert "C_crit" in params


def test_gcd_tolerances(gcd_contract):
    """Test GCD contract defines tolerances."""
    tier1 = gcd_contract["contract"]["tier_1_kernel"]
    assert "tolerances" in tier1
    
    tol = tier1["tolerances"]
    
    # UMCP tolerances
    assert tol["tol_seam"] == 0.005
    assert tol["tol_id"] == 1.0e-9
    
    # GCD tolerances
    assert "tol_energy" in tol
    assert "tol_flux" in tol
    assert "tol_resonance" in tol


def test_gcd_axioms(gcd_contract):
    """Test GCD contract defines axioms."""
    assert "axioms" in gcd_contract["contract"]
    axioms = gcd_contract["contract"]["axioms"]
    
    assert len(axioms) >= 3
    axiom_ids = [ax["id"] for ax in axioms]
    assert "AX-0" in axiom_ids
    assert "AX-1" in axiom_ids
    assert "AX-2" in axiom_ids


def test_gcd_regime_classification(gcd_contract):
    """Test GCD contract defines regime classifications."""
    assert "regime_classification" in gcd_contract["contract"]
    regimes = gcd_contract["contract"]["regime_classification"]
    
    # GCD-specific regime types
    assert "energy" in regimes
    assert "collapse" in regimes
    assert "flux" in regimes
    assert "resonance" in regimes


def test_gcd_mathematical_identities(gcd_contract):
    """Test GCD contract defines mathematical identities."""
    assert "mathematical_identities" in gcd_contract["contract"]
    identities = gcd_contract["contract"]["mathematical_identities"]
    
    identity_names = [id_["name"] for id_ in identities]
    
    # UMCP identities
    assert "Fidelity-drift duality" in identity_names
    assert "Integrity-collapse relation" in identity_names
    
    # GCD identities
    assert "Energy decomposition" in identity_names
    assert "Resonance factorization" in identity_names


def test_gcd_closures(gcd_contract):
    """Test GCD contract references closures."""
    assert "closures" in gcd_contract["contract"]
    closures = gcd_contract["contract"]["closures"]
    
    assert "energy_potential" in closures
    assert "entropic_collapse" in closures
    assert "generative_flux" in closures
    assert "field_resonance" in closures


def test_gcd_closure_paths(gcd_contract, repo_root):
    """Test GCD closure paths exist."""
    closures = gcd_contract["contract"]["closures"]
    
    for closure_name, closure_spec in closures.items():
        path = repo_root / closure_spec["path"]
        assert path.exists(), f"Closure not found: {path}"


def test_gcd_provenance(gcd_contract):
    """Test GCD contract has provenance metadata."""
    assert "provenance" in gcd_contract["contract"]
    prov = gcd_contract["contract"]["provenance"]
    
    assert "created" in prov
    assert "author" in prov
    assert "canonical_anchor" in prov
    assert prov["canonical_anchor"] == "canon/gcd_anchors.yaml"


def test_gcd_typed_censoring(gcd_contract):
    """Test GCD contract extends typed censoring."""
    assert "typed_censoring" in gcd_contract["contract"]
    tc = gcd_contract["contract"]["typed_censoring"]
    
    assert "gcd_regime_enums" in tc
    enums = tc["gcd_regime_enums"]
    
    assert "energy" in enums
    assert "collapse" in enums
    assert "flux" in enums
    assert "resonance" in enums


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
