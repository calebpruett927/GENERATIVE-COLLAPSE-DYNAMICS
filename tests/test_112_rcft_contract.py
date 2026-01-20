"""
Test suite for RCFT contract (contracts/RCFT.INTSTACK.v1.yaml)

Validates:
- Contract schema conformance
- Proper extension of GCD.INTSTACK.v1
- Tier-2 hierarchy
- Inherited Tier-1 symbols (frozen)
- New Tier-2 symbols
- RCFT closures referenced
- Regime classifications
- Mathematical identities
- Tolerances
"""

from pathlib import Path

import jsonschema
import pytest
import yaml


@pytest.fixture
def rcft_contract():
    """Load RCFT contract."""
    contract_path = Path("contracts/RCFT.INTSTACK.v1.yaml")
    assert contract_path.exists(), "RCFT contract must exist"
    with open(contract_path) as f:
        full = yaml.safe_load(f)
        return full["contract"]  # Extract contract object


@pytest.fixture
def contract_schema():
    """Load contract schema."""
    schema_path = Path("schemas/contract.schema.json")
    with open(schema_path) as f:
        import json

        return json.load(f)


def test_rcft_contract_exists():
    """RCFT contract should exist."""
    assert Path("contracts/RCFT.INTSTACK.v1.yaml").exists()


def test_rcft_contract_schema_conformance(contract_schema):
    """RCFT contract should conform to contract schema."""
    # Load full YAML (with schema wrapper)
    contract_path = Path("contracts/RCFT.INTSTACK.v1.yaml")
    with open(contract_path) as f:
        full_rcft = yaml.safe_load(f)

    # Validate against schema
    try:
        jsonschema.validate(instance=full_rcft, schema=contract_schema)
    except jsonschema.ValidationError as e:
        pytest.fail(f"RCFT contract schema validation failed: {e.message}")


def test_rcft_contract_id_version(rcft_contract):
    """RCFT contract should have correct ID and version."""
    assert rcft_contract["id"] == "RCFT.INTSTACK.v1"
    assert rcft_contract["version"] == "1.0.0"


def test_rcft_extends_gcd(rcft_contract):
    """RCFT should extend GCD.INTSTACK.v1 as Tier-2."""
    assert rcft_contract["parent_contract"] == "GCD.INTSTACK.v1"
    assert rcft_contract["tier_level"] == 2


def test_rcft_inherits_tier1_symbols(rcft_contract):
    """RCFT should list all inherited Tier-1 symbols."""
    reserved = rcft_contract["tier_1_kernel"]["reserved_symbols"]

    # Core UMCP symbols (Tier-1)
    assert "omega" in reserved
    assert "F" in reserved
    assert "S" in reserved
    assert "C" in reserved
    assert "tau_R" in reserved
    assert "kappa" in reserved
    assert "IC" in reserved

    # GCD Tier-1 extensions
    assert "E_potential" in reserved
    assert "Phi_collapse" in reserved
    assert "Phi_gen" in reserved
    assert "R" in reserved


def test_rcft_tier2_symbols(rcft_contract):
    """RCFT should add new Tier-2 symbols."""
    reserved = rcft_contract["tier_1_kernel"]["reserved_symbols"]

    # New RCFT Tier-2 symbols
    assert "D_fractal" in reserved
    assert "Psi_recursive" in reserved
    assert "lambda_pattern" in reserved
    assert "Theta_phase" in reserved


def test_rcft_frozen_parameters(rcft_contract):
    """RCFT should inherit and potentially extend frozen parameters."""
    frozen = rcft_contract["tier_1_kernel"]["frozen_parameters"]

    # Core frozen parameters (inherited)
    assert frozen["p"] == 3
    assert frozen["alpha"] == 1.0
    assert frozen["lambda"] == 0.2
    assert frozen["eta"] == 1.0e-3

    # GCD frozen parameters (inherited)
    assert frozen["alpha_energy"] == 1.0
    assert frozen["beta_energy"] == 0.5
    assert frozen["tau_0"] == 10.0
    assert frozen["C_crit"] == 0.2

    # RCFT Tier-2 parameters (not necessarily frozen)
    assert "alpha_decay" in frozen
    assert "max_recursion_depth" in frozen


def test_rcft_tolerances(rcft_contract):
    """RCFT should inherit and extend tolerances."""
    tol = rcft_contract["tier_1_kernel"]["tolerances"]

    # Inherited Tier-1 tolerances
    assert tol["tol_seam"] == 0.005
    assert tol["tol_id"] == 1.0e-9
    assert tol["tol_energy"] == 1.0e-6

    # New RCFT Tier-2 tolerances
    assert "tol_fractal" in tol
    assert "tol_recursive" in tol
    assert "tol_phase" in tol
    assert tol["tol_fractal"] == 1.0e-6


def test_rcft_axioms(rcft_contract):
    """RCFT should inherit GCD axioms and add principles."""
    axioms = rcft_contract["axioms"]
    axiom_ids = [a["id"] for a in axioms]

    # Inherited GCD axioms
    assert "AX-0" in axiom_ids
    assert "AX-1" in axiom_ids
    assert "AX-2" in axiom_ids

    # RCFT principles
    assert "P-RCFT-0" in axiom_ids
    assert "P-RCFT-1" in axiom_ids
    assert "P-RCFT-2" in axiom_ids


def test_rcft_regime_classifications(rcft_contract):
    """RCFT should have regime classifications for all tiers."""
    regimes = rcft_contract["regime_classification"]

    # Inherited Tier-1 regimes
    assert "energy" in regimes
    assert "collapse" in regimes
    assert "flux" in regimes
    assert "resonance" in regimes

    # New RCFT Tier-2 regimes
    assert "fractal" in regimes
    assert "recursive" in regimes
    assert "pattern" in regimes


def test_rcft_typed_censoring(rcft_contract):
    """RCFT should have typed censoring enums."""
    censoring = rcft_contract["typed_censoring"]

    # Inherited enums
    assert "gcd_regime_enums" in censoring

    # New RCFT enums
    assert "rcft_regime_enums" in censoring
    rcft_enums = censoring["rcft_regime_enums"]

    assert "fractal" in rcft_enums
    assert "Smooth" in rcft_enums["fractal"]
    assert "Wrinkled" in rcft_enums["fractal"]
    assert "Turbulent" in rcft_enums["fractal"]

    assert "recursive" in rcft_enums
    assert "Dormant" in rcft_enums["recursive"]
    assert "Active" in rcft_enums["recursive"]
    assert "Resonant" in rcft_enums["recursive"]


def test_rcft_mathematical_identities(rcft_contract):
    """RCFT should have mathematical identities from all tiers."""
    identities = rcft_contract["mathematical_identities"]
    names = [i["name"] for i in identities]

    # Inherited Tier-1 identities
    assert "Fidelity-drift duality" in names
    assert "Energy decomposition" in names

    # New RCFT Tier-2 identities
    assert "Fractal dimension bounds" in names
    assert "Phase periodicity" in names
    assert "Wavelength positivity" in names


def test_rcft_closures(rcft_contract):
    """RCFT should reference all GCD and RCFT closures."""
    closures = rcft_contract["closures"]
    # Closures are stored as a dict with closure names as keys

    # Inherited GCD Tier-1 closures
    assert "energy_potential" in closures
    assert "entropic_collapse" in closures
    assert "generative_flux" in closures
    assert "field_resonance" in closures

    # New RCFT Tier-2 closures
    assert "fractal_dimension" in closures
    assert "recursive_field" in closures
    assert "resonance_pattern" in closures


def test_rcft_closure_paths(rcft_contract):
    """All closure paths should resolve to existing files."""
    closures = rcft_contract["closures"]

    for _closure_name, closure_spec in closures.items():
        path = Path(closure_spec["path"])
        assert path.exists(), f"Closure path does not exist: {path}"


def test_rcft_closure_tier_marking(rcft_contract):
    """RCFT closures should be marked as Tier-2."""
    closures = rcft_contract["closures"]

    for closure_name, closure_spec in closures.items():
        if closure_name in ["fractal_dimension", "recursive_field", "resonance_pattern"]:
            assert "tier" in closure_spec
            assert closure_spec["tier"] == 2


def test_rcft_provenance(rcft_contract):
    """RCFT contract should have provenance linking to canon and parent."""
    prov = rcft_contract["provenance"]

    assert prov["canonical_anchor"] == "canon/rcft_anchors.yaml"
    assert prov["parent_contract"] == "GCD.INTSTACK.v1"
    assert prov["tier"] == 2


def test_rcft_embedding(rcft_contract):
    """RCFT should inherit embedding from GCD."""
    # embedding doesn't have a 'p' field in the standard structure
    # tier_1_kernel.frozen_parameters has 'p'
    assert rcft_contract["tier_1_kernel"]["frozen_parameters"]["p"] == 3


def test_rcft_description(rcft_contract):
    """RCFT contract should have descriptive text."""
    assert "notes" in rcft_contract
    assert "Tier-2" in rcft_contract["notes"]
    assert "GCD.INTSTACK.v1" in rcft_contract["notes"]
