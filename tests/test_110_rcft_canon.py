"""
Test suite for RCFT canon anchor (canon/rcft_anchors.yaml)

Validates:
- Canon file structure and schema reference
- RCFT ID and version
- Tier-2 hierarchy and parent framework
- RCFT principles (P-RCFT-0, P-RCFT-1, P-RCFT-2)
- Tier-2 extensions (D_fractal, Psi_recursive, lambda_pattern, Theta_phase)
- Regime classifications (fractal, recursive, pattern)
- Mathematical identities
- Tolerances
- Typed censoring enums
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def rcft_canon():
    """Load RCFT canon anchor."""
    canon_path = Path("canon/rcft_anchors.yaml")
    assert canon_path.exists(), "RCFT canon file must exist"
    with open(canon_path) as f:
        return yaml.safe_load(f)


def test_rcft_canon_exists():
    """RCFT canon file should exist at canon/rcft_anchors.yaml."""
    assert Path("canon/rcft_anchors.yaml").exists()


def test_rcft_canon_schema_reference(rcft_canon):
    """RCFT canon should reference the correct schema."""
    assert "schema" in rcft_canon
    assert rcft_canon["schema"] == "canon.anchors.schema.json"


def test_rcft_canon_id_and_version(rcft_canon):
    """RCFT canon should have correct ID and version."""
    assert rcft_canon["id"] == "UMCP.RCFT.v1"
    assert rcft_canon["version"] == "1.0.0"


def test_rcft_canon_tier_hierarchy(rcft_canon):
    """RCFT should declare Tier-2 hierarchy."""
    scope = rcft_canon["scope"]
    assert scope["tier"] == 2
    assert scope["parent_framework"] == "UMCP.GCD.v1"
    assert "RCFT > GCD > UMCP" in scope["hierarchy"]


def test_rcft_tier1_frozen_symbols(rcft_canon):
    """RCFT should list all Tier-1 frozen symbols from GCD."""
    tier_hierarchy = rcft_canon["tier_hierarchy"]
    frozen = tier_hierarchy["tier_1_frozen_symbols"]

    # Check core GCD Tier-1 symbols
    assert "ω" in frozen or "omega" in str(frozen)
    assert "F" in frozen
    assert "S" in frozen
    assert "C" in frozen
    assert "τ_R" in frozen or "tau_R" in str(frozen)
    assert "κ" in frozen or "kappa" in str(frozen)
    assert "IC" in frozen

    # Check GCD extensions (Tier-1)
    assert "E_potential" in frozen
    assert "Φ_collapse" in frozen or "Phi_collapse" in str(frozen)
    assert "Φ_gen" in frozen or "Phi_gen" in str(frozen)
    assert "R" in frozen


def test_rcft_principles(rcft_canon):
    """RCFT should define 3 principles (P-RCFT-0, P-RCFT-1, P-RCFT-2)."""
    principles = rcft_canon["principles"]
    assert len(principles) == 3

    principle_ids = [p["id"] for p in principles]
    assert "P-RCFT-0" in principle_ids
    assert "P-RCFT-1" in principle_ids
    assert "P-RCFT-2" in principle_ids

    # Check P-RCFT-0: "Tier-2 augments, never overrides"
    p0 = next(p for p in principles if p["id"] == "P-RCFT-0")
    assert "augments" in p0["statement"].lower()
    assert "override" in p0["statement"].lower()


def test_rcft_tier2_reserved_symbols(rcft_canon):
    """RCFT should define Tier-2 reserved symbols."""
    extensions = rcft_canon["tier_2_extensions"]
    symbols = extensions["reserved_symbols"]

    symbol_names = [s["symbol"] for s in symbols]
    assert "D_fractal" in symbol_names
    assert "Psi_recursive" in symbol_names
    assert "lambda_pattern" in symbol_names
    assert "Theta_phase" in symbol_names

    # Check all are marked as Tier-2
    for symbol in symbols:
        assert symbol["tier"] == 2


def test_rcft_symbol_properties(rcft_canon):
    """RCFT symbols should have complete properties."""
    symbols = rcft_canon["tier_2_extensions"]["reserved_symbols"]

    for symbol in symbols:
        assert "symbol" in symbol
        assert "latex" in symbol
        assert "description" in symbol
        assert "formula" in symbol
        assert "domain" in symbol
        assert "interpretation" in symbol
        assert "tier" in symbol


def test_rcft_regime_classifications(rcft_canon):
    """RCFT should define regime classifications."""
    regimes = rcft_canon["regime_classification"]

    # Fractal complexity regime
    assert "fractal_complexity" in regimes
    fractal = regimes["fractal_complexity"]
    assert len(fractal) == 3
    labels = [r["label"] for r in fractal]
    assert "Smooth" in labels
    assert "Wrinkled" in labels
    assert "Turbulent" in labels

    # Recursive strength regime
    assert "recursive_strength" in regimes
    recursive = regimes["recursive_strength"]
    assert len(recursive) == 3
    labels = [r["label"] for r in recursive]
    assert "Dormant" in labels
    assert "Active" in labels
    assert "Resonant" in labels


def test_rcft_mathematical_identities(rcft_canon):
    """RCFT should define mathematical identities."""
    identities = rcft_canon["mathematical_identities"]
    assert len(identities) >= 3

    names = [i["name"] for i in identities]
    assert "Fractal dimension bounds" in names
    assert "Phase periodicity" in names
    assert "Wavelength-frequency relation" in names

    # Check tolerances
    for identity in identities:
        assert "tolerance" in identity
        assert identity["tolerance"] > 0


def test_rcft_tolerances(rcft_canon):
    """RCFT should define Tier-2 tolerances."""
    tolerances = rcft_canon["tolerances"]
    gates = tolerances["gates"]

    gate_names = [g["name"] for g in gates]
    assert "tol_fractal" in gate_names
    assert "tol_recursive" in gate_names
    assert "tol_phase" in gate_names

    # Check tolerance values
    for gate in gates:
        assert gate["value"] > 0
        assert gate["value"] <= 1e-3  # Should be small


def test_rcft_typed_censoring(rcft_canon):
    """RCFT should define typed censoring enums."""
    censoring = rcft_canon["typed_censoring"]

    assert "fractal_regime_enum" in censoring
    assert "recursive_regime_enum" in censoring
    assert "pattern_type_enum" in censoring

    # Check fractal regime enum
    fractal_enum = censoring["fractal_regime_enum"]
    assert "Smooth" in fractal_enum
    assert "Wrinkled" in fractal_enum
    assert "Turbulent" in fractal_enum

    # Check recursive regime enum
    recursive_enum = censoring["recursive_regime_enum"]
    assert "Dormant" in recursive_enum
    assert "Active" in recursive_enum
    assert "Resonant" in recursive_enum

    # Check pattern type enum
    pattern_enum = censoring["pattern_type_enum"]
    assert "Standing" in pattern_enum
    assert "Traveling" in pattern_enum
    assert "Mixed" in pattern_enum


def test_rcft_computational_notes(rcft_canon):
    """RCFT should provide computational guidance."""
    notes = rcft_canon["computational_notes"]

    assert "fractal_dimension" in notes
    assert "recursive_field" in notes
    assert "pattern_wavelength" in notes

    # Check box-counting mentioned for fractal dimension
    assert "box-counting" in notes["fractal_dimension"].lower()

    # Check FFT mentioned for pattern wavelength
    assert (
        "FFT" in notes["pattern_wavelength"]
        or "fft" in notes["pattern_wavelength"].lower()
    )


def test_rcft_provenance(rcft_canon):
    """RCFT canon should have provenance information."""
    provenance = rcft_canon["provenance"]

    assert "created_by" in provenance
    assert "created_date" in provenance
    assert "parent_framework" in provenance
    assert provenance["parent_framework"] == "UMCP.GCD.v1"
