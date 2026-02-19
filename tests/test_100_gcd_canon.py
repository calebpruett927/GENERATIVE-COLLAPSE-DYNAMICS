#!/usr/bin/env python3
"""
Test suite for GCD (Generative Collapse Dynamics) canon anchor.

Validates:
- GCD canon anchor schema conformance
- Reserved symbol definitions
- Axiom specifications (AX-0, AX-1, AX-2)
- Regime classification thresholds
- Mathematical identities
- Tolerance specifications
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def gcd_canon(repo_root):
    """Load GCD canon anchor."""
    canon_path = repo_root / "canon" / "gcd_anchors.yaml"
    assert canon_path.exists(), f"GCD canon not found: {canon_path}"

    with open(canon_path) as f:
        return yaml.safe_load(f)


def test_gcd_canon_exists(repo_root):
    """Test that GCD canon anchor file exists."""
    canon_path = repo_root / "canon" / "gcd_anchors.yaml"
    assert canon_path.exists(), "GCD canon anchor must exist"


def test_gcd_canon_schema(gcd_canon):
    """Test GCD canon has correct schema reference."""
    assert "schema" in gcd_canon, "Canon must reference schema"
    assert "canon.domain_anchors.schema.json" in gcd_canon["schema"]


def test_gcd_canon_id(gcd_canon):
    """Test GCD canon has correct ID."""
    assert "id" in gcd_canon
    assert gcd_canon["id"] == "UMCP.CANON.GCD.v1"


def test_gcd_canon_version(gcd_canon):
    """Test GCD canon has version."""
    assert "version" in gcd_canon
    assert gcd_canon["version"] == "1.0.0"


def test_gcd_axioms(gcd_canon):
    """Test GCD axioms are defined."""
    assert "axioms" in gcd_canon
    axioms = gcd_canon["axioms"]

    # Should have 3 axioms
    assert len(axioms) >= 3, "GCD must define at least 3 axioms"

    axiom_ids = [ax["id"] for ax in axioms]
    assert "AX-0" in axiom_ids, "Missing AX-0: Collapse is generative"
    assert "AX-1" in axiom_ids, "Missing AX-1: Boundary defines interior"
    assert "AX-2" in axiom_ids, "Missing AX-2: Entropy measures determinacy"


def test_gcd_axiom_ax0(gcd_canon):
    """Test AX-0: Collapse is generative."""
    axioms = {ax["id"]: ax for ax in gcd_canon["axioms"]}
    ax0 = axioms["AX-0"]

    assert "statement" in ax0
    assert "collapse" in ax0["statement"].lower()
    assert "generative" in ax0["statement"].lower()


def test_gcd_axiom_ax1(gcd_canon):
    """Test AX-1: Boundary defines interior."""
    axioms = {ax["id"]: ax for ax in gcd_canon["axioms"]}
    ax1 = axioms["AX-1"]

    assert "statement" in ax1
    assert "boundary" in ax1["statement"].lower()
    assert "interior" in ax1["statement"].lower()


def test_gcd_axiom_ax2(gcd_canon):
    """Test AX-2: Entropy measures determinacy."""
    axioms = {ax["id"]: ax for ax in gcd_canon["axioms"]}
    ax2 = axioms["AX-2"]

    assert "statement" in ax2
    assert "entropy" in ax2["statement"].lower()
    assert "determinacy" in ax2["statement"].lower()


def test_gcd_reserved_symbols(gcd_canon):
    """Test GCD reserved symbols are defined."""
    assert "tier_1_invariants" in gcd_canon
    tier1 = gcd_canon["tier_1_invariants"]
    assert "reserved_symbols" in tier1
    symbols = tier1["reserved_symbols"]

    # Check symbol list is present
    assert len(symbols) >= 8, "Should have at least 8 Tier-1 invariants"


def test_gcd_regime_classification(gcd_canon):
    """Test GCD regime classification is defined."""
    assert "regime_classification" in gcd_canon
    regimes = gcd_canon["regime_classification"]

    # Check regime_classification has regimes list
    assert "regimes" in regimes
    regime_labels = [r["label"] for r in regimes["regimes"]]
    assert "Stable" in regime_labels
    assert "Watch" in regime_labels
    assert "Collapse" in regime_labels


def test_gcd_regime_thresholds(gcd_canon):
    """Test regime thresholds are specified."""
    regimes = gcd_canon["regime_classification"]["regimes"]

    # Find Stable regime
    stable = next(r for r in regimes if r["label"] == "Stable")
    assert "condition" in stable or "thresholds" in stable


def test_gcd_mathematical_identities(gcd_canon):
    """Test GCD mathematical identities are defined."""
    assert "mathematical_identities" in gcd_canon
    identities = gcd_canon["mathematical_identities"]

    # Core identities exist
    assert len(identities) >= 2


def test_gcd_tolerances(gcd_canon):
    """Test GCD tolerances are specified."""
    assert "tolerances" in gcd_canon
    tol = gcd_canon["tolerances"]

    # Check for gates structure
    assert "gates" in tol
    gate_names = [g["name"] for g in tol["gates"]]
    assert "tol_seam" in gate_names or "seam_continuity" in " ".join([g.get("description", "") for g in tol["gates"]])


def test_gcd_extensions(gcd_canon):
    """Test GCD extensions are defined."""
    # Check for gcd_extensions section (lowercase)
    assert "gcd_extensions" in gcd_canon or "GCD_extensions" in gcd_canon or "tier_2_extensions" in gcd_canon
    if "gcd_extensions" in gcd_canon:
        ext = gcd_canon["gcd_extensions"]
        assert len(ext) >= 2


def test_gcd_provenance(gcd_canon):
    """Test GCD canon has provenance metadata."""
    assert "provenance" in gcd_canon
    prov = gcd_canon["provenance"]

    assert "created_date" in prov or "created" in prov
    assert "created_by" in prov or "author" in prov


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
