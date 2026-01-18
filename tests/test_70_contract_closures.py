"""
Test contract and closure functionality.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


class TestContracts:
    """Test contract definitions and validation."""

    def test_contract_has_required_fields(self):
        """All contracts should have required fields."""
        contracts_dir = REPO_ROOT / "contracts"
        
        for contract_file in contracts_dir.glob("*.yaml"):
            with contract_file.open("r") as f:
                contract = yaml.safe_load(f)
            
            assert "contract_id" in contract, f"{contract_file.name} missing contract_id"
            assert "version" in contract, f"{contract_file.name} missing version"

    def test_contract_id_format(self):
        """Contract IDs should follow naming convention."""
        contracts_dir = REPO_ROOT / "contracts"
        
        for contract_file in contracts_dir.glob("*.yaml"):
            with contract_file.open("r") as f:
                contract = yaml.safe_load(f)
            
            contract_id = contract.get("contract_id", "")
            # Should be like "UMA.SOMETHING.v1"
            parts = contract_id.split(".")
            assert len(parts) >= 2, f"Contract ID {contract_id} should have at least 2 parts"

    def test_tier1_symbols_defined(self):
        """UMA.INTSTACK contract should define Tier-1 symbols."""
        contract_path = REPO_ROOT / "contracts" / "UMA.INTSTACK.v1.yaml"
        
        with contract_path.open("r") as f:
            contract = yaml.safe_load(f)
        
        # Look for tier1 or kernel symbols
        content = str(contract)
        tier1_symbols = ["ω", "omega", "F", "S", "C", "τ_R", "tau_R", "κ", "kappa", "IC"]
        
        found_symbols = [s for s in tier1_symbols if s in content]
        assert len(found_symbols) > 0, "No Tier-1 symbols found in contract"


class TestClosures:
    """Test closure registry and definitions."""

    def test_closure_registry_exists(self):
        """Closure registry should exist."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        assert registry_path.exists()

    def test_closure_registry_has_closures(self):
        """Closure registry should define closures."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)
        
        assert "closures" in registry
        assert len(registry["closures"]) > 0

    def test_closure_files_exist(self):
        """All referenced closure files should exist."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)
        
        closures_dir = REPO_ROOT / "closures"
        
        for closure in registry.get("closures", []):
            if "file" in closure:
                closure_file = closures_dir / closure["file"]
                assert closure_file.exists(), f"Closure file not found: {closure['file']}"

    def test_closure_ids_unique(self):
        """All closure IDs should be unique."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)
        
        closure_ids = [c.get("closure_id") for c in registry.get("closures", [])]
        assert len(closure_ids) == len(set(closure_ids)), "Duplicate closure IDs found"


class TestCanon:
    """Test canon anchor definitions."""

    def test_canon_anchors_exist(self):
        """Canon anchors should exist."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"
        assert anchors_path.exists()

    def test_canon_has_required_sections(self):
        """Canon should have required sections."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"
        
        with anchors_path.open("r") as f:
            canon = yaml.safe_load(f)
        
        assert canon is not None
        # Should be a non-empty structure
        assert len(canon) > 0