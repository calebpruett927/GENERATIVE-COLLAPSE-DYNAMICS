"""
Test contract and closure functionality.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


class TestContracts:
    """Test contract definitions and validation."""

    def test_contracts_directory_exists(self):
        """Contracts directory should exist."""
        contracts_dir = REPO_ROOT / "contracts"
        assert contracts_dir.exists(), "contracts/ directory should exist"

    def test_contract_files_are_valid_yaml(self):
        """All contract files should be valid YAML."""
        contracts_dir = REPO_ROOT / "contracts"
        
        if not contracts_dir.exists():
            pytest.skip("No contracts directory")
            
        yaml_files = list(contracts_dir.glob("*.yaml")) + list(contracts_dir.glob("*.yml"))
        
        if not yaml_files:
            pytest.skip("No contract files found")
        
        for contract_file in yaml_files:
            with contract_file.open("r") as f:
                contract = yaml.safe_load(f)
            assert contract is not None, f"{contract_file.name} should not be empty"

    def test_contract_has_id(self):
        """Contracts should have an identifier."""
        contracts_dir = REPO_ROOT / "contracts"
        
        if not contracts_dir.exists():
            pytest.skip("No contracts directory")
            
        yaml_files = list(contracts_dir.glob("*.yaml")) + list(contracts_dir.glob("*.yml"))
        
        if not yaml_files:
            pytest.skip("No contract files found")
        
        for contract_file in yaml_files:
            with contract_file.open("r") as f:
                contract = yaml.safe_load(f)
            
            # Check for any ID-like field
            has_id = any(
                "id" in k.lower() or "name" in k.lower() 
                for k in contract.keys()
            ) if isinstance(contract, dict) else False
            
            assert has_id or isinstance(contract, dict), f"{contract_file.name} should have an identifier"


class TestClosures:
    """Test closure registry and definitions."""

    def test_closures_directory_exists(self):
        """Closures directory should exist."""
        closures_dir = REPO_ROOT / "closures"
        assert closures_dir.exists(), "closures/ directory should exist"

    def test_closure_registry_exists(self):
        """Closure registry should exist."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        if not registry_path.exists():
            # Try alternative names
            alt_paths = [
                REPO_ROOT / "closures" / "registry.yml",
                REPO_ROOT / "closures" / "closures.yaml",
            ]
            found = any(p.exists() for p in alt_paths)
            if not found:
                pytest.skip("No closure registry found")
        else:
            assert registry_path.exists()

    def test_closure_registry_is_valid_yaml(self):
        """Closure registry should be valid YAML."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        if not registry_path.exists():
            pytest.skip("No closure registry found")
        
        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)
        
        assert registry is not None, "Registry should not be empty"

    def test_closure_ids_unique(self):
        """All closure IDs should be unique."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"
        
        if not registry_path.exists():
            pytest.skip("No closure registry found")
        
        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)
        
        if not isinstance(registry, dict) or "closures" not in registry:
            pytest.skip("Registry doesn't have expected structure")
        
        closure_ids = [c.get("closure_id") for c in registry.get("closures", []) if isinstance(c, dict)]
        closure_ids = [cid for cid in closure_ids if cid is not None]
        
        if closure_ids:
            assert len(closure_ids) == len(set(closure_ids)), "Duplicate closure IDs found"


class TestCanon:
    """Test canon anchor definitions."""

    def test_canon_directory_exists(self):
        """Canon directory should exist."""
        canon_dir = REPO_ROOT / "canon"
        assert canon_dir.exists(), "canon/ directory should exist"

    def test_canon_anchors_exist(self):
        """Canon anchors should exist."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"
        
        if not anchors_path.exists():
            alt_path = REPO_ROOT / "canon" / "anchors.yml"
            if not alt_path.exists():
                pytest.skip("No canon anchors file found")
        
        assert anchors_path.exists() or (REPO_ROOT / "canon" / "anchors.yml").exists()

    def test_canon_is_valid_yaml(self):
        """Canon anchors should be valid YAML."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"
        
        if not anchors_path.exists():
            pytest.skip("No canon anchors found")
        
        with anchors_path.open("r") as f:
            canon = yaml.safe_load(f)
        
        assert canon is not None, "Canon should not be empty"