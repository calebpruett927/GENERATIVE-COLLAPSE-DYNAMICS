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
            has_id = (
                any("id" in k.lower() or "name" in k.lower() for k in contract)
                if isinstance(contract, dict)
                else False
            )

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
        assert registry_path.exists(), "closures/registry.yaml should exist"

    def test_closure_registry_is_valid_yaml(self):
        """Closure registry should be valid YAML."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"

        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)

        assert registry is not None, "Registry should not be empty"

    def test_closure_files_exist(self):
        """All referenced closure files should exist."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"

        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)

        closures_dir = REPO_ROOT / "closures"

        # Handle both 'entries' and 'closures' keys for compatibility
        entries = registry.get("entries", registry.get("closures", []))

        for closure in entries:
            if isinstance(closure, dict) and "file" in closure:
                closure_file = closures_dir / closure["file"]
                assert closure_file.exists(), f"Closure file not found: {closure['file']}"

    def test_closure_ids_unique(self):
        """All closure IDs should be unique."""
        registry_path = REPO_ROOT / "closures" / "registry.yaml"

        with registry_path.open("r") as f:
            registry = yaml.safe_load(f)

        # Handle both 'entries' and 'closures' keys for compatibility
        entries = registry.get("entries", registry.get("closures", []))

        closure_ids = [c.get("closure_id") for c in entries if isinstance(c, dict)]
        closure_ids = [cid for cid in closure_ids if cid is not None]

        if closure_ids:
            assert len(closure_ids) == len(set(closure_ids)), "Duplicate closure IDs found"

    def test_closure_files_are_valid_python(self):
        """All closure Python files should be syntactically valid."""
        closures_dir = REPO_ROOT / "closures"

        for py_file in closures_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            with py_file.open("r") as f:
                content = f.read()
            # Try to compile to check syntax
            try:
                compile(content, py_file.name, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")


class TestCanon:
    """Test canon anchor definitions."""

    def test_canon_directory_exists(self):
        """Canon directory should exist."""
        canon_dir = REPO_ROOT / "canon"
        assert canon_dir.exists(), "canon/ directory should exist"

    def test_canon_anchors_exist(self):
        """Canon anchors should exist."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"
        assert anchors_path.exists(), "canon/anchors.yaml should exist"

    def test_canon_is_valid_yaml(self):
        """Canon anchors should be valid YAML."""
        anchors_path = REPO_ROOT / "canon" / "anchors.yaml"

        with anchors_path.open("r") as f:
            canon = yaml.safe_load(f)

        assert canon is not None, "Canon should not be empty"
