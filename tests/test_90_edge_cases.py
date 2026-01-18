"""
Test edge cases and error handling.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_yaml_handling(self, tmp_path):
        """Empty YAML files should be handled gracefully."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        
        with empty_file.open("r") as f:
            result = yaml.safe_load(f)
        
        assert result is None

    def test_invalid_yaml_handling(self, tmp_path):
        """Invalid YAML should raise appropriate error."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("{ invalid: yaml: content: }")
        
        with pytest.raises(yaml.YAMLError):
            with invalid_file.open("r") as f:
                yaml.safe_load(f)

    def test_empty_json_handling(self, tmp_path):
        """Empty JSON object should be valid."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")
        
        with empty_file.open("r") as f:
            result = json.load(f)
        
        assert result == {}

    def test_unicode_in_files(self, tmp_path):
        """Unicode characters should be handled correctly."""
        unicode_file = tmp_path / "unicode.yaml"
        unicode_content = """
        name: "Test with ω, κ, τ_R symbols"
        description: "Ψ phase space"
        """
        unicode_file.write_text(unicode_content)
        
        with unicode_file.open("r", encoding="utf-8") as f:
            result = yaml.safe_load(f)
        
        assert "ω" in result["name"]
        assert "Ψ" in result["description"]

    def test_large_number_handling(self):
        """Large numbers in JSON should be handled."""
        large_data = {"value": 10**100}
        serialized = json.dumps(large_data)
        deserialized = json.loads(serialized)
        
        assert deserialized["value"] == 10**100

    def test_deeply_nested_structure(self):
        """Deeply nested structures should be handled."""
        depth = 50
        nested = {"level": 0}
        current = nested
        
        for i in range(1, depth):
            current["child"] = {"level": i}
            current = current["child"]
        
        serialized = json.dumps(nested)
        deserialized = json.loads(serialized)
        
        assert deserialized["level"] == 0


class TestPathHandling:
    """Test file path edge cases."""

    def test_relative_path_resolution(self):
        """Relative paths should resolve correctly."""
        path = Path("casepacks/hello_world")
        full_path = REPO_ROOT / path
        
        assert full_path.exists()

    def test_absolute_path_handling(self):
        """Absolute paths should work."""
        abs_path = REPO_ROOT.absolute()
        assert abs_path.exists()
        assert abs_path.is_absolute()

    def test_path_with_spaces(self, tmp_path):
        """Paths with spaces should work."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        
        test_file = spaced_dir / "test.json"
        test_file.write_text("{}")
        
        assert test_file.exists()
        with test_file.open("r") as f:
            result = json.load(f)
        assert result == {}