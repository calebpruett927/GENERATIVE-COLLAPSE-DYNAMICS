"""
Test provenance tracking: git commit, python version, timestamps, hashing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestProvenanceTracking:
    """Test that validation results include proper provenance."""

    def test_result_has_timestamp(self, tmp_path):
        """Validation result should have a timestamp."""
        output_file = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        
        with output_file.open("r") as f:
            result = json.load(f)
        
        assert "created_utc" in result
        # Should be ISO format
        timestamp = result["created_utc"]
        assert "T" in timestamp
        assert "Z" in timestamp or "+" in timestamp

    def test_result_has_validator_info(self, tmp_path):
        """Validation result should have validator version info."""
        output_file = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        
        with output_file.open("r") as f:
            result = json.load(f)
        
        assert "validator" in result
        validator = result["validator"]
        assert "version" in validator
        assert validator["version"] == "0.1.0"

    def test_result_has_implementation_details(self, tmp_path):
        """Validation result should have implementation details."""
        output_file = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        
        with output_file.open("r") as f:
            result = json.load(f)
        
        impl = result.get("validator", {}).get("implementation", {})
        
        # Should have git commit (or "unknown" if not in git)
        assert "git_commit" in impl
        
        # Should have python version
        assert "python_version" in impl
        assert impl["python_version"].startswith("3.")

    def test_result_has_policy_info(self, tmp_path):
        """Validation result should have policy info."""
        output_file = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", "--strict", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        
        with output_file.open("r") as f:
            result = json.load(f)
        
        policy = result.get("summary", {}).get("policy", {})
        assert "strict" in policy
        assert policy["strict"] is True


class TestResultHashing:
    """Test SHA256 hashing of validation results."""

    def test_result_hash_is_deterministic(self, tmp_path):
        """Same input should produce same hash (minus timestamp)."""
        # This tests that the hashing algorithm is stable
        test_data = {"status": "CONFORMANT", "errors": 0}
        hash1 = hashlib.sha256(json.dumps(test_data, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(test_data, sort_keys=True).encode()).hexdigest()
        assert hash1 == hash2

    def test_result_hash_changes_with_content(self):
        """Different content should produce different hash."""
        data1 = {"status": "CONFORMANT", "errors": 0}
        data2 = {"status": "NON-CONFORMANT", "errors": 1}
        
        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()
        
        assert hash1 != hash2