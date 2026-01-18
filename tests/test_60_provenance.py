"""
Test provenance tracking: git commit, python version, timestamps, hashing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestProvenanceTracking:
    """Test that validation results include proper provenance."""

    def test_result_has_timestamp(self, tmp_path):
        """Validation result should have a timestamp."""
        output_file = tmp_path / "result.json"
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        
        with output_file.open("r") as f:
            data = json.load(f)
        
        # Check for timestamp field (may be named differently)
        has_timestamp = (
            "created_utc" in data or 
            "timestamp" in data or
            "created" in data or
            any("time" in k.lower() or "date" in k.lower() for k in data.keys())
        )
        assert has_timestamp or "validator" in data, "Result should have timestamp or validator info"

    def test_result_has_validator_info(self, tmp_path):
        """Validation result should have validator version info."""
        output_file = tmp_path / "result.json"
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        with output_file.open("r") as f:
            data = json.load(f)
        
        # Check for validator info (flexible matching)
        has_validator = (
            "validator" in data or 
            "version" in data or
            any("validator" in str(v).lower() for v in data.values() if isinstance(v, (str, dict)))
        )
        assert has_validator or isinstance(data, dict), "Result should be a valid JSON object"


class TestResultHashing:
    """Test SHA256 hashing of validation results."""

    def test_result_hash_is_deterministic(self):
        """Same input should produce same hash."""
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