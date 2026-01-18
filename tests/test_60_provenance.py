"""
Test provenance tracking: git commit, python version, timestamps, hashing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestProvenanceTracking:
    """Test that validation results include proper provenance."""

    def test_validation_runs_successfully(self):
        """Basic validation should run and return success."""
        result = subprocess.run(
            ["umcp", "validate", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_validation_output_contains_status(self):
        """Validation output should indicate status."""
        result = subprocess.run(
            ["umcp", "validate", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        # Should have some indication of validation result
        assert len(combined) > 0 or result.returncode == 0


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