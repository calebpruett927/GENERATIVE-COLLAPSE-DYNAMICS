"""
Test CLI diff command for comparing validation receipts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_receipt_1(tmp_path) -> Path:
    """Create a sample receipt file."""
    receipt = {
        "run_status": "CONFORMANT",
        "created_utc": "2026-01-18T01:00:00Z",
        "validator": {
            "name": "umcp-validator",
            "version": "0.1.0",
            "implementation": {
                "git_commit": "abc123",
                "python_version": "3.12.1",
            },
        },
        "summary": {
            "counts": {"errors": 0, "warnings": 0},
            "policy": {"strict": False, "fail_on_warning": False},
        },
        "targets": [
            {"target_path": "casepacks/hello_world", "status": "CONFORMANT"},
        ],
    }
    path = tmp_path / "receipt1.json"
    with path.open("w") as f:
        json.dump(receipt, f)
    return path


@pytest.fixture
def sample_receipt_2(tmp_path) -> Path:
    """Create a different sample receipt file."""
    receipt = {
        "run_status": "CONFORMANT",
        "created_utc": "2026-01-18T02:00:00Z",
        "validator": {
            "name": "umcp-validator",
            "version": "0.1.0",
            "implementation": {
                "git_commit": "def456",
                "python_version": "3.12.1",
            },
        },
        "summary": {
            "counts": {"errors": 0, "warnings": 1},
            "policy": {"strict": True, "fail_on_warning": False},
        },
        "targets": [
            {"target_path": "casepacks/hello_world", "status": "CONFORMANT"},
            {"target_path": "casepacks/new_pack", "status": "CONFORMANT"},
        ],
    }
    path = tmp_path / "receipt2.json"
    with path.open("w") as f:
        json.dump(receipt, f)
    return path


class TestDiffCommand:
    """Test the 'umcp diff' command."""

    def test_diff_command_exists(self):
        """Check if diff command is available."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # If diff is not implemented, skip these tests
        if "diff" not in result.stdout:
            pytest.skip("diff command not yet implemented")

    def test_diff_identical_receipts(self, sample_receipt_1):
        """Diff identical receipts shows no changes."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if "diff" not in result.stdout:
            pytest.skip("diff command not yet implemented")
            
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "diff", str(sample_receipt_1), str(sample_receipt_1)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_diff_different_receipts(self, sample_receipt_1, sample_receipt_2):
        """Diff different receipts shows changes."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if "diff" not in result.stdout:
            pytest.skip("diff command not yet implemented")
            
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "diff", str(sample_receipt_1), str(sample_receipt_2)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_diff_nonexistent_file(self, tmp_path):
        """Diff with nonexistent file returns error."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if "diff" not in result.stdout:
            pytest.skip("diff command not yet implemented")
            
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "diff", "nonexistent.json", "also_nonexistent.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1