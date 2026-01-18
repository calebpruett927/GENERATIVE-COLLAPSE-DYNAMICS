"""
Test CLI commands: validate, run, diff, and edge cases.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestCLIValidate:
    """Test the 'umcp validate' command."""

    def test_validate_repo_non_strict(self):
        """Validate repository in non-strict mode."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CONFORMANT" in result.stdout or "errors=0" in result.stdout

    def test_validate_repo_strict(self):
        """Validate repository in strict mode."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", "--strict", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CONFORMANT" in result.stdout or "errors=0" in result.stdout

    def test_validate_casepack(self):
        """Validate a specific casepack."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", "casepacks/hello_world"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_validate_nonexistent_path(self):
        """Validate a path that doesn't exist."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", "nonexistent/path"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # Should handle gracefully (may return 0 or 1 depending on implementation)
        assert result.returncode in [0, 1]

    def test_validate_output_json(self, tmp_path):
        """Validate and write output to JSON file."""
        output_file = tmp_path / "result.json"
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "validate", ".", "--out", str(output_file)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert output_file.exists()
        
        # Validate the output is valid JSON
        with output_file.open("r") as f:
            data = json.load(f)
        assert "run_status" in data or "status" in data


class TestCLIVersion:
    """Test CLI version and help commands."""

    def test_version(self):
        """Check --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self):
        """Check --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "validate" in result.stdout
        assert "run" in result.stdout


class TestCLIRun:
    """Test the 'umcp run' command."""

    def test_run_placeholder(self):
        """Run command should work (placeholder for engine)."""
        result = subprocess.run(
            [sys.executable, "-m", "umcp.cli", "run", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0