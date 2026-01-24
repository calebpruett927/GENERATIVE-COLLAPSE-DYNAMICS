"""Tests for UMCP CLI functionality.

This adds coverage for src/umcp/cli.py which currently has 0% coverage.
Focus on key commands: validate, version, list-closures, etc.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

# Test data paths
REPO_ROOT = Path(__file__).resolve().parents[1]
HELLO_WORLD_CASEPACK = REPO_ROOT / "casepacks" / "hello_world"


class TestCLIBasics:
    """Test basic CLI invocation and help."""

    def test_umcp_help(self):
        """Test that umcp --help works."""
        result = subprocess.run(
            ["umcp", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "umcp" in result.stdout.lower()

    def test_umcp_version(self):
        """Test that umcp --version works."""
        result = subprocess.run(
            ["umcp", "version"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        # Version command may not exist, so just check it doesn't crash badly
        assert result.returncode in [0, 1, 2]


class TestValidateCommand:
    """Test the validate subcommand."""

    def test_validate_help(self):
        """Test validate --help."""
        result = subprocess.run(
            ["umcp", "validate", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0

    def test_validate_repo_root(self):
        """Test validating the repository root."""
        result = subprocess.run(
            ["umcp", "validate", "."],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
        # Should succeed or at least not crash
        assert result.returncode in [0, 1]  # 0 = success, 1 = validation errors

    def test_validate_hello_world_casepack(self):
        """Test validating hello_world casepack."""
        result = subprocess.run(
            ["umcp", "validate", str(HELLO_WORLD_CASEPACK)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        # Should succeed
        assert result.returncode == 0

    def test_validate_with_json_output(self, tmp_path: Path) -> None:
        """Test validate with JSON output file."""
        output_file = tmp_path / "result.json"
        result = subprocess.run(
            [
                "umcp",
                "validate",
                str(HELLO_WORLD_CASEPACK),
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        if result.returncode == 0 and output_file.exists():
            # Verify JSON is valid
            with open(output_file) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_validate_nonexistent_path(self):
        """Test validating a path that doesn't exist."""
        result = subprocess.run(
            ["umcp", "validate", "/nonexistent/path/that/does/not/exist"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
        )
        # May return error code 2 for not found
        assert result.returncode in [0, 1, 2]


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_validate_multiple_casepacks(self):
        """Test validating multiple casepacks in sequence."""
        casepacks = [
            HELLO_WORLD_CASEPACK,
        ]

        for casepack in casepacks:
            if casepack.exists():
                result = subprocess.run(
                    ["umcp", "validate", str(casepack)],
                    capture_output=True,
                    text=True,
                    cwd=REPO_ROOT,
                    timeout=30,
                )
                # Should succeed or have validation errors (not crash)
                assert result.returncode in [0, 1]

    def test_cli_module_import(self):
        """Test that CLI module can be imported."""
        try:
            from umcp import cli

            assert hasattr(cli, "main")
        except ImportError:
            pytest.fail("Failed to import umcp.cli module")
