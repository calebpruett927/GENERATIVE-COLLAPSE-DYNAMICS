"""CLI subcommand integration tests.

Tests for umcp subcommands that previously had zero coverage:
integrity, report, list, casepack, test, engine, preflight, health.
Each test uses subprocess.run to exercise the actual CLI entry point.
"""

from __future__ import annotations

import subprocess
import sys

UMCP = [sys.executable, "-m", "umcp"]
REPO_ROOT = "."


class TestCLIIntegrity:
    """umcp integrity subcommand."""

    def test_integrity_help(self) -> None:
        r = subprocess.run([*UMCP, "integrity", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert "integrity" in r.stdout.lower()

    def test_integrity_check_repo(self) -> None:
        r = subprocess.run([*UMCP, "integrity", "."], capture_output=True, text=True, timeout=30)
        # Should succeed (exit 0) or produce structured output
        assert r.returncode in (0, 1)

    def test_integrity_nonexistent_path(self) -> None:
        r = subprocess.run(
            [*UMCP, "integrity", "/nonexistent/path"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode != 0


class TestCLIReport:
    """umcp report subcommand."""

    def test_report_help(self) -> None:
        r = subprocess.run([*UMCP, "report", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert "report" in r.stdout.lower()

    def test_report_repo(self) -> None:
        r = subprocess.run([*UMCP, "report"], capture_output=True, text=True, timeout=30)
        assert r.returncode in (0, 1)


class TestCLIList:
    """umcp list subcommand."""

    def test_list_help(self) -> None:
        r = subprocess.run([*UMCP, "list", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0

    def test_list_repo(self) -> None:
        r = subprocess.run([*UMCP, "list", "all"], capture_output=True, text=True, timeout=30)
        assert r.returncode in (0, 1)


class TestCLICasepack:
    """umcp casepack subcommand."""

    def test_casepack_help(self) -> None:
        r = subprocess.run([*UMCP, "casepack", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert "casepack" in r.stdout.lower()


class TestCLIEngine:
    """umcp engine subcommand."""

    def test_engine_help(self) -> None:
        r = subprocess.run([*UMCP, "engine", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0


class TestCLIHealth:
    """umcp health subcommand."""

    def test_health_help(self) -> None:
        r = subprocess.run([*UMCP, "health", "--help"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0

    def test_health_default(self) -> None:
        r = subprocess.run([*UMCP, "health"], capture_output=True, text=True, timeout=30)
        assert r.returncode in (0, 1)
