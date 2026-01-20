# Test package for UMCP validator
"""
Test package initializer for UMCP-Metadata-Runnable-Code.

Goals:
- Make `umcp` importable when running tests from repo root (and in CI).
- Keep side effects minimal, deterministic, and cross-platform.
- Provide a couple of small helpers for test modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_syspath() -> None:
    """Ensure repo root is on sys.path so `import umcp` works consistently."""
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        # Prepend so local package wins over any installed copy.
        sys.path.insert(0, root_str)


def _set_test_timezone() -> None:
    """
    UMCP project timezone default is America/Chicago.
    Set TZ for deterministic time formatting in tests that render timestamps.
    """
    os.environ.setdefault("TZ", "America/Chicago")
    # tzset is POSIX; guard for platforms where it doesn't exist.
    try:
        import time

        if hasattr(time, "tzset"):
            time.tzset()
    except Exception:
        # Do not fail tests because timezone cannot be forced on this platform.
        pass


def repo_root() -> Path:
    """Return the repository root path."""
    return REPO_ROOT


def testdata_path(*parts: str) -> Path:
    """
    Convenience helper for test modules to locate fixtures.
    Example: testdata_path("casepacks", "hello_world", "manifest.json")
    """
    return REPO_ROOT.joinpath(*parts)


# Apply minimal, safe initialization.
_ensure_repo_root_on_syspath()
_set_test_timezone()

__all__ = ["REPO_ROOT", "repo_root", "testdata_path"]
