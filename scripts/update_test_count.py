#!/usr/bin/env python3
"""
Update test count in README.md and copilot-instructions.md.

This script runs pytest --collect-only to get the actual test count,
then updates the badge and table entries in documentation files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def get_test_count() -> int:
    """Get the total number of tests by running pytest --collect-only."""
    try:
        result = subprocess.run(
            ["pytest", "--collect-only"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Look for "N tests collected" in output
        match = re.search(r"(\d+) tests? collected", result.stdout)
        if match:
            return int(match.group(1))

        print("Warning: Could not parse test count from pytest output", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error running pytest: {e}", file=sys.stderr)
        return 0


def count_test_files() -> int:
    """Count test files in tests/ directory."""
    test_dir = Path("tests")
    if not test_dir.exists():
        return 0
    return len(list(test_dir.glob("test_*.py")))


def update_file(filepath: Path, test_count: int, test_files: int) -> bool:
    """Update test count in a documentation file."""
    if not filepath.exists():
        print(f"Warning: {filepath} not found", file=sys.stderr)
        return False

    content = filepath.read_text(encoding="utf-8")
    original = content

    # Update badge (README.md)
    content = re.sub(r"tests-\d+%2B?%20passing", f"tests-{test_count}%20passing", content)
    content = re.sub(r"Tests: \d+\+? passing", f"Tests: {test_count} passing", content)

    # Update table entries (both README and copilot instructions)
    # Format: | **Tests** | 1900+ passing (80 files) |
    content = re.sub(
        r"\| \*\*Tests\*\* \| \d+\+? passing \(\d+ files\) \|",
        f"| **Tests** | {test_count} passing ({test_files} files) |",
        content,
    )

    # Update references in text like "1900+ tests"
    content = re.sub(r"\b\d+\+? tests?\b", f"{test_count} tests", content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        print(f"✓ Updated {filepath}")
        return True
    else:
        print(f"  No changes needed in {filepath}")
        return False


def main() -> int:
    """Update test counts in all documentation files."""
    repo_root = Path.cwd()

    # Get counts
    print("Collecting tests...")
    test_count = get_test_count()
    test_files = count_test_files()

    if test_count == 0:
        print("Error: No tests collected. Is pytest installed?", file=sys.stderr)
        return 1

    print(f"Found {test_count} tests in {test_files} files")

    # Update files
    files_to_update = [
        repo_root / "README.md",
        repo_root / ".github" / "copilot-instructions.md",
    ]

    updated = False
    for filepath in files_to_update:
        if update_file(filepath, test_count, test_files):
            updated = True

    if updated:
        print(f"\n✓ Test count updated to {test_count}")
        return 0
    else:
        print("\nNo updates needed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
