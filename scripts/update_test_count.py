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

        # Primary: count lines matching test item format (path::class::test)
        test_lines = [
            line for line in result.stdout.splitlines()
            if line.strip().startswith("tests/") and "::" in line
        ]
        if test_lines:
            return len(test_lines)

        # Fallback: look for "N tests collected" or "N passed"
        for pattern in [r"(\d+) tests? collected", r"(\d+) passed"]:
            match = re.search(pattern, result.stdout)
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
    """Update test count in a documentation file.

    Uses targeted patterns to avoid corrupting comma-formatted numbers
    or unrelated numeric references (e.g., "74/2476 tests", "10,162 tests").
    """
    if not filepath.exists():
        print(f"Warning: {filepath} not found", file=sys.stderr)
        return False

    content = filepath.read_text(encoding="utf-8")
    original = content

    # Badge URL: tests-NNNN%20passing
    content = re.sub(r"tests-\d+%2B?%20passing", f"tests-{test_count}%20passing", content)
    # Badge alt text: Tests: NNNN passing
    content = re.sub(r"Tests: \d+\+? passing", f"Tests: {test_count} passing", content)

    # Table: | **Tests** | NNNN passing (NN files) |
    content = re.sub(
        r"\| \*\*Tests\*\* \| [\d,]+\+? passing \(\d+\+? files\) \|",
        f"| **Tests** | {test_count:,} passing ({test_files} files) |",
        content,
    )

    # "Run NNNN tests" or "Run all NNNN tests" (with possible comma formatting)
    content = re.sub(
        r"(Run(?:\s+all)?\s+)[\d,]+(\s+tests?\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # "# All NNNN tests" (comment headers)
    content = re.sub(
        r"(#\s+All\s+)[\d,]+(\s+tests?\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # "All NNNN pass" (quality table)
    content = re.sub(
        r"(All\s+)[\d,]+(\s+pass\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # "Test Distribution (NN files, NNNN tests)"
    content = re.sub(
        r"(Test Distribution\s*\()[\d,]+ files,\s*[\d,]+ tests(\))",
        rf"\g<1>{test_files} files, {test_count:,} tests\2",
        content,
    )

    # "Test suite (NN files, NNNN tests)" in tree/structure diagrams
    content = re.sub(
        r"(Test suite\s*\()[\d,]+ files,\s*[\d,]+ tests(\))",
        rf"\g<1>{test_files} files, {test_count:,} tests\2",
        content,
    )

    # "tests/ for NNNN examples" (QUICKSTART)
    content = re.sub(
        r"(tests/['\"` ]+for\s+)[\d,]+(\s+examples?\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # Footer: "NNNN tests •" (with bullet)
    content = re.sub(
        r"[\d,]+ tests(\s*&bull;|\s*•)",
        f"{test_count:,} tests\\1",
        content,
    )

    # "Should show NNNN tests" (CONTRIBUTING)
    content = re.sub(
        r"(Should show\s+)[\d,]+(\s+tests?\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # "Suite (NNNN+ tests)" or "suite (NNNN tests)"
    content = re.sub(
        r"(suite\s*\()[\d,]+\+?(\s+tests?\))",
        rf"\g<1>{test_count:,}\2",
        content,
    )

    # "update test count to NNNN" (example commit messages)
    content = re.sub(
        r"(update test count to\s+)[\d,]+",
        rf"\g<1>{test_count:,}",
        content,
    )

    # "tests/ for NNNN examples" (looser match for QUICKSTART)
    content = re.sub(
        r"(tests/[`'\"]?\s+for\s+)[\d,]+(\s+examples?\b)",
        rf"\g<1>{test_count:,}\2",
        content,
    )

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
        repo_root / "CONTRIBUTING.md",
        repo_root / "COMMIT_PROTOCOL.md",
        repo_root / "QUICKSTART_TUTORIAL.md",
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
