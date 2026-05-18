#!/usr/bin/env python3
"""Sync Ground Truth — propagate frozen metrics to every file in the repo.

Reads metrics from ``ground_truth.py``, then sweeps all files that reference
those metrics and updates them in-place.  Designed to be idempotent — running
it twice produces no diff.

This script replaces manual regex hunts.  Every file × pattern is declared
once in the ``_SYNC_RULES`` table.  To add coverage for a new file, add a
rule.  To add a new metric, add it to ``ground_truth.py`` and reference it
here via ``{metric_name}`` in replacement templates.

Usage:
    python scripts/sync_ground_truth.py              # Full sync (runs pytest)
    python scripts/sync_ground_truth.py --skip-pytest # Fast sync (cached test count)
    python scripts/sync_ground_truth.py --dry-run     # Report only, no writes
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Add scripts/ to path so we can import ground_truth
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ground_truth import GroundTruth

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Rule definitions ─────────────────────────────────────────────


@dataclass
class SyncRule:
    """A single find-and-replace rule for propagating a ground-truth value.

    Parameters
    ----------
    file : str
        Path relative to repo root.
    pattern : str
        Regex with a capture group around the VALUE to replace.
        The regex matches the surrounding context; group 1 is replaced.
    replacement : str
        Template string using ``{field}`` placeholders from GroundTruth.
        This replaces the captured group 1.
    description : str
        Human-readable description for --dry-run output.
    """

    file: str
    pattern: str
    replacement: str
    description: str = ""


def _rules() -> list[SyncRule]:
    """All sync rules — the complete mapping from ground truth to files.

    Naming conventions for replacement templates:
        {test_count}          → raw int: 20221
        {test_count_comma}    → formatted: 20,221
        {test_count_url}      → URL-safe: 20%2C221
        {domain_count}        → raw int: 23
        {closure_count}       → raw int: 245
        {test_file_count}     → raw int: 231
        {theorem_count}       → raw int: 746
        {identity_count}      → raw int: 44
        {lemma_count}         → raw int: 47
        {version}             → string: 2.3.1
        {casepack_count}      → raw int: 29
        {canon_count}         → raw int: 22
        {contract_count}      → raw int: 23
        {schema_count}        → raw int: 17
        {c_assertions}        → raw int: 326
        {cpp_assertions}      → raw int: 434
        {total_c_cpp_assertions} → raw int: 760
    """
    return [
        # ══════════════════════════════════════════════════════════
        # WEB: metrics.ts — the web layer's single source of truth
        # ══════════════════════════════════════════════════════════
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const TEST_COUNT = '([\d,]+)'",
            "{test_count_comma}",
            "metrics.ts TEST_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const TEST_COUNT_RAW = ([\d_]+)",
            "{test_count_raw_ts}",
            "metrics.ts TEST_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const DOMAIN_COUNT = '(\d+)'",
            "{domain_count}",
            "metrics.ts DOMAIN_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const DOMAIN_COUNT_RAW = (\d+)",
            "{domain_count}",
            "metrics.ts DOMAIN_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const IDENTITY_COUNT = '(\d+)'",
            "{identity_count}",
            "metrics.ts IDENTITY_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const IDENTITY_COUNT_RAW = (\d+)",
            "{identity_count}",
            "metrics.ts IDENTITY_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts", r"export const LEMMA_COUNT = '(\d+)'", "{lemma_count}", "metrics.ts LEMMA_COUNT"
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const LEMMA_COUNT_RAW = (\d+)",
            "{lemma_count}",
            "metrics.ts LEMMA_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const CLOSURE_COUNT = '(\d+)'",
            "{closure_count}",
            "metrics.ts CLOSURE_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const CLOSURE_COUNT_RAW = (\d+)",
            "{closure_count}",
            "metrics.ts CLOSURE_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const THEOREM_COUNT = '(\d+)'",
            "{theorem_count}",
            "metrics.ts THEOREM_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const THEOREM_COUNT_RAW = (\d+)",
            "{theorem_count}",
            "metrics.ts THEOREM_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const TEST_FILE_COUNT = '(\d+)'",
            "{test_file_count}",
            "metrics.ts TEST_FILE_COUNT",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const TEST_FILE_COUNT_RAW = (\d+)",
            "{test_file_count}",
            "metrics.ts TEST_FILE_COUNT_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const C_ASSERTIONS = '(\d+)'",
            "{c_assertions}",
            "metrics.ts C_ASSERTIONS",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const C_ASSERTIONS_RAW = (\d+)",
            "{c_assertions}",
            "metrics.ts C_ASSERTIONS_RAW",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const CPP_ASSERTIONS = '(\d+)'",
            "{cpp_assertions}",
            "metrics.ts CPP_ASSERTIONS",
        ),
        SyncRule(
            "web/src/lib/metrics.ts",
            r"export const CPP_ASSERTIONS_RAW = (\d+)",
            "{cpp_assertions}",
            "metrics.ts CPP_ASSERTIONS_RAW",
        ),
        # ══════════════════════════════════════════════════════════
        # WEB: IndexLayout.astro — main landing page
        # ══════════════════════════════════════════════════════════
        SyncRule(
            "web/src/layouts/IndexLayout.astro",
            r"across (\d+) scientific domains",
            "{domain_count}",
            "IndexLayout description",
        ),
        SyncRule(
            "web/src/layouts/IndexLayout.astro",
            r"Domain catalog: (\d+) closure domains",
            "{domain_count}",
            "IndexLayout catalog comment",
        ),
        SyncRule(
            "web/src/layouts/IndexLayout.astro",
            r"const totalTheorems = (\d+);",
            "{theorem_count}",
            "IndexLayout totalTheorems",
        ),
        # ══════════════════════════════════════════════════════════
        # WEB: components
        # ══════════════════════════════════════════════════════════
        SyncRule(
            "web/src/components/RosettaTranslator.astro",
            r"across all (\d+) domains",
            "{domain_count}",
            "RosettaTranslator domain count",
        ),
        # ══════════════════════════════════════════════════════════
        # WEB: llms.txt (LLM-readable summary)
        # ══════════════════════════════════════════════════════════
        SyncRule("web/public/llms.txt", r"Version: ([\d.]+)", "{version}", "llms.txt version"),
        SyncRule("web/public/llms.txt", r"([\d,]+) automated tests", "{test_count_comma}", "llms.txt test count"),
        SyncRule("web/public/llms.txt", r"(\d+) test files", "{test_file_count}", "llms.txt test file count"),
        SyncRule("web/public/llms.txt", r"## (\d+) Domain Closures", "{domain_count}", "llms.txt domain count"),
        # ══════════════════════════════════════════════════════════
        # PYTHON: version strings
        # ══════════════════════════════════════════════════════════
        SyncRule("pyproject.toml", r'^version = "([\d.]+)"', "{version}", "pyproject.toml version"),
        SyncRule("src/umcp/__init__.py", r'__version__ = "([\d.]+)"', "{version}", "__init__.py version"),
        SyncRule("src/umcp/api_umcp.py", r'__version__ = "([\d.]+)"', "{version}", "api_umcp.py version"),
        # ══════════════════════════════════════════════════════════
        # README.md — badges and key narrative references
        # ══════════════════════════════════════════════════════════
        SyncRule("README.md", r"tests-([\d%2C]+)-brightgreen", "{test_count_url}", "README badge: tests"),
        SyncRule("README.md", r"theorems-(\d+)-purple", "{theorem_count}", "README badge: theorems"),
        SyncRule("README.md", r"domains-(\d+)-teal", "{domain_count}", "README badge: domains"),
        SyncRule("README.md", r"closures-(\d+)-blue", "{closure_count}", "README badge: closures"),
        SyncRule("README.md", r"test.files-(\d+)-green", "{test_file_count}", "README badge: test files"),
        # ══════════════════════════════════════════════════════════
        # README_PYPI.md — badges and key references
        # ══════════════════════════════════════════════════════════
        SyncRule("README_PYPI.md", r"tests-([\d%2C]+)-brightgreen", "{test_count_url}", "PyPI badge: tests"),
        SyncRule("README_PYPI.md", r"!\[Tests: ([\d,]+)\]", "{test_count_comma}", "PyPI badge alt: tests"),
        SyncRule("README_PYPI.md", r"Theorems: (\d+)", "{theorem_count}", "PyPI badge: theorems"),
        SyncRule("README_PYPI.md", r"Domains: (\d+)", "{domain_count}", "PyPI badge: domains"),
        SyncRule("README_PYPI.md", r"\*\*(\d+) scientific domains\*\*", "{domain_count}", "PyPI key features: domains"),
        SyncRule("README_PYPI.md", r"\*\*(\d+) proven theorems\*\*", "{theorem_count}", "PyPI key features: theorems"),
        SyncRule("README_PYPI.md", r"\*\*([\d,]+) tests\*\*", "{test_count_comma}", "PyPI key features: tests"),
        SyncRule("README_PYPI.md", r"across (\d+) files with", "{test_file_count}", "PyPI test files"),
        SyncRule("README_PYPI.md", r"with (\d+) closure modules", "{closure_count}", "PyPI closure count"),
        SyncRule("README_PYPI.md", r"## (\d+) Scientific Domains", "{domain_count}", "PyPI domain header"),
        SyncRule(
            "README_PYPI.md",
            r"# (\d+) domain closure modules \(",
            "{domain_count}",
            "PyPI project structure: domain count",
        ),
        SyncRule(
            "README_PYPI.md",
            r"modules \((\d+) \.py files\)",
            "{closure_count}",
            "PyPI project structure: closure .py files",
        ),
        SyncRule("README_PYPI.md", r"(\d+) canonical anchor files", "{canon_count}", "PyPI canon files"),
        SyncRule("README_PYPI.md", r"(\d+) versioned mathematical contracts", "{contract_count}", "PyPI contracts"),
        SyncRule("README_PYPI.md", r"(\d+) JSON Schema", "{schema_count}", "PyPI schemas"),
        SyncRule("README_PYPI.md", r"(\d+) self-contained validation", "{casepack_count}", "PyPI casepacks"),
        SyncRule(
            "README_PYPI.md",
            r"\*\*(\d+) casepacks\*\*",
            "{casepack_count}",
            "PyPI key features: casepacks",
        ),
        # ══════════════════════════════════════════════════════════
        # CONTRIBUTING.md
        # ══════════════════════════════════════════════════════════
        SyncRule("CONTRIBUTING.md", r"tests-([\d%2C]+)-brightgreen", "{test_count_url}", "CONTRIBUTING badge"),
        SyncRule("CONTRIBUTING.md", r"([\d,]+) tests", "{test_count_comma}", "CONTRIBUTING test count"),
        SyncRule("CONTRIBUTING.md", r"(\d+) test files", "{test_file_count}", "CONTRIBUTING test files"),
        # ══════════════════════════════════════════════════════════
        # Agent instruction files (.github/copilot-instructions.md, AGENTS.md, CLAUDE.md)
        # ══════════════════════════════════════════════════════════
        SyncRule(".github/copilot-instructions.md", r"UMCP v([\d.]+)", "{version}", "copilot-instructions version"),
        SyncRule(
            ".github/copilot-instructions.md",
            r"\*\*([\d,]+) tests\*\*",
            "{test_count_comma}",
            "copilot-instructions tests",
        ),
        SyncRule(
            ".github/copilot-instructions.md",
            r"\*\*(\d+) domains\*\*",
            "{domain_count}",
            "copilot-instructions domains",
        ),
        SyncRule(
            ".github/copilot-instructions.md",
            r"\*\*(\d+) closure modules\*\*",
            "{closure_count}",
            "copilot-instructions closures",
        ),
        SyncRule(
            ".github/copilot-instructions.md", r"\*\*(\d+) lemmas\*\*", "{lemma_count}", "copilot-instructions lemmas"
        ),
        SyncRule(
            ".github/copilot-instructions.md",
            r"\*\*(\d+) structural identities\*\*",
            "{identity_count}",
            "copilot-instructions identities",
        ),
        SyncRule(
            ".github/copilot-instructions.md",
            r"([\d,]+) test cases",
            "{test_count_comma}",
            "copilot-instructions test cases",
        ),
        SyncRule(
            ".github/copilot-instructions.md",
            r"(\d+) test files",
            "{test_file_count}",
            "copilot-instructions test files",
        ),
        SyncRule("AGENTS.md", r"Tests \(([\d,]+)\)", "{test_count_comma}", "AGENTS tests"),
        SyncRule("CLAUDE.md", r"UMCP v([\d.]+)", "{version}", "CLAUDE version"),
        # ══════════════════════════════════════════════════════════
        # QUICKSTART_TUTORIAL.md
        # ══════════════════════════════════════════════════════════
        SyncRule("QUICKSTART_TUTORIAL.md", r"([\d,]+) examples", "{test_count_comma}", "QUICKSTART examples"),
        # ══════════════════════════════════════════════════════════
        # CATALOGUE.md
        # ══════════════════════════════════════════════════════════
        SyncRule("CATALOGUE.md", r"UMCP v([\d.]+)", "{version}", "CATALOGUE version"),
        # ══════════════════════════════════════════════════════════
        # pyproject.toml description line
        # ══════════════════════════════════════════════════════════
        SyncRule("pyproject.toml", r"(\d+) scientific domains", "{domain_count}", "pyproject description: domains"),
        SyncRule("pyproject.toml", r"(\d+) proven theorems", "{theorem_count}", "pyproject description: theorems"),
        SyncRule(
            "pyproject.toml", r"([\d,]+) tests, three-layer", "{test_count_comma}", "pyproject description: tests"
        ),
        SyncRule("pyproject.toml", r"(\d+) closure modules", "{closure_count}", "pyproject description: closures"),
        # ══════════════════════════════════════════════════════════
        # MANIFEST.in version comment
        # ══════════════════════════════════════════════════════════
        SyncRule("MANIFEST.in", r"UMCP v([\d.]+)", "{version}", "MANIFEST.in version"),
        # ══════════════════════════════════════════════════════════
        # scripts/test_count.txt — canonical oracle for test count
        # ══════════════════════════════════════════════════════════
        SyncRule("scripts/test_count.txt", r"^(\d+)$", "{test_count}", "test_count.txt oracle"),
    ]


# ── Sync engine ──────────────────────────────────────────────────


def _format_value(template: str, gt: GroundTruth) -> str:
    """Expand a replacement template against GroundTruth fields."""
    # Special computed fields not directly in the dataclass
    extras = {
        "test_count_comma": gt.test_count_comma,
        "test_count_url": gt.test_count_url,
        "test_count_raw_ts": f"{gt.test_count:_}".replace("_", "_"),  # e.g. 20_221
        "total_c_cpp_assertions": str(gt.total_c_cpp_assertions),
    }
    result = template
    # Direct dataclass fields
    for field_name in [
        "version",
        "test_count",
        "domain_count",
        "closure_count",
        "test_file_count",
        "theorem_count",
        "identity_count",
        "lemma_count",
        "casepack_count",
        "canon_count",
        "contract_count",
        "schema_count",
        "c_assertions",
        "cpp_assertions",
    ]:
        result = result.replace(f"{{{field_name}}}", str(getattr(gt, field_name)))
    # Extras
    for key, val in extras.items():
        result = result.replace(f"{{{key}}}", val)
    return result


def sync_file(
    filepath: Path,
    rules: list[SyncRule],
    gt: GroundTruth,
    dry_run: bool = False,
) -> list[str]:
    """Apply all rules for a single file.  Returns list of changes made."""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    original = content
    changes: list[str] = []

    for rule in rules:
        new_val = _format_value(rule.replacement, gt)

        def _replacer(m: re.Match, _nv: str = new_val, _rule: SyncRule = rule) -> str:
            full = m.group(0)
            old_val = m.group(1)
            if old_val == _nv:
                return full
            changes.append(f"  {_rule.description}: {old_val!r} → {_nv!r}")
            return full[: m.start(1) - m.start(0)] + _nv + full[m.end(1) - m.start(0) :]

        content = re.sub(rule.pattern, _replacer, content, flags=re.MULTILINE)

    if content != original and not dry_run:
        filepath.write_text(content, encoding="utf-8")

    return changes


def sync_all(
    gt: GroundTruth,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Sync all files.  Returns {relative_path: [changes]}."""
    repo = root or _REPO_ROOT
    rules = _rules()

    # Group rules by file
    by_file: dict[str, list[SyncRule]] = {}
    for rule in rules:
        by_file.setdefault(rule.file, []).append(rule)

    all_changes: dict[str, list[str]] = {}
    for rel_path, file_rules in by_file.items():
        filepath = repo / rel_path
        changes = sync_file(filepath, file_rules, gt, dry_run=dry_run)
        if changes:
            all_changes[rel_path] = changes

    return all_changes


# ── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    skip_pytest = "--skip-pytest" in sys.argv

    print("Computing ground truth...")
    gt = GroundTruth.compute(skip_pytest=skip_pytest)

    # Validate
    warnings = gt.validate()
    if warnings:
        print("\n⚠ Validation warnings:")
        for w in warnings:
            print(f"  {w}")
        print()

    print(gt.summary())
    print()

    # Sync
    mode_label = "DRY RUN" if dry_run else "SYNC"
    print(f"── {mode_label}: propagating to all files ──")

    changes = sync_all(gt, dry_run=dry_run)

    if changes:
        total = sum(len(c) for c in changes.values())
        for filepath, file_changes in sorted(changes.items()):
            print(f"\n  {filepath}:")
            for ch in file_changes:
                print(f"    {ch}")
        print(f"\n{'Would update' if dry_run else 'Updated'} {total} value(s) in {len(changes)} file(s)")
    else:
        print("  All files already in sync ✓")

    # Write test_count.txt oracle (always, for backward compat)
    if not dry_run:
        count_file = _REPO_ROOT / "scripts" / "test_count.txt"
        count_file.write_text(f"{gt.test_count}\n", encoding="utf-8")

    print(f"\nFound {gt.test_count} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
