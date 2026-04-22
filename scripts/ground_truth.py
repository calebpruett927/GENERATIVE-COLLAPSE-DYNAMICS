#!/usr/bin/env python3
"""Ground Truth — Single Source of Truth for all repository-wide metrics.

Every number that appears in documentation, web templates, badges, agent
instructions, or metadata MUST be derived from this file.  When a metric
changes, update it HERE and run ``python scripts/sync_ground_truth.py``
(or let the pre-commit protocol do it automatically).

Architecture:
    COMPUTED metrics    — auto-derived from the repo (domains, closures, tests, test files)
    MANUAL metrics      — set by hand when content changes (theorems, version, identities, lemmas)
    FROZEN metrics      — immutable structural constants (identity count, lemma count)

    Computed metrics are refreshed every pre-commit run.
    Manual metrics are validated against plausible bounds.
    Frozen metrics are checked for accidental modification.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Manual / Frozen Metrics ──────────────────────────────────────
# Update these when content changes.  The sync script propagates them.

VERSION = "2.3.1"
"""Package version — canonical source is pyproject.toml, but this must match."""

THEOREM_COUNT = 746
"""Proven theorems across all domain closures.  Update after adding theorems."""

IDENTITY_COUNT = 44
"""Structural identities derived from Axiom-0.  Frozen — changes are rare."""

LEMMA_COUNT = 47
"""Proven lemmas (OPT-* tagged).  Frozen — changes are rare."""

CASEPACK_COUNT = 26
"""Self-contained validation casepacks.  Update after adding casepacks."""

CANON_FILE_COUNT = 22
"""Canonical anchor files in canon/.  Update after adding canon files."""

CONTRACT_COUNT = 23
"""Versioned mathematical contracts.  Update after adding contracts."""

SCHEMA_COUNT = 17
"""JSON Schema files.  Update after adding schemas."""

C_LINES = 1900
"""Approximate lines in the C99 orchestration core."""

C_ASSERTIONS = 326
"""C kernel + orchestration test assertions (166 + 160)."""

CPP_ASSERTIONS = 434
"""C++ Catch2 test assertions."""


# ── Computed Metrics ─────────────────────────────────────────────
# These are derived from the repo at runtime.  Do NOT hardcode.


def compute_domain_count(root: Path | None = None) -> int:
    """Count closure domain directories (excludes __pycache__, etc.)."""
    closures = (root or _REPO_ROOT) / "closures"
    return len([d for d in closures.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])


def compute_closure_count(root: Path | None = None) -> int:
    """Count closure .py modules (excludes __init__.py)."""
    closures = (root or _REPO_ROOT) / "closures"
    return len([f for f in closures.rglob("*.py") if f.name != "__init__.py" and "__pycache__" not in str(f)])


def compute_test_count(root: Path | None = None) -> int:
    """Run pytest --collect-only to get actual test count."""
    repo = root or _REPO_ROOT
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo),
        )
        # Try standard patterns first
        for pattern in [r"(\d+) tests? collected", r"(\d+) items?"]:
            m = re.search(pattern, result.stdout)
            if m:
                return int(m.group(1))
        # Fallback: sum all "tests/file.py: N" counts (parametrization-aware)
        total = sum(1 for ln in result.stdout.splitlines() if "::" in ln)
        if total > 0:
            return total
        # Alternative fallback: parse "tests/file.py: N" format and sum
        count_pattern = re.compile(r"tests/[^:]+: (\d+)")
        counts = [int(m.group(1)) for m in count_pattern.finditer(result.stdout)]
        if counts:
            return sum(counts)
        return 0
    except Exception as e:
        print(f"Warning: pytest collection failed: {e}", file=sys.stderr)
        return 0


def compute_test_file_count(root: Path | None = None) -> int:
    """Count test files: top-level + closures/ subdirectory."""
    tests = (root or _REPO_ROOT) / "tests"
    top = len(list(tests.glob("test_*.py")))
    closure = len(list((tests / "closures").glob("test_*.py"))) if (tests / "closures").exists() else 0
    return top + closure


def compute_casepack_count(root: Path | None = None) -> int:
    """Count casepack directories."""
    casepacks = (root or _REPO_ROOT) / "casepacks"
    return (
        len([d for d in casepacks.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])
        if casepacks.exists()
        else 0
    )


def compute_canon_count(root: Path | None = None) -> int:
    """Count canonical anchor YAML files."""
    canon = (root or _REPO_ROOT) / "canon"
    return len(list(canon.glob("*.yaml"))) if canon.exists() else 0


# ── GroundTruth dataclass ────────────────────────────────────────


@dataclass(frozen=True)
class GroundTruth:
    """All repository metrics in one frozen object."""

    # Identity
    version: str

    # Computed (refreshed every run)
    test_count: int
    domain_count: int
    closure_count: int
    test_file_count: int

    # Manual (set when content changes)
    theorem_count: int
    casepack_count: int
    canon_count: int
    contract_count: int
    schema_count: int

    # Frozen (structural constants — very rarely change)
    identity_count: int
    lemma_count: int

    # C/C++ layer
    c_lines: int
    c_assertions: int
    cpp_assertions: int

    # ── Derived / formatted helpers ──────────────────────────────

    @property
    def test_count_comma(self) -> str:
        return f"{self.test_count:,}"

    @property
    def test_count_url(self) -> str:
        return self.test_count_comma.replace(",", "%2C")

    @property
    def total_c_cpp_assertions(self) -> int:
        return self.c_assertions + self.cpp_assertions

    @classmethod
    def compute(cls, root: Path | None = None, skip_pytest: bool = False) -> GroundTruth:
        """Build a GroundTruth by computing live metrics from the repo.

        Parameters
        ----------
        root : Path, optional
            Repo root.  Defaults to the parent of ``scripts/``.
        skip_pytest : bool
            If True, read test count from ``scripts/test_count.txt``
            instead of running pytest (much faster for non-test-count updates).
        """
        repo = root or _REPO_ROOT

        if skip_pytest:
            count_file = repo / "scripts" / "test_count.txt"
            if count_file.exists():
                txt = count_file.read_text().strip()
                test_count = int(txt) if txt.isdigit() else 0
            else:
                test_count = 0
        else:
            test_count = compute_test_count(repo)

        return cls(
            version=VERSION,
            test_count=test_count,
            domain_count=compute_domain_count(repo),
            closure_count=compute_closure_count(repo),
            test_file_count=compute_test_file_count(repo),
            theorem_count=THEOREM_COUNT,
            casepack_count=compute_casepack_count(repo),
            canon_count=compute_canon_count(repo),
            contract_count=CONTRACT_COUNT,
            schema_count=SCHEMA_COUNT,
            identity_count=IDENTITY_COUNT,
            lemma_count=LEMMA_COUNT,
            c_lines=C_LINES,
            c_assertions=C_ASSERTIONS,
            cpp_assertions=CPP_ASSERTIONS,
        )

    def validate(self) -> list[str]:
        """Return a list of warnings if metrics are outside plausible bounds."""
        warnings: list[str] = []
        if self.test_count < 1000 or self.test_count > 50000:
            warnings.append(f"test_count={self.test_count} outside [1000, 50000]")
        if self.domain_count < 10 or self.domain_count > 100:
            warnings.append(f"domain_count={self.domain_count} outside [10, 100]")
        if self.closure_count < 50 or self.closure_count > 1000:
            warnings.append(f"closure_count={self.closure_count} outside [50, 1000]")
        if self.test_file_count < 50 or self.test_file_count > 500:
            warnings.append(f"test_file_count={self.test_file_count} outside [50, 500]")
        if self.theorem_count < 100 or self.theorem_count > 5000:
            warnings.append(f"theorem_count={self.theorem_count} outside [100, 5000]")
        if self.identity_count != 44:
            warnings.append(f"identity_count={self.identity_count} != 44 (frozen)")
        if self.lemma_count != 47:
            warnings.append(f"lemma_count={self.lemma_count} != 47 (frozen)")
        return warnings

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"  version:        {self.version}\n"
            f"  tests:          {self.test_count_comma}\n"
            f"  domains:        {self.domain_count}\n"
            f"  closures:       {self.closure_count}\n"
            f"  test files:     {self.test_file_count}\n"
            f"  theorems:       {self.theorem_count}\n"
            f"  identities:     {self.identity_count}\n"
            f"  lemmas:         {self.lemma_count}\n"
            f"  casepacks:      {self.casepack_count}\n"
            f"  canon files:    {self.canon_count}\n"
            f"  contracts:      {self.contract_count}\n"
            f"  schemas:        {self.schema_count}\n"
            f"  C assertions:   {self.c_assertions}\n"
            f"  C++ assertions: {self.cpp_assertions}"
        )


# ── CLI: print current ground truth ─────────────────────────────

if __name__ == "__main__":
    gt = GroundTruth.compute(skip_pytest="--skip-pytest" in sys.argv)
    print("Ground Truth:")
    print(gt.summary())
    warnings = gt.validate()
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
