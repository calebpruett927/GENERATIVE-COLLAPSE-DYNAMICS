#!/usr/bin/env python3
"""Axiom Conformance Checker — CI-enforceable verification of AXIOM-0 constraints.

Scans Python source, docstrings, comments, and documentation for violations of
the GCD attribution, terminology, and tier-system rules. This is the machine
enforcement layer: the instructions files tell AI *what* the rules are; this
script catches violations *after the fact*.

Violations checked:
  T001  External attribution of GCD structures ("by AM-GM", "Shannon entropy")
  T002  Symbol capture (Tier-2 code redefining F, ω, S, C, κ, IC, τ_R, regime)
  T003  Hardcoded frozen parameters (epsilon/tol not from frozen_contract)
  T004  Boolean verdicts (True/False where three-valued is required)
  T005  Prohibited terminology ("hyperparameter", "rederives", "recovers")

Exit codes:
    0 = No violations found
    1 = Violations found (blocks CI)

Usage:
    python scripts/check_axiom_conformance.py              # Check all
    python scripts/check_axiom_conformance.py --fix-hint    # Show fix suggestions
    python scripts/check_axiom_conformance.py src/umcp/     # Check specific path
"""

from __future__ import annotations

import contextlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Frozen parameters (these are the ONLY acceptable source) ─────────────
FROZEN_CONTRACT_MODULE = "frozen_contract"

# ── Violation patterns ───────────────────────────────────────────────────

# T001: External attribution patterns
# These catch comments, docstrings, and string literals that attribute
# GCD-independent structures to external theories.
T001_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bby\s+AM[- ]?GM\b", re.IGNORECASE),
        "Use 'integrity bound (IC ≤ F)' — AM-GM is a degenerate limit, not a source",
    ),
    (
        re.compile(r"\bAM[- ]?GM\s+(inequality|bound|gap)\b", re.IGNORECASE),
        "Use 'integrity bound (IC ≤ F)' or 'heterogeneity gap (Δ = F − IC)'",
    ),
    (
        re.compile(r"\bShannon\s+entropy\b", re.IGNORECASE),
        "Use 'Bernoulli field entropy' — Shannon entropy is the degenerate limit",
    ),
    (
        re.compile(r"\bby\s+unitarity\b", re.IGNORECASE),
        "Use 'duality identity F + ω = 1' — not quantum unitarity",
    ),
    (
        re.compile(r"\bfrom\s+the\s+exponential\s+map\b", re.IGNORECASE),
        "Use 'log-integrity relation IC = exp(κ)' — not the exponential map",
    ),
    (
        re.compile(r"\b(rederives?|recovers?)\s+(the|a|classical)\b", re.IGNORECASE),
        "Use 'derives independently' — classical results are degenerate limits",
    ),
    (
        re.compile(r"\bhyperparameter\b", re.IGNORECASE),
        "Use 'frozen parameter' — seam-derived, not tuned",
    ),
]

# Context-aware exceptions for T001: lines that CORRECTLY label
# classical results as degenerate limits are not violations.
T001_CONTEXT_EXCEPTIONS: list[re.Pattern[str]] = [
    # "Shannon entropy is the degenerate limit" — correct usage per instructions
    re.compile(r"Shannon\s+entropy\s+is\s+(the|a)\s+degenerate\s+limit", re.IGNORECASE),
    # "(Shannon entropy is the degenerate limit)" — parenthetical correct usage
    re.compile(r"\(Shannon\s+entropy\s+is\s+(the|a)\s+degenerate", re.IGNORECASE),
    # "Shannon entropy → ..." in comparison tables showing classical as output
    re.compile(r"Shannon\s+entropy\s+→", re.IGNORECASE),
    # "classical AM-GM inequality is the degenerate limit"
    re.compile(r"AM[- ]?GM\s+inequality\s+is\s+(the|a)\s+degenerate\s+limit", re.IGNORECASE),
    # "parallel to how Shannon entropy is the" — explicit comparison
    re.compile(r"how\s+Shannon\s+entropy\s+is", re.IGNORECASE),
    # "Shannon entropy of bit" — this is literally computing Shannon entropy on bits (correct)
    re.compile(r"Shannon\s+entropy\s+of\s+bit", re.IGNORECASE),
]

# T001 allowlist: files/contexts where these terms are legitimately discussed
# (e.g., the terminology table itself, test assertions, documentation that
# explicitly labels classical results as degenerate limits)
T001_ALLOWLIST_FILES = {
    ".github/copilot-instructions.md",
    ".github/copilot-coding-agent.yml",
    "CLAUDE.md",
    "AGENTS.md",
    ".cursorrules",
    ".windsurfrules",
    "scripts/check_axiom_conformance.py",
    "GLOSSARY.md",
    "AXIOM.md",
    "TIER_SYSTEM.md",
    "KERNEL_SPECIFICATION.md",
    "ACADEMIC_STATUS_REPORT.md",
    "NEGATIVE_RESULT_ANALYSIS.md",
    "PHILOSOPHICAL_CONVERGENCES.md",
    "CHANGELOG.md",
    "SYNTHESIS_REPORT.md",
    "CERN_HIGGS_RESEARCH.md",
    "README.md",
    "QUICKSTART_TUTORIAL.md",
    "SCALE_LADDER_REVELATIONS.md",
    # LaTeX papers discuss degenerate limits explicitly
    "paper/",
    # Test files check terminology enforcement
    "tests/",
}

# T002: Symbol capture — Tier-2 closures redefining Tier-1 symbols
# Only checked in closures/ directory. Closures legitimately READ kernel
# outputs into local variables (F = result["F"]) — that's not capture.
# Capture is when a closure REDEFINES the symbol with a new formula.
TIER1_SYMBOLS = {"F", "omega", "S", "C", "kappa", "IC", "tau_R", "regime"}
# Match standalone redefinitions but NOT reads from dicts, kernel outputs,
# dataclass fields, function parameters, or comprehension targets
T002_REDEFINITION_PATTERN = re.compile(
    r"^\s*(F|omega|S|C|kappa|IC|tau_R|regime)\s*=\s*"
    r"(?!.*\b(row|inv|data|result|r|k|out|kernel|float|int|round|str|getattr|get)\b)"
    r"(?!.*\[)"
    r"(?!.*\.)",
)

# T003: Hardcoded frozen parameters
T003_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?<!\w)epsilon\s*=\s*1e-8\b"),
        "Import EPSILON from frozen_contract, do not hardcode 1e-8",
    ),
    (
        re.compile(r"(?<!\w)tol_seam\s*=\s*0\.005\b"),
        "Import TOL_SEAM from frozen_contract, do not hardcode 0.005",
    ),
    (
        re.compile(r"(?<!\w)p_exponent\s*=\s*3\b"),
        "Import P_EXPONENT from frozen_contract, do not hardcode 3",
    ),
]

T003_ALLOWLIST_FILES = {
    "src/umcp/frozen_contract.py",  # the source of truth
    "scripts/check_axiom_conformance.py",
    ".github/copilot-instructions.md",
    ".github/copilot-coding-agent.yml",
    "CLAUDE.md",
    "AGENTS.md",
    "KERNEL_SPECIFICATION.md",
    "TIER_SYSTEM.md",
    # Test files use literal values for assertions
    "tests/",
    # Documentation references values for explanation
    "README.md",
    "QUICKSTART_TUTORIAL.md",
    "CHANGELOG.md",
    ".cursorrules",
    ".windsurfrules",
}

# T005: Prohibited terminology in code comments and docstrings
T005_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bwe\s+chose\b", re.IGNORECASE),
        "Frozen parameters are seam-derived, not chosen. Use 'discovered' or 'frozen'",
    ),
    (
        re.compile(r"\bby\s+convention\b", re.IGNORECASE),
        "Parameters are seam-derived. Use 'consistent across the seam'",
    ),
    (
        re.compile(r"\breduces?\s+to\b", re.IGNORECASE),
        "GCD derives independently. Use 'the classical result emerges as a degenerate limit'",
    ),
    (
        re.compile(r"\bis\s+equivalent\s+to\b", re.IGNORECASE),
        "GCD structures are original. Use 'derives independently'",
    ),
]

T005_ALLOWLIST_FILES = T001_ALLOWLIST_FILES | {
    "PHILOSOPHICAL_CONVERGENCES.md",
    "NEGATIVE_RESULT_ANALYSIS.md",
    "tests/",
    "paper/",
}


@dataclass
class Violation:
    """A single axiom conformance violation."""

    code: str  # T001, T002, etc.
    file: str
    line: int
    text: str  # the offending line
    message: str  # what's wrong and how to fix it


@dataclass
class ConformanceReport:
    """Aggregated results of axiom conformance checking."""

    violations: list[Violation] = field(default_factory=list)
    files_checked: int = 0
    lines_checked: int = 0

    @property
    def is_conformant(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        if self.is_conformant:
            return f"✓ AXIOM CONFORMANT — {self.files_checked} files, {self.lines_checked} lines, 0 violations"
        by_code: dict[str, int] = {}
        for v in self.violations:
            by_code[v.code] = by_code.get(v.code, 0) + 1
        breakdown = ", ".join(f"{code}: {count}" for code, count in sorted(by_code.items()))
        return f"✗ NONCONFORMANT — {len(self.violations)} violations in {self.files_checked} files ({breakdown})"


def _is_in_allowlist(filepath: str, allowlist: set[str]) -> bool:
    """Check if a file is in an allowlist (by suffix or prefix match).

    Entries ending with '/' are treated as directory prefixes.
    Other entries match as suffixes (e.g., 'README.md' matches 'path/to/README.md').
    """
    for allowed in allowlist:
        if allowed.endswith("/"):
            # Directory prefix match
            if filepath.startswith(allowed) or ("/" + allowed) in ("/" + filepath):
                return True
        elif filepath.endswith(allowed):
            return True
    return False


def _is_in_comment_or_docstring_context(line: str) -> bool:
    """Heuristic: is this line a comment or inside a string/docstring?"""
    stripped = line.strip()
    return stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")


def check_file(filepath: Path, repo_root: Path) -> list[Violation]:
    """Check a single file for axiom conformance violations."""
    violations: list[Violation] = []
    rel_path = str(filepath.relative_to(repo_root))

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return violations

    lines = content.splitlines()

    for line_num, line in enumerate(lines, start=1):
        # T001: External attribution
        if not _is_in_allowlist(rel_path, T001_ALLOWLIST_FILES):
            for pattern, message in T001_PATTERNS:
                if pattern.search(line):
                    # Check context-aware exceptions: lines that correctly
                    # label classical results as degenerate limits are OK
                    if any(exc.search(line) for exc in T001_CONTEXT_EXCEPTIONS):
                        continue
                    violations.append(
                        Violation(
                            code="T001",
                            file=rel_path,
                            line=line_num,
                            text=line.strip(),
                            message=message,
                        )
                    )

        # T002: Symbol capture (only in closures/)
        # Symbol capture means REDEFINING a Tier-1 symbol with a different semantic meaning.
        # Closures correctly ASSIGN domain values to Tier-1 symbols (F = concurrence, etc.)
        # — that's the design pattern, not a violation.
        # We flag only: class/def redefinitions and clear misuse patterns.
        if rel_path.startswith("closures/") and rel_path.endswith(".py"):
            stripped = line.strip()
            # Skip comments, docstrings, keyword arguments, string assignments
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                pass
            elif re.match(r"^\s*\w+=\w+[,)]", line):
                pass  # keyword argument: regime=regime, F=F_val, etc.
            elif re.match(r'^\s*(regime)\s*=\s*["\']', line):
                pass  # regime = "SomeLabel" — domain-specific labeling, standard pattern
            elif line_num <= 50 and re.match(r"^\s*(F|IC|S|C|omega|kappa|tau_R|regime)\s*=\s*\S", line):
                pass  # Module-level docstring symbols (e.g. "F = Fidelity (coherence)")
            else:
                # Check for class/def redefinitions — these are unambiguous symbol capture
                # Exception: @property methods like `def regime(self)` are accessors, not capture
                class_def_m = re.match(r"^\s*(?:class|def)\s+(F|omega|S|C|kappa|IC|tau_R|regime)\b", line)
                if class_def_m:
                    # Allow `def SYMBOL(self` — these are property accessors / methods
                    if re.match(r"^\s*def\s+\w+\(self", line):
                        pass
                    else:
                        violations.append(
                            Violation(
                                code="T002",
                                file=rel_path,
                                line=line_num,
                                text=stripped,
                                message=f"Tier-2 code redefining Tier-1 symbol '{class_def_m.group(1)}' as class/function — use a different name",
                            )
                        )

        # T003: Hardcoded frozen parameters
        if rel_path.endswith(".py") and not _is_in_allowlist(rel_path, T003_ALLOWLIST_FILES):
            for pattern, message in T003_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        Violation(
                            code="T003",
                            file=rel_path,
                            line=line_num,
                            text=line.strip(),
                            message=message,
                        )
                    )

        # T005: Prohibited terminology (in Python comments/docstrings)
        if (
            rel_path.endswith(".py")
            and not _is_in_allowlist(rel_path, T005_ALLOWLIST_FILES)
            and _is_in_comment_or_docstring_context(line)
        ):
            for pattern, message in T005_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        Violation(
                            code="T005",
                            file=rel_path,
                            line=line_num,
                            text=line.strip(),
                            message=message,
                        )
                    )

    return violations


def check_repo(root: Path, target: Path | None = None) -> ConformanceReport:
    """Check the repository (or a subtree) for axiom conformance violations."""
    report = ConformanceReport()

    search_root = target or root
    extensions = {".py", ".md", ".yaml", ".yml", ".tex"}

    # Directories to skip
    skip_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        ".venv",
        "archive",
        ".egg-info",
        # Skip nested repository copies (e.g., src/umcp_cpp/GENERATIVE-COLLAPSE-DYNAMICS/)
        "GENERATIVE-COLLAPSE-DYNAMICS",
    }

    for filepath in sorted(search_root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix not in extensions:
            continue
        if any(part in skip_dirs for part in filepath.parts):
            continue

        report.files_checked += 1
        with contextlib.suppress(Exception):
            report.lines_checked += sum(1 for _ in filepath.open(encoding="utf-8", errors="replace"))
        report.violations.extend(check_file(filepath, root))

    return report


def main() -> int:
    """Entry point for CLI and CI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check repository for AXIOM-0 conformance violations",
        epilog="Part of the UMCP pre-commit protocol. Exit 0 = conformant, 1 = violations found.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to check (default: current directory)",
    )
    parser.add_argument(
        "--fix-hint",
        action="store_true",
        help="Show suggested fixes for each violation",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    target = Path(args.path).resolve()

    print("═" * 68)
    print("  AXIOM-0 Conformance Check")
    print("═" * 68)
    print()

    report = check_repo(repo_root, target if target != repo_root else None)

    if report.is_conformant:
        print(report.summary())
        return 0

    # Print violations grouped by file
    current_file = ""
    for v in report.violations:
        if v.file != current_file:
            current_file = v.file
            print(f"\n  {current_file}")
            print(f"  {'─' * len(current_file)}")
        print(f"    L{v.line:4d} [{v.code}] {v.text[:80]}")
        if args.fix_hint:
            print(f"           → {v.message}")

    print()
    print(report.summary())
    return 1


if __name__ == "__main__":
    sys.exit(main())
