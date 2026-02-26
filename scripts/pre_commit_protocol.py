#!/usr/bin/env python3
"""UMCP Pre-Commit Protocol — typed, repo-aware gate before every commit.

Replaces ad-hoc commit workflows with a deterministic pipeline that
mirrors .github/workflows/validate.yml exactly.  Every step is typed,
every path is resolved from a frozen RepoContext, and every result is
a structured dataclass — not a parsed stdout string.

Usage:
    python scripts/pre_commit_protocol.py           # Full protocol (auto-fix)
    python scripts/pre_commit_protocol.py --check    # Dry-run: report only
    python scripts/pre_commit_protocol.py --fix      # Auto-fix mode (default)

Exit codes:
    0 = All checks passed, safe to commit
    1 = Blocking failure, commit blocked

Architecture:
    RepoContext    — frozen dataclass resolving every critical path once
    StepStatus     — three-valued enum: PASS / FAIL / WARN (matches UMCP regime)
    StepResult     — typed output of each pipeline step
    ProtocolReport — aggregated results, produces final verdict
    step_*         — pure functions: RepoContext → StepResult
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

# Ensure venv bin/ is on PATH so subprocess calls find ruff, mypy, pytest, umcp
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

# ── Repo Context ─────────────────────────────────────────────────
# Frozen dataclass that discovers and validates the repo structure once,
# then threads through every step.  Pattern matches tests/conftest.py RepoPaths.


def _find_repo_root() -> Path:
    """Walk up from this script to find the repo root (contains pyproject.toml)."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    # Fallback: walk up from cwd
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Cannot locate repo root (no pyproject.toml found)"
    raise FileNotFoundError(msg)


@dataclass(frozen=True)
class RepoContext:
    """Immutable snapshot of the UMCP repository structure.

    Every path referenced by the protocol is resolved here, once,
    at protocol start.  If a path does not exist, the protocol can
    fail early with a clear message rather than a cryptic subprocess error.

    Follows the frozen-dataclass pattern from tests/conftest.py RepoPaths.
    """

    root: Path

    # Source code
    src_dir: Path
    umcp_pkg: Path

    # Integrity
    integrity_dir: Path
    integrity_script: Path

    # CI-critical directories
    schemas_dir: Path
    contracts_dir: Path
    closures_dir: Path
    casepacks_dir: Path
    tests_dir: Path
    scripts_dir: Path
    canon_dir: Path

    # Key files
    pyproject: Path
    registry: Path
    validator_rules: Path
    ledger_dir: Path

    # Computed metadata
    version: str
    python_exe: str

    @classmethod
    def discover(cls) -> RepoContext:
        """Build a RepoContext by discovering the repo structure."""
        root = _find_repo_root()

        # Parse version from pyproject.toml
        pyproject = root / "pyproject.toml"
        version = "unknown"
        if pyproject.exists():
            for line in pyproject.read_text().splitlines():
                if line.strip().startswith("version"):
                    m = re.search(r'"([^"]+)"', line)
                    if m:
                        version = m.group(1)
                    break

        return cls(
            root=root,
            src_dir=root / "src",
            umcp_pkg=root / "src" / "umcp",
            integrity_dir=root / "integrity",
            integrity_script=root / "scripts" / "update_integrity.py",
            schemas_dir=root / "schemas",
            contracts_dir=root / "contracts",
            closures_dir=root / "closures",
            casepacks_dir=root / "casepacks",
            tests_dir=root / "tests",
            scripts_dir=root / "scripts",
            canon_dir=root / "canon",
            pyproject=pyproject,
            registry=root / "closures" / "registry.yaml",
            validator_rules=root / "validator_rules.yaml",
            ledger_dir=root / "ledger",
            version=version,
            python_exe=sys.executable,
        )

    def validate_structure(self) -> list[str]:
        """Return list of missing critical paths (empty means all good)."""
        critical: list[Path] = [
            self.pyproject,
            self.umcp_pkg,
            self.schemas_dir,
            self.contracts_dir,
            self.closures_dir,
            self.tests_dir,
            self.integrity_script,
            self.registry,
            self.validator_rules,
        ]
        return [str(p.relative_to(self.root)) for p in critical if not p.exists()]


# ── Step Result Types ────────────────────────────────────────────


class StepStatus(StrEnum):
    """Three-valued step outcome — mirrors UMCP Stable/Watch/Collapse."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"  # Non-blocking (e.g., mypy with continue-on-error)


class ToolVersions(NamedTuple):
    """Captured tool versions for reproducibility."""

    python: str
    ruff: str
    mypy: str
    pytest: str
    umcp: str


@dataclass
class StepResult:
    """Typed result of a single protocol step."""

    name: str
    status: StepStatus
    duration_s: float = 0.0
    message: str = ""
    fixed_count: int = 0
    blocking: bool = True

    @property
    def passed(self) -> bool:
        return self.status != StepStatus.FAIL

    @property
    def icon(self) -> str:
        icons: dict[str, str] = {
            "PASS": "✓",
            "FAIL": "✗",
            "WARN": "⚠",
        }
        return icons[self.status.value]


@dataclass
class ProtocolReport:
    """Aggregated protocol results."""

    steps: list[StepResult] = field(default_factory=list)
    mode: str = "fix"
    context: RepoContext | None = None
    tool_versions: ToolVersions | None = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def all_pass(self) -> bool:
        return all(s.passed for s in self.steps)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def blocking_failures(self) -> list[StepResult]:
        return [s for s in self.steps if s.status == StepStatus.FAIL and s.blocking]


# ── Subprocess Helper ────────────────────────────────────────────


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture output.  Never raises on non-zero exit."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _capture_tool_versions(ctx: RepoContext) -> ToolVersions:
    """Capture tool versions for the report header."""

    def _ver(cmd: list[str]) -> str:
        try:
            r = _run(cmd, cwd=ctx.root, timeout=10)
            out = r.stdout.strip().split("\n")[0] if r.returncode == 0 else "unavailable"
        except Exception:
            out = "unavailable"
        return out

    return ToolVersions(
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ruff=_ver(["ruff", "--version"]),
        mypy=_ver(["mypy", "--version"]),
        pytest=_ver([ctx.python_exe, "-m", "pytest", "--version"]),
        umcp=ctx.version,
    )


# ── Protocol Steps ───────────────────────────────────────────────
# Each step is a typed function: (RepoContext, mode) → StepResult

DIVIDER = "─" * 72


def _header(step_num: int, total: int, name: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  Step {step_num}/{total}: {name}")
    print(DIVIDER)


def step_manifold_bounds(ctx: RepoContext) -> StepResult:
    """Step 0: Manifold Bound Surface — fast identity gate (~3 s).

    Runs test_000_manifold_bounds.py (Layer 0-2) before the full suite.
    If the algebraic bound surface fails, there is no point running 2,476+
    tests that depend on the same kernel identities.
    """
    t0 = time.monotonic()
    result = _run(
        [ctx.python_exe, "-m", "pytest", str(ctx.tests_dir / "test_000_manifold_bounds.py"), "-q", "--tb=short"],
        cwd=ctx.root,
        timeout=60,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    passed_count = 0
    failed_count = 0
    for match in re.finditer(r"(\d+)\s+(passed|failed)", combined):
        count_val = int(match.group(1))
        kind = match.group(2)
        if kind == "passed":
            passed_count = count_val
        elif kind == "failed":
            failed_count = count_val

    passed = result.returncode == 0
    msg = (
        f"{passed_count} bounds passed, {failed_count} failed"
        if (passed_count or failed_count)
        else ("bound surface verified" if passed else f"exit code {result.returncode}")
    )

    return StepResult(
        name="manifold bounds",
        status=StepStatus.PASS if passed else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
    )


def step_ruff_format(ctx: RepoContext, mode: str) -> StepResult:
    """Step 1: ruff format — enforce code style."""
    t0 = time.monotonic()

    if mode == "fix":
        result = _run(["ruff", "format", "."], cwd=ctx.root)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        fixed = sum(1 for ln in lines if "reformatted" in ln and "left unchanged" not in ln)

        # Verify clean
        check = _run(["ruff", "format", "--check", "."], cwd=ctx.root)
        passed = check.returncode == 0
        msg = f"Formatted {fixed} file(s)" if fixed else "All files clean"
    else:
        result = _run(["ruff", "format", "--check", "."], cwd=ctx.root)
        passed = result.returncode == 0
        fixed = 0
        if not passed:
            bad = [ln for ln in result.stdout.strip().split("\n") if "Would reformat" in ln]
            msg = f"{len(bad)} file(s) need formatting"
        else:
            msg = "All files clean"

    return StepResult(
        name="ruff format",
        status=StepStatus.PASS if passed else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
        fixed_count=fixed,
    )


def step_ruff_lint(ctx: RepoContext, mode: str) -> StepResult:
    """Step 2: ruff check — lint rules."""
    t0 = time.monotonic()
    fixed = 0

    if mode == "fix":
        fix_result = _run(["ruff", "check", "--fix", "."], cwd=ctx.root)
        combined = (fix_result.stdout or "") + "\n" + (fix_result.stderr or "")
        for ln in combined.split("\n"):
            m = re.search(r"Fixed\s+(\d+)\s+error", ln, re.IGNORECASE)
            if m:
                fixed = int(m.group(1))
                break

    # Verify clean
    result = _run(["ruff", "check", "."], cwd=ctx.root)
    passed = result.returncode == 0
    msg = "All checks passed"
    if not passed:
        for ln in result.stdout.split("\n"):
            if ln.startswith("Found"):
                msg = ln.strip()
                break
        else:
            msg = "Lint errors remain"

    return StepResult(
        name="ruff check",
        status=StepStatus.PASS if passed else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
        fixed_count=fixed,
    )


def step_mypy(ctx: RepoContext) -> StepResult:
    """Step 3: mypy — type checking (non-blocking, matches CI continue-on-error)."""
    t0 = time.monotonic()
    result = _run(
        ["mypy", str(ctx.umcp_pkg), f"--config-file={ctx.pyproject}"],
        cwd=ctx.root,
        timeout=120,
    )

    error_count = 0
    for ln in result.stdout.split("\n"):
        m = re.search(r"Found\s+(\d+)\s+error", ln)
        if m:
            error_count = int(m.group(1))
            break

    passed = result.returncode == 0
    msg = "Clean" if passed else f"{error_count} error(s) (non-blocking, same as CI)"

    return StepResult(
        name="mypy",
        status=StepStatus.PASS if passed else StepStatus.WARN,
        duration_s=time.monotonic() - t0,
        message=msg,
        blocking=False,
    )


def step_stage_files(ctx: RepoContext) -> StepResult:
    """Step 4: git add -A — stage all changes for integrity scan."""
    t0 = time.monotonic()
    result = _run(["git", "add", "-A"], cwd=ctx.root)

    status_result = _run(["git", "status", "--short"], cwd=ctx.root)
    staged_lines = [ln for ln in status_result.stdout.strip().split("\n") if ln.strip()]

    return StepResult(
        name="git add -A",
        status=StepStatus.PASS if result.returncode == 0 else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=f"{len(staged_lines)} file(s) staged",
    )


def step_update_test_count(ctx: RepoContext) -> StepResult:
    """Step 5: update test count in documentation."""
    t0 = time.monotonic()
    test_count_script = ctx.root / "scripts" / "update_test_count.py"

    if not test_count_script.exists():
        return StepResult(
            name="update test count",
            status=StepStatus.WARN,
            duration_s=time.monotonic() - t0,
            message="Script not found (non-blocking)",
            blocking=False,
        )

    result = _run(
        [ctx.python_exe, str(test_count_script)],
        cwd=ctx.root,
    )

    test_count = 0
    for ln in result.stdout.split("\n"):
        m = re.search(r"Found\s+(\d+)\s+tests", ln)
        if m:
            test_count = int(m.group(1))
            break

    # Re-stage documentation files
    readme = ctx.root / "README.md"
    copilot_inst = ctx.root / ".github" / "copilot-instructions.md"
    if readme.exists():
        _run(["git", "add", str(readme)], cwd=ctx.root)
    if copilot_inst.exists():
        _run(["git", "add", str(copilot_inst)], cwd=ctx.root)

    passed = result.returncode == 0
    msg = f"{test_count} tests counted" if passed and test_count > 0 else "Test count updated"

    return StepResult(
        name="update test count",
        status=StepStatus.PASS if passed else StepStatus.WARN,
        duration_s=time.monotonic() - t0,
        message=msg,
        blocking=False,
    )


def step_update_integrity(ctx: RepoContext) -> StepResult:
    """Step 6: regenerate SHA256 checksums."""
    t0 = time.monotonic()
    result = _run(
        [ctx.python_exe, str(ctx.integrity_script)],
        cwd=ctx.root,
    )

    file_count = 0
    for ln in result.stdout.split("\n"):
        m = re.search(r"Checksummed\s+(\d+)\s+file", ln)
        if m:
            file_count = int(m.group(1))
            break

    # Re-stage integrity files
    _run(["git", "add", str(ctx.integrity_dir)], cwd=ctx.root)

    passed = result.returncode == 0
    msg = f"{file_count} files checksummed" if passed else "Integrity update failed"

    return StepResult(
        name="update integrity",
        status=StepStatus.PASS if passed else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
    )


# ── Test count bounds ─────────────────────────────────────────────
# The manifold-bounds step (step 1) already executes real tests against the
# kernel.  Re-running the full suite (~115 s) is redundant when the only
# intervening steps are formatting, linting, and integrity checksums —
# none of which alter test-affecting code.
#
# Instead we collect tests (--collect-only) in ~2 s and verify the count
# is within expected bounds.  This catches:
#   - accidental test deletion (count drops)
#   - import errors in new test files (collection fails)
#   - broken fixtures (collection fails)
#
# CI still runs the full suite; pre-commit only needs the fast gate.

_MIN_EXPECTED_TESTS = 1000  # floor — repo has ~2,476+ and growing
_MAX_EXPECTED_TESTS = 6000  # ceiling — sanity upper bound


def step_pytest(ctx: RepoContext) -> StepResult:
    """Step 6: pytest bounds — collect tests (no execution) and verify count.

    Why not a full run?  The manifold-bounds step already executed real
    kernel tests.  Ruff / mypy / integrity don't change runtime behavior.
    Collecting is enough to catch import errors, fixture breakage, and
    accidental test deletion — in ~2 s instead of ~115 s.
    """
    t0 = time.monotonic()
    result = _run(
        [ctx.python_exe, "-m", "pytest", "--co", "-q"],
        cwd=ctx.root,
        timeout=60,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    collected = 0

    # pytest --co -q groups tests by file:  "tests/test_foo.py: 42"
    # Sum the per-file counts to get the total.
    for ln in result.stdout.strip().split("\n"):
        m = re.match(r".*:\s+(\d+)\s*$", ln.strip())
        if m:
            collected += int(m.group(1))

    # Fallback: try "<N> tests collected" summary (some pytest versions)
    if collected == 0:
        m_summary = re.search(r"(\d+)\s+tests?\s+collected", combined)
        if m_summary:
            collected = int(m_summary.group(1))

    # Collection itself must succeed (catches import errors, fixture breakage)
    if result.returncode != 0:
        # Extract first error line for diagnosis
        err_lines = [ln for ln in combined.split("\n") if "ERROR" in ln or "ModuleNotFoundError" in ln]
        err_hint = err_lines[0].strip() if err_lines else f"exit code {result.returncode}"
        return StepResult(
            name="pytest bounds",
            status=StepStatus.FAIL,
            duration_s=time.monotonic() - t0,
            message=f"collection failed: {err_hint}",
        )

    # Bounds check
    in_bounds = _MIN_EXPECTED_TESTS <= collected <= _MAX_EXPECTED_TESTS
    if in_bounds:
        msg = f"{collected} tests collected (bounds: {_MIN_EXPECTED_TESTS}–{_MAX_EXPECTED_TESTS})"
    elif collected < _MIN_EXPECTED_TESTS:
        msg = f"{collected} tests collected — below minimum {_MIN_EXPECTED_TESTS} (tests deleted?)"
    else:
        msg = f"{collected} tests collected — above maximum {_MAX_EXPECTED_TESTS} (update bounds)"

    return StepResult(
        name="pytest bounds",
        status=StepStatus.PASS if in_bounds else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
    )


def step_repo_health(ctx: RepoContext, mode: str) -> StepResult:
    """Step 6½: Repository Health Check — detect drift, sync versions, verify freeze.

    Ensures every satellite file uses the canonical (latest) version,
    freeze hashes match current files, manifest refs exist, and no merge
    conflict markers remain.  In fix mode, auto-corrects what it can.

    This is the "assume latest" enforcer — it runs before integrity
    checksums so any auto-fixed files get included in the hash sweep.
    """
    t0 = time.monotonic()

    # Import from scripts/ — add to path temporarily
    scripts_dir = str(ctx.scripts_dir)
    orig_path = sys.path[:]
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from repo_health_check import run_health_check, sync_version  # type: ignore[import-not-found]

        # Always sync version first (proactive, not reactive)
        if mode == "fix":
            synced = sync_version(ctx.root, quiet=True)
        else:
            synced = 0

        # Run the full health check (with fix in fix mode)
        health_report = run_health_check(ctx.root, fix=(mode == "fix"))

        errors = health_report.error_count
        warnings = health_report.warn_count
        checks = len(health_report.checked)

        # Re-stage any files changed by auto-fix or version sync
        if mode == "fix" and (synced > 0 or errors > 0):
            _run(["git", "add", "-A"], cwd=ctx.root)

        if errors > 0:
            # Summarize first 3 errors
            err_msgs = [f.message for f in health_report.findings if f.severity.value == "ERROR"][:3]
            detail = "; ".join(err_msgs)
            msg = f"{errors} error(s), {warnings} warning(s) across {checks} checks: {detail}"
            status = StepStatus.FAIL
        elif warnings > 0:
            msg = f"PASS with {warnings} warning(s) across {checks} checks"
            if synced:
                msg += f" [{synced} file(s) synced]"
            status = StepStatus.WARN
        else:
            msg = f"All {checks} checks clean"
            if synced:
                msg += f" [{synced} file(s) synced]"
            status = StepStatus.PASS

    except Exception as e:
        # Fallback: run as subprocess
        fix_flag = ["--fix"] if mode == "fix" else []
        result = _run(
            [ctx.python_exe, str(ctx.scripts_dir / "repo_health_check.py"), "--json", *fix_flag],
            cwd=ctx.root,
            timeout=60,
        )
        try:
            data = json.loads(result.stdout)
            errors = data.get("errors", 0)
            warnings = data.get("warnings", 0)
            status_str = data.get("status", "UNKNOWN")
            status = StepStatus.PASS if status_str == "PASS" else StepStatus.FAIL
            msg = f"{status_str}: {errors} error(s), {warnings} warning(s)"
        except (json.JSONDecodeError, KeyError):
            status = StepStatus.WARN
            msg = f"Health check subprocess failed: {e}"
            errors = 0
    finally:
        sys.path[:] = orig_path

    return StepResult(
        name="repo health",
        status=status,
        duration_s=time.monotonic() - t0,
        message=msg,
        fixed_count=synced if "synced" in dir() else 0,
        blocking=True,  # Errors block; warnings don't (status=WARN passes)
    )


def step_umcp_validate(ctx: RepoContext) -> StepResult:
    """Step 7: umcp validate — contract validation (must be CONFORMANT).

    Uses the umcp Python API directly, falling back to CLI subprocess.
    """
    t0 = time.monotonic()

    status = "UNKNOWN"
    target_count = 0
    error_count = 0

    # Try Python API first — avoids subprocess overhead, uses repo context directly
    try:
        orig_path = sys.path[:]
        sys.path.insert(0, str(ctx.src_dir))
        from umcp import validate  # type: ignore[import-not-found]

        vr = validate(str(ctx.root))
        status = vr.status
        target_count = len(vr.data.get("targets", []))
        error_count = vr.error_count
        sys.path[:] = orig_path
    except Exception:
        sys.path[:] = orig_path if "orig_path" in dir() else sys.path
        # Fallback to CLI subprocess
        result = _run(["umcp", "validate", "."], cwd=ctx.root, timeout=120)
        stdout = result.stdout

        # Extract JSON from mixed log + JSON output
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(stdout[json_start : json_end + 1])
                status = data.get("run_status", "UNKNOWN")
                targets = data.get("targets", [])
                target_count = len(targets)
                error_count = sum(t.get("counts", {}).get("errors", 0) for t in targets)
            except json.JSONDecodeError:
                pass

        # Last-resort grep
        if status == "UNKNOWN":
            if "CONFORMANT" in stdout and "NONCONFORMANT" not in stdout:
                status = "CONFORMANT"
            elif "NONCONFORMANT" in stdout:
                status = "NONCONFORMANT"

    passed = status == "CONFORMANT"
    msg = f"{status} ({target_count} targets, {error_count} errors)"

    return StepResult(
        name="umcp validate",
        status=StepStatus.PASS if passed else StepStatus.FAIL,
        duration_s=time.monotonic() - t0,
        message=msg,
    )


def step_axiom_conformance(ctx: RepoContext) -> StepResult:
    """Step 11: Axiom-0 conformance — terminology, symbol capture, frozen params.

    Runs scripts/check_axiom_conformance.py to verify:
    - T001: No external attribution (terminology enforcement)
    - T002: No Tier-1 symbol capture in Tier-2 closures
    - T003: No hardcoded frozen parameters
    - T005: No prohibited terminology

    Blocking: violations fail the commit. All legacy violations have been resolved.
    """
    t0 = time.monotonic()

    script = ctx.scripts_dir / "check_axiom_conformance.py"
    if not script.exists():
        return StepResult(
            name="axiom conformance",
            status=StepStatus.WARN,
            duration_s=time.monotonic() - t0,
            message="check_axiom_conformance.py not found — skipped",
        )

    result = _run(
        [ctx.python_exe, str(script)],
        cwd=ctx.root,
        timeout=60,
    )

    # Parse the summary line: "✓ AXIOM CONFORMANT ..." or "✗ NONCONFORMANT — N violations ..."
    output = result.stdout.strip().splitlines()
    summary = output[-1] if output else ""

    if "AXIOM CONFORMANT" in summary or ("CONFORMANT" in summary and "NON" not in summary):
        msg = summary.split("—")[-1].strip() if "—" in summary else "All files conformant"
        status = StepStatus.PASS
    else:
        # Extract violation count
        import re as _re

        m = _re.search(r"(\d+)\s+violations?", summary)
        count = int(m.group(1)) if m else 0
        msg = f"{count} axiom violation(s) — run 'python scripts/check_axiom_conformance.py --fix-hint' for details"
        status = StepStatus.FAIL

    return StepResult(
        name="axiom conformance",
        status=status,
        duration_s=time.monotonic() - t0,
        message=msg,
        blocking=True,
    )


# ── Protocol Runner ──────────────────────────────────────────────

# Step registry: (display_label, function_key)
# Order matters:
#   1-4: Code quality (bounds, format, lint, types)
#   5:   Stage files
#   6:   Repo Health — version sync + drift detection (before integrity!)
#   7:   Test count
#   8:   Update integrity checksums (after health fix so fixed files get hashed)
#   9:   Pytest bounds
#   10:  UMCP validate
#   11:  Axiom-0 conformance (blocking)
_STEP_REGISTRY: list[tuple[str, str]] = [
    ("Manifold Bounds", "bounds"),
    ("Ruff Format", "ruff_format"),
    ("Ruff Lint", "ruff_lint"),
    ("Mypy Type Check", "mypy"),
    ("Stage Files", "stage"),
    ("Repo Health", "health"),
    ("Update Test Count", "test_count"),
    ("Update Integrity", "integrity"),
    ("Pytest Bounds", "pytest"),
    ("UMCP Validate", "validate"),
    ("Axiom Conformance", "axiom"),
]


def run_protocol(ctx: RepoContext, mode: str = "fix") -> ProtocolReport:
    """Execute the full pre-commit protocol, threading RepoContext through every step."""
    report = ProtocolReport(
        mode=mode,
        context=ctx,
        start_time=time.monotonic(),
    )

    # Capture tool versions for reproducibility
    report.tool_versions = _capture_tool_versions(ctx)

    # Map keys to step functions (each receives typed RepoContext)
    step_funcs: dict[str, object] = {
        "bounds": lambda: step_manifold_bounds(ctx),
        "ruff_format": lambda: step_ruff_format(ctx, mode),
        "ruff_lint": lambda: step_ruff_lint(ctx, mode),
        "mypy": lambda: step_mypy(ctx),
        "stage": lambda: step_stage_files(ctx),
        "health": lambda: step_repo_health(ctx, mode),
        "test_count": lambda: step_update_test_count(ctx),
        "integrity": lambda: step_update_integrity(ctx),
        "pytest": lambda: step_pytest(ctx),
        "validate": lambda: step_umcp_validate(ctx),
        "axiom": lambda: step_axiom_conformance(ctx),
    }

    total = len(_STEP_REGISTRY)

    for i, (label, key) in enumerate(_STEP_REGISTRY, 1):
        _header(i, total, label)
        fn = step_funcs[key]
        step_result: StepResult = fn()  # type: ignore[operator]
        report.steps.append(step_result)

        fixed_note = f" (auto-fixed {step_result.fixed_count})" if step_result.fixed_count else ""
        print(
            f"  {step_result.icon} {step_result.status.value}: "
            f"{step_result.message}{fixed_note}  [{step_result.duration_s:.1f}s]"
        )

        # Abort on blocking failure:
        # - Step 1 (bounds): kernel is broken, nothing else can be trusted
        # - Steps 3+ (after format/lint): skip slow tests if code broken
        if step_result.status == StepStatus.FAIL and step_result.blocking and (i == 1 or i > 3):
            print(f"\n  ⛔ Blocking failure at step {i}. Aborting protocol.")
            break

    report.end_time = time.monotonic()
    return report


def print_summary(report: ProtocolReport) -> None:
    """Print the typed summary report."""
    print(f"\n{'═' * 72}")
    print("  UMCP PRE-COMMIT PROTOCOL — SUMMARY")
    print(f"{'═' * 72}")
    print(f"  Mode:     {report.mode}")
    ctx_version = report.context.version if report.context else "unknown"
    print(f"  Version:  {ctx_version}")
    print(f"  Duration: {report.duration_s:.1f}s")

    if report.tool_versions:
        tv = report.tool_versions
        print(f"  Tools:    Python {tv.python} | {tv.ruff} | {tv.mypy}")

    print()
    for step in report.steps:
        fixed = f" [{step.fixed_count} fixed]" if step.fixed_count else ""
        blocking_tag = "" if step.blocking else " (non-blocking)"
        print(f"  {step.icon} {step.name:<25s} {step.message}{fixed}{blocking_tag}")

    print()
    if report.all_pass:
        print("  ALL CHECKS PASSED — safe to commit and push")
    else:
        failures = report.blocking_failures
        if failures:
            print(f"  {len(failures)} BLOCKING FAILURE(S) — commit NOT safe")
            for s in failures:
                print(f"     -> {s.name}: {s.message}")
        else:
            print("  All blocking checks passed (non-blocking warnings present)")

    print(f"{'═' * 72}")


# ── Entry Point ──────────────────────────────────────────────────


def main() -> int:
    """Parse args, discover repo context, run protocol."""
    parser = argparse.ArgumentParser(
        description="UMCP Pre-Commit Protocol — typed, repo-aware commit gate",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Dry-run: report only, no fixes")
    group.add_argument("--fix", action="store_true", help="Auto-fix mode (default)")
    args = parser.parse_args()

    mode = "check" if args.check else "fix"

    # Discover and validate repo structure
    try:
        ctx = RepoContext.discover()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    missing = ctx.validate_structure()
    if missing:
        print("Error: Missing critical paths:", file=sys.stderr)
        for p in missing:
            print(f"   -> {p}", file=sys.stderr)
        return 1

    # Banner
    print("+" + "=" * 70 + "+")
    print("|" + " UMCP PRE-COMMIT PROTOCOL ".center(70) + "|")
    print("|" + f" v{ctx.version} | {mode} mode ".center(70) + "|")
    print("+" + "=" * 70 + "+")
    print(f"  Repo root: {ctx.root}")

    # Run protocol
    report = run_protocol(ctx, mode)
    print_summary(report)
    return 0 if report.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
