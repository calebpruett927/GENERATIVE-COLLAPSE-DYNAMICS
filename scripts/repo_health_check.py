#!/usr/bin/env python3
"""UMCP Repository Health Check — proactive drift and corruption detection.

Catches the exact class of problems that accumulate silently:
  - Freeze drift (frozen hashes no longer match current files)
  - Broken file references (manifest/config points at nonexistent files)
  - Version inconsistency (pyproject, __init__, report, trace_meta disagree)
  - CHANGELOG gaps (git tags with no changelog entry)
  - Stale sidecar hashes (freeze/*.sha256 disagree with freeze manifest)
  - Merge conflict residue (conflict markers in tracked files)

Usage:
    python scripts/repo_health_check.py           # Full check (exit 0=ok, 1=problems)
    python scripts/repo_health_check.py --json     # Machine-readable output
    python scripts/repo_health_check.py --fix      # Auto-fix what can be fixed

Designed to run in:
  - pre-commit protocol (step_repo_health)
  - CI pipeline (.github/workflows/validate.yml)
  - Scheduled GitHub Action (weekly)
  - Local development (ad-hoc)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# Ensure venv bin/ is on PATH
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")


# ── Types ────────────────────────────────────────────────────────


class Severity(StrEnum):
    ERROR = "ERROR"  # Blocks commit / CI
    WARN = "WARN"  # Should be fixed soon
    INFO = "INFO"  # Informational


@dataclass
class Finding:
    """A single detected problem."""

    check: str
    severity: Severity
    message: str
    detail: str = ""
    fixable: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.detail:
            d["detail"] = self.detail
        if self.fixable:
            d["fixable"] = True
        return d


@dataclass
class HealthReport:
    """Aggregated health check result."""

    findings: list[Finding] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    checked: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARN)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.ok else "FAIL",
            "timestamp": self.timestamp,
            "errors": self.error_count,
            "warnings": self.warn_count,
            "checks_run": self.checked,
            "findings": [f.to_dict() for f in self.findings],
        }


# ── Utilities ────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_yaml(path: Path) -> Any:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _find_repo_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


# ── Checks ───────────────────────────────────────────────────────
# Each check function takes (root, report) and appends findings.


def check_freeze_drift(root: Path, report: HealthReport) -> None:
    """Verify freeze manifest hashes match current file content."""
    report.checked.append("freeze_drift")
    manifest_path = root / "freeze" / "freeze_manifest.json"
    if not manifest_path.exists():
        report.findings.append(
            Finding("freeze_drift", Severity.WARN, "No freeze manifest found at freeze/freeze_manifest.json")
        )
        return

    manifest = json.loads(manifest_path.read_text())
    hashes: dict[str, str] = manifest.get("hashes", {})
    drifted: list[str] = []

    for fname, frozen_hash in hashes.items():
        fpath = root / fname
        if not fpath.exists():
            report.findings.append(
                Finding(
                    "freeze_drift",
                    Severity.ERROR,
                    f"Frozen file missing: {fname}",
                    detail=f"freeze_manifest.json references {fname} but file does not exist",
                )
            )
            continue
        current_hash = _sha256(fpath)
        if current_hash != frozen_hash:
            drifted.append(fname)
            report.findings.append(
                Finding(
                    "freeze_drift",
                    Severity.ERROR,
                    f"Freeze drift: {fname}",
                    detail=f"frozen={frozen_hash[:16]}… current={current_hash[:16]}…",
                    fixable=True,
                )
            )

    if not drifted:
        pass  # All clean — no finding needed


def check_freeze_sidecars(root: Path, report: HealthReport) -> None:
    """Verify .sha256 sidecar files agree with freeze manifest."""
    report.checked.append("freeze_sidecars")
    manifest_path = root / "freeze" / "freeze_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    hashes: dict[str, str] = manifest.get("hashes", {})

    for fname, manifest_hash in hashes.items():
        sidecar_name = fname.replace("/", "_") + ".sha256"
        sidecar_path = root / "freeze" / sidecar_name
        if not sidecar_path.exists():
            report.findings.append(Finding("freeze_sidecars", Severity.WARN, f"Missing sidecar: freeze/{sidecar_name}"))
            continue
        sidecar_hash = sidecar_path.read_text().strip()
        if sidecar_hash != manifest_hash:
            report.findings.append(
                Finding(
                    "freeze_sidecars",
                    Severity.WARN,
                    f"Sidecar disagrees with manifest: {sidecar_name}",
                    detail=f"sidecar={sidecar_hash[:16]}… manifest={manifest_hash[:16]}…",
                    fixable=True,
                )
            )


def check_version_consistency(root: Path, report: HealthReport) -> None:
    """Check that version strings agree across pyproject.toml, __init__.py, report, trace_meta."""
    report.checked.append("version_consistency")
    versions: dict[str, str] = {}

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
            if m:
                versions["pyproject.toml"] = m.group(1)
                break

    # src/umcp/__init__.py
    init_py = root / "src" / "umcp" / "__init__.py"
    if init_py.exists():
        for line in init_py.read_text().splitlines():
            m = re.match(r'^__version__\s*=\s*"([^"]+)"', line.strip())
            if m:
                versions["__init__.py"] = m.group(1)
                break

    # integrity/code_version.txt
    ver_file = root / "integrity" / "code_version.txt"
    if ver_file.exists():
        # First line only — file may have commit metadata on subsequent lines
        first_line = ver_file.read_text().splitlines()[0].strip()
        versions["code_version.txt"] = first_line.lstrip("v")

    # outputs/report.txt
    report_file = root / "outputs" / "report.txt"
    if report_file.exists():
        for line in report_file.read_text().splitlines():
            m = re.match(r"Validator:\s*umcp-validator\s+v(.+)", line.strip())
            if m:
                versions["outputs/report.txt"] = m.group(1)
                break

    # derived/trace_meta.yaml
    trace_meta = root / "derived" / "trace_meta.yaml"
    if trace_meta.exists():
        try:
            meta = _load_yaml(trace_meta)
            v = meta.get("trace_meta", {}).get("generated", {}).get("validator_version")
            if v:
                versions["derived/trace_meta.yaml"] = str(v)
        except Exception:
            pass

    # Compare
    unique = set(versions.values())
    if len(unique) > 1:
        canonical = versions.get("pyproject.toml", "unknown")
        for source, ver in versions.items():
            if ver != canonical:
                report.findings.append(
                    Finding(
                        "version_consistency",
                        Severity.ERROR,
                        f"Version mismatch in {source}: {ver} (expected {canonical})",
                        fixable=True,
                    )
                )


def check_manifest_refs(root: Path, report: HealthReport) -> None:
    """Check that manifest.yaml references only files that exist."""
    report.checked.append("manifest_refs")
    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        return

    try:
        manifest = _load_yaml(manifest_path)
    except Exception:
        report.findings.append(Finding("manifest_refs", Severity.ERROR, "manifest.yaml is not valid YAML"))
        return

    # Walk the manifest looking for "path" keys
    def _find_paths(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "path" and isinstance(v, str):
                    paths.append((prefix, v))
                else:
                    paths.extend(_find_paths(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                paths.extend(_find_paths(item, f"{prefix}[{i}]"))
        return paths

    for location, path_str in _find_paths(manifest):
        full_path = root / path_str
        if not full_path.exists():
            report.findings.append(
                Finding(
                    "manifest_refs",
                    Severity.ERROR,
                    f"Broken reference in manifest.yaml: {path_str}",
                    detail=f"Referenced at {location}",
                )
            )


def check_merge_conflicts(root: Path, report: HealthReport) -> None:
    """Scan tracked files for unresolved merge conflict markers."""
    report.checked.append("merge_conflicts")
    conflict_pattern = re.compile(r"^(<{7}\s|={7}$|>{7}\s)", re.MULTILINE)

    # Use git to list tracked text files
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = result.stdout.strip().splitlines()
    except Exception:
        # Fallback: scan key directories
        files = []
        for ext in ("*.yaml", "*.yml", "*.json", "*.csv", "*.py", "*.md", "*.txt"):
            files.extend(str(p.relative_to(root)) for p in root.rglob(ext) if ".git" not in str(p))

    binary_exts = {".png", ".jpg", ".npz", ".gz", ".pdf", ".lock", ".whl"}
    for fname in files:
        fpath = root / fname
        if not fpath.exists() or fpath.suffix in binary_exts:
            continue
        try:
            content = fpath.read_text(errors="replace")
            if conflict_pattern.search(content):
                report.findings.append(
                    Finding(
                        "merge_conflicts",
                        Severity.ERROR,
                        f"Merge conflict markers in {fname}",
                    )
                )
        except Exception:
            pass


def check_changelog_tags(root: Path, report: HealthReport) -> None:
    """Check that every git tag has a corresponding CHANGELOG entry."""
    report.checked.append("changelog_tags")
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return

    content = changelog.read_text()
    # Find all [x.y.z] entries
    changelog_versions = set(re.findall(r"\[(\d+\.\d+\.\d+)\]", content))

    # Get git tags
    try:
        result = subprocess.run(
            ["git", "tag", "-l", "v*"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        tags = result.stdout.strip().splitlines()
    except Exception:
        return

    for tag in tags:
        ver = tag.lstrip("v")
        # Only check semver tags (x.y.z format, no suffixes)
        if not re.match(r"\d+\.\d+\.\d+$", ver):
            continue
        # Only check tags from v1.3.0 onward (CHANGELOG started then)
        parts = ver.split(".")
        if int(parts[0]) < 1 or (int(parts[0]) == 1 and int(parts[1]) < 3):
            continue
        if ver not in changelog_versions:
            report.findings.append(
                Finding(
                    "changelog_tags",
                    Severity.WARN,
                    f"Git tag {tag} has no CHANGELOG entry",
                    detail=f"Add a ## [{ver}] section to CHANGELOG.md",
                )
            )


def check_required_files(root: Path, report: HealthReport) -> None:
    """Check that the 16 root validation files exist."""
    report.checked.append("required_files")
    required = [
        "contract.yaml",
        "embedding.yaml",
        "observables.yaml",
        "return.yaml",
        "closures.yaml",
        "manifest.yaml",
        "weights.csv",
        "derived/trace.csv",
        "derived/trace_meta.yaml",
        "outputs/invariants.csv",
        "outputs/regimes.csv",
        "outputs/report.txt",
        "outputs/welds.csv",
        "validator_rules.yaml",
        "closures/registry.yaml",
        "integrity/sha256.txt",
    ]
    for fname in required:
        if not (root / fname).exists():
            report.findings.append(Finding("required_files", Severity.ERROR, f"Required file missing: {fname}"))


def check_schema_consistency(root: Path, report: HealthReport) -> None:
    """Check all JSON schemas use the same draft version."""
    report.checked.append("schema_consistency")
    schema_dir = root / "schemas"
    if not schema_dir.exists():
        return

    drafts: dict[str, str] = {}
    for schema_file in sorted(schema_dir.glob("*.json")):
        try:
            schema = json.loads(schema_file.read_text())
            draft = schema.get("$schema", "")
            if draft:
                drafts[schema_file.name] = draft
        except Exception:
            report.findings.append(
                Finding("schema_consistency", Severity.ERROR, f"Invalid JSON in schema: {schema_file.name}")
            )

    unique_drafts = set(drafts.values())
    if len(unique_drafts) > 1:
        for name, draft in drafts.items():
            # Find the majority draft
            from collections import Counter

            majority = Counter(drafts.values()).most_common(1)[0][0]
            if draft != majority:
                report.findings.append(
                    Finding(
                        "schema_consistency",
                        Severity.WARN,
                        f"Schema {name} uses different draft",
                        detail=f"has {draft}, majority uses {majority}",
                    )
                )


# ── Auto-fix ─────────────────────────────────────────────────────


def _read_canonical_version(root: Path) -> str | None:
    """Read the single source of truth for version: pyproject.toml."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None
    for line in pyproject.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
        if m:
            return m.group(1)
    return None


def sync_version(root: Path, *, quiet: bool = False) -> int:
    """Propagate canonical version from pyproject.toml to ALL satellite files.

    This is the core "assume latest" mechanism — every satellite file is
    unconditionally updated to match the single source of truth.  Called by
    auto_fix (reactive) and can also be called proactively.

    Returns count of files changed.
    """
    canonical = _read_canonical_version(root)
    if not canonical:
        return 0

    changed = 0

    # 1. integrity/code_version.txt — first line is the version
    ver_file = root / "integrity" / "code_version.txt"
    if ver_file.exists():
        lines = ver_file.read_text().splitlines(keepends=True)
        expected_first = f"v{canonical}\n" if lines and lines[0].endswith("\n") else f"v{canonical}"
        if lines and lines[0].strip() != f"v{canonical}":
            lines[0] = expected_first
            ver_file.write_text("".join(lines))
            changed += 1
            if not quiet:
                print(f"    ✓ Synced integrity/code_version.txt → v{canonical}")

    # 2. outputs/report.txt — Validator: umcp-validator vX.Y.Z
    report_file = root / "outputs" / "report.txt"
    if report_file.exists():
        content = report_file.read_text()
        new_content = re.sub(r"umcp-validator v[\d.]+", f"umcp-validator v{canonical}", content)
        if new_content != content:
            report_file.write_text(new_content)
            changed += 1
            if not quiet:
                print(f"    ✓ Synced outputs/report.txt → v{canonical}")

    # 3. derived/trace_meta.yaml — validator_version: "X.Y.Z"
    trace_meta = root / "derived" / "trace_meta.yaml"
    if trace_meta.exists():
        content = trace_meta.read_text()
        new_content = re.sub(
            r'validator_version:\s*["\']?[\d.]+["\']?',
            f'validator_version: "{canonical}"',
            content,
        )
        if new_content != content:
            trace_meta.write_text(new_content)
            changed += 1
            if not quiet:
                print(f"    ✓ Synced derived/trace_meta.yaml → {canonical}")

    # 4. src/umcp/__init__.py — __version__ = "X.Y.Z"
    init_py = root / "src" / "umcp" / "__init__.py"
    if init_py.exists():
        content = init_py.read_text()
        new_content = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{canonical}"', content)
        if new_content != content:
            init_py.write_text(new_content)
            changed += 1
            if not quiet:
                print(f"    ✓ Synced src/umcp/__init__.py → {canonical}")

    return changed


def auto_fix(root: Path, report: HealthReport) -> int:
    """Attempt to auto-fix fixable findings. Returns count of fixes applied."""
    fixes = 0

    # 1. Proactive version sync — always runs if ANY version inconsistency detected
    if any(f.check == "version_consistency" and f.fixable for f in report.findings):
        print("  → Syncing satellite files to canonical version...")
        fixes += sync_version(root)

    # 2. Fix freeze drift by re-running freeze
    if any(f.check == "freeze_drift" and f.fixable for f in report.findings):
        print("  → Re-freezing baseline...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/freeze_baseline.py",
                    "--run-id",
                    f"AUTOFIX-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            fixes += 1
            print("    ✓ Freeze updated")
        except Exception as e:
            print(f"    ✗ Freeze update failed: {e}")

    # 3. Re-stage any fixed files so downstream steps (integrity, commit) see them
    if fixes > 0:
        with contextlib.suppress(Exception):
            subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True, timeout=10)

    return fixes


# ── Runner ───────────────────────────────────────────────────────


def run_health_check(root: Path | None = None, fix: bool = False) -> HealthReport:
    """Run all health checks and return the report."""
    if root is None:
        root = _find_repo_root()

    report = HealthReport()

    checks = [
        check_freeze_drift,
        check_freeze_sidecars,
        check_version_consistency,
        check_manifest_refs,
        check_merge_conflicts,
        check_changelog_tags,
        check_required_files,
        check_schema_consistency,
    ]

    for check_fn in checks:
        try:
            check_fn(root, report)
        except Exception as e:
            report.findings.append(Finding(check_fn.__name__, Severity.ERROR, f"Check crashed: {e}"))

    if fix and report.findings:
        auto_fix(root, report)

    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="UMCP Repository Health Check — proactive drift and corruption detection"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--fix", action="store_true", help="Auto-fix what can be fixed")
    parser.add_argument("--root", default=None, help="Repo root (default: auto-detect)")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else _find_repo_root()

    if not args.json:
        print("=" * 70)
        print("UMCP Repository Health Check")
        print("=" * 70)
        print(f"Root: {root}")
        print()

    report = run_health_check(root, fix=args.fix)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        if not report.findings:
            print("  ✓ All checks passed — no problems detected")
        else:
            for f in report.findings:
                symbol = "✗" if f.severity == Severity.ERROR else "⚠" if f.severity == Severity.WARN else "ℹ"
                print(f"  {symbol} [{f.severity.value}] {f.message}")
                if f.detail:
                    print(f"    {f.detail}")

        print()
        print(f"Checks run: {len(report.checked)}")
        print(f"Errors: {report.error_count}  Warnings: {report.warn_count}")
        print(f"Status: {'PASS' if report.ok else 'FAIL'}")
        print("=" * 70)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
