"""UMCP CLI - Command Line Interface for validation."""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportConstantRedefinition=false

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from umcp import VALIDATOR_NAME, __version__
from umcp.logging_utils import HealthCheck, get_logger

# Import optimized kernel validation (OPT-1, Lemma 1 bounds)
try:
    from umcp.kernel_optimized import validate_kernel_bounds

    _HAS_KERNEL_OPTIMIZATIONS = True
except ImportError:
    _HAS_KERNEL_OPTIMIZATIONS = False
    validate_kernel_bounds = None  # type: ignore[assignment]


# Ensure src is in sys.path for absolute imports when running as a script
def _ensure_repo_root_in_syspath() -> None:  # pyright: ignore[reportUnusedFunction]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# -----------------------------
# Internal codes (stable)
# -----------------------------
# These are validator-internal issue codes (not part of semantic rules).
E_MISSING = "E001"
E_SCHEMA_INVALID = "E002"
E_SCHEMA_FAIL = "E003"
E_PARSE = "E004"


RE_CK = re.compile(r"^c_[0-9]+$")
RE_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
RE_INT = re.compile(r"[-+]?\d+$")

# Pre-compiled patterns for strict validation
RE_POSITIVE_WELD_CLAIM = re.compile(r"\bweld\s+(?:PASS|validated|demonstrated|confirmed|verified)\b", re.IGNORECASE)
RE_POSITIVE_SEAM_CLAIM = re.compile(r"\bseam\s+(?:PASS|validated|demonstrated|confirmed|verified)\b", re.IGNORECASE)
RE_POSITIVE_CONTINUITY_CLAIM = re.compile(
    r"\bcontinuity\s+claim\s+(?:PASS|validated|demonstrated|confirmed)\b", re.IGNORECASE
)

# -----------------------------
# Persistent cache management
# -----------------------------
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}
_FILE_CONTENT_CACHE: dict[str, Any] = {}
_FILE_HASH_CACHE: dict[str, str] = {}
_CACHE_STATS = {"hits": 0, "misses": 0, "schema_reuse": 0, "file_reuse": 0}


def _get_cache_dir(repo_root: Path) -> Path:
    """Get or create .umcp_cache directory at repo root."""
    cache_dir = repo_root / ".umcp_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_cache_metadata(repo_root: Path) -> dict[str, Any]:
    """Load cache metadata from disk."""
    cache_file = _get_cache_dir(repo_root) / "validation_cache.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                return (
                    dict(data)
                    if isinstance(data, dict)
                    else {"file_hashes": {}, "schema_hashes": {}, "stats": {"total_runs": 0}}
                )
        except Exception:
            pass
    return {"file_hashes": {}, "schema_hashes": {}, "stats": {"total_runs": 0}}


def _save_cache_metadata(repo_root: Path, metadata: dict[str, Any]) -> None:
    """Save cache metadata to disk."""
    cache_file = _get_cache_dir(repo_root) / "validation_cache.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(metadata, f)
    except Exception:
        pass  # Silent fail on cache write


def _get_cached_validator(schema: dict[str, Any], schema_id: str) -> Draft202012Validator:
    """Get cached validator or create and cache new one."""
    if schema_id not in _VALIDATOR_CACHE:
        _VALIDATOR_CACHE[schema_id] = Draft202012Validator(schema)
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
        _CACHE_STATS["schema_reuse"] += 1
    return _VALIDATOR_CACHE[schema_id]


class LazySchemaLoader:
    """Lazy-load schemas only when first accessed. Reduces overhead for targeted validation."""

    def __init__(self, repo_root: Path, repo_target: TargetResult):
        self.repo_root = repo_root
        self.repo_target = repo_target
        self._schemas: dict[str, dict[str, Any] | None] = {}
        self._loaded: set[str] = set()

    def get(self, schema_name: str) -> dict[str, Any] | None:
        """Load and validate schema on first access."""
        if schema_name in self._loaded:
            return self._schemas.get(schema_name)

        self._loaded.add(schema_name)
        schema_path = self.repo_root / "schemas" / f"{schema_name}.schema.json"
        schema = _validate_schema_json(self.repo_target, self.repo_root, schema_path)
        self._schemas[schema_name] = schema
        return schema


# -----------------------------
# Result model (matches schemas/validator.result.schema.json)
# -----------------------------
@dataclass
class Issue:
    severity: str  # ERROR|WARN|INFO
    code: str  # e.g. E101, W201, E001
    message: str
    path: str | None = None
    json_pointer: str | None = None
    hint: str | None = None
    rule: str | None = None

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            obj["path"] = self.path
        if self.json_pointer is not None:
            obj["json_pointer"] = self.json_pointer
        if self.hint is not None:
            obj["hint"] = self.hint
        if self.rule is not None:
            obj["rule"] = self.rule
        return obj


@dataclass
class TargetResult:
    target_type: str  # repo|casepack|file|directory
    target_path: str
    run_status: str = "CONFORMANT"
    counts: dict[str, int] = field(default_factory=lambda: {"errors": 0, "warnings": 0, "info": 0})
    issues: list[Issue] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.counts["errors"] += 1
        elif issue.severity == "WARN":
            self.counts["warnings"] += 1
        else:
            self.counts["info"] += 1

    def finalize_status(self, fail_on_warning: bool) -> None:
        if self.counts["errors"] > 0 or (fail_on_warning and self.counts["warnings"] > 0):
            self.run_status = "NONCONFORMANT"
        else:
            self.run_status = "CONFORMANT"

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "target_type": self.target_type,
            "target_path": self.target_path,
            "run_status": self.run_status,
            "counts": dict(self.counts),
            "issues": [i.to_json() for i in self.issues],
        }
        if self.artifacts:
            obj["artifacts"] = self.artifacts
        return obj


# -----------------------------
# Basic helpers
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash. Returns 'unknown' if not in git repo or error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unknown"


def _get_python_version() -> str:
    """Get Python version string (e.g., '3.12.1')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> Any:
    """Load JSON with content-based caching."""
    path_str = str(path)

    # Check if file exists in cache and hasn't changed
    if path_str in _FILE_HASH_CACHE:
        try:
            current_hash = _compute_file_hash(path)
            if current_hash == _FILE_HASH_CACHE[path_str] and path_str in _FILE_CONTENT_CACHE:
                _CACHE_STATS["file_reuse"] += 1
                return _FILE_CONTENT_CACHE[path_str]
        except Exception:
            pass

    # Load and cache
    content = json.loads(_read_text(path))
    try:
        _FILE_CONTENT_CACHE[path_str] = content
        _FILE_HASH_CACHE[path_str] = _compute_file_hash(path)
    except Exception:
        pass  # Silent fail on cache
    return content


def _load_yaml(path: Path) -> Any:
    """Load YAML with content-based caching. Fallback to minimal parser if PyYAML unavailable and file is simple."""
    path_str = str(path)

    # Check if file exists in cache and hasn't changed
    if path_str in _FILE_HASH_CACHE:
        try:
            current_hash = _compute_file_hash(path)
            if current_hash == _FILE_HASH_CACHE[path_str] and path_str in _FILE_CONTENT_CACHE:
                _CACHE_STATS["file_reuse"] += 1
                return _FILE_CONTENT_CACHE[path_str]
        except Exception:
            pass

    if yaml is not None:
        # Load and cache using PyYAML
        content = yaml.safe_load(_read_text(path))
        try:
            _FILE_CONTENT_CACHE[path_str] = content
            _FILE_HASH_CACHE[path_str] = _compute_file_hash(path)
        except Exception:
            pass  # Silent fail on cache
        return content
    else:
        # Minimal YAML parser for simple key: value pairs (no nested structures)
        result = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
        try:
            _FILE_CONTENT_CACHE[path_str] = result
            _FILE_HASH_CACHE[path_str] = _compute_file_hash(path)
        except Exception:
            pass
        return result


@lru_cache(maxsize=256)
def _get_resolved_path(path_str: str) -> Path:
    """Cache resolved paths to avoid repeated filesystem operations."""
    return Path(path_str).resolve()


def _relpath(repo_root: Path, p: Path) -> str:
    try:
        repo_resolved = _get_resolved_path(str(repo_root))
        p_resolved = _get_resolved_path(str(p))
        return p_resolved.relative_to(repo_resolved).as_posix()
    except Exception:
        return p.as_posix()


def _append_to_ledger(repo_root: Path, run_status: str, invariants_data: dict[str, Any] | None = None) -> None:
    """
    Append validation result to continuous ledger at ledger/return_log.csv.
    Records: timestamp, run_status, Δκ (delta_kappa), s (stiffness), and optional observables.
    """
    ledger_dir = repo_root / "ledger"
    ledger_path = ledger_dir / "return_log.csv"

    # Ensure ledger directory exists
    ledger_dir.mkdir(exist_ok=True)

    # Check if we need to write header
    write_header = not ledger_path.exists() or ledger_path.stat().st_size == 0

    # Prepare row data
    timestamp = _utc_now_iso()
    row = {
        "timestamp": timestamp,
        "run_status": run_status,
        "delta_kappa": "",
        "stiffness": "",
        "omega": "",
        "curvature": "",
    }

    # Extract invariants if available
    if invariants_data:
        row["delta_kappa"] = invariants_data.get("delta_kappa", "")
        row["stiffness"] = invariants_data.get("S", "")
        row["omega"] = invariants_data.get("omega", "")
        row["curvature"] = invariants_data.get("C", "")

    # Append to ledger
    with open(ledger_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "run_status",
            "delta_kappa",
            "stiffness",
            "omega",
            "curvature",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _require_file(target: TargetResult, repo_root: Path, p: Path, kind_hint: str = "") -> bool:
    if p.exists() and p.is_file():
        return True
    target.add_issue(
        Issue(
            severity="ERROR",
            code=E_MISSING,
            message=f"Missing required file: {_relpath(repo_root, p)}",
            path=_relpath(repo_root, p),
            json_pointer=None,
            hint=(f"Create the file at this exact path. {kind_hint}".strip() or None),
            rule="require_file",
        )
    )
    return False


def _require_dir(target: TargetResult, repo_root: Path, p: Path, kind_hint: str = "") -> bool:
    if p.exists() and p.is_dir():
        return True
    target.add_issue(
        Issue(
            severity="ERROR",
            code=E_MISSING,
            message=f"Missing required directory: {_relpath(repo_root, p)}",
            path=_relpath(repo_root, p),
            json_pointer=None,
            hint=(f"Create the directory at this exact path. {kind_hint}".strip() or None),
            rule="require_dir",
        )
    )
    return False


def _validate_schema_json(target: TargetResult, repo_root: Path, schema_path: Path) -> dict[str, Any] | None:
    """
    Load and Draft202012Validator.check_schema(schema). On failure, emit E002.
    """
    if not _require_file(target, repo_root, schema_path, "Schemas must exist under schemas/*.json"):
        return None
    try:
        schema = _load_json(schema_path)
    except Exception as e:
        target.add_issue(
            Issue(
                severity="ERROR",
                code=E_PARSE,
                message=f"Schema JSON parse failed: {_relpath(repo_root, schema_path)}",
                path=_relpath(repo_root, schema_path),
                json_pointer=None,
                hint=str(e),
                rule="schema_parse",
            )
        )
        return None

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        target.add_issue(
            Issue(
                severity="ERROR",
                code=E_SCHEMA_INVALID,
                message=f"Invalid JSON Schema: {_relpath(repo_root, schema_path)}",
                path=_relpath(repo_root, schema_path),
                json_pointer=None,
                hint=str(e),
                rule="check_schema",
            )
        )
        return None
    return dict(schema)


def _validate_instance_against_schema(
    target: TargetResult,
    repo_root: Path,
    instance: Any,
    instance_path: Path,
    schema: dict[str, Any],
    schema_name: str,
) -> None:
    v = _get_cached_validator(schema, schema_name)
    errors = sorted(v.iter_errors(instance), key=lambda e: (e.json_path, e.message))
    if not errors:
        return

    # Emit one issue per schema error to keep debugging concrete.
    for e in errors:
        jp = e.json_path if e.json_path else "/"
        target.add_issue(
            Issue(
                severity="ERROR",
                code=E_SCHEMA_FAIL,
                message=f"Schema validation failed ({schema_name}): {_relpath(repo_root, instance_path)}",
                path=_relpath(repo_root, instance_path),
                json_pointer=jp,
                hint=e.message,
                rule="schema_validate",
            )
        )


def _coerce_scalar(v: Any) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    low = s.lower()
    if low in {"true", "t", "yes", "y"}:
        return True
    if low in {"false", "f", "no", "n"}:
        return False
    # Use pre-compiled regex patterns
    if RE_INT.fullmatch(s):
        try:
            return int(s)
        except Exception:
            return s
    if RE_FLOAT.fullmatch(s):
        try:
            return float(s)
        except Exception:
            return s
    return s


def _parse_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        rows: list[dict[str, Any]] = []
        for r in reader:
            rows.append({k: _coerce_scalar(v) for k, v in r.items()})
        return rows


def _infer_psi_format(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "psi_trace_csv_long"
    keys_union: set[str] = set()
    for r in rows:
        keys_union.update(r.keys())
    if "dim" in keys_union and "c" in keys_union:
        return "psi_trace_csv_long"
    if any(RE_CK.match(k) for k in keys_union):
        return "psi_trace_csv_wide"
    return "psi_trace_csv_long"


def _close(lhs: float, rhs: float, atol: float, rtol: float) -> bool:
    # Canonical closeness: abs(lhs-rhs) <= atol + rtol*abs(rhs)
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return False
    return abs(lhs - rhs) <= (atol + rtol * abs(rhs))


def _dot_get(obj: Any, dotpath: str) -> Any:
    cur = obj
    for part in dotpath.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# -----------------------------
# Semantic rule execution (built-in types)
# -----------------------------
def _load_validator_rules(
    target: TargetResult,
    repo_root: Path,
    rules_path: Path,
    schema_rules: dict[str, Any],
) -> dict[str, Any] | None:
    if not _require_file(
        target,
        repo_root,
        rules_path,
        "This repo expects validator_rules.yaml at the root.",
    ):
        return None
    try:
        rules_doc = _load_yaml(rules_path)
    except Exception as e:
        target.add_issue(
            Issue(
                severity="ERROR",
                code=E_PARSE,
                message=f"YAML parse failed: {_relpath(repo_root, rules_path)}",
                path=_relpath(repo_root, rules_path),
                json_pointer=None,
                hint=str(e),
                rule="yaml_parse",
            )
        )
        return None

    _validate_instance_against_schema(
        target,
        repo_root,
        rules_doc,
        rules_path,
        schema_rules,
        "validator.rules.schema.json",
    )
    if target.counts["errors"] > 0:
        return None
    return dict(rules_doc)


def _rule_by_id(rules_doc: dict[str, Any], rule_id: str) -> dict[str, Any] | None:
    for r in rules_doc.get("rules", []):
        if r.get("id") == rule_id:
            result: dict[str, Any] = dict(r)
            return result
    return None


def _emit_rule_issue(
    target: TargetResult,
    repo_root: Path,
    severity: str,
    code: str,
    message: str,
    file_path: Path,
    json_pointer: str,
    hint: str,
    rule_name: str,
) -> None:
    target.add_issue(
        Issue(
            severity=severity,
            code=code,
            message=message,  # MUST be exact rule.message
            path=_relpath(repo_root, file_path),
            json_pointer=json_pointer,
            hint=hint,
            rule=rule_name,
        )
    )


def _expected_regime_label(omega: float, F: float, S: float, C: float, regimes: dict[str, Any]) -> str:
    # OPT-1: Use optimized kernel validation if available (Lemma 1 bounds)
    if _HAS_KERNEL_OPTIMIZATIONS and validate_kernel_bounds is not None:
        # Approximate κ and IC from ω for validation
        import math

        IC = max(1 - omega, 1e-10)
        kappa = math.log(IC)
        # Silent validation - log on failure but don't raise
        try:
            validate_kernel_bounds(F=F, omega=omega, C=C, IC=IC, kappa=kappa)
        except (ValueError, TypeError):
            pass  # Continue with standard regime classification

    # Canonical expected label:
    # - Collapse if omega >= omega_gte
    # - Stable if omega < omega_lt AND F > F_gt AND S < S_lt AND C < C_lt
    # - else Watch
    if omega >= float(regimes["collapse"]["omega_gte"]):
        return "Collapse"
    st = regimes["stable"]
    if (
        (omega < float(st["omega_lt"]))
        and (float(st["F_gt"]) < F)
        and (float(st["S_lt"]) > S)
        and (float(st["C_lt"]) > C)
    ):
        return "Stable"
    return "Watch"


def _apply_semantic_rules_to_casepack(
    target: TargetResult,
    repo_root: Path,
    rules_doc: dict[str, Any],
    canon_doc: dict[str, Any],
    psi_csv_path: Path | None,
    invariants_json_path: Path | None,
) -> None:
    # -------------------------
    # E101: ψ wide must have at least one c_k column
    # -------------------------
    rule = _rule_by_id(rules_doc, "E101")
    if rule and rule.get("enabled", True) and psi_csv_path and psi_csv_path.exists():
        rows = _parse_csv_rows(psi_csv_path)
        fmt = _infer_psi_format(rows)
        if fmt == "psi_trace_csv_wide":
            keys_union: set[str] = set()
            for r in rows:
                keys_union.update(r.keys())
            pattern = rule["check"]["pattern"]
            min_matches = int(rule["check"]["min_matches"])
            rx = re.compile(pattern)
            matches = sum(1 for k in keys_union if rx.match(k))
            if matches < min_matches:
                hint = f"Observed {matches} keys matching {pattern} across rows. Add at least one coordinate column (c_1, c_2, ...)."
                _emit_rule_issue(
                    target=target,
                    repo_root=repo_root,
                    severity=rule["severity"],
                    code=rule["id"],
                    message=rule["message"],
                    file_path=psi_csv_path,
                    json_pointer="/rows",
                    hint=hint,
                    rule_name="pattern_min_matches",
                )

    # If no invariants file, nothing else to do.
    if not invariants_json_path or not invariants_json_path.exists():
        return

    inv = _load_json(invariants_json_path)
    inv_rows = inv.get("rows", [])

    # -------------------------
    # W201: F ≈ 1 − ω
    # -------------------------
    rule = _rule_by_id(rules_doc, "W201")
    if rule and rule.get("enabled", True):
        atol = float(rule.get("atol", 1.0e-9))
        rtol = float(rule.get("rtol", 0.0))
        omega_path = rule["check"]["fields"]["omega"]
        F_path = rule["check"]["fields"]["F"]
        on_missing = rule["check"].get("on_missing", "warn")

        for i, row in enumerate(inv_rows):
            omega = _dot_get(row, omega_path)
            F = _dot_get(row, F_path)
            jp = f"/rows/{i}"

            if not isinstance(omega, (int, float)) or not isinstance(F, (int, float)):
                if on_missing != "skip":
                    sev = "ERROR" if on_missing == "error" else rule["severity"]
                    hint = f"Missing or non-numeric fields for identity check. Required: omega, F. Observed omega={omega!r}, F={F!r}."
                    _emit_rule_issue(
                        target,
                        repo_root,
                        sev,
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        jp,
                        hint,
                        "tier1_identity_F_equals_one_minus_omega",
                    )
                continue

            omega_f = float(omega)
            F_f = float(F)
            rhs = 1.0 - omega_f
            if not _close(F_f, rhs, atol=atol, rtol=rtol):
                delta = F_f - rhs
                hint = (
                    f"Row identity failed: F vs (1-omega). "
                    f"F={F_f}, omega={omega_f}, rhs={rhs}, delta={delta}, atol={atol}, rtol={rtol}."
                )
                _emit_rule_issue(
                    target,
                    repo_root,
                    rule["severity"],
                    rule["id"],
                    rule["message"],
                    invariants_json_path,
                    jp,
                    hint,
                    "tier1_identity_F_equals_one_minus_omega",
                )

    # -------------------------
    # W202: IC ≈ exp(κ)
    # -------------------------
    rule = _rule_by_id(rules_doc, "W202")
    if rule and rule.get("enabled", True):
        atol = float(rule.get("atol", 1.0e-9))
        rtol = float(rule.get("rtol", 1.0e-9))
        IC_path = rule["check"]["fields"]["IC"]
        kappa_path = rule["check"]["fields"]["kappa"]
        on_missing = rule["check"].get("on_missing", "warn")

        for i, row in enumerate(inv_rows):
            IC = _dot_get(row, IC_path)
            kappa = _dot_get(row, kappa_path)
            jp = f"/rows/{i}"

            if not isinstance(IC, (int, float)) or not isinstance(kappa, (int, float)):
                if on_missing != "skip":
                    sev = "ERROR" if on_missing == "error" else rule["severity"]
                    hint = f"Missing or non-numeric fields for identity check. Required: IC, kappa. Observed IC={IC!r}, kappa={kappa!r}."
                    _emit_rule_issue(
                        target,
                        repo_root,
                        sev,
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        jp,
                        hint,
                        "tier1_identity_IC_equals_exp_kappa",
                    )
                continue

            IC_f = float(IC)
            kappa_f = float(kappa)
            rhs = math.exp(kappa_f)
            if not math.isfinite(rhs):
                hint = f"exp(kappa) is non-finite (overflow/NaN). kappa={kappa_f}, exp(kappa)={rhs!r}."
                _emit_rule_issue(
                    target,
                    repo_root,
                    rule["severity"],
                    rule["id"],
                    rule["message"],
                    invariants_json_path,
                    jp,
                    hint,
                    "tier1_identity_IC_equals_exp_kappa",
                )
                continue

            if not _close(IC_f, rhs, atol=atol, rtol=rtol):
                delta = IC_f - rhs
                hint = (
                    f"Row identity failed: IC vs exp(kappa). "
                    f"IC={IC_f}, kappa={kappa_f}, rhs={rhs}, delta={delta}, atol={atol}, rtol={rtol}."
                )
                _emit_rule_issue(
                    target,
                    repo_root,
                    rule["severity"],
                    rule["id"],
                    rule["message"],
                    invariants_json_path,
                    jp,
                    hint,
                    "tier1_identity_IC_equals_exp_kappa",
                )

    # -------------------------
    # W301: regime.label consistent with canon thresholds; critical overlay when checkable
    # -------------------------
    rule = _rule_by_id(rules_doc, "W301")
    if rule and rule.get("enabled", True):
        regimes = canon_doc["umcp_canon"]["regimes"]

        fields = rule["check"]["fields"]
        policies = rule["check"]["policies"]
        on_missing_regime = policies.get("on_missing_regime", "warn")
        on_missing_icmin = policies.get("on_missing_IC_min", "skip")

        omega_path = fields["omega"]
        F_path = fields["F"]
        S_path = fields["S"]
        C_path = fields["C"]
        label_path = fields["regime_label"]
        crit_path = fields["critical_overlay"]
        icmin_path = fields["IC_min"]

        for i, row in enumerate(inv_rows):
            omega = _dot_get(row, omega_path)
            F = _dot_get(row, F_path)
            S = _dot_get(row, S_path)
            C = _dot_get(row, C_path)

            if not all(isinstance(x, (int, float)) for x in [omega, F, S, C]):
                if on_missing_regime != "skip":
                    sev = "ERROR" if on_missing_regime == "error" else rule["severity"]
                    hint = (
                        "Cannot compute expected regime label (missing/non-numeric omega/F/S/C). "
                        f"Observed omega={omega!r}, F={F!r}, S={S!r}, C={C!r}."
                    )
                    _emit_rule_issue(
                        target,
                        repo_root,
                        sev,
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        f"/rows/{i}",
                        hint,
                        "regime_label_consistency",
                    )
                continue

            omega_f = float(omega)
            F_f = float(F)
            S_f = float(S)
            C_f = float(C)

            exp_label = _expected_regime_label(omega_f, F_f, S_f, C_f, regimes)

            provided_label = _dot_get(row, label_path)
            if provided_label is None:
                if on_missing_regime != "skip":
                    sev = "ERROR" if on_missing_regime == "error" else rule["severity"]
                    hint = f"Missing regime.label. Expected label would be '{exp_label}'."
                    _emit_rule_issue(
                        target,
                        repo_root,
                        sev,
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        f"/rows/{i}",
                        hint,
                        "regime_label_consistency",
                    )
            else:
                if provided_label not in {"Stable", "Watch", "Collapse"}:
                    hint = f"Invalid regime.label value: {provided_label!r}. Expected one of Stable|Watch|Collapse."
                    _emit_rule_issue(
                        target,
                        repo_root,
                        rule["severity"],
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        f"/rows/{i}/regime/label",
                        hint,
                        "regime_label_consistency",
                    )
                elif provided_label != exp_label:
                    hint = (
                        f"Regime mismatch. expected='{exp_label}', observed='{provided_label}'. "
                        "Computed from omega/F/S/C against canon thresholds."
                    )
                    _emit_rule_issue(
                        target,
                        repo_root,
                        rule["severity"],
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        f"/rows/{i}/regime/label",
                        hint,
                        "regime_label_consistency",
                    )

            # Critical overlay check (only if IC_min numeric/finite; else policy)
            IC_min = _dot_get(row, icmin_path)
            crit = _dot_get(row, crit_path)

            if (IC_min is None) or (not isinstance(IC_min, (int, float))) or (not math.isfinite(float(IC_min))):
                if on_missing_icmin in {"warn", "error"} and crit is not None:
                    sev = "ERROR" if on_missing_icmin == "error" else rule["severity"]
                    hint = (
                        "critical_overlay is present but IC_min is missing/non-numeric; cannot verify. "
                        f"Observed IC_min={IC_min!r}."
                    )
                    _emit_rule_issue(
                        target,
                        repo_root,
                        sev,
                        rule["id"],
                        rule["message"],
                        invariants_json_path,
                        f"/rows/{i}/regime",
                        hint,
                        "regime_label_consistency",
                    )
            else:
                min_ic_lt = float(regimes["collapse"]["critical_overlay"]["min_IC_lt"])
                expected_crit = float(IC_min) < min_ic_lt
                if crit is not None:
                    if not isinstance(crit, bool):
                        hint = f"critical_overlay must be boolean when present. Observed={crit!r}."
                        _emit_rule_issue(
                            target,
                            repo_root,
                            rule["severity"],
                            rule["id"],
                            rule["message"],
                            invariants_json_path,
                            f"/rows/{i}/regime/critical_overlay",
                            hint,
                            "regime_label_consistency",
                        )
                    elif crit != expected_crit:
                        hint = (
                            f"Critical overlay mismatch. expected={expected_crit}, observed={crit}. "
                            f"Based on IC_min={float(IC_min)} and canon min_IC_lt={min_ic_lt}."
                        )
                        _emit_rule_issue(
                            target,
                            repo_root,
                            rule["severity"],
                            rule["id"],
                            rule["message"],
                            invariants_json_path,
                            f"/rows/{i}/regime/critical_overlay",
                            hint,
                            "regime_label_consistency",
                        )


# -----------------------------
# Validation workflow
# -----------------------------
def _should_skip_casepack(cache_metadata: dict[str, Any], manifest_path: Path, case_path: str) -> bool:
    """Check if casepack can be skipped based on unchanged manifest hash."""
    if not manifest_path.exists():
        return False

    try:
        # Check if we have cached result for this casepack
        cached_casepacks = cache_metadata.get("casepack_results", {})
        if case_path not in cached_casepacks:
            return False

        cached_entry = cached_casepacks[case_path]

        # Only skip if previous result was CONFORMANT
        if cached_entry.get("status") != "CONFORMANT":
            return False

        # Check if manifest hash matches
        current_hash = _compute_file_hash(manifest_path)
        cached_hash = cached_entry.get("manifest_hash")

        result: bool = current_hash == cached_hash
        return result
    except Exception:
        return False


def _cache_casepack_result(cache_metadata: dict[str, Any], case_path: str, manifest_path: Path, status: str) -> None:
    """Cache casepack validation result with manifest hash."""
    try:
        if "casepack_results" not in cache_metadata:
            cache_metadata["casepack_results"] = {}

        cache_metadata["casepack_results"][case_path] = {
            "status": status,
            "manifest_hash": (_compute_file_hash(manifest_path) if manifest_path.exists() else None),
            "last_validated": _utc_now_iso(),
        }
    except Exception:
        pass  # Silent fail on cache update


def _validate_casepack_strict(target: TargetResult, repo_root: Path, case_dir: Path, fail_on_warning: bool) -> None:
    """
    Apply strict validation rules to a CasePack.
    Strict rules (emitted as warnings in baseline, errors in strict):
    - contracts/ must exist with all required files (contract.yaml, embedding.yaml, return.yaml, weights.yaml)
    - contract.yaml must include all UMA.INTSTACK.v1 required fields
    - weights.yaml must sum to 1.0 within tolerance
    - closures/closure_registry.yaml must exist
    - receipts/ss1m.json must include manifest root hash
    - If README contains weld/seam/continuity language, require seam_receipt.json with PASS
    """
    severity = "ERROR" if fail_on_warning else "WARN"

    # Check contracts directory and required files
    contracts_dir = case_dir / "contracts"
    if not contracts_dir.exists():
        target.add_issue(
            Issue(
                severity=severity,
                code="W101",
                message=f"Strict: contracts/ directory missing in {_relpath(repo_root, case_dir)}",
                path=_relpath(repo_root, case_dir),
                hint="Create contracts/ with contract.yaml, embedding.yaml, return.yaml, weights.yaml",
                rule="strict_casepack_structure",
            )
        )
    else:
        # Check required contract files
        required_files = [
            "contract.yaml",
            "embedding.yaml",
            "return.yaml",
            "weights.yaml",
        ]
        for fname in required_files:
            fpath = contracts_dir / fname
            if not fpath.exists():
                target.add_issue(
                    Issue(
                        severity=severity,
                        code="W102",
                        message=f"Strict: Missing required contract file: {fname}",
                        path=_relpath(repo_root, contracts_dir),
                        hint=f"Create {fname} with explicit contract specifications",
                        rule="strict_contract_files",
                    )
                )

        # Validate contract.yaml contains required UMA.INTSTACK.v1 fields
        contract_path = contracts_dir / "contract.yaml"
        if contract_path.exists():
            try:
                contract_doc = _load_yaml(contract_path)
                frozen_params = contract_doc.get("contract", {}).get("tier_1_kernel", {}).get("frozen_parameters", {})
                required_params = [
                    "a",
                    "b",
                    "face",
                    "epsilon",
                    "p",
                    "alpha",
                    "lambda",
                    "eta",
                    "tol_seam",
                    "tol_id",
                ]
                for param in required_params:
                    if param not in frozen_params:
                        target.add_issue(
                            Issue(
                                severity=severity,
                                code="W103",
                                message=f"Strict: contract.yaml missing required parameter: {param}",
                                path=_relpath(repo_root, contract_path),
                                hint=f"Add {param} to contract.tier_1_kernel.frozen_parameters",
                                rule="strict_contract_completeness",
                            )
                        )
            except Exception:
                pass  # Parse errors already caught by baseline validation

        # Validate weights.yaml sums to 1.0
        weights_path = contracts_dir / "weights.yaml"
        if weights_path.exists():
            try:
                weights_doc = _load_yaml(weights_path)
                channels = weights_doc.get("weights", {}).get("channels", [])
                if channels:
                    weight_sum = sum(ch.get("weight", 0) for ch in channels)
                    tol = weights_doc.get("weights", {}).get("validation", {}).get("tolerance", 1e-9)
                    if abs(weight_sum - 1.0) > tol:
                        target.add_issue(
                            Issue(
                                severity=severity,
                                code="W104",
                                message=f"Strict: weights do not sum to 1.0 (sum={weight_sum:.12f})",
                                path=_relpath(repo_root, weights_path),
                                hint=f"Adjust weights to sum to 1.0 within tolerance {tol}",
                                rule="strict_weights_normalization",
                            )
                        )
            except Exception:
                pass

    # Check closures directory
    closures_dir = case_dir / "closures"
    closure_registry = closures_dir / "closure_registry.yaml"
    if not closure_registry.exists():
        target.add_issue(
            Issue(
                severity=severity,
                code="W105",
                message="Strict: closures/closure_registry.yaml missing",
                path=_relpath(repo_root, case_dir),
                hint="Create closure_registry.yaml declaring weld budget terms",
                rule="strict_closures",
            )
        )

    # Check receipts directory and SS1M manifest hash
    receipts_dir = case_dir / "receipts"
    ss1m_path = receipts_dir / "ss1m.json"
    if ss1m_path.exists():
        try:
            ss1m_doc = _load_json(ss1m_path)
            manifest_hash = ss1m_doc.get("receipt", {}).get("manifest", {}).get("root_sha256")
            if not manifest_hash or manifest_hash == "pending":
                target.add_issue(
                    Issue(
                        severity=severity,
                        code="W106",
                        message="Strict: ss1m.json missing manifest root_sha256",
                        path=_relpath(repo_root, ss1m_path),
                        hint="Run generate_manifest.py to update receipt with manifest hash",
                        rule="strict_manifest_integrity",
                    )
                )

            # Check environment metadata
            if "environment" not in ss1m_doc.get("receipt", {}):
                target.add_issue(
                    Issue(
                        severity=severity,
                        code="W107",
                        message="Strict: ss1m.json missing environment metadata",
                        path=_relpath(repo_root, ss1m_path),
                        hint="Add environment metadata (python_version, platform, hostname) to receipt",
                        rule="strict_environment_metadata",
                    )
                )
        except Exception:
            pass

    # Check README for continuity claims requiring seam_receipt
    readme_path = case_dir / "README.md"
    if readme_path.exists():
        try:
            readme_text = readme_path.read_text()
            # Use pre-compiled regex patterns for better performance
            has_positive_claim = (
                RE_POSITIVE_WELD_CLAIM.search(readme_text)
                or RE_POSITIVE_SEAM_CLAIM.search(readme_text)
                or RE_POSITIVE_CONTINUITY_CLAIM.search(readme_text)
            )

            if has_positive_claim:
                seam_receipt_path = case_dir / "receipts" / "seam_receipt.json"
                if not seam_receipt_path.exists():
                    target.add_issue(
                        Issue(
                            severity=severity,
                            code="W108",
                            message="Strict: README makes continuity claim but seam_receipt.json missing",
                            path=_relpath(repo_root, readme_path),
                            hint="Either remove continuity claims or add seam_receipt.json with PASS status",
                            rule="strict_continuity_claim",
                        )
                    )
                else:
                    # Validate seam receipt status is PASS
                    try:
                        seam_doc = _load_json(seam_receipt_path)
                        status = seam_doc.get("receipt", {}).get("status")
                        if status != "PASS":
                            target.add_issue(
                                Issue(
                                    severity=severity,
                                    code="W109",
                                    message=f"Strict: seam_receipt.json status is {status}, expected PASS",
                                    path=_relpath(repo_root, seam_receipt_path),
                                    hint="Fix seam calculation or remove continuity claims from README",
                                    rule="strict_continuity_integrity",
                                )
                            )
                    except Exception:
                        pass
        except Exception:
            pass


def _find_repo_root(start: Path) -> Path | None:
    """
    Find repo root by searching upward for pyproject.toml.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return None


def _validate_repo(repo_root: Path, fail_on_warning: bool) -> dict[str, Any]:
    # Load persistent cache metadata
    cache_metadata = _load_cache_metadata(repo_root)
    cache_metadata["stats"]["total_runs"] = cache_metadata["stats"].get("total_runs", 0) + 1

    repo_target = TargetResult(target_type="repo", target_path=".")
    casepack_targets: list[TargetResult] = []

    # Basic directories
    _require_dir(repo_target, repo_root, repo_root / "schemas")
    _require_dir(repo_target, repo_root, repo_root / "canon")
    _require_dir(repo_target, repo_root, repo_root / "contracts")
    _require_dir(repo_target, repo_root, repo_root / "closures")
    _require_dir(repo_target, repo_root, repo_root / "casepacks")

    # Use lazy schema loader - only loads schemas when actually needed
    schemas = LazySchemaLoader(repo_root, repo_target)

    # Always load result schema (needed for self-validation) and rules/canon (commonly used)
    schema_result = schemas.get("validator.result")
    schema_rules = schemas.get("validator.rules")
    schema_canon = schemas.get("canon.anchors")

    # Validate all schemas present are structurally valid (best effort).
    schemas_dir = repo_root / "schemas"
    if schemas_dir.exists():
        for sp in sorted(schemas_dir.glob("*.json")):
            _validate_schema_json(repo_target, repo_root, sp)

    # Load and validate anchors.yaml (already loaded in lazy loader)
    canon_path = repo_root / "canon" / "anchors.yaml"
    canon_doc = None
    if _require_file(repo_target, repo_root, canon_path) and schema_canon:
        try:
            canon_doc = _load_yaml(canon_path)
        except Exception as e:
            repo_target.add_issue(
                Issue(
                    severity="ERROR",
                    code=E_PARSE,
                    message=f"YAML parse failed: {_relpath(repo_root, canon_path)}",
                    path=_relpath(repo_root, canon_path),
                    hint=str(e),
                    rule="yaml_parse",
                )
            )
        else:
            _validate_instance_against_schema(
                repo_target,
                repo_root,
                canon_doc,
                canon_path,
                schema_canon,
                "canon.anchors.schema.json",
            )

    # Load and validate contract (lazy load)
    contract_path = repo_root / "contracts" / "UMA.INTSTACK.v1.yaml"
    schema_contract = schemas.get("contract")
    if _require_file(repo_target, repo_root, contract_path) and schema_contract:
        try:
            contract_doc = _load_yaml(contract_path)
        except Exception as e:
            repo_target.add_issue(
                Issue(
                    severity="ERROR",
                    code=E_PARSE,
                    message=f"YAML parse failed: {_relpath(repo_root, contract_path)}",
                    path=_relpath(repo_root, contract_path),
                    hint=str(e),
                    rule="yaml_parse",
                )
            )
        else:
            _validate_instance_against_schema(
                repo_target,
                repo_root,
                contract_doc,
                contract_path,
                schema_contract,
                "contract.schema.json",
            )

    # Load and validate closures registry + referenced closure files (lazy load)
    closures_registry_path = repo_root / "closures" / "registry.yaml"
    schema_closures = schemas.get("closures")
    if _require_file(repo_target, repo_root, closures_registry_path) and schema_closures:
        try:
            registry_doc = _load_yaml(closures_registry_path)
        except Exception as e:
            repo_target.add_issue(
                Issue(
                    severity="ERROR",
                    code=E_PARSE,
                    message=f"YAML parse failed: {_relpath(repo_root, closures_registry_path)}",
                    path=_relpath(repo_root, closures_registry_path),
                    hint=str(e),
                    rule="yaml_parse",
                )
            )
            registry_doc = None
        else:
            _validate_instance_against_schema(
                repo_target,
                repo_root,
                registry_doc,
                closures_registry_path,
                schema_closures,
                "closures.schema.json",
            )

        if registry_doc and isinstance(registry_doc, dict):
            ref_paths = []
            closures_map = (registry_doc.get("registry", {}) or {}).get("closures", {}) or {}
            if isinstance(closures_map, dict):
                for key in [
                    "gamma",
                    "return_domain",
                    "norms",
                    "curvature_neighborhood",
                ]:
                    val = closures_map.get(key)
                    if isinstance(val, dict) and "path" in val and isinstance(val["path"], str):
                        ref_paths.append(val["path"])

            for rp in ref_paths:
                cp = (repo_root / rp).resolve()
                if _require_file(repo_target, repo_root, cp) and schema_closures:
                    try:
                        cdoc = _load_yaml(cp)
                    except Exception as e:
                        repo_target.add_issue(
                            Issue(
                                severity="ERROR",
                                code=E_PARSE,
                                message=f"YAML parse failed: {_relpath(repo_root, cp)}",
                                path=_relpath(repo_root, cp),
                                hint=str(e),
                                rule="yaml_parse",
                            )
                        )
                    else:
                        _validate_instance_against_schema(
                            repo_target,
                            repo_root,
                            cdoc,
                            cp,
                            schema_closures,
                            "closures.schema.json",
                        )

    # Load and validate validator_rules.yaml
    rules_doc = None
    rules_path = repo_root / "validator_rules.yaml"
    if schema_rules:
        rules_doc = _load_validator_rules(repo_target, repo_root, rules_path, schema_rules)

    # Validate casepacks
    casepacks_dir = repo_root / "casepacks"
    casepacks_skipped = 0
    if casepacks_dir.exists():
        for case_dir in sorted(p for p in casepacks_dir.iterdir() if p.is_dir()):
            manifest_path = case_dir / "manifest.json"
            expected_dir = case_dir / "expected"
            psi_csv_path = expected_dir / "psi.csv"
            invariants_path = expected_dir / "invariants.json"
            ss1m_path = expected_dir / "ss1m_receipt.json"

            case_path = _relpath(repo_root, case_dir)

            # Smart casepack skipping: Skip if manifest unchanged and previously CONFORMANT
            if _should_skip_casepack(cache_metadata, manifest_path, case_path):
                casepacks_skipped += 1
                t = TargetResult(target_type="casepack", target_path=case_path)
                t.run_status = "CONFORMANT"
                casepack_targets.append(t)
                continue

            t = TargetResult(target_type="casepack", target_path=case_path)
            # Required structure
            _require_file(t, repo_root, manifest_path, "CasePack requires manifest.json")
            _require_dir(
                t,
                repo_root,
                expected_dir,
                "CasePack requires expected/ outputs for regression/publication",
            )

            # Validate manifest schema (lazy load)
            schema_manifest = schemas.get("manifest")
            if manifest_path.exists() and schema_manifest:
                try:
                    mdoc = _load_json(manifest_path)
                except Exception as e:
                    t.add_issue(
                        Issue(
                            severity="ERROR",
                            code=E_PARSE,
                            message=f"JSON parse failed: {_relpath(repo_root, manifest_path)}",
                            path=_relpath(repo_root, manifest_path),
                            hint=str(e),
                            rule="json_parse",
                        )
                    )
                else:
                    _validate_instance_against_schema(
                        t,
                        repo_root,
                        mdoc,
                        manifest_path,
                        schema_manifest,
                        "manifest.schema.json",
                    )

            # Validate psi.csv via schema (parsed, lazy load)
            if psi_csv_path.exists():
                schema_psi = schemas.get("trace.psi")
                if not schema_psi:
                    continue
                try:
                    rows = _parse_csv_rows(psi_csv_path)
                    fmt = _infer_psi_format(rows)
                    psi_doc = {
                        "schema": "schemas/trace.psi.schema.json",
                        "format": fmt,
                        "rows": rows,
                    }
                except Exception as e:
                    t.add_issue(
                        Issue(
                            severity="ERROR",
                            code=E_PARSE,
                            message=f"CSV parse failed: {_relpath(repo_root, psi_csv_path)}",
                            path=_relpath(repo_root, psi_csv_path),
                            hint=str(e),
                            rule="csv_parse",
                        )
                    )
                else:
                    _validate_instance_against_schema(
                        t,
                        repo_root,
                        psi_doc,
                        psi_csv_path,
                        schema_psi,
                        "trace.psi.schema.json",
                    )

            # Validate invariants.json (lazy load)
            if invariants_path.exists():
                schema_invariants = schemas.get("invariants")
                if not schema_invariants:
                    continue
                try:
                    inv_doc = _load_json(invariants_path)
                except Exception as e:
                    t.add_issue(
                        Issue(
                            severity="ERROR",
                            code=E_PARSE,
                            message=f"JSON parse failed: {_relpath(repo_root, invariants_path)}",
                            path=_relpath(repo_root, invariants_path),
                            hint=str(e),
                            rule="json_parse",
                        )
                    )
                else:
                    _validate_instance_against_schema(
                        t,
                        repo_root,
                        inv_doc,
                        invariants_path,
                        schema_invariants,
                        "invariants.schema.json",
                    )

            # Validate SS1m receipt if present (lazy load)
            if ss1m_path.exists():
                schema_ss1m = schemas.get("receipt.ss1m")
                if not schema_ss1m:
                    continue
                try:
                    ss1m_doc = _load_json(ss1m_path)
                except Exception as e:
                    t.add_issue(
                        Issue(
                            severity="ERROR",
                            code=E_PARSE,
                            message=f"JSON parse failed: {_relpath(repo_root, ss1m_path)}",
                            path=_relpath(repo_root, ss1m_path),
                            hint=str(e),
                            rule="json_parse",
                        )
                    )
                else:
                    _validate_instance_against_schema(
                        t,
                        repo_root,
                        ss1m_doc,
                        ss1m_path,
                        schema_ss1m,
                        "receipt.ss1m.schema.json",
                    )

            # Apply semantic rules if rules + canon available
            if rules_doc and canon_doc and (psi_csv_path.exists() or invariants_path.exists()):
                _apply_semantic_rules_to_casepack(
                    target=t,
                    repo_root=repo_root,
                    rules_doc=rules_doc,
                    canon_doc=canon_doc,
                    psi_csv_path=psi_csv_path if psi_csv_path.exists() else None,
                    invariants_json_path=(invariants_path if invariants_path.exists() else None),
                )

            # Apply strict validation rules
            _validate_casepack_strict(t, repo_root, case_dir, fail_on_warning)

            t.finalize_status(fail_on_warning=fail_on_warning)

            # Cache result for future smart skipping
            _cache_casepack_result(cache_metadata, case_path, manifest_path, t.run_status)

            casepack_targets.append(t)

    # Finalize repo status (aggregate counts)
    repo_target.counts["errors"] += sum(t.counts["errors"] for t in casepack_targets)
    repo_target.counts["warnings"] += sum(t.counts["warnings"] for t in casepack_targets)
    repo_target.counts["info"] += sum(t.counts["info"] for t in casepack_targets)
    repo_target.finalize_status(fail_on_warning=fail_on_warning)

    # Summary block
    targets_total = 1 + len(casepack_targets)
    targets_failed = sum(1 for t in [repo_target, *casepack_targets] if t.run_status != "CONFORMANT")

    # Save cache metadata and stats
    cache_metadata["file_hashes"] = _FILE_HASH_CACHE.copy()
    cache_metadata["last_run"] = _utc_now_iso()
    cache_metadata["cache_stats"] = _CACHE_STATS.copy()
    _save_cache_metadata(repo_root, cache_metadata)

    targets_data: list[dict[str, Any]] = [repo_target.to_json(), *[t.to_json() for t in casepack_targets]]
    result: dict[str, Any] = {
        "schema": "schemas/validator.result.schema.json",
        "created_utc": _utc_now_iso(),
        "validator": {
            "name": VALIDATOR_NAME,
            "version": __version__,
            "implementation": {
                "language": "python",
                "python_version": _get_python_version(),
                "git_commit": _get_git_commit(repo_root),
                "build": "repo",
            },
        },
        "run_status": repo_target.run_status,
        "summary": {
            "counts": {
                "errors": repo_target.counts["errors"],
                "warnings": repo_target.counts["warnings"],
                "info": repo_target.counts["info"],
                "targets_total": targets_total,
                "targets_failed": targets_failed,
            },
            "policy": {
                "strict": bool(fail_on_warning),
                "fail_on_warning": bool(fail_on_warning),
            },
            "cache_stats": {
                "schema_validators_cached": len(_VALIDATOR_CACHE),
                "files_cached": len(_FILE_CONTENT_CACHE),
                "cache_hits": _CACHE_STATS["hits"],
                "cache_misses": _CACHE_STATS["misses"],
                "file_reuse": _CACHE_STATS["file_reuse"],
                "schema_reuse": _CACHE_STATS["schema_reuse"],
                "casepacks_skipped": casepacks_skipped,
                "total_validation_runs": cache_metadata["stats"]["total_runs"],
            },
        },
        "targets": targets_data,
        "issues": [],  # optional flattened list; leaving empty avoids duplication
        "notes": "UMCP repository validation",
    }

    # Append to continuous ledger if CONFORMANT
    if repo_target.run_status == "CONFORMANT":
        # Try to extract invariants from outputs/invariants.csv
        invariants_csv = repo_root / "outputs" / "invariants.csv"
        invariants_data = None
        if invariants_csv.exists():
            try:
                with open(invariants_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        invariants_data = rows[0]
            except Exception:
                pass  # Ledger is best-effort
        _append_to_ledger(repo_root, repo_target.run_status, invariants_data)

    # Self-validate validator.result.json output if schema_result loaded successfully
    if schema_result is not None:
        v = Draft202012Validator(schema_result)
        errs = sorted(v.iter_errors(result), key=lambda e: (e.json_path, e.message))
        if errs:
            # If the validator cannot validate its own output, mark NON_EVALUABLE and attach internal error.
            repo_target.add_issue(
                Issue(
                    severity="ERROR",
                    code=E_SCHEMA_FAIL,
                    message="validator.result.json output does not conform to validator.result.schema.json",
                    path=None,
                    json_pointer=errs[0].json_path or "/",
                    hint=errs[0].message,
                    rule="self_validate_output",
                )
            )
            repo_target.finalize_status(fail_on_warning=fail_on_warning)
            result["run_status"] = "NON_EVALUABLE"
            result["summary"]["counts"]["errors"] += 1
            # Update repo target in result - targets_data was defined earlier with proper typing
            targets_data[0] = repo_target.to_json()

    return result


# -----------------------------
# CLI
# -----------------------------
def _cmd_validate(args: argparse.Namespace) -> int:
    # Initialize logger
    json_output = bool(os.environ.get("UMCP_JSON_LOGS", ""))
    verbose = bool(getattr(args, "verbose", False))
    logger = get_logger(json_output=json_output, include_metrics=verbose)

    start = Path(args.path).resolve()
    repo_root = _find_repo_root(start)
    if repo_root is None:
        logger.error("Could not find repo root", path=str(start))
        print(
            "ERROR: Could not find repo root (no pyproject.toml found in parents).",
            file=sys.stderr,
        )
        return 2

    # Determine strict mode: --strict flag OR --fail-on-warning (legacy)
    strict_mode = bool(getattr(args, "strict", False)) or bool(args.fail_on_warning)

    logger.info("Starting validation", repo_root=str(repo_root), strict=strict_mode)

    with logger.operation("validate_repo", path=str(repo_root)):
        result = _validate_repo(repo_root=repo_root, fail_on_warning=strict_mode)

    out_json = json.dumps(result, indent=2, sort_keys=False)

    # Compute sha256 of result
    result_hash = hashlib.sha256(out_json.encode("utf-8")).hexdigest()

    # Extract provenance details
    created_utc = result.get("created_utc", "unknown")
    git_commit = result.get("validator", {}).get("implementation", {}).get("git_commit", "unknown")
    python_version = result.get("validator", {}).get("implementation", {}).get("python_version", "unknown")
    error_count = result.get("summary", {}).get("counts", {}).get("errors", 0)
    warning_count = result.get("summary", {}).get("counts", {}).get("warnings", 0)

    # Generate governance summary with full provenance
    governance_note = (
        f"UMCP validation: {result['run_status']} (repo + casepacks/hello_world), "
        f"errors={error_count} warnings={warning_count}; "
        f"validator={VALIDATOR_NAME} v{__version__} (build=repo, commit={git_commit[:8] if git_commit != 'unknown' else git_commit}, python={python_version}); "
        f"policy strict={str(strict_mode).lower()}; "
        f"created_utc={created_utc}; "
        f"sha256={result_hash[:16]}...\n"
        f"Note: non-strict = baseline structural validity; strict = publication lint gate."
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.write_text(out_json + "\n", encoding="utf-8")
        print(f"Wrote validator result: {_relpath(repo_root, out_path)}")
        print(f"\n{governance_note}")
    else:
        print(out_json)
        print(f"\n{governance_note}", file=sys.stderr)

    return 0 if result.get("run_status") == "CONFORMANT" else 1


def _cmd_run(args: argparse.Namespace) -> int:
    """
    Operational placeholder:
    - Performs validation of the target path (repo or casepack).
    - Does not generate Ψ/invariants from raw inputs (engine not implemented in this repo build).
    """
    # For now, run == validate with optional output path.
    return _cmd_validate(args)


def _cmd_diff(args: argparse.Namespace) -> int:
    """Compare two validation receipts and show differences."""
    try:
        r1_file = Path(args.receipt1)
        r2_file = Path(args.receipt2)

        if not r1_file.exists():
            print(f"Error: Receipt not found: {args.receipt1}", file=sys.stderr)
            return 1
        if not r2_file.exists():
            print(f"Error: Receipt not found: {args.receipt2}", file=sys.stderr)
            return 1

        with r1_file.open("r") as f:
            receipt1 = json.load(f)
        with r2_file.open("r") as f:
            receipt2 = json.load(f)

        # Display header
        print("=" * 80)
        print("UMCP Receipt Comparison")
        print("=" * 80)
        print(f"Receipt 1: {args.receipt1}")
        print(f"Receipt 2: {args.receipt2}")
        print()

        # Compare basic info
        print("📋 Basic Information")
        print("-" * 80)
        _compare_field(receipt1, receipt2, "run_status", "Status")
        _compare_field(receipt1, receipt2, "created_utc", "Created UTC")
        print()

        # Compare validation results
        print("✅ Validation Results")
        print("-" * 80)
        summary1 = receipt1.get("summary", {}).get("counts", {})
        summary2 = receipt2.get("summary", {}).get("counts", {})
        _compare_dict_field(summary1, summary2, "errors", "Errors")
        _compare_dict_field(summary1, summary2, "warnings", "Warnings")

        if getattr(args, "verbose", False):
            targets1 = receipt1.get("targets", [])
            targets2 = receipt2.get("targets", [])
            if len(targets1) != len(targets2):
                print(f"  Target count changed: {len(targets1)} → {len(targets2)}")
        print()

        # Compare implementation
        print("🔧 Implementation")
        print("-" * 80)
        impl1 = receipt1.get("validator", {}).get("implementation", {})
        impl2 = receipt2.get("validator", {}).get("implementation", {})
        _compare_dict_field(impl1, impl2, "git_commit", "Git Commit")
        _compare_dict_field(impl1, impl2, "python_version", "Python Version")
        print()

        # Compare policy
        print("⚖️  Policy")
        print("-" * 80)
        policy1 = receipt1.get("summary", {}).get("policy", {})
        policy2 = receipt2.get("summary", {}).get("policy", {})
        _compare_dict_field(policy1, policy2, "strict", "Strict Mode")
        _compare_dict_field(policy1, policy2, "fail_on_warning", "Fail on Warning")
        print()

        # Compare targets validated
        print("📦 Targets Validated")
        print("-" * 80)
        targets1 = {t.get("target_path") for t in receipt1.get("targets", []) if isinstance(t, dict)}
        targets2 = {t.get("target_path") for t in receipt2.get("targets", []) if isinstance(t, dict)}

        added = targets2 - targets1
        removed = targets1 - targets2
        common = targets1 & targets2

        print(f"  Common: {len(common)}")
        if added:
            print(f"  Added in Receipt 2: {len(added)}")
            if getattr(args, "verbose", False):
                for target in sorted(t for t in added if t):
                    print(f"    + {target}")
        if removed:
            print(f"  Removed from Receipt 1: {len(removed)}")
            if getattr(args, "verbose", False):
                for target in sorted(t for t in removed if t):
                    print(f"    - {target}")
        print()

        # Summary
        print("📊 Summary")
        print("-" * 80)

        changes = []
        if receipt1.get("run_status") != receipt2.get("run_status"):
            changes.append("Status changed")
        if impl1.get("git_commit") != impl2.get("git_commit"):
            changes.append("Git commit changed")
        if policy1.get("strict") != policy2.get("strict"):
            changes.append("Policy mode changed")
        if added or removed:
            changes.append("Targets changed")
        if summary1.get("errors") != summary2.get("errors"):
            changes.append("Error count changed")

        if changes:
            print("Changes detected:")
            for change in changes:
                print(f"  • {change}")
        else:
            print("No significant changes detected.")

        print("=" * 80)

        return 0

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error comparing receipts: {e}", file=sys.stderr)
        return 1


def _compare_field(dict1: dict[str, Any], dict2: dict[str, Any], key: str, label: str) -> None:
    """Compare a single field between two dictionaries."""
    val1 = dict1.get(key)
    val2 = dict2.get(key)

    if val1 == val2:
        print(f"  {label}: {val1} (unchanged)")
    else:
        print(f"  {label}: {val1} → {val2}")


def _compare_dict_field(dict1: dict[str, Any], dict2: dict[str, Any], key: str, label: str) -> None:
    """Compare a nested field between two dictionaries."""
    val1 = dict1.get(key)
    val2 = dict2.get(key)

    if val1 == val2:
        print(f"  {label}: {val1} (unchanged)")
    else:
        print(f"  {label}: {val1} → {val2}")


def _cmd_health(args: argparse.Namespace) -> int:
    """Check system health and readiness for production operation."""
    start = Path(args.path).resolve()
    repo_root = _find_repo_root(start)
    if repo_root is None:
        print(
            "ERROR: Could not find repo root (no pyproject.toml found in parents).",
            file=sys.stderr,
        )
        return 2

    health = HealthCheck.check(repo_root)

    if args.json:
        print(json.dumps(health, indent=2))
    else:
        # Human-readable output
        print("=" * 80)
        print("UMCP System Health Check")
        print("=" * 80)
        print(f"Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        print()

        print("Checks:")
        for check_name, check_data in health.get("checks", {}).items():
            if isinstance(check_data, dict):
                status = check_data.get("status", "unknown")
                symbol = "✓" if status == "pass" else "✗"
                print(f"  {symbol} {check_name}: {status}")
                if status == "fail" and "error" in check_data:
                    print(f"    Error: {check_data['error']}")
        print()

        if "metrics" in health:
            print("Metrics:")
            if "schemas_count" in health["metrics"]:
                print(f"  Schemas: {health['metrics']['schemas_count']}")
            if "system" in health["metrics"]:
                sys_metrics = health["metrics"]["system"]
                print(f"  CPU: {sys_metrics.get('cpu_percent', 'N/A')}%")
                print(f"  Memory: {sys_metrics.get('memory_percent', 'N/A')}%")
                print(f"  Disk: {sys_metrics.get('disk_percent', 'N/A')}%")
            print()

        print("=" * 80)

    return 0 if health["status"] == "healthy" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="umcp", description="UMCP contract-first validator CLI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser(
        "validate",
        help="Validate UMCP repo artifacts, CasePacks, schemas, and semantic rules",
    )
    v.add_argument("path", nargs="?", default=".", help="Path inside repo (default: .)")
    v.add_argument("--out", default=None, help="Write validator result JSON to this file")
    v.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode: warnings become errors",
    )
    v.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="(Legacy) Treat warnings as failing",
    )
    v.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show performance metrics and detailed logging",
    )
    v.set_defaults(func=_cmd_validate)

    r = sub.add_parser("run", help="Operational placeholder: validates the target")
    r.add_argument("path", nargs="?", default=".", help="Path inside repo (default: .)")
    r.add_argument("--out", default=None, help="Write validator result JSON to this file")
    r.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode: warnings become errors",
    )
    r.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="(Legacy) Treat warnings as failing",
    )
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("diff", help="Compare two validation receipts")
    d.add_argument("receipt1", help="Path to first receipt JSON file")
    d.add_argument("receipt2", help="Path to second receipt JSON file")
    d.add_argument("--verbose", "-v", action="store_true", help="Show detailed differences")
    d.set_defaults(func=_cmd_diff)

    h = sub.add_parser("health", help="Check system health and production readiness")
    h.add_argument("path", nargs="?", default=".", help="Path inside repo (default: .)")
    h.add_argument("--json", action="store_true", help="Output as JSON for monitoring systems")
    h.set_defaults(func=_cmd_health)

    return p


def main() -> int:
    """Main CLI entry point."""
    print("[DEBUG] Entered main()")
    parser = build_parser()
    print("[DEBUG] Built parser")
    try:
        print("[DEBUG] About to parse args")
        args = parser.parse_args()
        print(f"[DEBUG] Parsed args: {args}")
        if not hasattr(args, "func"):
            print("Error: No command provided. Use --help for usage.", file=sys.stderr)
            parser.print_help()
            return 2
        print("[DEBUG] About to call command handler")
        result: int = args.func(args)
        return result
    except Exception as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
