from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from jsonschema import Draft202012Validator

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

from . import VALIDATOR_NAME, __version__


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


# -----------------------------
# Result model (matches schemas/validator.result.schema.json)
# -----------------------------
@dataclass
class Issue:
    severity: str  # ERROR|WARN|INFO
    code: str      # e.g. E101, W201, E001
    message: str
    path: Optional[str] = None
    json_pointer: Optional[str] = None
    hint: Optional[str] = None
    rule: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
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
    counts: Dict[str, int] = field(default_factory=lambda: {"errors": 0, "warnings": 0, "info": 0})
    issues: List[Issue] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.counts["errors"] += 1
        elif issue.severity == "WARN":
            self.counts["warnings"] += 1
        else:
            self.counts["info"] += 1

    def finalize_status(self, fail_on_warning: bool) -> None:
        if self.counts["errors"] > 0:
            self.run_status = "NONCONFORMANT"
        elif fail_on_warning and self.counts["warnings"] > 0:
            self.run_status = "NONCONFORMANT"
        else:
            self.run_status = "CONFORMANT"

    def to_json(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash. Returns 'unknown' if not in git repo or error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False
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
    return json.loads(_read_text(path))


def _load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml).")
    return yaml.safe_load(_read_text(path))


def _relpath(repo_root: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return p.as_posix()


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


def _validate_schema_json(target: TargetResult, repo_root: Path, schema_path: Path) -> Optional[Dict[str, Any]]:
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
    return schema


def _validate_instance_against_schema(
    target: TargetResult,
    repo_root: Path,
    instance: Any,
    instance_path: Path,
    schema: Dict[str, Any],
    schema_name: str,
) -> None:
    v = Draft202012Validator(schema)
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
    if re.fullmatch(r"[-+]?\d+", s):
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


def _parse_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        rows: List[Dict[str, Any]] = []
        for r in reader:
            rows.append({k: _coerce_scalar(v) for k, v in r.items()})
        return rows


def _infer_psi_format(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "psi_trace_csv_long"
    keys_union = set()
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
    schema_rules: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _require_file(target, repo_root, rules_path, "This repo expects validator_rules.yaml at the root."):
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

    _validate_instance_against_schema(target, repo_root, rules_doc, rules_path, schema_rules, "validator.rules.schema.json")
    if target.counts["errors"] > 0:
        return None
    return rules_doc


def _rule_by_id(rules_doc: Dict[str, Any], rule_id: str) -> Optional[Dict[str, Any]]:
    for r in rules_doc.get("rules", []):
        if r.get("id") == rule_id:
            return r
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


def _expected_regime_label(omega: float, F: float, S: float, C: float, regimes: Dict[str, Any]) -> str:
    # Canonical expected label:
    # - Collapse if omega >= omega_gte
    # - Stable if omega < omega_lt AND F > F_gt AND S < S_lt AND C < C_lt
    # - else Watch
    if omega >= float(regimes["collapse"]["omega_gte"]):
        return "Collapse"
    st = regimes["stable"]
    if (omega < float(st["omega_lt"])) and (F > float(st["F_gt"])) and (S < float(st["S_lt"])) and (C < float(st["C_lt"])):
        return "Stable"
    return "Watch"


def _apply_semantic_rules_to_casepack(
    target: TargetResult,
    repo_root: Path,
    rules_doc: Dict[str, Any],
    canon_doc: Dict[str, Any],
    psi_csv_path: Optional[Path],
    invariants_json_path: Optional[Path],
) -> None:
    # -------------------------
    # E101: ψ wide must have at least one c_k column
    # -------------------------
    rule = _rule_by_id(rules_doc, "E101")
    if rule and rule.get("enabled", True) and psi_csv_path and psi_csv_path.exists():
        rows = _parse_csv_rows(psi_csv_path)
        fmt = _infer_psi_format(rows)
        if fmt == "psi_trace_csv_wide":
            keys_union = set()
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
                        target, repo_root, sev, rule["id"], rule["message"],
                        invariants_json_path, jp, hint,
                        "tier1_identity_F_equals_one_minus_omega"
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
                    target, repo_root, rule["severity"], rule["id"], rule["message"],
                    invariants_json_path, jp, hint,
                    "tier1_identity_F_equals_one_minus_omega"
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
                        target, repo_root, sev, rule["id"], rule["message"],
                        invariants_json_path, jp, hint,
                        "tier1_identity_IC_equals_exp_kappa"
                    )
                continue

            IC_f = float(IC)
            kappa_f = float(kappa)
            rhs = math.exp(kappa_f)
            if not math.isfinite(rhs):
                hint = f"exp(kappa) is non-finite (overflow/NaN). kappa={kappa_f}, exp(kappa)={rhs!r}."
                _emit_rule_issue(
                    target, repo_root, rule["severity"], rule["id"], rule["message"],
                    invariants_json_path, jp, hint,
                    "tier1_identity_IC_equals_exp_kappa"
                )
                continue

            if not _close(IC_f, rhs, atol=atol, rtol=rtol):
                delta = IC_f - rhs
                hint = (
                    f"Row identity failed: IC vs exp(kappa). "
                    f"IC={IC_f}, kappa={kappa_f}, rhs={rhs}, delta={delta}, atol={atol}, rtol={rtol}."
                )
                _emit_rule_issue(
                    target, repo_root, rule["severity"], rule["id"], rule["message"],
                    invariants_json_path, jp, hint,
                    "tier1_identity_IC_equals_exp_kappa"
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
                        target, repo_root, sev, rule["id"], rule["message"],
                        invariants_json_path, f"/rows/{i}", hint,
                        "regime_label_consistency"
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
                        target, repo_root, sev, rule["id"], rule["message"],
                        invariants_json_path, f"/rows/{i}", hint,
                        "regime_label_consistency"
                    )
            else:
                if provided_label not in {"Stable", "Watch", "Collapse"}:
                    hint = f"Invalid regime.label value: {provided_label!r}. Expected one of Stable|Watch|Collapse."
                    _emit_rule_issue(
                        target, repo_root, rule["severity"], rule["id"], rule["message"],
                        invariants_json_path, f"/rows/{i}/regime/label", hint,
                        "regime_label_consistency"
                    )
                elif provided_label != exp_label:
                    hint = (
                        f"Regime mismatch. expected='{exp_label}', observed='{provided_label}'. "
                        "Computed from omega/F/S/C against canon thresholds."
                    )
                    _emit_rule_issue(
                        target, repo_root, rule["severity"], rule["id"], rule["message"],
                        invariants_json_path, f"/rows/{i}/regime/label", hint,
                        "regime_label_consistency"
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
                        target, repo_root, sev, rule["id"], rule["message"],
                        invariants_json_path, f"/rows/{i}/regime", hint,
                        "regime_label_consistency"
                    )
            else:
                min_ic_lt = float(regimes["collapse"]["critical_overlay"]["min_IC_lt"])
                expected_crit = float(IC_min) < min_ic_lt
                if crit is not None:
                    if not isinstance(crit, bool):
                        hint = f"critical_overlay must be boolean when present. Observed={crit!r}."
                        _emit_rule_issue(
                            target, repo_root, rule["severity"], rule["id"], rule["message"],
                            invariants_json_path, f"/rows/{i}/regime/critical_overlay", hint,
                            "regime_label_consistency"
                        )
                    elif crit != expected_crit:
                        hint = (
                            f"Critical overlay mismatch. expected={expected_crit}, observed={crit}. "
                            f"Based on IC_min={float(IC_min)} and canon min_IC_lt={min_ic_lt}."
                        )
                        _emit_rule_issue(
                            target, repo_root, rule["severity"], rule["id"], rule["message"],
                            invariants_json_path, f"/rows/{i}/regime/critical_overlay", hint,
                            "regime_label_consistency"
                        )


# -----------------------------
# Validation workflow
# -----------------------------
def _find_repo_root(start: Path) -> Optional[Path]:
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


def _validate_repo(repo_root: Path, fail_on_warning: bool) -> Dict[str, Any]:
    repo_target = TargetResult(target_type="repo", target_path=".")
    casepack_targets: List[TargetResult] = []

    # Basic directories
    _require_dir(repo_target, repo_root, repo_root / "schemas")
    _require_dir(repo_target, repo_root, repo_root / "canon")
    _require_dir(repo_target, repo_root, repo_root / "contracts")
    _require_dir(repo_target, repo_root, repo_root / "closures")
    _require_dir(repo_target, repo_root, repo_root / "casepacks")

    # Load core schemas needed for validation
    schema_rules = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "validator.rules.schema.json")
    schema_canon = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "canon.anchors.schema.json")
    schema_contract = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "contract.schema.json")
    schema_closures = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "closures.schema.json")
    schema_manifest = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "manifest.schema.json")
    schema_psi = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "trace.psi.schema.json")
    schema_invariants = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "invariants.schema.json")
    schema_ss1m = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "receipt.ss1m.schema.json")
    schema_result = _validate_schema_json(repo_target, repo_root, repo_root / "schemas" / "validator.result.schema.json")

    # Validate all schemas present are structurally valid (best effort).
    schemas_dir = repo_root / "schemas"
    if schemas_dir.exists():
        for sp in sorted(schemas_dir.glob("*.json")):
            _validate_schema_json(repo_target, repo_root, sp)

    # Load and validate anchors.yaml
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
            _validate_instance_against_schema(repo_target, repo_root, canon_doc, canon_path, schema_canon, "canon.anchors.schema.json")

    # Load and validate contract
    contract_path = repo_root / "contracts" / "UMA.INTSTACK.v1.yaml"
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
            _validate_instance_against_schema(repo_target, repo_root, contract_doc, contract_path, schema_contract, "contract.schema.json")

    # Load and validate closures registry + referenced closure files
    closures_registry_path = repo_root / "closures" / "registry.yaml"
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
            _validate_instance_against_schema(repo_target, repo_root, registry_doc, closures_registry_path, schema_closures, "closures.schema.json")

        if registry_doc and isinstance(registry_doc, dict):
            ref_paths = []
            closures_map = (registry_doc.get("registry", {}) or {}).get("closures", {}) or {}
            if isinstance(closures_map, dict):
                for key in ["gamma", "return_domain", "norms", "curvature_neighborhood"]:
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
                        _validate_instance_against_schema(repo_target, repo_root, cdoc, cp, schema_closures, "closures.schema.json")

    # Load and validate validator_rules.yaml
    rules_doc = None
    rules_path = repo_root / "validator_rules.yaml"
    if schema_rules:
        rules_doc = _load_validator_rules(repo_target, repo_root, rules_path, schema_rules)

    # Validate casepacks
    casepacks_dir = repo_root / "casepacks"
    if casepacks_dir.exists():
        for case_dir in sorted(p for p in casepacks_dir.iterdir() if p.is_dir()):
            manifest_path = case_dir / "manifest.json"
            expected_dir = case_dir / "expected"
            psi_csv_path = expected_dir / "psi.csv"
            invariants_path = expected_dir / "invariants.json"
            ss1m_path = expected_dir / "ss1m_receipt.json"

            t = TargetResult(target_type="casepack", target_path=_relpath(repo_root, case_dir))
            # Required structure
            _require_file(t, repo_root, manifest_path, "CasePack requires manifest.json")
            _require_dir(t, repo_root, expected_dir, "CasePack requires expected/ outputs for regression/publication")

            # Validate manifest schema
            if manifest_path.exists() and schema_manifest:
                try:
                    mdoc = _load_json(manifest_path)
                except Exception as e:
                    t.add_issue(Issue(
                        severity="ERROR",
                        code=E_PARSE,
                        message=f"JSON parse failed: {_relpath(repo_root, manifest_path)}",
                        path=_relpath(repo_root, manifest_path),
                        hint=str(e),
                        rule="json_parse",
                    ))
                else:
                    _validate_instance_against_schema(t, repo_root, mdoc, manifest_path, schema_manifest, "manifest.schema.json")

            # Validate psi.csv via schema (parsed)
            if psi_csv_path.exists() and schema_psi:
                try:
                    rows = _parse_csv_rows(psi_csv_path)
                    fmt = _infer_psi_format(rows)
                    psi_doc = {"schema": "schemas/trace.psi.schema.json", "format": fmt, "rows": rows}
                except Exception as e:
                    t.add_issue(Issue(
                        severity="ERROR",
                        code=E_PARSE,
                        message=f"CSV parse failed: {_relpath(repo_root, psi_csv_path)}",
                        path=_relpath(repo_root, psi_csv_path),
                        hint=str(e),
                        rule="csv_parse",
                    ))
                else:
                    _validate_instance_against_schema(t, repo_root, psi_doc, psi_csv_path, schema_psi, "trace.psi.schema.json")

            # Validate invariants.json
            if invariants_path.exists() and schema_invariants:
                try:
                    inv_doc = _load_json(invariants_path)
                except Exception as e:
                    t.add_issue(Issue(
                        severity="ERROR",
                        code=E_PARSE,
                        message=f"JSON parse failed: {_relpath(repo_root, invariants_path)}",
                        path=_relpath(repo_root, invariants_path),
                        hint=str(e),
                        rule="json_parse",
                    ))
                else:
                    _validate_instance_against_schema(t, repo_root, inv_doc, invariants_path, schema_invariants, "invariants.schema.json")

            # Validate SS1m receipt if present
            if ss1m_path.exists() and schema_ss1m:
                try:
                    ss1m_doc = _load_json(ss1m_path)
                except Exception as e:
                    t.add_issue(Issue(
                        severity="ERROR",
                        code=E_PARSE,
                        message=f"JSON parse failed: {_relpath(repo_root, ss1m_path)}",
                        path=_relpath(repo_root, ss1m_path),
                        hint=str(e),
                        rule="json_parse",
                    ))
                else:
                    _validate_instance_against_schema(t, repo_root, ss1m_doc, ss1m_path, schema_ss1m, "receipt.ss1m.schema.json")

            # Apply semantic rules if rules + canon available
            if rules_doc and canon_doc:
                if psi_csv_path.exists() or invariants_path.exists():
                    _apply_semantic_rules_to_casepack(
                        target=t,
                        repo_root=repo_root,
                        rules_doc=rules_doc,
                        canon_doc=canon_doc,
                        psi_csv_path=psi_csv_path if psi_csv_path.exists() else None,
                        invariants_json_path=invariants_path if invariants_path.exists() else None,
                    )

            t.finalize_status(fail_on_warning=fail_on_warning)
            casepack_targets.append(t)

    # Finalize repo status (aggregate counts)
    repo_target.counts["errors"] += sum(t.counts["errors"] for t in casepack_targets)
    repo_target.counts["warnings"] += sum(t.counts["warnings"] for t in casepack_targets)
    repo_target.counts["info"] += sum(t.counts["info"] for t in casepack_targets)
    repo_target.finalize_status(fail_on_warning=fail_on_warning)

    # Summary block
    targets_total = 1 + len(casepack_targets)
    targets_failed = sum(1 for t in [repo_target, *casepack_targets] if t.run_status != "CONFORMANT")

    result = {
        "schema": "schemas/validator.result.schema.json",
        "created_utc": _utc_now_iso(),
        "validator": {
            "name": VALIDATOR_NAME,
            "version": __version__,
            "implementation": {
                "language": "python",
                "python_version": _get_python_version(),
                "git_commit": _get_git_commit(repo_root),
                "build": "repo"
            }
        },
        "run_status": repo_target.run_status,
        "summary": {
            "counts": {
                "errors": repo_target.counts["errors"],
                "warnings": repo_target.counts["warnings"],
                "info": repo_target.counts["info"],
                "targets_total": targets_total,
                "targets_failed": targets_failed
            },
            "policy": {
                "strict": bool(fail_on_warning),
                "fail_on_warning": bool(fail_on_warning)
            }
        },
        "targets": [repo_target.to_json(), *[t.to_json() for t in casepack_targets]],
        "issues": [],  # optional flattened list; leaving empty avoids duplication
        "notes": "UMCP repository validation"
    }

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
            # Update repo target in result
            result["targets"][0] = repo_target.to_json()

    return result


# -----------------------------
# CLI
# -----------------------------
def _cmd_validate(args: argparse.Namespace) -> int:
    start = Path(args.path).resolve()
    repo_root = _find_repo_root(start)
    if repo_root is None:
        print("ERROR: Could not find repo root (no pyproject.toml found in parents).", file=sys.stderr)
        return 2

    # Determine strict mode: --strict flag OR --fail-on-warning (legacy)
    strict_mode = bool(getattr(args, 'strict', False)) or bool(args.fail_on_warning)
    
    result = _validate_repo(repo_root=repo_root, fail_on_warning=strict_mode)

    out_json = json.dumps(result, indent=2, sort_keys=False)
    
    # Compute sha256 of result
    result_hash = hashlib.sha256(out_json.encode('utf-8')).hexdigest()
    
    # Extract provenance details
    created_utc = result.get("created_utc", "unknown")
    git_commit = result.get("validator", {}).get("implementation", {}).get("git_commit", "unknown")
    python_version = result.get("validator", {}).get("implementation", {}).get("python_version", "unknown")
    error_count = result.get("summary", {}).get("counts", {}).get("errors", 0)
    warning_count = result.get("summary", {}).get("counts", {}).get("warnings", 0)
    
    # Generate governance summary with full provenance
    policy_mode = "strict" if strict_mode else "non-strict"
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
        
        if getattr(args, 'verbose', False):
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
            if getattr(args, 'verbose', False):
                for target in sorted(t for t in added if t):
                    print(f"    + {target}")
        if removed:
            print(f"  Removed from Receipt 1: {len(removed)}")
            if getattr(args, 'verbose', False):
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


def _compare_field(dict1: dict, dict2: dict, key: str, label: str) -> None:
    """Compare a single field between two dictionaries."""
    val1 = dict1.get(key)
    val2 = dict2.get(key)
    
    if val1 == val2:
        print(f"  {label}: {val1} (unchanged)")
    else:
        print(f"  {label}: {val1} → {val2}")


def _compare_dict_field(dict1: dict, dict2: dict, key: str, label: str) -> None:
    """Compare a nested field between two dictionaries."""
    val1 = dict1.get(key)
    val2 = dict2.get(key)
    
    if val1 == val2:
        print(f"  {label}: {val1} (unchanged)")
    else:
        print(f"  {label}: {val1} → {val2}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="umcp", description="UMCP contract-first validator CLI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate UMCP repo artifacts, CasePacks, schemas, and semantic rules")
    v.add_argument("path", nargs="?", default=".", help="Path inside repo (default: .)")
    v.add_argument("--out", default=None, help="Write validator result JSON to this file")
    v.add_argument("--strict", action="store_true", help="Enable strict mode: warnings become errors")
    v.add_argument("--fail-on-warning", action="store_true", help="(Legacy) Treat warnings as failing")
    v.set_defaults(func=_cmd_validate)

    r = sub.add_parser("run", help="Operational placeholder: validates the target")
    r.add_argument("path", nargs="?", default=".", help="Path inside repo (default: .)")
    r.add_argument("--out", default=None, help="Write validator result JSON to this file")
    r.add_argument("--strict", action="store_true", help="Enable strict mode: warnings become errors")
    r.add_argument("--fail-on-warning", action="store_true", help="(Legacy) Treat warnings as failing")
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("diff", help="Compare two validation receipts")
    d.add_argument("receipt1", help="Path to first receipt JSON file")
    d.add_argument("receipt2", help="Path to second receipt JSON file")
    d.add_argument("--verbose", "-v", action="store_true", help="Show detailed differences")
    d.set_defaults(func=_cmd_diff)

    return p


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
