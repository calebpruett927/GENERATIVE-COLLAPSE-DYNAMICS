from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pytest

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from jsonschema import Draft202012Validator  # type: ignore
except Exception as e:  # pragma: no cover
    Draft202012Validator = None  # type: ignore


RE_CK = re.compile(r"^c_[0-9]+$")
RE_OORK = re.compile(r"^oor_[0-9]+$")
RE_MISSK = re.compile(r"^miss_[0-9]+$")


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    schemas_dir: Path
    canon_anchors: Path
    contract: Path
    closures_registry: Path
    validator_rules: Path
    hello_world_dir: Path
    hello_manifest: Path
    hello_expected_dir: Path
    hello_psi_csv: Path
    hello_invariants_json: Path
    hello_ss1m_receipt_json: Path


@pytest.fixture(scope="session")
def repo_paths() -> RepoPaths:
    root = Path(__file__).resolve().parents[1]
    return RepoPaths(
        root=root,
        schemas_dir=root / "schemas",
        canon_anchors=root / "canon" / "anchors.yaml",
        contract=root / "contracts" / "UMA.INTSTACK.v1.yaml",
        closures_registry=root / "closures" / "registry.yaml",
        validator_rules=root / "validator_rules.yaml",
        hello_world_dir=root / "casepacks" / "hello_world",
        hello_manifest=root / "casepacks" / "hello_world" / "manifest.json",
        hello_expected_dir=root / "casepacks" / "hello_world" / "expected",
        hello_psi_csv=root / "casepacks" / "hello_world" / "expected" / "psi.csv",
        hello_invariants_json=root / "casepacks" / "hello_world" / "expected" / "invariants.json",
        hello_ss1m_receipt_json=root / "casepacks" / "hello_world" / "expected" / "ss1m_receipt.json",
    )


def require_file(path: Path) -> None:
    assert path.exists(), f"Missing required file: {path.as_posix()}"


def require_dir(path: Path) -> None:
    assert path.exists() and path.is_dir(), f"Missing required directory: {path.as_posix()}"


def load_text(path: Path) -> str:
    require_file(path)
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    require_file(path)
    return json.loads(load_text(path))


def load_yaml(path: Path) -> Any:
    require_file(path)
    assert yaml is not None, "PyYAML is required for these tests (pip install pyyaml)."
    return yaml.safe_load(load_text(path))


def ensure_jsonschema_available() -> None:
    assert Draft202012Validator is not None, "jsonschema is required (pip install jsonschema)."


def load_schema(repo_paths: RepoPaths, schema_filename: str) -> Dict[str, Any]:
    ensure_jsonschema_available()
    schema_path = repo_paths.schemas_dir / schema_filename
    require_file(schema_path)
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_instance(instance: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Returns list of error strings (empty if valid).
    """
    ensure_jsonschema_available()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.json_path)
    return [f"{e.json_path}: {e.message}" for e in errors]


def coerce_scalar(v: Any) -> Any:
    """
    Coerce CSV string scalars to bool/int/float when safe. Otherwise keep as-is.
    """
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v

    s = str(v).strip()
    if s == "":
        return None

    low = s.lower()
    if low in {"true", "t", "yes", "y"}:
        return True
    if low in {"false", "f", "no", "n"}:
        return False

    # int?
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            pass

    # float?
    if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except Exception:
            pass

    return s


def parse_csv_as_rows(path: Path) -> List[Dict[str, Any]]:
    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, f"No header row found in CSV: {path.as_posix()}"
        rows: List[Dict[str, Any]] = []
        for r in reader:
            rows.append({k: coerce_scalar(v) for k, v in r.items()})
        return rows


def infer_psi_format(rows: List[Dict[str, Any]]) -> str:
    """
    Infer psi trace format based on available keys.
    - long: has ('dim' and 'c') at least in first row
    - wide: has any c_<k> column in union of keys
    """
    if not rows:
        return "psi_trace_csv_long"

    keys_union = set()
    for r in rows:
        keys_union.update(r.keys())

    if "dim" in keys_union and "c" in keys_union:
        return "psi_trace_csv_long"

    if any(RE_CK.match(k) for k in keys_union):
        return "psi_trace_csv_wide"

    # fallback: treat as long (will fail schema if missing columns)
    return "psi_trace_csv_long"


def build_psi_doc(rows: List[Dict[str, Any]], fmt: str) -> Dict[str, Any]:
    """
    Build a JSON document conforming to schemas/trace.psi.schema.json from parsed CSV rows.
    """
    return {
        "schema": "schemas/trace.psi.schema.json",
        "format": fmt,
        "rows": rows
    }


def close(lhs: float, rhs: float, atol: float, rtol: float) -> bool:
    """
    Canonical closeness: abs(lhs-rhs) <= atol + rtol*abs(rhs)
    """
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return False
    return abs(lhs - rhs) <= (atol + rtol * abs(rhs))


def compute_expected_regime_label(
    omega: float,
    F: float,
    S: float,
    C: float,
    thresholds: Dict[str, Any]
) -> str:
    """
    Canonical expected label:
    - Collapse if omega >= omega_gte
    - Stable if omega < omega_lt AND F > F_gt AND S < S_lt AND C < C_lt
    - else Watch
    """
    collapse_omega_gte = thresholds["collapse"]["omega_gte"]
    stable = thresholds["stable"]

    if omega >= collapse_omega_gte:
        return "Collapse"

    if (omega < stable["omega_lt"]) and (F > stable["F_gt"]) and (S < stable["S_lt"]) and (C < stable["C_lt"]):
        return "Stable"

    return "Watch"


def load_rule_by_id(rules_doc: Dict[str, Any], rule_id: str) -> Dict[str, Any]:
    for r in rules_doc.get("rules", []):
        if r.get("id") == rule_id:
            return r
    raise KeyError(f"Rule id not found in validator_rules.yaml: {rule_id}")


def dot_get(obj: Any, dotpath: str) -> Any:
    cur = obj
    for part in dotpath.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur
