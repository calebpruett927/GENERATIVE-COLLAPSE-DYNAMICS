from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from .conftest import (
    RepoPaths,
    close,
    compute_expected_regime_label,
    dot_get,
    infer_psi_format,
    load_rule_by_id,
    load_json,
    load_yaml,
    parse_csv_as_rows,
    repo_paths,
    require_file,
)

RE_CK = re.compile(r"^c_[0-9]+$")


def test_validator_rules_file_exists_and_conforms(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.validator_rules)
    rules_doc = load_yaml(repo_paths.validator_rules)
    schema_path = repo_paths.root / "schemas" / "validator.rules.schema.json"
    assert schema_path.exists(), f"Missing rules schema: {schema_path.as_posix()}"

    # Validate rules_doc against schema
    from jsonschema import Draft202012Validator  # type: ignore
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    errors = sorted(v.iter_errors(rules_doc), key=lambda e: e.json_path)
    assert not errors, "validator_rules.yaml failed schema validation:\n" + "\n".join(f"{e.json_path}: {e.message}" for e in errors)


def test_E101_wide_psi_has_at_least_one_coordinate_column_when_wide(repo_paths: RepoPaths) -> None:
    """
    Implements the E101 semantic check:
      If psi trace is wide, require at least one c_<k> column.
    """
    rules_doc = load_yaml(repo_paths.validator_rules)
    rule = load_rule_by_id(rules_doc, "E101")

    rows = parse_csv_as_rows(repo_paths.hello_psi_csv)
    fmt = infer_psi_format(rows)
    if fmt != "psi_trace_csv_wide":
        return  # Not applicable to long/JSON formats.

    keys_union = set()
    for r in rows:
        keys_union.update(r.keys())

    matches = sum(1 for k in keys_union if RE_CK.match(k))
    assert matches >= rule["check"]["min_matches"], (
        f"E101 failed: expected >= {rule['check']['min_matches']} coordinate columns matching {rule['check']['pattern']}, "
        f"observed {matches}."
    )


def test_W201_F_equals_one_minus_omega(repo_paths: RepoPaths) -> None:
    """
    Implements W201 semantic check with canonical closeness definition.
    """
    rules_doc = load_yaml(repo_paths.validator_rules)
    rule = load_rule_by_id(rules_doc, "W201")
    inv = load_json(repo_paths.hello_invariants_json)

    atol = float(rule.get("atol", 1.0e-9))
    rtol = float(rule.get("rtol", 0.0))

    omega_path = rule["check"]["fields"]["omega"]
    F_path = rule["check"]["fields"]["F"]

    rows: List[Dict[str, Any]] = inv["rows"]
    failures = []

    for i, row in enumerate(rows):
        omega = dot_get(row, omega_path)
        F = dot_get(row, F_path)

        if omega is None or F is None or not isinstance(omega, (int, float)) or not isinstance(F, (int, float)):
            failures.append((i, "missing/non-numeric", omega, F))
            continue

        omega_f = float(omega)
        F_f = float(F)
        if not (math.isfinite(omega_f) and math.isfinite(F_f)):
            failures.append((i, "non-finite", omega_f, F_f))
            continue

        rhs = 1.0 - omega_f
        if not close(F_f, rhs, atol=atol, rtol=rtol):
            failures.append((i, "mismatch", omega_f, F_f, rhs, F_f - rhs))

    assert not failures, "W201 failed (F ≈ 1 − ω) on rows:\n" + "\n".join(map(str, failures))


def test_W202_IC_equals_exp_kappa(repo_paths: RepoPaths) -> None:
    """
    Implements W202 semantic check with canonical closeness definition.
    """
    rules_doc = load_yaml(repo_paths.validator_rules)
    rule = load_rule_by_id(rules_doc, "W202")
    inv = load_json(repo_paths.hello_invariants_json)

    atol = float(rule.get("atol", 1.0e-9))
    rtol = float(rule.get("rtol", 1.0e-9))

    IC_path = rule["check"]["fields"]["IC"]
    kappa_path = rule["check"]["fields"]["kappa"]

    rows: List[Dict[str, Any]] = inv["rows"]
    failures = []

    for i, row in enumerate(rows):
        IC = dot_get(row, IC_path)
        kappa = dot_get(row, kappa_path)

        if IC is None or kappa is None or not isinstance(IC, (int, float)) or not isinstance(kappa, (int, float)):
            failures.append((i, "missing/non-numeric", IC, kappa))
            continue

        IC_f = float(IC)
        kappa_f = float(kappa)
        if not (math.isfinite(IC_f) and math.isfinite(kappa_f)):
            failures.append((i, "non-finite", IC_f, kappa_f))
            continue

        rhs = math.exp(kappa_f)
        if not math.isfinite(rhs):
            failures.append((i, "exp non-finite", kappa_f, rhs))
            continue

        if not close(IC_f, rhs, atol=atol, rtol=rtol):
            failures.append((i, "mismatch", IC_f, kappa_f, rhs, IC_f - rhs))

    assert not failures, "W202 failed (IC ≈ exp(κ)) on rows:\n" + "\n".join(map(str, failures))


def test_W301_regime_label_consistency_with_canon(repo_paths: RepoPaths) -> None:
    """
    Implements W301 semantic check:
      - compute expected label from canon thresholds
      - compare to regime.label when present; for publication-grade expected/invariants.json it SHOULD be present
    """
    rules_doc = load_yaml(repo_paths.validator_rules)
    rule = load_rule_by_id(rules_doc, "W301")
    inv = load_json(repo_paths.hello_invariants_json)
    canon = load_yaml(repo_paths.canon_anchors)

    thresholds = canon["umcp_canon"]["regimes"]

    omega_path = rule["check"]["fields"]["omega"]
    F_path = rule["check"]["fields"]["F"]
    S_path = rule["check"]["fields"]["S"]
    C_path = rule["check"]["fields"]["C"]

    label_path = rule["check"]["fields"]["regime_label"]
    crit_path = rule["check"]["fields"]["critical_overlay"]
    icmin_path = rule["check"]["fields"]["IC_min"]

    policies = rule["check"]["policies"]
    on_missing_regime = policies.get("on_missing_regime", "warn")
    on_missing_icmin = policies.get("on_missing_IC_min", "skip")

    rows: List[Dict[str, Any]] = inv["rows"]
    failures = []

    for i, row in enumerate(rows):
        omega = dot_get(row, omega_path)
        F = dot_get(row, F_path)
        S = dot_get(row, S_path)
        C = dot_get(row, C_path)

        if not all(isinstance(x, (int, float)) for x in [omega, F, S, C]):
            # cannot compute expected label
            if on_missing_regime != "skip":
                failures.append((i, "cannot compute expected label", omega, F, S, C))
            continue

        exp_label = compute_expected_regime_label(float(omega), float(F), float(S), float(C), thresholds)

        provided_label = dot_get(row, label_path)
        if provided_label is None:
            # For a published expected/invariants.json, require this to be present even if policy is warn.
            failures.append((i, "missing regime.label", exp_label))
        else:
            if provided_label not in {"Stable", "Watch", "Collapse"}:
                failures.append((i, "invalid label", provided_label))
            elif provided_label != exp_label:
                failures.append((i, "label mismatch", exp_label, provided_label, float(omega), float(F), float(S), float(C)))

        # Critical overlay check is optional and only checkable if IC_min exists and numeric
        IC_min = dot_get(row, icmin_path)
        crit = dot_get(row, crit_path)

        if IC_min is None or not isinstance(IC_min, (int, float)) or not math.isfinite(float(IC_min)):
            if on_missing_icmin != "skip" and crit is not None:
                failures.append((i, "critical_overlay present but IC_min missing/non-numeric", IC_min, crit))
        else:
            expected_crit = float(IC_min) < float(thresholds["collapse"]["critical_overlay"]["min_IC_lt"])
            if crit is not None:
                if not isinstance(crit, bool):
                    failures.append((i, "critical_overlay not boolean", crit))
                elif crit != expected_crit:
                    failures.append((i, "critical_overlay mismatch", expected_crit, crit, float(IC_min)))

    assert not failures, "W301 regime label consistency failed:\n" + "\n".join(map(str, failures))
