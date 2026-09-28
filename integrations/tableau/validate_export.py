#!/usr/bin/env python3
"""Validate the Structura Reditus Tableau export contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REQUIRED = {
    "runs.csv",
    "kernel_observations.csv",
    "return_observations.csv",
    "missingness_events.csv",
    "seam_events.csv",
    "contract_fields.csv",
    "receipt_fields.csv",
    "casepacks.csv",
    "dashboard_run_summary.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate(bundle: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = bundle / "dataset_manifest.json"
    if not manifest_path.exists():
        return ["dataset_manifest.json is missing"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evidence_origin") != "DERIVED_RESTATEMENT":
        errors.append("evidence_origin must remain DERIVED_RESTATEMENT")
    if manifest.get("stance_boundary") != "The exporter derives no final stance and no weld verdict.":
        errors.append("stance/weld boundary changed")

    available = {path.name for path in bundle.glob("*.csv")}
    missing = sorted(REQUIRED - available)
    if missing:
        errors.append("missing required tables: " + ", ".join(missing))
        return errors

    runs = _read_csv(bundle / "runs.csv")
    keys = [row["run_key"] for row in runs]
    if len(keys) != len(set(keys)):
        errors.append("runs.csv contains duplicate run_key values")
    for row in runs:
        if row.get("authority_status") != "NO_PROMOTION_BY_EXPORT":
            errors.append(f"{row.get('run_key')}: authority was promoted")
        if row.get("semantic_status") != "SOURCE_DECLARED_UNASSESSED":
            errors.append(f"{row.get('run_key')}: semantic status was promoted")

    for table_name in ["kernel_observations.csv", "return_observations.csv"]:
        for line, row in enumerate(_read_csv(bundle / table_name), start=2):
            raw = row.get("tau_R_raw", "")
            numeric = row.get("tau_R_numeric", "")
            cls = row.get("tau_R_token_class", "")
            upper = raw.strip().upper()
            if upper in {"INF_REC", "UNIDENTIFIABLE", "OOR"}:
                if numeric:
                    errors.append(f"{table_name}:{line}: canonical return token coerced to numeric")
                if cls != upper:
                    errors.append(f"{table_name}:{line}: canonical return token class changed")
            if upper in {"INF", "+INF", "INFINITY", "+INFINITY"}:
                if numeric:
                    errors.append(f"{table_name}:{line}: source infinity token coerced to numeric")
                if cls != "SOURCE_INFINITY_TOKEN":
                    errors.append(f"{table_name}:{line}: source infinity token class changed")
            if numeric and not _finite(numeric):
                errors.append(f"{table_name}:{line}: numeric helper is non-finite")

    for line, row in enumerate(_read_csv(bundle / "seam_events.csv"), start=2):
        if row.get("weld_status") != "NOT_DERIVED_BY_EXPORT":
            errors.append(f"seam_events.csv:{line}: export asserted weld")

    for line, row in enumerate(_read_csv(bundle / "dashboard_run_summary.csv"), start=2):
        if row.get("summary_origin") != "DERIVED_RESTATEMENT":
            errors.append(f"dashboard_run_summary.csv:{line}: summary origin changed")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    errors = validate(args.bundle)
    if errors:
        print("NONCONFORMANT export bundle")
        for error in errors:
            print(f"- {error}")
        return 1
    print("CONFORMANT export bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
