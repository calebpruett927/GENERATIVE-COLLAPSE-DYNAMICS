#!/usr/bin/env python3
"""Build a source-faithful Tableau bundle from the active GCD/UMCP repository."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPORT_ID = "SR.TABLEAU.DATASET.v0.2"


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    source_scope: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count


def _repo_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def _iter_runs(root: Path, include_archive: bool) -> Iterator[SourceRun]:
    for scope, parent in [
        ("current", root / "runs"),
        ("archive", root / "archive" / "runs"),
    ]:
        if scope == "archive" and not include_archive:
            continue
        if not parent.exists():
            continue
        for run_dir in sorted(path for path in parent.iterdir() if path.is_dir()):
            yield SourceRun(run_dir=run_dir, source_scope=scope)


def _run_key(source_scope: str, run_id: str) -> str:
    return f"{source_scope}:{run_id}"


def _finite_number(raw: str | None) -> float | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _return_class(raw: str | None) -> str:
    value = (raw or "").strip()
    upper = value.upper()
    if upper in {"INF_REC", "UNIDENTIFIABLE", "OOR"}:
        return upper
    if upper in {"INF", "+INF", "INFINITY", "+INFINITY"}:
        return "SOURCE_INFINITY_TOKEN"
    if upper in {"-INF", "-INFINITY"}:
        return "SOURCE_NEGATIVE_INFINITY_TOKEN"
    if _finite_number(value) is not None:
        return "NUMERIC"
    if not value:
        return "MISSING"
    return "SOURCE_TOKEN"


def _stringify(value: Any) -> tuple[str, str]:
    if value is None:
        return "", "null"
    if isinstance(value, bool):
        return ("true" if value else "false"), "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value), "number"
    if isinstance(value, str):
        return value, "string"
    return json.dumps(value, sort_keys=True, separators=(",", ":")), "json"


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value[key], child)
        return
    if isinstance(value, list):
        yield prefix, value
        return
    yield prefix, value


def _run_identity(root: Path, source: SourceRun) -> dict[str, Any]:
    manifest_path = source.run_dir / "manifest.json"
    frozen_path = source.run_dir / "config" / "frozen.json"
    manifest = _load_json(manifest_path)
    frozen = _load_json(frozen_path)
    run_id = str(manifest.get("run_id") or frozen.get("run_id") or source.run_dir.name)
    adapter = manifest.get("adapter") or frozen.get("adapter") or {}
    adapter_name = adapter.get("name", "") if isinstance(adapter, dict) else ""
    return {
        "run_key": _run_key(source.source_scope, run_id),
        "run_id": run_id,
        "source_scope": source.source_scope,
        "casepack_id": str(manifest.get("casepack_id") or frozen.get("casepack_id") or ""),
        "created_utc": manifest.get("created_utc", frozen.get("created_utc", "")),
        "contract_id": str(manifest.get("contract") or frozen.get("contract") or ""),
        "timezone": manifest.get("timezone", frozen.get("timezone", "")),
        "git_commit_declared": manifest.get("git_commit", frozen.get("git_commit", "")),
        "package_version_declared": manifest.get("package_version", frozen.get("package_version", "")),
        "pipeline": manifest.get("pipeline", frozen.get("pipeline", "")),
        "adapter_name": adapter_name,
        "semantic_status": "SOURCE_DECLARED_UNASSESSED",
        "authority_status": "NO_PROMOTION_BY_EXPORT",
        "evidence_origin": "DERIVED_RESTATEMENT",
        "manifest_path": str(manifest_path.relative_to(root)) if manifest_path.exists() else "",
        "frozen_path": str(frozen_path.relative_to(root)) if frozen_path.exists() else "",
    }


def _kernel_path(source: SourceRun) -> Path:
    gate = source.run_dir / "kernel" / "kernel_gate.csv"
    if gate.exists():
        return gate
    return source.run_dir / "kernel" / "kernel.csv"


def _kernel_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    core = {"t", "omega", "F", "S", "C", "tau_R", "IC", "kappa", "kappa_cum", "regime", "regime_gate"}
    for source in sources:
        identity = _run_identity(root, source)
        path = _kernel_path(source)
        for row_index, row in enumerate(_read_csv(path)):
            tau_raw = row.get("tau_R", "")
            tau_number = _finite_number(tau_raw)
            payload = {key: value for key, value in row.items() if key not in core}
            yield {
                **identity,
                "row_index": row_index,
                "t": row.get("t", ""),
                "omega": row.get("omega", ""),
                "F": row.get("F", ""),
                "S": row.get("S", ""),
                "C": row.get("C", ""),
                "tau_R_raw": tau_raw,
                "tau_R_numeric": "" if tau_number is None else repr(tau_number),
                "tau_R_token_class": _return_class(tau_raw),
                "IC": row.get("IC", ""),
                "kappa": row.get("kappa", ""),
                "kappa_cum": row.get("kappa_cum", ""),
                "source_regime": row.get("regime_gate", row.get("regime", "")),
                "source_payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                "source_path": str(path.relative_to(root)) if path.exists() else "",
            }


def _return_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        candidates = [
            source.run_dir / "tables" / "tauR_series.csv",
            source.run_dir / "tables" / "taur_series.csv",
            source.run_dir / "derived" / "tauR_series.csv",
            source.run_dir / "derived" / "taur_series.csv",
            source.run_dir / "casepacks" / "KIN.CP.SHM" / "tables" / "tauR_series.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        for row_index, row in enumerate(_read_csv(path)):
            raw = row.get("tau_R", "")
            number = _finite_number(raw)
            yield {
                **identity,
                "row_index": row_index,
                "t": row.get("t", ""),
                "tau_R_raw": raw,
                "tau_R_numeric": "" if number is None else repr(number),
                "tau_R_token_class": _return_class(raw),
                "source_path": str(path.relative_to(root)) if path.exists() else "",
            }


def _missingness_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        for kind, filename, flag_name in [
            ("CENSOR", "censor.csv", "censored"),
            ("OOR", "oor.csv", "oor"),
        ]:
            path = source.run_dir / "logs" / filename
            for row_index, row in enumerate(_read_csv(path)):
                yield {
                    **identity,
                    "row_index": row_index,
                    "t": row.get("t", ""),
                    "channel": row.get("channel", ""),
                    "missingness_kind": kind,
                    "flag_raw": row.get(flag_name, ""),
                    "rate_raw": row.get("rate", ""),
                    "source_path": str(path.relative_to(root)) if path.exists() else "",
                }


def _seam_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        path = source.run_dir / "seams" / "seam_ledger.csv"
        for row_index, row in enumerate(_read_csv(path)):
            yield {
                **identity,
                "row_index": row_index,
                "seam_id": row.get("seam_id", str(row_index)),
                "source_type": row.get("type", ""),
                "t_raw": row.get("t", row.get("t_s", "")),
                "payload_json": json.dumps(row, sort_keys=True, separators=(",", ":")),
                "source_path": str(path.relative_to(root)) if path.exists() else "",
                "weld_status": "NOT_DERIVED_BY_EXPORT",
            }


def _contract_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        for source_kind, path in [
            ("manifest", source.run_dir / "manifest.json"),
            ("frozen_contract", source.run_dir / "config" / "frozen.json"),
        ]:
            payload = _load_json(path)
            for field_path, value in _flatten(payload):
                value_text, value_type = _stringify(value)
                yield {
                    **identity,
                    "source_kind": source_kind,
                    "field_path": field_path,
                    "value": value_text,
                    "value_type": value_type,
                    "source_path": str(path.relative_to(root)) if path.exists() else "",
                }


def _receipt_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        path = source.run_dir / "receipts" / "weld_receipt.json"
        payload = _load_json(path)
        for field_path, value in _flatten(payload):
            value_text, value_type = _stringify(value)
            yield {
                **identity,
                "field_path": field_path,
                "value": value_text,
                "value_type": value_type,
                "source_path": str(path.relative_to(root)) if path.exists() else "",
                "receipt_authority_status": "SOURCE_RECEIPT_ONLY",
            }


def _casepack_rows(root: Path) -> Iterator[dict[str, Any]]:
    casepacks = root / "casepacks"
    if not casepacks.exists():
        return
    for path in sorted(casepacks.rglob("manifest.json")):
        manifest = _load_json(path)
        cp = manifest.get("casepack", {}) if isinstance(manifest.get("casepack"), dict) else {}
        refs = manifest.get("refs", {}) if isinstance(manifest.get("refs"), dict) else {}
        contract = refs.get("contract", {}) if isinstance(refs.get("contract"), dict) else {}
        closures = refs.get("closures_registry", {}) if isinstance(refs.get("closures_registry"), dict) else {}
        yield {
            "casepack_id": str(cp.get("id") or path.parent.name),
            "version": cp.get("version", ""),
            "title": cp.get("title", ""),
            "description": cp.get("description", ""),
            "created_utc": cp.get("created_utc", ""),
            "timezone": cp.get("timezone", ""),
            "contract_id": contract.get("id", ""),
            "closure_registry_id": closures.get("id", ""),
            "manifest_path": str(path.relative_to(root)),
            "semantic_status": "SOURCE_DECLARED_UNASSESSED",
            "authority_status": "NO_PROMOTION_BY_EXPORT",
        }


def _float_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        number = _finite_number(row.get(field))
        if number is not None:
            values.append(number)
    return values


def _mean(values: list[float]) -> str:
    return "" if not values else repr(sum(values) / len(values))


def _run_summary_rows(root: Path, sources: list[SourceRun]) -> Iterator[dict[str, Any]]:
    for source in sources:
        identity = _run_identity(root, source)
        kernel_path = _kernel_path(source)
        kernel = _read_csv(kernel_path)
        latest = kernel[-1] if kernel else {}
        seams = _read_csv(source.run_dir / "seams" / "seam_ledger.csv")
        censor = _read_csv(source.run_dir / "logs" / "censor.csv")
        oor = _read_csv(source.run_dir / "logs" / "oor.csv")
        latest_tau = latest.get("tau_R", "")
        latest_tau_num = _finite_number(latest_tau)
        yield {
            **identity,
            "kernel_rows": len(kernel),
            "seam_count": len(seams),
            "censor_event_count": len(censor),
            "oor_event_count": len(oor),
            "latest_t": latest.get("t", ""),
            "latest_F": latest.get("F", ""),
            "latest_omega": latest.get("omega", ""),
            "latest_S": latest.get("S", ""),
            "latest_C": latest.get("C", ""),
            "latest_IC": latest.get("IC", ""),
            "latest_kappa": latest.get("kappa", ""),
            "latest_tau_R_raw": latest_tau,
            "latest_tau_R_numeric": "" if latest_tau_num is None else repr(latest_tau_num),
            "latest_tau_R_token_class": _return_class(latest_tau),
            "latest_source_regime": latest.get("regime_gate", latest.get("regime", "")),
            "mean_F": _mean(_float_values(kernel, "F")),
            "mean_omega": _mean(_float_values(kernel, "omega")),
            "mean_S": _mean(_float_values(kernel, "S")),
            "mean_C": _mean(_float_values(kernel, "C")),
            "mean_IC": _mean(_float_values(kernel, "IC")),
            "summary_origin": "DERIVED_RESTATEMENT",
        }


def export_dataset(root: Path, output_dir: Path, include_archive: bool = True) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [source for source in _iter_runs(root, include_archive) if source.run_dir.resolve() != output_dir]
    run_rows = [_run_identity(root, source) for source in sources]

    identity_fields = [
        "run_key",
        "run_id",
        "source_scope",
        "casepack_id",
        "created_utc",
        "contract_id",
        "timezone",
        "git_commit_declared",
        "package_version_declared",
        "pipeline",
        "adapter_name",
        "semantic_status",
        "authority_status",
        "evidence_origin",
    ]

    counts: dict[str, int] = {}
    counts["runs"] = _write_csv(
        output_dir / "runs.csv",
        identity_fields + ["manifest_path", "frozen_path"],
        run_rows,
    )
    counts["kernel_observations"] = _write_csv(
        output_dir / "kernel_observations.csv",
        identity_fields
        + [
            "row_index",
            "t",
            "omega",
            "F",
            "S",
            "C",
            "tau_R_raw",
            "tau_R_numeric",
            "tau_R_token_class",
            "IC",
            "kappa",
            "kappa_cum",
            "source_regime",
            "source_payload_json",
            "source_path",
        ],
        _kernel_rows(root, sources),
    )
    counts["return_observations"] = _write_csv(
        output_dir / "return_observations.csv",
        identity_fields
        + [
            "row_index",
            "t",
            "tau_R_raw",
            "tau_R_numeric",
            "tau_R_token_class",
            "source_path",
        ],
        _return_rows(root, sources),
    )
    counts["missingness_events"] = _write_csv(
        output_dir / "missingness_events.csv",
        identity_fields
        + [
            "row_index",
            "t",
            "channel",
            "missingness_kind",
            "flag_raw",
            "rate_raw",
            "source_path",
        ],
        _missingness_rows(root, sources),
    )
    counts["seam_events"] = _write_csv(
        output_dir / "seam_events.csv",
        identity_fields
        + [
            "row_index",
            "seam_id",
            "source_type",
            "t_raw",
            "payload_json",
            "source_path",
            "weld_status",
        ],
        _seam_rows(root, sources),
    )
    counts["contract_fields"] = _write_csv(
        output_dir / "contract_fields.csv",
        identity_fields
        + ["source_kind", "field_path", "value", "value_type", "source_path"],
        _contract_rows(root, sources),
    )
    counts["receipt_fields"] = _write_csv(
        output_dir / "receipt_fields.csv",
        identity_fields
        + [
            "field_path",
            "value",
            "value_type",
            "source_path",
            "receipt_authority_status",
        ],
        _receipt_rows(root, sources),
    )
    counts["casepacks"] = _write_csv(
        output_dir / "casepacks.csv",
        [
            "casepack_id",
            "version",
            "title",
            "description",
            "created_utc",
            "timezone",
            "contract_id",
            "closure_registry_id",
            "manifest_path",
            "semantic_status",
            "authority_status",
        ],
        _casepack_rows(root),
    )
    counts["dashboard_run_summary"] = _write_csv(
        output_dir / "dashboard_run_summary.csv",
        identity_fields
        + [
            "kernel_rows",
            "seam_count",
            "censor_event_count",
            "oor_event_count",
            "latest_t",
            "latest_F",
            "latest_omega",
            "latest_S",
            "latest_C",
            "latest_IC",
            "latest_kappa",
            "latest_tau_R_raw",
            "latest_tau_R_numeric",
            "latest_tau_R_token_class",
            "latest_source_regime",
            "mean_F",
            "mean_omega",
            "mean_S",
            "mean_C",
            "mean_IC",
            "summary_origin",
        ],
        _run_summary_rows(root, sources),
    )

    manifest = {
        "dataset_id": EXPORT_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _repo_commit(root),
        "source_root": str(root),
        "include_archive": include_archive,
        "evidence_origin": "DERIVED_RESTATEMENT",
        "semantic_boundary": (
            "Source rows remain under their declared local semantics. The export does not "
            "assert current Tier-1/Tier-0 compatibility or promote repository presence into authority."
        ),
        "return_boundary": (
            "tau_R is preserved verbatim. Canonical tokens, source infinity tokens, finite numeric "
            "values, missing values, and other source tokens remain distinguishable."
        ),
        "missingness_boundary": "Missing values are not imputed as zero.",
        "stance_boundary": "The exporter derives no final stance and no weld verdict.",
        "table_counts": counts,
        "tables": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    manifest = export_dataset(args.repo_root, args.output_dir, include_archive=not args.no_archive)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "table_counts": manifest["table_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
