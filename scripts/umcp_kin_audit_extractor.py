#!/usr/bin/env python3
"""
UMCP.KIN audit extractor

Run from repo root. It will:
- discover run folders (default: ./runs/* or ./runs/**)
- read frozen config + manifest + kernel + psi + logs + tauR series
- compute summary stats for {omega,F,S,C,tau_R,IC,kappa}
- for ballistic: extract seam ledger + weld receipt rows
- output a single JSON report you can paste into ChatGPT

No values are invented: missing items become null and are listed in "gaps".
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float_series(s: pd.Series) -> pd.Series:
    """Convert series to numeric, coercing strings like 'inf', 'Infinity', 'nan'."""
    return pd.to_numeric(s, errors="coerce")


def is_finite_array(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def quantiles(a: np.ndarray, qs: tuple[float, ...] = (0.25, 0.5, 0.75, 0.05, 0.95)) -> dict[str, float]:
    if a.size == 0:
        return {"q25": math.nan, "median": math.nan, "q75": math.nan, "p05": math.nan, "p95": math.nan}
    q25, med, q75, p05, p95 = np.quantile(a, [0.25, 0.5, 0.75, 0.05, 0.95])
    return {"q25": float(q25), "median": float(med), "q75": float(q75), "p05": float(p05), "p95": float(p95)}


def stat_block(a: np.ndarray) -> dict[str, Any]:
    if a.size == 0:
        return {"n": 0, "min": None, "max": None, "median": None, "q25": None, "q75": None, "iqr": None}
    q = quantiles(a, qs=(0.25, 0.5, 0.75, 0.05, 0.95))
    mn, mx = float(np.min(a)), float(np.max(a))
    iqr = q["q75"] - q["q25"]
    return {
        "n": int(a.size),
        "min": mn,
        "max": mx,
        "median": q["median"],
        "q25": q["q25"],
        "q75": q["q75"],
        "iqr": float(iqr),
        "p05": q["p05"],
        "p95": q["p95"],
    }


def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def discover_run_dirs(repo_root: Path) -> list[Path]:
    runs_root = repo_root / "runs"
    if not runs_root.exists():
        return []
    # Look for runs/**/config/frozen.json
    frozen_files = list(runs_root.glob("**/config/frozen.json"))
    return sorted({p.parent.parent for p in frozen_files})


def normalize_columns(cols: list[str]) -> dict[str, str]:
    """
    Heuristic mapping from your CSV headers to canonical names we want.
    We never guess values, only names.
    """
    # Lowercase normalized
    lower = {c: c.strip().lower() for c in cols}

    def pick(candidates: list[str]) -> str | None:
        for cand in candidates:
            for orig, low in lower.items():
                if low == cand:
                    return orig
        return None

    mapping: dict[str, str] = {}
    # time
    t = pick(["t", "time", "timestamp"])
    if t:
        mapping["t"] = t

    mapping_candidates = {
        "omega": ["omega", "komega", "ω"],
        "F": ["f", "kf"],
        "S": ["s", "ks"],
        "C": ["c", "kc"],
        "tau_R": ["tau_r", "taur", "ktau_r", "ktaur", "tau r", "tau"],
        "IC": ["ic", "kic", "integrity"],
        "kappa": ["kappa", "kkappa", "κ", "log_integrity", "logintegrity"],
    }

    for canon, cands in mapping_candidates.items():
        found = pick(cands)
        if found:
            mapping[canon] = found

    return mapping


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # Try comma, then fallback to tab
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return None


def extract_log_rates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Flexible: supports either
      - row-per-sample with boolean columns
      - row-per-channel summary with rate fields
    """
    out: dict[str, Any] = {"overall": None, "by_channel": {}}
    if df.empty:
        return out

    cols = [c.lower() for c in df.columns]

    # Common boolean flags
    bool_candidates = ["clipped", "is_clipped", "oor", "is_oor", "flag", "flagged", "censored", "is_censored"]
    channel_candidates = ["channel", "chan", "id", "name"]

    # Case A: summary rows with explicit rate
    for rate_col in ["rate", "clipping_rate", "oor_rate", "percent", "pct"]:
        if rate_col in cols:
            rate_name = df.columns[cols.index(rate_col)]
            # If channel present, map by channel; else overall mean
            ch_col = next((df.columns[cols.index(cc)] for cc in channel_candidates if cc in cols), None)
            if ch_col:
                for _, r in df.iterrows():
                    ch = str(r.get(ch_col))
                    val = r.get(rate_name)
                    if pd.notna(val):
                        out["by_channel"][ch] = float(val)
                # overall as mean of channels if not present
                if out["by_channel"]:
                    out["overall"] = float(np.mean(list(out["by_channel"].values())))
            else:
                vals = pd.to_numeric(df[rate_name], errors="coerce").dropna()
                if len(vals):
                    out["overall"] = float(vals.mean())
            return out

    # Case B: per-sample boolean
    flag_col = next((df.columns[cols.index(fc)] for fc in bool_candidates if fc in cols), None)
    ch_col = next((df.columns[cols.index(cc)] for cc in channel_candidates if cc in cols), None)

    if flag_col:
        flags = pd.to_numeric(df[flag_col], errors="coerce")
        flags = flags.dropna()
        if len(flags):
            out["overall"] = float(flags.mean())

        if ch_col:
            for ch, g in df.groupby(ch_col):
                f = pd.to_numeric(g[flag_col], errors="coerce").dropna()
                if len(f):
                    out["by_channel"][str(ch)] = float(f.mean())

    return out


# ----------------------------
# Main extractor
# ----------------------------


def extract_run(run_dir: Path) -> dict[str, Any]:
    gaps: list[str] = []

    frozen = read_json(run_dir / "config" / "frozen.json")
    if frozen is None:
        gaps.append("config/frozen.json missing")
        frozen = {}

    manifest = read_json(run_dir / "manifest.json")
    if manifest is None:
        gaps.append("manifest.json missing")
        manifest = {}

    kernel_path = run_dir / "kernel" / "kernel.csv"
    psi_path = run_dir / "derived" / "psi.csv"

    kernel_df = load_csv(kernel_path)
    if kernel_df is None:
        gaps.append(f"kernel missing or unreadable: {kernel_path}")

    psi_df = load_csv(psi_path)
    if psi_df is None:
        gaps.append(f"psi missing or unreadable: {psi_path}")

    # tauR series location (try both patterns)
    tauR_path = find_first_existing(
        [
            run_dir / "tables" / "tauR_series.csv",
            run_dir / "tables" / "taur_series.csv",
            run_dir / "derived" / "tauR_series.csv",
            run_dir / "derived" / "taur_series.csv",
            run_dir / "casepacks" / "KIN.CP.SHM" / "tables" / "tauR_series.csv",  # legacy fallback
        ]
    )
    tauR_df = load_csv(tauR_path) if tauR_path else None
    if tauR_df is None:
        gaps.append("tauR_series.csv missing/unreadable (checked run-local tables/derived)")

    # logs
    oor_df = load_csv(run_dir / "logs" / "oor.csv")
    if oor_df is None:
        gaps.append("logs/oor.csv missing/unreadable")

    censor_df = load_csv(run_dir / "logs" / "censor.csv")
    if censor_df is None:
        gaps.append("logs/censor.csv missing/unreadable")

    # seams + weld
    seam_df = load_csv(run_dir / "seams" / "seam_ledger.csv")
    weld_csv = run_dir / "receipts" / "weld_receipt.csv"
    weld_json = run_dir / "receipts" / "weld_receipt.json"
    weld_df = load_csv(weld_csv)
    weld_j = read_json(weld_json)

    # Identity block
    casepack_id = frozen.get("casepack_id") or frozen.get("casepackId")
    run_id = frozen.get("run_id") or frozen.get("runId") or run_dir.name

    identity = {
        "casepack_id": casepack_id,
        "run_id": run_id,
        "created_utc": manifest.get("created_utc"),
        "timezone": frozen.get("timezone") or "America/Chicago",
        "code_provenance": {
            "git_commit": frozen.get("git_commit") or frozen.get("gitCommit"),
            "package_version": frozen.get("package_version") or frozen.get("packageVersion"),
        },
        "contract": frozen.get("contract"),
        "pipeline": frozen.get("pipeline"),
        "adapter": frozen.get("adapter"),
        "return": frozen.get("return"),
        "weights": frozen.get("weights"),
        "closures": frozen.get("closures"),
        # Canon-final additions (v5+)
        "frozen_scales": frozen.get("frozen_scales"),
        "identity_validation": frozen.get("identity_validation"),
        "tier1_identities": frozen.get("tier1_identities"),
        "IC_pass_rate": frozen.get("IC_pass_rate"),
    }

    # Hashes / artifacts
    artifacts = []
    if isinstance(manifest.get("artifacts"), list):
        for a in manifest["artifacts"]:
            artifacts.append(
                {
                    "path": a.get("path"),
                    "sha256": a.get("sha256"),
                    "role": a.get("role"),
                    "bytes": a.get("bytes"),
                }
            )

    # Column maps
    columns: dict[str, Any] = {}
    kernel_map = {}
    if kernel_df is not None:
        columns["kernel"] = list(kernel_df.columns)
        kernel_map = normalize_columns(list(kernel_df.columns))
    else:
        columns["kernel"] = None

    if psi_df is not None:
        columns["psi"] = list(psi_df.columns)
    else:
        columns["psi"] = None

    if tauR_df is not None:
        columns["tauR_series"] = list(tauR_df.columns)
    else:
        columns["tauR_series"] = None

    # Summary stats from kernel
    kernel_stats: dict[str, Any] = {}
    tauR_coverage: dict[str, Any] = {"finite_coverage_percent": None, "censoring_percent": None}

    if kernel_df is not None and kernel_map:
        # Stats for invariants
        for inv in ["omega", "F", "S", "C", "tau_R", "IC", "kappa"]:
            col = kernel_map.get(inv)
            if not col:
                kernel_stats[inv] = None
                gaps.append(f"kernel column missing for {inv} (could not map)")
                continue
            arr = safe_float_series(kernel_df[col]).to_numpy()
            arr = arr[is_finite_array(arr)]
            kernel_stats[inv] = stat_block(arr)

        # tau_R coverage from kernel tau_R if present
        tr_col = kernel_map.get("tau_R")
        if tr_col:
            raw = safe_float_series(kernel_df[tr_col]).to_numpy()
            finite_mask = is_finite_array(raw)
            n = raw.size
            if n > 0:
                finite_pct = 100.0 * float(np.sum(finite_mask)) / float(n)
                tauR_coverage["finite_coverage_percent"] = finite_pct
                tauR_coverage["censoring_percent"] = 100.0 - finite_pct

    # tau_R distribution stats from tauR_series.csv if present
    tauR_dist_stats = None
    if tauR_df is not None and not tauR_df.empty:
        # try to find tau_R column
        lower = {c.lower(): c for c in tauR_df.columns}
        tr_col = lower.get("tau_r") or lower.get("taur") or lower.get("ktau_r") or lower.get("tau")
        if tr_col:
            raw = safe_float_series(tauR_df[tr_col]).to_numpy()
            finite = raw[is_finite_array(raw)]
            tauR_dist_stats = stat_block(finite)
            # If kernel coverage missing, compute from this file too
            if tauR_coverage["finite_coverage_percent"] is None and raw.size > 0:
                tauR_coverage["finite_coverage_percent"] = 100.0 * float(np.sum(is_finite_array(raw))) / float(raw.size)
                tauR_coverage["censoring_percent"] = 100.0 - tauR_coverage["finite_coverage_percent"]
        else:
            gaps.append("tauR_series.csv: could not find tau_R column")

    # Logs
    oor_rates = extract_log_rates(oor_df) if oor_df is not None else {"overall": None, "by_channel": {}}
    censor_rates = extract_log_rates(censor_df) if censor_df is not None else {"overall": None, "by_channel": {}}

    # Ballistic seam/weld extraction (if present)
    seam_row = None
    if seam_df is not None and not seam_df.empty and "seam_id" in seam_df.columns:
        # Try to find KIN.SEAM.IMP.001 first, else first IMP row
        hit = seam_df[seam_df["seam_id"].astype(str) == "KIN.SEAM.IMP.001"]
        if hit.empty:
            hit = seam_df[seam_df["seam_id"].astype(str).str.contains("IMP", na=False)]
        if not hit.empty:
            seam_row = hit.iloc[0].to_dict()

    weld_row = None
    if weld_df is not None and not weld_df.empty:
        # Try to find seam id row if column exists
        if "seam_id" in weld_df.columns:
            hit = weld_df[weld_df["seam_id"].astype(str) == "KIN.SEAM.IMP.001"]
            if hit.empty:
                hit = weld_df
            weld_row = hit.iloc[0].to_dict()
    elif weld_j is not None:
        # Keep whatever JSON structure you use; include raw + attempt common keys
        weld_row = weld_j

    report = {
        "identity": identity,
        "hashes": {
            "frozen_json": str(run_dir / "config" / "frozen.json"),
            "manifest_json": str(run_dir / "manifest.json"),
            "kernel_csv": str(kernel_path),
            "psi_csv": str(psi_path),
            "tauR_series_csv": str(tauR_path) if tauR_path else None,
        },
        "artifacts": artifacts,
        "columns": columns,
        "kernel_column_map": kernel_map,
        "kernel_stats": kernel_stats,
        "tauR_coverage": tauR_coverage,
        "tauR_distribution_stats": tauR_dist_stats,
        "logs": {
            "oor": oor_rates,
            "censor": censor_rates,
        },
        "seams": {
            "seam_ledger_row": seam_row,
        },
        "receipts": {
            "weld_receipt_row": weld_row,
        },
        "gaps": sorted(set(gaps)),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="UMCP.KIN audit extractor - extract run data to JSON report")
    ap.add_argument("--repo", type=str, default=".", help="Path to repo root (default: .)")
    ap.add_argument("--run", type=str, default=None, help="Specific run dir (e.g., runs/<Run_ID>)")
    ap.add_argument("--casepack", type=str, default=None, help="Filter by casepack_id substring (e.g., KIN.CP.SHM)")
    ap.add_argument("--out", type=str, default=None, help="Write JSON report to file")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()

    if args.run:
        run_dirs = [Path(args.run).resolve()]
    else:
        run_dirs = discover_run_dirs(repo_root)

    reports = []
    for rd in run_dirs:
        r = extract_run(rd)
        if args.casepack:
            cp = (r.get("identity", {}) or {}).get("casepack_id") or ""
            if args.casepack not in str(cp):
                continue
        reports.append(r)

    out_obj = {
        "umcp_kin_extract_version": "1.0",
        "repo_root": str(repo_root),
        "generated_utc": datetime.now(UTC).isoformat(),
        "runs": reports,
    }

    txt = json.dumps(out_obj, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
