#!/usr/bin/env python3
"""
UMCP Baseline Freeze Script

Creates a frozen snapshot of current artifacts for:
1. Phase B drift detection (FN-001, FN-002, FN-008, FN-009)
2. Reproducibility guarantees
3. Seam accounting across runs

The freeze directory contains hashes and values that become the
"known good" baseline for comparison.

Usage:
    python scripts/freeze_baseline.py [--run-id RUN_ID]
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def freeze_baseline(root: Path, run_id: str | None = None) -> dict[str, str]:
    """
    Freeze current state as baseline.

    Creates freeze/ directory with:
    - SHA256 hashes of critical artifacts
    - Snapshot of bounds, return params, adapter settings
    - Baseline κ, IC values
    """
    freeze_dir = root / "freeze"
    freeze_dir.mkdir(exist_ok=True)

    if run_id is None:
        run_id = f"BASELINE-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    hashes: dict[str, str] = {}

    # 1. Hash critical YAML files
    critical_files = [
        "embedding.yaml",
        "return.yaml",
        "contract.yaml",
        "weights.csv",
        "observables.yaml",
        "closures/registry.yaml",
    ]

    for fname in critical_files:
        fpath = root / fname
        if fpath.exists():
            h = sha256_file(fpath)
            hashes[fname] = h
            hash_file = freeze_dir / f"{fname.replace('/', '_')}.sha256"
            hash_file.write_text(h)
            print(f"  ✓ Froze {fname}: {h[:16]}...")

    # 2. Extract and freeze bounds from observables
    obs_path = root / "observables.yaml"
    if obs_path.exists():
        with open(obs_path) as f:
            obs = yaml.safe_load(f)
        bounds = obs.get("bounds", obs.get("coordinate_bounds", {}))
        bounds_file = freeze_dir / "bounds.json"
        bounds_file.write_text(json.dumps(bounds, indent=2))
        print(f"  ✓ Froze bounds: {bounds_file}")

    # 3. Extract and freeze return parameters
    return_path = root / "return.yaml"
    if return_path.exists():
        with open(return_path) as f:
            ret = yaml.safe_load(f)
        params = {
            "eta": ret.get("return", {}).get("eta"),
            "H_rec": ret.get("return", {}).get("H_rec"),
            "norm": ret.get("return", {}).get("norm"),
        }
        params_file = freeze_dir / "return_params.json"
        params_file.write_text(json.dumps(params, indent=2))
        print(f"  ✓ Froze return params: {params_file}")

    # 4. Freeze norms closure hash
    registry_path = root / "closures" / "registry.yaml"
    if registry_path.exists():
        with open(registry_path) as f:
            reg = yaml.safe_load(f)
        closures: dict[str, Any] = reg.get("registry", {}).get("closures", {})
        norms_def: dict[str, Any] = closures.get("norms", {})
        if norms_def.get("path"):
            norms_path: Path = root / str(norms_def["path"])
            if norms_path.exists():
                h = sha256_file(norms_path)
                (freeze_dir / "norms_closure.sha256").write_text(h)
                print(f"  ✓ Froze norms closure: {h[:16]}...")

    # 5. Freeze current invariants as baseline
    inv_path = root / "outputs" / "invariants.csv"
    if inv_path.exists():
        with open(inv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        baseline_invariants: list[dict[str, Any]] = []
        for row in rows:
            baseline_invariants.append(
                {
                    "t": row.get("t"),
                    "kappa": float(row.get("kappa", 0)),
                    "IC": float(row.get("IC", 1)),
                    "omega": float(row.get("omega", 0)),
                    "F": float(row.get("F", 1)),
                }
            )

        baseline_file = freeze_dir / "baseline_invariants.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "frozen_utc": datetime.now(UTC).isoformat(),
                    "invariants": baseline_invariants,
                },
                indent=2,
            )
        )
        print(f"  ✓ Froze baseline invariants: {baseline_file}")

    # 6. Create freeze manifest
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "frozen_utc": datetime.now(UTC).isoformat(),
        "hashes": hashes,
        "files_frozen": list(hashes.keys()),
        "purpose": "Baseline for drift detection and seam accounting",
    }
    manifest_file = freeze_dir / "freeze_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"  ✓ Created freeze manifest: {manifest_file}")

    return hashes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Freeze current state as baseline")
    parser.add_argument("--run-id", default=None, help="Run ID for this baseline")
    args = parser.parse_args()

    # Find repo root
    root = Path.cwd()
    while root != root.parent:
        if (root / "pyproject.toml").exists():
            break
        root = root.parent

    print("=" * 60)
    print("UMCP Baseline Freeze")
    print("=" * 60)
    print(f"Root: {root}")
    print()

    hashes = freeze_baseline(root, args.run_id)

    print()
    print("=" * 60)
    print(f"Baseline frozen: {len(hashes)} artifacts")
    print("=" * 60)


if __name__ == "__main__":
    main()
