#!/usr/bin/env python3
"""
freeze_gate_scales.py - Compute and freeze gate alignment scales from SHM control.

This script computes the gate normalization scales (Ω, Σ, C) from the SHM baseline
so that legacy regime thresholds (Stable: ω<0.038, S<0.15, C<0.14) become meaningful
under the current [0,1] normalized outputs.

Mathematical definition:
    Ω = p95(ω_raw) / 0.038
    Σ = p95(S_raw) / 0.15
    C = p95(C_raw) / 0.14

Gate-aligned normalization:
    ω_gate = clip(ω_raw / Ω, 0, 1)
    S_gate = clip(S_raw / Σ, 0, 1)
    C_gate = clip(C_raw / C, 0, 1)

Under this normalization, values < 1.0 are within Stable threshold.

Usage:
    python scripts/freeze_gate_scales.py [--run RUN004] [--out frozen_scales.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default legacy regime thresholds (from original kernel spec)
LEGACY_THRESHOLDS = {
    "omega": 0.038,  # Stable if ω < 0.038
    "S": 0.15,  # Stable if S < 0.15
    "C": 0.14,  # Stable if C < 0.14
}


def compute_frozen_scales(
    shm_kernel_csv: Path,
    legacy_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute frozen gate scales from SHM control kernel.csv.

    Parameters
    ----------
    shm_kernel_csv : Path
        Path to SHM kernel/kernel.csv
    legacy_thresholds : dict, optional
        Override legacy regime thresholds

    Returns
    -------
    dict
        Frozen scales with metadata
    """
    if legacy_thresholds is None:
        legacy_thresholds = LEGACY_THRESHOLDS

    # Load SHM kernel data
    df: pd.DataFrame = pd.read_csv(shm_kernel_csv)  # type: ignore[call-overload]

    # Compute p95 of each measure
    p95_omega = float(np.percentile(df["omega"].dropna(), 95))
    p95_S = float(np.percentile(df["S"].dropna(), 95))
    p95_C = float(np.percentile(df["C"].dropna(), 95))

    # Compute frozen scales: Ω = p95(ω) / threshold
    Omega = p95_omega / legacy_thresholds["omega"]
    Sigma = p95_S / legacy_thresholds["S"]
    Cal_C = p95_C / legacy_thresholds["C"]

    # Also compute median and mean for reference
    med_omega = float(np.median(df["omega"].dropna()))
    med_S = float(np.median(df["S"].dropna()))
    med_C = float(np.median(df["C"].dropna()))

    return {
        "frozen_from": str(shm_kernel_csv),
        "frozen_at": datetime.now(UTC).isoformat(),
        "contract": "UMA.INTSTACK.v1",
        "control_casepack": "KIN.CP.SHM",
        "legacy_thresholds": {
            "omega_stable": legacy_thresholds["omega"],
            "S_stable": legacy_thresholds["S"],
            "C_stable": legacy_thresholds["C"],
        },
        "p95_raw": {
            "omega": p95_omega,
            "S": p95_S,
            "C": p95_C,
        },
        "median_raw": {
            "omega": med_omega,
            "S": med_S,
            "C": med_C,
        },
        "frozen_scales": {
            "Omega": Omega,
            "Sigma": Sigma,
            "Cal_C": Cal_C,
        },
        "interpretation": {
            "gate_aligned_omega": f"clip(omega_raw / {Omega:.6f}, 0, 1)",
            "gate_aligned_S": f"clip(S_raw / {Sigma:.6f}, 0, 1)",
            "gate_aligned_C": f"clip(C_raw / {Cal_C:.6f}, 0, 1)",
            "stable_if": "gate_aligned < 1.0 for all three measures",
        },
    }


def apply_gate_alignment(
    kernel_csv: Path,
    frozen_scales: dict[str, Any],
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Apply gate alignment normalization to a kernel.csv.

    Parameters
    ----------
    kernel_csv : Path
        Path to kernel/kernel.csv
    frozen_scales : dict
        Frozen scales from compute_frozen_scales()
    output_csv : Path, optional
        If provided, write gate-aligned CSV

    Returns
    -------
    pd.DataFrame
        DataFrame with added gate-aligned columns
    """
    df: pd.DataFrame = pd.read_csv(kernel_csv)  # type: ignore[call-overload]
    scales = frozen_scales["frozen_scales"]

    # Compute gate-aligned values
    df["omega_gate"] = np.clip(df["omega"] / scales["Omega"], 0, 1)
    df["S_gate"] = np.clip(df["S"] / scales["Sigma"], 0, 1)
    df["C_gate"] = np.clip(df["C"] / scales["Cal_C"], 0, 1)

    # Classify regime based on gate-aligned values
    def classify_regime(row: pd.Series) -> str:
        if row["omega_gate"] < 1.0 and row["S_gate"] < 1.0 and row["C_gate"] < 1.0:
            return "Stable"
        elif row["omega_gate"] > 1.0 or row["S_gate"] > 1.0:
            return "Transitional"
        else:
            return "Drift"

    df["regime_gate"] = df.apply(classify_regime, axis=1)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Wrote gate-aligned kernel to {output_csv}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and freeze gate alignment scales from SHM control")
    parser.add_argument(
        "--run",
        default="RUN004",
        help="Run ID to use for SHM control (default: RUN004)",
    )
    parser.add_argument(
        "--out",
        default="freeze/frozen_scales.json",
        help="Output path for frozen scales JSON",
    )
    parser.add_argument(
        "--apply-all",
        action="store_true",
        help="Apply gate alignment to all casepacks and write _gate.csv files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    runs_dir = repo_root / "runs"

    # Find SHM kernel.csv
    shm_kernel = runs_dir / f"KIN.CP.SHM.{args.run}" / "kernel" / "kernel.csv"
    if not shm_kernel.exists():
        print(f"ERROR: SHM kernel not found: {shm_kernel}")
        return

    print(f"Computing frozen scales from {shm_kernel}...")
    frozen = compute_frozen_scales(shm_kernel)

    # Display summary
    print("\n=== FROZEN GATE SCALES (from SHM control) ===")
    print(f"  Ω (omega scale): {frozen['frozen_scales']['Omega']:.6f}")
    print(f"  Σ (S scale):     {frozen['frozen_scales']['Sigma']:.6f}")
    print(f"  𝒞 (C scale):     {frozen['frozen_scales']['Cal_C']:.6f}")
    print("\n  p95 raw values:")
    print(f"    ω_p95 = {frozen['p95_raw']['omega']:.6f}")
    print(f"    S_p95 = {frozen['p95_raw']['S']:.6f}")
    print(f"    C_p95 = {frozen['p95_raw']['C']:.6f}")

    # Write frozen scales JSON
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nWrote frozen scales to {out_path}")

    # Optionally apply gate alignment to all casepacks
    if args.apply_all:
        print("\n=== APPLYING GATE ALIGNMENT TO ALL CASEPACKS ===")
        casepacks = ["KIN.CP.SHM", "KIN.CP.GAIT", "KIN.CP.BALLISTIC"]

        for cp in casepacks:
            kernel_csv = runs_dir / f"{cp}.{args.run}" / "kernel" / "kernel.csv"
            if not kernel_csv.exists():
                print(f"  SKIP {cp}: kernel.csv not found")
                continue

            output_csv = kernel_csv.parent / "kernel_gate.csv"
            df = apply_gate_alignment(kernel_csv, frozen, output_csv)

            # Report regime distribution
            regime_counts = df["regime_gate"].value_counts()
            print(f"\n  {cp}:")
            print(
                f"    Stable:       {regime_counts.get('Stable', 0):4d} ({100 * regime_counts.get('Stable', 0) / len(df):.1f}%)"
            )
            print(
                f"    Transitional: {regime_counts.get('Transitional', 0):4d} ({100 * regime_counts.get('Transitional', 0) / len(df):.1f}%)"
            )
            print(
                f"    Drift:        {regime_counts.get('Drift', 0):4d} ({100 * regime_counts.get('Drift', 0) / len(df):.1f}%)"
            )


if __name__ == "__main__":
    main()
