#!/usr/bin/env python3
"""
Generate KIN.REF.PHASE Expected Outputs

This script generates the canonical expected outputs for the KIN.REF.PHASE
oscillator test vector. It loads raw_measurements.csv, applies the phase-anchor
selection rules, and outputs:

  - expected/ref_phase_expected.csv (anchor selections per time index)
  - expected/censor_expected.json (typed censor events)

Usage:
    python scripts/generate_kin_ref_phase_expected.py

The script is deterministic: running it twice produces identical output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# Repository root
REPO_ROOT = Path(__file__).parent.parent

# Load the KIN.REF.PHASE closure module dynamically
closure_path = REPO_ROOT / "casepacks" / "kin_ref_phase_oscillator" / "closures" / "kin_ref_phase.py"
spec = importlib.util.spec_from_file_location("kin_ref_phase", closure_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from {closure_path}")
kin_ref_phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kin_ref_phase)

# Import the required items from the module
DEBOUNCE = kin_ref_phase.DEBOUNCE
DELTA_PHI_MAX = kin_ref_phase.DELTA_PHI_MAX
WINDOW = kin_ref_phase.WINDOW
get_frozen_config_sha256 = kin_ref_phase.get_frozen_config_sha256
process_trajectory = kin_ref_phase.process_trajectory


def load_raw_measurements(casepack_dir: Path) -> tuple[list[float], list[float]]:
    """
    Load raw measurements from CSV.

    Args:
        casepack_dir: Path to casepack directory

    Returns:
        Tuple of (x_series, v_series)
    """
    csv_path = casepack_dir / "raw_measurements.csv"

    x_series: list[float] = []
    v_series: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_series.append(float(row["x"]))
            v_series.append(float(row["v"]))

    return x_series, v_series


def write_expected_csv(
    results: list[Any],
    output_path: Path,
) -> None:
    """
    Write expected results to CSV.

    Args:
        results: List of PhaseAnchorResult
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = ["idx", "phi", "eligible_count", "anchor_u", "delta_phi", "undefined_reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "idx": r.idx,
                    "phi": f"{r.phi:.10f}",
                    "eligible_count": r.eligible_count,
                    "anchor_u": r.anchor_u if r.anchor_u is not None else "",
                    "delta_phi": f"{r.delta_phi:.10f}" if r.delta_phi is not None else "",
                    "undefined_reason": r.undefined_reason.value,
                }
            )


def write_censor_json(
    censors: list[Any],
    output_path: Path,
) -> None:
    """
    Write censor events to JSON.

    Args:
        censors: List of CensorEvent
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "schema": "KIN.REF.PHASE.CENSOR.v1",
        "description": "Typed censor events for undefined reference anchors",
        "parameters": {
            "delta_phi_max": round(DELTA_PHI_MAX, 10),
            "window": WINDOW,
            "debounce": DEBOUNCE,
        },
        "censors": [
            {
                "idx": c.idx,
                "phi": round(c.phi, 10),
                "reason": c.reason,
                "eligible_count": c.eligible_count,
            }
            for c in censors
        ],
        "total_censored": len(censors),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> int:
    """Generate expected outputs."""
    print("=" * 60)
    print("KIN.REF.PHASE Expected Output Generator")
    print("=" * 60)

    # Paths
    casepack_dir = REPO_ROOT / "casepacks" / "kin_ref_phase_oscillator"
    expected_csv = casepack_dir / "expected" / "ref_phase_expected.csv"
    censor_json = casepack_dir / "expected" / "censor_expected.json"

    # Print frozen config
    print("\nFrozen Configuration:")
    print(f"  delta_phi_max: {DELTA_PHI_MAX:.10f} (π/6)")
    print(f"  window: {WINDOW}")
    print(f"  debounce: {DEBOUNCE}")
    print(f"  frozen_config_sha256: {get_frozen_config_sha256()}")

    # Load data
    print(f"\nLoading data from: {casepack_dir / 'raw_measurements.csv'}")
    x_series, v_series = load_raw_measurements(casepack_dir)
    print(f"  Loaded {len(x_series)} rows")

    # Process trajectory
    print("\nProcessing trajectory...")
    results, censors = process_trajectory(x_series, v_series)

    # Statistics
    defined_count = sum(1 for r in results if r.anchor_u is not None)
    undefined_count = len(results) - defined_count
    empty_eligible = sum(1 for c in censors if c.reason == "EMPTY_ELIGIBLE")
    phase_mismatch = sum(1 for c in censors if c.reason == "PHASE_MISMATCH")

    print("\nResults:")
    print(f"  Total rows: {len(results)}")
    print(f"  Defined anchors: {defined_count}")
    print(f"  Undefined: {undefined_count}")
    print(f"    EMPTY_ELIGIBLE: {empty_eligible}")
    print(f"    PHASE_MISMATCH: {phase_mismatch}")

    # Write outputs
    print("\nWriting expected outputs:")
    print(f"  CSV: {expected_csv}")
    write_expected_csv(results, expected_csv)

    print(f"  JSON: {censor_json}")
    write_censor_json(censors, censor_json)

    # Verify tie-breaker case
    print("\nVerifying edge cases:")

    # Check for tie-breaker (rows with same phi)
    tie_found = False
    for i in range(len(results) - 1):
        r1, r2 = results[i], results[i + 1]
        if r1.anchor_u is not None and r2.anchor_u is not None and abs(r1.phi - r2.phi) < 1e-9:
            print(f"  Tie-breaker case at rows {i}, {i+1}:")
            print(f"    phi = {r1.phi:.10f}")
            print(f"    anchor_u[{i}] = {r1.anchor_u}, anchor_u[{i+1}] = {r2.anchor_u}")
            tie_found = True

    if not tie_found:
        print("  Note: No adjacent rows with identical phi found in defined region")

    # Check for empty eligible
    if empty_eligible > 0:
        print(f"  Empty eligible cases: rows {[c.idx for c in censors if c.reason == 'EMPTY_ELIGIBLE']}")

    # Check for phase mismatch
    if phase_mismatch > 0:
        print(f"  Phase mismatch cases: rows {[c.idx for c in censors if c.reason == 'PHASE_MISMATCH']}")

    print("\n" + "=" * 60)
    print("Done. Expected outputs generated successfully.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
