#!/usr/bin/env python3
"""
Migrate ledger to include all kernel invariants.

This script upgrades ledger/return_log.csv from the old schema:
  timestamp, run_status, delta_kappa, stiffness, omega, curvature

To the new schema with all kernel invariants:
  timestamp, run_status, F, omega, kappa, IC, C, S, tau_R, delta_kappa

The migration:
1. Reads the existing ledger
2. Maps old columns to new schema
3. Computes missing invariants where possible
4. Writes to new ledger format

Usage:
    python scripts/migrate_ledger.py
"""

import csv
import math
import shutil
from datetime import datetime
from pathlib import Path


def migrate_ledger():
    """Migrate ledger to new schema with all kernel invariants."""
    repo_root = Path(__file__).parent.parent
    ledger_path = repo_root / "ledger" / "return_log.csv"
    backup_path = repo_root / "ledger" / f"return_log.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    if not ledger_path.exists():
        print("No ledger found. Nothing to migrate.")
        return

    # Backup original
    shutil.copy(ledger_path, backup_path)
    print(f"Backed up to: {backup_path}")

    # Read existing data
    rows = []
    with open(ledger_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows with columns: {old_fieldnames}")

    # New schema
    new_fieldnames = [
        "timestamp",
        "run_status",
        "F",
        "omega",
        "kappa",
        "IC",
        "C",
        "S",
        "tau_R",
        "delta_kappa",
    ]

    # Migrate rows
    migrated_rows = []
    for row in rows:
        new_row = {
            "timestamp": row.get("timestamp", ""),
            "run_status": row.get("run_status", ""),
            "F": "",  # Will compute if omega available
            "omega": row.get("omega", ""),
            "kappa": "",  # Will compute if IC available
            "IC": "",
            "C": row.get("curvature", row.get("C", "")),
            "S": row.get("stiffness", row.get("S", "")),
            "tau_R": row.get("tau_R", ""),
            "delta_kappa": row.get("delta_kappa", ""),
        }

        # Compute F from omega if available (F = 1 - omega)
        omega_str = new_row["omega"]
        if omega_str and omega_str.strip():
            try:
                omega = float(omega_str)
                F = 1.0 - omega
                new_row["F"] = f"{F:.6f}"

                # Compute IC and kappa if we have curvature as proxy
                # IC ≈ F * (1 - C) as rough estimate when no direct measurement
                C_str = new_row["C"]
                if C_str and C_str.strip():
                    try:
                        C = float(C_str)
                        IC_estimate = F * (1 - C * 0.5)  # Conservative estimate
                        IC_estimate = max(0.01, min(0.99, IC_estimate))
                        new_row["IC"] = f"{IC_estimate:.6f}"
                        new_row["kappa"] = f"{math.log(IC_estimate):.6f}"
                    except (ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                pass

        migrated_rows.append(new_row)

    # Write new ledger
    with open(ledger_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(migrated_rows)

    print(f"Migrated {len(migrated_rows)} rows to new schema")
    print(f"New columns: {new_fieldnames}")
    print("Done!")


if __name__ == "__main__":
    migrate_ledger()
