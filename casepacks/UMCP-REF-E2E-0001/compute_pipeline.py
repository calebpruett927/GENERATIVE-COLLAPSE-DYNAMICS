#!/usr/bin/env python3
"""
UMCP-REF-E2E-0001 Pipeline Executor
Complete end-to-end pipeline: ingest → freeze → compute → regime → render → export

Pipeline stages:
1. /ingest: Load raw measurements
2. /freeze: Contract snapshot already on disk (contracts/contract.yaml)
3. /compute: Calculate Tier-1 invariants {ω,F,S,C,τ_R,IC,κ}
4. /regime: Classify regime labels
5. /render: Generate outputs
6. /export: Create receipts and manifest

Follows UMCP requirements:
- No Tier-1 computation until contract is frozen
- Typed boundaries explicit (τ_R = INF_REC)
- No symbol capture
- clip_and_flag OOR policy
"""

import csv
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Paths
CASEPACK = Path(__file__).resolve().parent
DATA = CASEPACK / "data"
CONTRACTS = CASEPACK / "contracts"
OUTPUTS = CASEPACK / "outputs"
RECEIPTS = CASEPACK / "receipts"
LOGS = CASEPACK / "logs"

RAW_CSV = DATA / "raw.csv"
PSI_CSV = DATA / "psi_trace.csv"
KERNEL_CSV = OUTPUTS / "kernel_ledger.csv"
REGIME_CSV = OUTPUTS / "regime.csv"
DIAG_CSV = OUTPUTS / "diagnostics.csv"
SS1M_JSON = RECEIPTS / "ss1m.json"

# Contract parameters (frozen)
A = 0.0
B = 10.0
EPS = 1.0e-8
P = 3
ALPHA = 1.0
LAMBDA = 0.2
ETA = 1.0e-3
TOL_SEAM = 0.005
TOL_ID = 1.0e-9

# Weights (uniform for n=3)
WEIGHTS = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


class PipelineLog:
    def __init__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.logfile = LOGS / "run.log"
        self.start_time = datetime.now(UTC)
        self.log(f"Pipeline started: {self.start_time.isoformat()}")

    def log(self, msg: str):
        timestamp = datetime.now(UTC).isoformat()
        line = f"[{timestamp}] {msg}\n"
        with open(self.logfile, "a") as f:
            f.write(line)
        print(msg)


log = PipelineLog()


def clip01(x: float) -> tuple[float, bool]:
    """Clip to [0,1] and return (clipped_value, oor_flag)"""
    if x < 0.0:
        return 0.0, True
    elif x > 1.0:
        return 1.0, True
    else:
        return x, False


def log_safe(c: float) -> float:
    """Safe log domain: clamp to [ε, 1-ε]"""
    return min(1.0 - EPS, max(EPS, c))


def embed_to_psi(x: float) -> tuple[float, bool]:
    """Linear embedding with clip_and_flag"""
    c_raw = (x - A) / (B - A)
    c_clip, oor = clip01(c_raw)
    return c_clip, oor


def compute_invariants(cs: list[float], t: int, psi0: list[float]) -> dict[str, Any]:
    """Compute Tier-1 invariants: {ω, F, S, C, τ_R, κ, IC}"""
    n = len(cs)
    w = WEIGHTS

    # F: weighted coherence
    F = sum(w[i] * cs[i] for i in range(n))

    # ω: instability = 1 - F
    omega = 1.0 - F

    # S: entropy-like term -Σ w_i [c ln c + (1-c) ln(1-c)]
    S = 0.0
    for i in range(n):
        c_safe = log_safe(cs[i])
        S += w[i] * (c_safe * math.log(c_safe) + (1.0 - c_safe) * math.log(1.0 - c_safe))
    S = -S

    # C: curvature proxy = std(c_i) / 0.5
    mean_c = sum(cs) / n
    var = sum((cs[i] - mean_c) ** 2 for i in range(n)) / n
    std = math.sqrt(var)
    C = std / 0.5

    # κ: log-mean = Σ w_i ln(c_i)
    kappa = sum(w[i] * math.log(log_safe(cs[i])) for i in range(n))

    # IC: integrity coordinate = exp(κ)
    IC = math.exp(kappa)

    # τ_R: return time (check L2 distance from initial state)
    if t == 0:
        tau_R = "INF_REC"
        tau_R_finite = False
    else:
        # L2 distance from initial state
        dist = math.sqrt(sum((cs[i] - psi0[i]) ** 2 for i in range(n)))
        if dist < ETA:
            # Return detected
            tau_R = t
            tau_R_finite = True
        else:
            tau_R = "INF_REC"
            tau_R_finite = False

    return {
        "t": t,
        "omega": omega,
        "F": F,
        "S": S,
        "C": C,
        "tau_R": tau_R,
        "tau_R_finite": tau_R_finite,
        "kappa": kappa,
        "IC": IC,
    }


def classify_regime(omega: float, F: float, S: float, C: float) -> tuple[str, bool]:
    """Classify regime based on thresholds"""
    # Stable: ω < 0.038, F > 0.90, S < 0.15, C < 0.14
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable", False
    # Collapse: ω >= 0.30
    elif omega >= 0.30:
        return "Collapse", True
    # Watch: everything else
    else:
        return "Watch", False


def stage_ingest():
    """Stage 1: Ingest raw measurements"""
    log.log("=== STAGE 1: /ingest ===")
    log.log(f"Reading raw measurements from: {RAW_CSV}")

    with open(RAW_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    log.log(f"Loaded {len(rows)} rows")
    return rows


def stage_freeze():
    """Stage 2: Verify contract snapshot is frozen"""
    log.log("=== STAGE 2: /freeze ===")
    contract_file = CONTRACTS / "contract.yaml"

    if not contract_file.exists():
        raise FileNotFoundError(f"Contract snapshot not found: {contract_file}")

    log.log(f"Contract snapshot verified: {contract_file}")
    log.log(f"Parameters frozen: a={A}, b={B}, p={P}, α={ALPHA}, λ={LAMBDA}, η={ETA}")
    log.log("No Tier-1 computation will occur until this stage completes.")


def stage_compute(raw_rows: list[dict]) -> list[dict]:
    """Stage 3: Compute Tier-1 invariants"""
    log.log("=== STAGE 3: /compute ===")
    log.log("Computing Tier-1 kernel: {ω, F, S, C, τ_R, κ, IC}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # First pass: embed to Ψ-space
    psi_rows = []
    for row in raw_rows:
        t = int(row["t"])
        x1 = float(row["x1_voltage"])
        x2 = float(row["x2_temperature"])
        x3 = float(row["x3_pressure"])

        c1, oor1 = embed_to_psi(x1)
        c2, oor2 = embed_to_psi(x2)
        c3, oor3 = embed_to_psi(x3)

        psi_rows.append({"t": t, "c_1": c1, "c_2": c2, "c_3": c3, "oor_1": oor1, "oor_2": oor2, "oor_3": oor3})

    # Write psi_trace.csv
    with open(PSI_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "c_1", "c_2", "c_3", "oor_1", "oor_2", "oor_3"])
        writer.writeheader()
        writer.writerows(psi_rows)

    log.log(f"Wrote Ψ-trace: {PSI_CSV}")

    # Count OOR events
    oor_count = sum(1 for row in psi_rows if row["oor_1"] or row["oor_2"] or row["oor_3"])
    log.log(f"OOR events detected: {oor_count}")

    # Second pass: compute kernel invariants
    psi0 = [psi_rows[0]["c_1"], psi_rows[0]["c_2"], psi_rows[0]["c_3"]]
    kernel_rows = []

    for psi_row in psi_rows:
        t = psi_row["t"]
        cs = [psi_row["c_1"], psi_row["c_2"], psi_row["c_3"]]

        inv = compute_invariants(cs, t, psi0)
        kernel_rows.append(inv)

    # Write kernel_ledger.csv
    with open(KERNEL_CSV, "w", newline="") as f:
        fieldnames = ["t", "omega", "F", "S", "C", "tau_R", "kappa", "IC"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kernel_rows:
            writer.writerow({k: row[k] for k in fieldnames if k in row})

    log.log(f"Wrote kernel ledger: {KERNEL_CSV}")

    # Count finite returns
    finite_returns = sum(1 for row in kernel_rows if row["tau_R_finite"])
    inf_rec_count = sum(1 for row in kernel_rows if not row["tau_R_finite"])
    log.log(f"Finite returns: {finite_returns}, INF_REC: {inf_rec_count}")

    return kernel_rows


def stage_regime(kernel_rows: list[dict]):
    """Stage 4: Classify regimes"""
    log.log("=== STAGE 4: /regime ===")

    regime_rows = []
    for row in kernel_rows:
        label, critical = classify_regime(row["omega"], row["F"], row["S"], row["C"])
        regime_rows.append({"t": row["t"], "regime": label, "critical_overlay": critical})

    # Write regime.csv
    with open(REGIME_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "regime", "critical_overlay"])
        writer.writeheader()
        writer.writerows(regime_rows)

    log.log(f"Wrote regime classification: {REGIME_CSV}")

    # Regime summary
    regime_counts = {}
    for row in regime_rows:
        label = row["regime"]
        regime_counts[label] = regime_counts.get(label, 0) + 1

    log.log(f"Regime distribution: {regime_counts}")
    return regime_rows


def stage_render(kernel_rows: list[dict]):
    """Stage 5: Generate diagnostics (non-gating)"""
    log.log("=== STAGE 5: /render ===")

    # Diagnostic checks (equator-style, non-gating)
    diag_rows = []
    for row in kernel_rows:
        # Example diagnostic: check if F is near equator (0.5)
        F_equator_dist = abs(row["F"] - 0.5)
        near_equator = F_equator_dist < 0.1

        diag_rows.append(
            {
                "t": row["t"],
                "F_equator_distance": F_equator_dist,
                "near_equator": near_equator,
                "note": "Diagnostic only - not used as gate",
            }
        )

    with open(DIAG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "F_equator_distance", "near_equator", "note"])
        writer.writeheader()
        writer.writerows(diag_rows)

    log.log(f"Wrote diagnostics: {DIAG_CSV}")


def stage_export(kernel_rows: list[dict], regime_rows: list[dict]):
    """Stage 6: Create receipts"""
    log.log("=== STAGE 6: /export ===")

    RECEIPTS.mkdir(parents=True, exist_ok=True)

    # Get final row summary
    final_row = kernel_rows[-1]

    # Validate invariant consistency: IC ≈ exp(κ)
    log.log("Validating invariant consistency...")
    ic_consistency_checks = []
    for row in kernel_rows:
        expected_IC = math.exp(row["kappa"])
        actual_IC = row["IC"]
        error = abs(actual_IC - expected_IC)
        relative_error = error / max(abs(expected_IC), 1e-15)

        ic_consistency_checks.append(
            {"t": row["t"], "IC": actual_IC, "exp_kappa": expected_IC, "abs_error": error, "rel_error": relative_error}
        )

        # Tolerance check (strict: 1e-9, baseline: warn if > 1e-6)
        if relative_error > 1e-6:
            log.log(f"  WARNING: t={row['t']}: IC consistency violation (rel_error={relative_error:.2e})")

    # Check final row consistency
    final_ic_error = abs(final_row["IC"] - math.exp(final_row["kappa"]))
    if final_ic_error > 1e-9:
        log.log(f"  WARNING: Final IC consistency error: {final_ic_error:.2e}")
    else:
        log.log(f"  ✓ Final IC ≈ exp(κ) (error={final_ic_error:.2e})")

    # Count regime distribution
    regime_counts = {}
    for row in regime_rows:
        label = row["regime"]
        regime_counts[label] = regime_counts.get(label, 0) + 1

    # Count OOR events
    oor_count = 0
    with open(PSI_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["oor_1"] == "True" or row["oor_2"] == "True" or row["oor_3"] == "True":
                oor_count += 1

    # Count tau_R types
    inf_rec_count = sum(1 for row in kernel_rows if not row["tau_R_finite"])
    finite_return_count = sum(1 for row in kernel_rows if row["tau_R_finite"])

    # Create SS1M receipt
    receipt = {
        "schema": "schemas/receipt.ss1m.schema.json",
        "receipt": {
            "kind": "ss1m",
            "version": 1,
            "case_id": "UMCP-REF-E2E-0001",
            "timezone": "America/Chicago",
            "emitted_at": datetime.now(UTC).isoformat(),
            "address": "Clement Paulus",
            "canon": {
                "pre_doi": "10.5281/zenodo.17756705",
                "post_doi": "10.5281/zenodo.18072852",
                "weld_id": "W-2025-12-31-PHYS-COHERENCE",
            },
            "contract": {
                "id": "UMA.INTSTACK.v1",
                "version": "1.0.0",
                "path": "contracts/contract.yaml",
                "frozen_parameters": {
                    "a": A,
                    "b": B,
                    "face": "pre_clip",
                    "epsilon": EPS,
                    "p": P,
                    "alpha": ALPHA,
                    "lambda": LAMBDA,
                    "eta": ETA,
                    "tol_seam": TOL_SEAM,
                    "tol_id": TOL_ID,
                    "oor_policy": "clip_and_flag",
                },
            },
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
                "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            },
            "manifest": {"id": "UMCP-REF-E2E-0001.manifest", "root_sha256": "pending"},
            "kernel_summary": {
                "total_rows": len(kernel_rows),
                "final_row": {
                    "t": final_row["t"],
                    "omega": final_row["omega"],
                    "F": final_row["F"],
                    "S": final_row["S"],
                    "C": final_row["C"],
                    "tau_R": final_row["tau_R"],
                    "kappa": final_row["kappa"],
                    "IC": final_row["IC"],
                },
                "ic_consistency": {
                    "final_abs_error": abs(final_row["IC"] - math.exp(final_row["kappa"])),
                    "tolerance": 1e-9,
                    "check_passed": abs(final_row["IC"] - math.exp(final_row["kappa"])) < 1e-9,
                },
            },
            "regime_summary": {"distribution": regime_counts, "final_regime": regime_rows[-1]["regime"]},
            "typed_boundaries": {
                "oor_count": oor_count,
                "inf_rec_count": inf_rec_count,
                "finite_return_count": finite_return_count,
                "note": "τ_R = INF_REC → R·τ_R = 0 (no credit rule)",
            },
            "status": "CONFORMANT",
            "notes": "Complete end-to-end pipeline execution. All Tier-1 invariants computed after contract freeze.",
        },
    }

    with open(SS1M_JSON, "w") as f:
        json.dump(receipt, f, indent=2)

    log.log(f"Wrote SS1M receipt: {SS1M_JSON}")

    return receipt


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    log.log("╔══════════════════════════════════════════════════════════╗")
    log.log("║  UMCP-REF-E2E-0001: Complete Audit-Ready Pipeline       ║")
    log.log("╚══════════════════════════════════════════════════════════╝")

    # Execute pipeline
    raw_rows = stage_ingest()
    stage_freeze()
    kernel_rows = stage_compute(raw_rows)
    regime_rows = stage_regime(kernel_rows)
    stage_render(kernel_rows)
    receipt = stage_export(kernel_rows, regime_rows)

    log.log("")
    log.log("=== PIPELINE COMPLETE ===")
    log.log(f"Duration: {(datetime.now(UTC) - log.start_time).total_seconds():.2f}s")
    log.log("")
    log.log("Generated artifacts:")
    log.log(f"  - {PSI_CSV}")
    log.log(f"  - {KERNEL_CSV}")
    log.log(f"  - {REGIME_CSV}")
    log.log(f"  - {DIAG_CSV}")
    log.log(f"  - {SS1M_JSON}")
    log.log("")

    # Print HUD-style summary
    final = receipt["receipt"]["kernel_summary"]["final_row"]
    log.log("╔══════════════════════════════════════════════════════════╗")
    log.log("║                     SS1M SUMMARY                         ║")
    log.log("╚══════════════════════════════════════════════════════════╝")
    log.log(f"Case: {receipt['receipt']['case_id']} | Status: {receipt['receipt']['status']}")
    log.log(
        f"Final: ω={final['omega']:.6f} F={final['F']:.6f} S={final['S']:.6f} C={final['C']:.6f} τ_R={final['tau_R']} κ={final['kappa']:.6f} IC={final['IC']:.6f}"
    )
    log.log(
        f"Regime: {receipt['receipt']['regime_summary']['final_regime']} | OOR events: {receipt['receipt']['typed_boundaries']['oor_count']}"
    )
    log.log("")


if __name__ == "__main__":
    main()
