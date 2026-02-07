"""Generate expected outputs for the finance_continuity casepack."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASEPACK = Path(__file__).resolve().parent
EXPECTED = CASEPACK / "expected"
RAW = CASEPACK / "raw_measurements.csv"
PSI_CSV = EXPECTED / "psi.csv"
INV_JSON = EXPECTED / "invariants.json"

# Frozen targets (Tier-0 declarations)
REVENUE_TARGET = 200_000.0
EXPENSE_BUDGET = 150_000.0
CASHFLOW_TARGET = 40_000.0

# Weights: [revenue_perf, expense_control, gross_margin, cashflow_health]
WEIGHTS = [0.30, 0.25, 0.25, 0.20]

EPS = 1e-8


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def log_safe(c: float) -> float:
    return min(1.0 - EPS, max(EPS, c))


def embed(revenue: float, expenses: float, cogs: float, cashflow: float) -> list[float]:
    """Embed raw financials to [ε, 1-ε]^4."""
    c1 = clip01(min(revenue / REVENUE_TARGET, 1.0))
    c2 = clip01(min(EXPENSE_BUDGET / expenses, 1.0) if expenses > 0 else 1.0)
    c3 = clip01((revenue - cogs) / revenue if revenue > 0 else 0.0)
    c4 = clip01(min(cashflow / CASHFLOW_TARGET, 1.0))
    return [max(EPS, min(1.0 - EPS, c)) for c in [c1, c2, c3, c4]]


def main() -> None:
    EXPECTED.mkdir(parents=True, exist_ok=True)

    # Read raw measurements
    rows = []
    with RAW.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Generate psi.csv and invariant rows
    psi_rows: list[dict[str, object]] = []
    inv_rows: list[dict[str, object]] = []
    trace_coords: list[list[float]] = []

    for row in rows:
        t = int(row["t"])
        revenue = float(row["revenue"])
        expenses = float(row["expenses"])
        cogs = float(row["cogs"])
        cashflow = float(row["cashflow"])

        cs = embed(revenue, expenses, cogs, cashflow)
        trace_coords.append(cs)
        n = len(cs)

        # OOR flags
        oor = [
            revenue / REVENUE_TARGET > 1.0 or revenue < 0,
            expenses > EXPENSE_BUDGET,
            (revenue - cogs) / revenue < 0 if revenue > 0 else True,
            cashflow / CASHFLOW_TARGET > 1.0 or cashflow < 0,
        ]

        psi_rows.append(
            {
                "t": t,
                "c_1": cs[0],
                "c_2": cs[1],
                "c_3": cs[2],
                "c_4": cs[3],
                "oor_1": str(oor[0]).lower(),
                "oor_2": str(oor[1]).lower(),
                "oor_3": str(oor[2]).lower(),
                "oor_4": str(oor[3]).lower(),
                "miss_1": "false",
                "miss_2": "false",
                "miss_3": "false",
                "miss_4": "false",
            }
        )

        # Kernel invariants
        wts = WEIGHTS
        F = sum(wts[i] * cs[i] for i in range(n))
        omega = 1.0 - F

        # Entropy
        S = 0.0
        for i in range(n):
            c = log_safe(cs[i])
            S += wts[i] * (c * math.log(c) + (1.0 - c) * math.log(1.0 - c))
        S = -S

        # Curvature
        mean_c = sum(cs) / n
        var_c = sum((c - mean_c) ** 2 for c in cs) / n
        C = math.sqrt(var_c) / 0.5

        # Log-integrity and IC (unweighted — matching core kernel)
        kappa = sum(math.log(log_safe(cs[i])) for i in range(n))
        IC = math.exp(kappa)

        # Return time (lookback over trace)
        tau_R: float | str = "INF_REC"
        if t > 0:
            current = cs
            eta = 0.05
            H_rec = min(t, 64)
            for delta in range(1, H_rec + 1):
                past = trace_coords[t - delta]
                dist = math.sqrt(sum((current[i] - past[i]) ** 2 for i in range(n)))
                if dist < eta:
                    tau_R = delta
                    break

        # Regime classification — finance-calibrated thresholds
        # IC is unweighted product of coordinates, so lower baseline than weighted
        if IC < 0.10:
            regime_label = "Collapse"
            critical_overlay = True
        elif omega >= 0.30:
            regime_label = "Collapse"
            critical_overlay = False
        elif omega < 0.18 and F > 0.80:
            regime_label = "Stable"
            critical_overlay = False
        else:
            regime_label = "Watch"
            critical_overlay = False

        inv_rows.append(
            {
                "t": t,
                "omega": omega,
                "F": F,
                "S": S,
                "C": C,
                "tau_R": tau_R,
                "kappa": kappa,
                "IC": IC,
                "regime": {
                    "label": regime_label,
                    "critical_overlay": critical_overlay,
                },
                "kernel_optional": {"IC_min": min(cs)},
            }
        )

    # Write psi.csv
    with PSI_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "c_1",
                "c_2",
                "c_3",
                "c_4",
                "oor_1",
                "oor_2",
                "oor_3",
                "oor_4",
                "miss_1",
                "miss_2",
                "miss_3",
                "miss_4",
            ]
        )
        for pr in psi_rows:
            w.writerow(
                [
                    pr["t"],
                    pr["c_1"],
                    pr["c_2"],
                    pr["c_3"],
                    pr["c_4"],
                    pr["oor_1"],
                    pr["oor_2"],
                    pr["oor_3"],
                    pr["oor_4"],
                    pr["miss_1"],
                    pr["miss_2"],
                    pr["miss_3"],
                    pr["miss_4"],
                ]
            )

    # Write invariants.json
    inv = {
        "schema": "schemas/invariants.schema.json",
        "format": "tier1_invariants",
        "contract_id": "FINANCE.INTSTACK.v1",
        "closure_registry_id": "UMCP.CLOSURES.DEFAULT.v1",
        "canon": {
            "pre_doi": "10.5281/zenodo.17756705",
            "post_doi": "10.5281/zenodo.18072852",
            "weld_id": "W-2026-02-07-FINANCE-CONTINUITY",
            "timezone": "America/Chicago",
        },
        "rows": inv_rows,
        "notes": "Generated by casepacks/finance_continuity/generate_expected.py",
    }

    with INV_JSON.open("w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

    print(f"Wrote: {PSI_CSV}")
    print(f"Wrote: {INV_JSON}")
    print(f"Processed {len(rows)} months of financial data")

    # Summary
    for r in inv_rows:
        regime_info = r["regime"]
        regime_label = str(regime_info["label"]) if isinstance(regime_info, dict) else str(regime_info)
        tr = r["tau_R"]
        t_idx = int(str(r["t"]))
        print(
            f"  t={t_idx:2d} ({rows[t_idx]['month']}): "
            f"ω={r['omega']:.4f} F={r['F']:.4f} IC={r['IC']:.4f} "
            f"τ_R={tr!s:>7s} → {regime_label}"
        )


if __name__ == "__main__":
    main()
