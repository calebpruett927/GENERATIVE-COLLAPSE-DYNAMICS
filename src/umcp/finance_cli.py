"""
UMCP Finance CLI — Financial Continuity Validation Tool

Records, stores, and validates business financial continuity using the
UMCP kernel. Every month of financial data is a measurement; every report
is a re-demonstration of AXIOM-0.

Subcommands:
    init      Initialize a finance workspace with targets
    record    Record a month of financial data
    analyze   Compute kernel invariants and regime for all recorded months
    report    Generate a continuity report with seam accounting
    dashboard Launch the interactive Streamlit dashboard

Data flow:
    record → raw_finance.csv (append-only)
    analyze → finance_trace.csv + finance_invariants.csv + finance_ledger.csv
    report  → continuity_report.json + regime_summary

Cross-references:
    - contracts/FINANCE.INTSTACK.v1.yaml
    - closures/finance/finance_embedding.py
    - src/umcp/frozen_contract.py (kernel, seam accounting)
    - src/umcp/kernel_optimized.py (OPT-1..15)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

# UMCP core imports
from umcp.frozen_contract import (
    EPSILON,
    Regime,
    RegimeThresholds,
    classify_regime,
    compute_kernel,
    compute_tau_R,
    gamma_omega,
    cost_curvature,
    compute_seam_residual,
    compute_budget_delta_kappa,
    check_seam_pass,
    P_EXPONENT,
    TOL_SEAM,
)
from umcp.kernel_optimized import OptimizedKernelComputer

# Finance-specific imports — add closures root to path for direct import
_closures_root = str(Path(__file__).resolve().parents[2])
if _closures_root not in sys.path:
    sys.path.insert(0, _closures_root)
from closures.finance.finance_embedding import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_WEIGHTS,
    COORDINATE_NAMES,
    FinanceTargets,
    FinanceRecord,
    embed_finance,
)

# Finance-calibrated regime thresholds
# Business context: ω < 0.18 with F > 0.80 is "Stable"
# IC is unweighted product of coordinates, so critical threshold lower than default
FINANCE_THRESHOLDS = RegimeThresholds(
    omega_stable_max=0.18,
    F_stable_min=0.80,
    S_stable_max=0.80,
    C_stable_max=0.60,
    omega_watch_min=0.18,
    omega_watch_max=0.30,
    omega_collapse_min=0.30,
    I_critical_max=0.10,
)


# ============================================================================
# Constants
# ============================================================================

FINANCE_DIR_NAME = ".umcp-finance"
RAW_CSV = "raw_finance.csv"
TRACE_CSV = "finance_trace.csv"
INVARIANTS_CSV = "finance_invariants.csv"
LEDGER_CSV = "finance_ledger.csv"
CONFIG_JSON = "finance_config.json"
REPORT_JSON = "continuity_report.json"

# Finance-calibrated seam tolerance (monthly cadence has wider natural variation)
FINANCE_TOL_SEAM = 0.50

RAW_HEADERS = ["t", "month", "revenue", "expenses", "cogs", "cashflow", "recorded_utc"]
TRACE_HEADERS = ["t", "month", "c_1", "c_2", "c_3", "c_4", "oor_1", "oor_2", "oor_3", "oor_4"]
INVARIANT_HEADERS = ["t", "month", "F", "omega", "S", "C", "kappa", "IC", "tau_R", "regime", "critical"]
LEDGER_HEADERS = [
    "weld_id", "t0", "t1", "month_from", "month_to",
    "dk_ledger", "dk_budget", "D_omega", "D_C", "R", "tau_R",
    "residual_s", "tol_seam", "pass",
]


# ============================================================================
# Workspace management
# ============================================================================


def _get_workspace(workspace: str | None = None) -> Path:
    """Resolve the finance workspace directory."""
    if workspace:
        ws = Path(workspace)
    else:
        ws = Path.cwd() / FINANCE_DIR_NAME
    return ws


def _ensure_workspace(ws: Path) -> None:
    """Ensure workspace directory exists."""
    ws.mkdir(parents=True, exist_ok=True)


def _load_config(ws: Path) -> dict[str, Any]:
    """Load finance configuration."""
    config_path = ws / CONFIG_JSON
    if not config_path.exists():
        print(f"Error: No finance workspace found at {ws}")
        print("Run 'umcp-finance init' first.")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data)


def _get_targets(config: dict[str, Any]) -> FinanceTargets:
    """Extract targets from config."""
    t = config["targets"]
    return FinanceTargets(
        revenue_target=t["revenue_target"],
        expense_budget=t["expense_budget"],
        cashflow_target=t["cashflow_target"],
    )


# ============================================================================
# INIT subcommand
# ============================================================================


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a finance workspace with frozen targets."""
    ws = _get_workspace(args.workspace)
    _ensure_workspace(ws)

    config = {
        "contract": "FINANCE.INTSTACK.v1",
        "version": "1.0.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "targets": {
            "revenue_target": args.revenue_target,
            "expense_budget": args.expense_budget,
            "cashflow_target": args.cashflow_target,
        },
        "weights": args.weights if args.weights else DEFAULT_WEIGHTS,
        "cadence": "monthly",
        "frozen": True,
    }

    config_path = ws / CONFIG_JSON
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Create empty raw CSV
    raw_path = ws / RAW_CSV
    if not raw_path.exists():
        with open(raw_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(RAW_HEADERS)

    print(f"Finance workspace initialized at {ws}")
    print(f"  Contract: FINANCE.INTSTACK.v1")
    print(f"  Revenue target:  ${args.revenue_target:,.2f}")
    print(f"  Expense budget:  ${args.expense_budget:,.2f}")
    print(f"  Cash flow target: ${args.cashflow_target:,.2f}")
    print(f"  Weights: {config['weights']}")
    print()
    print("Next: record monthly data with 'umcp-finance record'")


# ============================================================================
# RECORD subcommand
# ============================================================================


def cmd_record(args: argparse.Namespace) -> None:
    """Record a month of financial data (append-only)."""
    ws = _get_workspace(args.workspace)
    config = _load_config(ws)
    raw_path = ws / RAW_CSV

    # Determine t index
    t = 0
    existing_months: set[str] = set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = int(row["t"]) + 1
                existing_months.add(row["month"])

    if args.month in existing_months:
        print(f"Error: Month {args.month} already recorded.")
        print("The ledger is append-only. To correct, record a new month with corrected values.")
        sys.exit(1)

    # Append to raw CSV
    with open(raw_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            t, args.month, args.revenue, args.expenses, args.cogs, args.cashflow,
            datetime.now(UTC).isoformat(),
        ])

    # Quick embed and report
    targets = _get_targets(config)
    record = FinanceRecord(
        month=args.month,
        revenue=args.revenue,
        expenses=args.expenses,
        cogs=args.cogs,
        cashflow=args.cashflow,
    )
    embedded = embed_finance(record, targets)

    print(f"Recorded: {args.month} (t={t})")
    print(f"  Revenue:  ${args.revenue:>12,.2f}  →  c₁ = {embedded.c[0]:.4f}")
    print(f"  Expenses: ${args.expenses:>12,.2f}  →  c₂ = {embedded.c[1]:.4f}")
    print(f"  COGS:     ${args.cogs:>12,.2f}  →  c₃ = {embedded.c[2]:.4f}")
    print(f"  Cash Flow:${args.cashflow:>12,.2f}  →  c₄ = {embedded.c[3]:.4f}")

    oor_count = sum(embedded.oor_flags)
    if oor_count > 0:
        print(f"  ⚠ {oor_count} dimension(s) out of range (clipped)")

    print()
    print("Run 'umcp-finance analyze' to compute invariants.")


# ============================================================================
# ANALYZE subcommand
# ============================================================================


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute kernel invariants and regime classification for all recorded months."""
    ws = _get_workspace(args.workspace)
    config = _load_config(ws)
    targets = _get_targets(config)
    weights_list = config.get("weights", DEFAULT_WEIGHTS)
    weights = np.array(weights_list, dtype=float)

    # Read raw data
    raw_path = ws / RAW_CSV
    if not raw_path.exists():
        print("Error: No financial data recorded yet.")
        sys.exit(1)

    records: list[dict[str, Any]] = []
    with open(raw_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            records.append(row)

    if not records:
        print("Error: No financial data recorded yet.")
        sys.exit(1)

    # Embed all months
    trace_data: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    inv_rows: list[dict[str, Any]] = []

    for row in records:
        t = int(row["t"])
        record = FinanceRecord(
            month=row["month"],
            revenue=float(row["revenue"]),
            expenses=float(row["expenses"]),
            cogs=float(row["cogs"]),
            cashflow=float(row["cashflow"]),
        )
        embedded = embed_finance(record, targets)
        trace_data.append(embedded.c)

        # Trace row
        trace_rows.append({
            "t": t, "month": row["month"],
            "c_1": embedded.c[0], "c_2": embedded.c[1],
            "c_3": embedded.c[2], "c_4": embedded.c[3],
            "oor_1": embedded.oor_flags[0], "oor_2": embedded.oor_flags[1],
            "oor_3": embedded.oor_flags[2], "oor_4": embedded.oor_flags[3],
        })

        # Compute tau_R
        tau_R_val = float("inf")
        if t > 0:
            trace_array = np.array(trace_data)
            tau_R_val = compute_tau_R(trace_array, t, eta=0.05, H_rec=min(t, 64), norm="L2")

        # Compute kernel invariants
        kernel = compute_kernel(embedded.c, weights, tau_R=tau_R_val, epsilon=EPSILON)

        # Classify regime
        regime = classify_regime(
            omega=kernel.omega,
            F=kernel.F,
            S=kernel.S,
            C=kernel.C,
            integrity=kernel.IC,
            thresholds=FINANCE_THRESHOLDS,
        )

        tau_display = "INF_REC" if math.isinf(kernel.tau_R) else str(int(kernel.tau_R))

        inv_rows.append({
            "t": t, "month": row["month"],
            "F": kernel.F, "omega": kernel.omega,
            "S": kernel.S, "C": kernel.C,
            "kappa": kernel.kappa, "IC": kernel.IC,
            "tau_R": tau_display,
            "regime": regime.value,
            "critical": regime == Regime.CRITICAL,
        })

    # Write trace CSV
    trace_path = ws / TRACE_CSV
    with open(trace_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_HEADERS)
        w.writeheader()
        w.writerows(trace_rows)

    # Write invariants CSV
    inv_path = ws / INVARIANTS_CSV
    with open(inv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INVARIANT_HEADERS)
        w.writeheader()
        w.writerows(inv_rows)

    # Compute seam accounting (ledger)
    ledger_rows: list[dict[str, Any]] = []
    for i in range(1, len(inv_rows)):
        prev = inv_rows[i - 1]
        curr = inv_rows[i]

        tau_R_curr = float("inf") if curr["tau_R"] == "INF_REC" else float(curr["tau_R"])

        # Ledger change
        dk_ledger = curr["kappa"] - prev["kappa"]

        # Budget components
        D_omega = gamma_omega(curr["omega"], p=P_EXPONENT, epsilon=EPSILON)
        D_C = cost_curvature(curr["C"])

        # Return credit
        R = 1.0 - curr["omega"] if not math.isinf(tau_R_curr) else 0.0

        # Budget
        dk_budget = compute_budget_delta_kappa(R, tau_R_curr if not math.isinf(tau_R_curr) else 0.0, D_omega, D_C)

        # Residual
        residual = compute_seam_residual(dk_budget, dk_ledger)

        # Pass check
        i_ratio = curr["IC"] / prev["IC"] if prev["IC"] > 0 else 0.0
        passed, _ = check_seam_pass(
            residual=residual,
            tau_R=tau_R_curr,
            I_ratio=i_ratio,
            delta_kappa=dk_ledger,
            tol_seam=FINANCE_TOL_SEAM,
        )

        weld_id = f"W-{curr['month']}"

        ledger_rows.append({
            "weld_id": weld_id,
            "t0": prev["t"], "t1": curr["t"],
            "month_from": prev["month"], "month_to": curr["month"],
            "dk_ledger": dk_ledger, "dk_budget": dk_budget,
            "D_omega": D_omega, "D_C": D_C,
            "R": R, "tau_R": curr["tau_R"],
            "residual_s": residual, "tol_seam": FINANCE_TOL_SEAM,
            "pass": "PASS" if passed else "FAIL",
        })

    # Write ledger
    ledger_path = ws / LEDGER_CSV
    with open(ledger_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_HEADERS)
        w.writeheader()
        w.writerows(ledger_rows)

    # Print summary
    print(f"Financial Analysis — {len(records)} months")
    print("=" * 80)
    print(f"{'Month':>8s}  {'ω':>7s}  {'F':>7s}  {'IC':>7s}  {'τ_R':>7s}  {'Regime':>10s}")
    print("-" * 80)
    for r in inv_rows:
        regime_str = r["regime"]
        if regime_str == "CRITICAL":
            regime_str = "⚠ CRITICAL"
        elif regime_str == "COLLAPSE":
            regime_str = "▼ COLLAPSE"
        elif regime_str == "WATCH":
            regime_str = "◆ WATCH"
        else:
            regime_str = "● STABLE"
        print(f"{r['month']:>8s}  {r['omega']:7.4f}  {r['F']:7.4f}  "
              f"{r['IC']:7.4f}  {r['tau_R']:>7s}  {regime_str:>10s}")

    print()
    if ledger_rows:
        print("Seam Accounting")
        print("-" * 80)
        pass_count = sum(1 for lr in ledger_rows if lr["pass"] == "PASS")
        fail_count = len(ledger_rows) - pass_count
        print(f"  Transitions: {len(ledger_rows)}")
        print(f"  PASS: {pass_count}  |  FAIL: {fail_count}")

        for lr in ledger_rows:
            status = "✓" if lr["pass"] == "PASS" else "✗"
            print(f"  {status} {lr['month_from']} → {lr['month_to']}: "
                  f"Δκ={lr['dk_ledger']:+.5f}  s={lr['residual_s']:+.5f}  "
                  f"τ_R={lr['tau_R']}")

    print()
    print(f"Trace:      {trace_path}")
    print(f"Invariants: {inv_path}")
    print(f"Ledger:     {ledger_path}")


# ============================================================================
# REPORT subcommand
# ============================================================================


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a JSON continuity report."""
    ws = _get_workspace(args.workspace)
    config = _load_config(ws)

    # Read invariants
    inv_path = ws / INVARIANTS_CSV
    if not inv_path.exists():
        print("Error: Run 'umcp-finance analyze' first.")
        sys.exit(1)

    inv_rows: list[dict[str, Any]] = []
    with open(inv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            inv_rows.append(row)

    # Read ledger
    ledger_path = ws / LEDGER_CSV
    ledger_rows: list[dict[str, Any]] = []
    if ledger_path.exists():
        with open(ledger_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ledger_rows.append(row)

    # Regime counts
    regime_counts: dict[str, int] = {}
    for r in inv_rows:
        regime = r["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # Continuity assessment
    pass_count = sum(1 for lr in ledger_rows if lr["pass"] == "PASS")
    total_seams = len(ledger_rows)
    continuity_ratio = pass_count / total_seams if total_seams > 0 else 0.0

    # Overall verdict
    last_regime = inv_rows[-1]["regime"] if inv_rows else "UNKNOWN"
    if continuity_ratio >= 0.8 and last_regime in ("STABLE", "WATCH"):
        verdict = "CONFORMANT"
    elif last_regime == "COLLAPSE" or last_regime == "CRITICAL":
        verdict = "NONCONFORMANT"
    else:
        verdict = "NON_EVALUABLE"

    report = {
        "contract": "FINANCE.INTSTACK.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "months_analyzed": len(inv_rows),
        "targets": config["targets"],
        "weights": config.get("weights", DEFAULT_WEIGHTS),
        "verdict": verdict,
        "regime_summary": regime_counts,
        "continuity": {
            "total_transitions": total_seams,
            "pass": pass_count,
            "fail": total_seams - pass_count,
            "continuity_ratio": continuity_ratio,
        },
        "current_state": {
            "month": inv_rows[-1]["month"] if inv_rows else None,
            "regime": last_regime,
            "omega": float(inv_rows[-1]["omega"]) if inv_rows else None,
            "F": float(inv_rows[-1]["F"]) if inv_rows else None,
            "IC": float(inv_rows[-1]["IC"]) if inv_rows else None,
        },
        "invariants": inv_rows,
        "seam_ledger": ledger_rows,
    }

    report_path = ws / REPORT_JSON
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Continuity Report — {report['months_analyzed']} months")
    print("=" * 60)
    print(f"  Verdict:    {verdict}")
    print(f"  Regime now: {last_regime}")
    print(f"  Continuity: {pass_count}/{total_seams} seams passed ({continuity_ratio:.0%})")
    print(f"  Regimes:    {regime_counts}")
    print(f"\n  Report: {report_path}")


# ============================================================================
# DASHBOARD subcommand
# ============================================================================


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit finance dashboard."""
    ws = _get_workspace(args.workspace)
    config_path = ws / CONFIG_JSON
    if not config_path.exists():
        print("Error: No finance workspace found. Run 'umcp-finance init' first.")
        sys.exit(1)

    dashboard_path = Path(__file__).parent / "finance_dashboard.py"
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    port = args.port if hasattr(args, "port") else 8502

    try:
        import streamlit as _st  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("Error: Streamlit not installed. Install with: pip install umcp[viz]")
        sys.exit(1)

    os.environ["UMCP_FINANCE_WORKSPACE"] = str(ws)
    os.system(f"streamlit run {dashboard_path} --server.port {port}")


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="umcp-finance",
        description="UMCP Financial Continuity Tool — Record, store, and validate business financial health.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = sub.add_parser("init", help="Initialize a finance workspace with frozen targets")
    p_init.add_argument("--revenue-target", type=float, required=True, help="Monthly revenue target ($)")
    p_init.add_argument("--expense-budget", type=float, required=True, help="Monthly expense budget ($)")
    p_init.add_argument("--cashflow-target", type=float, required=True, help="Monthly cash flow target ($)")
    p_init.add_argument("--weights", type=float, nargs=4, default=None,
                        help="Coordinate weights [rev, exp, margin, cf] (default: 0.30 0.25 0.25 0.20)")
    p_init.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    # record
    p_rec = sub.add_parser("record", help="Record a month of financial data")
    p_rec.add_argument("--month", type=str, required=True, help="Month in YYYY-MM format")
    p_rec.add_argument("--revenue", type=float, required=True, help="Monthly revenue ($)")
    p_rec.add_argument("--expenses", type=float, required=True, help="Monthly total expenses ($)")
    p_rec.add_argument("--cogs", type=float, required=True, help="Cost of goods sold ($)")
    p_rec.add_argument("--cashflow", type=float, required=True, help="Operating cash flow ($)")
    p_rec.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    # analyze
    p_ana = sub.add_parser("analyze", help="Compute kernel invariants and regime classification")
    p_ana.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    # report
    p_rep = sub.add_parser("report", help="Generate a continuity report")
    p_rep.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch interactive dashboard")
    p_dash.add_argument("--workspace", type=str, default=None, help="Workspace directory")
    p_dash.add_argument("--port", type=int, default=8502, help="Dashboard port (default: 8502)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
