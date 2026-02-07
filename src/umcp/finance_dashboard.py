"""
UMCP Finance Dashboard

Interactive Streamlit dashboard for financial continuity visualization.
Displays kernel invariants, regime timeline, seam accounting, and
continuity metrics for business financial data.

Launch:
    umcp-finance dashboard
    streamlit run src/umcp/finance_dashboard.py

Cross-references:
    - src/umcp/finance_cli.py (data generation)
    - contracts/FINANCE.INTSTACK.v1.yaml
    - closures/finance/finance_embedding.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# Optional visualization dependencies (matches dashboard.py pattern)
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false
# pyright: reportAssignmentType=false
_has_viz_deps = False
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    _has_viz_deps = True
except ImportError:
    np = None
    pd = None
    go = None
    st = None
    make_subplots = None


FINANCE_DIR_NAME = ".umcp-finance"

REGIME_COLORS = {
    "STABLE": "#4caf50",
    "WATCH": "#ff9800",
    "COLLAPSE": "#f44336",
    "CRITICAL": "#9c27b0",
}

COORDINATE_LABELS = {
    "c_1": "Revenue Performance",
    "c_2": "Expense Control",
    "c_3": "Gross Margin",
    "c_4": "Cash Flow Health",
}


def _get_workspace() -> Path:
    """Resolve workspace from env or cwd."""
    env_ws = os.environ.get("UMCP_FINANCE_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    return Path.cwd() / FINANCE_DIR_NAME


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load CSV as list of dicts."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data)


def main() -> None:
    if not _has_viz_deps:
        print("Error: Streamlit not installed. Install with: pip install umcp[viz]")
        sys.exit(1)

    # Narrow types for Pylance after _has_viz_deps guard
    assert st is not None
    assert go is not None
    assert make_subplots is not None

    st.set_page_config(
        page_title="UMCP Finance — Continuity Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 UMCP Financial Continuity Dashboard")
    st.caption("Contract: FINANCE.INTSTACK.v1 | Axiom: Only what returns through collapse is real")

    ws = _get_workspace()
    config = _load_json(ws / "finance_config.json")
    if not config:
        st.error("No finance workspace found. Run `umcp-finance init` first.")
        return

    # Load data
    invariants = _load_csv(ws / "finance_invariants.csv")
    trace = _load_csv(ws / "finance_trace.csv")
    ledger = _load_csv(ws / "finance_ledger.csv")
    report = _load_json(ws / "continuity_report.json")

    if not invariants:
        st.warning("No analysis data. Run `umcp-finance analyze` first.")
        return

    # ---- Header metrics ----
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)

    last = invariants[-1]
    verdict = report.get("verdict", "N/A")
    verdict_color = {"CONFORMANT": "🟢", "NONCONFORMANT": "🔴", "NON_EVALUABLE": "🟡"}.get(verdict, "⚪")

    col1.metric("Verdict", f"{verdict_color} {verdict}")
    col2.metric("Current Regime", last["regime"])
    col3.metric("Drift (ω)", f"{float(last['omega']):.4f}")
    col4.metric("Fidelity (F)", f"{float(last['F']):.4f}")
    col5.metric("Integrity (IC)", f"{float(last['IC']):.4f}")

    # ---- Targets ----
    st.divider()
    targets = config.get("targets", {})
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Revenue Target", f"${targets.get('revenue_target', 0):,.0f}")
    tc2.metric("Expense Budget", f"${targets.get('expense_budget', 0):,.0f}")
    tc3.metric("Cash Flow Target", f"${targets.get('cashflow_target', 0):,.0f}")

    if go is None or pd is None:
        st.warning("Install plotly and pandas for charts: `pip install umcp[viz]`")
        # Fallback: show table
        st.subheader("Invariants")
        st.table(invariants)
        return

    # ---- Regime Timeline ----
    st.divider()
    st.subheader("Regime Timeline")

    df_inv = pd.DataFrame(invariants)
    for col in ["F", "omega", "S", "C", "kappa", "IC"]:
        df_inv[col] = pd.to_numeric(df_inv[col], errors="coerce")

    fig_regime = go.Figure()
    regime_y = {"STABLE": 1, "WATCH": 2, "COLLAPSE": 3, "CRITICAL": 4}
    colors = [REGIME_COLORS.get(r, "#999") for r in df_inv["regime"]]
    fig_regime.add_trace(go.Bar(
        x=df_inv["month"],
        y=[regime_y.get(r, 0) for r in df_inv["regime"]],
        marker_color=colors,
        text=df_inv["regime"],
        textposition="auto",
        hovertemplate="Month: %{x}<br>Regime: %{text}<extra></extra>",
    ))
    fig_regime.update_layout(
        yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=["STABLE", "WATCH", "COLLAPSE", "CRITICAL"]),
        height=250,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_regime, use_container_width=True)

    # ---- Kernel Invariants ----
    st.subheader("Kernel Invariants")

    fig_inv = make_subplots(rows=2, cols=2,
                            subplot_titles=("Drift (ω)", "Fidelity (F)", "Entropy (S)", "Integrity (IC)"))

    fig_inv.add_trace(go.Scatter(x=df_inv["month"], y=df_inv["omega"], mode="lines+markers",
                                  name="ω", line=dict(color="#f44336")), row=1, col=1)
    fig_inv.add_hline(y=0.038, line_dash="dash", line_color="green", annotation_text="Stable threshold",
                      row=1, col=1)
    fig_inv.add_hline(y=0.30, line_dash="dash", line_color="red", annotation_text="Collapse threshold",
                      row=1, col=1)

    fig_inv.add_trace(go.Scatter(x=df_inv["month"], y=df_inv["F"], mode="lines+markers",
                                  name="F", line=dict(color="#4caf50")), row=1, col=2)

    fig_inv.add_trace(go.Scatter(x=df_inv["month"], y=df_inv["S"], mode="lines+markers",
                                  name="S", line=dict(color="#2196f3")), row=2, col=1)

    fig_inv.add_trace(go.Scatter(x=df_inv["month"], y=df_inv["IC"], mode="lines+markers",
                                  name="IC", line=dict(color="#9c27b0")), row=2, col=2)
    fig_inv.add_hline(y=0.30, line_dash="dash", line_color="red", annotation_text="Critical threshold",
                      row=2, col=2)

    fig_inv.update_layout(height=500, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_inv, use_container_width=True)

    # ---- Coordinates ----
    st.subheader("Embedded Coordinates")
    if trace:
        df_trace = pd.DataFrame(trace)
        for col in ["c_1", "c_2", "c_3", "c_4"]:
            df_trace[col] = pd.to_numeric(df_trace[col], errors="coerce")

        fig_coords = go.Figure()
        for col, label in COORDINATE_LABELS.items():
            fig_coords.add_trace(go.Scatter(
                x=df_trace["month"], y=df_trace[col],
                mode="lines+markers", name=label,
            ))
        fig_coords.update_layout(
            yaxis_title="Coordinate value [0, 1]",
            height=350, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_coords, use_container_width=True)

    # ---- Seam Accounting ----
    st.divider()
    st.subheader("Seam Accounting (Month-over-Month Continuity)")

    if ledger:
        df_ledger = pd.DataFrame(ledger)
        for col in ["dk_ledger", "dk_budget", "residual_s", "D_omega", "D_C"]:
            df_ledger[col] = pd.to_numeric(df_ledger[col], errors="coerce")

        pass_count = (df_ledger["pass"] == "PASS").sum()
        fail_count = (df_ledger["pass"] == "FAIL").sum()

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Total Seams", len(df_ledger))
        lc2.metric("PASS", int(pass_count))
        lc3.metric("FAIL", int(fail_count))

        # Residual chart
        fig_seam = go.Figure()
        colors_seam = ["#4caf50" if p == "PASS" else "#f44336" for p in df_ledger["pass"]]
        fig_seam.add_trace(go.Bar(
            x=df_ledger["month_to"],
            y=df_ledger["residual_s"],
            marker_color=colors_seam,
            text=df_ledger["pass"],
            hovertemplate="To: %{x}<br>Residual: %{y:.5f}<br>%{text}<extra></extra>",
        ))
        fig_seam.add_hline(y=0.005, line_dash="dash", line_color="orange", annotation_text="+tol_seam")
        fig_seam.add_hline(y=-0.005, line_dash="dash", line_color="orange", annotation_text="-tol_seam")
        fig_seam.update_layout(
            yaxis_title="Seam residual (s)",
            height=300, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_seam, use_container_width=True)

        # Ledger table
        st.dataframe(df_ledger, use_container_width=True, hide_index=True)

    else:
        st.info("No seam data available. Need at least 2 months of data.")

    # ---- Footer ----
    st.divider()
    st.caption(
        "UMCP Finance v1.0 | Contract: FINANCE.INTSTACK.v1 | "
        "Workspace: " + str(ws) + " | "
        "Axiom-0: What returns through collapse is real"
    )


if __name__ == "__main__":
    main()
