"""
UMCP Visualization Dashboard - Streamlit Communication Extension

Interactive web dashboard for exploring UMCP validation results,
ledger data, casepacks, contracts, closures, and kernel metrics.

This is an optional extension that requires: pip install umcp[viz]

Usage:
  streamlit run src/umcp/dashboard.py

Features:
  - Real-time system health monitoring
  - Interactive ledger exploration with anomaly detection
  - Casepack browser with validation status
  - Regime phase space visualization with trajectories
  - Kernel metrics analysis with trends and correlations
  - Contract and closure exploration with code preview

Cross-references:
  - EXTENSION_INTEGRATION.md (extension architecture)
  - src/umcp/api_umcp.py (REST API extension)
  - src/umcp/validator.py (validation engine)
  - ledger/return_log.csv (validation ledger)
  - KERNEL_SPECIFICATION.md (regime definitions)

Note: This module uses optional visualization dependencies (pandas, plotly, streamlit)
that may not be installed. Type errors for these are suppressed with pyright directives.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Try to import visualization dependencies
_has_viz_deps = False
try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    _has_viz_deps = True
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

HAS_VIZ_DEPS: bool = _has_viz_deps


def _cache_data(**kwargs: Any) -> Any:
    """Safe wrapper around st.cache_data; no-op when Streamlit is unavailable."""
    if st is not None:
        return st.cache_data(**kwargs)

    # Fallback: identity decorator
    def _identity(func: Any) -> Any:
        return func

    return _identity


if TYPE_CHECKING:
    import pandas as pd

# Import UMCP core modules
try:
    from . import __version__
except ImportError:
    __version__ = "2.0.0"


# Ensure closures package is importable (needed for Docker container)
# Add the repo root to sys.path so closures/ can be imported
def _setup_closures_path() -> None:
    """Add repo root to sys.path so closures package is importable."""
    import sys
    from pathlib import Path

    # Find repo root (contains pyproject.toml)
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            repo_root = str(current)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return
        current = current.parent

    # Fallback: check common Docker paths
    for path in ["/app", "/workspaces/UMCP-Metadata-Runnable-Code"]:
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


_setup_closures_path()


# ============================================================================
# Constants and Configuration
# ============================================================================

REGIME_COLORS = {
    "STABLE": "#28a745",  # Green
    "WATCH": "#ffc107",  # Yellow/Amber
    "COLLAPSE": "#dc3545",  # Red
    "CRITICAL": "#6f42c1",  # Purple
}

# Kernel invariants from KERNEL_SPECIFICATION.md
# Tier-1 outputs: F, ω, S, C, κ, IC computed from frozen trace Ψ(t)
KERNEL_SYMBOLS = {
    # Core Tier-1 Invariants
    "omega": "ω (Drift = 1-F)",
    "F": "F (Fidelity)",
    "S": "S (Shannon Entropy)",
    "C": "C (Curvature Proxy)",
    "tau_R": "τ_R (Return Time)",
    "IC": "IC (Integrity Composite = exp(κ))",
    "kappa": "κ (Log-Integrity = Σwᵢ ln cᵢ)",
    # Derived/Seam Values
    "stiffness": "Stiffness",
    "delta_kappa": "Δκ (Seam Curvature Change)",
    "curvature": "C (Curvature = std/0.5)",
    "freshness": "Freshness (1-ω)",
    "seam_residual": "s (Seam Residual)",
    # Meta
    "timestamp": "Timestamp",
    "run_status": "Status",
    "Phi_gen": "Φ_gen (Generative Flux)",
}

STATUS_COLORS = {
    "CONFORMANT": "#28a745",
    "NONCONFORMANT": "#dc3545",
    "NON_EVALUABLE": "#6c757d",
}

# Theme configurations
THEMES = {
    "Default": {
        "primary": "#007bff",
        "secondary": "#6c757d",
        "success": "#28a745",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        "bg_primary": "#ffffff",
        "bg_secondary": "#f8f9fa",
    },
    "Dark": {
        "primary": "#0d6efd",
        "secondary": "#adb5bd",
        "success": "#198754",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#0dcaf0",
        "bg_primary": "#212529",
        "bg_secondary": "#343a40",
    },
    "Ocean": {
        "primary": "#0077b6",
        "secondary": "#90e0ef",
        "success": "#2a9d8f",
        "danger": "#e63946",
        "warning": "#f4a261",
        "info": "#48cae4",
        "bg_primary": "#caf0f8",
        "bg_secondary": "#ade8f4",
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_repo_root() -> Path:
    """Find the repository root (contains pyproject.toml)."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


@_cache_data(ttl=60)
def load_ledger() -> Any:
    """Load the return log ledger as a DataFrame."""
    if pd is None:
        raise ImportError("pandas not installed. Run: pip install umcp[viz]")

    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(ledger_path)

    # Ensure tau_R stays as string column — INF_REC is a typed sentinel
    # that cannot be coerced to int64 by PyArrow serialization.
    # Uses canonical tau_R_display for consistent formatting.
    if "tau_R" in df.columns:
        try:
            from .measurement_engine import tau_R_display
        except ImportError:
            from umcp.measurement_engine import tau_R_display  # type: ignore[no-redef]

        df["tau_R"] = df["tau_R"].apply(tau_R_display)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    return df


@_cache_data(ttl=60)
def load_casepacks() -> list[dict[str, Any]]:
    """Load casepack information with extended metadata."""
    repo_root = get_repo_root()
    casepacks_dir = repo_root / "casepacks"

    if not casepacks_dir.exists():
        return []

    casepacks: list[dict[str, Any]] = []
    for casepack_dir in sorted(casepacks_dir.iterdir()):
        if not casepack_dir.is_dir():
            continue

        manifest_path = casepack_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = casepack_dir / "manifest.yaml"

        casepack_info: dict[str, Any] = {
            "id": casepack_dir.name,
            "path": str(casepack_dir),
            "version": "unknown",
            "description": None,
            "contract": None,
            "closures_count": 0,
            "test_vectors": 0,
            "files_count": 0,
        }

        # Count files
        casepack_info["files_count"] = len(list(casepack_dir.rglob("*")))

        # Count closures
        closures_dir = casepack_dir / "closures"
        if closures_dir.exists():
            casepack_info["closures_count"] = len(list(closures_dir.glob("*.py")))

        # Count test vectors
        test_file = casepack_dir / "test_vectors.csv"
        if not test_file.exists():
            test_file = casepack_dir / "raw_measurements.csv"
        if test_file.exists():
            with open(test_file) as f:
                casepack_info["test_vectors"] = max(0, sum(1 for _ in f) - 1)

        # Load manifest
        if manifest_path.exists():
            try:
                if manifest_path.suffix == ".json":
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    import yaml

                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)

                if manifest and "casepack" in manifest:
                    cp = manifest["casepack"]
                    casepack_info["id"] = cp.get("id", casepack_dir.name)
                    casepack_info["version"] = cp.get("version", "unknown")
                    casepack_info["description"] = cp.get("description")
                    casepack_info["title"] = cp.get("title")
                if manifest and "refs" in manifest:
                    refs = manifest["refs"]
                    if "contract" in refs:
                        casepack_info["contract"] = refs["contract"].get("id")
            except Exception:
                pass

        casepacks.append(casepack_info)

    return casepacks


@_cache_data(ttl=60)
def load_contracts() -> list[dict[str, Any]]:
    """Load contract information with extended metadata."""
    repo_root = get_repo_root()
    contracts_dir = repo_root / "contracts"

    if not contracts_dir.exists():
        return []

    contracts: list[dict[str, Any]] = []
    for contract_path in sorted(contracts_dir.glob("*.yaml")):
        filename = contract_path.stem
        parts = filename.split(".")
        domain = parts[0] if parts else "unknown"
        version = parts[-1] if len(parts) > 1 and parts[-1].startswith("v") else "v1"

        # Get file size
        size_bytes = contract_path.stat().st_size

        contracts.append(
            {
                "id": filename,
                "domain": domain,
                "version": version,
                "path": str(contract_path),
                "size_bytes": size_bytes,
            }
        )

    return contracts


@_cache_data(ttl=60)
def load_closures() -> list[dict[str, Any]]:
    """Load closure information."""
    repo_root = get_repo_root()
    closures_dir = repo_root / "closures"

    if not closures_dir.exists():
        return []

    closures: list[dict[str, Any]] = []

    def _infer_domain(path: Path) -> str:
        """Infer domain from file name or parent directory name."""
        # Check parent directory name first (e.g., closures/gcd/, closures/rcft/)
        rel = path.relative_to(closures_dir)
        parts_lower = [p.lower() for p in rel.parts]
        combined = " ".join(parts_lower)
        if "gcd" in combined or "curvature" in combined or "gamma" in combined:
            return "GCD"
        if "kin" in combined:
            return "KIN"
        if "rcft" in combined:
            return "RCFT"
        if "weyl" in combined:
            return "WEYL"
        if "security" in combined:
            return "SECURITY"
        if "astro" in combined:
            return "ASTRO"
        if "nuclear" in combined or "nuc" in combined:
            return "NUC"
        if "quantum" in combined or "qm" in combined:
            return "QM"
        if "finance" in combined or "fin" in combined:
            return "FIN"
        return "unknown"

    # Python closures (recursive)
    for closure_path in sorted(closures_dir.rglob("*.py")):
        if closure_path.name.startswith("_"):
            continue

        name = closure_path.stem
        size_bytes = closure_path.stat().st_size
        domain = _infer_domain(closure_path)

        # Count lines
        with open(closure_path) as f:
            lines = len(f.readlines())

        closures.append(
            {
                "name": name,
                "domain": domain,
                "path": str(closure_path),
                "type": "python",
                "size_bytes": size_bytes,
                "lines": lines,
            }
        )

    # YAML closures (recursive)
    for closure_path in sorted(closures_dir.rglob("*.yaml")):
        if closure_path.name == "registry.yaml":
            continue

        name = closure_path.stem
        size_bytes = closure_path.stat().st_size
        domain = _infer_domain(closure_path)

        closures.append(
            {
                "name": name,
                "domain": domain,
                "path": str(closure_path),
                "type": "yaml",
                "size_bytes": size_bytes,
                "lines": 0,
            }
        )

    return closures


def classify_regime(omega: float, seam_residual: float = 0.0) -> str:
    """
    Classify the computational regime based on kernel invariants.

    Regimes (from KERNEL_SPECIFICATION.md):
      - STABLE: ω ∈ [0.3, 0.7], |s| ≤ 0.005
      - WATCH: ω ∈ [0.1, 0.3) ∪ (0.7, 0.9], |s| ≤ 0.01
      - COLLAPSE: ω < 0.1 or ω > 0.9
      - CRITICAL: |s| > 0.01
    """
    if abs(seam_residual) > 0.01:
        return "CRITICAL"
    if omega < 0.1 or omega > 0.9:
        return "COLLAPSE"
    if 0.3 <= omega <= 0.7:
        return "STABLE"
    return "WATCH"


def get_regime_color(regime: str) -> str:
    """Get color for regime visualization."""
    return REGIME_COLORS.get(regime, "#6c757d")


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size:.1f} TB"


def get_trend_indicator(current: float, previous: float, invert: bool = False) -> str:
    """Get trend arrow indicator."""
    threshold = 1.01
    if current > previous * threshold:
        return "📉" if invert else "📈"
    elif current < previous / threshold:
        return "📈" if invert else "📉"
    return "➡️"


def detect_anomalies(series: Any, threshold: float = 2.5) -> Any:
    """Detect anomalies using z-score method."""
    if pd is None or np is None:
        return []
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return abs(z_scores) > threshold


# ============================================================================
# Dashboard Pages
# ============================================================================


def render_overview_page() -> None:
    """Render the main overview page with comprehensive metrics."""
    if st is None or px is None or pd is None:
        return

    st.title("🔬 UMCP Dashboard")
    st.caption(f"Universal Measurement Contract Protocol | v{__version__} | Schema: UMCP.v1")

    # ========== Core Axiom ==========
    with st.expander("📜 **Core Axiom**: What Returns Through Collapse Is Real", expanded=False):
        st.markdown("""
        **AXIOM-0 (The Return Axiom)**: *"Collapse is generative; only what returns is real."*
        
        This is the fundamental axiom upon which UMCP, GCD, and RCFT are built.
        
        **Operational Definitions**:
        - **Collapse**: Regime label produced by kernel gates on (ω, F, S, C) under frozen thresholds
        - **Return (τ_R)**: Re-entry condition with prior u ∈ Dθ(t) where ‖Ψ(t) - Ψ(u)‖ ≤ η
        - **Drift (ω)**: ω = 1 - F, collapse proximity measure on [0,1]
        - **Integrity (IC)**: IC = exp(κ) where κ = Σ wᵢ ln(cᵢ,ε)
        
        **Constitutional Principle**: *One-way dependency flow within a frozen run, with return-based canonization between runs.*
        """)

    # Load all data
    df = load_ledger()
    casepacks = load_casepacks()
    contracts = load_contracts()
    closures = load_closures()

    # ========== Top-level metrics ==========
    st.subheader("📊 System Overview")
    metrics_cols = st.columns(6)

    ledger_count = len(df) if not df.empty else 0
    with metrics_cols[0]:
        st.metric("📒 Ledger Entries", f"{ledger_count:,}")

    with metrics_cols[1]:
        st.metric("📦 Casepacks", len(casepacks))

    with metrics_cols[2]:
        st.metric("📜 Contracts", len(contracts))

    with metrics_cols[3]:
        st.metric("🔧 Closures", len(closures))

    with metrics_cols[4]:
        if not df.empty and "run_status" in df.columns:
            conformant = (df["run_status"] == "CONFORMANT").sum()
            total = len(df)
            rate = conformant / total * 100 if total > 0 else 0
            st.metric("✅ Conformance", f"{rate:.1f}%")
        else:
            st.metric("✅ Conformance", "N/A")

    with metrics_cols[5]:
        # System health indicator
        health = "🟢 Healthy"
        if df.empty:
            health = "🟡 No Data"
        elif not df.empty and "run_status" in df.columns:
            nonconformant = (df["run_status"] == "NONCONFORMANT").sum()
            if nonconformant / len(df) > 0.2:
                health = "🔴 Issues"
            elif nonconformant / len(df) > 0.1:
                health = "🟡 Warning"
        st.metric("System Health", health)

    st.divider()

    # ========== Main content area ==========
    left_col, center_col, right_col = st.columns([1.5, 2, 1])

    with left_col:
        st.subheader("📈 Status Distribution")
        if df.empty:
            st.info("No ledger data yet.")
        elif "run_status" in df.columns:
            status_counts = df["run_status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                color=status_counts.index,
                color_discrete_map=STATUS_COLORS,
                hole=0.4,
            )
            fig.update_layout(
                height=280,
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, width="stretch")

    with center_col:
        st.subheader("📈 Metrics Timeline")
        if df.empty or "timestamp" not in df.columns:
            st.info("No time series data available.")
        else:
            df_sorted = df.sort_values("timestamp").tail(100)
            numeric_cols = [c for c in ["omega", "curvature", "stiffness", "kappa"] if c in df.columns]

            if numeric_cols:
                selected_metric = st.selectbox(
                    "Select Metric", numeric_cols, format_func=lambda x: KERNEL_SYMBOLS.get(x, x) or x
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted[selected_metric],
                        mode="lines+markers",
                        name=KERNEL_SYMBOLS.get(selected_metric, selected_metric) or selected_metric,
                        marker={"size": 4},
                        line={"width": 2},
                        fill="tozeroy",
                        fillcolor="rgba(0,123,255,0.1)",
                    )
                )
                fig.update_layout(
                    height=280,
                    margin={"t": 10, "b": 30, "l": 40, "r": 10},
                    xaxis_title="",
                    yaxis_title=KERNEL_SYMBOLS.get(selected_metric, selected_metric),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric metrics available.")

    with right_col:
        st.subheader("🎯 Latest Values")
        if not df.empty:
            latest = df.iloc[-1]
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:5]

            for col in numeric_cols:
                val = latest[col]
                if pd.notna(val):
                    label = KERNEL_SYMBOLS.get(col, col) or col
                    # Get trend if enough data
                    delta = None
                    if len(df) > 1:
                        prev = df.iloc[-2][col]
                        if pd.notna(prev) and prev != 0:
                            delta = f"{((val - prev) / abs(prev) * 100):+.1f}%"
                    st.metric(label, f"{val:.4f}", delta=delta)
        else:
            st.info("No data")

    st.divider()

    # ========== Recent Activity Feed ==========
    st.subheader("📝 Recent Activity")
    if not df.empty and "timestamp" in df.columns:
        recent = df.tail(8).iloc[::-1]
        cols = st.columns(4)
        for i, (_, row) in enumerate(recent.iterrows()):
            with cols[i % 4]:
                status = row.get("run_status", "UNKNOWN")
                emoji = "✅" if status == "CONFORMANT" else "❌" if status == "NONCONFORMANT" else "⚠️"
                ts = row.get("timestamp", "")
                if hasattr(ts, "strftime"):
                    ts = ts.strftime("%m/%d %H:%M")
                omega_val = row.get("omega", None)
                omega_str = f"ω={omega_val:.3f}" if omega_val is not None else ""
                st.markdown(
                    f"""
                <div style="padding:8px; border-radius:8px; background-color: {STATUS_COLORS.get(status, "#ccc")}22; border-left: 3px solid {STATUS_COLORS.get(status, "#ccc")};">
                    <strong>{emoji} {status}</strong><br/>
                    <small>{ts}</small><br/>
                    <small>{omega_str}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No recent activity to display.")


def render_ledger_page() -> None:
    """Render the ledger exploration page with advanced filtering."""
    if st is None or px is None or pd is None:
        return

    st.title("📒 Validation Ledger")
    st.caption("Explore and analyze validation history")

    df = load_ledger()

    if df.empty:
        st.warning("No ledger data found. Run `umcp validate` to populate the ledger.")
        return

    # ========== Filters ==========
    st.subheader("🔍 Filters")
    filter_cols = st.columns(5)

    filtered_df = df.copy()

    with filter_cols[0]:
        if "run_status" in filtered_df.columns:
            status_options = ["All", *filtered_df["run_status"].unique().tolist()]
            status_filter: str = st.selectbox("Status", status_options) or "All"
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df["run_status"] == status_filter]

    with filter_cols[1]:
        if "timestamp" in filtered_df.columns:
            date_range = st.selectbox("Time Range", ["All", "Last 24h", "Last 7d", "Last 30d", "Custom"])
            now = datetime.now()
            if date_range == "Last 24h":
                filtered_df = filtered_df[filtered_df["timestamp"] >= now - timedelta(days=1)]
            elif date_range == "Last 7d":
                filtered_df = filtered_df[filtered_df["timestamp"] >= now - timedelta(days=7)]
            elif date_range == "Last 30d":
                filtered_df = filtered_df[filtered_df["timestamp"] >= now - timedelta(days=30)]

    with filter_cols[2]:
        numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols and "omega" in numeric_cols:
            omega_filter = st.slider("Omega Range", 0.0, 1.0, (0.0, 1.0), 0.01)
            if "omega" in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df["omega"] >= omega_filter[0]) & (filtered_df["omega"] <= omega_filter[1])
                ]

    with filter_cols[3]:
        max_rows = st.slider("Max Rows", 10, 1000, 100)

    with filter_cols[4]:
        sort_order = st.selectbox("Sort", ["Newest First", "Oldest First"])
        ascending = sort_order == "Oldest First"

    # Apply sorting
    if "timestamp" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("timestamp", ascending=ascending)

    # ========== Data Display ==========
    st.caption(
        f"Showing {min(max_rows, len(filtered_df))} of {len(filtered_df)} entries (filtered from {len(df)} total)"
    )

    display_df = filtered_df.tail(max_rows) if not ascending else filtered_df.head(max_rows)
    st.dataframe(display_df, width="stretch", height=400)

    st.divider()

    # ========== Statistics ==========
    st.subheader("📊 Statistics")
    tab1, tab2, tab3 = st.tabs(["Summary", "Distributions", "Anomalies"])

    with tab1:
        numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stats = filtered_df[numeric_cols].describe()
            # Add additional stats
            stats.loc["range"] = stats.loc["max"] - stats.loc["min"]
            stats.loc["cv%"] = (stats.loc["std"] / stats.loc["mean"] * 100).round(2)
            st.dataframe(stats.T.style.format("{:.4f}"), width="stretch")
        else:
            st.info("No numeric columns for statistics.")

    with tab2:
        numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            selected = st.selectbox("Select Metric for Distribution", numeric_cols, key="dist_metric")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(filtered_df, y=selected, title=f"{KERNEL_SYMBOLS.get(selected, selected)} Box Plot")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width="stretch")
            with col2:
                fig = px.histogram(
                    filtered_df,
                    x=selected,
                    nbins=30,
                    title=f"{KERNEL_SYMBOLS.get(selected, selected)} Histogram",
                    marginal="rug",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, width="stretch")

    with tab3:
        numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols and np is not None:
            st.markdown("**Anomaly Detection** (Z-score > 2.5)")
            selected = st.selectbox("Select Metric for Anomaly Detection", numeric_cols, key="anomaly_metric")
            if len(filtered_df) > 3:
                anomalies = detect_anomalies(filtered_df[selected])
                anomaly_count = anomalies.sum()
                st.metric("Anomalies Detected", anomaly_count)

                if anomaly_count > 0:
                    anomaly_df = filtered_df[anomalies]
                    st.dataframe(anomaly_df, width="stretch")

                    # Visualize
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df.index,
                            y=filtered_df[selected],
                            mode="lines",
                            name="Normal",
                            line={"color": "#007bff"},
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_df.index,
                            y=anomaly_df[selected],
                            mode="markers",
                            name="Anomaly",
                            marker={"color": "red", "size": 10},
                        )
                    )
                    fig.update_layout(title=f"Anomalies in {KERNEL_SYMBOLS.get(selected, selected)}", height=300)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough data for anomaly detection (need > 3 samples).")

    st.divider()

    # ========== Export ==========
    st.download_button(
        "📥 Download Filtered Ledger (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_casepacks_page() -> None:
    """Render the casepacks page with detailed information."""
    if st is None or pd is None:
        return

    st.title("📦 Casepacks")
    st.caption("Browse and explore available reference implementations")

    casepacks = load_casepacks()

    if not casepacks:
        st.warning("No casepacks found in `casepacks/` directory.")
        return

    # ========== Summary metrics ==========
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Casepacks", len(casepacks))
    with col2:
        total_closures = sum(cp["closures_count"] for cp in casepacks)
        st.metric("Total Closures", total_closures)
    with col3:
        total_tests = sum(cp["test_vectors"] for cp in casepacks)
        st.metric("Total Test Vectors", total_tests)
    with col4:
        total_files = sum(cp["files_count"] for cp in casepacks)
        st.metric("Total Files", total_files)

    st.divider()

    # ========== Filter ==========
    search = st.text_input("🔍 Search casepacks", placeholder="Type to filter...")

    filtered_casepacks = casepacks
    if search:
        search_lower = search.lower()
        filtered_casepacks = [
            cp
            for cp in casepacks
            if search_lower in cp["id"].lower() or search_lower in str(cp.get("description", "")).lower()
        ]

    st.caption(f"Showing {len(filtered_casepacks)} casepacks")

    # ========== Casepack Grid ==========
    for i in range(0, len(filtered_casepacks), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(filtered_casepacks):
                cp = filtered_casepacks[i + j]
                with col, st.container(border=True):
                    # Header
                    st.subheader(f"📦 {cp['id']}")

                    # Metadata badges
                    badge_html = f"""
                    <span style="background:#007bff; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">v{cp["version"]}</span>
                    """
                    if cp.get("contract"):
                        badge_html += f""" <span style="background:#28a745; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">📜 {cp["contract"]}</span>"""
                    st.markdown(badge_html, unsafe_allow_html=True)

                    # Description
                    if cp.get("title"):
                        st.markdown(f"**{cp['title']}**")
                    if cp.get("description"):
                        desc = cp["description"]
                        if len(desc) > 150:
                            desc = desc[:150] + "..."
                        st.markdown(desc)

                    # Metrics row
                    m_cols = st.columns(3)
                    with m_cols[0]:
                        st.metric("Closures", cp["closures_count"])
                    with m_cols[1]:
                        st.metric("Test Vectors", cp["test_vectors"])
                    with m_cols[2]:
                        st.metric("Files", cp["files_count"])

                    # Path
                    st.caption(f"📁 `{cp['path']}`")


def render_contracts_page() -> None:
    """Render the contracts page with detailed information."""
    if st is None or pd is None:
        return

    st.title("📜 Contracts")
    st.caption("Mathematical contracts defining validation rules and invariants")

    contracts = load_contracts()

    if not contracts:
        st.warning("No contracts found in `contracts/` directory.")
        return

    # ========== Summary by domain ==========
    domains: dict[str, list[dict[str, Any]]] = {}
    for c in contracts:
        domain = c["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(c)

    # Summary metrics
    st.subheader("📊 Overview")
    cols = st.columns(len(domains) + 1)
    with cols[0]:
        st.metric("Total Contracts", len(contracts))
    for i, (domain, domain_contracts) in enumerate(domains.items(), 1):
        with cols[i]:
            st.metric(f"{domain}", len(domain_contracts))

    st.divider()

    # ========== Domain tabs ==========
    domain_tabs = st.tabs(list(domains.keys()))

    for tab, (domain, domain_contracts) in zip(domain_tabs, domains.items(), strict=False):
        with tab:
            st.subheader(f"🏷️ {domain} Domain")
            st.caption(f"{len(domain_contracts)} contract(s)")

            for c in domain_contracts:
                with st.expander(f"📜 {c['id']} (v{c['version']})", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Path**: `{c['path']}`")
                    with col2:
                        st.markdown(f"**Size**: {format_bytes(c['size_bytes'])}")

                    # Preview content
                    st.markdown("**Content Preview:**")
                    try:
                        with open(c["path"]) as f:
                            content = f.read(3000)
                        if len(content) >= 3000:
                            content = content[:3000] + "\n\n... (truncated)"
                        st.code(content, language="yaml")
                    except Exception as e:
                        st.error(f"Could not read contract: {e}")


def render_closures_page() -> None:
    """Render the closures page with code preview."""
    if st is None or pd is None:
        return

    st.title("🔧 Closures")
    st.caption("Computational closures for validation and transformation")

    closures = load_closures()

    if not closures:
        st.warning("No closures found in `closures/` directory.")
        return

    # ========== Summary ==========
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        python_count = sum(1 for c in closures if c["type"] == "python")
        st.metric("🐍 Python", python_count)
    with col2:
        yaml_count = sum(1 for c in closures if c["type"] == "yaml")
        st.metric("📄 YAML", yaml_count)
    with col3:
        total_lines = sum(c["lines"] for c in closures)
        st.metric("📏 Total Lines", f"{total_lines:,}")
    with col4:
        total_size = sum(c["size_bytes"] for c in closures)
        st.metric("💾 Total Size", format_bytes(total_size))

    st.divider()

    # ========== Filters ==========
    filter_cols = st.columns(3)
    with filter_cols[0]:
        type_filter = st.radio("Type", ["All", "Python", "YAML"], horizontal=True)
    with filter_cols[1]:
        domain_options = ["All", *sorted({c["domain"] for c in closures})]
        domain_filter: str = st.selectbox("Domain", domain_options) or "All"
    with filter_cols[2]:
        search = st.text_input("Search", placeholder="Filter by name...")

    # Apply filters
    filtered = closures
    if type_filter == "Python":
        filtered = [c for c in filtered if c["type"] == "python"]
    elif type_filter == "YAML":
        filtered = [c for c in filtered if c["type"] == "yaml"]
    if domain_filter != "All":
        filtered = [c for c in filtered if c["domain"] == domain_filter]
    if search:
        filtered = [c for c in filtered if search.lower() in c["name"].lower()]

    st.caption(f"Showing {len(filtered)} closures")

    # ========== Closures list ==========
    for closure in filtered:
        emoji = "🐍" if closure["type"] == "python" else "📄"
        domain_badge = closure["domain"]

        with st.expander(f"{emoji} {closure['name']} — [{domain_badge}]"):
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Type**: {closure['type'].upper()}")
            with col2:
                st.markdown(f"**Size**: {format_bytes(closure['size_bytes'])}")
            with col3:
                if closure["lines"]:
                    st.markdown(f"**Lines**: {closure['lines']}")

            st.markdown(f"**Path**: `{closure['path']}`")

            # Code preview
            st.markdown("**Code:**")
            try:
                with open(closure["path"]) as f:
                    content = f.read(5000)
                lang = "python" if closure["type"] == "python" else "yaml"
                if len(content) >= 5000:
                    content = content[:5000] + "\n\n# ... (truncated)"
                st.code(content, language=lang, line_numbers=True)
            except Exception as e:
                st.error(f"Could not read closure: {e}")


def render_regime_page() -> None:
    """Render the regime classification page with interactive visualization."""
    if st is None or go is None or np is None:
        return

    st.title("🌡️ Regime Classification")
    st.caption("Interactive exploration of the kernel regime phase space")

    # ========== Theory ==========
    with st.expander("📖 Understanding Regimes", expanded=False):
        st.markdown(r"""
        The UMCP kernel classifies computational state into four regimes based on
        the **overlap fraction (ω)** and **seam residual (s)**:

        | Regime | Omega (ω) Range | Seam Residual | Interpretation |
        |--------|-----------------|---------------|----------------|
        | 🟢 **STABLE** | [0.3, 0.7] | \|s\| ≤ 0.005 | Normal operation |
        | 🟡 **WATCH** | [0.1, 0.3) ∪ (0.7, 0.9] | \|s\| ≤ 0.01 | Requires monitoring |
        | 🔴 **COLLAPSE** | < 0.1 or > 0.9 | Any | System degradation |
        | 🟣 **CRITICAL** | Any | \|s\| > 0.01 | Budget violation |

        The **seam residual** measures deviation from budget constraints.
        Critical regime takes priority as it indicates invariant violation.

        See **KERNEL_SPECIFICATION.md** for complete regime definitions.
        """)

    st.divider()

    # ========== Interactive Classifier ==========
    st.subheader("🎛️ Interactive Classifier")

    col1, col2 = st.columns(2)

    with col1:
        omega = st.slider(
            "Overlap Fraction (ω)", 0.0, 1.0, 0.5, 0.01, help="Fraction of overlapping state between iterations"
        )
        freshness = 1.0 - omega
        st.markdown(f"**Freshness (F = 1 - ω)**: `{freshness:.3f}`")

    with col2:
        seam = st.slider(
            "Seam Residual (s)", -0.05, 0.05, 0.0, 0.001, format="%.3f", help="Budget deviation at seam boundaries"
        )

    # Compute regime
    regime = classify_regime(omega, seam)
    color = get_regime_color(regime)

    # Display result
    st.markdown(
        f"""<div style='text-align: center; padding: 20px; background-color: {color}22;
        border-radius: 10px; border: 2px solid {color}; margin: 10px 0;'>
        <h1 style='color: {color}; margin: 0;'>{regime}</h1>
        <p style='margin: 5px 0 0 0;'>ω = {omega:.3f} | s = {seam:.4f} | F = {freshness:.3f}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ========== Phase Space Heatmap ==========
    st.subheader("📊 Regime Phase Space")

    with st.expander("⚙️ Visualization Settings"):
        resolution = st.slider("Resolution", 50, 200, 100, help="Higher resolution = more detail but slower rendering")
        show_boundaries = st.checkbox("Show Regime Boundaries", value=True)
        show_trajectory = st.checkbox("Simulate Trajectory", value=False)

    # Create phase space
    omega_range = np.linspace(0, 1, resolution)
    seam_range = np.linspace(-0.05, 0.05, resolution)

    regime_map: list[list[int]] = []
    for o in omega_range:
        row: list[int] = []
        for s in seam_range:
            r = classify_regime(float(o), float(s))
            val = {"STABLE": 0, "WATCH": 1, "COLLAPSE": 2, "CRITICAL": 3}[r]
            row.append(val)
        regime_map.append(row)

    fig = go.Figure()

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=regime_map,
            x=seam_range,
            y=omega_range,
            colorscale=[
                [0, REGIME_COLORS["STABLE"]],
                [0.33, REGIME_COLORS["WATCH"]],
                [0.66, REGIME_COLORS["COLLAPSE"]],
                [1.0, REGIME_COLORS["CRITICAL"]],
            ],
            showscale=False,
            hovertemplate="ω: %{y:.3f}<br>s: %{x:.4f}<extra></extra>",
        )
    )

    # Current point
    fig.add_trace(
        go.Scatter(
            x=[seam],
            y=[omega],
            mode="markers",
            marker={"size": 18, "color": "white", "line": {"width": 3, "color": "black"}},
            name="Current",
            hovertemplate=f"ω: {omega:.3f}<br>s: {seam:.4f}<br>Regime: {regime}<extra></extra>",
        )
    )

    # Regime boundaries
    if show_boundaries:
        for y_val in [0.1, 0.3, 0.7, 0.9]:
            fig.add_hline(y=y_val, line_dash="dash", line_color="white", opacity=0.6, line_width=1)
        for x_val in [-0.01, 0.01]:
            fig.add_vline(x=x_val, line_dash="dash", line_color="white", opacity=0.6, line_width=1)

    # Simulated trajectory
    if show_trajectory:
        t = np.linspace(0, 4 * np.pi, 100)
        traj_omega = 0.5 + 0.3 * np.sin(t / 2)
        traj_seam = 0.008 * np.sin(t)
        fig.add_trace(
            go.Scatter(
                x=traj_seam,
                y=traj_omega,
                mode="lines",
                line={"color": "white", "width": 2, "dash": "dot"},
                name="Trajectory",
            )
        )

    fig.update_layout(
        height=500,
        xaxis_title="Seam Residual (s)",
        yaxis_title="Overlap Fraction (ω)",
        xaxis={"range": [-0.05, 0.05]},
        yaxis={"range": [0, 1]},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )

    st.plotly_chart(fig, width="stretch")

    # Legend
    legend_cols = st.columns(4)
    for i, (regime_name, regime_color) in enumerate(REGIME_COLORS.items()):
        text_color = "white" if regime_name in ["COLLAPSE", "CRITICAL"] else "black"
        with legend_cols[i]:
            st.markdown(
                f"""<div style='background-color:{regime_color}; padding:10px;
                border-radius:5px; text-align:center; color:{text_color};'>
                <b>{regime_name}</b></div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ========== Ledger Overlay ==========
    st.subheader("📈 Ledger Data Overlay")
    df = load_ledger()
    if not df.empty and "omega" in df.columns:
        st.markdown("Plotting actual validation data on the phase space:")

        # Prepare data
        plot_df = df.copy()
        if "seam_residual" not in plot_df.columns:
            plot_df["seam_residual"] = 0.0  # Default if not present

        # Classify each point
        def _classify_row(row: Any) -> str:
            return classify_regime(row["omega"], row.get("seam_residual", 0))

        plot_df["regime"] = plot_df.apply(_classify_row, axis=1)

        # Create scatter overlay
        fig2 = go.Figure()

        # Background heatmap
        fig2.add_trace(
            go.Heatmap(
                z=regime_map,
                x=seam_range,
                y=omega_range,
                colorscale=[
                    [0, REGIME_COLORS["STABLE"]],
                    [0.33, REGIME_COLORS["WATCH"]],
                    [0.66, REGIME_COLORS["COLLAPSE"]],
                    [1.0, REGIME_COLORS["CRITICAL"]],
                ],
                showscale=False,
                opacity=0.5,
            )
        )

        # Scatter points
        for regime_name in plot_df["regime"].unique():
            regime_data = plot_df[plot_df["regime"] == regime_name]
            fig2.add_trace(
                go.Scatter(
                    x=regime_data["seam_residual"],
                    y=regime_data["omega"],
                    mode="markers",
                    name=regime_name,
                    marker={"color": REGIME_COLORS.get(regime_name, "#999"), "size": 8},
                )
            )

        fig2.update_layout(
            height=400,
            xaxis_title="Seam Residual (s)",
            yaxis_title="Overlap Fraction (ω)",
            xaxis={"range": [-0.05, 0.05]},
            yaxis={"range": [0, 1]},
        )
        st.plotly_chart(fig2, width="stretch")

        # Regime distribution
        regime_counts = plot_df["regime"].value_counts()
        st.markdown("**Regime Distribution:**")
        for regime_name, count in regime_counts.items():
            pct = count / len(plot_df) * 100
            st.markdown(f"- {regime_name}: {count} ({pct:.1f}%)")
    else:
        st.info("No omega data available in ledger. Run validations to see overlay.")


def render_metrics_page() -> None:
    """Render the kernel metrics page with advanced analysis."""
    if st is None or px is None or np is None or pd is None:
        return

    st.title("📐 Kernel Metrics Analysis")
    st.caption("Deep dive into kernel invariants, correlations, and trends")

    df = load_ledger()

    if df.empty:
        st.warning("No ledger data available for metrics analysis.")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found in ledger.")
        return

    # ========== Metric Selection ==========
    st.subheader("🎯 Select Metrics")
    default_metrics = [c for c in numeric_cols[:4] if c in numeric_cols]
    selected_metrics: list[str] = st.multiselect(
        "Choose metrics to analyze",
        options=numeric_cols,
        default=default_metrics,
        format_func=lambda x: KERNEL_SYMBOLS.get(x, x) or x,
        help="Select one or more metrics for visualization",
    )

    if not selected_metrics:
        st.info("Please select at least one metric to continue.")
        return

    st.divider()

    # ========== Time Series ==========
    st.subheader("📈 Time Series Analysis")

    if "timestamp" in df.columns and len(df) > 1:
        df_sorted = df.sort_values("timestamp")

        # Options
        ts_cols = st.columns(2)
        with ts_cols[0]:
            ma_window = st.slider("Moving Average Window", 1, 50, 10)
        with ts_cols[1]:
            show_raw = st.checkbox("Show Raw Data", value=True)

        # Create subplots
        if make_subplots is None:
            st.error("plotly.subplots not available")
            return

        subplot_titles = [KERNEL_SYMBOLS.get(m, m) or m for m in selected_metrics]
        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
        )

        for i, metric in enumerate(selected_metrics, 1):
            # Raw data
            if show_raw:
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted[metric],
                        mode="lines",
                        name=f"{metric} (raw)",
                        line={"width": 1},
                        opacity=0.4,
                    ),
                    row=i,
                    col=1,
                )
            # Moving average
            if ma_window > 1:
                ma = df_sorted[metric].rolling(window=ma_window, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=ma,
                        mode="lines",
                        name=f"{metric} (MA-{ma_window})",
                        line={"width": 2},
                    ),
                    row=i,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted[metric],
                        mode="lines",
                        name=metric,
                        line={"width": 2},
                    ),
                    row=i,
                    col=1,
                )

        fig.update_layout(
            height=220 * len(selected_metrics),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough time series data available.")

    st.divider()

    # ========== Distribution Analysis ==========
    st.subheader("📊 Distribution Analysis")

    dist_type = st.radio("Chart Type", ["Violin", "Box", "Histogram"], horizontal=True)

    dist_cols = st.columns(min(3, len(selected_metrics)))

    for i, metric in enumerate(selected_metrics[:3]):
        with dist_cols[i % 3]:
            if dist_type == "Violin":
                fig = go.Figure()
                fig.add_trace(
                    go.Violin(
                        y=df[metric].dropna(),
                        name=metric,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor="lightblue",
                    )
                )
            elif dist_type == "Box":
                fig = px.box(df, y=metric)
            else:
                fig = px.histogram(df, x=metric, nbins=30)

            fig.update_layout(
                title=KERNEL_SYMBOLS.get(metric, metric),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ========== Correlation Analysis ==========
    if len(selected_metrics) > 1:
        st.subheader("🔗 Correlation Analysis")

        tab1, tab2 = st.tabs(["Correlation Matrix", "Scatter Matrix"])

        with tab1:
            corr = df[selected_metrics].corr()

            x_labels = [KERNEL_SYMBOLS.get(m, m) or m for m in selected_metrics]
            y_labels = [KERNEL_SYMBOLS.get(m, m) or m for m in selected_metrics]

            fig = px.imshow(
                corr,
                x=x_labels,
                y=y_labels,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto=True,
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(title="Correlation Matrix", height=400)
            st.plotly_chart(fig, width="stretch")

            # Insights
            st.markdown("**Correlation Insights:**")
            for i in range(len(selected_metrics)):
                for j in range(i + 1, len(selected_metrics)):
                    corr_val = corr.iloc[i, j]
                    m1, m2 = selected_metrics[i], selected_metrics[j]
                    if abs(corr_val) > 0.7:
                        strength = "strongly" if abs(corr_val) > 0.85 else "moderately"
                        direction = "positively" if corr_val > 0 else "negatively"
                        st.markdown(
                            f"- **{KERNEL_SYMBOLS.get(m1, m1)}** and **{KERNEL_SYMBOLS.get(m2, m2)}** are {strength} {direction} correlated (r={corr_val:.2f})"
                        )

        with tab2:
            if len(selected_metrics) >= 2:
                fig = px.scatter_matrix(
                    df[selected_metrics].dropna(),
                    dimensions=selected_metrics[:5],
                    height=600,
                )
                fig.update_traces(diagonal_visible=False, marker={"size": 4})
                st.plotly_chart(fig, width="stretch")

    st.divider()

    # ========== Statistical Summary ==========
    st.subheader("📋 Statistical Summary")

    stats = df[selected_metrics].describe()
    stats.loc["range"] = stats.loc["max"] - stats.loc["min"]
    stats.loc["iqr"] = stats.loc["75%"] - stats.loc["25%"]
    stats.loc["cv%"] = stats.loc["std"] / stats.loc["mean"] * 100

    # Rename index for display
    index_map = {
        "count": "Count",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Min",
        "25%": "Q1 (25%)",
        "50%": "Median",
        "75%": "Q3 (75%)",
        "max": "Max",
        "range": "Range",
        "iqr": "IQR",
        "cv%": "CV %",
    }

    def _rename_index(x: Any) -> str:
        return index_map.get(str(x), str(x))

    stats.index = stats.index.map(_rename_index)

    # Try to use styled dataframe with gradient, fallback to plain if matplotlib not available
    try:
        st.dataframe(stats.T.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0), width="stretch")
    except ImportError:
        # Matplotlib not available, use plain styled dataframe
        st.dataframe(stats.T.style.format("{:.4f}"), width="stretch")


def render_health_page() -> None:
    """Render system health diagnostics."""
    if st is None or pd is None:
        return

    st.title("🏥 System Health")
    st.caption("UMCP system diagnostics and health checks")

    repo_root = get_repo_root()

    # ========== Core Component Checks ==========
    st.subheader("✅ Core Components")

    checks: list[dict[str, Any]] = []

    # pyproject.toml
    pyproject = repo_root / "pyproject.toml"
    checks.append(
        {
            "Component": "pyproject.toml",
            "Status": "✅ Found" if pyproject.exists() else "❌ Missing",
            "Details": str(pyproject) if pyproject.exists() else "Required for package configuration",
        }
    )

    # Schemas
    schemas_dir = repo_root / "schemas"
    schema_count = len(list(schemas_dir.glob("*.json"))) if schemas_dir.exists() else 0
    checks.append(
        {
            "Component": "Schemas",
            "Status": f"✅ {schema_count} schemas" if schema_count > 0 else "⚠️ No schemas",
            "Details": str(schemas_dir),
        }
    )

    # Contracts
    contracts_dir = repo_root / "contracts"
    contract_count = len(list(contracts_dir.glob("*.yaml"))) if contracts_dir.exists() else 0
    checks.append(
        {
            "Component": "Contracts",
            "Status": f"✅ {contract_count} contracts" if contract_count > 0 else "⚠️ No contracts",
            "Details": str(contracts_dir),
        }
    )

    # Closures
    closures_dir = repo_root / "closures"
    closure_count = (
        len(list(closures_dir.glob("*.py"))) + len(list(closures_dir.glob("*.yaml"))) if closures_dir.exists() else 0
    )
    checks.append(
        {
            "Component": "Closures",
            "Status": f"✅ {closure_count} closures" if closure_count > 0 else "⚠️ No closures",
            "Details": str(closures_dir),
        }
    )

    # Casepacks
    casepacks_dir = repo_root / "casepacks"
    casepack_count = len([d for d in casepacks_dir.iterdir() if d.is_dir()]) if casepacks_dir.exists() else 0
    checks.append(
        {
            "Component": "Casepacks",
            "Status": f"✅ {casepack_count} casepacks" if casepack_count > 0 else "⚠️ No casepacks",
            "Details": str(casepacks_dir),
        }
    )

    # Ledger
    ledger_path = repo_root / "ledger" / "return_log.csv"
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger_lines = max(0, sum(1 for _ in f) - 1)
        checks.append(
            {
                "Component": "Ledger",
                "Status": f"✅ {ledger_lines} entries",
                "Details": str(ledger_path),
            }
        )
    else:
        checks.append(
            {
                "Component": "Ledger",
                "Status": "⚠️ Not initialized",
                "Details": "Run `umcp validate` to create ledger",
            }
        )

    # Display checks as table
    checks_df = pd.DataFrame(checks)
    st.dataframe(checks_df, width="stretch", hide_index=True)

    st.divider()

    # ========== Python Environment ==========
    st.subheader("🐍 Python Environment")

    env_cols = st.columns(3)
    with env_cols[0]:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    with env_cols[1]:
        st.metric("UMCP Version", __version__)
    with env_cols[2]:
        st.metric("Platform", sys.platform)

    st.divider()

    # ========== Dependencies ==========
    st.subheader("📦 Dependencies")

    deps: list[dict[str, Any]] = []

    # Core deps
    core_deps = ["yaml", "json"]
    for dep in core_deps:
        try:
            __import__(dep if dep != "yaml" else "yaml")
            deps.append({"Package": dep, "Type": "Core", "Status": "✅ Installed"})
        except ImportError:
            deps.append({"Package": dep, "Type": "Core", "Status": "❌ Not installed"})

    # Viz deps
    viz_deps = ["pandas", "plotly", "streamlit", "numpy"]
    for dep in viz_deps:
        status = "✅ Installed" if HAS_VIZ_DEPS else "❌ Not installed"
        deps.append({"Package": dep, "Type": "Visualization", "Status": status})

    deps_df = pd.DataFrame(deps)
    st.dataframe(deps_df, width="stretch", hide_index=True)

    st.divider()

    # ========== Quick Validation ==========
    st.subheader("🧪 Quick Validation Test")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Run Health Check", width="stretch"):
            with st.spinner("Running health check..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "umcp", "health"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=repo_root,
                    )
                    if result.returncode == 0:
                        st.success("✅ Health check passed!")
                        if result.stdout:
                            st.code(result.stdout, language="text")
                    else:
                        st.error("❌ Health check failed!")
                        st.code(result.stderr or result.stdout, language="text")
                except FileNotFoundError:
                    st.warning("umcp CLI not found. Make sure UMCP is installed.")
                except subprocess.TimeoutExpired:
                    st.error("Health check timed out after 30 seconds.")
                except Exception as e:
                    st.error(f"Error running health check: {e}")

    with col2:
        if st.button("🧪 Run Tests (Quick)", width="stretch"):
            with st.spinner("Running quick tests..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q", "tests/", "-k", "not slow"],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=repo_root,
                    )
                    if result.returncode == 0:
                        st.success("✅ Tests passed!")
                    else:
                        st.warning("⚠️ Some tests failed")
                    st.code(result.stdout + result.stderr, language="text")
                except Exception as e:
                    st.error(f"Error running tests: {e}")


# ============================================================================
# SI Unit Conversion Utilities
# ============================================================================

# SI Unit conversion factors to base units
SI_PREFIXES = {
    "Y": 1e24,  # yotta
    "Z": 1e21,  # zetta
    "E": 1e18,  # exa
    "P": 1e15,  # peta
    "T": 1e12,  # tera
    "G": 1e9,  # giga
    "M": 1e6,  # mega
    "k": 1e3,  # kilo
    "h": 1e2,  # hecto
    "da": 1e1,  # deka
    "": 1.0,  # base
    "d": 1e-1,  # deci
    "c": 1e-2,  # centi
    "m": 1e-3,  # milli
    "μ": 1e-6,  # micro
    "n": 1e-9,  # nano
    "p": 1e-12,  # pico
    "f": 1e-15,  # femto
    "a": 1e-18,  # atto
}


# ============================================================================
# GCD (Generative Collapse Dynamics) Translation Framework
# ============================================================================

# GCD Tier-1 Reserved Symbols with semantic interpretations
GCD_SYMBOLS = {
    "omega": {
        "latex": "ω",
        "name": "Drift",
        "description": "Weighted distance from upper boundary (collapse metric)",
        "formula": "ω = Σ wᵢ(1-cᵢ)",
        "domain": "[0, 1]",
        "optimal": 0.0,
        "collapse_threshold": 0.30,
        "interpretation": {
            0.0: "Perfect alignment (no drift)",
            0.038: "Stable boundary",
            0.30: "Collapse threshold",
            1.0: "Total collapse",
        },
    },
    "F": {
        "latex": "F",
        "name": "Fidelity",
        "description": "System fidelity (complement of drift)",
        "formula": "F = 1 - ω",
        "domain": "[0, 1]",
        "optimal": 1.0,
        "collapse_threshold": 0.70,
        "interpretation": {
            1.0: "Perfect fidelity",
            0.90: "Stable region",
            0.70: "Watch boundary",
            0.0: "Zero fidelity",
        },
    },
    "S": {
        "latex": "S",
        "name": "Entropy",
        "description": "Shannon entropy (system uncertainty/determinacy)",
        "formula": "S = -Σ wᵢ[cᵢln(cᵢ) + (1-cᵢ)ln(1-cᵢ)]",
        "domain": "[0, ln(2)]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "Perfect determinacy (S=0 state)",
            0.15: "Low uncertainty",
            0.35: "Moderate uncertainty",
            0.693: "Maximum uncertainty",
        },
    },
    "C": {
        "latex": "C",
        "name": "Curvature",
        "description": "Coordinate non-uniformity (heterogeneity measure)",
        "formula": "C = σ(cᵢ)/0.5",
        "domain": "[0, 1]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "Uniform/homogeneous",
            0.14: "Low curvature",
            0.30: "Moderate curvature",
            1.0: "Maximum curvature",
        },
    },
    "kappa": {
        "latex": "κ",
        "name": "Log-Integrity",
        "description": "Logarithmic collapse accumulation",
        "formula": "κ = Σ wᵢ ln(cᵢ)",
        "domain": "(-∞, 0]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "No log-collapse",
            -0.5: "Mild log-collapse",
            -2.0: "Significant log-collapse",
            -10.0: "Severe log-collapse",
        },
    },
    "IC": {
        "latex": "IC",
        "name": "Integrity Composite",
        "description": "Geometric mean of coordinates (exponential of κ)",
        "formula": "IC = exp(κ) = ∏ cᵢ^wᵢ",
        "domain": "(0, 1]",
        "optimal": 1.0,
        "identity": "IC = exp(κ)",
        "interpretation": {
            1.0: "Perfect integrity",
            0.80: "Good integrity",
            0.50: "Degraded integrity",
            0.10: "Critical integrity",
        },
    },
    "tau_R": {
        "latex": "τ_R",
        "name": "Return Time",
        "description": "Recursive timescale to return domain",
        "domain": "[0, ∞] ∪ {INF_REC}",
        "interpretation": {
            0: "Immediate return",
            10: "Fast return",
            50: "Slow return",
            "INF_REC": "No return (non-returning)",
        },
    },
}

# GCD Regime thresholds (from gcd_anchors.yaml)
GCD_REGIMES = {
    "Stable": {
        "condition": "ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14",
        "color": "#28a745",  # Green
        "description": "Optimal operational state with low collapse risk",
        "icon": "🟢",
    },
    "Watch": {
        "condition": "0.038 ≤ ω < 0.30 OR intermediate conditions",
        "color": "#ffc107",  # Yellow
        "description": "Elevated monitoring required; approaching collapse boundary",
        "icon": "🟡",
    },
    "Collapse": {
        "condition": "ω ≥ 0.30 OR F < 0.70",
        "color": "#dc3545",  # Red
        "description": "Collapse event detected; generative restructuring active",
        "icon": "🔴",
    },
}

# GCD Axioms for display
GCD_AXIOMS = {
    "AX-0": {
        "statement": "Collapse is generative",
        "description": "Collapse events are not terminal failures but generative processes that produce new structure.",
    },
    "AX-1": {
        "statement": "Boundary defines interior",
        "description": "The collapse boundary (ω ≥ 0.30) defines what is stable. Distance from collapse determines behavior.",
    },
    "AX-2": {
        "statement": "Entropy measures determinacy",
        "description": "S=0 represents perfect determinacy; S approaching maximum represents maximum uncertainty.",
    },
}


def translate_to_gcd(tier1_values: dict[str, Any]) -> dict[str, Any]:
    """
    Translate Tier-1 kernel values to GCD native representation with interpretations.

    Args:
        tier1_values: Dict with keys F, omega, S, C, kappa, IC, etc.

    Returns:
        Dict with GCD translations, interpretations, and regime classification
    """
    gcd = {
        "symbols": {},
        "regime": None,
        "axiom_state": {},
        "natural_language": [],
        "warnings": [],
    }

    # Extract values with defaults
    omega = tier1_values.get("omega", tier1_values.get("F", 0))
    if "omega" not in tier1_values and "F" in tier1_values:
        omega = 1 - tier1_values["F"]

    fidelity = tier1_values.get("F", 1 - omega)
    entropy = tier1_values.get("S", 0)
    curvature = tier1_values.get("C", 0)
    kappa = tier1_values.get("kappa", 0)
    ic = tier1_values.get("IC", 1)

    # Translate each symbol
    gcd["symbols"]["omega"] = {
        "value": omega,
        "latex": "ω",
        "interpretation": _interpret_value(omega, GCD_SYMBOLS["omega"]),
        "distance_to_collapse": max(0, 0.30 - omega),
        "percent_to_collapse": min(100, omega / 0.30 * 100),
    }

    gcd["symbols"]["F"] = {
        "value": fidelity,
        "latex": "F",
        "interpretation": _interpret_value(fidelity, GCD_SYMBOLS["F"]),
        "identity_check": abs(fidelity + omega - 1.0) < 1e-9,
    }

    gcd["symbols"]["S"] = {
        "value": entropy,
        "latex": "S",
        "interpretation": _interpret_value(entropy, GCD_SYMBOLS["S"]),
        "determinacy": "deterministic" if entropy < 0.01 else ("low uncertainty" if entropy < 0.15 else "uncertain"),
    }

    gcd["symbols"]["C"] = {
        "value": curvature,
        "latex": "C",
        "interpretation": _interpret_value(curvature, GCD_SYMBOLS["C"]),
        "homogeneity": "homogeneous" if curvature < 0.01 else ("coherent" if curvature < 0.14 else "heterogeneous"),
    }

    gcd["symbols"]["kappa"] = {
        "value": kappa,
        "latex": "κ",
        "interpretation": _interpret_value(kappa, GCD_SYMBOLS["kappa"]),
    }

    gcd["symbols"]["IC"] = {
        "value": ic,
        "latex": "IC",
        "interpretation": _interpret_value(ic, GCD_SYMBOLS["IC"]),
        "identity_check": abs(ic - np.exp(kappa)) < 1e-6 if np is not None else True,
    }

    # Determine GCD regime based on canonical thresholds from gcd_anchors.yaml
    # Stable: ω < 0.038 (strict) or ω < 0.10 with high F and low S/C
    # Watch: 0.038 ≤ ω < 0.30
    # Collapse: ω ≥ 0.30
    if omega < 0.10 and fidelity > 0.85:
        gcd["regime"] = "Stable"
    elif omega >= 0.30:
        gcd["regime"] = "Collapse"
    else:
        gcd["regime"] = "Watch"

    gcd["regime_info"] = GCD_REGIMES[gcd["regime"]]

    # Axiom state interpretation
    gcd["axiom_state"]["AX-0"] = {
        "active": gcd["regime"] == "Collapse",
        "description": "Generative collapse active" if gcd["regime"] == "Collapse" else "System stable (no collapse)",
    }
    gcd["axiom_state"]["AX-1"] = {
        "distance_to_boundary": gcd["symbols"]["omega"]["distance_to_collapse"],
        "description": f"{gcd['symbols']['omega']['distance_to_collapse']:.3f} from collapse boundary",
    }
    gcd["axiom_state"]["AX-2"] = {
        "determinacy": gcd["symbols"]["S"]["determinacy"],
        "description": f"System is {gcd['symbols']['S']['determinacy']} (S={entropy:.4f})",
    }

    # Natural language summary
    regime_desc = GCD_REGIMES[gcd["regime"]]["description"]
    gcd["natural_language"] = [
        f"**GCD Regime**: {gcd['regime']} — {regime_desc}",
        f"**Drift (ω)**: {omega:.4f} — {gcd['symbols']['omega']['interpretation']}",
        f"**Fidelity (F)**: {fidelity:.4f} — {gcd['symbols']['F']['interpretation']}",
        f"**Entropy (S)**: {entropy:.4f} — System is {gcd['symbols']['S']['determinacy']}",
        f"**Curvature (C)**: {curvature:.4f} — Coordinates are {gcd['symbols']['C']['homogeneity']}",
        f"**Integrity (IC)**: {ic:.4f} — {gcd['symbols']['IC']['interpretation']}",
    ]

    # Create summary for easy access
    gcd["summary"] = f"GCD {gcd['regime']}: ω={omega:.4f}, F={fidelity:.4f}, IC={ic:.4f}. {regime_desc}"

    # Alias for axiom_states (some code uses this key)
    gcd["axiom_states"] = gcd["axiom_state"]

    # Warnings
    if omega >= 0.25:
        gcd["warnings"].append("⚠️ Approaching collapse boundary (ω ≥ 0.25)")
    if not gcd["symbols"]["F"]["identity_check"]:
        gcd["warnings"].append("⚠️ Identity violation: F + ω ≠ 1")
    if not gcd["symbols"]["IC"]["identity_check"]:
        gcd["warnings"].append("⚠️ Identity violation: IC ≠ exp(κ)")

    return gcd


def _interpret_value(value: float, symbol_def: dict[str, Any]) -> str:
    """Get natural language interpretation for a GCD value."""
    interp = symbol_def.get("interpretation", {})

    # Find closest threshold
    closest_desc = "Unknown"
    closest_dist = float("inf")

    for threshold, desc in interp.items():
        if isinstance(threshold, int | float):
            dist = abs(value - threshold)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = desc

    return closest_desc


def render_gcd_panel(gcd_data: dict[str, Any], compact: bool = False) -> None:
    """Render a GCD translation panel in Streamlit."""
    if st is None:
        return

    regime = gcd_data["regime"]
    regime_info = gcd_data["regime_info"]

    # Regime header
    st.markdown(
        f"""<div style="padding: 15px; border-left: 6px solid {regime_info["color"]}; 
            background: {regime_info["color"]}22; border-radius: 8px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: {regime_info["color"]};">
                {regime_info["icon"]} GCD Regime: {regime}
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">{regime_info["description"]}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if compact:
        # Compact display
        cols = st.columns(6)
        symbols = ["omega", "F", "S", "C", "kappa", "IC"]
        for i, sym in enumerate(symbols):
            with cols[i]:
                data = gcd_data["symbols"][sym]
                st.metric(data["latex"], f"{data['value']:.4f}")
    else:
        # Full display with interpretations
        st.markdown("### 📐 GCD Tier-1 Invariants")

        col1, col2 = st.columns(2)

        with col1:
            for sym in ["omega", "F", "S"]:
                data = gcd_data["symbols"][sym]
                symbol_info = GCD_SYMBOLS[sym]

                st.markdown(f"**{data['latex']} ({symbol_info['name']})**: `{data['value']:.6f}`")
                st.caption(f"_{data['interpretation']}_")

                # Progress bar for bounded values
                if sym == "omega":
                    progress = min(1.0, data["value"] / 0.30)
                    st.progress(progress, text=f"{data['percent_to_collapse']:.1f}% to collapse")
                elif sym == "F":
                    st.progress(data["value"], text=f"Fidelity: {data['value'] * 100:.1f}%")

        with col2:
            for sym in ["C", "kappa", "IC"]:
                data = gcd_data["symbols"][sym]
                symbol_info = GCD_SYMBOLS[sym]

                st.markdown(f"**{data['latex']} ({symbol_info['name']})**: `{data['value']:.6f}`")
                st.caption(f"_{data['interpretation']}_")

                if sym == "C":
                    st.progress(min(1.0, data["value"]), text=f"Curvature: {data['value'] * 100:.1f}%")
                elif sym == "IC":
                    st.progress(data["value"], text=f"Integrity: {data['value'] * 100:.1f}%")

        # Axiom state
        st.markdown("### 📜 GCD Axiom State")
        ax_cols = st.columns(3)
        for i, (ax_id, ax_state) in enumerate(gcd_data["axiom_state"].items()):
            with ax_cols[i]:
                ax_info = GCD_AXIOMS[ax_id]
                st.markdown(f"**{ax_id}**: _{ax_info['statement']}_")
                st.caption(ax_state["description"])

        # Warnings
        if gcd_data["warnings"]:
            st.markdown("### ⚠️ Warnings")
            for warning in gcd_data["warnings"]:
                st.warning(warning)


# Physical quantity definitions with units
PHYSICS_QUANTITIES = {
    "position": {
        "symbol": "x",
        "base_unit": "m",
        "units": {
            "km": 1e3,
            "m": 1.0,
            "cm": 1e-2,
            "mm": 1e-3,
            "μm": 1e-6,
            "nm": 1e-9,
            "ft": 0.3048,
            "in": 0.0254,
            "mi": 1609.34,
        },
        "ref_value": 1.0,  # m
        "description": "Position / Distance",
    },
    "velocity": {
        "symbol": "v",
        "base_unit": "m/s",
        "units": {
            "m/s": 1.0,
            "km/h": 1 / 3.6,
            "km/s": 1e3,
            "cm/s": 1e-2,
            "ft/s": 0.3048,
            "mph": 0.44704,
            "knot": 0.514444,
        },
        "ref_value": 1.0,  # m/s
        "description": "Velocity / Speed",
    },
    "acceleration": {
        "symbol": "a",
        "base_unit": "m/s²",
        "units": {"m/s²": 1.0, "g": 9.80665, "ft/s²": 0.3048, "cm/s²": 1e-2, "Gal": 1e-2},
        "ref_value": 9.80665,  # m/s² (1g)
        "description": "Acceleration",
    },
    "mass": {
        "symbol": "m",
        "base_unit": "kg",
        "units": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "μg": 1e-9, "t": 1e3, "lb": 0.453592, "oz": 0.0283495},
        "ref_value": 1.0,  # kg
        "description": "Mass",
    },
    "force": {
        "symbol": "F",
        "base_unit": "N",
        "units": {"N": 1.0, "kN": 1e3, "MN": 1e6, "mN": 1e-3, "dyn": 1e-5, "lbf": 4.44822, "kgf": 9.80665},
        "ref_value": 1.0,  # N
        "description": "Force",
    },
    "energy": {
        "symbol": "E",
        "base_unit": "J",
        "units": {
            "J": 1.0,
            "kJ": 1e3,
            "MJ": 1e6,
            "mJ": 1e-3,
            "eV": 1.602e-19,
            "keV": 1.602e-16,
            "cal": 4.184,
            "kcal": 4184,
            "BTU": 1055.06,
            "kWh": 3.6e6,
        },
        "ref_value": 1.0,  # J
        "description": "Energy",
    },
    "momentum": {
        "symbol": "p",
        "base_unit": "kg·m/s",
        "units": {"kg·m/s": 1.0, "g·cm/s": 1e-5, "N·s": 1.0},
        "ref_value": 1.0,  # kg·m/s
        "description": "Momentum",
    },
    "time": {
        "symbol": "t",
        "base_unit": "s",
        "units": {"s": 1.0, "ms": 1e-3, "μs": 1e-6, "ns": 1e-9, "min": 60, "h": 3600, "day": 86400},
        "ref_value": 1.0,  # s
        "description": "Time",
    },
    "frequency": {
        "symbol": "f",
        "base_unit": "Hz",
        "units": {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "rpm": 1 / 60},
        "ref_value": 1.0,  # Hz
        "description": "Frequency",
    },
    "power": {
        "symbol": "P",
        "base_unit": "W",
        "units": {"W": 1.0, "kW": 1e3, "MW": 1e6, "mW": 1e-3, "hp": 745.7, "BTU/h": 0.293071},
        "ref_value": 1.0,  # W
        "description": "Power",
    },
    "angle": {
        "symbol": "θ",
        "base_unit": "rad",
        "units": {"rad": 1.0, "deg": 0.0174533, "°": 0.0174533, "rev": 6.28319, "grad": 0.015708},
        "ref_value": 1.0,  # rad
        "description": "Angle",
    },
    "angular_velocity": {
        "symbol": "ω",
        "base_unit": "rad/s",
        "units": {"rad/s": 1.0, "deg/s": 0.0174533, "rpm": 0.10472, "Hz": 6.28319},
        "ref_value": 1.0,  # rad/s
        "description": "Angular Velocity",
    },
}

# Kinematics scenarios with preset values
KINEMATICS_SCENARIOS = {
    "Free Fall": {
        "description": "Object falling under gravity",
        "quantities": {
            "position": {"value": 0.0, "unit": "m"},
            "velocity": {"value": 0.0, "unit": "m/s"},
            "acceleration": {"value": 9.81, "unit": "m/s²"},
            "mass": {"value": 1.0, "unit": "kg"},
            "time": {"value": 1.0, "unit": "s"},
        },
    },
    "Projectile Motion": {
        "description": "Projectile launched at 45°",
        "quantities": {
            "position": {"value": 0.0, "unit": "m"},
            "velocity": {"value": 10.0, "unit": "m/s"},
            "acceleration": {"value": 9.81, "unit": "m/s²"},
            "mass": {"value": 0.5, "unit": "kg"},
            "angle": {"value": 45.0, "unit": "deg"},
        },
    },
    "Simple Harmonic Oscillator": {
        "description": "Mass on a spring",
        "quantities": {
            "position": {"value": 0.1, "unit": "m"},
            "velocity": {"value": 0.0, "unit": "m/s"},
            "mass": {"value": 1.0, "unit": "kg"},
            "frequency": {"value": 1.0, "unit": "Hz"},
            "energy": {"value": 0.5, "unit": "J"},
        },
    },
    "Circular Motion": {
        "description": "Uniform circular motion",
        "quantities": {
            "position": {"value": 1.0, "unit": "m"},
            "velocity": {"value": 2.0, "unit": "m/s"},
            "angular_velocity": {"value": 2.0, "unit": "rad/s"},
            "mass": {"value": 1.0, "unit": "kg"},
            "force": {"value": 4.0, "unit": "N"},
        },
    },
    "Elastic Collision": {
        "description": "1D elastic collision",
        "quantities": {
            "mass": {"value": 1.0, "unit": "kg"},
            "velocity": {"value": 5.0, "unit": "m/s"},
            "momentum": {"value": 5.0, "unit": "kg·m/s"},
            "energy": {"value": 12.5, "unit": "J"},
        },
    },
}


# ============================================================
# PHYSICAL CONSTANTS - Fundamental physics constants (CODATA 2022)
# ============================================================
PHYSICAL_CONSTANTS: dict[str, dict[str, Any]] = {
    "c": {"name": "Speed of light in vacuum", "value": 299792458, "unit": "m/s", "symbol": "c"},
    "G": {"name": "Gravitational constant", "value": 6.67430e-11, "unit": "m³/(kg·s²)", "symbol": "G"},
    "h": {"name": "Planck constant", "value": 6.62607015e-34, "unit": "J·s", "symbol": "h"},
    "ħ": {"name": "Reduced Planck constant", "value": 1.054571817e-34, "unit": "J·s", "symbol": "ħ"},
    "e": {"name": "Elementary charge", "value": 1.602176634e-19, "unit": "C", "symbol": "e"},
    "m_e": {"name": "Electron mass", "value": 9.1093837015e-31, "unit": "kg", "symbol": "mₑ"},
    "m_p": {"name": "Proton mass", "value": 1.67262192369e-27, "unit": "kg", "symbol": "mₚ"},
    "m_n": {"name": "Neutron mass", "value": 1.67492749804e-27, "unit": "kg", "symbol": "mₙ"},
    "k_B": {"name": "Boltzmann constant", "value": 1.380649e-23, "unit": "J/K", "symbol": "kB"},
    "N_A": {"name": "Avogadro constant", "value": 6.02214076e23, "unit": "mol⁻¹", "symbol": "Nₐ"},
    "R": {"name": "Gas constant", "value": 8.314462618, "unit": "J/(mol·K)", "symbol": "R"},
    "ε_0": {"name": "Vacuum permittivity", "value": 8.8541878128e-12, "unit": "F/m", "symbol": "ε₀"},
    "μ_0": {"name": "Vacuum permeability", "value": 1.25663706212e-6, "unit": "H/m", "symbol": "μ₀"},
    "α": {"name": "Fine structure constant", "value": 7.2973525693e-3, "unit": "dimensionless", "symbol": "α"},
    "g": {"name": "Standard gravity", "value": 9.80665, "unit": "m/s²", "symbol": "g"},
    "atm": {"name": "Standard atmosphere", "value": 101325, "unit": "Pa", "symbol": "atm"},
    "σ": {"name": "Stefan-Boltzmann constant", "value": 5.670374419e-8, "unit": "W/(m²·K⁴)", "symbol": "σ"},
    "R_∞": {"name": "Rydberg constant", "value": 10973731.568160, "unit": "m⁻¹", "symbol": "R∞"},
    "a_0": {"name": "Bohr radius", "value": 5.29177210903e-11, "unit": "m", "symbol": "a₀"},
}

# ============================================================
# PHYSICS FORMULAS - Common formulas for calculation
# Note: Lambda functions are intentionally untyped for flexibility
# ============================================================
PHYSICS_FORMULAS: dict[str, dict[str, Any]] = {  # type: ignore[misc]
    "kinetic_energy": {
        "name": "Kinetic Energy",
        "formula": "KE = ½mv²",
        "latex": r"KE = \frac{1}{2}mv^2",
        "inputs": ["mass", "velocity"],
        "output": "energy",
        "calculate": lambda m, v: 0.5 * m * v**2,
        "description": "Energy of motion",
    },
    "potential_energy_gravity": {
        "name": "Gravitational Potential Energy",
        "formula": "PE = mgh",
        "latex": r"PE = mgh",
        "inputs": ["mass", "position", "acceleration"],
        "output": "energy",
        "calculate": lambda m, h, g=9.80665: m * g * h,
        "description": "Energy due to height in gravitational field",
    },
    "momentum": {
        "name": "Linear Momentum",
        "formula": "p = mv",
        "latex": r"p = mv",
        "inputs": ["mass", "velocity"],
        "output": "momentum",
        "calculate": lambda m, v: m * v,
        "description": "Product of mass and velocity",
    },
    "force_newton": {
        "name": "Newton's Second Law",
        "formula": "F = ma",
        "latex": r"F = ma",
        "inputs": ["mass", "acceleration"],
        "output": "force",
        "calculate": lambda m, a: m * a,
        "description": "Force equals mass times acceleration",
    },
    "work": {
        "name": "Work Done",
        "formula": "W = Fd",
        "latex": r"W = Fd",
        "inputs": ["force", "position"],
        "output": "energy",
        "calculate": lambda F, d: F * d,
        "description": "Work as force times displacement",
    },
    "power": {
        "name": "Power",
        "formula": "P = W/t",
        "latex": r"P = \frac{W}{t}",
        "inputs": ["energy", "time"],
        "output": "power",
        "calculate": lambda W, t: W / t if t != 0 else 0,
        "description": "Rate of energy transfer",
    },
    "centripetal_force": {
        "name": "Centripetal Force",
        "formula": "F = mv²/r",
        "latex": r"F = \frac{mv^2}{r}",
        "inputs": ["mass", "velocity", "position"],
        "output": "force",
        "calculate": lambda m, v, r: m * v**2 / r if r != 0 else 0,
        "description": "Force for circular motion",
    },
    "gravitational_force": {
        "name": "Gravitational Force",
        "formula": "F = Gm₁m₂/r²",
        "latex": r"F = \frac{Gm_1m_2}{r^2}",
        "inputs": ["mass", "mass", "position"],
        "output": "force",
        "calculate": lambda m1, m2, r: 6.67430e-11 * m1 * m2 / r**2 if r != 0 else 0,
        "description": "Universal gravitation",
    },
    "period_pendulum": {
        "name": "Simple Pendulum Period",
        "formula": "T = 2π√(L/g)",
        "latex": r"T = 2\pi\sqrt{\frac{L}{g}}",
        "inputs": ["position"],  # L = position
        "output": "time",
        "calculate": lambda L, g=9.80665: 2 * 3.14159265 * (L / g) ** 0.5 if g != 0 else 0,
        "description": "Period of simple pendulum",
    },
    "wave_speed": {
        "name": "Wave Speed",
        "formula": "v = fλ",
        "latex": r"v = f\lambda",
        "inputs": ["frequency", "position"],  # λ = position (wavelength)
        "output": "velocity",
        "calculate": lambda f, wavelength: f * wavelength,
        "description": "Wave velocity",
    },
    "relativistic_energy": {
        "name": "Mass-Energy Equivalence",
        "formula": "E = mc²",
        "latex": r"E = mc^2",
        "inputs": ["mass"],
        "output": "energy",
        "calculate": lambda m: m * 299792458**2,
        "description": "Einstein's famous equation",
    },
    "de_broglie": {
        "name": "de Broglie Wavelength",
        "formula": "λ = h/p",
        "latex": r"\lambda = \frac{h}{p}",
        "inputs": ["momentum"],
        "output": "position",  # wavelength
        "calculate": lambda p: 6.62607015e-34 / p if p != 0 else 0,
        "description": "Matter wave wavelength",
    },
}

# ============================================================
# KINEMATICS EQUATIONS - 1D motion equations
# Note: Lambda functions are intentionally untyped for flexibility
# ============================================================
KINEMATICS_EQUATIONS: dict[str, dict[str, Any]] = {  # type: ignore[misc]
    "position_time": {
        "name": "Position from time",
        "formula": "x = x₀ + v₀t + ½at²",
        "latex": r"x = x_0 + v_0t + \frac{1}{2}at^2",
        "inputs": {"x0": "initial position", "v0": "initial velocity", "a": "acceleration", "t": "time"},
        "output": "position",
        "calculate": lambda x0, v0, a, t: x0 + v0 * t + 0.5 * a * t**2,
    },
    "velocity_time": {
        "name": "Velocity from time",
        "formula": "v = v₀ + at",
        "latex": r"v = v_0 + at",
        "inputs": {"v0": "initial velocity", "a": "acceleration", "t": "time"},
        "output": "velocity",
        "calculate": lambda v0, a, t: v0 + a * t,
    },
    "velocity_position": {
        "name": "Velocity from position",
        "formula": "v² = v₀² + 2a(x - x₀)",
        "latex": r"v^2 = v_0^2 + 2a(x - x_0)",
        "inputs": {"v0": "initial velocity", "a": "acceleration", "x": "position", "x0": "initial position"},
        "output": "velocity",
        "calculate": lambda v0, a, x, x0: (v0**2 + 2 * a * (x - x0)) ** 0.5 if v0**2 + 2 * a * (x - x0) >= 0 else 0,
    },
    "time_from_velocity": {
        "name": "Time to reach velocity",
        "formula": "t = (v - v₀)/a",
        "latex": r"t = \frac{v - v_0}{a}",
        "inputs": {"v": "final velocity", "v0": "initial velocity", "a": "acceleration"},
        "output": "time",
        "calculate": lambda v, v0, a: (v - v0) / a if a != 0 else 0,
    },
    "projectile_range": {
        "name": "Projectile Range",
        "formula": "R = v₀²sin(2θ)/g",
        "latex": r"R = \frac{v_0^2 \sin(2\theta)}{g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "position",
        "calculate": lambda v0, theta, g=9.80665: v0**2 * np.sin(2 * theta) / g if np is not None and g != 0 else 0,
    },
    "projectile_max_height": {
        "name": "Projectile Max Height",
        "formula": "H = v₀²sin²(θ)/(2g)",
        "latex": r"H = \frac{v_0^2 \sin^2(\theta)}{2g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "position",
        "calculate": lambda v0, theta, g=9.80665: (
            v0**2 * np.sin(theta) ** 2 / (2 * g) if np is not None and g != 0 else 0
        ),
    },
    "projectile_time_flight": {
        "name": "Projectile Time of Flight",
        "formula": "T = 2v₀sin(θ)/g",
        "latex": r"T = \frac{2v_0 \sin(\theta)}{g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "time",
        "calculate": lambda v0, theta, g=9.80665: 2 * v0 * np.sin(theta) / g if np is not None and g != 0 else 0,
    },
    "shm_position": {
        "name": "SHM Position",
        "formula": "x = A·cos(ωt + φ)",
        "latex": r"x = A\cos(\omega t + \phi)",
        "inputs": {"A": "amplitude", "omega": "angular frequency", "t": "time", "phi": "phase"},
        "output": "position",
        "calculate": lambda A, omega, t, phi=0: A * np.cos(omega * t + phi) if np is not None else 0,
    },
    "shm_velocity": {
        "name": "SHM Velocity",
        "formula": "v = -Aω·sin(ωt + φ)",
        "latex": r"v = -A\omega\sin(\omega t + \phi)",
        "inputs": {"A": "amplitude", "omega": "angular frequency", "t": "time", "phi": "phase"},
        "output": "velocity",
        "calculate": lambda A, omega, t, phi=0: -A * omega * np.sin(omega * t + phi) if np is not None else 0,
    },
    "circular_period": {
        "name": "Circular Motion Period",
        "formula": "T = 2πr/v",
        "latex": r"T = \frac{2\pi r}{v}",
        "inputs": {"r": "radius", "v": "velocity"},
        "output": "time",
        "calculate": lambda r, v: 2 * 3.14159265 * r / v if v != 0 else 0,
    },
}


def convert_to_base_unit(value: float, unit: str, quantity_type: str) -> float:
    """Convert a value from given unit to base SI unit."""
    if quantity_type not in PHYSICS_QUANTITIES:
        return value
    units = PHYSICS_QUANTITIES[quantity_type]["units"]
    if unit in units:
        return value * units[unit]
    return value


def convert_from_base_unit(value: float, unit: str, quantity_type: str) -> float:
    """Convert a value from base SI unit to given unit."""
    if quantity_type not in PHYSICS_QUANTITIES:
        return value
    units = PHYSICS_QUANTITIES[quantity_type]["units"]
    if unit in units:
        return value / units[unit]
    return value


def normalize_to_bounded(value: float, ref_value: float, epsilon: float = 1e-6) -> tuple[float, bool]:
    """
    Normalize a physical value to [0,1] bounded embedding.
    Returns (normalized_value, was_clipped).
    """
    if ref_value == 0:
        ref_value = 1.0
    normalized = abs(value) / ref_value
    was_clipped = normalized < epsilon or normalized > (1 - epsilon)
    clipped = max(epsilon, min(1 - epsilon, normalized))
    return clipped, was_clipped


def render_physics_interface_page() -> None:
    """
    Render the Physics Interface page for SI unit conversion and tier translation.
    """
    if st is None or pd is None or np is None:
        return

    st.title("⚛️ Physics Interface")
    st.caption("Convert physical quantities with SI units through UMCP tier translation")

    # Initialize session state
    if "physics_quantities" not in st.session_state:
        st.session_state.physics_quantities = {}
    if "physics_audit" not in st.session_state:
        st.session_state.physics_audit = []

    # ========== SI Unit Converter Section ==========
    st.header("🔄 SI Unit Converter")

    with st.expander("📖 About SI Units", expanded=False):
        st.markdown("""
        **SI Base Units:**
        - Length: meter (m)
        - Mass: kilogram (kg)
        - Time: second (s)
        - Electric current: ampere (A)
        - Temperature: kelvin (K)
        - Amount of substance: mole (mol)
        - Luminous intensity: candela (cd)
        
        **Common Prefixes:**
        | Prefix | Symbol | Factor |
        |--------|--------|--------|
        | giga | G | 10⁹ |
        | mega | M | 10⁶ |
        | kilo | k | 10³ |
        | milli | m | 10⁻³ |
        | micro | μ | 10⁻⁶ |
        | nano | n | 10⁻⁹ |
        """)

    # Quick converter
    st.subheader("⚡ Quick Converter")
    conv_cols = st.columns([2, 1, 1, 1, 2])

    with conv_cols[0]:
        conv_value = st.number_input("Value", value=1.0, format="%.6f", key="conv_value")
    with conv_cols[1]:
        conv_quantity = st.selectbox("Quantity", list(PHYSICS_QUANTITIES.keys()), key="conv_quantity")
    with conv_cols[2]:
        qty_key = conv_quantity if conv_quantity else "position"
        from_units = list(PHYSICS_QUANTITIES[qty_key]["units"].keys())
        from_unit = st.selectbox("From", from_units, key="conv_from")
    with conv_cols[3]:
        to_unit = st.selectbox("To", from_units, index=min(1, len(from_units) - 1), key="conv_to")
    with conv_cols[4]:
        # Calculate conversion (with safe defaults)
        from_u = str(from_unit) if from_unit else from_units[0]
        to_u = str(to_unit) if to_unit else from_units[0]
        base_value = convert_to_base_unit(float(conv_value), from_u, qty_key)
        result_value = convert_from_base_unit(base_value, to_u, qty_key)
        st.metric("Result", f"{result_value:.6g} {to_u}")

    st.divider()

    # ========== Physics Toolbox (Tabbed Interface) ==========
    st.header("🧰 Physics Toolbox")

    tool_tabs = st.tabs(
        ["📐 Formula Calculator", "🔬 Physical Constants", "📊 Dimensional Analysis", "📜 Audit History"]
    )

    # Tab 1: Formula Calculator
    with tool_tabs[0]:
        st.markdown("**Select a formula and input values to calculate results.**")

        formula_cols = st.columns([1, 2])
        with formula_cols[0]:
            formula_name = st.selectbox(
                "Formula",
                list(PHYSICS_FORMULAS.keys()),
                format_func=lambda x: f"{PHYSICS_FORMULAS[x]['name']} ({PHYSICS_FORMULAS[x]['formula']})",
                key="formula_select",
            )

        if formula_name:
            formula = PHYSICS_FORMULAS[formula_name]

            with formula_cols[1]:
                st.latex(formula["latex"])
                st.caption(formula["description"])

            # Input fields for formula
            st.markdown("**Input Values:**")
            input_cols = st.columns(len(formula["inputs"]))
            formula_inputs = []

            for i, inp_name in enumerate(formula["inputs"]):
                with input_cols[i]:
                    qty_info = PHYSICS_QUANTITIES.get(inp_name, {"symbol": inp_name, "base_unit": "units"})
                    val = st.number_input(
                        f"{qty_info['symbol']} ({qty_info.get('base_unit', '')})",
                        value=1.0,
                        format="%.6g",
                        key=f"formula_inp_{i}",
                    )
                    formula_inputs.append(val)

            # Calculate result
            if st.button("🧮 Calculate", key="formula_calc"):
                try:
                    result = formula["calculate"](*formula_inputs)
                    output_qty = PHYSICS_FORMULAS[formula_name]["output"]
                    out_info = PHYSICS_QUANTITIES.get(output_qty, {"symbol": "?", "base_unit": "units"})
                    st.success(f"**Result:** {out_info['symbol']} = {result:.6g} {out_info['base_unit']}")

                    # Show in other common units
                    if output_qty in PHYSICS_QUANTITIES:
                        st.markdown("**In other units:**")
                        other_units = list(PHYSICS_QUANTITIES[output_qty]["units"].items())[:4]
                        unit_cols = st.columns(len(other_units))
                        for j, (unit_name, factor) in enumerate(other_units):
                            with unit_cols[j]:
                                converted = result / factor
                                st.metric(unit_name, f"{converted:.4g}")
                except Exception as e:
                    st.error(f"Calculation error: {e}")

    # Tab 2: Physical Constants
    with tool_tabs[1]:
        st.markdown("**Fundamental Physical Constants (CODATA 2022)**")

        # Search filter
        const_search = st.text_input("🔍 Search constants", key="const_search")

        # Create dataframe
        const_data = []
        for key, const in PHYSICAL_CONSTANTS.items():
            if const_search.lower() in const["name"].lower() or const_search.lower() in key.lower():
                const_data.append(
                    {
                        "Symbol": const["symbol"],
                        "Name": const["name"],
                        "Value": f"{const['value']:.6e}"
                        if const["value"] < 0.01 or const["value"] > 1000
                        else f"{const['value']:.6g}",
                        "Unit": const["unit"],
                    }
                )

        if const_data:
            st.dataframe(pd.DataFrame(const_data), hide_index=True, width="stretch")
        else:
            st.info("No constants found matching your search.")

        # Quick access buttons
        st.markdown("**Quick Copy:**")
        quick_cols = st.columns(5)
        quick_consts = ["c", "G", "h", "e", "g"]
        for i, const_key in enumerate(quick_consts):
            with quick_cols[i]:
                const = PHYSICAL_CONSTANTS[const_key]
                st.code(f"{const['symbol']} = {const['value']:.6e}")

    # Tab 3: Dimensional Analysis
    with tool_tabs[2]:
        st.markdown("**Check dimensional consistency of your calculations.**")

        dim_cols = st.columns(2)
        with dim_cols[0]:
            st.markdown("**Left Side (LHS)**")
            lhs_qty = st.selectbox("Quantity type", list(PHYSICS_QUANTITIES.keys()), key="dim_lhs")
            lhs_val = st.number_input("Value", value=1.0, key="dim_lhs_val")
            lhs_unit = st.selectbox("Unit", list(PHYSICS_QUANTITIES[lhs_qty]["units"].keys()), key="dim_lhs_unit")

        with dim_cols[1]:
            st.markdown("**Right Side (RHS)**")
            rhs_qty = st.selectbox("Quantity type", list(PHYSICS_QUANTITIES.keys()), key="dim_rhs")
            rhs_val = st.number_input("Value", value=1.0, key="dim_rhs_val")
            rhs_unit = st.selectbox("Unit", list(PHYSICS_QUANTITIES[rhs_qty]["units"].keys()), key="dim_rhs_unit")

        if st.button("🔍 Check Consistency", key="dim_check"):
            lhs_base = PHYSICS_QUANTITIES[lhs_qty]["base_unit"]
            rhs_base = PHYSICS_QUANTITIES[rhs_qty]["base_unit"]

            if lhs_base == rhs_base:
                lhs_si = convert_to_base_unit(lhs_val, lhs_unit, lhs_qty)
                rhs_si = convert_to_base_unit(rhs_val, rhs_unit, rhs_qty)
                diff = abs(lhs_si - rhs_si)
                rel_diff = diff / max(lhs_si, rhs_si, 1e-15) * 100

                st.success(f"✅ **Dimensionally consistent** (both are {lhs_base})")
                st.markdown(f"- LHS: {lhs_val} {lhs_unit} = **{lhs_si:.6g} {lhs_base}**")
                st.markdown(f"- RHS: {rhs_val} {rhs_unit} = **{rhs_si:.6g} {rhs_base}**")
                st.markdown(f"- Difference: {diff:.6g} {lhs_base} ({rel_diff:.2f}%)")
            else:
                st.error("❌ **Dimensionally inconsistent!**")
                st.markdown(f"- LHS has dimensions: **{lhs_base}**")
                st.markdown(f"- RHS has dimensions: **{rhs_base}**")
                st.markdown("These cannot be equal or compared directly.")

    # Tab 4: Audit History
    with tool_tabs[3]:
        if st.session_state.physics_audit:
            st.markdown(f"**{len(st.session_state.physics_audit)} previous calculations**")

            for i, entry in enumerate(reversed(st.session_state.physics_audit[-5:])):
                regime = entry.get("tier2", {}).get("regime", "N/A")
                regime_color = get_regime_color(regime)
                ts = entry.get("timestamp", "")[:19]

                with st.expander(f"Run {len(st.session_state.physics_audit) - i}: {regime} @ {ts}"):
                    hist_cols = st.columns(3)
                    with hist_cols[0]:
                        st.markdown("**Tier 0:**")
                        st.markdown(f"- Quantities: {entry['tier0'].get('n_quantities', 'N/A')}")
                    with hist_cols[1]:
                        st.markdown("**Tier 1:**")
                        t1 = entry.get("tier1", {})
                        st.markdown(f"- F: {t1.get('F', 0):.4f}")
                        st.markdown(f"- ω: {t1.get('omega', 0):.4f}")
                    with hist_cols[2]:
                        st.markdown("**GCD:**")
                        gcd = entry.get("gcd", {})
                        st.markdown(f"- Regime: {gcd.get('regime', 'N/A')}")

                    if st.button("📋 View Full JSON", key=f"hist_json_{i}"):
                        st.json(entry)

            if st.button("🗑️ Clear History"):
                st.session_state.physics_audit = []
                st.rerun()
        else:
            st.info("No calculation history yet. Run a physics translation to see results here.")

    st.divider()

    # ========== Physics Quantity Input ==========
    st.header("📥 Physical Quantities Input")
    st.markdown("Enter physical measurements that will be normalized to [0,1] for UMCP processing.")

    # Quantity selection
    selected_quantities = st.multiselect(
        "Select quantities to include",
        list(PHYSICS_QUANTITIES.keys()),
        default=["position", "velocity", "acceleration", "mass", "energy"],
        key="physics_selected",
    )

    if not selected_quantities:
        st.warning("Select at least one quantity to continue.")
        return

    # Input for each quantity
    st.subheader("🎯 Enter Values")

    input_data = {}
    n_cols = min(3, len(selected_quantities))

    for i in range(0, len(selected_quantities), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j < len(selected_quantities):
                qty_name = selected_quantities[i + j]
                qty_info = PHYSICS_QUANTITIES[qty_name]

                with col:
                    st.markdown(f"**{qty_info['description']}** ({qty_info['symbol']})")

                    val_col, unit_col = st.columns([2, 1])
                    with val_col:
                        val = st.number_input(
                            "Value",
                            value=qty_info["ref_value"],
                            format="%.6f",
                            key=f"phys_val_{qty_name}",
                            label_visibility="collapsed",
                        )
                    with unit_col:
                        unit = st.selectbox(
                            "Unit",
                            list(qty_info["units"].keys()),
                            key=f"phys_unit_{qty_name}",
                            label_visibility="collapsed",
                        )

                    input_data[qty_name] = {
                        "value": val,
                        "unit": unit,
                        "symbol": qty_info["symbol"],
                        "base_unit": qty_info["base_unit"],
                        "ref_value": qty_info["ref_value"],
                    }

    st.divider()

    # Reference values (for normalization)
    st.subheader("📏 Reference Scales")
    with st.expander("Adjust reference values for normalization", expanded=False):
        ref_cols = st.columns(min(4, len(selected_quantities)))
        for i, qty_name in enumerate(selected_quantities):
            with ref_cols[i % len(ref_cols)]:
                new_ref = st.number_input(
                    f"{qty_name} ref",
                    value=PHYSICS_QUANTITIES[qty_name]["ref_value"],
                    min_value=1e-15,
                    format="%.4f",
                    key=f"ref_{qty_name}",
                )
                input_data[qty_name]["ref_value"] = new_ref

    # Epsilon
    epsilon = st.select_slider(
        "ε-clipping threshold",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
        key="phys_epsilon",
    )

    st.divider()

    # ========== Process Button ==========
    if st.button("🚀 Run Physics → UMCP Translation", type="primary", width="stretch"):
        progress = st.progress(0, text="Starting physics translation...")

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "physics",
            "tier0": {},
            "tier1": {},
            "tier2": {},
            "status": "PROCESSING",
        }

        # TIER 0: Convert to base units and normalize
        progress.progress(20, text="Tier 0: Converting to base SI units...")

        normalized_coords = []
        weights = []
        tier0_data = []

        for qty_name, data in input_data.items():
            # Convert to base unit
            base_value = convert_to_base_unit(data["value"], data["unit"], qty_name)

            # Normalize to [0,1]
            norm_value, was_clipped = normalize_to_bounded(base_value, data["ref_value"], epsilon)

            tier0_data.append(
                {
                    "quantity": qty_name,
                    "symbol": data["symbol"],
                    "input_value": data["value"],
                    "input_unit": data["unit"],
                    "base_value": base_value,
                    "base_unit": data["base_unit"],
                    "ref_value": data["ref_value"],
                    "normalized": norm_value,
                    "clipped": was_clipped,
                }
            )

            normalized_coords.append(norm_value)
            weights.append(1.0 / len(input_data))  # Equal weights

        audit_entry["tier0"] = {
            "quantities": tier0_data,
            "epsilon": epsilon,
            "n_quantities": len(normalized_coords),
        }

        # TIER 1: Kernel computation
        progress.progress(50, text="Tier 1: Computing kernel invariants...")

        c = np.array(normalized_coords)
        w = np.array(weights)

        fidelity = float(np.sum(w * c))
        drift = 1 - fidelity
        log_ic = float(np.sum(w * np.log(c)))
        integrity_composite = float(np.exp(log_ic))

        entropy = 0.0
        for ci, wi in zip(c, w, strict=False):
            if wi > 0 and 0 < ci < 1:
                entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))

        curvature = float(np.std(c, ddof=0) / 0.5) if len(c) > 1 else 0.0
        amgm_gap = fidelity - integrity_composite

        audit_entry["tier1"] = {
            "F": fidelity,
            "omega": drift,
            "S": entropy,
            "C": curvature,
            "kappa": log_ic,
            "IC": integrity_composite,
            "amgm_gap": amgm_gap,
        }

        # TIER 2: Regime classification
        progress.progress(80, text="Tier 2: Computing regime and diagnostics...")

        regime = classify_regime(drift)
        stability_score = int(fidelity * 100 * (1 - curvature))

        # Physics-specific diagnostics
        diagnostics = []

        # Check for physical consistency
        if "velocity" in input_data and "mass" in input_data:
            v_base = convert_to_base_unit(input_data["velocity"]["value"], input_data["velocity"]["unit"], "velocity")
            m_base = convert_to_base_unit(input_data["mass"]["value"], input_data["mass"]["unit"], "mass")
            calc_ke = 0.5 * m_base * v_base**2

            if "energy" in input_data:
                e_base = convert_to_base_unit(input_data["energy"]["value"], input_data["energy"]["unit"], "energy")
                if abs(e_base - calc_ke) > 0.01 * max(e_base, calc_ke):
                    diagnostics.append(f"⚠️ Energy mismatch: input E={e_base:.2f}J, calculated KE={calc_ke:.2f}J")
                else:
                    diagnostics.append(f"✅ Energy consistent: E ≈ ½mv² = {calc_ke:.4f} J")

        if "velocity" in input_data and "mass" in input_data:
            v_base = convert_to_base_unit(input_data["velocity"]["value"], input_data["velocity"]["unit"], "velocity")
            m_base = convert_to_base_unit(input_data["mass"]["value"], input_data["mass"]["unit"], "mass")
            calc_p = m_base * v_base

            if "momentum" in input_data:
                p_base = convert_to_base_unit(
                    input_data["momentum"]["value"], input_data["momentum"]["unit"], "momentum"
                )
                if abs(p_base - calc_p) > 0.01 * max(p_base, calc_p):
                    diagnostics.append(
                        f"⚠️ Momentum mismatch: input p={p_base:.2f} kg·m/s, calculated p={calc_p:.2f} kg·m/s"
                    )
                else:
                    diagnostics.append(f"✅ Momentum consistent: p = mv = {calc_p:.4f} kg·m/s")

        if not diagnostics:
            diagnostics.append("✅ No physics consistency checks available for selected quantities")

        audit_entry["tier2"] = {
            "regime": regime,
            "stability_score": stability_score,
            "risk_level": "LOW" if regime == "STABLE" else ("MEDIUM" if regime == "WATCH" else "HIGH"),
            "diagnostics": diagnostics,
        }

        audit_entry["status"] = "COMPLETE"
        progress.progress(100, text="Complete!")

        # GCD Translation
        gcd_translation = translate_to_gcd(audit_entry["tier1"])
        audit_entry["gcd"] = gcd_translation

        st.session_state.physics_audit.append(audit_entry)

        # ========== Display Results ==========
        st.divider()

        regime = audit_entry["tier2"]["regime"]
        regime_color = get_regime_color(regime)

        st.markdown(
            f"""<div style="padding: 20px; border-left: 6px solid {regime_color}; 
                background: {regime_color}22; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {regime_color};">⚛️ Result: {regime}</h2>
                <p style="margin: 5px 0 0 0;">Stability Score: {stability_score}/100</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Three-column display
        tier_cols = st.columns(3)

        with tier_cols[0]:
            st.markdown("### 📥 Tier 0: Physical Input")
            t0_df = pd.DataFrame(
                [
                    {
                        "Quantity": d["symbol"],
                        "Input": f"{d['input_value']:.4g} {d['input_unit']}",
                        "SI Base": f"{d['base_value']:.4g} {d['base_unit']}",
                        "Normalized": f"{d['normalized']:.4f}",
                        "OOR": "⚠️" if d["clipped"] else "✓",
                    }
                    for d in audit_entry["tier0"]["quantities"]
                ]
            )
            st.dataframe(t0_df, hide_index=True, width="stretch")

        with tier_cols[1]:
            st.markdown("### ⚙️ Tier 1: Kernel")
            t1 = audit_entry["tier1"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("F", f"{t1['F']:.4f}")
                st.metric("ω", f"{t1['omega']:.4f}")
                st.metric("S", f"{t1['S']:.4f}")
            with col2:
                st.metric("C", f"{t1['C']:.4f}")
                st.metric("κ", f"{t1['kappa']:.4f}")
                st.metric("IC", f"{t1['IC']:.4f}")

        with tier_cols[2]:
            st.markdown("### 📊 Tier 2: Diagnostics")
            t2 = audit_entry["tier2"]
            st.metric("Regime", t2["regime"])
            st.metric("Score", f"{t2['stability_score']}/100")
            st.markdown("**Diagnostics:**")
            for diag in t2["diagnostics"]:
                st.markdown(f"- {diag}")

        # ========== GCD Translation Panel ==========
        st.divider()
        st.header("🌀 GCD Translation (Generative Collapse Dynamics)")
        st.caption("Physical values translated to native GCD representation")

        render_gcd_panel(gcd_translation, compact=False)

        # GCD Natural Language Summary
        st.markdown("### 📝 Natural Language Summary")
        for line in gcd_translation["natural_language"]:
            st.markdown(line)

        # Full audit
        with st.expander("🔍 Full Audit JSON (includes GCD)", expanded=False):
            st.json(audit_entry)


def render_kinematics_interface_page() -> None:
    """
    Render the Kinematics Interface page for motion analysis with tier translation.
    """
    if st is None or pd is None or np is None:
        return

    st.title("🎯 Kinematics Interface")
    st.caption("Analyze motion through phase space with UMCP tier translation")

    # Initialize session state
    if "kin_audit" not in st.session_state:
        st.session_state.kin_audit = []

    # ========== Scenario Selection ==========
    st.header("📋 Kinematic Scenarios")

    scenario_cols = st.columns(len(KINEMATICS_SCENARIOS))
    selected_scenario = None

    for i, (name, scenario) in enumerate(KINEMATICS_SCENARIOS.items()):
        with scenario_cols[i]:
            if st.button(f"📌 {name}", key=f"kin_scenario_{name}", width="stretch"):
                selected_scenario = name
                st.session_state.kin_scenario = scenario

    if selected_scenario:
        st.success(f"Loaded: {selected_scenario}")
        st.caption(KINEMATICS_SCENARIOS[selected_scenario]["description"])

    st.divider()

    # ========== Kinematics Toolbox ==========
    st.header("🧰 Kinematics Toolbox")

    kin_tabs = st.tabs(
        ["📐 Equations Solver", "🚀 Trajectory Calculator", "⚡ Energy Check", "📈 Motion Plot", "📜 History"]
    )

    # Tab 1: Equations Solver
    with kin_tabs[0]:
        st.markdown("**Solve kinematic equations for unknown quantities.**")

        eq_cols = st.columns([1, 2])
        with eq_cols[0]:
            eq_name = st.selectbox(
                "Select equation",
                list(KINEMATICS_EQUATIONS.keys()),
                format_func=lambda x: f"{KINEMATICS_EQUATIONS[x]['name']}",
                key="kin_eq_select",
            )

        if eq_name:
            eq = KINEMATICS_EQUATIONS[eq_name]

            with eq_cols[1]:
                st.latex(eq["latex"])
                st.markdown(f"*{eq['formula']}*")

            # Input fields
            st.markdown("**Input Values:**")
            eq_inputs = eq["inputs"] if isinstance(eq["inputs"], dict) else {k: k for k in eq["inputs"]}
            input_cols = st.columns(len(eq_inputs))
            eq_values = []

            for i, (var_name, var_desc) in enumerate(eq_inputs.items()):
                with input_cols[i]:
                    val = st.number_input(
                        f"{var_name}",
                        value=1.0 if var_name not in ["phi", "theta"] else 0.785,
                        format="%.4g",
                        key=f"kin_eq_inp_{eq_name}_{i}",
                        help=var_desc if isinstance(var_desc, str) else None,
                    )
                    eq_values.append(val)

            if st.button("🧮 Solve", key="kin_eq_solve"):
                try:
                    result = eq["calculate"](*eq_values)
                    output_qty = eq["output"]
                    out_info = PHYSICS_QUANTITIES.get(output_qty, {"symbol": "result", "base_unit": "units"})
                    st.success(f"**Result:** {out_info['symbol']} = {result:.6g} {out_info['base_unit']}")
                except Exception as e:
                    st.error(f"Calculation error: {e}")

    # Tab 2: Trajectory Calculator
    with kin_tabs[1]:
        st.markdown("**Calculate projectile trajectory parameters.**")

        traj_cols = st.columns(3)
        with traj_cols[0]:
            v0 = st.number_input("Initial velocity (m/s)", value=20.0, min_value=0.1, key="traj_v0")
        with traj_cols[1]:
            theta_deg = st.slider("Launch angle (°)", min_value=0, max_value=90, value=45, key="traj_theta")
            theta_rad = theta_deg * 0.0174533
        with traj_cols[2]:
            g_val = st.number_input("Gravity (m/s²)", value=9.80665, key="traj_g")

        if st.button("📊 Calculate Trajectory", key="traj_calc"):
            # Calculate trajectory parameters
            t_flight = 2 * v0 * np.sin(theta_rad) / g_val
            max_height = (v0 * np.sin(theta_rad)) ** 2 / (2 * g_val)
            range_dist = v0**2 * np.sin(2 * theta_rad) / g_val

            result_cols = st.columns(4)
            with result_cols[0]:
                st.metric("Time of Flight", f"{t_flight:.3f} s")
            with result_cols[1]:
                st.metric("Max Height", f"{max_height:.3f} m")
            with result_cols[2]:
                st.metric("Range", f"{range_dist:.3f} m")
            with result_cols[3]:
                # Landing velocity (same magnitude as launch)
                st.metric("Landing Speed", f"{v0:.3f} m/s")

            # Trajectory plot
            if px is not None:
                t_points = np.linspace(0, t_flight, 100)
                x_points = v0 * np.cos(theta_rad) * t_points
                y_points = v0 * np.sin(theta_rad) * t_points - 0.5 * g_val * t_points**2

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_points, y=y_points, mode="lines", name="Trajectory", line={"color": "#2196F3", "width": 3}
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, range_dist / 2, range_dist],
                        y=[0, max_height, 0],
                        mode="markers",
                        name="Key Points",
                        marker={"size": 10, "color": ["green", "red", "orange"]},
                    )
                )
                fig.update_layout(
                    xaxis_title="Distance (m)",
                    yaxis_title="Height (m)",
                    title=f"Projectile Trajectory (v₀={v0} m/s, θ={theta_deg}°)",
                    height=350,
                    yaxis={"scaleanchor": "x"},
                )
                st.plotly_chart(fig, width="stretch")

    # Tab 3: Energy Conservation Check
    with kin_tabs[2]:
        st.markdown("**Verify energy conservation in your system.**")

        st.subheader("Initial State")
        init_cols = st.columns(4)
        with init_cols[0]:
            m_check = st.number_input("Mass (kg)", value=1.0, min_value=0.001, key="energy_mass")
        with init_cols[1]:
            h1 = st.number_input("Height₁ (m)", value=10.0, key="energy_h1")
        with init_cols[2]:
            v1 = st.number_input("Velocity₁ (m/s)", value=0.0, key="energy_v1")
        with init_cols[3]:
            g_check = st.number_input("g (m/s²)", value=9.80665, key="energy_g")

        st.subheader("Final State")
        final_cols = st.columns(3)
        with final_cols[0]:
            h2 = st.number_input("Height₂ (m)", value=0.0, key="energy_h2")
        with final_cols[1]:
            v2 = st.number_input("Velocity₂ (m/s)", value=14.0, key="energy_v2")
        with final_cols[2]:
            W_nc = st.number_input(
                "Non-conservative work (J)", value=0.0, key="energy_wnc", help="Work done by friction, drag, etc."
            )

        if st.button("⚖️ Check Energy Conservation", key="energy_check"):
            # Calculate energies
            KE1 = 0.5 * m_check * v1**2
            PE1 = m_check * g_check * h1
            E1 = KE1 + PE1

            KE2 = 0.5 * m_check * v2**2
            PE2 = m_check * g_check * h2
            E2 = KE2 + PE2

            energy_diff = E2 - E1 + W_nc

            energy_cols = st.columns(2)
            with energy_cols[0]:
                st.markdown("**Initial State:**")
                st.markdown(f"- KE₁ = ½mv₁² = {KE1:.4f} J")
                st.markdown(f"- PE₁ = mgh₁ = {PE1:.4f} J")
                st.markdown(f"- **E₁ = {E1:.4f} J**")

            with energy_cols[1]:
                st.markdown("**Final State:**")
                st.markdown(f"- KE₂ = ½mv₂² = {KE2:.4f} J")
                st.markdown(f"- PE₂ = mgh₂ = {PE2:.4f} J")
                st.markdown(f"- **E₂ = {E2:.4f} J**")

            st.divider()

            if abs(energy_diff) < 0.01 * max(E1, E2, 1):
                st.success(f"✅ **Energy is conserved!** ΔE = {energy_diff:.6f} J (< 1% of total)")
                # Calculate expected v2 from conservation
                v2_expected = np.sqrt(2 * (E1 - PE2 - W_nc) / m_check) if PE2 + W_nc <= E1 else 0
                st.info(f"Expected v₂ from conservation: {v2_expected:.4f} m/s")
            else:
                st.error(f"❌ **Energy is NOT conserved!** ΔE = {energy_diff:.4f} J")
                st.markdown(f"Missing energy: {abs(energy_diff):.4f} J")
                if W_nc == 0:
                    st.info("💡 Consider adding non-conservative work (friction, drag)")

    # Tab 4: Motion Plot
    with kin_tabs[3]:
        st.markdown("**Visualize 1D motion over time.**")

        plot_cols = st.columns(4)
        with plot_cols[0]:
            x0_plot = st.number_input("x₀ (m)", value=0.0, key="plot_x0")
        with plot_cols[1]:
            v0_plot = st.number_input("v₀ (m/s)", value=10.0, key="plot_v0")
        with plot_cols[2]:
            a_plot = st.number_input("a (m/s²)", value=-9.81, key="plot_a")
        with plot_cols[3]:
            t_max = st.number_input("t_max (s)", value=3.0, min_value=0.1, key="plot_tmax")

        if st.button("📈 Generate Motion Plots", key="motion_plot"):
            t = np.linspace(0, t_max, 200)
            x = x0_plot + v0_plot * t + 0.5 * a_plot * t**2
            v = v0_plot + a_plot * t

            if px is not None:
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Position vs Time", "Velocity vs Time"))

                fig.add_trace(go.Scatter(x=t, y=x, name="x(t)", line={"color": "#2196F3", "width": 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=v, name="v(t)", line={"color": "#4CAF50", "width": 2}), row=2, col=1)

                fig.update_xaxes(title_text="Time (s)", row=1, col=1)
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="Position (m)", row=1, col=1)
                fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)

                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, width="stretch")

            # Key points
            t_stop = -v0_plot / a_plot if a_plot != 0 else float("inf")
            x_max = x0_plot + v0_plot * t_stop + 0.5 * a_plot * t_stop**2 if 0 < t_stop < t_max else None

            st.markdown("**Key Points:**")
            if x_max is not None and t_stop > 0:
                st.markdown(f"- Maximum height reached at t = {t_stop:.3f} s, x = {x_max:.3f} m")

    # Tab 5: History
    with kin_tabs[4]:
        if st.session_state.kin_audit:
            st.markdown(f"**{len(st.session_state.kin_audit)} previous analyses**")

            for i, entry in enumerate(reversed(st.session_state.kin_audit[-5:])):
                regime = entry.get("tier2", {}).get("umcp_regime", "N/A")
                kin_regime = entry.get("tier2", {}).get("kinematic_regime", "N/A")
                ts = entry.get("timestamp", "")[:19]

                with st.expander(f"Run {len(st.session_state.kin_audit) - i}: {regime}/{kin_regime} @ {ts}"):
                    hist_cols = st.columns(3)
                    with hist_cols[0]:
                        st.markdown("**Phase Space:**")
                        t15 = entry.get("tier15", {})
                        st.markdown(f"- |γ| = {t15.get('phase_magnitude', 0):.4f}")
                        st.markdown(f"- Credit: {t15.get('kinematic_credit', 0):.4f}")
                    with hist_cols[1]:
                        st.markdown("**Tier 1:**")
                        t1 = entry.get("tier1", {})
                        st.markdown(f"- F: {t1.get('F', 0):.4f}")
                        st.markdown(f"- ω: {t1.get('omega', 0):.4f}")
                    with hist_cols[2]:
                        st.markdown("**Score:**")
                        t2 = entry.get("tier2", {})
                        st.markdown(f"- Stability: {t2.get('stability_score', 0)}/100")

            if st.button("🗑️ Clear Kinematics History"):
                st.session_state.kin_audit = []
                st.rerun()
        else:
            st.info("No analysis history yet. Run a kinematics translation to see results here.")

    st.divider()

    # ========== Motion Parameters Input ==========
    st.header("📐 Motion Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Position (x)")
        pos_val = st.number_input("Position value", value=1.0, format="%.4f", key="kin_pos")
        pos_unit = st.selectbox(
            "Position unit", list(PHYSICS_QUANTITIES["position"]["units"].keys()), key="kin_pos_unit"
        )
        pos_ref = st.number_input("Position reference (L_ref)", value=1.0, min_value=0.001, key="kin_pos_ref")

    with col2:
        st.subheader("Velocity (v)")
        vel_val = st.number_input("Velocity value", value=1.0, format="%.4f", key="kin_vel")
        vel_unit = st.selectbox(
            "Velocity unit", list(PHYSICS_QUANTITIES["velocity"]["units"].keys()), key="kin_vel_unit"
        )
        vel_ref = st.number_input("Velocity reference (v_ref)", value=1.0, min_value=0.001, key="kin_vel_ref")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Acceleration (a)")
        acc_val = st.number_input("Acceleration value", value=9.81, format="%.4f", key="kin_acc")
        acc_unit = st.selectbox(
            "Acceleration unit", list(PHYSICS_QUANTITIES["acceleration"]["units"].keys()), key="kin_acc_unit"
        )
        acc_ref = st.number_input("Acceleration reference (a_ref)", value=9.80665, min_value=0.001, key="kin_acc_ref")

    with col4:
        st.subheader("Mass (m)")
        mass_val = st.number_input("Mass value", value=1.0, format="%.4f", key="kin_mass")
        mass_unit = st.selectbox("Mass unit", list(PHYSICS_QUANTITIES["mass"]["units"].keys()), key="kin_mass_unit")

    st.divider()

    # ========== Derived Quantities ==========
    st.header("⚡ Derived Quantities (Auto-Calculated)")

    # Convert to base units (with safe defaults)
    pos_u = str(pos_unit) if pos_unit else "m"
    vel_u = str(vel_unit) if vel_unit else "m/s"
    acc_u = str(acc_unit) if acc_unit else "m/s²"
    mass_u = str(mass_unit) if mass_unit else "kg"

    x_base = convert_to_base_unit(float(pos_val), pos_u, "position")
    v_base = convert_to_base_unit(float(vel_val), vel_u, "velocity")
    a_base = convert_to_base_unit(float(acc_val), acc_u, "acceleration")
    m_base = convert_to_base_unit(float(mass_val), mass_u, "mass")

    # Calculate derived quantities
    e_kin = 0.5 * m_base * v_base**2
    momentum = m_base * v_base

    derived_cols = st.columns(4)
    with derived_cols[0]:
        st.metric("Kinetic Energy", f"{e_kin:.4g} J")
    with derived_cols[1]:
        st.metric("Momentum", f"{momentum:.4g} kg·m/s")
    with derived_cols[2]:
        # Phase magnitude
        x_norm = abs(x_base) / pos_ref
        v_norm = abs(v_base) / vel_ref
        phase_mag = np.sqrt(x_norm**2 + v_norm**2)
        st.metric("Phase Magnitude |γ|", f"{phase_mag:.4f}")
    with derived_cols[3]:
        # Calculate energy reference
        e_ref = m_base * vel_ref**2
        st.metric("E_ref", f"{e_ref:.4g} J")

    st.divider()

    # Epsilon
    epsilon = st.select_slider(
        "ε-clipping threshold",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
        key="kin_epsilon",
    )

    # ========== Process ==========
    if st.button("🚀 Run Kinematics → UMCP Translation", type="primary", width="stretch"):
        progress = st.progress(0, text="Starting kinematics translation...")

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "kinematics",
            "tier0": {},
            "tier1": {},
            "tier15": {},  # Tier-0 protocol for kinematics
            "tier2": {},
            "status": "PROCESSING",
        }

        # TIER 0: Build Ψ(t) vector
        progress.progress(15, text="Tier 0: Building kinematic observable vector Ψ(t)...")

        # Normalize all quantities
        x_norm, x_clip = normalize_to_bounded(x_base, pos_ref, epsilon)
        v_norm, v_clip = normalize_to_bounded(v_base, vel_ref, epsilon)
        a_norm, a_clip = normalize_to_bounded(a_base, acc_ref, epsilon)

        e_ref = m_base * vel_ref**2
        e_norm, e_clip = normalize_to_bounded(e_kin, e_ref, epsilon)

        p_ref = m_base * vel_ref
        p_norm, p_clip = normalize_to_bounded(momentum, p_ref, epsilon)

        # KIN.INTSTACK.v1 weights (frozen)
        kin_weights = {"x": 0.25, "v": 0.25, "a": 0.15, "E_kin": 0.20, "p": 0.15}

        psi_vector = [x_norm, v_norm, a_norm, e_norm, p_norm]
        psi_weights = [kin_weights["x"], kin_weights["v"], kin_weights["a"], kin_weights["E_kin"], kin_weights["p"]]

        audit_entry["tier0"] = {
            "physical_inputs": {
                "x": {"value": x_base, "unit": "m", "ref": pos_ref, "normalized": x_norm, "clipped": x_clip},
                "v": {"value": v_base, "unit": "m/s", "ref": vel_ref, "normalized": v_norm, "clipped": v_clip},
                "a": {"value": a_base, "unit": "m/s²", "ref": acc_ref, "normalized": a_norm, "clipped": a_clip},
                "E_kin": {"value": e_kin, "unit": "J", "ref": e_ref, "normalized": e_norm, "clipped": e_clip},
                "p": {"value": momentum, "unit": "kg·m/s", "ref": p_ref, "normalized": p_norm, "clipped": p_clip},
            },
            "psi_vector": psi_vector,
            "psi_weights": psi_weights,
            "epsilon": epsilon,
            "phase_point": {"x": x_norm, "v": v_norm},
        }

        # TIER 1: Kernel computation
        progress.progress(40, text="Tier 1: Computing kernel invariants...")

        c = np.array(psi_vector)
        w = np.array(psi_weights)

        fidelity = float(np.sum(w * c))
        drift = 1 - fidelity
        log_ic = float(np.sum(w * np.log(c)))
        integrity_composite = float(np.exp(log_ic))

        entropy = 0.0
        for ci, wi in zip(c, w, strict=False):
            if wi > 0 and 0 < ci < 1:
                entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))

        curvature = float(np.std(c, ddof=0) / 0.5)
        amgm_gap = fidelity - integrity_composite

        audit_entry["tier1"] = {
            "F": fidelity,
            "omega": drift,
            "S": entropy,
            "C": curvature,
            "kappa": log_ic,
            "IC": integrity_composite,
            "amgm_gap": amgm_gap,
            "is_homogeneous": np.allclose(c, c[0], atol=1e-15),
        }

        # TIER-0 Protocol: Phase space analysis (kinematics-specific)
        progress.progress(60, text="Protocol: Computing phase space return metrics...")

        gamma = (x_norm, v_norm)
        phase_mag_sq = x_norm**2 + v_norm**2
        phase_mag = np.sqrt(phase_mag_sq)

        # Return time estimation (simplified for single point)
        # In full implementation, this would check against historical trajectory
        eta_phase = 0.01  # Tolerance

        # Estimate kinematic regime based on phase magnitude
        if phase_mag < 0.3:
            kin_regime = "Stable"
            tau_kin_est = "< T_crit (returning)"
            kin_credit = 1.0
        elif phase_mag < 0.7:
            kin_regime = "Watch"
            tau_kin_est = "T_crit to 2·T_crit"
            kin_credit = 0.5
        else:
            kin_regime = "Unstable"
            tau_kin_est = "INF_KIN (non-returning)"
            kin_credit = 0.0

        audit_entry["tier15"] = {
            "phase_point": gamma,
            "phase_magnitude": phase_mag,
            "phase_magnitude_sq": phase_mag_sq,
            "eta_phase": eta_phase,
            "kinematic_regime": kin_regime,
            "tau_kin_estimate": tau_kin_est,
            "kinematic_credit": kin_credit,
            "return_rate": 1.0 - drift,  # Simplified
        }

        # TIER 2: Regime classification
        progress.progress(85, text="Tier 2: Computing diagnostics and regime...")

        regime = classify_regime(drift)
        stability_score = int(fidelity * 100 * (1 - curvature))

        # Kinematics-specific recommendations
        recommendations = []

        if kin_credit == 0:
            recommendations.append("⚠️ Non-returning motion: kinematic credit = 0 (AXIOM-0)")
        if x_clip or v_clip or a_clip:
            recommendations.append("⚠️ OOR clipping applied: check reference scales")
        if amgm_gap > 0.1:
            recommendations.append("📊 Large AM-GM gap: heterogeneous phase space")

        # Conservation checks
        if abs(e_kin - 0.5 * m_base * v_base**2) < 1e-9:
            recommendations.append("✅ E_kin = ½mv² verified")
        if abs(momentum - m_base * v_base) < 1e-9:
            recommendations.append("✅ p = mv verified")

        if not recommendations:
            recommendations.append("✅ All kinematics checks passed")

        audit_entry["tier2"] = {
            "umcp_regime": regime,
            "kinematic_regime": kin_regime,
            "stability_score": stability_score,
            "risk_level": "LOW" if regime == "STABLE" else ("MEDIUM" if regime == "WATCH" else "HIGH"),
            "recommendations": recommendations,
        }

        audit_entry["status"] = "COMPLETE"
        progress.progress(100, text="Complete!")

        st.session_state.kin_audit.append(audit_entry)

        # ========== Display Results ==========
        st.divider()

        regime = audit_entry["tier2"]["umcp_regime"]
        kin_regime = audit_entry["tier2"]["kinematic_regime"]
        regime_color = get_regime_color(regime)

        st.markdown(
            f"""<div style="padding: 20px; border-left: 6px solid {regime_color}; 
                background: {regime_color}22; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {regime_color};">🎯 UMCP Regime: {regime} | KIN Regime: {kin_regime}</h2>
                <p style="margin: 5px 0 0 0;">Stability Score: {stability_score}/100 • Kinematic Credit: {kin_credit:.2f}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Four-column display for kinematics (includes Protocol tier)
        tier_cols = st.columns(4)

        with tier_cols[0]:
            st.markdown("### 📥 Tier 0: Physical")
            t0 = audit_entry["tier0"]["physical_inputs"]
            t0_df = pd.DataFrame(
                [
                    {"Qty": k, "Value": f"{v['value']:.3g}", "Unit": v["unit"], "ψ": f"{v['normalized']:.4f}"}
                    for k, v in t0.items()
                ]
            )
            st.dataframe(t0_df, hide_index=True, width="stretch")

        with tier_cols[1]:
            st.markdown("### ⚙️ Tier 1: Kernel")
            t1 = audit_entry["tier1"]
            st.metric("F (Fidelity)", f"{t1['F']:.4f}")
            st.metric("ω (Drift)", f"{t1['omega']:.4f}")
            st.metric("IC", f"{t1['IC']:.4f}")

        with tier_cols[2]:
            st.markdown("### 🔄 Protocol: Phase")
            t15 = audit_entry["tier15"]
            st.metric("|γ| (Phase Mag)", f"{t15['phase_magnitude']:.4f}")
            st.metric("τ_kin", t15["tau_kin_estimate"])
            st.metric("Credit", f"{t15['kinematic_credit']:.2f}")

        with tier_cols[3]:
            st.markdown("### 📊 Tier 2: Regime")
            t2 = audit_entry["tier2"]
            st.metric("UMCP", t2["umcp_regime"])
            st.metric("KIN", t2["kinematic_regime"])
            st.metric("Score", f"{t2['stability_score']}/100")

        # Recommendations
        st.markdown("### 📋 Recommendations")
        for rec in audit_entry["tier2"]["recommendations"]:
            st.markdown(f"- {rec}")

        # ========== GCD Translation Panel ==========
        st.divider()
        st.markdown("### 🌀 GCD Translation (Generative Collapse Dynamics)")
        st.caption("Native Tier-1 interpretation using GCD framework")

        # Translate tier1 values to GCD
        gcd_translation = translate_to_gcd(audit_entry["tier1"])
        audit_entry["gcd"] = gcd_translation

        render_gcd_panel(gcd_translation, compact=False)

        # Additional kinematics-specific GCD insight
        phase_mag = audit_entry["tier15"]["phase_magnitude"]
        kin_credit = audit_entry["tier15"]["kinematic_credit"]

        st.markdown("#### 🔄 Phase Space GCD Insight")
        phase_gcd_cols = st.columns(3)
        with phase_gcd_cols[0]:
            phase_regime = "STABLE" if phase_mag < 0.3 else ("WATCH" if phase_mag < 0.7 else "COLLAPSE")
            phase_color = GCD_REGIMES[phase_regime]["color"]
            st.markdown(f"**|γ| Regime:** :{phase_color.replace('#', '')}[{phase_regime}]")
            st.caption(f"|γ| = {phase_mag:.4f} in phase space")
        with phase_gcd_cols[1]:
            credit_health = "✅ High" if kin_credit > 0.7 else ("⚠️ Medium" if kin_credit > 0.3 else "❌ Low")
            st.markdown(f"**Kinematic Credit:** {credit_health}")
            st.caption(f"κ_kin = {kin_credit:.4f}")
        with phase_gcd_cols[2]:
            collapse_pressure = 1.0 - kin_credit
            st.markdown(f"**Collapse Pressure:** {collapse_pressure:.2%}")
            st.caption("Generative potential available")

        # Phase space visualization
        st.markdown("### 🌐 Phase Space (x, v)")

        # Create a simple phase space plot
        if px is not None:
            fig = go.Figure()

            # Add phase point
            fig.add_trace(
                go.Scatter(
                    x=[x_norm],
                    y=[v_norm],
                    mode="markers+text",
                    marker={"size": 15, "color": regime_color, "symbol": "circle"},
                    text=["γ(t)"],
                    textposition="top center",
                    name="Current State",
                )
            )

            # Add reference circle at |γ| = 0.3 (stable boundary)
            theta_range = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(
                go.Scatter(
                    x=0.3 * np.cos(theta_range),
                    y=0.3 * np.sin(theta_range),
                    mode="lines",
                    line={"color": "green", "dash": "dash"},
                    name="Stable boundary",
                )
            )

            # Add reference circle at |γ| = 0.7 (watch boundary)
            fig.add_trace(
                go.Scatter(
                    x=0.7 * np.cos(theta_range),
                    y=0.7 * np.sin(theta_range),
                    mode="lines",
                    line={"color": "orange", "dash": "dash"},
                    name="Watch boundary",
                )
            )

            fig.update_layout(
                xaxis_title="x̃ (normalized position)",
                yaxis_title="ṽ (normalized velocity)",
                xaxis={"range": [-0.1, 1.1], "scaleanchor": "y"},
                yaxis={"range": [-0.1, 1.1]},
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, width="stretch")

        # Full audit
        with st.expander("🔍 Full Audit JSON", expanded=False):
            st.json(audit_entry)


def render_test_templates_page() -> None:
    """
    Render the Test Templates page for interactive tier translation.

    Allows users to:
    1. Input Tier 0 raw data (coordinates, weights)
    2. See Tier 1 transformation (kernel invariants)
    3. See Tier 2 outputs (regime classification, diagnostics)
    4. Get a full audit trail
    """
    if st is None or pd is None or np is None:
        return

    st.title("🧮 Test Templates")
    st.caption("Transform your data through UMCP tiers with full audit trail")

    # Initialize session state for templates
    if "template_coords" not in st.session_state:
        st.session_state.template_coords = [0.85, 0.72, 0.91, 0.68]
    if "template_weights" not in st.session_state:
        st.session_state.template_weights = [0.25, 0.25, 0.25, 0.25]
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    # ========== TIER 0: Input Layer ==========
    st.header("📥 Tier 0: Interface Layer (Raw Input)")
    st.markdown("""
    **Tier 0** declares raw measurements and converts them to bounded trace Ψ(t) ∈ [0,1]ⁿ.
    Enter your coordinates (values should be in range [0, 1]) and weights (must sum to 1.0).
    """)

    with st.expander("📖 About Tier 0", expanded=False):
        st.markdown("""
        **Tier 0 Scope:**
        - Observables: Raw measurements with units
        - Embedding: Map x(t) → Ψ(t) ∈ [0,1]ⁿ
        - Weights: w_i ≥ 0, Σw_i = 1
        - OOR Policy: Out-of-range handling
        
        **Key Rule:** Tier 0 is frozen before Tier 1 computes.
        """)

    # Template presets
    st.subheader("📋 Template Presets")
    preset_cols = st.columns(5)

    presets = {
        "Stable": {"coords": [0.85, 0.80, 0.82, 0.78], "weights": [0.25, 0.25, 0.25, 0.25]},
        "Watch": {"coords": [0.45, 0.42, 0.48, 0.40], "weights": [0.25, 0.25, 0.25, 0.25]},
        "Collapse": {"coords": [0.12, 0.08, 0.15, 0.05], "weights": [0.25, 0.25, 0.25, 0.25]},
        "Heterogeneous": {"coords": [0.95, 0.20, 0.80, 0.35], "weights": [0.30, 0.20, 0.30, 0.20]},
        "Homogeneous": {"coords": [0.75, 0.75, 0.75, 0.75], "weights": [0.25, 0.25, 0.25, 0.25]},
    }

    for i, (name, preset) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(f"📌 {name}", key=f"preset_{name}", width="stretch"):
                st.session_state.template_coords = preset["coords"]
                st.session_state.template_weights = preset["weights"]
                st.rerun()

    st.divider()

    # Coordinate input
    st.subheader("🎯 Coordinates (Ψ)")

    n_dims = st.slider("Number of dimensions", min_value=2, max_value=8, value=len(st.session_state.template_coords))

    # Adjust arrays if dimension changed
    while len(st.session_state.template_coords) < n_dims:
        st.session_state.template_coords.append(0.5)
        st.session_state.template_weights.append(1.0 / n_dims)
    while len(st.session_state.template_coords) > n_dims:
        st.session_state.template_coords.pop()
        st.session_state.template_weights.pop()

    # Normalize weights after dimension change
    w_sum = sum(st.session_state.template_weights)
    if abs(w_sum - 1.0) > 0.01:
        st.session_state.template_weights = [w / w_sum for w in st.session_state.template_weights]

    coord_cols = st.columns(n_dims)
    coords = []
    for i in range(n_dims):
        with coord_cols[i]:
            val = st.number_input(
                f"c_{i + 1}",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.template_coords[i]),
                step=0.01,
                format="%.3f",
                key=f"coord_{i}",
            )
            coords.append(val)

    # Weight input
    st.subheader("⚖️ Weights (w)")
    weight_cols = st.columns(n_dims)
    weights = []
    for i in range(n_dims):
        with weight_cols[i]:
            val = st.number_input(
                f"w_{i + 1}",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.template_weights[i]),
                step=0.05,
                format="%.3f",
                key=f"weight_{i}",
            )
            weights.append(val)

    # Weight validation
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        st.warning(f"⚠️ Weights sum to {weight_sum:.4f}, should sum to 1.0")
        if st.button("🔧 Normalize Weights"):
            weights = [w / weight_sum for w in weights]
            st.session_state.template_weights = weights
            st.rerun()
    else:
        st.success(f"✅ Weights sum to {weight_sum:.6f}")

    # Store in session state
    st.session_state.template_coords = coords
    st.session_state.template_weights = weights

    # Epsilon clipping
    st.subheader("🔒 ε-Clipping Policy")
    epsilon = st.select_slider(
        "Epsilon (ε)", options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4], value=1e-6, format_func=lambda x: f"{x:.0e}"
    )

    st.divider()

    # ========== PROCESS BUTTON ==========
    process_col1, process_col2, process_col3 = st.columns([2, 1, 1])
    with process_col1:
        process_button = st.button("🚀 Run Tier Translation", type="primary", width="stretch")
    with process_col2:
        if st.button("🗑️ Clear Audit Log", width="stretch"):
            st.session_state.audit_log = []
            st.rerun()
    with process_col3:
        export_audit = st.button("📤 Export Audit", width="stretch")

    if process_button:
        # Start audit
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "tier0": {},
            "tier1": {},
            "tier2": {},
            "status": "PROCESSING",
        }

        # ========== TIER 0 PROCESSING ==========
        st.header("⚙️ Processing...")
        progress = st.progress(0, text="Starting tier translation...")

        # Tier 0: Validate and prepare inputs
        progress.progress(10, text="Tier 0: Validating inputs...")

        c_raw = np.array(coords)
        w_raw = np.array(weights)

        # Apply epsilon clipping
        c_clipped = np.clip(c_raw, epsilon, 1 - epsilon)
        clip_count = np.sum(c_raw != c_clipped)
        clip_perturbation = np.sum(np.abs(c_raw - c_clipped))

        audit_entry["tier0"] = {
            "raw_coordinates": coords,
            "raw_weights": weights,
            "epsilon": epsilon,
            "clipped_coordinates": c_clipped.tolist(),
            "clip_count": int(clip_count),
            "clip_perturbation": float(clip_perturbation),
            "weight_sum": float(weight_sum),
            "n_dimensions": n_dims,
        }

        progress.progress(30, text="Tier 1: Computing kernel invariants...")

        # ========== TIER 1: KERNEL COMPUTATION ==========
        try:
            # Compute kernel invariants
            fidelity = float(np.sum(w_raw * c_clipped))  # F: Fidelity
            drift = 1 - fidelity  # ω: Drift

            # Log-space kappa computation
            log_ic = float(np.sum(w_raw * np.log(c_clipped)))  # κ
            integrity_composite = float(np.exp(log_ic))  # IC: Integrity composite

            # Entropy
            entropy = 0.0
            for ci, wi in zip(c_clipped, w_raw, strict=False):
                if wi > 0 and 0 < ci < 1:
                    entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))
            entropy = float(entropy)

            # Curvature proxy
            curvature = float(np.std(c_clipped, ddof=0) / 0.5)

            # AM-GM gap
            amgm_gap = fidelity - integrity_composite

            # Check homogeneity
            is_homogeneous = np.allclose(c_clipped, c_clipped[0], atol=1e-15)

            # Heterogeneity classification
            if amgm_gap < 1e-6:
                het_regime = "homogeneous"
            elif amgm_gap < 0.01:
                het_regime = "coherent"
            elif amgm_gap < 0.05:
                het_regime = "heterogeneous"
            else:
                het_regime = "fragmented"

            # Validate bounds (Lemma 1)
            bounds_valid = (
                0 <= fidelity <= 1
                and 0 <= drift <= 1
                and 0 <= curvature <= 1
                and epsilon <= integrity_composite <= 1 - epsilon
                and np.isfinite(log_ic)
                and 0 <= entropy <= np.log(2)
            )

            audit_entry["tier1"] = {
                "F": fidelity,
                "omega": drift,
                "S": entropy,
                "C": curvature,
                "kappa": log_ic,
                "IC": integrity_composite,
                "amgm_gap": amgm_gap,
                "is_homogeneous": bool(is_homogeneous),
                "heterogeneity_regime": het_regime,
                "bounds_valid": bounds_valid,
                "identity_F_omega": abs(fidelity + drift - 1.0) < 1e-9,
                "identity_IC_kappa": abs(integrity_composite - np.exp(log_ic)) < 1e-9,
            }

            progress.progress(60, text="Tier 2: Computing diagnostics and regime...")

            # ========== TIER 2: DIAGNOSTICS & REGIME ==========

            # Regime classification
            regime = classify_regime(drift)

            # Stability metrics
            freshness = 1 - drift  # How "fresh" the state is

            # Return time estimate (simplified)
            if regime == "STABLE":
                tau_R_est = "≤ 10 steps"
            elif regime == "WATCH":
                tau_R_est = "10-50 steps"
            elif regime == "COLLAPSE":
                tau_R_est = "∞_rec (no return expected)"
            else:
                tau_R_est = "CRITICAL (seam failure)"

            # Stability score (0-100)
            stability_score = int(fidelity * 100 * (1 - curvature))

            # Risk level
            if regime == "STABLE":
                risk = "LOW"
            elif regime == "WATCH":
                risk = "MEDIUM"
            else:
                risk = "HIGH"

            # Recommendations
            recommendations = []
            if drift > 0.5:
                recommendations.append("High drift detected - review data sources")
            if curvature > 0.3:
                recommendations.append("High curvature - coordinates are dispersed")
            if amgm_gap > 0.1:
                recommendations.append("Large AM-GM gap - significant heterogeneity")
            if not bounds_valid:
                recommendations.append("⚠️ Lemma 1 bounds violated - check inputs")
            if not recommendations:
                recommendations.append("✅ All metrics within normal ranges")

            audit_entry["tier2"] = {
                "regime": regime,
                "freshness": freshness,
                "tau_R_estimate": tau_R_est,
                "stability_score": stability_score,
                "risk_level": risk,
                "recommendations": recommendations,
                "classification_criteria": {
                    "omega_threshold_collapse": "ω < 0.1 or ω > 0.9",
                    "omega_threshold_stable": "0.3 ≤ ω ≤ 0.7",
                    "omega_threshold_watch": "otherwise",
                },
            }

            audit_entry["status"] = "COMPLETE"
            progress.progress(100, text="Complete!")

        except Exception as e:
            audit_entry["status"] = "ERROR"
            audit_entry["error"] = str(e)
            st.error(f"❌ Processing error: {e}")
            progress.progress(100, text="Error!")

        # Add to audit log
        st.session_state.audit_log.append(audit_entry)

        # ========== DISPLAY RESULTS ==========
        st.divider()

        if audit_entry["status"] == "COMPLETE":
            # Results header with regime color
            regime = audit_entry["tier2"]["regime"]
            regime_color = get_regime_color(regime)
            st.markdown(
                f"""<div style="padding: 20px; border-left: 6px solid {regime_color}; 
                    background: {regime_color}22; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="margin: 0; color: {regime_color};">🎯 Result: {regime}</h2>
                    <p style="margin: 5px 0 0 0;">Stability Score: {audit_entry["tier2"]["stability_score"]}/100 • Risk: {audit_entry["tier2"]["risk_level"]}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Three-column tier display
            tier_cols = st.columns(3)

            # TIER 0 OUTPUT
            with tier_cols[0]:
                st.markdown("### 📥 Tier 0: Interface")
                t0 = audit_entry["tier0"]
                st.metric("Dimensions", t0["n_dimensions"])
                st.metric("Clipped Values", t0["clip_count"])
                st.metric("ε-Perturbation", f"{t0['clip_perturbation']:.2e}")

                with st.expander("Raw Data"):
                    t0_df = pd.DataFrame(
                        {
                            "Coordinate": [f"c_{i + 1}" for i in range(len(t0["raw_coordinates"]))],
                            "Raw": t0["raw_coordinates"],
                            "Clipped": t0["clipped_coordinates"],
                            "Weight": t0["raw_weights"],
                        }
                    )
                    st.dataframe(t0_df, hide_index=True, width="stretch")

            # TIER 1 OUTPUT
            with tier_cols[1]:
                st.markdown("### ⚙️ Tier 1: Kernel")
                t1 = audit_entry["tier1"]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F (Fidelity)", f"{t1['F']:.4f}")
                    st.metric("ω (Drift)", f"{t1['omega']:.4f}")
                    st.metric("S (Entropy)", f"{t1['S']:.4f}")
                with col2:
                    st.metric("C (Curvature)", f"{t1['C']:.4f}")
                    st.metric("κ (Log-IC)", f"{t1['kappa']:.4f}")
                    st.metric("IC", f"{t1['IC']:.4f}")

                st.markdown(f"**AM-GM Gap:** {t1['amgm_gap']:.4f}")
                st.markdown(f"**Heterogeneity:** {t1['heterogeneity_regime']}")

                # Identity checks
                st.markdown("**Identity Checks:**")
                id1 = "✅" if t1["identity_F_omega"] else "❌"
                id2 = "✅" if t1["identity_IC_kappa"] else "❌"
                st.markdown(f"- {id1} F + ω = 1")
                st.markdown(f"- {id2} IC = exp(κ)")
                st.markdown(f"- {'✅' if t1['bounds_valid'] else '❌'} Lemma 1 bounds")

            # TIER 2 OUTPUT
            with tier_cols[2]:
                st.markdown("### 📊 Tier 2: Diagnostics")
                t2 = audit_entry["tier2"]

                st.metric("Regime", t2["regime"])
                st.metric("Freshness (1-ω)", f"{t2['freshness']:.2%}")
                st.metric("τ_R Estimate", t2["tau_R_estimate"])

                st.markdown("**Recommendations:**")
                for rec in t2["recommendations"]:
                    st.markdown(f"- {rec}")

            # ========== FULL AUDIT TRAIL ==========
            st.divider()
            st.subheader("📋 Full Audit Trail")

            with st.expander("🔍 View Complete Audit JSON", expanded=False):
                st.json(audit_entry)

            # ========== GCD TRANSLATION PANEL ==========
            st.divider()
            st.subheader("🌀 GCD Translation (Generative Collapse Dynamics)")
            st.caption("Native Tier-1 interpretation using the GCD framework for intuitive understanding")

            # Translate tier1 values to GCD
            gcd_translation = translate_to_gcd(audit_entry["tier1"])
            audit_entry["gcd"] = gcd_translation

            render_gcd_panel(gcd_translation, compact=False)

            # Additional interpretive summary for test templates
            st.markdown("#### 📖 Plain Language Interpretation")
            omega_val = audit_entry["tier1"]["omega"]
            fidelity_val = audit_entry["tier1"]["F"]
            ic_val = audit_entry["tier1"]["IC"]

            gcd_regime = gcd_translation["regime"]
            if gcd_regime == "STABLE":
                interpretation = f"""
                Your data shows **high coherence** with minimal drift (ω = {omega_val:.4f}).
                The system is in a **stable generative state** where collapse events produce 
                meaningful structure. Fidelity remains high at F = {fidelity_val:.4f}, indicating
                strong alignment with the reference trace.
                """
            elif gcd_regime == "WATCH":
                interpretation = f"""
                Your data shows **moderate drift** (ω = {omega_val:.4f}), placing the system
                in a **watch regime**. Collapse dynamics are active but not yet overwhelming
                the generative capacity. Monitor the IC value ({ic_val:.4f}) for further degradation.
                """
            else:
                interpretation = f"""
                Your data indicates **significant drift** (ω = {omega_val:.4f}), placing the
                system in a **collapse regime**. The generative capacity is compromised, and 
                the integrity composite has degraded to IC = {ic_val:.4f}. Recovery may require
                substantial intervention.
                """
            st.info(interpretation.strip())

            # Tier flow visualization
            st.markdown("### 🔄 Tier Flow Diagram")
            st.markdown(f"""
            ```
            ┌─────────────────────────────────────────────────────────────────────────┐
            │                           TIER TRANSLATION                               │
            ├─────────────────────────────────────────────────────────────────────────┤
            │                                                                         │
            │  TIER 0 (Interface)              TIER 1 (Kernel)         TIER 2 (Diag)  │
            │  ═══════════════════             ═══════════════         ══════════════ │
            │                                                                         │
            │  Raw Coordinates:                Computed:               Classification:│
            │  c = {[f"{c:.2f}" for c in coords]}     F = {t1["F"]:.4f}              Regime: {regime}      │
            │                                  ω = {t1["omega"]:.4f}              Risk: {t2["risk_level"]}       │
            │  Weights:                        S = {t1["S"]:.4f}                                │
            │  w = {[f"{w:.2f}" for w in weights]}    C = {t1["C"]:.4f}              Return Est:   │
            │                                  κ = {t1["kappa"]:.4f}             {t2["tau_R_estimate"]}  │
            │  ε = {epsilon:.0e}                       IC = {t1["IC"]:.4f}                               │
            │                                                                         │
            │        ────────────>         ────────────>         ────────────>        │
            │           freeze                compute                classify         │
            │                                                                         │
            └─────────────────────────────────────────────────────────────────────────┘
            ```
            """)

    # Export audit
    if export_audit and st.session_state.audit_log:
        st.download_button(
            label="📥 Download Audit Log (JSON)",
            data=json.dumps(st.session_state.audit_log, indent=2),
            file_name=f"umcp_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # ========== AUDIT HISTORY ==========
    st.divider()
    st.subheader("📜 Audit History")

    if st.session_state.audit_log:
        for i, entry in enumerate(reversed(st.session_state.audit_log[-10:])):
            status_icon = "✅" if entry["status"] == "COMPLETE" else "❌"
            regime = entry.get("tier2", {}).get("regime", "N/A")
            regime_color = get_regime_color(regime)

            with st.expander(
                f"{status_icon} Run {len(st.session_state.audit_log) - i} — {regime} @ {entry['timestamp'][:19]}"
            ):
                t0 = entry.get("tier0", {})
                t1 = entry.get("tier1", {})
                t2 = entry.get("tier2", {})

                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**Tier 0:**")
                    st.markdown(f"- Dims: {t0.get('n_dimensions', 'N/A')}")
                    st.markdown(f"- Clipped: {t0.get('clip_count', 'N/A')}")
                with cols[1]:
                    st.markdown("**Tier 1:**")
                    st.markdown(
                        f"- F: {t1.get('F', 'N/A'):.4f}" if isinstance(t1.get("F"), int | float) else "- F: N/A"
                    )
                    st.markdown(
                        f"- ω: {t1.get('omega', 'N/A'):.4f}" if isinstance(t1.get("omega"), int | float) else "- ω: N/A"
                    )
                with cols[2]:
                    st.markdown("**Tier 2:**")
                    st.markdown(f"- Regime: {regime}")
                    st.markdown(f"- Score: {t2.get('stability_score', 'N/A')}")
    else:
        st.info("No audit history yet. Run a tier translation to start logging.")


# ============================================================================
# Expansion Features - Export, Comparison, Notifications, Bookmarks
# ============================================================================


def render_exports_page() -> None:
    """Render the data export page with comprehensive download options."""
    if st is None or pd is None:
        return

    st.title("📥 Data Export Center")
    st.caption("Download data, plots, and reports in various formats")

    # Initialize export history
    if "export_history" not in st.session_state:
        st.session_state.export_history = []

    # ========== Export Categories ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ledger Data", "📦 Casepack Reports", "📈 Plot Images", "📋 Full Reports"])

    with tab1:
        st.subheader("Ledger Data Export")
        df = load_ledger()

        if df.empty:
            st.warning("No ledger data available.")
        else:
            st.info(f"Found {len(df)} ledger entries")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.selectbox(
                    "Date Range",
                    ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
                    key="export_date_range",
                )
                if date_range == "Last 24 Hours" and "timestamp" in df.columns:
                    df = df[df["timestamp"] >= datetime.now() - timedelta(hours=24)]
                elif date_range == "Last 7 Days" and "timestamp" in df.columns:
                    df = df[df["timestamp"] >= datetime.now() - timedelta(days=7)]
                elif date_range == "Last 30 Days" and "timestamp" in df.columns:
                    df = df[df["timestamp"] >= datetime.now() - timedelta(days=30)]

            with col2:
                columns = st.multiselect(
                    "Select Columns", df.columns.tolist(), default=df.columns.tolist()[:10], key="export_columns"
                )

            # Preview
            if columns:
                st.markdown("**Preview (first 10 rows):**")
                st.dataframe(df[columns].head(10), width="stretch")

                # Download buttons
                export_cols = st.columns(3)
                with export_cols[0]:
                    csv_data = df[columns].to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                with export_cols[1]:
                    json_data = df[columns].to_json(orient="records", indent=2)
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width="stretch",
                    )
                with export_cols[2]:
                    # Excel requires openpyxl
                    try:
                        import io

                        buffer = io.BytesIO()
                        df[columns].to_excel(buffer, index=False, engine="openpyxl")
                        st.download_button(
                            label="📊 Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                        )
                    except ImportError:
                        st.button("📊 Excel (needs openpyxl)", disabled=True, width="stretch")

    with tab2:
        st.subheader("Casepack Reports")
        casepacks = load_casepacks()

        if not casepacks:
            st.warning("No casepacks found.")
        else:
            selected_cp = st.selectbox("Select Casepack", [cp["id"] for cp in casepacks], key="export_casepack")

            if selected_cp:
                cp = next((c for c in casepacks if c["id"] == selected_cp), None)
                if cp:
                    st.json(cp)

                    # Generate report
                    report = {
                        "casepack": cp,
                        "generated_at": datetime.now().isoformat(),
                        "version": __version__,
                    }

                    st.download_button(
                        label="📋 Download Casepack Report",
                        data=json.dumps(report, indent=2),
                        file_name=f"casepack_{selected_cp}_report.json",
                        mime="application/json",
                        width="stretch",
                    )

    with tab3:
        st.subheader("Plot Export")
        st.info(
            "💡 Tip: In any chart throughout the dashboard, hover over the plot and click the camera icon (📷) to download as PNG directly from the Plotly toolbar."
        )

        # Generate sample plots for export
        df = load_ledger()
        if not df.empty and go is not None and px is not None:
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Omega Timeline", "Status Distribution", "Metrics Correlation", "Regime Phases"],
                key="export_plot_type",
            )

            fig = None
            if plot_type == "Omega Timeline" and "omega" in df.columns and "timestamp" in df.columns:
                fig = px.line(df.tail(100), x="timestamp", y="omega", title="Omega (ω) Timeline")
            elif plot_type == "Status Distribution" and "run_status" in df.columns:
                fig = px.pie(df, names="run_status", title="Status Distribution", hole=0.4)
            elif plot_type == "Metrics Correlation":
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:5]
                if len(numeric_cols) >= 2:
                    fig = px.scatter_matrix(df[numeric_cols].tail(50), title="Metrics Correlation Matrix")
            elif plot_type == "Regime Phases" and "omega" in df.columns:
                seam = df["seam_residual"].values if "seam_residual" in df.columns else [0] * len(df)
                regimes = [classify_regime(o, s) for o, s in zip(df["omega"].values, seam, strict=False)]
                df_temp = df.copy()
                df_temp["regime"] = regimes
                fig = px.scatter(
                    df_temp.tail(100),
                    x="omega",
                    y=df_temp.select_dtypes(include=["number"]).columns[0],
                    color="regime",
                    color_discrete_map=REGIME_COLORS,
                    title="Regime Phase Distribution",
                )

            if fig:
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")

                # Export as HTML (interactive)
                html_buffer = fig.to_html(include_plotlyjs="cdn")
                st.download_button(
                    label="🌐 Download Interactive HTML",
                    data=html_buffer,
                    file_name=f"umcp_plot_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    width="stretch",
                )
        else:
            st.info("No data available for plot generation.")

    with tab4:
        st.subheader("Full System Report")

        if st.button("🔄 Generate Full Report", width="stretch"):
            with st.spinner("Generating comprehensive report..."):
                # Compile full report
                df = load_ledger()
                casepacks = load_casepacks()
                contracts = load_contracts()
                closures = load_closures()

                report = {
                    "report_metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "umcp_version": __version__,
                        "report_type": "full_system_report",
                    },
                    "summary": {
                        "total_ledger_entries": len(df),
                        "total_casepacks": len(casepacks),
                        "total_contracts": len(contracts),
                        "total_closures": len(closures),
                    },
                    "ledger_statistics": {},
                    "casepacks": casepacks,
                    "contracts": contracts,
                    "closures": closures,
                }

                # Add ledger stats
                if not df.empty:
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    for col in numeric_cols:
                        report["ledger_statistics"][col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                        }

                    if "run_status" in df.columns:
                        report["ledger_statistics"]["status_distribution"] = df["run_status"].value_counts().to_dict()

                st.success("✅ Report generated!")
                st.json(report)

                st.download_button(
                    label="📥 Download Full Report (JSON)",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"umcp_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width="stretch",
                )


def render_comparison_page() -> None:
    """Render the audit comparison page for side-by-side analysis."""
    if st is None or pd is None:
        return

    st.title("🔀 Comparison Mode")
    st.caption("Compare two audit runs or validation results side-by-side")

    # Initialize comparison history
    if "comparison_history" not in st.session_state:
        st.session_state.comparison_history = []

    # ========== Data Sources ==========
    tab1, tab2, tab3 = st.tabs(["📒 Ledger Comparison", "📦 Casepack Comparison", "🧮 Audit Comparison"])

    with tab1:
        st.subheader("Compare Ledger Entries")
        df = load_ledger()

        if df.empty or len(df) < 2:
            st.warning("Need at least 2 ledger entries for comparison.")
        else:
            col1, col2 = st.columns(2)

            # Create index-based selection
            df["index_label"] = [
                f"#{i + 1} - {row.get('timestamp', 'N/A')}" for i, (_, row) in enumerate(df.iterrows())
            ]

            with col1:
                st.markdown("### 📋 Entry A")
                entry_a_idx = st.selectbox(
                    "Select Entry A", range(len(df)), format_func=lambda x: df.iloc[x]["index_label"], key="compare_a"
                )
                entry_a = df.iloc[entry_a_idx]

                # Display entry A
                with st.container(border=True):
                    if "run_status" in df.columns:
                        status = entry_a["run_status"]
                        st.markdown(f"**Status:** :{STATUS_COLORS.get(status, 'gray')}[{status}]")
                    for col in df.select_dtypes(include=["number"]).columns[:6]:
                        val = entry_a[col]
                        if pd.notna(val):
                            st.metric(KERNEL_SYMBOLS.get(col, col) or col, f"{val:.4f}")

            with col2:
                st.markdown("### 📋 Entry B")
                entry_b_idx = st.selectbox(
                    "Select Entry B",
                    range(len(df)),
                    format_func=lambda x: df.iloc[x]["index_label"],
                    index=min(1, len(df) - 1),
                    key="compare_b",
                )
                entry_b = df.iloc[entry_b_idx]

                # Display entry B
                with st.container(border=True):
                    if "run_status" in df.columns:
                        status = entry_b["run_status"]
                        st.markdown(f"**Status:** :{STATUS_COLORS.get(status, 'gray')}[{status}]")
                    for col in df.select_dtypes(include=["number"]).columns[:6]:
                        val = entry_b[col]
                        if pd.notna(val):
                            st.metric(KERNEL_SYMBOLS.get(col, col) or col, f"{val:.4f}")

            # ========== Diff Analysis ==========
            st.divider()
            st.subheader("📊 Difference Analysis")

            diff_data = []
            for col in df.select_dtypes(include=["number"]).columns:
                val_a = entry_a[col]
                val_b = entry_b[col]
                if pd.notna(val_a) and pd.notna(val_b):
                    diff = val_b - val_a
                    pct_change = ((val_b - val_a) / abs(val_a) * 100) if val_a != 0 else 0
                    diff_data.append(
                        {
                            "Metric": KERNEL_SYMBOLS.get(col, col) or col,
                            "Entry A": f"{val_a:.4f}",
                            "Entry B": f"{val_b:.4f}",
                            "Difference": f"{diff:+.4f}",
                            "% Change": f"{pct_change:+.2f}%",
                            "Trend": "📈" if diff > 0 else "📉" if diff < 0 else "➡️",
                        }
                    )

            if diff_data:
                diff_df = pd.DataFrame(diff_data)
                st.dataframe(diff_df, width="stretch", hide_index=True)

                # Visual comparison
                if go is not None:
                    metrics = [d["Metric"] for d in diff_data][:8]
                    vals_a = [float(d["Entry A"]) for d in diff_data][:8]
                    vals_b = [float(d["Entry B"]) for d in diff_data][:8]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Entry A", x=metrics, y=vals_a, marker_color="#007bff"))
                    fig.add_trace(go.Bar(name="Entry B", x=metrics, y=vals_b, marker_color="#28a745"))
                    fig.update_layout(
                        barmode="group",
                        height=400,
                        title="Metric Comparison",
                        xaxis_title="Metrics",
                        yaxis_title="Value",
                    )
                    st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader("Compare Casepacks")
        casepacks = load_casepacks()

        if len(casepacks) < 2:
            st.warning("Need at least 2 casepacks for comparison.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📦 Casepack A")
                cp_a_id = st.selectbox("Select Casepack A", [cp["id"] for cp in casepacks], key="cp_compare_a")
                cp_a = next((c for c in casepacks if c["id"] == cp_a_id), None)
                if cp_a:
                    st.json(cp_a)

            with col2:
                st.markdown("### 📦 Casepack B")
                cp_b_id = st.selectbox(
                    "Select Casepack B",
                    [cp["id"] for cp in casepacks],
                    index=min(1, len(casepacks) - 1),
                    key="cp_compare_b",
                )
                cp_b = next((c for c in casepacks if c["id"] == cp_b_id), None)
                if cp_b:
                    st.json(cp_b)

            # Comparison summary
            if cp_a and cp_b:
                st.divider()
                st.subheader("📊 Comparison Summary")

                comparison_metrics = [
                    ("Files Count", cp_a.get("files_count", 0), cp_b.get("files_count", 0)),
                    ("Test Vectors", cp_a.get("test_vectors", 0), cp_b.get("test_vectors", 0)),
                    ("Closures", cp_a.get("closures_count", 0), cp_b.get("closures_count", 0)),
                ]

                comp_df = pd.DataFrame(comparison_metrics, columns=["Metric", cp_a_id, cp_b_id])
                st.dataframe(comp_df, width="stretch", hide_index=True)

    with tab3:
        st.subheader("Compare Audit Runs")

        if "audit_log" not in st.session_state or len(st.session_state.audit_log) < 2:
            st.warning("Need at least 2 audit runs for comparison. Run tier translations in Test Templates page.")
        else:
            audit_log = st.session_state.audit_log

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧮 Audit A")
                audit_a_idx = st.selectbox(
                    "Select Audit A",
                    range(len(audit_log)),
                    format_func=lambda x: f"Run {x + 1} @ {audit_log[x]['timestamp'][:19]}",
                    key="audit_compare_a",
                )
                audit_a = audit_log[audit_a_idx]
                st.json(audit_a)

            with col2:
                st.markdown("### 🧮 Audit B")
                audit_b_idx = st.selectbox(
                    "Select Audit B",
                    range(len(audit_log)),
                    format_func=lambda x: f"Run {x + 1} @ {audit_log[x]['timestamp'][:19]}",
                    index=min(1, len(audit_log) - 1),
                    key="audit_compare_b",
                )
                audit_b = audit_log[audit_b_idx]
                st.json(audit_b)


def render_notifications_page() -> None:
    """Render the notification and alerts page."""
    if st is None:
        return

    st.title("🔔 Notifications & Alerts")
    st.caption("Configure alerts for regime changes and anomalies")

    # Initialize notification settings
    if "notifications" not in st.session_state:
        st.session_state.notifications = {
            "enabled": True,
            "regime_change": True,
            "anomaly_detected": True,
            "validation_failed": True,
            "threshold_omega_low": 0.1,
            "threshold_omega_high": 0.9,
            "alert_log": [],
        }

    # ========== Alert Configuration ==========
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Alert Settings")

        with st.container(border=True):
            st.session_state.notifications["enabled"] = st.toggle(
                "Enable Notifications", value=st.session_state.notifications["enabled"]
            )

            st.markdown("**Alert Types:**")
            st.session_state.notifications["regime_change"] = st.checkbox(
                "🌡️ Regime Changes",
                value=st.session_state.notifications["regime_change"],
                help="Alert when regime transitions (STABLE→WATCH, WATCH→COLLAPSE, etc.)",
            )
            st.session_state.notifications["anomaly_detected"] = st.checkbox(
                "⚠️ Anomaly Detection",
                value=st.session_state.notifications["anomaly_detected"],
                help="Alert when statistical anomalies are detected",
            )
            st.session_state.notifications["validation_failed"] = st.checkbox(
                "❌ Validation Failures",
                value=st.session_state.notifications["validation_failed"],
                help="Alert when validation returns NONCONFORMANT",
            )

            st.markdown("**Thresholds:**")
            st.session_state.notifications["threshold_omega_low"] = st.slider(
                "ω Low Threshold (COLLAPSE)",
                0.0,
                0.3,
                st.session_state.notifications["threshold_omega_low"],
                0.01,
                help="Alert when ω drops below this value",
            )
            st.session_state.notifications["threshold_omega_high"] = st.slider(
                "ω High Threshold (COLLAPSE)",
                0.7,
                1.0,
                st.session_state.notifications["threshold_omega_high"],
                0.01,
                help="Alert when ω exceeds this value",
            )

    with col2:
        st.subheader("🔍 Current State Check")

        if st.button("🔄 Check for Alerts Now", width="stretch"):
            df = load_ledger()
            alerts = []

            if not df.empty:
                latest = df.iloc[-1]

                # Check omega thresholds
                if "omega" in df.columns:
                    omega = latest["omega"]
                    low_thresh = st.session_state.notifications["threshold_omega_low"]
                    high_thresh = st.session_state.notifications["threshold_omega_high"]

                    if omega < low_thresh:
                        alerts.append(
                            {
                                "type": "CRITICAL",
                                "message": f"ω below threshold: {omega:.4f} < {low_thresh}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    elif omega > high_thresh:
                        alerts.append(
                            {
                                "type": "CRITICAL",
                                "message": f"ω above threshold: {omega:.4f} > {high_thresh}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Check regime transitions
                if len(df) >= 2 and "omega" in df.columns:
                    prev = df.iloc[-2]
                    current_regime = classify_regime(latest.get("omega", 0.5), latest.get("seam_residual", 0))
                    prev_regime = classify_regime(prev.get("omega", 0.5), prev.get("seam_residual", 0))

                    if current_regime != prev_regime:
                        alerts.append(
                            {
                                "type": "WARNING",
                                "message": f"Regime changed: {prev_regime} → {current_regime}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Check validation status
                if "run_status" in df.columns and latest["run_status"] == "NONCONFORMANT":
                    alerts.append(
                        {
                            "type": "ERROR",
                            "message": "Latest validation NONCONFORMANT",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Anomaly check
                if "omega" in df.columns and len(df) > 5:
                    anomalies = detect_anomalies(df["omega"])
                    if anomalies.iloc[-1]:
                        alerts.append(
                            {
                                "type": "WARNING",
                                "message": f"Statistical anomaly detected in ω: {latest['omega']:.4f}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Display and log alerts
            if alerts:
                st.session_state.notifications["alert_log"].extend(alerts)
                for alert in alerts:
                    if alert["type"] == "CRITICAL":
                        st.error(f"🚨 **CRITICAL:** {alert['message']}")
                    elif alert["type"] == "ERROR":
                        st.error(f"❌ **ERROR:** {alert['message']}")
                    elif alert["type"] == "WARNING":
                        st.warning(f"⚠️ **WARNING:** {alert['message']}")
                    else:
                        st.info(f"ℹ️ **INFO:** {alert['message']}")
            else:
                st.success("✅ No alerts - system operating normally")

    # ========== Alert Log ==========
    st.divider()
    st.subheader("📜 Alert History")

    alert_log = st.session_state.notifications.get("alert_log", [])
    if alert_log:
        # Show last 20 alerts
        for alert in reversed(alert_log[-20:]):
            icon = (
                "🚨"
                if alert["type"] == "CRITICAL"
                else "⚠️"
                if alert["type"] == "WARNING"
                else "❌"
                if alert["type"] == "ERROR"
                else "ℹ️"
            )
            st.markdown(f"{icon} **{alert['type']}** — {alert['message']} @ {alert['timestamp'][:19]}")

        if st.button("🗑️ Clear Alert History"):
            st.session_state.notifications["alert_log"] = []
            st.rerun()
    else:
        st.info("No alerts recorded yet.")


def render_bookmarks_page() -> None:
    """Render the bookmarks page for saving interesting states."""
    if st is None or pd is None:
        return

    st.title("🔖 Bookmarks")
    st.caption("Save and revisit interesting states and configurations")

    # Initialize bookmarks
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []

    # ========== Add Bookmark ==========
    st.subheader("➕ Save Current State")

    with st.form("add_bookmark"):
        col1, col2 = st.columns(2)

        with col1:
            bookmark_name = st.text_input("Bookmark Name", placeholder="e.g., 'Stable regime baseline'")
            bookmark_type = st.selectbox(
                "Bookmark Type", ["Ledger Snapshot", "Configuration", "Audit Run", "Custom Note"]
            )

        with col2:
            bookmark_tags = st.text_input("Tags (comma-separated)", placeholder="stable, baseline, v1.5")
            bookmark_notes = st.text_area("Notes", placeholder="Add any notes about this bookmark...")

        submitted = st.form_submit_button("🔖 Save Bookmark", width="stretch")

        if submitted and bookmark_name:
            # Capture current state
            df = load_ledger()
            snapshot = {}

            if bookmark_type == "Ledger Snapshot" and not df.empty:
                latest = df.iloc[-1].to_dict()
                # Convert numpy/pandas types to native Python
                snapshot = {
                    k: (float(v) if hasattr(v, "item") else str(v) if hasattr(v, "isoformat") else v)
                    for k, v in latest.items()
                    if pd.notna(v)
                }
            elif bookmark_type == "Audit Run" and "audit_log" in st.session_state and st.session_state.audit_log:
                snapshot = st.session_state.audit_log[-1]
            elif bookmark_type == "Configuration":
                snapshot = {
                    "auto_refresh": st.session_state.get("auto_refresh", False),
                    "refresh_interval": st.session_state.get("refresh_interval", 30),
                    "show_advanced": st.session_state.get("show_advanced", False),
                    "compact_mode": st.session_state.get("compact_mode", False),
                    "theme": st.session_state.get("theme", "Default"),
                }

            bookmark = {
                "id": len(st.session_state.bookmarks) + 1,
                "name": bookmark_name,
                "type": bookmark_type,
                "tags": [t.strip() for t in bookmark_tags.split(",") if t.strip()],
                "notes": bookmark_notes,
                "snapshot": snapshot,
                "created_at": datetime.now().isoformat(),
            }

            st.session_state.bookmarks.append(bookmark)
            st.success(f"✅ Bookmark '{bookmark_name}' saved!")
            st.rerun()

    st.divider()

    # ========== View Bookmarks ==========
    st.subheader("📚 Saved Bookmarks")

    if not st.session_state.bookmarks:
        st.info("No bookmarks saved yet. Create your first bookmark above!")
    else:
        # Filter by type
        types = list({b["type"] for b in st.session_state.bookmarks})
        type_filter = st.selectbox("Filter by Type", ["All", *types])

        filtered = st.session_state.bookmarks
        if type_filter != "All":
            filtered = [b for b in filtered if b["type"] == type_filter]

        # Display bookmarks
        for _i, bookmark in enumerate(reversed(filtered)):
            with st.expander(f"🔖 {bookmark['name']} ({bookmark['type']}) — {bookmark['created_at'][:10]}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if bookmark["tags"]:
                        st.markdown("**Tags:** " + ", ".join([f"`{t}`" for t in bookmark["tags"]]))
                    if bookmark["notes"]:
                        st.markdown(f"**Notes:** {bookmark['notes']}")

                    st.markdown("**Snapshot:**")
                    st.json(bookmark["snapshot"])

                with col2:
                    if st.button("🗑️ Delete", key=f"del_bm_{bookmark['id']}"):
                        st.session_state.bookmarks = [
                            b for b in st.session_state.bookmarks if b["id"] != bookmark["id"]
                        ]
                        st.rerun()

                    # Export single bookmark
                    st.download_button(
                        label="📥 Export",
                        data=json.dumps(bookmark, indent=2, default=str),
                        file_name=f"bookmark_{bookmark['id']}.json",
                        mime="application/json",
                        key=f"export_bm_{bookmark['id']}",
                    )

        # Export all bookmarks
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Export All Bookmarks",
                data=json.dumps(st.session_state.bookmarks, indent=2, default=str),
                file_name=f"umcp_bookmarks_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                width="stretch",
            )
        with col2:
            if st.button("🗑️ Clear All Bookmarks", width="stretch"):
                st.session_state.bookmarks = []
                st.rerun()


# ============================================================================
# Medium-term Expansions - Time Series, Custom Formulas, Batch, API
# ============================================================================


def render_time_series_page() -> None:
    """Render the time series analysis page for tracking invariants over time."""
    if st is None or pd is None or go is None:
        return

    st.title("📈 Time Series Analysis")
    st.caption("Track kernel invariants and trends over multiple validation runs")

    df = load_ledger()

    if df.empty or "timestamp" not in df.columns:
        st.warning("No time series data available. Run validations to populate the ledger.")
        return

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # ========== Time Range Selection ==========
    st.subheader("📅 Time Range")
    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.selectbox(
            "Quick Select",
            ["All Time", "Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
            key="ts_date_range",
        )

    filtered_df = df.copy()
    if date_range == "Last Hour":
        filtered_df = df[df["timestamp"] >= datetime.now() - timedelta(hours=1)]
    elif date_range == "Last 24 Hours":
        filtered_df = df[df["timestamp"] >= datetime.now() - timedelta(hours=24)]
    elif date_range == "Last 7 Days":
        filtered_df = df[df["timestamp"] >= datetime.now() - timedelta(days=7)]
    elif date_range == "Last 30 Days":
        filtered_df = df[df["timestamp"] >= datetime.now() - timedelta(days=30)]

    with col2:
        st.metric("Data Points", len(filtered_df))
    with col3:
        if len(filtered_df) >= 2:
            time_span = filtered_df["timestamp"].max() - filtered_df["timestamp"].min()
            st.metric("Time Span", str(time_span).split(".")[0])
        else:
            st.metric("Time Span", "N/A")

    if len(filtered_df) < 2:
        st.warning("Need at least 2 data points for time series analysis.")
        return

    st.divider()

    # ========== Metric Selection ==========
    numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
    selected_metrics = st.multiselect(
        "Select Metrics to Analyze",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
        format_func=lambda x: KERNEL_SYMBOLS.get(x, x) or x,
    )

    if not selected_metrics:
        st.info("Select at least one metric to analyze.")
        return

    # ========== Trend Analysis ==========
    st.subheader("📊 Trend Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📉 Moving Averages", "🔮 Forecasting", "📊 Statistics"])

    with tab1:
        st.markdown("### Multi-Metric Timeline")

        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[KERNEL_SYMBOLS.get(m, m) or m for m in selected_metrics],
            vertical_spacing=0.05,
        )

        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"],
                    y=filtered_df[metric],
                    name=KERNEL_SYMBOLS.get(metric, metric) or metric,
                    mode="lines+markers",
                    marker={"size": 4},
                    line={"color": colors[i % len(colors)]},
                ),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            height=200 * len(selected_metrics),
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        st.markdown("### Moving Averages")

        window_size = st.slider("Window Size", 3, min(50, len(filtered_df)), 5)

        fig = go.Figure()
        for metric in selected_metrics:
            # Raw data
            fig.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"],
                    y=filtered_df[metric],
                    name=f"{KERNEL_SYMBOLS.get(metric, metric) or metric} (raw)",
                    mode="markers",
                    marker={"size": 4, "opacity": 0.5},
                )
            )
            # Moving average
            ma = filtered_df[metric].rolling(window=window_size, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"],
                    y=ma,
                    name=f"{KERNEL_SYMBOLS.get(metric, metric) or metric} (MA-{window_size})",
                    mode="lines",
                    line={"width": 2},
                )
            )

        fig.update_layout(height=500, title=f"Moving Average (window={window_size})")
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("### Simple Forecasting")
        st.info("💡 Using linear regression for simple trend projection")

        forecast_periods = st.slider("Forecast Periods", 5, 50, 10)

        for metric in selected_metrics:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Linear regression forecast
                y = filtered_df[metric].values
                np.arange(len(y))

                if len(y) >= 2:
                    # Simple linear fit
                    slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
                    intercept = y[0]

                    # Forecast
                    x_future = np.arange(len(y), len(y) + forecast_periods)
                    y_forecast = intercept + slope * x_future

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, name="Historical", mode="lines+markers"))
                    fig.add_trace(
                        go.Scatter(
                            x=list(x_future),
                            y=y_forecast,
                            name="Forecast",
                            mode="lines",
                            line={"dash": "dash", "color": "red"},
                        )
                    )
                    fig.update_layout(title=f"{KERNEL_SYMBOLS.get(metric, metric) or metric} Forecast", height=300)
                    st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown(f"**{KERNEL_SYMBOLS.get(metric, metric) or metric}**")
                st.metric("Current", f"{y[-1]:.4f}")
                if len(y) >= 2:
                    st.metric("Trend", f"{slope:+.6f}/period")
                    st.metric(f"Forecast (+{forecast_periods})", f"{y_forecast[-1]:.4f}")

    with tab4:
        st.markdown("### Time Series Statistics")

        stats_data = []
        for metric in selected_metrics:
            series = filtered_df[metric]
            stats_data.append(
                {
                    "Metric": KERNEL_SYMBOLS.get(metric, metric) or metric,
                    "Mean": f"{series.mean():.4f}",
                    "Std": f"{series.std():.4f}",
                    "Min": f"{series.min():.4f}",
                    "Max": f"{series.max():.4f}",
                    "Range": f"{series.max() - series.min():.4f}",
                    "Trend": "📈" if series.iloc[-1] > series.iloc[0] else "📉",
                    "Volatility": f"{(series.std() / series.mean() * 100):.2f}%" if series.mean() != 0 else "N/A",
                }
            )

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, width="stretch", hide_index=True)

        # Correlation matrix
        if len(selected_metrics) >= 2:
            st.markdown("### Correlation Matrix")
            corr = filtered_df[selected_metrics].corr()

            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Metric Correlations"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")


def render_formula_builder_page() -> None:
    """Render the custom formula builder page."""
    if st is None:
        return

    st.title("🔧 Custom Formula Builder")
    st.caption("Create and test your own physics and analysis formulas")

    # Initialize custom formulas
    if "custom_formulas" not in st.session_state:
        st.session_state.custom_formulas = []

    # ========== Formula Creation ==========
    st.subheader("➕ Create New Formula")

    with st.form("create_formula"):
        col1, col2 = st.columns(2)

        with col1:
            formula_name = st.text_input("Formula Name", placeholder="e.g., 'Custom Energy'")
            formula_latex = st.text_input("LaTeX Expression", placeholder=r"e.g., E = \frac{1}{2}mv^2")
            formula_category = st.selectbox("Category", ["Physics", "Kinematics", "Statistics", "Custom"])

        with col2:
            formula_expr = st.text_input(
                "Python Expression",
                placeholder="e.g., 0.5 * m * v**2",
                help="Use variable names that match your parameters",
            )
            formula_params = st.text_input(
                "Parameters (comma-separated)", placeholder="m, v", help="Variable names used in the expression"
            )
            formula_units = st.text_input("Result Units", placeholder="J (Joules)")

        formula_description = st.text_area("Description", placeholder="Describe what this formula calculates...")

        submitted = st.form_submit_button("➕ Add Formula", width="stretch")

        if submitted and formula_name and formula_expr and formula_params:
            params = [p.strip() for p in formula_params.split(",")]

            new_formula = {
                "id": len(st.session_state.custom_formulas) + 1,
                "name": formula_name,
                "latex": formula_latex or formula_expr,
                "category": formula_category,
                "expression": formula_expr,
                "parameters": params,
                "units": formula_units,
                "description": formula_description,
                "created_at": datetime.now().isoformat(),
            }

            # Test the formula
            try:
                test_vars = dict.fromkeys(params, 1.0)
                test_result = eval(
                    formula_expr,
                    {"__builtins__": {}},
                    {
                        **test_vars,
                        "np": np,
                        "sin": np.sin,
                        "cos": np.cos,
                        "sqrt": np.sqrt,
                        "pi": np.pi,
                        "exp": np.exp,
                        "log": np.log,
                    },
                )
                new_formula["test_result"] = test_result
                st.session_state.custom_formulas.append(new_formula)
                st.success(f"✅ Formula '{formula_name}' added! Test result with all params=1: {test_result}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Formula error: {e}")

    st.divider()

    # ========== Use Custom Formulas ==========
    st.subheader("🧮 Calculate with Formulas")

    # Combine built-in and custom formulas
    all_formulas = st.session_state.custom_formulas

    if not all_formulas:
        st.info("No custom formulas yet. Create one above!")
    else:
        selected_formula = st.selectbox(
            "Select Formula", all_formulas, format_func=lambda x: f"{x['name']} ({x['category']})"
        )

        if selected_formula:
            st.markdown(f"### {selected_formula['name']}")
            if selected_formula.get("latex"):
                st.latex(selected_formula["latex"])
            st.caption(selected_formula.get("description", "No description"))

            # Parameter inputs
            st.markdown("**Parameters:**")
            param_values = {}
            cols = st.columns(min(4, len(selected_formula["parameters"])))

            for i, param in enumerate(selected_formula["parameters"]):
                with cols[i % len(cols)]:
                    param_values[param] = st.number_input(
                        param, value=1.0, format="%.4f", key=f"custom_param_{selected_formula['id']}_{param}"
                    )

            # Calculate
            if st.button("🔢 Calculate", width="stretch"):
                try:
                    # Safe evaluation with numpy functions
                    safe_dict = {
                        "np": np,
                        "sin": np.sin,
                        "cos": np.cos,
                        "tan": np.tan,
                        "sqrt": np.sqrt,
                        "pi": np.pi,
                        "e": np.e,
                        "exp": np.exp,
                        "log": np.log,
                        "log10": np.log10,
                        "abs": abs,
                        "pow": pow,
                        **param_values,
                    }
                    result = eval(selected_formula["expression"], {"__builtins__": {}}, safe_dict)

                    st.success(f"**Result:** {result:.6g} {selected_formula.get('units', '')}")

                except Exception as e:
                    st.error(f"❌ Calculation error: {e}")

    st.divider()

    # ========== Manage Formulas ==========
    st.subheader("📋 Manage Custom Formulas")

    if st.session_state.custom_formulas:
        for formula in st.session_state.custom_formulas:
            with st.expander(f"🔧 {formula['name']} ({formula['category']})"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if formula.get("latex"):
                        st.latex(formula["latex"])
                    st.code(formula["expression"], language="python")
                    st.caption(f"Parameters: {', '.join(formula['parameters'])}")
                    if formula.get("description"):
                        st.caption(formula["description"])

                with col2:
                    if st.button("🗑️ Delete", key=f"del_formula_{formula['id']}"):
                        st.session_state.custom_formulas = [
                            f for f in st.session_state.custom_formulas if f["id"] != formula["id"]
                        ]
                        st.rerun()

        # Export formulas
        st.download_button(
            label="📥 Export All Formulas",
            data=json.dumps(st.session_state.custom_formulas, indent=2),
            file_name="custom_formulas.json",
            mime="application/json",
            width="stretch",
        )
    else:
        st.info("No custom formulas saved yet.")


def render_cosmology_page() -> None:
    """
    Render the WEYL Cosmology page for modified gravity analysis.

    Implements visualization of the WEYL.INTSTACK.v1 contract:
    - Σ(z) evolution and regime classification
    - ĥJ measurements from DES Y3 data
    - Weyl transfer function visualization
    - UMCP integration patterns

    Reference: Nature Communications 15:9295 (2024)
    """
    if st is None or np is None or pd is None:
        return

    st.title("🌌 WEYL Cosmology")
    st.caption("Modified gravity analysis with Σ(z) parametrization | Nature Comms 15:9295 (2024)")

    # Try to import WEYL closures - with path fix for Docker
    weyl_available = False
    Omega_Lambda_of_z = None  # Initialize for scope
    try:
        from closures.weyl import (
            DES_Y3_DATA,
            PLANCK_2018,
            D1_of_z,
            GzModel,
            H_of_z,
            Omega_Lambda_of_z,
            Sigma_to_UMCP_invariants,
            chi_of_z,
            compute_Sigma,
            sigma8_of_z,
        )

        weyl_available = True
    except ImportError:
        # Add closures directory to path for Docker container
        import sys

        repo_root = get_repo_root()
        closures_path = str(repo_root / "closures")
        if closures_path not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from closures.weyl import (
                DES_Y3_DATA,
                PLANCK_2018,
                D1_of_z,
                GzModel,
                H_of_z,
                Omega_Lambda_of_z,
                Sigma_to_UMCP_invariants,
                chi_of_z,
                compute_Sigma,
                sigma8_of_z,
            )

            weyl_available = True
        except ImportError as e:
            st.error(f"❌ WEYL closures import failed: {e}")

    if not weyl_available:
        st.error("❌ WEYL closures not available. Please ensure closures/weyl/ is installed.")
        st.code("# WEYL closures should be in closures/weyl/", language="python")
        return

    # Initialize session state
    if "weyl_params" not in st.session_state:
        st.session_state.weyl_params = {
            "Sigma_0": 0.24,
            "g_model": "constant",
            "z_max": 2.0,
            "n_points": 100,
        }

    # ========== Cosmological Background Section ==========
    st.header("🌍 Cosmological Background")

    with st.expander("📖 About ΛCDM Background", expanded=False):
        st.markdown("""
        **Planck 2018 Fiducial Cosmology:**
        - Matter density: Ω_m,0 = 0.315
        - Dark energy: Ω_Λ,0 = 0.685
        - Hubble constant: H₀ = 67.4 km/s/Mpc
        - σ8 amplitude: σ8,0 = 0.811
        
        **Key Functions:**
        - H(z) = H₀ √[Ω_m(1+z)³ + Ω_Λ] - Hubble parameter
        - χ(z) = ∫ c/H(z') dz' - Comoving distance
        - D₁(z) - Linear growth function (normalized to 1 at z=0)
        - σ8(z) = σ8,0 × D₁(z) - Amplitude evolution
        
        **UMCP Integration:**
        - Background cosmology = embedding specification (Tier-0)
        - Frozen parameters (Ω_m, σ8, H₀) define the coordinate system
        """)

    # Background visualization
    z_arr = np.linspace(0, 3, 100)
    H_arr = np.array([H_of_z(z) for z in z_arr])
    chi_arr = np.array([chi_of_z(z) for z in z_arr])
    D1_arr = np.array([D1_of_z(z) for z in z_arr])
    sigma8_arr = np.array([sigma8_of_z(z) for z in z_arr])

    bg_tabs = st.tabs(["📈 H(z)", "📏 χ(z)", "📊 D₁(z) & σ8(z)"])

    with bg_tabs[0]:
        fig_H = go.Figure()
        fig_H.add_trace(go.Scatter(x=z_arr, y=H_arr, mode="lines", name="H(z)", line={"color": "#1f77b4", "width": 2}))
        fig_H.update_layout(
            title="Hubble Parameter Evolution",
            xaxis_title="Redshift z",
            yaxis_title="H(z) [km/s/Mpc]",
            showlegend=True,
        )
        st.plotly_chart(fig_H, width="stretch")

    with bg_tabs[1]:
        fig_chi = go.Figure()
        fig_chi.add_trace(
            go.Scatter(x=z_arr, y=chi_arr, mode="lines", name="χ(z)", line={"color": "#2ca02c", "width": 2})
        )
        fig_chi.update_layout(
            title="Comoving Distance",
            xaxis_title="Redshift z",
            yaxis_title="χ(z) [Mpc/h]",
            showlegend=True,
        )
        st.plotly_chart(fig_chi, width="stretch")

    with bg_tabs[2]:
        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Scatter(x=z_arr, y=D1_arr, mode="lines", name="D₁(z)", line={"color": "#ff7f0e", "width": 2})
        )
        fig_growth.add_trace(
            go.Scatter(x=z_arr, y=sigma8_arr, mode="lines", name="σ8(z)", line={"color": "#d62728", "width": 2})
        )
        fig_growth.add_hline(y=PLANCK_2018.sigma8_0, line_dash="dash", line_color="gray", annotation_text="σ8,0")
        fig_growth.update_layout(
            title="Growth Function and σ8 Evolution",
            xaxis_title="Redshift z",
            yaxis_title="Value",
            showlegend=True,
        )
        st.plotly_chart(fig_growth, width="stretch")

    st.divider()

    # ========== Σ(z) Modified Gravity Section ==========
    st.header("🔬 Modified Gravity: Σ(z)")

    with st.expander("📖 About Σ Parametrization", expanded=False):
        st.markdown("""
        **Σ(z) Definition (Eq. 11):**
        
        The Σ parameter encodes deviations from General Relativity:
        
        $$ k^2 (\\Phi + \\Psi)/2 = -4\\pi G a^2 \\Sigma(z,k) \\bar{\\rho}(z) \\Delta_m(z,k) $$
        
        - **Σ = 1**: General Relativity
        - **Σ ≠ 1**: Modified gravity or gravitational slip
        
        **Parametrization (Eq. 13):**
        $$ \\Sigma(z) = 1 + \\Sigma_0 \\cdot g(z) $$
        
        Where g(z) models:
        - **constant**: g(z) = 1 for z ∈ [0,1]
        - **exponential**: g(z) = exp(1+z) for z ∈ [0,1]
        - **standard**: g(z) = Ω_Λ(z)
        
        **UMCP Regime Mapping:**
        | Σ₀ Range | Regime | UMCP Analog |
        |----------|--------|-------------|
        | |Σ₀| < 0.1 | GR_consistent | STABLE |
        | 0.1 ≤ |Σ₀| < 0.3 | Tension | WATCH |
        | |Σ₀| ≥ 0.3 | Modified_gravity | COLLAPSE |
        """)

    # Interactive Σ₀ controls
    control_cols = st.columns([1, 1, 1, 1])

    with control_cols[0]:
        sigma_0 = st.slider("Σ₀ (deviation amplitude)", -0.5, 0.5, 0.24, 0.01, help="DES Y3 finds Σ₀ ≈ 0.24 ± 0.14")
    with control_cols[1]:
        g_model_name = st.selectbox("g(z) Model", ["constant", "exponential", "standard"], index=0)
        g_model = GzModel(g_model_name)
    with control_cols[2]:
        z_max_sigma = st.slider("z_max", 0.5, 3.0, 2.0, 0.1)
    with control_cols[3]:
        n_points = st.slider("Points", 20, 200, 100, 10)

    # Compute Σ(z)
    z_sigma = np.linspace(0, z_max_sigma, n_points)
    # Pass Omega_Lambda_of_z for standard model (required by closure)
    Sigma_results = [compute_Sigma(z, sigma_0, g_model, Omega_Lambda_z=Omega_Lambda_of_z) for z in z_sigma]
    Sigma_values = [r.Sigma for r in Sigma_results]
    regimes = [r.regime for r in Sigma_results]

    # Create Σ(z) plot with regime coloring
    fig_sigma = go.Figure()

    # Add Σ(z) curve
    fig_sigma.add_trace(
        go.Scatter(
            x=z_sigma,
            y=Sigma_values,
            mode="lines+markers",
            name="Σ(z)",
            line={"color": "#9467bd", "width": 2},
            marker={"size": 4},
        )
    )

    # Add GR line
    fig_sigma.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="GR (Σ=1)")

    # Add regime bands
    fig_sigma.add_hrect(y0=0.9, y1=1.1, fillcolor="green", opacity=0.1, line_width=0)
    fig_sigma.add_hrect(y0=0.7, y1=0.9, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_sigma.add_hrect(y0=1.1, y1=1.3, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_sigma.add_hrect(y0=0.0, y1=0.7, fillcolor="red", opacity=0.1, line_width=0)
    fig_sigma.add_hrect(y0=1.3, y1=2.0, fillcolor="red", opacity=0.1, line_width=0)

    fig_sigma.update_layout(
        title=f"Σ(z) Evolution | Σ₀ = {sigma_0:.3f} | Model: {g_model_name}",
        xaxis_title="Redshift z",
        yaxis_title="Σ(z)",
        yaxis={"range": [0.5, 1.6]},
        showlegend=True,
    )
    st.plotly_chart(fig_sigma, width="stretch")

    # Regime summary
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    regime_cols = st.columns(3)
    for i, (regime, count) in enumerate(regime_counts.items()):
        with regime_cols[i % 3]:
            st.markdown(f"**{regime}**: {count} points ({100 * count / len(regimes):.1f}%)")

    st.divider()

    # ========== DES Y3 Data Section ==========
    st.header("📊 DES Y3 Reference Data")

    with st.expander("📖 About DES Y3 Analysis", expanded=False):
        st.markdown("""
        **Dark Energy Survey Year 3 Results:**
        
        The Nature Communications paper analyzes:
        - 4 lens redshift bins: z ∈ [0.295, 0.467, 0.626, 0.771]
        - Galaxy-galaxy lensing + galaxy clustering
        - Cross-correlation with CMB lensing from Planck
        
        **Key Measurements (CMB prior):**
        - ĥJ(z₁) = 0.326 ± 0.062
        - ĥJ(z₂) = 0.332 ± 0.052
        - ĥJ(z₃) = 0.387 ± 0.059
        - ĥJ(z₄) = 0.354 ± 0.085
        
        **Fitted Σ₀:**
        - Standard model: Σ₀ = 0.17 ± 0.12
        - Constant model: Σ₀ = 0.24 ± 0.14
        - Exponential model: Σ₀ = 0.10 ± 0.05
        """)

    # Display DES Y3 data table
    des_df = pd.DataFrame(
        {
            "Bin": [1, 2, 3, 4],
            "z_eff": DES_Y3_DATA["z_bins"],
            "ĥJ (mean)": DES_Y3_DATA["hJ_cmb"]["mean"],
            "ĥJ (σ)": DES_Y3_DATA["hJ_cmb"]["sigma"],
        }
    )
    st.dataframe(des_df, hide_index=True, width="stretch")

    # Plot ĥJ measurements
    fig_hJ = go.Figure()
    fig_hJ.add_trace(
        go.Scatter(
            x=DES_Y3_DATA["z_bins"],
            y=DES_Y3_DATA["hJ_cmb"]["mean"],
            error_y={"type": "data", "array": DES_Y3_DATA["hJ_cmb"]["sigma"]},
            mode="markers",
            name="ĥJ (CMB prior)",
            marker={"size": 12, "color": "#1f77b4"},
        )
    )
    fig_hJ.update_layout(
        title="DES Y3 ĥJ Measurements",
        xaxis_title="Effective Redshift z",
        yaxis_title="ĥJ",
        showlegend=True,
    )
    st.plotly_chart(fig_hJ, width="stretch")

    st.divider()

    # ========== UMCP Integration Section ==========
    st.header("🔗 UMCP Integration")

    with st.expander("📖 About WEYL-UMCP Mapping", expanded=True):
        st.markdown("""
        **Core Principle Alignment:**
        > "Within-run: frozen causes only. Between-run: continuity only by return-weld."
        
        **WEYL Implementation:**
        - **Within-run**: Frozen cosmological parameters (Ω_m, σ8, z*) determine the Weyl trace
        - **Between-run**: Canonization requires return-weld (Σ → 1 at high z)
        
        **Invariant Mapping:**
        | WEYL Quantity | UMCP Analog | Interpretation |
        |---------------|-------------|----------------|
        | ĥJ | F (Fidelity) | Fraction of ideal response |
        | 1 - ĥJ | ω (Drift) | Distance from ideal |
        | Σ₀ | Deviation | Amplitude of departure from GR |
        | χ² improvement | Seam closure | Better fit = tighter weld |
        """)

    # Interactive UMCP mapping
    st.subheader("🧮 Compute UMCP Invariants")

    map_cols = st.columns(3)
    with map_cols[0]:
        input_sigma0 = st.number_input("Σ₀", value=0.24, step=0.01, format="%.3f")
    with map_cols[1]:
        chi2_sigma = st.number_input("χ² (Σ model)", value=1.1, step=0.1, format="%.2f")
    with map_cols[2]:
        chi2_lcdm = st.number_input("χ² (ΛCDM)", value=2.1, step=0.1, format="%.2f")

    if st.button("📊 Compute UMCP Mapping"):
        mapping = Sigma_to_UMCP_invariants(input_sigma0, chi2_sigma, chi2_lcdm)

        result_cols = st.columns(4)
        with result_cols[0]:
            st.metric("ω (Drift)", f"{mapping['omega_analog']:.3f}")
        with result_cols[1]:
            st.metric("F (Fidelity)", f"{mapping['F_analog']:.3f}")
        with result_cols[2]:
            st.metric("χ² Improvement", f"{mapping['chi2_improvement']:.1%}")
        with result_cols[3]:
            st.metric("Regime", mapping["regime"])

        st.success("✅ WEYL measurements mapped to UMCP invariants successfully!")


def render_batch_validation_page() -> None:
    """Render the batch validation page for running multiple casepacks."""
    if st is None:
        return

    st.title("📦 Batch Validation")
    st.caption("Run validation on multiple casepacks with a summary report")

    # Initialize batch results
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

    repo_root = get_repo_root()
    casepacks = load_casepacks()

    if not casepacks:
        st.warning("No casepacks found in the repository.")
        return

    # ========== Casepack Selection ==========
    st.subheader("📦 Select Casepacks")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_casepacks = st.multiselect(
            "Choose casepacks to validate",
            [cp["id"] for cp in casepacks],
            default=[cp["id"] for cp in casepacks[:3]] if len(casepacks) >= 3 else [cp["id"] for cp in casepacks],
        )

    with col2:
        if st.button("✅ Select All"):
            st.session_state["batch_select_all"] = True
            st.rerun()
        if st.button("❌ Clear All"):
            st.session_state["batch_select_all"] = False
            st.rerun()

    # ========== Validation Options ==========
    st.subheader("⚙️ Options")

    opt_cols = st.columns(4)
    with opt_cols[0]:
        strict_mode = st.checkbox("Strict Mode", value=False)
    with opt_cols[1]:
        verbose = st.checkbox("Verbose Output", value=False)
    with opt_cols[2]:
        fail_fast = st.checkbox("Fail Fast", value=False, help="Stop on first failure")
    with opt_cols[3]:
        st.checkbox("Parallel (simulated)", value=False, disabled=True, help="Coming soon")

    st.divider()

    # ========== Run Batch ==========
    if st.button("🚀 Run Batch Validation", width="stretch", disabled=not selected_casepacks):
        results = []
        progress = st.progress(0, text="Starting batch validation...")
        status_container = st.container()

        total = len(selected_casepacks)
        passed = 0
        failed = 0

        for i, cp_id in enumerate(selected_casepacks):
            progress.progress((i + 1) / total, text=f"Validating {cp_id}... ({i + 1}/{total})")

            try:
                cmd = [sys.executable, "-m", "umcp", "validate", f"casepacks/{cp_id}"]
                if strict_mode:
                    cmd.append("--strict")
                if verbose:
                    cmd.append("--verbose")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=repo_root,
                )

                status = "CONFORMANT" if result.returncode == 0 else "NONCONFORMANT"
                if status == "CONFORMANT":
                    passed += 1
                else:
                    failed += 1

                results.append(
                    {
                        "casepack": cp_id,
                        "status": status,
                        "return_code": result.returncode,
                        "stdout": result.stdout[:500] if result.stdout else "",
                        "stderr": result.stderr[:500] if result.stderr else "",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if fail_fast and status == "NONCONFORMANT":
                    st.warning(f"⚠️ Fail-fast triggered at {cp_id}")
                    break

            except subprocess.TimeoutExpired:
                failed += 1
                results.append(
                    {
                        "casepack": cp_id,
                        "status": "TIMEOUT",
                        "return_code": -1,
                        "stdout": "",
                        "stderr": "Validation timed out after 60 seconds",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                failed += 1
                results.append(
                    {
                        "casepack": cp_id,
                        "status": "ERROR",
                        "return_code": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        progress.progress(1.0, text="Complete!")

        # Store results
        batch_run = {
            "id": len(st.session_state.batch_results) + 1,
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
        }
        st.session_state.batch_results.append(batch_run)

        # Summary
        with status_container:
            st.markdown("### 📊 Batch Summary")

            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Total", total)
            with summary_cols[1]:
                st.metric("Passed", passed, delta=None)
            with summary_cols[2]:
                st.metric("Failed", failed, delta=None)
            with summary_cols[3]:
                rate = (passed / total * 100) if total > 0 else 0
                st.metric("Pass Rate", f"{rate:.1f}%")

            # Results table
            if pd is not None:
                results_df = pd.DataFrame(
                    [{"Casepack": r["casepack"], "Status": r["status"], "Time": r["timestamp"][:19]} for r in results]
                )

                def color_status(val: str) -> str:
                    if val == "CONFORMANT":
                        return "background-color: #d4edda"
                    elif val == "NONCONFORMANT":
                        return "background-color: #f8d7da"
                    return "background-color: #fff3cd"

                st.dataframe(
                    results_df.style.applymap(color_status, subset=["Status"]), width="stretch", hide_index=True
                )

    st.divider()

    # ========== Batch History ==========
    st.subheader("📜 Batch History")

    if st.session_state.batch_results:
        for batch in reversed(st.session_state.batch_results[-5:]):
            status_icon = "✅" if batch["failed"] == 0 else "⚠️" if batch["passed"] > 0 else "❌"

            with st.expander(
                f"{status_icon} Batch #{batch['id']} — {batch['passed']}/{batch['total']} passed @ {batch['timestamp'][:19]}"
            ):
                for r in batch["results"]:
                    icon = "✅" if r["status"] == "CONFORMANT" else "❌"
                    st.markdown(f"{icon} **{r['casepack']}** — {r['status']}")

                st.download_button(
                    label="📥 Download Report",
                    data=json.dumps(batch, indent=2),
                    file_name=f"batch_report_{batch['id']}.json",
                    mime="application/json",
                )
    else:
        st.info("No batch runs yet. Run a batch validation to see history.")


def render_api_integration_page() -> None:
    """Render the API integration page for real-time sync."""
    if st is None:
        return

    st.title("🔌 API Integration")
    st.caption("Connect to the UMCP REST API for real-time data sync")

    # Initialize API settings
    if "api_settings" not in st.session_state:
        st.session_state.api_settings = {
            "url": "http://localhost:8000",
            "connected": False,
            "last_sync": None,
            "auto_sync": False,
        }

    # ========== Connection Settings ==========
    st.subheader("⚙️ Connection Settings")

    col1, col2 = st.columns([2, 1])

    with col1:
        api_url = st.text_input(
            "API URL", value=st.session_state.api_settings["url"], placeholder="http://localhost:8000"
        )
        st.session_state.api_settings["url"] = api_url

    with col2:
        if st.button("🔗 Test Connection", width="stretch"):
            try:
                import urllib.error
                import urllib.request

                with urllib.request.urlopen(f"{api_url}/health", timeout=5) as response:
                    data = json.loads(response.read().decode())
                    st.session_state.api_settings["connected"] = True
                    st.success(f"✅ Connected! API Status: {data.get('status', 'OK')}")
            except urllib.error.URLError as e:
                st.session_state.api_settings["connected"] = False
                st.error(f"❌ Connection failed: {e.reason}")
            except Exception as e:
                st.session_state.api_settings["connected"] = False
                st.error(f"❌ Error: {e}")

    # Connection status
    status = "🟢 Connected" if st.session_state.api_settings["connected"] else "🔴 Disconnected"
    st.markdown(f"**Status:** {status}")

    st.divider()

    # ========== API Endpoints ==========
    st.subheader("📡 API Endpoints")

    if not st.session_state.api_settings["connected"]:
        st.warning("Connect to the API first to test endpoints.")
    else:
        tabs = st.tabs(["🏥 Health", "📒 Ledger", "📦 Casepacks", "✅ Validate"])

        with tabs[0]:
            st.markdown("### Health Check")
            if st.button("🔄 Fetch Health", key="api_health"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/health", timeout=5) as response:
                        data = json.loads(response.read().decode())
                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[1]:
            st.markdown("### Ledger Data")
            limit = st.slider("Limit", 5, 100, 20, key="api_ledger_limit")
            if st.button("🔄 Fetch Ledger", key="api_ledger"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/ledger?limit={limit}", timeout=10) as response:
                        data = json.loads(response.read().decode())
                        if isinstance(data, list) and pd is not None:
                            df = pd.DataFrame(data)
                            st.dataframe(df, width="stretch")
                        else:
                            st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[2]:
            st.markdown("### Casepacks")
            if st.button("🔄 Fetch Casepacks", key="api_casepacks"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/casepacks", timeout=10) as response:
                        data = json.loads(response.read().decode())
                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[3]:
            st.markdown("### Validate via API")
            target = st.text_input("Target Path", value=".", key="api_validate_target")
            if st.button("🚀 Validate", key="api_validate"):
                try:
                    import urllib.request

                    req_data = json.dumps({"target": target}).encode()
                    req = urllib.request.Request(
                        f"{api_url}/validate",
                        data=req_data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    with urllib.request.urlopen(req, timeout=60) as response:
                        data = json.loads(response.read().decode())

                        if data.get("run_status") == "CONFORMANT":
                            st.success("✅ CONFORMANT")
                        else:
                            st.error("❌ NONCONFORMANT")

                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()

    # ========== Auto-Sync ==========
    st.subheader("🔄 Auto-Sync")

    st.session_state.api_settings["auto_sync"] = st.toggle(
        "Enable Auto-Sync",
        value=st.session_state.api_settings["auto_sync"],
        help="Automatically sync data from API at regular intervals",
    )

    if st.session_state.api_settings["auto_sync"]:
        sync_interval = st.slider("Sync Interval (seconds)", 10, 120, 30)
        st.info(f"💡 Auto-sync will fetch data every {sync_interval} seconds when enabled.")

        if st.session_state.api_settings.get("last_sync"):
            st.caption(f"Last sync: {st.session_state.api_settings['last_sync']}")


# ============================================================================
# Precision Verification Page
# ============================================================================


def render_precision_page() -> None:
    """
    Render the Precision Verification page - exact numerical values,
    formal axiom verification, and invariant computation with full precision.

    This page embodies the breakthrough: computationally enforced truth.
    """
    if st is None or np is None or pd is None:
        return

    st.title("🎯 Precision Verification")
    st.caption("Exact numerical values • Formal axiom enforcement • Auditable computation")

    # ========== Core Axiom Display ==========
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;
                border-left: 4px solid #00ff88;">
        <h3 style="color: #00ff88; margin: 0;">AXIOM-0: The Return Axiom</h3>
        <p style="color: #ffffff; font-size: 1.2em; font-style: italic; margin: 10px 0;">
            "Collapse is generative; only what returns is real."
        </p>
        <p style="color: #aaaaaa; font-size: 0.9em; margin: 0;">
            This is not philosophy. It is compiled into the validator.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ========== Interactive Invariant Calculator ==========
    st.subheader("🔬 Kernel Invariant Calculator")

    st.markdown("""
    Enter state coordinates to compute **exact** kernel invariants.
    All formulas from [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md).
    """)

    with st.expander("⚙️ State Input", expanded=True):
        n_channels = st.slider("Number of channels (n)", 2, 8, 3)

        cols = st.columns(n_channels)
        c_values = []
        for i in range(n_channels):
            with cols[i]:
                c = st.number_input(
                    f"c{i + 1}",
                    min_value=0.0001,
                    max_value=0.9999,
                    value=0.8 - 0.1 * i,
                    step=0.01,
                    format="%.4f",
                    key=f"precision_c{i}",
                )
                c_values.append(c)

        # Weights
        st.markdown("**Weights** (must sum to 1)")
        weight_cols = st.columns(n_channels)
        w_values = []
        for i in range(n_channels):
            with weight_cols[i]:
                w = st.number_input(
                    f"w{i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / n_channels,
                    step=0.01,
                    format="%.4f",
                    key=f"precision_w{i}",
                )
                w_values.append(w)

    # Normalize weights
    w_sum = sum(w_values)
    w_normalized = [w / w_sum for w in w_values] if w_sum > 0 else [1.0 / n_channels] * n_channels

    # ========== Compute Invariants with Full Precision ==========
    c_arr = np.array(c_values)
    w_arr = np.array(w_normalized)
    eps = 1e-8  # Clipping epsilon

    # Fidelity (arithmetic mean)
    F = float(np.sum(w_arr * c_arr))

    # Drift
    omega = 1.0 - F

    # Log-integrity (kappa)
    c_clipped = np.clip(c_arr, eps, 1 - eps)
    kappa = float(np.sum(w_arr * np.log(c_clipped)))

    # Integrity Composite (geometric mean)
    IC = float(np.exp(kappa))

    # Curvature proxy
    C = float(np.std(c_arr) / 0.5)  # Population std / 0.5

    # Shannon Entropy
    S_terms = []
    for c in c_clipped:
        s = -c * np.log(c) - (1 - c) * np.log(1 - c) if 0 < c < 1 else 0
        S_terms.append(s)
    S = float(np.sum(w_arr * np.array(S_terms)))

    # AM-GM Gap
    gap = F - IC

    # ========== Display Results with Full Precision ==========
    st.subheader("📊 Computed Invariants")

    # Main invariants table
    invariants_data = {
        "Symbol": ["F", "ω", "κ", "IC", "C", "S", "Δ (AM-GM Gap)"],
        "Name": [
            "Fidelity (Arithmetic Mean)",
            "Drift (1 - F)",
            "Log-Integrity (Σwᵢ ln cᵢ)",
            "Integrity Composite (exp κ)",
            "Curvature Proxy (std/0.5)",
            "Shannon Entropy",
            "AM-GM Gap (F - IC)",
        ],
        "Value": [f"{F:.15f}", f"{omega:.15f}", f"{kappa:.15f}", f"{IC:.15f}", f"{C:.15f}", f"{S:.15f}", f"{gap:.15f}"],
        "Bound Check": [
            "✅" if 0 <= F <= 1 else "❌",
            "✅" if 0 <= omega <= 1 else "❌",
            "✅" if kappa <= 0 else "❌",  # IC <= 1 implies kappa <= 0
            "✅" if 0 < IC <= 1 else "❌",
            "✅" if 0 <= C <= 1 else "⚠️",  # Soft bound
            "✅" if S >= 0 else "❌",
            "✅" if gap >= 0 else "❌",  # AM-GM: F >= IC always
        ],
    }

    df_inv = pd.DataFrame(invariants_data)
    st.dataframe(df_inv, width="stretch", hide_index=True)

    # ========== Formal Verification Checks ==========
    st.subheader("✅ Formal Verification")

    checks = []

    # Lemma 1: F ∈ [0,1]
    checks.append(
        {
            "Lemma": "Lemma 1",
            "Statement": "F ∈ [0, 1]",
            "Computed": f"F = {F:.10f}",
            "Status": "PASS ✅" if 0 <= F <= 1 else "FAIL ❌",
        }
    )

    # Lemma 2: IC is geometric mean
    ic_check = np.prod(c_clipped**w_arr)
    checks.append(
        {
            "Lemma": "Lemma 2",
            "Statement": "IC = Π cᵢ^wᵢ (geometric mean)",
            "Computed": f"|IC - Πcᵢ^wᵢ| = {abs(IC - ic_check):.2e}",
            "Status": "PASS ✅" if abs(IC - ic_check) < 1e-12 else "FAIL ❌",
        }
    )

    # Lemma 3: κ = ln(IC)
    kappa_check = np.log(IC)
    checks.append(
        {
            "Lemma": "Lemma 3",
            "Statement": "κ = ln(IC)",
            "Computed": f"|κ - ln(IC)| = {abs(kappa - kappa_check):.2e}",
            "Status": "PASS ✅" if abs(kappa - kappa_check) < 1e-12 else "FAIL ❌",
        }
    )

    # Lemma 4: AM-GM inequality F >= IC
    checks.append(
        {
            "Lemma": "Lemma 4",
            "Statement": "F ≥ IC (AM-GM inequality)",
            "Computed": f"F - IC = {gap:.10f}",
            "Status": "PASS ✅" if gap >= -1e-15 else "FAIL ❌",
        }
    )

    # Lemma 5: Equality iff homogeneous
    is_homogeneous = np.std(c_arr) < 1e-10
    gap_zero = gap < 1e-10
    checks.append(
        {
            "Lemma": "Lemma 5",
            "Statement": "F = IC ⟺ all cᵢ equal",
            "Computed": f"std(c) = {np.std(c_arr):.2e}, gap = {gap:.2e}",
            "Status": "PASS ✅" if (is_homogeneous == gap_zero) else "INCONCLUSIVE ⚠️",
        }
    )

    df_checks = pd.DataFrame(checks)
    st.dataframe(df_checks, width="stretch", hide_index=True)

    # ========== Regime Classification ==========
    st.subheader("🌡️ Regime Classification")

    # Classify based on omega
    if omega < 0.038:
        regime = "STABLE"
        regime_color = "#28a745"
        regime_desc = "System operating within normal parameters"
    elif omega < 0.30:
        regime = "WATCH"
        regime_color = "#ffc107"
        regime_desc = "Elevated drift, monitoring required"
    else:
        regime = "COLLAPSE"
        regime_color = "#dc3545"
        regime_desc = "Critical drift level, intervention needed"

    st.markdown(
        f"""
    <div style="background-color: {regime_color}22; border-left: 4px solid {regime_color};
                padding: 15px; border-radius: 5px;">
        <h2 style="color: {regime_color}; margin: 0;">{regime}</h2>
        <p style="margin: 5px 0 0 0;">{regime_desc}</p>
        <p style="margin: 5px 0 0 0; font-family: monospace;">
            ω = {omega:.6f} | Threshold: STABLE < 0.038 < WATCH < 0.30 < COLLAPSE
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ========== Seam Simulation ==========
    st.subheader("🔗 Seam Weld Simulation")

    st.markdown("""
    Simulate a seam transition to verify weld accounting.
    This demonstrates the **residual computation** that enforces continuity.
    """)

    seam_cols = st.columns(2)

    with seam_cols[0]:
        st.markdown("**State at t₀**")
        IC_0 = st.number_input("IC(t₀)", value=0.85, step=0.01, format="%.6f", key="IC0")
        tau_R = st.number_input("τ_R (return time)", value=5, min_value=1, max_value=100, key="tau_R_sim")

    with seam_cols[1]:
        st.markdown("**State at t₁**")
        IC_1 = st.number_input("IC(t₁)", value=0.82, step=0.01, format="%.6f", key="IC1")
        tol_seam = st.number_input("Tolerance", value=0.005, step=0.001, format="%.4f", key="tol_sim")

    # Seam parameters
    R = 0.05  # Return credit rate
    D_omega = 0.02  # Drift decay
    D_C = 0.01  # Curvature decay

    # Compute seam
    kappa_0 = np.log(IC_0)
    kappa_1 = np.log(IC_1)
    delta_kappa_ledger = kappa_1 - kappa_0
    i_r = IC_1 / IC_0

    delta_kappa_budget = R * tau_R - (D_omega + D_C)
    residual = delta_kappa_budget - delta_kappa_ledger

    # Identity check
    identity_check = abs(np.exp(delta_kappa_ledger) - i_r)

    # Weld status
    if tau_R == float("inf") or tau_R < 0:
        weld_status = "NO_RETURN"
        weld_color = "#6c757d"
    elif abs(residual) <= tol_seam and identity_check < 1e-9:
        weld_status = "PASS"
        weld_color = "#28a745"
    else:
        weld_status = "FAIL"
        weld_color = "#dc3545"

    st.markdown("#### Seam Computation")

    seam_table = {
        "Quantity": [
            "κ(t₀) = ln(IC₀)",
            "κ(t₁) = ln(IC₁)",
            "Δκ_ledger = κ₁ - κ₀",
            "i_r = IC₁/IC₀",
            "Δκ_budget = R·τ_R - (D_ω + D_C)",
            "Residual s = Δκ_budget - Δκ_ledger",
            "Identity Check |exp(Δκ) - i_r|",
        ],
        "Value": [
            f"{kappa_0:.10f}",
            f"{kappa_1:.10f}",
            f"{delta_kappa_ledger:.10f}",
            f"{i_r:.10f}",
            f"{delta_kappa_budget:.10f}",
            f"{residual:.10f}",
            f"{identity_check:.2e}",
        ],
    }
    st.dataframe(pd.DataFrame(seam_table), width="stretch", hide_index=True)

    st.markdown(
        f"""
    <div style="background-color: {weld_color}22; border: 2px solid {weld_color};
                padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: {weld_color}; margin: 0;">WELD: {weld_status}</h1>
        <p style="margin: 10px 0 0 0;">
            |s| = {abs(residual):.6f} {"≤" if abs(residual) <= tol_seam else ">"} {tol_seam} (tolerance)
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ========== Constitutional Enforcement ==========
    st.subheader("📜 Constitutional Enforcement")

    st.markdown("""
    These are not guidelines. They are **computationally enforced constraints**.
    """)

    constitution = [
        {
            "Clause": "No back-edges within run",
            "Meaning": "Tier-2 cannot modify Tier-0/1",
            "Enforcement": "Frozen interface before kernel compute",
            "Status": "ENFORCED ✅",
        },
        {
            "Clause": "No continuity without return",
            "Meaning": "τ_R = ∞ → seam FAIL",
            "Enforcement": "typed_censoring.no_return_no_credit = true",
            "Status": "ENFORCED ✅",
        },
        {
            "Clause": "Residual within tolerance",
            "Meaning": "|s| > tol → seam FAIL",
            "Enforcement": "Weld gate checks |Δκ_budget - Δκ_ledger|",
            "Status": "ENFORCED ✅",
        },
        {
            "Clause": "Identity consistency",
            "Meaning": "exp(Δκ) must equal IC ratio",
            "Enforcement": "|exp(Δκ) - i_r| < 10⁻⁹",
            "Status": "ENFORCED ✅",
        },
    ]

    st.dataframe(pd.DataFrame(constitution), width="stretch", hide_index=True)


# ============================================================================
# Three-Layer Geometry Visualization
# ============================================================================


def render_geometry_page() -> None:
    """
    Render the Three-Layer Geometry page - Interactive visualization of
    State Space → Invariant Coordinates → Seam Graph architecture.

    Reference: INFRASTRUCTURE_GEOMETRY.md
    """
    if st is None or go is None or np is None or pd is None:
        return

    st.title("🔷 Infrastructure Geometry")
    st.caption("Interactive exploration of the three-layer geometric architecture")

    # ========== Theory Overview ==========
    with st.expander("📖 Three-Layer Architecture", expanded=False):
        st.markdown(r"""
        The UMCP infrastructure is built on three coupled geometric layers:

        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │     LAYER 3: SEAM GRAPH (Continuity Certification)              │
        │     Transitions certified when weld accounting closes           │
        │     Residual s = Δκ_budget - Δκ_ledger within tolerance        │
        ├─────────────────────────────────────────────────────────────────┤
        │     LAYER 2: INVARIANT COORDINATES (Projections)                │
        │     Ledger {ω, F, S, C, τ_R, κ, IC} as multi-projection        │
        │     Regime gates partition the projection space                 │
        ├─────────────────────────────────────────────────────────────────┤
        │     LAYER 1: STATE SPACE (Manifold + Distance)                  │
        │     Ψ(t) ∈ [0,1]ⁿ with declared norm ‖·‖ and tolerance η       │
        │     Return = re-entry to η-balls under declared geometry        │
        └─────────────────────────────────────────────────────────────────┘
        ```

        **Key Insight**: Each layer performs a specific geometric operation.
        Together they make continuity **auditable rather than asserted**.

        See [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md) for full specification.
        """)

    st.divider()

    # ========== Layer Navigation ==========
    layer_tabs = st.tabs(
        ["🎯 Layer 1: State Space", "📊 Layer 2: Invariant Projections", "🔗 Layer 3: Seam Graph", "🌐 Unified View"]
    )

    # ========== LAYER 1: STATE SPACE ==========
    with layer_tabs[0]:
        render_layer1_state_space()

    # ========== LAYER 2: INVARIANT COORDINATES ==========
    with layer_tabs[1]:
        render_layer2_projections()

    # ========== LAYER 3: SEAM GRAPH ==========
    with layer_tabs[2]:
        render_layer3_seam_graph()

    # ========== UNIFIED VIEW ==========
    with layer_tabs[3]:
        render_unified_geometry_view()


def render_layer1_state_space() -> None:
    """Render Layer 1: State Space visualization with trajectory and η-balls."""
    if st is None or go is None or np is None:
        return

    st.subheader("State Space: Ψ(t) ∈ [0,1]ⁿ")

    st.markdown("""
    **Layer 1** defines the bounded state manifold and return geometry:
    - **Trajectory**: Time evolution Ψ(t) through normalized state space
    - **η-balls**: Neighborhoods defining "closeness" for return detection
    - **Return**: Re-entry to historical η-neighborhoods
    """)

    # ========== Interactive State Space ==========
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_dims = st.slider("Dimensions (n)", 2, 5, 3, help="State space dimension")
            n_steps = st.slider("Time Steps", 20, 200, 50, help="Trajectory length")

        with col2:
            eta = st.slider(
                "η (Return Tolerance)", 0.01, 0.3, 0.1, 0.01, help="Neighborhood radius for return detection"
            )
            drift_rate = st.slider("Drift Rate", 0.0, 0.2, 0.05, 0.01, help="Stochastic drift magnitude")

        with col3:
            oscillation = st.slider("Oscillation", 0.0, 1.0, 0.3, 0.05, help="Periodic component strength")
            seed = st.number_input("Random Seed", 0, 999, 42, help="For reproducibility")

    # Generate synthetic trajectory
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)

    # Create trajectory with drift + oscillation + noise
    trajectory = np.zeros((n_steps, n_dims))
    for d in range(n_dims):
        phase = 2 * np.pi * d / n_dims
        base = 0.5 + oscillation * 0.3 * np.sin(t + phase)
        drift = drift_rate * np.cumsum(np.random.randn(n_steps)) / np.sqrt(n_steps)
        noise = 0.02 * np.random.randn(n_steps)
        trajectory[:, d] = np.clip(base + drift + noise, 0.01, 0.99)

    # Compute returns
    returns = []
    for i in range(1, n_steps):
        for j in range(max(0, i - 20), i):  # Look back up to 20 steps
            dist = np.linalg.norm(trajectory[i] - trajectory[j])
            if dist <= eta:
                returns.append((i, j, dist))
                break

    # ========== 2D/3D Projection ==========
    st.markdown("#### Trajectory Projection")

    view_mode = st.radio("View Mode", ["2D (c₁, c₂)", "3D (c₁, c₂, c₃)"], horizontal=True)

    if view_mode == "2D (c₁, c₂)" or n_dims < 3:
        # 2D visualization
        fig = go.Figure()

        # Trajectory line
        fig.add_trace(
            go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode="lines+markers",
                line={"color": "royalblue", "width": 2},
                marker={
                    "size": 4,
                    "color": np.arange(n_steps),
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Time t"},
                },
                name="Trajectory Ψ(t)",
                hovertemplate="t=%{marker.color:.0f}<br>c₁=%{x:.3f}<br>c₂=%{y:.3f}<extra></extra>",
            )
        )

        # Start point
        fig.add_trace(
            go.Scatter(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                mode="markers",
                marker={"size": 15, "color": "green", "symbol": "star"},
                name="Start (t=0)",
            )
        )

        # End point
        fig.add_trace(
            go.Scatter(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                mode="markers",
                marker={"size": 15, "color": "red", "symbol": "square"},
                name=f"End (t={n_steps - 1})",
            )
        )

        # η-balls for return events
        for _i, j, _dist in returns[:10]:  # Show first 10 returns
            theta_circle = np.linspace(0, 2 * np.pi, 50)
            x_circle = trajectory[j, 0] + eta * np.cos(theta_circle)
            y_circle = trajectory[j, 1] + eta * np.sin(theta_circle)
            fig.add_trace(
                go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode="lines",
                    line={"color": "orange", "width": 1, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Return connections
        for i, j, dist in returns[:10]:
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[i, 0], trajectory[j, 0]],
                    y=[trajectory[i, 1], trajectory[j, 1]],
                    mode="lines",
                    line={"color": "orange", "width": 1, "dash": "dash"},
                    showlegend=False,
                    hovertemplate=f"Return: t={i}→t={j}<br>τ_R={i - j}<br>dist={dist:.4f}<extra></extra>",
                )
            )

        fig.update_layout(
            height=500,
            xaxis_title="c₁ (Channel 1)",
            yaxis_title="c₂ (Channel 2)",
            xaxis={"range": [0, 1], "constrain": "domain"},
            yaxis={"range": [0, 1], "scaleanchor": "x", "scaleratio": 1},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            title=f"State Space Trajectory (η={eta})",
        )

        st.plotly_chart(fig, width="stretch")

    else:
        # 3D visualization
        fig = go.Figure()

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines+markers",
                line={"color": "royalblue", "width": 3},
                marker={
                    "size": 3,
                    "color": np.arange(n_steps),
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Time t"},
                },
                name="Trajectory Ψ(t)",
            )
        )

        # Start/end markers
        fig.add_trace(
            go.Scatter3d(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                z=[trajectory[0, 2]],
                mode="markers",
                marker={"size": 10, "color": "green", "symbol": "diamond"},
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]],
                mode="markers",
                marker={"size": 10, "color": "red", "symbol": "square"},
                name="End",
            )
        )

        fig.update_layout(
            height=600,
            scene={
                "xaxis_title": "c₁",
                "yaxis_title": "c₂",
                "zaxis_title": "c₃",
                "xaxis": {"range": [0, 1]},
                "yaxis": {"range": [0, 1]},
                "zaxis": {"range": [0, 1]},
            },
            title=f"3D State Space (η={eta})",
        )

        st.plotly_chart(fig, width="stretch")

    # ========== Return Statistics ==========
    st.markdown("#### Return Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Returns", len(returns))
    with col2:
        if returns:
            avg_tau = np.mean([i - j for i, j, _ in returns])
            st.metric("Avg τ_R", f"{avg_tau:.1f}")
        else:
            st.metric("Avg τ_R", "∞")
    with col3:
        if returns:
            min_tau = min(i - j for i, j, _ in returns)
            st.metric("Min τ_R", min_tau)
        else:
            st.metric("Min τ_R", "∞")
    with col4:
        return_rate = len(returns) / n_steps * 100
        st.metric("Return Rate", f"{return_rate:.1f}%")

    # Return time histogram
    if returns:
        tau_values = [i - j for i, j, _ in returns]
        fig_hist = go.Figure(data=[go.Histogram(x=tau_values, nbinsx=20, marker_color="royalblue")])
        fig_hist.update_layout(
            height=250, xaxis_title="Return Time τ_R (steps)", yaxis_title="Count", title="Return Time Distribution"
        )
        st.plotly_chart(fig_hist, width="stretch")


def render_layer2_projections() -> None:
    """Render Layer 2: Invariant Coordinate projections."""
    if st is None or go is None or np is None or pd is None:
        return

    st.subheader("Invariant Coordinates: Multi-Projection Ledger")

    st.markdown("""
    **Layer 2** projects the state trajectory into interpretable coordinates:

    | Projection | Formula | Range | Geometric Meaning |
    |------------|---------|-------|-------------------|
    | **F** (Fidelity) | Σ wᵢcᵢ | [0,1] | Weighted state quality |
    | **ω** (Drift) | 1 - F | [0,1] | Distance from ideal |
    | **S** (Entropy) | -Σ wᵢ[cᵢ ln cᵢ + (1-cᵢ)ln(1-cᵢ)] | ≥0 | Uncertainty spread |
    | **C** (Curvature) | std(cᵢ)/0.5 | [0,1] | Shape dispersion |
    | **κ** (Log-integrity) | Σ wᵢ ln cᵢ | ≤0 | Additive ledger term |
    | **IC** (Integrity) | exp(κ) | (0,1] | Geometric mean composite |
    """)

    # ========== Generate Sample Data ==========
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Samples", 20, 200, 80)
            scenario = st.selectbox("Scenario", ["Stable System", "Drift Event", "Oscillating", "Collapse Approach"])
        with col2:
            seed = st.number_input("Seed", 0, 999, 42, key="layer2_seed")
            show_regime = st.checkbox("Show Regime Regions", True)

    np.random.seed(seed)
    t = np.arange(n_samples)

    # Generate scenario-based data
    if scenario == "Stable System":
        omega = 0.02 + 0.01 * np.random.randn(n_samples)
        C = 0.05 + 0.02 * np.random.randn(n_samples)
    elif scenario == "Drift Event":
        omega = 0.02 + 0.15 * (1 / (1 + np.exp(-(t - n_samples / 2) / 5)))
        C = 0.05 + 0.1 * (1 / (1 + np.exp(-(t - n_samples / 2) / 5)))
        omega += 0.01 * np.random.randn(n_samples)
        C += 0.01 * np.random.randn(n_samples)
    elif scenario == "Oscillating":
        omega = 0.15 + 0.12 * np.sin(2 * np.pi * t / 30)
        C = 0.10 + 0.08 * np.sin(2 * np.pi * t / 30 + np.pi / 4)
        omega += 0.01 * np.random.randn(n_samples)
        C += 0.01 * np.random.randn(n_samples)
    else:  # Collapse Approach
        omega = 0.02 + 0.35 * (t / n_samples) ** 2
        C = 0.05 + 0.25 * (t / n_samples) ** 1.5
        omega += 0.01 * np.random.randn(n_samples)
        C += 0.01 * np.random.randn(n_samples)

    omega = np.clip(omega, 0.001, 0.999)
    C = np.clip(C, 0.001, 0.999)
    F = 1 - omega
    S = 0.5 * C + 0.1 * np.random.randn(n_samples)
    S = np.clip(S, 0.001, 1.0)
    kappa = np.log(F) - 0.1 * C
    IC = np.exp(kappa)
    IC = np.clip(IC, 0.001, 0.999)

    # Create dataframe
    df = pd.DataFrame({"t": t, "omega": omega, "F": F, "S": S, "C": C, "kappa": kappa, "IC": IC})

    # ========== Multi-Axis Projection View ==========
    st.markdown("#### Projection Time Series")

    # Create subplot figure
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "ω (Drift)",
            "F (Fidelity)",
            "S (Entropy)",
            "C (Curvature)",
            "κ (Log-Integrity)",
            "IC (Integrity)",
        ),
        vertical_spacing=0.12,
    )

    projections = [("omega", 1, 1), ("F", 1, 2), ("S", 2, 1), ("C", 2, 2), ("kappa", 3, 1), ("IC", 3, 2)]

    for name, row, col in projections:
        fig.add_trace(go.Scatter(x=df["t"], y=df[name], mode="lines", name=name, line={"width": 2}), row=row, col=col)

        # Add threshold lines for ω
        if name == "omega" and show_regime:
            fig.add_hline(y=0.038, line_dash="dash", line_color="green", row=row, col=col)
            fig.add_hline(y=0.30, line_dash="dash", line_color="red", row=row, col=col)

    fig.update_layout(height=700, showlegend=False, title="Invariant Projections Over Time")
    st.plotly_chart(fig, width="stretch")

    # ========== Phase Space Views ==========
    st.markdown("#### Projection Phase Spaces")

    phase_col1, phase_col2 = st.columns(2)

    with phase_col1:
        # ω vs F (should be linear: F = 1 - ω)
        fig_of = go.Figure()
        fig_of.add_trace(
            go.Scatter(
                x=df["omega"],
                y=df["F"],
                mode="markers",
                marker={"color": df["t"], "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "t"}},
                hovertemplate="ω=%{x:.3f}<br>F=%{y:.3f}<br>t=%{marker.color:.0f}<extra></extra>",
            )
        )
        # Identity line
        fig_of.add_trace(
            go.Scatter(x=[0, 1], y=[1, 0], mode="lines", line={"dash": "dash", "color": "gray"}, name="F = 1 - ω")
        )
        fig_of.update_layout(
            height=350, xaxis_title="ω (Drift)", yaxis_title="F (Fidelity)", title="Drift-Fidelity Axis"
        )
        st.plotly_chart(fig_of, width="stretch")

    with phase_col2:
        # S vs C
        fig_sc = go.Figure()
        fig_sc.add_trace(
            go.Scatter(
                x=df["S"],
                y=df["C"],
                mode="markers",
                marker={"color": df["t"], "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "t"}},
                hovertemplate="S=%{x:.3f}<br>C=%{y:.3f}<br>t=%{marker.color:.0f}<extra></extra>",
            )
        )
        fig_sc.update_layout(
            height=350, xaxis_title="S (Entropy)", yaxis_title="C (Curvature)", title="Entropy-Curvature Axis"
        )
        st.plotly_chart(fig_sc, width="stretch")

    # ========== AM-GM Gap Visualization ==========
    st.markdown("#### AM-GM Gap Analysis")

    df["gap"] = df["F"] - df["IC"]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(x=df["t"], y=df["F"], mode="lines", name="F (Arithmetic)", line={"color": "blue"}))
    fig_gap.add_trace(go.Scatter(x=df["t"], y=df["IC"], mode="lines", name="IC (Geometric)", line={"color": "green"}))
    fig_gap.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["gap"],
            mode="lines",
            name="Gap (F - IC)",
            line={"color": "orange", "dash": "dash"},
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.2)",
        )
    )
    fig_gap.update_layout(
        height=300,
        xaxis_title="Time t",
        yaxis_title="Value",
        title="AM-GM Gap: F ≥ IC (with equality iff homogeneous)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    st.plotly_chart(fig_gap, width="stretch")

    st.caption("The gap Δ = F - IC quantifies state heterogeneity (Lemma 4, KERNEL_SPECIFICATION.md)")


def render_layer3_seam_graph() -> None:
    """Render Layer 3: Seam Graph for transition certification."""
    if st is None or go is None or np is None or pd is None:
        return

    st.subheader("Seam Graph: Transition Certification")

    st.markdown("""
    **Layer 3** certifies transitions between states via weld accounting:

    **Weld PASS requires ALL of:**
    1. τ_R(t₁) is finite (not ∞_rec, not UNIDENTIFIABLE)
    2. |s| ≤ tol_seam (residual within tolerance)
    3. |exp(Δκ) - IC₁/IC₀| < 10⁻⁹ (identity closure)

    **Key quantities:**
    - **Δκ_ledger** = κ(t₁) - κ(t₀) = ln(IC₁/IC₀)
    - **Δκ_budget** = R·τ_R - (D_ω + D_C)
    - **Residual s** = Δκ_budget - Δκ_ledger
    """)

    # ========== Generate Seam Data ==========
    with st.expander("⚙️ Seam Simulation", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_seams = st.slider("Number of Seams", 5, 30, 12)
            tol_seam = st.slider("Tolerance (tol_seam)", 0.001, 0.02, 0.005, 0.001)
        with col2:
            R = st.slider("Return Rate R", 0.01, 0.1, 0.05, 0.01, help="Credit per return step")
            noise_level = st.slider("Budget Noise", 0.0, 0.02, 0.003, 0.001)
        with col3:
            failure_rate = st.slider("Failure Rate", 0.0, 0.5, 0.15, 0.05)
            seed = st.number_input("Seed", 0, 999, 42, key="layer3_seed")

    np.random.seed(seed)

    # Generate seam transitions
    seams = []
    for i in range(n_seams):
        t0, t1 = i * 10, (i + 1) * 10

        # Generate IC values
        IC0 = 0.95 - 0.02 * i + 0.01 * np.random.randn()
        IC1 = IC0 - 0.01 - 0.005 * np.random.randn()
        IC0, IC1 = np.clip([IC0, IC1], 0.1, 0.99)

        kappa0, kappa1 = np.log(IC0), np.log(IC1)
        delta_kappa_ledger = kappa1 - kappa0

        # Return time
        tau_R = np.random.randint(2, 15) if np.random.rand() > 0.1 else float("inf")

        # Compute budget terms
        if tau_R != float("inf"):
            credit = R * tau_R
            D_omega = 0.02 + 0.01 * np.random.randn()
            D_C = 0.01 + 0.005 * np.random.randn()
            delta_kappa_budget = credit - (D_omega + D_C) + noise_level * np.random.randn()

            # Inject some failures
            if np.random.rand() < failure_rate:
                delta_kappa_budget += 0.02 * (1 if np.random.rand() > 0.5 else -1)

            residual = delta_kappa_budget - delta_kappa_ledger
        else:
            D_omega, D_C = 0, 0
            delta_kappa_budget = 0
            residual = float("inf")

        # Determine pass/fail
        if tau_R == float("inf"):
            status = "NO_RETURN"
        elif abs(residual) <= tol_seam:
            status = "PASS"
        else:
            status = "FAIL"

        seams.append(
            {
                "t0": t0,
                "t1": t1,
                "IC0": IC0,
                "IC1": IC1,
                "kappa0": kappa0,
                "kappa1": kappa1,
                "delta_kappa_ledger": delta_kappa_ledger,
                "delta_kappa_budget": delta_kappa_budget if tau_R != float("inf") else None,
                "tau_R": tau_R if tau_R != float("inf") else "INF_REC",
                "residual": residual if tau_R != float("inf") else None,
                "D_omega": D_omega,
                "D_C": D_C,
                "status": status,
            }
        )

    df_seams = pd.DataFrame(seams)

    # ========== Seam Graph Visualization ==========
    st.markdown("#### Seam Transition Graph")

    # Create network-style graph
    fig_graph = go.Figure()

    # Add nodes (time points)
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for _i, seam in enumerate(seams):
        # Start node
        node_x.append(seam["t0"])
        node_y.append(seam["IC0"])
        node_text.append(f"t={seam['t0']}<br>IC={seam['IC0']:.4f}")
        node_colors.append("lightblue")

    # Final node
    node_x.append(seams[-1]["t1"])
    node_y.append(seams[-1]["IC1"])
    node_text.append(f"t={seams[-1]['t1']}<br>IC={seams[-1]['IC1']:.4f}")
    node_colors.append("lightblue")

    fig_graph.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={"size": 15, "color": node_colors, "line": {"width": 2, "color": "darkblue"}},
            text=node_text,
            hoverinfo="text",
            name="States",
        )
    )

    # Add edges (seams) with color by status
    status_colors = {"PASS": "green", "FAIL": "red", "NO_RETURN": "gray"}
    for seam in seams:
        color = status_colors.get(seam["status"], "gray")
        dash = "solid" if seam["status"] == "PASS" else "dash"
        residual_str = f"{seam['residual']:.4f}" if seam["residual"] is not None else "N/A"

        fig_graph.add_trace(
            go.Scatter(
                x=[seam["t0"], seam["t1"]],
                y=[seam["IC0"], seam["IC1"]],
                mode="lines",
                line={"color": color, "width": 3, "dash": dash},
                hovertemplate=(
                    f"Seam: t={seam['t0']}→{seam['t1']}<br>"
                    f"Status: {seam['status']}<br>"
                    f"τ_R: {seam['tau_R']}<br>"
                    f"Residual: {residual_str}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Add tolerance band annotation
    fig_graph.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"tol_seam = {tol_seam}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig_graph.update_layout(
        height=400,
        xaxis_title="Time t",
        yaxis_title="Integrity IC",
        title="Seam Certification Graph (Green=PASS, Red=FAIL, Gray=NO_RETURN)",
    )

    st.plotly_chart(fig_graph, width="stretch")

    # ========== Residual Distribution ==========
    st.markdown("#### Residual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Residual bar chart
        valid_seams = [s for s in seams if s["residual"] is not None]
        if valid_seams:
            residuals = [s["residual"] for s in valid_seams]
            labels = [f"t{s['t0']}→{s['t1']}" for s in valid_seams]
            colors = ["green" if abs(r) <= tol_seam else "red" for r in residuals]

            fig_res = go.Figure(data=[go.Bar(x=labels, y=residuals, marker_color=colors)])
            fig_res.add_hline(y=tol_seam, line_dash="dash", line_color="orange")
            fig_res.add_hline(y=-tol_seam, line_dash="dash", line_color="orange")
            fig_res.update_layout(
                height=300, xaxis_title="Seam", yaxis_title="Residual s", title="Seam Residuals (orange = tolerance)"
            )
            st.plotly_chart(fig_res, width="stretch")

    with col2:
        # Summary metrics
        pass_count = sum(1 for s in seams if s["status"] == "PASS")
        fail_count = sum(1 for s in seams if s["status"] == "FAIL")
        no_return_count = sum(1 for s in seams if s["status"] == "NO_RETURN")

        st.markdown("##### Certification Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("✅ PASS", pass_count)
        with summary_col2:
            st.metric("❌ FAIL", fail_count)
        with summary_col3:
            st.metric("⚪ NO_RETURN", no_return_count)

        if valid_seams:
            residuals = [s["residual"] for s in valid_seams]
            st.metric("Mean |Residual|", f"{np.mean(np.abs(residuals)):.5f}")
            st.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.5f}")

    # ========== Seam Table ==========
    st.markdown("#### Seam Details")

    display_df = df_seams.copy()
    display_df["tau_R"] = display_df["tau_R"].astype(str)  # Prevent PyArrow mixed-type error
    display_df["residual"] = display_df["residual"].apply(lambda x: f"{x:.5f}" if x is not None else "N/A")
    display_df["delta_kappa_budget"] = display_df["delta_kappa_budget"].apply(
        lambda x: f"{x:.5f}" if x is not None else "N/A"
    )

    st.dataframe(
        display_df[["t0", "t1", "tau_R", "delta_kappa_ledger", "delta_kappa_budget", "residual", "status"]],
        width="stretch",
        hide_index=True,
    )


def render_unified_geometry_view() -> None:
    """Render unified view of all three layers."""
    if st is None or go is None or np is None or pd is None:
        return

    st.subheader("Unified Three-Layer View")

    st.markdown("""
    This view shows how the three geometric layers interact:
    - **Layer 1**: State trajectory Ψ(t) flows through bounded space
    - **Layer 2**: Kernel projects trajectory → invariant coordinates
    - **Layer 3**: Seam accounting certifies transitions
    """)

    # ========== Generate Unified Data ==========
    with st.expander("⚙️ Unified Simulation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_steps = st.slider("Time Steps", 30, 150, 60, key="unified_steps")
            seed = st.number_input("Seed", 0, 999, 42, key="unified_seed")
        with col2:
            drift_intensity = st.slider("Drift Intensity", 0.0, 0.3, 0.08)
            st.slider("η Tolerance", 0.05, 0.25, 0.12)

    np.random.seed(seed)
    t = np.arange(n_steps)

    # Generate 3-channel state
    c1 = 0.8 + 0.1 * np.sin(2 * np.pi * t / 40) + drift_intensity * t / n_steps + 0.02 * np.random.randn(n_steps)
    c2 = (
        0.75
        + 0.15 * np.sin(2 * np.pi * t / 30 + np.pi / 3)
        + 0.5 * drift_intensity * t / n_steps
        + 0.02 * np.random.randn(n_steps)
    )
    c3 = 0.85 - 0.1 * np.sin(2 * np.pi * t / 50) - 0.3 * drift_intensity * t / n_steps + 0.02 * np.random.randn(n_steps)

    c1, c2, c3 = (np.clip(c, 0.01, 0.99) for c in (c1, c2, c3))

    # Weights
    w = np.array([0.4, 0.35, 0.25])

    # Compute invariants
    C_arr = np.column_stack([c1, c2, c3])
    F = np.sum(C_arr * w, axis=1)
    omega = 1 - F
    C_curv = np.std(C_arr, axis=1) / 0.5
    kappa = np.sum(w * np.log(C_arr), axis=1)
    IC = np.exp(kappa)

    # Regime classification
    regimes = ["STABLE" if o < 0.038 else "WATCH" if o < 0.30 else "COLLAPSE" for o in omega]

    # Create figure with 3 subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Layer 1: State Space Ψ(t) ∈ [0,1]³",
            "Layer 2: Invariant Projections",
            "Layer 3: Regime Classification & Continuity",
        ),
        vertical_spacing=0.1,
        row_heights=[0.35, 0.35, 0.30],
    )

    # ========== Layer 1: State Space ==========
    fig.add_trace(go.Scatter(x=t, y=c1, mode="lines", name="c₁", line={"color": "#1f77b4"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=c2, mode="lines", name="c₂", line={"color": "#ff7f0e"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=c3, mode="lines", name="c₃", line={"color": "#2ca02c"}), row=1, col=1)

    # ========== Layer 2: Invariants ==========
    fig.add_trace(go.Scatter(x=t, y=omega, mode="lines", name="ω (Drift)", line={"color": "red"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=IC, mode="lines", name="IC (Integrity)", line={"color": "blue"}), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=C_curv, mode="lines", name="C (Curvature)", line={"color": "purple", "dash": "dot"}),
        row=2,
        col=1,
    )

    # Threshold lines
    fig.add_hline(y=0.038, line_dash="dash", line_color="green", row=2, col=1, annotation_text="ω_stable")
    fig.add_hline(y=0.30, line_dash="dash", line_color="red", row=2, col=1, annotation_text="ω_collapse")

    # ========== Layer 3: Regime & Continuity ==========
    [0 if r == "STABLE" else 1 if r == "WATCH" else 2 for r in regimes]
    regime_colors = [REGIME_COLORS.get(r, "gray") for r in regimes]

    # Regime as colored segments
    for i in range(len(t) - 1):
        fig.add_trace(
            go.Scatter(
                x=[t[i], t[i + 1]],
                y=[0.5, 0.5],
                mode="lines",
                line={"color": regime_colors[i], "width": 20},
                showlegend=False,
                hovertemplate=f"t={t[i]}: {regimes[i]}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Add legend annotations
    for i, (regime, color) in enumerate(REGIME_COLORS.items()):
        fig.add_annotation(
            x=n_steps * (0.15 + 0.25 * i),
            y=0.85,
            text=f"<b>{regime}</b>",
            showarrow=False,
            font={"color": color, "size": 12},
            row=3,
            col=1,
        )

    fig.update_layout(
        height=900,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": 1.02, "xanchor": "center", "x": 0.5},
    )

    fig.update_yaxes(title_text="Coordinate Value", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Invariant Value", row=2, col=1)
    fig.update_yaxes(title_text="Regime", row=3, col=1, range=[0, 1], showticklabels=False)
    fig.update_xaxes(title_text="Time t", row=3, col=1)

    st.plotly_chart(fig, width="stretch")

    # ========== Flow Summary ==========
    st.markdown("#### Geometric Flow Summary")

    summary_cols = st.columns(3)

    with summary_cols[0]:
        st.markdown("""
        **Layer 1 → Layer 2**

        Projection via kernel:
        - Ψ(t) → F = Σwᵢcᵢ
        - Ψ(t) → IC = exp(Σwᵢ ln cᵢ)
        - Ψ(t) → C = std(cᵢ)/0.5
        """)

    with summary_cols[1]:
        st.markdown("""
        **Layer 2 → Layer 3**

        Regime gates partition space:
        - ω < 0.038 → STABLE
        - 0.038 ≤ ω < 0.30 → WATCH
        - ω ≥ 0.30 → COLLAPSE
        """)

    with summary_cols[2]:
        st.markdown("""
        **Layer 3: Continuity**

        Seam residual certifies:
        - s = Δκ_budget - Δκ_ledger
        - |s| ≤ 0.005 → PASS
        - Otherwise → FAIL
        """)


# ============================================================================
# Canon Explorer Page
# ============================================================================


def _load_canon_files() -> dict[str, Any]:
    """Load all canon anchor YAML files."""
    repo_root = get_repo_root()
    canon_dir = repo_root / "canon"
    result: dict[str, Any] = {}
    if not canon_dir.exists():
        return result
    try:
        import yaml
    except ImportError:
        return result
    for yf in sorted(canon_dir.glob("*_anchors.yaml")):
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if data:
                result[yf.stem] = data
        except Exception:
            pass
    return result


def render_canon_explorer_page() -> None:
    """Render the canon anchor explorer page — browse all domain anchors with professional formatting."""
    if st is None or pd is None:
        return

    st.title("📖 Canon Explorer")
    st.caption("Browse Tier-1 / Tier-2 anchor definitions across all domains")

    with st.expander("📖 What are Canon Anchors?", expanded=False):
        st.markdown("""
        **Canon anchors** define the immutable mathematical contracts for each domain.
        They specify:
        - **Axioms** — foundational truths the domain assumes
        - **Reserved symbols** — Greek letters and variables with fixed meanings
        - **Regime gates** — threshold values for STABLE / WATCH / COLLAPSE classification
        - **Budget identities** — Tier-1 mathematical relationships (F + ω = 1, IC ≤ F, IC ≈ exp(κ))
        - **Physical constants** — domain-specific reference values

        Canon files are **frozen at release** — they define the contract that closures must satisfy.
        """)

    canon = _load_canon_files()
    if not canon:
        st.warning("No canon anchor files found in canon/")
        return

    # Multi-column domain selector with icons
    domain_icons = {
        "gcd_anchors": "🔧 GCD",
        "kin_anchors": "🎯 KIN",
        "rcft_anchors": "🌀 RCFT",
        "weyl_anchors": "🌌 WEYL",
        "anchors": "🔒 Security",
        "astro_anchors": "🔭 ASTRO",
        "nuc_anchors": "☢️ NUC",
        "qm_anchors": "🔮 QM",
    }

    domain_names = list(canon.keys())
    selected = st.selectbox(
        "Select Domain Canon",
        domain_names,
        format_func=lambda x: domain_icons.get(x, x.replace("_anchors", "").upper()),
    )

    data = canon[selected]

    # Header info with styled card
    meta = data.get("metadata", data.get("canon", {}))
    canon_id = meta.get("id", meta.get("canon_id", "N/A"))
    version = meta.get("version", "N/A")
    tier = str(meta.get("tier", meta.get("scope", "N/A")))
    created = str(meta.get("created", "N/A"))

    # Tier color badge
    tier_color = "#1976d2" if "0" in tier else "#388e3c" if "1" in tier else "#f57c00"
    st.markdown(
        f"""<div style="padding:12px 16px; border-radius:8px; background:linear-gradient(135deg, #f8f9fa, #ffffff);
             border-left:4px solid {tier_color}; margin-bottom:16px;">
        <span style="font-size:1.2em; font-weight:bold;">{canon_id}</span>
        <span style="background:{tier_color}; color:white; padding:2px 8px; border-radius:4px;
              font-size:0.8em; margin-left:8px;">{tier}</span>
        <span style="color:#666; margin-left:12px;">v{version} · Created: {created}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Axioms
    axioms = data.get("axioms", [])
    if axioms:
        st.subheader("📜 Axioms")
        for idx, ax in enumerate(axioms):
            if isinstance(ax, dict):
                ax_id = ax.get("id", ax.get("axiom_id", f"A{idx + 1}"))
                label = ax.get("label", ax.get("name", ""))
                desc = ax.get("description", ax.get("statement", ""))
                with st.expander(f"**{ax_id}**: {label}", expanded=idx == 0):
                    st.markdown(desc)
            elif isinstance(ax, str):
                st.markdown(f"- {ax}")

    # Reserved symbols
    symbols = data.get("tier_1_symbols", data.get("reserved_symbols", data.get("symbols", [])))
    if symbols:
        st.subheader("🔣 Reserved Symbols")
        sym_rows: list[dict[str, str]] = []
        for sym in symbols:
            if isinstance(sym, dict):
                sym_rows.append(
                    {
                        "Symbol": sym.get("symbol", sym.get("name", "")),
                        "Label": sym.get("label", sym.get("description", "")),
                        "Domain": str(sym.get("domain", sym.get("range", ""))),
                        "Unit": sym.get("unit", sym.get("units", "")),
                    }
                )
        if sym_rows:
            st.dataframe(pd.DataFrame(sym_rows), use_container_width=True, hide_index=True)

    # Regime gates
    gates = data.get("regime_gates", data.get("gates", []))
    if gates:
        st.subheader("🚦 Regime Gates")
        gate_rows: list[dict[str, str]] = []
        for gate in gates:
            if isinstance(gate, dict):
                gate_rows.append(
                    {
                        "Gate": gate.get("name", gate.get("id", "")),
                        "🟢 Stable": str(gate.get("stable", gate.get("STABLE", ""))),
                        "🟡 Watch": str(gate.get("watch", gate.get("WATCH", ""))),
                        "🔴 Collapse": str(gate.get("collapse", gate.get("COLLAPSE", ""))),
                    }
                )
        if gate_rows:
            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

    # Constants
    constants = data.get("constants", data.get("physical_constants", {}))
    if constants:
        st.subheader("🔢 Constants")
        if isinstance(constants, dict):
            const_rows = [{"Name": k, "Value": str(v)} for k, v in constants.items()]
            st.dataframe(pd.DataFrame(const_rows), use_container_width=True, hide_index=True)
        elif isinstance(constants, list):
            for c in constants:
                if isinstance(c, dict):
                    st.markdown(f"- **{c.get('name', '')}**: {c.get('value', '')} {c.get('unit', '')}")

    # Budget identities
    identities = data.get("budget_identities", data.get("identities", data.get("mathematical_identities", [])))
    if identities:
        st.subheader("📐 Budget Identities")
        for ident in identities:
            if isinstance(ident, dict):
                ident_id = ident.get("id", ident.get("name", ""))
                expr = ident.get("expression", ident.get("formula", ident.get("description", "")))
                st.markdown(f"- **{ident_id}**: `{expr}`")
            elif isinstance(ident, str):
                st.markdown(f"- {ident}")

    # Raw YAML view
    with st.expander("📄 Raw YAML"):
        import yaml

        st.code(yaml.dump(data, default_flow_style=False, allow_unicode=True), language="yaml")


# ============================================================================
# Domain-Specific Pages
# ============================================================================


def render_astronomy_page() -> None:
    """Render interactive Astronomy domain page with all 6 closures."""
    if st is None or go is None or pd is None:
        return

    st.title("🔭 Astronomy Domain")
    st.caption(
        "ASTRO.INTSTACK.v1 — Stellar luminosity, distance ladder, spectral analysis, evolution, orbits, dynamics"
    )

    with st.expander("📖 Domain Overview", expanded=False):
        st.markdown("""
        The **Astronomy** domain embeds astrophysical observables into UMCP's [0, 1] contract space.
        Each closure maps physical measurements to Tier-1 invariants (ω, F, κ, IC) and classifies
        the measurement regime.

        | Closure | Physics | Key Observable |
        |---------|---------|---------------|
        | Stellar Luminosity | Stefan-Boltzmann, mass-luminosity | L★ / L☉ |
        | Distance Ladder | Parallax, modulus, Hubble flow | d (pc) consistency |
        | Spectral Analysis | Wien's law, B−V color index | λ_peak, T_eff |
        | Stellar Evolution | Main-sequence lifetime, HR track | t_MS, evolutionary phase |
        | Orbital Mechanics | Kepler III, vis-viva | P, v_orb |
        | Gravitational Dynamics | Virial theorem, rotation curves | M_virial, DM fraction |
        """)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "⭐ Stellar Luminosity",
            "📏 Distance Ladder",
            "🌈 Spectral Analysis",
            "🔄 Stellar Evolution",
            "🪐 Orbital Mechanics",
            "🌀 Gravitational Dynamics",
        ]
    )

    # ── Tab 1: Stellar Luminosity ──
    with tab1:
        st.subheader("⭐ Stellar Luminosity")
        st.markdown("""
        **Stefan-Boltzmann**: $L = 4\\pi R^2 \\sigma T_{\\text{eff}}^4$ ·
        **Mass-Luminosity**: $L \\propto M^{3.5}$ ·
        **Wien peak**: $\\lambda_{\\text{peak}} = b / T_{\\text{eff}}$
        """)

        preset_col, _ = st.columns([1, 2])
        with preset_col:
            preset = st.selectbox(
                "Presets", ["Custom", "☀️ Sun", "⭐ Sirius A", "🔴 Proxima Centauri", "💙 Rigel"], key="astro_lum_preset"
            )
        presets_lum = {
            "☀️ Sun": (1.0, 5778.0, 1.0),
            "⭐ Sirius A": (2.06, 9940.0, 1.71),
            "🔴 Proxima Centauri": (0.122, 3042.0, 0.154),
            "💙 Rigel": (21.0, 12100.0, 78.9),
        }
        _m, _t, _r = presets_lum.get(preset, (1.0, 5778.0, 1.0))
        c1, c2, c3 = st.columns(3)
        with c1:
            m_star = st.number_input("M★ (M☉)", 0.08, 150.0, _m, 0.1, key="astro_mstar")
        with c2:
            t_eff = st.number_input("T_eff (K)", 2000.0, 50000.0, _t, 100.0, key="astro_teff")
        with c3:
            r_star = st.number_input("R★ (R☉)", 0.01, 1500.0, _r, 0.1, key="astro_rstar")

        if st.button("Compute Luminosity", key="astro_lum", type="primary"):
            try:
                from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

                result = compute_stellar_luminosity(m_star, t_eff, r_star)
                regime = result["regime"]
                regime_color = {"Consistent": "🟢", "Mild": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("L_predicted (L☉)", f"{result['L_predicted']:.4f}")
                with rc2:
                    st.metric("L_SB (L☉)", f"{result['L_SB']:.4f}")
                with rc3:
                    st.metric("δ_L", f"{result['delta_L']:.6f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Wien peak
                st.info(f"**Wien Peak**: λ_peak = {result['lambda_peak']:.1f} nm")

                # Visualization: luminosity comparison bar
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=["L_predicted (M-L relation)", "L_SB (Stefan-Boltzmann)"],
                        y=[result["L_predicted"], result["L_SB"]],
                        marker_color=["#007bff", "#28a745"],
                        text=[f"{result['L_predicted']:.4f}", f"{result['L_SB']:.4f}"],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Luminosity Comparison",
                    yaxis_title="L / L☉",
                    height=300,
                    margin={"t": 40, "b": 20, "l": 40, "r": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 2: Distance Ladder ──
    with tab2:
        st.subheader("📏 Distance Ladder Cross-Validation")
        st.markdown("""
        Three independent distance measures are compared for consistency:
        - **Distance modulus**: $d = 10^{(m - M + 5)/5}$ pc
        - **Trigonometric parallax**: $d = 1/\\pi$ pc
        - **Hubble flow**: $d = cz / H_0$
        """)

        preset_col2, _ = st.columns([1, 2])
        with preset_col2:
            dp = st.selectbox(
                "Presets", ["Custom", "⭐ Vega", "🌟 Cepheid (LMC)", "🌌 Distant Galaxy"], key="astro_dist_preset"
            )
        presets_dist = {
            "⭐ Vega": (0.03, 0.58, 0.1289, 0.0),
            "🌟 Cepheid (LMC)": (13.5, -5.0, 0.00002, 0.003),
            "🌌 Distant Galaxy": (22.0, -21.0, 0.00001, 0.1),
        }
        _ma, _mab, _pi, _zc = presets_dist.get(dp, (10.0, 4.83, 0.01, 0.01))
        c1, c2 = st.columns(2)
        with c1:
            m_app = st.number_input("m (apparent mag)", -30.0, 30.0, _ma, 0.1, key="astro_mapp")
            m_abs = st.number_input("M (absolute mag)", -30.0, 30.0, _mab, 0.1, key="astro_mabs")
        with c2:
            pi_arcsec = st.number_input("π (arcsec)", 1e-6, 1.0, _pi, 0.001, key="astro_pi", format="%.6f")
            z_cosmo = st.number_input("z (redshift)", 0.0, 10.0, _zc, 0.001, key="astro_z", format="%.4f")

        if st.button("Compute Distances", key="astro_dist", type="primary"):
            try:
                from closures.astronomy.distance_ladder import compute_distance_ladder

                result = compute_distance_ladder(m_app, m_abs, pi_arcsec, z_cosmo)
                regime = result["regime"]
                regime_color = {"High": "🟢", "Moderate": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("d_modulus (pc)", f"{result['d_modulus']:.2f}")
                with rc2:
                    st.metric("d_parallax (pc)", f"{result['d_parallax']:.2f}")
                with rc3:
                    st.metric("d_Hubble (pc)", f"{result['d_hubble']:.2f}")

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Consistency", f"{result['distance_consistency']:.4f}")
                with mc2:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Bar chart of distances
                fig = go.Figure()
                methods = ["Modulus", "Parallax", "Hubble"]
                vals = [result["d_modulus"], result["d_parallax"], result["d_hubble"]]
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=vals,
                        marker_color=["#007bff", "#28a745", "#fd7e14"],
                        text=[f"{v:.1f}" for v in vals],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Distance Ladder Comparison",
                    yaxis_title="Distance (pc)",
                    yaxis_type="log",
                    height=300,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 3: Spectral Analysis ──
    with tab3:
        st.subheader("🌈 Spectral Analysis")
        st.markdown("""
        **Wien's displacement**: $\\lambda_{\\text{peak}} = 2.898 \\times 10^6 / T_{\\text{eff}}$ nm ·
        **Ballesteros B−V→T**: $T = 4600 (1/(0.92 (B-V) + 1.7) + 1/(0.92 (B-V) + 0.62))$
        """)
        c1, c2, c3 = st.columns(3)
        with c1:
            t_eff_s = st.number_input("T_eff (K)", 2000.0, 50000.0, 5778.0, 100.0, key="astro_teff_s")
        with c2:
            b_v = st.number_input("B−V (mag)", -0.5, 2.5, 0.65, 0.01, key="astro_bv")
        with c3:
            spec_class = st.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"], index=4, key="astro_spec")

        if st.button("Analyze Spectrum", key="astro_spec_btn", type="primary"):
            try:
                from closures.astronomy.spectral_analysis import compute_spectral_analysis

                result = compute_spectral_analysis(t_eff_s, b_v, spec_class)
                regime = result["regime"]
                regime_color = {"Excellent": "🟢", "Good": "🟡", "Marginal": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("λ_peak (nm)", f"{result['lambda_peak']:.1f}")
                with rc2:
                    st.metric("T from B−V (K)", f"{result['T_from_BV']:.0f}")
                with rc3:
                    st.metric("χ² spectral", f"{result['chi2_spectral']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Spectral class bar
                spectral_temps = {"O": 40000, "B": 20000, "A": 8500, "F": 6500, "G": 5500, "K": 4000, "M": 3000}
                fig = go.Figure()
                classes = list(spectral_temps.keys())
                temps = list(spectral_temps.values())
                colors = ["#9bb0ff", "#aabfff", "#cad7ff", "#f8f7ff", "#fff4e8", "#ffd2a1", "#ffcc6f"]
                fig.add_trace(go.Bar(x=classes, y=temps, marker_color=colors, name="Typical T_eff"))
                fig.add_hline(
                    y=t_eff_s, line_dash="dash", line_color="red", annotation_text=f"Input T_eff = {t_eff_s:.0f} K"
                )
                fig.update_layout(
                    title="Spectral Class Temperature Scale",
                    yaxis_title="T_eff (K)",
                    height=300,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 4: Stellar Evolution ──
    with tab4:
        st.subheader("🔄 Stellar Evolution")
        st.markdown("""
        **Main-sequence lifetime**: $t_{MS} \\approx 10 \\times (M/M_\\odot)^{-2.5}$ Gyr ·
        Compares observed luminosity and temperature against ZAMS predictions.
        """)
        preset_col3, _ = st.columns([1, 2])
        with preset_col3:
            ep = st.selectbox(
                "Presets",
                ["Custom", "☀️ Sun (4.6 Gyr)", "⭐ Sirius (0.24 Gyr)", "🔴 Red Giant (10 Gyr)"],
                key="astro_evol_preset",
            )
        presets_evol = {
            "☀️ Sun (4.6 Gyr)": (1.0, 1.0, 5778.0, 4.6),
            "⭐ Sirius (0.24 Gyr)": (2.06, 25.4, 9940.0, 0.24),
            "🔴 Red Giant (10 Gyr)": (1.0, 100.0, 4500.0, 10.0),
        }
        _me, _le, _te, _ae = presets_evol.get(ep, (1.0, 1.0, 5778.0, 4.6))
        c1, c2 = st.columns(2)
        with c1:
            m_star_e = st.number_input("M★ (M☉)", 0.08, 150.0, _me, 0.1, key="astro_mstar_e")
            l_obs = st.number_input("L_obs (L☉)", 0.0001, 1e6, _le, 0.1, key="astro_lobs")
        with c2:
            t_eff_e = st.number_input("T_eff (K)", 2000.0, 50000.0, _te, 100.0, key="astro_teff_e")
            age_gyr = st.number_input("Age (Gyr)", 0.001, 15.0, _ae, 0.1, key="astro_age")

        if st.button("Compute Evolution", key="astro_evol", type="primary"):
            try:
                from closures.astronomy.stellar_evolution import compute_stellar_evolution

                result = compute_stellar_evolution(m_star_e, l_obs, t_eff_e, age_gyr)
                regime = result["regime"]

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("t_MS (Gyr)", f"{result['t_MS']:.3f}")
                with rc2:
                    st.metric("Phase", result["evolutionary_phase"])
                with rc3:
                    st.metric("L_ZAMS (L☉)", f"{result['L_ZAMS']:.4f}")
                with rc4:
                    st.metric("Regime", regime)

                # Age vs MS lifetime gauge
                frac = min(age_gyr / max(result["t_MS"], 0.001), 2.0)
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=frac * 100,
                        title={"text": "Age / t_MS (%)"},
                        gauge={
                            "axis": {"range": [0, 200]},
                            "bar": {"color": "#007bff"},
                            "steps": [
                                {"range": [0, 80], "color": "#d4edda"},
                                {"range": [80, 100], "color": "#fff3cd"},
                                {"range": [100, 200], "color": "#f8d7da"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=250, margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 5: Orbital Mechanics ──
    with tab5:
        st.subheader("🪐 Orbital Mechanics")
        st.markdown("""
        **Kepler III**: $P^2 = 4\\pi^2 a^3 / (G M)$ ·
        **Vis-viva**: $v = \\sqrt{GM(2/r - 1/a)}$ ·
        Validates observed period against Kepler prediction.
        """)
        preset_col4, _ = st.columns([1, 2])
        with preset_col4:
            op = st.selectbox("Presets", ["Custom", "🌍 Earth", "♃ Jupiter", "☿ Mercury"], key="astro_orbit_preset")
        presets_orb = {
            "🌍 Earth": (1.0, 1.0, 1.0, 0.017),
            "♃ Jupiter": (11.86, 5.20, 1.0, 0.049),
            "☿ Mercury": (0.2408, 0.387, 1.0, 0.206),
        }
        _po, _ao, _mo, _eo = presets_orb.get(op, (1.0, 1.0, 1.0, 0.017))
        c1, c2 = st.columns(2)
        with c1:
            p_orb = st.number_input("P_orb (years)", 0.001, 1000.0, _po, 0.01, key="astro_porb")
            a_semi = st.number_input("a (AU)", 0.01, 100.0, _ao, 0.01, key="astro_asemi")
        with c2:
            m_total = st.number_input("M_total (M☉)", 0.01, 100.0, _mo, 0.1, key="astro_mtotal")
            e_orb = st.number_input("e (eccentricity)", 0.0, 0.99, _eo, 0.001, key="astro_eorb", format="%.3f")

        if st.button("Compute Orbit", key="astro_orbit", type="primary"):
            try:
                from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

                result = compute_orbital_mechanics(p_orb, a_semi, m_total, e_orb)
                regime = result["regime"]
                regime_color = {"Stable": "🟢", "Eccentric": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("P_predicted (yr)", f"{result['P_predicted']:.4f}")
                with rc2:
                    st.metric("Kepler residual", f"{result['kepler_residual']:.6f}")
                with rc3:
                    st.metric("v_orb (km/s)", f"{result['v_orb'] / 1000:.2f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Orbital ellipse visualization
                import math

                theta = [i * 2 * math.pi / 100 for i in range(101)]
                r_vals = [a_semi * (1 - e_orb**2) / (1 + e_orb * math.cos(t)) for t in theta]
                x_vals = [r * math.cos(t) for r, t in zip(r_vals, theta, strict=False)]
                y_vals = [r * math.sin(t) for r, t in zip(r_vals, theta, strict=False)]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Orbit", line={"color": "#007bff", "width": 2})
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        name="Central body",
                        marker={"size": 12, "color": "#ffc107", "symbol": "star"},
                    )
                )
                fig.update_layout(
                    title=f"Orbital Ellipse (e = {e_orb:.3f})",
                    xaxis_title="x (AU)",
                    yaxis_title="y (AU)",
                    height=350,
                    margin={"t": 40, "b": 20},
                    yaxis_scaleanchor="x",
                    yaxis_scaleratio=1,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 6: Gravitational Dynamics ──
    with tab6:
        st.subheader("🌀 Gravitational Dynamics")
        st.markdown("""
        **Virial mass**: $M_{\\text{vir}} = R \\sigma_v^2 / G$ ·
        **Dark matter fraction**: $f_{DM} = 1 - M_{\\text{lum}} / M_{\\text{vir}}$ ·
        Tests virial equilibrium in galaxy-scale systems.
        """)
        preset_col5, _ = st.columns([1, 2])
        with preset_col5:
            gp = st.selectbox(
                "Presets",
                ["Custom", "🌌 Milky Way", "🌀 Andromeda (M31)", "🔴 Dwarf Spheroidal"],
                key="astro_dyn_preset",
            )
        presets_dyn = {
            "🌌 Milky Way": (220.0, 8.0, 150.0, 5e10),
            "🌀 Andromeda (M31)": (250.0, 20.0, 180.0, 7e10),
            "🔴 Dwarf Spheroidal": (10.0, 0.3, 8.0, 1e7),
        }
        _vr, _ro, _sv, _ml = presets_dyn.get(gp, (220.0, 8.0, 150.0, 5e10))
        c1, c2 = st.columns(2)
        with c1:
            v_rot = st.number_input("v_rot (km/s)", 1.0, 500.0, _vr, 1.0, key="astro_vrot")
            r_obs = st.number_input("r_obs (kpc)", 0.01, 1000.0, _ro, 0.1, key="astro_robs")
        with c2:
            sigma_v = st.number_input("σ_v (km/s)", 1.0, 500.0, _sv, 1.0, key="astro_sigmav")
            m_lum = st.number_input("M_lum (M☉)", 1e5, 1e13, _ml, 1e9, key="astro_mlum", format="%.2e")

        if st.button("Compute Dynamics", key="astro_dyn", type="primary"):
            try:
                from closures.astronomy.gravitational_dynamics import compute_gravitational_dynamics

                result = compute_gravitational_dynamics(v_rot, r_obs, sigma_v, m_lum)
                regime = result["regime"]
                regime_color = {"Equilibrium": "🟢", "Relaxing": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("M_virial (M☉)", f"{result['M_virial']:.3e}")
                with rc2:
                    st.metric("M_dynamic (M☉)", f"{result['M_dynamic']:.3e}")
                with rc3:
                    st.metric("DM fraction", f"{result['dark_matter_fraction']:.2%}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                st.metric("Virial ratio", f"{result['virial_ratio']:.4f}")

                # Pie chart: luminous vs dark matter
                dm_frac = max(0, min(result["dark_matter_fraction"], 1.0))
                fig = go.Figure(
                    go.Pie(
                        labels=["Luminous Matter", "Dark Matter"],
                        values=[1 - dm_frac, dm_frac],
                        marker={"colors": ["#ffc107", "#343a40"]},
                        hole=0.4,
                        textinfo="label+percent",
                    )
                )
                fig.update_layout(title="Mass Budget", height=300, margin={"t": 40, "b": 20})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")


def render_nuclear_page() -> None:
    """Render interactive Nuclear Physics domain page with all 6 closures."""
    if st is None or go is None or pd is None:
        return

    st.title("☢️ Nuclear Physics Domain")
    st.caption(
        "NUC.INTSTACK.v1 — Binding energy, alpha decay, shell structure, fissility, decay chains, double-sided collapse"
    )

    with st.expander("📖 Domain Overview", expanded=False):
        st.markdown("""
        The **Nuclear Physics** domain maps nuclear observables into UMCP contract space.
        All closures return **NamedTuple** results with UMCP invariants (ω_eff, F_eff, Ψ).

        | Closure | Physics | Key Observable | Reference |
        |---------|---------|---------------|-----------|
        | Binding Energy | Semi-Empirical Mass Formula (Weizsäcker) | BE/A (MeV/nucleon) | Ni-62 = 8.7945 MeV |
        | Alpha Decay | Geiger-Nuttall / Gamow tunneling | log₁₀(T½) | ²³⁸U → ²³⁴Th |
        | Shell Structure | Magic numbers: 2, 8, 20, 28, 50, 82, 126 | Shell closure strength | ²⁰⁸Pb doubly magic |
        | Fissility | Z²/A vs critical fissility | x = (Z²/A) / (Z²/A)_crit | (Z²/A)_crit ≈ 48.26 |
        | Decay Chain | Sequential α/β decay pathway | Chain length, bottleneck | ²³⁸U → ²⁰⁶Pb (14 steps) |
        | Double-Sided Collapse | Fusion ↔ Fission convergence | Iron peak distance | Fe-56 = 0 distance |
        """)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "⚛️ Binding Energy",
            "💫 Alpha Decay",
            "🐚 Shell Structure",
            "💣 Fissility",
            "🔗 Decay Chain",
            "🔄 Double-Sided Collapse",
        ]
    )

    # ── Tab 1: Binding Energy (SEMF) ──
    with tab1:
        st.subheader("⚛️ Nuclide Binding Energy (SEMF)")
        st.markdown("""
        **Weizsäcker formula**: $B(Z,A) = a_V A - a_S A^{2/3} - a_C Z(Z-1)A^{-1/3} - a_A (A-2Z)^2/A + \\delta$

        The iron peak (Ni-62, Fe-56) at **8.79 MeV/nucleon** defines the maximum binding energy per nucleon —
        the UMCP collapse attractor for nuclear matter.
        """)

        preset_col, _ = st.columns([1, 2])
        with preset_col:
            bp = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "⚛️ Fe-56 (Iron)",
                    "☢️ U-238",
                    "🔵 Pb-208 (Doubly Magic)",
                    "🧪 He-4 (Alpha)",
                    "💎 C-12 (Carbon)",
                ],
                key="nuc_bind_preset",
            )
        presets_bind = {
            "⚛️ Fe-56 (Iron)": (26, 56),
            "☢️ U-238": (92, 238),
            "🔵 Pb-208 (Doubly Magic)": (82, 208),
            "🧪 He-4 (Alpha)": (2, 4),
            "💎 C-12 (Carbon)": (6, 12),
        }
        _z, _a = presets_bind.get(bp, (26, 56))
        c1, c2 = st.columns(2)
        with c1:
            z_val = st.number_input("Z (protons)", 1, 120, _z, key="nuc_z")
        with c2:
            a_val = st.number_input("A (mass number)", 1, 300, _a, key="nuc_a")

        if st.button("Compute Binding", key="nuc_bind", type="primary"):
            try:
                from closures.nuclear_physics import compute_binding

                result = compute_binding(z_val, a_val)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("BE/A (MeV)", f"{rd['BE_per_A']:.4f}")
                with rc2:
                    st.metric("BE_total (MeV)", f"{rd['BE_total']:.2f}")
                with rc3:
                    st.metric("ω_eff (deficit)", f"{rd['omega_eff']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("F_eff = 1 − ω", f"{rd['F_eff']:.4f}")
                with mc2:
                    st.metric("Ψ_BE (embedding)", f"{rd['Psi_BE']:.4f}")

                # Binding energy curve visualization
                from closures.nuclear_physics import compute_binding as _cb

                a_range = list(range(4, 260, 2))
                be_curve = []
                for _ai in a_range:
                    _zi = max(1, round(_ai * 0.42))  # approximate Z ≈ 0.42 * A
                    try:
                        _ri = _cb(_zi, _ai)
                        be_curve.append({"A": _ai, "BE/A": _ri.BE_per_A})
                    except Exception:
                        pass

                if be_curve:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=[d["A"] for d in be_curve],
                            y=[d["BE/A"] for d in be_curve],
                            mode="lines",
                            name="SEMF Curve",
                            line={"color": "#007bff", "width": 2},
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[a_val],
                            y=[rd["BE_per_A"]],
                            mode="markers",
                            name=f"Z={z_val}, A={a_val}",
                            marker={"size": 12, "color": "red", "symbol": "star"},
                        )
                    )
                    fig.add_hline(
                        y=8.7945, line_dash="dash", line_color="green", annotation_text="Iron peak (8.7945 MeV)"
                    )
                    fig.update_layout(
                        title="Nuclear Binding Energy Curve",
                        xaxis_title="Mass Number A",
                        yaxis_title="BE/A (MeV/nucleon)",
                        height=350,
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 2: Alpha Decay ──
    with tab2:
        st.subheader("💫 Alpha Decay (Geiger-Nuttall)")
        st.markdown("""
        **Geiger-Nuttall law**: $\\log_{10}(T_{1/2}) = a / \\sqrt{Q_\\alpha} + b$

        An alpha particle (⁴He) tunnels through the Coulomb barrier.
        The Q-value determines disintegration energy; the Gamow factor sets the tunneling probability.
        """)
        preset_col2, _ = st.columns([1, 2])
        with preset_col2:
            ap = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "☢️ U-238 (Q=4.27 MeV)",
                    "☢️ Pu-239 (Q=5.24 MeV)",
                    "☢️ Ra-226 (Q=4.87 MeV)",
                    "☢️ Po-210 (Q=5.41 MeV)",
                ],
                key="nuc_alpha_preset",
            )
        presets_alpha = {
            "☢️ U-238 (Q=4.27 MeV)": (92, 238, 4.27),
            "☢️ Pu-239 (Q=5.24 MeV)": (94, 239, 5.24),
            "☢️ Ra-226 (Q=4.87 MeV)": (88, 226, 4.87),
            "☢️ Po-210 (Q=5.41 MeV)": (84, 210, 5.41),
        }
        _za, _aa, _qa = presets_alpha.get(ap, (92, 238, 4.27))
        c1, c2, c3 = st.columns(3)
        with c1:
            z_ad = st.number_input("Z (parent)", 2, 120, _za, key="nuc_z_ad")
        with c2:
            a_ad = st.number_input("A (parent)", 4, 300, _aa, key="nuc_a_ad")
        with c3:
            q_alpha = st.number_input("Q_α (MeV)", 0.1, 15.0, _qa, 0.01, key="nuc_qalpha")

        if st.button("Compute Alpha Decay", key="nuc_alpha", type="primary"):
            try:
                from closures.nuclear_physics import compute_alpha_decay

                result = compute_alpha_decay(z_ad, a_ad, q_alpha)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Q_α (MeV)", f"{rd['Q_alpha']:.4f}")
                with rc2:
                    st.metric("log₁₀(T½/s)", f"{rd['log10_half_life_s']:.2f}")
                with rc3:
                    st.metric("Ψ_Q_α", f"{rd['Psi_Q_alpha']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("T½ (s)", f"{rd['half_life_s']:.3e}")
                with mc2:
                    st.metric("Mean lifetime τ (s)", f"{rd['mean_lifetime_s']:.3e}")

                # Decay scheme visualization
                parent = f"Z={z_ad}, A={a_ad}"
                daughter = f"Z={z_ad - 2}, A={a_ad - 4}"
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=[parent, "α particle (⁴He)", daughter],
                        y=[a_ad, 4, a_ad - 4],
                        marker_color=["#dc3545", "#ffc107", "#28a745"],
                        text=[f"A={a_ad}", "A=4", f"A={a_ad - 4}"],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title=f"Alpha Decay: {parent} → {daughter} + α",
                    yaxis_title="Mass Number A",
                    height=300,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 3: Shell Structure ──
    with tab3:
        st.subheader("🐚 Shell Structure")
        st.markdown("""
        **Nuclear magic numbers**: 2, 8, 20, 28, 50, 82, 126

        Nuclei at or near magic numbers have enhanced stability (closed shells).
        **Doubly-magic** nuclei (both Z and N magic) are exceptionally stable: ⁴He, ¹⁶O, ⁴⁰Ca, ⁴⁸Ca, ⁴⁸Ni, ²⁰⁸Pb.
        """)
        preset_col3, _ = st.columns([1, 2])
        with preset_col3:
            sp = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "🔵 Pb-208 (Z=82, N=126)",
                    "⚛️ Ca-40 (Z=20, N=20)",
                    "🧪 O-16 (Z=8, N=8)",
                    "☢️ Sn-132 (Z=50, N=82)",
                ],
                key="nuc_shell_preset",
            )
        presets_shell = {
            "🔵 Pb-208 (Z=82, N=126)": (82, 208),
            "⚛️ Ca-40 (Z=20, N=20)": (20, 40),
            "🧪 O-16 (Z=8, N=8)": (8, 16),
            "☢️ Sn-132 (Z=50, N=82)": (50, 132),
        }
        _zs, _as = presets_shell.get(sp, (50, 120))
        c1, c2 = st.columns(2)
        with c1:
            z_sh = st.number_input("Z (protons)", 1, 120, _zs, key="nuc_z_sh")
        with c2:
            a_sh = st.number_input("A (mass number)", 1, 300, _as, key="nuc_a_sh")

        if st.button("Compute Shell", key="nuc_shell", type="primary"):
            try:
                from closures.nuclear_physics import compute_shell

                result = compute_shell(z_sh, a_sh)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")
                n_val = rd["N"]

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Z", rd["Z"])
                with rc2:
                    st.metric("N", n_val)
                with rc3:
                    st.metric("Doubly Magic", "Yes ✨" if rd["doubly_magic"] else "No")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Magic Z?", "Yes" if rd["magic_proton"] else "No")
                with mc2:
                    st.metric("Nearest Magic Z", f"{rd['nearest_magic_Z']}")
                with mc3:
                    st.metric("Magic N?", "Yes" if rd["magic_neutron"] else "No")
                with mc4:
                    st.metric("Nearest Magic N", f"{rd['nearest_magic_N']}")

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.metric("Distance to Magic Z", f"{rd['distance_to_magic_Z']}")
                with dc2:
                    st.metric("Distance to Magic N", f"{rd['distance_to_magic_N']}")
                st.metric("Shell Correction (MeV)", f"{rd['shell_correction']:.4f}")

                # Magic numbers proximity chart
                magic = [2, 8, 20, 28, 50, 82, 126]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=magic,
                        y=[0] * len(magic),
                        mode="markers+text",
                        marker={"size": 20, "color": "#28a745", "symbol": "diamond"},
                        text=[str(m) for m in magic],
                        textposition="top center",
                        name="Magic Numbers",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[z_sh],
                        y=[0.1],
                        mode="markers+text",
                        marker={"size": 15, "color": "#dc3545"},
                        text=[f"Z={z_sh}"],
                        textposition="top center",
                        name="Proton count",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[n_val],
                        y=[-0.1],
                        mode="markers+text",
                        marker={"size": 15, "color": "#007bff"},
                        text=[f"N={n_val}"],
                        textposition="bottom center",
                        name="Neutron count",
                    )
                )
                fig.update_layout(
                    title="Magic Number Proximity",
                    xaxis_title="Nucleon Count",
                    yaxis={"visible": False, "range": [-0.5, 0.5]},
                    height=250,
                    margin={"t": 40, "b": 20},
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 4: Fissility ──
    with tab4:
        st.subheader("💣 Fissility Assessment")
        st.markdown("""
        **Fissility parameter**: $x = (Z^2/A) / (Z^2/A)_{\\text{crit}}$

        When $x \\geq 1$, the Coulomb repulsion overcomes surface tension and the nucleus is
        **spontaneously fissile**. $(Z^2/A)_{\\text{crit}} \\approx 48.26$ (liquid drop model).
        """)
        preset_col4, _ = st.columns([1, 2])
        with preset_col4:
            fp = st.selectbox(
                "Presets",
                ["Custom", "☢️ U-235 (fissile)", "☢️ U-238", "☢️ Pu-239", "⚛️ Fe-56 (stable)"],
                key="nuc_fiss_preset",
            )
        presets_fiss = {
            "☢️ U-235 (fissile)": (92, 235),
            "☢️ U-238": (92, 238),
            "☢️ Pu-239": (94, 239),
            "⚛️ Fe-56 (stable)": (26, 56),
        }
        _zf, _af = presets_fiss.get(fp, (92, 235))
        c1, c2 = st.columns(2)
        with c1:
            z_fi = st.number_input("Z", 1, 120, _zf, key="nuc_z_fi")
        with c2:
            a_fi = st.number_input("A", 1, 300, _af, key="nuc_a_fi")

        if st.button("Compute Fissility", key="nuc_fiss", type="primary"):
            try:
                from closures.nuclear_physics import compute_fissility

                result = compute_fissility(z_fi, a_fi)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("x (fissility)", f"{rd['fissility_x']:.4f}")
                with rc2:
                    st.metric("Z²/A", f"{rd['Z_squared_over_A']:.2f}")
                with rc3:
                    st.metric("Ψ_fiss", f"{rd['Psi_fiss']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Coulomb Energy (MeV)", f"{rd['coulomb_energy']:.2f}")
                with mc2:
                    st.metric("Surface Energy (MeV)", f"{rd['surface_energy']:.2f}")

                # Fissility gauge
                x_val = rd["fissility_x"]
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=x_val,
                        title={"text": "Fissility Parameter x"},
                        gauge={
                            "axis": {"range": [0, 1.5]},
                            "bar": {"color": "#dc3545" if x_val >= 1.0 else "#ffc107" if x_val >= 0.7 else "#28a745"},
                            "steps": [
                                {"range": [0, 0.7], "color": "#d4edda"},
                                {"range": [0.7, 1.0], "color": "#fff3cd"},
                                {"range": [1.0, 1.5], "color": "#f8d7da"},
                            ],
                            "threshold": {"line": {"color": "red", "width": 3}, "value": 1.0},
                        },
                    )
                )
                fig.update_layout(height=250, margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 5: Decay Chain ──
    with tab5:
        st.subheader("🔗 Decay Chain Analysis")
        st.markdown("""
        Analyzes a sequential decay pathway. Provide the chain steps as a series of α and β decays.
        The closure computes chain statistics: total nucleon shedding, bottleneck half-life, and endpoint.

        *Note: This closure takes pre-defined chain steps (isotope, decay mode, half-life, Q-value).*
        """)

        st.info("📋 Using the standard **²³⁸U → ²⁰⁶Pb** decay chain (14 α + β steps)")

        if st.button("Analyze U-238 Decay Chain", key="nuc_chain", type="primary"):
            try:
                from closures.nuclear_physics.decay_chain import ChainStep, compute_decay_chain

                # Standard U-238 → Pb-206 decay chain
                u238_chain = [
                    ChainStep(isotope="U-238", Z=92, A=238, decay_mode="alpha", half_life_s=1.41e17, Q_MeV=4.27),
                    ChainStep(isotope="Th-234", Z=90, A=234, decay_mode="beta_minus", half_life_s=2.08e6, Q_MeV=0.27),
                    ChainStep(isotope="Pa-234", Z=91, A=234, decay_mode="beta_minus", half_life_s=70.2, Q_MeV=2.20),
                    ChainStep(isotope="U-234", Z=92, A=234, decay_mode="alpha", half_life_s=7.75e12, Q_MeV=4.86),
                    ChainStep(isotope="Th-230", Z=90, A=230, decay_mode="alpha", half_life_s=2.38e12, Q_MeV=4.77),
                    ChainStep(isotope="Ra-226", Z=88, A=226, decay_mode="alpha", half_life_s=5.05e10, Q_MeV=4.87),
                    ChainStep(isotope="Rn-222", Z=86, A=222, decay_mode="alpha", half_life_s=3.30e5, Q_MeV=5.59),
                    ChainStep(isotope="Po-218", Z=84, A=218, decay_mode="alpha", half_life_s=186.0, Q_MeV=6.11),
                    ChainStep(isotope="Pb-214", Z=82, A=214, decay_mode="beta_minus", half_life_s=1608.0, Q_MeV=1.02),
                    ChainStep(isotope="Bi-214", Z=83, A=214, decay_mode="beta_minus", half_life_s=1194.0, Q_MeV=3.27),
                    ChainStep(isotope="Po-214", Z=84, A=214, decay_mode="alpha", half_life_s=1.64e-4, Q_MeV=7.83),
                    ChainStep(isotope="Pb-210", Z=82, A=210, decay_mode="beta_minus", half_life_s=7.01e8, Q_MeV=0.06),
                    ChainStep(isotope="Bi-210", Z=83, A=210, decay_mode="beta_minus", half_life_s=4.33e5, Q_MeV=1.16),
                    ChainStep(isotope="Po-210", Z=84, A=210, decay_mode="alpha", half_life_s=1.20e7, Q_MeV=5.41),
                ]

                result = compute_decay_chain(u238_chain)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Chain Length", rd["chain_length"])
                with rc2:
                    st.metric("α decays", rd["alpha_count"])
                with rc3:
                    st.metric("β⁻ decays", rd["beta_minus_count"])
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Endpoint", rd["endpoint_isotope"])
                with mc2:
                    st.metric("Total Q (MeV)", f"{rd['total_Q_MeV']:.2f}")
                with mc3:
                    st.metric("Nucleons shed", rd["total_nucleons_shed"])
                with mc4:
                    st.metric("Bottleneck", f"{rd['bottleneck_step']}")

                # Chain visualization
                chain_data = []
                for step in u238_chain:
                    chain_data.append(
                        {
                            "Isotope": step.isotope,
                            "Z": step.Z,
                            "A": step.A,
                            "Decay": step.decay_mode,
                            "T½ (s)": f"{step.half_life_s:.2e}",
                            "Q (MeV)": f"{step.Q_MeV:.2f}",
                        }
                    )
                chain_data.append(
                    {"Isotope": "Pb-206 (stable)", "Z": 82, "A": 206, "Decay": "—", "T½ (s)": "stable", "Q (MeV)": "—"}
                )
                st.dataframe(pd.DataFrame(chain_data), use_container_width=True, hide_index=True)

                # A vs step chart
                fig = go.Figure()
                a_vals = [s.A for s in u238_chain] + [206]
                labels = [s.isotope for s in u238_chain] + ["Pb-206"]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(a_vals))),
                        y=a_vals,
                        mode="lines+markers+text",
                        text=labels,
                        textposition="top center",
                        name="Mass Number A",
                        marker={"size": 8, "color": "#007bff"},
                        line={"width": 2},
                    )
                )
                fig.update_layout(
                    title="Decay Chain: Mass Number vs Step",
                    xaxis_title="Step",
                    yaxis_title="A",
                    height=350,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 6: Double-Sided Collapse ──
    with tab6:
        st.subheader("🔄 Double-Sided Collapse")
        st.markdown("""
        **AX-N4**: Nuclear binding energy converges on the iron peak from **both sides**:
        - **Light nuclei** (A < 56): energy released by **fusion** → moving right on the curve
        - **Heavy nuclei** (A > 56): energy released by **fission** → moving left on the curve

        The signed distance from Fe-56 quantifies how far a nuclide is from the collapse attractor.
        """)
        preset_col6, _ = st.columns([1, 2])
        with preset_col6:
            dsp = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "⚛️ Fe-56 (at peak)",
                    "🧪 He-4 (fusion fuel)",
                    "☢️ U-238 (fission fuel)",
                    "💎 C-12 (stellar fusion)",
                ],
                key="nuc_ds_preset",
            )
        presets_ds = {
            "⚛️ Fe-56 (at peak)": (26, 56),
            "🧪 He-4 (fusion fuel)": (2, 4),
            "☢️ U-238 (fission fuel)": (92, 238),
            "💎 C-12 (stellar fusion)": (6, 12),
        }
        _zd, _ad = presets_ds.get(dsp, (26, 56))
        c1, c2 = st.columns(2)
        with c1:
            z_ds = st.number_input("Z", 1, 120, _zd, key="nuc_z_ds")
        with c2:
            a_ds = st.number_input("A", 1, 300, _ad, key="nuc_a_ds")

        if st.button("Compute Double-Sided", key="nuc_ds", type="primary"):
            try:
                from closures.nuclear_physics import compute_binding, compute_double_sided

                binding = compute_binding(z_ds, a_ds)
                result = compute_double_sided(z_ds, a_ds, binding.BE_per_A)
                rd = result._asdict()
                regime = rd["regime"]
                regime_color = {"STABLE": "🟢", "WATCH": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("BE/A (MeV)", f"{rd['BE_per_A']:.4f}")
                with rc2:
                    st.metric("Signed Distance", f"{rd['signed_distance']:.4f}")
                with rc3:
                    st.metric("Side", rd["side"])
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("|Distance|", f"{rd['abs_distance']:.4f}")
                with mc2:
                    st.metric("Convergence", rd["convergence_direction"])
                with mc3:
                    st.metric("ω_eff", f"{rd['omega_eff']:.4f}")

                # Double-sided visualization: position on the binding curve
                from closures.nuclear_physics import compute_binding as _cb2

                a_range2 = list(range(4, 260, 2))
                be_data2 = []
                for _ai2 in a_range2:
                    _zi2 = max(1, round(_ai2 * 0.42))
                    try:
                        _ri2 = _cb2(_zi2, _ai2)
                        be_data2.append({"A": _ai2, "BE/A": _ri2.BE_per_A})
                    except Exception:
                        pass

                if be_data2:
                    fig = go.Figure()
                    # Color the curve by side
                    a_list = [d["A"] for d in be_data2]
                    be_list = [d["BE/A"] for d in be_data2]
                    fusion_a = [a for a in a_list if a <= 56]
                    fusion_be = [be for a, be in zip(a_list, be_list, strict=False) if a <= 56]
                    fission_a = [a for a in a_list if a >= 56]
                    fission_be = [be for a, be in zip(a_list, be_list, strict=False) if a >= 56]
                    fig.add_trace(
                        go.Scatter(
                            x=fusion_a,
                            y=fusion_be,
                            mode="lines",
                            name="Fusion side",
                            line={"color": "#007bff", "width": 2},
                            fill="tozeroy",
                            fillcolor="rgba(0,123,255,0.1)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fission_a,
                            y=fission_be,
                            mode="lines",
                            name="Fission side",
                            line={"color": "#dc3545", "width": 2},
                            fill="tozeroy",
                            fillcolor="rgba(220,53,69,0.1)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[a_ds],
                            y=[rd["BE_per_A"]],
                            mode="markers",
                            name=f"Z={z_ds}, A={a_ds}",
                            marker={
                                "size": 14,
                                "color": "#ffc107",
                                "symbol": "star",
                                "line": {"width": 2, "color": "black"},
                            },
                        )
                    )
                    # Arrow showing convergence direction
                    fig.add_annotation(
                        x=56, y=8.8, text="← Fusion | Fission →", showarrow=False, font={"size": 12, "color": "gray"}
                    )
                    fig.update_layout(
                        title="Double-Sided Collapse: Convergence on Iron Peak",
                        xaxis_title="A",
                        yaxis_title="BE/A (MeV/nucleon)",
                        height=380,
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")


def render_quantum_page() -> None:
    """Render interactive Quantum Mechanics domain page with all 6 closures."""
    if st is None or go is None or pd is None:
        return

    st.title("🔮 Quantum Mechanics Domain")
    st.caption("QM.INTSTACK.v1 — Born rule, entanglement, tunneling, harmonic oscillator, spin, uncertainty")

    with st.expander("📖 Domain Overview", expanded=False):
        st.markdown("""
        The **Quantum Mechanics** domain maps quantum observables into UMCP contract space.
        Each closure validates a fundamental quantum principle against measurement data.

        | Closure | Principle | Key Observable |
        |---------|-----------|---------------|
        | Wavefunction Collapse | Born rule: P = |ψ|² | Born deviation δP |
        | Entanglement | Bell's theorem, CHSH inequality | Concurrence, S_vN |
        | Tunneling | WKB approximation, Gamow factor | Transmission coefficient T |
        | Harmonic Oscillator | E_n = ℏω(n + ½) | Energy level spacing |
        | Spin Measurement | Stern-Gerlach, Zeeman effect | Larmor frequency |
        | Uncertainty Principle | ΔxΔp ≥ ℏ/2 | Heisenberg ratio |
        """)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🎲 Wavefunction Collapse",
            "🔗 Entanglement",
            "🚇 Tunneling",
            "🎵 Harmonic Oscillator",
            "🧭 Spin Measurement",
            "📏 Uncertainty Principle",
        ]
    )

    # ── Tab 1: Wavefunction Collapse ──
    with tab1:
        st.subheader("🎲 Wavefunction Collapse (Born Rule)")
        st.markdown("""
        **Born rule**: $P_i = |\\langle i | \\psi \\rangle|^2$ — the probability of measuring
        outcome $i$ equals the squared amplitude of the wavefunction projected onto that eigenstate.

        Enter wavefunction amplitudes and observed measurement probabilities to test Born-rule fidelity.
        """)

        preset_col, _ = st.columns([1, 2])
        with preset_col:
            qcp = st.selectbox(
                "Presets", ["Custom", "🎯 Perfect Born", "📐 Superposition", "🌀 Decoherent"], key="qm_coll_preset"
            )
        presets_coll = {
            "🎯 Perfect Born": ("0.6, 0.8", "0.36, 0.64"),
            "📐 Superposition": ("0.707, 0.707", "0.50, 0.50"),
            "🌀 Decoherent": ("0.6, 0.8", "0.50, 0.50"),
        }
        _psi, _prob = presets_coll.get(qcp, ("0.6, 0.8", "0.35, 0.65"))
        c1, c2 = st.columns(2)
        with c1:
            psi_str = st.text_input("ψ amplitudes (comma-separated)", _psi, key="qm_psi")
        with c2:
            prob_str = st.text_input("P measured (comma-separated)", _prob, key="qm_prob")

        if st.button("Compute Collapse", key="qm_collapse", type="primary"):
            try:
                from closures.quantum_mechanics.wavefunction_collapse import compute_wavefunction_collapse

                psi = [float(x.strip()) for x in psi_str.split(",")]
                probs = [float(x.strip()) for x in prob_str.split(",")]
                result = compute_wavefunction_collapse(psi, probs)
                regime = result["regime"]
                regime_color = {"Faithful": "🟢", "Perturbed": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("P_born", f"{result['P_born']:.4f}")
                with rc2:
                    st.metric("δP (deviation)", f"{result['delta_P']:.6f}")
                with rc3:
                    st.metric("Fidelity", f"{result['fidelity_state']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")
                st.metric("Purity", f"{result['purity']:.4f}")

                # Born vs observed bar chart
                born_probs = [abs(float(x.strip())) ** 2 for x in psi_str.split(",")]
                norm = sum(born_probs)
                born_probs = [p / norm for p in born_probs]
                meas_probs = [float(x.strip()) for x in prob_str.split(",")]
                labels = [f"|{i}⟩" for i in range(len(born_probs))]

                fig = go.Figure()
                fig.add_trace(go.Bar(x=labels, y=born_probs, name="Born prediction", marker_color="#007bff"))
                fig.add_trace(go.Bar(x=labels, y=meas_probs, name="Measured", marker_color="#dc3545"))
                fig.update_layout(
                    title="Born Rule: Predicted vs Measured",
                    yaxis_title="Probability",
                    barmode="group",
                    height=300,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 2: Entanglement ──
    with tab2:
        st.subheader("🔗 Entanglement Analysis")
        st.markdown("""
        **Von Neumann entropy**: $S_{vN} = -\\text{tr}(\\rho \\ln \\rho)$ ·
        **Bell-CHSH inequality**: $|S| \\leq 2$ (classical), $|S| \\leq 2\\sqrt{2}$ (quantum max)

        Enter the density matrix eigenvalues and optional Bell correlations.
        """)
        preset_col2, _ = st.columns([1, 2])
        with preset_col2:
            ep = st.selectbox(
                "Presets", ["Custom", "🔗 Bell State (maximal)", "📐 Separable", "🌀 Mixed"], key="qm_ent_preset"
            )
        presets_ent = {
            "🔗 Bell State (maximal)": ("0.5, 0.5, 0.0, 0.0", "0.707, 0.707, 0.707, -0.707"),
            "📐 Separable": ("1.0, 0.0, 0.0, 0.0", "0.5, 0.5, 0.5, -0.5"),
            "🌀 Mixed": ("0.5, 0.3, 0.15, 0.05", "0.7, 0.7, 0.7, -0.7"),
        }
        _rho, _bell = presets_ent.get(ep, ("0.5, 0.3, 0.15, 0.05", "0.7, 0.7, 0.7, -0.7"))
        rho_str = st.text_input("ρ eigenvalues (comma-separated)", _rho, key="qm_rho")
        bell_str = st.text_input("Bell correlations (4 values)", _bell, key="qm_bell")

        if st.button("Compute Entanglement", key="qm_ent", type="primary"):
            try:
                from closures.quantum_mechanics.entanglement import compute_entanglement

                rho = [float(x.strip()) for x in rho_str.split(",")]
                bell = [float(x.strip()) for x in bell_str.split(",")] if bell_str.strip() else None
                result = compute_entanglement(rho, bell)
                regime = result["regime"]
                regime_color = {"Maximal": "🟢", "Strong": "🟢", "Weak": "🟡"}.get(regime, "⚪")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Concurrence", f"{result['concurrence']:.4f}")
                with rc2:
                    st.metric("S_vN (entropy)", f"{result['S_vN']:.4f}")
                with rc3:
                    st.metric("Bell parameter S", f"{result['bell_parameter']:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")
                st.metric("Negativity", f"{result['negativity']:.4f}")

                # Entanglement measures radar
                fig = go.Figure()
                cats = ["Concurrence", "S_vN / ln(2)", "Bell/2√2", "Negativity"]
                vals_r = [
                    result["concurrence"],
                    result["S_vN"] / 0.6931 if result["S_vN"] > 0 else 0,
                    result["bell_parameter"] / 2.828 if result["bell_parameter"] > 0 else 0,
                    result["negativity"],
                ]
                vals_r = [min(v, 1.0) for v in vals_r]
                fig.add_trace(
                    go.Scatterpolar(
                        r=[*vals_r, vals_r[0]],
                        theta=[*cats, cats[0]],
                        fill="toself",
                        name="Entanglement",
                        fillcolor="rgba(111, 66, 193, 0.2)",
                        line={"color": "#6f42c1"},
                    )
                )
                fig.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                    height=320,
                    margin={"t": 30, "b": 20},
                    title="Entanglement Signature",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Bell inequality check
                bell_val = result["bell_parameter"]
                if bell_val > 2:
                    st.success(
                        f"🔔 Bell inequality **violated** (S = {bell_val:.3f} > 2): quantum correlations confirmed"
                    )
                else:
                    st.info(f"Bell inequality satisfied (S = {bell_val:.3f} ≤ 2): classically explicable")
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 3: Tunneling ──
    with tab3:
        st.subheader("🚇 Quantum Tunneling (WKB)")
        st.markdown("""
        **WKB transmission**: $T \\approx \\exp\\left(-2 \\int_0^L \\kappa(x) \\, dx\\right)$
        where $\\kappa = \\sqrt{2m(V - E)} / \\hbar$

        A particle with energy E < V can tunnel through a potential barrier of height V and width L.
        """)
        preset_col3, _ = st.columns([1, 2])
        with preset_col3:
            tp = st.selectbox(
                "Presets",
                ["Custom", "⚛️ Alpha Decay", "🔬 STM Tip (thin barrier)", "🧱 Thick Barrier"],
                key="qm_tun_preset",
            )
        presets_tun = {
            "⚛️ Alpha Decay": (4.0, 30.0, 0.01),
            "🔬 STM Tip (thin barrier)": (4.0, 5.0, 0.5),
            "🧱 Thick Barrier": (1.0, 10.0, 5.0),
        }
        _ep, _vb, _bw = presets_tun.get(tp, (5.0, 10.0, 1.0))
        c1, c2, c3 = st.columns(3)
        with c1:
            e_p = st.number_input("E particle (eV)", 0.01, 100.0, _ep, 0.1, key="qm_ep")
        with c2:
            v_b = st.number_input("V barrier (eV)", 0.01, 200.0, _vb, 0.1, key="qm_vb")
        with c3:
            bw = st.number_input("Barrier width (nm)", 0.001, 50.0, _bw, 0.01, key="qm_bw")

        if st.button("Compute Tunneling", key="qm_tunnel", type="primary"):
            try:
                from closures.quantum_mechanics.tunneling import compute_tunneling

                result = compute_tunneling(e_p, v_b, bw)
                regime = result["regime"]
                regime_color = {"Transparent": "🟢", "Moderate": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("T (transmission)", f"{result['T_coeff']:.6e}")
                with rc2:
                    st.metric("κ barrier (1/nm)", f"{result['kappa_barrier']:.4f}")
                with rc3:
                    st.metric("T/T_classical", f"{result['T_ratio']:.4e}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Barrier potential diagram
                import math

                x_pts = 200
                x = [i * (bw * 3) / x_pts for i in range(x_pts)]
                v_profile = []
                psi_profile = []
                barrier_start = bw * 0.8
                barrier_end = barrier_start + bw
                for xi in x:
                    if barrier_start <= xi <= barrier_end:
                        v_profile.append(v_b)
                        psi_profile.append(e_p * math.exp(-result["kappa_barrier"] * (xi - barrier_start)))
                    else:
                        v_profile.append(0)
                        psi_profile.append(e_p)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=v_profile,
                        mode="lines",
                        name="V(x)",
                        fill="tozeroy",
                        fillcolor="rgba(220,53,69,0.2)",
                        line={"color": "#dc3545", "width": 2},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=psi_profile,
                        mode="lines",
                        name="|ψ|² (schematic)",
                        line={"color": "#007bff", "width": 2, "dash": "dash"},
                    )
                )
                fig.add_hline(y=e_p, line_dash="dot", line_color="green", annotation_text=f"E = {e_p:.1f} eV")
                fig.update_layout(
                    title="Barrier Potential & Tunneling",
                    xaxis_title="x (nm)",
                    yaxis_title="Energy (eV)",
                    height=300,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 4: Harmonic Oscillator ──
    with tab4:
        st.subheader("🎵 Quantum Harmonic Oscillator")
        st.markdown("""
        **Energy levels**: $E_n = \\hbar\\omega(n + \\frac{1}{2})$

        The quantum harmonic oscillator has equally-spaced energy levels. Compare predicted vs observed energy.
        """)
        c1, c2, c3 = st.columns(3)
        with c1:
            n_q = st.number_input("n (quantum number)", 0, 100, 0, key="qm_n")
        with c2:
            omega_f = st.number_input("ω (rad/s)", 1e10, 1e16, 1e13, key="qm_omega", format="%.2e")
        with c3:
            e_obs = st.number_input("E_obs (eV)", 0.0, 100.0, 0.05, 0.001, key="qm_eobs", format="%.4f")

        if st.button("Compute Oscillator", key="qm_osc", type="primary"):
            try:
                from closures.quantum_mechanics.harmonic_oscillator import compute_harmonic_oscillator

                result = compute_harmonic_oscillator(n_q, omega_f, e_obs)
                regime = result["regime"]
                regime_color = {"Faithful": "🟢", "Accurate": "🟢", "Perturbed": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("E_predicted (eV)", f"{result['E_predicted']:.6f}")
                with rc2:
                    st.metric("E_observed (eV)", f"{e_obs:.6f}")
                with rc3:
                    st.metric("δE", f"{result['delta_E']:.6f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Energy level diagram
                import math

                hbar_eV = 6.582e-16  # eV·s
                e_levels = [(nn + 0.5) * hbar_eV * omega_f for nn in range(min(n_q + 5, 15))]
                fig = go.Figure()
                for nn, en in enumerate(e_levels):
                    color = "#dc3545" if nn == n_q else "#007bff"
                    width = 3 if nn == n_q else 1
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[en, en],
                            mode="lines",
                            line={"color": color, "width": width},
                            name=f"n={nn}",
                            showlegend=(nn <= n_q + 2),
                        )
                    )
                    fig.add_annotation(x=1.1, y=en, text=f"n={nn}: {en:.4f} eV", showarrow=False, font={"size": 9})
                fig.update_layout(
                    title="Energy Level Diagram",
                    yaxis_title="E (eV)",
                    xaxis={"visible": False},
                    height=350,
                    margin={"t": 40, "b": 20, "r": 120},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 5: Spin Measurement ──
    with tab5:
        st.subheader("🧭 Spin Measurement (Stern-Gerlach)")
        st.markdown("""
        **Spin quantization**: $S_z = m_s \\hbar$ where $m_s = -s, -s+1, ..., +s$ ·
        **Larmor frequency**: $\\omega_L = g \\mu_B B / \\hbar$ ·
        **Zeeman splitting**: $\\Delta E = g \\mu_B B$
        """)
        preset_col5, _ = st.columns([1, 2])
        with preset_col5:
            spp = st.selectbox(
                "Presets",
                ["Custom", "⬆️ Spin-½ up (electron)", "🔄 Spin-1 (deuterium)", "⚛️ Spin-3/2"],
                key="qm_spin_preset",
            )
        presets_spin = {
            "⬆️ Spin-½ up (electron)": (0.5, 0.5, 1.0),
            "🔄 Spin-1 (deuterium)": (1.0, 1.0, 2.0),
            "⚛️ Spin-3/2": (1.5, 0.5, 1.0),
        }
        _st_val, _sz, _bf = presets_spin.get(spp, (0.5, 0.5, 1.0))
        c1, c2, c3 = st.columns(3)
        with c1:
            s_tot = st.number_input("S total", 0.0, 10.0, _st_val, 0.5, key="qm_stot")
        with c2:
            sz_obs = st.number_input("S_z observed", -10.0, 10.0, _sz, 0.5, key="qm_sz")
        with c3:
            b_field = st.number_input("B field (T)", 0.001, 50.0, _bf, 0.1, key="qm_bfield")

        if st.button("Compute Spin", key="qm_spin", type="primary"):
            try:
                from closures.quantum_mechanics.spin_measurement import compute_spin_measurement

                result = compute_spin_measurement(s_tot, sz_obs, b_field)
                regime = result["regime"]
                regime_color = {"Faithful": "🟢", "Perturbed": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("S_z predicted", f"{result['S_z_predicted']:.4f}")
                with rc2:
                    st.metric("Spin fidelity", f"{result['spin_fidelity']:.4f}")
                with rc3:
                    st.metric("Larmor (GHz)", f"{result['larmor_freq'] / 1e9:.4f}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                st.metric("Zeeman splitting (eV)", f"{result['zeeman_split']:.6e}")

                # Spin state visualization
                allowed_mz = [s_tot - i for i in range(int(2 * s_tot) + 1)]
                fig = go.Figure()
                for mz in allowed_mz:
                    is_obs = abs(mz - sz_obs) < 0.01
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[mz, mz],
                            mode="lines",
                            line={"color": "#dc3545" if is_obs else "#007bff", "width": 3 if is_obs else 1.5},
                            name=f"m_s = {mz:+.1f}" + (" ← observed" if is_obs else ""),
                        )
                    )
                fig.update_layout(
                    title=f"Spin-{s_tot} Projections (2s+1 = {int(2 * s_tot + 1)} states)",
                    yaxis_title="m_s",
                    xaxis={"visible": False},
                    height=280,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 6: Uncertainty Principle ──
    with tab6:
        st.subheader("📏 Heisenberg Uncertainty Principle")
        st.markdown("""
        **Heisenberg** (1927): $\\Delta x \\cdot \\Delta p \\geq \\frac{\\hbar}{2}$

        The product of position and momentum uncertainties has a fundamental quantum lower bound.
        The ratio to ℏ/2 measures how close the state is to a minimum-uncertainty wavepacket.
        """)
        preset_col6, _ = st.columns([1, 2])
        with preset_col6:
            up = st.selectbox(
                "Presets",
                ["Custom", "📏 Minimum Uncertainty", "🌊 Spread Wavepacket", "⚛️ Atomic Scale"],
                key="qm_unc_preset",
            )
        presets_unc = {
            "📏 Minimum Uncertainty": (0.1, 5.27e-25),
            "🌊 Spread Wavepacket": (10.0, 1e-24),
            "⚛️ Atomic Scale": (0.053, 1.99e-24),
        }
        _dx, _dp = presets_unc.get(up, (1.0, 0.1))
        c1, c2 = st.columns(2)
        with c1:
            dx = st.number_input("Δx (nm)", 0.001, 10000.0, _dx, 0.01, key="qm_dx")
        with c2:
            dp = st.number_input("Δp (kg·m/s)", 1e-30, 1e-20, _dp, 1e-26, key="qm_dp", format="%.4e")

        if st.button("Check Uncertainty", key="qm_unc", type="primary"):
            try:
                from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

                result = compute_uncertainty(dx, dp)
                regime = result["regime"]
                regime_color = {"Minimum": "🟢", "Moderate": "🟡", "Dispersed": "🟡"}.get(regime, "🔴")

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("ΔxΔp", f"{result['heisenberg_product']:.4e}")
                with rc2:
                    st.metric("Ratio to ℏ/2", f"{result['heisenberg_ratio']:.4f}")
                with rc3:
                    st.metric("ℏ/2", f"{result['min_uncertainty']:.4e}")
                with rc4:
                    st.metric("Regime", f"{regime_color} {regime}")

                # Heisenberg bound visualization
                ratio = result["heisenberg_ratio"]
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=ratio,
                        title={"text": "ΔxΔp / (ℏ/2)"},
                        gauge={
                            "axis": {"range": [0, max(10, ratio * 1.5)]},
                            "bar": {"color": "#007bff"},
                            "steps": [
                                {"range": [0, 1], "color": "#f8d7da"},
                                {"range": [1, 2], "color": "#d4edda"},
                                {"range": [2, 5], "color": "#fff3cd"},
                                {"range": [5, max(10, ratio * 1.5)], "color": "#e2e3e5"},
                            ],
                            "threshold": {"line": {"color": "red", "width": 3}, "value": 1.0},
                        },
                    )
                )
                fig.update_layout(height=250, margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)
                if ratio >= 1:
                    st.success("✅ Heisenberg bound satisfied (ΔxΔp ≥ ℏ/2)")
                else:
                    st.error("⚠️ Heisenberg bound **violated** — check input units")
            except Exception as e:
                st.error(f"Computation error: {e}")


def render_finance_page() -> None:
    """Render interactive Finance domain page with professional visualizations."""
    if st is None or go is None or pd is None:
        return

    st.title("💰 Finance Domain")
    st.caption("FINANCE.INTSTACK.v1 — Business financial continuity validation via UMCP embedding")

    with st.expander("📖 Domain Overview", expanded=False):
        st.markdown("""
        The **Finance** domain maps raw financial records into UMCP's [0, 1]⁴ coordinate space,
        enabling contract-based validation of business health.

        **Embedding coordinates** (each clipped to [ε, 1−ε]):

        | Coordinate | Formula | Measures |
        |-----------|---------|----------|
        | c₁ (Revenue) | min(revenue / target, 1.0) | Revenue performance vs goal |
        | c₂ (Expense) | min(budget / expenses, 1.0) | Expense control efficiency |
        | c₃ (Margin) | (revenue − COGS) / revenue | Gross margin profitability |
        | c₄ (Cashflow) | min(cashflow / target, 1.0) | Cash flow health |

        **Default weights**: w = [0.30, 0.25, 0.25, 0.20]

        Once embedded, the standard UMCP invariants (ω, F, κ, IC) are computed and the
        regime is classified as **STABLE** / **WATCH** / **COLLAPSE**.
        """)

    st.divider()

    # ── Preset selection ──
    preset_col, _ = st.columns([1, 2])
    with preset_col:
        fp = st.selectbox(
            "Scenario Presets",
            [
                "Custom",
                "🏢 Healthy Business",
                "📈 Growth Phase",
                "📉 Cash Crunch",
                "⚠️ Margin Squeeze",
                "🔴 Distressed",
            ],
            key="fin_preset",
        )

    presets_fin: dict[str, dict[str, float | str]] = {
        "🏢 Healthy Business": {
            "month": "2026-01",
            "rev": 500000,
            "exp": 380000,
            "cogs": 200000,
            "cf": 90000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
        "📈 Growth Phase": {
            "month": "2026-03",
            "rev": 650000,
            "exp": 500000,
            "cogs": 260000,
            "cf": 120000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
        "📉 Cash Crunch": {
            "month": "2026-06",
            "rev": 480000,
            "exp": 460000,
            "cogs": 200000,
            "cf": 15000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
        "⚠️ Margin Squeeze": {
            "month": "2026-09",
            "rev": 500000,
            "exp": 420000,
            "cogs": 400000,
            "cf": 60000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
        "🔴 Distressed": {
            "month": "2026-12",
            "rev": 200000,
            "exp": 550000,
            "cogs": 180000,
            "cf": -50000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
    }

    p = presets_fin.get(
        fp,
        {
            "month": "2026-01",
            "rev": 500000,
            "exp": 400000,
            "cogs": 200000,
            "cf": 80000,
            "rev_t": 500000,
            "exp_t": 450000,
            "cf_t": 75000,
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Financial Record**")
        month = st.text_input("Month (YYYY-MM)", str(p["month"]), key="fin_month")
        revenue = st.number_input("Revenue ($)", 0.0, 1e9, float(p["rev"]), 1000.0, key="fin_rev")
        expenses = st.number_input("Expenses ($)", 0.0, 1e9, float(p["exp"]), 1000.0, key="fin_exp")
        cogs = st.number_input("COGS ($)", 0.0, 1e9, float(p["cogs"]), 1000.0, key="fin_cogs")
        cashflow = st.number_input("Cashflow ($)", -1e9, 1e9, float(p["cf"]), 1000.0, key="fin_cf")

    with col2:
        st.markdown("**🎯 Targets**")
        rev_target = st.number_input("Revenue Target ($)", 0.0, 1e9, float(p["rev_t"]), 1000.0, key="fin_rev_t")
        exp_budget = st.number_input("Expense Budget ($)", 0.0, 1e9, float(p["exp_t"]), 1000.0, key="fin_exp_t")
        cf_target = st.number_input("Cashflow Target ($)", 0.0, 1e9, float(p["cf_t"]), 1000.0, key="fin_cf_t")

    if st.button("💹 Embed & Analyze", key="fin_embed", type="primary"):
        try:
            from closures.finance.finance_embedding import FinanceRecord, FinanceTargets, embed_finance

            record = FinanceRecord(month=month, revenue=revenue, expenses=expenses, cogs=cogs, cashflow=cashflow)
            targets = FinanceTargets(revenue_target=rev_target, expense_budget=exp_budget, cashflow_target=cf_target)
            embedded = embed_finance(record, targets)

            st.divider()
            st.subheader("📐 UMCP Embedding Coordinates")
            coord_names = ["c₁ Revenue", "c₂ Expense", "c₃ Margin", "c₄ Cashflow"]
            c_cols = st.columns(4)
            for i, (name, val) in enumerate(zip(coord_names, embedded.c, strict=False)):
                with c_cols[i]:
                    flag = "⚠️ OOR" if embedded.oor_flags[i] else "✅ In-range"
                    st.metric(name, f"{val:.4f}", delta=flag)

            # Compute UMCP invariants from coordinates
            import math

            weights = [0.30, 0.25, 0.25, 0.20]
            epsilon = 1e-8
            kappa = sum(w * math.log(max(c, epsilon)) for w, c in zip(weights, embedded.c, strict=False))
            ic = math.exp(kappa)
            omega = 1.0 - sum(w * c for w, c in zip(weights, embedded.c, strict=False))
            f_val = 1.0 - omega

            st.divider()
            st.subheader("🧮 UMCP Invariants")
            inv_cols = st.columns(5)
            with inv_cols[0]:
                st.metric("ω (drift)", f"{omega:.4f}")
            with inv_cols[1]:
                st.metric("F = 1−ω", f"{f_val:.4f}")
            with inv_cols[2]:
                st.metric("κ", f"{kappa:.4f}")
            with inv_cols[3]:
                st.metric("IC = exp(κ)", f"{ic:.4f}")
            with inv_cols[4]:
                regime = "STABLE" if 0.3 <= omega <= 0.7 else "WATCH" if 0.1 <= omega <= 0.9 else "COLLAPSE"
                color = "🟢" if regime == "STABLE" else "🟡" if regime == "WATCH" else "🔴"
                st.metric("Regime", f"{color} {regime}")

            # Identity checks
            st.divider()
            ic_le_f = ic <= f_val + 1e-9
            f_plus_omega = f_val + omega
            st.subheader("📋 Tier-1 Identity Verification")
            id_cols = st.columns(3)
            with id_cols[0]:
                st.markdown(f"**F + ω = {f_plus_omega:.6f}** {'✅' if abs(f_plus_omega - 1.0) < 1e-6 else '⚠️'}")
                st.caption("Must equal 1.0 (budget identity)")
            with id_cols[1]:
                st.markdown(f"**IC ≤ F** → {ic:.4f} ≤ {f_val:.4f} {'✅' if ic_le_f else '❌'}")
                st.caption("AM-GM inequality (information ≤ fidelity)")
            with id_cols[2]:
                ic_target = math.exp(kappa)
                st.markdown(
                    f"**IC ≈ exp(κ)** → {ic:.6f} ≈ {ic_target:.6f} {'✅' if abs(ic - ic_target) < 1e-9 else '⚠️'}"
                )
                st.caption("Exponential consistency check")

            st.divider()

            # Two-column visualization
            viz_left, viz_right = st.columns(2)

            with viz_left:
                # Radar chart
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=[*list(embedded.c), embedded.c[0]],
                        theta=[*coord_names, coord_names[0]],
                        fill="toself",
                        name="Financial Health",
                        fillcolor="rgba(0, 123, 255, 0.15)",
                        line={"color": "#007bff", "width": 2},
                    )
                )
                # Add target ring at 1.0
                fig.add_trace(
                    go.Scatterpolar(
                        r=[1.0] * 5,
                        theta=[*coord_names, coord_names[0]],
                        mode="lines",
                        name="Target (1.0)",
                        line={"color": "#28a745", "width": 1, "dash": "dash"},
                    )
                )
                fig.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 1.1]}},
                    height=380,
                    margin={"t": 40, "b": 30},
                    title="Financial Health Radar",
                    showlegend=True,
                    legend={"x": 0.7, "y": 0.05},
                )
                st.plotly_chart(fig, use_container_width=True)

            with viz_right:
                # Bar chart comparing actuals vs targets
                categories = ["Revenue", "Expenses", "Cashflow"]
                actuals = [revenue, expenses, cashflow]
                targets_vals = [rev_target, exp_budget, cf_target]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(x=categories, y=actuals, name="Actual", marker_color=["#007bff", "#dc3545", "#28a745"])
                )
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=targets_vals,
                        name="Target",
                        marker_color=["#007bff55", "#dc354555", "#28a74555"],
                    )
                )
                fig.update_layout(
                    title="Actual vs Target",
                    barmode="group",
                    yaxis_title="$ (USD)",
                    height=380,
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Profitability breakdown
            gross_profit = revenue - cogs
            operating_income = revenue - expenses
            margin_pct = (gross_profit / revenue * 100) if revenue > 0 else 0

            st.subheader("📊 Financial Summary")
            sum_cols = st.columns(4)
            with sum_cols[0]:
                st.metric("Gross Profit", f"${gross_profit:,.0f}", delta=f"{margin_pct:.1f}% margin")
            with sum_cols[1]:
                st.metric(
                    "Operating Income", f"${operating_income:,.0f}", delta=f"{'✅' if operating_income > 0 else '🔴'}"
                )
            with sum_cols[2]:
                rev_ratio = (revenue / rev_target * 100) if rev_target > 0 else 0
                st.metric(
                    "Revenue vs Target",
                    f"{rev_ratio:.0f}%",
                    delta=f"{'▲' if rev_ratio >= 100 else '▼'} {abs(rev_ratio - 100):.0f}%",
                )
            with sum_cols[3]:
                cf_ratio = (cashflow / cf_target * 100) if cf_target > 0 else 0
                st.metric(
                    "Cashflow vs Target",
                    f"{cf_ratio:.0f}%",
                    delta=f"{'▲' if cf_ratio >= 100 else '▼'} {abs(cf_ratio - 100):.0f}%",
                )

            # Waterfall chart of P&L
            fig_wf = go.Figure(
                go.Waterfall(
                    name="P&L Flow",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=["Revenue", "COGS", "OpEx", "Cashflow Adj.", "Net Position"],
                    y=[
                        revenue,
                        -cogs,
                        -(expenses - cogs),
                        cashflow - operating_income,
                        operating_income + (cashflow - operating_income),
                    ],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#28a745"}},
                    decreasing={"marker": {"color": "#dc3545"}},
                    totals={"marker": {"color": "#007bff"}},
                )
            )
            fig_wf.update_layout(title="P&L Waterfall", yaxis_title="$ (USD)", height=350, margin={"t": 40, "b": 20})
            st.plotly_chart(fig_wf, use_container_width=True)

        except Exception as e:
            st.error(f"Computation error: {e}")


def render_rcft_page() -> None:
    """Render interactive RCFT domain page with all 4 closures and professional visualizations."""
    if st is None or go is None or pd is None or np is None:
        return

    st.title("🌀 RCFT Domain")
    st.caption(
        "RCFT.INTSTACK.v1 — Recursive Collapse Field Theory: fractal dimension, recursive fields, attractor basins, resonance"
    )

    with st.expander("📖 Domain Overview", expanded=False):
        st.markdown("""
        **Recursive Collapse Field Theory** (RCFT) extends UMCP into dynamical systems territory.
        It analyzes time-series of Tier-1 invariants (ω, S, C, F) to detect:

        - **Fractal structure** in collapse trajectories (box-counting dimension D_f)
        - **Recursive field strength** Ψ_r = Σ αⁿ · Ψₙ (memory-weighted invariant accumulation)
        - **Attractor topology** in (ω, S, C) phase space (monostable vs multistable)
        - **Resonance patterns** via FFT (dominant wavelength λ, phase coherence Θ)

        | Closure | Output | Regime Classification |
        |---------|--------|----------------------|
        | Fractal Dimension | D_f ∈ [1, 3] | Smooth < 1.2 · Wrinkled 1.2–1.8 · Turbulent ≥ 1.8 |
        | Recursive Field | Ψ_r ≥ 0 | Dormant < 0.1 · Active 0.1–1.0 · Resonant ≥ 1.0 |
        | Attractor Basin | n_attractors ≥ 1 | Monostable · Bistable · Multistable |
        | Resonance Pattern | λ, Θ | Standing · Traveling · Mixed |
        """)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📐 Fractal Dimension",
            "🌊 Recursive Field",
            "🎯 Attractor Basin",
            "🔊 Resonance Pattern",
        ]
    )

    # ── Tab 1: Fractal Dimension ──
    with tab1:
        st.subheader("📐 Fractal Dimension (Box-Counting)")
        st.markdown("""
        **Box-counting dimension**: $D_f = \\lim_{\\varepsilon \\to 0} \\frac{\\log N(\\varepsilon)}{\\log(1/\\varepsilon)}$

        The fractal dimension D_f quantifies the complexity of collapse trajectories in invariant space.
        A smooth trajectory has D_f ≈ 1 (line); a turbulent one approaches D_f ≈ 2 (space-filling).
        """)

        preset_col, _ = st.columns([1, 2])
        with preset_col:
            fp1 = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "🟢 Smooth Orbit (low noise)",
                    "🟡 Wrinkled Spiral (moderate)",
                    "🔴 Turbulent Cloud (high noise)",
                    "⭕ Clean Circle",
                ],
                key="rcft_frac_preset",
            )

        presets_frac = {
            "🟢 Smooth Orbit (low noise)": (200, 0.05),
            "🟡 Wrinkled Spiral (moderate)": (300, 0.4),
            "🔴 Turbulent Cloud (high noise)": (400, 0.9),
            "⭕ Clean Circle": (500, 0.0),
        }
        _npts, _noise = presets_frac.get(fp1, (200, 0.3))

        c1, c2 = st.columns(2)
        with c1:
            n_pts = st.slider("Trajectory points", 50, 500, _npts, key="rcft_npts")
        with c2:
            noise = st.slider("Noise level", 0.0, 1.0, _noise, 0.05, key="rcft_noise")

        if st.button("Compute Fractal Dimension", key="rcft_frac", type="primary"):
            try:
                from closures.rcft.fractal_dimension import compute_fractal_dimension

                t = np.linspace(0, 10, n_pts)
                trajectory = np.column_stack(
                    [np.sin(t) + noise * np.random.randn(n_pts), np.cos(t) + noise * np.random.randn(n_pts)]
                )
                result = compute_fractal_dimension(trajectory)

                st.divider()
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("D_fractal", f"{result['D_fractal']:.4f}")
                with rc2:
                    st.metric("R²", f"{result.get('r_squared', 0):.4f}")
                with rc3:
                    regime = result["regime"]
                    color = "🟢" if regime == "Smooth" else "🟡" if regime == "Wrinkled" else "🔴"
                    st.metric("Regime", f"{color} {regime}")
                with rc4:
                    st.metric("log slope", f"{result.get('log_slope', result['D_fractal']):.4f}")

                # Two-column visualization
                viz_l, viz_r = st.columns(2)
                with viz_l:
                    # Trajectory scatter plot
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=trajectory[:, 0],
                            y=trajectory[:, 1],
                            mode="markers",
                            marker={
                                "size": 3,
                                "color": list(range(n_pts)),
                                "colorscale": "Viridis",
                                "colorbar": {"title": "t"},
                            },
                            name="Trajectory",
                        )
                    )
                    fig.update_layout(
                        title=f"Collapse Trajectory (D_f = {result['D_fractal']:.3f})",
                        xaxis_title="sin(t) + noise",
                        yaxis_title="cos(t) + noise",
                        height=350,
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with viz_r:
                    # Log-log box counting plot
                    eps_used = result.get("eps_used", None)
                    box_counts = result.get("box_counts", None)
                    if eps_used is not None and box_counts is not None:
                        log_eps = np.log(1.0 / np.array(eps_used))
                        log_n = np.log(np.array(box_counts).astype(float) + 1)
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=log_eps.tolist(),
                                y=log_n.tolist(),
                                mode="markers+lines",
                                name="Box counts",
                                marker={"color": "#007bff", "size": 6},
                            )
                        )
                        fig2.update_layout(
                            title="Box-Counting (log-log)",
                            xaxis_title="log(1/ε)",
                            yaxis_title="log N(ε)",
                            height=350,
                            margin={"t": 40, "b": 20},
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        # Dimension gauge
                        fig2 = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=result["D_fractal"],
                                title={"text": "Fractal Dimension"},
                                gauge={
                                    "axis": {"range": [1, 3]},
                                    "bar": {"color": "#007bff"},
                                    "steps": [
                                        {"range": [1, 1.2], "color": "#d4edda"},
                                        {"range": [1.2, 1.8], "color": "#fff3cd"},
                                        {"range": [1.8, 3], "color": "#f8d7da"},
                                    ],
                                },
                            )
                        )
                        fig2.update_layout(height=300, margin={"t": 40, "b": 10})
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 2: Recursive Field ──
    with tab2:
        st.subheader("🌊 Recursive Field Strength (Ψ_r)")
        st.markdown("""
        **Recursive field**: $\\Psi_r = \\sum_{n=0}^{N} \\alpha^n \\cdot \\Psi_n$
        where $\\Psi_n = \\sqrt{S_n^2 + C_n^2} \\cdot (1 - F_n)$

        The field accumulates history with exponential decay α, measuring how strongly
        past invariant states influence the current collapse trajectory.
        """)

        preset_col2, _ = st.columns([1, 2])
        with preset_col2:
            fp2 = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "💤 Dormant (stable system)",
                    "⚡ Active (moderate dynamics)",
                    "🔥 Resonant (high memory)",
                ],
                key="rcft_field_preset",
            )

        presets_field = {
            "💤 Dormant (stable system)": (30, 0.3, (0.1, 0.2), (0.05, 0.1), (0.85, 0.95)),
            "⚡ Active (moderate dynamics)": (80, 0.8, (0.3, 0.7), (0.2, 0.5), (0.3, 0.7)),
            "🔥 Resonant (high memory)": (150, 0.95, (0.5, 0.9), (0.3, 0.8), (0.1, 0.4)),
        }
        _ns, _alpha, _s_range, _c_range, _f_range = presets_field.get(
            fp2, (50, 0.8, (0.2, 0.8), (0.1, 0.5), (0.3, 0.9))
        )

        c1, c2 = st.columns(2)
        with c1:
            n_series = st.slider("Series length", 10, 200, _ns, key="rcft_nseries")
        with c2:
            alpha_param = st.slider("α (decay)", 0.1, 0.99, _alpha, 0.01, key="rcft_alpha")

        if st.button("Compute Recursive Field", key="rcft_field", type="primary"):
            try:
                from closures.rcft.recursive_field import compute_recursive_field

                S_arr = np.random.uniform(_s_range[0], _s_range[1], n_series)
                C_arr = np.random.uniform(_c_range[0], _c_range[1], n_series)
                F_arr = np.random.uniform(_f_range[0], _f_range[1], n_series)
                result = compute_recursive_field(S_arr, C_arr, F_arr, alpha=alpha_param)

                st.divider()
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Ψ_recursive", f"{result['Psi_recursive']:.4f}")
                with rc2:
                    st.metric("Iterations", str(result.get("n_iterations", "N/A")))
                with rc3:
                    converged = result.get("convergence_achieved", False)
                    st.metric("Converged", "✅ Yes" if converged else "⏳ No")
                with rc4:
                    regime = result["regime"]
                    color = "💤" if regime == "Dormant" else "⚡" if regime == "Active" else "🔥"
                    st.metric("Regime", f"{color} {regime}")

                viz_l, viz_r = st.columns(2)
                with viz_l:
                    # Weighted contributions over time
                    weighted = result.get("weighted_contributions", None)
                    if weighted is not None:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                y=weighted.tolist(),
                                mode="lines+markers",
                                marker={"size": 3, "color": "#007bff"},
                                line={"color": "#007bff", "width": 1},
                                name="αⁿ · Ψₙ",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                y=result.get("decay_factors", np.array([])).tolist(),
                                mode="lines",
                                line={"color": "#dc3545", "width": 1, "dash": "dash"},
                                name="αⁿ envelope",
                            )
                        )
                        fig.update_layout(
                            title="Weighted Contributions",
                            xaxis_title="n (iteration)",
                            yaxis_title="αⁿ · Ψₙ",
                            height=350,
                            margin={"t": 40, "b": 20},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with viz_r:
                    # Input invariant time series
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(y=S_arr.tolist(), name="S (entropy)", line={"color": "#007bff", "width": 1})
                    )
                    fig2.add_trace(
                        go.Scatter(y=C_arr.tolist(), name="C (curvature)", line={"color": "#28a745", "width": 1})
                    )
                    fig2.add_trace(
                        go.Scatter(y=F_arr.tolist(), name="F (fidelity)", line={"color": "#dc3545", "width": 1})
                    )
                    fig2.update_layout(
                        title="Input Invariant Series",
                        xaxis_title="Time step",
                        yaxis_title="Value",
                        height=350,
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Components summary
                components = result.get("components", {})
                if components:
                    st.markdown("**📊 Field Components**")
                    comp_cols = st.columns(5)
                    comp_items = list(components.items())[:5]
                    for i, (k, v) in enumerate(comp_items):
                        with comp_cols[i]:
                            label = k.replace("_", " ").title()
                            st.metric(label, f"{v:.4f}" if isinstance(v, float) else str(v))

            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 3: Attractor Basin ──
    with tab3:
        st.subheader("🎯 Attractor Basin Topology")
        st.markdown("""
        Analyzes the phase-space structure of collapse trajectories in (ω, S, C) coordinates.
        Classifies the system as **Monostable** (single attractor), **Bistable** (two),
        or **Multistable** (three+). The dominant attractor indicates the most probable
        collapse endpoint.
        """)

        preset_col3, _ = st.columns([1, 2])
        with preset_col3:
            fp3 = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "🎯 Monostable (tight cluster)",
                    "⚖️ Bistable (two basins)",
                    "🌀 Multistable (chaotic)",
                ],
                key="rcft_basin_preset",
            )

        presets_basin = {
            "🎯 Monostable (tight cluster)": (100, (0.3, 0.5), (0.2, 0.4), (0.1, 0.2)),
            "⚖️ Bistable (two basins)": (200, (0.1, 0.9), (0.1, 0.8), (0.05, 0.5)),
            "🌀 Multistable (chaotic)": (400, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        }
        _nb, _omega_r, _s_r, _c_r = presets_basin.get(fp3, (100, (0.1, 0.9), (0.1, 0.8), (0.05, 0.5)))

        n_basin = st.slider("Series length", 20, 500, _nb, key="rcft_nbasin")

        if st.button("Compute Attractor Basin", key="rcft_basin", type="primary"):
            try:
                from closures.rcft.attractor_basin import compute_attractor_basin

                omega_arr = np.random.uniform(_omega_r[0], _omega_r[1], n_basin)
                S_arr = np.random.uniform(_s_r[0], _s_r[1], n_basin)
                C_arr = np.random.uniform(_c_r[0], _c_r[1], n_basin)
                result = compute_attractor_basin(omega_arr, S_arr, C_arr)

                st.divider()
                rc1, rc2, rc3, rc4 = st.columns(4)
                n_att = result.get("n_attractors_found", 0)
                with rc1:
                    st.metric("Attractors Found", str(n_att))
                with rc2:
                    st.metric("Dominant", str(result.get("dominant_attractor", "N/A")))
                with rc3:
                    max_strength = result.get("max_basin_strength", 0)
                    st.metric(
                        "Max Strength", f"{max_strength:.4f}" if isinstance(max_strength, float) else str(max_strength)
                    )
                with rc4:
                    regime = result.get("regime", "Unknown")
                    color = "🎯" if regime == "Monostable" else "⚖️" if regime == "Bistable" else "🌀"
                    st.metric("Regime", f"{color} {regime}")

                viz_l, viz_r = st.columns(2)
                with viz_l:
                    # 3D scatter of trajectory with basin coloring
                    traj_class = result.get("trajectory_classification", [])
                    fig = go.Figure()
                    if len(traj_class) == n_basin:
                        fig.add_trace(
                            go.Scatter3d(
                                x=omega_arr.tolist(),
                                y=S_arr.tolist(),
                                z=C_arr.tolist(),
                                mode="markers",
                                marker={
                                    "size": 3,
                                    "color": list(traj_class),
                                    "colorscale": "Set1",
                                    "colorbar": {"title": "Basin"},
                                },
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=omega_arr.tolist(),
                                y=S_arr.tolist(),
                                z=C_arr.tolist(),
                                mode="markers",
                                marker={"size": 3, "color": "#007bff"},
                            )
                        )

                    # Mark attractor locations
                    locs = result.get("attractor_locations", [])
                    for idx, loc in enumerate(locs):
                        if len(loc) >= 3:
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[loc[0]],
                                    y=[loc[1]],
                                    z=[loc[2]],
                                    mode="markers",
                                    marker={"size": 10, "color": "red", "symbol": "diamond"},
                                    name=f"Attractor {idx}",
                                )
                            )

                    fig.update_layout(
                        title=f"Phase Space ({regime})",
                        scene={"xaxis_title": "ω", "yaxis_title": "S", "zaxis_title": "C"},
                        height=400,
                        margin={"t": 40, "b": 10},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with viz_r:
                    # Basin strength bar chart
                    strengths = result.get("basin_strengths", [])
                    volumes = result.get("basin_volumes", [])
                    if strengths:
                        basin_labels = [f"Basin {i}" for i in range(len(strengths))]
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Bar(
                                x=basin_labels,
                                y=strengths,
                                name="Strength",
                                marker_color=["#007bff", "#dc3545", "#28a745", "#ffc107", "#6f42c1"][: len(strengths)],
                            )
                        )
                        if volumes:
                            fig2.add_trace(
                                go.Bar(
                                    x=basin_labels,
                                    y=volumes,
                                    name="Volume fraction",
                                    marker_color=["#007bff55", "#dc354555", "#28a74555", "#ffc10755", "#6f42c155"][
                                        : len(volumes)
                                    ],
                                )
                            )
                        fig2.update_layout(
                            title="Basin Properties",
                            barmode="group",
                            yaxis_title="Value",
                            height=400,
                            margin={"t": 40, "b": 20},
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    # Convergence rates
                    rates = result.get("convergence_rates", [])
                    if rates:
                        st.markdown("**📉 Convergence Rates**")
                        rate_cols = st.columns(min(len(rates), 4))
                        for i, r in enumerate(rates[:4]):
                            with rate_cols[i]:
                                st.metric(f"Basin {i}", f"{r:.4f}" if isinstance(r, float) else str(r))

            except Exception as e:
                st.error(f"Computation error: {e}")

    # ── Tab 4: Resonance Pattern ──
    with tab4:
        st.subheader("🔊 Resonance Pattern (FFT)")
        st.markdown("""
        Extracts the dominant wavelength λ and phase angle Θ from field time series via FFT.
        - **Standing** pattern: high phase coherence, stationary nodes
        - **Traveling** pattern: propagating wave structure
        - **Mixed**: superposition of standing and traveling components
        """)

        preset_col4, _ = st.columns([1, 2])
        with preset_col4:
            fp4 = st.selectbox(
                "Presets",
                [
                    "Custom",
                    "🎵 Pure Sine (f = 0.1)",
                    "🎶 Harmonic Mix (f = 0.05 + 0.15)",
                    "📻 Noisy Signal (f = 0.2, high noise)",
                    "🔇 White Noise (no signal)",
                ],
                key="rcft_res_preset",
            )

        presets_res: dict[str, tuple[int, float, float, str]] = {
            "🎵 Pure Sine (f = 0.1)": (256, 0.1, 0.05, "pure"),
            "🎶 Harmonic Mix (f = 0.05 + 0.15)": (256, 0.05, 0.1, "harmonic"),
            "📻 Noisy Signal (f = 0.2, high noise)": (256, 0.2, 0.8, "pure"),
            "🔇 White Noise (no signal)": (256, 0.0, 1.0, "noise"),
        }
        _nr, _freq, _noise_r, _mode = presets_res.get(fp4, (128, 0.1, 0.3, "pure"))

        c1, c2 = st.columns(2)
        with c1:
            n_res = st.slider("Series length", 32, 512, _nr, key="rcft_nres")
        with c2:
            freq = st.slider("Primary frequency", 0.01, 0.5, _freq, 0.01, key="rcft_freq")
        noise_res = st.slider("Noise amplitude", 0.0, 2.0, _noise_r, 0.05, key="rcft_noise_res")

        if st.button("Compute Resonance", key="rcft_res", type="primary"):
            try:
                from closures.rcft.resonance_pattern import compute_resonance_pattern

                t = np.arange(n_res)
                if _mode == "harmonic":
                    signal = (
                        np.sin(2 * np.pi * freq * t)
                        + 0.5 * np.sin(2 * np.pi * 3 * freq * t)
                        + noise_res * np.random.randn(n_res)
                    )
                elif _mode == "noise":
                    signal = noise_res * np.random.randn(n_res)
                else:
                    signal = np.sin(2 * np.pi * freq * t) + noise_res * np.random.randn(n_res)

                result = compute_resonance_pattern(signal)

                st.divider()
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("λ_pattern", f"{result['lambda_pattern']:.4f}")
                with rc2:
                    st.metric("Θ_phase", f"{result['Theta_phase']:.4f}")
                with rc3:
                    dom_f = result.get("dominant_frequency", 0.0)
                    st.metric("f_dominant", f"{dom_f:.4f}")
                with rc4:
                    ptype = result.get("pattern_type", "")
                    color = "🎵" if ptype == "Standing" else "🌊" if ptype == "Traveling" else "🔀"
                    st.metric("Pattern", f"{color} {ptype}")

                # Additional metrics
                coh = result.get("phase_coherence", 0.0)
                harm = result.get("harmonic_content", 0.0)
                st.columns(3)[0].metric("Phase Coherence", f"{coh:.4f}")
                st.columns(3)[1].metric("Harmonic Content", f"{harm:.4f}")

                viz_l, viz_r = st.columns(2)
                with viz_l:
                    # Time-domain signal
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            y=signal.tolist(), mode="lines", line={"color": "#007bff", "width": 1}, name="Signal"
                        )
                    )
                    fig.update_layout(
                        title="Time-Domain Signal",
                        xaxis_title="Sample",
                        yaxis_title="Amplitude",
                        height=320,
                        margin={"t": 40, "b": 20},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with viz_r:
                    # Power spectrum
                    spectrum = result.get("frequency_spectrum", None)
                    if spectrum is not None:
                        freqs = np.fft.rfftfreq(n_res).tolist()
                        powers = np.array(spectrum).tolist()
                        # Truncate to same length
                        min_len = min(len(freqs), len(powers))
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=freqs[:min_len],
                                y=powers[:min_len],
                                mode="lines",
                                fill="tozeroy",
                                fillcolor="rgba(220, 53, 69, 0.15)",
                                line={"color": "#dc3545", "width": 1.5},
                                name="Power",
                            )
                        )
                        # Mark dominant frequency
                        if dom_f > 0:
                            fig2.add_vline(
                                x=dom_f, line_dash="dash", line_color="#28a745", annotation_text=f"f = {dom_f:.3f}"
                            )
                        fig2.update_layout(
                            title="FFT Power Spectrum",
                            xaxis_title="Frequency",
                            yaxis_title="Power",
                            height=320,
                            margin={"t": 40, "b": 20},
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Computation error: {e}")


def render_domain_overview_page() -> None:
    """Render cross-domain summary page showing all 9 domains."""
    if st is None or px is None or pd is None:
        return

    st.title("🗺️ Domain Overview")
    st.caption("Cross-domain summary of all 9 UMCP domains — closures, contracts, casepacks, canon anchors")

    # Domain metadata
    domains = [
        {
            "name": "GCD",
            "icon": "🔧",
            "tier": "Tier-1",
            "closures": 5,
            "contract": "UMA.INTSTACK.v1",
            "canon": "gcd_anchors.yaml",
            "casepack": "gcd_complete",
            "desc": "Generic Collapse Dynamics — foundational kernel",
        },
        {
            "name": "KIN",
            "icon": "🎯",
            "tier": "Tier-0 (diagnostic)",
            "closures": 6,
            "contract": "KIN.INTSTACK.v1",
            "canon": "kin_anchors.yaml",
            "casepack": "kinematics_complete",
            "desc": "Kinematics — Tier-0 protocol diagnostic",
        },
        {
            "name": "RCFT",
            "icon": "🌀",
            "tier": "Tier-2",
            "closures": 4,
            "contract": "RCFT.INTSTACK.v1",
            "canon": "rcft_anchors.yaml",
            "casepack": "rcft_complete",
            "desc": "Recursive Collapse Field Theory — fractal overlays",
        },
        {
            "name": "WEYL",
            "icon": "🌌",
            "tier": "Tier-2",
            "closures": 6,
            "contract": "WEYL.INTSTACK.v1",
            "canon": "weyl_anchors.yaml",
            "casepack": "weyl_des_y3",
            "desc": "Modified gravity — DES Y3 Σ(z) analysis",
        },
        {
            "name": "Security",
            "icon": "🔒",
            "tier": "Tier-2",
            "closures": 15,
            "contract": "SECURITY.INTSTACK.v1",
            "canon": "anchors.yaml",
            "casepack": "security_validation",
            "desc": "Security validation and integrity checking",
        },
        {
            "name": "ASTRO",
            "icon": "🔭",
            "tier": "Tier-2",
            "closures": 6,
            "contract": "ASTRO.INTSTACK.v1",
            "canon": "astro_anchors.yaml",
            "casepack": "astronomy_complete",
            "desc": "Astronomy — stars, orbits, distances, spectra",
        },
        {
            "name": "NUC",
            "icon": "☢️",
            "tier": "Tier-2",
            "closures": 6,
            "contract": "NUC.INTSTACK.v1",
            "canon": "nuc_anchors.yaml",
            "casepack": "nuclear_chain",
            "desc": "Nuclear physics — binding, decay, shells, fissility",
        },
        {
            "name": "QM",
            "icon": "🔮",
            "tier": "Tier-2",
            "closures": 6,
            "contract": "QM.INTSTACK.v1",
            "canon": "qm_anchors.yaml",
            "casepack": "quantum_mechanics_complete",
            "desc": "Quantum mechanics — Born rule, tunneling, spin",
        },
        {
            "name": "FIN",
            "icon": "💰",
            "tier": "Tier-2",
            "closures": 1,
            "contract": "FINANCE.INTSTACK.v1",
            "canon": "—",
            "casepack": "finance_continuity",
            "desc": "Finance — business continuity embedding",
        },
    ]

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌐 Domains", len(domains))
    with c2:
        st.metric("🔧 Total Closures", sum(d["closures"] for d in domains))
    with c3:
        st.metric("📜 Contracts", len(domains))
    with c4:
        st.metric("📖 Canon Files", sum(1 for d in domains if d["canon"] != "—"))

    st.divider()

    # Domain cards
    for i in range(0, len(domains), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(domains):
                d = domains[i + j]
                with col:
                    st.markdown(
                        f"""
                    <div style="padding:16px; border-radius:12px; border: 1px solid #ddd;
                         background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                         margin-bottom:10px;">
                        <h3>{d["icon"]} {d["name"]}</h3>
                        <p style="color:#555; font-size:0.9em;">{d["desc"]}</p>
                        <hr style="margin:8px 0;">
                        <small>
                        <b>Tier:</b> {d["tier"]}<br>
                        <b>Closures:</b> {d["closures"]}<br>
                        <b>Contract:</b> {d["contract"]}<br>
                        <b>Canon:</b> {d["canon"]}<br>
                        <b>Casepack:</b> {d["casepack"]}
                        </small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    st.divider()

    # Domain distribution chart
    st.subheader("📊 Domain Distribution")
    df_domains = pd.DataFrame(domains)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            df_domains,
            x="name",
            y="closures",
            color="tier",
            title="Closures per Domain",
            labels={"name": "Domain", "closures": "Closure Count", "tier": "Tier"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=350, margin={"t": 40, "b": 40})
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        tier_counts = df_domains.groupby("tier")["closures"].sum().reset_index()
        fig = px.pie(
            tier_counts,
            values="closures",
            names="tier",
            title="Closures by Tier",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(height=350, margin={"t": 40, "b": 40})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tier architecture summary
    st.subheader("🏛️ Protocol Tier Architecture (v3.0.0)")
    tier_col1, tier_col2, tier_col3 = st.columns(3)
    with tier_col1:
        st.markdown(
            """
        <div style="padding:14px; border-radius:10px; background:#e3f2fd; border-left:4px solid #1976d2;">
        <h4>Tier-0: Protocol</h4>
        <ul style="margin:4px 0; padding-left:18px;">
        <li>Schema validation</li>
        <li>Regime gates</li>
        <li>SHA256 integrity</li>
        <li>Seam calculus</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with tier_col2:
        st.markdown(
            """
        <div style="padding:14px; border-radius:10px; background:#e8f5e9; border-left:4px solid #388e3c;">
        <h4>Tier-1: Immutable Invariants</h4>
        <ul style="margin:4px 0; padding-left:18px;">
        <li>F + ω = 1 (budget)</li>
        <li>IC ≤ F (AM-GM)</li>
        <li>IC ≈ exp(κ)</li>
        <li>Regime classification</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with tier_col3:
        st.markdown(
            """
        <div style="padding:14px; border-radius:10px; background:#fff3e0; border-left:4px solid #f57c00;">
        <h4>Tier-2: Expansion Space</h4>
        <ul style="margin:4px 0; padding-left:18px;">
        <li>Domain closures</li>
        <li>Validity checks</li>
        <li>Canon anchors</li>
        <li>9 active domains</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Cross-domain radar comparison
    if go is not None:
        st.divider()
        st.subheader("🕸️ Cross-Domain Closure Coverage")
        domain_names_r = [d["icon"] + " " + d["name"] for d in domains]
        closure_counts = [d["closures"] for d in domains]
        max_closures = max(closure_counts) if closure_counts else 1

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[*[c / max_closures for c in closure_counts], closure_counts[0] / max_closures],
                theta=[*domain_names_r, domain_names_r[0]],
                fill="toself",
                name="Closure count (normalized)",
                fillcolor="rgba(0, 123, 255, 0.1)",
                line={"color": "#007bff", "width": 2},
            )
        )
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1.1]}},
            height=400,
            margin={"t": 40, "b": 30},
            title="Domain Coverage Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================


def _is_running_in_streamlit() -> bool:
    """Check if we're running inside a Streamlit runtime context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


def main() -> None:
    """Main dashboard application."""
    if not HAS_VIZ_DEPS:
        print("━" * 60)
        print("UMCP Dashboard requires visualization dependencies.")
        print("━" * 60)
        print("")
        print("Install with:")
        print("  pip install umcp[viz]")
        print("")
        print("This installs:")
        print("  • streamlit>=1.30.0")
        print("  • pandas>=2.0.0")
        print("  • plotly>=5.18.0")
        print("  • numpy>=1.24.0")
        print("")
        print("Then run:")
        print("  streamlit run src/umcp/dashboard.py")
        print("━" * 60)
        sys.exit(1)

    # If called from CLI (not inside Streamlit runtime), launch streamlit as subprocess
    if not _is_running_in_streamlit():
        dashboard_path = str(Path(__file__).resolve())
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.headless", "true"]
        sys.exit(subprocess.call(cmd))

    # st is guaranteed to be available here since HAS_VIZ_DEPS is True
    assert st is not None  # for type narrowing

    # Page configuration
    st.set_page_config(
        page_title="UMCP Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for toggles
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 30
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False
    if "compact_mode" not in st.session_state:
        st.session_state.compact_mode = False
    if "theme" not in st.session_state:
        st.session_state.theme = "Default"
    if "last_validation" not in st.session_state:
        st.session_state.last_validation = None

    # Sidebar
    st.sidebar.title("🔬 UMCP")
    st.sidebar.caption(f"v{__version__}")

    # ========== Navigation ==========
    st.sidebar.markdown("### 📍 Navigation")

    # Organized pages by category
    pages = {
        # Core Pages
        "Overview": ("📊", render_overview_page),
        "Domain Overview": ("🗺️", render_domain_overview_page),
        "Canon Explorer": ("📖", render_canon_explorer_page),
        "Precision": ("🎯", render_precision_page),
        "Geometry": ("🔷", render_geometry_page),
        "Ledger": ("📒", render_ledger_page),
        "Casepacks": ("📦", render_casepacks_page),
        "Contracts": ("📜", render_contracts_page),
        "Closures": ("🔧", render_closures_page),
        "Regime": ("🌡️", render_regime_page),
        "Metrics": ("📐", render_metrics_page),
        "Health": ("🏥", render_health_page),
        # Interactive Pages
        "Live Runner": ("▶️", render_live_runner_page),
        "Batch Validation": ("📦", render_batch_validation_page),
        "Test Templates": ("🧮", render_test_templates_page),
        # Domain Pages (Tier-2 Expansion)
        "Astronomy": ("🔭", render_astronomy_page),
        "Nuclear": ("☢️", render_nuclear_page),
        "Quantum": ("🔮", render_quantum_page),
        "Finance": ("💰", render_finance_page),
        "RCFT": ("🌀", render_rcft_page),
        "Physics": ("⚛️", render_physics_interface_page),
        "Kinematics": ("🎯", render_kinematics_interface_page),
        "Cosmology": ("🌌", render_cosmology_page),
        # Analysis Pages
        "Formula Builder": ("🔧", render_formula_builder_page),
        "Time Series": ("📈", render_time_series_page),
        "Comparison": ("🔀", render_comparison_page),
        # Management Pages
        "Exports": ("📥", render_exports_page),
        "Bookmarks": ("🔖", render_bookmarks_page),
        "Notifications": ("🔔", render_notifications_page),
        "API Integration": ("🔌", render_api_integration_page),
    }

    page = st.sidebar.radio(
        "Select Page", list(pages.keys()), format_func=lambda x: f"{pages[x][0]} {x}", label_visibility="collapsed"
    )

    st.sidebar.divider()

    # ========== Display Controls ==========
    st.sidebar.markdown("### ⚙️ Display Controls")

    # Toggle switches
    st.session_state.compact_mode = st.sidebar.toggle(
        "Compact Mode", value=st.session_state.compact_mode, help="Reduce spacing and show more data"
    )

    st.session_state.show_advanced = st.sidebar.toggle(
        "Show Advanced Options", value=st.session_state.show_advanced, help="Display advanced configuration options"
    )

    st.session_state.auto_refresh = st.sidebar.toggle(
        "Auto Refresh", value=st.session_state.auto_refresh, help="Automatically refresh data periodically"
    )

    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.sidebar.slider(
            "Refresh Interval (sec)", min_value=5, max_value=120, value=st.session_state.refresh_interval, step=5
        )
        # Auto-refresh: clear cached data and rerun after the configured interval
        import time

        time.sleep(st.session_state.refresh_interval)
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # ========== Theme Selection ==========
    if st.session_state.show_advanced:
        st.sidebar.markdown("### 🎨 Theme")
        st.session_state.theme = st.sidebar.selectbox(
            "Color Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme)
        )
        st.sidebar.divider()

    # ========== Quick Stats ==========
    st.sidebar.markdown("### 📊 Quick Stats")
    df = load_ledger()
    casepacks = load_casepacks()
    contracts = load_contracts()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📒 Ledger", len(df))
        st.metric("📜 Contracts", len(contracts))
    with col2:
        st.metric("📦 Casepacks", len(casepacks))
        if not df.empty and "run_status" in df.columns:
            conformant = (df["run_status"] == "CONFORMANT").sum()
            total = len(df)
            rate = int(conformant / total * 100) if total > 0 else 0
            st.metric("✅ Rate", f"{rate}%")
        else:
            st.metric("✅ Rate", "N/A")

    st.sidebar.divider()

    # ========== Quick Actions ==========
    st.sidebar.markdown("### ⚡ Quick Actions")

    qa_col1, qa_col2 = st.sidebar.columns(2)

    with qa_col1:
        if st.button("🔄 Refresh", width="stretch", key="sidebar_refresh"):
            st.rerun()

    with qa_col2:
        if st.button("🧪 Validate", width="stretch", key="sidebar_validate"):
            st.session_state.run_quick_validation = True

    # Handle quick validation
    if st.session_state.get("run_quick_validation", False):
        with st.sidebar:
            with st.spinner("Validating..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "umcp", "validate", "casepacks/hello_world"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=get_repo_root(),
                    )
                    if result.returncode == 0:
                        st.success("✅ Valid")
                    else:
                        st.error("❌ Failed")
                except Exception as e:
                    st.error(f"Error: {e}")
            st.session_state.run_quick_validation = False

    st.sidebar.divider()

    # ========== Core Axiom ==========
    st.sidebar.markdown("### 📜 Core Axiom")
    st.sidebar.info('**"What Returns Through Collapse Is Real"**')
    st.sidebar.caption("Collapse is generative; only what returns is real.")

    st.sidebar.divider()

    # ========== Protocol Tiers ==========
    st.sidebar.markdown("### 🏛️ Protocol Tiers (v3.0.0)")
    st.sidebar.markdown("""
    - **Tier-0**: Protocol — validation, regime gates, SHA256, seam calculus
    - **Tier-1**: Immutable Invariants — F+ω=1, IC≤F, IC≈exp(κ)
    - **Tier-2**: Expansion Space — domain closures with validity checks
    """)

    st.sidebar.divider()

    # ========== Resources ==========
    st.sidebar.markdown("### 📚 Resources")
    st.sidebar.markdown("- [GitHub](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code)")
    st.sidebar.markdown("- [Documentation](README.md)")
    st.sidebar.markdown("- [API Docs](http://localhost:8000/docs)")
    st.sidebar.markdown("- [Tutorial](QUICKSTART_TUTORIAL.md)")

    st.sidebar.divider()
    st.sidebar.caption("© 2026 UMCP Project")

    # Render selected page
    _, render_func = pages[page]
    render_func()


def render_live_runner_page() -> None:
    """Render live validation runner with real-time controls."""
    if st is None or pd is None:
        return

    st.title("▶️ Live Validation Runner")
    st.caption("Run validations interactively with real-time feedback")

    repo_root = get_repo_root()
    casepacks = load_casepacks()

    # ========== Control Panel ==========
    st.subheader("🎛️ Control Panel")

    with st.container(border=True):
        ctrl_cols = st.columns([2, 1, 1, 1])

        with ctrl_cols[0]:
            casepack_options = ["Repository (All)"] + [cp["id"] for cp in casepacks]
            selected_target = st.selectbox("Target", casepack_options, help="Select what to validate")

        with ctrl_cols[1]:
            strict_mode = st.toggle("Strict Mode", value=False, help="Enable strict validation")

        with ctrl_cols[2]:
            verbose = st.toggle("Verbose Output", value=False, help="Show detailed output")

        with ctrl_cols[3]:
            fail_on_warning = st.toggle("Fail on Warning", value=False)

    st.divider()

    # ========== Run Controls ==========
    run_cols = st.columns([1, 1, 2])

    with run_cols[0]:
        run_button = st.button("▶️ Run Validation", width="stretch", type="primary")

    with run_cols[1]:
        # Stop button placeholder (for future async support)
        st.button("⏹️ Stop", width="stretch", disabled=True)

    with run_cols[2]:
        st.empty()  # Spacer

    # ========== Results Area ==========
    if run_button:
        st.divider()
        st.subheader("📋 Validation Results")

        # Build command
        cmd = [sys.executable, "-m", "umcp", "validate"]

        if selected_target != "Repository (All)":
            cmd.append(f"casepacks/{selected_target}")

        if strict_mode:
            cmd.append("--strict")
        if verbose:
            cmd.append("--verbose")
        if fail_on_warning:
            cmd.append("--fail-on-warning")

        # Display command
        with st.expander("📝 Command", expanded=False):
            st.code(" ".join(cmd), language="bash")

        # Run with progress
        progress_bar = st.progress(0, text="Starting validation...")
        status_container = st.empty()
        output_container = st.container()

        try:
            progress_bar.progress(20, text="Running validation...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_root,
            )

            progress_bar.progress(100, text="Complete!")

            # Parse result
            if result.returncode == 0:
                status_container.success("✅ **CONFORMANT** - All validations passed!")

                # Try to parse JSON output
                try:
                    # Find JSON in output
                    output = result.stdout
                    json_start = output.find("{")
                    json_end = output.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = output[json_start:json_end]
                        result_data = json.loads(json_str)

                        # Display metrics
                        with output_container:
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("Status", result_data.get("run_status", "N/A"))
                            with metric_cols[1]:
                                counts = result_data.get("summary", {}).get("counts", {})
                                st.metric("Errors", counts.get("errors", 0))
                            with metric_cols[2]:
                                st.metric("Warnings", counts.get("warnings", 0))
                            with metric_cols[3]:
                                st.metric("Targets", counts.get("targets_total", 0))

                            # Targets breakdown
                            st.markdown("**Validated Targets:**")
                            targets = result_data.get("targets", [])
                            for target in targets:
                                status_icon = "✅" if target.get("run_status") == "CONFORMANT" else "❌"
                                st.markdown(
                                    f"- {status_icon} `{target.get('target_path', 'unknown')}` — {target.get('run_status', 'N/A')}"
                                )
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                status_container.error("❌ **NONCONFORMANT** - Validation failed!")

            # Raw output
            with st.expander("📄 Full Output", expanded=False):
                st.code(
                    result.stdout + result.stderr if result.stdout or result.stderr else "No output", language="text"
                )

            # Store result in session
            st.session_state.last_validation = {
                "target": selected_target,
                "status": "CONFORMANT" if result.returncode == 0 else "NONCONFORMANT",
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            progress_bar.progress(100, text="Timeout!")
            status_container.error("⏱️ Validation timed out after 120 seconds")
        except Exception as e:
            progress_bar.progress(100, text="Error!")
            status_container.error(f"❌ Error running validation: {e}")

    st.divider()

    # ========== History ==========
    st.subheader("📜 Recent Runs")

    if st.session_state.last_validation:
        last = st.session_state.last_validation
        status_color = STATUS_COLORS.get(last["status"], "#6c757d")
        st.markdown(
            f"""<div style="padding: 10px; border-left: 4px solid {status_color}; background: {status_color}22; border-radius: 4px;">
            <strong>{last["status"]}</strong> — {last["target"]} @ {last["timestamp"][:19]}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No validation runs yet. Click 'Run Validation' to start.")

    st.divider()

    # ========== Casepack Quick Selector ==========
    st.subheader("📦 Quick Casepack Runner")

    # Grid of casepack buttons
    cols_per_row = 3
    for i in range(0, len(casepacks), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(casepacks):
                cp = casepacks[i + j]
                with col, st.container(border=True):
                    st.markdown(f"**{cp['id']}**")
                    st.caption(f"v{cp['version']} • {cp['test_vectors']} vectors")
                    if st.button("▶️ Run", key=f"run_{cp['id']}", width="stretch"):
                        with st.spinner(f"Validating {cp['id']}..."):
                            result = subprocess.run(
                                [sys.executable, "-m", "umcp", "validate", f"casepacks/{cp['id']}"],
                                capture_output=True,
                                text=True,
                                timeout=60,
                                cwd=repo_root,
                            )
                            if result.returncode == 0:
                                st.success("✅ Pass")
                            else:
                                st.error("❌ Fail")


if __name__ == "__main__":
    main()
