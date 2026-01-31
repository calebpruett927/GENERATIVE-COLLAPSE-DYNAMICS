"""
UMCP Visualization Dashboard - Streamlit Communication Extension

Interactive web dashboard for exploring UMCP validation results,
ledger data, casepacks, and kernel metrics.

This is an optional extension that requires: pip install umcp[viz]

Usage:
  streamlit run src/umcp/dashboard.py

Cross-references:
  - EXTENSION_INTEGRATION.md (extension architecture)
  - src/umcp/api_umcp.py (REST API extension)
  - src/umcp/validator.py (validation engine)
  - ledger/return_log.csv (validation ledger)

Note: This module uses optional visualization dependencies (pandas, plotly, streamlit)
that may not be installed. Type errors for these are suppressed with # type: ignore.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Try to import visualization dependencies
try:
    import pandas as pd  # type: ignore[import-untyped]
    import plotly.express as px  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    import streamlit as st  # type: ignore[import-untyped]

    HAS_VIZ_DEPS = True
except ImportError:
    HAS_VIZ_DEPS = False  # type: ignore[misc]
    pd = None  # type: ignore[assignment]
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd

# Import UMCP core modules
try:
    from . import __version__
except ImportError:
    __version__ = "1.5.0"


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


def load_ledger() -> Any:
    """Load the return log ledger as a DataFrame."""
    if pd is None:
        raise ImportError("pandas not installed. Run: pip install umcp[viz]")

    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        return pd.DataFrame()  # type: ignore[union-attr]

    df = pd.read_csv(ledger_path)  # type: ignore[union-attr]

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  # type: ignore[union-attr]

    return df


def load_casepacks() -> list[dict[str, Any]]:
    """Load casepack information."""
    repo_root = get_repo_root()
    casepacks_dir = repo_root / "casepacks"

    if not casepacks_dir.exists():
        return []

    casepacks: list[dict[str, Any]] = []
    for casepack_dir in sorted(casepacks_dir.iterdir()):
        if not casepack_dir.is_dir():
            continue

        # Try to load manifest
        manifest_path = casepack_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = casepack_dir / "manifest.yaml"

        casepack_info: dict[str, Any] = {
            "id": casepack_dir.name,
            "path": str(casepack_dir),
            "version": "unknown",
            "description": None,
        }

        if manifest_path.exists():
            try:
                if manifest_path.suffix == ".json":
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    # Simple YAML parsing
                    import yaml

                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)

                if manifest and "casepack" in manifest:
                    cp = manifest["casepack"]
                    casepack_info["id"] = cp.get("id", casepack_dir.name)
                    casepack_info["version"] = cp.get("version", "unknown")
                    casepack_info["description"] = cp.get("description")
            except Exception:
                pass

        casepacks.append(casepack_info)

    return casepacks


def load_contracts() -> list[dict[str, str]]:
    """Load contract information."""
    repo_root = get_repo_root()
    contracts_dir = repo_root / "contracts"

    if not contracts_dir.exists():
        return []

    contracts: list[dict[str, str]] = []
    for contract_path in sorted(contracts_dir.glob("*.yaml")):
        filename = contract_path.stem
        parts = filename.split(".")
        domain = parts[0] if parts else "unknown"
        version = parts[-1] if len(parts) > 1 and parts[-1].startswith("v") else "v1"

        contracts.append(
            {
                "id": filename,
                "domain": domain,
                "version": version,
                "path": str(contract_path),
            }
        )

    return contracts


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
    colors = {
        "STABLE": "#28a745",  # Green
        "WATCH": "#ffc107",  # Yellow
        "COLLAPSE": "#dc3545",  # Red
        "CRITICAL": "#6f42c1",  # Purple
    }
    return colors.get(regime, "#6c757d")


# ============================================================================
# Dashboard Pages
# ============================================================================


def render_overview_page() -> None:
    """Render the main overview page."""
    if st is None:
        return

    st.title("🔬 UMCP Dashboard")
    st.markdown(f"**Version**: {__version__} | **Schema**: UMCP.v1")

    # Load data
    df = load_ledger()
    casepacks = load_casepacks()
    contracts = load_contracts()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Ledger Entries", len(df) if not df.empty else 0)

    with col2:
        st.metric("Casepacks", len(casepacks))

    with col3:
        st.metric("Contracts", len(contracts))

    with col4:
        conformant_count = 0
        if not df.empty and "run_status" in df.columns:
            conformant_count = int((df["run_status"] == "CONFORMANT").sum())
        st.metric("Conformant Runs", conformant_count)

    st.divider()

    # Recent activity
    st.subheader("📊 Recent Validation Activity")

    if df.empty:
        st.info("No ledger data available. Run some validations to populate the ledger.")
    else:
        # Status distribution
        if "run_status" in df.columns:
            status_counts = df["run_status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Validation Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Timeline
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df_sorted = df.sort_values("timestamp")
            fig = px.line(
                df_sorted.tail(100),
                x="timestamp",
                y="omega" if "omega" in df.columns else df.columns[1],
                title="Recent Validation Timeline (Last 100)",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_ledger_page() -> None:
    """Render the ledger exploration page."""
    if st is None:
        return

    st.title("📒 Validation Ledger")

    df = load_ledger()

    if df.empty:
        st.warning("No ledger data found.")
        return

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        if "run_status" in df.columns:
            status_filter = st.multiselect(
                "Filter by Status",
                options=df["run_status"].unique().tolist(),
                default=df["run_status"].unique().tolist(),
            )
            df = df[df["run_status"].isin(status_filter)]

    with col2:
        limit = st.slider("Rows to Display", 10, 500, 100)

    # Display table
    st.dataframe(df.tail(limit), use_container_width=True)

    # Stats
    st.subheader("📈 Statistics")
    col1, col2, col3 = st.columns(3)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for i, col in enumerate(numeric_cols[:3]):
        with [col1, col2, col3][i]:
            st.metric(f"Mean {col}", f"{df[col].mean():.4f}")
            st.metric(f"Std {col}", f"{df[col].std():.4f}")

    # Download option
    st.download_button(
        "Download Filtered Ledger (CSV)",
        data=df.to_csv(index=False),
        file_name="umcp_ledger_filtered.csv",
        mime="text/csv",
    )


def render_casepacks_page() -> None:
    """Render the casepacks page."""
    if st is None:
        return

    st.title("📦 Casepacks")

    casepacks = load_casepacks()

    if not casepacks:
        st.warning("No casepacks found.")
        return

    # Display as cards
    for cp in casepacks:
        with st.expander(f"📦 {cp['id']} (v{cp['version']})"):
            st.write(f"**Path**: `{cp['path']}`")
            if cp.get("description"):
                st.write(f"**Description**: {cp['description']}")

            # Check for test vectors
            test_file = Path(cp["path"]) / "test_vectors.csv"
            if test_file.exists():
                with open(test_file) as f:
                    test_count = sum(1 for _ in f) - 1
                st.write(f"**Test Vectors**: {test_count}")

            # Check for closures
            closures_dir = Path(cp["path"]) / "closures"
            if closures_dir.exists():
                closure_files = list(closures_dir.glob("*.py"))
                st.write(f"**Closures**: {len(closure_files)}")


def render_contracts_page() -> None:
    """Render the contracts page."""
    if st is None:
        return

    st.title("📜 Contracts")

    contracts = load_contracts()

    if not contracts:
        st.warning("No contracts found.")
        return

    # Group by domain
    domains: dict[str, list[dict[str, str]]] = {}
    for c in contracts:
        domain = c["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(c)

    for domain, domain_contracts in domains.items():
        st.subheader(f"🏷️ {domain}")
        for c in domain_contracts:
            st.write(f"- **{c['id']}** (v{c['version']})")


def render_regime_page() -> None:
    """Render the regime classification page."""
    if st is None or go is None:
        return

    st.title("🌡️ Regime Classification")

    st.markdown("""
    Classify the computational regime based on kernel invariants.
    The regime determines system stability and operational parameters.
    """)

    # Interactive classifier
    col1, col2 = st.columns(2)

    with col1:
        omega = st.slider("Overlap Fraction (ω)", 0.0, 1.0, 0.5, 0.01)
    with col2:
        seam = st.slider("Seam Residual (s)", -0.05, 0.05, 0.0, 0.001)

    regime = classify_regime(omega, seam)
    color = get_regime_color(regime)

    st.markdown(
        f"<h2 style='color: {color}; text-align: center;'>Regime: {regime}</h2>",
        unsafe_allow_html=True,
    )

    # Regime boundaries visualization
    st.subheader("📊 Regime Phase Space")

    # Create phase space plot
    import numpy as np

    omega_range = np.linspace(0, 1, 100)
    seam_range = np.linspace(-0.05, 0.05, 100)

    regime_map = []
    for o in omega_range:
        row = []
        for s in seam_range:
            r = classify_regime(float(o), float(s))
            if r == "STABLE":
                row.append(1)
            elif r == "WATCH":
                row.append(2)
            elif r == "COLLAPSE":
                row.append(3)
            else:  # CRITICAL
                row.append(4)
        regime_map.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=regime_map,
            x=seam_range,
            y=omega_range,
            colorscale=[
                [0, "#28a745"],  # STABLE
                [0.33, "#ffc107"],  # WATCH
                [0.66, "#dc3545"],  # COLLAPSE
                [1, "#6f42c1"],  # CRITICAL
            ],
            showscale=False,
        )
    )

    # Add current point
    fig.add_trace(
        go.Scatter(
            x=[seam],
            y=[omega],
            mode="markers",
            marker={"size": 15, "color": "white", "line": {"width": 3, "color": "black"}},
            name="Current",
        )
    )

    fig.update_layout(
        title="Regime Phase Space",
        xaxis_title="Seam Residual (s)",
        yaxis_title="Overlap Fraction (ω)",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    **Regime Definitions:**
    - 🟢 **STABLE**: ω ∈ [0.3, 0.7], |s| ≤ 0.005
    - 🟡 **WATCH**: ω ∈ [0.1, 0.3) ∪ (0.7, 0.9], |s| ≤ 0.01
    - 🔴 **COLLAPSE**: ω < 0.1 or ω > 0.9
    - 🟣 **CRITICAL**: |s| > 0.01
    """)


def render_metrics_page() -> None:
    """Render the kernel metrics page."""
    if st is None:
        return

    st.title("📐 Kernel Metrics")

    df = load_ledger()

    if df.empty:
        st.warning("No ledger data available for metrics analysis.")
        return

    # Select metrics to visualize
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found in ledger.")
        return

    selected_metrics = st.multiselect(
        "Select Metrics to Visualize",
        options=numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
    )

    if not selected_metrics:
        return

    # Time series
    st.subheader("📈 Time Series")
    if "timestamp" in df.columns:
        for metric in selected_metrics:
            fig = px.line(
                df.sort_values("timestamp"),
                x="timestamp",
                y=metric,
                title=f"{metric} Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Distributions
    st.subheader("📊 Distributions")
    cols = st.columns(min(3, len(selected_metrics)))
    for i, metric in enumerate(selected_metrics[:3]):
        with cols[i]:
            fig = px.histogram(df, x=metric, title=f"{metric} Distribution", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    if len(selected_metrics) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr = df[selected_metrics].corr()
        fig = px.imshow(
            corr,
            title="Metric Correlations",
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================


def main() -> None:
    """Main dashboard application."""
    if not HAS_VIZ_DEPS:
        print("Visualization dependencies not installed.")
        print("Install with: pip install umcp[viz]")
        print("")
        print("This requires:")
        print("  - streamlit>=1.30.0")
        print("  - pandas>=2.0.0")
        print("  - plotly>=5.18.0")
        sys.exit(1)

    if st is None:
        return

    # Page configuration
    st.set_page_config(
        page_title="UMCP Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("🔬 UMCP")
    st.sidebar.markdown(f"v{__version__}")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Ledger", "Casepacks", "Contracts", "Regime", "Metrics"],
    )

    # Render selected page
    if page == "Overview":
        render_overview_page()
    elif page == "Ledger":
        render_ledger_page()
    elif page == "Casepacks":
        render_casepacks_page()
    elif page == "Contracts":
        render_contracts_page()
    elif page == "Regime":
        render_regime_page()
    elif page == "Metrics":
        render_metrics_page()

    # Footer
    st.sidebar.divider()  # type: ignore[union-attr]
    st.sidebar.markdown(  # type: ignore[union-attr]
        "© 2026 UMCP | [Docs](/docs) | [API](/api)"
    )


if __name__ == "__main__":
    main()
