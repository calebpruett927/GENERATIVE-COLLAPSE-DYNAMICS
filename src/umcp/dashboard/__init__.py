"""
UMCP Visualization Dashboard - Streamlit Communication Extension

Interactive web dashboard for exploring UMCP validation results,
ledger data, casepacks, contracts, closures, and kernel metrics.

This is an optional extension that requires: pip install umcp[viz]

Usage:
  umcp-dashboard
  streamlit run src/umcp/dashboard/__init__.py

Features:
  - Real-time system health monitoring (31 pages)
  - Interactive ledger exploration with anomaly detection
  - Casepack browser with validation status
  - Regime phase space visualization with trajectories
  - Kernel metrics analysis with trends and correlations
  - Contract and closure exploration with code preview
  - Physics, Kinematics, Cosmology, Astronomy domain interfaces
  - Geometry, Canon Explorer, Precision analysis
  - Live validation runner, batch validation, test templates

Cross-references:
  - EXTENSION_INTEGRATION.md (extension architecture)
  - src/umcp/api_umcp.py (REST API extension)
  - src/umcp/validator.py (validation engine)
  - ledger/return_log.csv (validation ledger)
  - KERNEL_SPECIFICATION.md (regime definitions)

Package structure:
  - _deps.py: Optional dependency imports (streamlit, pandas, plotly, numpy)
  - _utils.py: Shared constants, data loaders, helper functions
  - pages_core.py: Overview, Ledger, Casepacks, Contracts, Closures, Regime, Metrics, Health
  - pages_physics.py: GCD framework, Physics interface, Kinematics interface
  - pages_interactive.py: Test Templates, Batch Validation, Live Runner
  - pages_analysis.py: Exports, Comparison, Time Series, Formula Builder
  - pages_management.py: Notifications, Bookmarks, API Integration
  - pages_science.py: Cosmology, Astronomy, Nuclear, Quantum, Finance, RCFT
  - pages_advanced.py: Precision, Geometry, Canon Explorer, Domain Overview
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Shared deps & utilities ──────────────────────────────────────────────────
from umcp.dashboard._deps import HAS_VIZ_DEPS, st
from umcp.dashboard._utils import (
    KERNEL_SYMBOLS,
    REGIME_COLORS,
    STATUS_COLORS,
    THEMES,
    _cache_data,
    _ensure_closures_path,
    classify_regime,
    detect_anomalies,
    format_bytes,
    get_regime_color,
    get_repo_root,
    get_trend_indicator,
    load_casepacks,
    load_closures,
    load_contracts,
    load_ledger,
)

# ── Import UMCP version ─────────────────────────────────────────────────────
try:
    from umcp import __version__
except ImportError:
    __version__ = "2.0.0"

# ── Page render functions (lazy-imported from submodules) ────────────────────
from umcp.dashboard.pages_advanced import (
    render_canon_explorer_page,
    render_domain_overview_page,
    render_geometry_page,
    render_layer1_state_space,
    render_layer2_projections,
    render_layer3_seam_graph,
    render_precision_page,
    render_unified_geometry_view,
)
from umcp.dashboard.pages_analysis import (
    render_comparison_page,
    render_exports_page,
    render_formula_builder_page,
    render_time_series_page,
)
from umcp.dashboard.pages_core import (
    render_casepacks_page,
    render_closures_page,
    render_contracts_page,
    render_health_page,
    render_ledger_page,
    render_metrics_page,
    render_overview_page,
    render_regime_page,
)
from umcp.dashboard.pages_interactive import (
    render_batch_validation_page,
    render_live_runner_page,
    render_test_templates_page,
)
from umcp.dashboard.pages_management import (
    render_api_integration_page,
    render_bookmarks_page,
    render_notifications_page,
)
from umcp.dashboard.pages_physics import (
    GCD_AXIOMS,
    GCD_REGIMES,
    GCD_SYMBOLS,
    PHYSICS_QUANTITIES,
    convert_from_base_unit,
    convert_to_base_unit,
    normalize_to_bounded,
    render_gcd_panel,
    render_kinematics_interface_page,
    render_physics_interface_page,
    translate_to_gcd,
)
from umcp.dashboard.pages_science import (
    render_astronomy_page,
    render_cosmology_page,
    render_finance_page,
    render_nuclear_page,
    render_quantum_page,
    render_rcft_page,
)

# ── Public API surface (backward-compatible) ─────────────────────────────────
__all__ = [
    "GCD_AXIOMS",
    "GCD_REGIMES",
    "GCD_SYMBOLS",
    "HAS_VIZ_DEPS",
    # Constants
    "KERNEL_SYMBOLS",
    "PHYSICS_QUANTITIES",
    "REGIME_COLORS",
    "STATUS_COLORS",
    "THEMES",
    "_cache_data",
    "_ensure_closures_path",
    # Utilities
    "classify_regime",
    "convert_from_base_unit",
    "convert_to_base_unit",
    "detect_anomalies",
    "format_bytes",
    "get_regime_color",
    "get_repo_root",
    "get_trend_indicator",
    # Data loaders
    "load_casepacks",
    "load_closures",
    "load_contracts",
    "load_ledger",
    # Entry point
    "main",
    "normalize_to_bounded",
    "render_api_integration_page",
    "render_astronomy_page",
    "render_batch_validation_page",
    "render_bookmarks_page",
    "render_canon_explorer_page",
    "render_casepacks_page",
    "render_closures_page",
    "render_comparison_page",
    "render_contracts_page",
    # Science pages
    "render_cosmology_page",
    "render_domain_overview_page",
    # Analysis pages
    "render_exports_page",
    "render_finance_page",
    "render_formula_builder_page",
    "render_gcd_panel",
    "render_geometry_page",
    "render_health_page",
    "render_kinematics_interface_page",
    "render_layer1_state_space",
    "render_layer2_projections",
    "render_layer3_seam_graph",
    "render_ledger_page",
    "render_live_runner_page",
    "render_metrics_page",
    # Management pages
    "render_notifications_page",
    "render_nuclear_page",
    # Core pages
    "render_overview_page",
    # Physics pages
    "render_physics_interface_page",
    # Advanced pages
    "render_precision_page",
    "render_quantum_page",
    "render_rcft_page",
    "render_regime_page",
    # Interactive pages
    "render_test_templates_page",
    "render_time_series_page",
    "render_unified_geometry_view",
    # Physics helpers
    "translate_to_gcd",
]


# ── Streamlit runtime detection ──────────────────────────────────────────────
def _is_running_in_streamlit() -> bool:
    """Check if we're running inside a Streamlit runtime context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


# ── Main application ─────────────────────────────────────────────────────────
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
        print("  umcp-dashboard")
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
    pages: dict[str, tuple[str, Any]] = {
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
    st.sidebar.markdown("- [GitHub](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS)")
    st.sidebar.markdown("- [Documentation](README.md)")
    st.sidebar.markdown("- [API Docs](http://localhost:8000/docs)")
    st.sidebar.markdown("- [Tutorial](QUICKSTART_TUTORIAL.md)")

    st.sidebar.divider()
    st.sidebar.caption("© 2026 UMCP Project")

    # Render selected page
    _, render_func = pages[page]
    render_func()


# ── Streamlit direct execution ───────────────────────────────────────────────
if _is_running_in_streamlit():
    main()
