"""
UMCP Visualization Dashboard - Streamlit Communication Extension

Interactive web dashboard for exploring UMCP validation results,
ledger data, casepacks, contracts, closures, and kernel metrics.

This is an optional extension that requires: pip install umcp[viz]

Usage:
  umcp-dashboard
  streamlit run src/umcp/dashboard/__init__.py

Features:
  - Real-time system health monitoring (45 pages)
  - Interactive ledger exploration with anomaly detection
  - Casepack browser with validation status
  - Regime phase space visualization with trajectories
  - Kernel metrics analysis with trends and correlations
  - Contract and closure exploration with code preview
  - Physics, Kinematics, Cosmology, Astronomy domain interfaces
  - Evolution, Brain Kernel, Awareness Manifold, Cognitive Traversal
  - Geometry, Canon Explorer, Precision analysis
  - Live validation runner, batch validation, test templates

Cross-references:
  - docs/EXTENSION_INTEGRATION.md (extension architecture)
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
  - pages_science.py: Cosmology, Astronomy, Nuclear, Quantum, Finance, RCFT, Materials Science, Security
  - pages_advanced.py: Precision, Geometry, Canon Explorer, Domain Overview
  - pages_diagnostic.py: τ_R* Diagnostic, Epistemic Classification, Insights Engine
  - pages_exploration.py: Rosetta Translation, Orientation Protocol, Everyday Physics
  - pages_evolution.py: Evolution Kernel, Brain Kernel, Awareness Manifold, Cognitive Traversal
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Shared deps & utilities ──────────────────────────────────────────────────
from umcp.dashboard._deps import HAS_VIZ_DEPS, st
from umcp.dashboard._utils import (
    KERNEL_SYMBOLS,
    NAV_CATEGORIES,
    NAV_CATEGORY_ICONS,
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
    inject_custom_css,
    load_casepacks,
    load_closures,
    load_contracts,
    load_ledger,
    metric_row,
    page_header,
    regime_badge,
    section_divider,
)

# ── Import UMCP version ─────────────────────────────────────────────────────
try:
    from umcp import __version__
except ImportError:
    __version__ = "2.2.0"

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
from umcp.dashboard.pages_closures import render_closure_explorer_page
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
from umcp.dashboard.pages_diagnostic import (
    render_epistemic_page,
    render_insights_page,
    render_tau_r_star_page,
)
from umcp.dashboard.pages_evolution import (
    render_awareness_manifold_page,
    render_brain_kernel_page,
    render_cognitive_traversal_page,
    render_evolution_kernel_page,
)
from umcp.dashboard.pages_exploration import (
    render_everyday_physics_page,
    render_orientation_page,
    render_rosetta_page,
)
from umcp.dashboard.pages_interactive import (
    render_batch_validation_page,
    render_live_runner_page,
    render_test_templates_page,
    render_work_template_page,
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
    render_atomic_physics_page,
    render_cosmology_page,
    render_finance_page,
    render_materials_science_page,
    render_nuclear_page,
    render_quantum_page,
    render_rcft_page,
    render_security_page,
    render_standard_model_page,
)

# ── Public API surface (backward-compatible) ─────────────────────────────────
__all__ = [
    "GCD_AXIOMS",
    "GCD_REGIMES",
    "GCD_SYMBOLS",
    "HAS_VIZ_DEPS",
    "KERNEL_SYMBOLS",
    "NAV_CATEGORIES",
    "NAV_CATEGORY_ICONS",
    "PHYSICS_QUANTITIES",
    "REGIME_COLORS",
    "STATUS_COLORS",
    "THEMES",
    "_cache_data",
    "_ensure_closures_path",
    "classify_regime",
    "convert_from_base_unit",
    "convert_to_base_unit",
    "detect_anomalies",
    "format_bytes",
    "get_regime_color",
    "get_repo_root",
    "get_trend_indicator",
    "inject_custom_css",
    "load_casepacks",
    "load_closures",
    "load_contracts",
    "load_ledger",
    "main",
    "metric_row",
    "normalize_to_bounded",
    "page_header",
    "regime_badge",
    "render_api_integration_page",
    "render_astronomy_page",
    "render_atomic_physics_page",
    "render_awareness_manifold_page",
    "render_batch_validation_page",
    "render_bookmarks_page",
    "render_brain_kernel_page",
    "render_canon_explorer_page",
    "render_casepacks_page",
    "render_closure_explorer_page",
    "render_closures_page",
    "render_cognitive_traversal_page",
    "render_comparison_page",
    "render_contracts_page",
    "render_cosmology_page",
    "render_domain_overview_page",
    "render_epistemic_page",
    "render_everyday_physics_page",
    "render_evolution_kernel_page",
    "render_exports_page",
    "render_finance_page",
    "render_formula_builder_page",
    "render_gcd_panel",
    "render_geometry_page",
    "render_health_page",
    "render_insights_page",
    "render_kinematics_interface_page",
    "render_layer1_state_space",
    "render_layer2_projections",
    "render_layer3_seam_graph",
    "render_ledger_page",
    "render_live_runner_page",
    "render_materials_science_page",
    "render_metrics_page",
    "render_notifications_page",
    "render_nuclear_page",
    "render_orientation_page",
    "render_overview_page",
    "render_physics_interface_page",
    "render_precision_page",
    "render_quantum_page",
    "render_rcft_page",
    "render_regime_page",
    "render_rosetta_page",
    "render_security_page",
    "render_standard_model_page",
    "render_tau_r_star_page",
    "render_test_templates_page",
    "render_time_series_page",
    "render_unified_geometry_view",
    "render_work_template_page",
    "section_divider",
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
        port = os.environ.get("UMCP_DASHBOARD_PORT", "8001")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path,
            "--server.headless",
            "true",
            "--server.port",
            port,
            "--server.address",
            "0.0.0.0",
        ]
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

    # Inject shared CSS
    inject_custom_css()

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

    # ── Page registry (icon → render function) ───────────────────────────────
    pages: dict[str, tuple[str, Any]] = {
        # Core
        "Overview": ("📊", render_overview_page),
        "Domain Overview": ("🗺️", render_domain_overview_page),
        "Health": ("🏥", render_health_page),
        "Ledger": ("📒", render_ledger_page),
        "Metrics": ("📐", render_metrics_page),
        # Science Domains
        "Cosmology": ("🌌", render_cosmology_page),
        "Astronomy": ("🔭", render_astronomy_page),
        "Nuclear": ("☢️", render_nuclear_page),
        "Quantum": ("🔮", render_quantum_page),
        "Atomic Physics": ("⚛️", render_atomic_physics_page),
        "Standard Model": ("🔬", render_standard_model_page),
        "Materials Science": ("🧱", render_materials_science_page),
        "Finance": ("💰", render_finance_page),
        "RCFT": ("🌀", render_rcft_page),
        "Security": ("🛡️", render_security_page),
        "Everyday Physics": ("🌡️", render_everyday_physics_page),
        # Evolution & Cognition
        "Evolution Kernel": ("🧬", render_evolution_kernel_page),
        "Brain Kernel": ("🧠", render_brain_kernel_page),
        "Awareness Manifold": ("💠", render_awareness_manifold_page),
        "Cognitive Traversal": ("🚀", render_cognitive_traversal_page),
        # Analysis
        "Regime": ("🌡️", render_regime_page),
        "Time Series": ("📈", render_time_series_page),
        "Comparison": ("🔀", render_comparison_page),
        "Formula Builder": ("🧮", render_formula_builder_page),
        "Precision": ("🎯", render_precision_page),
        # Exploration
        "Canon Explorer": ("📖", render_canon_explorer_page),
        "Geometry": ("🔷", render_geometry_page),
        "Rosetta Translation": ("🌐", render_rosetta_page),
        "Orientation Protocol": ("🧭", render_orientation_page),
        "Physics": ("⚗️", render_physics_interface_page),
        "Kinematics": ("🎯", render_kinematics_interface_page),
        # Tools
        "Casepacks": ("📦", render_casepacks_page),
        "Contracts": ("📜", render_contracts_page),
        "Closures": ("🔧", render_closures_page),
        "Closure Explorer": ("🗂️", render_closure_explorer_page),
        "Live Runner": ("▶️", render_live_runner_page),
        "Batch Validation": ("📋", render_batch_validation_page),
        "Test Templates": ("🧪", render_test_templates_page),
        "Work Templates": ("📐", render_work_template_page),
        # Diagnostics
        "τ_R* Diagnostic": ("🩺", render_tau_r_star_page),
        "Epistemic": ("🧿", render_epistemic_page),
        "Insights": ("💡", render_insights_page),
        # Manage
        "Exports": ("📥", render_exports_page),
        "Bookmarks": ("🔖", render_bookmarks_page),
        "Notifications": ("🔔", render_notifications_page),
        "API Integration": ("🔌", render_api_integration_page),
    }

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════

    # ── Branding ─────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="title">🔬 UMCP</div>
            <div class="version">v{__version__} · 46 pages · 15 domains</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick Actions (compact) ──────────────────────────────────────────────
    qa_col1, qa_col2 = st.sidebar.columns(2)
    with qa_col1:
        if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh"):
            st.rerun()
    with qa_col2:
        if st.button("🧪 Validate", use_container_width=True, key="sidebar_validate"):
            st.session_state.run_quick_validation = True

    # Handle quick validation inline
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
                        st.success("✅ CONFORMANT")
                    else:
                        st.error("❌ NONCONFORMANT")
                except Exception as e:
                    st.error(f"Error: {e}")
            st.session_state.run_quick_validation = False

    st.sidebar.divider()

    # ── Categorized Navigation ───────────────────────────────────────────────
    # Build flat list of all page names for sidebar radio
    all_page_names: list[str] = []
    for cat_name in NAV_CATEGORIES:
        all_page_names.extend(NAV_CATEGORIES[cat_name])

    # Initialize selected page in session state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Overview"

    # Render categorized navigation using expanders
    for cat_name, cat_pages in NAV_CATEGORIES.items():
        cat_icon = NAV_CATEGORY_ICONS.get(cat_name, "📁")
        is_active_cat = st.session_state.selected_page in cat_pages
        with st.sidebar.expander(
            f"{cat_icon} {cat_name}",
            expanded=is_active_cat,
        ):
            for pg_name in cat_pages:
                if pg_name not in pages:
                    continue
                icon, _ = pages[pg_name]
                is_current = st.session_state.selected_page == pg_name
                label = f"**{icon} {pg_name}**" if is_current else f"{icon} {pg_name}"
                if st.button(
                    label,
                    key=f"nav_{pg_name}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.selected_page = pg_name
                    st.rerun()

    st.sidebar.divider()

    # ── Quick Stats (compact) ────────────────────────────────────────────────
    df = load_ledger()
    casepacks = load_casepacks()

    with st.sidebar.container():
        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            st.metric("📒", len(df), help="Ledger entries")
        with c2:
            st.metric("📦", len(casepacks), help="Casepacks")
        with c3:
            if not df.empty and "run_status" in df.columns:
                conformant = (df["run_status"] == "CONFORMANT").sum()
                total = len(df)
                rate = int(conformant / total * 100) if total > 0 else 0
                st.metric("✅", f"{rate}%", help="Conformance rate")
            else:
                st.metric("✅", "N/A", help="Conformance rate")

    # ── Display Controls ─────────────────────────────────────────────────────
    with st.sidebar.expander("⚙️ Display", expanded=False):
        st.session_state.compact_mode = st.toggle(
            "Compact Mode",
            value=st.session_state.compact_mode,
            help="Reduce spacing and show more data",
        )
        st.session_state.show_advanced = st.toggle(
            "Advanced Options",
            value=st.session_state.show_advanced,
            help="Display advanced configuration options",
        )
        st.session_state.auto_refresh = st.toggle(
            "Auto Refresh",
            value=st.session_state.auto_refresh,
            help="Automatically refresh data periodically",
        )
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider(
                "Interval (sec)",
                5,
                120,
                st.session_state.refresh_interval,
                5,
            )
            import time

            time.sleep(st.session_state.refresh_interval)
            st.cache_data.clear()
            st.rerun()

        if st.session_state.show_advanced:
            st.session_state.theme = st.selectbox(
                "Theme",
                list(THEMES.keys()),
                index=list(THEMES.keys()).index(st.session_state.theme),
            )

    # ── Axiom & Resources (collapsed) ────────────────────────────────────────
    with st.sidebar.expander("📜 Axiom-0", expanded=False):
        st.info('**"Collapse is generative; only what returns is real."**')
        st.caption("F+ω=1 · IC≤F · IC=exp(κ)\n\nTier-0: Protocol · Tier-1: Invariants · Tier-2: Closures")

    with st.sidebar.expander("📚 Resources", expanded=False):
        st.markdown(
            "- [GitHub](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS)\n"
            "- [Tutorial](QUICKSTART_TUTORIAL.md)\n"
            "- [API Docs](http://localhost:8000/docs)"
        )

    st.sidebar.caption(f"© 2026 UMCP · v{__version__}")

    # ── Render selected page ─────────────────────────────────────────────────
    selected = st.session_state.selected_page
    if selected in pages:
        _, render_func = pages[selected]
        render_func()
    else:
        # Fallback if selected page no longer exists
        st.session_state.selected_page = "Overview"
        render_overview_page()


# ── Streamlit direct execution ───────────────────────────────────────────────
# When `streamlit run __init__.py` executes this file, it also triggers a
# package import of umcp.dashboard (via the top-level imports from _deps/_utils),
# running __init__.py a second time under __name__ == "umcp.dashboard".
# Only call main() from the Streamlit-exec'd instance (where __name__ differs).
if _is_running_in_streamlit() and __name__ != "umcp.dashboard":
    main()
