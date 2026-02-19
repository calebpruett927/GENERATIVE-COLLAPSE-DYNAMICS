"""
Core dashboard pages: Overview, Ledger, Casepacks, Contracts, Closures, Regime, Metrics, Health.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from typing import Any

from umcp.dashboard._deps import HAS_VIZ_DEPS, go, make_subplots, np, pd, px, st
from umcp.dashboard._utils import (
    KERNEL_SYMBOLS,
    REGIME_COLORS,
    STATUS_COLORS,
    classify_regime,
    detect_anomalies,
    format_bytes,
    get_regime_color,
    get_repo_root,
    load_casepacks,
    load_closures,
    load_contracts,
    load_ledger,
)

try:
    from umcp import __version__
except ImportError:
    __version__ = "2.0.0"


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
