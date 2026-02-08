"""
Analysis dashboard pages: Exports, Comparison, Time Series, Formula Builder.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from datetime import datetime, timedelta

from umcp.dashboard._deps import go, make_subplots, np, pd, px, st
from umcp.dashboard._utils import (
    KERNEL_SYMBOLS,
    REGIME_COLORS,
    STATUS_COLORS,
    classify_regime,
    load_casepacks,
    load_closures,
    load_contracts,
    load_ledger,
)

try:
    from umcp import __version__
except ImportError:
    __version__ = "2.0.0"


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
