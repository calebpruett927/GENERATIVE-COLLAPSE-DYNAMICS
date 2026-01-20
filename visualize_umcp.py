#!/usr/bin/env python3
"""
UMCP Visualization Dashboard - Production Edition

Advanced Streamlit app for UMCP validation monitoring:
- Real-time validation metrics
- Interactive phase space analysis
- Regime transition tracking
- Historical trend analysis
- Anomaly detection
- Export capabilities

Usage:
    streamlit run visualize_umcp.py
    umcp-visualize  # Using entry point
    
Features:
    - Auto-refresh capability
    - Statistical analysis
    - Regime prediction
    - Custom filtering
    - Multi-metric correlations
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Please install required packages: pip install streamlit pandas plotly numpy")
    st.stop()


# UMCP Extension Entry Points
class UMCPVisualization:
    """UMCP Extension: Interactive Visualization Dashboard
    
    Provides Streamlit-based visualization of UMCP validation data.
    Automatically registered with UMCP extension system.
    
    Attributes:
        name: Extension name
        version: Extension version
        description: Extension description
        requires: Required dependencies
    """
    
    name = "visualization"
    version = "1.0.0"
    description = "Interactive Streamlit dashboard for UMCP validation monitoring"
    requires = ["streamlit>=1.30.0", "pandas>=2.0.0", "plotly>=5.18.0", "numpy>=1.24.0"]
    
    @staticmethod
    def install():
        """Install extension dependencies"""
        import subprocess
        subprocess.check_call(["pip", "install"] + UMCPVisualization.requires)
    
    @staticmethod
    def run():
        """Run the extension"""
        main()
    
    @staticmethod
    def info():
        """Return extension metadata"""
        return {
            "name": UMCPVisualization.name,
            "version": UMCPVisualization.version,
            "description": UMCPVisualization.description,
            "requires": UMCPVisualization.requires,
            "features": [
                "Real-time validation metrics",
                "Interactive phase space plots",
                "Regime transition tracking",
                "Historical trend analysis",
                "Anomaly detection",
                "Export capabilities"
            ]
        }


# Enhanced color scheme matching UMCP regimes
REGIME_COLORS = {
    "Stable": "#28a745",    # Green
    "Watch": "#ffc107",     # Yellow/Amber
    "Collapse": "#dc3545",  # Red
    "Unknown": "#6c757d"    # Gray
}

# Threshold definitions for regime classification
REGIME_THRESHOLDS = {
    "Stable": {"omega_max": 0.038, "F_min": 0.90, "S_max": 0.15, "C_max": 0.14},
    "Collapse": {"omega_min": 0.30},
    "Watch": {}  # Default fallback
}


def load_ledger(repo_root: Path) -> Optional[pd.DataFrame]:
    """Load the continuous ledger from ledger/return_log.csv with enhanced processing"""
    ledger_path = repo_root / "ledger" / "return_log.csv"
    if not ledger_path.exists():
        return None
    
    try:
        df = pd.read_csv(ledger_path)
        if df.empty:
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns with error handling
        numeric_cols = ['omega', 'stiffness', 'curvature', 'delta_kappa']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add derived metrics
        if len(df) > 1:
            df['omega_change'] = df['omega'].diff()
            df['curvature_change'] = df['curvature'].diff()
            df['time_delta'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        return df
    except Exception as e:
        st.error(f"Error loading ledger: {e}")
        return None


def load_all_receipts(repo_root: Path) -> List[Dict[str, Any]]:
    """Load all available receipt files for historical comparison"""
    receipts = []
    
    # Look for receipt*.json files
    for receipt_file in repo_root.glob("receipt*.json"):
        try:
            with open(receipt_file, 'r') as f:
                receipt = json.load(f)
                receipt['_filename'] = receipt_file.name
                receipts.append(receipt)
        except Exception:
            continue
    
    # Sort by created_utc
    receipts.sort(key=lambda r: r.get('created_utc', ''), reverse=True)
    return receipts


def load_latest_receipt(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load the most recent validation receipt"""
    receipt_path = repo_root / "receipt.json"
    if not receipt_path.exists():
        return None
    
    try:
        with open(receipt_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading receipt: {e}")
        return None


def load_invariants(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load current invariants from outputs/invariants.csv"""
    inv_path = repo_root / "outputs" / "invariants.csv"
    if not inv_path.exists():
        return None
    
    try:
        with open(inv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[0] if rows else None
    except Exception as e:
        st.error(f"Error loading invariants: {e}")
        return None


def load_regimes(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load regime data from outputs/regimes.csv"""
    reg_path = repo_root / "outputs" / "regimes.csv"
    if not reg_path.exists():
        return None
    
    try:
        with open(reg_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[0] if rows else None
    except Exception:
        return None


def load_welds(repo_root: Path) -> Optional[pd.DataFrame]:
    """Load weld data from outputs/welds.csv"""
    weld_path = repo_root / "outputs" / "welds.csv"
    if not weld_path.exists():
        return None
    
    try:
        return pd.read_csv(weld_path)
    except Exception:
        return None


def classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Classify regime based on UMCP thresholds"""
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    elif omega >= 0.30:
        return "Collapse"
    else:
        return "Watch"


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistical metrics from ledger data"""
    if df is None or df.empty:
        return {}
    
    stats = {
        "total_validations": len(df),
        "conformant": len(df[df['run_status'] == 'CONFORMANT']),
        "nonconformant": len(df[df['run_status'] != 'CONFORMANT']),
        "omega_mean": df['omega'].mean(),
        "omega_std": df['omega'].std(),
        "omega_trend": df['omega'].iloc[-1] - df['omega'].iloc[0] if len(df) > 1 else 0,
        "curvature_mean": df['curvature'].mean(),
        "curvature_std": df['curvature'].std(),
        "stiffness_mean": df['stiffness'].mean(),
        "stiffness_std": df['stiffness'].std(),
    }
    
    # Calculate regime distribution
    df_with_regime = df.copy()
    df_with_regime['regime'] = df_with_regime.apply(
        lambda row: classify_regime(
            row['omega'], 
            1.0 - row['omega'], 
            row['stiffness'], 
            row['curvature']
        ),
        axis=1
    )
    
    regime_counts = df_with_regime['regime'].value_counts().to_dict()
    stats['regime_distribution'] = regime_counts
    stats['current_regime'] = df_with_regime['regime'].iloc[-1] if not df_with_regime.empty else "Unknown"
    
    # Detect regime transitions
    if len(df_with_regime) > 1:
        regime_changes = (df_with_regime['regime'] != df_with_regime['regime'].shift()).sum() - 1
        stats['regime_transitions'] = regime_changes
    else:
        stats['regime_transitions'] = 0
    
    return stats


def detect_anomalies(df: pd.DataFrame, col: str = 'omega', threshold: float = 2.0) -> pd.Series:
    """Detect anomalies using z-score method"""
    if df is None or df.empty or col not in df.columns:
        return pd.Series([False] * len(df)) if df is not None else pd.Series()
    
    mean = df[col].mean()
    std = df[col].std()
    
    if std == 0:
        return pd.Series([False] * len(df))
    
    z_scores = np.abs((df[col] - mean) / std)
    return z_scores > threshold


def plot_omega_vs_curvature(df: pd.DataFrame, show_anomalies: bool = True):
    """Enhanced phase space plot with regime boundaries and anomalies"""
    if df is None or df.empty:
        st.warning("No data to plot")
        return
    
    # Add regime classification
    df = df.copy()
    df['regime'] = df.apply(
        lambda row: classify_regime(
            float(row['omega']),
            1.0 - float(row['omega']),
            float(row['stiffness']),
            float(row['curvature'])
        ),
        axis=1
    )
    
    fig = go.Figure()
    
    # Plot regime boundaries
    fig.add_shape(
        type="line",
        x0=0.038, y0=0, x1=0.038, y1=df['curvature'].max() * 1.1,
        line=dict(color="red", width=1, dash="dash"),
        name="Stable/Watch boundary (ω=0.038)"
    )
    
    fig.add_shape(
        type="line",
        x0=0.30, y0=0, x1=0.30, y1=df['curvature'].max() * 1.1,
        line=dict(color="darkred", width=2, dash="dash"),
        name="Collapse boundary (ω=0.30)"
    )
    
    # Plot points by regime
    for regime in ["Stable", "Watch", "Collapse", "Unknown"]:
        regime_data = df[df['regime'] == regime]
        if not regime_data.empty:
            fig.add_trace(go.Scatter(
                x=regime_data['omega'],
                y=regime_data['curvature'],
                mode='markers+lines',
                name=regime,
                marker=dict(
                    color=REGIME_COLORS[regime],
                    size=12,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                line=dict(color=REGIME_COLORS[regime], width=1, dash='dot'),
                text=regime_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                customdata=regime_data['stiffness'],
                hovertemplate='<b>%{text}</b><br>ω: %{x:.6f}<br>C: %{y:.6f}<br>S: %{customdata:.6f}<extra></extra>'
            ))
    
    # Mark anomalies
    if show_anomalies:
        anomalies_omega = detect_anomalies(df, 'omega')
        anomalies_curvature = detect_anomalies(df, 'curvature')
        anomaly_mask = anomalies_omega | anomalies_curvature
        
        if anomaly_mask.any():
            anomaly_df = df[anomaly_mask]
            fig.add_trace(go.Scatter(
                x=anomaly_df['omega'],
                y=anomaly_df['curvature'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='black',
                    size=16,
                    symbol='x-open',
                    line=dict(width=2)
                ),
                text=anomaly_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                hovertemplate='<b>ANOMALY</b><br>%{text}<br>ω: %{x:.6f}<br>C: %{y:.6f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': "UMCP Phase Space: ω vs Curvature (C)<br><sub>with Regime Boundaries & Anomaly Detection</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="ω (Structural Instability)",
        yaxis_title="C (Curvature)",
        template="plotly_white",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # Add annotations for regime regions
    fig.add_annotation(
        x=0.019, y=df['curvature'].max() * 0.9,
        text="Stable",
        showarrow=False,
        font=dict(size=12, color=REGIME_COLORS["Stable"]),
        bgcolor="rgba(255, 255, 255, 0.7)"
    )
    
    fig.add_annotation(
        x=0.15, y=df['curvature'].max() * 0.9,
        text="Watch",
        showarrow=False,
        font=dict(size=12, color=REGIME_COLORS["Watch"]),
        bgcolor="rgba(255, 255, 255, 0.7)"
    )
    
    if df['omega'].max() >= 0.30:
        fig.add_annotation(
            x=0.35, y=df['curvature'].max() * 0.9,
            text="Collapse",
            showarrow=False,
            font=dict(size=12, color=REGIME_COLORS["Collapse"]),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
    
    st.plotly_chart(fig, width="stretch")


def plot_time_series(df: pd.DataFrame, show_moving_avg: bool = True):
    """Enhanced time series with moving averages and trend lines"""
    if df is None or df.empty:
        st.warning("No data to plot")
        return
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("ω (Instability)", "C (Curvature)", "S (Stiffness)"),
        vertical_spacing=0.08
    )
    
    # ω over time
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['omega'],
        mode='lines+markers',
        name='ω',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    if show_moving_avg and len(df) >= 3:
        ma_window = min(5, len(df) // 2)
        omega_ma = df['omega'].rolling(window=ma_window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=omega_ma,
            mode='lines',
            name=f'ω MA({ma_window})',
            line=dict(color='#1976D2', width=3, dash='dash'),
            opacity=0.7
        ), row=1, col=1)
    
    # Curvature over time
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['curvature'],
        mode='lines+markers',
        name='C',
        line=dict(color='#FF9800', width=2),
        marker=dict(size=6)
    ), row=2, col=1)
    
    if show_moving_avg and len(df) >= 3:
        ma_window = min(5, len(df) // 2)
        curv_ma = df['curvature'].rolling(window=ma_window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=curv_ma,
            mode='lines',
            name=f'C MA({ma_window})',
            line=dict(color='#F57C00', width=3, dash='dash'),
            opacity=0.7
        ), row=2, col=1)
    
    # Stiffness over time
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['stiffness'],
        mode='lines+markers',
        name='S',
        line=dict(color='#4CAF50', width=2),
        marker=dict(size=6)
    ), row=3, col=1)
    
    if show_moving_avg and len(df) >= 3:
        ma_window = min(5, len(df) // 2)
        stiff_ma = df['stiffness'].rolling(window=ma_window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=stiff_ma,
            mode='lines',
            name=f'S MA({ma_window})',
            line=dict(color='#388E3C', width=3, dash='dash'),
            opacity=0.7
        ), row=3, col=1)
    
    # Add threshold lines
    fig.add_hline(y=0.038, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
                  annotation_text="Stable threshold", annotation_position="right")
    fig.add_hline(y=0.30, line_dash="solid", line_color="darkred", opacity=0.7, row=1, col=1,
                  annotation_text="Collapse threshold", annotation_position="right")
    
    fig.update_layout(
        title={
            'text': "UMCP Invariants Time Series<br><sub>with Moving Averages & Thresholds</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        template="plotly_white",
        hovermode='x unified',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    fig.update_yaxes(title_text="ω", row=1, col=1)
    fig.update_yaxes(title_text="C", row=2, col=1)
    fig.update_yaxes(title_text="S", row=3, col=1)
    
    st.plotly_chart(fig, width="stretch")


def plot_regime_timeline(df: pd.DataFrame):
    """Plot regime classification over time as a timeline"""
    if df is None or df.empty:
        st.warning("No data to plot")
        return
    
    df = df.copy()
    df['regime'] = df.apply(
        lambda row: classify_regime(
            row['omega'], 1.0 - row['omega'], 
            row['stiffness'], row['curvature']
        ),
        axis=1
    )
    
    # Create regime timeline
    regime_map = {"Stable": 0, "Watch": 1, "Collapse": 2, "Unknown": 3}
    df['regime_num'] = df['regime'].map(regime_map)
    
    fig = go.Figure()
    
    for regime in ["Stable", "Watch", "Collapse", "Unknown"]:
        regime_df = df[df['regime'] == regime]
        if not regime_df.empty:
            fig.add_trace(go.Scatter(
                x=regime_df['timestamp'],
                y=regime_df['regime_num'],
                mode='markers+lines',
                name=regime,
                marker=dict(
                    color=REGIME_COLORS[regime],
                    size=15,
                    symbol='square'
                ),
                line=dict(color=REGIME_COLORS[regime], width=3),
                hovertemplate='<b>%{text}</b><br>Timestamp: %{x}<extra></extra>',
                text=regime_df['regime']
            ))
    
    fig.update_layout(
        title="Regime Classification Timeline",
        xaxis_title="Timestamp",
        yaxis=dict(
            title="Regime",
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=["Stable", "Watch", "Collapse", "Unknown"]
        ),
        template="plotly_white",
        height=300,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap of invariants"""
    if df is None or df.empty:
        st.warning("No data to plot")
        return
    
    # Select numeric columns
    numeric_cols = ['omega', 'stiffness', 'curvature']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        st.warning("Not enough numeric data for correlation analysis")
        return
    
    # Compute correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Format text for display
    text_values = [[f"{val:.3f}" for val in row] for row in corr_matrix.values]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale='RdBu',
        zmid=0,
        text=text_values,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Invariant Correlation Matrix",
        template="plotly_white",
        height=400,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left')
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_distribution_violin(df: pd.DataFrame):
    """Plot violin plots for invariant distributions"""
    if df is None or df.empty:
        st.warning("No data to plot")
        return
    
    df = df.copy()
    df['regime'] = df.apply(
        lambda row: classify_regime(
            row['omega'], 1.0 - row['omega'], 
            row['stiffness'], row['curvature']
        ),
        axis=1
    )
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("ω Distribution", "C Distribution", "S Distribution")
    )
    
    for regime in ["Stable", "Watch", "Collapse"]:
        regime_df = df[df['regime'] == regime]
        if not regime_df.empty:
            fig.add_trace(go.Violin(
                y=regime_df['omega'],
                name=regime,
                marker_color=REGIME_COLORS[regime],
                showlegend=True
            ), row=1, col=1)
            
            fig.add_trace(go.Violin(
                y=regime_df['curvature'],
                name=regime,
                marker_color=REGIME_COLORS[regime],
                showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Violin(
                y=regime_df['stiffness'],
                name=regime,
                marker_color=REGIME_COLORS[regime],
                showlegend=False
            ), row=1, col=3)
    
    fig.update_layout(
        title="Invariant Distributions by Regime",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="ω", row=1, col=1)
    fig.update_yaxes(title_text="C", row=1, col=2)
    fig.update_yaxes(title_text="S", row=1, col=3)
    
    st.plotly_chart(fig, width="stretch")


def display_latest_receipt(receipt: Optional[Dict[str, Any]]):
    """Display latest validation receipt summary"""
    if receipt is None:
        st.warning("No receipt found")
        return
    
    st.subheader("Latest Validation Receipt")
    
    col1, col2, col3, col4 = st.columns(4)
    
    status = receipt.get("run_status", "UNKNOWN")
    status_color = {
        "CONFORMANT": "🟢",
        "NONCONFORMANT": "🔴",
        "NON_EVALUABLE": "🟠"
    }.get(status, "⚪")
    
    with col1:
        st.metric("Status", f"{status_color} {status}")
    
    with col2:
        errors = receipt.get("summary", {}).get("counts", {}).get("errors", 0)
        st.metric("Errors", errors, delta=None, delta_color="inverse")
    
    with col3:
        warnings = receipt.get("summary", {}).get("counts", {}).get("warnings", 0)
        st.metric("Warnings", warnings, delta=None, delta_color="inverse")
    
    with col4:
        created = receipt.get("created_utc", "N/A")
        st.metric("Created", created)
    
    # Show targets
    with st.expander("Targets Validated"):
        targets = receipt.get("targets", [])
        for target in targets:
            target_type = target.get("target_type", "unknown")
            target_path = target.get("target_path", "unknown")
            target_status = target.get("run_status", "unknown")
            st.write(f"- **{target_type}**: `{target_path}` → {target_status}")


def display_current_invariants(invariants: Optional[Dict[str, Any]]):
    """Display current invariant values"""
    if invariants is None:
        st.warning("No invariants found")
        return
    
    st.subheader("Current Invariants")
    
    col1, col2, col3, col4 = st.columns(4)
    
    omega = float(invariants.get('omega', 0))
    F = float(invariants.get('F', 0))
    S = float(invariants.get('S', 0))
    C = float(invariants.get('C', 0))
    
    regime = classify_regime(omega, F, S, C)
    regime_color = REGIME_COLORS.get(regime, REGIME_COLORS["Unknown"])
    
    with col1:
        st.metric("ω (Instability)", f"{omega:.6f}")
    
    with col2:
        st.metric("F (Fidelity)", f"{F:.6f}")
    
    with col3:
        st.metric("S (Stiffness)", f"{S:.6f}")
    
    with col4:
        st.metric("C (Curvature)", f"{C:.6f}")
    
    # Display regime with color
    st.markdown(f"### Current Regime: <span style='color:{regime_color}; font-weight:bold;'>{regime}</span>", unsafe_allow_html=True)
    
    # Additional invariants in expander
    with st.expander("All Invariants"):
        for key, value in invariants.items():
            st.write(f"**{key}**: {value}")


def main():
    st.set_page_config(
        page_title="UMCP Production Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .big-metric { font-size: 24px; font-weight: bold; }
        .status-success { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 UMCP Production Dashboard")
    st.markdown("**Real-time monitoring of UMCP validation, regime transitions, and system health**")
    
    # Detect repository root
    repo_root = Path.cwd()
    if not (repo_root / "pyproject.toml").exists():
        current = repo_root
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                repo_root = current
                break
            current = current.parent
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.code(str(repo_root), language="text")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 30 seconds")
        import time
        time.sleep(30)
        st.rerun()
    
    # Visualization options
    st.sidebar.header("📊 Visualization Options")
    show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
    show_moving_avg = st.sidebar.checkbox("Show Moving Averages", value=True)
    show_advanced_stats = st.sidebar.checkbox("Show Advanced Statistics", value=False)
    
    # Load data
    with st.spinner("Loading UMCP data..."):
        ledger_df = load_ledger(repo_root)
        receipt = load_latest_receipt(repo_root)
        receipts = load_all_receipts(repo_root)
        invariants = load_invariants(repo_root)
        regimes_data = load_regimes(repo_root)
        welds_df = load_welds(repo_root)
    
    # Compute statistics if data available
    stats = compute_statistics(ledger_df) if ledger_df is not None else {}
    
    # Header metrics row
    if stats:
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Validations",
                stats.get('total_validations', 0),
                help="Total number of validation runs"
            )
        
        with col2:
            conformant_rate = (stats.get('conformant', 0) / max(stats.get('total_validations', 1), 1)) * 100
            st.metric(
                "Conformance Rate",
                f"{conformant_rate:.1f}%",
                help="Percentage of successful validations"
            )
        
        with col3:
            current_regime = stats.get('current_regime', 'Unknown')
            regime_color = REGIME_COLORS.get(current_regime, "#6c757d")
            st.markdown(
                f"<div style='text-align: center;'>"
                f"<div style='color: #666; font-size: 14px;'>Current Regime</div>"
                f"<div style='color: {regime_color}; font-size: 24px; font-weight: bold;'>{current_regime}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            omega_trend = stats.get('omega_trend', 0)
            trend_arrow = "↑" if omega_trend > 0 else "↓" if omega_trend < 0 else "→"
            trend_color = "red" if omega_trend > 0 else "green" if omega_trend < 0 else "gray"
            st.metric(
                "ω Trend",
                f"{trend_arrow} {abs(omega_trend):.6f}",
                delta=f"{omega_trend:.6f}",
                delta_color="inverse"
            )
        
        with col5:
            transitions = stats.get('regime_transitions', 0)
            st.metric(
                "Regime Transitions",
                transitions,
                help="Number of regime changes detected"
            )
    
    st.markdown("---")
    
    # Latest receipt and current state
    col1, col2 = st.columns(2)
    
    with col1:
        display_latest_receipt(receipt)
    
    with col2:
        display_current_invariants(invariants)
    
    st.markdown("---")
    
    # Main visualizations
    if ledger_df is not None and not ledger_df.empty:
        st.header("📈 Interactive Analytics")
        
        # Time range filter
        if len(ledger_df) > 1:
            min_date = ledger_df['timestamp'].min().to_pydatetime()
            max_date = ledger_df['timestamp'].max().to_pydatetime()
            
            date_range = st.slider(
                "Filter by Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD HH:mm"
            )
            
            ledger_df_filtered = ledger_df[
                (ledger_df['timestamp'] >= pd.Timestamp(date_range[0])) &
                (ledger_df['timestamp'] <= pd.Timestamp(date_range[1]))
            ]
        else:
            ledger_df_filtered = ledger_df
        
        # Tab organization
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Phase Space",
            "📈 Time Series",
            "⏱️ Timeline",
            "📊 Statistics",
            "🔗 Correlations",
            "📋 Raw Data"
        ])
        
        with tab1:
            st.markdown("### Phase Space: Regime Classification")
            st.markdown("Explore the ω-C phase space with regime boundaries and anomaly detection")
            plot_omega_vs_curvature(ledger_df_filtered, show_anomalies=show_anomalies)
            
            if show_advanced_stats and not ledger_df_filtered.empty:
                st.markdown("#### Phase Space Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ω Range", f"{ledger_df_filtered['omega'].min():.6f} - {ledger_df_filtered['omega'].max():.6f}")
                with col2:
                    st.metric("C Range", f"{ledger_df_filtered['curvature'].min():.6f} - {ledger_df_filtered['curvature'].max():.6f}")
                with col3:
                    st.metric("S Range", f"{ledger_df_filtered['stiffness'].min():.6f} - {ledger_df_filtered['stiffness'].max():.6f}")
        
        with tab2:
            st.markdown("### Invariants Over Time")
            st.markdown("Track ω, C, and S metrics with moving averages and threshold indicators")
            plot_time_series(ledger_df_filtered, show_moving_avg=show_moving_avg)
            
            if show_advanced_stats and len(ledger_df_filtered) > 1:
                st.markdown("#### Trend Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    omega_change = ledger_df_filtered['omega'].iloc[-1] - ledger_df_filtered['omega'].iloc[0]
                    st.metric("ω Change", f"{omega_change:+.6f}", delta=omega_change)
                with col2:
                    curv_change = ledger_df_filtered['curvature'].iloc[-1] - ledger_df_filtered['curvature'].iloc[0]
                    st.metric("C Change", f"{curv_change:+.6f}", delta=curv_change)
                with col3:
                    stiff_change = ledger_df_filtered['stiffness'].iloc[-1] - ledger_df_filtered['stiffness'].iloc[0]
                    st.metric("S Change", f"{stiff_change:+.6f}", delta=stiff_change)
        
        with tab3:
            st.markdown("### Regime Timeline")
            st.markdown("Visualize regime transitions over time")
            plot_regime_timeline(ledger_df_filtered)
            
            # Regime distribution pie chart
            if not ledger_df_filtered.empty:
                st.markdown("#### Regime Distribution")
                df_temp = ledger_df_filtered.copy()
                df_temp['regime'] = df_temp.apply(
                    lambda row: classify_regime(
                        row['omega'], 1.0 - row['omega'], 
                        row['stiffness'], row['curvature']
                    ),
                    axis=1
                )
                regime_counts = df_temp['regime'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=regime_counts.index,
                    values=regime_counts.values,
                    marker=dict(colors=[REGIME_COLORS.get(r, "#6c757d") for r in regime_counts.index]),
                    hole=0.4
                )])
                fig.update_layout(
                    title="Time Spent in Each Regime",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")
        
        with tab4:
            st.markdown("### Statistical Analysis")
            
            # Distributions
            if len(ledger_df_filtered) >= 3:
                st.markdown("#### Invariant Distributions by Regime")
                plot_distribution_violin(ledger_df_filtered)
            
            # Summary statistics table
            st.markdown("#### Summary Statistics")
            summary_data = {
                "Metric": ["ω (Instability)", "C (Curvature)", "S (Stiffness)"],
                "Mean": [
                    f"{ledger_df_filtered['omega'].mean():.6f}",
                    f"{ledger_df_filtered['curvature'].mean():.6f}",
                    f"{ledger_df_filtered['stiffness'].mean():.6f}"
                ],
                "Std Dev": [
                    f"{ledger_df_filtered['omega'].std():.6f}",
                    f"{ledger_df_filtered['curvature'].std():.6f}",
                    f"{ledger_df_filtered['stiffness'].std():.6f}"
                ],
                "Min": [
                    f"{ledger_df_filtered['omega'].min():.6f}",
                    f"{ledger_df_filtered['curvature'].min():.6f}",
                    f"{ledger_df_filtered['stiffness'].min():.6f}"
                ],
                "Max": [
                    f"{ledger_df_filtered['omega'].max():.6f}",
                    f"{ledger_df_filtered['curvature'].max():.6f}",
                    f"{ledger_df_filtered['stiffness'].max():.6f}"
                ]
            }
            st.table(pd.DataFrame(summary_data))
        
        with tab5:
            st.markdown("### Correlation Analysis")
            if len(ledger_df_filtered) >= 3:
                plot_correlation_heatmap(ledger_df_filtered)
                
                st.markdown("#### Interpretation")
                st.info("""
                - **Strong positive correlation (>0.7)**: Metrics move together
                - **Strong negative correlation (<-0.7)**: Metrics move oppositely
                - **Weak correlation (-0.3 to 0.3)**: Little to no linear relationship
                """)
        
        with tab6:
            st.markdown("### Raw Ledger Data")
            
            # Search/filter
            search_term = st.text_input("🔍 Search ledger data", "")
            if search_term:
                mask = ledger_df_filtered.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False).any(),
                    axis=1
                )
                display_df = ledger_df_filtered[mask]
            else:
                display_df = ledger_df_filtered
            
            st.dataframe(
                display_df.style.highlight_max(axis=0, subset=['omega', 'curvature']),
                width="stretch",
                height=400
            )
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                json_str = display_df.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"umcp_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            with col3:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
        
        # Additional insights
        if show_advanced_stats:
            st.markdown("---")
            st.header("🔬 Advanced Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Anomaly Detection")
                anomalies = detect_anomalies(ledger_df_filtered, 'omega')
                if anomalies.any():
                    st.warning(f"⚠️ {anomalies.sum()} anomalies detected in ω values")
                    anomaly_df = ledger_df_filtered[anomalies][['timestamp', 'omega', 'curvature', 'stiffness']]
                    st.dataframe(anomaly_df)
                else:
                    st.success("✅ No anomalies detected")
            
            with col2:
                st.markdown("#### Recent Activity")
                if len(ledger_df_filtered) > 5:
                    recent = ledger_df_filtered.tail(5)[['timestamp', 'run_status', 'omega']]
                    st.dataframe(recent)
    
    else:
        st.info("📭 No ledger data available. Run `umcp validate` to start collecting data.")
        st.code("umcp validate --out receipt.json", language="bash")
        st.markdown("### Getting Started")
        st.markdown("""
        1. Run UMCP validation in your terminal
        2. Refresh this dashboard to see results
        3. Historical data will accumulate automatically
        4. Use filters and visualization options in the sidebar
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 12px;'>"
        f"UMCP Dashboard v1.1.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Built with Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
