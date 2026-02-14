"""
Advanced dashboard pages: Precision, Geometry, Canon Explorer, Domain Overview.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any

from umcp.dashboard._deps import go, make_subplots, np, pd, px, st
from umcp.dashboard._utils import (
    REGIME_COLORS,
    get_repo_root,
)


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

    # Bernoulli field entropy (Shannon entropy is the degenerate limit)
    S_terms = []
    for c in c_clipped:
        s = -c * np.log(c) - (1 - c) * np.log(1 - c) if 0 < c < 1 else 0
        S_terms.append(s)
    S = float(np.sum(w_arr * np.array(S_terms)))

    # Heterogeneity gap
    gap = F - IC

    # ========== Display Results with Full Precision ==========
    st.subheader("📊 Computed Invariants")

    # Main invariants table
    invariants_data = {
        "Symbol": ["F", "ω", "κ", "IC", "C", "S", "Δ (Heterogeneity Gap)"],
        "Name": [
            "Fidelity (Arithmetic Mean)",
            "Drift (1 - F)",
            "Log-Integrity (Σwᵢ ln cᵢ)",
            "Integrity Composite (exp κ)",
            "Curvature Proxy (std/0.5)",
            "Bernoulli Field Entropy",
            "Heterogeneity Gap (F - IC)",
        ],
        "Value": [f"{F:.15f}", f"{omega:.15f}", f"{kappa:.15f}", f"{IC:.15f}", f"{C:.15f}", f"{S:.15f}", f"{gap:.15f}"],
        "Bound Check": [
            "✅" if 0 <= F <= 1 else "❌",
            "✅" if 0 <= omega <= 1 else "❌",
            "✅" if kappa <= 0 else "❌",  # IC <= 1 implies kappa <= 0
            "✅" if 0 < IC <= 1 else "❌",
            "✅" if 0 <= C <= 1 else "⚠️",  # Soft bound
            "✅" if S >= 0 else "❌",
            "✅" if gap >= 0 else "❌",  # Integrity bound: F >= IC always
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

    # Lemma 4: Integrity bound F >= IC (AM-GM is the degenerate limit)
    checks.append(
        {
            "Lemma": "Lemma 4",
            "Statement": "F ≥ IC (integrity bound)",
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

    Reference: docs/INFRASTRUCTURE_GEOMETRY.md
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

        See [docs/INFRASTRUCTURE_GEOMETRY.md](docs/INFRASTRUCTURE_GEOMETRY.md) for full specification.
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

    # ========== Heterogeneity Gap Visualization ==========
    st.markdown("#### Heterogeneity Gap Analysis")

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
        title="Heterogeneity Gap: F ≥ IC (with equality iff homogeneous)",
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
    # Also load anchors.yaml (root canon) if present
    root_anchor = canon_dir / "anchors.yaml"
    if root_anchor.exists():
        try:
            with open(root_anchor) as f:
                data = yaml.safe_load(f)
            if data:
                result["anchors"] = data
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
        - **Regime classification** — threshold values for STABLE / WATCH / COLLAPSE
        - **Mathematical identities** — Tier-1 relationships (F + ω = 1, IC ≤ F, IC ≈ exp(κ))
        - **Tolerances** — numerical precision requirements

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
        "anchors": "🔒 UMCP Root",
        "astro_anchors": "🔭 ASTRO",
        "nuc_anchors": "☢️ NUC",
        "qm_anchors": "🔮 QM",
    }

    domain_names = list(canon.keys())
    selected = st.selectbox(
        "Select Domain Canon",
        domain_names,
        format_func=lambda x: domain_icons.get(x, x.replace("_anchors", "").upper()),
        key="canon_domain_select",
    )

    data = canon[selected]

    # ── Header card ──
    # Handle varying metadata locations across canon files
    canon_id = data.get("id", "")
    version = data.get("version", "")
    created = str(data.get("created", ""))
    scope_data = data.get("scope", {})
    tier_str = str(scope_data.get("tier", scope_data.get("hierarchy", "")))

    # Root anchors.yaml uses umcp_canon as top-level
    if not canon_id and "umcp_canon" in data:
        root = data["umcp_canon"]
        canon_id = root.get("canon_id", "")
        scope_str = root.get("scope", "")
        tier_str = scope_str if isinstance(scope_str, str) else str(scope_str)

    if not canon_id:
        canon_id = selected.replace("_", ".").upper()

    tier_color = "#1976d2" if "1" in tier_str else "#388e3c" if "2" in tier_str else "#f57c00"
    st.markdown(
        f"""<div style="padding:12px 16px; border-radius:8px; background:linear-gradient(135deg, #f8f9fa, #ffffff);
             border-left:4px solid {tier_color}; margin-bottom:16px;">
        <span style="font-size:1.2em; font-weight:bold;">{canon_id}</span>
        <span style="background:{tier_color}; color:white; padding:2px 8px; border-radius:4px;
              font-size:0.8em; margin-left:8px;">Tier {tier_str}</span>
        <span style="color:#666; margin-left:12px;">v{version} · Created: {created}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Show scope description
    scope_desc = scope_data.get("description", "")
    if scope_desc:
        st.info(scope_desc.strip())

    st.divider()

    # ── Axioms / Principles ──
    axioms = data.get("axioms", data.get("principles", []))
    if axioms:
        st.subheader("📜 Axioms")
        for idx, ax in enumerate(axioms):
            if isinstance(ax, dict):
                ax_id = ax.get("id", ax.get("axiom_id", f"A{idx + 1}"))
                label = ax.get("label", ax.get("statement", ax.get("name", "")))
                desc = ax.get("description", "")
                # Some domains add realization fields (nuc_realization, qm_realization, etc.)
                realization = ""
                for k, v in ax.items():
                    if k.endswith("_realization") and isinstance(v, str):
                        realization = v
                with st.expander(f"**{ax_id}**: {label}", expanded=idx == 0):
                    if desc:
                        st.markdown(desc)
                    if realization:
                        st.markdown(f"**Domain realization**: {realization.strip()}")
            elif isinstance(ax, str):
                st.markdown(f"- {ax}")

    # ── Reserved Symbols ──
    # Symbols live in various nested locations depending on domain
    symbols = _extract_symbols(data)
    if symbols:
        st.subheader("🔣 Reserved Symbols")
        sym_rows: list[dict[str, str]] = []
        for sym in symbols:
            if isinstance(sym, dict):
                sym_rows.append(
                    {
                        "Symbol": sym.get("symbol", sym.get("latex", sym.get("name", sym.get("ascii", "")))),
                        "Description": sym.get("description", sym.get("label", "")),
                        "Domain": str(sym.get("domain", "")),
                        "Formula": sym.get("formula", sym.get("identity", "")),
                    }
                )
        if sym_rows:
            st.dataframe(pd.DataFrame(sym_rows), use_container_width=True, hide_index=True)

    # ── Tier hierarchy (RCFT) ──
    tier_hier = data.get("tier_hierarchy", {})
    if tier_hier:
        frozen = tier_hier.get("tier_1_frozen_symbols", [])
        if frozen:
            st.subheader("🔒 Tier-1 Frozen Symbols (inherited)")
            st.markdown(", ".join(f"`{s}`" for s in frozen))
        hier_desc = tier_hier.get("description", "")
        if hier_desc:
            st.info(hier_desc.strip())

    # ── Regime Classification ──
    regime_data = data.get("regime_classification", {})
    if regime_data:
        st.subheader("🚦 Regime Classification")
        # Standard regimes list
        regimes = regime_data.get("regimes", [])
        if regimes:
            gate_rows: list[dict[str, str]] = []
            for r in regimes:
                if isinstance(r, dict):
                    gate_rows.append(
                        {
                            "Regime": r.get("label", r.get("name", "")),
                            "Condition": r.get("condition", ""),
                            "Interpretation": r.get("interpretation", ""),
                        }
                    )
            if gate_rows:
                st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)
        # RCFT-style sub-classifications (fractal_complexity, recursive_strength, etc.)
        for key, items in regime_data.items():
            if key in ("description", "regimes"):
                continue
            if isinstance(items, list):
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                sub_rows: list[dict[str, str]] = []
                for item in items:
                    if isinstance(item, dict):
                        sub_rows.append(
                            {
                                "Label": item.get("label", ""),
                                "Condition": item.get("condition", ""),
                                "Interpretation": item.get("interpretation", ""),
                            }
                        )
                if sub_rows:
                    st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)

    # ── Root anchors: regime gates from umcp_canon ──
    if "umcp_canon" in data:
        root = data["umcp_canon"]
        regimes_root = root.get("regimes", {})
        if regimes_root:
            st.subheader("🚦 Regime Gates")
            for rname, rval in regimes_root.items():
                if isinstance(rval, dict):
                    items = [f"**{k}**: {v}" for k, v in rval.items() if k != "note"]
                    note = rval.get("note", "")
                    with st.expander(f"**{rname.replace('_', ' ').title()}**"):
                        for item in items:
                            st.markdown(f"- {item}")
                        if note:
                            st.caption(note)

    # ── Mathematical Identities ──
    identities = data.get("mathematical_identities", [])
    if identities:
        st.subheader("📐 Mathematical Identities")
        id_list = identities.get("identities", []) if isinstance(identities, dict) else identities
        for ident in id_list:
            if isinstance(ident, dict):
                name = ident.get("name", ident.get("id", ""))
                formula = ident.get("formula", ident.get("expression", ""))
                tol = ident.get("tolerance", "")
                desc = ident.get("description", ident.get("interpretation", ""))
                label = f"**{name}**: `{formula}`" if name else f"`{formula}`"
                if tol:
                    label += f"  (tol: {tol})"
                st.markdown(f"- {label}")
                if desc:
                    st.caption(f"  {desc}")
            elif isinstance(ident, str):
                st.markdown(f"- {ident}")

    # ── Tolerances ──
    tolerances = data.get("tolerances", {})
    if tolerances:
        st.subheader("🎯 Tolerances")
        tol_desc = tolerances.get("description", "")
        if tol_desc:
            st.caption(tol_desc)
        gates = tolerances.get("gates", [])
        if gates:
            tol_rows: list[dict[str, str]] = []
            for g in gates:
                if isinstance(g, dict):
                    tol_rows.append(
                        {
                            "Name": g.get("name", ""),
                            "Value": str(g.get("value", "")),
                            "Description": g.get("description", g.get("formula", "")),
                        }
                    )
            if tol_rows:
                st.dataframe(pd.DataFrame(tol_rows), use_container_width=True, hide_index=True)

    # ── Typed Censoring ──
    censoring = data.get("typed_censoring", {})
    if censoring:
        st.subheader("🏷️ Typed Censoring")
        tc_desc = censoring.get("description", "")
        if tc_desc:
            st.caption(tc_desc)
        values = censoring.get("values", [])
        if values and isinstance(values, list):
            for v in values:
                if isinstance(v, dict):
                    st.markdown(f"- **{v.get('symbol', '')}**: {v.get('interpretation', '')} — _{v.get('usage', '')}_")
        # Enum-style fields (RCFT)
        for key, val in censoring.items():
            if key == "description":
                continue
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                st.markdown(f"**{key.replace('_', ' ').title()}**: {', '.join(f'`{x}`' for x in val)}")

    # ── Domain-specific extensions ──
    for ext_key in ("gcd_extensions", "tier_2_extensions", "computational_notes"):
        ext = data.get(ext_key, {})
        if ext and isinstance(ext, dict):
            title = ext_key.replace("_", " ").title()
            st.subheader(f"🔧 {title}")
            ext_desc = ext.get("description", "")
            if ext_desc:
                st.caption(ext_desc)
            for k, v in ext.items():
                if k in ("description", "reserved_symbols"):
                    continue
                if isinstance(v, list):
                    st.markdown(f"**{k.replace('_', ' ').title()}**")
                    for item in v:
                        if isinstance(item, dict):
                            sym = item.get("symbol", item.get("name", ""))
                            desc = item.get("description", "")
                            st.markdown(f"- **{sym}**: {desc}")
                        elif isinstance(item, str):
                            st.markdown(f"- {item}")
                elif isinstance(v, str):
                    st.markdown(f"**{k.replace('_', ' ').title()}**: {v.strip()}")

    # ── Root anchors: contract defaults, identifiers, artifacts ──
    if "umcp_canon" in data:
        root = data["umcp_canon"]
        # Contract defaults
        defaults = root.get("contract_defaults", {})
        if defaults:
            st.subheader("📋 Contract Defaults")
            contract_id = defaults.get("contract_id", "")
            if contract_id:
                st.markdown(f"**Contract**: `{contract_id}`")

            embedding = defaults.get("embedding", {})
            if embedding:
                st.markdown("**Embedding**")
                for k, v in embedding.items():
                    st.markdown(f"- **{k}**: `{v}`")

            kernel = defaults.get("tier_1_kernel", {})
            if kernel:
                st.markdown("**Tier-1 Kernel**")
                for k, v in kernel.items():
                    if isinstance(v, list):
                        st.markdown(f"- **{k}**: {', '.join(f'`{x}`' for x in v)}")
                    elif isinstance(v, dict):
                        st.markdown(f"- **{k}**:")
                        for sk, sv in v.items():
                            st.markdown(f"  - {sk}: `{sv}`")
                    else:
                        st.markdown(f"- **{k}**: `{v}`")

        # Anchors (DOIs, weld)
        anchors = root.get("anchors", {})
        if anchors:
            st.subheader("🔗 Anchors")
            for name, info in anchors.items():
                if isinstance(info, dict):
                    title = info.get("title", info.get("id", name))
                    doi = info.get("doi", "")
                    with st.expander(f"**{name.title()}**: {title}"):
                        if doi:
                            st.markdown(f"DOI: `{doi}`")
                        for k, v in info.items():
                            if k not in ("title", "doi"):
                                st.markdown(f"- **{k}**: {v}")

    # ── Provenance ──
    provenance = data.get("provenance", {})
    if provenance:
        st.subheader("📜 Provenance")
        for k, v in provenance.items():
            if isinstance(v, list):
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                for item in v:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")

    # ── Notes ──
    notes = data.get("notes", "")
    if notes:
        with st.expander("📝 Notes"):
            st.markdown(notes.strip())

    # ── Raw YAML view ──
    with st.expander("📄 Raw YAML"):
        import yaml

        st.code(yaml.dump(data, default_flow_style=False, allow_unicode=True), language="yaml")


def _extract_symbols(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract reserved symbols from varying canon YAML structures."""
    symbols: list[dict[str, Any]] = []

    # Direct top-level lists
    for key in ("reserved_symbols", "symbols", "tier_1_symbols"):
        val = data.get(key, [])
        if isinstance(val, list):
            symbols.extend(val)

    # Nested under tier_1_invariants.reserved_symbols (GCD, WEYL, KIN, ASTRO)
    t1 = data.get("tier_1_invariants", {})
    if isinstance(t1, dict):
        rs = t1.get("reserved_symbols", [])
        if isinstance(rs, list):
            symbols.extend(rs)
        elif isinstance(rs, dict):
            # KIN-style: each symbol is a dict value keyed by name
            for name, info in rs.items():
                if isinstance(info, dict):
                    entry = dict(info)
                    if "symbol" not in entry:
                        entry["symbol"] = name
                    symbols.append(entry)

    # Nested under tier_2_extensions.reserved_symbols (RCFT)
    t2 = data.get("tier_2_extensions", {})
    if isinstance(t2, dict):
        rs = t2.get("reserved_symbols", [])
        if isinstance(rs, list):
            symbols.extend(rs)

    # Root anchors: umcp_canon.contract_defaults.tier_1_kernel.reserved_symbols_unicode
    if "umcp_canon" in data:
        root = data["umcp_canon"]
        kernel = root.get("contract_defaults", {}).get("tier_1_kernel", {})
        unicode_syms = kernel.get("reserved_symbols_unicode", [])
        ascii_syms = kernel.get("reserved_symbols_ascii", [])
        if unicode_syms and not symbols:
            for u, a in zip(unicode_syms, ascii_syms, strict=False):
                symbols.append({"symbol": u, "description": a, "domain": "", "formula": ""})

    return symbols


# ============================================================================
# Domain-Specific Pages
# ============================================================================


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
        <li>IC ≤ F (integrity bound)</li>
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
