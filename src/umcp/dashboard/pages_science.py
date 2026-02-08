"""
Science domain dashboard pages: Cosmology, Astronomy, Nuclear, Quantum, Finance, RCFT.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from umcp.dashboard._deps import go, np, pd, st
from umcp.dashboard._utils import (
    _ensure_closures_path,
)


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

    # Ensure closures path is available
    _ensure_closures_path()

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


def render_astronomy_page() -> None:
    """Render interactive Astronomy domain page with all 6 closures."""
    if st is None or go is None or pd is None:
        return

    _ensure_closures_path()

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

    _ensure_closures_path()

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
                regime_color = {"Peak": "🟢", "Plateau": "🟢", "Slope": "🟡"}.get(regime, "🔴")

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
                regime_color = {"Stable": "🟢", "Geological": "🟢", "Laboratory": "🟡", "Eternal": "🟢"}.get(
                    regime, "🔴"
                )

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
                regime_color = {"DoublyMagic": "🟢", "SinglyMagic": "🟢", "NearMagic": "🟡"}.get(regime, "🔴")
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
                regime_color = {"Subfissile": "🟢", "Transitional": "🟡", "Fissile": "🔴"}.get(regime, "🔴")

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
                regime_color = {"ZeroStep": "🟢", "Dominated": "🟢", "Cascade": "🟡"}.get(regime, "🔴")

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
                regime_color = {"AtPeak": "🟢", "NearPeak": "🟢", "Convergent": "🟡"}.get(regime, "🔴")

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

    _ensure_closures_path()

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
                    _t_ratio = result["T_ratio"]
                    _t_str = "INF_REC" if _t_ratio == "INF_REC" else f"{_t_ratio:.4e}"
                    st.metric("T/T_classical", _t_str)
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
                regime_color = {"Pure": "🟢", "High": "🟢", "Mixed": "🟡"}.get(regime, "🔴")

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
        _dx, _dp = presets_unc.get(up, (1.0, 5.27e-25))
        c1, c2 = st.columns(2)
        with c1:
            dx = st.number_input("Δx (nm)", 0.001, 10000.0, _dx, 0.01, key="qm_dx")
        with c2:
            dp = st.number_input("Δp (kg·m/s)", 1e-30, 1e-18, _dp, 1e-26, key="qm_dp", format="%.4e")

        if st.button("Check Uncertainty", key="qm_unc", type="primary"):
            try:
                from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

                result = compute_uncertainty(dx * 1e-9, dp)  # Convert nm → meters
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

    _ensure_closures_path()

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

    _ensure_closures_path()

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
            "🎯 Monostable (tight cluster)": (100, "monostable"),
            "⚖️ Bistable (two basins)": (200, "bistable"),
            "🌀 Multistable (chaotic)": (400, "multistable"),
        }
        _nb, _mode_basin = presets_basin.get(fp3, (100, "bistable"))

        n_basin = st.slider("Series length", 20, 500, _nb, key="rcft_nbasin")

        if st.button("Compute Attractor Basin", key="rcft_basin", type="primary"):
            try:
                from closures.rcft.attractor_basin import compute_attractor_basin

                # Generate time-correlated trajectories so the pathway connects
                t_basin = np.linspace(0, 10, n_basin)
                if _mode_basin == "monostable":
                    # Exponential convergence to a single fixed point
                    omega_arr = 0.05 + 0.25 * np.exp(-t_basin / 2) + 0.01 * np.random.randn(n_basin)
                    S_arr = 0.10 + 0.15 * np.exp(-t_basin / 2) + 0.01 * np.random.randn(n_basin)
                    C_arr = 0.03 + 0.10 * np.exp(-t_basin / 2) + 0.005 * np.random.randn(n_basin)
                elif _mode_basin == "bistable":
                    # Oscillation between two attractor basins
                    omega_arr = 0.15 + 0.12 * np.sin(2 * np.pi * t_basin / 5) + 0.02 * np.random.randn(n_basin)
                    S_arr = 0.20 + 0.10 * np.sin(2 * np.pi * t_basin / 5 + np.pi / 2) + 0.02 * np.random.randn(n_basin)
                    C_arr = 0.10 + 0.08 * np.sin(2 * np.pi * t_basin / 5) + 0.01 * np.random.randn(n_basin)
                else:
                    # Random walk — chaotic / multistable
                    omega_arr = np.clip(0.20 + 0.05 * np.random.randn(n_basin).cumsum() / 10, 0, 0.5)
                    S_arr = np.clip(0.30 + 0.08 * np.random.randn(n_basin).cumsum() / 10, 0, 1)
                    C_arr = np.clip(0.15 + 0.05 * np.random.randn(n_basin).cumsum() / 10, 0, 0.5)
                omega_arr = np.clip(omega_arr, 0, 1)
                S_arr = np.clip(S_arr, 0, 1)
                C_arr = np.clip(C_arr, 0, 1)
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
                    # 3D trajectory with connected pathway and basin coloring
                    traj_class = result.get("trajectory_classification", [])
                    fig = go.Figure()
                    # Connected trajectory line (the pathway)
                    fig.add_trace(
                        go.Scatter3d(
                            x=omega_arr.tolist(),
                            y=S_arr.tolist(),
                            z=C_arr.tolist(),
                            mode="lines",
                            line={"color": "rgba(0,123,255,0.3)", "width": 2},
                            name="Trajectory path",
                            showlegend=True,
                        )
                    )
                    # Points colored by basin assignment
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
                                name="Points (basin)",
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
                                name="Points",
                            )
                        )
                    # Mark start and end of trajectory
                    fig.add_trace(
                        go.Scatter3d(
                            x=[omega_arr[0]],
                            y=[S_arr[0]],
                            z=[C_arr[0]],
                            mode="markers",
                            marker={"size": 8, "color": "#28a745", "symbol": "circle"},
                            name="Start",
                        )
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=[omega_arr[-1]],
                            y=[S_arr[-1]],
                            z=[C_arr[-1]],
                            mode="markers",
                            marker={"size": 8, "color": "#ffc107", "symbol": "square"},
                            name="End",
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
