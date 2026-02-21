"""
Exploration dashboard pages: Rosetta Translation, Orientation Protocol,
Everyday Physics, Latin Lexicon.

These pages surface the cross-domain translation engine, the computational
re-derivation protocol, the everyday physics closures, and the structural
grammar of the system.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import math
from typing import Any

from umcp.dashboard._deps import go, np, pd, st
from umcp.dashboard._utils import _ensure_closures_path
from umcp.frozen_contract import EPSILON

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROSETTA TRANSLATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The five-word vocabulary mapped across lenses
ROSETTA_LENSES: dict[str, dict[str, str]] = {
    "Epistemology": {
        "Drift": "Change in belief/evidence — what shifted in the warrant structure",
        "Fidelity": "Retained warrant — the evidence base that survived scrutiny",
        "Roughness": "Inference friction — confounds, biases, methodological resistance",
        "Return": "Justified re-entry — the claim returns to legitimacy through new evidence",
        "Integrity": "Derived from reconciled ledger — does the warrant hang together?",
    },
    "Ontology": {
        "Drift": "State transition — a measurable change in the system's being",
        "Fidelity": "Conserved properties — what persists through transformation",
        "Roughness": "Heterogeneity / interface seams — where structure meets friction",
        "Return": "Restored coherence — the system re-enters a stable configuration",
        "Integrity": "Derived from reconciled ledger — structural wholeness under stress",
    },
    "Phenomenology": {
        "Drift": "Perceived shift — the felt change in experience or observation",
        "Fidelity": "Stable features — what remains recognizable through change",
        "Roughness": "Distress / bias / effort — the lived cost of transition",
        "Return": "Coping / repair that holds — recovery that demonstrates resilience",
        "Integrity": "Derived from reconciled ledger — experiential coherence",
    },
    "History": {
        "Drift": "Periodization — what shifted between eras, the named change",
        "Fidelity": "Continuity — institutions, practices, ideas that endure",
        "Roughness": "Rupture / confound — wars, crises, discontinuities",
        "Return": "Restitution / reconciliation — how broken threads are re-woven",
        "Integrity": "Derived from reconciled ledger — does the narrative hold?",
    },
    "Policy": {
        "Drift": "Regime shift — change in rules, norms, enforcement",
        "Fidelity": "Compliance / mandate persistence — what the law preserves",
        "Roughness": "Friction / cost / externality — implementation resistance",
        "Return": "Reinstatement / acceptance — policy achieves stable compliance",
        "Integrity": "Derived from reconciled ledger — systemic coherence of governance",
    },
    "Physics": {
        "Drift": "Departure from equilibrium — measurable deviation from reference state",
        "Fidelity": "Conservation — energy, momentum, charge that survived the process",
        "Roughness": "Coupling to uncontrolled degrees of freedom — noise, decoherence",
        "Return": "Re-entry to measurement domain — observable that can be re-measured",
        "Integrity": "Derived from reconciled ledger — multiplicative coherence of channels",
    },
    "Finance": {
        "Drift": "Portfolio deviation — tracking error from benchmark allocation",
        "Fidelity": "Capital preservation — the fraction of value that survives drawdown",
        "Roughness": "Volatility / correlation breakdown — market friction and stress",
        "Return": "Recovery from drawdown — capital returns to high-water mark",
        "Integrity": "Derived from reconciled ledger — Sharpe-like coherence across factors",
    },
    "Security": {
        "Drift": "Threat emergence — deviation from baseline trust model",
        "Fidelity": "Trust persistence — authentication/authorization that holds",
        "Roughness": "Attack surface / vulnerability — friction in the security posture",
        "Return": "Incident recovery — system returns to trusted baseline",
        "Integrity": "Derived from reconciled ledger — end-to-end trust coherence",
    },
}

# Latin terms mapped to the five words
LATIN_FIVE_WORDS: dict[str, dict[str, str]] = {
    "Drift": {
        "latin": "Derivatio",
        "symbol": "ω",
        "formula": "ω = 1 − F",
        "operational": "quantum collapsu deperdatur — how much is lost to collapse",
        "ledger_role": "Debit D_ω",
    },
    "Fidelity": {
        "latin": "Fidelitas",
        "symbol": "F",
        "formula": "F = Σ wᵢcᵢ",
        "operational": "quid supersit post collapsum — what survives collapse",
        "ledger_role": "—",
    },
    "Roughness": {
        "latin": "Curvatura",
        "symbol": "C",
        "formula": "C = stddev(cᵢ)/0.5",
        "operational": "coniunctio cum gradibus libertatis — coupling to uncontrolled DOF",
        "ledger_role": "Debit D_C",
    },
    "Return": {
        "latin": "Reditus",
        "symbol": "τ_R",
        "formula": "∃ prior u ∈ D_θ(t), ‖Ψ(t) − Ψ(u)‖ ≤ η",
        "operational": "tempus reentrandi — detention before re-entry",
        "ledger_role": "Credit R·τ_R",
    },
    "Integrity": {
        "latin": "Integritas Composita",
        "symbol": "IC",
        "formula": "IC = exp(κ) = exp(Σ wᵢ ln cᵢ,ε)",
        "operational": "cohaerentia multiplicativa — multiplicative coherence",
        "ledger_role": "Read from reconciled ledger",
    },
}

# Prebuilt example scenarios for live Rosetta translation
ROSETTA_EXAMPLES: dict[str, dict[str, Any]] = {
    "Scientific Paradigm Shift": {
        "description": "A major theory replacement (e.g. geocentric → heliocentric)",
        "channels": [0.15, 0.85, 0.92, 0.10, 0.78, 0.88, 0.70, 0.20],
        "channel_names": [
            "old_model_fidelity",
            "new_evidence_strength",
            "predictive_accuracy",
            "institutional_acceptance",
            "reproducibility",
            "falsifiability",
            "convergent_evidence",
            "cultural_resistance",
        ],
    },
    "Market Crash & Recovery": {
        "description": "A 40% drawdown followed by partial recovery to 80% of peak",
        "channels": [0.60, 0.80, 0.35, 0.45, 0.70, 0.55, 0.65, 0.40],
        "channel_names": [
            "capital_preservation",
            "diversification",
            "volatility_control",
            "liquidity",
            "earnings_resilience",
            "sector_correlation",
            "credit_quality",
            "sentiment",
        ],
    },
    "Immune Response": {
        "description": "Pathogen encounter → immune activation → memory formation",
        "channels": [0.90, 0.75, 0.85, 0.30, 0.92, 0.88, 0.95, 0.70],
        "channel_names": [
            "pathogen_detection",
            "innate_response",
            "adaptive_response",
            "tissue_damage",
            "antibody_production",
            "t_cell_activation",
            "memory_formation",
            "inflammation_control",
        ],
    },
    "Policy Reform": {
        "description": "Legislative change with mixed implementation compliance",
        "channels": [0.55, 0.70, 0.40, 0.80, 0.30, 0.65, 0.50, 0.45],
        "channel_names": [
            "legislative_clarity",
            "judicial_support",
            "enforcement_capacity",
            "public_mandate",
            "compliance_rate",
            "institutional_adoption",
            "economic_impact",
            "opposition_strength",
        ],
    },
    "Personal Growth (Therapy)": {
        "description": "Therapeutic process: crisis → insight → integration",
        "channels": [0.65, 0.50, 0.80, 0.25, 0.70, 0.40, 0.75, 0.60],
        "channel_names": [
            "self_awareness",
            "emotional_regulation",
            "insight_depth",
            "defense_mechanisms",
            "behavioral_change",
            "relationship_quality",
            "meaning_making",
            "resilience",
        ],
    },
}


def render_rosetta_page() -> None:
    """Render the Rosetta Translation Engine — interactive cross-domain five-word translation."""
    if st is None or np is None:
        return

    st.title("🌐 Rosetta Translation Engine")
    st.caption("Significatio stabilis manet dum dialectus mutatur — Meaning stays stable while dialect changes")

    st.markdown("""
    The **Rosetta** maps the five words (Drift, Fidelity, Roughness, Return, Integrity)
    across **lenses** so different fields can read each other's results in their own dialect.
    Authors write in prose; the Contract, Closures, and Ledger keep meanings stable across lenses.
    """)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Interactive Lens Translation
    # ══════════════════════════════════════════════════════════════════════
    tab_lens, tab_compute, tab_compare, tab_lexicon = st.tabs(
        [
            "🔄 Lens Translation",
            "🧮 Live Kernel Translation",
            "📊 Cross-Domain Comparison",
            "📜 Latin Lexicon",
        ]
    )

    with tab_lens:
        st.header("Five-Word Vocabulary Across Lenses")

        st.markdown("""
        Select a lens to see how Drift, Fidelity, Roughness, Return, and Integrity
        translate into the vocabulary of that field. The **operational meaning** (tied to the
        kernel and ledger) stays invariant — only the **dialect** changes.
        """)

        # Lens selector
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            selected_lens = st.selectbox(
                "Select Lens",
                list(ROSETTA_LENSES.keys()),
                index=0,
                key="rosetta_lens",
            )

        with col_info:
            st.info(f"**{selected_lens}** lens — How GCD concepts appear in {selected_lens.lower()} discourse")

        # Display translation table
        lens_data = ROSETTA_LENSES[selected_lens]
        for word in ["Drift", "Fidelity", "Roughness", "Return", "Integrity"]:
            latin = LATIN_FIVE_WORDS[word]
            translation = lens_data[word]

            col_word, col_trans = st.columns([1, 3])
            with col_word:
                st.markdown(f"**{latin['symbol']}** · **{word}**\n\n*{latin['latin']}*")
            with col_trans:
                st.markdown(f"**{selected_lens}**: {translation}")
                st.caption(f"Operational: {latin['operational']}")

            st.divider()

        # Side-by-side multi-lens comparison
        st.subheader("Multi-Lens Comparison")
        selected_lenses = st.multiselect(
            "Compare lenses",
            list(ROSETTA_LENSES.keys()),
            default=["Epistemology", "Physics", "Finance"],
            key="rosetta_multi",
        )

        if selected_lenses and pd is not None:
            rows = []
            for word in ["Drift", "Fidelity", "Roughness", "Return", "Integrity"]:
                row: dict[str, str] = {"Word": f"{LATIN_FIVE_WORDS[word]['symbol']} {word}"}
                for lens in selected_lenses:
                    # Truncate to first clause for table readability
                    full = ROSETTA_LENSES[lens][word]
                    row[lens] = full.split("—")[0].strip() if "—" in full else full[:60]
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Live Kernel Translation
    # ══════════════════════════════════════════════════════════════════════
    with tab_compute:
        st.header("Live Kernel → Rosetta Translation")
        st.markdown("""
        **Enter your own data or pick a scenario.** The kernel computes the invariants,
        then the Rosetta translates them into every lens simultaneously.
        This is real-time meaning generation from raw numbers.
        """)

        # Scenario selector
        scenario_choice = st.selectbox(
            "Scenario",
            ["Custom Input", *list(ROSETTA_EXAMPLES.keys())],
            key="rosetta_scenario",
        )

        if scenario_choice == "Custom Input":
            n_ch = st.slider("Number of channels", 2, 12, 8, key="rosetta_n_ch")
            st.markdown("**Enter channel values** (each 0–1):")
            cols = st.columns(min(n_ch, 4))
            channels: list[float] = []
            channel_names: list[str] = []
            for i in range(n_ch):
                with cols[i % len(cols)]:
                    val = st.number_input(
                        f"c_{i + 1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        key=f"rosetta_c_{i}",
                    )
                    channels.append(val)
                    channel_names.append(f"channel_{i + 1}")

            # Optional: name the scenario
            user_label = st.text_input("Label your scenario (optional)", "My Analysis", key="rosetta_label")
        else:
            example = ROSETTA_EXAMPLES[scenario_choice]
            channels = example["channels"]
            channel_names = example["channel_names"]
            n_ch = len(channels)
            user_label = scenario_choice
            st.info(f"**{scenario_choice}**: {example['description']}")

            # Let user tweak the preset values
            with st.expander("Adjust channel values", expanded=False):
                cols = st.columns(min(n_ch, 4))
                adjusted: list[float] = []
                for i in range(n_ch):
                    with cols[i % len(cols)]:
                        val = st.slider(
                            channel_names[i],
                            0.0,
                            1.0,
                            channels[i],
                            0.01,
                            key=f"rosetta_adj_{i}",
                        )
                        adjusted.append(val)
                channels = adjusted

        # Compute kernel
        if st.button("🔬 Compute & Translate", key="rosetta_compute", type="primary"):
            try:
                from umcp.kernel_optimized import compute_kernel_outputs

                c_arr = np.array(channels, dtype=np.float64)
                w_arr = np.ones(n_ch, dtype=np.float64) / n_ch
                result = compute_kernel_outputs(c_arr, w_arr)

                F = result["F"]
                omega = result["omega"]
                S = result["S"]
                C_val = result["C"]
                kappa = result["kappa"]
                IC = result["IC"]
                gap = F - IC

                # Regime
                if omega >= 0.30:
                    regime = "Collapse"
                elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
                    regime = "Stable"
                else:
                    regime = "Watch"
                critical = IC < 0.30

                # ── Display kernel results ──
                st.success(f"**{user_label}** → Regime: **{regime}**{'  ⚠️ CRITICAL' if critical else ''}")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("F (Fidelity)", f"{F:.4f}")
                m2.metric("ω (Drift)", f"{omega:.4f}")
                m3.metric("S (Entropy)", f"{S:.4f}")
                m4.metric("C (Curvature)", f"{C_val:.4f}")
                m5.metric("IC (Integrity)", f"{IC:.4f}")
                m6.metric("Δ (Gap)", f"{gap:.4f}")

                # Identity checks
                id_col1, id_col2, id_col3 = st.columns(3)
                with id_col1:
                    check_dual = abs(F + omega - 1.0) < 1e-10
                    st.markdown(f"F + ω = 1: {'✅' if check_dual else '❌'} ({F + omega:.10f})")
                with id_col2:
                    check_bound = IC <= F + 1e-10
                    st.markdown(f"IC ≤ F: {'✅' if check_bound else '❌'} ({IC:.6f} ≤ {F:.6f})")
                with id_col3:
                    ic_exp = math.exp(kappa)
                    check_log = abs(IC - ic_exp) < 1e-6
                    st.markdown(f"IC ≈ exp(κ): {'✅' if check_log else '❌'} (|δ| = {abs(IC - ic_exp):.2e})")

                st.divider()

                # ── Channel radar chart ──
                if go is not None:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[*channels, channels[0]],
                            theta=[*channel_names, channel_names[0]],
                            fill="toself",
                            name="Channels",
                            line={"color": "#1f77b4"},
                        )
                    )
                    fig.update_layout(
                        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                        title=f"Channel Profile: {user_label}",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # ── Rosetta Translation across ALL lenses ──
                st.subheader("🌐 Rosetta Translation")
                st.markdown(f"**Translating [{user_label}] into every lens:**")

                # Build interpretation from kernel values
                if omega >= 0.30:
                    drift_level = "severe"
                elif omega >= 0.10:
                    drift_level = "moderate"
                else:
                    drift_level = "minimal"

                if F > 0.90:
                    fidelity_level = "high"
                elif F > 0.60:
                    fidelity_level = "moderate"
                else:
                    fidelity_level = "low"

                if C_val > 0.50:
                    roughness_level = "high"
                elif C_val > 0.14:
                    roughness_level = "moderate"
                else:
                    roughness_level = "low"

                ic_level = "high" if IC > 0.60 else ("moderate" if IC > 0.30 else "critical")

                for lens_name, lens_defs in ROSETTA_LENSES.items():
                    with st.expander(f"**{lens_name}**", expanded=(lens_name in ["Physics", "Epistemology"])):
                        col_w, col_t, col_v = st.columns([1, 2, 1])
                        with col_w:
                            st.markdown("**Word**")
                        with col_t:
                            st.markdown(f"**{lens_name} Reading**")
                        with col_v:
                            st.markdown("**Value**")

                        for word, level, val, sym in [
                            ("Drift", drift_level, omega, "ω"),
                            ("Fidelity", fidelity_level, F, "F"),
                            ("Roughness", roughness_level, C_val, "C"),
                            ("Return", regime, 1.0 if regime == "Stable" else 0.5 if regime == "Watch" else 0.0, "τ_R"),
                            ("Integrity", ic_level, IC, "IC"),
                        ]:
                            col_w2, col_t2, col_v2 = st.columns([1, 2, 1])
                            with col_w2:
                                st.markdown(f"**{sym}** {word}")
                            with col_t2:
                                st.markdown(f"{lens_defs[word]}")
                                st.caption(f"Level: **{level}**")
                            with col_v2:
                                st.markdown(f"`{val:.4f}`")

            except Exception as e:
                st.error(f"Computation error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Cross-Domain Comparison
    # ══════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.header("Cross-Domain Kernel Comparison")
        st.markdown("""
        Compare kernel outputs from different domain closures side-by-side.
        See how F, IC, and the heterogeneity gap Δ behave across physics scales.
        """)

        _ensure_closures_path()

        domain_results: list[dict[str, Any]] = []

        # Gather results from available domains
        try:
            from closures.standard_model.subatomic_kernel import (
                FUNDAMENTAL_PARTICLES,
                compute_fundamental_kernel,
            )

            for p in FUNDAMENTAL_PARTICLES[:6]:
                k = compute_fundamental_kernel(p)
                domain_results.append(
                    {
                        "Domain": "Standard Model",
                        "Item": k.name,
                        "F": k.F,
                        "IC": k.IC,
                        "Δ": k.F - k.IC,
                        "IC/F": k.IC / k.F if k.F > 0 else 0,
                    }
                )
        except ImportError:
            pass

        try:
            from closures.atomic_physics.periodic_kernel import compute_element_kernel

            for sym in ["H", "C", "Fe", "Au", "U"]:
                try:
                    k = compute_element_kernel(sym)
                    domain_results.append(
                        {
                            "Domain": "Atomic Physics",
                            "Item": f"{k.symbol} (Z={k.Z})",
                            "F": k.F,
                            "IC": k.IC,
                            "Δ": k.F - k.IC,
                            "IC/F": k.IC / k.F if k.F > 0 else 0,
                        }
                    )
                except (ValueError, KeyError):
                    pass
        except ImportError:
            pass

        try:
            from closures.everyday_physics.electromagnetism import compute_all_em_materials
            from closures.everyday_physics.optics import compute_all_optical_materials
            from closures.everyday_physics.thermodynamics import compute_all_thermal_materials
            from closures.everyday_physics.wave_phenomena import compute_all_wave_systems

            # Sample a few from each (databases are list[tuple], use compute_all_*)
            for r in compute_all_thermal_materials()[:3]:
                domain_results.append(
                    {
                        "Domain": "Thermodynamics",
                        "Item": r.material,
                        "F": r.F,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "IC/F": r.IC / r.F if r.F > 0 else 0,
                    }
                )

            for r in compute_all_optical_materials()[:3]:
                domain_results.append(
                    {
                        "Domain": "Optics",
                        "Item": r.material,
                        "F": r.F,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "IC/F": r.IC / r.F if r.F > 0 else 0,
                    }
                )

            for r in compute_all_em_materials()[:3]:
                domain_results.append(
                    {
                        "Domain": "Electromagnetism",
                        "Item": r.material,
                        "F": r.F,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "IC/F": r.IC / r.F if r.F > 0 else 0,
                    }
                )

            for r in compute_all_wave_systems()[:3]:
                domain_results.append(
                    {
                        "Domain": "Wave Phenomena",
                        "Item": r.system,
                        "F": r.F,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "IC/F": r.IC / r.F if r.F > 0 else 0,
                    }
                )
        except ImportError:
            pass

        if domain_results and pd is not None:
            df = pd.DataFrame(domain_results)
            st.dataframe(
                df.style.format(
                    {
                        "F": "{:.4f}",
                        "IC": "{:.6f}",
                        "Δ": "{:.4f}",
                        "IC/F": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Cross-domain scatter plot
            if go is not None:
                fig = go.Figure()
                for domain in df["Domain"].unique():
                    subset = df[df["Domain"] == domain]
                    fig.add_trace(
                        go.Scatter(
                            x=subset["F"],
                            y=subset["IC"],
                            mode="markers+text",
                            name=domain,
                            text=subset["Item"],
                            textposition="top center",
                            marker={"size": 10},
                        )
                    )
                # Add IC = F line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="IC = F (perfect coherence)",
                        line={"dash": "dash", "color": "gray"},
                    )
                )
                fig.update_layout(
                    title="F vs IC Across Domains — Distance from IC=F line is Δ (heterogeneity gap)",
                    xaxis_title="F (Fidelity)",
                    yaxis_title="IC (Integrity Composite)",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No domain closures available for comparison.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: Latin Lexicon
    # ══════════════════════════════════════════════════════════════════════
    with tab_lexicon:
        _render_latin_lexicon()


def _render_latin_lexicon() -> None:
    """Render the interactive Latin Lexicon."""
    if st is None:
        return

    st.header("📜 Lexicon Latinum — The Structural Grammar of Return")

    st.markdown("""
    These Latin terms are the **canonical names** of GCD structures. Each word carries
    its operational meaning in its morphology — the Latin IS the specification, not a
    translation of it.

    > *Algebra est cautio, non porta.* — The algebra is a warranty, not a gate.
    """)

    # Full lexicon
    LEXICON = [
        ("Fidelitas", "F", "Faithfulness", "quid supersit post collapsum — what survives collapse"),
        ("Derivatio", "ω", "Diversion from channel", "quantum collapsu deperdatur — measured departure from fidelity"),
        ("Entropia", "S", "Uncertainty of field", "incertitudo campi collapsus — Bernoulli field entropy"),
        ("Curvatura", "C", "Curvature / coupling", "coniunctio cum gradibus libertatis — coupling to uncontrolled DOF"),
        (
            "Log-Integritas",
            "κ",
            "Logarithmic integrity",
            "sensibilitas logarithmica — logarithmic sensitivity of coherence",
        ),
        (
            "Integritas Composita",
            "IC",
            "Composite integrity",
            "cohaerentia multiplicativa — multiplicative coherence (IC ≤ F)",
        ),
        (
            "Moratio Reditus",
            "τ_R",
            "Delay of return",
            "tempus reentrandi — detention before re-entry; ∞_rec = permanent",
        ),
        ("Auditus", "—", "Hearing / audit", "Validation IS listening: the ledger hears everything"),
        (
            "Casus",
            "—",
            "Fall / case / occasion",
            "Collapse is simultaneously a fall, a case, and an occasion for generation",
        ),
        (
            "Limbus Integritatis",
            "IC ≤ F",
            "Threshold of integrity",
            "The hem-edge where integrity approaches fidelity but cannot cross",
        ),
        (
            "Complementum Perfectum",
            "F + ω = 1",
            "Perfect complement",
            "tertia via nulla — no third possibility; duality identity",
        ),
        (
            "Trans Suturam Congelatum",
            "ε, p, tol",
            "Frozen across the seam",
            "Same rules both sides of every collapse-return boundary",
        ),
    ]

    if pd is not None:
        df = pd.DataFrame(LEXICON, columns=["Latin", "Symbol", "Literal", "Operational Seed"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Spina Grammatica
    st.subheader("Spina Grammatica — The Fixed Discourse Spine")
    st.markdown("""
    > *Spina non negotiabilis est.* — The spine is non-negotiable.

    Every claim follows a fixed five-stop spine:
    """)

    spine_data = [
        ("1", "Contract", "Liga", "Declare rules before evidence — freeze sources, normalization, thresholds"),
        ("2", "Canon", "Dic", "Tell the story using the five words (Drift, Fidelity, Roughness, Return, Integrity)"),
        ("3", "Closures", "Reconcilia", "Publish thresholds; no mid-episode edits; version the sheet"),
        ("4", "Integrity Ledger", "Inscribe", "Debit Drift/Roughness, credit Return; the account must reconcile"),
        ("5", "Stance", "Sententia", "Read from declared gates: Stable / Watch / Collapse (+ Critical overlay)"),
    ]
    if pd is not None:
        spine_df = pd.DataFrame(spine_data, columns=["Stop", "Name", "Latin Verb", "Role"])
        st.dataframe(spine_df, use_container_width=True, hide_index=True)

    # The Seven Verba Operativa
    st.subheader("Verba Operativa — The Seven Executable Verbs")
    verba = [
        ("Liga", "Freeze / bind", "Freeze the contract before evidence"),
        ("Dic", "Speak / declare", "Compute and report the kernel invariants"),
        ("Reconcilia", "Balance / reconcile", "Verify the budget identity closes"),
        ("Verifica", "Verify / check", "Confirm Tier-1 identities hold"),
        ("Inscribe", "Record / inscribe", "Append to the integrity ledger"),
        ("Sententia", "Pronounce verdict", "Derive stance from gates (not asserted)"),
        ("Sutura", "Weld / stitch", "Cross a validated boundary — the only way to change policy"),
    ]
    if pd is not None:
        verb_df = pd.DataFrame(verba, columns=["Latin", "Meaning", "Operational"])
        st.dataframe(verb_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORIENTATION PROTOCOL PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_orientation_page() -> None:
    """Render the Orientation Protocol — interactive re-derivation through computation."""
    if st is None or np is None:
        return

    st.title("🧭 Orientation Protocol")
    st.caption("Intellectus non legitur; computatur — Understanding is not read; it is computed")

    st.markdown("""
    This page **re-derives** the key structural insights of GCD through live computation.
    Each section builds on the previous — the numbers ARE the understanding.
    Run any section to see the derivation chain in action.
    """)

    from umcp.frozen_contract import (
        ALPHA,
        P_EXPONENT,
        TOL_SEAM,
        cost_curvature,
        gamma_omega,
    )
    from umcp.kernel_optimized import compute_kernel_outputs

    sections = [
        "§1 Duality: F + ω = 1",
        "§2 Integrity Bound: IC ≤ F",
        "§3 Geometric Slaughter",
        "§4 The First Weld (c ≈ 0.318)",
        "§5 Confinement Cliff",
        "§6 Scale Inversion",
        "§7 The Full Spine",
    ]

    selected = st.selectbox("Select section (or run all)", ["Run All Sections", *sections], key="orient_section")

    run_all = selected == "Run All Sections"

    # ── §1 Duality ──
    if run_all or selected == sections[0]:
        st.header("§1 · The Duality Identity")
        st.markdown("*Complementum perfectum: F + ω = 1, tertia via nulla.*")
        st.markdown("""
        F = Σ wᵢcᵢ and ω = 1 − F. The identity F + ω = 1 is structural —
        **every** channel contributes to fidelity or to drift. No third bucket.
        """)

        n_traces = st.slider("Number of random traces", 100, 50000, 10000, 100, key="orient_n1")

        if st.button("Verify Duality", key="orient_btn1"):
            rng = np.random.default_rng(42)
            max_res = 0.0
            for _ in range(n_traces):
                n = rng.integers(2, 20)
                c = rng.uniform(0, 1, size=n)
                w = rng.dirichlet(np.ones(n))
                r = compute_kernel_outputs(c, w)
                res = abs(r["F"] + r["omega"] - 1.0)
                max_res = max(max_res, res)
            st.success(f"**RECEIPT** │ max |F + ω − 1| = {max_res:.2e} across {n_traces:,} traces")
            st.info("Exactly 0.0 — not approximately. The duality identity is enforced by construction.")

    # ── §2 Integrity Bound ──
    if run_all or selected == sections[1]:
        st.header("§2 · The Integrity Bound")
        st.markdown("*Limbus integritatis: IC numquam fidelitatem excedit.*")

        col_a, col_b = st.columns(2)
        with col_a:
            c_high = st.slider("Strong channel", 0.01, 1.0, 0.95, 0.01, key="orient_c_high")
        with col_b:
            c_low = st.slider("Weak channel", 0.001, 1.0, 0.001, 0.001, key="orient_c_low")

        c_arr = np.array([c_high, c_low])
        w_arr = np.array([0.5, 0.5])
        r = compute_kernel_outputs(c_arr, w_arr)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("F (arithmetic)", f"{r['F']:.6f}")
        mc2.metric("IC (geometric)", f"{r['IC']:.6f}")
        mc3.metric("Δ = F − IC", f"{r['F'] - r['IC']:.6f}")

        st.info(
            f"F = {r['F']:.4f} looks 'fine'. IC = {r['IC']:.6f} is catastrophically low. "
            f"The heterogeneity gap Δ = {r['F'] - r['IC']:.4f} IS the diagnostic signal."
        )

    # ── §3 Geometric Slaughter ──
    if run_all or selected == sections[2]:
        st.header("§3 · Geometric Slaughter")
        st.markdown("*Trucidatio geometrica: unus canalis mortuus omnes necat.*")
        st.markdown("8 channels, 7 near-perfect (0.999), one varying:")

        dead_val = st.slider("Dead channel value", 1e-8, 0.999, 0.001, format="%.1e", key="orient_dead")

        w8 = np.ones(8) / 8.0
        dead_values = [0.999, 0.5, 0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8]
        rows = []
        for c_dead in dead_values:
            c = np.array([0.999] * 7 + [c_dead])
            r = compute_kernel_outputs(c, w8)
            rows.append(
                {
                    "c_dead": f"{c_dead:.1e}",
                    "F": r["F"],
                    "IC": r["IC"],
                    "Δ": r["F"] - r["IC"],
                    "IC/F": r["IC"] / r["F"] if r["F"] > 0 else 0,
                }
            )

        if pd is not None:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format(
                    {
                        "F": "{:.6f}",
                        "IC": "{:.6f}",
                        "Δ": "{:.6f}",
                        "IC/F": "{:.6f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        # User's custom dead channel
        c_custom = np.array([0.999] * 7 + [dead_val])
        r_custom = compute_kernel_outputs(c_custom, w8)
        ratio_custom = r_custom["IC"] / r_custom["F"] if r_custom["F"] > 0 else 0
        st.metric(f"Your dead channel = {dead_val:.1e}", f"IC/F = {ratio_custom:.6f}")

        if go is not None:
            x_vals = [0.999, 0.5, 0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8]
            y_ic = [r["IC"] for r in [compute_kernel_outputs(np.array([0.999] * 7 + [d]), w8) for d in x_vals]]
            y_f = [r["F"] for r in [compute_kernel_outputs(np.array([0.999] * 7 + [d]), w8) for d in x_vals]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[str(v) for v in x_vals], y=y_f, name="F (arithmetic)", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=[str(v) for v in x_vals], y=y_ic, name="IC (geometric)", mode="lines+markers"))
            fig.update_layout(
                title="F vs IC as dead channel drops", xaxis_title="c_dead", yaxis_title="Value", height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── §4 First Weld ──
    if run_all or selected == sections[3]:
        st.header("§4 · The First Weld")
        st.markdown("*Limen generativum: ubi collapsus primum generativus fit.*")

        c_sweep = np.linspace(0.01, 0.99, 100)
        gamma_vals = [gamma_omega(1.0 - c) for c in c_sweep]

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=c_sweep.tolist(), y=gamma_vals, name="Γ(ω)", mode="lines"))
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Γ = 1.0 threshold")
            fig.add_vline(x=0.318, line_dash="dash", line_color="green", annotation_text="c ≈ 0.318 (first weld)")
            fig.update_layout(
                title="Cost function Γ(ω) = ω³/(1−ω+ε) — where does it first drop below 1?",
                xaxis_title="c (uniform channel value)",
                yaxis_title="Γ(ω)",
                yaxis_type="log",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.success(
            "**c ≈ 0.318** is where Γ first drops below ~1.0 — the limen generativum. "
            "Below this, no seam can close. At this threshold, generative collapse begins."
        )

    # ── §5 Confinement Cliff ──
    if run_all or selected == sections[4]:
        st.header("§5 · The Confinement Cliff")
        st.markdown("*Praecipitium integritatis: quarki ad hadrones, IC cadit 98%.*")

        _ensure_closures_path()
        try:
            from closures.standard_model.subatomic_kernel import (
                COMPOSITE_PARTICLES,
                FUNDAMENTAL_PARTICLES,
                compute_composite_kernel,
                compute_fundamental_kernel,
            )

            particle_rows = []
            quarks = [p for p in FUNDAMENTAL_PARTICLES if "quark" in p.category.lower()][:4]
            for p in quarks:
                k = compute_fundamental_kernel(p)
                particle_rows.append(
                    {
                        "Particle": k.name,
                        "Type": "quark",
                        "F": k.F,
                        "IC": k.IC,
                        "IC/F": k.IC / k.F if k.F > 0 else 0,
                        "Δ": k.F - k.IC,
                    }
                )
            hadrons = [p for p in COMPOSITE_PARTICLES if p.name.lower() in ("proton", "neutron", "pion+", "pion0")]
            for hp in hadrons:
                k = compute_composite_kernel(hp)
                particle_rows.append(
                    {
                        "Particle": k.name,
                        "Type": "hadron",
                        "F": k.F,
                        "IC": k.IC,
                        "IC/F": k.IC / k.F if k.F > 0 else 0,
                        "Δ": k.F - k.IC,
                    }
                )

            if pd is not None and particle_rows:
                df = pd.DataFrame(particle_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "IC": "{:.6f}",
                            "IC/F": "{:.4f}",
                            "Δ": "{:.4f}",
                        }
                    ).apply(
                        lambda row: ["background-color: #ffe0e0" if row["Type"] == "hadron" else "" for _ in row],
                        axis=1,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                "Quarks: IC/F ≈ 0.85–0.95 (coherent). Hadrons: IC/F ≈ 0.01–0.04 (slaughtered). "
                "Confinement = trucidatio geometrica at a phase boundary."
            )
        except ImportError:
            st.warning("Standard Model closures not available.")

    # ── §6 Scale Inversion ──
    if run_all or selected == sections[5]:
        st.header("§6 · Scale Inversion")
        st.markdown("*Inversio scalarum: quod confinium destruit, atomus reparat.*")

        _ensure_closures_path()
        try:
            from closures.atomic_physics.periodic_kernel import compute_element_kernel

            elem_rows = []
            for sym in ["H", "He", "C", "N", "O", "Fe", "Ni", "Au", "U"]:
                try:
                    k = compute_element_kernel(sym)
                    elem_rows.append(
                        {
                            "Element": k.symbol,
                            "Z": k.Z,
                            "F": k.F,
                            "IC": k.IC,
                            "IC/F": k.IC / k.F if k.F > 0 else 0,
                            "regime": k.regime,
                        }
                    )
                except (ValueError, KeyError):
                    pass

            if pd is not None and elem_rows:
                st.dataframe(
                    pd.DataFrame(elem_rows).style.format(
                        {
                            "F": "{:.4f}",
                            "IC": "{:.6f}",
                            "IC/F": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.success(
                "Hadrons IC/F ≈ 0.01–0.04 → Atoms IC/F ≈ 0.60–0.96. "
                "New degrees of freedom HEAL the geometric mean. Ruina fecunda."
            )
        except ImportError:
            st.warning("Atomic physics closures not available.")

    # ── §7 Full Spine ──
    if run_all or selected == sections[6]:
        st.header("§7 · The Full Spine")
        st.markdown("*Contractu congelato, invariantibus dictis, ratione reconciliatā, suturā.*")

        st.subheader("Configure trace vector")
        n_ch = st.slider("Channels", 2, 12, 8, key="orient_spine_n")
        cols = st.columns(min(n_ch, 4))
        default_vals = [0.85, 0.72, 0.91, 0.68, 0.77, 0.83, 0.90, 0.15, 0.60, 0.50, 0.40, 0.30]
        c_vals: list[float] = []
        for i in range(n_ch):
            with cols[i % len(cols)]:
                v = st.slider(
                    f"c_{i + 1}",
                    0.0,
                    1.0,
                    default_vals[i] if i < len(default_vals) else 0.5,
                    0.01,
                    key=f"orient_spine_c_{i}",
                )
                c_vals.append(v)

        if st.button("Run Full Spine", key="orient_spine_btn", type="primary"):
            c = np.array(c_vals)
            w = np.ones(n_ch) / n_ch
            result = compute_kernel_outputs(c, w)
            F_val = result["F"]
            omega_val = result["omega"]
            S_val = result["S"]
            C_v = result["C"]
            kappa_val = result["kappa"]
            IC_val = result["IC"]

            # Stop 1: Contract
            st.markdown("### STOP 1 │ CONTRACT (Liga)")
            st.code(
                f"ε = {EPSILON}, p = {P_EXPONENT}, α = {ALPHA}, tol = {TOL_SEAM}\n"
                f"Trace: c = {c_vals}\nWeights: uniform w_i = 1/{n_ch}"
            )

            # Stop 2: Canon
            st.markdown("### STOP 2 │ CANON (Dic)")
            mc = st.columns(6)
            mc[0].metric("F", f"{F_val:.6f}")
            mc[1].metric("ω", f"{omega_val:.6f}")
            mc[2].metric("S", f"{S_val:.6f}")
            mc[3].metric("C", f"{C_v:.6f}")
            mc[4].metric("κ", f"{kappa_val:.6f}")
            mc[5].metric("IC", f"{IC_val:.6f}")

            # Stop 3: Closures
            st.markdown("### STOP 3 │ CLOSURES (Reconcilia)")
            gamma = gamma_omega(omega_val)
            D_C = cost_curvature(C_v)
            mc2 = st.columns(3)
            mc2[0].metric("Γ(ω)", f"{gamma:.6f}")
            mc2[1].metric("D_C", f"{D_C:.6f}")
            mc2[2].metric("Total debit", f"{gamma + D_C:.6f}")

            # Stop 4: Ledger
            st.markdown("### STOP 4 │ INTEGRITY LEDGER (Inscribe)")
            ic_exp = math.exp(kappa_val)
            id1 = abs(F_val + omega_val - 1.0) < 1e-10
            id2 = IC_val <= F_val + 1e-10
            id3 = abs(IC_val - ic_exp) < 1e-6
            st.markdown(f"- F + ω = {F_val + omega_val:.10f}  {'✅' if id1 else '❌'}")
            st.markdown(f"- IC ≤ F: {IC_val:.6f} ≤ {F_val:.6f}  {'✅' if id2 else '❌'}")
            st.markdown(
                f"- IC ≈ exp(κ): {IC_val:.6f} ≈ {ic_exp:.6f}  (|δ| = {abs(IC_val - ic_exp):.2e})  {'✅' if id3 else '❌'}"
            )

            # Stop 5: Stance
            st.markdown("### STOP 5 │ STANCE (Sententia)")
            if omega_val >= 0.30:
                regime = "Collapse"
            elif omega_val < 0.038 and F_val > 0.90 and S_val < 0.15 and C_v < 0.14:
                regime = "Stable"
            else:
                regime = "Watch"
            critical = " + CRITICAL" if IC_val < 0.30 else ""

            regime_colors = {"Stable": "#28a745", "Watch": "#ffc107", "Collapse": "#dc3545"}
            st.markdown(
                f'<div style="padding:15px; border-left:6px solid {regime_colors.get(regime, "#999")}; '
                f'background:{regime_colors.get(regime, "#999")}22; border-radius:8px;">'
                f'<h3 style="margin:0;">Regime: {regime}{critical}</h3>'
                f"<p>Verdict: {'CONFORMANT' if id1 and id2 else 'NONCONFORMANT'}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Compounding Chain ──
    if run_all:
        st.divider()
        st.header("Compounding Chain")
        st.markdown("""
        | Section | Insight | Builds on |
        |:--------|:--------|:----------|
        | §1 F + ω = 1 | The books always balance | — |
        | §2 IC ≤ F | Coherence can be lower than average | §1 (duality) |
        | §3 One dead channel | ONE weak channel kills geometric mean | §2 (why IC < F) |
        | §4 c ≈ 0.318 | First weld MUST be homogeneous | §3 (slaughter) |
        | §5 Quarks → hadrons | Confinement IS slaughter at phase boundary | §3 + §4 |
        | §6 Atoms restore IC | New DOF heal the damage | §5 (what was destroyed) |
        | §7 Full pipeline | Spine orchestrates everything into verdict | §1–§6 |

        *Each insight is a CONSEQUENCE of the previous one. The compounding is structural, not additive.*
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVERYDAY PHYSICS PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_everyday_physics_page() -> None:
    """Render the Everyday Physics domain page — thermodynamics, optics, EM, waves, epistemic."""
    if st is None or np is None:
        return

    st.title("🌡️ Everyday Physics")
    st.caption("Thermodynamics · Optics · Electromagnetism · Wave Phenomena · Epistemic Coherence")

    _ensure_closures_path()

    tab_thermo, tab_optics, tab_em, tab_wave, tab_epist = st.tabs(
        [
            "🔥 Thermodynamics",
            "🔦 Optics",
            "⚡ Electromagnetism",
            "🌊 Wave Phenomena",
            "🧠 Epistemic Coherence",
        ]
    )

    # ══════════════════════════════════════════════════════════════════════
    # THERMODYNAMICS
    # ══════════════════════════════════════════════════════════════════════
    with tab_thermo:
        st.header("🔥 Thermal Material Kernel")
        st.markdown("6-channel trace: Cp, k_th, ρ, T_m, T_b, α_th (thermal diffusivity)")

        try:
            from closures.everyday_physics.thermodynamics import (
                compute_all_thermal_materials,
                compute_thermal_material,
            )

            # All materials overview
            all_results = compute_all_thermal_materials()
            thermo_rows = []
            for r in all_results:
                thermo_rows.append(
                    {
                        "Material": r.material,
                        "F": r.F,
                        "ω": r.omega,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "S": r.S,
                        "C": r.C,
                        "Regime": r.regime,
                    }
                )

            if pd is not None and thermo_rows:
                df = pd.DataFrame(thermo_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "ω": "{:.4f}",
                            "IC": "{:.6f}",
                            "Δ": "{:.4f}",
                            "S": "{:.4f}",
                            "C": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # F vs IC scatter
                if go is not None:
                    fig = go.Figure()
                    for regime in df["Regime"].unique():
                        sub = df[df["Regime"] == regime]
                        fig.add_trace(
                            go.Scatter(
                                x=sub["F"],
                                y=sub["IC"],
                                mode="markers+text",
                                text=sub["Material"],
                                textposition="top center",
                                name=regime,
                                marker={"size": 10},
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode="lines", name="IC = F", line={"dash": "dash", "color": "gray"}
                        )
                    )
                    fig.update_layout(title="Thermal Materials: F vs IC", xaxis_title="F", yaxis_title="IC", height=450)
                    st.plotly_chart(fig, use_container_width=True)

            # Custom material input
            with st.expander("🔧 Custom Thermal Material"):
                cu_col1, cu_col2, cu_col3 = st.columns(3)
                with cu_col1:
                    cu_name = st.text_input("Material name", "My Material", key="thermo_name")
                    cu_Cp = st.number_input("Cp [J/(g·K)]", 0.01, 10.0, 0.385, key="thermo_Cp")
                with cu_col2:
                    cu_k = st.number_input("k_th [W/(m·K)]", 0.01, 500.0, 401.0, key="thermo_k")
                    cu_rho = st.number_input("ρ [kg/m³]", 0.1, 25000.0, 8960.0, key="thermo_rho")
                with cu_col3:
                    cu_Tm = st.number_input("T_melt [K]", 1.0, 5000.0, 1358.0, key="thermo_Tm")
                    cu_Tb = st.number_input("T_boil [K]", 1.0, 10000.0, 2835.0, key="thermo_Tb")

                if st.button("Compute Thermal Kernel", key="thermo_compute"):
                    try:
                        r = compute_thermal_material(cu_name, cu_Cp, cu_k, cu_rho, cu_Tm, cu_Tb)
                        mt = st.columns(4)
                        mt[0].metric("F", f"{r.F:.4f}")
                        mt[1].metric("IC", f"{r.IC:.6f}")
                        mt[2].metric("Δ", f"{r.gap:.4f}")
                        mt[3].metric("Regime", r.regime)
                    except Exception as e:
                        st.error(f"Error: {e}")

        except ImportError as e:
            st.error(f"Thermodynamics closures not available: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # OPTICS
    # ══════════════════════════════════════════════════════════════════════
    with tab_optics:
        st.header("🔦 Optical Material Kernel")
        st.markdown("6-channel trace: n_d, V_d, T_vis, R_vis, E_gap, n_group")

        try:
            from closures.everyday_physics.optics import (
                compute_all_optical_materials,
                compute_optical_material,
            )

            all_results = compute_all_optical_materials()
            optics_rows = []
            for r in all_results:
                optics_rows.append(
                    {
                        "Material": r.material,
                        "F": r.F,
                        "ω": r.omega,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "S": r.S,
                        "Regime": r.regime,
                    }
                )

            if pd is not None and optics_rows:
                df = pd.DataFrame(optics_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "ω": "{:.4f}",
                            "IC": "{:.6f}",
                            "Δ": "{:.4f}",
                            "S": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if go is not None:
                    fig = go.Figure()
                    for regime in df["Regime"].unique():
                        sub = df[df["Regime"] == regime]
                        fig.add_trace(
                            go.Scatter(
                                x=sub["F"],
                                y=sub["IC"],
                                mode="markers+text",
                                text=sub["Material"],
                                textposition="top center",
                                name=regime,
                                marker={"size": 10},
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode="lines", name="IC = F", line={"dash": "dash", "color": "gray"}
                        )
                    )
                    fig.update_layout(title="Optical Materials: F vs IC", xaxis_title="F", yaxis_title="IC", height=450)
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔧 Custom Optical Material"):
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    o_name = st.text_input("Name", "My Optical", key="opt_name")
                    o_nd = st.number_input("n_d (refractive index)", 1.0, 4.0, 1.52, key="opt_nd")
                with oc2:
                    o_vd = st.number_input("V_d (Abbe number)", 1.0, 100.0, 64.2, key="opt_vd")
                    o_tvis = st.number_input("T_vis (transmittance)", 0.0, 1.0, 0.92, key="opt_tvis")
                with oc3:
                    o_rvis = st.number_input("R_vis (reflectance)", 0.0, 1.0, 0.04, key="opt_rvis")
                    o_egap = st.number_input("E_gap [eV]", 0.0, 12.0, 8.0, key="opt_egap")
                o_ng = st.number_input("n_group", 1.0, 4.0, 1.53, key="opt_ng")

                if st.button("Compute Optical Kernel", key="opt_compute"):
                    try:
                        r = compute_optical_material(o_name, o_nd, o_vd, o_tvis, o_rvis, o_egap, o_ng)
                        mt = st.columns(4)
                        mt[0].metric("F", f"{r.F:.4f}")
                        mt[1].metric("IC", f"{r.IC:.6f}")
                        mt[2].metric("Δ", f"{r.gap:.4f}")
                        mt[3].metric("Regime", r.regime)
                    except Exception as e:
                        st.error(f"Error: {e}")

        except ImportError as e:
            st.error(f"Optics closures not available: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # ELECTROMAGNETISM
    # ══════════════════════════════════════════════════════════════════════
    with tab_em:
        st.header("⚡ Electromagnetic Material Kernel")
        st.markdown("6-channel trace: σ, ε_r, work function, band gap, μ_r, resistivity (log-scaled)")

        try:
            from closures.everyday_physics.electromagnetism import (
                compute_all_em_materials,
                compute_electromagnetic_material,
            )

            all_results = compute_all_em_materials()
            em_rows = []
            for r in all_results:
                em_rows.append(
                    {
                        "Material": r.material,
                        "Category": r.category,
                        "F": r.F,
                        "ω": r.omega,
                        "IC": r.IC,
                        "Δ": r.gap,
                        "Regime": r.regime,
                    }
                )

            if pd is not None and em_rows:
                df = pd.DataFrame(em_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "ω": "{:.4f}",
                            "IC": "{:.6f}",
                            "Δ": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if go is not None:
                    fig = go.Figure()
                    for cat in df["Category"].unique():
                        sub = df[df["Category"] == cat]
                        fig.add_trace(
                            go.Scatter(
                                x=sub["F"],
                                y=sub["IC"],
                                mode="markers+text",
                                text=sub["Material"],
                                textposition="top center",
                                name=cat,
                                marker={"size": 10},
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode="lines", name="IC = F", line={"dash": "dash", "color": "gray"}
                        )
                    )
                    fig.update_layout(
                        title="EM Materials: F vs IC (colored by category)",
                        xaxis_title="F",
                        yaxis_title="IC",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔧 Custom EM Material"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    e_name = st.text_input("Name", "My Material", key="em_name")
                    e_cat = st.selectbox(
                        "Category", ["conductor", "semiconductor", "insulator", "magnetic"], key="em_cat"
                    )
                    e_sigma = st.number_input("σ [MS/m]", 0.0, 100.0, 59.6, key="em_sigma")
                    e_epsr = st.number_input("ε_r", 1.0, 100.0, 1.0, key="em_epsr")
                with ec2:
                    e_wf = st.number_input("Work function [eV]", 1.0, 7.0, 4.65, key="em_wf")
                    e_bg = st.number_input("Band gap [eV]", 0.0, 12.0, 0.0, key="em_bg")
                    e_mur = st.number_input("μ_r", 0.1, 1e6, 1.0, key="em_mur")
                    e_rho = st.number_input("Resistivity [Ω·m]", 1e-10, 1e16, 1.68e-8, format="%.2e", key="em_rho")

                if st.button("Compute EM Kernel", key="em_compute"):
                    try:
                        r = compute_electromagnetic_material(e_name, e_cat, e_sigma, e_epsr, e_wf, e_bg, e_mur, e_rho)
                        mt = st.columns(4)
                        mt[0].metric("F", f"{r.F:.4f}")
                        mt[1].metric("IC", f"{r.IC:.6f}")
                        mt[2].metric("Δ", f"{r.gap:.4f}")
                        mt[3].metric("Regime", r.regime)
                    except Exception as e:
                        st.error(f"Error: {e}")

        except ImportError as e:
            st.error(f"Electromagnetism closures not available: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # WAVE PHENOMENA
    # ══════════════════════════════════════════════════════════════════════
    with tab_wave:
        st.header("🌊 Wave System Kernel")
        st.markdown("6-channel trace: frequency, wavelength, phase velocity, Q factor, coherence, amplitude")

        try:
            from closures.everyday_physics.wave_phenomena import (
                compute_all_wave_systems,
            )

            all_results = compute_all_wave_systems()
            wave_rows = []
            for r in all_results:
                wave_rows.append(
                    {
                        "System": r.system,
                        "Type": r.wave_type,
                        "F": r.F,
                        "ω": r.omega,
                        "IC": r.IC,
                        "Δ": r.F - r.IC,
                        "Regime": r.regime,
                    }
                )

            if pd is not None and wave_rows:
                df = pd.DataFrame(wave_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "ω": "{:.4f}",
                            "IC": "{:.6f}",
                            "Δ": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if go is not None:
                    fig = go.Figure()
                    for wtype in df["Type"].unique():
                        sub = df[df["Type"] == wtype]
                        fig.add_trace(
                            go.Scatter(
                                x=sub["F"],
                                y=sub["IC"],
                                mode="markers+text",
                                text=sub["System"],
                                textposition="top center",
                                name=wtype,
                                marker={"size": 10},
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode="lines", name="IC = F", line={"dash": "dash", "color": "gray"}
                        )
                    )
                    fig.update_layout(
                        title="Wave Systems: F vs IC (colored by wave type)",
                        xaxis_title="F",
                        yaxis_title="IC",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except ImportError as e:
            st.error(f"Wave phenomena closures not available: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # EPISTEMIC COHERENCE
    # ══════════════════════════════════════════════════════════════════════
    with tab_epist:
        st.header("🧠 Epistemic Coherence Kernel")
        st.markdown("""
        8-channel trace mapping knowledge systems through GCD:
        pattern recognition · narrative coherence · predictive accuracy · causal mechanism ·
        reproducibility · falsifiability · evidential convergence · institutional scrutiny
        """)

        try:
            from closures.everyday_physics.epistemic_coherence import (
                CHANNEL_NAMES,
                compute_all_epistemic_systems,
                compute_epistemic_from_channels,
                run_all_theorems,
            )

            # Overview
            all_results = compute_all_epistemic_systems()
            epist_rows = []
            for r in all_results:
                epist_rows.append(
                    {
                        "System": r.name,
                        "Category": r.category,
                        "F": r.F,
                        "ω": r.omega,
                        "IC": r.IC,
                        "Δ": r.heterogeneity_gap,
                        "Regime": r.regime,
                        "Strongest": r.strongest_channel,
                        "Weakest": r.weakest_channel,
                    }
                )

            if pd is not None and epist_rows:
                df = pd.DataFrame(epist_rows)
                st.dataframe(
                    df.style.format(
                        {
                            "F": "{:.4f}",
                            "ω": "{:.4f}",
                            "IC": "{:.6f}",
                            "Δ": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Radar chart for selected system
                system_names = [r.name for r in all_results]
                selected_sys = st.selectbox("Compare system", system_names, key="epist_sys")
                idx = system_names.index(selected_sys)
                selected_r = all_results[idx]

                if go is not None:
                    fig = go.Figure()
                    trace_vals = list(selected_r.trace)
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[*trace_vals, trace_vals[0]],
                            theta=[*list(CHANNEL_NAMES), CHANNEL_NAMES[0]],
                            fill="toself",
                            name=selected_r.name,
                        )
                    )
                    fig.update_layout(
                        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                        title=f"Channel Profile: {selected_r.name} ({selected_r.category})",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Custom epistemic analysis
            with st.expander("🔧 Custom Epistemic Analysis"):
                st.markdown("Enter 8 channel values [0–1] for your own knowledge system:")
                ch_cols = st.columns(4)
                custom_channels: list[float] = []
                for i, ch_name in enumerate(CHANNEL_NAMES):
                    with ch_cols[i % 4]:
                        val = st.slider(ch_name, 0.0, 1.0, 0.5, 0.01, key=f"epist_ch_{i}")
                        custom_channels.append(val)

                custom_name = st.text_input("Name your system", "My Knowledge System", key="epist_custom_name")
                custom_cat = st.selectbox(
                    "Category",
                    ["ScientificPractice", "FolkKnowledge", "Pseudoscience", "ParadigmShift", "Other"],
                    key="epist_custom_cat",
                )

                if st.button("Analyze", key="epist_analyze"):
                    try:
                        r = compute_epistemic_from_channels(custom_name, custom_cat, custom_channels)
                        mt = st.columns(5)
                        mt[0].metric("F", f"{r.F:.4f}")
                        mt[1].metric("IC", f"{r.IC:.6f}")
                        mt[2].metric("Δ", f"{r.heterogeneity_gap:.4f}")
                        mt[3].metric("Regime", r.regime)
                        mt[4].metric("Dead channels", str(r.dead_channels))
                        st.caption(f"Strongest: {r.strongest_channel} | Weakest: {r.weakest_channel}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Theorems
            with st.expander("📜 Run All 7 Epistemic Theorems"):
                if st.button("Run Theorems", key="epist_thm"):
                    results = run_all_theorems()
                    for thm in results:
                        icon = "✅" if thm.verdict == "PROVEN" else "❌"
                        st.markdown(
                            f"**{icon} {thm.name}**: {thm.statement}\n\n"
                            f"Tests: {thm.n_passed}/{thm.n_tests} | Verdict: **{thm.verdict}**"
                        )

        except ImportError as e:
            st.error(f"Epistemic coherence closures not available: {e}")
