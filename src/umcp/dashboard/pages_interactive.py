"""
Interactive dashboard pages: Test Templates, Batch Validation, Live Runner.
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
from datetime import datetime

from umcp.dashboard._deps import np, pd, st
from umcp.dashboard._utils import (
    STATUS_COLORS,
    classify_regime,
    get_regime_color,
    get_repo_root,
    load_casepacks,
)
from umcp.dashboard.pages_physics import render_gcd_panel, translate_to_gcd

# Import optimized kernel computer for real computation
try:
    from umcp.kernel_optimized import OptimizedKernelComputer

    _HAS_KERNEL = True
except ImportError:  # pragma: no cover
    _HAS_KERNEL = False


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
            # Use OptimizedKernelComputer for auditable, lemma-verified computation
            if _HAS_KERNEL:
                computer = OptimizedKernelComputer(epsilon=epsilon)
                outputs = computer.compute(c_clipped, w_raw)
                fidelity = outputs.F
                drift = outputs.omega
                log_ic = outputs.kappa
                integrity_composite = outputs.IC
                entropy = outputs.S
                curvature = outputs.C
                heterogeneity_gap = outputs.heterogeneity_gap
                is_homogeneous = outputs.is_homogeneous
                het_regime = outputs.regime
                computation_mode = outputs.computation_mode

                # Lipschitz error bounds (OPT-12)
                error_bounds = computer.propagate_coordinate_error(epsilon)
            else:
                # Fallback: inline computation
                fidelity = float(np.sum(w_raw * c_clipped))
                drift = 1 - fidelity
                log_ic = float(np.sum(w_raw * np.log(c_clipped)))
                integrity_composite = float(np.exp(log_ic))
                entropy = 0.0
                for ci, wi in zip(c_clipped, w_raw, strict=False):
                    if wi > 0 and 0 < ci < 1:
                        entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))
                entropy = float(entropy)
                curvature = float(np.std(c_clipped, ddof=0) / 0.5)
                heterogeneity_gap = fidelity - integrity_composite
                is_homogeneous = np.allclose(c_clipped, c_clipped[0], atol=1e-15)
                if heterogeneity_gap < 1e-6:
                    het_regime = "homogeneous"
                elif heterogeneity_gap < 0.01:
                    het_regime = "coherent"
                elif heterogeneity_gap < 0.05:
                    het_regime = "heterogeneous"
                else:
                    het_regime = "fragmented"
                computation_mode = "inline_fallback"
                error_bounds = None

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
                "heterogeneity_gap": heterogeneity_gap,
                "is_homogeneous": bool(is_homogeneous),
                "heterogeneity_regime": het_regime,
                "computation_mode": computation_mode,
                "bounds_valid": bounds_valid,
                "identity_F_omega": abs(fidelity + drift - 1.0) < 1e-9,
                "identity_IC_kappa": abs(integrity_composite - np.exp(log_ic)) < 1e-9,
                "error_bounds": {
                    "F": error_bounds.F,
                    "omega": error_bounds.omega,
                    "kappa": error_bounds.kappa,
                    "S": error_bounds.S,
                }
                if error_bounds is not None
                else None,
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
            if heterogeneity_gap > 0.1:
                recommendations.append("Large heterogeneity gap - significant heterogeneity")
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

                st.markdown(f"**Heterogeneity Gap:** {t1['heterogeneity_gap']:.4f}")
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

                st.dataframe(results_df.style.map(color_status, subset=["Status"]), width="stretch", hide_index=True)

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
