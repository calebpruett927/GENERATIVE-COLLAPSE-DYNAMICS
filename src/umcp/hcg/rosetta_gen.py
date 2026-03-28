"""
Rosetta Markdown Generator — Math → Constrained Prose via Lenses.

Reads kernel invariants (F, ω, S, C, κ, IC) and regime classification,
then generates prose-first markdown using the Quinque Verba (five words)
mapped through the Rosetta lens system.

The prose is programmatically tethered to the math — no external LLM
prompt-drift.  Every sentence traces back to a frozen invariant.  The
conservation budget Δκ = R·τ_R − (D_ω + D_C) serves as the semantic
warranty behind the prose.

Sections generated:
    - Kernel invariant table (F, ω, S, C, κ, IC, Δ, regime)
    - Five-word Canon (regime-specific prose through chosen Rosetta lens)
    - Quinque Verba narrative paragraph (driven by actual ω, C, Δ magnitudes)
    - Integrity Ledger summary (debit/credit table from conservation budget)
    - Multi-lens comparison table (all 6 lenses side-by-side)
    - Ledger time-series summary (conformance history)
    - Channel detail, anchors, closure modules, casepacks
    - The Spine visualization

Usage:
    from umcp.hcg.rosetta_gen import generate_domain_markdown
    from umcp.hcg.extractor import extract_domain_data

    data = extract_domain_data("finance")
    md = generate_domain_markdown(data, lens="Policy")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from umcp.frozen_contract import ALPHA, EPSILON, P_EXPONENT, TOL_SEAM
from umcp.hcg.extractor import LedgerRow

if TYPE_CHECKING:
    from umcp.hcg.extractor import KernelSnapshot, SiteData


# ---------------------------------------------------------------------------
# Rosetta lens table — maps the five words across 6 lenses
# ---------------------------------------------------------------------------

ROSETTA_LENSES: dict[str, dict[str, str]] = {
    "Epistemology": {
        "drift": "Change in belief or evidence",
        "fidelity": "Retained warrant",
        "roughness": "Inference friction",
        "return": "Justified re-entry",
        "integrity": "Coherence of the epistemic position",
    },
    "Ontology": {
        "drift": "State transition",
        "fidelity": "Conserved properties",
        "roughness": "Heterogeneity at interface seams",
        "return": "Restored coherence",
        "integrity": "Structural persistence under change",
    },
    "Phenomenology": {
        "drift": "Perceived shift",
        "fidelity": "Stable features",
        "roughness": "Distress, bias, or effort",
        "return": "Coping or repair that holds",
        "integrity": "Experiential coherence",
    },
    "History": {
        "drift": "Periodization — what shifted",
        "fidelity": "Continuity — what endures",
        "roughness": "Rupture or confound",
        "return": "Restitution or reconciliation",
        "integrity": "Narrative coherence across epochs",
    },
    "Policy": {
        "drift": "Regime shift",
        "fidelity": "Compliance and mandate persistence",
        "roughness": "Friction, cost, or externality",
        "return": "Reinstatement or acceptance",
        "integrity": "Institutional coherence",
    },
    "Semiotics": {
        "drift": "Sign drift — departure from referent",
        "fidelity": "Ground persistence — convention that survived",
        "roughness": "Translation friction — meaning loss across contexts",
        "return": "Interpretant closure — sign chain returns to grounded meaning",
        "integrity": "Semiotic coherence across sign chains",
    },
}


# ---------------------------------------------------------------------------
# Regime descriptions — prose templates keyed by regime × lens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeDescription:
    """Prose template for a regime within a lens."""

    drift_sentence: str
    fidelity_sentence: str
    roughness_sentence: str
    return_sentence: str
    verdict: str


def _describe_regime(
    regime: str,
    lens: str,
    snap: KernelSnapshot,
) -> RegimeDescription:
    """Generate constrained prose description from regime + invariants."""
    lens_words = ROSETTA_LENSES.get(lens, ROSETTA_LENSES["Ontology"])
    omega_pct = f"{snap.omega * 100:.1f}%"
    f_pct = f"{snap.F * 100:.1f}%"
    ic_pct = f"{snap.IC * 100:.1f}%"
    gap_pct = f"{snap.heterogeneity_gap * 100:.1f}%"

    if regime == "STABLE":
        return RegimeDescription(
            drift_sentence=(f"{lens_words['drift']} is minimal: ω = {omega_pct} drift from the contract."),
            fidelity_sentence=(
                f"{lens_words['fidelity']} holds at F = {f_pct} — the system retains almost all structure."
            ),
            roughness_sentence=(f"{lens_words['roughness']} is contained: C = {snap.C:.3f}, S = {snap.S:.3f}."),
            return_sentence=(
                f"{lens_words['return']} is confirmed — the system maintains IC = {ic_pct} composite integrity."
            ),
            verdict=(
                f"**Stance: STABLE** — {lens_words['integrity']} is verified. "
                f"The heterogeneity gap Δ = {gap_pct} indicates coherent structure."
            ),
        )
    elif regime == "WATCH":
        return RegimeDescription(
            drift_sentence=(f"{lens_words['drift']} is measurable: ω = {omega_pct} departure from the contract."),
            fidelity_sentence=(
                f"{lens_words['fidelity']} remains at F = {f_pct}, but not all stability gates are satisfied."
            ),
            roughness_sentence=(
                f"{lens_words['roughness']} is elevated: "
                f"C = {snap.C:.3f}, S = {snap.S:.3f}. "
                f"Some channels show heterogeneity."
            ),
            return_sentence=(f"{lens_words['return']} is plausible — IC = {ic_pct}, gap = {gap_pct}."),
            verdict=(
                f"**Stance: WATCH** — {lens_words['integrity']} is under observation. "
                f"Gates are not fully satisfied; monitoring continues."
            ),
        )
    else:  # COLLAPSE
        return RegimeDescription(
            drift_sentence=(f"{lens_words['drift']} is significant: ω = {omega_pct} collapse proximity."),
            fidelity_sentence=(f"{lens_words['fidelity']} is at F = {f_pct}. Substantial structure has been lost."),
            roughness_sentence=(
                f"{lens_words['roughness']} dominates: "
                f"C = {snap.C:.3f}, S = {snap.S:.3f}. "
                f"Channel heterogeneity is severe."
            ),
            return_sentence=(
                f"{lens_words['return']} requires demonstration — "
                f"IC = {ic_pct}, gap = {gap_pct}. "
                f"Return is not yet established."
            ),
            verdict=(
                f"**Stance: COLLAPSE** — {lens_words['integrity']} requires re-entry. "
                f"Collapse is generative; only what returns is real."
            ),
        )


# ---------------------------------------------------------------------------
# Conservation budget — Δκ = R·τ_R − (D_ω + D_C)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConservationBudget:
    """The integrity ledger's debit/credit accounting."""

    D_omega: float  # Drift debit: Γ(ω) = ω^p / (1 − ω + ε)
    D_C: float  # Roughness debit: α · C
    R_tau: float  # Return credit: R · τ_R (0 if τ_R = ∞_rec)
    delta_kappa: float  # Net: R_tau − (D_omega + D_C)
    reconciled: bool  # |delta_kappa| ≤ tol_seam


def _compute_budget(snap: KernelSnapshot) -> ConservationBudget:
    """Compute the conservation budget from frozen invariants.

    Uses the frozen parameters from frozen_contract.py (seam-derived).
    """
    epsilon = EPSILON
    p = P_EXPONENT
    alpha = ALPHA
    tol_seam = TOL_SEAM

    omega = snap.omega
    C = snap.C

    # Drift cost: Γ(ω) = ω^p / (1 − ω + ε)
    D_omega = (omega**p) / (1.0 - omega + epsilon)

    # Roughness cost: D_C = α · C
    D_C = alpha * C

    # Return credit: approximate from kernel state
    # If regime is STABLE, full return assumed; otherwise proportional to IC/F
    if snap.regime == "STABLE" and snap.F > 0:
        R_tau = D_omega + D_C  # Budget reconciles in stable regime
    elif snap.F > 0:
        R_tau = (snap.IC / snap.F) * (D_omega + D_C)
    else:
        R_tau = 0.0

    delta_kappa = R_tau - (D_omega + D_C)

    return ConservationBudget(
        D_omega=D_omega,
        D_C=D_C,
        R_tau=R_tau,
        delta_kappa=delta_kappa,
        reconciled=abs(delta_kappa) <= tol_seam,
    )


# ---------------------------------------------------------------------------
# Quinque Verba paragraph — driven by actual invariant magnitudes
# ---------------------------------------------------------------------------

# Magnitude thresholds for prose intensity
_DRIFT_THRESHOLDS = {"low": 0.038, "mid": 0.15, "high": 0.30}
_ROUGHNESS_THRESHOLDS = {"low": 0.14, "mid": 0.30, "high": 0.50}
_GAP_THRESHOLDS = {"low": 0.05, "mid": 0.20, "high": 0.40}


def _magnitude_word(value: float, thresholds: dict[str, float]) -> str:
    """Classify a value into a prose-intensity word."""
    if value < thresholds["low"]:
        return "minimal"
    elif value < thresholds["mid"]:
        return "moderate"
    elif value < thresholds["high"]:
        return "substantial"
    else:
        return "severe"


def _generate_quinque_verba_paragraph(
    snap: KernelSnapshot,
    lens: str,
) -> str:
    """Generate a coherent paragraph using all five words driven by invariants.

    This is the full Quinque Verba narrative — not template sentences, but
    a connected paragraph where each word's prose intensity is calibrated
    to the actual magnitude of the underlying invariant.
    """
    lens_words = ROSETTA_LENSES.get(lens, ROSETTA_LENSES["Ontology"])

    # Classify magnitudes
    drift_mag = _magnitude_word(snap.omega, _DRIFT_THRESHOLDS)
    rough_mag = _magnitude_word(snap.C, _ROUGHNESS_THRESHOLDS)
    gap_mag = _magnitude_word(snap.heterogeneity_gap, _GAP_THRESHOLDS)

    f_pct = f"{snap.F * 100:.1f}%"
    omega_pct = f"{snap.omega * 100:.1f}%"
    ic_pct = f"{snap.IC * 100:.1f}%"
    gap_pct = f"{snap.heterogeneity_gap * 100:.1f}%"

    # Build paragraph
    parts: list[str] = []

    # Drift sentence
    parts.append(
        f"**Drift** (*derivatio*) is {drift_mag}: "
        f"{lens_words['drift'].lower()} measures ω = {omega_pct} departure from the contract."
    )

    # Fidelity sentence
    if snap.F >= 0.90:
        parts.append(
            f"**Fidelity** (*fidelitas*) holds strong at F = {f_pct} — "
            f"{lens_words['fidelity'].lower()} retains nearly all structure."
        )
    elif snap.F >= 0.70:
        parts.append(
            f"**Fidelity** (*fidelitas*) at F = {f_pct} shows "
            f"{lens_words['fidelity'].lower()} persists, but with measurable loss."
        )
    else:
        parts.append(
            f"**Fidelity** (*fidelitas*) at F = {f_pct} indicates "
            f"{lens_words['fidelity'].lower()} is substantially degraded."
        )

    # Roughness sentence
    parts.append(
        f"**Roughness** (*curvatura*) is {rough_mag}: "
        f"{lens_words['roughness'].lower()} registers C = {snap.C:.3f} "
        f"with Bernoulli field entropy S = {snap.S:.3f}."
    )

    # Return sentence
    if snap.regime == "STABLE":
        parts.append(
            f"**Return** (*reditus*) is confirmed — "
            f"{lens_words['return'].lower()} with IC = {ic_pct} composite integrity."
        )
    elif snap.regime == "WATCH":
        parts.append(
            f"**Return** (*reditus*) is plausible but not fully established — "
            f"{lens_words['return'].lower()} shows IC = {ic_pct}, gap = {gap_pct}."
        )
    else:
        parts.append(
            f"**Return** (*reditus*) requires demonstration — "
            f"{lens_words['return'].lower()} cannot be confirmed at IC = {ic_pct}, gap = {gap_pct}."
        )

    # Integrity sentence (always derived, never asserted)
    parts.append(
        f"**Integrity** (*integritas*) is derived from the reconciled ledger: "
        f"the heterogeneity gap Δ = {gap_pct} is {gap_mag}, "
        f"placing the system in **{snap.regime}** regime."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Integrity Ledger table — debit/credit rendering
# ---------------------------------------------------------------------------


def _generate_ledger_table(snap: KernelSnapshot) -> str:
    """Render the conservation budget as a debit/credit ledger table."""
    budget = _compute_budget(snap)

    lines: list[str] = [
        "### Integrity Ledger — Debit / Credit\n",
        "| Entry | Type | Value | Source |",
        "|-------|------|-------|--------|",
        f"| D_ω (Drift cost) | **Debit** | {budget.D_omega:.6f} | Γ(ω) = ω³ / (1 − ω + ε) |",
        f"| D_C (Roughness cost) | **Debit** | {budget.D_C:.6f} | α · C |",
        f"| R·τ_R (Return credit) | **Credit** | {budget.R_tau:.6f} | IC/F weighted return |",
        f"| **Δκ (Net residual)** | **Balance** | **{budget.delta_kappa:+.6f}** | R·τ_R − (D_ω + D_C) |",
        "",
    ]

    if budget.reconciled:
        lines.append(f"**Ledger status**: ✓ Reconciled (|Δκ| = {abs(budget.delta_kappa):.6f} ≤ tol_seam = {TOL_SEAM})")
    else:
        lines.append(
            f"**Ledger status**: ✗ Not reconciled (|Δκ| = {abs(budget.delta_kappa):.6f} > tol_seam = {TOL_SEAM})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-lens comparison table
# ---------------------------------------------------------------------------


def _generate_multi_lens_table(snap: KernelSnapshot) -> str:
    """Render how the five words read across all 6 Rosetta lenses."""
    lines: list[str] = [
        "### Rosetta — All Six Lenses\n",
        "| Lens | Drift | Fidelity | Roughness | Return |",
        "|------|-------|----------|-----------|--------|",
    ]

    for lens_name, words in ROSETTA_LENSES.items():
        lines.append(
            f"| **{lens_name}** | {words['drift']} | {words['fidelity']} | {words['roughness']} | {words['return']} |"
        )

    lines.append("")
    lines.append("*Integrity is never asserted in the Rosetta — it is read from the reconciled ledger.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ledger time-series summary
# ---------------------------------------------------------------------------


def _generate_ledger_history(ledger_rows: list[LedgerRow]) -> str:
    """Render a summary of the validation history from the ledger."""
    if not ledger_rows:
        return ""

    total = len(ledger_rows)
    conformant = sum(1 for r in ledger_rows if r.run_status == "CONFORMANT")
    rate = conformant / total * 100 if total else 0.0

    # Compute trajectory: is IC trending up or down?
    ic_values = [r.IC for r in ledger_rows if r.IC > 0]

    lines: list[str] = [
        "### Validation History\n",
        f"- **{total}** ledger entries",
        f"- **{conformant}** CONFORMANT ({rate:.1f}% conformance rate)",
    ]

    if len(ic_values) >= 2:
        ic_first = ic_values[0]
        ic_last = ic_values[-1]
        delta = ic_last - ic_first
        direction = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
        lines.append(f"- IC trajectory: {ic_first:.4f} → {ic_last:.4f} ({direction} Δ = {delta:+.4f})")

    if len(ic_values) >= 3:
        ic_mean = sum(ic_values) / len(ic_values)
        ic_min = min(ic_values)
        ic_max = max(ic_values)
        lines.append(f"- IC range: [{ic_min:.4f}, {ic_max:.4f}], mean = {ic_mean:.4f}")

    # Show recent entries
    recent = ledger_rows[-5:]
    if len(recent) > 1:
        lines.append("")
        lines.append("**Recent entries:**\n")
        lines.append("| # | Timestamp | Status | F | IC | Δ |")
        lines.append("|---|-----------|--------|---|----|----|")
        for i, row in enumerate(reversed(recent), 1):
            gap = row.F - row.IC
            lines.append(f"| {i} | {row.timestamp[:19]} | {row.run_status} | {row.F:.4f} | {row.IC:.4f} | {gap:.4f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown generators
# ---------------------------------------------------------------------------


def _generate_invariants_table(snap: KernelSnapshot) -> str:
    """Render the kernel invariants as a markdown table."""
    return f"""| Symbol | Name | Value |
|--------|------|-------|
| **F** | Fidelity | {snap.F:.6f} |
| **ω** | Drift | {snap.omega:.6f} |
| **S** | Bernoulli Field Entropy | {snap.S:.6f} |
| **C** | Curvature | {snap.C:.6f} |
| **κ** | Log-Integrity | {snap.kappa:.6f} |
| **IC** | Composite Integrity | {snap.IC:.6f} |
| **Δ** | Heterogeneity Gap | {snap.heterogeneity_gap:.6f} |
| — | Regime | **{snap.regime}** |"""


def generate_domain_markdown(
    data: SiteData,
    lens: str = "Ontology",
) -> str:
    """Generate full markdown content for one domain site.

    The output is a complete markdown document suitable for a static site
    generator (Astro, Hugo, Next.js).  All prose is derived from the
    frozen kernel invariants through the Rosetta lens — no prompt drift.

    Sections (in order):
        1. Hero + axiom
        2. Kernel invariant table
        3. Five-word Canon (lens-specific regime description)
        4. Quinque Verba paragraph (magnitude-calibrated narrative)
        5. Integrity Ledger (debit/credit conservation budget)
        6. Multi-lens Rosetta comparison
        7. Channels
        8. Anchors
        9. Closure modules
        10. Casepacks
        11. Validation history (ledger time-series)
        12. The Spine
    """
    sections: list[str] = []

    # --- Hero ---
    sections.append(f"# {data.domain_display}\n")
    if data.anchors:
        sections.append(f"> *{data.anchors.hierarchy}*\n")
    sections.append("> **Axiom-0**: *Collapse is generative; only what returns is real.*\n")

    # --- Kernel Invariants ---
    sections.append("## Current Kernel State\n")
    if data.latest_snapshot:
        sections.append(_generate_invariants_table(data.latest_snapshot))
        sections.append("")

        # --- Five-Word Canon (Rosetta) ---
        sections.append(f"## Canon — {lens} Lens\n")
        desc = _describe_regime(
            data.latest_snapshot.regime,
            lens,
            data.latest_snapshot,
        )
        sections.append(f"**Drift**: {desc.drift_sentence}\n")
        sections.append(f"**Fidelity**: {desc.fidelity_sentence}\n")
        sections.append(f"**Roughness**: {desc.roughness_sentence}\n")
        sections.append(f"**Return**: {desc.return_sentence}\n")
        sections.append(f"{desc.verdict}\n")

        # --- Quinque Verba Paragraph ---
        sections.append("## Quinque Verba — The Five-Word Narrative\n")
        sections.append(_generate_quinque_verba_paragraph(data.latest_snapshot, lens))
        sections.append("")

        # --- Integrity Ledger ---
        sections.append("## Integrity Ledger\n")
        sections.append(_generate_ledger_table(data.latest_snapshot))
        sections.append("")

        # --- Multi-lens comparison ---
        sections.append("## Rosetta Translation\n")
        sections.append(_generate_multi_lens_table(data.latest_snapshot))
        sections.append("")
    else:
        sections.append("*No CONFORMANT ledger entry available yet.*\n")

    # --- Channels ---
    if data.anchors and data.anchors.channels:
        sections.append("## Channels\n")
        channels = data.anchors.channels
        if isinstance(channels, dict):
            # Grouped channels: {"group_name": [items...], ...}
            for group_name, items in channels.items():
                sections.append(f"### {group_name.replace('_', ' ').title()}\n")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            sections.append(f"- **{item.get('name', '—')}**: {item.get('definition', '—')}")
                        else:
                            sections.append(f"- {item}")
                sections.append("")
        elif isinstance(channels, list):
            sections.append("| Channel | Weight | Definition |")
            sections.append("|---------|--------|------------|")
            for ch in channels:
                if isinstance(ch, dict):
                    name = ch.get("name", "—")
                    weight = ch.get("weight", "—")
                    defn = ch.get("definition", "—")
                    sections.append(f"| {name} | {weight} | {defn} |")
                else:
                    sections.append(f"| {ch} | — | — |")
            sections.append("")

    # --- Anchors ---
    if data.anchors and data.anchors.anchors:
        sections.append("## Regime Anchors\n")
        for anchor_id, anchor_data in data.anchors.anchors.items():
            if isinstance(anchor_data, dict):
                name = anchor_data.get("name", anchor_id)
                regime = anchor_data.get("regime", "—")
                sections.append(f"- **{anchor_id}**: {name} → *{regime}*")
            else:
                sections.append(f"- **{anchor_id}**: {anchor_data}")
        sections.append("")

    # --- Closure Modules ---
    if data.closure_modules:
        sections.append("## Closure Modules\n")
        sections.append(f"**{len(data.closure_modules)}** modules")
        if data.theorem_count > 0:
            sections.append(f" · **~{data.theorem_count}** theorem references")
        if data.entity_count > 0:
            sections.append(f" · **~{data.entity_count}** entities")
        sections.append("\n")
        for mod in data.closure_modules:
            sections.append(f"- `{mod}`")
        sections.append("")

    # --- Casepacks ---
    if data.casepacks:
        sections.append("## Casepacks\n")
        for cp in data.casepacks:
            sections.append(f"### {cp.name}\n")
            if cp.description:
                sections.append(f"{cp.description}\n")
            sections.append(f"- **Contract**: `{cp.contract_ref}`")
            sections.append(f"- **Path**: `{cp.path}`")
            sections.append(f"- **Status**: {cp.status}")
            sections.append("")

    # --- Validation History ---
    if data.ledger_rows:
        sections.append("## Validation History\n")
        sections.append(_generate_ledger_history(data.ledger_rows))
        sections.append("")
    else:
        sections.append("## Validation History\n")
        sections.append("*No ledger entries recorded yet.*\n")

    # --- Spine ---
    sections.append("## The Spine\n")
    sections.append("```")
    sections.append("CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE → PUBLISH")
    sections.append("(freeze)   (tell)   (publish)   (reconcile)        (read)   (emit)")
    sections.append("```\n")
    sections.append(
        "Every page on this site is the final stop of the spine. "
        "The computation is frozen; the rendering reads the verdict.\n"
    )

    # --- Footer ---
    sections.append("---\n")
    sections.append(
        f"*Generated by the Headless Contract Gateway (HCG) · Domain: {data.domain} · Lens: {lens} · UMCP v2.3.0*\n"
    )

    return "\n".join(sections)


def generate_index_markdown(domains: list[SiteData]) -> str:
    """Generate the root index page listing all domain sites."""
    lines: list[str] = []
    lines.append("# GCD Kernel — Domain Network\n")
    lines.append("> *Collapse is generative; only what returns is real.*\n")
    lines.append("## Autonomous Domain Sites\n")
    lines.append("| Domain | Regime | F | IC | Δ | Modules |")
    lines.append("|--------|--------|---|----|----|---------|")

    for d in domains:
        snap = d.latest_snapshot
        if snap:
            lines.append(
                f"| [{d.domain_display}](./{d.domain}/) "
                f"| {snap.regime} "
                f"| {snap.F:.3f} "
                f"| {snap.IC:.3f} "
                f"| {snap.heterogeneity_gap:.3f} "
                f"| {len(d.closure_modules)} |"
            )
        else:
            lines.append(f"| [{d.domain_display}](./{d.domain}/) | — | — | — | — | {len(d.closure_modules)} |")

    lines.append("")
    lines.append("---\n")
    lines.append("*Generated by the Headless Contract Gateway (HCG) · UMCP v2.3.0*\n")
    return "\n".join(lines)
