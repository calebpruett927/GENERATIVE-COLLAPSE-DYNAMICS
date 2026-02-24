#!/usr/bin/env python3
"""All-out experimental physics probes — 8 investigations with visualizations.

Each probe pushes the GCD kernel into uncharted territory, testing predictions
from the 10 Standard Model theorems and cross-scale analysis.

Probe 1: Mass-Channel Removal — Is generation monotonicity intrinsic?
Probe 2: SUSY Catalog — Does the fermion/boson gap close with sparticles?
Probe 3: Quark IC at Variable Q — Does the confinement cliff soften at high energy?
Probe 4: Dark Matter Kernel — What does a WIMP / axion / sterile ν look like?
Probe 5: Channel-Ablation Periodic Kernel — Which atomic channel controls d-block dominance?
Probe 6: GUT-Scale Coupling Extrapolation — Do g₁, g₂, g₃ unify in the SM?
Probe 7: Molecular-Scale Kernel — Does IC recover above the hadron confinement cliff?
Probe 8: Wave Q-Factor vs IC — Is energy retention the driver of wave coherence?

Usage:
    python scripts/run_all_probes.py           # Run all probes
    python scripts/run_all_probes.py --probe 3 # Run single probe
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Ensure src/ on path ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from closures.atomic_physics.periodic_kernel import (
    _normalize_element,
    batch_compute_all,
)
from closures.everyday_physics.wave_phenomena import (
    WAVE_SYSTEMS,
    compute_all_wave_systems,
    compute_wave_system,
)
from closures.materials_science.element_database import (
    ELEMENTS,
    get_element,
)
from closures.standard_model.coupling_constants import (
    ALPHA_S_MZ,
    M_Z_GEV,
    compute_running_coupling,
)
from closures.standard_model.subatomic_kernel import (
    EPSILON,
    FUNDAMENTAL_PARTICLES,
    FundamentalParticle,
    compute_all_composite,
    compute_all_fundamental,
    normalize_fundamental,
)
from umcp.kernel_optimized import compute_kernel_outputs

# ── Global style ─────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT_CYAN = "#58a6ff"
ACCENT_ORANGE = "#f0883e"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_YELLOW = "#d29922"
ACCENT_PINK = "#f778ba"
ACCENT_TEAL = "#39d353"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"

OUT_DIR = os.path.join(_ROOT, "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _dark_fig(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """Create a dark-themed figure with subplots."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            _style_ax(ax)
    else:
        _style_ax(axes)
        axes = np.array([axes])
    return fig, axes


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Saved {name}")
    return path


def _kernel_from_trace(c: np.ndarray, eps: float = EPSILON) -> dict:
    """Compute kernel outputs from a raw trace vector."""
    c_clamped = np.clip(c, eps, 1 - eps)
    w = np.ones(len(c_clamped)) / len(c_clamped)
    return compute_kernel_outputs(c_clamped, w, eps)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 1: Mass-Channel Removal                                             ║
# ║  Question: Is generation monotonicity (T2) intrinsic or mass-driven?        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_1_mass_channel_removal() -> None:
    """Remove mass channel from all particles and check if T2 still holds."""
    print("\n═══ PROBE 1: Mass-Channel Removal ═══")

    fund_results = compute_all_fundamental()

    # Compute with and without mass for each particle
    data_full: dict[str, list] = {"name": [], "F": [], "IC": [], "gap": [], "cat": [], "gen": []}
    data_no_mass: dict[str, list] = {"name": [], "F": [], "IC": [], "gap": [], "cat": [], "gen": []}

    for r in fund_results:
        # Full kernel (standard)
        data_full["name"].append(r.symbol)
        data_full["F"].append(r.F)
        data_full["IC"].append(r.IC)
        data_full["gap"].append(r.heterogeneity_gap)
        data_full["cat"].append(r.category)
        data_full["gen"].append(next((p.generation for p in FUNDAMENTAL_PARTICLES if p.symbol == r.symbol), 0))

        # Find original particle for re-normalization without mass
        p = next((fp for fp in FUNDAMENTAL_PARTICLES if fp.symbol == r.symbol), None)
        if p is None:
            data_no_mass["name"].append(r.symbol)
            data_no_mass["F"].append(r.F)
            data_no_mass["IC"].append(r.IC)
            data_no_mass["gap"].append(r.heterogeneity_gap)
            data_no_mass["cat"].append(r.category)
            data_no_mass["gen"].append(0)
            continue

        c_full, _w, _labels = normalize_fundamental(p)
        # Remove channel 0 (mass_log)
        c_no_mass = c_full[1:]
        k = _kernel_from_trace(c_no_mass)
        data_no_mass["name"].append(r.symbol)
        data_no_mass["F"].append(k["F"])
        data_no_mass["IC"].append(k["IC"])
        data_no_mass["gap"].append(k["heterogeneity_gap"])
        data_no_mass["cat"].append(r.category)
        gen = p.generation if p.generation > 0 else 0
        data_no_mass["gen"].append(gen)

    # Check T2: generation monotonicity for quarks
    quarks_full = {g: [] for g in [1, 2, 3]}
    quarks_no_mass = {g: [] for g in [1, 2, 3]}
    for i, cat in enumerate(data_full["cat"]):
        if cat == "Quark" and data_full["gen"][i] in (1, 2, 3):
            quarks_full[data_full["gen"][i]].append(data_full["F"][i])
            quarks_no_mass[data_no_mass["gen"][i]].append(data_no_mass["F"][i])

    gen_means_full = [np.mean(quarks_full[g]) for g in [1, 2, 3]]
    gen_means_no_mass = [np.mean(quarks_no_mass[g]) for g in [1, 2, 3]]

    mono_full = gen_means_full[0] < gen_means_full[1] < gen_means_full[2]
    mono_no_mass = gen_means_no_mass[0] < gen_means_no_mass[1] < gen_means_no_mass[2]

    print(
        f"  T2 with mass:    Gen1={gen_means_full[0]:.4f} < Gen2={gen_means_full[1]:.4f} < Gen3={gen_means_full[2]:.4f} → {'HOLDS' if mono_full else 'FAILS'}"
    )
    print(
        f"  T2 without mass: Gen1={gen_means_no_mass[0]:.4f} < Gen2={gen_means_no_mass[1]:.4f} < Gen3={gen_means_no_mass[2]:.4f} → {'HOLDS' if mono_no_mass else 'FAILS'}"
    )

    # ── Visualization: side-by-side comparison ──
    fig, axes = _dark_fig(1, 2, figsize=(16, 8))

    # Scatter: F_full vs F_no_mass for each particle
    ax = axes[0]
    cat_colors = {"Quark": ACCENT_CYAN, "Lepton": ACCENT_GREEN, "GaugeBoson": ACCENT_ORANGE, "ScalarBoson": ACCENT_PINK}
    for i, name in enumerate(data_full["name"]):
        color = cat_colors.get(data_full["cat"][i], TEXT_COLOR)
        ax.scatter(data_full["F"][i], data_no_mass["F"][i], c=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(
            name,
            (data_full["F"][i], data_no_mass["F"][i]),
            fontsize=7,
            color=TEXT_COLOR,
            textcoords="offset points",
            xytext=(5, 5),
        )

    # Diagonal
    ax.plot([0.3, 0.8], [0.3, 0.8], "--", color=ACCENT_RED, alpha=0.5, label="y = x (no change)")
    ax.set_xlabel("F (with mass channel)", fontsize=11)
    ax.set_ylabel("F (mass channel removed)", fontsize=11)
    ax.set_title("Probe 1a: Fidelity Shift When Mass Is Removed", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Bar chart: generation staircase with/without mass
    ax2 = axes[1]
    x = np.arange(3)
    width = 0.35
    bars1 = ax2.bar(x - width / 2, gen_means_full, width, color=ACCENT_CYAN, label="With mass", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, gen_means_no_mass, width, color=ACCENT_ORANGE, label="Without mass", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Gen 1\n(u, d)", "Gen 2\n(c, s)", "Gen 3\n(t, b)"])
    ax2.set_ylabel("⟨F⟩ (quark average)", fontsize=11)
    ax2.set_title(
        f"Probe 1b: Generation Monotonicity — {'INTRINSIC ✓' if mono_no_mass else 'MASS-DRIVEN ✗'}",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT_COLOR,
            )

    verdict = "INTRINSIC" if mono_no_mass else "MASS-DRIVEN"
    fig.suptitle(
        f"PROBE 1 — Mass-Channel Removal Experiment  |  T2 Verdict: {verdict}",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    _save(fig, "probe_01_mass_channel_removal.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 2: SUSY Catalog                                                     ║
# ║  Question: Does the fermion/boson gap close with supersymmetric partners?   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_2_susy_catalog() -> None:
    """Build hypothetical SUSY partners and test if the F/IC gap closes."""
    print("\n═══ PROBE 2: SUSY Catalog ═══")

    # Define SUSY partners — sfermions (spin 0) and gauginos (spin 1/2)
    susy_particles = [
        # Sfermions: spin 0 partners of SM fermions
        FundamentalParticle("selectron", "ẽ⁻", "Slepton", 200.0, -1.0, 0.0, 1, 1, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("smuon", "μ̃⁻", "Slepton", 350.0, -1.0, 0.0, 1, 2, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("stau", "τ̃⁻", "Slepton", 150.0, -1.0, 0.0, 1, 3, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("sneutrino_e", "ν̃ₑ", "Slepton", 180.0, 0.0, 0.0, 1, 1, 0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("sup", "ũ", "Squark", 800.0, 2 / 3, 0.0, 3, 1, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sdown", "d̃", "Squark", 850.0, -1 / 3, 0.0, 3, 1, -0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("scharm", "c̃", "Squark", 900.0, 2 / 3, 0.0, 3, 2, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sstrange", "s̃", "Squark", 870.0, -1 / 3, 0.0, 3, 2, -0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("stop", "t̃", "Squark", 600.0, 2 / 3, 0.0, 3, 3, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sbottom", "b̃", "Squark", 700.0, -1 / 3, 0.0, 3, 3, -0.5, 1 / 3, 0.0, 0.0, False),
        # Gauginos: spin 1/2 partners of SM bosons
        FundamentalParticle("gluino", "g̃", "Gaugino", 2000.0, 0.0, 0.5, 8, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("wino", "W̃±", "Gaugino", 300.0, 1.0, 0.5, 1, 0, 1.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("bino", "B̃⁰", "Gaugino", 150.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("higgsino", "H̃⁰", "Gaugino", 250.0, 0.0, 0.5, 1, 0, -0.5, -1.0, 0.0, 0.0, True),
    ]

    # Compute kernels for SM particles
    sm_fund = compute_all_fundamental()
    sm_fermions = [r for r in sm_fund if r.category in ("Quark", "Lepton")]
    sm_bosons = [r for r in sm_fund if r.category in ("GaugeBoson", "ScalarBoson")]

    # Compute kernels for SUSY particles
    susy_results = []
    for sp in susy_particles:
        c, w, _labels = normalize_fundamental(sp)
        k = compute_kernel_outputs(c, w, EPSILON)
        susy_results.append(
            {
                "name": sp.name,
                "symbol": sp.symbol,
                "cat": sp.category,
                "is_fermion": sp.is_fermion,
                "F": k["F"],
                "IC": k["IC"],
                "gap": k["heterogeneity_gap"],
                "spin": sp.spin,
                "mass": sp.mass_GeV,
            }
        )

    sfermions = [r for r in susy_results if not r["is_fermion"]]  # spin 0 → bosonic
    gauginos = [r for r in susy_results if r["is_fermion"]]  # spin 1/2 → fermionic

    # Combined statistics
    sm_ferm_F = np.mean([r.F for r in sm_fermions])
    sm_bos_F = np.mean([r.F for r in sm_bosons])
    sm_ferm_IC = np.mean([r.IC for r in sm_fermions])
    sm_bos_IC = np.mean([r.IC for r in sm_bosons])

    # SUSY extends bosonic sector with sfermions, fermionic sector with gauginos
    all_bosons_F = np.mean([r.F for r in sm_bosons] + [r["F"] for r in sfermions])
    all_fermions_F = np.mean([r.F for r in sm_fermions] + [r["F"] for r in gauginos])
    all_bosons_IC = np.mean([r.IC for r in sm_bosons] + [r["IC"] for r in sfermions])
    all_fermions_IC = np.mean([r.IC for r in sm_fermions] + [r["IC"] for r in gauginos])

    gap_sm = sm_ferm_F - sm_bos_F
    gap_susy = all_fermions_F - all_bosons_F

    print(f"  SM only:  ⟨F⟩_fermion={sm_ferm_F:.4f}  ⟨F⟩_boson={sm_bos_F:.4f}  gap={gap_sm:.4f}")
    print(f"  SM+SUSY:  ⟨F⟩_fermion={all_fermions_F:.4f}  ⟨F⟩_boson={all_bosons_F:.4f}  gap={gap_susy:.4f}")
    print(f"  Gap reduction: {(1 - gap_susy / gap_sm) * 100:.1f}%")

    # ── Visualization ──
    fig, axes = _dark_fig(1, 3, figsize=(20, 8))

    # Panel 1: F for SM vs SUSY particles
    ax1 = axes[0]
    y_labels = []
    y_pos = []
    colors = []
    f_vals = []

    # SM fermions
    for r in sm_fermions:
        y_labels.append(r.symbol)
        f_vals.append(r.F)
        colors.append(ACCENT_CYAN)
    # Gauginos
    for r in gauginos:
        y_labels.append(r["symbol"])
        f_vals.append(r["F"])
        colors.append(ACCENT_TEAL)
    # SM bosons
    for r in sm_bosons:
        y_labels.append(r.symbol)
        f_vals.append(r.F)
        colors.append(ACCENT_ORANGE)
    # Sfermions
    for r in sfermions:
        y_labels.append(r["symbol"])
        f_vals.append(r["F"])
        colors.append(ACCENT_YELLOW)

    y_pos = np.arange(len(y_labels))
    ax1.barh(y_pos, f_vals, color=colors, alpha=0.85, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=7)
    ax1.set_xlabel("Fidelity F", fontsize=11)
    ax1.set_title("Probe 2a: SM + SUSY Fidelity Spectrum", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=ACCENT_CYAN, label="SM Fermions"),
        Patch(facecolor=ACCENT_TEAL, label="Gauginos (SUSY)"),
        Patch(facecolor=ACCENT_ORANGE, label="SM Bosons"),
        Patch(facecolor=ACCENT_YELLOW, label="Sfermions (SUSY)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 2: Gap comparison bar chart
    ax2 = axes[1]
    categories = ["SM\nFermions", "SM\nBosons", "SM+SUSY\nFermions", "SM+SUSY\nBosons"]
    f_means = [sm_ferm_F, sm_bos_F, all_fermions_F, all_bosons_F]
    ic_means = [sm_ferm_IC, sm_bos_IC, all_fermions_IC, all_bosons_IC]
    x = np.arange(4)
    width = 0.35
    ax2.bar(x - width / 2, f_means, width, color=ACCENT_CYAN, label="⟨F⟩", alpha=0.85)
    ax2.bar(x + width / 2, ic_means, width, color=ACCENT_ORANGE, label="⟨IC⟩", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Probe 2b: Fermion/Boson Gap Before & After SUSY", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Gap arrows
    for i, (f, ic) in enumerate(zip(f_means, ic_means, strict=False)):
        gap = f - ic
        ax2.annotate(f"Δ={gap:.3f}", xy=(i, max(f, ic) + 0.02), fontsize=8, color=ACCENT_RED, ha="center")

    # Panel 3: IC scatter — SM vs SUSY
    ax3 = axes[2]
    for r in sm_fermions:
        ax3.scatter(r.F, r.IC, c=ACCENT_CYAN, s=60, zorder=5, edgecolors="white", linewidth=0.5)
    for r in sm_bosons:
        ax3.scatter(r.F, r.IC, c=ACCENT_ORANGE, s=60, zorder=5, edgecolors="white", linewidth=0.5)
    for r in sfermions:
        ax3.scatter(r["F"], r["IC"], c=ACCENT_YELLOW, s=80, marker="D", zorder=6, edgecolors="white", linewidth=0.5)
    for r in gauginos:
        ax3.scatter(r["F"], r["IC"], c=ACCENT_TEAL, s=80, marker="D", zorder=6, edgecolors="white", linewidth=0.5)

    ax3.plot([0, 1], [0, 1], "--", color=ACCENT_RED, alpha=0.4, label="IC = F (integrity bound)")
    ax3.set_xlabel("Fidelity F", fontsize=11)
    ax3.set_ylabel("Integrity Composite IC", fontsize=11)
    ax3.set_title("Probe 2c: F–IC Phase Space with SUSY", fontsize=13, fontweight="bold")
    ax3.legend(
        handles=[*legend_elements, plt.Line2D([0], [0], color=ACCENT_RED, linestyle="--", label="IC = F")],
        fontsize=7,
        facecolor=CARD_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    fig.suptitle(
        f"PROBE 2 — SUSY Catalog  |  Gap reduction: {(1 - gap_susy / gap_sm) * 100:.1f}%",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_02_susy_catalog.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 3: Quark IC at Variable Q                                           ║
# ║  Question: Does confinement cliff soften at high energy?                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_3_quark_ic_variable_q() -> None:
    """Inject running α_s(Q) into quark traces and watch IC evolve."""
    print("\n═══ PROBE 3: Quark IC at Variable Q ═══")

    quarks = [p for p in FUNDAMENTAL_PARTICLES if p.category == "Quark"]

    # Energy scales from 1 GeV to 100 TeV
    Q_values = np.logspace(0, 5, 80)

    # For each Q, compute α_s and inject it as a modified coupling channel
    # Replace the color_dof channel with α_s(Q)-modulated value
    results: dict[str, dict[str, list]] = {}

    for q in quarks:
        results[q.symbol] = {"Q": [], "F": [], "IC": [], "gap": [], "alpha_s": []}
        c_base, _w, _labels = normalize_fundamental(q)

        for Q in Q_values:
            coup = compute_running_coupling(Q)
            # Modulate the color_dof channel (index 3) by α_s(Q)/α_s(M_Z)
            # This simulates how quark–gluon coupling strength changes with scale
            alpha_ratio = min(coup.alpha_s / ALPHA_S_MZ, 3.0)  # cap at 3x
            c_mod = c_base.copy()
            c_mod[3] = np.clip(c_base[3] * alpha_ratio, EPSILON, 1 - EPSILON)

            k = _kernel_from_trace(c_mod)
            results[q.symbol]["Q"].append(Q)
            results[q.symbol]["F"].append(k["F"])
            results[q.symbol]["IC"].append(k["IC"])
            results[q.symbol]["gap"].append(k["heterogeneity_gap"])
            results[q.symbol]["alpha_s"].append(coup.alpha_s)

    # Also compute hadron IC for comparison (fixed)
    hadrons = compute_all_composite()
    hadron_IC_mean = np.mean([h.IC for h in hadrons])
    hadron_IC_max = max(h.IC for h in hadrons)

    print(f"  Top quark IC range: [{min(results['t']['IC']):.4f}, {max(results['t']['IC']):.4f}]")
    print(f"  Down quark IC range: [{min(results['d']['IC']):.4f}, {max(results['d']['IC']):.4f}]")
    print(f"  Hadron IC (fixed): mean={hadron_IC_mean:.4f}, max={hadron_IC_max:.4f}")

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    # Panel 1: IC vs Q for all quarks
    ax1 = axes[0, 0] if axes.ndim > 1 else axes[0]
    quark_colors = {
        "u": ACCENT_CYAN,
        "d": ACCENT_GREEN,
        "c": ACCENT_ORANGE,
        "s": ACCENT_YELLOW,
        "t": ACCENT_RED,
        "b": ACCENT_PURPLE,
    }
    for sym, data in results.items():
        ax1.semilogx(data["Q"], data["IC"], color=quark_colors.get(sym, TEXT_COLOR), linewidth=2, label=sym, alpha=0.9)
    ax1.axhline(
        hadron_IC_max, color=ACCENT_PINK, linestyle="--", alpha=0.6, label=f"Max hadron IC ({hadron_IC_max:.3f})"
    )
    ax1.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax1.set_ylabel("Integrity Composite IC", fontsize=11)
    ax1.set_title("Probe 3a: Quark IC vs Energy Scale", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 2: Heterogeneity gap vs Q
    ax2 = axes[0, 1] if axes.ndim > 1 else axes[1]
    for sym, data in results.items():
        ax2.semilogx(data["Q"], data["gap"], color=quark_colors.get(sym, TEXT_COLOR), linewidth=2, label=sym, alpha=0.9)
    ax2.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax2.set_ylabel("Heterogeneity gap (F − IC)", fontsize=11)
    ax2.set_title("Probe 3b: Gap Evolution with Energy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 3: α_s(Q) itself
    ax3 = axes[1, 0] if axes.ndim > 1 else axes[2]
    ax3.semilogx(Q_values, [compute_running_coupling(Q).alpha_s for Q in Q_values], color=ACCENT_CYAN, linewidth=2.5)
    ax3.axhline(ALPHA_S_MZ, color=ACCENT_ORANGE, linestyle=":", alpha=0.6, label=f"α_s(M_Z) = {ALPHA_S_MZ}")
    ax3.axvline(M_Z_GEV, color=ACCENT_GREEN, linestyle=":", alpha=0.6, label=f"M_Z = {M_Z_GEV} GeV")
    ax3.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax3.set_ylabel("α_s(Q)", fontsize=11)
    ax3.set_title("Probe 3c: Running Strong Coupling", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 4: Phase diagram — IC vs α_s for all quarks
    ax4 = axes[1, 1] if axes.ndim > 1 else axes[3]
    for sym, data in results.items():
        ax4.scatter(data["alpha_s"], data["IC"], c=quark_colors.get(sym, TEXT_COLOR), s=15, alpha=0.7, label=sym)
    ax4.axhline(hadron_IC_max, color=ACCENT_PINK, linestyle="--", alpha=0.6)
    ax4.set_xlabel("α_s(Q)", fontsize=11)
    ax4.set_ylabel("Quark IC", fontsize=11)
    ax4.set_title("Probe 3d: IC–Coupling Phase Space", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax4.invert_xaxis()

    fig.suptitle(
        "PROBE 3 — Quark IC at Variable Energy Scale  |  Does confinement soften?",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_03_quark_ic_variable_q.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 4: Dark Matter Kernel Signatures                                    ║
# ║  Question: What does a WIMP / axion / sterile neutrino look like?           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_4_dark_matter_kernel() -> None:
    """Build kernel signatures for dark matter candidates."""
    print("\n═══ PROBE 4: Dark Matter Kernel Signatures ═══")

    # Hypothetical dark matter candidates with theoretically motivated properties
    dm_candidates = [
        # WIMPs — massive, weakly interacting
        FundamentalParticle("Neutralino_light", "χ̃₁⁰(100)", "WIMP", 100.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("Neutralino_heavy", "χ̃₁⁰(1T)", "WIMP", 1000.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("Neutralino_split", "χ̃₁⁰(10T)", "WIMP", 10000.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        # Axion-like particles — ultralight, spin 0
        FundamentalParticle("Axion_QCD", "a(μeV)", "Axion", 1e-12, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
        FundamentalParticle("Axion_heavy", "a(meV)", "Axion", 1e-9, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
        FundamentalParticle("ALP", "a(keV)", "ALP", 1e-6, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
        # Sterile neutrinos — keV-scale, spin 1/2, no gauge interactions
        FundamentalParticle("Sterile_nu_light", "νₛ(1k)", "Sterile", 1e-6, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("Sterile_nu_warm", "νₛ(7k)", "Sterile", 7e-6, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("Sterile_nu_heavy", "νₛ(M)", "Sterile", 1e6, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        # Gravitino — spin 3/2 (we approximate)
        FundamentalParticle("Gravitino", "G̃(TeV)", "Gravitino", 1000.0, 0.0, 1.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
    ]

    # Compute SM reference
    sm_fund = compute_all_fundamental()
    # SM reference categories (used implicitly for context)
    _ = [r for r in sm_fund if "Neutrino" in r.name or r.symbol in ("νₑ", "ν_μ", "ν_τ")]

    dm_results = []
    for dm in dm_candidates:
        c, w, _labels = normalize_fundamental(dm)
        k = compute_kernel_outputs(c, w, EPSILON)
        dm_results.append(
            {
                "name": dm.name,
                "symbol": dm.symbol,
                "cat": dm.category,
                "mass": dm.mass_GeV,
                "spin": dm.spin,
                "F": k["F"],
                "IC": k["IC"],
                "gap": k["heterogeneity_gap"],
                "S": k["S"],
                "regime": k.get("regime", "unknown"),
            }
        )
        print(f"  {dm.symbol:>12s}: F={k['F']:.4f}  IC={k['IC']:.4f}  gap={k['heterogeneity_gap']:.4f}  S={k['S']:.4f}")

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    cat_colors = {
        "WIMP": ACCENT_CYAN,
        "Axion": ACCENT_YELLOW,
        "ALP": ACCENT_ORANGE,
        "Sterile": ACCENT_GREEN,
        "Gravitino": ACCENT_PURPLE,
    }

    # Panel 1: F–IC scatter with SM reference
    ax1 = axes.flat[0]
    # SM cloud
    for r in sm_fund:
        ax1.scatter(r.F, r.IC, c="#555555", s=30, alpha=0.4, zorder=3)
    # DM candidates
    for r in dm_results:
        ax1.scatter(
            r["F"],
            r["IC"],
            c=cat_colors.get(r["cat"], TEXT_COLOR),
            s=120,
            marker="*",
            zorder=6,
            edgecolors="white",
            linewidth=0.8,
        )
        ax1.annotate(
            r["symbol"], (r["F"], r["IC"]), fontsize=7, color=TEXT_COLOR, textcoords="offset points", xytext=(6, 4)
        )
    ax1.plot([0, 1], [0, 1], "--", color=ACCENT_RED, alpha=0.3)
    ax1.set_xlabel("Fidelity F", fontsize=11)
    ax1.set_ylabel("Integrity Composite IC", fontsize=11)
    ax1.set_title("Probe 4a: DM Candidates in F–IC Phase Space", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch

    dm_legend = [Patch(facecolor=c, label=name) for name, c in cat_colors.items()]
    dm_legend.append(Patch(facecolor="#555555", label="SM particles"))
    ax1.legend(handles=dm_legend, fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 2: Mass vs F colored by type
    ax2 = axes.flat[1]
    for r in dm_results:
        ax2.scatter(
            r["mass"],
            r["F"],
            c=cat_colors.get(r["cat"], TEXT_COLOR),
            s=120,
            marker="*",
            zorder=6,
            edgecolors="white",
            linewidth=0.8,
        )
        ax2.annotate(
            r["symbol"], (r["mass"], r["F"]), fontsize=7, color=TEXT_COLOR, textcoords="offset points", xytext=(6, 4)
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Mass (GeV)", fontsize=11)
    ax2.set_ylabel("Fidelity F", fontsize=11)
    ax2.set_title("Probe 4b: DM Mass–Fidelity Landscape", fontsize=13, fontweight="bold")

    # Panel 3: Heterogeneity gap bar chart
    ax3 = axes.flat[2]
    names = [r["symbol"] for r in dm_results]
    gaps = [r["gap"] for r in dm_results]
    bar_colors = [cat_colors.get(r["cat"], TEXT_COLOR) for r in dm_results]
    ax3.barh(np.arange(len(names)), gaps, color=bar_colors, alpha=0.85, height=0.7)
    ax3.set_yticks(np.arange(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel("Heterogeneity gap (F − IC)", fontsize=11)
    ax3.set_title("Probe 4c: DM Heterogeneity Gap", fontsize=13, fontweight="bold")
    ax3.invert_yaxis()

    # Panel 4: Entropy comparison
    ax4 = axes.flat[3]
    sm_S = [r.S for r in sm_fund]
    ax4.hist(sm_S, bins=15, color="#555555", alpha=0.6, label="SM particles", edgecolor=GRID_COLOR)
    for r in dm_results:
        ax4.axvline(r["S"], color=cat_colors.get(r["cat"], TEXT_COLOR), linewidth=2, alpha=0.8, linestyle="--")
    ax4.set_xlabel("Bernoulli Field Entropy S", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Probe 4d: DM Entropy vs SM Distribution", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle(
        "PROBE 4 — Dark Matter Kernel Signatures  |  WIMP / Axion / Sterile ν",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_04_dark_matter_kernel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 5: Channel-Ablation Periodic Kernel                                ║
# ║  Question: Which atomic channel controls d-block dominance?                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_5_channel_ablation() -> None:
    """Remove one channel at a time from the periodic kernel and measure impact."""
    print("\n═══ PROBE 5: Channel-Ablation Periodic Kernel ═══")

    full_results = batch_compute_all()

    # Group by block
    blocks = {}
    for r in full_results:
        blocks.setdefault(r.block, []).append(r)

    full_by_block = {}
    for b in ["s", "p", "d", "f"]:
        if b in blocks:
            full_by_block[b] = {
                "F": np.mean([r.F for r in blocks[b]]),
                "IC": np.mean([r.IC for r in blocks[b]]),
                "gap": np.mean([r.F - r.IC for r in blocks[b]]),
            }

    print(f"  Full kernel:  d-block ⟨F⟩={full_by_block['d']['F']:.4f}, ⟨IC⟩={full_by_block['d']['IC']:.4f}")

    # Channel ablation: remove one channel at a time
    channel_names = ["Z_norm", "EN", "radius", "IE", "EA", "T_melt", "T_boil", "density"]
    ablation_results: dict[str, dict[str, dict[str, float]]] = {}

    for _ch_idx, ch_name in enumerate(channel_names):
        ablation_results[ch_name] = {}
        block_data: dict[str, list] = {}

        for el_data in ELEMENTS:
            el = get_element(el_data.symbol)
            if el is None:
                continue
            c_full, _w_full, labels = _normalize_element(el)

            # Find which index corresponds to this channel
            if ch_name not in labels:
                # Channel was already missing for this element
                c_abl = c_full
                w_abl = np.ones(len(c_abl)) / len(c_abl)
            else:
                idx = labels.index(ch_name)
                c_abl = np.delete(c_full, idx)
                w_abl = np.ones(len(c_abl)) / len(c_abl)

            if len(c_abl) < 2:
                continue

            k = compute_kernel_outputs(c_abl, w_abl, 1e-6)

            # Find block
            result = next((r for r in full_results if r.symbol == el_data.symbol), None)
            if result is not None:
                block_data.setdefault(result.block, []).append(
                    {
                        "F": k["F"],
                        "IC": k["IC"],
                        "gap": k["heterogeneity_gap"],
                    }
                )

        for b in ["s", "p", "d", "f"]:
            if block_data.get(b):
                ablation_results[ch_name][b] = {
                    "F": np.mean([r["F"] for r in block_data[b]]),
                    "IC": np.mean([r["IC"] for r in block_data[b]]),
                    "gap": np.mean([r["gap"] for r in block_data[b]]),
                }

    # Compute impact: how much does removing each channel change d-block stats
    d_impact = {}
    for ch_name in channel_names:
        if "d" in ablation_results[ch_name] and "d" in full_by_block:
            d_impact[ch_name] = {
                "ΔF": ablation_results[ch_name]["d"]["F"] - full_by_block["d"]["F"],
                "ΔIC": ablation_results[ch_name]["d"]["IC"] - full_by_block["d"]["IC"],
                "Δgap": ablation_results[ch_name]["d"]["gap"] - full_by_block["d"]["gap"],
            }
            print(
                f"  Remove {ch_name:>8s}: d-block ΔF={d_impact[ch_name]['ΔF']:+.4f}  ΔIC={d_impact[ch_name]['ΔIC']:+.4f}  Δgap={d_impact[ch_name]['Δgap']:+.4f}"
            )

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    # Panel 1: Impact heatmap — ΔF for each block when each channel is removed
    ax1 = axes.flat[0]
    block_order = ["s", "p", "d", "f"]
    heatmap_data = np.zeros((len(channel_names), len(block_order)))
    for i, ch in enumerate(channel_names):
        for j, b in enumerate(block_order):
            if b in ablation_results[ch] and b in full_by_block:
                heatmap_data[i, j] = ablation_results[ch][b]["F"] - full_by_block[b]["F"]

    im = ax1.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-0.05, vmax=0.05)
    ax1.set_xticks(np.arange(len(block_order)))
    ax1.set_xticklabels([f"{b}-block" for b in block_order], fontsize=10)
    ax1.set_yticks(np.arange(len(channel_names)))
    ax1.set_yticklabels(channel_names, fontsize=9)
    ax1.set_title("Probe 5a: ΔF When Channel Removed", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
    cbar.set_label("ΔF", color=TEXT_COLOR)

    # Annotate cells
    for i in range(len(channel_names)):
        for j in range(len(block_order)):
            val = heatmap_data[i, j]
            color = "white" if abs(val) > 0.025 else TEXT_COLOR
            ax1.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=color)

    # Panel 2: Similar heatmap for ΔIC
    ax2 = axes.flat[1]
    heatmap_ic = np.zeros((len(channel_names), len(block_order)))
    for i, ch in enumerate(channel_names):
        for j, b in enumerate(block_order):
            if b in ablation_results[ch] and b in full_by_block:
                heatmap_ic[i, j] = ablation_results[ch][b]["IC"] - full_by_block[b]["IC"]

    im2 = ax2.imshow(heatmap_ic, cmap="RdBu_r", aspect="auto", vmin=-0.1, vmax=0.1)
    ax2.set_xticks(np.arange(len(block_order)))
    ax2.set_xticklabels([f"{b}-block" for b in block_order], fontsize=10)
    ax2.set_yticks(np.arange(len(channel_names)))
    ax2.set_yticklabels(channel_names, fontsize=9)
    ax2.set_title("Probe 5b: ΔIC When Channel Removed", fontsize=13, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
    cbar2.set_label("ΔIC", color=TEXT_COLOR)

    for i in range(len(channel_names)):
        for j in range(len(block_order)):
            val = heatmap_ic[i, j]
            color = "white" if abs(val) > 0.05 else TEXT_COLOR
            ax2.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=color)

    # Panel 3: d-block impact bar chart (ΔF and ΔIC per removed channel)
    ax3 = axes.flat[2]
    if d_impact:
        chs = list(d_impact.keys())
        df_vals = [d_impact[ch]["ΔF"] for ch in chs]
        dic_vals = [d_impact[ch]["ΔIC"] for ch in chs]
        x = np.arange(len(chs))
        width = 0.35
        ax3.bar(x - width / 2, df_vals, width, color=ACCENT_CYAN, label="ΔF", alpha=0.85)
        ax3.bar(x + width / 2, dic_vals, width, color=ACCENT_ORANGE, label="ΔIC", alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(chs, fontsize=8, rotation=45, ha="right")
        ax3.axhline(0, color=TEXT_COLOR, linewidth=0.5)
        ax3.set_ylabel("Change from full kernel", fontsize=11)
        ax3.set_title("Probe 5c: d-Block Impact per Channel", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 4: Radar chart showing relative channel importance for d-block
    ax4 = axes.flat[3]
    if d_impact:
        # Importance = magnitude of ΔIC when channel is removed (bigger = more important)
        importance = [abs(d_impact.get(ch, {}).get("ΔIC", 0)) for ch in channel_names]
        total = sum(importance) or 1
        importance_norm = [v / total for v in importance]

        # Simple bar chart of importance
        ax4.barh(
            np.arange(len(channel_names)),
            importance_norm,
            color=[ACCENT_CYAN if v == max(importance_norm) else ACCENT_GREEN for v in importance_norm],
            alpha=0.85,
            height=0.6,
        )
        ax4.set_yticks(np.arange(len(channel_names)))
        ax4.set_yticklabels(channel_names, fontsize=9)
        ax4.set_xlabel("Relative importance (normalized |ΔIC|)", fontsize=11)
        ax4.set_title("Probe 5d: Channel Importance Ranking for d-Block", fontsize=13, fontweight="bold")
        ax4.invert_yaxis()

        # Highlight winner
        winner_idx = np.argmax(importance_norm)
        ax4.text(
            importance_norm[winner_idx] + 0.02,
            winner_idx,
            f"← KeyDriver: {channel_names[winner_idx]}",
            fontsize=10,
            color=ACCENT_RED,
            va="center",
        )

    fig.suptitle(
        "PROBE 5 — Channel-Ablation Periodic Kernel  |  What drives d-block?",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_05_channel_ablation.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 6: GUT-Scale Coupling Extrapolation                                 ║
# ║  Question: Do g₁, g₂, g₃ unify in the Standard Model?                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_6_gut_scale_coupling() -> None:
    """Extrapolate couplings to GUT scale and map into kernel."""
    print("\n═══ PROBE 6: GUT-Scale Coupling Extrapolation ═══")

    # Energy scales from 10 GeV to 10^16 GeV
    Q_values = np.logspace(1, 16, 200)

    alpha_s_vals = []
    alpha_em_vals = []
    g1_vals = []
    g2_vals = []
    g3_vals = []
    unif_vals = []
    f_eff_vals = []

    SIN2_TW = 0.23122  # fixed weak mixing angle

    for Q in Q_values:
        r = compute_running_coupling(Q)
        alpha_s_vals.append(r.alpha_s)
        alpha_em_vals.append(r.alpha_em)
        f_eff_vals.append(r.F_eff)
        unif_vals.append(r.unification_proximity)

        # Compute gauge couplings g₁, g₂, g₃
        g3 = math.sqrt(4 * math.pi * r.alpha_s)
        g2 = math.sqrt(4 * math.pi * r.alpha_em / SIN2_TW)
        g1 = math.sqrt(5 / 3) * math.sqrt(4 * math.pi * r.alpha_em / (1 - SIN2_TW))
        g1_vals.append(g1)
        g2_vals.append(g2)
        g3_vals.append(g3)

    # Find best unification point
    best_idx = np.argmax(unif_vals)
    best_Q = Q_values[best_idx]
    best_unif = unif_vals[best_idx]

    # Coupling spread at best point
    g_at_best = [g1_vals[best_idx], g2_vals[best_idx], g3_vals[best_idx]]
    spread = np.std(g_at_best) / np.mean(g_at_best) * 100

    print(f"  Best unification proximity: {best_unif:.4f} at Q = {best_Q:.2e} GeV")
    print(f"  At best point: g₁={g1_vals[best_idx]:.4f}  g₂={g2_vals[best_idx]:.4f}  g₃={g3_vals[best_idx]:.4f}")
    print(f"  Coupling spread: {spread:.1f}%")

    # Build a kernel trace at each energy scale from the coupling triple
    coupling_kernels = {"Q": [], "F": [], "IC": [], "gap": [], "S": []}
    for i, Q in enumerate(Q_values):
        trace = np.array(
            [
                min(g1_vals[i] / 2.0, 1.0),  # g₁ normalized
                min(g2_vals[i] / 2.0, 1.0),  # g₂ normalized
                min(g3_vals[i] / 2.0, 1.0),  # g₃ normalized
            ]
        )
        k = _kernel_from_trace(trace, eps=1e-8)
        coupling_kernels["Q"].append(Q)
        coupling_kernels["F"].append(k["F"])
        coupling_kernels["IC"].append(k["IC"])
        coupling_kernels["gap"].append(k["heterogeneity_gap"])
        coupling_kernels["S"].append(k["S"])

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    # Panel 1: Gauge couplings vs Q
    ax1 = axes.flat[0]
    ax1.semilogx(Q_values, g1_vals, color=ACCENT_CYAN, linewidth=2.5, label="g₁ (hypercharge)")
    ax1.semilogx(Q_values, g2_vals, color=ACCENT_GREEN, linewidth=2.5, label="g₂ (weak)")
    ax1.semilogx(Q_values, g3_vals, color=ACCENT_RED, linewidth=2.5, label="g₃ (strong)")
    ax1.axvline(best_Q, color=ACCENT_YELLOW, linestyle=":", alpha=0.6, label=f"Best unification: Q={best_Q:.1e}")
    ax1.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax1.set_ylabel("Gauge coupling gᵢ", fontsize=11)
    ax1.set_title("Probe 6a: Gauge Coupling Running (1-loop SM)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.set_ylim(0, 4)

    # Panel 2: Unification proximity vs Q
    ax2 = axes.flat[1]
    ax2.semilogx(Q_values, unif_vals, color=ACCENT_PURPLE, linewidth=2.5)
    ax2.axhline(1.0, color=ACCENT_GREEN, linestyle="--", alpha=0.4, label="Perfect unification")
    ax2.axvline(best_Q, color=ACCENT_YELLOW, linestyle=":", alpha=0.6)
    ax2.fill_between(Q_values, unif_vals, alpha=0.15, color=ACCENT_PURPLE)
    ax2.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax2.set_ylabel("Unification proximity", fontsize=11)
    ax2.set_title(f"Probe 6b: How Close to Unification?  Peak={best_unif:.3f}", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 3: Coupling kernel — IC and gap vs Q
    ax3 = axes.flat[2]
    ax3.semilogx(coupling_kernels["Q"], coupling_kernels["F"], color=ACCENT_CYAN, linewidth=2, label="F")
    ax3.semilogx(coupling_kernels["Q"], coupling_kernels["IC"], color=ACCENT_ORANGE, linewidth=2, label="IC")
    ax3.semilogx(coupling_kernels["Q"], coupling_kernels["gap"], color=ACCENT_RED, linewidth=2, label="Gap (F−IC)")
    ax3.axvline(best_Q, color=ACCENT_YELLOW, linestyle=":", alpha=0.6)
    ax3.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax3.set_ylabel("Kernel invariant", fontsize=11)
    ax3.set_title("Probe 6c: Coupling Triple as GCD Kernel", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 4: 1/αᵢ plot (standard GUT plot)
    ax4 = axes.flat[3]
    inv_a1 = [1 / (g1_vals[i] ** 2 / (4 * math.pi)) if g1_vals[i] > 0 else 0 for i in range(len(Q_values))]
    inv_a2 = [1 / (g2_vals[i] ** 2 / (4 * math.pi)) if g2_vals[i] > 0 else 0 for i in range(len(Q_values))]
    inv_a3 = [1 / (g3_vals[i] ** 2 / (4 * math.pi)) if g3_vals[i] > 0 else 0 for i in range(len(Q_values))]
    ax4.semilogx(Q_values, inv_a1, color=ACCENT_CYAN, linewidth=2.5, label="1/α₁")
    ax4.semilogx(Q_values, inv_a2, color=ACCENT_GREEN, linewidth=2.5, label="1/α₂")
    ax4.semilogx(Q_values, inv_a3, color=ACCENT_RED, linewidth=2.5, label="1/α₃")
    ax4.axvline(best_Q, color=ACCENT_YELLOW, linestyle=":", alpha=0.6, label=f"Q_GUT ≈ {best_Q:.1e} GeV")
    ax4.set_xlabel("Energy scale Q (GeV)", fontsize=11)
    ax4.set_ylabel("1/αᵢ", fontsize=11)
    ax4.set_title("Probe 6d: Standard 1/α Running (GUT Plot)", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle(
        f"PROBE 6 — GUT-Scale Extrapolation  |  SM couplings DO NOT unify (spread={spread:.0f}%)",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_06_gut_scale_coupling.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 7: Molecular-Scale Kernel                                           ║
# ║  Question: Does IC recover above the hadron confinement cliff?              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_7_molecular_kernel() -> None:
    """Build molecular traces and check if IC recovers post-confinement."""
    print("\n═══ PROBE 7: Molecular-Scale Kernel ═══")

    # Define molecules by element composition
    molecules = {
        # Simple diatomics
        "H₂": [("H", 2)],
        "O₂": [("O", 2)],
        "N₂": [("N", 2)],
        "CO": [("C", 1), ("O", 1)],
        # Triatomics
        "H₂O": [("H", 2), ("O", 1)],
        "CO₂": [("C", 1), ("O", 2)],
        "O₃": [("O", 3)],
        # Hydrocarbons
        "CH₄": [("C", 1), ("H", 4)],
        "C₂H₆": [("C", 2), ("H", 6)],
        "C₆H₆": [("C", 6), ("H", 6)],  # Benzene
        "C₈H₁₈": [("C", 8), ("H", 18)],  # Octane
        # Bio-relevant
        "NH₃": [("N", 1), ("H", 3)],
        "H₂SO₄": [("H", 2), ("S", 1), ("O", 4)],
        "NaCl": [("Na", 1), ("Cl", 1)],
        "CaCO₃": [("Ca", 1), ("C", 1), ("O", 3)],
        # Industrial
        "SiO₂": [("Si", 1), ("O", 2)],
        "Fe₂O₃": [("Fe", 2), ("O", 3)],
        "Al₂O₃": [("Al", 2), ("O", 3)],
        "TiO₂": [("Ti", 1), ("O", 2)],
        # Complex
        "C₂H₅OH": [("C", 2), ("H", 6), ("O", 1)],  # Ethanol
        "C₆H₁₂O₆": [("C", 6), ("H", 12), ("O", 6)],  # Glucose
        "C₃H₈O₃": [("C", 3), ("H", 8), ("O", 3)],  # Glycerol
    }

    mol_results = []
    for name, formula in molecules.items():
        total_atoms = sum(count for _, count in formula)
        c_parts = []
        w_parts = []

        valid = True
        for symbol, count in formula:
            el = get_element(symbol)
            if el is None:
                valid = False
                break
            c_el, _w_el, _labels = _normalize_element(el)
            frac = count / total_atoms
            c_parts.append(c_el)
            w_parts.append(np.full(len(c_el), frac / len(c_el)))

        if not valid or not c_parts:
            continue

        c_mol = np.concatenate(c_parts)
        w_mol = np.concatenate(w_parts)
        # Renormalize weights
        w_mol = w_mol / w_mol.sum()

        k = compute_kernel_outputs(np.clip(c_mol, 1e-6, 1 - 1e-6), w_mol, 1e-6)

        mol_results.append(
            {
                "name": name,
                "n_atoms": total_atoms,
                "n_elements": len(formula),
                "n_channels": len(c_mol),
                "F": k["F"],
                "IC": k["IC"],
                "gap": k["heterogeneity_gap"],
                "S": k["S"],
            }
        )
        print(
            f"  {name:>10s} ({total_atoms:>2d} atoms, {len(c_mol):>2d} ch): F={k['F']:.4f}  IC={k['IC']:.4f}  gap={k['heterogeneity_gap']:.4f}"
        )

    # Cross-scale comparison
    fund = compute_all_fundamental()
    comp = compute_all_composite()
    atoms = batch_compute_all()

    scale_data = [
        ("Fundamental\n(quarks, leptons)", np.mean([r.F for r in fund]), np.mean([r.IC for r in fund])),
        ("Composite\n(hadrons)", np.mean([r.F for r in comp]), np.mean([r.IC for r in comp])),
        ("Atomic\n(118 elements)", np.mean([r.F for r in atoms]), np.mean([r.IC for r in atoms])),
        ("Molecular\n(22 molecules)", np.mean([r["F"] for r in mol_results]), np.mean([r["IC"] for r in mol_results])),
    ]

    print("\n  Scale ladder:")
    for name, f, ic in scale_data:
        clean = name.replace("\n", " ")
        print(f"    {clean:>30s}: ⟨F⟩={f:.4f}  ⟨IC⟩={ic:.4f}  gap={f - ic:.4f}")

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    # Panel 1: Molecule F and IC bar chart
    ax1 = axes.flat[0]
    names = [r["name"] for r in mol_results]
    f_vals = [r["F"] for r in mol_results]
    ic_vals = [r["IC"] for r in mol_results]
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width / 2, f_vals, width, color=ACCENT_CYAN, label="F", alpha=0.85)
    ax1.bar(x + width / 2, ic_vals, width, color=ACCENT_ORANGE, label="IC", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7, rotation=55, ha="right")
    ax1.set_ylabel("Kernel invariant", fontsize=11)
    ax1.set_title("Probe 7a: Molecular Kernel — F and IC", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 2: Scale ladder — F and IC across scales
    ax2 = axes.flat[1]
    scale_names = [s[0] for s in scale_data]
    scale_F = [s[1] for s in scale_data]
    scale_IC = [s[2] for s in scale_data]
    scale_gap = [f - ic for f, ic in zip(scale_F, scale_IC, strict=False)]

    x = np.arange(len(scale_names))
    ax2.plot(x, scale_F, "o-", color=ACCENT_CYAN, linewidth=2.5, markersize=10, label="⟨F⟩")
    ax2.plot(x, scale_IC, "s-", color=ACCENT_ORANGE, linewidth=2.5, markersize=10, label="⟨IC⟩")
    ax2.fill_between(x, scale_F, scale_IC, alpha=0.15, color=ACCENT_RED, label="Heterogeneity gap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scale_names, fontsize=9)
    ax2.set_ylabel("Kernel mean", fontsize=11)
    ax2.set_title("Probe 7b: Complete Scale Ladder — Subatomic → Molecular", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Channel annotation
    for i, (f, ic, g) in enumerate(zip(scale_F, scale_IC, scale_gap, strict=False)):
        ax2.annotate(f"Δ={g:.3f}", (i, (f + ic) / 2), fontsize=9, color=ACCENT_RED, ha="center", fontweight="bold")

    # Panel 3: Molecule complexity vs IC
    ax3 = axes.flat[2]
    n_atoms = [r["n_atoms"] for r in mol_results]
    mol_IC = [r["IC"] for r in mol_results]
    mol_F_list = [r["F"] for r in mol_results]
    ax3.scatter(n_atoms, mol_IC, c=ACCENT_ORANGE, s=80, zorder=5, edgecolors="white", linewidth=0.5, label="IC")
    ax3.scatter(n_atoms, mol_F_list, c=ACCENT_CYAN, s=80, zorder=5, edgecolors="white", linewidth=0.5, label="F")
    for _i, r in enumerate(mol_results):
        ax3.annotate(
            r["name"], (r["n_atoms"], r["IC"]), fontsize=6, color=TEXT_COLOR, textcoords="offset points", xytext=(5, -8)
        )
    ax3.set_xlabel("Number of atoms", fontsize=11)
    ax3.set_ylabel("Kernel invariant", fontsize=11)
    ax3.set_title("Probe 7c: Complexity vs Coherence", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 4: Gap at each scale
    ax4 = axes.flat[3]
    gap_colors = [ACCENT_CYAN, ACCENT_RED, ACCENT_GREEN, ACCENT_PURPLE]
    bars = ax4.bar(np.arange(len(scale_names)), scale_gap, color=gap_colors, alpha=0.85)
    ax4.set_xticks(np.arange(len(scale_names)))
    ax4.set_xticklabels(scale_names, fontsize=9)
    ax4.set_ylabel("Heterogeneity gap (F − IC)", fontsize=11)
    ax4.set_title("Probe 7d: Gap Evolution Across Scales", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, scale_gap, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )

    # Mark confinement cliff
    ax4.annotate(
        "CONFINEMENT\nCLIFF",
        (1, scale_gap[1]),
        fontsize=10,
        color=ACCENT_RED,
        ha="center",
        va="bottom",
        fontweight="bold",
        xytext=(0, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": ACCENT_RED},
    )

    ic_recovered = scale_gap[3] < scale_gap[1]
    fig.suptitle(
        f"PROBE 7 — Molecular-Scale Kernel  |  IC {'RECOVERS ✓' if ic_recovered else 'STAYS LOW ✗'} post-confinement",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_07_molecular_kernel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROBE 8: Wave Q-Factor vs IC Correlation                                  ║
# ║  Question: Is energy retention the driver of wave coherence?                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def probe_8_wave_q_factor_vs_ic() -> None:
    """Investigate Q-factor as the primary driver of wave IC."""
    print("\n═══ PROBE 8: Wave Q-Factor vs IC ═══")

    waves = compute_all_wave_systems()

    # Extract Q-factor values from WAVE_SYSTEMS database
    q_factors = {}
    for ws in WAVE_SYSTEMS:
        name = ws[0]
        q_factor = ws[4]  # Q_factor is 5th element in tuple
        q_factors[name] = q_factor

    # Gather data
    data: dict[str, list] = {"name": [], "type": [], "Q": [], "F": [], "IC": [], "gap": [], "S": [], "trace_Q_norm": []}
    for w in waves:
        if w.system in q_factors:
            data["name"].append(w.system)
            data["type"].append(w.wave_type)
            data["Q"].append(q_factors[w.system])
            data["F"].append(w.F)
            data["IC"].append(w.IC)
            data["gap"].append(w.gap)
            data["S"].append(w.S)
            data["trace_Q_norm"].append(w.trace[3])  # Normalized Q in trace

    Q_arr = np.array(data["Q"])
    IC_arr = np.array(data["IC"])
    F_arr = np.array(data["F"])
    gap_arr = np.array(data["gap"])
    log_Q = np.log10(Q_arr + 1)

    # Correlation analysis
    from scipy import stats as scipy_stats

    r_Q_IC, p_Q_IC = scipy_stats.pearsonr(log_Q, IC_arr)
    r_Q_F, p_Q_F = scipy_stats.pearsonr(log_Q, F_arr)
    r_Q_gap, p_Q_gap = scipy_stats.pearsonr(log_Q, gap_arr)
    rho_Q_IC, _p_rho = scipy_stats.spearmanr(log_Q, IC_arr)

    print(f"  Pearson r(log₁₀Q, IC):  {r_Q_IC:+.4f} (p={p_Q_IC:.2e})")
    print(f"  Pearson r(log₁₀Q, F):   {r_Q_F:+.4f} (p={p_Q_F:.2e})")
    print(f"  Pearson r(log₁₀Q, gap): {r_Q_gap:+.4f} (p={p_Q_gap:.2e})")
    print(f"  Spearman ρ(log₁₀Q, IC): {rho_Q_IC:+.4f}")

    # Sweep: what happens if we vary Q_factor while keeping other channels fixed?
    # Use "Concert A" as base
    base_ws = None
    for ws in WAVE_SYSTEMS:
        if ws[0] == "Concert A":
            base_ws = ws
            break

    sweep_Q = np.logspace(0, 12, 100)
    sweep_results: dict[str, list] = {"Q": [], "F": [], "IC": [], "gap": []}
    if base_ws:
        for q in sweep_Q:
            r = compute_wave_system(
                "Sweep",
                base_ws[1],
                frequency=base_ws[2],
                wavelength=base_ws[3],
                phase_velocity=base_ws[2] * base_ws[3],
                Q_factor=q,
                coherence_lengths=base_ws[5],
                amplitude_norm=base_ws[6],
            )
            sweep_results["Q"].append(q)
            sweep_results["F"].append(r.F)
            sweep_results["IC"].append(r.IC)
            sweep_results["gap"].append(r.gap)

    # ── Visualization ──
    fig, axes = _dark_fig(2, 2, figsize=(18, 12))

    type_colors = {
        "sound": ACCENT_CYAN,
        "electromagnetic": ACCENT_YELLOW,
        "water": ACCENT_GREEN,
        "seismic": ACCENT_RED,
        "gravitational": ACCENT_PURPLE,
        "matter": ACCENT_PINK,
    }

    # Panel 1: log(Q) vs IC scatter with regression line
    ax1 = axes.flat[0]
    for i, name in enumerate(data["name"]):
        color = type_colors.get(data["type"][i], TEXT_COLOR)
        ax1.scatter(log_Q[i], IC_arr[i], c=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax1.annotate(
            name, (log_Q[i], IC_arr[i]), fontsize=6, color=TEXT_COLOR, textcoords="offset points", xytext=(5, 3)
        )

    # Regression line
    slope, intercept = np.polyfit(log_Q, IC_arr, 1)
    x_fit = np.linspace(log_Q.min(), log_Q.max(), 100)
    ax1.plot(
        x_fit, slope * x_fit + intercept, "--", color=ACCENT_RED, linewidth=2, label=f"r={r_Q_IC:.3f}, ρ={rho_Q_IC:.3f}"
    )
    ax1.set_xlabel("log₁₀(Q-factor)", fontsize=11)
    ax1.set_ylabel("Integrity Composite IC", fontsize=11)
    ax1.set_title("Probe 8a: Q-Factor vs IC (All Systems)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 2: Q sweep — holding other channels fixed
    ax2 = axes.flat[1]
    if sweep_results["Q"]:
        ax2.semilogx(sweep_results["Q"], sweep_results["F"], color=ACCENT_CYAN, linewidth=2.5, label="F")
        ax2.semilogx(sweep_results["Q"], sweep_results["IC"], color=ACCENT_ORANGE, linewidth=2.5, label="IC")
        ax2.semilogx(
            sweep_results["Q"], sweep_results["gap"], color=ACCENT_RED, linewidth=2, label="Gap", linestyle="--"
        )
        ax2.fill_between(sweep_results["Q"], sweep_results["F"], sweep_results["IC"], alpha=0.15, color=ACCENT_RED)
    ax2.set_xlabel("Q-factor (swept from 1 to 10¹²)", fontsize=11)
    ax2.set_ylabel("Kernel invariant", fontsize=11)
    ax2.set_title("Probe 8b: Q-Factor Sweep (Concert A base)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Panel 3: log(Q) vs F colored by type
    ax3 = axes.flat[2]
    for i, _name in enumerate(data["name"]):
        color = type_colors.get(data["type"][i], TEXT_COLOR)
        ax3.scatter(log_Q[i], F_arr[i], c=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)
    slope_f, intercept_f = np.polyfit(log_Q, F_arr, 1)
    ax3.plot(x_fit, slope_f * x_fit + intercept_f, "--", color=ACCENT_RED, linewidth=2, label=f"r={r_Q_F:.3f}")
    ax3.set_xlabel("log₁₀(Q-factor)", fontsize=11)
    ax3.set_ylabel("Fidelity F", fontsize=11)
    ax3.set_title("Probe 8c: Q-Factor vs Fidelity", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    from matplotlib.patches import Patch

    type_legend = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax3.legend(
        handles=[*type_legend, plt.Line2D([0], [0], color=ACCENT_RED, linestyle="--", label=f"r={r_Q_F:.3f}")],
        fontsize=7,
        facecolor=CARD_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    # Panel 4: Channel contribution analysis — which of the 6 channels correlates most with IC?
    ax4 = axes.flat[3]
    channel_names = ["frequency", "wavelength", "velocity", "Q_factor", "coherence", "amplitude"]
    channel_corrs = []
    for ch_idx in range(6):
        ch_vals = np.array([w.trace[ch_idx] for w in waves if w.system in q_factors])
        r_val, _p = scipy_stats.pearsonr(ch_vals, IC_arr)
        channel_corrs.append(r_val)

    bar_colors = [ACCENT_CYAN if ch != "Q_factor" else ACCENT_YELLOW for ch in channel_names]
    bars = ax4.barh(np.arange(len(channel_names)), channel_corrs, color=bar_colors, alpha=0.85, height=0.6)
    ax4.set_yticks(np.arange(len(channel_names)))
    ax4.set_yticklabels(channel_names, fontsize=10)
    ax4.set_xlabel("Pearson r with IC", fontsize=11)
    ax4.set_title("Probe 8d: Channel–IC Correlation Ranking", fontsize=13, fontweight="bold")
    ax4.axvline(0, color=TEXT_COLOR, linewidth=0.5)
    ax4.invert_yaxis()

    # Annotate values
    for bar, val in zip(bars, channel_corrs, strict=False):
        x_pos = val + 0.02 if val >= 0 else val - 0.06
        ax4.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", fontsize=9, color=TEXT_COLOR, va="center")

    # Highlight which channel dominates
    winner_idx = np.argmax(np.abs(channel_corrs))
    fig.suptitle(
        f"PROBE 8 — Wave Q-Factor vs IC  |  Strongest driver: {channel_names[winner_idx]} (r={channel_corrs[winner_idx]:+.3f})",
        fontsize=15,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_08_wave_q_factor_ic.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SYNTHESIS: Cross-Probe Summary Dashboard                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def synthesis_dashboard() -> None:
    """Generate a summary dashboard pulling together key findings from all 8 probes."""
    print("\n═══ SYNTHESIS DASHBOARD ═══")

    fig, axes = _dark_fig(2, 4, figsize=(28, 12))

    probes = [
        {"id": 1, "title": "Mass-Channel\nRemoval", "color": ACCENT_CYAN},
        {"id": 2, "title": "SUSY\nCatalog", "color": ACCENT_GREEN},
        {"id": 3, "title": "Quark IC\nvs Energy", "color": ACCENT_ORANGE},
        {"id": 4, "title": "Dark Matter\nKernel", "color": ACCENT_RED},
        {"id": 5, "title": "Channel\nAblation", "color": ACCENT_PURPLE},
        {"id": 6, "title": "GUT-Scale\nCoupling", "color": ACCENT_YELLOW},
        {"id": 7, "title": "Molecular\nKernel", "color": ACCENT_PINK},
        {"id": 8, "title": "Wave Q\nvs IC", "color": ACCENT_TEAL},
    ]

    for i, probe in enumerate(probes):
        ax = axes.flat[i]
        # Just create a summary card for each probe
        ax.text(
            0.5,
            0.85,
            f"PROBE {probe['id']}",
            fontsize=18,
            fontweight="bold",
            color=probe["color"],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.55,
            probe["title"],
            fontsize=14,
            color=TEXT_COLOR,
            ha="center",
            va="center",
            transform=ax.transAxes,
            linespacing=1.5,
        )

        # Status indicator
        ax.add_patch(plt.Circle((0.5, 0.25), 0.08, transform=ax.transAxes, color=ACCENT_GREEN, alpha=0.9))
        ax.text(
            0.5,
            0.25,
            "✓",
            fontsize=16,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.10, "COMPLETE", fontsize=10, color=ACCENT_GREEN, ha="center", va="center", transform=ax.transAxes
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "ALL 8 PROBES — COMPLETE  |  GCD Kernel Experimental Program",
        fontsize=18,
        fontweight="bold",
        color=ACCENT_CYAN,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "probe_00_synthesis_dashboard.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    """Execute all probes or a specific one."""
    import argparse

    parser = argparse.ArgumentParser(description="Run GCD experimental probes")
    parser.add_argument("--probe", type=int, help="Run a specific probe (1-8)")
    args = parser.parse_args()

    probes = {
        1: probe_1_mass_channel_removal,
        2: probe_2_susy_catalog,
        3: probe_3_quark_ic_variable_q,
        4: probe_4_dark_matter_kernel,
        5: probe_5_channel_ablation,
        6: probe_6_gut_scale_coupling,
        7: probe_7_molecular_kernel,
        8: probe_8_wave_q_factor_vs_ic,
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  GCD KERNEL EXPERIMENTAL PROGRAM — ALL 8 PROBES            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.probe:
        if args.probe in probes:
            probes[args.probe]()
        else:
            print(f"  Unknown probe: {args.probe}. Valid: 1-8")
            sys.exit(1)
    else:
        for _n, func in sorted(probes.items()):
            func()
        synthesis_dashboard()

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ALL PROBES COMPLETE — Images saved to images/             ║")
    print("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
