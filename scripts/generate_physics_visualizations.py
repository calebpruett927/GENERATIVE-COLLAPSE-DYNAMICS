#!/usr/bin/env python3
"""Generate Eye-Opening Physics Visualizations from GCD Kernel Data.

Produces 10 publication-quality figures revealing deep structural
patterns across the Standard Model, nuclear physics, atomic physics,
everyday wave phenomena, and electroweak symmetry breaking — all
through the lens of the GCD kernel (F, ω, IC, S, C, κ).

Usage:
    python scripts/generate_physics_visualizations.py

Output:
    images/physics_viz_*.png  (10 figures)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Ensure workspace root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from closures.atomic_physics.periodic_kernel import batch_compute_all
from closures.everyday_physics.wave_phenomena import compute_all_wave_systems
from closures.nuclear_physics.element_data import ELEMENTS
from closures.standard_model.coupling_constants import compute_running_coupling
from closures.standard_model.subatomic_kernel import (
    FUNDAMENTAL_PARTICLES,
    compute_all,
    compute_all_composite,
    compute_all_fundamental,
)
from closures.standard_model.symmetry_breaking import (
    FERMION_MASSES,
    compute_higgs_mechanism,
)

OUT = _ROOT / "images"
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Color palettes
# ══════════════════════════════════════════════════════════════════
CAT_COLORS = {
    "Quark": "#e74c3c",
    "Lepton": "#3498db",
    "GaugeBoson": "#f39c12",
    "ScalarBoson": "#2ecc71",
    "Baryon": "#9b59b6",
    "Meson": "#1abc9c",
}
WAVE_COLORS = {
    "Sound": "#e67e22",
    "Electromagnetic": "#e74c3c",
    "Water": "#3498db",
    "Seismic": "#795548",
    "Gravitational": "#9b59b6",
    "Matter": "#2ecc71",
}
REGIME_COLORS = {
    "Stable": "#27ae60",
    "Watch": "#f39c12",
    "Collapse": "#e74c3c",
}
BLOCK_COLORS = {"s": "#e74c3c", "p": "#3498db", "d": "#f39c12", "f": "#2ecc71"}


def _style():
    """Set publication-quality style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "text.color": "#c9d1d9",
            "axes.labelcolor": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "axes.edgecolor": "#30363d",
            "grid.color": "#21262d",
            "grid.alpha": 0.6,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#0d1117",
        }
    )


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: The Confinement Cliff — IC drops 98% quarks → hadrons
# ══════════════════════════════════════════════════════════════════
def fig1_confinement_cliff():
    """T3: The most dramatic feature — confinement as IC collapse."""
    fund = compute_all_fundamental()
    comp = compute_all_composite()

    quarks = [r for r in fund if r.category == "Quark"]
    hadrons = comp  # All composites are hadrons

    quarks_sorted = sorted(quarks, key=lambda r: r.IC, reverse=True)
    hadrons_sorted = sorted(hadrons, key=lambda r: r.IC, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: IC waterfall
    names_q = [q.symbol for q in quarks_sorted]
    ics_q = [q.IC for q in quarks_sorted]
    names_h = [h.symbol for h in hadrons_sorted]
    ics_h = [h.IC for h in hadrons_sorted]

    x_q = np.arange(len(names_q))
    x_h = np.arange(len(names_h)) + len(names_q) + 1

    ax1.bar(x_q, ics_q, color="#e74c3c", alpha=0.85, label="Quarks (confined)")
    ax1.bar(x_h, ics_h, color="#9b59b6", alpha=0.85, label="Hadrons (composite)")

    # Draw the cliff line
    cliff_x = len(names_q) + 0.5
    ax1.axvline(cliff_x, color="#f39c12", linewidth=2, linestyle="--", alpha=0.8)
    ax1.text(
        cliff_x, max(ics_q) * 0.9, "CONFINEMENT\nBOUNDARY", ha="center", fontsize=10, fontweight="bold", color="#f39c12"
    )

    ax1.set_xticks(list(x_q) + list(x_h))
    ax1.set_xticklabels(names_q + names_h, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("IC (Integrity Composite)")
    ax1.set_title("The Confinement Cliff: IC Drops 98% at Quark → Hadron Boundary")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Right: F vs IC scatter showing the gap explosion
    all_res = fund + comp
    for r in all_res:
        color = CAT_COLORS.get(r.category, "#888")
        ax2.scatter(r.F, r.IC, c=color, s=100, alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3)
        ax2.annotate(
            r.symbol,
            (r.F, r.IC),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            color="#c9d1d9",
        )

    # IC = F line (perfect integrity)
    ax2.plot([0, 1], [0, 1], "--", color="#888", alpha=0.5, label="IC = F (no gap)")
    ax2.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color="#e74c3c")
    ax2.set_xlabel("Fidelity (F)")
    ax2.set_ylabel("Integrity Composite (IC)")
    ax2.set_title("Heterogeneity Gap: Δ = F − IC")
    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax2.legend(handles=handles, loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.2, 0.85)
    ax2.set_ylim(-0.02, 0.75)

    fig.suptitle(
        "INSIGHT 1: Confinement is Visible in the Kernel — One Dead Channel Kills IC",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_01_confinement_cliff.png")
    plt.close(fig)
    print("  [1/10] Confinement cliff ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Fermion vs Boson Split (T1: Spin-Statistics)
# ══════════════════════════════════════════════════════════════════
def fig2_fermion_boson():
    """T1: Fermions have systematically higher fidelity than bosons."""
    fund = compute_all_fundamental()
    fermions = [
        r
        for r in fund
        if FUNDAMENTAL_PARTICLES[
            next(i for i, p in enumerate(FUNDAMENTAL_PARTICLES) if p.symbol == r.symbol)
        ].is_fermion
    ]
    bosons = [r for r in fund if r not in fermions]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: F distribution
    f_ferm = [r.F for r in fermions]
    f_bos = [r.F for r in bosons]
    ax = axes[0]
    positions = [1, 2]
    parts = ax.violinplot([f_ferm, f_bos], positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#3498db", "#f39c12"][i])
        pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Fermions\n(½-integer spin)", "Bosons\n(integer spin)"])
    ax.set_ylabel("Fidelity (F)")
    ax.set_title("Fidelity by Spin-Statistics")
    f_mean_ferm = np.mean(f_ferm)
    f_mean_bos = np.mean(f_bos)
    ax.text(1, f_mean_ferm + 0.02, f"⟨F⟩ = {f_mean_ferm:.3f}", ha="center", fontsize=10, color="#3498db")
    ax.text(2, f_mean_bos + 0.02, f"⟨F⟩ = {f_mean_bos:.3f}", ha="center", fontsize=10, color="#f39c12")
    ax.axhline(0.5, color="#888", linestyle=":", alpha=0.4)
    ax.grid(alpha=0.3)

    # Middle: IC distribution
    ic_ferm = [r.IC for r in fermions]
    ic_bos = [r.IC for r in bosons]
    ax = axes[1]
    parts2 = ax.violinplot([ic_ferm, ic_bos], positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts2["bodies"]):
        pc.set_facecolor(["#3498db", "#f39c12"][i])
        pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Fermions", "Bosons"])
    ax.set_ylabel("Integrity Composite (IC)")
    ax.set_title("Integrity by Spin-Statistics")
    ax.grid(alpha=0.3)

    # Right: Individual particles
    ax = axes[2]
    for r in fund:
        cat = r.category
        color = CAT_COLORS.get(cat, "#888")
        marker = "o" if r in fermions else "s"
        ax.scatter(r.F, r.IC, c=color, s=120, marker=marker, alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(
            r.symbol, (r.F, r.IC), fontsize=8, ha="center", va="bottom", xytext=(0, 6), textcoords="offset points"
        )
    ax.plot([0, 1], [0, 1], "--", color="#888", alpha=0.4)
    ax.set_xlabel("F")
    ax.set_ylabel("IC")
    ax.set_title("All 17 Fundamental Particles")
    handles = [
        mpatches.Patch(color=c, label=k)
        for k, c in CAT_COLORS.items()
        if k in {"Quark", "Lepton", "GaugeBoson", "ScalarBoson"}
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "INSIGHT 2: Nature's Deepest Divide — Fermions Have Higher Fidelity Than Bosons",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_02_fermion_boson.png")
    plt.close(fig)
    print("  [2/10] Fermion/Boson split ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Generation Monotonicity Staircase (T2)
# ══════════════════════════════════════════════════════════════════
def fig3_generation_staircase():
    """T2: ⟨F⟩ increases monotonically across generations."""
    fund = compute_all_fundamental()
    quarks = [(r, p) for r, p in zip(fund, FUNDAMENTAL_PARTICLES, strict=False) if p.category == "Quark"]
    leptons = [(r, p) for r, p in zip(fund, FUNDAMENTAL_PARTICLES, strict=False) if p.category == "Lepton"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (data, title, color_base) in enumerate(
        [
            (quarks, "Quarks", ["#ff6b6b", "#ee5a5a", "#cc4444"]),
            (leptons, "Leptons", ["#74b9ff", "#5dade2", "#3498db"]),
        ]
    ):
        ax = axes[idx]
        gen_data = {1: [], 2: [], 3: []}
        for r, p in data:
            gen_data[p.generation].append((r, p))

        gen_means_F = {}
        gen_means_IC = {}
        for g in [1, 2, 3]:
            particles = gen_data[g]
            names = [p.symbol for _, p in particles]
            Fs = [r.F for r, _ in particles]
            ICs = [r.IC for r, _ in particles]
            gen_means_F[g] = np.mean(Fs)
            gen_means_IC[g] = np.mean(ICs)

            x_base = (g - 1) * 3
            xs = [x_base + j for j in range(len(names))]
            ax.bar(xs, Fs, color=color_base[g - 1], alpha=0.85, width=0.8)
            for x, name, f in zip(xs, names, Fs, strict=False):
                ax.text(x, f + 0.01, name, ha="center", fontsize=9, color="#c9d1d9")

        # Connect generation means with staircase line
        gen_xs = [1, 4, 7]
        gen_fs = [gen_means_F[1], gen_means_F[2], gen_means_F[3]]
        ax.plot(
            gen_xs, gen_fs, "o-", color="#f39c12", linewidth=2.5, markersize=10, zorder=5, label="⟨F⟩ by generation"
        )

        ax.set_xticks([1, 4, 7])
        ax.set_xticklabels(["Gen 1", "Gen 2", "Gen 3"], fontsize=11)
        ax.set_ylabel("Fidelity (F)")
        ax.set_title(f"{title}: Generation Monotonicity")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # Annotate means
        for _g, x, f in zip([1, 2, 3], gen_xs, gen_fs, strict=False):
            ax.annotate(
                f"⟨F⟩={f:.3f}",
                (x, f),
                fontsize=9,
                fontweight="bold",
                color="#f39c12",
                xytext=(15, -15),
                textcoords="offset points",
            )

    fig.suptitle(
        "INSIGHT 3: Heavier Generations Have Higher Fidelity — Mass Log Drives the Staircase",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_03_generation_staircase.png")
    plt.close(fig)
    print("  [3/10] Generation staircase ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Running Coupling Constants — α_s, α_em vs. Energy
# ══════════════════════════════════════════════════════════════════
def fig4_running_couplings():
    """T9: The strong force weakens and EM strengthens with energy."""
    Qs = np.logspace(0.7, 4.2, 500)  # 5 GeV to ~16 TeV
    alphas_s = []
    alphas_em = []
    regimes = []

    for Q in Qs:
        try:
            r = compute_running_coupling(Q)
            alphas_s.append(r.alpha_s)
            alphas_em.append(r.alpha_em)
            regimes.append(r.regime)
        except Exception:
            alphas_s.append(np.nan)
            alphas_em.append(np.nan)
            regimes.append("Error")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color background by regime
    regime_spans = []
    current_regime = regimes[0]
    start_idx = 0
    for i, reg in enumerate(regimes):
        if reg != current_regime:
            regime_spans.append((Qs[start_idx], Qs[i - 1], current_regime))
            current_regime = reg
            start_idx = i
    regime_spans.append((Qs[start_idx], Qs[-1], current_regime))

    regime_bg = {"Perturbative": "#1a332a", "Transitional": "#33301a", "NonPerturbative": "#331a1a"}
    for q_start, q_end, reg in regime_spans:
        ax.axvspan(q_start, q_end, alpha=0.3, color=regime_bg.get(reg, "#1a1a1a"))

    # Plot couplings
    ax.semilogy(Qs, alphas_s, color="#e74c3c", linewidth=2.5, label=r"$\alpha_s$ (strong)")
    ax.semilogy(Qs, alphas_em, color="#3498db", linewidth=2.5, label=r"$\alpha_{em}$ (electromagnetic)")

    # Mark key energy scales
    scales = [
        (91.2, r"$M_Z$", "#f39c12"),
        (80.4, r"$M_W$", "#f39c12"),
        (125.3, r"$M_H$", "#2ecc71"),
        (172.7, r"$m_t$", "#e74c3c"),
    ]
    for E, label, color in scales:
        ax.axvline(E, color=color, linestyle=":", alpha=0.6)
        ax.text(E * 1.05, 0.2, label, fontsize=10, color=color, rotation=90, va="top")

    # Mark quark thresholds
    thresholds = [(1.27, "c"), (4.18, "b"), (172.69, "t")]
    for m, _name in thresholds:
        if 5 < m < 15000:
            ax.axvline(m, color="#888", linestyle="--", alpha=0.3)

    ax.set_xlabel("Energy Scale Q (GeV)")
    ax.set_ylabel("Coupling Constant α")
    ax.set_title("Running Coupling Constants: α_s Gets Weaker, α_em Gets Stronger")
    ax.legend(fontsize=12, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_xlim(Qs[0], Qs[-1])
    ax.set_ylim(1e-3, 1)

    # Annotate convergence
    ax.annotate(
        "← Couplings converge\n     at high energy",
        xy=(5000, 0.03),
        fontsize=11,
        color="#58a6ff",
        fontweight="bold",
        ha="center",
    )

    fig.suptitle(
        "INSIGHT 4: Asymptotic Freedom — The Strong Force Weakens at High Energy",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_04_running_couplings.png")
    plt.close(fig)
    print("  [4/10] Running couplings ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Nuclear Binding Energy Curve with Iron Peak
# ══════════════════════════════════════════════════════════════════
def fig5_binding_curve():
    """The most famous curve in nuclear physics, with GCD regime overlay."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get data — filter out H (BE/A = 0)
    elements = [(e.Z, e.symbol, e.A, e.BE_per_A) for e in ELEMENTS if e.BE_per_A > 0]

    [e[0] for e in elements]
    [e[1] for e in elements]
    As = [e[2] for e in elements]
    BEs = [e[3] for e in elements]

    # Color by distance from iron peak (double-sided collapse)
    iron_peak = 8.7945
    distances = [(be - iron_peak) / iron_peak for be in BEs]

    # Color map: green for near peak, red for far
    norm_dist = [abs(d) for d in distances]
    max(norm_dist) if norm_dist else 1
    colors = []
    for d, _be, A in zip(distances, BEs, As, strict=False):
        if abs(d) < 0.01:
            colors.append("#2ecc71")  # At peak
        elif abs(d) < 0.05:
            colors.append("#f39c12")  # Near peak
        elif A < 62:
            colors.append("#3498db")  # Fusion side
        else:
            colors.append("#e74c3c")  # Fission side

    ax.scatter(As, BEs, c=colors, s=40, alpha=0.8, edgecolors="none", zorder=3)

    # Label key elements
    key_elements = {"He": 4, "C": 12, "O": 16, "Fe": 56, "Ni": 62, "U": 238, "Pb": 208}
    for sym, _A_ref in key_elements.items():
        matches = [(z, s, a, be) for z, s, a, be in elements if s == sym]
        if matches:
            _, s, a, be = matches[0]
            ax.annotate(
                f"{s}-{a}",
                (a, be),
                fontsize=10,
                fontweight="bold",
                color="#c9d1d9",
                xytext=(8, 8),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "#8b949e", "lw": 0.8},
            )

    # Iron peak marker
    ax.axhline(iron_peak, color="#2ecc71", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(260, iron_peak + 0.1, f"Fe/Ni peak: {iron_peak} MeV/nucleon", color="#2ecc71", fontsize=10)

    # Fusion/fission arrows
    ax.annotate("", xy=(30, 8.0), xytext=(5, 4.0), arrowprops={"arrowstyle": "->", "color": "#3498db", "lw": 2.5})
    ax.text(8, 5.5, "FUSION →", color="#3498db", fontsize=12, fontweight="bold", rotation=40)

    ax.annotate("", xy=(180, 7.8), xytext=(280, 7.1), arrowprops={"arrowstyle": "->", "color": "#e74c3c", "lw": 2.5})
    ax.text(215, 7.2, "← FISSION", color="#e74c3c", fontsize=12, fontweight="bold", rotation=15)

    # Shell closure annotations
    magic_As = [4, 16, 40, 56, 88, 208]
    magic_labels = ["He-4\n(doubly magic)", "O-16", "Ca-40", "Fe-56\n(iron peak)", "Sr-88", "Pb-208\n(doubly magic)"]
    for a, _label in zip(magic_As, magic_labels, strict=False):
        matches = [(z, s, aa, be) for z, s, aa, be in elements if abs(aa - a) <= 1]
        if matches:
            _, _, aa, be = matches[0]
            ax.plot(aa, be, "D", color="#f39c12", markersize=8, zorder=5, markeredgecolor="white")

    ax.set_xlabel("Mass Number (A)", fontsize=13)
    ax.set_ylabel("Binding Energy per Nucleon (MeV)", fontsize=13)
    ax.set_title("Nuclear Binding Energy Curve — Double-Sided Collapse Toward Iron Peak")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 9.5)

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498db", label="Fusion side (A < 62)"),
        mpatches.Patch(color="#2ecc71", label="Near iron peak"),
        mpatches.Patch(color="#e74c3c", label="Fission/decay side (A > 62)"),
        mpatches.Patch(color="#f39c12", label="Shell closures"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10)

    fig.suptitle(
        "INSIGHT 5: All of Nuclear Physics Converges on Iron — Fusion From Below, Fission From Above",
        fontsize=14,
        fontweight="bold",
        y=1.01,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_05_binding_curve.png")
    plt.close(fig)
    print("  [5/10] Binding energy curve ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6: Yukawa Coupling Hierarchy — Mass Generation
# ══════════════════════════════════════════════════════════════════
def fig6_yukawa_hierarchy():
    """The staggering 6-OOM hierarchy of Higgs couplings."""
    result = compute_higgs_mechanism()
    yuks = result.yukawa_couplings

    # Sort by coupling strength
    sorted_yuks = sorted(yuks.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in sorted_yuks]
    values = [v for _, v in sorted_yuks]
    masses = [FERMION_MASSES[k] for k in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Yukawa couplings (log scale)
    colors_yuk = []
    for n in names:
        if n in {"up", "down", "charm", "strange", "top", "bottom"}:
            colors_yuk.append("#e74c3c")
        else:
            colors_yuk.append("#3498db")

    ax1.barh(range(len(names)), values, color=colors_yuk, alpha=0.85, height=0.7)
    ax1.set_xscale("log")
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([n.capitalize() for n in names], fontsize=11)
    ax1.set_xlabel("Yukawa Coupling Strength (y_f = √2 · m_f / v)")
    ax1.set_title("The Yukawa Hierarchy: Who Talks to the Higgs?")

    # Annotate values
    for i, (_n, v) in enumerate(zip(names, values, strict=False)):
        ax1.text(v * 1.5, i, f"{v:.2e}", va="center", fontsize=9, color="#c9d1d9")

    ax1.grid(axis="x", alpha=0.3)

    # Annotate the ratio
    ratio = values[0] / values[-1]
    ax1.text(
        1e-3,
        len(names) // 2,
        f"Top/electron ratio:\n{ratio:,.0f}×",
        fontsize=14,
        fontweight="bold",
        color="#f39c12",
        ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#21262d", "edgecolor": "#f39c12"},
    )

    # Right: Mass vs Yukawa (showing perfect linearity)
    ax2.scatter(masses, values, s=150, zorder=5, edgecolors="white", linewidth=0.5)
    for name, m, y in zip(names, masses, values, strict=False):
        cat_col = "#e74c3c" if name in {"up", "down", "charm", "strange", "top", "bottom"} else "#3498db"
        ax2.scatter(m, y, c=cat_col, s=150, zorder=5, edgecolors="white", linewidth=0.5)
        ax2.annotate(name.capitalize(), (m, y), fontsize=9, ha="left", xytext=(8, 3), textcoords="offset points")

    # Perfect linear relation line
    m_line = np.logspace(-4, 3, 100)
    y_line = np.sqrt(2) * m_line / 246.22
    ax2.loglog(m_line, y_line, "--", color="#888", alpha=0.5, label=r"$y_f = \sqrt{2} \cdot m_f / v$")

    ax2.set_xlabel("Fermion Mass (GeV)")
    ax2.set_ylabel("Yukawa Coupling")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Mass = Yukawa × VEV: Perfect Linearity")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    handles = [mpatches.Patch(color="#e74c3c", label="Quarks"), mpatches.Patch(color="#3498db", label="Leptons")]
    ax2.legend(handles=handles, fontsize=10, loc="upper left")

    fig.suptitle(
        "INSIGHT 6: The Yukawa Hierarchy Spans 6 Orders of Magnitude — Why?",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_06_yukawa_hierarchy.png")
    plt.close(fig)
    print("  [6/10] Yukawa hierarchy ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7: Wave Spectrum — 24 Systems Spanning 23 Orders of Magnitude
# ══════════════════════════════════════════════════════════════════
def fig7_wave_spectrum():
    """The full wave universe from gravitational waves to gamma rays."""
    waves = compute_all_wave_systems()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [1.5, 1]})

    # Top: Frequency spectrum
    waves_sorted = sorted(waves, key=lambda w: -w.F)
    names = [w.system for w in waves_sorted]
    types = [w.wave_type for w in waves_sorted]
    Fs = [w.F for w in waves_sorted]
    ICs = [w.IC for w in waves_sorted]
    gaps = [w.gap for w in waves_sorted]
    regimes = [w.regime for w in waves_sorted]

    x = np.arange(len(names))
    colors = [WAVE_COLORS.get(t, "#888") for t in types]

    ax1.bar(x, Fs, color=colors, alpha=0.85, width=0.7, label="F")
    ax1.bar(x, ICs, color=colors, alpha=0.4, width=0.5, label="IC")

    # Regime markers
    for i, reg in enumerate(regimes):
        marker_color = REGIME_COLORS.get(reg, "#888")
        ax1.plot(i, Fs[i] + 0.02, "v", color=marker_color, markersize=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=65, ha="right", fontsize=8)
    ax1.set_ylabel("Kernel Invariants")
    ax1.set_title("Fidelity (solid) and Integrity (faded) Across 24 Wave Systems")
    ax1.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=c, label=k) for k, c in WAVE_COLORS.items()]
    regime_handles = [mpatches.Patch(color=c, label=f"▼ {k}") for k, c in REGIME_COLORS.items()]
    ax1.legend(handles=handles + regime_handles, loc="upper right", fontsize=8, ncol=3)

    # Bottom: Heterogeneity gap
    ax2.bar(x, gaps, color=colors, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=65, ha="right", fontsize=8)
    ax2.set_ylabel("Heterogeneity Gap (Δ = F − IC)")
    ax2.set_title("Channel Heterogeneity: Where Does Integrity Break Down?")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "INSIGHT 7: From Gravitational Waves to Gamma Rays — The Universal Kernel Sees All Waves",
        fontsize=14,
        fontweight="bold",
        y=1.01,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_07_wave_spectrum.png")
    plt.close(fig)
    print("  [7/10] Wave spectrum ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 8: Periodic Table Kernel Heatmap
# ══════════════════════════════════════════════════════════════════
def fig8_periodic_heatmap():
    """118 elements through the GCD kernel — periodic patterns emerge."""
    results = batch_compute_all()

    # Build periodic table layout
    # Standard periodic table positions: (period, group) → element
    # We'll plot F and IC as heatmaps

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    for ax_idx, (metric, label, _cmap) in enumerate(
        [
            ("F", "Fidelity (F)", "YlOrRd"),
            ("IC", "Integrity Composite (IC)", "YlGnBu"),
        ]
    ):
        ax = axes[ax_idx]
        vals = [getattr(r, metric) for r in results]
        [r.symbol for r in results]
        [r.Z for r in results]
        blocks = [r.block for r in results]

        # Simple bar plot colored by block
        x = np.arange(len(results))
        block_colors = [BLOCK_COLORS.get(b, "#888") for b in blocks]
        ax.bar(x, vals, color=block_colors, alpha=0.8, width=1.0, edgecolor="none")

        # Mark noble gases
        noble_gas_Z = [2, 10, 18, 36, 54, 86]
        for ng_z in noble_gas_Z:
            idx = next((i for i, r in enumerate(results) if ng_z == r.Z), None)
            if idx is not None:
                ax.axvline(idx, color="#2ecc71", linewidth=1.5, alpha=0.5, linestyle="--")
                ax.text(
                    idx,
                    vals[idx] + 0.01,
                    results[idx].symbol,
                    fontsize=8,
                    ha="center",
                    color="#2ecc71",
                    fontweight="bold",
                )

        # Mark every 10th element
        for i, r in enumerate(results):
            if r.Z % 10 == 0:
                ax.text(i, -0.03, r.symbol, fontsize=7, ha="center", rotation=90, color="#8b949e")

        ax.set_ylabel(label)
        ax.set_xlim(-1, len(results))
        ax.grid(axis="y", alpha=0.3)

        if ax_idx == 0:
            ax.set_title("Fidelity Across the Periodic Table — Noble Gas Peaks Visible")
        else:
            ax.set_xlabel("Atomic Number (Z)")
            ax.set_title("Integrity Shows the Heterogeneity Penalty — One Bad Channel Kills IC")

    handles = [mpatches.Patch(color=c, label=f"{k}-block") for k, c in BLOCK_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="#2ecc71", linestyle="--", label="Noble gases"))
    axes[0].legend(handles=handles, loc="upper right", fontsize=9, ncol=5)

    fig.suptitle(
        "INSIGHT 8: The Periodic Table Through the Kernel — Block Structure and Noble Gas Peaks Emerge",
        fontsize=14,
        fontweight="bold",
        y=1.01,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_08_periodic_kernel.png")
    plt.close(fig)
    print("  [8/10] Periodic kernel heatmap ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 9: Mass-Kernel Log Mapping (T4)
# ══════════════════════════════════════════════════════════════════
def fig9_mass_kernel_mapping():
    """T4: 13 orders of magnitude in mass compress to F ∈ [0.37, 0.73]."""
    all_results = compute_all()

    fig, ax = plt.subplots(figsize=(14, 8))

    for r in all_results:
        color = CAT_COLORS.get(r.category, "#888")
        mass = r.mass_GeV
        if mass <= 0:
            mass = 1e-11  # Neutrino floor

        ax.scatter(mass, r.F, c=color, s=130, alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(
            r.symbol, (mass, r.F), fontsize=8, ha="left", xytext=(6, 4), textcoords="offset points", color="#c9d1d9"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mass (GeV/c²)", fontsize=13)
    ax.set_ylabel("Fidelity (F)", fontsize=13)

    # Show the compression factor
    ax.axhline(0.37, color="#888", linestyle=":", alpha=0.4)
    ax.axhline(0.73, color="#888", linestyle=":", alpha=0.4)
    ax.fill_between([1e-11, 200], 0.37, 0.73, alpha=0.05, color="#58a6ff")
    ax.text(
        1e-5,
        0.75,
        "F ∈ [0.37, 0.73]  ← 13 OOM compressed to 0.36 range",
        fontsize=11,
        color="#58a6ff",
        fontweight="bold",
    )

    # Mass scale annotations
    ax.axvline(0.000511, color="#3498db", linestyle=":", alpha=0.3)
    ax.text(0.000511, 0.3, "e⁻", fontsize=9, color="#3498db")
    ax.axvline(0.938, color="#9b59b6", linestyle=":", alpha=0.3)
    ax.text(0.938, 0.3, "p", fontsize=9, color="#9b59b6")
    ax.axvline(172.69, color="#e74c3c", linestyle=":", alpha=0.3)
    ax.text(172.69, 0.3, "t", fontsize=9, color="#e74c3c")

    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1e-11, 500)
    ax.set_ylim(0.25, 0.85)

    fig.suptitle(
        "INSIGHT 9: 13 Orders of Magnitude in Mass Map to a Narrow Band of Fidelity",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_09_mass_kernel_mapping.png")
    plt.close(fig)
    print("  [9/10] Mass-kernel mapping ✓")


# ══════════════════════════════════════════════════════════════════
# FIGURE 10: The Scale Ladder — Cross-Scale Universality (T6)
# ══════════════════════════════════════════════════════════════════
def fig10_scale_ladder():
    """T6: The kernel speaks the same language at every scale."""
    # Compute averages at each scale
    fund = compute_all_fundamental()
    comp = compute_all_composite()
    periodic = batch_compute_all()
    waves = compute_all_wave_systems()

    scales = {
        "Quarks": ([r for r in fund if r.category == "Quark"], "#e74c3c"),
        "Leptons": ([r for r in fund if r.category == "Lepton"], "#3498db"),
        "Gauge Bosons": ([r for r in fund if r.category in ("GaugeBoson", "ScalarBoson")], "#f39c12"),
        "Hadrons": (comp, "#9b59b6"),
        "Elements\n(118)": (periodic, "#2ecc71"),
        "Wave Systems\n(24)": (waves, "#e67e22"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    scale_names = list(scales.keys())
    colors_list = [scales[k][1] for k in scale_names]
    x = np.arange(len(scale_names))

    # Compute averages
    avg_F = []
    avg_IC = []
    avg_gap = []
    for name in scale_names:
        data, _ = scales[name]
        Fs = [r.F for r in data]
        ICs = [r.IC for r in data]
        gaps = [r.F - r.IC for r in data]
        avg_F.append(np.mean(Fs))
        avg_IC.append(np.mean(ICs))
        avg_gap.append(np.mean(gaps))

    # Left: Average F
    axes[0].bar(x, avg_F, color=colors_list, alpha=0.85, width=0.6)
    axes[0].set_ylabel("⟨F⟩ (Average Fidelity)")
    axes[0].set_title("Fidelity Across Scales")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scale_names, fontsize=9, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(avg_F):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Middle: Average IC
    axes[1].bar(x, avg_IC, color=colors_list, alpha=0.85, width=0.6)
    axes[1].set_ylabel("⟨IC⟩ (Average Integrity)")
    axes[1].set_title("Integrity Across Scales")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scale_names, fontsize=9, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(avg_IC):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Right: Average heterogeneity gap
    axes[2].bar(x, avg_gap, color=colors_list, alpha=0.85, width=0.6)
    axes[2].set_ylabel("⟨Δ⟩ = ⟨F − IC⟩ (Heterogeneity Gap)")
    axes[2].set_title("Heterogeneity Gap Across Scales")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scale_names, fontsize=9, rotation=20, ha="right")
    axes[2].grid(axis="y", alpha=0.3)
    for i, v in enumerate(avg_gap):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "INSIGHT 10: The Scale Ladder — Same Kernel, Same Invariants, Every Scale of Physics",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#58a6ff",
    )
    plt.tight_layout()
    fig.savefig(OUT / "physics_viz_10_scale_ladder.png")
    plt.close(fig)
    print("  [10/10] Scale ladder ✓")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    """Generate all 10 physics visualizations."""
    _style()
    print(f"Generating physics visualizations → {OUT}/")
    print("=" * 60)

    fig1_confinement_cliff()
    fig2_fermion_boson()
    fig3_generation_staircase()
    fig4_running_couplings()
    fig5_binding_curve()
    fig6_yukawa_hierarchy()
    fig7_wave_spectrum()
    fig8_periodic_heatmap()
    fig9_mass_kernel_mapping()
    fig10_scale_ladder()

    print("=" * 60)
    print(f"✓ All 10 figures saved to {OUT}/")
    print("\nGenerated files:")
    for f in sorted(OUT.glob("physics_viz_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
