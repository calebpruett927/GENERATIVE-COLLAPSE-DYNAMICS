#!/usr/bin/env python3
"""Generate publication-quality figures for the confinement paper.

Produces 4 figures:
  1. confinement_cliff.pdf  — The IC cliff: quarks vs hadrons vs exotics
  2. channel_death.pdf      — Trace vector heatmap showing channel death
  3. epsilon_robustness.pdf — Guard band sensitivity plot
  4. gap_amplification.pdf  — F vs IC scatter showing the gap blowup

All figures use the paper's CanonLink (#337799) colour palette.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Colour palette (matches LaTeX preamble) ──────────────────
CANON = "#337799"
CLIFF_RED = "#CC3333"
WELD_GREEN = "#2E8B57"
EPS_BLUE = "#4477AA"
TIER_GOLD = "#B8860B"
HUD_GRAY = "#888888"
BG_CREAM = "#FAFAF5"

# ── Data ─────────────────────────────────────────────────────
quarks = [
    ("u", 0.595, 0.561, 0.034),
    ("d", 0.556, 0.517, 0.040),
    ("c", 0.662, 0.634, 0.029),
    ("s", 0.610, 0.573, 0.037),
    ("t", 0.638, 0.589, 0.049),
    ("b", 0.667, 0.615, 0.052),
]

hadrons = [
    ("p", 0.550, 0.0204, "B", 2),
    ("n", 0.395, 0.0035, "B", 3),
    ("Λ⁰", 0.407, 0.0153, "B", 2),
    ("Σ⁺", 0.526, 0.0229, "B", 2),
    ("Ξ⁰", 0.445, 0.0045, "B", 2),
    ("Ω⁻", 0.674, 0.0287, "B", 2),
    ("Λc⁺", 0.544, 0.0239, "B", 2),
    ("π⁺", 0.476, 0.0047, "M", 3),
    ("π⁰", 0.334, 0.0008, "M", 4),
    ("K⁺", 0.473, 0.0212, "M", 1),
    ("K⁰", 0.349, 0.0038, "M", 2),
    ("J/ψ", 0.364, 0.0020, "M", 4),
    ("Υ", 0.369, 0.0008, "M", 4),
    ("D⁰", 0.317, 0.0025, "M", 2),
]

exotics = [
    ("Pc(4312)⁺", 0.526, 0.0228, "P", 2),
    ("Pc(4440)⁺", 0.526, 0.0228, "P", 2),
    ("Pc(4457)⁺", 0.589, 0.0249, "P", 2),
    ("Tcc⁺", 0.651, 0.0271, "T", 2),
    ("X(3872)", 0.401, 0.0009, "T", 4),
    ("Zc(3900)⁺", 0.526, 0.0048, "T", 3),
]

# Trace vector data for heatmap
trace_channels = ["mass", "charge", "spin", "ch.4*", "ch.5*", "ch.6*", "ch.7*", "ch.8*"]
trace_particles = ["u (quark)", "p (baryon)", "π⁰ (meson)", "J/ψ (meson)"]
trace_matrix = np.array(
    [
        [0.627, 0.667, 0.500, 0.631, 0.667, 0.333, 0.333, 1.000],  # u
        [0.825, 1.000, 0.500, 1.000, 1e-6, 1e-6, 1.000, 0.073],  # p
        [0.762, 1e-6, 1e-6, 0.667, 1e-6, 1e-6, 0.446, 0.799],  # π⁰
        [0.864, 1e-6, 1.000, 0.667, 1e-6, 1e-6, 0.380, 1e-3],  # J/ψ
    ]
)

# Epsilon sensitivity data
eps_data = [
    (1e-2, 0.580, 0.155, 73.3),
    (1e-3, 0.581, 0.077, 86.8),
    (1e-4, 0.581, 0.040, 93.2),
    (1e-5, 0.581, 0.021, 96.4),
    (1e-6, 0.581, 0.011, 98.1),
    (1e-7, 0.581, 0.011, 98.1),
    (1e-8, 0.581, 0.011, 98.1),
    (1e-10, 0.581, 0.011, 98.1),
    (1e-12, 0.581, 0.011, 98.1),
]

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def style_ax(ax: plt.Axes) -> None:
    """Apply consistent styling."""
    ax.set_facecolor(BG_CREAM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HUD_GRAY)
    ax.spines["bottom"].set_color(HUD_GRAY)
    ax.tick_params(colors=HUD_GRAY, labelsize=9)


# ══════════════════════════════════════════════════════════════
# FIGURE 1: The Confinement Cliff
# ══════════════════════════════════════════════════════════════
def fig_confinement_cliff() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    style_ax(ax)

    # Collect all IC values with categories
    q_names = [q[0] for q in quarks]
    q_ic = [q[2] for q in quarks]

    h_names = [h[0] for h in hadrons]
    h_ic = [h[2] for h in hadrons]

    e_names = [e[0] for e in exotics]
    e_ic = [e[2] for e in exotics]

    # Sort each group by IC descending
    q_sorted = sorted(zip(q_names, q_ic, strict=False), key=lambda x: -x[1])
    h_sorted = sorted(zip(h_names, h_ic, strict=False), key=lambda x: -x[1])
    e_sorted = sorted(zip(e_names, e_ic, strict=False), key=lambda x: -x[1])

    # Positions
    gap1, gap2 = 1.5, 1.0
    q_x = np.arange(len(q_sorted))
    h_x = np.arange(len(h_sorted)) + len(q_sorted) + gap1
    e_x = np.arange(len(e_sorted)) + len(q_sorted) + gap1 + len(h_sorted) + gap2

    all_x = np.concatenate([q_x, h_x, e_x])
    all_names = [n for n, _ in q_sorted] + [n for n, _ in h_sorted] + [n for n, _ in e_sorted]

    # Plot bars
    ax.bar(
        q_x,
        [ic for _, ic in q_sorted],
        color=CANON,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        label="Quarks",
        zorder=3,
    )
    ax.bar(
        h_x,
        [ic for _, ic in h_sorted],
        color=CLIFF_RED,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Hadrons",
        zorder=3,
    )
    ax.bar(
        e_x,
        [ic for _, ic in e_sorted],
        color=TIER_GOLD,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Exotics (LHCb)",
        zorder=3,
    )

    # Min quark IC line
    min_q_ic = min(q_ic)
    ax.axhline(y=min_q_ic, color=CANON, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(
        len(q_sorted) + gap1 + len(h_sorted) / 2,
        min_q_ic + 0.015,
        f"IC$_q^{{\\mathrm{{min}}}}$ = {min_q_ic:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=CANON,
        fontstyle="italic",
    )

    # Mean lines
    mean_q = np.mean(q_ic)
    mean_h = np.mean(h_ic)
    ax.axhline(y=mean_q, color=CANON, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=mean_h, color=CLIFF_RED, linestyle=":", linewidth=0.8, alpha=0.5)

    # Cliff annotation
    mid_x = len(q_sorted) + gap1 / 2 - 0.5
    ax.annotate(
        "", xy=(mid_x, mean_h), xytext=(mid_x, mean_q), arrowprops={"arrowstyle": "<->", "color": CLIFF_RED, "lw": 2}
    )
    ax.text(
        mid_x + 0.3,
        (mean_q + mean_h) / 2,
        "98.1%\ndrop",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=CLIFF_RED,
    )

    # Vertical separator
    sep_x = len(q_sorted) + gap1 / 2 - 0.75
    ax.axvline(x=sep_x, color=HUD_GRAY, linestyle="-", linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_names, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Integrity Composite (IC)", fontsize=11, color="#333333")
    ax.set_title(
        "The Confinement Cliff: IC Collapse at the Quark → Hadron Boundary",
        fontsize=13,
        fontweight="bold",
        color="#222222",
        pad=12,
    )

    # Log scale for clarity (the cliff is 2 OOM)
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.0)
    ax.set_xlim(-0.7, max(e_x) + 0.7)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, edgecolor=HUD_GRAY)
    ax.grid(axis="y", alpha=0.2, color=HUD_GRAY, zorder=1)

    fig.tight_layout()
    fig.savefig(OUT / "confinement_cliff.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "confinement_cliff.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ confinement_cliff.pdf/png")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Channel Death Heatmap
# ══════════════════════════════════════════════════════════════
def fig_channel_death() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    # Use log scale for the colour to reveal the dead channels
    log_matrix = np.log10(trace_matrix + 1e-8)

    # Custom colourmap: dead channels in red, live channels in blue-green
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "channel_death",
        [(0.0, CLIFF_RED), (0.35, "#FFCC66"), (0.6, "#88CC88"), (1.0, CANON)],
    )

    im = ax.imshow(log_matrix, cmap=cmap, aspect="auto", vmin=-6.5, vmax=0)

    # Annotate each cell with the actual value
    for i in range(trace_matrix.shape[0]):
        for j in range(trace_matrix.shape[1]):
            val = trace_matrix[i, j]
            if val < 0.001:
                txt = "ε"
                clr = "white"
                fw = "bold"
            else:
                txt = f"{val:.3f}"
                clr = "white" if val < 0.2 else "#222222"
                fw = "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=clr, fontweight=fw)

    # IC values on the right
    ic_vals = [0.561, 0.020, 0.0008, 0.002]
    for i, ic in enumerate(ic_vals):
        ax.text(
            8.3,
            i,
            f"IC={ic:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            color=CLIFF_RED if ic < 0.1 else CANON,
            fontweight="bold",
        )

    ax.set_xticks(range(8))
    ax.set_xticklabels(trace_channels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(trace_particles, fontsize=10)

    ax.set_title(
        "Channel Death: Trace Vectors Across the Confinement Boundary",
        fontsize=12,
        fontweight="bold",
        color="#222222",
        pad=10,
    )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label("log₁₀(channel value)", fontsize=9, color=HUD_GRAY)
    cbar.ax.tick_params(labelsize=8, colors=HUD_GRAY)

    # Mark dead channels with X overlay
    for i in range(trace_matrix.shape[0]):
        for j in range(trace_matrix.shape[1]):
            if trace_matrix[i, j] < 0.001:
                ax.plot(j, i, marker="x", markersize=18, color="white", markeredgewidth=2.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "channel_death.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "channel_death.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ channel_death.pdf/png")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Guard Band Robustness
# ══════════════════════════════════════════════════════════════
def fig_epsilon_robustness() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    style_ax(ax1)
    style_ax(ax2)

    eps_vals = [d[0] for d in eps_data]
    ic_q = [d[1] for d in eps_data]
    ic_h = [d[2] for d in eps_data]
    drop = [d[3] for d in eps_data]

    # LEFT: IC_quark and IC_hadron vs epsilon
    ax1.semilogx(eps_vals, ic_q, "o-", color=CANON, linewidth=2, markersize=7, label="⟨IC⟩ quarks", zorder=3)
    ax1.semilogx(eps_vals, ic_h, "s-", color=CLIFF_RED, linewidth=2, markersize=7, label="⟨IC⟩ hadrons", zorder=3)

    # Shade the gap
    ax1.fill_between(eps_vals, ic_h, ic_q, alpha=0.12, color=CLIFF_RED)

    ax1.set_xlabel("Guard band ε", fontsize=10, color="#333333")
    ax1.set_ylabel("Average IC", fontsize=10, color="#333333")
    ax1.set_title("IC Values vs Guard Band", fontsize=11, fontweight="bold", color="#222222")
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor=HUD_GRAY)
    ax1.set_ylim(-0.02, 0.7)
    ax1.grid(alpha=0.2, color=HUD_GRAY)
    ax1.invert_xaxis()

    # Annotation for saturation
    ax1.axvspan(1e-12, 1e-6, alpha=0.06, color=WELD_GREEN)
    ax1.text(
        1e-9, 0.35, "saturation\nregion", ha="center", va="center", fontsize=8, color=WELD_GREEN, fontstyle="italic"
    )

    # RIGHT: Drop percentage vs epsilon
    ax2.semilogx(eps_vals, drop, "D-", color=TIER_GOLD, linewidth=2.5, markersize=8, zorder=3)

    ax2.axhline(y=98.1, color=WELD_GREEN, linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(1e-1, 98.8, "98.1% (saturated)", fontsize=8, color=WELD_GREEN, ha="right")

    # Annotation: 14/14 at all epsilon
    ax2.fill_between(eps_vals, 70, drop, alpha=0.1, color=TIER_GOLD)
    ax2.text(
        1e-7,
        77,
        "14/14 bound holds\nat all ε values",
        ha="center",
        va="center",
        fontsize=9,
        color=TIER_GOLD,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": TIER_GOLD, "alpha": 0.9},
    )

    ax2.set_xlabel("Guard band ε", fontsize=10, color="#333333")
    ax2.set_ylabel("IC Drop (%)", fontsize=10, color="#333333")
    ax2.set_title("Cliff Depth vs Guard Band", fontsize=11, fontweight="bold", color="#222222")
    ax2.set_ylim(68, 102)
    ax2.grid(alpha=0.2, color=HUD_GRAY)
    ax2.invert_xaxis()

    fig.suptitle(
        "Guard Band Robustness: The Cliff Survives Across 9 Orders of Magnitude",
        fontsize=12,
        fontweight="bold",
        color="#222222",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "epsilon_robustness.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "epsilon_robustness.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ epsilon_robustness.pdf/png")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: F vs IC Scatter — Gap Amplification
# ══════════════════════════════════════════════════════════════
def fig_gap_amplification() -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    style_ax(ax)

    # Quarks: F vs IC
    q_F = [q[1] for q in quarks]
    q_IC = [q[2] for q in quarks]
    q_names = [q[0] for q in quarks]

    # Hadrons
    h_F = [h[1] for h in hadrons]
    h_IC = [h[2] for h in hadrons]
    [h[0] for h in hadrons]

    # Exotics
    e_F = [e[1] for e in exotics]
    e_IC = [e[2] for e in exotics]
    [e[0] for e in exotics]

    # IC = F line (zero gap)
    f_line = np.linspace(0, 0.75, 100)
    ax.plot(f_line, f_line, "-", color=HUD_GRAY, linewidth=1, alpha=0.4, label="IC = F (zero gap)", zorder=1)

    # Plot points
    ax.scatter(q_F, q_IC, c=CANON, s=120, marker="o", zorder=4, edgecolors="white", linewidth=0.8, label="Quarks")
    ax.scatter(h_F, h_IC, c=CLIFF_RED, s=90, marker="s", zorder=4, edgecolors="white", linewidth=0.8, label="Hadrons")
    ax.scatter(
        e_F, e_IC, c=TIER_GOLD, s=100, marker="D", zorder=4, edgecolors="white", linewidth=0.8, label="Exotics (LHCb)"
    )

    # Label quarks
    for name, f, ic in zip(q_names, q_F, q_IC, strict=False):
        ax.annotate(name, (f, ic), textcoords="offset points", xytext=(6, 4), fontsize=8, color=CANON)

    # Label selected hadrons
    for name, f, ic, _, _ in hadrons:
        if name in ("p", "π⁰", "J/ψ", "Ω⁻", "n"):
            ax.annotate(name, (f, ic), textcoords="offset points", xytext=(6, -8), fontsize=7.5, color=CLIFF_RED)

    # Gap arrows for one quark and one hadron
    # up quark: show small gap
    ax.annotate(
        "",
        xy=(0.595, 0.561),
        xytext=(0.595, 0.595),
        arrowprops={"arrowstyle": "-", "color": CANON, "lw": 1.5, "ls": "--"},
    )
    ax.text(0.600, 0.578, "Δ=0.034", fontsize=7, color=CANON, rotation=90)

    # proton: show large gap
    ax.annotate(
        "",
        xy=(0.550, 0.0204),
        xytext=(0.550, 0.550),
        arrowprops={"arrowstyle": "-", "color": CLIFF_RED, "lw": 1.5, "ls": "--"},
    )
    ax.text(0.555, 0.20, "Δ=0.529", fontsize=7.5, color=CLIFF_RED, rotation=90, fontweight="bold")

    # Horizontal line at min quark IC
    ax.axhline(y=0.517, color=CANON, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.32, 0.53, "IC$_q^{\\mathrm{min}}$ = 0.517", fontsize=8, color=CANON, fontstyle="italic")

    ax.set_xlabel("Fidelity (F)", fontsize=11, color="#333333")
    ax.set_ylabel("Integrity Composite (IC)", fontsize=11, color="#333333")
    ax.set_title(
        "Gap Amplification: Δ = F − IC Explodes Upon Binding", fontsize=12, fontweight="bold", color="#222222", pad=10
    )

    ax.set_xlim(0.28, 0.72)
    ax.set_ylim(-0.02, 0.72)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor=HUD_GRAY)
    ax.grid(alpha=0.15, color=HUD_GRAY)

    # 10.8x annotation
    ax.text(
        0.42,
        0.08,
        "⟨Δ⟩ amplifies 10.8×\nupon binding",
        fontsize=10,
        fontweight="bold",
        color=CLIFF_RED,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": CLIFF_RED, "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(OUT / "gap_amplification.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "gap_amplification.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ gap_amplification.pdf/png")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    fig_confinement_cliff()
    fig_channel_death()
    fig_epsilon_robustness()
    fig_gap_amplification()
    print(f"\n✓ All 4 figures generated in {OUT}/")
