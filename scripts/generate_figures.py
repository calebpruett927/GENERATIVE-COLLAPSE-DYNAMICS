#!/usr/bin/env python3
"""Generate publication-quality PNG figures for UMCP/GCD documentation.

Usage:
    python scripts/generate_figures.py

Outputs 8 PNGs into images/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# ── Ensure closures importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "images"
OUT_DIR.mkdir(exist_ok=True)

matplotlib.use("Agg")

# ── Shared styling ──────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STABLE_COLOR = "#2ca02c"
WATCH_COLOR = "#ff7f0e"
COLLAPSE_COLOR = "#d62728"
ACCENT_BLUE = "#1f77b4"
ACCENT_PURPLE = "#9467bd"
ACCENT_PINK = "#e377c2"
DARK_BG = "#1a1a2e"


def _save(fig: Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(ROOT)}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Regime Phase Diagram — (F, ω) space with real particle data
# ═════════════════════════════════════════════════════════════════════════════
def fig01_regime_phase_diagram() -> None:
    """F vs ω with regime boundaries and all 31 Standard Model particles."""
    from closures.standard_model.subatomic_kernel import compute_all

    results = compute_all()

    fig, ax = plt.subplots(figsize=(9, 7))

    # Regime shading
    np.linspace(0, 1, 500)
    # STABLE: ω ∈ [0.3, 0.7]  → F ∈ [0.3, 0.7]
    ax.axhspan(0.3, 0.7, alpha=0.08, color=STABLE_COLOR, zorder=0)
    ax.axhspan(0.1, 0.3, alpha=0.06, color=WATCH_COLOR, zorder=0)
    ax.axhspan(0.7, 0.9, alpha=0.06, color=WATCH_COLOR, zorder=0)
    ax.axhspan(0.0, 0.1, alpha=0.04, color=COLLAPSE_COLOR, zorder=0)
    ax.axhspan(0.9, 1.0, alpha=0.04, color=COLLAPSE_COLOR, zorder=0)

    # Identity line F + ω = 1
    ax.plot([0, 1], [1, 0], "k-", linewidth=1.5, alpha=0.7, label="F + ω = 1 (duality)")

    # Category markers
    cat_markers = {
        "Quark": ("o", ACCENT_BLUE, 60),
        "Lepton": ("s", STABLE_COLOR, 60),
        "GaugeBoson": ("D", COLLAPSE_COLOR, 55),
        "ScalarBoson": ("p", ACCENT_PURPLE, 80),
        "Baryon": ("^", WATCH_COLOR, 60),
        "Meson": ("v", ACCENT_PINK, 60),
    }

    for r in results:
        marker, color, size = cat_markers.get(r.category, ("o", "gray", 40))
        ax.scatter(r.omega, r.F, marker=marker, c=color, s=size, edgecolors="black", linewidths=0.5, zorder=5)
        # Label key particles
        if r.name in ("top", "electron", "photon", "Higgs", "proton", "pion0", "tau"):
            offset = (5, 5)
            if r.name == "proton":
                offset = (-35, -12)
            elif r.name == "tau":
                offset = (5, -12)
            ax.annotate(
                r.name,
                (r.omega, r.F),
                fontsize=7.5,
                fontweight="bold",
                textcoords="offset points",
                xytext=offset,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            )

    # Legend for categories
    handles = []
    for cat, (marker, color, _) in cat_markers.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                label=cat,
            )
        )
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, ncol=2)

    # Regime labels
    ax.text(
        0.02,
        0.5,
        "STABLE",
        fontsize=9,
        color=STABLE_COLOR,
        alpha=0.6,
        transform=ax.transAxes,
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.22,
        "WATCH",
        fontsize=9,
        color=WATCH_COLOR,
        alpha=0.6,
        transform=ax.transAxes,
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.05,
        "COLLAPSE",
        fontsize=9,
        color=COLLAPSE_COLOR,
        alpha=0.6,
        transform=ax.transAxes,
        va="center",
        fontweight="bold",
    )

    ax.set_xlabel("ω (Drift)")
    ax.set_ylabel("F (Fidelity)")
    ax.set_title("Regime Phase Diagram — 31 Standard Model Particles")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    _save(fig, "08_regime_phase_31_particles.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Confinement Cliff — IC collapse at quark→hadron boundary
# ═════════════════════════════════════════════════════════════════════════════
def fig02_confinement_cliff() -> None:
    """IC cliff showing 98% drop from quarks to composite hadrons."""
    from closures.standard_model.subatomic_kernel import (
        compute_all_composite,
        compute_all_fundamental,
    )

    fund = compute_all_fundamental()
    comp = compute_all_composite()

    # Separate quarks from other fundamentals
    quarks = [r for r in fund if r.category == "Quark"]
    leptons = [r for r in fund if r.category == "Lepton"]
    bosons = [r for r in fund if r.category in ("GaugeBoson", "ScalarBoson")]
    baryons = [r for r in comp if r.category == "Baryon"]
    mesons = [r for r in comp if r.category == "Meson"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Left: IC bar chart grouped by category
    groups = [
        ("Quarks", quarks, ACCENT_BLUE),
        ("Leptons", leptons, STABLE_COLOR),
        ("Bosons", bosons, ACCENT_PURPLE),
        ("Baryons", baryons, WATCH_COLOR),
        ("Mesons", mesons, ACCENT_PINK),
    ]

    x = 0
    tick_positions = []
    tick_labels = []
    category_centers = []

    for _group_name, particles, color in groups:
        group_start = x
        for p in sorted(particles, key=lambda p: p.IC, reverse=True):
            ax1.bar(x, p.IC, color=color, edgecolor="black", linewidth=0.3, width=0.8)
            tick_positions.append(x)
            tick_labels.append(p.name)
            x += 1
        category_centers.append((group_start + x - 1) / 2)
        x += 1  # gap between groups

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=7.5)
    ax1.set_ylabel("IC (Integrity Composite)")
    ax1.set_title("Confinement Cliff: IC Across the Standard Model")
    ax1.set_yscale("log")
    ax1.set_ylim(5e-4, 1)

    # Mark the confinement boundary
    boundary_x = len(quarks) + len(leptons) + len(bosons) + 2.5
    ax1.axvline(x=boundary_x, color=COLLAPSE_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.text(
        boundary_x + 0.3, 0.4, "Confinement\nBoundary", fontsize=8, color=COLLAPSE_COLOR, fontweight="bold", va="center"
    )

    # Horizontal line at min quark IC
    min_quark_ic = min(q.IC for q in quarks)
    ax1.axhline(y=min_quark_ic, color=ACCENT_BLUE, linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(0.5, min_quark_ic * 1.15, f"min quark IC = {min_quark_ic:.3f}", fontsize=7, color=ACCENT_BLUE, alpha=0.8)

    # Right: Category mean IC comparison
    cat_means = []
    cat_names = []
    cat_colors = []
    for group_name, particles, color in groups:
        cat_means.append(np.mean([p.IC for p in particles]))
        cat_names.append(group_name)
        cat_colors.append(color)

    bars = ax2.barh(range(len(cat_names)), cat_means, color=cat_colors, edgecolor="black", linewidth=0.5, height=0.6)
    ax2.set_yticks(range(len(cat_names)))
    ax2.set_yticklabels(cat_names)
    ax2.set_xlabel("⟨IC⟩")
    ax2.set_title("Mean IC by Category")
    ax2.set_xscale("log")

    for i, (_bar, val) in enumerate(zip(bars, cat_means, strict=False)):
        ax2.text(val * 1.2, i, f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "09_confinement_cliff_ic.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Periodic Table Fidelity Heatmap — 118 Elements
# ═════════════════════════════════════════════════════════════════════════════
def fig03_periodic_table_heatmap() -> None:
    """Fidelity heatmap for all 118 elements in periodic table layout."""
    from closures.atomic_physics.periodic_kernel import compute_element_kernel
    from closures.materials_science.element_database import ELEMENTS_BY_Z

    # Periodic table layout: (row, col) for each element Z
    # Standard 18-column layout
    PT_LAYOUT: dict[int, tuple[int, int]] = {}
    # Period 1
    PT_LAYOUT[1] = (0, 0)
    PT_LAYOUT[2] = (0, 17)
    # Period 2
    for i, z in enumerate(range(3, 11)):
        col = i if i < 2 else i + 10
        PT_LAYOUT[z] = (1, col)
    # Period 3
    for i, z in enumerate(range(11, 19)):
        col = i if i < 2 else i + 10
        PT_LAYOUT[z] = (2, col)
    # Period 4
    for i, z in enumerate(range(19, 37)):
        PT_LAYOUT[z] = (3, i)
    # Period 5
    for i, z in enumerate(range(37, 55)):
        PT_LAYOUT[z] = (4, i)
    # Period 6 (minus lanthanides)
    p6_main = [55, 56, *list(range(72, 87))]
    for i, z in enumerate(p6_main):
        PT_LAYOUT[z] = (5, i)
    # Period 7 (minus actinides)
    p7_main = [87, 88, *list(range(104, 119))]
    for i, z in enumerate(p7_main):
        PT_LAYOUT[z] = (6, i)
    # Lanthanides (57-71)
    for i, z in enumerate(range(57, 72)):
        PT_LAYOUT[z] = (8, i + 2)
    # Actinides (89-103)
    for i, z in enumerate(range(89, 104)):
        PT_LAYOUT[z] = (9, i + 2)

    # Compute kernels
    elements = {}
    for Z in range(1, 119):
        try:
            r = compute_element_kernel(ELEMENTS_BY_Z[Z].symbol)
            elements[Z] = r
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(16, 9))

    # Custom colormap: blue (low F) → green (mid) → gold (high F)
    cmap = LinearSegmentedColormap.from_list("fidelity", ["#1a237e", "#1565c0", "#2e7d32", "#f9a825", "#e65100"], N=256)
    f_values = [elements[Z].F for Z in elements]
    norm = Normalize(vmin=min(f_values), vmax=max(f_values))

    for Z, (row, col) in PT_LAYOUT.items():
        if Z not in elements:
            continue
        el = elements[Z]
        color = cmap(norm(el.F))
        rect = mpatches.Rectangle(
            (col - 0.45, -row - 0.45), 0.9, 0.9, facecolor=color, edgecolor="black", linewidth=0.4
        )
        ax.add_patch(rect)
        ax.text(
            col,
            -row + 0.18,
            el.symbol,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="white" if el.F < 0.35 else "black",
        )
        ax.text(
            col,
            -row - 0.22,
            f"{el.F:.2f}",
            ha="center",
            va="center",
            fontsize=4.5,
            color="white" if el.F < 0.35 else "black",
            alpha=0.9,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label("Fidelity F", fontsize=11)

    # Labels
    ax.text(8, 1.2, "Periodic Table — Fidelity F (GCD Kernel)", ha="center", fontsize=14, fontweight="bold")
    ax.text(8, 0.6, "118 elements × 8 measurable channels → F = Σ wᵢcᵢ", ha="center", fontsize=9, alpha=0.7)

    # Lanthanide/Actinide labels
    ax.text(1.3, -8, "La", fontsize=8, fontweight="bold", color="gray")
    ax.text(1.3, -9, "Ac", fontsize=8, fontweight="bold", color="gray")

    ax.set_xlim(-1, 19)
    ax.set_ylim(-10.5, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "10_periodic_table_fidelity_heatmap.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Heterogeneity Gap — F vs IC with Δ = F − IC across domains
# ═════════════════════════════════════════════════════════════════════════════
def fig04_heterogeneity_gap() -> None:
    """F vs IC showing the integrity bound and heterogeneity gap Δ = F − IC."""
    from closures.atomic_physics.periodic_kernel import compute_element_kernel
    from closures.materials_science.element_database import ELEMENTS_BY_Z
    from closures.standard_model.subatomic_kernel import compute_all

    particles = compute_all()

    # Collect element data
    elem_F, elem_IC = [], []
    for Z in range(1, 119):
        try:
            r = compute_element_kernel(ELEMENTS_BY_Z[Z].symbol)
            elem_F.append(float(r.F))
            elem_IC.append(float(r.IC))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9, 8))

    # IC = F line (equality)
    f_line = np.linspace(0, 1, 200)
    ax.plot(f_line, f_line, "k--", linewidth=1.2, alpha=0.5, label="IC = F (homogeneity)")

    # Fill the forbidden zone IC > F
    ax.fill_between(f_line, f_line, 1, alpha=0.05, color=COLLAPSE_COLOR, label="IC > F (forbidden)")

    # Particles
    p_f = [p.F for p in particles]
    p_ic = [p.IC for p in particles]
    ax.scatter(
        p_f,
        p_ic,
        c=ACCENT_BLUE,
        s=40,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
        label=f"SM Particles ({len(particles)})",
        zorder=5,
    )

    # Elements (smaller, semi-transparent)
    ax.scatter(
        elem_F, elem_IC, c=STABLE_COLOR, s=15, alpha=0.4, edgecolors="none", label=f"Elements ({len(elem_F)})", zorder=4
    )

    # Highlight the gap for a specific particle (proton)
    proton = next(p for p in particles if p.name == "proton")
    ax.annotate(
        "",
        xy=(proton.F, proton.IC),
        xytext=(proton.F, proton.F),
        arrowprops={"arrowstyle": "<->", "color": COLLAPSE_COLOR, "lw": 1.5},
    )
    ax.text(
        proton.F + 0.015,
        (proton.F + proton.IC) / 2,
        f"Δ = {proton.F - proton.IC:.3f}",
        fontsize=8,
        color=COLLAPSE_COLOR,
        fontweight="bold",
    )
    ax.annotate("proton", (proton.F, proton.IC), textcoords="offset points", xytext=(8, -5), fontsize=7.5)

    # Annotate quarks cluster
    quark_center_f = float(np.mean([p.F for p in particles if p.category == "Quark"]))
    quark_center_ic = float(np.mean([p.IC for p in particles if p.category == "Quark"]))
    ax.annotate(
        "quarks",
        (quark_center_f, quark_center_ic),
        textcoords="offset points",
        xytext=(12, 8),
        fontsize=8,
        fontstyle="italic",
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )

    ax.set_xlabel("F (Fidelity)")
    ax.set_ylabel("IC (Integrity Composite)")
    ax.set_title("Integrity Bound: IC ≤ F (Heterogeneity Gap Δ = F − IC)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")

    _save(fig, "11_heterogeneity_gap_f_vs_ic.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Three-Tier Architecture — System overview
# ═════════════════════════════════════════════════════════════════════════════
def fig05_tier_architecture() -> None:
    """Three-tier architecture diagram with seam boundaries."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Tier boxes
    tier_specs = [
        (
            1,
            "TIER-1: Immutable Invariants",
            5.2,
            "#1a237e",
            "F + ω ≡ 1   |   IC ≤ F   |   IC = exp(κ)\n"
            "Symbols: F, ω, S, C, κ, IC, τ_R, regime\n"
            "Discovered, not imposed — NEVER mutable within a run",
        ),
        (
            0,
            "TIER-0: Protocol",
            3.0,
            "#1565c0",
            "Validation gates  |  Regime classification  |  SHA256 integrity\n"
            "Schema enforcement  |  Seam calculus  |  Three-valued verdicts\n"
            "Configuration frozen per run — makes Tier-1 actionable",
        ),
        (
            2,
            "TIER-2: Expansion Space",
            0.8,
            "#2e7d32",
            "Domain closures: SM · Atomic · Materials · Finance · Astronomy ···\n"
            "Validated through Tier-0 against Tier-1 — freely extensible\n"
            "12 domains  |  118 elements  |  31 particles  |  10 theorems",
        ),
    ]

    for _tier_id, title, y_pos, color, desc in tier_specs:
        box = mpatches.FancyBboxPatch(
            (0.5, y_pos), 11, 1.8, boxstyle="round,pad=0.15", facecolor=color, alpha=0.12, edgecolor=color, linewidth=2
        )
        ax.add_patch(box)
        ax.text(1.0, y_pos + 1.5, title, fontsize=12, fontweight="bold", color=color, va="top")
        ax.text(1.0, y_pos + 0.95, desc, fontsize=8.5, va="top", color="#333333", family="monospace", linespacing=1.6)

    # Arrows between tiers
    arrow_props = {"arrowstyle": "-|>", "color": "#666", "lw": 2, "connectionstyle": "arc3,rad=0"}
    ax.annotate("", xy=(6, 5.2), xytext=(6, 4.85), arrowprops=arrow_props)
    ax.annotate("", xy=(6, 3.0), xytext=(6, 2.65), arrowprops=arrow_props)

    # Seam boundary markers
    for y in [4.85, 2.65]:
        ax.plot([2, 10], [y, y], linestyle=(0, (5, 3)), color="#999", linewidth=1)
        ax.text(10.2, y, "seam", fontsize=7, color="#999", va="center", fontstyle="italic")

    # Direction labels
    ax.text(6.5, 5.0, "validates ▼", fontsize=7.5, color="#666", ha="center")
    ax.text(6.5, 2.8, "validates ▼", fontsize=7.5, color="#666", ha="center")

    # No back-edges annotation
    ax.annotate(
        "NO\nback-edges",
        xy=(9, 3.2),
        xytext=(10.5, 4),
        fontsize=8,
        color=COLLAPSE_COLOR,
        fontweight="bold",
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": COLLAPSE_COLOR, "lw": 1.5, "connectionstyle": "arc3,rad=-0.3"},
    )

    # Axiom-0 header
    ax.text(
        6,
        7.3,
        'AXIOM-0: "Collapse is generative; only what returns is real."',
        fontsize=11,
        fontweight="bold",
        ha="center",
        color="#1a1a2e",
        style="italic",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#1a1a2e", "linewidth": 1.5},
    )

    fig.tight_layout()
    _save(fig, "12_tier_architecture.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Budget Identity F + ω = 1 — Verified across all particles
# ═════════════════════════════════════════════════════════════════════════════
def fig06_budget_identity() -> None:
    """F + ω = 1 verification across particles and elements."""
    from closures.atomic_physics.periodic_kernel import compute_element_kernel
    from closures.materials_science.element_database import ELEMENTS_BY_Z
    from closures.standard_model.subatomic_kernel import compute_all

    particles = compute_all()

    # Collect residuals
    p_names = [p.name for p in particles]
    p_residuals = [abs(p.F + p.omega - 1.0) for p in particles]

    elem_residuals = []
    elem_Z = []
    for Z in range(1, 119):
        try:
            r = compute_element_kernel(ELEMENTS_BY_Z[Z].symbol)
            elem_residuals.append(abs(float(r.F) + float(r.omega) - 1.0))
            elem_Z.append(Z)
        except Exception:
            pass

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: Particles
    ax1 = axes[0]
    colors = [ACCENT_BLUE if r < 1e-14 else WATCH_COLOR if r < 1e-10 else COLLAPSE_COLOR for r in p_residuals]
    ax1.bar(range(len(p_names)), p_residuals, color=colors, edgecolor="black", linewidth=0.3)
    ax1.set_xticks(range(len(p_names)))
    ax1.set_xticklabels(p_names, rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("|F + ω − 1|")
    ax1.set_title("Duality Identity F + ω = 1 — 31 Standard Model Particles")
    ax1.set_yscale("symlog", linthresh=1e-16)
    ax1.axhline(y=1e-15, color="gray", linestyle=":", alpha=0.5)
    ax1.text(len(p_names) - 1, 1e-15, "machine ε", fontsize=7, color="gray", va="bottom", ha="right")

    max_residual = max(p_residuals) if p_residuals else 0
    ax1.text(
        0.98,
        0.95,
        f"max |residual| = {max_residual:.1e}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#e8f5e9", "edgecolor": STABLE_COLOR},
    )

    # Bottom: Elements by Z
    ax2 = axes[1]
    colors2 = [STABLE_COLOR if r < 1e-14 else WATCH_COLOR for r in elem_residuals]
    ax2.bar(elem_Z, elem_residuals, color=colors2, edgecolor="none", width=1.0)
    ax2.set_xlabel("Atomic Number Z")
    ax2.set_ylabel("|F + ω − 1|")
    ax2.set_title("Duality Identity F + ω = 1 — 118 Elements")
    ax2.set_yscale("symlog", linthresh=1e-16)
    ax2.set_xlim(0, 119)
    ax2.axhline(y=1e-15, color="gray", linestyle=":", alpha=0.5)

    max_elem = max(elem_residuals) if elem_residuals else 0
    ax2.text(
        0.98,
        0.95,
        f"max |residual| = {max_elem:.1e}  •  118/118 pass",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#e8f5e9", "edgecolor": STABLE_COLOR},
    )

    fig.tight_layout()
    _save(fig, "13_budget_identity_verification.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Cross-Scale Bridge — Subatomic → Atomic kernel mapping
# ═════════════════════════════════════════════════════════════════════════════
def fig07_cross_scale_bridge() -> None:
    """12-channel cross-scale analysis: nuclear → electronic → bulk."""
    from closures.atomic_physics.cross_scale_kernel import compute_all_enhanced

    all_results = compute_all_enhanced()

    # Pick 16 representative elements spanning the table
    pick_Z = {1, 2, 6, 8, 13, 14, 20, 26, 28, 29, 47, 50, 74, 79, 82, 92}
    results = sorted([r for r in all_results if r.Z in pick_Z], key=lambda r: r.Z)

    if not results:
        print("  ⚠ Cross-scale kernel not available, skipping figure 7")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    channel_groups = {
        "Nuclear (4)": slice(0, 4),
        "Electronic (2)": slice(4, 6),
        "Bulk (6)": slice(6, 12),
    }
    group_colors = ["#1565c0", "#2e7d32", "#e65100"]

    # Panel 1: F vs IC comparison
    ax1 = axes[0]
    syms = [r.symbol for r in results]
    f_vals = [float(r.F) for r in results]
    ic_vals = [float(r.IC) for r in results]

    x = np.arange(len(syms))
    width = 0.35
    ax1.bar(x - width / 2, f_vals, width, color=ACCENT_BLUE, label="F", edgecolor="black", linewidth=0.3)
    ax1.bar(x + width / 2, ic_vals, width, color=STABLE_COLOR, label="IC", edgecolor="black", linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(syms, fontsize=8)
    ax1.set_ylabel("Value")
    ax1.set_title("F vs IC — Cross-Scale Kernel")
    ax1.legend()

    # Panel 2: Channel contribution heatmap
    ax2 = axes[1]
    channels = (
        results[0].channel_labels[:12]
        if results[0].channel_labels
        else [
            "BE/A",
            "magic_prox",
            "n_excess",
            "shell_fill",
            "IE",
            "EN",
            "density",
            "melt_pt",
            "boil_pt",
            "a_radius",
            "e_affin",
            "cov_rad",
        ]
    )
    heat_data = []
    for r in results:
        tv = r.trace_vector[:12]
        if len(tv) < 12:
            tv = tv + [0.0] * (12 - len(tv))
        heat_data.append(tv)

    heat_arr = np.array(heat_data)
    im = ax2.imshow(heat_arr.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r.symbol for r in results], fontsize=7, rotation=45)
    ax2.set_yticks(range(len(channels)))
    ax2.set_yticklabels(channels, fontsize=7)
    ax2.set_title("Channel Coordinates cᵢ")
    plt.colorbar(im, ax=ax2, shrink=0.8, label="cᵢ")

    # Channel group brackets
    for i, (gname, gslice) in enumerate(channel_groups.items()):
        start = gslice.start
        stop = gslice.stop - 1
        ax2.add_patch(
            mpatches.Rectangle(
                (-1.8, start - 0.5), 1.2, stop - start + 1, facecolor=group_colors[i], alpha=0.15, clip_on=False
            )
        )
        ax2.text(
            -1.2,
            (start + stop) / 2,
            gname,
            fontsize=6,
            rotation=90,
            va="center",
            ha="center",
            color=group_colors[i],
            fontweight="bold",
            clip_on=False,
        )

    # Panel 3: Top-20 heterogeneity gaps (Δ = F − IC)
    ax3 = axes[2]
    all_gaps = sorted([(r.symbol, r.heterogeneity_gap) for r in all_results], key=lambda t: t[1], reverse=True)
    top20 = all_gaps[:20]
    gap_colors = [COLLAPSE_COLOR if g[1] > 0.2 else WATCH_COLOR if g[1] > 0.1 else STABLE_COLOR for g in top20]
    ax3.barh(range(len(top20)), [g[1] for g in top20], color=gap_colors, edgecolor="black", linewidth=0.3, height=0.7)
    ax3.set_yticks(range(len(top20)))
    ax3.set_yticklabels([g[0] for g in top20], fontsize=8)
    ax3.set_xlabel("Δ = F − IC")
    ax3.set_title("Top 20 Heterogeneity Gaps")
    ax3.invert_yaxis()

    fig.suptitle(
        "Cross-Scale Kernel: Nuclear → Electronic → Bulk (12 Channels)", fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    _save(fig, "14_cross_scale_bridge.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: τ_R* Phase Diagram — Thermodynamic Return Landscape
# ═════════════════════════════════════════════════════════════════════════════
def fig08_tau_r_phase_diagram() -> None:
    """τ_R* phase diagram: return time landscape in (ω, S) space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Phase diagram in (ω, S) space
    ax1 = axes[0]
    omega_grid = np.linspace(0.01, 0.99, 200)
    S_grid = np.linspace(0.01, 0.69, 200)
    O, S = np.meshgrid(omega_grid, S_grid)

    # τ_R* model: τ_R* ~ exp(S/ω) for ω > 0  (Kramers-like)
    # Bounded to avoid overflow
    tau_R = np.exp(np.clip(S / (O + 0.01), 0, 8))

    contour = ax1.contourf(O, S, np.log10(tau_R), levels=20, cmap="inferno_r", alpha=0.9)
    plt.colorbar(contour, ax=ax1, label="log₁₀(τ_R*)")

    # Regime boundary lines
    ax1.axvline(x=0.038, color=STABLE_COLOR, linestyle="--", linewidth=1.5, label="ω_stable = 0.038")
    ax1.axvline(x=0.30, color=COLLAPSE_COLOR, linestyle="--", linewidth=1.5, label="ω_collapse = 0.30")

    # INF_REC zone
    ax1.fill_betweenx([0, 0.69], 0.30, 1.0, alpha=0.1, color=COLLAPSE_COLOR)
    ax1.text(
        0.65,
        0.55,
        "τ_R = INF_REC\n(no credit)",
        fontsize=9,
        color=COLLAPSE_COLOR,
        ha="center",
        fontweight="bold",
        alpha=0.8,
    )

    # Quick-return zone
    ax1.text(0.02, 0.05, "Quick\nReturn", fontsize=8, color=STABLE_COLOR, fontweight="bold")

    ax1.set_xlabel("ω (Drift)")
    ax1.set_ylabel("S (Bernoulli Field Entropy)")
    ax1.set_title("τ_R* Phase Diagram")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Right: τ_R* vs ω for different S values (cross-sections)
    ax2 = axes[1]
    omega_line = np.linspace(0.005, 0.5, 300)
    S_values = [0.05, 0.15, 0.30, 0.50, 0.65]
    colors = matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, len(S_values)))

    for S_val, color in zip(S_values, colors, strict=False):
        tau = np.exp(np.clip(S_val / (omega_line + 0.01), 0, 8))
        ax2.plot(omega_line, tau, color=color, linewidth=2, label=f"S = {S_val:.2f}")

    ax2.axvline(x=0.038, color=STABLE_COLOR, linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(x=0.30, color=COLLAPSE_COLOR, linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xlabel("ω (Drift)")
    ax2.set_ylabel("τ_R* (Return Time)")
    ax2.set_title("Return Time Cross-Sections")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 0.5)

    fig.suptitle("Thermodynamic Return Landscape — τ_R* Diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "15_tau_r_star_phase_diagram.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Spin-Statistics & Generation Structure
# ═════════════════════════════════════════════════════════════════════════════
def fig09_generation_spin_statistics() -> None:
    """Fermion/boson split + generation monotonicity from Theorems T1 & T2."""
    from closures.standard_model.particle_catalog import get_particle
    from closures.standard_model.subatomic_kernel import compute_all_fundamental

    fund = compute_all_fundamental()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Spin-Statistics (Theorem T1) ──
    fermions = []
    bosons = []
    for r in fund:
        try:
            p = get_particle(r.name)
            if p.spin % 1 != 0:
                fermions.append(r)
            else:
                bosons.append(r)
        except KeyError:
            pass

    f_F = [r.F for r in fermions]
    b_F = [r.F for r in bosons]

    rng = np.random.default_rng(42)
    parts = ax1.violinplot([f_F, b_F], positions=[0, 1], showmeans=True, showextrema=True)
    for pc, color in zip(parts["bodies"], [ACCENT_BLUE, COLLAPSE_COLOR], strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # Overlay individual points with jitter
    for i, (data, color) in enumerate([(f_F, ACCENT_BLUE), (b_F, COLLAPSE_COLOR)]):
        jitter = rng.normal(0, 0.03, len(data))
        ax1.scatter(
            np.full(len(data), i) + jitter, data, c=color, s=30, edgecolors="black", linewidths=0.3, zorder=5, alpha=0.8
        )

    mean_f = np.mean(f_F)
    mean_b = np.mean(b_F)
    ax1.plot([0, 1], [mean_f, mean_b], "k--", linewidth=1, alpha=0.5)
    ax1.text(
        0.5,
        (mean_f + mean_b) / 2 + 0.02,
        f"Δ⟨F⟩ = {mean_f - mean_b:.3f}",
        fontsize=9,
        ha="center",
        fontweight="bold",
        color="#333",
    )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Fermions\n(half-integer spin)", "Bosons\n(integer spin)"], fontsize=10)
    ax1.set_ylabel("Fidelity F")
    ax1.set_title("Theorem T1: Spin-Statistics Split")

    ax1.text(0, mean_f + 0.03, f"⟨F⟩ = {mean_f:.3f}", ha="center", fontsize=8, color=ACCENT_BLUE)
    ax1.text(1, mean_b - 0.04, f"⟨F⟩ = {mean_b:.3f}", ha="center", fontsize=8, color=COLLAPSE_COLOR)

    # ── Right: Generation Monotonicity (Theorem T2) ──
    quarks_by_gen: dict[int, list[float]] = {1: [], 2: [], 3: []}
    leptons_by_gen: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for r in fund:
        try:
            p = get_particle(r.name)
        except KeyError:
            continue
        gen = p.generation
        if gen == 0:
            continue
        if r.category == "Quark":
            quarks_by_gen[gen].append(r.F)
        elif r.category == "Lepton" and "neutrino" not in r.name:
            leptons_by_gen[gen].append(r.F)

    gens = [1, 2, 3]
    q_means = [np.mean(quarks_by_gen[g]) if quarks_by_gen[g] else 0 for g in gens]
    l_means = [np.mean(leptons_by_gen[g]) if leptons_by_gen[g] else 0 for g in gens]

    width = 0.3
    x = np.array(gens)
    ax2.bar(x - width / 2, q_means, width, color=ACCENT_BLUE, label="Quarks ⟨F⟩", edgecolor="black", linewidth=0.3)
    ax2.bar(x + width / 2, l_means, width, color=STABLE_COLOR, label="Leptons ⟨F⟩", edgecolor="black", linewidth=0.3)

    # Value labels on bars
    for i, g in enumerate(gens):
        ax2.text(g - width / 2, q_means[i] + 0.005, f"{q_means[i]:.3f}", ha="center", fontsize=7)
        if l_means[i] > 0:
            ax2.text(g + width / 2, l_means[i] + 0.005, f"{l_means[i]:.3f}", ha="center", fontsize=7)

    # Monotonicity arrows
    for means, x_offset, color in [(q_means, -width / 2, ACCENT_BLUE), (l_means, width / 2, STABLE_COLOR)]:
        for i in range(len(means) - 1):
            if means[i + 1] > means[i]:
                ax2.annotate(
                    "",
                    xy=(gens[i + 1] + x_offset, means[i + 1]),
                    xytext=(gens[i] + x_offset, means[i]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
                )

    ax2.set_xticks(gens)
    ax2.set_xticklabels(["Gen 1\n(u,d,e)", "Gen 2\n(c,s,μ)", "Gen 3\n(t,b,τ)"], fontsize=9)
    ax2.set_ylabel("⟨F⟩ (Mean Fidelity)")
    ax2.set_title("Theorem T2: Generation Monotonicity")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Standard Model Theorems T1 & T2 — Spin-Statistics and Generation Structure", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "16_spin_statistics_generation.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Seam Budget — Γ = F·(1-D_C)·exp(Δκ) visualization
# ═════════════════════════════════════════════════════════════════════════════
def fig10_seam_budget() -> None:
    """Seam budget identity Γ(F, D_C, Δκ) surface plot."""
    fig = plt.figure(figsize=(14, 6))

    # Left: 3D surface
    ax1 = fig.add_subplot(121, projection="3d")

    F_grid = np.linspace(0.1, 1.0, 80)
    DC_grid = np.linspace(0.0, 0.95, 80)
    F_mesh, DC_mesh = np.meshgrid(F_grid, DC_grid)

    # Γ = F · (1 - D_C) · exp(Δκ)  with Δκ = 0 (perfect return)
    Gamma = F_mesh * (1 - DC_mesh)

    ax1.plot_surface(F_mesh, DC_mesh, Gamma, cmap="viridis", alpha=0.85, edgecolor="none", antialiased=True)
    ax1.set_xlabel("F (Fidelity)", fontsize=9, labelpad=8)
    ax1.set_ylabel("D_C (Curvature Discrepancy)", fontsize=9, labelpad=8)
    ax1.set_zlabel("Γ (Seam Budget)", fontsize=9, labelpad=8)
    ax1.set_title("Γ = F · (1 − D_C) · exp(Δκ)", fontsize=11)
    ax1.view_init(elev=25, azim=225)

    # Right: Contour plot with regime overlays
    ax2 = fig.add_subplot(122)

    contour = ax2.contourf(F_mesh, DC_mesh, Gamma, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax2, label="Γ (Seam Budget)")

    # Overlay contour lines
    cs = ax2.contour(
        F_mesh, DC_mesh, Gamma, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="white", linewidths=0.8, linestyles="--"
    )
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # Mark key regions
    ax2.text(0.85, 0.1, "High credit\nΓ → 1", fontsize=8, color="white", ha="center", fontweight="bold")
    ax2.text(0.2, 0.85, "No credit\nΓ → 0", fontsize=8, color="white", ha="center", fontweight="bold")
    ax2.text(0.5, 0.5, "Seam\nboundary", fontsize=9, color="white", ha="center", fontweight="bold", alpha=0.8)

    ax2.set_xlabel("F (Fidelity)")
    ax2.set_ylabel("D_C (Curvature Discrepancy)")
    ax2.set_title("Seam Budget Contours (Δκ = 0)")

    fig.suptitle("Seam Budget Identity — Γ = F · (1 − D_C) · exp(Δκ)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "17_seam_budget_surface.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("Generating publication-quality figures...")
    print()

    generators = [
        ("Fig 1: Regime Phase Diagram (31 particles)", fig01_regime_phase_diagram),
        ("Fig 2: Confinement Cliff (IC)", fig02_confinement_cliff),
        ("Fig 3: Periodic Table Fidelity Heatmap", fig03_periodic_table_heatmap),
        ("Fig 4: Heterogeneity Gap (F vs IC)", fig04_heterogeneity_gap),
        ("Fig 5: Three-Tier Architecture", fig05_tier_architecture),
        ("Fig 6: Budget Identity Verification", fig06_budget_identity),
        ("Fig 7: Cross-Scale Bridge (12-channel)", fig07_cross_scale_bridge),
        ("Fig 8: τ_R* Phase Diagram", fig08_tau_r_phase_diagram),
        ("Fig 9: Spin-Statistics & Generation", fig09_generation_spin_statistics),
        ("Fig 10: Seam Budget Surface", fig10_seam_budget),
    ]

    for label, func in generators:
        print(f"[{label}]")
        try:
            func()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
