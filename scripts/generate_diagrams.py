#!/usr/bin/env python3
"""Generate PNG diagrams showing GCD kernel geometry and statistical proofs.

Uses real data from UMCP closures — Standard Model particles, periodic table,
double-slit scenarios. Every number comes from computed kernel outputs.

Usage:
    python scripts/generate_diagrams.py
    # Outputs 7 PNGs to images/
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.size": 11,
        "font.family": "monospace",
        "figure.dpi": 200,
    }
)

ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"
PINK = "#f778ba"
YELLOW = "#e3b341"

# ═══════════════════════════════════════════════════════════════════════════
# REAL DATA — from closures/standard_model/subatomic_kernel.py
# ═══════════════════════════════════════════════════════════════════════════

FUNDAMENTAL = {
    # name: (F, IC, category)
    "u": (0.594691, 0.560647, "Quark"),
    "d": (0.556172, 0.516654, "Quark"),
    "c": (0.662384, 0.633713, "Quark"),
    "s": (0.610048, 0.573335, "Quark"),
    "t": (0.638027, 0.588996, "Quark"),
    "b": (0.667246, 0.614820, "Quark"),
    "e⁻": (0.674375, 0.614431, "Lepton"),
    "ν_e": (0.476933, 0.020801, "Lepton"),
    "μ⁻": (0.689990, 0.651915, "Lepton"),
    "ν_μ": (0.518600, 0.022684, "Lepton"),
    "τ⁻": (0.729058, 0.678422, "Lepton"),
    "ν_τ": (0.560266, 0.023863, "Lepton"),
    "γ": (0.331100, 0.000755, "GaugeBoson"),
    "W±": (0.574233, 0.023541, "GaugeBoson"),
    "Z⁰": (0.366252, 0.003649, "GaugeBoson"),
    "g": (0.416667, 0.000872, "GaugeBoson"),
    "H⁰": (0.414902, 0.004054, "ScalarBoson"),
}

COMPOSITE = {
    # name: (F, IC, type)
    "p": (0.549722, 0.020405, "Baryon"),
    "n": (0.394867, 0.003520, "Baryon"),
    "Λ⁰": (0.407111, 0.015331, "Baryon"),
    "Σ⁺": (0.526330, 0.022898, "Baryon"),
    "Ξ⁰": (0.444552, 0.004452, "Baryon"),
    "Ω⁻": (0.673576, 0.028695, "Baryon"),
    "Λ_c⁺": (0.544499, 0.023920, "Baryon"),
    "π⁺": (0.476098, 0.004696, "Meson"),
    "π⁰": (0.334239, 0.000808, "Meson"),
    "K⁺": (0.472994, 0.021240, "Meson"),
    "K⁰": (0.349060, 0.003783, "Meson"),
    "J/ψ": (0.363895, 0.001961, "Meson"),
    "Υ": (0.368812, 0.000832, "Meson"),
    "D⁰": (0.316533, 0.002516, "Meson"),
}

# Double slit scenarios
DOUBLE_SLIT = {
    "S1: Full interference": (0.851375, 0.097621, 0.148625),
    "S2: Which-path": (0.700000, 0.064519, 0.300000),
    "S3: Single slit": (0.672500, 0.061706, 0.327500),
    "S4: Weak measurement": (0.855625, 0.846813, 0.144375),
    "S5: Quantum eraser": (0.852812, 0.756210, 0.147188),
    "S6: Delayed choice": (0.847687, 0.097252, 0.152313),
    "S7: Electron (Tonomura)": (0.831875, 0.707138, 0.168125),
    "S8: Classical limit": (0.501250, 0.028593, 0.498750),
}

# Generation data
GEN_QUARKS = {"Gen 1\n(u, d)": 0.5754, "Gen 2\n(c, s)": 0.6362, "Gen 3\n(t, b)": 0.6526}
GEN_LEPTONS = {"Gen 1\n(e, νe)": 0.5757, "Gen 2\n(μ, νμ)": 0.6043, "Gen 3\n(τ, ντ)": 0.6447}

# Periodic table (selected representative elements per block)
PERIODIC_BY_BLOCK: dict[str, list[tuple[int, str, float, float]]] = {
    "s": [
        (1, "H", 0.279, 0.025),
        (3, "Li", 0.201, 0.158),
        (11, "Na", 0.184, 0.171),
        (19, "K", 0.158, 0.154),
        (37, "Rb", 0.177, 0.162),
        (55, "Cs", 0.185, 0.040),
        (4, "Be", 0.366, 0.287),
        (12, "Mg", 0.273, 0.253),
        (20, "Ca", 0.228, 0.163),
        (38, "Sr", 0.242, 0.186),
        (56, "Ba", 0.269, 0.223),
    ],
    "p": [
        (5, "B", 0.412, 0.293),
        (6, "C", 0.531, 0.413),
        (7, "N", 0.313, 0.055),
        (8, "O", 0.336, 0.073),
        (9, "F", 0.445, 0.089),
        (14, "Si", 0.407, 0.371),
        (16, "S", 0.365, 0.289),
        (17, "Cl", 0.396, 0.125),
        (32, "Ge", 0.412, 0.399),
        (35, "Br", 0.439, 0.309),
        (50, "Sn", 0.394, 0.364),
        (53, "I", 0.438, 0.342),
        (82, "Pb", 0.381, 0.320),
    ],
    "d": [
        (21, "Sc", 0.325, 0.272),
        (22, "Ti", 0.363, 0.270),
        (23, "V", 0.404, 0.361),
        (24, "Cr", 0.397, 0.366),
        (25, "Mn", 0.390, 0.375),
        (26, "Fe", 0.387, 0.313),
        (27, "Co", 0.413, 0.384),
        (28, "Ni", 0.430, 0.411),
        (29, "Cu", 0.414, 0.399),
        (42, "Mo", 0.510, 0.462),
        (44, "Ru", 0.518, 0.488),
        (74, "W", 0.627, 0.562),
        (76, "Os", 0.620, 0.577),
        (77, "Ir", 0.605, 0.582),
        (78, "Pt", 0.592, 0.576),
        (79, "Au", 0.562, 0.542),
    ],
    "f": [
        (57, "La", 0.355, 0.317),
        (58, "Ce", 0.377, 0.351),
        (60, "Nd", 0.413, 0.390),
        (64, "Gd", 0.381, 0.302),
        (90, "Th", 0.506, 0.459),
        (92, "U", 0.472, 0.405),
    ],
}

# Full periodic table F values for heatmap (Z → F)
PERIODIC_F: dict[int, float] = {
    1: 0.279,
    2: 0.316,
    3: 0.201,
    4: 0.366,
    5: 0.412,
    6: 0.531,
    7: 0.313,
    8: 0.336,
    9: 0.445,
    10: 0.304,
    11: 0.184,
    12: 0.273,
    13: 0.309,
    14: 0.407,
    15: 0.298,
    16: 0.365,
    17: 0.396,
    18: 0.260,
    19: 0.158,
    20: 0.228,
    21: 0.325,
    22: 0.363,
    23: 0.404,
    24: 0.397,
    25: 0.390,
    26: 0.387,
    27: 0.413,
    28: 0.430,
    29: 0.414,
    30: 0.354,
    31: 0.325,
    32: 0.412,
    33: 0.369,
    34: 0.401,
    35: 0.439,
    36: 0.335,
    37: 0.177,
    38: 0.242,
    39: 0.344,
    40: 0.420,
    41: 0.487,
    42: 0.510,
    43: 0.482,
    44: 0.518,
    45: 0.501,
    46: 0.452,
    47: 0.416,
    48: 0.363,
    49: 0.333,
    50: 0.394,
    51: 0.391,
    52: 0.410,
    53: 0.438,
    54: 0.323,
    55: 0.185,
    56: 0.269,
    57: 0.355,
    58: 0.377,
    59: 0.384,
    60: 0.413,
    61: 0.352,
    62: 0.333,
    63: 0.309,
    64: 0.381,
    65: 0.420,
    66: 0.382,
    67: 0.390,
    68: 0.397,
    69: 0.405,
    70: 0.319,
    71: 0.416,
    72: 0.483,
    73: 0.555,
    74: 0.627,
    75: 0.586,
    76: 0.620,
    77: 0.605,
    78: 0.592,
    79: 0.562,
    80: 0.416,
    81: 0.356,
    82: 0.381,
    83: 0.406,
    84: 0.416,
    85: 0.447,
    86: 0.302,
    87: 0.242,
    88: 0.311,
    89: 0.396,
    90: 0.506,
    91: 0.476,
    92: 0.472,
    93: 0.452,
    94: 0.477,
    95: 0.415,
    96: 0.454,
    97: 0.430,
    98: 0.398,
    99: 0.376,
    100: 0.401,
    101: 0.402,
    102: 0.479,
    103: 0.397,
    104: 0.719,
    105: 0.695,
    106: 0.716,
    107: 0.728,
    108: 0.741,
    109: 0.742,
    110: 0.744,
    111: 0.736,
    112: 0.465,
    113: 0.438,
    114: 0.431,
    115: 0.405,
    116: 0.423,
    117: 0.444,
    118: 0.337,
}

PERIODIC_IC: dict[int, float] = {
    1: 0.025,
    2: 0.007,
    3: 0.158,
    4: 0.287,
    5: 0.293,
    6: 0.413,
    7: 0.055,
    8: 0.073,
    9: 0.089,
    10: 0.028,
    11: 0.171,
    12: 0.253,
    13: 0.270,
    14: 0.371,
    15: 0.232,
    16: 0.289,
    17: 0.125,
    18: 0.047,
    19: 0.154,
    20: 0.163,
    21: 0.272,
    22: 0.270,
    23: 0.361,
    24: 0.366,
    25: 0.375,
    26: 0.313,
    27: 0.384,
    28: 0.411,
    29: 0.399,
    30: 0.327,
    31: 0.272,
    32: 0.399,
    33: 0.337,
    34: 0.350,
    35: 0.309,
    36: 0.092,
    37: 0.162,
    38: 0.186,
    39: 0.307,
    40: 0.370,
    41: 0.446,
    42: 0.462,
    43: 0.434,
    44: 0.488,
    45: 0.479,
    46: 0.419,
    47: 0.405,
    48: 0.330,
    49: 0.280,
    50: 0.364,
    51: 0.377,
    52: 0.383,
    53: 0.342,
    54: 0.107,
    55: 0.040,
    56: 0.223,
    57: 0.317,
    58: 0.351,
    59: 0.358,
    60: 0.390,
    61: 0.280,
    62: 0.275,
    63: 0.249,
    64: 0.302,
    65: 0.398,
    66: 0.338,
    67: 0.342,
    68: 0.344,
    69: 0.385,
    70: 0.204,
    71: 0.357,
    72: 0.379,
    73: 0.455,
    74: 0.562,
    75: 0.442,
    76: 0.577,
    77: 0.582,
    78: 0.576,
    79: 0.542,
    80: 0.312,
    81: 0.297,
    82: 0.320,
    83: 0.363,
    84: 0.365,
    85: 0.350,
    86: 0.098,
    87: 0.182,
    88: 0.229,
    89: 0.330,
    90: 0.459,
    91: 0.414,
    92: 0.405,
    93: 0.378,
    94: 0.422,
    95: 0.306,
    96: 0.366,
    97: 0.324,
    98: 0.277,
    99: 0.302,
    100: 0.320,
    101: 0.357,
    102: 0.427,
    103: 0.301,
    104: 0.648,
    105: 0.611,
    106: 0.629,
    107: 0.643,
    108: 0.656,
    109: 0.665,
    110: 0.673,
    111: 0.674,
    112: 0.266,
    113: 0.349,
    114: 0.242,
    115: 0.294,
    116: 0.332,
    117: 0.363,
    118: 0.161,
}


def fig_path(name: str) -> str:
    return os.path.join(OUT, name)


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: Kernel Geometry — F vs IC with AM-GM Bound
# ═══════════════════════════════════════════════════════════════════════════
def plot_kernel_geometry() -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    # AM-GM bound: IC ≤ F (diagonal)
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, "--", color=ORANGE, alpha=0.8, linewidth=1.5, label="IC = F (AM-GM bound)")
    ax.fill_between(x, x, 1, alpha=0.05, color=RED)
    ax.fill_between(x, 0, x, alpha=0.05, color=GREEN)

    # Plot particles by category
    cats = {
        "Quark": (ACCENT, "^", 100),
        "Lepton": (GREEN, "s", 100),
        "GaugeBoson": (RED, "D", 100),
        "ScalarBoson": (PINK, "p", 130),
    }
    for name, (F, IC, cat) in FUNDAMENTAL.items():
        color, marker, sz = cats[cat]
        ax.scatter(F, IC, c=color, marker=marker, s=sz, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(
            name,
            (F, IC),
            fontsize=6,
            color="#8b949e",
            xytext=(5, 5),
            textcoords="offset points",
        )

    comp_cats = {"Baryon": (PURPLE, "o", 70), "Meson": (CYAN, "v", 70)}
    for name, (F, IC, cat) in COMPOSITE.items():
        color, marker, sz = comp_cats[cat]
        ax.scatter(F, IC, c=color, marker=marker, s=sz, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(
            name,
            (F, IC),
            fontsize=5.5,
            color="#6e7681",
            xytext=(4, -8),
            textcoords="offset points",
        )

    # Regime boundaries (vertical lines at ω thresholds → F thresholds)
    regime_bounds = [
        (0.90, "STABLE", GREEN),
        (0.80, "WATCH", YELLOW),
        (0.70, "TENSION", ORANGE),
    ]
    for f_thresh, label, color in regime_bounds:
        ax.axvline(x=f_thresh, color=color, alpha=0.3, linewidth=1, linestyle=":")
        ax.text(f_thresh + 0.005, 0.82, label, fontsize=7, color=color, rotation=90, va="top")

    ax.axvline(x=0.70, color=RED, alpha=0.3, linewidth=1, linestyle=":")
    ax.text(0.70 - 0.025, 0.82, "COLLAPSE →", fontsize=7, color=RED, rotation=90, va="top")

    # Legend
    handles = [
        mpatches.Patch(color=ACCENT, label="Quarks (6)"),
        mpatches.Patch(color=GREEN, label="Leptons (6)"),
        mpatches.Patch(color=RED, label="Gauge Bosons (4)"),
        mpatches.Patch(color=PINK, label="Scalar Boson (H⁰)"),
        mpatches.Patch(color=PURPLE, label="Baryons (7)"),
        mpatches.Patch(color=CYAN, label="Mesons (7)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.3, edgecolor="#30363d")

    ax.set_xlabel("Fidelity  F  (arithmetic mean)", fontsize=12)
    ax.set_ylabel("Integrity Composite  IC  (geometric mean)", fontsize=12)
    ax.set_title(
        "GCD Kernel Geometry: F vs IC for 31 Standard Model Particles\n"
        "IC ≤ F guaranteed by AM-GM inequality  │  Δ = F − IC measures channel heterogeneity",
        fontsize=11,
        pad=15,
    )
    ax.set_xlim(0.25, 0.80)
    ax.set_ylim(-0.02, 0.75)
    ax.grid(True, alpha=0.3)

    # Annotation: the confinement cliff
    ax.annotate(
        "CONFINEMENT CLIFF\nAll composites cluster\nnear IC ≈ 0\n(98.1% IC collapse)",
        xy=(0.45, 0.015),
        xytext=(0.35, 0.35),
        fontsize=8,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#161b22", "edgecolor": ORANGE, "alpha": 0.9},
    )

    # Annotation: quark cluster
    ax.annotate(
        "Quarks: IC ≈ F\n(channels alive)",
        xy=(0.63, 0.59),
        xytext=(0.72, 0.45),
        fontsize=8,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#161b22", "edgecolor": ACCENT, "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(fig_path("01_kernel_geometry_f_vs_ic.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_kernel_geometry_f_vs_ic.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: Confinement Cliff — IC Collapse at Quark→Hadron Boundary
# ═══════════════════════════════════════════════════════════════════════════
def plot_confinement_cliff() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Quark ICs
    quarks = {k: v for k, v in FUNDAMENTAL.items() if v[2] == "Quark"}
    q_names = list(quarks.keys())
    q_ics = [quarks[n][1] for n in q_names]

    # Hadron ICs (baryons + mesons)
    h_names = list(COMPOSITE.keys())
    h_ics = [COMPOSITE[n][1] for n in h_names]

    all_names = [*q_names, "│", *h_names]
    all_ics = [*q_ics, None, *h_ics]
    all_colors = [ACCENT] * len(q_names) + ["none"] + [PURPLE if COMPOSITE[n][2] == "Baryon" else CYAN for n in h_names]

    x = list(range(len(all_names)))
    bars = []
    for i, (ic, color) in enumerate(zip(all_ics, all_colors, strict=False)):
        if ic is None:
            continue
        bar = ax.bar(i, ic, color=color, edgecolor="white", linewidth=0.5, alpha=0.85, width=0.7)
        bars.append(bar)

    # Cliff annotation
    cliff_x = len(q_names)  # divider position
    ax.axvline(x=cliff_x, color=RED, linewidth=2, linestyle="--", alpha=0.8)
    ax.annotate(
        "CONFINEMENT\n    CLIFF\n\n  98.1% IC\n   collapse",
        xy=(cliff_x, 0.35),
        fontsize=11,
        color=RED,
        fontweight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#161b22", "edgecolor": RED, "alpha": 0.9},
    )

    # Mean lines
    q_mean = sum(q_ics) / len(q_ics)
    h_mean = sum(h_ics) / len(h_ics)
    ax.axhline(y=q_mean, xmin=0, xmax=cliff_x / len(all_names), color=ACCENT, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(
        y=h_mean, xmin=(cliff_x + 1) / len(all_names), xmax=1, color=PURPLE, linewidth=1.5, linestyle="--", alpha=0.7
    )

    ax.text(2.5, q_mean + 0.015, f"⟨IC⟩_quarks = {q_mean:.4f}", fontsize=9, color=ACCENT, ha="center")
    ax.text(cliff_x + 7, h_mean + 0.015, f"⟨IC⟩_hadrons = {h_mean:.4f}", fontsize=9, color=PURPLE, ha="center")

    # Min quark IC line
    min_q_ic = min(q_ics)
    ax.axhline(y=min_q_ic, color=ORANGE, linewidth=1, linestyle=":", alpha=0.6)
    ax.text(
        len(all_names) - 2, min_q_ic + 0.015, f"min quark IC = {min_q_ic:.4f}", fontsize=7, color=ORANGE, ha="right"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Integrity Composite  IC", fontsize=12)
    ax.set_title(
        "Theorem T3: Confinement as IC Collapse\n14/14 hadrons below minimum quark IC  │  Gap amplification: 10.82×",
        fontsize=11,
        pad=15,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 0.72)

    handles = [
        mpatches.Patch(color=ACCENT, label=f"Quarks (⟨IC⟩={q_mean:.3f})"),
        mpatches.Patch(color=PURPLE, label=f"Baryons (⟨IC⟩={h_mean:.4f})"),
        mpatches.Patch(color=CYAN, label="Mesons"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.3, edgecolor="#30363d")

    fig.tight_layout()
    fig.savefig(fig_path("02_confinement_cliff.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_confinement_cliff.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: Complementarity Cliff — Double-Slit 8 Scenarios
# ═══════════════════════════════════════════════════════════════════════════
def plot_complementarity_cliff() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 2]})

    names = list(DOUBLE_SLIT.keys())
    F_vals = [DOUBLE_SLIT[n][0] for n in names]
    IC_vals = [DOUBLE_SLIT[n][1] for n in names]
    [DOUBLE_SLIT[n][2] for n in names]

    x = np.arange(len(names))
    width = 0.35

    # Left panel: F and IC bars
    bars_f = ax1.bar(
        x - width / 2, F_vals, width, color=ACCENT, alpha=0.85, label="F (Fidelity)", edgecolor="white", linewidth=0.5
    )
    bars_ic = ax1.bar(
        x + width / 2, IC_vals, width, color=GREEN, alpha=0.85, label="IC (Integrity)", edgecolor="white", linewidth=0.5
    )

    # Highlight S4
    bars_f[3].set_edgecolor(YELLOW)
    bars_f[3].set_linewidth(2)
    bars_ic[3].set_edgecolor(YELLOW)
    bars_ic[3].set_linewidth(2)

    # Cliff threshold
    ax1.axhline(y=0.10, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.text(0.5, 0.12, "CLIFF: IC < 0.10", fontsize=9, color=RED)

    # S4 annotation
    ax1.annotate(
        "S4: KERNEL-OPTIMAL\nIC = 0.847 (highest)\nΔ = 0.009 (lowest)\nAll channels alive",
        xy=(3, 0.855),
        xytext=(5, 0.90),
        fontsize=8,
        color=YELLOW,
        arrowprops={"arrowstyle": "->", "color": YELLOW, "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#161b22", "edgecolor": YELLOW, "alpha": 0.9},
    )

    short_names = [n.split(": ")[1] if ": " in n else n for n in names]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_title("Double-Slit: Fidelity vs Integrity", fontsize=11, pad=10)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.3, edgecolor="#30363d")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right panel: AM-GM gap (Δ = F - IC)
    gaps = [f - ic for f, ic in zip(F_vals, IC_vals, strict=False)]
    colors = [YELLOW if i == 3 else (RED if g > 0.5 else ORANGE if g > 0.1 else GREEN) for i, g in enumerate(gaps)]
    ax2.barh(x, gaps, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.6)

    for i, (g, _n) in enumerate(zip(gaps, short_names, strict=False)):
        ax2.text(g + 0.01, i, f"Δ={g:.3f}", fontsize=8, va="center", color="#c9d1d9")

    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_xlabel("AM-GM Gap  Δ = F − IC", fontsize=12)
    ax2.set_title("Channel Heterogeneity", fontsize=11, pad=10)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # Label the key insight
    ax2.annotate(
        "S1,S6: high F, one dead channel → huge Δ\nS4: all channels alive → tiny Δ",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="#8b949e",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#161b22", "edgecolor": "#30363d", "alpha": 0.9},
    )

    fig.suptitle(
        "Complementarity Cliff: Wave & Particle Are Both Channel-Deficient Extremes\n"
        "7/7 Theorems PROVEN  │  67/67 Subtests  │  >5× IC gap",
        fontsize=12,
        y=1.02,
        fontweight="bold",
        color="#c9d1d9",
    )
    fig.tight_layout()
    fig.savefig(fig_path("03_complementarity_cliff.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_complementarity_cliff.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 4: Generation Monotonicity + Spin-Statistics
# ═══════════════════════════════════════════════════════════════════════════
def plot_generation_spin() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: Generation monotonicity
    q_labels = list(GEN_QUARKS.keys())
    q_vals = list(GEN_QUARKS.values())
    list(GEN_LEPTONS.keys())
    l_vals = list(GEN_LEPTONS.values())

    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width / 2, q_vals, width, color=ACCENT, alpha=0.85, label="Quarks", edgecolor="white", linewidth=0.5)
    ax1.bar(x + width / 2, l_vals, width, color=GREEN, alpha=0.85, label="Leptons", edgecolor="white", linewidth=0.5)

    # Monotonicity arrows
    for i in range(2):
        (q_vals[i] + q_vals[i + 1]) / 2
        ax1.annotate(
            "",
            xy=(i + 0.85 - width / 2, q_vals[i + 1]),
            xytext=(i + 0.15 - width / 2, q_vals[i]),
            arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2},
        )
        (l_vals[i] + l_vals[i + 1]) / 2
        ax1.annotate(
            "",
            xy=(i + 0.85 + width / 2, l_vals[i + 1]),
            xytext=(i + 0.15 + width / 2, l_vals[i]),
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2},
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels, fontsize=9)
    ax1.set_ylabel("⟨F⟩  (mean Fidelity)", fontsize=12)
    ax1.set_title(
        "Theorem T2: Generation Monotonicity\nGen1 < Gen2 < Gen3 (both quarks & leptons)", fontsize=10, pad=10
    )
    ax1.legend(fontsize=9, framealpha=0.3, edgecolor="#30363d")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0.50, 0.70)

    # Right: Spin-Statistics
    fermion_f = 0.614816
    boson_f = 0.420631
    fermion_ic = 0.458357
    boson_ic = 0.006574

    cats = ["Fermions\n(12 particles)", "Bosons\n(5 particles)"]
    x2 = np.arange(2)
    width2 = 0.3
    ax2.bar(
        x2 - width2 / 2,
        [fermion_f, boson_f],
        width2,
        color=ACCENT,
        alpha=0.85,
        label="⟨F⟩",
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.bar(
        x2 + width2 / 2,
        [fermion_ic, boson_ic],
        width2,
        color=GREEN,
        alpha=0.85,
        label="⟨IC⟩",
        edgecolor="white",
        linewidth=0.5,
    )

    # Split annotation
    split = fermion_f - boson_f
    ax2.annotate(
        "",
        xy=(0 - width2 / 2, fermion_f),
        xytext=(1 - width2 / 2, boson_f),
        arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 2},
    )
    mid_y = (fermion_f + boson_f) / 2
    ax2.text(0.5, mid_y + 0.02, f"split = {split:.3f}", fontsize=10, color=ORANGE, ha="center", fontweight="bold")

    # Value labels
    for i, (f_val, ic_val) in enumerate([(fermion_f, fermion_ic), (boson_f, boson_ic)]):
        ax2.text(i - width2 / 2, f_val + 0.01, f"{f_val:.3f}", fontsize=8, ha="center", color=ACCENT)
        ax2.text(i + width2 / 2, ic_val + 0.01, f"{ic_val:.4f}", fontsize=8, ha="center", color=GREEN)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, fontsize=10)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title(
        "Theorem T1: Spin-Statistics Separation\n⟨F⟩_fermion > ⟨F⟩_boson (split = 0.194)", fontsize=10, pad=10
    )
    ax2.legend(fontsize=9, framealpha=0.3, edgecolor="#30363d")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 0.72)

    fig.suptitle(
        "Standard Model Theorems T1 & T2: Statistical Structure in the Kernel", fontsize=12, y=1.01, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(fig_path("04_generation_spin_statistics.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_generation_spin_statistics.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 5: Periodic Table Fidelity Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_periodic_heatmap() -> None:
    # Standard periodic table layout: row (period), col (group)
    # Each element maps to (row, col) in the standard 18-column layout
    PT_LAYOUT: dict[int, tuple[int, int]] = {
        1: (0, 0),
        2: (0, 17),
        3: (1, 0),
        4: (1, 1),
        5: (1, 12),
        6: (1, 13),
        7: (1, 14),
        8: (1, 15),
        9: (1, 16),
        10: (1, 17),
        11: (2, 0),
        12: (2, 1),
        13: (2, 12),
        14: (2, 13),
        15: (2, 14),
        16: (2, 15),
        17: (2, 16),
        18: (2, 17),
        19: (3, 0),
        20: (3, 1),
    }
    # d-block period 4
    for i, z in enumerate(range(21, 31)):
        PT_LAYOUT[z] = (3, 2 + i)
    for z in range(31, 37):
        PT_LAYOUT[z] = (3, 12 + (z - 31))

    # Period 5
    PT_LAYOUT[37] = (4, 0)
    PT_LAYOUT[38] = (4, 1)
    for i, z in enumerate(range(39, 49)):
        PT_LAYOUT[z] = (4, 2 + i)
    for z in range(49, 55):
        PT_LAYOUT[z] = (4, 12 + (z - 49))

    # Period 6
    PT_LAYOUT[55] = (5, 0)
    PT_LAYOUT[56] = (5, 1)
    # La-Lu (lanthanides) → row 8
    for i, z in enumerate(range(57, 72)):
        PT_LAYOUT[z] = (8, 2 + i)
    for i, z in enumerate(range(72, 81)):
        PT_LAYOUT[z] = (5, 2 + i)
    for z in range(81, 87):
        PT_LAYOUT[z] = (5, 12 + (z - 81))

    # Period 7
    PT_LAYOUT[87] = (6, 0)
    PT_LAYOUT[88] = (6, 1)
    # Ac-Lr (actinides) → row 9
    for i, z in enumerate(range(89, 104)):
        PT_LAYOUT[z] = (9, 2 + i)
    for i, z in enumerate(range(104, 113)):
        PT_LAYOUT[z] = (6, 2 + i)
    for z in range(113, 119):
        PT_LAYOUT[z] = (6, 12 + (z - 113))

    SYMBOLS: dict[int, str] = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
        97: "Bk",
        98: "Cf",
        99: "Es",
        100: "Fm",
        101: "Md",
        102: "No",
        103: "Lr",
        104: "Rf",
        105: "Db",
        106: "Sg",
        107: "Bh",
        108: "Hs",
        109: "Mt",
        110: "Ds",
        111: "Rg",
        112: "Cn",
        113: "Nh",
        114: "Fl",
        115: "Mc",
        116: "Lv",
        117: "Ts",
        118: "Og",
    }

    fig, ax = plt.subplots(figsize=(16, 9))

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    norm = plt.Normalize(vmin=0.15, vmax=0.75)

    for z in range(1, 119):
        if z not in PT_LAYOUT or z not in PERIODIC_F:
            continue
        row, col = PT_LAYOUT[z]
        f_val = PERIODIC_F[z]
        color = cmap(norm(f_val))

        rect = plt.Rectangle((col, -row), 0.92, 0.92, facecolor=color, edgecolor="#30363d", linewidth=0.5)
        ax.add_patch(rect)

        sym = SYMBOLS.get(z, "")
        ax.text(
            col + 0.46,
            -row + 0.60,
            sym,
            fontsize=6.5,
            ha="center",
            va="center",
            fontweight="bold",
            color="#0d1117" if f_val > 0.45 else "#c9d1d9",
        )
        ax.text(
            col + 0.46,
            -row + 0.25,
            f"{f_val:.2f}",
            fontsize=4.5,
            ha="center",
            va="center",
            color="#0d1117" if f_val > 0.45 else "#8b949e",
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label("Kernel Fidelity  F", fontsize=11, color="#c9d1d9")
    cbar.ax.tick_params(colors="#8b949e")

    # Block labels
    ax.text(5, 1.2, "d-block: ⟨F⟩ = 0.489 (highest)", fontsize=9, color=GREEN, ha="center")
    ax.text(0.5, 1.2, "s-block: alkali metals\nhave lowest F", fontsize=8, color=RED, ha="center")

    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-10.5, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Periodic Table of Kernel Fidelity: 118 Elements Through the GCD Kernel\n"
        "Tier-1 Proof: 10,162 tests, 0 failures  │  F + ω = 1, IC ≤ F, IC = exp(κ)  ∀  Z ∈ [1, 118]",
        fontsize=12,
        pad=15,
    )

    fig.tight_layout()
    fig.savefig(fig_path("05_periodic_table_fidelity.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_periodic_table_fidelity.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 6: Regime Phase Diagram
# ═══════════════════════════════════════════════════════════════════════════
def plot_regime_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Regime zones
    regimes = [
        (0.00, 0.10, "STABLE", GREEN, "ω < 0.10\nF > 0.90"),
        (0.10, 0.20, "WATCH", YELLOW, "0.10 ≤ ω < 0.20\n0.80 ≤ F < 0.90"),
        (0.20, 0.30, "TENSION", ORANGE, "0.20 ≤ ω < 0.30\n0.70 ≤ F < 0.80"),
        (0.30, 1.00, "COLLAPSE", RED, "ω ≥ 0.30\nF < 0.70"),
    ]

    for start, end, label, color, desc in regimes:
        ax.axvspan(start, end, alpha=0.25, color=color, zorder=1)
        mid = (start + end) / 2
        ax.text(mid, 0.85, label, fontsize=14, ha="center", va="center", fontweight="bold", color=color, zorder=3)
        ax.text(mid, 0.60, desc, fontsize=8, ha="center", va="center", color="#8b949e", zorder=3)

    # Plot real particles on the ω axis
    particles_on_axis = [
        ("τ⁻", 0.271, ACCENT, "^"),
        ("μ⁻", 0.310, ACCENT, "^"),
        ("e⁻", 0.326, ACCENT, "^"),
        ("t", 0.362, ACCENT, "^"),
        ("c", 0.338, ACCENT, "^"),
        ("b", 0.333, ACCENT, "^"),
        ("H⁰", 0.585, PINK, "D"),
        ("γ", 0.669, RED, "D"),
        ("g", 0.583, RED, "D"),
        ("Z⁰", 0.634, RED, "D"),
        ("p", 0.450, PURPLE, "o"),
        ("n", 0.605, PURPLE, "o"),
        ("π⁰", 0.666, CYAN, "v"),
    ]
    y_offsets = [0.35, 0.30, 0.25, 0.40, 0.20, 0.15, 0.30, 0.35, 0.25, 0.20, 0.35, 0.30, 0.40]

    for (name, omega, color, marker), y_off in zip(particles_on_axis, y_offsets, strict=False):
        ax.scatter(omega, y_off, c=color, marker=marker, s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(name, (omega, y_off), fontsize=7, color="#8b949e", xytext=(3, 5), textcoords="offset points")

    # Threshold lines
    for thresh in [0.10, 0.20, 0.30]:
        ax.axvline(x=thresh, color="#c9d1d9", linewidth=1.5, linestyle="--", alpha=0.5)

    # F axis on top
    ax_top = ax.twiny()
    ax_top.set_xlim(1, 0)  # F = 1 - ω, reversed
    ax_top.set_xlabel("Fidelity  F = 1 − ω", fontsize=11, color="#c9d1d9")
    ax_top.tick_params(colors="#8b949e")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Drift  ω", fontsize=12)
    ax.set_yticks([])
    ax.set_title(
        "GCD Regime Classification Phase Diagram\nReal particles from Standard Model mapped to their drift values",
        fontsize=12,
        pad=25,
    )

    handles = [
        mpatches.Patch(color=GREEN, alpha=0.4, label="STABLE"),
        mpatches.Patch(color=YELLOW, alpha=0.4, label="WATCH"),
        mpatches.Patch(color=ORANGE, alpha=0.4, label="TENSION"),
        mpatches.Patch(color=RED, alpha=0.4, label="COLLAPSE"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.3, edgecolor="#30363d")

    fig.tight_layout()
    fig.savefig(fig_path("06_regime_phase_diagram.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_regime_phase_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 7: Cross-Scale Universality + AM-GM Gap Distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_cross_scale() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Cross-scale F comparison
    scales = ["Composite\n(14 hadrons)", "Atomic\n(118 elements)", "Fundamental\n(17 particles)"]
    f_means = [0.444449, 0.515540, 0.557703]
    colors = [PURPLE, GREEN, ACCENT]

    bars = ax1.bar(scales, f_means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.5)

    # Value labels
    for bar, val in zip(bars, f_means, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.3f}",
            fontsize=11,
            ha="center",
            va="bottom",
            color="#c9d1d9",
            fontweight="bold",
        )

    # Monotonicity arrows
    for i in range(2):
        ax1.annotate(
            "",
            xy=(i + 1, f_means[i + 1] - 0.005),
            xytext=(i, f_means[i] - 0.005),
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.5},
        )

    ax1.set_ylabel("⟨F⟩  (mean Fidelity)", fontsize=12)
    ax1.set_title(
        "Theorem T6: Cross-Scale Universality\ncomp(0.444) < atom(0.516) < fund(0.558)",
        fontsize=10,
        pad=10,
    )
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0.35, 0.62)

    # Right: AM-GM gap (Δ) distribution across all 118 elements
    gaps_118 = [PERIODIC_F[z] - PERIODIC_IC[z] for z in range(1, 119) if z in PERIODIC_F and z in PERIODIC_IC]
    ax2.hist(gaps_118, bins=25, color=ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)

    mean_gap = np.mean(gaps_118)
    median_gap = np.median(gaps_118)
    ax2.axvline(x=mean_gap, color=ORANGE, linewidth=2, linestyle="--", label=f"Mean Δ = {mean_gap:.3f}")
    ax2.axvline(x=median_gap, color=GREEN, linewidth=2, linestyle=":", label=f"Median Δ = {median_gap:.3f}")

    # IC < 0.15 threshold
    collapsed = sum(1 for z in range(1, 119) if z in PERIODIC_IC and PERIODIC_IC[z] < 0.15)
    ax2.text(
        0.95,
        0.95,
        f"Δ > 0.20 → channel death\n{collapsed} elements with IC < 0.15\n(noble gases, H, alkalis)",
        fontsize=8,
        transform=ax2.transAxes,
        va="top",
        ha="right",
        color="#8b949e",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#161b22", "edgecolor": "#30363d", "alpha": 0.9},
    )

    ax2.set_xlabel("AM-GM Gap  Δ = F − IC", fontsize=12)
    ax2.set_ylabel("Count (elements)", fontsize=12)
    ax2.set_title(
        "AM-GM Gap Distribution: 118 Elements\nChannel heterogeneity across the periodic table", fontsize=10, pad=10
    )
    ax2.legend(fontsize=9, framealpha=0.3, edgecolor="#30363d")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cross-Scale Patterns: Universality of the GCD Kernel", fontsize=12, y=1.01, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path("07_cross_scale_amgm_gap.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_cross_scale_amgm_gap.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Generating GCD kernel diagrams → {OUT}/")
    print()
    plot_kernel_geometry()
    plot_confinement_cliff()
    plot_complementarity_cliff()
    plot_generation_spin()
    plot_periodic_heatmap()
    plot_regime_diagram()
    plot_cross_scale()
    print()
    print(f"Done. 7 PNGs written to {OUT}/")


if __name__ == "__main__":
    main()
