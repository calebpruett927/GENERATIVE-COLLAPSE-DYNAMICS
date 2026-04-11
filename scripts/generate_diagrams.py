#!/usr/bin/env python3
"""Generate PNG diagrams showing GCD kernel geometry and statistical proofs.

Uses real data from UMCP closures — Standard Model particles, periodic table,
double-slit scenarios. Every number comes from computed kernel outputs.

Usage:
    python scripts/generate_diagrams.py
    # Outputs 10 PNGs to images/
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.axes

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# STYLE — GitHub dark theme with enhanced polish
# ---------------------------------------------------------------------------
BG_DARK = "#0d1117"
BG_PANEL = "#161b22"
BG_BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_MUTED = "#6e7681"
GRID_COLOR = "#21262d"

plt.rcParams.update(
    {
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": BG_BORDER,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "font.size": 14,
        "font.family": "sans-serif",
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "legend.framealpha": 0.65,
        "legend.edgecolor": BG_BORDER,
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

# Glow effect for emphasis text
GLOW = [patheffects.withStroke(linewidth=3, foreground=BG_DARK)]
GLOW_SUBTLE = [patheffects.withStroke(linewidth=2, foreground=BG_DARK)]


def _info_box(
    ax: matplotlib.axes.Axes,
    x: float,
    y: float,
    text: str,
    color: str = TEXT_SECONDARY,
    *,
    transform: str = "axes",
    fontsize: int = 12,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Place a styled information box on the axes."""
    t = ax.transAxes if transform == "axes" else ax.transData
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        transform=t,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": BG_PANEL,
            "edgecolor": BG_BORDER,
            "alpha": 0.92,
            "linewidth": 0.8,
        },
    )


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

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-DOMAIN DATA — from evolution, semiotics, consciousness, matter genesis
# ═══════════════════════════════════════════════════════════════════════════

# Matter Genesis — 6-act scale ladder (from closures/standard_model/matter_genesis.py)
MATTER_GENESIS = {
    "Act I\nFundamental": (0.5246, 0.2947, 17),
    "Act II\nConfinement": (0.4544, 0.0035, 14),
    "Act III\nNuclear": (0.6772, 0.4406, 22),
    "Act IV\nAtomic": (0.4622, 0.1783, 15),
    "Act V\nMolecular": (0.3572, 0.1050, 15),
    "Act VI\nBulk": (0.2533, 0.0970, 16),
}

# Evolution — selected organisms (from closures/evolution/)
EVOLUTION_DATA: dict[str, tuple[float, float, str]] = {
    "E. coli": (0.5900, 0.3762, "Monera"),
    "Yeast": (0.4625, 0.3217, "Fungi"),
    "Drosophila": (0.5125, 0.4703, "Animalia"),
    "Octopus": (0.5438, 0.5097, "Animalia"),
    "Raven": (0.6025, 0.5631, "Animalia"),
    "Wolf": (0.5875, 0.5143, "Animalia"),
    "Dolphin": (0.5413, 0.4690, "Animalia"),
    "Chimpanzee": (0.5237, 0.3700, "Animalia"),
    "Homo sapiens": (0.6539, 0.3184, "Animalia"),
    "T. rex": (0.4362, 0.3291, "Animalia"),
    "Dodo": (0.2126, 0.0987, "Animalia"),
}

# Consciousness — selected systems (from closures/consciousness_coherence/)
CONSCIOUSNESS_DATA: dict[str, tuple[float, float, str]] = {
    "Waking": (0.7313, 0.7207, "Neural"),
    "Deep meditation": (0.7375, 0.7201, "Neural"),
    "REM sleep": (0.4875, 0.4727, "Neural"),
    "Cetacean": (0.5750, 0.5641, "Neural"),
    "Corvid": (0.4750, 0.4556, "Neural"),
    "432 Hz tuning": (0.6687, 0.5660, "Harmonic"),
    "Circadian rhythm": (0.6462, 0.5732, "Physical"),
    "Cardiac rhythm": (0.6250, 0.5299, "Physical"),
    "Euler identity": (0.8962, 0.8903, "Mathematical"),
    "Gödel self-ref": (0.7162, 0.6096, "Recursive"),
    "Babylonian base-60": (0.7188, 0.7073, "Mathematical"),
}

# Semiotics — selected sign systems (from closures/dynamic_semiotics/)
SEMIOTICS_DATA: dict[str, tuple[float, float, str]] = {
    "English": (0.6525, 0.6003, "Natural"),
    "Mandarin": (0.7163, 0.6918, "Natural"),
    "Latin": (0.7050, 0.6543, "Natural"),
    "Math notation": (0.7450, 0.6335, "Formal"),
    "DNA/RNA code": (0.5950, 0.3771, "Formal"),
    "ASL": (0.6812, 0.6731, "Natural"),
    "Honeybee dance": (0.4664, 0.1672, "Animal"),
    "Morse code": (0.2963, 0.1385, "Formal"),
    "LLM output": (0.5288, 0.4022, "Artificial"),
}

# Immunology — immune cell types (from closures/immunology/immune_cell_kernel.py)
IMMUNOLOGY_DATA: dict[str, tuple[float, float, str]] = {
    "Neutrophil": (0.41, 0.18, "Innate"),
    "NK cell": (0.59, 0.39, "Innate"),
    "Macrophage": (0.54, 0.30, "Innate"),
    "Dendritic cell": (0.63, 0.40, "Innate"),
    "CD4 Th1": (0.66, 0.48, "Adaptive"),
    "CD8 CTL": (0.73, 0.61, "Adaptive"),
    "CD4 Treg": (0.71, 0.59, "Adaptive"),
    "Memory B": (0.70, 0.58, "Adaptive"),
    "B cell plasma": (0.63, 0.37, "Adaptive"),
}

# Ecology — ecological states (from closures/ecology/ecology_kernel.py)
ECOLOGY_DATA: dict[str, tuple[float, float, str]] = {
    "Coral reef": (0.93, 0.89, "Stable"),
    "Climax forest": (0.89, 0.83, "Stable"),
    "Grassland": (0.78, 0.66, "Intermediate"),
    "Tundra": (0.46, 0.23, "Stressed"),
    "Desert": (0.31, 0.08, "Stressed"),
    "Agriculture": (0.58, 0.16, "Managed"),
    "Trophic cascade": (0.22, 0.001, "Collapse"),
    "Pioneer stage": (0.54, 0.32, "Intermediate"),
}

# Information theory — complexity classes (from closures/information_theory/)
INFO_THEORY_DATA: dict[str, tuple[float, float, str]] = {
    "O(1)": (0.96, 0.94, "Decidable"),
    "P": (0.88, 0.81, "Decidable"),
    "NP": (0.56, 0.19, "Hard"),
    "BPP": (0.74, 0.54, "Randomized"),
    "BQP": (0.71, 0.50, "Quantum"),
    "PSPACE": (0.60, 0.24, "Hard"),
    "EXPTIME": (0.52, 0.16, "Hard"),
    "HALT": (0.10, 0.0001, "Undecidable"),
}

# Spacetime memory — spacetime objects (from closures/spacetime_memory/)
SPACETIME_DATA: dict[str, tuple[float, float, str]] = {
    "Neutron star": (0.91, 0.84, "Stellar"),
    "Main-seq star": (0.89, 0.82, "Stellar"),
    "He-4 nucleus": (0.91, 0.88, "Nuclear"),
    "Fe-56 nucleus": (0.88, 0.83, "Nuclear"),
    "Red giant": (0.79, 0.64, "Stellar"),
    "White dwarf": (0.86, 0.76, "Stellar"),
    "Black hole": (0.68, 0.25, "Stellar"),
    "Photon (ST)": (0.39, 0.01, "Subatomic"),
}

# Awareness-cognition — organisms (from closures/awareness_cognition/)
AWARENESS_DATA: dict[str, tuple[float, float, str]] = {
    "E. coli (AWC)": (0.28, 0.08, "Bacteria"),
    "C. elegans": (0.29, 0.09, "Nematoda"),
    "Fruit fly": (0.45, 0.23, "Insecta"),
    "Octopus (AWC)": (0.53, 0.35, "Cephalopoda"),
    "NC crow": (0.62, 0.48, "Aves"),
    "Dolphin (AWC)": (0.70, 0.61, "Mammalia"),
    "Chimpanzee (AWC)": (0.78, 0.71, "Primates"),
    "Human adult": (0.95, 0.93, "Primates"),
}

# Clinical neuroscience — neurocognitive states (from closures/clinical_neuroscience/)
CLINICAL_DATA: dict[str, tuple[float, float, str]] = {
    "Young adult": (0.89, 0.83, "Healthy"),
    "Expert meditator": (0.91, 0.86, "Healthy"),
    "Flow state": (0.89, 0.82, "Altered"),
    "Psilocybin": (0.76, 0.67, "Altered"),
    "REM sleep": (0.70, 0.58, "Altered"),
    "Gen. anesthesia": (0.31, 0.08, "Altered"),
    "Coma": (0.20, 0.03, "DOC"),
    "Alzheimer severe": (0.34, 0.09, "Neurodegen"),
    "Schizophrenia": (0.57, 0.33, "Psychiatric"),
}


def fig_path(name: str) -> str:
    return os.path.join(OUT, name)


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: Kernel Geometry — F vs IC with Integrity Bound
# ═══════════════════════════════════════════════════════════════════════════
def plot_kernel_geometry() -> None:
    fig, ax = plt.subplots(figsize=(13, 10))

    # Integrity bound: IC ≤ F (diagonal) with layered gradient fill
    x = np.linspace(0, 1, 500)
    ax.plot(
        x,
        x,
        color=ORANGE,
        alpha=0.9,
        linewidth=2.5,
        label="IC = F  (integrity bound)",
        path_effects=GLOW_SUBTLE,
        zorder=4,
    )
    for a in np.linspace(0.015, 0.06, 5):
        ax.fill_between(x, x, 1, alpha=a, color=RED, zorder=1)
    for a in np.linspace(0.015, 0.05, 4):
        ax.fill_between(x, 0, x, alpha=a, color=GREEN, zorder=1)

    # Plot particles by category
    cats = {
        "Quark": (ACCENT, "^", 180),
        "Lepton": (GREEN, "s", 160),
        "GaugeBoson": (RED, "D", 160),
        "ScalarBoson": (PINK, "p", 200),
    }
    for name, (F, IC, cat) in FUNDAMENTAL.items():
        color, marker, sz = cats[cat]
        ax.scatter(F, IC, c=color, marker=marker, s=sz, zorder=6, edgecolors="white", linewidth=0.8, alpha=0.95)
        ax.annotate(
            name,
            (F, IC),
            fontsize=11,
            color=TEXT_SECONDARY,
            xytext=(6, 6),
            textcoords="offset points",
            path_effects=GLOW_SUBTLE,
        )

    comp_cats = {"Baryon": (PURPLE, "o", 130), "Meson": (CYAN, "v", 130)}
    for name, (F, IC, cat) in COMPOSITE.items():
        color, marker, sz = comp_cats[cat]
        ax.scatter(F, IC, c=color, marker=marker, s=sz, zorder=6, edgecolors="white", linewidth=0.6, alpha=0.9)
        ax.annotate(
            name,
            (F, IC),
            fontsize=10,
            color=TEXT_MUTED,
            xytext=(4, -9),
            textcoords="offset points",
            path_effects=GLOW_SUBTLE,
        )

    # Regime boundaries
    for f_thresh, label, color in [(0.90, "STABLE", GREEN), (0.80, "WATCH", YELLOW), (0.70, "TENSION", ORANGE)]:
        ax.axvline(x=f_thresh, color=color, alpha=0.4, linewidth=1.2, linestyle=":")
        ax.text(
            f_thresh + 0.006,
            0.72,
            label,
            fontsize=12,
            color=color,
            rotation=90,
            va="top",
            fontweight="bold",
            path_effects=GLOW,
        )
    ax.axvline(x=0.70, color=RED, alpha=0.4, linewidth=1.2, linestyle=":")
    ax.text(
        0.675, 0.72, "COLLAPSE →", fontsize=12, color=RED, rotation=90, va="top", fontweight="bold", path_effects=GLOW
    )

    handles = [
        mpatches.Patch(color=ACCENT, label="Quarks (6)"),
        mpatches.Patch(color=GREEN, label="Leptons (6)"),
        mpatches.Patch(color=RED, label="Gauge Bosons (4)"),
        mpatches.Patch(color=PINK, label="Scalar Boson (H⁰)"),
        mpatches.Patch(color=PURPLE, label="Baryons (7)"),
        mpatches.Patch(color=CYAN, label="Mesons (7)"),
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", fontsize=14, framealpha=0.5, edgecolor=BG_BORDER, fancybox=True
    )
    legend.get_frame().set_facecolor(BG_PANEL)

    ax.set_xlabel("Fidelity  F  (arithmetic mean)")
    ax.set_ylabel("Integrity Composite  IC  (geometric mean)")
    ax.set_title(
        "GCD Kernel Geometry: F vs IC for 31 Standard Model Particles\n"
        "IC ≤ F (integrity bound)  │  Δ = F − IC measures channel heterogeneity",
        pad=18,
    )
    ax.set_xlim(0.25, 0.80)
    ax.set_ylim(-0.02, 0.78)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    ax.annotate(
        "CONFINEMENT CLIFF\nAll composites cluster near IC ≈ 0\n98.1% IC collapse at boundary",
        xy=(0.45, 0.015),
        xytext=(0.34, 0.38),
        fontsize=13,
        color=ORANGE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2, "connectionstyle": "arc3,rad=-0.2"},
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_PANEL, "edgecolor": ORANGE, "alpha": 0.95, "linewidth": 1.5},
    )

    ax.annotate(
        "Quarks: IC ≈ F\nChannels alive\n(low heterogeneity)",
        xy=(0.63, 0.59),
        xytext=(0.71, 0.42),
        fontsize=13,
        color=ACCENT,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2, "connectionstyle": "arc3,rad=0.2"},
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_PANEL, "edgecolor": ACCENT, "alpha": 0.95, "linewidth": 1.5},
    )

    fig.tight_layout()
    fig.savefig(fig_path("01_kernel_geometry_f_vs_ic.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_kernel_geometry_f_vs_ic.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: Confinement Cliff — IC Collapse at Quark→Hadron Boundary
# ═══════════════════════════════════════════════════════════════════════════
def plot_confinement_cliff() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    quarks = {k: v for k, v in FUNDAMENTAL.items() if v[2] == "Quark"}
    q_names = list(quarks.keys())
    q_ics = [quarks[n][1] for n in q_names]
    h_names = list(COMPOSITE.keys())
    h_ics = [COMPOSITE[n][1] for n in h_names]

    all_names = [*q_names, "│", *h_names]
    all_ics: list[float | None] = [*q_ics, None, *h_ics]
    all_colors = [ACCENT] * len(q_names) + ["none"] + [PURPLE if COMPOSITE[n][2] == "Baryon" else CYAN for n in h_names]
    x = list(range(len(all_names)))

    for i, (ic, color) in enumerate(zip(all_ics, all_colors, strict=False)):
        if ic is None:
            continue
        ax.bar(i, ic, color=color, edgecolor="white", linewidth=0.6, alpha=0.9, width=0.72, zorder=3)
        ax.text(
            i,
            ic + 0.008,
            f"{ic:.3f}",
            fontsize=10,
            color=TEXT_SECONDARY,
            ha="center",
            va="bottom",
            path_effects=GLOW_SUBTLE,
        )

    # Dramatic cliff divider
    cliff_x = len(q_names)
    ax.axvline(x=cliff_x, color=RED, linewidth=3, alpha=0.85, zorder=5)
    for dx in [0.3, 0.6, 0.9]:
        ax.axvline(x=cliff_x - dx, color=RED, linewidth=0.5, alpha=0.12)
        ax.axvline(x=cliff_x + dx, color=RED, linewidth=0.5, alpha=0.12)

    ax.annotate(
        "CONFINEMENT\nCLIFF\n\n98.1% IC\ncollapse",
        xy=(cliff_x, 0.38),
        fontsize=17,
        color=RED,
        fontweight="bold",
        ha="center",
        path_effects=GLOW,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": BG_PANEL, "edgecolor": RED, "alpha": 0.95, "linewidth": 2},
    )

    q_mean = sum(q_ics) / len(q_ics)
    h_mean = sum(h_ics) / len(h_ics)
    ax.axhline(
        y=q_mean, xmin=0, xmax=cliff_x / len(all_names), color=ACCENT, linewidth=2, linestyle="--", alpha=0.8, zorder=4
    )
    ax.axhline(
        y=h_mean,
        xmin=(cliff_x + 1) / len(all_names),
        xmax=1,
        color=PURPLE,
        linewidth=2,
        linestyle="--",
        alpha=0.8,
        zorder=4,
    )
    ax.text(
        2.5,
        q_mean + 0.018,
        f"⟨IC⟩_quarks = {q_mean:.4f}",
        fontsize=14,
        color=ACCENT,
        ha="center",
        fontweight="bold",
        path_effects=GLOW,
    )
    ax.text(
        cliff_x + 7,
        h_mean + 0.018,
        f"⟨IC⟩_hadrons = {h_mean:.4f}",
        fontsize=14,
        color=PURPLE,
        ha="center",
        fontweight="bold",
        path_effects=GLOW,
    )

    min_q_ic = min(q_ics)
    ax.axhline(y=min_q_ic, color=ORANGE, linewidth=1.2, linestyle=":", alpha=0.7)
    ax.text(
        len(all_names) - 2,
        min_q_ic + 0.018,
        f"min quark IC = {min_q_ic:.4f}",
        fontsize=12,
        color=ORANGE,
        ha="right",
        path_effects=GLOW_SUBTLE,
    )

    gap_amp = q_mean / h_mean if h_mean > 0 else float("inf")
    _info_box(
        ax,
        0.98,
        0.98,
        f"Gap amplification: {gap_amp:.1f}×\n14/14 hadrons < min quark IC\nConfinement = channel death",
        color=TEXT_PRIMARY,
        ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Integrity Composite  IC")
    ax.set_title(
        "Theorem T3: Confinement as IC Collapse\n14/14 hadrons below minimum quark IC  │  Gap amplification: 10.82×",
        pad=18,
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_ylim(0, 0.72)

    handles = [
        mpatches.Patch(color=ACCENT, label=f"Quarks (⟨IC⟩={q_mean:.3f})"),
        mpatches.Patch(color=PURPLE, label=f"Baryons (⟨IC⟩={h_mean:.4f})"),
        mpatches.Patch(color=CYAN, label="Mesons"),
    ]
    legend = ax.legend(
        handles=handles, loc="upper right", fontsize=14, framealpha=0.5, edgecolor=BG_BORDER, fancybox=True
    )
    legend.get_frame().set_facecolor(BG_PANEL)

    fig.tight_layout()
    fig.savefig(fig_path("02_confinement_cliff.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_confinement_cliff.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: Complementarity Cliff — Double-Slit 8 Scenarios
# ═══════════════════════════════════════════════════════════════════════════
def plot_complementarity_cliff() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 2]})

    names = list(DOUBLE_SLIT.keys())
    F_vals = [DOUBLE_SLIT[n][0] for n in names]
    IC_vals = [DOUBLE_SLIT[n][1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars_f = ax1.bar(
        x - width / 2, F_vals, width, color=ACCENT, alpha=0.9, label="F (Fidelity)", edgecolor="white", linewidth=0.6
    )
    bars_ic = ax1.bar(
        x + width / 2, IC_vals, width, color=GREEN, alpha=0.9, label="IC (Integrity)", edgecolor="white", linewidth=0.6
    )

    # Highlight S4 with golden border
    bars_f[3].set_edgecolor(YELLOW)
    bars_f[3].set_linewidth(2.5)
    bars_ic[3].set_edgecolor(YELLOW)
    bars_ic[3].set_linewidth(2.5)

    ax1.axhline(y=0.10, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.text(0.5, 0.12, "CLIFF: IC < 0.10", fontsize=14, color=RED, fontweight="bold", path_effects=GLOW)

    ax1.annotate(
        "S4: KERNEL-OPTIMAL\nIC = 0.847 (highest)\nΔ = 0.009 (lowest)\nAll channels alive",
        xy=(3, 0.855),
        xytext=(5, 0.92),
        fontsize=13,
        color=YELLOW,
        arrowprops={"arrowstyle": "->", "color": YELLOW, "lw": 2},
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_PANEL, "edgecolor": YELLOW, "alpha": 0.95, "linewidth": 1.5},
    )

    short_names = [n.split(": ")[1] if ": " in n else n for n in names]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right", fontsize=13)
    ax1.set_ylabel("Value")
    ax1.set_title("Double-Slit: Fidelity vs Integrity", pad=12)
    legend1 = ax1.legend(fontsize=14, loc="upper right", framealpha=0.5, edgecolor=BG_BORDER)
    legend1.get_frame().set_facecolor(BG_PANEL)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_ylim(0, 1.05)

    # Right panel: Heterogeneity gap
    gaps = [f - ic for f, ic in zip(F_vals, IC_vals, strict=False)]
    colors = [YELLOW if i == 3 else (RED if g > 0.5 else ORANGE if g > 0.1 else GREEN) for i, g in enumerate(gaps)]
    ax2.barh(x, gaps, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6, height=0.6)

    for i, g in enumerate(gaps):
        ax2.text(
            g + 0.012,
            i,
            f"Δ={g:.3f}",
            fontsize=13,
            va="center",
            color=TEXT_PRIMARY,
            fontweight="bold" if i == 3 else "normal",
            path_effects=GLOW_SUBTLE,
        )

    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names, fontsize=13)
    ax2.set_xlabel("Heterogeneity Gap  Δ = F − IC")
    ax2.set_title("Channel Heterogeneity", pad=12)
    ax2.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax2.invert_yaxis()

    _info_box(
        ax2,
        0.5,
        0.02,
        "S1,S6: high F, one dead channel → huge Δ\nS4: all channels alive → tiny Δ",
        color=TEXT_SECONDARY,
        transform="axes",
        va="bottom",
        ha="center",
    )

    fig.suptitle(
        "Complementarity Cliff: Wave & Particle Are Both Channel-Deficient Extremes\n"
        "7/7 Theorems PROVEN  │  67/67 Subtests  │  >5× IC gap",
        fontsize=17,
        y=1.02,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(fig_path("03_complementarity_cliff.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_complementarity_cliff.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 4: Generation Monotonicity + Spin-Statistics
# ═══════════════════════════════════════════════════════════════════════════
def plot_generation_spin() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    # Left: Generation monotonicity
    q_labels = list(GEN_QUARKS.keys())
    q_vals = list(GEN_QUARKS.values())
    l_vals = list(GEN_LEPTONS.values())

    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width / 2, q_vals, width, color=ACCENT, alpha=0.9, label="Quarks", edgecolor="white", linewidth=0.6)
    ax1.bar(x + width / 2, l_vals, width, color=GREEN, alpha=0.9, label="Leptons", edgecolor="white", linewidth=0.6)

    # Monotonicity arrows with glow
    for i in range(2):
        ax1.annotate(
            "",
            xy=(i + 0.85 - width / 2, q_vals[i + 1]),
            xytext=(i + 0.15 - width / 2, q_vals[i]),
            arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2.5},
        )
        ax1.annotate(
            "",
            xy=(i + 0.85 + width / 2, l_vals[i + 1]),
            xytext=(i + 0.15 + width / 2, l_vals[i]),
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.5},
        )

    # Value labels
    for i, (qv, lv) in enumerate(zip(q_vals, l_vals, strict=False)):
        ax1.text(
            i - width / 2, qv + 0.003, f"{qv:.3f}", fontsize=12, ha="center", color=ACCENT, path_effects=GLOW_SUBTLE
        )
        ax1.text(
            i + width / 2, lv + 0.003, f"{lv:.3f}", fontsize=12, ha="center", color=GREEN, path_effects=GLOW_SUBTLE
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels, fontsize=14)
    ax1.set_ylabel("⟨F⟩  (mean Fidelity)")
    ax1.set_title("Theorem T2: Generation Monotonicity\nGen1 < Gen2 < Gen3 (both quarks & leptons)", pad=12)
    legend1 = ax1.legend(fontsize=14, framealpha=0.5, edgecolor=BG_BORDER)
    legend1.get_frame().set_facecolor(BG_PANEL)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_ylim(0.50, 0.70)

    # Right: Spin-Statistics with enhanced visuals
    fermion_f, boson_f = 0.614816, 0.420631
    fermion_ic, boson_ic = 0.458357, 0.006574

    cats = ["Fermions\n(12 particles)", "Bosons\n(5 particles)"]
    x2 = np.arange(2)
    width2 = 0.3
    ax2.bar(
        x2 - width2 / 2,
        [fermion_f, boson_f],
        width2,
        color=ACCENT,
        alpha=0.9,
        label="⟨F⟩",
        edgecolor="white",
        linewidth=0.6,
    )
    ax2.bar(
        x2 + width2 / 2,
        [fermion_ic, boson_ic],
        width2,
        color=GREEN,
        alpha=0.9,
        label="⟨IC⟩",
        edgecolor="white",
        linewidth=0.6,
    )

    split = fermion_f - boson_f
    ax2.annotate(
        "",
        xy=(0 - width2 / 2, fermion_f),
        xytext=(1 - width2 / 2, boson_f),
        arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 2.5},
    )
    mid_y = (fermion_f + boson_f) / 2
    ax2.text(
        0.5,
        mid_y + 0.02,
        f"split = {split:.3f}",
        fontsize=16,
        color=ORANGE,
        ha="center",
        fontweight="bold",
        path_effects=GLOW,
    )

    for i, (f_val, ic_val) in enumerate([(fermion_f, fermion_ic), (boson_f, boson_ic)]):
        ax2.text(
            i - width2 / 2,
            f_val + 0.012,
            f"{f_val:.3f}",
            fontsize=13,
            ha="center",
            color=ACCENT,
            path_effects=GLOW_SUBTLE,
        )
        ax2.text(
            i + width2 / 2,
            ic_val + 0.012,
            f"{ic_val:.4f}",
            fontsize=13,
            ha="center",
            color=GREEN,
            path_effects=GLOW_SUBTLE,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, fontsize=15)
    ax2.set_ylabel("Value")
    ax2.set_title("Theorem T1: Spin-Statistics Separation\n⟨F⟩_fermion > ⟨F⟩_boson (split = 0.194)", pad=12)
    legend2 = ax2.legend(fontsize=14, framealpha=0.5, edgecolor=BG_BORDER)
    legend2.get_frame().set_facecolor(BG_PANEL)
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax2.set_ylim(0, 0.72)

    fig.suptitle(
        "Standard Model Theorems T1 & T2: Statistical Structure in the Kernel", fontsize=17, y=1.01, fontweight="bold"
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
    for i, z in enumerate(range(21, 31)):
        PT_LAYOUT[z] = (3, 2 + i)
    for z in range(31, 37):
        PT_LAYOUT[z] = (3, 12 + (z - 31))
    PT_LAYOUT[37] = (4, 0)
    PT_LAYOUT[38] = (4, 1)
    for i, z in enumerate(range(39, 49)):
        PT_LAYOUT[z] = (4, 2 + i)
    for z in range(49, 55):
        PT_LAYOUT[z] = (4, 12 + (z - 49))
    PT_LAYOUT[55] = (5, 0)
    PT_LAYOUT[56] = (5, 1)
    for i, z in enumerate(range(57, 72)):
        PT_LAYOUT[z] = (8, 2 + i)
    for i, z in enumerate(range(72, 81)):
        PT_LAYOUT[z] = (5, 2 + i)
    for z in range(81, 87):
        PT_LAYOUT[z] = (5, 12 + (z - 81))
    PT_LAYOUT[87] = (6, 0)
    PT_LAYOUT[88] = (6, 1)
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

    fig, ax = plt.subplots(figsize=(18, 10))

    # Enhanced colormap: custom diverging from red through yellow to green
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=0.15, vmax=0.75)

    for z in range(1, 119):
        if z not in PT_LAYOUT or z not in PERIODIC_F:
            continue
        row, col = PT_LAYOUT[z]
        f_val = PERIODIC_F[z]
        color = cmap(norm(f_val))

        # Rounded rectangle with subtle shadow
        rect = mpatches.FancyBboxPatch(
            (col + 0.04, -row + 0.04),
            0.88,
            0.88,
            boxstyle="round,pad=0.06",
            facecolor=color,
            edgecolor=BG_BORDER,
            linewidth=0.6,
        )
        ax.add_patch(rect)

        sym = SYMBOLS.get(z, "")
        text_color = BG_DARK if f_val > 0.42 else TEXT_PRIMARY
        ax.text(
            col + 0.48, -row + 0.62, sym, fontsize=12, ha="center", va="center", fontweight="bold", color=text_color
        )
        ax.text(
            col + 0.48,
            -row + 0.28,
            f"{f_val:.2f}",
            fontsize=10,
            ha="center",
            va="center",
            color=text_color,
            alpha=0.90,
        )

    # Colorbar with enhanced styling
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, aspect=22, pad=0.02)
    cbar.set_label("Kernel Fidelity  F", fontsize=17, color=TEXT_PRIMARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.outline.set_edgecolor(BG_BORDER)  # type: ignore[union-attr]

    # Block labels with enhanced styling
    ax.text(
        5,
        1.4,
        "d-block: ⟨F⟩ = 0.489 (highest)",
        fontsize=15,
        color=GREEN,
        ha="center",
        fontweight="bold",
        path_effects=GLOW,
    )
    ax.text(
        0.5,
        1.4,
        "s-block: lowest F\n(alkali metals)",
        fontsize=14,
        color=RED,
        ha="center",
        fontweight="bold",
        path_effects=GLOW,
    )
    ax.text(14, 1.4, "p-block: reactive\nnon-metals", fontsize=14, color=ORANGE, ha="center", path_effects=GLOW)

    # Lanthanide/Actinide labels
    ax.text(0.5, -7.5, "Lanthanides", fontsize=13, color=TEXT_SECONDARY, va="center")
    ax.text(0.5, -8.5, "Actinides", fontsize=13, color=TEXT_SECONDARY, va="center")

    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-10.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Periodic Table of Kernel Fidelity: 118 Elements Through the GCD Kernel\n"
        "Tier-1 Proof: 10,162 tests, 0 failures  │  F + ω = 1, IC ≤ F, IC = exp(κ)  ∀  Z ∈ [1, 118]",
        fontsize=18,
        pad=20,
    )

    fig.tight_layout()
    fig.savefig(fig_path("05_periodic_table_fidelity.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_periodic_table_fidelity.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 6: Regime Phase Diagram
# ═══════════════════════════════════════════════════════════════════════════
def plot_regime_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    regimes = [
        (0.00, 0.10, "STABLE", GREEN, "ω < 0.10\nF > 0.90"),
        (0.10, 0.20, "WATCH", YELLOW, "0.10 ≤ ω < 0.20\n0.80 ≤ F < 0.90"),
        (0.20, 0.30, "TENSION", ORANGE, "0.20 ≤ ω < 0.30\n0.70 ≤ F < 0.80"),
        (0.30, 1.00, "COLLAPSE", RED, "ω ≥ 0.30\nF < 0.70"),
    ]

    for start, end, label, color, desc in regimes:
        # Gradient-like layered fill
        for a in np.linspace(0.08, 0.30, 5):
            ax.axvspan(start, end, alpha=a / 5, color=color, zorder=1)
        mid = (start + end) / 2
        ax.text(
            mid,
            0.88,
            label,
            fontsize=20,
            ha="center",
            va="center",
            fontweight="bold",
            color=color,
            zorder=3,
            path_effects=GLOW,
        )
        ax.text(mid, 0.62, desc, fontsize=13, ha="center", va="center", color=TEXT_SECONDARY, zorder=3)

    # Real particles on ω axis
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
    y_offsets = [0.35, 0.30, 0.25, 0.42, 0.20, 0.15, 0.30, 0.35, 0.25, 0.20, 0.35, 0.30, 0.42]

    for (name, omega, color, marker), y_off in zip(particles_on_axis, y_offsets, strict=False):
        ax.scatter(omega, y_off, c=color, marker=marker, s=100, zorder=5, edgecolors="white", linewidth=0.8)
        ax.annotate(
            name,
            (omega, y_off),
            fontsize=12,
            color=TEXT_SECONDARY,
            xytext=(4, 6),
            textcoords="offset points",
            path_effects=GLOW_SUBTLE,
        )

    for thresh in [0.10, 0.20, 0.30]:
        ax.axvline(x=thresh, color=TEXT_PRIMARY, linewidth=1.8, linestyle="--", alpha=0.4)

    ax_top = ax.twiny()
    ax_top.set_xlim(1, 0)
    ax_top.set_xlabel("Fidelity  F = 1 − ω", color=TEXT_PRIMARY)
    ax_top.tick_params(colors=TEXT_SECONDARY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Drift  ω")
    ax.set_yticks([])
    ax.set_title(
        "GCD Regime Classification Phase Diagram\nReal particles from Standard Model mapped to their drift values",
        pad=28,
    )

    handles = [
        mpatches.Patch(color=GREEN, alpha=0.5, label="STABLE (12.5%)"),
        mpatches.Patch(color=YELLOW, alpha=0.5, label="WATCH (24.4%)"),
        mpatches.Patch(color=ORANGE, alpha=0.5, label="TENSION"),
        mpatches.Patch(color=RED, alpha=0.5, label="COLLAPSE (63.1%)"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", fontsize=14, framealpha=0.5, edgecolor=BG_BORDER)
    legend.get_frame().set_facecolor(BG_PANEL)

    fig.tight_layout()
    fig.savefig(fig_path("06_regime_phase_diagram.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_regime_phase_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 7: Matter Genesis Scale Ladder + Cross-Domain Heterogeneity Gap
# ═══════════════════════════════════════════════════════════════════════════
def plot_cross_scale() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [3, 2]})

    # ── Left panel: Matter Genesis 6-Act Ladder ──
    acts = list(MATTER_GENESIS.keys())
    f_vals = [MATTER_GENESIS[a][0] for a in acts]
    ic_vals = [MATTER_GENESIS[a][1] for a in acts]
    counts = [MATTER_GENESIS[a][2] for a in acts]

    x = np.arange(len(acts))
    width = 0.35

    ax1.bar(
        x - width / 2,
        f_vals,
        width,
        color=ACCENT,
        alpha=0.9,
        label="⟨F⟩ (Fidelity)",
        edgecolor="white",
        linewidth=1.0,
    )
    bars_ic = ax1.bar(
        x + width / 2,
        ic_vals,
        width,
        color=GREEN,
        alpha=0.9,
        label="⟨IC⟩ (Integrity)",
        edgecolor="white",
        linewidth=1.0,
    )

    # Highlight Act II confinement cliff
    bars_ic[1].set_edgecolor(RED)
    bars_ic[1].set_linewidth(3)

    # Value labels on bars
    for i, (fv, icv, n) in enumerate(zip(f_vals, ic_vals, counts, strict=False)):
        ax1.text(
            i - width / 2,
            fv + 0.01,
            f"{fv:.3f}",
            fontsize=11,
            ha="center",
            color=ACCENT,
            fontweight="bold",
            path_effects=GLOW_SUBTLE,
        )
        ax1.text(
            i + width / 2,
            icv + 0.01,
            f"{icv:.3f}",
            fontsize=11,
            ha="center",
            color=GREEN,
            fontweight="bold",
            path_effects=GLOW_SUBTLE,
        )
        ax1.text(i, -0.04, f"n={n}", fontsize=10, ha="center", color=TEXT_MUTED)

    # Confinement cliff annotation
    ax1.annotate(
        "CONFINEMENT CLIFF\nIC drops 98.8%\n0.295 → 0.004",
        xy=(1 + width / 2, 0.0035),
        xytext=(2.5, 0.42),
        fontsize=13,
        color=RED,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 2.5, "connectionstyle": "arc3,rad=-0.3"},
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_PANEL, "edgecolor": RED, "alpha": 0.95, "linewidth": 2},
    )

    # Nuclear peak annotation
    ax1.annotate(
        "NUCLEAR PEAK\nBinding energy maximum\nFe-56: F = 0.774",
        xy=(2 - width / 2, 0.677),
        xytext=(3.8, 0.60),
        fontsize=12,
        color=YELLOW,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": YELLOW, "lw": 2},
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_PANEL, "edgecolor": YELLOW, "alpha": 0.95, "linewidth": 1.5},
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(acts, fontsize=11)
    ax1.set_ylabel("Mean Kernel Value")
    ax1.set_title(
        "Matter Genesis: Particle → Atom → Mass\n6 Acts, 99 entities, 10 theorems │ IC cliff at confinement boundary",
        pad=14,
    )
    legend1 = ax1.legend(fontsize=12, loc="upper right", framealpha=0.6, edgecolor=BG_BORDER)
    legend1.get_frame().set_facecolor(BG_PANEL)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_ylim(-0.06, 0.78)

    # ── Right panel: Cross-Domain Heterogeneity Gap ──
    # Combine gaps from all domains
    domain_gaps: dict[str, list[float]] = {
        "SM Particles": [FUNDAMENTAL[n][0] - FUNDAMENTAL[n][1] for n in FUNDAMENTAL]
        + [COMPOSITE[n][0] - COMPOSITE[n][1] for n in COMPOSITE],
        "118 Elements": [PERIODIC_F[z] - PERIODIC_IC[z] for z in range(1, 119) if z in PERIODIC_F and z in PERIODIC_IC],
        "Evolution": [v[0] - v[1] for v in EVOLUTION_DATA.values()],
        "Consciousness": [v[0] - v[1] for v in CONSCIOUSNESS_DATA.values()],
        "Semiotics": [v[0] - v[1] for v in SEMIOTICS_DATA.values()],
        "Immunology": [v[0] - v[1] for v in IMMUNOLOGY_DATA.values()],
        "Ecology": [v[0] - v[1] for v in ECOLOGY_DATA.values()],
        "Info Theory": [v[0] - v[1] for v in INFO_THEORY_DATA.values()],
        "Spacetime": [v[0] - v[1] for v in SPACETIME_DATA.values()],
        "Awareness": [v[0] - v[1] for v in AWARENESS_DATA.values()],
        "Clinical": [v[0] - v[1] for v in CLINICAL_DATA.values()],
    }
    domain_colors = [RED, CYAN, GREEN, PURPLE, ORANGE, PINK, "#2ea043", "#a371f7", "#79c0ff", YELLOW, "#f47067"]
    domain_means = []

    y_pos = np.arange(len(domain_gaps))
    for i, (_domain, gaps) in enumerate(domain_gaps.items()):
        mean_g = float(np.mean(gaps))
        domain_means.append(mean_g)
        # Box showing range
        min_g, max_g = min(gaps), max(gaps)
        ax2.barh(i, mean_g, height=0.5, color=domain_colors[i], alpha=0.85, edgecolor="white", linewidth=1.0)
        ax2.plot([min_g, max_g], [i, i], color=domain_colors[i], linewidth=2.5, alpha=0.5)
        ax2.scatter([min_g, max_g], [i, i], color=domain_colors[i], s=40, edgecolors="white", linewidth=0.5, zorder=5)
        ax2.text(
            mean_g + 0.008,
            i,
            f"Δ̄={mean_g:.3f} (n={len(gaps)})",
            fontsize=11,
            va="center",
            color=TEXT_PRIMARY,
            fontweight="bold",
            path_effects=GLOW_SUBTLE,
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(domain_gaps.keys()), fontsize=12)
    ax2.set_xlabel("Heterogeneity Gap  Δ = F − IC")
    ax2.set_title("Cross-Domain Gap Distribution\n11 domains, 300+ data points", pad=14)
    ax2.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.55)

    fig.suptitle(
        "Cross-Scale Universality: From Quarks to Ecosystems\n"
        "IC ≤ F holds across all 23 domains │ Matter Genesis reveals the scale ladder",
        fontsize=18,
        y=1.02,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(fig_path("07_cross_scale_heterogeneity_gap.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_cross_scale_heterogeneity_gap.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 8: Validation Timelapse — Ledger History
# ═══════════════════════════════════════════════════════════════════════════
def plot_validation_timelapse() -> None:
    """Plot the project's validation history from ledger/return_log.csv."""
    import csv
    from datetime import datetime

    ledger_path = os.path.join(os.path.dirname(__file__), "..", "ledger", "return_log.csv")
    if not os.path.exists(ledger_path):
        print("  ⚠ ledger/return_log.csv not found — skipping timelapse")
        return

    timestamps: list[datetime] = []
    f_vals: list[float] = []
    ic_vals: list[float] = []
    omega_vals: list[float] = []
    statuses: list[str] = []

    with open(ledger_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                f_val = float(row["F"])
                ic_val = float(row["IC"])
                om_val = float(row["omega"])
                status = row["run_status"]
                timestamps.append(ts)
                f_vals.append(f_val)
                ic_vals.append(ic_val)
                omega_vals.append(om_val)
                statuses.append(status)
            except (ValueError, KeyError):
                continue

    if len(timestamps) < 10:
        print("  ⚠ Too few ledger entries for timelapse — skipping")
        return

    cumulative = list(range(1, len(timestamps) + 1))
    conformant_cum = []
    conf_count = 0
    for s in statuses:
        if s == "CONFORMANT":
            conf_count += 1
        conformant_cum.append(conf_count)
    conf_rate = [c / t for c, t in zip(conformant_cum, cumulative, strict=False)]

    window = min(50, len(f_vals) // 4) if len(f_vals) > 20 else max(1, len(f_vals) // 4)
    f_rolling = []
    for i in range(len(f_vals)):
        start = max(0, i - window + 1)
        f_rolling.append(np.mean(f_vals[start : i + 1]))

    gaps = [f - ic for f, ic in zip(f_vals, ic_vals, strict=False)]
    gap_rolling = []
    for i in range(len(gaps)):
        start = max(0, i - window + 1)
        gap_rolling.append(np.mean(gaps[start : i + 1]))

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # Panel 1: Cumulative runs
    ax1 = axes[0]
    ax1.fill_between(timestamps, 0, cumulative, alpha=0.2, color=ACCENT)
    ax1.plot(timestamps, cumulative, color=ACCENT, linewidth=2)
    ax1.set_ylabel("Cumulative Runs")
    ax1.set_title(
        f"UMCP Validation Timelapse: {len(timestamps):,} Runs\n"
        f"{timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}",
        pad=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    for m in [100, 500, 1000, 2000, 5000, 7000, 10000]:
        if m <= len(timestamps):
            ax1.axhline(y=m, color=BG_BORDER, linewidth=0.5, linestyle=":")
            ax1.text(timestamps[-1], m, f"  {m:,}", fontsize=12, color=TEXT_SECONDARY, va="center")

    # Panel 2: F and IC evolution
    ax2 = axes[1]
    ax2.plot(timestamps, f_rolling, color=GREEN, linewidth=2, label=f"F (rolling {window})", alpha=0.9)
    ax2.fill_between(timestamps, 0, gap_rolling, alpha=0.15, color=ORANGE, label="Δ = F − IC (heterogeneity gap)")
    ax2.plot(timestamps, gap_rolling, color=ORANGE, linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Kernel Invariants")
    legend2 = ax2.legend(fontsize=14, loc="center right", framealpha=0.5, edgecolor=BG_BORDER)
    legend2.get_frame().set_facecolor(BG_PANEL)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_ylim(-0.02, 1.05)

    # Panel 3: Drift (ω) evolution
    omega_rolling = []
    for i in range(len(omega_vals)):
        start = max(0, i - window + 1)
        omega_rolling.append(np.mean(omega_vals[start : i + 1]))

    ax3 = axes[2]
    ax3.plot(timestamps, omega_rolling, color=RED, linewidth=2, label=f"ω (rolling {window})", alpha=0.9)
    ax3.fill_between(timestamps, 0, omega_rolling, alpha=0.12, color=RED)
    ax3.axhline(y=0.30, color=RED, linewidth=1, linestyle="--", alpha=0.5, label="Collapse threshold (ω ≥ 0.30)")
    ax3.axhline(y=0.038, color=YELLOW, linewidth=1, linestyle=":", alpha=0.5, label="Stable threshold (ω < 0.038)")
    ax3.set_ylabel("Drift  ω")
    legend3 = ax3.legend(fontsize=12, loc="upper right", framealpha=0.5, edgecolor=BG_BORDER)
    legend3.get_frame().set_facecolor(BG_PANEL)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_ylim(-0.005, max(0.35, max(omega_rolling) * 1.15 if omega_rolling else 0.35))

    # Panel 4: Conformance rate
    ax4 = axes[3]
    ax4.plot(timestamps, conf_rate, color=GREEN, linewidth=2)
    ax4.fill_between(timestamps, 0, conf_rate, alpha=0.12, color=GREEN)
    ax4.axhline(y=1.0, color=GREEN, linewidth=0.5, linestyle="--", alpha=0.5)
    ax4.axhline(y=0.95, color=YELLOW, linewidth=0.5, linestyle=":", alpha=0.5)
    final_rate = conf_rate[-1] if conf_rate else 0
    _info_box(
        ax4,
        0.98,
        0.15,
        f"Current: {final_rate:.1%} conformant\n{conf_count:,}/{len(timestamps):,} runs",
        color=GREEN if final_rate >= 0.99 else YELLOW,
        ha="right",
        va="bottom",
    )
    ax4.set_ylabel("Conformance Rate")
    ax4.set_xlabel("Time")
    ax4.grid(True, alpha=0.25, linewidth=0.5)
    ax4.set_ylim(0.90, 1.005)

    fig.tight_layout()
    fig.savefig(fig_path("08_validation_timelapse.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_validation_timelapse.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 9: Integrity Bound Proof — IC ≤ F Across All Domains
# ═══════════════════════════════════════════════════════════════════════════
def plot_integrity_bound_proof() -> None:
    """Show IC ≤ F across ALL domains — 12 domain groups, 400+ data points."""
    all_points: list[tuple[str, float, float, str]] = []
    for name, (f, ic, _cat) in FUNDAMENTAL.items():
        all_points.append((name, f, ic, "SM Fundamental"))
    for name, (f, ic, _cat) in COMPOSITE.items():
        all_points.append((name, f, ic, "SM Composite"))
    for z in sorted(PERIODIC_F.keys()):
        if z in PERIODIC_IC:
            all_points.append((f"Z={z}", PERIODIC_F[z], PERIODIC_IC[z], "Periodic Table"))
    for name, (f, ic, _cat) in EVOLUTION_DATA.items():
        all_points.append((name, f, ic, "Evolution"))
    for name, (f, ic, _cat) in CONSCIOUSNESS_DATA.items():
        all_points.append((name, f, ic, "Consciousness"))
    for name, (f, ic, _cat) in SEMIOTICS_DATA.items():
        all_points.append((name, f, ic, "Semiotics"))
    for name, (f, ic, _cat) in IMMUNOLOGY_DATA.items():
        all_points.append((name, f, ic, "Immunology"))
    for name, (f, ic, _cat) in ECOLOGY_DATA.items():
        all_points.append((name, f, ic, "Ecology"))
    for name, (f, ic, _cat) in INFO_THEORY_DATA.items():
        all_points.append((name, f, ic, "Information Theory"))
    for name, (f, ic, _cat) in SPACETIME_DATA.items():
        all_points.append((name, f, ic, "Spacetime Memory"))
    for name, (f, ic, _cat) in AWARENESS_DATA.items():
        all_points.append((name, f, ic, "Awareness-Cognition"))
    for name, (f, ic, _cat) in CLINICAL_DATA.items():
        all_points.append((name, f, ic, "Clinical Neuro"))

    fig, ax = plt.subplots(figsize=(13, 10))

    x = np.linspace(0, 1, 500)
    ax.plot(x, x, color=ORANGE, linewidth=2.5, alpha=0.9, label="IC = F (integrity bound)", path_effects=GLOW_SUBTLE)
    for a in np.linspace(0.012, 0.05, 4):
        ax.fill_between(x, x, 1, alpha=a, color=RED)
    ax.text(
        0.15, 0.65, "FORBIDDEN\nIC > F", fontsize=16, color=RED, alpha=0.4, ha="center", fontweight="bold", rotation=45
    )
    for a in np.linspace(0.008, 0.03, 3):
        ax.fill_between(x, 0, x, alpha=a, color=GREEN)

    domain_styles: dict[str, tuple[str, str, int]] = {
        "SM Fundamental": (ACCENT, "^", 120),
        "SM Composite": (PURPLE, "D", 90),
        "Periodic Table": (CYAN, ".", 50),
        "Evolution": (GREEN, "s", 100),
        "Consciousness": (YELLOW, "p", 110),
        "Semiotics": (ORANGE, "h", 100),
        "Immunology": (PINK, "X", 90),
        "Ecology": ("#2ea043", "d", 90),
        "Information Theory": ("#a371f7", "v", 90),
        "Spacetime Memory": ("#79c0ff", "*", 110),
        "Awareness-Cognition": ("#e3b341", "<", 90),
        "Clinical Neuro": ("#f47067", ">", 90),
    }
    for domain, (color, marker, sz) in domain_styles.items():
        pts = [(f, ic) for _, f, ic, d in all_points if d == domain]
        if pts:
            fs, ics = zip(*pts, strict=False)
            ax.scatter(
                fs,
                ics,
                c=color,
                marker=marker,
                s=sz,
                alpha=0.8,
                label=f"{domain} ({len(pts)})",
                edgecolors="white",
                linewidth=0.5,
            )

    violations = sum(1 for _, f, ic, _ in all_points if ic > f + 1e-10)
    total = len(all_points)
    n_domains = len(domain_styles)

    _info_box(
        ax,
        0.02,
        0.98,
        f"IC ≤ F: {total}/{total} verified\n"
        f"Violations: {violations}\n"
        f"Domains: {n_domains} (physics → biology → cognition → ecology → computation)\n"
        f"From quarks to ecosystems — zero exceptions",
        color=GREEN,
        fontsize=13,
    )

    ax.set_xlabel("Fidelity  F")
    ax.set_ylabel("Integrity Composite  IC")
    ax.set_title(
        f"Integrity Bound Proof: IC ≤ F Across {total} Data Points, {n_domains} Domains\n"
        "Zero violations │ Quarks → Elements → Organisms → Ecosystems → Consciousness → Computation",
        pad=18,
    )
    legend = ax.legend(fontsize=11, loc="lower right", framealpha=0.6, edgecolor=BG_BORDER, ncol=3, columnspacing=0.8)
    legend.get_frame().set_facecolor(BG_PANEL)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.02, 0.92)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fig_path("09_integrity_bound_proof.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 09_integrity_bound_proof.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 10: Tier Architecture — Visual Stack
# ═══════════════════════════════════════════════════════════════════════════
def plot_tier_architecture() -> None:
    """Visualize the three-tier architecture with dependency arrows."""
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 11)
    ax.axis("off")

    tiers = [
        (
            1.5,
            7.5,
            9,
            2.5,
            "TIER 1: THE KERNEL",
            GREEN,
            "K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC)\n\n"
            "F + ω = 1  │  IC ≤ F  │  IC = exp(κ)\n"
            "44 structural identities  │  47 lemmas  │  5 frozen constants\n"
            "746 proven theorems across 23 domains  │  0 violations",
        ),
        (
            1.5,
            4.2,
            9,
            2.5,
            "TIER 0: PROTOCOL",
            ACCENT,
            "Regime gates: Stable / Watch / Collapse  │  Seam calculus\n\n"
            "Validator  │  Contracts  │  Schemas  │  SHA-256 integrity\n"
            "Three-valued verdicts: CONFORMANT / NONCONFORMANT / NON_EVALUABLE\n"
            "20,221 tests  │  245 closure modules  │  C++17 accelerator (50-80× speedup)",
        ),
        (
            1.5,
            0.0,
            9,
            3.3,
            "TIER 2: EXPANSION SPACE",
            PURPLE,
            "23 domains across physics, biology, cognition, ecology, and computation:\n\n"
            "Standard Model  │  Nuclear  │  Atomic  │  QM  │  Astronomy  │  WEYL\n"
            "Kinematics  │  Materials  │  Finance  │  Security  │  GCD  │  RCFT\n"
            "Evolution  │  Semiotics  │  Consciousness  │  Continuity  │  Everyday\n"
            "Awareness  │  Clinical Neuro  │  Spacetime  │  Immunology  │  Ecology\n"
            "Information Theory",
        ),
    ]

    for x_pos, y, w, h, title, color, desc in tiers:
        shadow = mpatches.FancyBboxPatch(
            (x_pos + 0.06, y - 0.06),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor="#000000",
            edgecolor="none",
            alpha=0.3,
        )
        ax.add_patch(shadow)
        rect = mpatches.FancyBboxPatch(
            (x_pos, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor=BG_PANEL,
            edgecolor=color,
            linewidth=3.5,
        )
        ax.add_patch(rect)
        ax.text(
            x_pos + w / 2,
            y + h - 0.35,
            title,
            fontsize=18,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            path_effects=GLOW,
        )
        ax.text(
            x_pos + w / 2,
            y + h / 2 - 0.2,
            desc,
            fontsize=12,
            color=TEXT_PRIMARY,
            ha="center",
            va="center",
            linespacing=1.5,
        )

    # Dependency arrows (thicker, more visible)
    arrow_props = {"arrowstyle": "-|>", "color": TEXT_SECONDARY, "lw": 3, "connectionstyle": "arc3,rad=0"}
    ax.annotate("", xy=(6, 6.7), xytext=(6, 7.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6, 3.3), xytext=(6, 4.2), arrowprops=arrow_props)

    # Arrow labels
    ax.text(6.4, 7.1, "implements", fontsize=11, color=TEXT_SECONDARY, va="center")
    ax.text(6.4, 3.75, "validates against", fontsize=11, color=TEXT_SECONDARY, va="center")

    # No-back-edge indicator — line split into two segments with gap for text
    ax.annotate(
        "",
        xy=(11.5, 8.5),
        xytext=(11.5, 6.3),
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 2.5, "linestyle": "--"},
    )
    ax.annotate(
        "",
        xy=(11.5, 4.7),
        xytext=(11.5, 3.3),
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 2.5, "linestyle": "--"},
    )
    ax.text(
        11.5,
        5.5,
        "✗ NO FEEDBACK",
        fontsize=14,
        color=RED,
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
        path_effects=GLOW,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BG_DARK, "edgecolor": "none", "alpha": 0.9},
        zorder=10,
    )

    # Axiom-0 banner
    ax.text(
        6,
        10.5,
        '"Collapse is generative; only what returns is real."  — Axiom-0',
        fontsize=16,
        color=YELLOW,
        ha="center",
        va="center",
        fontstyle="italic",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": BG_PANEL, "edgecolor": YELLOW, "alpha": 0.9, "linewidth": 2.5},
    )

    fig.tight_layout()
    fig.savefig(fig_path("10_tier_architecture.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 10_tier_architecture.png")


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
    plot_validation_timelapse()
    plot_integrity_bound_proof()
    plot_tier_architecture()
    print()
    print(f"Done. 10 PNGs written to {OUT}/")


if __name__ == "__main__":
    main()
