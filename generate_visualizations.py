"""
Generate PNG visualizations for UMCP system architecture and benchmarks.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Global style (professional, print-friendly)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.18,
    }
)

# Palette (subtle, modern)
PALETTE = {
    "tier0": "#3B82F6",
    "tier1": "#F59E0B",
    "tier2": "#EF4444",
    "accent": "#10B981",
    "light": "#F8FAFC",
    "dark": "#0F172A",
    "muted": "#64748B",
    "line": "#CBD5E1",
}


def draw_section(ax, x, y, w, h, title, subtitle, lines, color):
    # shadow
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.06),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0,
        facecolor="#000000",
        alpha=0.08,
        zorder=1,
    )
    ax.add_patch(shadow)

    # main card
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)

    # accent bar
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.22),
            w,
            0.22,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0,
            facecolor=color,
            alpha=0.9,
            zorder=3,
        )
    )

    ax.text(
        x + 0.35,
        y + h - 0.45,
        title,
        fontsize=12,
        fontweight="bold",
        color=PALETTE["dark"],
        zorder=4,
    )
    ax.text(
        x + 0.35, y + h - 0.75, subtitle, fontsize=9.5, color=PALETTE["muted"], zorder=4
    )

    text_y = y + h - 1.15
    for line in lines:
        ax.text(
            x + 0.45, text_y, f"• {line}", fontsize=9.3, color=PALETTE["dark"], zorder=4
        )
        text_y -= 0.34


def wrap_lines(lines, width=36):
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def create_architecture_diagram():
    """Create system architecture diagram."""
    _fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Background panel
    bg = FancyBboxPatch(
        (0.4, 0.3),
        15.2,
        8.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor=PALETTE["light"],
        zorder=0,
    )
    ax.add_patch(bg)

    # Title and subtitle
    ax.text(
        8,
        8.4,
        "UMCP System Architecture",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        8,
        8.05,
        "Contract-first validation across three tiers",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )

    draw_section(
        ax,
        0.9,
        6.0,
        14.2,
        1.9,
        title="Tier 0 — Base UMCP",
        subtitle="Core schema validation and provenance",
        lines=wrap_lines(
            [
                "canon/anchors.yaml",
                "contracts/UMA.INTSTACK.v1.yaml",
                "4 base closures",
                "casepacks/hello_world",
                "30 tests",
            ],
            width=42,
        ),
        color=PALETTE["tier0"],
    )

    draw_section(
        ax,
        0.9,
        3.6,
        14.2,
        2.1,
        title="Tier 1 — GCD (Generative Collapse Dynamics)",
        subtitle="Energy, collapse, flux, resonance",
        lines=wrap_lines(
            [
                "canon/gcd_anchors.yaml",
                "contracts/GCD.INTSTACK.v1.yaml",
                "energy_potential, entropic_collapse",
                "generative_flux, field_resonance",
                "casepacks/gcd_complete • 53 tests",
            ],
            width=42,
        ),
        color=PALETTE["tier1"],
    )

    draw_section(
        ax,
        0.9,
        1.0,
        14.2,
        2.1,
        title="Tier 2 — RCFT (Recursive Collapse Field Theory)",
        subtitle="Fractal, recursive, pattern analysis",
        lines=wrap_lines(
            [
                "canon/rcft_anchors.yaml",
                "contracts/RCFT.INTSTACK.v1.yaml",
                "fractal_dimension, recursive_field",
                "resonance_pattern",
                "inherits Tier 1 • casepacks/rcft_complete",
            ],
            width=42,
        ),
        color=PALETTE["tier2"],
    )

    # Vertical connectors
    ax.annotate(
        "",
        xy=(8, 5.9),
        xytext=(8, 5.7),
        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": PALETTE["muted"]},
    )
    ax.annotate(
        "",
        xy=(8, 3.3),
        xytext=(8, 3.1),
        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": PALETTE["muted"]},
    )

    # Math callouts (framework-accurate equations)
    eq_box = FancyBboxPatch(
        (7.1, 6.95),
        3.8,
        1.7,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1,
        edgecolor=PALETTE["line"],
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(eq_box)
    ax.text(
        7.25,
        8.25,
        "GCD Equations",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        7.25, 7.9, r"$E=\omega^2+\alpha S+\beta C^2$", fontsize=9, color=PALETTE["dark"]
    )
    ax.text(
        7.25,
        7.6,
        r"$\Phi_c=S(1-F)e^{-\tau_R/\tau_0}$",
        fontsize=9,
        color=PALETTE["dark"],
    )
    ax.text(
        7.25, 7.3, r"$\Phi_g=\kappa\sqrt{IC}(1+C^2)$", fontsize=9, color=PALETTE["dark"]
    )
    ax.text(
        7.25,
        7.0,
        r"$R=(1-|\omega|)(1-S)e^{-C/C_{crit}}$",
        fontsize=9,
        color=PALETTE["dark"],
    )

    rcft_box = FancyBboxPatch(
        (7.1, 2.2),
        3.8,
        1.4,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1,
        edgecolor=PALETTE["line"],
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(rcft_box)
    ax.text(
        7.25,
        3.4,
        "RCFT Equations",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        7.25,
        3.1,
        r"$D_f=\log N(\epsilon)/\log(1/\epsilon)$",
        fontsize=9,
        color=PALETTE["dark"],
    )
    ax.text(
        7.25, 2.8, r"$\Psi_r=\sum \alpha^n\Psi_n$", fontsize=9, color=PALETTE["dark"]
    )
    ax.text(
        7.25, 2.5, r"$(\lambda_p,\Theta)$ via FFT", fontsize=9, color=PALETTE["dark"]
    )

    # Footer summary
    footer = FancyBboxPatch(
        (0.9, 0.35),
        14.2,
        0.45,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1,
        edgecolor=PALETTE["accent"],
        facecolor="white",
    )
    ax.add_patch(footer)
    ax.text(
        1.2,
        0.53,
        "Totals: 221 tests • 11 closures • 3 contracts • 3 canon anchors",
        fontsize=9.5,
        color=PALETTE["dark"],
    )

    plt.tight_layout()
    plt.savefig(
        "architecture_diagram.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Created: architecture_diagram.png")
    plt.close()


def create_architecture_grid():
    """Create alternate architecture diagram (grid layout)."""
    _fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.4, 0.3),
        15.2,
        8.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor=PALETTE["light"],
        zorder=0,
    )
    ax.add_patch(bg)

    ax.text(
        8,
        8.4,
        "UMCP Architecture (Grid View)",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        8,
        8.05,
        "Components grouped by tier",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )

    col_w = 4.8
    row_h = 2.0
    draw_section(
        ax,
        0.9,
        5.6,
        col_w,
        row_h,
        title="Tier 0",
        subtitle="Base UMCP",
        lines=wrap_lines(
            [
                "Canon: anchors.yaml",
                "Contract: UMA.INTSTACK.v1",
                "Base closures (4)",
                "Casepack: hello_world",
            ],
            width=28,
        ),
        color=PALETTE["tier0"],
    )
    draw_section(
        ax,
        5.6,
        5.6,
        col_w,
        row_h,
        title="Tier 1",
        subtitle="GCD",
        lines=wrap_lines(
            [
                "Canon: gcd_anchors.yaml",
                "Contract: GCD.INTSTACK.v1",
                "Closures: energy, collapse",
                "flux, resonance",
            ],
            width=28,
        ),
        color=PALETTE["tier1"],
    )
    draw_section(
        ax,
        10.3,
        5.6,
        col_w,
        row_h,
        title="Tier 2",
        subtitle="RCFT",
        lines=wrap_lines(
            [
                "Canon: rcft_anchors.yaml",
                "Contract: RCFT.INTSTACK.v1",
                "Closures: fractal, recursive",
                "pattern",
            ],
            width=28,
        ),
        color=PALETTE["tier2"],
    )

    draw_section(
        ax,
        0.9,
        2.7,
        col_w,
        row_h,
        title="Validation",
        subtitle="Contract checks",
        lines=wrap_lines(
            [
                "Schema + semantic rules",
                "Identity checks",
                "Regime consistency",
            ],
            width=28,
        ),
        color=PALETTE["accent"],
    )
    draw_section(
        ax,
        5.6,
        2.7,
        col_w,
        row_h,
        title="Provenance",
        subtitle="Receipts",
        lines=wrap_lines(
            [
                "Git commit hash",
                "SHA256 checksums",
                "UTC timestamp",
            ],
            width=28,
        ),
        color=PALETTE["accent"],
    )
    draw_section(
        ax,
        10.3,
        2.7,
        col_w,
        row_h,
        title="Performance",
        subtitle="Benchmark",
        lines=wrap_lines(
            [
                "~7ms per validation",
                "221 tests pass",
                "Strict mode supported",
            ],
            width=28,
        ),
        color=PALETTE["accent"],
    )

    plt.savefig(
        "architecture_grid.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Created: architecture_grid.png")
    plt.close()


def create_benchmark_comparison():
    """Create benchmark comparison chart."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Speed comparison
    categories = ["Mean\nTime", "Median\nTime", "Max\nTime"]
    standard = [3.19, 2.76, 12.08]
    umcp = [7.26, 6.04, 17.63]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, standard, width, label="Standard", color="#2E86AB", alpha=0.8
    )
    bars2 = ax1.bar(
        x + width / 2, umcp, width, label="UMCP", color="#F77F00", alpha=0.8
    )

    ax1.set_ylabel("Time (milliseconds)", fontsize=11)
    ax1.set_title("Validation Speed Comparison", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Feature comparison
    features = [
        "Errors\nCaught",
        "Contract\nCheck",
        "Closure\nVerify",
        "Semantic\nRules",
        "Provenance\nTrack",
    ]
    standard_features = [400, 0, 0, 0, 0]
    umcp_features = [400, 400, 400, 400, 400]

    x2 = np.arange(len(features))

    ax2.bar(
        x2 - width / 2,
        standard_features,
        width,
        label="Standard",
        color="#2E86AB",
        alpha=0.8,
    )
    ax2.bar(
        x2 + width / 2, umcp_features, width, label="UMCP", color="#F77F00", alpha=0.8
    )

    ax2.set_ylabel("Checks Performed", fontsize=11)
    ax2.set_title("Validation Features Comparison", fontsize=13, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(features, fontsize=9)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "benchmark_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Created: benchmark_comparison.png")
    plt.close()


def create_workflow_diagram():
    """Create UMCP workflow diagram."""
    _fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    bg = FancyBboxPatch(
        (0.4, 0.3),
        13.2,
        8.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor=PALETTE["light"],
        zorder=0,
    )
    ax.add_patch(bg)

    ax.text(
        7,
        8.5,
        "UMCP Workflow",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        7,
        8.15,
        "From raw data to validated receipt",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )

    # Step 1: Input
    step1 = FancyBboxPatch(
        (1.0, 6.5),
        3.0,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        edgecolor=PALETTE["tier0"],
        facecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(step1)
    ax.text(2.5, 7.45, "Step 1", ha="center", fontweight="bold", fontsize=9)
    ax.text(2.5, 7.1, "Raw data", ha="center", fontsize=9)
    ax.text(2.5, 6.8, "CSV / JSON", ha="center", fontsize=8, color=PALETTE["muted"])

    # Arrow
    ax.annotate(
        "",
        xy=(4.5, 7.1),
        xytext=(4.1, 7.1),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": PALETTE["dark"]},
    )

    # Step 2: Invariants
    step2 = FancyBboxPatch(
        (4.7, 6.5),
        3.0,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        edgecolor=PALETTE["tier1"],
        facecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(step2)
    ax.text(6.2, 7.45, "Step 2", ha="center", fontweight="bold", fontsize=9)
    ax.text(6.2, 7.1, "Compute", ha="center", fontsize=9)
    ax.text(6.2, 6.8, "Invariants", ha="center", fontsize=8, color=PALETTE["muted"])

    # Arrow
    ax.annotate(
        "",
        xy=(8.2, 7.1),
        xytext=(7.8, 7.1),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": PALETTE["dark"]},
    )

    # Step 3: Closures
    step3 = FancyBboxPatch(
        (8.4, 6.5),
        3.0,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        edgecolor=PALETTE["tier2"],
        facecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(step3)
    ax.text(9.9, 7.45, "Step 3", ha="center", fontweight="bold", fontsize=9)
    ax.text(9.9, 7.1, "Execute", ha="center", fontsize=9)
    ax.text(9.9, 6.8, "Closures", ha="center", fontsize=8, color=PALETTE["muted"])

    # Arrow down from step 2
    ax.annotate(
        "",
        xy=(6.2, 5.8),
        xytext=(6.2, 6.4),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": PALETTE["dark"]},
    )

    # Step 4: Validation
    step4 = FancyBboxPatch(
        (4.7, 4.3),
        3.0,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        edgecolor=PALETTE["accent"],
        facecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(step4)
    ax.text(6.2, 5.25, "Step 4", ha="center", fontweight="bold", fontsize=9)
    ax.text(6.2, 4.9, "Validate", ha="center", fontsize=9)
    ax.text(6.2, 4.6, "Contract", ha="center", fontsize=8, color=PALETTE["muted"])

    # Arrow down
    ax.annotate(
        "",
        xy=(6.2, 3.5),
        xytext=(6.2, 4.1),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": PALETTE["dark"]},
    )

    # Step 5: Receipt
    step5 = FancyBboxPatch(
        (4.7, 2.1),
        3.0,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        edgecolor=PALETTE["tier0"],
        facecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(step5)
    ax.text(6.2, 3.05, "Step 5", ha="center", fontweight="bold", fontsize=9)
    ax.text(6.2, 2.7, "Generate", ha="center", fontsize=9)
    ax.text(6.2, 2.4, "Receipt", ha="center", fontsize=8, color=PALETTE["muted"])

    # Side info boxes
    # Canon
    canon_box = FancyBboxPatch(
        (0.9, 4.3),
        2.8,
        1.2,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        edgecolor=PALETTE["muted"],
        facecolor=PALETTE["light"],
        linewidth=1,
        linestyle="--",
    )
    ax.add_patch(canon_box)
    ax.text(2.3, 5.0, "Canon", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        2.3, 4.65, "anchors.yaml", ha="center", fontsize=7.5, color=PALETTE["muted"]
    )

    # Contract
    contract_box = FancyBboxPatch(
        (10.0, 4.3),
        2.8,
        1.2,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        edgecolor=PALETTE["muted"],
        facecolor=PALETTE["light"],
        linewidth=1,
        linestyle="--",
    )
    ax.add_patch(contract_box)
    ax.text(11.4, 5.0, "Contract", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        11.4,
        4.65,
        "GCD/RCFT.INTSTACK.v1",
        ha="center",
        fontsize=7.5,
        color=PALETTE["muted"],
    )

    # Registry
    registry_box = FancyBboxPatch(
        (10.0, 2.1),
        2.8,
        1.2,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        edgecolor=PALETTE["muted"],
        facecolor=PALETTE["light"],
        linewidth=1,
        linestyle="--",
    )
    ax.add_patch(registry_box)
    ax.text(11.4, 2.9, "Registry", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        11.4,
        2.55,
        "closures/registry.yaml",
        ha="center",
        fontsize=7.5,
        color=PALETTE["muted"],
    )

    # Connect to steps
    ax.plot([3.7, 4.7], [5.1, 6.1], color=PALETTE["muted"], alpha=0.4, linewidth=1)
    ax.plot([10.0, 7.7], [5.1, 6.1], color=PALETTE["muted"], alpha=0.4, linewidth=1)
    ax.plot([10.0, 7.7], [2.7, 3.1], color=PALETTE["muted"], alpha=0.4, linewidth=1)

    # Output description
    output_box = FancyBboxPatch(
        (1.0, 0.5),
        12.0,
        1.2,
        boxstyle="round,pad=0.06,rounding_size=0.06",
        edgecolor=PALETTE["accent"],
        facecolor="white",
        linewidth=1.2,
    )
    ax.add_patch(output_box)
    ax.text(7.0, 1.55, "Receipt Contents", ha="center", fontsize=10, fontweight="bold")
    ax.text(1.4, 1.05, "Git commit hash", fontsize=8.5)
    ax.text(3.9, 1.05, "SHA256 checksums", fontsize=8.5)
    ax.text(6.3, 1.05, "Validation status", fontsize=8.5)
    ax.text(8.7, 1.05, "Timestamp (UTC)", fontsize=8.5)
    ax.text(11.0, 1.05, "Contract version", fontsize=8.5)

    # Equation legend (ties steps to math)
    legend = FancyBboxPatch(
        (0.8, 3.9),
        2.6,
        1.4,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        edgecolor=PALETTE["line"],
        facecolor="white",
        linewidth=1.0,
    )
    ax.add_patch(legend)
    ax.text(
        1.0, 5.05, "Math checks", fontsize=9.2, fontweight="bold", color=PALETTE["dark"]
    )
    ax.text(1.0, 4.75, r"$IC\approx e^{\kappa}$", fontsize=8.8)
    ax.text(1.0, 4.45, r"$F\approx 1-\omega$", fontsize=8.8)
    ax.text(1.0, 4.15, r"$IC\approx e^{\kappa}$", fontsize=8.8)

    plt.tight_layout()
    plt.savefig(
        "workflow_diagram.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Created: workflow_diagram.png")
    plt.close()


def create_workflow_vertical():
    """Create alternate vertical workflow diagram."""
    _fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.4, 0.4),
        9.2,
        13.0,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor=PALETTE["light"],
        zorder=0,
    )
    ax.add_patch(bg)

    ax.text(
        5,
        13.3,
        "UMCP Workflow (Vertical)",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        5,
        12.9,
        "Optimized for documents and slides",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )

    steps = [
        ("Step 1", "Raw data", "CSV / JSON", PALETTE["tier0"]),
        ("Step 2", "Compute", "Invariants", PALETTE["tier1"]),
        ("Step 3", "Execute", "Closures", PALETTE["tier2"]),
        ("Step 4", "Validate", "Contract", PALETTE["accent"]),
        ("Step 5", "Generate", "Receipt", PALETTE["tier0"]),
    ]
    y = 10.7
    for i, (s, a, b, color) in enumerate(steps):
        card = FancyBboxPatch(
            (1.2, y),
            7.6,
            1.6,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            edgecolor=color,
            facecolor="white",
            linewidth=1.6,
        )
        ax.add_patch(card)
        ax.text(2.0, y + 1.1, s, fontsize=10, fontweight="bold", color=PALETTE["dark"])
        ax.text(4.5, y + 1.1, a, fontsize=10, color=PALETTE["dark"])
        ax.text(4.5, y + 0.7, b, fontsize=9, color=PALETTE["muted"])

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.3),
                xytext=(5, y),
                arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": PALETTE["muted"]},
            )
        y -= 2.2

    footer = FancyBboxPatch(
        (1.2, 0.9),
        7.6,
        1.4,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        edgecolor=PALETTE["accent"],
        facecolor="white",
        linewidth=1.2,
    )
    ax.add_patch(footer)
    ax.text(5, 1.85, "Receipt Contents", ha="center", fontsize=10, fontweight="bold")
    ax.text(2.0, 1.35, "Git commit hash", fontsize=8.5)
    ax.text(4.6, 1.35, "SHA256 checksums", fontsize=8.5)
    ax.text(7.0, 1.35, "Validation status", fontsize=8.5)

    plt.savefig(
        "workflow_vertical.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Created: workflow_vertical.png")
    plt.close()


if __name__ == "__main__":
    print("Generating UMCP visualizations...\n")

    create_architecture_diagram()
    create_architecture_grid()
    create_benchmark_comparison()
    create_workflow_diagram()
    create_workflow_vertical()

    print("\nAll visualizations created successfully.")
    print("\nFiles generated:")
    print("  1. architecture_diagram.png    - 3-tier system structure")
    print("  1b. architecture_grid.png      - Alternate grid layout")
    print("  2. benchmark_comparison.png    - Speed & features vs standard")
    print("  3. workflow_diagram.png        - Data processing pipeline")
    print("  3b. workflow_vertical.png      - Vertical workflow layout")
