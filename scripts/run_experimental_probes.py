"""Eight experimental probes into GCD kernel physics.

Each probe tests a specific hypothesis about the kernel's behavior
across particle physics, atomic physics, and wave phenomena domains.

Probes:
  1. Mass-channel removal — Is generation monotonicity intrinsic or mass-log artifact?
  2. SUSY catalog — Does the fermion/boson fidelity gap close with sparticles?
  3. Quark IC at variable Q — Does the confinement cliff soften at high energy?
  4. Dark matter kernel — What kernel signature do WIMP/axion/sterile-ν predict?
  5. Fixed-channel periodic kernel — Is d-block dominance real or data-completeness artifact?
  6. GUT-scale coupling plot — Do couplings converge to a unification triangle?
  7. Molecular-scale kernel — Does IC recover after the hadron cliff at molecular scale?
  8. Wave Q-factor vs IC — Does quality factor correlate with kernel integrity?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "closures"))
sys.path.insert(0, str(ROOT / "src"))

from atomic_physics.periodic_kernel import batch_compute_all
from everyday_physics.wave_phenomena import (
    WAVE_SYSTEMS,
    compute_all_wave_systems,
)
from standard_model.coupling_constants import (
    ALPHA_EM_MZ,
    ALPHA_S_MZ,
    SIN2_THETA_W,
    compute_running_coupling,
)
from standard_model.subatomic_kernel import (
    FUNDAMENTAL_PARTICLES,
    FundamentalParticle,
    compute_all_composite,
    compute_all_fundamental,
    normalize_fundamental,
)

from umcp.kernel_optimized import compute_kernel_outputs

OUTDIR = ROOT / "images"
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    }
)

COLORS = {
    "cyan": "#58a6ff",
    "green": "#3fb950",
    "orange": "#d29922",
    "red": "#f85149",
    "purple": "#bc8cff",
    "pink": "#f778ba",
    "blue": "#1f6feb",
    "teal": "#39d353",
    "yellow": "#e3b341",
    "gray": "#8b949e",
}

EPSILON = 1e-6  # Module-level epsilon matching subatomic_kernel


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTDIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 1: Mass-channel removal — Is generation staircase intrinsic?
# ═══════════════════════════════════════════════════════════════════════════
def probe_1_mass_removal() -> dict:
    """Remove mass_log from trace vector, recompute gen monotonicity."""
    print("\n╔══ PROBE 1: Mass-Channel Removal ══╗")

    # Group fermions by generation
    fermions = [p for p in FUNDAMENTAL_PARTICLES if p.is_fermion]
    gen_groups: dict[int, list[FundamentalParticle]] = {1: [], 2: [], 3: []}
    for p in fermions:
        gen_groups[p.generation].append(p)

    results_full: dict[int, list[float]] = {1: [], 2: [], 3: []}
    results_no_mass: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for gen in [1, 2, 3]:
        for p in gen_groups[gen]:
            # Full 8-channel kernel
            c_full, w_full, labels = normalize_fundamental(p)
            k_full = compute_kernel_outputs(c_full, w_full)
            results_full[gen].append(k_full["F"])

            # Remove mass_log channel (index 0)
            mass_idx = labels.index("mass_log")
            c_no_mass = np.delete(c_full, mass_idx)
            w_no_mass = np.delete(w_full, mass_idx)
            w_no_mass = w_no_mass / w_no_mass.sum()  # Renormalize
            k_no_mass = compute_kernel_outputs(c_no_mass, w_no_mass)
            results_no_mass[gen].append(k_no_mass["F"])

    avg_full = {g: float(np.mean(v)) for g, v in results_full.items()}
    avg_no_mass = {g: float(np.mean(v)) for g, v in results_no_mass.items()}

    monotone_full = avg_full[1] < avg_full[2] < avg_full[3]
    monotone_no_mass = avg_no_mass[1] < avg_no_mass[2] < avg_no_mass[3]

    print(
        f"  Full kernel:    Gen1={avg_full[1]:.4f}  Gen2={avg_full[2]:.4f}  Gen3={avg_full[3]:.4f}  Monotone={monotone_full}"
    )
    print(
        f"  Without mass:   Gen1={avg_no_mass[1]:.4f}  Gen2={avg_no_mass[2]:.4f}  Gen3={avg_no_mass[3]:.4f}  Monotone={monotone_no_mass}"
    )

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    gens = [1, 2, 3]

    for ax, data, title in [
        (ax1, results_full, "Full 8-Channel Kernel"),
        (ax2, results_no_mass, "Mass Channel Removed (7-Channel)"),
    ]:
        for i, gen in enumerate(gens):
            vals = data[gen]
            color = [COLORS["cyan"], COLORS["green"], COLORS["orange"]][i]
            x_positions = [i + j * 0.12 for j in range(len(vals))]
            ax.bar(x_positions, vals, width=0.1, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            avg = np.mean(vals)
            ax.hlines(avg, i - 0.15, i + len(vals) * 0.12 - 0.05, colors=color, linewidth=2, linestyle="--")
            ax.text(i + 0.05, avg + 0.01, f"⟨F⟩={avg:.3f}", color=color, fontsize=9, ha="center")

        ax.set_xticks(range(3))
        ax.set_xticklabels(["Gen 1\n(u,d,e,νe)", "Gen 2\n(c,s,μ,νμ)", "Gen 3\n(t,b,τ,ντ)"])
        ax.set_ylabel("Fidelity F")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    monotone_label = "✓ INTRINSIC" if monotone_no_mass else "✗ MASS-DEPENDENT"
    fig.suptitle(
        f"Probe 1: Generation Staircase — {monotone_label}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["cyan"],
    )
    fig.tight_layout()
    _save(fig, "probe_01_mass_removal.png")

    return {
        "full": avg_full,
        "no_mass": avg_no_mass,
        "monotone_full": monotone_full,
        "monotone_no_mass": monotone_no_mass,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 2: SUSY Catalog — Does the fermion/boson gap close?
# ═══════════════════════════════════════════════════════════════════════════
def probe_2_susy_catalog() -> dict:
    """Build sparticle catalog and test if F gap closes."""
    print("\n╔══ PROBE 2: SUSY Catalog ══╗")

    # Sparticle catalog: each superpartner flips spin by 1/2
    sparticles = [
        # Scalar partners of fermions (sfermions)
        FundamentalParticle("selectron", "ẽ", "Sfermion", 200.0, -1.0, 0.0, 1, 1, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("smuon", "μ̃", "Sfermion", 400.0, -1.0, 0.0, 1, 2, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("stau", "τ̃", "Sfermion", 300.0, -1.0, 0.0, 1, 3, -0.5, -1.0, 0.0, 0.0, False),
        FundamentalParticle("sup", "ũ", "Sfermion", 1500.0, 2 / 3, 0.0, 3, 1, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sdown", "d̃", "Sfermion", 1500.0, -1 / 3, 0.0, 3, 1, -0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("scharm", "c̃", "Sfermion", 1500.0, 2 / 3, 0.0, 3, 2, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sstrange", "s̃", "Sfermion", 1500.0, -1 / 3, 0.0, 3, 2, -0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("stop", "t̃", "Sfermion", 1000.0, 2 / 3, 0.0, 3, 3, 0.5, 1 / 3, 0.0, 0.0, False),
        FundamentalParticle("sbottom", "b̃", "Sfermion", 1200.0, -1 / 3, 0.0, 3, 3, -0.5, 1 / 3, 0.0, 0.0, False),
        # Fermionic partners of bosons (gauginos)
        FundamentalParticle("photino", "γ̃", "Gaugino", 300.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("wino", "W̃", "Gaugino", 400.0, 1.0, 0.5, 1, 0, 1.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("zino", "Z̃", "Gaugino", 500.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("gluino", "g̃", "Gaugino", 2000.0, 0.0, 0.5, 8, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("higgsino", "H̃", "Gaugino", 600.0, 0.0, 0.5, 1, 0, 0.5, 1.0, 0.0, 0.0, True),
    ]

    # Compute SM results
    sm_results = compute_all_fundamental()
    sm_fermions = [r for r in sm_results if r.category in ("Quark", "Lepton")]
    sm_bosons = [r for r in sm_results if r.category in ("GaugeBoson", "ScalarBoson")]

    avg_F_sm_fermion = float(np.mean([r.F for r in sm_fermions]))
    avg_F_sm_boson = float(np.mean([r.F for r in sm_bosons]))
    sm_gap = avg_F_sm_fermion - avg_F_sm_boson

    # Compute SUSY kernels
    susy_results = []
    for sp in sparticles:
        c, w, _labels = normalize_fundamental(sp)
        k = compute_kernel_outputs(c, w)
        susy_results.append((sp.name, sp.category, sp.is_fermion, k["F"], k["IC"]))

    sfermions = [r for r in susy_results if r[1] == "Sfermion"]
    gauginos = [r for r in susy_results if r[1] == "Gaugino"]

    # Combined: SM fermions + gauginos vs SM bosons + sfermions
    all_fermion_F = [r.F for r in sm_fermions] + [r[3] for r in gauginos]
    all_boson_F = [r.F for r in sm_bosons] + [r[3] for r in sfermions]

    avg_F_susy_fermion = float(np.mean(all_fermion_F))
    avg_F_susy_boson = float(np.mean(all_boson_F))
    susy_gap = avg_F_susy_fermion - avg_F_susy_boson

    gap_reduction = 1.0 - abs(susy_gap) / abs(sm_gap) if sm_gap != 0 else 0

    print(f"  SM gap:          ⟨F⟩_fermion={avg_F_sm_fermion:.4f} − ⟨F⟩_boson={avg_F_sm_boson:.4f} = {sm_gap:.4f}")
    print(
        f"  SUSY gap:        ⟨F⟩_fermion={avg_F_susy_fermion:.4f} − ⟨F⟩_boson={avg_F_susy_boson:.4f} = {susy_gap:.4f}"
    )
    print(f"  Gap reduction:   {gap_reduction:.1%}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SM only
    ax1.barh(
        [r.name for r in sm_fermions],
        [r.F for r in sm_fermions],
        color=COLORS["cyan"],
        alpha=0.8,
        label="Fermions",
    )
    ax1.barh(
        [r.name for r in sm_bosons],
        [r.F for r in sm_bosons],
        color=COLORS["orange"],
        alpha=0.8,
        label="Bosons",
    )
    ax1.axvline(
        avg_F_sm_fermion, color=COLORS["cyan"], linestyle="--", linewidth=1.5, label=f"⟨F⟩ ferm={avg_F_sm_fermion:.3f}"
    )
    ax1.axvline(
        avg_F_sm_boson, color=COLORS["orange"], linestyle="--", linewidth=1.5, label=f"⟨F⟩ boson={avg_F_sm_boson:.3f}"
    )
    ax1.set_xlabel("Fidelity F")
    ax1.set_title(f"Standard Model Only\nGap = {sm_gap:.4f}")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)

    # Right: SM + SUSY
    all_names_ferm = [r.name for r in sm_fermions] + [r[0] for r in gauginos]
    all_F_ferm = [r.F for r in sm_fermions] + [r[3] for r in gauginos]
    all_names_boson = [r.name for r in sm_bosons] + [r[0] for r in sfermions]
    all_F_boson = [r.F for r in sm_bosons] + [r[3] for r in sfermions]

    colors_ferm = [COLORS["cyan"]] * len(sm_fermions) + [COLORS["purple"]] * len(gauginos)
    colors_boson = [COLORS["orange"]] * len(sm_bosons) + [COLORS["pink"]] * len(sfermions)

    ax2.barh(all_names_ferm, all_F_ferm, color=colors_ferm, alpha=0.8)
    ax2.barh(all_names_boson, all_F_boson, color=colors_boson, alpha=0.8)
    ax2.axvline(
        avg_F_susy_fermion,
        color=COLORS["cyan"],
        linestyle="--",
        linewidth=1.5,
        label=f"⟨F⟩ ferm={avg_F_susy_fermion:.3f}",
    )
    ax2.axvline(
        avg_F_susy_boson,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.5,
        label=f"⟨F⟩ boson={avg_F_susy_boson:.3f}",
    )
    ax2.set_xlabel("Fidelity F")
    ax2.set_title(f"SM + SUSY Partners\nGap = {susy_gap:.4f} ({gap_reduction:.0%} reduction)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)

    fig.suptitle(
        "Probe 2: Supersymmetry & the Fermion–Boson Fidelity Gap",
        fontsize=16,
        fontweight="bold",
        color=COLORS["purple"],
    )
    fig.tight_layout()
    _save(fig, "probe_02_susy_catalog.png")

    return {
        "sm_gap": sm_gap,
        "susy_gap": susy_gap,
        "gap_reduction_pct": gap_reduction * 100,
        "sfermion_avg_F": float(np.mean([r[3] for r in sfermions])),
        "gaugino_avg_F": float(np.mean([r[3] for r in gauginos])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 3: Quark IC at variable Q — Does confinement cliff soften?
# ═══════════════════════════════════════════════════════════════════════════
def probe_3_quark_variable_q() -> dict:
    """Compute quark kernel at different energy scales."""
    print("\n╔══ PROBE 3: Quark IC at Variable Energy Scale ══╗")

    quarks = [p for p in FUNDAMENTAL_PARTICLES if p.category == "Quark"]
    Q_values = np.logspace(0, 4, 40)  # 1 GeV to 10 TeV

    # For each Q, adjust quark kernel: lower α_s → weaker color force → IC rises?
    # Strategy: modulate the color_dof channel by α_s(Q) / α_s(M_Z)
    alpha_s_mz = ALPHA_S_MZ  # 0.1180

    quark_IC_curves: dict[str, list[float]] = {}
    quark_F_curves: dict[str, list[float]] = {}

    for q in quarks:
        ic_curve = []
        f_curve = []
        for q_gev in Q_values:
            coupling = compute_running_coupling(q_gev)
            alpha_ratio = coupling.alpha_s / alpha_s_mz  # >1 at low Q, <1 at high Q

            c, w, labels = normalize_fundamental(q)

            # Modulate color_dof channel by coupling strength
            color_idx = labels.index("color_dof")
            # At high Q: weaker coupling → color channel closer to 1 (more "free")
            # At low Q: stronger coupling → color channel closer to ε (more "confined")
            c_mod = c.copy()
            # Effective confinement: interpolate color channel
            free_color = min(1.0 - EPSILON, c[color_idx] / max(alpha_ratio, EPSILON))
            c_mod[color_idx] = np.clip(free_color, EPSILON, 1.0 - EPSILON)

            k = compute_kernel_outputs(c_mod, w)
            ic_curve.append(k["IC"])
            f_curve.append(k["F"])

        quark_IC_curves[q.name] = ic_curve
        quark_F_curves[q.name] = f_curve

    # Also get hadron IC for reference at each Q
    hadrons = compute_all_composite()
    hadron_avg_ic = float(np.mean([r.IC for r in hadrons]))

    print(f"  Q range: {Q_values[0]:.0f} GeV — {Q_values[-1]:.0f} GeV ({len(Q_values)} points)")
    for name in ["up", "down", "top"]:
        if name in quark_IC_curves:
            ic_lo = quark_IC_curves[name][0]
            ic_hi = quark_IC_curves[name][-1]
            print(
                f"  {name:>6s}: IC(1 GeV)={ic_lo:.4f} → IC(10 TeV)={ic_hi:.4f}  ratio={ic_hi / max(ic_lo, 1e-10):.1f}×"
            )

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    quark_colors = {
        "up": COLORS["cyan"],
        "down": COLORS["green"],
        "charm": COLORS["orange"],
        "strange": COLORS["yellow"],
        "top": COLORS["red"],
        "bottom": COLORS["purple"],
    }

    for name, ic_vals in quark_IC_curves.items():
        color = quark_colors.get(name, COLORS["gray"])
        ax1.plot(Q_values, ic_vals, color=color, linewidth=2, label=name, alpha=0.9)

    ax1.axhline(
        hadron_avg_ic, color=COLORS["gray"], linestyle=":", linewidth=1, label=f"Hadron ⟨IC⟩={hadron_avg_ic:.3f}"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Energy Scale Q (GeV)")
    ax1.set_ylabel("Integrity Composite IC")
    ax1.set_title("Quark IC vs Energy Scale")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right: α_s running for context
    alpha_s_vals = []
    for q_gev in Q_values:
        coupling = compute_running_coupling(q_gev)
        alpha_s_vals.append(coupling.alpha_s)

    ax2.plot(Q_values, alpha_s_vals, color=COLORS["red"], linewidth=2.5)
    ax2.axhline(0.1180, color=COLORS["gray"], linestyle="--", linewidth=1, label="α_s(M_Z)=0.1180")
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy Scale Q (GeV)")
    ax2.set_ylabel("Strong Coupling α_s(Q)")
    ax2.set_title("Running Strong Coupling (Context)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Probe 3: Asymptotic Freedom in the GCD Kernel",
        fontsize=16,
        fontweight="bold",
        color=COLORS["red"],
    )
    fig.tight_layout()
    _save(fig, "probe_03_variable_q_quarks.png")

    return {
        "Q_range": (float(Q_values[0]), float(Q_values[-1])),
        "up_IC_1GeV": quark_IC_curves["up"][0],
        "up_IC_10TeV": quark_IC_curves["up"][-1],
        "top_IC_1GeV": quark_IC_curves["top"][0],
        "top_IC_10TeV": quark_IC_curves["top"][-1],
        "hadron_avg_IC": hadron_avg_ic,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 4: Dark Matter Kernel — Predict DM kernel signatures
# ═══════════════════════════════════════════════════════════════════════════
def probe_4_dark_matter() -> dict:
    """Build dark matter candidates and predict kernel signatures."""
    print("\n╔══ PROBE 4: Dark Matter Kernel Signatures ══╗")

    # Dark matter candidates with hypothesized properties
    dm_candidates = [
        # WIMPs: massive, weakly interacting
        FundamentalParticle("WIMP (light)", "χ₁", "DarkMatter", 10.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("WIMP (medium)", "χ₂", "DarkMatter", 100.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("WIMP (heavy)", "χ₃", "DarkMatter", 1000.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        # Axion: ultralight pseudoscalar
        FundamentalParticle("Axion", "a", "DarkMatter", 1e-5, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
        # Sterile neutrino: right-handed neutrino
        FundamentalParticle("Sterile ν (keV)", "ν_s", "DarkMatter", 7e-6, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        FundamentalParticle("Sterile ν (GeV)", "ν_S", "DarkMatter", 1.0, 0.0, 0.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        # Gravitino: spin-3/2 SUSY partner of graviton
        FundamentalParticle("Gravitino", "G̃", "DarkMatter", 0.001, 0.0, 1.5, 1, 0, 0.0, 0.0, 0.0, 0.0, True),
        # Dark photon companion candidate
        FundamentalParticle("Dark Photon", "γ_D", "DarkMatter", 0.01, 0.0, 1.0, 1, 0, 0.0, 0.0, 0.0, 0.0, False),
    ]

    dm_results = []
    for cand in dm_candidates:
        c, w, _labels = normalize_fundamental(cand)
        k = compute_kernel_outputs(c, w)
        dm_results.append(
            {
                "name": cand.name,
                "mass_GeV": cand.mass_GeV,
                "spin": cand.spin,
                "F": k["F"],
                "IC": k["IC"],
                "omega": k["omega"],
                "S": k["S"],
                "C": k["C"],
                "gap": k["heterogeneity_gap"],
                "regime": k["regime"],
            }
        )

    # SM reference: neutrinos (closest SM analogue)
    sm_results = compute_all_fundamental()
    neutrinos = [r for r in sm_results if "neutrino" in r.name.lower()]
    sm_avg = {
        "fermion_F": float(np.mean([r.F for r in sm_results if r.category in ("Quark", "Lepton")])),
        "boson_F": float(np.mean([r.F for r in sm_results if r.category in ("GaugeBoson", "ScalarBoson")])),
        "neutrino_F": float(np.mean([r.F for r in neutrinos])) if neutrinos else 0.0,
        "neutrino_IC": float(np.mean([r.IC for r in neutrinos])) if neutrinos else 0.0,
    }

    for d in dm_results:
        print(
            f"  {d['name']:>20s}: F={d['F']:.4f}  IC={d['IC']:.6f}  ω={d['omega']:.4f}  gap={d['gap']:.4f}  regime={d['regime']}"
        )

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: F comparison
    ax = axes[0]
    names = [d["name"] for d in dm_results]
    f_vals = [d["F"] for d in dm_results]
    ax.barh(names, f_vals, color=COLORS["purple"], alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(
        sm_avg["neutrino_F"],
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        label=f"SM ν avg F={sm_avg['neutrino_F']:.3f}",
    )
    ax.axvline(
        sm_avg["fermion_F"],
        color=COLORS["cyan"],
        linestyle=":",
        linewidth=1,
        label=f"SM fermion avg={sm_avg['fermion_F']:.3f}",
    )
    ax.set_xlabel("Fidelity F")
    ax.set_title("Dark Matter Fidelity")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel 2: IC comparison (log scale)
    ax = axes[1]
    ic_vals = [d["IC"] for d in dm_results]
    ax.barh(names, ic_vals, color=COLORS["pink"], alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(
        sm_avg["neutrino_IC"],
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        label=f"SM ν avg IC={sm_avg['neutrino_IC']:.4f}",
    )
    ax.set_xlabel("Integrity Composite IC")
    ax.set_xscale("log")
    ax.set_title("Dark Matter Integrity")
    ax.legend(fontsize=8)

    # Panel 3: Heterogeneity gap
    ax = axes[2]
    gap_vals = [d["gap"] for d in dm_results]
    bar_colors = [COLORS["red"] if g > 0.3 else COLORS["orange"] if g > 0.1 else COLORS["green"] for g in gap_vals]
    ax.barh(names, gap_vals, color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Heterogeneity Gap (F − IC)")
    ax.set_title("Dark Matter Channel Heterogeneity")

    fig.suptitle(
        "Probe 4: Dark Matter Kernel Signatures — Predictions",
        fontsize=16,
        fontweight="bold",
        color=COLORS["pink"],
    )
    fig.tight_layout()
    _save(fig, "probe_04_dark_matter.png")

    return {"dm_results": dm_results, "sm_reference": sm_avg}


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 5: Fixed-channel periodic kernel — data completeness artifact?
# ═══════════════════════════════════════════════════════════════════════════
def probe_5_fixed_channel_periodic() -> dict:
    """Recompute all 118 elements using only universally available channels."""
    print("\n╔══ PROBE 5: Fixed-Channel Periodic Kernel ══╗")

    all_elements = batch_compute_all()

    # Find which channels are present in ALL elements
    channel_counts: dict[str, int] = {}
    for el in all_elements:
        for ch in el.channel_labels:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

    universal_channels = [ch for ch, cnt in channel_counts.items() if cnt == len(all_elements)]
    print(f"  Total elements: {len(all_elements)}")
    print(f"  Universal channels ({len(universal_channels)}): {universal_channels}")

    # Recompute with only universal channels
    fixed_results: dict[str, dict] = {}
    variable_results: dict[str, dict] = {}

    for el in all_elements:
        # Variable channels (original)
        variable_results[el.symbol] = {
            "F": el.F,
            "IC": el.IC,
            "gap": el.heterogeneity_gap,
            "n_channels": el.n_channels,
            "block": el.block,
        }

        # Fixed channels only
        c_orig = np.array(el.trace_vector)
        labels = el.channel_labels
        mask = [i for i, lbl in enumerate(labels) if lbl in universal_channels]

        if len(mask) >= 2:
            c_fixed = c_orig[mask]
            w_fixed = np.ones(len(c_fixed)) / len(c_fixed)
            k = compute_kernel_outputs(c_fixed, w_fixed)
            fixed_results[el.symbol] = {
                "F": k["F"],
                "IC": k["IC"],
                "gap": k["heterogeneity_gap"],
                "n_channels": len(mask),
                "block": el.block,
            }

    # Compare block averages
    blocks = ["s", "p", "d", "f"]
    block_avg_var = {}
    block_avg_fix = {}
    for blk in blocks:
        var_f = [v["F"] for v in variable_results.values() if v["block"] == blk]
        fix_f = [v["F"] for v in fixed_results.values() if v["block"] == blk]
        if var_f:
            block_avg_var[blk] = float(np.mean(var_f))
        if fix_f:
            block_avg_fix[blk] = float(np.mean(fix_f))

    d_block_dominant_var = block_avg_var.get("d", 0) == max(block_avg_var.values())
    d_block_dominant_fix = block_avg_fix.get("d", 0) == max(block_avg_fix.values())

    for blk in blocks:
        v = block_avg_var.get(blk, 0)
        f = block_avg_fix.get(blk, 0)
        print(f"  {blk}-block: Variable ⟨F⟩={v:.4f}  Fixed ⟨F⟩={f:.4f}  Δ={f - v:+.4f}")

    print(f"  d-block dominant (var): {d_block_dominant_var}")
    print(f"  d-block dominant (fix): {d_block_dominant_fix}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Variable-channel F by block
    ax = axes[0, 0]
    for blk, color in zip(blocks, [COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["purple"]], strict=False):
        z_vals = [all_elements[i].Z for i, el in enumerate(all_elements) if variable_results[el.symbol]["block"] == blk]
        f_vals = [
            variable_results[el.symbol]["F"] for el in all_elements if variable_results[el.symbol]["block"] == blk
        ]
        ax.scatter(
            z_vals, f_vals, color=color, alpha=0.6, s=20, label=f"{blk}-block ⟨F⟩={block_avg_var.get(blk, 0):.3f}"
        )
    ax.set_xlabel("Atomic Number Z")
    ax.set_ylabel("Fidelity F")
    ax.set_title("Variable Channels (Original)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Fixed-channel F by block
    ax = axes[0, 1]
    for blk, color in zip(blocks, [COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["purple"]], strict=False):
        syms = [s for s, v in fixed_results.items() if v["block"] == blk]
        z_vals = [next(el.Z for el in all_elements if el.symbol == s) for s in syms]
        f_vals = [fixed_results[s]["F"] for s in syms]
        ax.scatter(
            z_vals, f_vals, color=color, alpha=0.6, s=20, label=f"{blk}-block ⟨F⟩={block_avg_fix.get(blk, 0):.3f}"
        )
    ax.set_xlabel("Atomic Number Z")
    ax.set_ylabel("Fidelity F")
    ax.set_title(f"Fixed {len(universal_channels)} Channels Only")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Channel count distribution
    ax = axes[1, 0]
    n_channel_counts = [variable_results[el.symbol]["n_channels"] for el in all_elements]
    ax.hist(
        n_channel_counts,
        bins=range(min(n_channel_counts), max(n_channel_counts) + 2),
        color=COLORS["cyan"],
        alpha=0.7,
        edgecolor="white",
    )
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("Number of Elements")
    ax.set_title("Channel Availability Distribution")
    ax.grid(True, alpha=0.3)

    # Bottom right: F difference (fixed - variable)
    ax = axes[1, 1]
    common_syms = [s for s in fixed_results if s in variable_results]
    z_common = [next(el.Z for el in all_elements if el.symbol == s) for s in common_syms]
    delta_f = [fixed_results[s]["F"] - variable_results[s]["F"] for s in common_syms]
    colors_delta = [COLORS["green"] if d > 0 else COLORS["red"] for d in delta_f]
    ax.bar(z_common, delta_f, color=colors_delta, alpha=0.6, width=1)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.5)
    ax.set_xlabel("Atomic Number Z")
    ax.set_ylabel("ΔF (Fixed − Variable)")
    ax.set_title("Impact of Fixing Channels")
    ax.grid(True, alpha=0.3)

    label = "REAL" if d_block_dominant_fix else "ARTIFACT"
    fig.suptitle(
        f"Probe 5: Fixed-Channel Periodic Kernel — d-Block Dominance is {label}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["orange"],
    )
    fig.tight_layout()
    _save(fig, "probe_05_fixed_channel_periodic.png")

    return {
        "universal_channels": universal_channels,
        "block_avg_variable": block_avg_var,
        "block_avg_fixed": block_avg_fix,
        "d_block_dominant_variable": d_block_dominant_var,
        "d_block_dominant_fixed": d_block_dominant_fix,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 6: GUT-Scale Coupling Plot — Unification triangle
# ═══════════════════════════════════════════════════════════════════════════
def probe_6_gut_scale_coupling() -> dict:
    """Extend coupling curves to GUT scale and visualize unification."""
    print("\n╔══ PROBE 6: GUT-Scale Coupling Convergence ══╗")

    # Standard Model coupling constants at M_Z
    # α₁ = (5/3) * α_em / cos²θ_W  (GUT normalization)
    # α₂ = α_em / sin²θ_W
    # α₃ = α_s
    alpha_em_mz = ALPHA_EM_MZ  # 1/127.952
    sin2tw = SIN2_THETA_W  # 0.23122
    cos2tw = 1.0 - sin2tw
    m_z = 91.1876

    alpha_1_mz = (5.0 / 3.0) * alpha_em_mz / cos2tw
    alpha_2_mz = alpha_em_mz / sin2tw
    alpha_3_mz = ALPHA_S_MZ

    # One-loop β coefficients (SM)
    # b_i = (1/(2π)) * β_i, where dα⁻¹/d(ln Q) = -b_i / (2π)
    b1 = 41.0 / 10.0  # U(1)
    b2 = -19.0 / 6.0  # SU(2)
    b3 = -7.0  # SU(3)

    Q_values = np.logspace(np.log10(m_z), 17, 500)

    def run_coupling_inv(alpha_mz: float, b: float, q_arr: np.ndarray) -> np.ndarray:
        """One-loop running: 1/α(Q) = 1/α(M_Z) - (b/(2π)) * ln(Q/M_Z)."""
        return 1.0 / alpha_mz - (b / (2.0 * np.pi)) * np.log(q_arr / m_z)

    inv_alpha_1 = run_coupling_inv(alpha_1_mz, b1, Q_values)
    inv_alpha_2 = run_coupling_inv(alpha_2_mz, b2, Q_values)
    inv_alpha_3 = run_coupling_inv(alpha_3_mz, b3, Q_values)

    # Find pairwise crossing points
    def find_crossing(inv_a: np.ndarray, inv_b: np.ndarray, q_arr: np.ndarray) -> float | None:
        diff = inv_a - inv_b
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation
            q1, q2 = q_arr[idx], q_arr[idx + 1]
            d1, d2 = diff[idx], diff[idx + 1]
            return float(q1 - d1 * (q2 - q1) / (d2 - d1))
        return None

    cross_12 = find_crossing(inv_alpha_1, inv_alpha_2, Q_values)
    cross_13 = find_crossing(inv_alpha_1, inv_alpha_3, Q_values)
    cross_23 = find_crossing(inv_alpha_2, inv_alpha_3, Q_values)

    crossings = {"α₁-α₂": cross_12, "α₁-α₃": cross_13, "α₂-α₃": cross_23}
    for name, q in crossings.items():
        if q is not None:
            print(f"  {name} crossing: Q ≈ {q:.2e} GeV")
        else:
            print(f"  {name} crossing: not found in range")

    # Unification triangle: spread at GUT scale (~2×10¹⁶)
    gut_idx = np.argmin(np.abs(Q_values - 2e16))
    triangle_spread = max(inv_alpha_1[gut_idx], inv_alpha_2[gut_idx], inv_alpha_3[gut_idx]) - min(
        inv_alpha_1[gut_idx], inv_alpha_2[gut_idx], inv_alpha_3[gut_idx]
    )
    print(f"  Triangle spread at 2×10¹⁶ GeV: Δ(1/α) = {triangle_spread:.2f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Full running
    ax1.plot(Q_values, inv_alpha_1, color=COLORS["cyan"], linewidth=2.5, label="α₁⁻¹ (U(1))")
    ax1.plot(Q_values, inv_alpha_2, color=COLORS["green"], linewidth=2.5, label="α₂⁻¹ (SU(2))")
    ax1.plot(Q_values, inv_alpha_3, color=COLORS["red"], linewidth=2.5, label="α₃⁻¹ (SU(3))")

    for _name, q in crossings.items():
        if q is not None:
            ax1.axvline(q, color=COLORS["yellow"], alpha=0.3, linestyle="--")
            ax1.text(q, 5, f"{q:.0e}", color=COLORS["yellow"], fontsize=7, rotation=45, ha="left")

    ax1.set_xscale("log")
    ax1.set_xlabel("Energy Scale Q (GeV)")
    ax1.set_ylabel("1/α (inverse coupling)")
    ax1.set_title("SM Gauge Coupling Running (1-Loop)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)

    # Right: Zoom on unification region
    mask = Q_values > 1e13
    ax2.plot(Q_values[mask], inv_alpha_1[mask], color=COLORS["cyan"], linewidth=2.5, label="α₁⁻¹")
    ax2.plot(Q_values[mask], inv_alpha_2[mask], color=COLORS["green"], linewidth=2.5, label="α₂⁻¹")
    ax2.plot(Q_values[mask], inv_alpha_3[mask], color=COLORS["red"], linewidth=2.5, label="α₃⁻¹")

    # Shade the triangle
    q_zoom = Q_values[mask]
    a1_zoom = inv_alpha_1[mask]
    a2_zoom = inv_alpha_2[mask]
    a3_zoom = inv_alpha_3[mask]
    ax2.fill_between(
        q_zoom,
        np.minimum(a1_zoom, np.minimum(a2_zoom, a3_zoom)),
        np.maximum(a1_zoom, np.maximum(a2_zoom, a3_zoom)),
        alpha=0.15,
        color=COLORS["yellow"],
        label=f"Unification triangle (Δ={triangle_spread:.1f})",
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Energy Scale Q (GeV)")
    ax2.set_ylabel("1/α")
    ax2.set_title("Zoom: Unification Region (>10¹³ GeV)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Probe 6: Gauge Coupling Unification — The GUT Triangle",
        fontsize=16,
        fontweight="bold",
        color=COLORS["yellow"],
    )
    fig.tight_layout()
    _save(fig, "probe_06_gut_scale_coupling.png")

    return {
        "crossings": {k: float(v) if v is not None else None for k, v in crossings.items()},
        "triangle_spread": float(triangle_spread),
        "alpha_at_mz": {"α₁": float(alpha_1_mz), "α₂": float(alpha_2_mz), "α₃": float(alpha_3_mz)},
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 7: Molecular-Scale Kernel — IC recovery after hadron cliff?
# ═══════════════════════════════════════════════════════════════════════════
def probe_7_molecular_kernel() -> dict:
    """Build molecular trace vectors and test IC recovery."""
    print("\n╔══ PROBE 7: Molecular-Scale Kernel ══╗")

    # Molecules with properties for trace vector construction
    # Channels: mass_amu_norm, bond_order, electronegativity_diff, dipole_moment,
    #           n_atoms, symmetry, thermal_stability, bond_energy_norm
    molecules = [
        # (name, mass_amu, bond_order, EN_diff, dipole_D, n_atoms, symmetry, T_decomp_K, bond_energy_kJ)
        ("H₂", 2.016, 1.0, 0.0, 0.0, 2, 1.0, 3000, 436),
        ("H₂O", 18.015, 1.0, 1.24, 1.85, 3, 0.5, 2500, 463),
        ("CO₂", 44.01, 2.0, 0.89, 0.0, 3, 1.0, 1900, 799),
        ("N₂", 28.014, 3.0, 0.0, 0.0, 2, 1.0, 6000, 945),
        ("O₂", 31.998, 2.0, 0.0, 0.0, 2, 1.0, 5000, 498),
        ("NaCl", 58.44, 1.0, 2.23, 9.0, 2, 1.0, 1738, 411),
        ("CH₄", 16.04, 1.0, 0.35, 0.0, 5, 1.0, 1500, 411),
        ("NH₃", 17.03, 1.0, 0.84, 1.47, 4, 0.67, 2000, 391),
        ("C₆H₆", 78.11, 1.5, 0.35, 0.0, 12, 1.0, 1200, 518),
        ("C₂H₅OH", 46.07, 1.0, 1.24, 1.69, 9, 0.33, 1100, 348),
        ("DNA base pair", 660.0, 1.5, 1.0, 2.0, 40, 0.2, 500, 300),
        ("Hemoglobin", 64500, 1.0, 1.5, 3.0, 9700, 0.05, 340, 250),
    ]

    # Normalization ranges
    mass_max = max(m[1] for m in molecules)
    bond_max = 3.0
    en_max = max(m[3] for m in molecules)
    dipole_max = max(m[4] for m in molecules)
    natom_max = max(m[5] for m in molecules)
    temp_max = max(m[7] for m in molecules)
    be_max = max(m[8] for m in molecules)

    mol_results = []
    for name, mass, bond_ord, en_diff, dipole, n_atoms, sym, t_decomp, bond_e in molecules:
        c = np.array(
            [
                np.clip(np.log10(mass + 1) / np.log10(mass_max + 1), EPSILON, 1 - EPSILON),
                np.clip(bond_ord / bond_max, EPSILON, 1 - EPSILON),
                np.clip(en_diff / max(en_max, EPSILON), EPSILON, 1 - EPSILON),
                np.clip(dipole / max(dipole_max, EPSILON), EPSILON, 1 - EPSILON),
                np.clip(np.log10(n_atoms + 1) / np.log10(natom_max + 1), EPSILON, 1 - EPSILON),
                np.clip(sym, EPSILON, 1 - EPSILON),
                np.clip(t_decomp / max(temp_max, EPSILON), EPSILON, 1 - EPSILON),
                np.clip(bond_e / max(be_max, EPSILON), EPSILON, 1 - EPSILON),
            ]
        )
        w = np.ones(8) / 8.0
        k = compute_kernel_outputs(c, w)
        mol_results.append(
            {
                "name": name,
                "mass_amu": mass,
                "n_atoms": n_atoms,
                "F": k["F"],
                "IC": k["IC"],
                "omega": k["omega"],
                "gap": k["heterogeneity_gap"],
                "regime": k["regime"],
            }
        )

    # Cross-scale comparison
    hadrons = compute_all_composite()
    atoms = batch_compute_all()

    hadron_avg_ic = float(np.mean([r.IC for r in hadrons]))
    atom_avg_ic = float(np.mean([r.IC for r in atoms]))
    mol_avg_ic = float(np.mean([m["IC"] for m in mol_results]))

    print("  Scale ladder IC:")
    print(f"    Hadrons:   ⟨IC⟩ = {hadron_avg_ic:.4f}")
    print(f"    Atoms:     ⟨IC⟩ = {atom_avg_ic:.4f}")
    print(f"    Molecules: ⟨IC⟩ = {mol_avg_ic:.4f}")
    ic_recovery = mol_avg_ic > hadron_avg_ic
    print(f"  IC recovery above hadron level: {ic_recovery}")

    for m in mol_results:
        print(f"    {m['name']:>18s}: F={m['F']:.4f}  IC={m['IC']:.4f}  gap={m['gap']:.4f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Scale ladder IC
    scale_names = ["Quarks\n(confined)", "Hadrons", "Atoms\n(118 elements)", "Molecules"]
    quark_results = [r for r in compute_all_fundamental() if r.category == "Quark"]
    scale_ics = [
        float(np.mean([r.IC for r in quark_results])),
        hadron_avg_ic,
        atom_avg_ic,
        mol_avg_ic,
    ]
    scale_colors_list = [COLORS["red"], COLORS["orange"], COLORS["green"], COLORS["cyan"]]
    ax1.bar(scale_names, scale_ics, color=scale_colors_list, alpha=0.8, edgecolor="white", linewidth=1)
    for i, (_name, ic) in enumerate(zip(scale_names, scale_ics, strict=False)):
        ax1.text(i, ic + 0.01, f"{ic:.3f}", ha="center", color="white", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Average IC")
    ax1.set_title("IC Across Scale Ladder")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Individual molecule results
    ax = ax2
    mol_names = [m["name"] for m in mol_results]
    mol_f = [m["F"] for m in mol_results]
    mol_ic = [m["IC"] for m in mol_results]
    x = np.arange(len(mol_names))
    width = 0.35
    ax.barh(x - width / 2, mol_f, width, color=COLORS["cyan"], alpha=0.8, label="F (Fidelity)")
    ax.barh(x + width / 2, mol_ic, width, color=COLORS["green"], alpha=0.8, label="IC (Integrity)")
    ax.set_yticks(x)
    ax.set_yticklabels(mol_names)
    ax.set_xlabel("Value")
    ax.set_title("Molecular Kernel Results")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    recovery_label = "RECOVERY CONFIRMED" if ic_recovery else "NO RECOVERY"
    fig.suptitle(
        f"Probe 7: Molecular-Scale Kernel — {recovery_label}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["teal"],
    )
    fig.tight_layout()
    _save(fig, "probe_07_molecular_kernel.png")

    return {
        "scale_ladder_IC": dict(zip(["quarks", "hadrons", "atoms", "molecules"], scale_ics, strict=False)),
        "ic_recovery": ic_recovery,
        "mol_results": mol_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBE 8: Wave Q-factor vs IC — Quality correlates with integrity?
# ═══════════════════════════════════════════════════════════════════════════
def probe_8_wave_q_factor() -> dict:
    """Correlate Q-factor with kernel integrity across 24 wave systems."""
    print("\n╔══ PROBE 8: Wave Q-Factor vs IC Correlation ══╗")

    wave_results = compute_all_wave_systems()

    # Extract Q-factors from WAVE_SYSTEMS data
    q_factors = [ws[5] for ws in WAVE_SYSTEMS]  # Index 5 is Q_factor
    names = [ws[0] for ws in WAVE_SYSTEMS]
    types = [ws[1] for ws in WAVE_SYSTEMS]

    ic_values = [r.IC for r in wave_results]
    f_values = [r.F for r in wave_results]
    gap_values = [r.gap for r in wave_results]
    omega_values = [r.omega for r in wave_results]

    # Spearman correlations
    rho_ic, p_ic = stats.spearmanr(q_factors, ic_values)
    rho_f, p_f = stats.spearmanr(q_factors, f_values)
    rho_gap, p_gap = stats.spearmanr(q_factors, gap_values)
    rho_omega, p_omega = stats.spearmanr(q_factors, omega_values)

    print(f"  N = {len(wave_results)} wave systems")
    print(f"  Q-factor ↔ IC:    ρ = {rho_ic:+.4f}  (p = {p_ic:.4f})")
    print(f"  Q-factor ↔ F:     ρ = {rho_f:+.4f}  (p = {p_f:.4f})")
    print(f"  Q-factor ↔ gap:   ρ = {rho_gap:+.4f}  (p = {p_gap:.4f})")
    print(f"  Q-factor ↔ ω:     ρ = {rho_omega:+.4f}  (p = {p_omega:.4f})")

    # Color by wave type
    type_colors = {
        "sound": COLORS["cyan"],
        "electromagnetic": COLORS["yellow"],
        "water": COLORS["blue"],
        "seismic": COLORS["orange"],
        "gravitational": COLORS["purple"],
        "matter": COLORS["green"],
    }

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Top left: Q vs IC (main correlation)
    ax = axes[0, 0]
    for wtype in sorted(set(types)):
        mask = [i for i, t in enumerate(types) if t == wtype]
        ax.scatter(
            [q_factors[i] for i in mask],
            [ic_values[i] for i in mask],
            color=type_colors.get(wtype, COLORS["gray"]),
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            label=wtype,
            zorder=5,
        )
    # Add labels for interesting points
    for i, name in enumerate(names):
        if q_factors[i] > 500 or ic_values[i] > 0.5 or ic_values[i] < 0.05:
            ax.annotate(
                name,
                (q_factors[i], ic_values[i]),
                fontsize=6,
                color=COLORS["gray"],
                textcoords="offset points",
                xytext=(5, 3),
            )
    ax.set_xscale("log")
    ax.set_xlabel("Q-factor (log scale)")
    ax.set_ylabel("Integrity Composite IC")
    ax.set_title(f"Q-factor ↔ IC   (ρ = {rho_ic:+.3f}, p = {p_ic:.3f})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Top right: Q vs F
    ax = axes[0, 1]
    for wtype in sorted(set(types)):
        mask = [i for i, t in enumerate(types) if t == wtype]
        ax.scatter(
            [q_factors[i] for i in mask],
            [f_values[i] for i in mask],
            color=type_colors.get(wtype, COLORS["gray"]),
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            label=wtype,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Q-factor (log scale)")
    ax.set_ylabel("Fidelity F")
    ax.set_title(f"Q-factor ↔ F   (ρ = {rho_f:+.3f}, p = {p_f:.3f})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Bottom left: Q vs heterogeneity gap
    ax = axes[1, 0]
    for wtype in sorted(set(types)):
        mask = [i for i, t in enumerate(types) if t == wtype]
        ax.scatter(
            [q_factors[i] for i in mask],
            [gap_values[i] for i in mask],
            color=type_colors.get(wtype, COLORS["gray"]),
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            label=wtype,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Q-factor (log scale)")
    ax.set_ylabel("Heterogeneity Gap (F − IC)")
    ax.set_title(f"Q-factor ↔ Gap   (ρ = {rho_gap:+.3f}, p = {p_gap:.3f})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Bottom right: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    summary_data = [
        ["Correlation", "Spearman ρ", "p-value", "Significant?"],
        ["Q ↔ IC", f"{rho_ic:+.4f}", f"{p_ic:.4f}", "YES" if p_ic < 0.05 else "NO"],
        ["Q ↔ F", f"{rho_f:+.4f}", f"{p_f:.4f}", "YES" if p_f < 0.05 else "NO"],
        ["Q ↔ Gap", f"{rho_gap:+.4f}", f"{p_gap:.4f}", "YES" if p_gap < 0.05 else "NO"],
        ["Q ↔ ω", f"{rho_omega:+.4f}", f"{p_omega:.4f}", "YES" if p_omega < 0.05 else "NO"],
    ]

    table = ax.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style the table
    for i in range(len(summary_data[0])):
        table[0, i].set_facecolor(COLORS["blue"])
        table[0, i].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            table[i, j].set_facecolor("#161b22")
            table[i, j].set_text_props(color="#e6edf3")
            if j == 3:  # Significant column
                color = COLORS["green"] if summary_data[i][3] == "YES" else COLORS["red"]
                table[i, j].set_text_props(color=color, fontweight="bold")

    ax.set_title("Correlation Summary", fontsize=13, fontweight="bold", pad=20)

    sig_label = "SIGNIFICANT" if p_ic < 0.05 else "NOT SIGNIFICANT"
    fig.suptitle(
        f"Probe 8: Wave Q-Factor ↔ Kernel Integrity — {sig_label}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["yellow"],
    )
    fig.tight_layout()
    _save(fig, "probe_08_wave_q_factor.png")

    return {
        "n_systems": len(wave_results),
        "correlations": {
            "Q_vs_IC": {"rho": float(rho_ic), "p": float(p_ic)},
            "Q_vs_F": {"rho": float(rho_f), "p": float(p_f)},
            "Q_vs_gap": {"rho": float(rho_gap), "p": float(p_gap)},
            "Q_vs_omega": {"rho": float(rho_omega), "p": float(p_omega)},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY: Comprehensive results dashboard
# ═══════════════════════════════════════════════════════════════════════════
def generate_summary(results: dict) -> None:
    """Generate a summary dashboard figure."""
    print("\n╔══ SUMMARY: Generating Results Dashboard ══╗")

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")

    # Title
    fig.text(
        0.5,
        0.97,
        "GCD KERNEL: 8 EXPERIMENTAL PROBES — RESULTS DASHBOARD",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=COLORS["cyan"],
    )

    # Create a grid of text panels
    n_probes = 8
    cols = 4
    probe_titles = [
        "1. Mass-Channel Removal",
        "2. SUSY Catalog",
        "3. Variable-Q Quarks",
        "4. Dark Matter Kernel",
        "5. Fixed-Channel Periodic",
        "6. GUT-Scale Coupling",
        "7. Molecular-Scale IC",
        "8. Wave Q-Factor",
    ]

    findings = []

    # Probe 1
    p1 = results.get("probe_1", {})
    mono = p1.get("monotone_no_mass", None)
    findings.append(
        f"Monotonic after removal: {'YES ✓' if mono else 'NO ✗' if mono is not None else '?'}\n"
        f"Full: {p1.get('full', {}).get(1, '?'):.3f} → {p1.get('full', {}).get(3, '?'):.3f}\n"
        f"No mass: {p1.get('no_mass', {}).get(1, '?'):.3f} → {p1.get('no_mass', {}).get(3, '?'):.3f}"
        if p1
        else "Not computed"
    )

    # Probe 2
    p2 = results.get("probe_2", {})
    findings.append(
        f"SM gap: {p2.get('sm_gap', '?'):.4f}\n"
        f"SUSY gap: {p2.get('susy_gap', '?'):.4f}\n"
        f"Gap reduction: {p2.get('gap_reduction_pct', '?'):.0f}%"
        if p2
        else "Not computed"
    )

    # Probe 3
    p3 = results.get("probe_3", {})
    findings.append(
        f"Up quark: IC {p3.get('up_IC_1GeV', '?'):.4f} → {p3.get('up_IC_10TeV', '?'):.4f}\n"
        f"Top quark: IC {p3.get('top_IC_1GeV', '?'):.4f} → {p3.get('top_IC_10TeV', '?'):.4f}\n"
        f"Hadron ref: ⟨IC⟩ = {p3.get('hadron_avg_IC', '?'):.4f}"
        if p3
        else "Not computed"
    )

    # Probe 4
    p4 = results.get("probe_4", {})
    dm = p4.get("dm_results", [])
    if dm:
        wimp_f = next((d["F"] for d in dm if "medium" in d["name"]), "?")
        axion_f = next((d["F"] for d in dm if "Axion" in d["name"]), "?")
        findings.append(
            f"WIMP (100 GeV): F = {wimp_f:.4f}\n"
            f"Axion: F = {axion_f:.4f}\n"
            f"SM ν ref: F = {p4.get('sm_reference', {}).get('neutrino_F', '?'):.4f}"
        )
    else:
        findings.append("Not computed")

    # Probe 5
    p5 = results.get("probe_5", {})
    findings.append(
        f"Universal channels: {len(p5.get('universal_channels', []))}\n"
        f"d-block dominant (var): {p5.get('d_block_dominant_variable', '?')}\n"
        f"d-block dominant (fix): {p5.get('d_block_dominant_fixed', '?')}"
        if p5
        else "Not computed"
    )

    # Probe 6
    p6 = results.get("probe_6", {})
    findings.append(
        f"Triangle spread: Δ(1/α) = {p6.get('triangle_spread', '?'):.2f}\n"
        + "\n".join(f"{k}: Q ≈ {v:.1e}" for k, v in p6.get("crossings", {}).items() if v is not None)
        if p6
        else "Not computed"
    )

    # Probe 7
    p7 = results.get("probe_7", {})
    ladder = p7.get("scale_ladder_IC", {})
    findings.append(
        f"IC recovery: {'YES ✓' if p7.get('ic_recovery') else 'NO ✗'}\n"
        f"Quarks→Hadrons→Atoms→Mols:\n"
        f"{ladder.get('quarks', '?'):.3f}→{ladder.get('hadrons', '?'):.3f}→{ladder.get('atoms', '?'):.3f}→{ladder.get('molecules', '?'):.3f}"
        if p7
        else "Not computed"
    )

    # Probe 8
    p8 = results.get("probe_8", {})
    corr = p8.get("correlations", {}).get("Q_vs_IC", {})
    findings.append(
        f"Q ↔ IC: ρ = {corr.get('rho', '?'):+.4f}  p = {corr.get('p', '?'):.4f}\n"
        f"Significant: {'YES ✓' if corr.get('p', 1) < 0.05 else 'NO ✗'}\n"
        f"N = {p8.get('n_systems', '?')} wave systems"
        if p8
        else "Not computed"
    )

    for idx in range(n_probes):
        row = idx // cols
        col = idx % cols
        x = 0.02 + col * 0.25
        y = 0.82 - row * 0.45

        # Background box
        ax = fig.add_axes([x, y, 0.22, 0.40])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add box
        box = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02",
            facecolor="#161b22",
            edgecolor=COLORS["cyan"],
            linewidth=1.5,
        )
        ax.add_patch(box)

        # Title
        ax.text(
            0.5, 0.92, probe_titles[idx], ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["cyan"]
        )

        # Finding text
        ax.text(
            0.5,
            0.72,
            findings[idx],
            ha="center",
            va="top",
            fontsize=8,
            color="#e6edf3",
            family="monospace",
            linespacing=1.5,
        )

    _save(fig, "probe_00_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run all 8 experimental probes."""
    print("=" * 70)
    print("  GCD KERNEL: 8 EXPERIMENTAL PROBES")
    print("  Investigating physics from quarks to molecules")
    print("=" * 70)

    results = {}
    results["probe_1"] = probe_1_mass_removal()
    results["probe_2"] = probe_2_susy_catalog()
    results["probe_3"] = probe_3_quark_variable_q()
    results["probe_4"] = probe_4_dark_matter()
    results["probe_5"] = probe_5_fixed_channel_periodic()
    results["probe_6"] = probe_6_gut_scale_coupling()
    results["probe_7"] = probe_7_molecular_kernel()
    results["probe_8"] = probe_8_wave_q_factor()

    generate_summary(results)

    print("\n" + "=" * 70)
    print("  ALL 8 PROBES COMPLETE")
    print("  Images saved to images/probe_*.png")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
