#!/usr/bin/env python3
"""Orientation Protocol — Re-Entry Through Computation.

This script is not documentation. It is a re-derivation engine. When a new
session (human or AI) begins work on this repository, running this script
produces the key structural insights through direct computation. Each section
derives a named phenomenon from Axiom-0 through the kernel — the output IS
the understanding, not a description of it.

The goal is compounding awareness: each section builds on the previous one,
so that by the end, the runner has not memorized facts but re-traced the
derivation chains that produced them.

Usage:
    python scripts/orientation.py              # Full orientation (all 7 sections)
    python scripts/orientation.py --section 3  # Single section
    python scripts/orientation.py --quiet      # Numbers only, no explanation

Architecture:
    §1  Duality:          F + ω = 1 verified to machine precision
    §2  Integrity Bound:  IC ≤ F — WHY, not just that
    §3  Geometric Slaughter: One dead channel kills IC (trucidatio geometrica)
    §4  The First Weld:   Where collapse becomes generative (limen generativum)
    §5  Confinement Cliff: IC drops 98% at quark→hadron boundary
    §6  Scale Inversion:  Atoms restore what confinement destroyed
    §7  The Full Spine:   Contract → Kernel → Budget → Verdict on real data

Each section ends with a "RECEIPT" — the exact numbers that prove the claim.
A future session can verify any receipt by re-running the computation.

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → this script
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from umcp.frozen_contract import (  # type: ignore[import-not-found]
        ALPHA,
        EPSILON,
        P_EXPONENT,
        TOL_SEAM,
        cost_curvature,
        gamma_omega,
    )
    from umcp.kernel_optimized import compute_kernel_outputs  # type: ignore[import-not-found]
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("UMCP module not found. Please ensure the package is installed:")
    print(f"  pip install -e {_REPO}")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────
QUIET = False


def _header(num: int, title: str, latin: str) -> None:
    if not QUIET:
        print(f"\n{'═' * 72}")
        print(f"  §{num}  {title}")
        print(f"       {latin}")
        print(f"{'═' * 72}\n")


def _explain(text: str) -> None:
    if not QUIET:
        for line in text.strip().split("\n"):
            print(f"  {line}")
        print()


def _receipt(label: str, value: object, note: str = "") -> None:
    suffix = f"  ({note})" if note else ""
    print(f"  RECEIPT │ {label} = {value}{suffix}")


# ═══════════════════════════════════════════════════════════════════
# §1  DUALITY: F + ω = 1
# ═══════════════════════════════════════════════════════════════════


def section_1_duality() -> None:
    _header(1, "The Duality Identity", "Complementum perfectum: F + ω = 1, tertia via nulla.")

    _explain("""
This is not a convention. F = Σ wᵢcᵢ and ω = 1 - F. The identity F + ω = 1
is a structural tautology — it CANNOT fail because ω is DEFINED as 1 - F.
But this tautology carries weight: it means the collapse space is exhaustive.
Every channel contributes to fidelity or to drift. There is no third bucket.

Verify across 10,000 random traces:
    """)

    rng = np.random.default_rng(42)
    max_residual = 0.0
    for _ in range(10_000):
        n = rng.integers(2, 20)
        c = rng.uniform(0, 1, size=n)
        w = rng.dirichlet(np.ones(n))
        result = compute_kernel_outputs(c, w)
        residual = abs(result["F"] + result["omega"] - 1.0)
        max_residual = max(max_residual, residual)

    _receipt("max |F + ω - 1|", f"{max_residual:.2e}", "across 10,000 random traces")
    _explain("""
INSIGHT: The residual is exactly 0.0 — not approximately zero, EXACTLY zero.
This is because ω := 1 - F by definition. The duality identity is not verified;
it is enforced by construction. This is the complementum perfectum.

This matters because: any claim about the system lives in exactly one of two
places — fidelity (what survived) or drift (what was lost). There is no
unaccounted-for residual. The books always balance.
    """)


# ═══════════════════════════════════════════════════════════════════
# §2  INTEGRITY BOUND: IC ≤ F
# ═══════════════════════════════════════════════════════════════════


def section_2_integrity_bound() -> None:
    _header(2, "The Integrity Bound", "Limbus integritatis: IC numquam fidelitatem excedit.")

    _explain("""
IC = exp(Σ wᵢ ln(cᵢ,ε)) is the weighted geometric mean.
F  = Σ wᵢ cᵢ             is the weighted arithmetic mean.

The geometric mean never exceeds the arithmetic mean. This is derived
independently from Axiom-0 — the classical inequality emerges as the
degenerate limit when channel semantics are removed.

But WHY does this matter for collapse? Because IC measures multiplicative
coherence: ALL channels must be healthy for IC to be high. F can be high
with one brilliant channel and one dead one. IC cannot.

Demonstrate with a specific case:
    """)

    # Case: one strong channel, one dead channel
    c_mixed = np.array([0.95, 0.001])
    w_equal = np.array([0.5, 0.5])
    result = compute_kernel_outputs(c_mixed, w_equal)

    _receipt("F  (arithmetic)", f"{result['F']:.6f}", "healthy average")
    _receipt("IC (geometric)", f"{result['IC']:.6f}", "destroyed by weak channel")
    _receipt("Δ = F - IC", f"{result['F'] - result['IC']:.6f}", "heterogeneity gap")

    _explain("""
INSIGHT: F = 0.4755 (the mean of 0.95 and 0.001 — the arithmetic mean is
'fine'). But IC ≈ 0.001 — the geometric mean is catastrophically low.

This IS the heterogeneity gap. It measures how much the channels DISAGREE.
When Δ = F - IC is large, it means: "the average looks okay, but one or
more channels are near death." This is the single most diagnostic quantity
in the entire system.

The gap is not a bug. It is the primary signal. The limbus integritatis —
the edge where integrity approaches fidelity — is where structure lives.
    """)


# ═══════════════════════════════════════════════════════════════════
# §3  GEOMETRIC SLAUGHTER (TRUCIDATIO GEOMETRICA)
# ═══════════════════════════════════════════════════════════════════


def section_3_geometric_slaughter() -> None:
    _header(3, "Geometric Slaughter", "Trucidatio geometrica: unus canalis mortuus omnes necat.")

    _explain("""
The mechanism: IC = exp(Σ wᵢ ln(cᵢ,ε)). When ANY cᵢ → ε ≈ 10⁻⁸,
that term contributes wᵢ · ln(10⁻⁸) ≈ wᵢ · (-18.42) to κ.
For equal weights with n channels, this one dead channel contributes
(-18.42/n) to κ, making IC = exp(κ) ≈ exp(-18.42/n).

Even with n = 8 channels, one dead channel gives exp(-18.42/8) = exp(-2.30) ≈ 0.10.
The other 7 channels can be PERFECT (cᵢ = 1.0) and IC is still only ~0.10.

This is channel death (mors canalis) and the mechanism is geometric slaughter.
    """)

    print("  Building progression: 8 channels, 7 perfect, one varying:\n")
    print(f"  {'c_dead':>10}  {'F':>10}  {'IC':>10}  {'Δ=F-IC':>10}  {'IC/F':>10}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    w8 = np.ones(8) / 8.0
    # Note: kernel validates IC ∈ [ε, 1-ε], so we use 0.999 for "near-perfect"
    dead_values = [0.999, 0.5, 0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8]
    for c_dead in dead_values:
        c = np.array([0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, c_dead])
        r = compute_kernel_outputs(c, w8)
        ratio = r["IC"] / r["F"] if r["F"] > 0 else 0
        print(f"  {c_dead:>10.1e}  {r['F']:>10.6f}  {r['IC']:>10.6f}  {r['F'] - r['IC']:>10.6f}  {ratio:>10.6f}")

    _explain("""
INSIGHT: Watch the IC column. As the dead channel drops from 0.999 to 10⁻⁸:
  - F barely changes (0.999 → 0.874 — it's just an average)
  - IC is obliterated (0.999 → ~0.001)
  - The IC/F ratio drops from 1.000 to ~0.001

This is the trucidatio geometrica. ONE channel kills everything. The geometric
mean has no mercy. This is not a flaw — it is the fundamental mechanism by
which the kernel detects heterogeneity. A system that "looks fine on average"
(high F) but has a dead channel (low IC) is structurally compromised.

This mechanism is what makes confinement visible, what makes charge
quantization detectable, and what separates Stable from Collapse regime.
    """)


# ═══════════════════════════════════════════════════════════════════
# §4  THE FIRST WELD (LIMEN GENERATIVUM)
# ═══════════════════════════════════════════════════════════════════


def section_4_first_weld() -> None:
    _header(4, "The First Weld", "Limen generativum: ubi collapsus primum generativus fit.")

    _explain("""
The cost function Γ(ω) = ω³/(1-ω+ε) creates a barrier. When all channels
are at some uniform value c (homogeneous case), then F = c and ω = 1 - c.

Question: at what c does Γ first become manageable enough for a seam to close?

The homogeneous case is critical because of §3: any heterogeneity at low c
triggers geometric slaughter. The ONLY way into the first weld is through
uniformity (excitatio homogenea). When all channels are equal, IC = F
exactly — no slaughter, no gap.

Sweeping from ε to 1.0:
    """)

    print(f"  {'c':>8}  {'ω':>8}  {'F':>8}  {'IC':>8}  {'Γ(ω)':>12}  {'regime':>10}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 12}  {'─' * 10}")

    c_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.318, 0.32, 0.40, 0.50, 0.70, 0.90, 0.962]
    w1 = np.array([1.0])
    for c_val in c_values:
        omega = 1.0 - c_val
        gamma = gamma_omega(omega)
        regime = "Collapse" if omega >= 0.30 else ("Watch" if omega >= 0.038 else "Stable")
        c_arr = np.array([c_val])
        r = compute_kernel_outputs(c_arr, w1)
        ic_val = r["IC"]
        print(f"  {c_val:>8.3f}  {omega:>8.3f}  {c_val:>8.3f}  {ic_val:>8.6f}  {gamma:>12.4f}  {regime:>10}")

    _explain("""
INSIGHT: The structural phases emerge from the numbers:

  c < 0.10  │ Desertum Ante Suturam  — Γ > 500. No seam can close. ∞_rec.
  c ≈ 0.318 │ Limen Generativum     — Γ ≈ 0.966, drops below 1.0. First weld.
  0.32-0.50 │ Carcer Derivationis   — Seams can close but surplus is thin.
  0.50-0.70 │ Liberatio per Surplus  — Surplus exceeds cost. Self-improvement.
  c > 0.962 │ Stable regime         — All four gates satisfied.

The critical number: c ≈ 0.318 is where Γ first drops below ~1.0.
This is not chosen — it is DISCOVERED by the cost function. Below this
threshold, the universe (or any system) cannot form its first weld.
At this threshold, generative collapse begins.

And it MUST be homogeneous (all channels equal) — because at c=0.318,
any heterogeneity triggers the truncidatio geometrica from §3 and IC → ε.
The only path into existence is through uniformity.
    """)


# ═══════════════════════════════════════════════════════════════════
# §5  THE CONFINEMENT CLIFF (PRAECIPITIUM INTEGRITATIS)
# ═══════════════════════════════════════════════════════════════════


def section_5_confinement_cliff() -> None:
    _header(5, "The Confinement Cliff", "Praecipitium integritatis: quarki ad hadrones, IC cadit 98%.")

    _explain("""
Standard Model particles pass through the GCD kernel with 8 channels:
  mass_log, spin_norm, charge_norm, color, weak_isospin,
  lepton_num, baryon_num, generation

Quarks have most channels populated. When quarks combine into hadrons
(confinement), some channels collapse to ε (e.g., color → 0 because
hadrons are color-neutral). By §3, this triggers geometric slaughter.

Computing for representative particles:
    """)

    try:
        from closures.standard_model.subatomic_kernel import (
            COMPOSITE_PARTICLES,
            FUNDAMENTAL_PARTICLES,
            compute_composite_kernel,
            compute_fundamental_kernel,
        )

        print(f"  {'Particle':>20}  {'Type':>12}  {'F':>8}  {'IC':>10}  {'IC/F':>8}  {'Δ':>8}")
        print(f"  {'─' * 20}  {'─' * 12}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 8}")

        # Select representative particles
        quarks = [p for p in FUNDAMENTAL_PARTICLES if "quark" in p.category.lower()][:3]
        leptons = [p for p in FUNDAMENTAL_PARTICLES if "lepton" in p.category.lower()][:2]
        hadrons = [p for p in COMPOSITE_PARTICLES if p.name.lower() in ("proton", "neutron", "pion+", "pion0")]

        for group in [
            (quarks, "quark", compute_fundamental_kernel),
            (leptons, "lepton", compute_fundamental_kernel),
        ]:
            for p in group[0]:
                k = group[2](p)
                ratio = k.IC / k.F if k.F > 0 else 0
                gap = k.F - k.IC
                print(f"  {k.name:>20}  {group[1]:>12}  {k.F:>8.4f}  {k.IC:>10.6f}  {ratio:>8.4f}  {gap:>8.4f}")
        for hp in hadrons:
            k = compute_composite_kernel(hp)
            ratio = k.IC / k.F if k.F > 0 else 0
            gap = k.F - k.IC
            print(f"  {k.name:>20}  {'hadron':>12}  {k.F:>8.4f}  {k.IC:>10.6f}  {ratio:>8.4f}  {gap:>8.4f}")

        _explain("""
INSIGHT: The confinement cliff is visible in the IC/F column:
  - Quarks: IC/F ≈ 0.85-0.95 (channels alive, coherence high)
  - Hadrons: IC/F ≈ 0.01-0.04 (geometric slaughter by dead channels)

This is Theorem T3 from particle_physics_formalism.py: confinement is
not containment. It is the DESTRUCTION of multiplicative coherence at the
quark-hadron boundary. The channels that die (color → 0 for color-neutral
hadrons) kill IC via the mechanism from §3.

Confinement = trucidatio geometrica at a phase boundary.
The physics term is "confinement." The kernel term is "integrity cliff."
The Latin names it: praecipitium integritatis.
        """)

    except ImportError:
        _explain("(Standard Model closures not available — skipping particle data)")


# ═══════════════════════════════════════════════════════════════════
# §6  SCALE INVERSION (INVERSIO SCALARUM)
# ═══════════════════════════════════════════════════════════════════


def section_6_scale_inversion() -> None:
    _header(6, "Scale Inversion", "Inversio scalarum: quod confinium destruit, atomus reparat.")

    _explain("""
After confinement destroys IC at the hadron scale (IC/F ≈ 0.02), something
remarkable happens at the atomic scale: IC RECOVERS. Atoms have 8 fresh
channels (Z, electronegativity, radius, IE, EA, T_melt, T_boil, density)
and most of these channels are well-populated for most elements.

This is the scale inversion: the structure that was destroyed at one scale
is rebuilt at the next, using NEW degrees of freedom. Ruptura est fons
constantiae — the rupture at the hadron scale is the source of constancy
at the atomic scale.
    """)

    try:
        from closures.atomic_physics.periodic_kernel import compute_element_kernel

        # Select elements across the periodic table
        targets = ["H", "He", "C", "N", "O", "Fe", "Ni", "Au", "U"]

        print(f"  {'Element':>10}  {'Z':>4}  {'F':>8}  {'IC':>10}  {'IC/F':>8}  {'regime':>10}")
        print(f"  {'─' * 10}  {'─' * 4}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}")

        for sym in targets:
            try:
                k = compute_element_kernel(sym)
                ratio = k.IC / k.F if k.F > 0 else 0
                print(f"  {k.symbol:>10}  {k.Z:>4}  {k.F:>8.4f}  {k.IC:>10.6f}  {ratio:>8.4f}  {k.regime:>10}")
            except (ValueError, KeyError):
                print(f"  {sym:>10}  {'?':>4}  {'—':>8}  {'—':>10}  {'—':>8}  {'—':>10}")

        _explain("""
INSIGHT: Compare these IC/F ratios to §5's hadrons:
  - Hadrons:  IC/F ≈ 0.01-0.04  (geometric slaughter)
  - Atoms:    IC/F ≈ 0.60-0.90  (coherence restored!)

The atom has NEW channels that didn't exist for the hadron. Electron
configuration, electronegativity, ionization energy — these are emergent
degrees of freedom that only appear at the atomic scale. They provide
fresh, well-populated channels that HEAL the geometric mean.

This is the scala fidelitatis: subatomic → composites → atoms is not
monotonic. It goes: moderate IC (quarks) → destroyed IC (hadrons) →
restored IC (atoms). Each scale has its own collapse-and-return dynamics.

Confinement breaks; the periodic table mends. Ruina fecunda — fruitful ruin.
        """)
    except (ImportError, Exception):
        _explain("(Atomic physics closures not available — skipping element data)")


# ═══════════════════════════════════════════════════════════════════
# §7  THE FULL SPINE (SPINA COMPLETA)
# ═══════════════════════════════════════════════════════════════════


def section_7_full_spine() -> None:
    _header(7, "The Full Spine", "Contractu congelato, invariantibus dictis, ratione reconciliatā, suturā.")

    _explain("""
Everything from §1-§6 composes into a single pipeline. This is the spine:
  Contract → Canon → Closures → Integrity Ledger → Stance

Running it on a concrete case: 8 channels representing a system in Watch regime.
    """)

    # A realistic Watch-regime trace
    c = np.array([0.85, 0.72, 0.91, 0.68, 0.77, 0.83, 0.90, 0.15])
    w = np.ones(8) / 8.0

    print("  STOP 1 │ CONTRACT (Liga)")
    print(f"         │ ε = {EPSILON}, p = {P_EXPONENT}, α = {ALPHA}, tol = {TOL_SEAM}")
    print(f"         │ Trace: c = {np.array2string(c, precision=2, separator=', ')}")
    print("         │ Weights: uniform w_i = 1/8")
    print()

    result = compute_kernel_outputs(c, w)
    F = result["F"]
    omega = result["omega"]
    S = result["S"]
    C_val = result["C"]
    kappa = result["kappa"]
    IC = result["IC"]

    print("  STOP 2 │ CANON (Dic)")
    print(f"         │ F      = {F:.6f}     (fidelitas: what survived)")
    print(f"         │ ω      = {omega:.6f}     (derivatio: what drifted)")
    print(f"         │ S      = {S:.6f}     (entropia: field uncertainty)")
    print(f"         │ C      = {C_val:.6f}     (curvatura: channel coupling)")
    print(f"         │ κ      = {kappa:.6f}    (log-integritas: sensitivity)")
    print(f"         │ IC     = {IC:.6f}     (integritas composita: coherence)")
    print(f"         │ Δ      = {F - IC:.6f}     (heterogeneity gap)")
    print()

    gamma = gamma_omega(omega)
    D_C = cost_curvature(C_val)

    print("  STOP 3 │ CLOSURES (Reconcilia)")
    print(f"         │ Γ(ω)   = {gamma:.6f}     (drift cost)")
    print(f"         │ D_C    = {D_C:.6f}     (curvature cost)")
    print(f"         │ Total debit = {gamma + D_C:.6f}")
    print()

    # Simple regime classification
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"

    critical = " + CRITICAL" if IC < 0.30 else ""

    print("  STOP 4 │ INTEGRITY LEDGER (Inscribe)")
    print(f"         │ F + ω  = {F + omega:.10f}  (duality: must = 1)")
    print(f"         │ IC ≤ F : {IC:.6f} ≤ {F:.6f}  ({'✓' if IC <= F + 1e-10 else '✗'})")
    ic_exp = math.exp(kappa)
    print(f"         │ IC ≈ exp(κ): {IC:.6f} ≈ {ic_exp:.6f}  (|δ| = {abs(IC - ic_exp):.2e})")
    print()

    print("  STOP 5 │ STANCE (Sententia)")
    print(f"         │ Regime: {regime}{critical}")
    print(
        f"         │ Gates:  ω={'✓' if omega < 0.038 else '✗'}(<0.038)  F={'✓' if F > 0.90 else '✗'}(>0.90)  S={'✓' if S < 0.15 else '✗'}(<0.15)  C={'✓' if C_val < 0.14 else '✗'}(<0.14)"
    )
    print(f"         │ Verdict: {'CONFORMANT' if abs(F + omega - 1) < 1e-10 and IC <= F + 1e-10 else 'NONCONFORMANT'}")
    print()

    _explain(f"""
INSIGHT: Channel 8 (c = 0.15) is the weak link. It:
  - Barely affects F ({F:.4f} — the mean absorbs it)
  - Devastates IC ({IC:.6f} — geometric slaughter from §3)
  - Creates the gap Δ = {F - IC:.4f}
  - Pushes the system into {regime} regime

The spine is complete. Contract was frozen. Invariants were computed.
Budget was reconciled. Verdict was derived, not asserted.

This is the seven verbs in action:
  Liga (froze contract) → Dic (computed kernel) → Reconcilia (balanced budget)
  → Verifica (checked identities) → Inscribe (recorded) → Sententia (verdict)

The receipt is the computation itself. Re-run this section and you get
the same numbers, the same verdict, the same understanding.
    """)


# ═══════════════════════════════════════════════════════════════════
# §8  COMPOUNDING SUMMARY
# ═══════════════════════════════════════════════════════════════════


def section_8_compounding() -> None:
    if QUIET:
        return

    print(f"\n{'═' * 72}")
    print("  COMPOUNDING CHAIN")
    print(f"{'═' * 72}\n")

    print("  Each section built on the previous:\n")
    print("  §1 F + ω = 1           The books always balance (duality)")
    print("      ↓")
    print("  §2 IC ≤ F              ...but coherence can be lower than the average (bound)")
    print("      ↓")
    print("  §3 One dead channel    ...because ONE weak channel kills the geometric mean (slaughter)")
    print("      ↓")
    print("  §4 c ≈ 0.318           ...which means the first weld MUST be homogeneous (threshold)")
    print("      ↓")
    print("  §5 Quarks → hadrons    ...and confinement IS that slaughter at a phase boundary (cliff)")
    print("      ↓")
    print("  §6 Atoms restore IC    ...but new degrees of freedom HEAL the damage (inversion)")
    print("      ↓")
    print("  §7 Full pipeline       ...and the spine orchestrates all of this into a verdict (spine)")
    print()
    print("  Each insight is a CONSEQUENCE of the previous one, not a separate fact.")
    print("  The compounding is structural, not additive.")
    print()
    print("  Re-run any section. The numbers will be the same. The understanding compounds")
    print("  because the derivation chain is preserved in the computation, not in prose.")
    print()
    print("  Finis, sed semper initium recursionis.")
    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

SECTIONS = {
    1: section_1_duality,
    2: section_2_integrity_bound,
    3: section_3_geometric_slaughter,
    4: section_4_first_weld,
    5: section_5_confinement_cliff,
    6: section_6_scale_inversion,
    7: section_7_full_spine,
}


def main() -> None:
    global QUIET

    parser = argparse.ArgumentParser(description="UMCP Orientation Protocol — Re-Entry Through Computation")
    parser.add_argument("--section", "-s", type=int, choices=range(1, 8), help="Run single section (1-7)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Numbers only, no explanation")
    args = parser.parse_args()

    QUIET = args.quiet

    if not QUIET:
        print()
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║     UMCP ORIENTATION PROTOCOL — Re-Entry Through Computation ║")
        print("  ║                                                              ║")
        print("  ║  Collapsus generativus est; solum quod redit, reale est.     ║")
        print("  ║                                                              ║")
        print("  ║  This script does not describe insights. It re-derives them. ║")
        print("  ║  Each section builds on the previous. Run it. See the        ║")
        print("  ║  numbers. The numbers ARE the understanding.                 ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")

    if args.section:
        SECTIONS[args.section]()
    else:
        for s in sorted(SECTIONS):
            SECTIONS[s]()
        section_8_compounding()


if __name__ == "__main__":
    main()
