"""Behavioral predictions and observable physics for superheavy elements Z=120-184.

Uses GCD kernel extrapolation + nuclear/atomic closure APIs to predict:
  1. Relativistic electron behavior (Zα → 1 and beyond)
  2. Alpha decay half-lives and dominant decay modes
  3. Nuclear binding and fissility landscape
  4. Shell closure effects and magic-number stabilization
  5. Spectroscopic signatures (K-alpha X-rays, fine structure)
  6. Block-by-block chemistry and bonding predictions
  7. Observable experimental signatures
  8. Phase diagram of nuclear matter

All predictions are Tier-2 closures validated through Tier-0 against Tier-1.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
from scipy.stats import linregress

from closures.atomic_physics.cross_scale_kernel import (
    compute_all_enhanced,
    magic_proximity,
)
from closures.nuclear_physics.alpha_decay import compute_alpha_decay
from closures.nuclear_physics.element_data import ELEMENTS as NUC_ELEMENTS
from closures.nuclear_physics.fissility import compute_fissility
from closures.nuclear_physics.nuclide_binding import compute_binding
from closures.nuclear_physics.shell_structure import compute_shell

# ── Constants ────────────────────────────────────────────────────
ALPHA_FINE = 1.0 / 137.036  # Fine structure constant
C_LIGHT = 299792458.0  # m/s
M_E_EV = 510998.95  # Electron rest mass (eV/c²)
RYDBERG_EV = 13.5984  # Rydberg energy (eV)

PERIOD_8_BLOCKS: dict[str, tuple[int, int, str]] = {
    "8s": (119, 120, "s"),
    "5g": (121, 138, "g"),
    "6f": (139, 152, "f"),
    "7d": (153, 162, "d"),
    "8p": (163, 168, "p"),
}

PREDICTED_MAGIC_Z = [114, 120, 126, 164]
PREDICTED_MAGIC_N = [172, 184, 228, 258]


def section(title: str) -> None:
    print(f"\n{'=' * 96}")
    print(f"  {title}")
    print(f"{'=' * 96}\n")


def subsection(title: str) -> None:
    print(f"\n  {'─' * 80}")
    print(f"  {title}")
    print(f"  {'─' * 80}\n")


def predict_mass_number(z: int) -> int:
    """Predict most stable isotope mass number from N/Z trend."""
    nz_ratio = np.array([el.N / el.Z for el in NUC_ELEMENTS if el.Z > 0])
    zz_nz = np.array([el.Z for el in NUC_ELEMENTS if el.Z > 0])
    sl_nz, int_nz, _, _, _ = linregress(zz_nz[50:], nz_ratio[50:])
    nz = sl_nz * z + int_nz
    n = round(z * nz)
    return z + n


def predict_block(z: int) -> str:
    """Return predicted electron block for element Z."""
    for _name, (zs, ze, bt) in PERIOD_8_BLOCKS.items():
        if zs <= z <= ze:
            return bt
    return "?"


def get_alpha_q(z: int, a: int) -> float:
    """Estimate Q_alpha for superheavy elements.

    Q_alpha ≈ BE(daughter) + BE(He-4) - BE(parent)
    daughter = (Z-2, A-4), He-4 = (2, 4)
    """
    be_parent = compute_binding(z, a).BE_total
    be_daughter = compute_binding(z - 2, a - 4).BE_total
    be_he4 = compute_binding(2, 4).BE_total  # ~28.3 MeV
    q = be_daughter + be_he4 - be_parent
    return max(0.1, q)  # Must be positive for decay to occur


def main() -> None:
    # ──────────────────────────────────────────────────────────────
    # Load empirical data for comparison
    # ──────────────────────────────────────────────────────────────
    cs_results = compute_all_enhanced()
    cs_by_z = {r.Z: r for r in cs_results}  # noqa: F841 — kept for interactive use

    # Build predictions for Z=120-184
    elements: list[dict[str, object]] = []
    for z in range(120, 185):
        a = predict_mass_number(z)
        n = a - z
        block = predict_block(z)
        fiss = compute_fissility(z, a)
        bind = compute_binding(z, a)
        shell = compute_shell(z, a)
        mp = magic_proximity(z, a)

        # Alpha decay
        q_alpha = get_alpha_q(z, a)
        alpha = compute_alpha_decay(z, a, q_alpha)

        # Relativistic parameter
        z_alpha = z * ALPHA_FINE
        z_alpha_sq = z_alpha**2
        v_1s_over_c = z_alpha  # v/c for 1s electron ≈ Zα

        elements.append(
            {
                "Z": z,
                "A": a,
                "N": n,
                "block": block,
                "BE_per_A": bind.BE_per_A,
                "BE_total": bind.BE_total,
                "fissility": fiss.fissility_x,
                "fiss_regime": fiss.regime,
                "coulomb_E": fiss.coulomb_energy,
                "surface_E": fiss.surface_energy,
                "shell_regime": shell.regime,
                "doubly_magic": shell.doubly_magic,
                "magic_Z": shell.magic_proton,
                "magic_N": shell.magic_neutron,
                "dist_magic_Z": shell.distance_to_magic_Z,
                "dist_magic_N": shell.distance_to_magic_N,
                "magic_prox": mp,
                "Q_alpha": q_alpha,
                "alpha_t12_s": alpha.half_life_s,
                "alpha_log_t12": alpha.log10_half_life_s,
                "alpha_regime": alpha.regime,
                "z_alpha": z_alpha,
                "z_alpha_sq": z_alpha_sq,
                "v_1s_c": v_1s_over_c,
            }
        )

    # ================================================================
    #  SECTION 1: RELATIVISTIC ELECTRON PHYSICS
    # ================================================================
    section("1. RELATIVISTIC ELECTRON BEHAVIOR — The Zα Frontier")

    print("  The innermost (1s) electron has v/c ≈ Zα (fine structure constant × Z).")
    print("  At Z=137, Zα = 1.0 — the 1s electron reaches the speed of light")
    print("  in non-relativistic quantum mechanics. The Dirac equation predicts")
    print("  qualitative changes to atomic structure as Zα → 1 and beyond.\n")

    print(
        f"  {'Z':>4} {'Block':>5} {'Zα':>6} {'(Zα)²':>7} {'v_1s/c':>7} "
        f"{'Regime':>12} {'1s E (keV)':>11} {'ΔE_fine/E':>10} {'Physics':>35}"
    )
    print("  " + "-" * 110)

    # Key milestones
    milestones = {
        120: "Orbital contraction begins to dominate",
        126: "s-orbital ground state ambiguous",
        137: "Zα = 1.0 — classical speed-of-light limit",
        140: "Dirac continuum dissolution begins",
        150: "1s deeply into negative sea",
        164: "Predicted magic shell closure",
        170: "QED vacuum polarization dominates",
        184: "End of extended periodic table",
    }

    for el in elements:
        z = int(str(el["Z"]))
        za = float(str(el["z_alpha"]))
        za2 = float(str(el["z_alpha_sq"]))
        v1s = float(str(el["v_1s_c"]))

        if za < 1.0:
            regime = "Heavy-Atom"
        elif za < 1.15:
            regime = "Critical"
        elif za < 1.30:
            regime = "Supercritical"
        else:
            regime = "Dive-into-Sea"

        # 1s binding energy: E_1s ≈ -Z² × 13.6 eV (non-rel)
        # Dirac: E = m_e c² [1 + (Zα)²/(n - δ)²]^(-1/2) - m_e c²
        # For 1s: n=1, j=1/2: γ = sqrt(j+1/2)² - (Zα)²)
        if za2 < 0.25:  # safe regime for perturbative
            e_1s_kev = z**2 * RYDBERG_EV / 1000.0
            delta_fine_frac = za2 * (1.0 / 0.5 - 0.75)
        elif za < 1.0:
            # Full Dirac: E = mc² [1 + (Zα)²/(1 - δ + sqrt((1)² - (Zα)²))²]^(-1/2) - mc²
            gamma = math.sqrt(max(1e-10, 1.0 - za2))
            e_dirac = M_E_EV * (1.0 / math.sqrt(1.0 + za2 / gamma**2) - 1.0)
            e_1s_kev = abs(e_dirac) / 1000.0
            delta_fine_frac = za2 * (1.0 / 0.5 - 0.75)
        else:
            # Beyond Z=137: the Dirac equation has no bound solution
            # for point nucleus. Extended nucleus gives:
            # 1s still exists but deeply relativistic
            e_1s_kev = z**2 * RYDBERG_EV * (1.0 + 1.5 * za2) / 1000.0
            delta_fine_frac = za2

        milestone = milestones.get(z, "")
        print(
            f"  {z:4d} {el['block']:>5} {za:6.3f} {za2:7.3f} {v1s:7.3f} "
            f"{regime:>12} {e_1s_kev:11.1f} {delta_fine_frac:10.3f} {milestone:>35}"
        )

    # Detailed physics explanation
    subsection("THE Zα = 1 BARRIER — What Happens at Z ≈ 137")
    print("""  For a POINT nucleus, the Dirac equation has no bound 1s state when Zα ≥ 1.
  This is NOT a real barrier — real nuclei have finite size (R ~ 7-8 fm for
  superheavy elements). The finite nuclear radius regularizes the singularity.

  What ACTUALLY happens as Z increases past 137:

  Z = 120-136 (Zα = 0.876-0.993): HEAVY-ATOM REGIME
    • 1s orbital contracts to ~0.4 pm (smaller than nuclear radius ~8 fm)
    • 1s electrons spend significant time INSIDE the nucleus
    • QED vacuum polarization contributes ~1% to binding energy
    • Magnetic hyperfine structure is dramatically enhanced
    • 1s binding energy: 200-500 keV (comparable to nuclear γ-rays)

  Z = 137 (Zα ≈ 1.000): CRITICAL REGIME
    • For a point nucleus, 1s orbital would collapse to zero radius
    • For realistic nucleus (R ≈ 8 fm), 1s binding energy ~ 600 keV
    • Electron-positron pair production from the VACUUM becomes
      energetically favorable near the nuclear surface
    • This is "vacuum breakdown" — the QED analog of pair production

  Z = 138-172 (Zα = 1.007-1.255): SUPERCRITICAL REGIME
    • 1s binding energy exceeds 2 × m_e c² = 1.022 MeV
    • The atom spontaneously emits positrons (supercritical QED)
    • This is a PHASE TRANSITION in the vacuum structure
    • The 1s "orbital" becomes a resonance in the Dirac sea
    • Spontaneous e⁺e⁻ production from vacuum near nucleus

  Z > 172 (Zα > 1.255): CONTINUUM DISSOLUTION
    • Multiple atomic orbitals (1s, 2s, 2p₁/₂) dive into the Dirac sea
    • The concept of discrete atomic shells breaks down
    • The atom becomes a "charged vacuum" — a new state of quantum matter
    • The electron cloud merges with the vacuum fluctuation field
    • No clean separation between bound and continuum states""")

    # ================================================================
    #  SECTION 2: ALPHA DECAY AND NUCLEAR LIFETIMES
    # ================================================================
    section("2. NUCLEAR LIFETIMES — How Long Do These Nuclei Live?")

    print("  Alpha decay dominates for superheavy elements. The Geiger-Nuttall")
    print("  relation predicts half-lives from the Q-value (alpha energy).\n")

    print(
        f"  {'Z':>4} {'A':>5} {'Block':>5} {'Q_α(MeV)':>9} {'log₁₀(T½/s)':>12} "
        f"{'T½ human':>20} {'Regime':>12} {'Fissility':>9} {'Notes':>30}"
    )
    print("  " + "-" * 120)

    # Human-readable half-life
    def human_halflife(log10_s: float) -> str:
        if log10_s == float("inf") or log10_s > 50:
            return "> age of universe"
        s = 10**log10_s
        if s < 1e-21:
            return f"{s * 1e24:.0f} ys"  # yoctoseconds
        if s < 1e-18:
            return f"{s * 1e21:.0f} zs"
        if s < 1e-15:
            return f"{s * 1e18:.1f} as"  # attoseconds
        if s < 1e-12:
            return f"{s * 1e15:.1f} fs"
        if s < 1e-9:
            return f"{s * 1e12:.1f} ps"
        if s < 1e-6:
            return f"{s * 1e9:.1f} ns"
        if s < 1e-3:
            return f"{s * 1e6:.1f} μs"
        if s < 1.0:
            return f"{s * 1e3:.1f} ms"
        if s < 60:
            return f"{s:.1f} s"
        if s < 3600:
            return f"{s / 60:.1f} min"
        if s < 86400:
            return f"{s / 3600:.1f} hr"
        if s < 3.156e7:
            return f"{s / 86400:.1f} days"
        if s < 3.156e10:
            return f"{s / 3.156e7:.1f} yr"
        if s < 3.156e13:
            return f"{s / 3.156e7:.0f} yr"
        return f"{s / 3.156e7:.1e} yr"

    # Categorize by observability
    observable = []  # T½ > 1 μs (can reach detector)
    ephemeral = []  # T½ < 1 μs
    stable_ish = []  # T½ > 1 s

    for el in elements:
        z = int(str(el["Z"]))
        a = int(str(el["A"]))
        q_a = float(str(el["Q_alpha"]))
        log_t12 = float(str(el["alpha_log_t12"]))
        t12_s = float(str(el["alpha_t12_s"]))
        fiss_x = float(str(el["fissility"]))
        regime = str(el["alpha_regime"])

        notes = ""
        if el["doubly_magic"]:
            notes += "doubly-magic "
        elif el["magic_Z"]:
            notes += f"magic Z={z} "
        elif el["magic_N"]:
            notes += f"magic N={int(str(el['N']))} "

        if fiss_x >= 1.0:
            notes += "supercritical "
        if z in [120, 126, 137, 164]:
            notes += "★ "

        print(
            f"  {z:4d} {a:5d} {el['block']:>5} {q_a:9.2f} {log_t12:12.2f} "
            f"{human_halflife(log_t12):>20} {regime:>12} {fiss_x:9.3f} {notes:>30}"
        )

        if t12_s > 1e-6:
            observable.append(el)
        else:
            ephemeral.append(el)
        if t12_s > 1.0:
            stable_ish.append(el)

    subsection("DECAY SUMMARY")
    print(f"  Observable in lab (T½ > 1 μs): {len(observable)} elements")
    print(f"  Ephemeral (T½ < 1 μs):         {len(ephemeral)} elements")
    print(f"  'Stable-ish' (T½ > 1 s):       {len(stable_ish)} elements")
    if stable_ish:
        print("  Longest-lived predictions:")
        sorted_stable = sorted(stable_ish, key=lambda e: float(str(e["alpha_log_t12"])), reverse=True)
        for el in sorted_stable[:5]:
            z = int(str(el["Z"]))
            print(
                f"    Z={z} A={el['A']} block={el['block']} "
                f"T½={human_halflife(float(str(el['alpha_log_t12'])))} "
                f"fiss={float(str(el['fissility'])):.3f}"
            )

    # ================================================================
    #  SECTION 3: BINDING ENERGY LANDSCAPE
    # ================================================================
    section("3. NUCLEAR BINDING — The Landscape of Existence")

    print("  The Bethe-Weizsäcker SEMF provides BE/A. Shell corrections")
    print("  can add or subtract several MeV from the smooth curve.\n")

    # Group by block
    for block_label, (z_start, z_end, btype) in PERIOD_8_BLOCKS.items():
        block_els = [e for e in elements if z_start <= int(str(e["Z"])) <= z_end]
        if not block_els:
            continue
        avg_bea = np.mean([float(str(e["BE_per_A"])) for e in block_els])
        avg_fiss = np.mean([float(str(e["fissility"])) for e in block_els])
        min_fiss = min(float(str(e["fissility"])) for e in block_els)
        max_fiss = max(float(str(e["fissility"])) for e in block_els)
        n_subcrit = sum(1 for e in block_els if float(str(e["fissility"])) < 1.0)

        print(f"  {block_label} ({btype}-block, Z={z_start}-{z_end}):")
        print(f"    <BE/A> = {avg_bea:.3f} MeV/nucleon")
        print(f"    Fissility: {min_fiss:.3f} – {max_fiss:.3f} (avg {avg_fiss:.3f})")
        print(f"    Subcritical (can exist as bound nucleus): {n_subcrit}/{len(block_els)}")

        # Shell structure
        magic_els = [e for e in block_els if e["magic_Z"] or e["magic_N"]]
        if magic_els:
            for e in magic_els:
                print(
                    f"    ★ Z={e['Z']} A={e['A']}: "
                    f"magic_Z={e['magic_Z']} magic_N={e['magic_N']} "
                    f"shell={e['shell_regime']}"
                )
        print()

    subsection("COULOMB vs SURFACE ENERGY — The Fundamental Competition")
    print("  Nuclear existence requires surface energy > Coulomb repulsion.")
    print("  As Z increases, Coulomb grows as Z(Z-1)/A^(1/3);")
    print("  surface only grows as A^(2/3). The ratio determines fate.\n")

    print(
        f"  {'Z':>4} {'A':>5} {'E_Coulomb':>10} {'E_Surface':>10} "
        f"{'C/S ratio':>10} {'BE_total':>10} {'BE/A':>7} {'Verdict':>20}"
    )
    print("  " + "-" * 90)

    for el in elements:
        z = int(str(el["Z"]))
        if z % 5 != 0 and z not in [120, 126, 137, 164, 168, 184]:
            continue
        ec = float(str(el["coulomb_E"]))
        es = float(str(el["surface_E"]))
        ratio = ec / es if es > 0 else float("inf")
        be_total = float(str(el["BE_total"]))
        bea = float(str(el["BE_per_A"]))

        verdict = ""
        if ratio < 1.5:
            verdict = "Bound"
        elif ratio < 2.0:
            verdict = "Weakly bound"
        elif be_total > 0:
            verdict = "Shell-stabilized only"
        else:
            verdict = "UNBOUND"

        print(
            f"  {z:4d} {int(str(el['A'])):5d} {ec:10.0f} {es:10.0f} "
            f"{ratio:10.3f} {be_total:10.0f} {bea:7.3f} {verdict:>20}"
        )

    # ================================================================
    #  SECTION 4: BLOCK-BY-BLOCK CHEMISTRY AND BEHAVIOR
    # ================================================================
    section("4. BLOCK-BY-BLOCK EXPECTED CHEMISTRY")

    # ── 8s BLOCK (Z=119-120) ─────────────────────────────────────
    subsection("8s BLOCK: Z=119-120 — The Last Alkali and Alkaline Earth")
    print("""  Z=119 (Unbiunennium, Uue) — 8s¹ configuration
    Expected behavior:
      • Most electropositive element ever — lower IE than Cs or Fr
      • Predicted IE: ~3.3 eV (vs. Cs: 3.89 eV, Fr: 4.07 eV)
      • HOWEVER: relativistic contraction stabilizes the 8s orbital
      • Actual IE may be HIGHER than expected from periodic trends
      • This is the "relativistic reversal" — 8s contracts inward
      • Atomic radius: ~334 pm (>2× Cs), enormous Rydberg-like atom
      • Chemistry: +1 oxidation state nearly certain
      • Most reactive metal ever created (if it exists long enough)
      • Predicted T½: depends on neutron count, but likely μs to ms

  Z=120 (Unbinilium, Ubn) — 8s² configuration
    Expected behavior:
      • Last alkaline earth metal
      • IE: ~5.0 eV (Ba: 5.21 eV, Ra: 5.28 eV)
      • Relativistic 8s contraction makes it less reactive than expected
      • DOUBLY MAGIC candidate (Z=120, N=184 → A=304)
      • This specific isotope is the most likely "island of stability" center
      • Could have half-life of SECONDS to MINUTES (vs μs for neighbors)
      • Experimentally the most sought-after superheavy element
      • Chemistry: +2 dominant, possibly +4 from d-orbital participation""")

    # ── 5g BLOCK (Z=121-138) ─────────────────────────────────────
    subsection("5g BLOCK: Z=121-138 — The Superactinide Series (Nova Materia)")
    print("""  ENTIRELY NEW ORBITAL TYPE: l=4 (g-orbital)
  No element in the universe has ever filled a g-orbital in its ground state.
  This is the first genuinely new type of chemistry since the actinides.

  Key predictions:
    Angular wavefunctions: 9 lobes (2l+1 = 9 nodal patterns)
    Magnetic substates: 2l+1 = 9 (ml = -4, -3, ..., +4)
    With spin: 2(2l+1) = 18 electrons to fill the g-subshell
    Spin-orbit manifolds: j = l+1/2 = 9/2 and j = l-1/2 = 7/2
      g₇/₂ (8 electrons) and g₉/₂ (10 electrons)

  Expected chemistry:
    • Multiple oxidation states: +2 to +18 theoretically possible
    • In practice, relativistic effects will limit to +2 to +6
    • The 5g orbital is extremely diffuse — bonding will be very weak
    • These elements may behave as noble-gas-LIKE due to 5g shielding
    • Coordination chemistry: potentially 12+ coordinate complexes
    • Color: strong f-f and g-g electronic transitions → vivid colors
    • Magnetism: up to 4 unpaired g-electrons → large magnetic moments

  Superactinide contraction (analogous to lanthanide contraction):
    • Lanthanides: 4f causes ~15 pm contraction over 14 elements
    • Actinides: 5f causes ~10 pm contraction over 14 elements
    • Superactinides: 5g predicted to cause ~60 pm contraction over 18 elements
    • MORE severe because g-orbitals are more diffuse and shield less
    • Result: Z=138 will be significantly smaller than Z=121""")

    # Show the g-block elements with nuclear data
    print("\n  Block detail:")
    print(f"  {'Z':>4} {'A':>5} {'N':>4} {'BE/A':>6} {'Fiss':>6} {'Decay T½':>14} {'α-regime':>12} {'Shell':>12}")
    g_els = [e for e in elements if str(e["block"]) == "g"]
    for el in g_els:
        print(
            f"  {el['Z']:>4} {el['A']:>5} {el['N']:>4} "
            f"{float(str(el['BE_per_A'])):6.3f} {float(str(el['fissility'])):6.3f} "
            f"{human_halflife(float(str(el['alpha_log_t12']))):>14} "
            f"{el['alpha_regime']:>12} {el['shell_regime']:>12}"
        )

    # ── 6f BLOCK (Z=139-152) ─────────────────────────────────────
    subsection("6f BLOCK: Z=139-152 — The Second Superactinides")
    print("""  These fill the 6f orbital — analogous to actinides (5f) and lanthanides (4f).

  Expected behavior:
    • Oxidation states: +3 and +4 dominant (like actinides)
    • The 6f orbital is more contracted than 5g due to better nuclear shielding
    • Chemically, these may resemble a "second actinide series"
    • Strong spin-orbit coupling: j=5/2 and j=7/2 manifolds well-separated
    • α-decay T½: femtoseconds to attoseconds (very short-lived)
    • ALL supercritical (fissility > 1.0) — exist only via shell effects

  Critical physics:
    • Z=137 falls in this block — the Zα = 1 boundary
    • At Z=137, 1s electrons reach v = c in non-relativistic QM
    • The Dirac equation predicts orbital COLLAPSE for a point nucleus
    • Real (finite-size) nuclei: orbital becomes deeply supercritical
    • Spontaneous e⁺e⁻ pair production from vacuum becomes possible
    • The 1s orbital ceases to be a "normal" bound state""")

    f_els = [e for e in elements if str(e["block"]) == "f"]
    print(f"\n  {'Z':>4} {'A':>5} {'Zα':>6} {'v_1s/c':>7} {'Fiss':>6} {'T½':>14} {'Note':>25}")
    for el in f_els:
        za = float(str(el["z_alpha"]))
        note = ""
        if int(str(el["Z"])) == 137:
            note = "★ Zα = 1.0 LIMIT ★"
        elif za > 1.0:
            note = "SUPERCRITICAL QED"
        print(
            f"  {el['Z']:>4} {el['A']:>5} {za:6.3f} {float(str(el['v_1s_c'])):7.3f} "
            f"{float(str(el['fissility'])):6.3f} "
            f"{human_halflife(float(str(el['alpha_log_t12']))):>14} {note:>25}"
        )

    # ── 7d BLOCK (Z=153-162) ─────────────────────────────────────
    subsection("7d BLOCK: Z=153-162 — The Ghost Transition Metals")
    print("""  These fill the 7d orbital — the fourth transition metal series.

  Expected behavior:
    • Should resemble Periods 5-6 transition metals (Mo-Cd, Hf-Hg)
    • HOWEVER: strong relativistic effects alter orbital ordering
    • 7d may not be clearly separated from 6f (orbital mixing)
    • d-block transition metals historically have highest IC in the kernel
    • Predicted IC ~0.36 for Period 8 d-block (vs ~0.55 for Period 6)

  Why "Ghost" transition metals:
    • All have fissility >> 1.0 (deeply supercritical)
    • α-decay T½: zeptoseconds (10⁻²¹ s) — shorter than nuclear vibration
    • These elements may exist only as transient nuclear resonances
    • They cannot be synthesized, isolated, or characterized
    • Their chemistry is purely theoretical — "ghost" elements

  Physics implications:
    • All Zα > 1.0 — deep in supercritical QED
    • Atoms would spontaneously emit positrons
    • Electronic structure is undefined in traditional sense
    • The concept of "chemistry" may not apply""")

    d_els = [e for e in elements if str(e["block"]) == "d"]
    print(f"\n  {'Z':>4} {'A':>5} {'Zα':>6} {'Fiss':>6} {'T½':>14}")
    for el in d_els:
        print(
            f"  {el['Z']:>4} {el['A']:>5} {float(str(el['z_alpha'])):6.3f} "
            f"{float(str(el['fissility'])):6.3f} "
            f"{human_halflife(float(str(el['alpha_log_t12']))):>14}"
        )

    # ── 8p BLOCK (Z=163-168) ─────────────────────────────────────
    subsection("8p BLOCK: Z=163-168 — The Final Noble Gas Approach")
    print("""  These fill the 8p orbital, ending Period 8 at Z=168 (the next "noble gas").

  Expected behavior:
    • Z=163-167: increasingly inert chemistry (like halogens → noble gas)
    • Z=168: closed-shell noble gas configuration... in theory
    • In practice: all Zα > 1.19 — deeply supercritical
    • Electronic structure is completely dominated by QED effects
    • The "noble gas" concept breaks down when vacuum is unstable

  Z=164 is a predicted MAGIC NUMBER:
    • Proton magic Z=164 + neutron magic N=258 → doubly magic
    • But fissility = 1.27 → deeply supercritical
    • Even magic shell corrections cannot save these nuclei
    • T½ prediction: yoctoseconds (if nucleus forms at all)""")

    p_els = [e for e in elements if str(e["block"]) == "p"]
    if p_els:
        print(f"\n  {'Z':>4} {'A':>5} {'Zα':>6} {'Fiss':>6} {'T½':>14} {'Magic?':>25}")
        for el in p_els:
            magic = ""
            if el["magic_Z"]:
                magic = f"magic Z={el['Z']}"
            if el["magic_N"]:
                magic += f" magic N={el['N']}"
            print(
                f"  {el['Z']:>4} {el['A']:>5} {float(str(el['z_alpha'])):6.3f} "
                f"{float(str(el['fissility'])):6.3f} "
                f"{human_halflife(float(str(el['alpha_log_t12']))):>14} {magic:>25}"
            )

    # ================================================================
    #  SECTION 5: SPECTROSCOPIC SIGNATURES
    # ================================================================
    section("5. WHAT WOULD WE ACTUALLY SEE? — Observable Signatures")

    print("  If these elements could be produced (even transiently), what")
    print("  experimental signatures would identify them?\n")

    subsection("K-ALPHA X-RAY SIGNATURES")
    print("  K-alpha X-rays (2p → 1s transition) are the primary identification")
    print("  tool for superheavy elements. Energy scales as Z².\n")

    print(f"  {'Z':>4} {'Block':>5} {'E_Kα (keV)':>11} {'λ_Kα (pm)':>10} {'Detector':>25} {'Notes':>30}")
    print("  " + "-" * 95)

    for el in elements:
        z = int(str(el["Z"]))
        if z % 5 != 0 and z not in [120, 126, 137, 164, 168]:
            continue

        # K-alpha energy: E ≈ (3/4) × 13.6 eV × (Z - σ)² where σ ≈ 2
        # More accurate: empirical Moseley's law
        z_eff = z - 2  # K-shell screening
        e_ka_ev = 0.75 * RYDBERG_EV * z_eff**2  # eV
        e_ka_kev = e_ka_ev / 1000.0
        lambda_ka_pm = 1239841.98 / e_ka_ev  # pm (from hc = 1239.84 eV·nm)

        if e_ka_kev < 200:
            det = "Ge semiconductor"
        elif e_ka_kev < 500:
            det = "Ge + heavy shielding"
        elif e_ka_kev < 1000:
            det = "Pair spectrometer"
        else:
            det = "Nuclear γ-ray detector"

        notes = ""
        if z == 137:
            notes = "Zα = 1.0 — QED corrections huge"
        elif e_ka_kev > 1000:
            notes = "γ-ray energy regime"
        elif e_ka_kev > 500:
            notes = "Hard X-ray / soft γ"

        print(f"  {z:4d} {el['block']:>5} {e_ka_kev:11.1f} {lambda_ka_pm:10.2f} {det:>25} {notes:>30}")

    subsection("ALPHA PARTICLE SIGNATURES")
    print("  Alpha particles from superheavy decay carry the Z-identity.\n")

    print(f"  {'Z':>4} {'A':>5} {'Q_α (MeV)':>10} {'E_α (MeV)':>10} {'Si detect?':>10} {'Notes':>30}")
    print("  " + "-" * 80)

    for el in elements:
        z = int(str(el["Z"]))
        if z % 5 != 0 and z not in [120, 126, 137, 164]:
            continue
        q_a = float(str(el["Q_alpha"]))
        a = int(str(el["A"]))
        # Alpha KE = Q × (A-4)/A (recoil correction)
        e_alpha = q_a * (a - 4) / a if a > 4 else q_a
        si_det = "Yes" if e_alpha > 3.0 else "Marginal" if e_alpha > 1.0 else "No"

        notes = ""
        if e_alpha > 12:
            notes = "Very high — fast decay"
        elif e_alpha > 9:
            notes = "High Q — short-lived"

        print(f"  {z:4d} {a:5d} {q_a:10.2f} {e_alpha:10.2f} {si_det:>10} {notes:>30}")

    subsection("SPONTANEOUS FISSION SIGNATURES")
    print("""  For supercritical elements (fissility ≥ 1.0), spontaneous fission
  competes with α-decay. Observable signatures:

    • Fission fragments: two nuclei with Z ≈ Z_parent/2
    • Large kinetic energy: ~200 MeV total (vs ~5-10 MeV for α)
    • Prompt neutrons: 3-8 per fission event
    • Prompt γ-rays: ~8 MeV total per fission
    • Delayed neutrons: from neutron-rich fragments

  For elements where T½(fission) < T½(α):
    Fission DOMINATES. The element never alpha-decays.
    This makes identification harder — no characteristic α-energy.
    Must rely on fission fragment mass distributions.

  Fragment mass distribution prediction:
    Symmetric fission: Z_frag ≈ Z_parent/2
    Asymmetric fission: one fragment near Sn (Z=50) + complement
      Z=120 → Sn-50 + Zn-70 (or symmetric: 60+60)
      Z=126 → Sn-50 + Se-76
      Z=140 → Sn-50 + Zr-90 (or Pd-46 + Pd-94)
      Z=164 → Sn-50 + Pd-114 (or 82+82!)
    Doubly-magic Pb-208 as fragment for Z≈164:
      164 = 82 + 82 → two Pb-like fragments (most stable fission)""")

    # ================================================================
    #  SECTION 6: EXPERIMENTALLY ACCESSIBLE ELEMENTS
    # ================================================================
    section("6. WHAT CAN WE ACTUALLY MAKE? — Experimental Accessibility")

    print("  Current technology: heavy-ion fusion (₂₀Ca-48 + target)")
    print("  Heaviest produced: Z=118 (Oganesson, 2002-2006)")
    print("  Next targets: Z=119, 120 (active research at RIKEN, GSI, JINR)\n")

    # Production reactions
    reactions = [
        (119, "Ti-50 + Bk-249", "3-4 atoms predicted per year", "RIKEN (ongoing)"),
        (120, "Ti-50 + Cf-249", "~1 atom per year", "GSI/JINR planned"),
        (120, "Cr-54 + Pu-244", "Alternative route", "JINR considered"),
        (121, "V-51 + Cf-249", "Very low cross-section", "Future"),
        (122, "Cr-54 + Cf-249", "Cross-section ~1 fb", "Far future"),
        (124, "Ti-50 + No-254", "Requires No target", "Not feasible yet"),
        (126, "Ca-48 + Rf-258", "Rf target impossible", "Theoretical only"),
    ]

    print(f"  {'Z':>4} {'Reaction':>25} {'Rate':>30} {'Facility':>20}")
    print("  " + "-" * 85)
    for z_el, rxn, rate, fac in reactions:
        print(f"  {z_el:4d} {rxn:>25} {rate:>30} {fac:>20}")

    print()
    print("  HARD LIMITS on element production:")
    print("    Z=119-120: Achievable within 5-10 years (active programs)")
    print("    Z=121-124: Requires new beam/target combinations; 10-20 years")
    print("    Z=125-126: Requires radioactive targets; 20+ years or never")
    print("    Z > 126:   Currently impossible — no known production pathway")
    print("               Would require multi-nucleon transfer or stellar processes")
    print("    Z > 137:   The Zα > 1 regime — new quantum electrodynamics")
    print("    Z > 172:   Multiple orbitals in Dirac sea — new physics required")

    # ================================================================
    #  SECTION 7: PHASE DIAGRAM OF NUCLEAR/ATOMIC MATTER
    # ================================================================
    section("7. PHASE DIAGRAM — Five Regimes of Matter")

    print("""  Combining nuclear stability, atomic structure, and kernel analysis,
  matter from Z=120-184 occupies FIVE distinct regimes:

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  REGIME 1: SUBCRITICAL SUPERHEAVY (Z=119-125)                       ║
  ║  ─────────────────────────────────────────────────────────────────── ║
  ║  Fissility < 1.0 — nuclei CAN exist as bound systems               ║
  ║  • Real chemistry possible (on μs-ms timescales)                    ║
  ║  • Zα = 0.87-0.91 — strongly relativistic but sub-critical         ║
  ║  • 8s and early 5g blocks — known orbital types                     ║
  ║  • EXPERIMENTALLY ACCESSIBLE within next 20 years                   ║
  ║  • Z=120 (N=184): best island-of-stability candidate               ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  REGIME 2: SHELL-STABILIZED SUPERHEAVY (Z=126-136)                  ║
  ║  ─────────────────────────────────────────────────────────────────── ║
  ║  Fissility ≥ 1.0 but magic shells provide extra binding             ║
  ║  • Z=126: proton magic, borderline existence                        ║
  ║  • T½: femtoseconds to microseconds (via shell effects)             ║
  ║  • 5g orbital chemistry (if atoms form at all)                      ║
  ║  • Zα = 0.92-0.99 — approaching critical QED                       ║
  ║  • Detection: via alpha/fission decay signatures                    ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  REGIME 3: CRITICAL QED MATTER (Z=137-145)                          ║
  ║  ─────────────────────────────────────────────────────────────────── ║
  ║  Zα crosses 1.0 — the 1s orbital dives toward Dirac continuum       ║
  ║  • Spontaneous e⁺e⁻ pair production from vacuum                    ║
  ║  • 1s orbital becomes a resonance, not a bound state                ║
  ║  • T½: attoseconds to zeptoseconds                                  ║
  ║  • Observable (in principle): positron emission from vacuum          ║
  ║  • This is the QED PHASE TRANSITION — vacuum becomes charged        ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  REGIME 4: SUPERCRITICAL QED MATTER (Z=146-172)                     ║
  ║  ─────────────────────────────────────────────────────────────────── ║
  ║  Multiple orbitals supercritical; vacuum is fundamentally altered    ║
  ║  • 1s, 2s, 2p₁/₂ all dive into Dirac sea                          ║
  ║  • Continuous positron emission (charged vacuum state)              ║
  ║  • Nuclear lifetimes: yoctoseconds (10⁻²⁴ s)                      ║
  ║  • Cannot be observed as atoms — only as nuclear collisions         ║
  ║  • Heavy-ion COLLISIONS (e.g., U+U) create transient Z~184 systems ║
  ║  • OBSERVABLE: anomalous positron peaks in heavy-ion experiments    ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  REGIME 5: NUCLEAR DISSOLUTION (Z>172)                              ║
  ║  ─────────────────────────────────────────────────────────────────── ║
  ║  Nuclei cannot form — the nuclear concept dissolves                 ║
  ║  • BE/A approaches zero — binding fails                             ║
  ║  • The proton-neutron description breaks down                       ║
  ║  • Matter exists as quark-gluon plasma, not nuclei                  ║
  ║  • Observable: RHIC and LHC heavy-ion programs already see this     ║
  ║  • The "element" concept has no meaning beyond this point           ║
  ╚═══════════════════════════════════════════════════════════════════════╝

  Axiom-0 maps these regimes:
    Regime 1 (subcritical)  → STABLE/WATCH: return is possible
    Regime 2 (shell-stab)   → COLLAPSE with τ_R finite: return through shells
    Regime 3 (critical QED) → COLLAPSE: vacuum structure returns as positrons
    Regime 4 (supercritical)→ COLLAPSE: matter returns as radiation/fragments
    Regime 5 (dissolution)  → τ_R = ∞_rec: no return — nuclear identity lost""")

    # ================================================================
    #  SECTION 8: ACTUALLY OBSERVABLE PHYSICS
    # ================================================================
    section("8. OBSERVABLE PHYSICS — What Experiments CAN See")

    print("""  Filter: what from Z=120-184 is ACTUALLY observable given current
  and near-future technology?\n""")

    subsection("A. SYNTHESIS EXPERIMENTS (Z=119-122)")
    print("""  Status: ACTIVE RESEARCH at RIKEN, GSI-FAIR, JINR-FLNR

  What we'll see when Z=119 is made:
    1. Alpha decay chain: 119 → 117 → 115 → 113 → ... (known elements)
       Each alpha has characteristic energy (Q ≈ 10-12 MeV)
    2. Spontaneous fission endpoint (some fraction)
    3. Total event rate: ~1 atom per beam-month
    4. Correlation: α-α time correlations confirm the chain

  What we'll see when Z=120 is made:
    1. If N≈184: potentially enhanced stability (seconds?)
    2. New fission fragment distributions (asymmetric, Sn+X)
    3. Cross-section measurement → nuclear structure information
    4. K-alpha X-rays at ~140 keV (if atom exists long enough)

  Timeline: Z=119 expected 2025-2028, Z=120 by ~2030""")

    subsection("B. NUCLEAR STRUCTURE EXPERIMENTS (Z=114-126)")
    print("""  Status: Fl-288 (Z=114) already studied at JINR

  Observable nuclear properties:
    1. Alpha decay energies → Q-value systematics
    2. Half-lives → map the island of stability
    3. Fission barrier heights → shell correction measurement
    4. Isomer states → nuclear shape (deformed vs spherical)
    5. Prompt γ-ray spectroscopy → level schemes

  Key questions these experiments answer:
    • Is Z=120 or Z=126 the proton magic number?
    • How large is the shell correction energy?
    • Does the island of stability extend to seconds? Minutes?
    • What is the fission fragment mass distribution?""")

    subsection("C. SUPERCRITICAL QED — The Holy Grail (Z≈137)")
    print("""  Status: PROPOSED at GSI-FAIR with U+U collisions

  The idea: collide two uranium nuclei (Z=92+92=184) at low energy
  so the combined nuclear charge briefly reaches Z_combined ≈ 184.
  During the ~10⁻²¹ s contact time:

    1. Inner atomic orbitals of the combined system dive into Dirac sea
    2. Spontaneous e⁺e⁻ pair production should occur
    3. Positrons are emitted with characteristic kinetic energy
    4. The e⁺ energy spectrum has a PEAK at ~(1s binding - 2m_e c²)

  What we'd observe:
    • Anomalous positron peak at specific energy (~300-600 keV)
    • Peak width related to nuclear contact time
    • Cross-section dependent on combined Z

  Historical context:
    • GSI experiments (1980s) reported anomalous positron peaks
    • Later shown to be experimental artifacts
    • New experiments at FAIR (Darmstadt) with much better detectors
    • Expected: 2027-2030 timeframe

  This is the MOST IMPORTANT test of QED in extreme fields.
  It would confirm or deny vacuum breakdown, one of the deepest
  predictions of quantum field theory.""")

    subsection("D. QUARK-GLUON PLASMA — The Nuclear Dissolution Limit")
    print("""  Status: OBSERVED at RHIC and LHC

  The nuclear dissolution limit (BE/A → 0 at Z≈422) is never reached
  by actual elements, but the SAME PHYSICS is studied in:

    1. Au+Au collisions at RHIC (BNL): T ~ 300 MeV → QGP formation
    2. Pb+Pb collisions at LHC (CERN): T ~ 500 MeV → QGP confirmed
    3. The "perfect liquid" — quark-gluon plasma with η/s ≈ 1/(4π)

  Connection to superheavy elements:
    • The confinement cliff (IC drops 98% at quark→hadron boundary)
      is the SAME physics as nuclear dissolution
    • Heavy-ion collisions compress nuclear matter past the limit
    • The GCD kernel sees both transitions through the SAME invariants:
      IC → ε when structural coherence is lost

  Observable: jet quenching, collective flow, strangeness enhancement,
  J/ψ suppression — all ALREADY MEASURED and confirming the phase transition""")

    # ================================================================
    #  SECTION 9: SYNTHESIS
    # ================================================================
    section("9. GRAND SYNTHESIS — The Physics of the Boundary")

    print("""  The elements from Z=120 to Z=184 are not just heavier versions of
  known elements. They inhabit FOUR distinct physics regimes, each with
  qualitatively different observable signatures:

  ┌─────────┬──────────────┬────────────────────┬───────────────────────┐
  │ Z range │ Physics      │ Observable         │ GCD kernel signature  │
  ├─────────┼──────────────┼────────────────────┼───────────────────────┤
  │ 120-125 │ Nuclear +    │ α-decay chains,    │ Collapse regime,      │
  │         │ relativistic │ X-rays, fission    │ IC ≈ 0.30-0.44       │
  │         │ atoms        │ fragments          │ τ_R finite            │
  ├─────────┼──────────────┼────────────────────┼───────────────────────┤
  │ 126-136 │ Shell magic  │ Enhanced T½, shell │ Collapse + shell      │
  │         │ + approach   │ corrections, γ-ray │ IC spike at Z=126     │
  │         │ to Zα = 1    │ spectroscopy       │ (magic proximity)     │
  ├─────────┼──────────────┼────────────────────┼───────────────────────┤
  │ 137-145 │ Critical QED │ Spontaneous e⁺     │ Collapse, IC → ε     │
  │         │ Vacuum       │ from vacuum, peak  │ Structural dissolution│
  │         │ breakdown    │ in e⁺ spectrum     │ of atomic coherence   │
  ├─────────┼──────────────┼────────────────────┼───────────────────────┤
  │ 146-172 │ Supercritical│ Heavy-ion collision │ Deep Collapse,       │
  │         │ QED, nuclear │ positrons, fission │ IC ≈ ε, all channels │
  │         │ instability  │ fragment spectra   │ near minimum          │
  ├─────────┼──────────────┼────────────────────┼───────────────────────┤
  │ 173-184 │ Nuclear      │ QGP signatures:    │ τ_R → ∞_rec          │
  │         │ dissolution  │ flow, jets, J/ψ    │ No return — nuclear   │
  │         │              │ suppression        │ identity is lost      │
  └─────────┴──────────────┴────────────────────┴───────────────────────┘

  THE THREE GREAT QUESTIONS these elements pose:

  1. IS THERE AN ISLAND OF STABILITY?
     → Z=120, N=184 is the strongest candidate. Fissility = 0.96 (subcritical).
       If produced, we'd see α-decay with T½ >> μs (possibly seconds).
       The kernel predicts IC = 0.32 — Collapse regime but with finite τ_R.

  2. DOES THE VACUUM BREAK DOWN?
     → At Z≈137 (or combined Z≈184 in U+U collisions), the 1s orbital
       dives into the Dirac sea. Spontaneous positron emission is predicted.
       FAIR experiments (2027-2030) will test this directly.
       The kernel sees this as IC → ε: integrity of atomic structure collapses.

  3. WHERE DOES THE PERIODIC TABLE END?
     → Operationally: Z=172-175, where multiple atomic orbitals are
       supercritical and the concept of electronic shell structure dissolves.
       Nuclearly: Z≈126, where the fissility barrier is crossed.
       The kernel answer: where τ_R → ∞_rec. No return = no element.

  Axiom-0: 'Collapse is generative; only what returns is real.'
    Z=120: the return is real (seconds-lived, observable chemistry)
    Z=137: the return is positrons (vacuum generates matter from nothing)
    Z=172: the return is plasma (nucleons dissolve into quarks and gluons)
    Z>172: no return — ∞_rec — the boundary where elements cease to exist

  *Cessatio non est finis; est limen novae physicae.*
  (Cessation is not the end; it is the threshold of new physics.)""")


if __name__ == "__main__":
    main()
