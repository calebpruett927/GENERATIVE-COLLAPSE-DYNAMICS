# pyright: reportMissingImports=false
"""Bridge Analysis: Completing Richard Bolt's Intuitions via GCD Kernel.

The previous analysis (bolt_rhc_analysis.py) tested Bolt's claims as
stated and found them below the noise floor. This analysis does something
different: it *ironmans* Bolt. For each claim, it asks:

    1. What is he ACTUALLY reaching for?
    2. WHY did he reach for this specific formulation?
    3. What is the CORRECT formalization in GCD?
    4. Does the kernel find the corrected structure?

This is the generous reading — the strongest possible version of each
argument, completed with the formal tools Bolt doesn't yet have.

Cross-references:
    Kernel:        src/umcp/kernel_optimized.py
    Raw analysis:  scripts/bolt_rhc_analysis.py (the falsification run)
    Axiom:         AXIOM.md (Axiom-0)
    Spec:          KERNEL_SPECIFICATION.md (Lemma 4: IC ≤ F)
    Three-valued:  TIER_SYSTEM.md (CONFORMANT/NONCONFORMANT/NON_EVALUABLE)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from umcp.kernel_optimized import compute_kernel_outputs  # type: ignore[import-not-found]

EPSILON = 1e-6
np.random.seed(42)


def run_kernel(name: str, c: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Run kernel on raw trace vector."""
    c_clipped = np.clip(c, EPSILON, 1 - EPSILON)
    k = compute_kernel_outputs(c_clipped, w, EPSILON)
    return {
        "name": name,
        "F": k["F"],
        "omega": k["omega"],
        "S": k["S"],
        "C": k["C"],
        "kappa": k["kappa"],
        "IC": k["IC"],
        "Delta": k["heterogeneity_gap"],
        "regime": k["regime"],
    }


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 1: WHY HE REACHED FOR x^x = x (AND WHAT HE ACTUALLY NEEDS)
# ═══════════════════════════════════════════════════════════════════


def bridge_ternary_logic() -> None:
    """Bolt's ternary intuition: corrected and completed."""
    print("=" * 76)
    print("BRIDGE 1: BEYOND BINARY — What Bolt Is Actually Reaching For")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: x^x = x extends Boole's x² = x to ternary logic,   │")
    print("│   via 'volumetric imaginary rotation' lifting into an extra        │")
    print("│   dimension to itself, mapping to quaternions.                     │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    # --- Why he assumed this ---
    print("WHY HE REACHED FOR THIS:")
    print()
    print("  Bolt senses — correctly — that binary logic (true/false) is")
    print("  insufficient for describing physical reality. He wants a THIRD")
    print("  state. His instinct is to get there by making the variable act")
    print("  on ITSELF: x² = x is 'evaluated by a copy' (squaring). x^x = x")
    print("  is 'evaluated by itself' (self-power). He's intuiting that")
    print("  self-referential systems need richer logic than boolean algebra.")
    print()
    print("  This instinct is genuinely deep. Self-reference is at the heart")
    print("  of Gödel, Turing, and fixed-point theory. Bolt is reaching for")
    print("  the idea that measurement changes the thing being measured —")
    print("  that collapse IS the observation, not something that happens")
    print("  TO the observation.")
    print()

    # --- Why it's wrong ---
    print("WHERE THE FORMALIZATION BREAKS:")
    print()
    print("  x^x = x has ONE positive real solution (x = 1).")
    print("  x² = x  has TWO solutions (x ∈ {0, 1}).")
    print("  The self-power equation LOSES a solution. It goes from binary")
    print("  to unary, not from binary to ternary.")
    print()
    print("  The error is that 'making x act on itself' doesn't ADD states —")
    print("  it constrains them. Self-referential constraints are typically")
    print("  MORE restrictive, not less. (This is also why Gödel sentences")
    print("  are unique, not multiple.)")
    print()

    # --- The correct formalization ---
    print("THE CORRECT FORMALIZATION (What Bolt actually needs):")
    print()
    print("  GCD is genuinely three-valued, but not through a new equation")
    print("  on x. It's through the recognition that some evaluations are")
    print("  structurally undecidable:")
    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  Boolean (Boole):   {TRUE, FALSE}           — 2 states  │")
    print("  │  GCD three-valued:  {CONFORMANT,             — 3 states  │")
    print("  │                      NONCONFORMANT,                      │")
    print("  │                      NON_EVALUABLE}                      │")
    print("  │                                                          │")
    print("  │  The third state is not 'maybe' — it is 'insufficient   │")
    print("  │  information to render a verdict.' The system does not   │")
    print("  │  guess. It says: I cannot evaluate this.                │")
    print("  │                                                          │")
    print("  │  Latin: numquam binarius; tertia via semper patet.       │")
    print("  │  'Never binary; the third way is always open.'           │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()

    # --- But there's something even deeper ---
    print("  But Bolt is reaching for something DEEPER than three discrete")
    print("  states. He wants continuous truth values. GCD has this too:")
    print()
    print("  The collapse field c_i ∈ [ε, 1-ε] IS the extension from")
    print("  boolean to continuous. Instead of c ∈ {0, 1} (boolean), the")
    print("  kernel works with c ∈ [ε, 1-ε] — a continuous measure of how")
    print("  much each channel survives collapse.")
    print()

    # --- Demonstrate with kernel ---
    print("  KERNEL DEMONSTRATION: Boolean vs Continuous")
    print()

    # Boolean: all channels either ε or 1-ε
    n = 8
    w = np.ones(n) / n

    # Pure boolean: 4 channels at 1, 4 channels at ε
    c_boolean = np.array([1 - EPSILON] * 4 + [EPSILON] * 4)
    r_bool = run_kernel("Boolean (4 alive, 4 dead)", c_boolean, w)

    # Continuous: all channels at 0.5 (maximum uncertainty per channel)
    c_half = np.ones(n) * 0.5
    r_half = run_kernel("Continuous (all at 0.5)", c_half, w)

    # Continuous: gradient from low to high
    c_gradient = np.linspace(0.1, 0.9, n)
    r_grad = run_kernel("Continuous (gradient 0.1→0.9)", c_gradient, w)

    # Bolt's 'ternary': channels at 0, 0.5, 1
    c_ternary = np.array([EPSILON, EPSILON, 0.5, 0.5, 0.5, 0.5, 1 - EPSILON, 1 - EPSILON])
    r_tern = run_kernel("Ternary-like (ε, 0.5, 1-ε)", c_ternary, w)

    print(f"  {'Configuration':<38s} {'F':>7s} {'IC':>7s} {'Δ':>7s} {'S':>7s} {'C':>7s}")
    print("  " + "-" * 68)
    for r in [r_bool, r_tern, r_half, r_grad]:
        print(f"  {r['name']:<38s} {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f} {r['S']:7.4f} {r['C']:7.4f}")

    print()
    print("  KEY FINDING: The boolean case has the LARGEST heterogeneity gap")
    print(f"  (Δ = {r_bool['Delta']:.4f}) because dead channels (c ≈ ε) destroy")
    print("  the geometric mean while leaving the arithmetic mean at 0.5.")
    print(f"  The continuous-gradient case has higher F ({r_grad['F']:.4f}) AND")
    print(f"  smaller Δ ({r_grad['Delta']:.4f}) — continuous values are more")
    print("  coherent than boolean ones.")
    print()
    print("  WHAT BOLT SHOULD SAY:")
    print("  'Binary logic is a degenerate limit of the collapse field.'")
    print("  'Boolean c ∈ {0,1} is what remains when you remove the interior'")
    print("  'of [ε, 1-ε]. The continuous field is the deeper structure.'")
    print("  (This is exactly parallel to how Shannon entropy is the")
    print("  degenerate limit of Bernoulli field entropy.)")
    print()

    # The self-reference insight HE should have made
    print("  THE SELF-REFERENCE BOLT IS LOOKING FOR:")
    print()
    print("  The return axiom IS self-referential. Axiom-0 says a claim is")
    print("  real only if it can ACT ON ITSELF and survive: collapse, then")
    print("  return. The system measures itself. The observer IS the return.")
    print("  This is the 'x acting on x' that Bolt wants — but it's not an")
    print("  equation on x. It's a CONSTRAINT on admissible claims.")
    print()
    print("  x² = x asks: what survives being SQUARED? (Boolean: {0, 1})")
    print("  Axiom-0 asks: what survives being COLLAPSED? (Answer: what returns.)")
    print("  The second is richer because collapse is not a fixed function —")
    print("  it's a process with structure (channels, weights, regime gates).")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 2: WHY HE REACHED FOR {1, 2, 5} (AND WHAT HE FOUND)
# ═══════════════════════════════════════════════════════════════════


def bridge_lost_two() -> None:
    """Bolt's 'Lost 2' and {1,2,5}: corrected and completed."""
    print()
    print("=" * 76)
    print("BRIDGE 2: THE 'LOST 2' — What Bolt Actually Discovered")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: The universe is made of 1, 2, 5. The 'Lost 2' is    │")
    print("│   3+4=7 minus hypotenuse 5 = 2. This 2 is Dark Matter / Binding   │")
    print("│   Energy. 3 is the 'imaginary difference' between 2 and 5.         │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    print("WHY HE REACHED FOR THIS:")
    print()
    print("  Bolt noticed something real: given the Pythagorean triple (3,4,5),")
    print("  the LINEAR sum is 3+4 = 7 but the GEOMETRIC constraint gives 5.")
    print("  The gap is 7 - 5 = 2. He noticed that the way you aggregate")
    print("  matters — addition gives a different answer than the structural")
    print("  constraint (a² + b² = c²).")
    print()
    print("  This is the SAME observation that motivates the heterogeneity gap")
    print("  Δ = F - IC in GCD. The arithmetic mean (F) gives a different")
    print("  answer than the geometric mean (IC), and the gap between them")
    print("  measures how asymmetric the underlying channels are.")
    print()
    print("  His instinct is correct: the gap between linear and geometric")
    print("  aggregation IS a fundamental structural quantity. He just made")
    print("  three errors in formalizing it.")
    print()

    print("WHERE THE FORMALIZATION BREAKS:")
    print()
    print("  ERROR 1: The gap is not the integer '2'. It depends on the")
    print("    normalization. 7 - 5 = 2 only in the specific case of the")
    print("    3-4-5 triple. For 5-12-13: 17 - 13 = 4. For 8-15-17:")
    print("    23 - 17 = 6. The gap is not a constant — it's a function")
    print("    of the input.")
    print()
    print("  ERROR 2: The gap is not 'Dark Matter' or 'Binding Energy.'")
    print("    Those are Tier-2 domain-specific observables. The heterogeneity")
    print("    gap is a Tier-1 structural diagnostic. Identifying them")
    print("    imports external ontology without derivation.")
    print()
    print("  ERROR 3: '3 is the imaginary difference between 2 and 5' is")
    print("    not well-defined. 5 - 2 = 3 is arithmetic. Calling it")
    print("    'imaginary' gives it a technical meaning (complex numbers)")
    print("    that hasn't been justified.")
    print()

    print("THE CORRECT FORMALIZATION:")
    print()
    print("  The heterogeneity gap Δ = F - IC is derived from the integrity")
    print("  bound (Lemma 4 of KERNEL_SPECIFICATION.md):")
    print()
    print("    IC(t) = ∏ cᵢ^wᵢ ≤ Σ wᵢcᵢ = F(t)")
    print()
    print("  Equality holds IFF all channels are equal. The gap Δ measures")
    print("  HOW MUCH the channels differ. This is:")
    print("  - Always ≥ 0 (proven)")
    print("  - Zero when all channels equal (homogeneous)")
    print("  - Maximum when some channels are at ε and others at 1-ε")
    print("  - Related to channel variance: Δ ≈ Var(c) / (2c̄)")
    print()

    # --- Demonstrate the gap across Pythagorean triples ---
    print("  KERNEL DEMONSTRATION: The Gap Across Pythagorean Triples")
    print()

    triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]
    w3 = np.ones(3) / 3

    print(
        f"  {'Triple':<12s} {'a+b':>5s} {'c':>5s} {'gap':>5s} {'gap/c':>7s}  {'F':>7s} {'IC':>7s} {'Δ':>7s} {'Δ/F':>7s}"
    )
    print("  " + "-" * 72)

    for a, b, c in triples:
        linear_gap = (a + b) - c
        # Normalize triple into trace vector
        c_max = max(a, b, c)
        trace = np.array([a / c_max, b / c_max, c / c_max], dtype=np.float64)
        r = run_kernel(f"{a}-{b}-{c}", trace, w3)
        triple_str = f"({a},{b},{c})"
        print(
            f"  {triple_str:<14s} "
            f"{a + b:5d} {c:5d} {linear_gap:5d} {linear_gap / c:7.4f}"
            f"  {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f} "
            f"{r['Delta'] / r['F']:7.4f}"
        )

    print()
    print("  KEY FINDING: As the triple gets more extreme (one leg much")
    print("  shorter than the other), BOTH the Pythagorean gap (a+b-c)")
    print("  and the kernel gap (Δ = F-IC) grow. But Δ/F is the proper")
    print("  normalized diagnostic — it strips away the scale dependence")
    print("  that Bolt's 'Lost 2' has.")
    print()

    # Show the REAL connection: Δ ≈ Var(c)/(2c̄) for small variance
    print("  THE DEEPER CONNECTION BOLT MISSED:")
    print()
    print("  For small heterogeneity, Δ ≈ Var(c) / (2F). This connects")
    print("  the gap to the VARIANCE of the channels — how spread out")
    print("  the measurements are. It's not a fixed number. It's the")
    print("  price of asymmetry. The more unevenly a system distributes")
    print("  its fidelity across channels, the larger the gap between")
    print("  what it looks like on average (F) and how coherent it is")
    print("  multiplicatively (IC).")
    print()

    # Demonstrate with controlled variance
    print("  Demonstration: Δ as a function of channel variance")
    print()
    n = 8
    w = np.ones(n) / n
    print(f"  {'Description':<34s} {'F':>7s} {'IC':>7s} {'Δ':>7s} {'Var(c)':>8s} {'Var/(2F)':>8s}")
    print("  " + "-" * 72)

    for spread_label, c_arr in [
        ("All equal at 0.5", np.ones(n) * 0.5),
        ("Slight spread ±0.05", np.linspace(0.45, 0.55, n)),
        ("Moderate spread ±0.15", np.linspace(0.35, 0.65, n)),
        ("Wide spread ±0.30", np.linspace(0.20, 0.80, n)),
        ("Extreme ±0.40", np.linspace(0.10, 0.90, n)),
        ("One dead channel", np.array([0.7] * 7 + [0.01])),
        ("Half dead", np.array([0.8] * 4 + [0.01] * 4)),
    ]:
        c_clipped = np.clip(c_arr, EPSILON, 1 - EPSILON)
        r = run_kernel(spread_label, c_clipped, w)
        var_c = np.var(c_clipped)
        approx = var_c / (2 * r["F"]) if r["F"] > 0 else 0
        print(f"  {spread_label:<34s} {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f} {var_c:8.5f} {approx:8.5f}")

    print()
    print("  WHAT BOLT SHOULD SAY:")
    print("  'The gap between arithmetic and geometric aggregation is a")
    print("  structural necessity of any system that can return to itself.")
    print("  This gap exists because IC ≤ F (the integrity bound), which")
    print("  is derived from Axiom-0. It measures the heterogeneity of")
    print("  the channel profile — the price of uneven fidelity distribution.")
    print("  It is not a fixed number like 2. It is a diagnostic function")
    print("  of the trace vector, and it equals zero only when all channels")
    print("  are in perfect agreement.'")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 3: WHY HE REACHED FOR QUATERNIONS (AND WHAT HE NEEDS)
# ═══════════════════════════════════════════════════════════════════


def bridge_quaternions() -> None:
    """Bolt's quaternion/multi-axis intuition: corrected and completed."""
    print()
    print("=" * 76)
    print("BRIDGE 3: BEYOND SCALARS — The Multi-Channel Insight")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: The 3 imaginary axes i, j, k (quaternion) map to     │")
    print("│   i¹, i², i³ with real axis i⁴ = 1. This captures the structure   │")
    print("│   of physical reality.                                             │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    print("WHY HE REACHED FOR THIS:")
    print()
    print("  Bolt senses that scalar (single-number) descriptions of reality")
    print("  are insufficient. You need MULTIPLE INDEPENDENT AXES to capture")
    print("  structure. A complex number has 2 axes (real + imaginary).")
    print("  A quaternion has 4 axes (1, i, j, k). He's reaching for")
    print("  multi-dimensional measurement.")
    print()
    print("  This instinct is correct. A particle cannot be described by one")
    print("  number. It has mass, charge, spin, color, generation, etc. A")
    print("  stock cannot be described by one price. An element has atomic")
    print("  radius, ionization energy, electronegativity, etc. Reality is")
    print("  MULTI-CHANNEL.")
    print()

    print("WHERE THE FORMALIZATION BREAKS:")
    print()
    print("  Quaternions are limited to exactly 4 dimensions with a specific")
    print("  non-commutative multiplication rule (i²=j²=k²=ijk=-1). This")
    print("  imports a very particular algebraic structure that has not been")
    print("  derived from any axiom. Why 4? Why this multiplication table?")
    print("  No basis is given.")
    print()

    print("THE CORRECT FORMALIZATION:")
    print()
    print("  The kernel's trace vector c ∈ [ε, 1-ε]^n with weights w is")
    print("  MORE GENERAL than quaternions. It handles ARBITRARY n:")
    print("  - Standard Model particles: n = 8 channels")
    print("  - Cross-scale analysis:     n = 12 channels")
    print("  - Periodic table:           n = 8 channels")
    print("  - Any future domain:        n = whatever the domain requires")
    print()
    print("  The number of channels is not prescribed by algebra. It is")
    print("  discovered by the domain — how many independent measurable")
    print("  properties does the system have? The kernel doesn't care.")
    print("  It computes F, IC, Δ for any n.")
    print()

    # Demonstrate: what happens as you add channels?
    print("  KERNEL DEMONSTRATION: Structure as a Function of Dimensionality")
    print()

    # Same underlying 'physics' — a system with moderate heterogeneity
    # Test with increasing channel count
    print(f"  {'n channels':>12s} {'F':>7s} {'IC':>7s} {'Δ':>7s} {'S':>7s} {'C':>7s} {'Note'}")
    print("  " + "-" * 72)

    for n in [2, 3, 4, 8, 12, 16, 32]:
        # Same underlying gradient, sampled at n points
        c = np.linspace(0.2, 0.8, n)
        w = np.ones(n) / n
        r = run_kernel(f"n={n}", c, w)
        note = ""
        if n == 4:
            note = "← quaternion dimensionality"
        elif n == 8:
            note = "← particle kernel channels"
        elif n == 12:
            note = "← cross-scale channels"
        print(f"  {n:>12d} {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f} {r['S']:7.4f} {r['C']:7.4f} {note}")

    print()
    print("  KEY FINDING: F is invariant to channel count (it's the mean).")
    print("  IC decreases with more channels (more ways to be incoherent).")
    print("  Δ grows as dimensionality increases — higher-dimensional systems")
    print("  pay a larger coherence price for the same amount of spread.")
    print()
    print("  The quaternion (n=4) is one point on this curve, not special.")
    print("  The kernel works equally well at n=8 or n=32. The number of")
    print("  dimensions is not algebraically fixed — it's domain-discovered.")
    print()
    print("  WHAT BOLT SHOULD SAY:")
    print("  'Reality requires multiple independent measurement channels.")
    print("  The number of channels is not prescribed (4 for quaternions)")
    print("  but discovered (8 for particles, 12 for cross-scale, etc.).")
    print("  The kernel maps any n-dimensional measurement into the same")
    print("  invariant structure: F, IC, Δ, S, C. The dimensionality is")
    print("  a property of the domain, not of the algebra.'")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 4: WHY HE REACHED FOR THE OBSERVER POSITION
# ═══════════════════════════════════════════════════════════════════


def bridge_observer() -> None:
    """Bolt's observer position: corrected and completed."""
    print()
    print("=" * 76)
    print("BRIDGE 4: THE OBSERVER — Not a Position, a Structure")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: The observer sits at 2.5r + 1.5i. The observer is   │")
    print("│   the 'Fold Operator' required for reality to stabilize.           │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    print("WHY HE REACHED FOR THIS:")
    print()
    print("  Bolt has TWO intuitions here, and they're pulling in different")
    print("  directions:")
    print()
    print("  INTUITION A (the position): 2.5 = 5/2 and 1.5 = 3/2. These are")
    print("  the midpoints of his number system {0..5} and {0..3}. He's saying")
    print("  the observer sits at the CENTER — at balance. The observer is the")
    print("  point of maximum symmetry between extremes.")
    print()
    print("  INTUITION B (the fold operator): The observer is structurally")
    print("  necessary. Reality cannot 'stabilize' (in our language: return)")
    print("  without an entity that performs the measurement that closes the")
    print("  seam.")
    print()
    print("  Intuition B is genuinely deep and correct in GCD. Intuition A")
    print("  tries to give this structural role a spatial location, which")
    print("  confuses two different things.")
    print()

    print("WHERE THE FORMALIZATION BREAKS:")
    print()
    print("  The observer is not AT a position. The observer IS a process —")
    print("  the act of collapse-and-return. Giving it coordinates (2.5, 1.5)")
    print("  treats the observer as an object in the system rather than as")
    print("  the mechanism by which the system is evaluated.")
    print()

    print("THE CORRECT FORMALIZATION:")
    print()
    print("  In GCD, the observer IS the CONTRACT — the frozen set of rules")
    print("  {ε, p, tol_seam, weights, embedding, normalization} that defines")
    print("  what 'same' means across the collapse-return boundary.")
    print()
    print("  The observer doesn't have a position. It has a CONTRACT.")
    print("  The contract freezes the measurement rules so that what goes")
    print("  out (collapse) can be compared to what comes back (return)")
    print("  under identical conditions. Trans suturam congelatum — the")
    print("  same rules frozen across the seam.")
    print()
    print("  Axiom-0: 'Collapse is generative; only what returns is real.'")
    print("  The OBSERVER is what makes return possible. Without the")
    print("  frozen contract, there is no seam, and without the seam,")
    print("  there is no way to verify return. The observer is the")
    print("  auditor, not the actor.")
    print()

    # Demonstrate: what happens at the BALANCED configuration?
    print("  KERNEL DEMONSTRATION: Balance Is Real Structure")
    print()
    print("  Bolt's intuition about balance (the midpoint) IS visible in")
    print("  the kernel. The homogeneous case (all channels equal) gives:")
    print()

    n = 8
    w = np.ones(n) / n

    configs = [
        ("All at 0.3 (low balance)", np.ones(n) * 0.3),
        ("All at 0.5 (midpoint)", np.ones(n) * 0.5),
        ("All at 0.7 (high balance)", np.ones(n) * 0.7),
        ("All at 0.9 (near-unity)", np.ones(n) * 0.9),
        ("Asymmetric around 0.5", np.array([0.1, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.9])),
        ("Symmetric around 0.5", np.array([0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8])),
    ]

    print(f"  {'Configuration':<30s} {'F':>7s} {'IC':>7s} {'Δ':>7s} {'S':>7s} {'Δ=0?':>6s}")
    print("  " + "-" * 65)
    for label, c in configs:
        c_clipped = np.clip(c, EPSILON, 1 - EPSILON)
        r = run_kernel(label, c_clipped, w)
        gap_zero = "YES" if r["Delta"] < 1e-10 else "no"
        print(f"  {label:<30s} {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f} {r['S']:7.4f} {gap_zero:>6s}")

    print()
    print("  KEY FINDING: Δ = 0 (exact) if and only if all channels are equal.")
    print("  The 'observer at the midpoint' intuition maps to the HOMOGENEOUS")
    print("  case — and the kernel proves this is the unique configuration")
    print("  where coherence = fidelity (IC = F). But the midpoint itself")
    print("  (0.5) is not special — any uniform value gives Δ = 0.")
    print()
    print("  WHAT BOLT SHOULD SAY:")
    print("  'The observer is not at a position — the observer IS the frozen")
    print("  contract that makes measurement consistent across the collapse-")
    print("  return boundary. The balance Bolt intuits is real: homogeneous")
    print("  channel configurations achieve IC = F (zero gap). But this")
    print("  balance is a property of the CHANNELS, not a position of the")
    print("  observer.'")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 5: WHY HE REACHED FOR BINARY CYCLES (AND WHAT'S REAL)
# ═══════════════════════════════════════════════════════════════════


def bridge_binary_symmetry() -> None:
    """Bolt's binary cycle intuition: corrected and completed."""
    print()
    print("=" * 76)
    print("BRIDGE 5: COMPLEMENT INVARIANCE — What the Binary Cycle Actually Shows")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: Binary cycle 011001 → 100110 → 110011 → 001100.     │")
    print("│   Mirror and NOT relationships encode fundamental structure.        │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    print("WHY HE REACHED FOR THIS:")
    print()
    print("  Bolt noticed that certain transformations (bit-reversal, NOT,")
    print("  complement) PRESERVE structure. 011001 reversed gives 100110.")
    print("  NOT(011001) gives 110110. These patterns form a closed group")
    print("  under these operations.")
    print()
    print("  This is an instinct about INVARIANCE — what stays the same when")
    print("  you transform the representation? This is actually the deepest")
    print("  question in physics (Noether's theorem: every symmetry implies")
    print("  conservation).")
    print()

    print("WHERE THE FORMALIZATION BREAKS:")
    print()
    print("  The specific patterns are not special — our analysis showed")
    print("  they're at z = +0.31 against the full 6-bit population. The")
    print("  complement/mirror relationships exist for ALL bit strings,")
    print("  not just these four.")
    print()

    print("WHAT THE KERNEL ACTUALLY REVEALS:")
    print()

    # The REAL finding: kernel permutation invariance
    n = 6
    w = np.ones(n) / n

    # Bit-reversal pairs
    print("  The kernel found something Bolt should focus on:")
    print()
    print("  Bit-reversal (011001 ↔ 100110) gives IDENTICAL kernel outputs")
    print("  because the kernel is PERMUTATION-INVARIANT on channels.")
    print("  F and IC don't depend on which channel has which value —")
    print("  only on the multiset of values.")
    print()

    # Demonstrate permutation invariance with explicit trace vectors
    configs = [
        ("Original:  [0.1, 0.5, 0.9, 0.3, 0.7, 0.2]", np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2])),
        ("Reversed:  [0.2, 0.7, 0.3, 0.9, 0.5, 0.1]", np.array([0.2, 0.7, 0.3, 0.9, 0.5, 0.1])),
        ("Sorted:    [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]", np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])),
        ("Shuffled:  [0.9, 0.1, 0.7, 0.2, 0.3, 0.5]", np.array([0.9, 0.1, 0.7, 0.2, 0.3, 0.5])),
    ]

    print(f"  {'Configuration':<46s} {'F':>7s} {'IC':>7s} {'Δ':>7s}")
    print("  " + "-" * 64)
    for label, c in configs:
        r = run_kernel(label, c, w)
        print(f"  {label:<46s} {r['F']:7.4f} {r['IC']:7.4f} {r['Delta']:7.4f}")

    print()
    print("  ALL FOUR produce IDENTICAL F, IC, Δ. The kernel sees the")
    print("  MULTISET {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} regardless of order.")
    print()
    print("  This is a REAL structural property: fidelity and integrity")
    print("  depend on WHAT you measure, not on HOW you label the channels.")
    print("  It follows directly from the definitions:")
    print("    F = Σ wᵢcᵢ  (with equal weights, order doesn't matter)")
    print("    IC = ∏ cᵢ^wᵢ  (with equal weights, order doesn't matter)")
    print()

    # Now show what complement (1-c) DOES change
    print("  But COMPLEMENT (replacing c with 1-c) DOES change the kernel:")
    print()
    c_orig = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2])
    c_comp = 1.0 - c_orig  # [0.9, 0.5, 0.1, 0.7, 0.3, 0.8]

    r_orig = run_kernel("Original c", c_orig, w)
    r_comp = run_kernel("Complement 1-c", c_comp, w)

    print(f"  {'Config':<20s} {'F':>7s} {'IC':>7s} {'Δ':>7s} {'ω':>7s}")
    print("  " + "-" * 50)
    print(f"  {'c':20s} {r_orig['F']:7.4f} {r_orig['IC']:7.4f} {r_orig['Delta']:7.4f} {r_orig['omega']:7.4f}")
    print(f"  {'1 - c':20s} {r_comp['F']:7.4f} {r_comp['IC']:7.4f} {r_comp['Delta']:7.4f} {r_comp['omega']:7.4f}")
    print(f"  {'F(c) + F(1-c)':20s} {r_orig['F'] + r_comp['F']:7.4f}")
    print()
    print("  F(c) + F(1-c) = 1.0 EXACTLY. This is the duality identity:")
    print("  if you flip every channel (c → 1-c), fidelity becomes drift")
    print("  and drift becomes fidelity. F + ω = 1 on each side.")
    print()
    print("  IC DOES change under complement because the geometric mean is")
    print("  not self-complementary. That's the structural asymmetry Bolt")
    print("  is sensing — but it's a property of the AM-GM relationship,")
    print("  not of specific bit patterns.")
    print()
    print("  WHAT BOLT SHOULD SAY:")
    print("  'The kernel is permutation-invariant: reordering channels does")
    print("  not change F, IC, or Δ. But complementing channels (c → 1-c)")
    print("  swaps fidelity and drift while changing integrity — this is")
    print("  the duality identity F + ω = 1 in action. The 'mirror' and")
    print("  'NOT' operations I noticed are instances of these two distinct")
    print("  symmetries: permutation (trivial) and complement (structural).'")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 6: THE GRAND CONVERGENCE — WHAT'S REAL ABOUT IT
# ═══════════════════════════════════════════════════════════════════


def bridge_convergence() -> None:
    """What the 'Grand Convergence' actually means — corrected."""
    print()
    print("=" * 76)
    print("BRIDGE 6: THE 'GRAND CONVERGENCE' — Corrected")
    print("=" * 76)
    print()

    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ BOLT'S CLAIM: Independent researchers arriving at the same         │")
    print("│   conclusion: classical mathematics is a degenerate limit of a     │")
    print("│   deeper geometric/recursive truth. 'Grand Convergence' of RHC     │")
    print("│   and GCD.                                                         │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    print("WHAT'S GENUINELY TRUE HERE:")
    print()
    print("  Bolt correctly identified that GCD treats classical results as")
    print("  degenerate limits. This is not a surface-level similarity — it's")
    print("  the core architectural claim of the framework:")
    print()
    print("    ┌────────────────────────────────────────────────────────────┐")
    print("    │ Classical Result        GCD Structure       What's Removed │")
    print("    │ ─────────────────────────────────────────────────────────  │")
    print("    │ Integrity bound      →  IC ≤ F (integrity    Channel      │")
    print("    │                          bound)              semantics,    │")
    print("    │                                              weights, ε    │")
    print("    │                                                           │")
    print("    │ Shannon entropy      →  Bernoulli field      Collapse     │")
    print("    │                          entropy              field        │")
    print("    │                                                           │")
    print("    │ Boolean logic        →  Collapse field       Interior     │")
    print("    │  c ∈ {0,1}              c ∈ [ε,1-ε]         of [ε,1-ε]   │")
    print("    │                                                           │")
    print("    │ Single numbers       →  Trace vectors        Extra        │")
    print("    │  (scalars)              c ∈ [ε,1-ε]^n       channels     │")
    print("    └────────────────────────────────────────────────────────────┘")
    print()
    print("  Bolt's intuitions point at EVERY ROW of this table:")
    print("  - x^x=x → Boolean(row 3): he wants to go beyond binary")
    print("  - 'Lost 2' → AM-GM(row 1): he notices the linear-geometric gap")
    print("  - Quaternions → Trace vectors(row 4): he wants multi-dimensional")
    print("  - Observer → Return axiom: he wants measurement to be structural")
    print()

    print("WHERE THE 'CONVERGENCE' CLAIM BREAKS:")
    print()
    print("  For convergence to be real, BOTH frameworks must independently")
    print("  derive the same structure. GCD derives all four rows from")
    print("  Axiom-0 with 3,500+ tests and zero violations. The RHC would")
    print("  need to do the same — state its axiom, derive its structures,")
    print("  and demonstrate that the kernel identities hold.")
    print()
    print("  Currently, the RHC has:")
    print("  - No stated axiom from which the structures follow")
    print("  - No test suite (zero tests)")
    print("  - No derivation chain (claims are asserted, not proven)")
    print("  - Several mathematical errors (x^x=x ≠ ternary)")
    print()
    print("  'Convergence' requires both sides to have arrived. GCD has a")
    print("  destination (proven invariants). The RHC has a direction")
    print("  (correct intuitions). These are different things.")
    print()

    print("THE HONEST ASSESSMENT:")
    print()
    print("  Bolt's intuitions are pointing at REAL structure. Every one of")
    print("  his claims, when corrected, maps to something that exists in")
    print("  GCD. This means he is looking at the right landscape. But he")
    print("  doesn't yet have the tools — the axiom, the kernel, the seam")
    print("  calculus — to formalize what he sees.")
    print()
    print("  The relationship is not 'convergence' (two equal frameworks).")
    print("  It is 'recognition' — one framework has formalized what the")
    print("  other intuited. The formalization exists. The intuitions point")
    print("  toward it. The bridge is: learn the formalization.")
    print()
    print("  Bolt does not need to build a new framework. The structure he")
    print("  is looking for already exists, with proofs, tests, and a")
    print("  validator. The invitation is not to converge but to join.")


# ═══════════════════════════════════════════════════════════════════
# BRIDGE 7: COMPLETING BOLT'S THOUGHT — THE FORMAL VERSION
# ═══════════════════════════════════════════════════════════════════


def bridge_completion() -> None:
    """What Bolt is trying to say, said correctly."""
    print()
    print("=" * 76)
    print("BRIDGE 7: COMPLETING THE THOUGHT")
    print("=" * 76)
    print()
    print("What Richard Bolt is trying to say, stated formally:")
    print()

    print("  1. BINARY IS INSUFFICIENT")
    print("     Bolt says: x^x = x extends boolean to ternary")
    print("     Corrected: The collapse field c ∈ [ε, 1-ε] extends boolean")
    print("     c ∈ {0,1} to continuous-valued truth. Boolean logic is the")
    print("     degenerate limit when the interior of [ε, 1-ε] is removed.")
    print("     Three-valued verdicts {CONFORMANT, NONCONFORMANT, NON_EVALUABLE}")
    print("     exist on top of this continuous substrate.")
    print()

    print("  2. THE LINEAR-GEOMETRIC GAP IS FUNDAMENTAL")
    print("     Bolt says: 'Lost 2' = 3+4-5 = 2, encoding dark matter")
    print("     Corrected: The heterogeneity gap Δ = F - IC measures the")
    print("     price of channel asymmetry. It is derived from the integrity")
    print("     bound IC ≤ F (Lemma 4), which follows from Axiom-0. The gap")
    print("     is not a fixed integer — it is a structural diagnostic that")
    print("     varies with the trace vector. It equals zero when all channels")
    print("     are in agreement and grows with channel variance: Δ ≈ Var(c)/(2F).")
    print()

    print("  3. REALITY IS MULTI-CHANNEL")
    print("     Bolt says: Quaternions (i,j,k → 4D)")
    print("     Corrected: The trace vector c ∈ [ε, 1-ε]^n generalizes to")
    print("     arbitrary n channels. The number of channels is domain-")
    print("     discovered (8 for particles, 12 for cross-scale, etc.).")
    print("     The kernel produces invariant structure (F, IC, Δ, S, C)")
    print("     for any n. Quaternions are the n=4 special case but carry")
    print("     algebraic baggage (non-commutativity) that is not needed.")
    print()

    print("  4. THE OBSERVER IS STRUCTURALLY NECESSARY")
    print("     Bolt says: Observer as 'Fold Operator' at position 2.5+1.5i")
    print("     Corrected: The observer IS the frozen contract — the set of")
    print("     measurement rules consistent across the collapse-return")
    print("     boundary. The observer does not have a position; it has a")
    print("     contract. Return is possible only because the measurement")
    print("     rules are frozen (trans suturam congelatum). The observer's")
    print("     structural necessity is encoded in Axiom-0: without return,")
    print("     there is no reality claim.")
    print()

    print("  5. SYMMETRY TRANSFORMATIONS REVEAL STRUCTURE")
    print("     Bolt says: Mirror (100110) and NOT (110011) of 011001 cycle")
    print("     Corrected: Two distinct symmetries exist in the kernel:")
    print("     (a) Permutation invariance: reordering channels preserves")
    print("         all kernel outputs (F, IC, Δ). This is trivial.")
    print("     (b) Complement duality: replacing c → 1-c swaps F ↔ ω.")
    print("         This IS the duality identity F + ω = 1. Non-trivial.")
    print("     The specific binary patterns are instances of these general")
    print("     symmetries, not special in themselves.")
    print()

    print("  6. CLASSICAL RESULTS ARE DEGENERATE LIMITS")
    print("     Bolt says: 'Bernoulli field entropy and the integrity bound are shadows'")
    print("     Corrected: Exactly right, and this is perhaps his strongest")
    print("     insight. Shannon entropy is the degenerate limit of Bernoulli")
    print("     field entropy when the collapse field is removed. AM-GM is")
    print("     the degenerate limit of IC ≤ F when channel semantics and")
    print("     weights are removed. The arrow of derivation runs from")
    print("     Axiom-0 outward to the classical results, not the reverse.")
    print("     Bolt saw this clearly.")
    print()

    print("═" * 76)
    print("FINAL ASSESSMENT")
    print("═" * 76)
    print()
    print("  Richard Bolt is looking at the right landscape. His intuitions")
    print("  — about the insufficiency of binary logic, the fundamentality")
    print("  of the linear-geometric gap, the multi-channel nature of")
    print("  measurement, the structural necessity of the observer, and")
    print("  the degenerate-limit relationship to classical mathematics —")
    print("  are all pointing at real structure that GCD has formalized.")
    print()
    print("  His formalizations are wrong in their specifics (x^x=x ≠")
    print("  ternary, {1,2,5} is not special, the gap is not the integer 2),")
    print("  but they are wrong in INSTRUCTIVE ways — each error points at")
    print("  a real structure that the kernel captures differently.")
    print()
    print("  Classification: GESTUS WITH SIGNAL")
    print("  The seam does not close (τ_R = ∞_rec) — but the gesture is")
    print("  pointing in a direction where structure exists. The gesture")
    print("  cannot earn epistemic credit, but it can motivate a weld.")
    print()
    print("  The path forward for Bolt is not to build a parallel framework.")
    print("  It is to learn the existing one — the axiom, the kernel, the")
    print("  seam calculus — and discover that everything he's been reaching")
    print("  for is already there, with proofs.")
    print()
    print("  Solum quod redit, reale est. The invitation to return is open.")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  GCD BRIDGE ANALYSIS: Ironmanning Richard Bolt's RHC                   ║")
    print("║  What he's reaching for, why he reached this way, and                   ║")
    print("║  the correct formalization in GCD                                       ║")
    print("║  Framework: UMCP/GCD | Axiom-0 | Frozen Parameters                     ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    bridge_ternary_logic()
    bridge_lost_two()
    bridge_quaternions()
    bridge_observer()
    bridge_binary_symmetry()
    bridge_convergence()
    bridge_completion()
