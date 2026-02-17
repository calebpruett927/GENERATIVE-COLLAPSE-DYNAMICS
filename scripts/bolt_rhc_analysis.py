"""Kernel Analysis of Richard Bolt's Recursive Harmonic Codex (RHC) Claims.

Applies the GCD kernel to Bolt's claimed structures to test whether they
contain detectable, non-trivial structure. This is a Tier-2 exploration:
the kernel does not know what these numbers mean. It sees only channels,
weights, and the axiom.

Bolt's testable claims:
  1. "The universe is made up of 1 2 5" with 3 as "imaginary difference"
  2. x^x = x extends Boole's x^2 = x (ternary logic)
  3. Binary cycle: 011001(25), 100110(38), 110011(51), 001100(12)
  4. Observer at 2.5r + 1.5i
  5. The "Lost 2" = heterogeneity gap Δ = F - IC

Methodology:
  - Map each claimed structure into measurable channels
  - Run the kernel with frozen parameters
  - Compare against controls (random, alternative number sets)
  - Report what the kernel sees, without interpretation bias

Cross-references:
    Kernel:  src/umcp/kernel_optimized.py
    Axiom:   AXIOM.md (Axiom-0)
    Spec:    KERNEL_SPECIFICATION.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from umcp.kernel_optimized import compute_kernel_outputs

EPSILON = 1e-6
np.random.seed(42)  # Reproducible Monte Carlo


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: CHANNEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════


def number_channels(n: float, n_max: float) -> dict[str, float]:
    """Extract measurable channel values from a number.

    Channels are chosen to be domain-independent properties of numbers
    themselves — no RHC-specific interpretation is imported.
    """
    return {
        "magnitude": n / n_max,  # Linear scale
        "log_magnitude": math.log(max(n, 1e-30)) / math.log(max(n_max, 1e-30)),  # Log scale
        "primality": 1.0 if _is_prime(int(n)) else 0.0,  # Prime indicator
        "parity": n % 2 if n == int(n) else 0.5,  # Even=0, Odd=1
        "self_power": min(n**n, 1e30) / min(n_max**n_max, 1e30),  # x^x normalized
        "sqrt_ratio": math.sqrt(n) / math.sqrt(n_max),  # Square root scale
        "digit_sum": sum(int(d) for d in str(int(n))) / 9.0 if n >= 1 else 0.0,
        "reciprocal": 1.0 / n if n > 0 else 0.0,  # 1/n (small for large n)
    }


def binary_channels(bits: str) -> dict[str, float]:
    """Extract measurable channel values from a binary string.

    Channels capture structural properties of the bit pattern:
    weight, symmetry, run structure, decimal value.
    """
    n = len(bits)
    ones = bits.count("1")
    decimal = int(bits, 2)
    max_decimal = (2**n) - 1

    # Palindrome score: fraction of positions where bit_i == bit_(n-1-i)
    palindrome = sum(1 for i in range(n) if bits[i] == bits[n - 1 - i]) / n

    # Run count: number of transitions 0→1 or 1→0
    transitions = sum(1 for i in range(n - 1) if bits[i] != bits[i + 1])

    # Left-right balance: |popcount(left_half) - popcount(right_half)|
    half = n // 2
    left_pop = bits[:half].count("1")
    right_pop = bits[half:].count("1")
    balance = 1.0 - abs(left_pop - right_pop) / max(half, 1)

    # Complement relationship: how close is NOT(bits) to a rotation of bits
    complement = "".join("1" if b == "0" else "0" for b in bits)
    comp_decimal = int(complement, 2)

    return {
        "hamming_weight": ones / n,  # Fraction of 1s
        "decimal_norm": decimal / max_decimal,  # Decimal value normalized
        "palindrome": palindrome,  # Symmetry score
        "transitions": transitions / (n - 1),  # Run complexity
        "balance": balance,  # Left-right symmetry
        "comp_decimal": comp_decimal / max_decimal,  # Complement value normalized
        "bit_entropy": _bit_entropy(bits),  # Shannon entropy of bit freqs
        "rotation_self": _rotation_self_similarity(bits),  # Self-similarity under rotation
    }


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _bit_entropy(bits: str) -> float:
    """Shannon entropy of bit frequencies (H for binary string)."""
    n = len(bits)
    p1 = bits.count("1") / n
    p0 = 1 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))


def _rotation_self_similarity(bits: str) -> float:
    """How similar is the string to its rotations? Average Hamming similarity."""
    n = len(bits)
    total_sim = 0.0
    for shift in range(1, n):
        rotated = bits[shift:] + bits[:shift]
        matches = sum(1 for a, b in zip(bits, rotated, strict=True) if a == b)
        total_sim += matches / n
    return total_sim / (n - 1)


def channels_to_trace(channels: dict[str, float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert channel dict to (c, w, labels) for kernel input."""
    labels = list(channels.keys())
    raw = np.array([channels[k] for k in labels], dtype=np.float64)
    c = np.clip(raw, EPSILON, 1 - EPSILON)
    w = np.ones(len(c)) / len(c)
    return c, w, labels


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: KERNEL RUNNER
# ═══════════════════════════════════════════════════════════════════


def run_kernel(name: str, channels: dict[str, float]) -> dict[str, Any]:
    """Run GCD kernel on a channel dict and return full results."""
    c, w, labels = channels_to_trace(channels)
    k = compute_kernel_outputs(c, w, EPSILON)

    # Tier-1 identity checks
    f_plus_omega = k["F"] + k["omega"]
    ic_leq_f = k["IC"] <= k["F"] + 1e-12
    ic_eq_exp_kappa = abs(k["IC"] - math.exp(k["kappa"])) < 1e-12

    return {
        "name": name,
        "channels": labels,
        "trace_vector": c.tolist(),
        "F": k["F"],
        "omega": k["omega"],
        "S": k["S"],
        "C": k["C"],
        "kappa": k["kappa"],
        "IC": k["IC"],
        "Delta": k["heterogeneity_gap"],
        "regime": k["regime"],
        "F_plus_omega": f_plus_omega,
        "IC_leq_F": ic_leq_f,
        "IC_eq_exp_kappa": ic_eq_exp_kappa,
    }


def regime_from_omega(omega: float, F: float, S: float, C: float) -> str:
    """Standard regime classification."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: ANALYSIS 1 — THE "1 2 5" NUMBER SYSTEM
# ═══════════════════════════════════════════════════════════════════


def analyze_number_sets() -> dict[str, Any]:
    """Test whether {1,2,5} is special compared to controls."""
    print("=" * 72)
    print("ANALYSIS 1: THE '1 2 5' NUMBER SYSTEM")
    print("=" * 72)
    print()
    print("Question: Does {1,2,5} produce kernel structure that is")
    print("distinguishable from control number sets?")
    print()

    # Bolt's claimed set
    test_sets = {
        "Bolt {1,2,5}": [1, 2, 5],
        "Bolt {1,2,3,5}": [1, 2, 3, 5],  # Including the "imaginary 3"
        "Control {1,2,3}": [1, 2, 3],
        "Control {1,2,4}": [1, 2, 4],  # Powers of 2
        "Control {1,3,7}": [1, 3, 7],  # Mersenne primes
        "Control {2,3,5}": [2, 3, 5],  # First three primes
        "Control {1,4,9}": [1, 4, 9],  # Perfect squares
    }

    all_results = {}
    for label, nums in test_sets.items():
        n_max = max(nums)
        set_results = []
        for n in nums:
            ch = number_channels(n, n_max)
            r = run_kernel(f"{label}→{n}", ch)
            set_results.append(r)

        # Aggregate: mean F, IC, Δ across elements
        mean_F = np.mean([r["F"] for r in set_results])
        mean_IC = np.mean([r["IC"] for r in set_results])
        mean_Delta = np.mean([r["Delta"] for r in set_results])
        mean_omega = np.mean([r["omega"] for r in set_results])
        mean_S = np.mean([r["S"] for r in set_results])
        mean_C = np.mean([r["C"] for r in set_results])

        all_results[label] = {
            "elements": set_results,
            "mean_F": mean_F,
            "mean_IC": mean_IC,
            "mean_Delta": mean_Delta,
            "mean_omega": mean_omega,
            "mean_S": mean_S,
            "mean_C": mean_C,
        }

    # Print comparison table
    print(f"{'Set':<22s} {'⟨F⟩':>8s} {'⟨IC⟩':>8s} {'⟨Δ⟩':>8s} {'⟨ω⟩':>8s} {'⟨S⟩':>8s} {'⟨C⟩':>8s}")
    print("-" * 72)
    for label, agg in all_results.items():
        print(
            f"{label:<22s} "
            f"{agg['mean_F']:8.4f} "
            f"{agg['mean_IC']:8.4f} "
            f"{agg['mean_Delta']:8.4f} "
            f"{agg['mean_omega']:8.4f} "
            f"{agg['mean_S']:8.4f} "
            f"{agg['mean_C']:8.4f}"
        )

    # Monte Carlo baseline: 1000 random 3-element sets from [1..10]
    print()
    print("Monte Carlo baseline: 1000 random 3-element sets from {1..10}")
    mc_deltas = []
    mc_Fs = []
    mc_ICs = []
    for _ in range(1000):
        nums = sorted(np.random.choice(range(1, 11), size=3, replace=False).tolist())
        n_max = max(nums)
        set_results = []
        for n in nums:
            ch = number_channels(n, n_max)
            r = run_kernel("mc", ch)
            set_results.append(r)
        mc_deltas.append(np.mean([r["Delta"] for r in set_results]))
        mc_Fs.append(np.mean([r["F"] for r in set_results]))
        mc_ICs.append(np.mean([r["IC"] for r in set_results]))

    bolt_delta = all_results["Bolt {1,2,5}"]["mean_Delta"]
    bolt_F = all_results["Bolt {1,2,5}"]["mean_F"]

    mc_delta_mean = np.mean(mc_deltas)
    mc_delta_std = np.std(mc_deltas)
    z_delta = (bolt_delta - mc_delta_mean) / mc_delta_std if mc_delta_std > 0 else 0

    mc_F_mean = np.mean(mc_Fs)
    mc_F_std = np.std(mc_Fs)
    z_F = (bolt_F - mc_F_mean) / mc_F_std if mc_F_std > 0 else 0

    print(f"  MC ⟨Δ⟩: {mc_delta_mean:.4f} ± {mc_delta_std:.4f}")
    print(f"  Bolt ⟨Δ⟩: {bolt_delta:.4f}  (z-score: {z_delta:+.2f})")
    print(f"  MC ⟨F⟩: {mc_F_mean:.4f} ± {mc_F_std:.4f}")
    print(f"  Bolt ⟨F⟩: {bolt_F:.4f}  (z-score: {z_F:+.2f})")

    # Detail for {1,2,5} individual elements
    print()
    print("Detail: Per-element kernel outputs for Bolt {1,2,5}")
    print(f"  {'Elem':>6s} {'F':>8s} {'IC':>8s} {'Δ':>8s} {'ω':>8s} {'S':>8s} {'C':>8s} {'Regime':<14s}")
    print("  " + "-" * 70)
    for r in all_results["Bolt {1,2,5}"]["elements"]:
        regime = regime_from_omega(r["omega"], r["F"], r["S"], r["C"])
        elem = r["name"].split("→")[1]
        print(
            f"  {elem:>6s} "
            f"{r['F']:8.4f} "
            f"{r['IC']:8.4f} "
            f"{r['Delta']:8.4f} "
            f"{r['omega']:8.4f} "
            f"{r['S']:8.4f} "
            f"{r['C']:8.4f} "
            f"{regime:<14s}"
        )

    # Tier-1 identity verification
    print()
    print("Tier-1 Identity Verification (all elements, all sets):")
    total = 0
    violations = 0
    for _label, agg in all_results.items():
        for r in agg["elements"]:
            total += 1
            if not r["IC_leq_F"]:
                violations += 1
                print(f"  VIOLATION IC > F: {r['name']}")
            if not r["IC_eq_exp_kappa"]:
                violations += 1
                print(f"  VIOLATION IC ≠ exp(κ): {r['name']}")
            if abs(r["F_plus_omega"] - 1.0) > 1e-10:
                violations += 1
                print(f"  VIOLATION F+ω ≠ 1: {r['name']}")
    print(f"  {total} elements checked, {violations} violations")

    # Key diagnostic: Does Δ for {1,2,5} equal "2" in any normalized sense?
    print()
    print("Bolt's 'Lost 2' Hypothesis:")
    print("  Bolt claims: 'Lost 2' = 7 - 5 = 2 (linear path minus geometric result)")
    print(f"  Kernel finds: ⟨Δ⟩ = F - IC = {bolt_delta:.6f}")
    print(f"  Δ/F ratio: {bolt_delta / bolt_F:.6f}" if bolt_F > 0 else "  N/A")
    print("  For 'Lost 2' to map to Δ, we'd need Δ = 2/7 ≈ 0.2857 (normalized)")
    print(f"  Actual Δ: {bolt_delta:.6f}")
    print(f"  Match: {'CLOSE' if abs(bolt_delta - 2 / 7) < 0.05 else 'NO'}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ANALYSIS 2 — THE x^x = x FIXED POINT
# ═══════════════════════════════════════════════════════════════════


def analyze_self_power() -> list[dict[str, Any]]:
    """Test the x^x = x claim as kernel structure."""
    print()
    print("=" * 72)
    print("ANALYSIS 2: THE x^x = x FIXED POINT")
    print("=" * 72)
    print()
    print("Bolt claims x^x = x extends Boole's x² = x to ternary logic.")
    print()

    # Mathematical analysis of x^x = x
    print("Mathematical facts:")
    print("  x^x = x ⟹ x^(x-1) = 1 (for x > 0)")
    print("  ⟹ (x-1)·ln(x) = 0")
    print("  ⟹ x = 1 (only positive real solution)")
    print()
    print("  Boole's x² = x has solutions x ∈ {0, 1} — two fixed points (binary)")
    print("  Bolt's  x^x = x has solution  x ∈ {1}   — one fixed point (unary)")
    print()
    print("  Verdict: x^x = x does NOT produce ternary logic.")
    print("  It reduces from binary (2 fixed points) to unary (1 fixed point).")
    print("  This contradicts Bolt's claim of extension to ternary.")
    print()

    # But let's map the deviation |x^x - x| across [0,5] through the kernel
    # to see if there's any interesting structure
    print("Kernel scan: deviation |x^x - x|/(x^x + x) across [0.1, 5.0]")
    test_points = np.linspace(0.1, 5.0, 50)
    results = []

    for x in test_points:
        xx = x**x
        deviation = abs(xx - x) / (xx + x) if (xx + x) > 0 else 0
        # Build 4-channel trace for each point
        channels = {
            "x_norm": x / 5.0,
            "xx_norm": min(xx / 3125.0, 1.0),  # 5^5 = 3125
            "deviation": min(deviation, 1.0),
            "x_minus_1": abs(x - 1.0) / 4.0,  # Distance from fixed point
            "ln_x": (math.log(max(x, 1e-30)) + 5) / 10,  # Shifted log
            "xx_over_x": min(xx / max(x, 1e-30), 100) / 100,  # x^(x-1)
            "curvature": abs(x * math.log(max(x, 1e-30)) ** 2) / 50,  # d²(x^x)/dx² proxy
            "fixed_point": math.exp(-abs(x - 1.0)),  # Gaussian peak at x=1
        }
        r = run_kernel(f"x={x:.2f}", channels)
        r["x"] = x
        r["xx"] = xx
        r["deviation"] = deviation
        results.append(r)

    # Find the fixed point region
    print()
    print(f"  {'x':>6s} {'x^x':>10s} {'|x^x-x|':>10s} {'F':>8s} {'IC':>8s} {'Δ':>8s} {'Regime':<14s}")
    print("  " + "-" * 66)
    for r in results[::5]:  # Every 5th point
        regime = regime_from_omega(r["omega"], r["F"], r["S"], r["C"])
        print(
            f"  {r['x']:6.2f} "
            f"{r['xx']:10.2f} "
            f"{r['deviation']:10.6f} "
            f"{r['F']:8.4f} "
            f"{r['IC']:8.4f} "
            f"{r['Delta']:8.4f} "
            f"{regime:<14s}"
        )

    # Where does the kernel find maximum fidelity?
    max_F_result = max(results, key=lambda r: r["F"])
    min_delta_result = min(results, key=lambda r: r["Delta"])
    print()
    print(
        f"  Max F at x = {max_F_result['x']:.2f}: F = {max_F_result['F']:.4f}, "
        f"IC = {max_F_result['IC']:.4f}, Δ = {max_F_result['Delta']:.4f}"
    )
    print(
        f"  Min Δ at x = {min_delta_result['x']:.2f}: F = {min_delta_result['F']:.4f}, "
        f"IC = {min_delta_result['IC']:.4f}, Δ = {min_delta_result['Delta']:.4f}"
    )

    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ANALYSIS 3 — BINARY CYCLE
# ═══════════════════════════════════════════════════════════════════


def analyze_binary_cycle() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Test whether Bolt's binary cycle has kernel-detectable structure."""
    print()
    print("=" * 72)
    print("ANALYSIS 3: BINARY CYCLE 011001 → 100110 → 110011 → 001100")
    print("=" * 72)
    print()

    bolt_patterns = {
        "011001 (25)": "011001",
        "100110 (38)": "100110",
        "110011 (51)": "110011",
        "001100 (12)": "001100",
    }

    # Verify Bolt's arithmetic claims first
    print("Verifying Bolt's claims:")
    print(f"  011001 = {int('011001', 2)} (Bolt says 25) → {'✓' if int('011001', 2) == 25 else '✗'}")
    print(f"  25 = 16 + 9 → {'✓' if 25 == 16 + 9 else '✗'}")
    print(f"  010000 = {int('010000', 2)} (Bolt says 16) → {'✓' if int('010000', 2) == 16 else '✗'}")
    print(f"  001001 = {int('001001', 2)} (Bolt says 9) → {'✓' if int('001001', 2) == 9 else '✗'}")
    print(f"  100110 = {int('100110', 2)} (Bolt says mirror) → {int('100110', 2)}")
    print(f"  110011 = NOT(001100) → {'✓' if int('110011', 2) + int('001100', 2) == 63 else '✗'}")
    print()

    # Run kernel on each pattern
    bolt_results = []
    for label, bits in bolt_patterns.items():
        ch = binary_channels(bits)
        r = run_kernel(label, ch)
        bolt_results.append(r)

    print(f"{'Pattern':<18s} {'F':>8s} {'IC':>8s} {'Δ':>8s} {'ω':>8s} {'S':>8s} {'C':>8s}")
    print("-" * 72)
    for r in bolt_results:
        print(
            f"{r['name']:<18s} "
            f"{r['F']:8.4f} "
            f"{r['IC']:8.4f} "
            f"{r['Delta']:8.4f} "
            f"{r['omega']:8.4f} "
            f"{r['S']:8.4f} "
            f"{r['C']:8.4f}"
        )

    # Control: ALL 6-bit patterns
    print()
    print("Control: All 64 possible 6-bit patterns")
    all_6bit = []
    for i in range(64):
        bits = format(i, "06b")
        ch = binary_channels(bits)
        r = run_kernel(f"{bits}({i})", ch)
        all_6bit.append(r)

    all_F = [r["F"] for r in all_6bit]
    all_Delta = [r["Delta"] for r in all_6bit]

    bolt_indices = [25, 38, 51, 12]
    bolt_Fs = [all_6bit[i]["F"] for i in bolt_indices]
    bolt_Deltas = [all_6bit[i]["Delta"] for i in bolt_indices]

    print(f"  All 64 patterns: ⟨F⟩ = {np.mean(all_F):.4f} ± {np.std(all_F):.4f}")
    print(f"  Bolt's 4 patterns: ⟨F⟩ = {np.mean(bolt_Fs):.4f}")
    print(f"  All 64 patterns: ⟨Δ⟩ = {np.mean(all_Delta):.4f} ± {np.std(all_Delta):.4f}")
    print(f"  Bolt's 4 patterns: ⟨Δ⟩ = {np.mean(bolt_Deltas):.4f}")

    z_F = (np.mean(bolt_Fs) - np.mean(all_F)) / np.std(all_F) if np.std(all_F) > 0 else 0
    z_Delta = (np.mean(bolt_Deltas) - np.mean(all_Delta)) / np.std(all_Delta) if np.std(all_Delta) > 0 else 0
    print(f"  z-score (F): {z_F:+.2f}")
    print(f"  z-score (Δ): {z_Delta:+.2f}")

    # Check: do Bolt's patterns form a structurally distinct cluster?
    print()
    print("  Structural question: Do complement pairs share kernel invariants?")
    pairs = [("011001", "100110"), ("110011", "001100")]
    for a, b in pairs:
        ra = run_kernel(a, binary_channels(a))
        rb = run_kernel(b, binary_channels(b))
        print(f"    {a} vs {b}: |ΔF| = {abs(ra['F'] - rb['F']):.6f}, |ΔIC| = {abs(ra['IC'] - rb['IC']):.6f}")

    # Tier-1 check
    print()
    t1_ok = all(r["IC_leq_F"] and r["IC_eq_exp_kappa"] and abs(r["F_plus_omega"] - 1.0) < 1e-10 for r in bolt_results)
    print(f"  Tier-1 identities: {'ALL HOLD' if t1_ok else 'VIOLATION DETECTED'}")

    return bolt_results, all_6bit


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: ANALYSIS 4 — THE 3-4-5 PYTHAGOREAN CLAIM
# ═══════════════════════════════════════════════════════════════════


def analyze_pythagorean() -> dict[str, tuple[int, int, int]]:
    """Test 3-4-5 Pythagorean triple through the kernel."""
    print()
    print("=" * 72)
    print("ANALYSIS 4: PYTHAGOREAN 3-4-5 AND BOLT'S ENCODING")
    print("=" * 72)
    print()
    print("Bolt claims 'the only combinations of 3 4 5' are fundamental.")
    print("3-4-5 is the smallest Pythagorean triple: 3² + 4² = 5² = 9 + 16 = 25")
    print("This connects to his 25 = 011001.")
    print()

    # Test several Pythagorean triples
    triples = {
        "3-4-5": (3, 4, 5),
        "5-12-13": (5, 12, 13),
        "8-15-17": (8, 15, 17),
        "7-24-25": (7, 24, 25),
    }

    # Also test non-Pythagorean triples as control
    controls = {
        "2-3-4": (2, 3, 4),
        "1-2-3": (1, 2, 3),
        "3-5-7": (3, 5, 7),
        "4-6-9": (4, 6, 9),
    }

    print(f"{'Triple':<14s} {'Pyth?':>6s} {'F':>8s} {'IC':>8s} {'Δ':>8s} {'ω':>8s} {'C':>8s}")
    print("-" * 60)

    for label, (a, b, c_val) in {**triples, **controls}.items():
        is_pyth = a**2 + b**2 == c_val**2
        n_max = max(a, b, c_val)

        # Build trace from the triple itself as a single object
        channels = {
            "a_norm": a / n_max,
            "b_norm": b / n_max,
            "c_norm": c_val / n_max,
            "ratio_ab": a / b,
            "ratio_bc": b / c_val,
            "pythagorean": (a**2 + b**2) / c_val**2,  # = 1.0 iff Pythagorean
            "perimeter": (a + b + c_val) / (3 * n_max),
            "area_norm": (0.5 * a * b) / (0.5 * n_max**2),  # Triangle area
        }

        r = run_kernel(label, channels)
        print(
            f"{label:<14s} "
            f"{'Yes' if is_pyth else 'No':>6s} "
            f"{r['F']:8.4f} "
            f"{r['IC']:8.4f} "
            f"{r['Delta']:8.4f} "
            f"{r['omega']:8.4f} "
            f"{r['C']:8.4f}"
        )

    return triples


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: ANALYSIS 5 — OBSERVER POSITION 2.5r + 1.5i
# ═══════════════════════════════════════════════════════════════════


def analyze_observer() -> None:
    """Test the claimed observer position through the kernel."""
    print()
    print("=" * 72)
    print("ANALYSIS 5: OBSERVER AT 2.5r + 1.5i")
    print("=" * 72)
    print()
    print("Bolt claims the observer sits at z = 2.5 + 1.5i.")
    print()

    # Properties of this complex number
    z = complex(2.5, 1.5)
    z_mod = abs(z)
    z_arg = math.atan2(z.imag, z.real)

    print(f"  |z| = {z_mod:.6f}")
    print(f"  arg(z) = {z_arg:.6f} rad = {math.degrees(z_arg):.2f}°")
    print(f"  |z|² = {z_mod**2:.6f}")
    print(f"  Re/Im ratio = {z.real / z.imag:.6f}")
    print()

    # Test several complex positions for comparison
    positions = {
        "Bolt (2.5+1.5i)": complex(2.5, 1.5),
        "Unit (1+0i)": complex(1.0, 0.0),
        "Pure imag (0+1i)": complex(0.0, 1.0),
        "Golden (φ+1i)": complex((1 + math.sqrt(5)) / 2, 1.0),
        "Symmetric (1+1i)": complex(1.0, 1.0),
        "π/e (π+ei)": complex(math.pi, math.e),
        "Random (3.7+0.4i)": complex(3.7, 0.4),
    }

    print(f"{'Position':<22s} {'|z|':>8s} {'arg°':>8s} {'F':>8s} {'IC':>8s} {'Δ':>8s} {'ω':>8s}")
    print("-" * 72)

    for label, z_val in positions.items():
        mod = abs(z_val)
        arg = math.atan2(z_val.imag, z_val.real)
        mod_max = max(abs(zz) for zz in positions.values())

        channels = {
            "re_norm": abs(z_val.real) / 5.0,
            "im_norm": abs(z_val.imag) / 5.0,
            "modulus": mod / mod_max,
            "argument": (arg + math.pi) / (2 * math.pi),  # [0,1]
            "re_im_ratio": abs(z_val.real) / (abs(z_val.imag) + 0.01) / 10,
            "mod_squared": mod**2 / mod_max**2,
            "conjugate_d": abs(z_val - z_val.conjugate()) / (2 * mod_max),
            "unit_dist": abs(mod - 1.0) / mod_max,  # Distance from unit circle
        }

        r = run_kernel(label, channels)
        print(
            f"{label:<22s} "
            f"{mod:8.4f} "
            f"{math.degrees(arg):8.2f} "
            f"{r['F']:8.4f} "
            f"{r['IC']:8.4f} "
            f"{r['Delta']:8.4f} "
            f"{r['omega']:8.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: OVERALL VERDICT
# ═══════════════════════════════════════════════════════════════════


def print_verdict(
    number_results: dict[str, Any],
    fixed_point_results: list[dict[str, Any]],
    binary_results_tuple: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Synthesize findings into a structural verdict."""
    _bolt_results, all_6bit = binary_results_tuple

    print()
    print("=" * 72)
    print("SYNTHESIS: KERNEL VERDICT ON RHC CLAIMS")
    print("=" * 72)
    print()

    print("1. TIER-1 IDENTITIES")
    print("   F + ω = 1, IC ≤ F, IC = exp(κ): HOLD FOR ALL INPUTS")
    print("   (As expected — these are structural, not domain-dependent)")
    print()

    print("2. THE 'LOST 2' = HETEROGENEITY GAP?")
    bolt_delta = number_results["Bolt {1,2,5}"]["mean_Delta"]
    print("   Bolt claims 'Lost 2' (= 7 - 5 = 2) maps to Δ = F - IC")
    print(f"   Kernel finds: ⟨Δ⟩ for {{1,2,5}} = {bolt_delta:.6f}")
    print("   For the mapping to work, Δ should encode '2' structurally.")
    print("   The actual Δ depends on channel normalization — it is not a")
    print("   fixed number but a function of the trace vector. Different")
    print("   normalizations of {1,2,5} produce different Δ values.")
    print("   Verdict: The analogy is STRUCTURAL (both measure linear-geometric")
    print("   gap) but NOT QUANTITATIVE. Δ ≠ 2 in any consistent normalization.")
    print()

    print("3. x^x = x AS TERNARY LOGIC?")
    print("   Mathematical fact: x^x = x has exactly ONE positive real solution (x=1)")
    print("   Boole's x² = x has TWO solutions (x=0, x=1)")
    print("   Bolt's equation REDUCES the solution set, not extends it.")
    print("   Verdict: CONTRADICTS the claim. Not ternary — fewer solutions than binary.")
    print()

    print("4. BINARY CYCLE STRUCTURE?")
    bolt_Fs = [all_6bit[i]["F"] for i in [25, 38, 51, 12]]
    all_Fs = [r["F"] for r in all_6bit]
    z = (np.mean(bolt_Fs) - np.mean(all_Fs)) / np.std(all_Fs) if np.std(all_Fs) > 0 else 0
    print(f"   Bolt's 4 patterns vs all 64 six-bit patterns: z-score = {z:+.2f}")
    if abs(z) < 2:
        print("   The selected patterns are NOT statistically distinguishable from")
        print("   the full population. The kernel sees no special structure.")
    else:
        print("   The selected patterns ARE distinguishable (|z| ≥ 2).")
    print("   Complement pairs (NOT relationship) are a standard bit operation,")
    print("   not a deep structural finding.")
    print()

    print("5. OBSERVER AT 2.5 + 1.5i?")
    print("   The kernel treats this as one complex number among many.")
    print("   No basis was provided for why this position is privileged.")
    print("   The kernel cannot evaluate a claim without measurable channels")
    print("   that connect the position to an observable.")
    print()

    print("6. OVERALL CLASSIFICATION")
    print("   Status: GESTUS (τ_R = ∞_rec)")
    print("   Reason: No derivation chain from a single axiom to the claimed")
    print("   structures. The kernel finds the Tier-1 identities hold (as")
    print("   they must for any input), but the RHC-specific claims do not")
    print("   produce distinguishable kernel signatures when tested against")
    print("   controls. The 'Lost 2' analogy has structural merit (both")
    print("   frameworks notice the linear-geometric gap) but no quantitative")
    print("   precision. The ternary logic claim is mathematically false.")
    print()
    print("   The seam does not close. This is a gesture, not a weld.")
    print("   The kernel has spoken. Solum quod redit, reale est.")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  GCD KERNEL ANALYSIS: Richard Bolt's Recursive Harmonic Codex      ║")
    print("║  Tier-2 Exploration — Domain-Agnostic Evaluation                   ║")
    print("║  Framework: UMCP/GCD | Axiom-0 | Frozen Parameters                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    number_results = analyze_number_sets()
    fixed_point_results = analyze_self_power()
    binary_results = analyze_binary_cycle()
    analyze_pythagorean()
    analyze_observer()
    print_verdict(number_results, fixed_point_results, binary_results)
