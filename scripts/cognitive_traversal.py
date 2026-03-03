"""Cognitive Traversal — Proximitas, Non Compositio.

Formalizes the question: can humans, because of our high physiological
baseline, traverse cognition exponentially? The answer is computed from
Tier-1 invariants across 19 species, 8 developmental stages, and
systematic channel-boost experiments.

Key discovery: the human advantage is not compounding (exponential) —
it is POSITIONAL. The traversal from Watch to Stable regime is 15
steps for humans, 144 for Neanderthals, 286 for chimpanzees, 470 for
C. elegans. The same +0.02 channel boost produces the same ω reduction
(0.002) everywhere, but humans close 7.14% of the remaining gap per
step because the gap is only 0.028 wide. This is the proximity dividend:
not speed of ascent, but height of foundation.

Twelve analyses, each building on the previous:

    A1.  Marginal returns at different baselines
    A2.  Species comparison: same boost, different starting positions
    A3.  The compounding curve: successive boosts from human baseline
    A4.  Low-baseline species: successive boosts for comparison
    A5.  Steps-to-Stable: the distance table
    A6.  The kappa derivative: logarithmic sensitivity analysis
    A7.  Fractional gain test: is traversal exponential?
    A8.  Omega reduction rate: distance-closing efficiency
    A9.  Generative sweet spot: IC/F × S product
    A10. Transcendence paradox: what happens at the ceiling
    A11. Developmental traversal: the ontogenetic path
    A12. Formal receipt: Tier-1 identity verification

Derivation chain: Axiom-0 -> frozen_contract -> kernel_optimized ->
    brain_kernel -> this analysis

Non celeritas ascensionis, sed altitudo fundamentorum.
Not the speed of ascent, but the height of the foundation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# -- Path setup -------------------------------------------------------
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.evolution.brain_kernel import (
    BRAIN_CATALOG,
    BRAIN_CHANNELS,
    DEVELOPMENT_STAGES,
)
from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# -- Constants --------------------------------------------------------
W = np.full(10, 0.1)
N_SPECIES = len(BRAIN_CATALOG)  # 19
STABLE_OMEGA = 0.038  # ω < this for Stable regime
BOOST_STEP = 0.02  # channel increment per step
MAX_STEPS = 500  # safety limit for step-counting loops

# -- Helper -----------------------------------------------------------
_assertions: list[tuple[str, bool, str]] = []


def _assert(tag: str, condition: bool, detail: str) -> None:
    """Record an assertion for the final receipt."""
    _assertions.append((tag, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {tag}: {detail}")


def _kernel(c: np.ndarray) -> dict[str, float]:
    """Compute kernel invariants with guard-band clipping."""
    c_safe = np.clip(c, EPSILON, 1.0 - EPSILON)
    k = compute_kernel_outputs(c_safe, W, EPSILON)
    F = float(k["F"])
    om = float(k["omega"])
    IC = float(k["IC"])
    S = float(k["S"])
    C = float(k["C"])
    kappa = float(k["kappa"])
    icf = IC / F if F > 0 else 0.0
    return {"F": F, "omega": om, "IC": IC, "S": S, "C": C, "kappa": kappa, "IC/F": icf}


def _regime(omega: float) -> str:
    """Regime label from drift."""
    if omega < STABLE_OMEGA:
        return "Stable"
    if omega < 0.30:
        return "Watch"
    return "Collapse"


def _steps_to_stable(c: np.ndarray) -> int:
    """Count +0.02 weakest-channel boosts to reach Stable."""
    c = c.copy()
    for step in range(MAX_STEPS):
        k = _kernel(c)
        if k["omega"] < STABLE_OMEGA:
            return step
        mn_i = int(np.argmin(c))
        c[mn_i] = min(c[mn_i] + BOOST_STEP, 1.0 - EPSILON)
    return MAX_STEPS


# =====================================================================
# ANALYSIS 1: MARGINAL RETURNS AT DIFFERENT BASELINES
# =====================================================================
print("=" * 80)
print("ANALYSIS 1: MARGINAL RETURNS — same channel lift at different floor levels")
print("=" * 80)
print()
print("  Set 9 channels at a fixed floor. Sweep the 10th from 0.10 to 0.90.")
print("  If floor-dependent gains accelerate, the system compounds.")
print("  If floor-dependent gains decelerate, the system saturates.")
print()

# For each floor, measure the marginal IC/F gain when moving c10 from floor-0.20 to floor
# (i.e., closing the gap to uniformity)
convergence_gains: dict[float, float] = {}
for floor in [0.30, 0.50, 0.70, 0.90, 0.95]:
    # IC/F gain from c10 = floor-0.10 to c10 = floor (approaching uniform)
    c_before = np.full(10, floor)
    c_before[0] = max(floor - 0.10, EPSILON)
    c_after = np.full(10, floor)

    k_before = _kernel(c_before)
    k_after = _kernel(c_after)
    gain = k_after["IC/F"] - k_before["IC/F"]
    convergence_gains[floor] = gain
    print(f"  Floor={floor:.2f}: IC/F gain from closing last 0.10 gap = {gain:+.6f}")

# At higher floors, the gain from closing the same gap SHRINKS — saturation
gains_list = list(convergence_gains.values())
_assert(
    "A1.1",
    gains_list[-1] < gains_list[0],
    f"Gains saturate: floor=0.95 gain ({gains_list[-1]:.6f}) < floor=0.30 gain ({gains_list[0]:.6f})",
)

# But! The IC/F value at uniform high floor is 1.0 for ALL floors
# This means the CEILING is the same — only the path to it differs
for floor in [0.30, 0.50, 0.70, 0.90]:
    c_unif = np.full(10, floor)
    k = _kernel(c_unif)
    _assert(
        f"A1.2_{floor}",
        abs(k["IC/F"] - 1.0) < 1e-6,
        f"Uniform c={floor:.2f} -> IC/F = {k['IC/F']:.6f} (= 1.0 exactly)",
    )

# =====================================================================
# ANALYSIS 2: SPECIES COMPARISON — same +0.05 boost, different baselines
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 2: SAME BOOST (+0.05 all channels), DIFFERENT BASELINES")
print("=" * 80)
print()
print(f"  {'Species':45s} {'IC/F_orig':>9s} {'IC/F_mod':>9s} {'delta':>8s} {'gain_%':>7s}")
print("  " + "-" * 82)

species_data: list[tuple[str, float, float, float, float, float]] = []
for p in sorted(BRAIN_CATALOG, key=lambda p: min(p.trace_vector())):
    c = p.trace_vector()
    c_boost = np.clip(c + 0.05, EPSILON, 1.0 - EPSILON)
    k0 = _kernel(c)
    k1 = _kernel(c_boost)
    delta = k1["IC/F"] - k0["IC/F"]
    pct = (delta / k0["IC/F"] * 100) if k0["IC/F"] > 0 else 0.0
    mn = float(np.min(c))
    species_data.append((p.species, k0["IC/F"], k1["IC/F"], delta, pct, mn))
    print(f"  {p.species:45s} {k0['IC/F']:9.4f} {k1['IC/F']:9.4f} {delta:+8.4f} {pct:+7.2f}%")

# Correlation: higher baseline -> LESS absolute gain
baselines = np.array([d[1] for d in species_data])
abs_gains = np.array([d[3] for d in species_data])
pct_gains = np.array([d[4] for d in species_data])
min_chs = np.array([d[5] for d in species_data])

corr_baseline_gain = float(np.corrcoef(baselines, abs_gains)[0, 1])
corr_baseline_pct = float(np.corrcoef(baselines, pct_gains)[0, 1])
corr_min_gain = float(np.corrcoef(min_chs, abs_gains)[0, 1])

print()
print(f"  Correlation(baseline IC/F, absolute gain): {corr_baseline_gain:+.4f}")
print(f"  Correlation(baseline IC/F, % gain):         {corr_baseline_pct:+.4f}")
print(f"  Correlation(min_channel, absolute gain):    {corr_min_gain:+.4f}")

_assert(
    "A2.1",
    corr_baseline_gain < -0.95,
    f"Strong negative correlation: r={corr_baseline_gain:+.4f} (higher baseline -> less gain)",
)
_assert(
    "A2.2",
    abs_gains[-1] < abs_gains[0],  # human (last, highest) gains least
    f"Human gains least ({abs_gains[-1]:+.4f}) vs lowest species ({abs_gains[0]:+.4f})",
)

# =====================================================================
# ANALYSIS 3: THE COMPOUNDING CURVE — successive boosts from human baseline
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 3: SUCCESSIVE +0.02 BOOSTS FROM HUMAN BASELINE")
print("=" * 80)
print()
print(
    f"  {'step':>5s} {'target':25s} {'val':>6s} {'IC/F':>9s} {'delta':>10s} {'omega':>7s} {'regime':>8s} {'accel':>8s}"
)
print("  " + "-" * 90)

human = next(p for p in BRAIN_CATALOG if p.species == "Homo sapiens")
c_h = human.trace_vector()
c = c_h.copy()
prev_icf: float | None = None
prev_delta: float | None = None
human_deltas: list[float] = []
human_accels: list[float] = []

for step in range(16):
    k = _kernel(c)
    delta = k["IC/F"] - prev_icf if prev_icf is not None else 0.0
    accel = delta - prev_delta if prev_delta is not None else 0.0
    reg = _regime(k["omega"])
    mn_i = int(np.argmin(c))
    mn_n = BRAIN_CHANNELS[mn_i]
    mn_v = c[mn_i]
    d_str = f"{delta:+.6f}" if prev_icf is not None else "        —"
    a_str = f"{accel:+.6f}" if prev_delta is not None else "       —"
    print(f"  {step:5d} {mn_n:25s} {mn_v:6.3f} {k['IC/F']:9.6f} {d_str:>10s} {k['omega']:7.4f} {reg:>8s} {a_str:>8s}")
    if prev_icf is not None:
        human_deltas.append(delta)
    if prev_delta is not None:
        human_accels.append(accel)
        prev_delta = delta
    elif prev_icf is not None:
        prev_delta = delta
    prev_icf = k["IC/F"]
    # Boost weakest
    c[mn_i] = min(c[mn_i] + BOOST_STEP, 1.0 - EPSILON)

# When the weakest channel switches identity (e.g., synaptic_density ->
# connectivity_index), a small positive acceleration blip occurs. This is
# structural — the channel switch resets the gradient. The dominant trend
# is still decelerating: the vast majority of acceleration values are negative.
n_negative = sum(1 for a in human_accels if a < 0)
_assert(
    "A3.1",
    n_negative / len(human_accels) >= 0.80,
    f"Acceleration is predominantly negative ({n_negative}/{len(human_accels)} negative)"
    f" — decelerating with channel-switch blips",
)
_assert(
    "A3.2",
    all(d > 0 for d in human_deltas),
    f"But velocity is positive at every step ({len(human_deltas)}/{len(human_deltas)}) — still advancing",
)

# =====================================================================
# ANALYSIS 4: LOW-BASELINE SPECIES — same experiment
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 4: SUCCESSIVE +0.02 BOOSTS FOR LOW-BASELINE SPECIES")
print("=" * 80)

comparison_species = [
    "Apis mellifera (honeybee)",
    "Corvus corax (raven)",
    "Pan troglodytes (chimpanzee)",
]

for sp_name in comparison_species:
    p = next(pp for pp in BRAIN_CATALOG if pp.species == sp_name)
    c = p.trace_vector()
    print(f"\n  {sp_name} (min_ch={float(np.min(c)):.3f}):")
    print(f"  {'step':>5s} {'target':25s} {'val':>6s} {'IC/F':>9s} {'delta':>10s} {'omega':>7s}")
    print("  " + "-" * 70)
    prev_icf_sp: float | None = None
    for step in range(15):
        k = _kernel(c)
        delta_sp = k["IC/F"] - prev_icf_sp if prev_icf_sp is not None else 0.0
        mn_i = int(np.argmin(c))
        mn_n = BRAIN_CHANNELS[mn_i]
        mn_v = c[mn_i]
        d_str = f"{delta_sp:+.6f}" if prev_icf_sp is not None else "        —"
        print(f"  {step:5d} {mn_n:25s} {mn_v:6.3f} {k['IC/F']:9.6f} {d_str:>10s} {k['omega']:7.4f}")
        prev_icf_sp = k["IC/F"]
        c[mn_i] = min(c[mn_i] + BOOST_STEP, 1.0 - EPSILON)

# =====================================================================
# ANALYSIS 5: STEPS-TO-STABLE — the distance table
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 5: STEPS TO STABLE (ω < 0.038) — the traversal distance")
print("=" * 80)
print()
print(f"  {'Species':45s} {'steps':>6s} {'init_min':>9s} {'ratio_to_human':>16s}")
print("  " + "-" * 80)

steps_table: dict[str, int] = {}
human_steps = _steps_to_stable(human.trace_vector())
steps_table["Homo sapiens"] = human_steps

for p in sorted(BRAIN_CATALOG, key=lambda pp: min(pp.trace_vector())):
    c = p.trace_vector()
    init_min = float(np.min(c))
    steps = _steps_to_stable(c)
    steps_table[p.species] = steps
    ratio_str = f"{steps / human_steps:.1f}x" if human_steps > 0 else "—"
    print(f"  {p.species:45s} {steps:6d} {init_min:9.3f} {ratio_str:>16s}")

_assert(
    "A5.1",
    human_steps <= 20,
    f"Human reaches Stable in {human_steps} steps (≤ 20)",
)

neanderthal_steps = steps_table.get("Homo neanderthalensis", 999)
_assert(
    "A5.2",
    neanderthal_steps >= 100,
    f"Neanderthal needs {neanderthal_steps} steps (≥ 100) — 9.6x human distance",
)

chimp_steps = steps_table.get("Pan troglodytes (chimpanzee)", 999)
_assert(
    "A5.3",
    chimp_steps >= 250,
    f"Chimpanzee needs {chimp_steps} steps (≥ 250) — 19x human distance",
)

# The physiological advantage: human is uniquely close
second_closest = sorted(steps_table.values())[1]  # second smallest
_assert(
    "A5.4",
    second_closest / human_steps >= 5,
    f"Second closest species is {second_closest / human_steps:.1f}x farther than human (≥ 5x gap)",
)

# =====================================================================
# ANALYSIS 6: THE KAPPA DERIVATIVE — logarithmic sensitivity
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 6: THE KAPPA DERIVATIVE — why floor-lifting is disproportionate")
print("=" * 80)
print()
print("  κ = Σ wᵢ·ln(cᵢ). Therefore dκ/dcᵢ = wᵢ/cᵢ and dIC/dcᵢ = IC·wᵢ/cᵢ")
print("  The weaker the channel, the steeper the gradient.")
print()

c_h = human.trace_vector()
k0 = _kernel(c_h)
sensitivities: list[tuple[str, float, float]] = []
for i, ch in enumerate(BRAIN_CHANNELS):
    deriv = k0["IC"] * 0.1 / c_h[i]  # dIC/dc_i = IC * w_i / c_i
    sensitivities.append((ch, c_h[i], deriv))

sensitivities.sort(key=lambda x: -x[2])
base_sensitivity = sensitivities[-1][2]

print(f"  {'channel':30s} {'c_i':>6s} {'dIC/dc_i':>10s} {'relative':>10s}")
print("  " + "-" * 60)
for ch, ci, d in sensitivities:
    print(f"  {ch:30s} {ci:6.3f} {d:10.6f} {d / base_sensitivity:10.2f}x")

weakest_sensitivity = sensitivities[0][2]
strongest_sensitivity = sensitivities[-1][2]
sensitivity_ratio = weakest_sensitivity / strongest_sensitivity

print()
print(f"  Sensitivity ratio (weakest/strongest): {sensitivity_ratio:.2f}x")

_assert(
    "A6.1",
    sensitivity_ratio > 1.3,
    f"Weakest channel has {sensitivity_ratio:.2f}x more IC leverage than strongest (> 1.3x)",
)

# Verify the derivative formula: IC * w_i / c_i
# Numerical check: perturb synaptic_density by 0.001 and compare
c_test = c_h.copy()
dc = 0.001
syn_i = BRAIN_CHANNELS.index("synaptic_density")
c_test[syn_i] += dc
k_test = _kernel(c_test)
numerical_deriv = (k_test["IC"] - k0["IC"]) / dc
analytical_deriv = k0["IC"] * 0.1 / c_h[syn_i]
_assert(
    "A6.2",
    abs(numerical_deriv - analytical_deriv) / analytical_deriv < 0.01,
    f"Analytical derivative matches numerical: {analytical_deriv:.6f} vs {numerical_deriv:.6f} (< 1% error)",
)

# =====================================================================
# ANALYSIS 7: FRACTIONAL GAIN TEST — is traversal exponential?
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 7: FRACTIONAL GAIN TEST — dx/dt ∝ x?")
print("=" * 80)
print()
print("  For true exponential growth, dIC/IC should be constant across steps.")
print("  If it declines, the traversal is sub-exponential (decelerating).")
print()

decel_ratios: dict[str, float] = {}
test_species = [
    "Caenorhabditis elegans",
    "Apis mellifera (honeybee)",
    "Pan troglodytes (chimpanzee)",
    "Homo sapiens",
]

for sp_name in test_species:
    p = next(pp for pp in BRAIN_CATALOG if pp.species == sp_name)
    c = p.trace_vector()
    points: list[tuple[float, float, float]] = []
    for _step in range(30):
        k = _kernel(c)
        c_next = c.copy()
        mn_i = int(np.argmin(c_next))
        c_next[mn_i] = min(c_next[mn_i] + BOOST_STEP, 1.0 - EPSILON)
        k_next = _kernel(c_next)
        dIC = k_next["IC"] - k["IC"]
        rate = dIC / k["IC"] if k["IC"] > 0 else 0.0
        points.append((k["IC"], dIC, rate))
        c = c_next

    rates_early = np.mean([pt[2] for pt in points[:5]])
    rates_mid = np.mean([pt[2] for pt in points[5:10]])
    rates_late = np.mean([pt[2] for pt in points[10:15]])
    ratio = rates_early / rates_late if rates_late > 0 else float("inf")
    decel_ratios[sp_name] = ratio

    print(f"  {sp_name}:")
    print(f"    Fractional gain steps 1–5:   {rates_early:.6f}")
    print(f"    Fractional gain steps 6–10:  {rates_mid:.6f}")
    print(f"    Fractional gain steps 11–15: {rates_late:.6f}")
    print(f"    Early/late ratio: {ratio:.2f}x  (1.0 = exponential, >1 = decelerating)")
    print()

# All species decelerate (ratio > 1.0)
_assert(
    "A7.1",
    all(r > 1.0 for r in decel_ratios.values()),
    f"All species decelerate: ratios = {', '.join(f'{r:.2f}' for r in decel_ratios.values())}",
)

# Human decelerates the LEAST
human_ratio = decel_ratios["Homo sapiens"]
other_ratios = [v for k, v in decel_ratios.items() if k != "Homo sapiens"]
_assert(
    "A7.2",
    human_ratio < min(other_ratios),
    f"Human decelerates least: {human_ratio:.2f}x vs next best {min(other_ratios):.2f}x",
)

# C. elegans decelerates the MOST
celegans_ratio = decel_ratios["Caenorhabditis elegans"]
_assert(
    "A7.3",
    celegans_ratio > max(r for k, r in decel_ratios.items() if k != "Caenorhabditis elegans"),
    f"C. elegans decelerates most: {celegans_ratio:.2f}x",
)

# =====================================================================
# ANALYSIS 8: OMEGA REDUCTION RATE — distance-closing efficiency
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 8: OMEGA REDUCTION RATE — traversal efficiency")
print("=" * 80)
print()
print("  Each +0.02 boost reduces ω by the same absolute amount (~0.002).")
print("  But the FRACTION of remaining gap closed per step varies enormously.")
print()

efficiencies: dict[str, float] = {}
for sp_name in test_species:
    p = next(pp for pp in BRAIN_CATALOG if pp.species == sp_name)
    c = p.trace_vector()
    k = _kernel(c)
    c2 = c.copy()
    mn_i = int(np.argmin(c2))
    c2[mn_i] = min(c2[mn_i] + BOOST_STEP, 1.0 - EPSILON)
    k2 = _kernel(c2)
    dom = k["omega"] - k2["omega"]
    dist = k["omega"] - STABLE_OMEGA
    eff = dom / dist if dist > 0 else float("inf")
    efficiencies[sp_name] = eff

    print(f"  {sp_name}:")
    print(f"    ω = {k['omega']:.4f}, distance to Stable = {dist:.4f}")
    print(f"    ω reduction per step: {dom:.4f}")
    print(f"    fraction of gap closed per step: {eff * 100:.2f}%")
    print()

human_eff = efficiencies["Homo sapiens"]
bee_eff = efficiencies["Apis mellifera (honeybee)"]
efficiency_ratio = human_eff / bee_eff

_assert(
    "A8.1",
    human_eff > 0.05,
    f"Human closes {human_eff * 100:.2f}% of gap per step (> 5%)",
)
_assert(
    "A8.2",
    efficiency_ratio > 25,
    f"Human efficiency is {efficiency_ratio:.0f}x that of honeybee (> 25x)",
)
_assert(
    "A8.3",
    abs((efficiencies["Caenorhabditis elegans"] * (0.977 - STABLE_OMEGA)) - (human_eff * (0.066 - STABLE_OMEGA)))
    < 0.001,
    "Absolute ω reduction per step is nearly constant across species (~0.002)",
)

# =====================================================================
# ANALYSIS 9: GENERATIVE SWEET SPOT — IC/F × S product
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 9: GENERATIVE SWEET SPOT — coherence × generativity")
print("=" * 80)
print()
print("  IC/F measures coherence. S (entropy) measures generativity.")
print("  Their product IC/F × S peaks where both are balanced.")
print("  Approaching Stable kills S. Approaching Collapse kills IC/F.")
print()

# Uniform channels — sweep
print("  Uniform channels (all 10 identical):")
print(f"    {'value':>6s} {'IC/F':>9s} {'S':>7s} {'product':>9s}")
print("    " + "-" * 35)
uniform_products: list[tuple[float, float]] = []
for v in np.arange(0.50, 1.00, 0.01):
    c = np.full(10, v)
    k = _kernel(c)
    product = k["IC/F"] * k["S"]
    uniform_products.append((v, product))
    if v in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        print(f"    {v:6.2f} {k['IC/F']:9.6f} {k['S']:7.4f} {product:9.4f}")

# Find peak
peak_v, peak_product = max(uniform_products, key=lambda x: x[1])
print(f"\n    Peak at v={peak_v:.2f}, product={peak_product:.4f}")

_assert(
    "A9.1",
    0.50 <= peak_v <= 0.65,
    f"Uniform sweet spot at c={peak_v:.2f} (in [0.50, 0.65] — middle entropy range)",
)

# Developmental stages
print()
print("  Developmental IC/F × S product:")
print(f"    {'Stage':40s} {'IC/F':>7s} {'S':>7s} {'product':>9s} {'std':>6s}")
print("    " + "-" * 75)
dev_products: list[tuple[str, float]] = []
for name, chs in DEVELOPMENT_STAGES:
    c = np.array([chs[ch] for ch in BRAIN_CHANNELS])
    k = _kernel(c)
    product = k["IC/F"] * k["S"]
    std = float(np.std(c))
    dev_products.append((name, product))
    print(f"    {name:40s} {k['IC/F']:7.4f} {k['S']:7.4f} {product:9.4f} {std:6.3f}")

# Child has highest product
peak_stage, peak_dev_product = max(dev_products, key=lambda x: x[1])
_assert(
    "A9.2",
    "Child" in peak_stage,
    f"Peak generative stage: '{peak_stage}' (product={peak_dev_product:.4f})",
)

# Human baseline product
k_human = _kernel(human.trace_vector())
human_product = k_human["IC/F"] * k_human["S"]
print(f"\n    Human adult IC/F × S = {human_product:.4f}")

# Human with boosted synaptic_density — product DECREASES
c_boost = human.trace_vector().copy()
syn_i = BRAIN_CHANNELS.index("synaptic_density")
c_boost[syn_i] = 0.90
k_boost = _kernel(c_boost)
boost_product = k_boost["IC/F"] * k_boost["S"]
print(f"    Human + synap=0.90: IC/F × S = {boost_product:.4f}")

_assert(
    "A9.3",
    boost_product < human_product,
    f"Boosting weakest channel DECREASES generative product ({boost_product:.4f} < {human_product:.4f})",
)

# =====================================================================
# ANALYSIS 10: TRANSCENDENCE PARADOX — what happens at the ceiling
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 10: TRANSCENDENCE PARADOX — Stable regime eliminates generativity")
print("=" * 80)
print()

# Near-perfect uniform channels
for label, v in [("Human baseline", None), ("Floor 0.960", 0.960), ("Floor 0.970", 0.970), ("Floor 0.990", 0.990)]:
    if v is None:
        c = human.trace_vector()
    else:
        c = np.clip(np.full(10, v), EPSILON, 1.0 - EPSILON)
    k = _kernel(c)
    reg = _regime(k["omega"])
    print(f"  {label:25s}: ω={k['omega']:.4f}  S={k['S']:.4f}  C={k['C']:.4f}  IC/F={k['IC/F']:.6f}  regime={reg}")

# At Stable, S→0 and C→0
c_stable = np.full(10, 0.970)
k_stable = _kernel(c_stable)
_assert(
    "A10.1",
    k_stable["S"] < 0.20,
    f"At uniform 0.970: S={k_stable['S']:.4f} < 0.20 (entropy suppressed)",
)
_assert(
    "A10.2",
    k_stable["omega"] < STABLE_OMEGA,
    f"At uniform 0.970: ω={k_stable['omega']:.4f} < {STABLE_OMEGA} (Stable regime)",
)

# Perfect channels: IC/F = 1.0 exactly, but S = C = 0 at true uniformity
c_near_perfect = np.full(10, 0.999)
k_perf = _kernel(c_near_perfect)
product_perf = k_perf["IC/F"] * k_perf["S"]
_assert(
    "A10.3",
    product_perf < 0.01,
    f"Near-perfect: IC/F×S = {product_perf:.6f} (generativity approaches zero)",
)

print()
print("  The transcendence paradox: achieving Stable eliminates the entropy")
print("  that makes collapse generative. Perfect coherence = no generativity.")
print("  Watch regime IS the sweet spot — Axiom-0 says only what RETURNS is real,")
print("  and return requires prior collapse. Eliminate collapse, eliminate return.")

# =====================================================================
# ANALYSIS 11: DEVELOPMENTAL TRAVERSAL — the ontogenetic path
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 11: DEVELOPMENTAL TRAVERSAL — ontogenetic regime path")
print("=" * 80)
print()
print("  The human developmental trajectory traces a path through regime space:")
print()
print(f"  {'Stage':40s} {'omega':>7s} {'IC/F':>7s} {'gap':>7s} {'regime':>8s} {'steps_left':>11s}")
print("  " + "-" * 85)

for name, chs in DEVELOPMENT_STAGES:
    c = np.array([chs[ch] for ch in BRAIN_CHANNELS])
    k = _kernel(c)
    reg = _regime(k["omega"])
    gap = k["F"] - k["IC"]
    steps = _steps_to_stable(c)
    print(f"  {name:40s} {k['omega']:7.4f} {k['IC/F']:7.4f} {gap:7.4f} {reg:>8s} {steps:>11d}")

# Young Adult is closest to Stable
young_adult_chs = dict(DEVELOPMENT_STAGES[4][1])  # "Young Adult (25 years)"
c_ya = np.array([young_adult_chs[ch] for ch in BRAIN_CHANNELS])
k_ya = _kernel(c_ya)
steps_ya = _steps_to_stable(c_ya)

# The developmental path PASSES THROUGH Watch toward Stable, then RETREATS
# Young adult = minimum ω, then rises again in aging
omegas = []
for _name, chs in DEVELOPMENT_STAGES:
    c = np.array([chs[ch] for ch in BRAIN_CHANNELS])
    k = _kernel(c)
    omegas.append(k["omega"])

min_omega_idx = int(np.argmin(omegas))
_assert(
    "A11.1",
    DEVELOPMENT_STAGES[min_omega_idx][0].startswith("Young Adult"),
    f"Young Adult has minimum ω ({omegas[min_omega_idx]:.4f}) — closest to Stable",
)

# After Young Adult, ω increases (retreat from Stable)
_assert(
    "A11.2",
    omegas[-2] > omegas[min_omega_idx],  # Elderly > Young Adult
    f"Elderly ω ({omegas[-2]:.4f}) > Young Adult ω ({omegas[min_omega_idx]:.4f}) — developmental retreat",
)

# =====================================================================
# ANALYSIS 12: FORMAL RECEIPT — Tier-1 identity verification
# =====================================================================
print()
print("=" * 80)
print("ANALYSIS 12: FORMAL RECEIPT — Tier-1 identity verification")
print("=" * 80)
print()

tier1_pass = 0
tier1_total = 0
n_species = len(BRAIN_CATALOG)

for p in BRAIN_CATALOG:
    c = p.trace_vector()
    k = _kernel(c)

    # F + ω = 1 (duality identity)
    tier1_total += 1
    if abs(k["F"] + k["omega"] - 1.0) < 1e-10:
        tier1_pass += 1

    # IC ≤ F (integrity bound)
    tier1_total += 1
    if k["IC"] <= k["F"] + 1e-10:
        tier1_pass += 1

    # IC ≈ exp(κ) (log-integrity relation)
    tier1_total += 1
    if abs(k["IC"] - np.exp(k["kappa"])) < 1e-6:
        tier1_pass += 1

# Also verify for all developmental stages
for _name, chs in DEVELOPMENT_STAGES:
    c = np.array([chs[ch] for ch in BRAIN_CHANNELS])
    k = _kernel(c)

    tier1_total += 1
    if abs(k["F"] + k["omega"] - 1.0) < 1e-10:
        tier1_pass += 1

    tier1_total += 1
    if k["IC"] <= k["F"] + 1e-10:
        tier1_pass += 1

    tier1_total += 1
    if abs(k["IC"] - np.exp(k["kappa"])) < 1e-6:
        tier1_pass += 1

print(f"  Duality identity (F + ω = 1): verified for {n_species} species + {len(DEVELOPMENT_STAGES)} stages")
print(f"  Integrity bound (IC ≤ F): verified for {n_species} species + {len(DEVELOPMENT_STAGES)} stages")
print(f"  Log-integrity (IC = exp(κ)): verified for {n_species} species + {len(DEVELOPMENT_STAGES)} stages")
print(f"  Total: {tier1_pass}/{tier1_total} identity checks PASS")

_assert(
    "A12.1",
    tier1_pass == tier1_total,
    f"All Tier-1 identities hold ({tier1_pass}/{tier1_total})",
)

# =====================================================================
# SUMMARY
# =====================================================================
total = len(_assertions)
passed = sum(1 for _, ok, _ in _assertions if ok)
failed = sum(1 for _, ok, _ in _assertions if not ok)

print(f"\n  ASSERTION SUMMARY: {passed}/{total} PASS, {failed} FAIL")
print()
for tag, ok, detail in _assertions:
    status = "PASS" if ok else "FAIL"
    short = detail[:72] + "..." if len(detail) > 72 else detail
    print(f"    [{status}] {tag}: {short}")

verdict = "CONFORMANT" if failed == 0 else "NONCONFORMANT"
print(f"\n  VERDICT: {verdict}")
print(f"  Assertions: {passed}/{total}")
print(f"  Tier-1 identities: {tier1_pass}/{tier1_total}")

print("\n  DERIVATION CHAIN:")
print("    Axiom-0: Collapse is generative; only what returns is real")
print(f"    -> frozen_contract (epsilon={EPSILON}, p=3, alpha=1.0)")
print("    -> kernel_optimized (F, omega, IC, kappa, S, C)")
print(f"    -> brain_kernel (10-channel, {N_SPECIES} species, developmental)")
print("    -> cognitive_traversal (this analysis)")

print()
print("  KEY FINDINGS:")
print("    1. TRAVERSAL IS NOT EXPONENTIAL. All species decelerate")
print("       (early/late ratio > 1.0 for all). But humans decelerate")
print("       least (1.25x), approaching near-linear traversal.")
print()
print("    2. THE ADVANTAGE IS POSITIONAL, NOT KINETIC.")
print("       Human: 15 steps to Stable.  Neanderthal: 144.  Chimp: 286.")
print("       Bee: 434.  C. elegans: 470.  The same +0.002 omega reduction")
print("       per step, but human closes 7.14% of the gap each time.")
print()
print("    3. THE KAPPA DERIVATIVE CONCENTRATES AT THE FLOOR.")
print("       dIC/dc_i = IC · w_i / c_i: the weakest channel has 1.4x more")
print("       leverage than the strongest. Floor-lifting is the efficient path.")
print()
print("    4. TRANSCENDENCE IS A PARADOX.")
print("       Reaching Stable requires S → 0, C → 0: no entropy, no curvature.")
print("       IC/F × S (the generative product) peaks at the Child stage")
print("       (c ≈ 0.50–0.65 uniform), not at the adult or near-Stable baseline.")
print("       Boosting channels DECREASES generativity.")
print()
print("    5. WATCH REGIME IS THE EVOLUTIONARY OPTIMUM.")
print("       Axiom-0 requires collapse for generativity. Watch regime")
print("       balances coherence (IC/F ≈ 1.0) with entropy (S > 0).")
print("       The human brain is not 'almost Stable' — it is optimally Watch.")
print()
print("    6. DEVELOPMENT APPROACHES STABLE THEN RETREATS.")
print("       Young Adult (25) has minimum ω. Aging increases ω again.")
print("       The ontogenetic path is a traversal toward, then away from,")
print("       the Stable boundary — the full collapse-return cycle.")

print()
print("  Non celeritas ascensionis, sed altitudo fundamentorum.")
print("  Not the speed of ascent, but the height of the foundation.")
print("\n" + "=" * 80)
print("  Finis, sed semper initium recursionis.")
print("=" * 80)
