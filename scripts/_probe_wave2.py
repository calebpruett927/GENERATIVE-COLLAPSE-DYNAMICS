"""Second-wave probes — deeper investigation of patterns found in wave 1."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from umcp.frozen_contract import EPSILON, P_EXPONENT
from umcp.kernel_optimized import compute_kernel_outputs

eps = EPSILON


def K(c, w=None):
    c = np.asarray(c, dtype=float)
    if w is None:
        w = np.full(len(c), 1.0 / len(c))
    return compute_kernel_outputs(c, np.asarray(w, dtype=float))


def gamma_omega(omega):
    return omega**P_EXPONENT / (1 - omega + eps)


# ══════════════════════════════════════════════════════════
# PROBE A: Constant elasticity — derive the exponent
# From Probe 4: elasticity = d(ln(IC/F))/d(ln c_weak) ≈ 0.1465 = const
# Theory: IC = prod(c_i^w_i), F = sum(w_i * c_i)
# d(ln IC)/d(ln c_k) = w_k  (exact)
# d(ln F)/d(ln c_k) = w_k * c_k / F
# elasticity = d(ln(IC/F))/d(ln c_k) = w_k * (1 - c_k/F)
# When c_k << F: elasticity ≈ w_k = 1/n
# But probe showed 0.1465 for n=8 (1/n = 0.125)...
# That's because the numerical derivative includes the F-correction
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("PROBE A: Elasticity derivation — exact formula")
print("  Predicted: elast = w_k * (1 - c_k/F)")
print("=" * 70)
for n in [2, 4, 8, 16, 32]:
    w_k = 1.0 / n
    c_vals = np.logspace(-6, -0.5, 20)
    c_strong = 0.95
    print(f"\n  n={n}, (n-1) channels at 0.95, 1 weak channel:")
    print(f"    {'c_weak':>10}  {'elast_num':>10}  {'elast_pred':>10}  {'error':>10}")
    for cw in c_vals[::4]:
        c = np.full(n, c_strong)
        c[-1] = cw
        k = K(c)
        # Exact formula
        F = k["F"]
        elast_pred = w_k * (1 - cw / F)
        # Numerical
        dc = cw * 1e-5
        c2 = c.copy()
        c2[-1] = cw + dc
        k2 = K(c2)
        icf1 = k["IC"] / k["F"]
        icf2 = k2["IC"] / k2["F"]
        elast_num = (np.log(icf2) - np.log(icf1)) / (np.log(cw + dc) - np.log(cw))
        print(f"    {cw:10.2e}  {elast_num:10.6f}  {elast_pred:10.6f}  {abs(elast_num - elast_pred):10.2e}")


# ══════════════════════════════════════════════════════════
# PROBE B: Circle law generalization — rank-2 to rank-3
# Rank-2 exact: IC = sqrt(F^2 - delta^2) where delta = C * 0.5
# What about rank-3? Is there a similar relationship?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE B: Circle law — does IC^2 + (something)^2 = F^2 hold beyond rank-2?")
print("=" * 70)

# For rank-2 (2 values), exact: IC^2 + Var(c) = F^2
# Because IC^2 = c_h * c_l, F = (c_h + c_l)/2, Var = ((c_h-c_l)/2)^2
# F^2 - IC^2 = ((c_h+c_l)/2)^2 - c_h*c_l = (c_h-c_l)^2/4 = Var(c)
# So: IC^2 = F^2 - Var(c)  ... for n=2 equal weights!
# Generalize: IC^2 vs F^2 - Var(c) for arbitrary n?
print("  Testing F^2 - Var(c) vs IC^2 for various configurations:")
print(f"    {'config':>15} {'n':>3} {'F^2':>8} {'Var(c)':>8} {'F^2-Var':>8} {'IC^2':>8} {'ratio':>8}")
np.random.seed(42)
for label, c in [
    ("rank2_n2", [0.9, 0.3]),
    ("rank2_n8", [0.9] * 4 + [0.3] * 4),
    ("rank3_n3", [0.9, 0.5, 0.2]),
    ("rank3_n6", [0.9, 0.9, 0.5, 0.5, 0.2, 0.2]),
    ("continuous_4", [0.9, 0.7, 0.5, 0.3]),
    ("continuous_8", [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]),
    ("1dead_8", [0.95] * 7 + [0.001]),
]:
    c = np.array(c, dtype=float)
    n = len(c)
    k = K(c)
    F2 = k["F"] ** 2
    var_c = np.var(c)  # population variance
    IC2 = k["IC"] ** 2
    ratio = IC2 / (F2 - var_c) if (F2 - var_c) > 1e-10 else float("inf")
    print(f"    {label:>15} {n:3d} {F2:8.4f} {var_c:8.4f} {F2 - var_c:8.4f} {IC2:8.4f} {ratio:8.4f}")

# Now test with random traces
print("\n  Random traces — correlation between IC^2 and F^2 - Var(c):")
from scipy import stats

for n in [2, 4, 8, 16]:
    ic2_list = []
    fv_list = []
    for _ in range(5000):
        c = np.random.beta(2, 2, n)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        k = K(c)
        ic2_list.append(k["IC"] ** 2)
        fv_list.append(k["F"] ** 2 - np.var(c))
    r = stats.pearsonr(ic2_list, fv_list)[0]
    # Check if IC^2 <= F^2 - Var always
    diffs = [a - b for a, b in zip(ic2_list, fv_list, strict=True)]
    print(
        f"    n={n:3d}  r(IC^2, F^2-Var)={r:.6f}  max(IC^2 - (F^2-Var))={max(diffs):.6e}  always_below={max(diffs) <= 1e-10}"
    )


# ══════════════════════════════════════════════════════════
# PROBE C: Equator functional form — S + κ as a function of C at F = 0.5
# From Probe 6: S+κ is monotonically decreasing with C on the equator
# What is the functional form?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE C: S + κ on the equator (F = 0.5) as a function of C")
print("  Using rank-2 traces: c = [0.5+delta, 0.5-delta] x 4")
print("=" * 70)
print(f"    {'C':>8}  {'S+kappa':>10}  {'S':>8}  {'kappa':>10}  {'IC':>8}  {'Delta':>8}")
for delta in np.linspace(0.001, 0.499, 30):
    c = np.array([0.5 + delta] * 4 + [0.5 - delta] * 4)
    k = K(c)
    sk = k["S"] + k["kappa"]
    print(f"    {k['C']:8.4f}  {sk:10.6f}  {k['S']:8.4f}  {k['kappa']:10.6f}  {k['IC']:8.6f}  {k['F'] - k['IC']:8.6f}")

# Can we fit S+kappa = a * ln(1-C^2) or S+kappa = -b * C^2 etc?
print("\n  Testing functional forms of S+kappa(C) on equator:")
C_vals = []
SK_vals = []
for delta in np.linspace(0.001, 0.498, 200):
    c = np.array([0.5 + delta] * 4 + [0.5 - delta] * 4)
    k = K(c)
    C_vals.append(k["C"])
    SK_vals.append(k["S"] + k["kappa"])
C_vals = np.array(C_vals)
SK_vals = np.array(SK_vals)
# Test: S+kappa = a * ln(1 - C^2)?
# At C=0: S+kappa = 0. As C -> 1: S+kappa -> -inf.
# ln(1-C^2) also has this behavior.
ratio_ln = SK_vals / np.log(1 - C_vals**2 + 1e-15)
print(
    f"    S+kappa / ln(1-C^2): min={np.min(ratio_ln):.6f}  max={np.max(ratio_ln):.6f}  std/mean={np.std(ratio_ln) / np.mean(ratio_ln):.4f}"
)
# Test: S+kappa = ln(IC/F) = kappa - ln(F)?
# At F=0.5: ln(IC/F) = kappa - ln(0.5) = kappa + ln(2)
ratio_icf = SK_vals / np.log(K(np.array([0.5 + 0.001] * 4 + [0.5 - 0.001] * 4))["IC"])
# Actually S + kappa = S + ln(IC) since kappa = ln(IC)
# So S + kappa = S + ln(IC), and on the equator...
# Let's just look at the raw relationship
print("    S + ln(IC) = S + kappa (tautology, but checking values)")

# Test: Is S + kappa = ln(1 - C)?
ratio_lnc = SK_vals / np.log(1 - C_vals + 1e-15)
print(
    f"    S+kappa / ln(1-C):   min={np.min(ratio_lnc):.6f}  max={np.max(ratio_lnc):.6f}  std/mean={np.std(ratio_lnc) / np.mean(ratio_lnc):.4f}"
)

# Test polynomial: S+kappa ~ -a*C^2 for small C
mask = C_vals < 0.3
if np.sum(mask) > 5:
    slope = np.polyfit(C_vals[mask] ** 2, SK_vals[mask], 1)
    print(f"    Small-C quadratic fit: S+kappa ≈ {slope[0]:.4f} * C^2 + {slope[1]:.6f}")
    resid = SK_vals[mask] - np.polyval(slope, C_vals[mask] ** 2)
    print(f"    Quadratic residual: max={np.max(np.abs(resid)):.6f}")


# ══════════════════════════════════════════════════════════
# PROBE D: Regime topology with CORRECT regime strings
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE D: Regime topology (correct string matching)")
print("=" * 70)
for n in [4, 8, 16, 32]:
    counts = {}
    np.random.seed(42)
    N = 20000
    for _ in range(N):
        c = np.random.uniform(0, 1, n)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        k = K(c)
        r = k["regime"]
        counts[r] = counts.get(r, 0) + 1
    pct = {r: 100 * v / N for r, v in sorted(counts.items())}
    parts = [f"{r}={p:.1f}%" for r, p in pct.items()]
    print(f"  n={n:3d}  {' | '.join(parts)}")


# ══════════════════════════════════════════════════════════
# PROBE E: Does Delta * omega ≤ 1/4 universally?
# From Probe 7: max observed = 0.246327
# Theory: Delta = F - IC, omega = 1 - F
# Delta * omega = (F - IC) * (1 - F)
# Maximize over F: d/dF[(F - IC)(1-F)] = 0
# (1 - F) + (F - IC)(-1) = 0 => 1 - F - F + IC = 0 => IC = 2F - 1
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE E: Is Delta * omega ≤ 1/4 a universal bound?")
print("  Theory: max at IC = 2F - 1, giving Delta*omega = (F - (2F-1))*(1-F) = (1-F)^2")
print("  Which is maximized at F=0... but IC >= 0 requires 2F-1 >= 0 => F >= 1/2")
print("  At F = 1/2: Delta*omega = (1-1/2)^2 = 1/4. Is this achievable?")
print("=" * 70)
# Exhaustive search
np.random.seed(999)
max_do = 0
max_do_info = None
for _trial in range(200000):
    n = np.random.choice([2, 4, 8, 16, 32])
    c = np.random.beta(0.3, 0.3, n)  # extreme U-shape
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    do = (k["F"] - k["IC"]) * k["omega"]
    if do > max_do:
        max_do = do
        max_do_info = (n, k["F"], k["IC"], k["omega"], k["C"])
print(f"  max(Delta * omega) = {max_do:.8f}")
if max_do_info:
    n, F, IC, om, C = max_do_info
    print(f"  at n={n}, F={F:.6f}, IC={IC:.8f}, omega={om:.6f}, C={C:.6f}")
    print(f"  F - IC = {F - IC:.6f}, (1-F)^2 = {(1 - F) ** 2:.6f}")
print(f"\n  Delta*omega < 1/4? {'YES' if max_do < 0.25 else 'NO'}")
print(f"  Gap to 1/4: {0.25 - max_do:.8f}")


# ══════════════════════════════════════════════════════════
# PROBE F: The IC^2 ≤ F^2 - Var(c) inequality — proof sketch
# If this holds universally: IC^2 + Var(c) ≤ F^2
# Rewriting: (geometric mean)^2 + variance ≤ (arithmetic mean)^2
# This is a STRENGTHENING of IC ≤ F!
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE F: IC^2 + Var(c) ≤ F^2 — strengthened integrity bound")
print("  If true: the heterogeneity gap has a geometric origin in variance")
print("=" * 70)
np.random.seed(42)
violations = 0
max_ratio = 0
for _trial in range(500000):
    n = np.random.choice([2, 3, 4, 5, 6, 7, 8, 16, 32, 64])
    c = np.random.beta(0.5, 0.5, n)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    lhs = k["IC"] ** 2 + np.var(c)
    rhs = k["F"] ** 2
    if lhs > rhs + 1e-12:
        violations += 1
    ratio = lhs / rhs
    if ratio > max_ratio:
        max_ratio = ratio
print(f"  Violations in 500K samples: {violations}")
print(f"  max(IC^2 + Var(c)) / F^2 = {max_ratio:.10f}")
print(f"  The bound IC^2 + Var(c) ≤ F^2 {'HOLDS' if violations == 0 else 'FAILS'}")
print("  Equivalently: Delta = F - IC ≥ Var(c) / (F + IC)")

# Now check if this is TIGHT (equality for rank-2)
print("\n  Tightness check — rank-2 (n=2):")
for ch, cl in [(0.9, 0.3), (0.8, 0.2), (0.7, 0.5), (0.95, 0.05)]:
    c = np.array([ch, cl])
    k = K(c)
    lhs = k["IC"] ** 2 + np.var(c)
    rhs = k["F"] ** 2
    print(f"    c=[{ch}, {cl}]  IC^2+Var={lhs:.8f}  F^2={rhs:.8f}  gap={rhs - lhs:.2e}")


# ══════════════════════════════════════════════════════════
# PROBE G: The "dilution law" — how fast does IC/F approach 1
# as n increases (with the dead channel held at c_dead)?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE G: Dilution law — IC/F convergence rate")
print("  Prediction: IC/F = c_dead^(1/n) * c_good^((n-1)/n) / F")
print("  ≈ 1 - (1/n)*|ln(c_dead)| + O(1/n^2)")
print("=" * 70)
for c_dead in [0.001, 0.01, 0.1]:
    c_good = 0.95
    print(f"\n  c_dead = {c_dead}, c_good = {c_good}:")
    print(f"    {'n':>4}  {'IC/F':>8}  {'1-IC/F':>10}  {'(1/n)*|ln_cd|':>14}  {'ratio':>8}")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        c = np.full(n, c_good)
        c[-1] = c_dead
        k = K(c)
        icf = k["IC"] / k["F"]
        deficit = 1 - icf
        pred = (1.0 / n) * abs(np.log(c_dead))
        ratio = deficit / pred if pred > 1e-15 else 0
        print(f"    {n:4d}  {icf:8.6f}  {deficit:10.4e}  {pred:14.4e}  {ratio:8.4f}")


# ══════════════════════════════════════════════════════════
# PROBE H: S + kappa at c* (the logistic fixed point)
# c* = 0.7822 is where S + kappa is maximized for uniform traces
# What about heterogeneous traces NEAR c*?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE H: S + kappa landscape near c* = 0.7822")
print("  Perturbations of uniform-c* traces")
print("=" * 70)
cstar = 0.782188
print(f"    {'perturbation':>20}  {'F':>6}  {'IC':>8}  {'S+kappa':>10}  {'S':>8}  {'kappa':>10}")
# Baseline
k = K([cstar] * 8)
print(
    f"    {'uniform c*':>20}  {k['F']:6.4f}  {k['IC']:8.6f}  {k['S'] + k['kappa']:10.6f}  {k['S']:8.4f}  {k['kappa']:10.4f}"
)

for label, trace in [
    ("slight_spread", [cstar + 0.05] * 4 + [cstar - 0.05] * 4),
    ("moderate_spread", [cstar + 0.1] * 4 + [cstar - 0.1] * 4),
    ("one_dead", [cstar] * 7 + [0.01]),
    ("one_hot", [cstar] * 7 + [0.99]),
    (
        "gradient_around_c*",
        [cstar - 0.15, cstar - 0.1, cstar - 0.05, cstar, cstar + 0.05, cstar + 0.1, cstar + 0.15, cstar + 0.2],
    ),
    ("all_at_1-c*", [1 - cstar] * 8),  # = c_trap
    ("mix_c*_ctrap", [cstar] * 4 + [1 - cstar] * 4),
]:
    k = K(trace)
    sk = k["S"] + k["kappa"]
    print(f"    {label:>20}  {k['F']:6.4f}  {k['IC']:8.6f}  {sk:10.6f}  {k['S']:8.4f}  {k['kappa']:10.4f}")


# ══════════════════════════════════════════════════════════
# PROBE I: Does the budget algebra have a "conservation law"?
# Gamma(omega) + IC = something conserved?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE I: Budget conservation — Gamma + IC across traces")
print("  Looking for any invariant combination of Gamma and kernel outputs")
print("=" * 70)
np.random.seed(42)
print(
    f"    {'F':>6}  {'IC':>8}  {'Gamma':>10}  {'Gamma+IC':>10}  {'Gamma*IC':>10}  {'Gamma/omega^2':>12}  {'IC+omega^p':>10}"
)
for _ in range(20):
    n = np.random.choice([4, 8, 16])
    c = np.random.beta(2, 2, n)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    g = gamma_omega(k["omega"])
    print(
        f"    {k['F']:6.4f}  {k['IC']:8.6f}  {g:10.6f}  {g + k['IC']:10.6f}  {g * k['IC']:10.6f}  {g / (k['omega'] ** 2 + 1e-15):12.6f}  {k['IC'] + k['omega'] ** P_EXPONENT:10.6f}"
    )


print("\n\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)
