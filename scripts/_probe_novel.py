"""Novel pattern probes — temporary exploration script."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from umcp.frozen_contract import EPSILON, P_EXPONENT
from umcp.kernel_optimized import compute_kernel_outputs

eps = EPSILON
p_exp = P_EXPONENT


def K(c, w=None):
    c = np.asarray(c, dtype=float)
    if w is None:
        w = np.full(len(c), 1.0 / len(c))
    return compute_kernel_outputs(c, np.asarray(w, dtype=float))


def gamma_omega(omega):
    return omega**p_exp / (1 - omega + eps)


# ══════════════════════════════════════════════════════════
# PROBE 1: Channel-count scaling of geometric slaughter
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("PROBE 1: Channel-count scaling of geometric slaughter")
print("  Fixed: (n-1) channels at 0.95, 1 channel at 0.01")
print("=" * 70)
print(f"  {'n':>3}  {'F':>8}  {'IC':>10}  {'IC/F':>8}  {'Delta':>8}  {'kappa':>10}  {'C':>8}")
for n in [2, 3, 4, 6, 8, 12, 16, 32, 64, 128]:
    c = np.full(n, 0.95)
    c[-1] = 0.01
    k = K(c)
    print(
        f"  {n:3d}  {k['F']:8.4f}  {k['IC']:10.6f}  {k['IC'] / k['F']:8.4f}  {k['F'] - k['IC']:8.4f}  {k['kappa']:10.4f}  {k['C']:8.4f}"
    )

# Key observation: IC/F as function of n
print("\n  Key: IC/F = exp((1/n)*ln(c_low)) * exp(((n-1)/n)*ln(c_high))")
print("  = c_low^(1/n) * c_high^((n-1)/n)")
print("  As n->inf: IC/F -> c_high/c_high = 1 (dead channel diluted)")
print("  The RATE at which slaughter diminishes with n is 1/n in the exponent")

# ══════════════════════════════════════════════════════════
# PROBE 2: S-C-F correlation vs channel count
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 2: S vs (F,C) correlation — does S lose independence?")
print("=" * 70)
from scipy import stats

for n in [4, 8, 16, 32, 64]:
    Fs, Ss, Cs, Ks = [], [], [], []
    np.random.seed(42)
    for _ in range(2000):
        c = np.random.beta(2, 2, n)
        k = K(c)
        Fs.append(k["F"])
        Ss.append(k["S"])
        Cs.append(k["C"])
        Ks.append(k["kappa"])
    r_SC = stats.pearsonr(Ss, Cs)[0]
    r_SF = stats.pearsonr(Ss, Fs)[0]
    r_SK = stats.pearsonr(Ss, Ks)[0]
    r_FC = stats.pearsonr(Fs, Cs)[0]
    print(f"  n={n:3d}  r(S,C)={r_SC:+.4f}  r(S,F)={r_SF:+.4f}  r(S,kappa)={r_SK:+.4f}  r(F,C)={r_FC:+.4f}")

# ══════════════════════════════════════════════════════════
# PROBE 3: Gamma/Delta ratio — is there a universal ratio?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 3: Gamma/Delta ratio across trace types")
print("=" * 70)
for label, trace in [
    ("mild_het", [0.9, 0.9, 0.9, 0.7]),
    ("moderate", [0.85, 0.85, 0.6, 0.5]),
    ("strong_het", [0.95, 0.95, 0.95, 0.05]),
    ("2ch_split", [0.99, 0.01]),
    ("8ch_1dead", [0.95] * 7 + [0.001]),
    ("gradient", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
    ("bimodal", [0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05]),
]:
    k = K(trace)
    delta = k["F"] - k["IC"]
    g = gamma_omega(k["omega"])
    ratio = g / delta if delta > 1e-10 else float("inf")
    print(
        f"  {label:12s}  omega={k['omega']:.4f}  Gamma={g:.6f}  Delta={delta:.6f}  Gamma/Delta={ratio:.6f}  regime={k['regime']}"
    )

# ══════════════════════════════════════════════════════════
# PROBE 4: The "coherence derivative" — d(IC/F)/d(c_weak)
# How sensitive is the coherence ratio to the weakest channel?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 4: Coherence sensitivity — d(IC/F)/d(c_weak)")
print("  8 channels, 7 at 0.95, varying c_weak")
print("=" * 70)
c_weak_vals = np.logspace(-8, -0.01, 50)
prev_icf = None
prev_cw = None
print(f"  {'c_weak':>10}  {'IC/F':>8}  {'d(IC/F)/d(c)':>14}  {'elasticity':>12}")
for cw in c_weak_vals:
    c = np.full(8, 0.95)
    c[-1] = cw
    k = K(c)
    icf = k["IC"] / k["F"]
    if prev_icf is not None and prev_cw is not None:
        dcw = cw - prev_cw
        dicf = icf - prev_icf
        deriv = dicf / dcw if abs(dcw) > 1e-15 else 0
        elast = deriv * cw / icf if abs(icf) > 1e-15 else 0
    else:
        deriv = 0
        elast = 0
    if cw < 0.001 or cw > 0.5 or abs(np.log10(cw) - round(np.log10(cw))) < 0.15:
        print(f"  {cw:10.2e}  {icf:8.4f}  {deriv:14.4f}  {elast:12.4f}")
    prev_icf = icf
    prev_cw = cw

# ══════════════════════════════════════════════════════════
# PROBE 5: Rank-2 manifold — when exactly 2 distinct channel
# values exist, what constraints emerge?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 5: Rank-2 manifold — 2-value traces")
print("  n=8, k channels at c_high, (8-k) at c_low")
print("=" * 70)
c_high, c_low = 0.9, 0.3
print(f"  {'k':>2}  {'F':>8}  {'IC':>10}  {'IC/F':>8}  {'Delta':>8}  {'S':>8}  {'C':>8}  {'S-predicted':>12}")
for k_hi in range(0, 9):
    c = np.array([c_high] * k_hi + [c_low] * (8 - k_hi))
    k = K(c)
    # Rank-2: F = (k*c_h + (n-k)*c_l)/n
    # IC = c_h^(k/n) * c_l^((n-k)/n)
    # S should be determined by F and C for rank-2
    # C = std(c)/0.5
    frac = k_hi / 8
    F_pred = frac * c_high + (1 - frac) * c_low
    IC_pred = c_high**frac * c_low ** (1 - frac)
    print(
        f"  {k_hi:2d}  {k['F']:8.4f}  {k['IC']:10.6f}  {k['IC'] / k['F']:8.4f}  {k['F'] - k['IC']:8.4f}  {k['S']:8.4f}  {k['C']:8.4f}  F_err={abs(k['F'] - F_pred):.1e}  IC_err={abs(k['IC'] - IC_pred):.1e}"
    )

# ══════════════════════════════════════════════════════════
# PROBE 6: Equator crossing — n-channel traces where F = 0.5
# exactly. What are the constraints on other invariants?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 6: Equator manifold — traces with F = 0.5 exactly")
print("  Different distributions with same mean")
print("=" * 70)
print(f"  {'label':>15}  {'F':>6}  {'IC':>10}  {'IC/F':>8}  {'S':>8}  {'C':>8}  {'kappa':>10}  {'S+kappa':>10}")
equator_traces = [
    ("uniform_0.5", [0.5] * 8),
    ("binary_clamped", [1.0 - 1e-8, 1e-8] * 4),
    ("spread_1", [0.7, 0.3] * 4),
    ("spread_2", [0.8, 0.2] * 4),
    ("spread_3", [0.9, 0.1] * 4),
    ("spread_4", [0.95, 0.05] * 4),
    ("gradient", [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]),
    ("3level", [0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1]),
]
for label, trace in equator_traces:
    k = K(trace)
    skappa = k["S"] + k["kappa"]
    print(
        f"  {label:>15}  {k['F']:6.4f}  {k['IC']:10.6f}  {k['IC'] / k['F']:8.4f}  {k['S']:8.4f}  {k['C']:8.4f}  {k['kappa']:10.4f}  {skappa:10.4f}"
    )

# ══════════════════════════════════════════════════════════
# PROBE 7: Does Delta * C have a universal bound?
# Product of heterogeneity gap and curvature
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 7: Delta * C — is the product bounded universally?")
print("=" * 70)
np.random.seed(123)
max_dc = 0
max_dc_trace = None
for _ in range(50000):
    n = np.random.choice([4, 8, 16])
    c = np.random.beta(0.5, 0.5, n)  # U-shaped — maximizes spread
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    dc = (k["F"] - k["IC"]) * k["C"]
    if dc > max_dc:
        max_dc = dc
        max_dc_trace = (n, k["F"], k["IC"], k["C"], k["F"] - k["IC"])
print(f"  max(Delta * C) = {max_dc:.6f}")
if max_dc_trace:
    n, F, IC, C, D = max_dc_trace
    print(f"  at n={n}, F={F:.4f}, IC={IC:.6f}, C={C:.4f}, Delta={D:.4f}")

# Also check Delta * S
max_ds = 0
for _ in range(50000):
    n = np.random.choice([4, 8, 16])
    c = np.random.beta(0.5, 0.5, n)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    ds = (k["F"] - k["IC"]) * k["S"]
    if ds > max_ds:
        max_ds = ds
print(f"  max(Delta * S) = {max_ds:.6f}")
print("  max(Delta * C * S) = ???  (need more probes)")

# Now the critical one: Delta * omega
max_do = 0
for _ in range(50000):
    n = np.random.choice([4, 8, 16])
    c = np.random.beta(0.5, 0.5, n)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    k = K(c)
    do = (k["F"] - k["IC"]) * k["omega"]
    if do > max_do:
        max_do = do
print(f"  max(Delta * omega) = {max_do:.6f}")

# ══════════════════════════════════════════════════════════
# PROBE 8: The "triple point" — where all 3 regimes meet
# Find the boundary manifold in (F, S, C) space
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 8: Regime boundary topology")
print("  Sampling 50K traces, counting regime populations by n")
print("=" * 70)
for n in [4, 8, 16, 32]:
    counts = {"Stable": 0, "Watch": 0, "Collapse": 0}
    np.random.seed(42)
    N = 20000
    for _ in range(N):
        c = np.random.uniform(0, 1, n)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        k = K(c)
        regime = k["regime"]
        if regime in counts:
            counts[regime] += 1
        else:
            counts[regime] = 1
    pct = {r: 100 * v / N for r, v in counts.items()}
    print(
        f"  n={n:3d}  Stable={pct.get('Stable', 0):5.1f}%  Watch={pct.get('Watch', 0):5.1f}%  Collapse={pct.get('Collapse', 0):5.1f}%"
    )

# ══════════════════════════════════════════════════════════
# PROBE 9: IC as a function of F at FIXED C
# Is there a universal curve IC(F) | C = const?
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROBE 9: IC(F) at fixed curvature — is there a universal curve?")
print("=" * 70)
# For rank-2 (2-value) traces, C = |c_h - c_l|/(2*0.5) = |c_h - c_l|
# We can parametrize: c_h = F + delta, c_l = F - delta (for n=2)
# Then C = 2*delta/0.5 = 4*delta... no, C = std/0.5
# For n=2 equal weight: std = delta, C = delta/0.5 = 2*delta
# IC = sqrt(c_h * c_l) = sqrt(F^2 - delta^2)
# So IC = sqrt(F^2 - (C*0.5/2)^2) = sqrt(F^2 - C^2/16)...
# Actually for n=2: std = sqrt(((c_h-F)^2 + (c_l-F)^2)/2) = sqrt(delta^2) = delta
# C = delta / 0.5 = 2*delta
# IC = (c_h * c_l)^(1/2) = sqrt((F+delta)(F-delta)) = sqrt(F^2 - delta^2)
# delta = C/2 * 0.5 = C*0.25... wait
# C = std(c) / 0.5
# std for 2 values: each deviates by delta from mean
# std = delta (population std)
# C = delta / 0.5 = 2*delta
# So delta = C/2 * 0.5... no. delta = C * 0.5 / 2... hmm
# C = std / 0.5. std = delta. So C = delta / 0.5 => delta = C * 0.5
# IC = sqrt(F^2 - delta^2) = sqrt(F^2 - (C*0.5)^2) = sqrt(F^2 - C^2/4)
print("  Rank-2 (n=2) prediction: IC = sqrt(F^2 - C^2/4)")
print(f"  {'F':>6}  {'C':>6}  {'IC_actual':>10}  {'IC_pred':>10}  {'error':>10}")
for F_target in [0.3, 0.5, 0.7, 0.9]:
    for C_target in [0.1, 0.3, 0.5]:
        delta = C_target * 0.5
        c_h = F_target + delta
        c_l = F_target - delta
        if c_l < 0.001 or c_h > 0.999:
            continue
        k = K([c_h, c_l])
        IC_pred = np.sqrt(F_target**2 - delta**2)
        print(f"  {k['F']:6.3f}  {k['C']:6.3f}  {k['IC']:10.6f}  {IC_pred:10.6f}  {abs(k['IC'] - IC_pred):10.2e}")

print("\n  INSIGHT: For n=2 rank-2, IC = sqrt(F^2 - (C/2)^2)")
print("  This is a CIRCLE in (F, IC) space at constant C!")
print("  The solvability condition IC <= F is the condition that")
print("  (C/2)^2 >= 0, which is always true.")
