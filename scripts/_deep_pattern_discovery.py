#!/usr/bin/env python3
"""Deep Pattern Discovery — Probe for previously unknown correlations.

This script runs 8 independent computational probes across the GCD kernel
manifold and across all 13 domains, hunting for structural patterns that
have not been documented or formalized.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from umcp.frozen_contract import ALPHA, EPSILON, P_EXPONENT

np.set_printoptions(precision=8, linewidth=120)
rng = np.random.default_rng(42)


def kernel(c, w):
    """Compute all 8 Tier-1 invariants from trace vector c with weights w."""
    c_eps = np.clip(c, EPSILON, 1.0 - EPSILON)
    F = float(np.dot(w, c_eps))
    omega = 1.0 - F
    kappa = float(np.sum(w * np.log(c_eps)))
    IC = float(np.exp(kappa))
    S = float(-np.sum(w * (c_eps * np.log(c_eps) + (1 - c_eps) * np.log(1 - c_eps))))
    C = float(np.sqrt(np.sum(w * (c_eps - F) ** 2)) / 0.5)
    Delta = F - IC
    gamma = omega**P_EXPONENT / (1 - omega + EPSILON)
    D_C = ALPHA * C
    # Weighted variance of channels
    Var_w = float(np.sum(w * (c_eps - F) ** 2))
    return {
        "F": F,
        "omega": omega,
        "S": S,
        "C": C,
        "kappa": kappa,
        "IC": IC,
        "Delta": Delta,
        "gamma": gamma,
        "D_C": D_C,
        "Var_w": Var_w,
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Generate 50,000 random traces across 2-32 channels
# ═══════════════════════════════════════════════════════════════════
N = 50_000
traces = []
for _ in range(N):
    n_ch = int(rng.integers(2, 33))
    c = rng.uniform(0, 1, n_ch)
    w = rng.dirichlet(np.ones(n_ch))
    inv = kernel(c, w)
    inv["n_ch"] = n_ch
    traces.append(inv)

# Vectorize
F_arr = np.array([t["F"] for t in traces])
omega_arr = np.array([t["omega"] for t in traces])
S_arr = np.array([t["S"] for t in traces])
C_arr = np.array([t["C"] for t in traces])
IC_arr = np.array([t["IC"] for t in traces])
Delta_arr = np.array([t["Delta"] for t in traces])
kappa_arr = np.array([t["kappa"] for t in traces])
gamma_arr = np.array([t["gamma"] for t in traces])
n_ch_arr = np.array([t["n_ch"] for t in traces])
Var_arr = np.array([t["Var_w"] for t in traces])

print("=" * 78)
print("  DEEP  PATTERN  DISCOVERY  —  8  PROBES  ACROSS  50K  TRACES")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════
# PROBE 1: Δ/F — Does heterogeneity gap normalize to a universal?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 1: Normalized Heterogeneity Gap  Δ/F  vs Channel Count")
print("═" * 78)
ratio = Delta_arr / np.maximum(F_arr, 1e-15)
print(f"Global: Δ/F = {ratio.mean():.6f} ± {ratio.std():.6f}  (median {np.median(ratio):.6f})")
print(f"{'n_ch':>5}  {'Δ/F mean':>10}  {'Δ/F std':>10}  {'IC/F':>8}  {'count':>6}")
for n in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32]:
    mask = n_ch_arr == n
    if mask.sum() > 50:
        print(
            f"{n:5d}  {ratio[mask].mean():10.6f}  {ratio[mask].std():10.6f}  "
            f"{(IC_arr[mask] / np.maximum(F_arr[mask], 1e-15)).mean():8.4f}  {mask.sum():6d}"
        )

# ═══════════════════════════════════════════════════════════════════
# PROBE 2: S-C Coupling — Entropy and Curvature independence
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 2: Entropy–Curvature Independence (S vs C)")
print("═" * 78)
r_global, p_global = pearsonr(S_arr, C_arr)
rho_global, p_rho = spearmanr(S_arr, C_arr)
print(f"Global: Pearson r(S,C) = {r_global:.6f} (p={p_global:.2e})")
print(f"        Spearman ρ(S,C) = {rho_global:.6f} (p={p_rho:.2e})")
print("\nConditioned on F band:")
print(f"{'F band':>14}  {'r(S,C)':>8}  {'ρ(S,C)':>8}  {'n':>6}")
for f_lo, f_hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    mask = (F_arr >= f_lo) & (F_arr < f_hi)
    if mask.sum() > 100:
        r, _ = pearsonr(S_arr[mask], C_arr[mask])
        rho, _ = spearmanr(S_arr[mask], C_arr[mask])
        print(f"  [{f_lo:.1f}, {f_hi:.1f})  {r:8.4f}  {rho:8.4f}  {mask.sum():6d}")
print("\nConditioned on n_ch:")
print(f"{'n_ch':>5}  {'r(S,C)':>8}  {'ρ(S,C)':>8}  {'n':>6}")
for n in [2, 4, 8, 16, 32]:
    mask = n_ch_arr == n
    if mask.sum() > 100:
        r, _ = pearsonr(S_arr[mask], C_arr[mask])
        rho, _ = spearmanr(S_arr[mask], C_arr[mask])
        print(f"{n:5d}  {r:8.4f}  {rho:8.4f}  {mask.sum():6d}")

# ═══════════════════════════════════════════════════════════════════
# PROBE 3: Jensen's Gap — Exact second-order expansion
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 3: Jensen's Gap — Δ vs Var(c)/(2F̄) Second-Order Identity")
print("═" * 78)
# IC = exp(E[ln c]) = geometric mean
# F = E[c] = arithmetic mean
# Δ = F - IC = AM - GM
# For small variance: AM - GM ≈ Var(c) / (2 * AM)
# This is the FISHER INFORMATION contribution from heterogeneity
fisher_est = Var_arr / (2 * np.maximum(F_arr, 1e-15))
residual = Delta_arr - fisher_est
print(f"Δ actual:       mean = {Delta_arr.mean():.8f}")
print(f"Var(c)/(2F):    mean = {fisher_est.mean():.8f}")
print(f"Residual:       mean = {residual.mean():.8f}, std = {residual.std():.8f}")
print(f"|Residual|:     mean = {np.abs(residual).mean():.8f}, max = {np.abs(residual).max():.8f}")
r_fisher, p_fisher = pearsonr(Delta_arr, fisher_est)
print(f"r(Δ, Var/2F) = {r_fisher:.8f} (p={p_fisher:.2e})")
print("\nBy channel count (checking where second-order breaks down):")
print(f"{'n_ch':>5}  {'|residual| mean':>16}  {'|residual| max':>16}  {'r(Δ,est)':>10}")
for n in [2, 4, 8, 16, 32]:
    mask = n_ch_arr == n
    if mask.sum() > 50:
        res_n = residual[mask]
        r_n, _ = pearsonr(Delta_arr[mask], fisher_est[mask])
        print(f"{n:5d}  {np.abs(res_n).mean():16.8f}  {np.abs(res_n).max():16.8f}  {r_n:10.6f}")

# ═══════════════════════════════════════════════════════════════════
# PROBE 4: κ Scaling Law — Log-integrity vs ln(n_ch)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 4: Log-Integrity Scaling Law — κ vs ln(n_ch)")
print("═" * 78)
# Theory: For uniform random c_i, E[ln c] ≈ -1 regardless of n
# But IC = exp(κ) where κ = Σ w_i ln(c_i)
# As n increases with random weights, this is a WEIGHTED average of ln(c_i)
# Each ln(c_i) ~ Uniform(-∞, 0) ... actually ln(U(0,1)) has mean -1, var 1
# So κ should converge to -1 for large n (by LLN on the weighted sum)
print(f"{'n_ch':>5}  {'κ mean':>10}  {'κ std':>10}  {'IC mean':>10}  {'IC/F':>8}  {'F mean':>8}")
for n in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32]:
    mask = n_ch_arr == n
    if mask.sum() > 50:
        k = kappa_arr[mask]
        ic = IC_arr[mask]
        f = F_arr[mask]
        print(
            f"{n:5d}  {k.mean():10.4f}  {k.std():10.4f}  {ic.mean():10.6f}  "
            f"{(ic / np.maximum(f, 1e-15)).mean():8.4f}  {f.mean():8.4f}"
        )

# Test: κ = a * ln(n) + b ?
unique_n = sorted(set(n_ch_arr))
k_means = [kappa_arr[n_ch_arr == n].mean() for n in unique_n]
ln_n = np.log(unique_n)
coeffs = np.polyfit(ln_n, k_means, 1)
print(f"\nLinear fit: κ ≈ {coeffs[0]:.4f} × ln(n) + {coeffs[1]:.4f}")
print("(Slope near 0 → κ does NOT scale with ln(n), it converges)")

# ═══════════════════════════════════════════════════════════════════
# PROBE 5: Cost Function Phase Portrait — Γ(ω) critical geometry
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 5: Cost Function Γ(ω) — Critical Points & Phase Geometry")
print("═" * 78)
omega_test = np.linspace(0.001, 0.999, 10000)
p = P_EXPONENT
gamma_test = omega_test**p / (1 - omega_test + EPSILON)

# Critical crossings
for threshold, name in [
    (0.01, "onset"),
    (0.1, "noticeable"),
    (1.0, "FIRST WELD"),
    (10.0, "severe"),
    (100.0, "catastrophic"),
]:
    cross = np.where(np.diff(np.sign(gamma_test - threshold)))[0]
    if len(cross) > 0:
        oc = omega_test[cross[0]]
        print(f"  Γ(ω) = {threshold:7.2f} ({name:>14s}) at ω = {oc:.6f}  (F = {1 - oc:.6f})")

# Derivative
dG = np.gradient(gamma_test, omega_test)
# Inflection point (where d²Γ/dω² = 0)
d2G = np.gradient(dG, omega_test)
inflection = np.where(np.diff(np.sign(d2G)))[0]
if len(inflection) > 0:
    omega_inf = omega_test[inflection[0]]
    print(f"\n  Inflection point: ω = {omega_inf:.6f} (F = {1 - omega_inf:.6f})")
    print(f"  dΓ/dω at inflection = {dG[inflection[0]]:.4f}")
    print("  This is where the cost acceleration changes — the 'knee' of collapse")

# What fraction of the 50K traces are in each Γ zone?
print("\n  Γ zone distribution (50K random traces):")
for lo, hi, name in [
    (0, 0.01, "Safe"),
    (0.01, 0.1, "Drift"),
    (0.1, 1.0, "Watch"),
    (1.0, 10.0, "Collapse"),
    (10.0, float("inf"), "Catastrophic"),
]:
    mask = (gamma_arr >= lo) & (gamma_arr < hi)
    print(f"    {name:>14s}: {mask.sum():6d} ({mask.sum() / N * 100:5.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# PROBE 6: Information Geometry — S as a function of (F, C)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 6: Information Geometry — S as a function of (F, C)")
print("═" * 78)
# Can S be predicted from F and C alone?
# Multiple regression: S = a*F + b*C + c*F*C + d
X = np.column_stack([F_arr, C_arr, F_arr * C_arr, np.ones(N)])
beta = np.linalg.lstsq(X, S_arr, rcond=None)[0]
S_pred = X @ beta
resid = S_arr - S_pred
R2 = 1 - np.var(resid) / np.var(S_arr)
print(f"  Linear model: S = {beta[0]:.4f}*F + {beta[1]:.4f}*C + {beta[2]:.4f}*F*C + {beta[3]:.4f}")
print(f"  R² = {R2:.6f}  (1.0 = perfect prediction)")
print(f"  Residual std = {resid.std():.6f}")

# Add quadratic terms
X2 = np.column_stack([F_arr, C_arr, F_arr * C_arr, F_arr**2, C_arr**2, np.ones(N)])
beta2 = np.linalg.lstsq(X2, S_arr, rcond=None)[0]
S_pred2 = X2 @ beta2
resid2 = S_arr - S_pred2
R2_q = 1 - np.var(resid2) / np.var(S_arr)
print(f"  Quadratic model R² = {R2_q:.6f}")

# Add IC and κ
X3 = np.column_stack([F_arr, C_arr, IC_arr, kappa_arr, F_arr * C_arr, np.ones(N)])
beta3 = np.linalg.lstsq(X3, S_arr, rcond=None)[0]
S_pred3 = X3 @ beta3
resid3 = S_arr - S_pred3
R2_full = 1 - np.var(resid3) / np.var(S_arr)
print(f"  Full model (F,C,IC,κ,F*C) R² = {R2_full:.6f}")
print(f"  → Entropy has {'strong' if R2_full > 0.95 else 'weak'} dependence on other invariants")

# ═══════════════════════════════════════════════════════════════════
# PROBE 7: Regime Boundary Topology
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 7: Regime Boundary Topology")
print("═" * 78)
# Classify each trace
regimes = []
for t in traces:
    o, f, s, c = t["omega"], t["F"], t["S"], t["C"]
    if o >= 0.30:
        reg = "Collapse"
    elif o < 0.038 and f > 0.90 and s < 0.15 and c < 0.14:
        reg = "Stable"
    else:
        reg = "Watch"
    regimes.append(reg)
regimes = np.array(regimes)

for reg in ["Stable", "Watch", "Collapse"]:
    mask = regimes == reg
    count = mask.sum()
    if count > 0:
        print(f"\n  {reg:>10s}: {count:5d} ({count / N * 100:5.1f}%)")
        print(
            f"    F  = {F_arr[mask].mean():.4f} ± {F_arr[mask].std():.4f}  [{F_arr[mask].min():.4f}, {F_arr[mask].max():.4f}]"
        )
        print(
            f"    IC = {IC_arr[mask].mean():.4f} ± {IC_arr[mask].std():.4f}  [{IC_arr[mask].min():.4f}, {IC_arr[mask].max():.4f}]"
        )
        print(f"    S  = {S_arr[mask].mean():.4f} ± {S_arr[mask].std():.4f}")
        print(f"    C  = {C_arr[mask].mean():.4f} ± {C_arr[mask].std():.4f}")
        print(f"    Δ  = {Delta_arr[mask].mean():.4f} ± {Delta_arr[mask].std():.4f}")
        print(f"    Δ/F= {(Delta_arr[mask] / np.maximum(F_arr[mask], 1e-15)).mean():.4f}")
        print(
            f"    Γ  = {gamma_arr[mask].mean():.4f} ± {gamma_arr[mask].std():.4f}  [median {np.median(gamma_arr[mask]):.4f}]"
        )

# ═══════════════════════════════════════════════════════════════════
# PROBE 8: Cross-Invariant Correlation Matrix
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 8: Full Cross-Invariant Correlation Matrix (Spearman)")
print("═" * 78)
names = ["F", "ω", "S", "C", "κ", "IC", "Δ", "Γ", "n_ch"]
arrays = [F_arr, omega_arr, S_arr, C_arr, kappa_arr, IC_arr, Delta_arr, gamma_arr, n_ch_arr.astype(float)]
n_inv = len(names)
corr_matrix = np.zeros((n_inv, n_inv))
for i in range(n_inv):
    for j in range(n_inv):
        corr_matrix[i, j], _ = spearmanr(arrays[i], arrays[j])

# Print header
print(f"{'':>5s}", end="")
for name in names:
    print(f"  {name:>6s}", end="")
print()
for i in range(n_inv):
    print(f"{names[i]:>5s}", end="")
    for j in range(n_inv):
        val = corr_matrix[i, j]
        print(f"  {val:6.3f}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════
# PROBE 9: The ENTROPY PEAK — Where does S maximize and why?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 9: Entropy Maximum — Where in (F, C) space does S peak?")
print("═" * 78)
# Bin by F and find S peak
print(f"{'F band':>14}  {'S mean':>8}  {'S max':>8}  {'C at S_max':>10}  {'IC at S_max':>10}")
for f_lo in np.arange(0.0, 1.0, 0.1):
    f_hi = f_lo + 0.1
    mask = (F_arr >= f_lo) & (F_arr < f_hi)
    if mask.sum() > 50:
        s_vals = S_arr[mask]
        s_max_idx = np.argmax(s_vals)
        abs_idx = np.where(mask)[0][s_max_idx]
        print(
            f"  [{f_lo:.1f}, {f_hi:.1f})  {s_vals.mean():8.4f}  {s_vals.max():8.4f}  "
            f"{C_arr[abs_idx]:10.4f}  {IC_arr[abs_idx]:10.6f}"
        )

# What F value maximizes S?
# Theory: Bernoulli entropy S = -c ln(c) - (1-c)ln(1-c) peaks at c = 0.5
# So S should peak when ALL channels are near 0.5, which gives F ≈ 0.5
top_S = np.argsort(S_arr)[-100:]  # Top 100 S values
print(
    f"\nTop 100 by S:  F={F_arr[top_S].mean():.4f}±{F_arr[top_S].std():.4f}, "
    f"C={C_arr[top_S].mean():.4f}±{C_arr[top_S].std():.4f}, "
    f"IC={IC_arr[top_S].mean():.4f}"
)
print("  → Entropy peaks at F ≈ 0.5 AND low C (channels clustered near 0.5)")

# ═══════════════════════════════════════════════════════════════════
# PROBE 10: The GOLDEN RATIO of collapse?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 10: Structural Constants of the Kernel")
print("═" * 78)
# The first weld (Γ = 1) is at ω* ≈ 0.317 for p=3
# Is there a deeper relationship?
# Solve ω^3 = 1 - ω exactly  → ω^3 + ω = 1
# This is a depressed cubic with one real root
# ω³ + ω - 1 = 0
coeffs_poly = [1, 0, 1, -1]  # standard form: x³ + 0x² + x - 1
roots = np.roots(coeffs_poly)
real_root = roots[np.isreal(roots)].real[0]
print(f"  First Weld: ω³ + ω = 1  →  ω* = {real_root:.10f}")
print(f"  Complementary: F* = 1 - ω* = {1 - real_root:.10f}")
print(f"  Γ(ω*) = ω*³/(1-ω*) = {real_root**3 / (1 - real_root):.10f}")
print(f"  Check: ω* = {real_root:.10f} ≈ 1/π = {1 / math.pi:.10f}?  Diff = {abs(real_root - 1 / math.pi):.6e}")
print(
    f"  Check: F* = {1 - real_root:.10f} ≈ φ−1 = {(1 + math.sqrt(5)) / 2 - 1:.10f}?  Diff = {abs(1 - real_root - ((1 + math.sqrt(5)) / 2 - 1)):.6e}"
)

# The ratio κ_stable / κ_collapse
stable_mask = regimes == "Stable"
collapse_mask = regimes == "Collapse"
if stable_mask.sum() > 0 and collapse_mask.sum() > 0:
    k_stable = kappa_arr[stable_mask].mean()
    k_collapse = kappa_arr[collapse_mask].mean()
    print(f"\n  κ_Stable = {k_stable:.6f},  κ_Collapse = {k_collapse:.6f}")
    print(f"  Ratio κ_Collapse/κ_Stable = {k_collapse / k_stable:.6f}")

# IC at the regime boundaries
print("\n  IC at regime boundaries:")
for threshold in [0.038, 0.10, 0.20, 0.30]:
    nearby = np.abs(omega_arr - threshold) < 0.005
    if nearby.sum() > 0:
        print(
            f"    ω = {threshold:.3f}: IC = {IC_arr[nearby].mean():.6f} ± {IC_arr[nearby].std():.6f}, "
            f"Γ = {gamma_arr[nearby].mean():.6f}"
        )

# ═══════════════════════════════════════════════════════════════════
# PROBE 11: MUTUAL INFORMATION between regime and each invariant
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 11: Which invariant is the best regime predictor?")
print("═" * 78)
# Use a simple metric: for each invariant, how well can we predict regime?
# Compute mean separation between Stable and Collapse distributions
# normalized by pooled std (= Cohen's d)
if stable_mask.sum() > 10 and collapse_mask.sum() > 10:
    print(f"{'Invariant':>10}  {'Cohen d':>10}  {'AUC est':>10}  {'Interpretation':>20}")
    for name, arr in [
        ("F", F_arr),
        ("ω", omega_arr),
        ("S", S_arr),
        ("C", C_arr),
        ("IC", IC_arr),
        ("Δ", Delta_arr),
        ("κ", kappa_arr),
        ("Γ", gamma_arr),
    ]:
        s_vals = arr[stable_mask]
        c_vals = arr[collapse_mask]
        pooled_std = np.sqrt((s_vals.std() ** 2 + c_vals.std() ** 2) / 2)
        d = abs(s_vals.mean() - c_vals.mean()) / max(pooled_std, 1e-15)
        # Rough AUC estimate from Cohen's d
        auc = 1 / (1 + np.exp(-0.71 * d))  # Logistic approximation
        interp = "Poor" if d < 0.5 else "Fair" if d < 1.0 else "Good" if d < 2.0 else "Excellent"
        print(f"{name:>10}  {d:10.4f}  {auc:10.4f}  {interp:>20}")

# ═══════════════════════════════════════════════════════════════════
# PROBE 12: Δ-IC Phase Space — Is there a forbidden region?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 12: Forbidden Regions in (Δ, IC) Phase Space")
print("═" * 78)
# Since IC ≤ F and Δ = F - IC, we have Δ ≥ 0 always
# Also IC = exp(κ) and F = IC + Δ
# So F ≤ 1 means IC + Δ ≤ 1, i.e., IC ≤ 1 - Δ
# The accessible region is: 0 ≤ IC ≤ 1-Δ, Δ ≥ 0
# But are there additional constraints?
print("  Theoretical boundary: IC + Δ ≤ 1  (since F = IC + Δ ≤ 1)")
violations = IC_arr + Delta_arr > 1.0 + 1e-10
print(f"  Violations: {violations.sum()} / {N} (should be 0)")

# Check the actual boundary
for delta_bin in np.arange(0.0, 0.6, 0.05):
    mask = (Delta_arr >= delta_bin) & (Delta_arr < delta_bin + 0.05)
    if mask.sum() > 50:
        ic_max = IC_arr[mask].max()
        ic_min = IC_arr[mask].min()
        theoretical_max = 1.0 - delta_bin
        print(
            f"  Δ ∈ [{delta_bin:.2f}, {delta_bin + 0.05:.2f}): IC ∈ [{ic_min:.4f}, {ic_max:.4f}], "
            f"theoretical max IC = {theoretical_max:.2f}, gap = {theoretical_max - ic_max:.4f}"
        )

# ═══════════════════════════════════════════════════════════════════
# PROBE 13: EQUAL-WEIGHT vs DIRICHLET — Does weight structure matter?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("PROBE 13: Equal Weights vs Dirichlet Weights")
print("═" * 78)
# Run 10K traces with n=8 channels, comparing equal vs Dirichlet weights
n_test = 10_000
eq_results = {"IC_F": [], "Delta": [], "C": [], "S": []}
dir_results = {"IC_F": [], "Delta": [], "C": [], "S": []}
for _ in range(n_test):
    c = rng.uniform(0, 1, 8)
    # Equal weights
    w_eq = np.ones(8) / 8
    inv_eq = kernel(c, w_eq)
    eq_results["IC_F"].append(inv_eq["IC"] / max(inv_eq["F"], 1e-15))
    eq_results["Delta"].append(inv_eq["Delta"])
    eq_results["C"].append(inv_eq["C"])
    eq_results["S"].append(inv_eq["S"])
    # Dirichlet weights (same channels)
    w_dir = rng.dirichlet(np.ones(8))
    inv_dir = kernel(c, w_dir)
    dir_results["IC_F"].append(inv_dir["IC"] / max(inv_dir["F"], 1e-15))
    dir_results["Delta"].append(inv_dir["Delta"])
    dir_results["C"].append(inv_dir["C"])
    dir_results["S"].append(inv_dir["S"])

print(f"  n=8 channels, {n_test} traces, same c_i for both:")
print(f"{'':>12}  {'Equal W':>12}  {'Dirichlet W':>12}  {'Diff':>10}")
for key in ["IC_F", "Delta", "C", "S"]:
    eq_mean = np.mean(eq_results[key])
    dir_mean = np.mean(dir_results[key])
    print(f"  {key:>10}  {eq_mean:12.6f}  {dir_mean:12.6f}  {eq_mean - dir_mean:10.6f}")

print(
    f"\n  → Equal weights {'preserve' if np.mean(eq_results['IC_F']) > np.mean(dir_results['IC_F']) else 'destroy'} more integrity"
)
print(
    f"  → Weight heterogeneity is {'an additional' if np.mean(eq_results['IC_F']) > np.mean(dir_results['IC_F']) else 'not a significant'} source of IC loss beyond channel heterogeneity"
)

print("\n" + "=" * 78)
print("  PROBES COMPLETE — 13 INDEPENDENT DISCOVERY VECTORS")
print("=" * 78)
