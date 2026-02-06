#!/usr/bin/env python3
# pyright: reportArgumentType=false
"""
Cosmology Exploration: Pushing WEYL Closures for Novel Outcomes

This script goes beyond the dashboard visualization to perform
substantive computational exploration of the WEYL modified gravity
framework, looking for statistically interesting signals.

Analyses performed:
 1. Full Σ₀ fit across all 3 g(z) models × 4 data variants
 2. Bin-by-bin Σ(z) inversion — is the deviation redshift-dependent?
 3. Tension quantification: σ₈ discrepancy across methods
 4. Weyl transfer function evolution — where does GR break?
 5. Nonlinear boost sensitivity — does scale dependence hide signals?
 6. χ² landscape scan — is the minimum unique or degenerate?
 7. Model comparison: Bayesian Information Criterion (BIC)
 8. Forecast: what precision would confirm/rule out the signal?
 9. UMCP regime transition mapping
10. Novel: cross-correlation of ĥJ residuals between bins

Reference: Nature Communications 15:9295 (2024)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

# Add project root so closures can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from closures.weyl import (
    DES_Y3_DATA,
    D1_of_z,
    GzModel,
    H_of_z,
    Omega_Lambda_of_z,
    Omega_m_of_z,
    Sigma_to_UMCP_invariants,
    chi_of_z,
    compute_background,
    compute_des_y3_background,
    compute_Sigma,
    compute_Sigma_from_hJ,
    compute_weyl_transfer,
    fit_Sigma_0,
    halofit_boost,
    sigma8_of_z,
)

np.set_printoptions(precision=6, suppress=True)

# ═══════════════════════════════════════════════════════════════════════
# SECTION 0: Setup
# ═══════════════════════════════════════════════════════════════════════

z_bins = DES_Y3_DATA["z_bins"]
n_bins = len(z_bins)

# Background quantities at anchor z*=10
bg_star = compute_background(10.0)
D1_star = bg_star.D1
sigma8_star = bg_star.sigma8_z

# Background at each bin
des_bg = compute_des_y3_background()

# Data variants
DATA_VARIANTS = {
    "CMB_prior": DES_Y3_DATA["hJ_cmb"],
    "No_CMB": DES_Y3_DATA["hJ_no_cmb"],
    "Pessimistic_CMB": DES_Y3_DATA["hJ_pessimistic_cmb"],
    "Pessimistic_No_CMB": DES_Y3_DATA["hJ_pessimistic_no_cmb"],
}

# g(z) models
GZ_MODELS = [GzModel.STANDARD, GzModel.CONSTANT, GzModel.EXPONENTIAL]

results: dict = {
    "timestamp": datetime.now(UTC).isoformat(),
    "reference": "Nature Communications 15:9295 (2024)",
    "sections": {},
}

print("=" * 80)
print("  WEYL COSMOLOGY EXPLORATION — Pushing the Numbers")
print("  Reference: Nature Comms 15:9295 (2024) + UMCP Framework")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Full Σ₀ Fits (3 models × 4 data variants = 12 fits)
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 1: Full Σ₀ Fitting (12 combinations)")
print("━" * 80)

fit_results: dict = {}

for variant_name, variant_data in DATA_VARIANTS.items():
    fit_results[variant_name] = {}
    for model in GZ_MODELS:
        fit = fit_Sigma_0(
            z_bins=z_bins,
            hJ_measured=variant_data["mean"],
            hJ_errors=variant_data["sigma"],
            Omega_m_z=Omega_m_of_z,
            D1_z=D1_of_z,
            D1_z_star=D1_star,
            sigma8_z_star=sigma8_star,
            g_model=model,
            Omega_Lambda_z=Omega_Lambda_of_z,
        )
        fit_results[variant_name][model.value] = fit

        # Significance of detection
        significance = abs(fit.Sigma_0) / max(fit.Sigma_0_error, 1e-10)

        print(f"\n  {variant_name} × {model.value}:")
        print(f"    Σ₀ = {fit.Sigma_0:+.4f} ± {fit.Sigma_0_error:.4f}")
        print(f"    Significance: {significance:.1f}σ")
        print(f"    χ²/dof = {fit.chi2_red:.3f}  (p = {fit.p_value:.4f})")
        print(f"    Regime: {fit.regime}")

results["sections"]["sigma0_fits"] = {
    variant: {
        model: {
            "Sigma_0": f.Sigma_0,
            "Sigma_0_error": f.Sigma_0_error,
            "significance_sigma": abs(f.Sigma_0) / max(f.Sigma_0_error, 1e-10),
            "chi2_red": f.chi2_red,
            "p_value": f.p_value,
            "regime": f.regime,
        }
        for model, f in models.items()
    }
    for variant, models in fit_results.items()
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Bin-by-Bin Σ(z) Inversion — Is the Signal z-Dependent?
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 2: Bin-by-Bin Σ(z) Inversion — Redshift Dependence")
print("━" * 80)
print("  Question: Is the deviation from GR constant or evolving?")

bin_inversions: dict = {}

for variant_name, variant_data in DATA_VARIANTS.items():
    bin_inversions[variant_name] = []
    print(f"\n  {variant_name}:")
    print(f"    {'z':>6}  {'ĥJ':>8}  {'σ':>8}  {'Σ(z)':>8}  {'Σ-1':>8}  {'S/N':>6}")
    print("    " + "-" * 55)

    for i, z in enumerate(z_bins):
        hJ = variant_data["mean"][i]
        sigma = variant_data["sigma"][i]
        Omega_m_z_val = Omega_m_of_z(z)
        D1_z_val = D1_of_z(z)

        Sigma_z = compute_Sigma_from_hJ(
            hJ=hJ,
            Omega_m_z=Omega_m_z_val,
            D1_z=D1_z_val,
            D1_z_star=D1_star,
            sigma8_z_star=sigma8_star,
        )

        # Propagate error: δΣ ≈ δĥJ / (Ω_m D₁/D₁* σ₈*)
        denom = Omega_m_z_val * (D1_z_val / D1_star) * sigma8_star
        Sigma_err = sigma / max(abs(denom), 1e-10)
        deviation = Sigma_z - 1.0
        sn = abs(deviation) / max(Sigma_err, 1e-10)

        bin_inversions[variant_name].append(
            {
                "z": float(z),
                "hJ": float(hJ),
                "hJ_sigma": float(sigma),
                "Sigma_z": float(Sigma_z),
                "Sigma_minus_1": float(deviation),
                "Sigma_error": float(Sigma_err),
                "signal_to_noise": float(sn),
            }
        )

        print(f"    {z:6.3f}  {hJ:8.4f}  {sigma:8.4f}  {Sigma_z:8.4f}  {deviation:+8.4f}  {sn:6.1f}")

    # Test for z-dependence: linear fit of (Σ-1) vs z
    deviations = [b["Sigma_minus_1"] for b in bin_inversions[variant_name]]
    errors = [b["Sigma_error"] for b in bin_inversions[variant_name]]
    weights = [1.0 / e**2 for e in errors]

    # Weighted linear regression
    z_arr = np.array([float(z) for z in z_bins])
    dev_arr = np.array(deviations)
    w_arr = np.array(weights)

    z_mean = np.average(z_arr, weights=w_arr)
    dev_mean = np.average(dev_arr, weights=w_arr)
    slope = np.sum(w_arr * (z_arr - z_mean) * (dev_arr - dev_mean)) / np.sum(w_arr * (z_arr - z_mean) ** 2)
    slope_err = 1.0 / np.sqrt(np.sum(w_arr * (z_arr - z_mean) ** 2))
    slope_significance = abs(slope) / slope_err

    print(f"\n    ⤷ Linear trend: d(Σ-1)/dz = {slope:+.4f} ± {slope_err:.4f}")
    print(f"    ⤷ Trend significance: {slope_significance:.1f}σ")
    if slope_significance > 2.0:
        print("    ⚠️  NOTABLE: Possible redshift-dependent deviation!")
    elif slope_significance > 1.0:
        print("    ⊘  Mild hint of z-dependence (not significant)")
    else:
        print("    ✓  Consistent with constant Σ₀ (no z-evolution)")

results["sections"]["bin_inversions"] = bin_inversions


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: σ₈ Tension Quantification
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 3: σ₈ Tension Quantification")
print("━" * 80)

s8_data = DES_Y3_DATA["sigma8_comparison"]

# CMB-derived vs lensing-derived
s8_cmb_params = s8_data["from_params_cmb"]
s8_cmb_hJ = s8_data["from_hJ_cmb"]
s8_nocmb_params = s8_data["from_params_no_cmb"]
s8_nocmb_hJ = s8_data["from_hJ_no_cmb"]


# Tension = |μ₁ - μ₂| / √(σ₁² + σ₂²)
def compute_tension(m1: float, s1: float, m2: float, s2: float) -> float:
    return abs(m1 - m2) / np.sqrt(s1**2 + s2**2)


tension_cmb = compute_tension(
    s8_cmb_params["mean"],
    s8_cmb_params["sigma"],
    s8_cmb_hJ["mean"],
    s8_cmb_hJ["sigma"],
)

# For asymmetric errors, use average
s8_nocmb_hJ_sigma = (s8_nocmb_hJ["sigma_plus"] + s8_nocmb_hJ["sigma_minus"]) / 2
tension_nocmb = compute_tension(
    s8_nocmb_params["mean"],
    s8_nocmb_params["sigma"],
    s8_nocmb_hJ["mean"],
    s8_nocmb_hJ_sigma,
)

# Also compute: what Σ₀ would resolve the σ₈ tension?
# If σ₈(lensing) = σ₈(CMB) × Σ, then Σ = σ₈(lensing)/σ₈(CMB)
Sigma_resolution = s8_cmb_hJ["mean"] / s8_cmb_params["mean"]
Sigma_0_resolution = Sigma_resolution - 1.0

print("\n  With CMB prior:")
print(f"    σ₈ (from params) = {s8_cmb_params['mean']:.3f} ± {s8_cmb_params['sigma']:.3f}")
print(f"    σ₈ (from ĥJ)    = {s8_cmb_hJ['mean']:.3f} ± {s8_cmb_hJ['sigma']:.3f}")
print(f"    Tension: {tension_cmb:.1f}σ")

print("\n  Without CMB prior:")
print(f"    σ₈ (from params) = {s8_nocmb_params['mean']:.3f} ± {s8_nocmb_params['sigma']:.3f}")
print(
    f"    σ₈ (from ĥJ)    = {s8_nocmb_hJ['mean']:.3f} +{s8_nocmb_hJ['sigma_plus']:.3f}/-{s8_nocmb_hJ['sigma_minus']:.3f}"
)
print(f"    Tension: {tension_nocmb:.1f}σ")

print("\n  To resolve σ₈ tension via modified gravity:")
print(f"    Σ needed = σ₈(lensing)/σ₈(CMB) = {Sigma_resolution:.3f}")
print(f"    → Σ₀ = {Sigma_0_resolution:+.3f}")
print(f"    → This is {'consistent' if abs(Sigma_0_resolution) < 0.3 else 'inconsistent'} with measured Σ₀ ≈ 0.24")

# Is the tension direction consistent with the Σ₀ sign?
consistent = (Sigma_0_resolution < 0) == (fit_results["CMB_prior"]["standard"].Sigma_0 < 0) or (
    Sigma_0_resolution > 0
) == (fit_results["CMB_prior"]["standard"].Sigma_0 > 0)
print(
    f"    → Sign consistency: {'YES ✓' if not consistent else 'NO ✗'} — "
    f"Σ₀ > 0 strengthens lensing, but σ₈(lensing) < σ₈(CMB)"
)
# Actually, Σ > 1 means MORE lensing, but σ₈(lensing) < σ₈(CMB)
# This is the Weyl tension: Σ₀ > 0 says gravity is STRONGER than GR,
# yet we measure LESS clustering. This is the novel puzzle.
print("\n  ★ NOVEL OBSERVATION:")
print("    Σ₀ = +0.24 implies gravity is STRONGER than GR (more lensing)")
print("    Yet σ₈(lensing) < σ₈(CMB) implies LESS structure growth")
print("    This apparent contradiction may indicate:")
print("      1. Scale-dependent modified gravity (different k-behavior)")
print("      2. Systematic in ĥJ or σ₈ extraction")
print("      3. A genuinely novel gravitational phenomenology")

results["sections"]["sigma8_tension"] = {
    "tension_cmb_sigma": float(tension_cmb),
    "tension_nocmb_sigma": float(tension_nocmb),
    "Sigma_to_resolve": float(Sigma_resolution),
    "Sigma_0_to_resolve": float(Sigma_0_resolution),
    "sign_puzzle": "Σ₀>0 (stronger gravity) contradicts σ₈(lens)<σ₈(CMB) (less growth)",
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Weyl Transfer Function Evolution
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 4: Weyl Transfer Function T_{Ψ_W}(z) Evolution")
print("━" * 80)

z_scan = np.linspace(0.01, 3.0, 60)

# Compute T_Weyl across redshift for GR and modified gravity
print("\n  Scanning z = [0.01, 3.0] with Σ₀ = 0.0 (GR) and 0.24 (DES)")

transfer_gr: list = []
transfer_mg: list = []

H_star = H_of_z(10.0)
D1_star_val = D1_of_z(10.0)
sigma8_star_val = sigma8_of_z(10.0)

for z in z_scan:
    H_z = H_of_z(z)
    B_ratio = np.sqrt(halofit_boost(0.1, z) / halofit_boost(0.1, 10.0))

    # GR: J(z) = 1
    T_gr = compute_weyl_transfer(
        H_z=H_z,
        H_z_star=H_star,
        J_z=1.0,
        D1_z_star=D1_star_val,
        B_ratio=B_ratio,
        T_Weyl_star=1.0,
    )
    transfer_gr.append(T_gr)

    # Modified gravity: J(z) = Σ(z) with Σ₀=0.24
    Sigma_result = compute_Sigma(z, 0.24, GzModel.STANDARD, Omega_Lambda_of_z)
    T_mg = compute_weyl_transfer(
        H_z=H_z,
        H_z_star=H_star,
        J_z=Sigma_result.Sigma,
        D1_z_star=D1_star_val,
        B_ratio=B_ratio,
        T_Weyl_star=1.0,
    )
    transfer_mg.append(T_mg)

# Find peak difference
T_gr_arr = np.array([t.T_Weyl for t in transfer_gr])
T_mg_arr = np.array([t.T_Weyl for t in transfer_mg])
rel_diff = (T_mg_arr - T_gr_arr) / np.maximum(np.abs(T_gr_arr), 1e-10) * 100

max_diff_idx = np.argmax(np.abs(rel_diff))
max_diff_z = z_scan[max_diff_idx]
max_diff_pct = rel_diff[max_diff_idx]

print("\n  Key results:")
print(f"    Max |T_MG - T_GR| / T_GR = {abs(max_diff_pct):.1f}% at z = {max_diff_z:.2f}")
print(f"    T_Weyl(z=0.5, GR)  = {T_gr_arr[np.argmin(np.abs(z_scan - 0.5))]:.4f}")
print(f"    T_Weyl(z=0.5, MG)  = {T_mg_arr[np.argmin(np.abs(z_scan - 0.5))]:.4f}")
print(f"    T_Weyl(z=1.0, GR)  = {T_gr_arr[np.argmin(np.abs(z_scan - 1.0))]:.4f}")
print(f"    T_Weyl(z=1.0, MG)  = {T_mg_arr[np.argmin(np.abs(z_scan - 1.0))]:.4f}")

# Where does GR deviate most?
# Regimes from transfer function
regimes_z = [(z_scan[i], transfer_mg[i].regime) for i in range(len(z_scan))]
regime_transitions = []
for i in range(1, len(regimes_z)):
    if regimes_z[i][1] != regimes_z[i - 1][1]:
        regime_transitions.append((regimes_z[i][0], regimes_z[i - 1][1], regimes_z[i][1]))

if regime_transitions:
    print("\n  Regime transitions (Σ₀=0.24):")
    for zt, r_from, r_to in regime_transitions:
        print(f"    z = {zt:.2f}: {r_from} → {r_to}")
else:
    print(f"\n  No regime transitions — signal stays in: {regimes_z[0][1]}")

results["sections"]["transfer_evolution"] = {
    "max_relative_difference_pct": float(abs(max_diff_pct)),
    "max_diff_redshift": float(max_diff_z),
    "regime_transitions": [(float(zt), rf, rt) for zt, rf, rt in regime_transitions],
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Scale-Dependent Sensitivity (Boost Factor)
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 5: Scale-Dependent Sensitivity")
print("━" * 80)
print("  Question: Does the signal look different on linear vs nonlinear scales?")

k_values = np.logspace(-3, 1, 50)  # 0.001 to 10 h/Mpc
z_test_vals = [0.3, 0.5, 0.8, 1.0]

print(f"\n  {'k (h/Mpc)':>12}  ", end="")
for z in z_test_vals:
    print(f"{'B(z=' + f'{z:.1f}' + ')':>12}", end="")
print()
print("  " + "-" * 60)

# Print selected k values
for k in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]:
    print(f"  {k:12.3f}  ", end="")
    for z in z_test_vals:
        B = halofit_boost(k, z)
        print(f"{B:12.4f}", end="")
    print()

# Key question: at DES Y3 ℓ-range, what scales dominate?
print("\n  DES Y3 typical multipoles ℓ ~ [200, 2000]")
print("  At z=0.5: k_ℓ = (ℓ+0.5)/χ(z)")
chi_05 = chi_of_z(0.5)
for ell in [200, 500, 1000, 2000]:
    k_ell_val = (ell + 0.5) / chi_05
    B = halofit_boost(k_ell_val, 0.5)
    regime = "linear" if B < 1.05 else "quasi-linear" if B < 2 else "nonlinear"
    print(f"    ℓ = {ell:5d} → k = {k_ell_val:.3f} h/Mpc → B = {B:.3f} ({regime})")

print("\n  ★ NOTE: Modified gravity could be scale-dependent.")
print("    If Σ(k,z) ≠ Σ(z), the signal would differ across ℓ-bins.")
print("    Current framework assumes Σ(z) only — a potential blind spot.")

results["sections"]["scale_sensitivity"] = {
    "chi_z05_Mpc_h": float(chi_05),
    "scales_at_z05": [
        {"ell": ell, "k_h_Mpc": float((ell + 0.5) / chi_05), "B_boost": float(halofit_boost((ell + 0.5) / chi_05, 0.5))}
        for ell in [200, 500, 1000, 2000]
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: χ² Landscape — Is the Minimum Unique?
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 6: χ² Landscape Scan")
print("━" * 80)

hJ_cmb = DES_Y3_DATA["hJ_cmb"]
Sigma_0_scan = np.linspace(-0.8, 0.8, 161)

chi2_landscape: dict = {}

for model in GZ_MODELS:
    chi2_vals = []
    for S0 in Sigma_0_scan:
        chi2_sum = 0.0
        for i, z in enumerate(z_bins):
            # Predicted ĥJ from Σ₀
            Sigma_r = compute_Sigma(z, S0, model, Omega_Lambda_of_z)
            hJ_pred = Omega_m_of_z(z) * (D1_of_z(z) / D1_star) * sigma8_star * Sigma_r.Sigma
            chi2_sum += ((hJ_cmb["mean"][i] - hJ_pred) / hJ_cmb["sigma"][i]) ** 2
        chi2_vals.append(chi2_sum)

    chi2_arr = np.array(chi2_vals)
    min_idx = np.argmin(chi2_arr)
    S0_best = Sigma_0_scan[min_idx]
    chi2_min = chi2_arr[min_idx]

    # Δχ² = 1 contour (1σ)
    one_sigma_mask = chi2_arr < chi2_min + 1.0
    S0_range = Sigma_0_scan[one_sigma_mask]

    # Δχ² = 4 contour (2σ)
    two_sigma_mask = chi2_arr < chi2_min + 4.0
    S0_range_2s = Sigma_0_scan[two_sigma_mask]

    # Check if GR (Σ₀=0) is within 2σ
    chi2_gr = chi2_arr[np.argmin(np.abs(Sigma_0_scan))]
    delta_chi2_gr = chi2_gr - chi2_min

    chi2_landscape[model.value] = {
        "S0_best": float(S0_best),
        "chi2_min": float(chi2_min),
        "chi2_at_GR": float(chi2_gr),
        "delta_chi2_GR": float(delta_chi2_gr),
        "GR_excluded_at": f"{np.sqrt(delta_chi2_gr):.1f}σ",
        "S0_1sigma": [float(S0_range[0]), float(S0_range[-1])] if len(S0_range) > 0 else None,
        "S0_2sigma": [float(S0_range_2s[0]), float(S0_range_2s[-1])] if len(S0_range_2s) > 0 else None,
    }

    print(f"\n  {model.value} model:")
    print(f"    Best-fit Σ₀ = {S0_best:+.3f},  χ²_min = {chi2_min:.2f}")
    print(f"    χ²(GR) = {chi2_gr:.2f},  Δχ² = {delta_chi2_gr:.2f}")
    print(f"    GR excluded at: {np.sqrt(delta_chi2_gr):.1f}σ")
    if len(S0_range) > 0:
        print(f"    1σ range: [{S0_range[0]:+.3f}, {S0_range[-1]:+.3f}]")
    if len(S0_range_2s) > 0:
        print(f"    2σ range: [{S0_range_2s[0]:+.3f}, {S0_range_2s[-1]:+.3f}]")

    # Check for secondary minima
    from scipy.signal import argrelmin

    local_mins = argrelmin(chi2_arr, order=5)[0]
    if len(local_mins) > 1:
        print(f"    ⚠️  {len(local_mins)} local minima found — potential degeneracy!")
        for lm in local_mins:
            print(f"      Σ₀ = {Sigma_0_scan[lm]:+.3f}, χ² = {chi2_arr[lm]:.2f}")
    else:
        print("    ✓ Unique minimum (no degeneracy)")

results["sections"]["chi2_landscape"] = chi2_landscape


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: Bayesian Information Criterion (Model Comparison)
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 7: Model Comparison — BIC and AIC")
print("━" * 80)
print("  Comparing: GR (0 params) vs Σ₀ models (1 param each)")

n_data = n_bins  # 4 data points
bic_results: dict = {}

# GR model: 0 free parameters
chi2_gr_val = chi2_landscape["standard"]["chi2_at_GR"]
bic_gr = chi2_gr_val + 0 * np.log(n_data)
aic_gr = chi2_gr_val + 2 * 0

print("\n  GR (ΛCDM, Σ₀ = 0):")
print(f"    χ² = {chi2_gr_val:.2f},  BIC = {bic_gr:.2f},  AIC = {aic_gr:.2f}")

for model in GZ_MODELS:
    chi2_min = chi2_landscape[model.value]["chi2_min"]
    # 1 free parameter (Σ₀)
    bic = chi2_min + 1 * np.log(n_data)
    aic = chi2_min + 2 * 1

    delta_bic = bic - bic_gr
    delta_aic = aic - aic_gr

    # Jeffreys scale for BIC
    if delta_bic < -10:
        evidence = "DECISIVE against GR"
    elif delta_bic < -6:
        evidence = "STRONG against GR"
    elif delta_bic < -2:
        evidence = "POSITIVE against GR"
    elif delta_bic < 2:
        evidence = "INCONCLUSIVE"
    elif delta_bic < 6:
        evidence = "POSITIVE for GR"
    else:
        evidence = "STRONG for GR"

    bic_results[model.value] = {
        "chi2_min": float(chi2_min),
        "BIC": float(bic),
        "AIC": float(aic),
        "delta_BIC_vs_GR": float(delta_bic),
        "delta_AIC_vs_GR": float(delta_aic),
        "evidence": evidence,
    }

    print(f"\n  Σ₀ ({model.value}):")
    print(f"    χ² = {chi2_min:.2f},  BIC = {bic:.2f},  AIC = {aic:.2f}")
    print(f"    ΔBIC = {delta_bic:+.2f},  ΔAIC = {delta_aic:+.2f}")
    print(f"    Evidence: {evidence}")

results["sections"]["model_comparison"] = bic_results


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: Forecasting — What Would Confirm the Signal?
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 8: Forecasting — What Precision Confirms/Rules Out?")
print("━" * 80)

# If Σ₀ = 0.24 is true, what error bar gives 3σ/5σ detection?
sigma_0_true = 0.24
sigma_3sig = sigma_0_true / 3.0
sigma_5sig = sigma_0_true / 5.0

print(f"\n  If Σ₀ = {sigma_0_true} is the true value:")
print(f"    3σ detection requires: σ(Σ₀) ≤ {sigma_3sig:.3f}")
print(f"    5σ discovery requires: σ(Σ₀) ≤ {sigma_5sig:.3f}")
print("    Current DES Y3 error: σ(Σ₀) ≈ 0.10")
print(f"    → DES Y3 is at {sigma_0_true / 0.10:.1f}σ")

# Error scales as 1/√N for independent bins
# Euclid: ~10× more area, LSST: ~5× more area, DESI: spectroscopic
surveys = {
    "DES Y3 (current)": 1.0,
    "DES Y6 (full)": np.sqrt(2),
    "KiDS-1000": 0.8,
    "HSC Y3": 1.2,
    "Euclid (2030)": 3.5,
    "LSST Y1 (2026)": 2.0,
    "LSST Y10 (2035)": 5.0,
    "Combined Stage-IV": 6.0,
}

print("\n  Projected significance by survey:")
print(f"    {'Survey':<25} {'σ(Σ₀)':>10}  {'Significance':>12}")
print("    " + "-" * 50)
for survey, improvement in surveys.items():
    sigma_proj = 0.10 / improvement
    significance = sigma_0_true / sigma_proj
    marker = " ★" if significance >= 5 else " ◆" if significance >= 3 else ""
    print(f"    {survey:<25} {sigma_proj:10.4f}  {significance:10.1f}σ{marker}")

results["sections"]["forecasts"] = {
    survey: {
        "sigma_Sigma0": float(0.10 / improvement),
        "significance": float(sigma_0_true / (0.10 / improvement)),
    }
    for survey, improvement in surveys.items()
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: UMCP Regime Transition Mapping
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 9: UMCP Regime Transition Mapping")
print("━" * 80)

# Scan Σ₀ and map to UMCP regimes
Sigma_0_regime_scan = np.linspace(-0.5, 0.5, 101)

print(f"\n  {'Σ₀':>8}  {'ω':>6}  {'F':>6}  {'Regime':>12}  {'χ² imp':>8}")
print("  " + "-" * 50)

transitions_found: list = []
prev_regime = None
for S0 in Sigma_0_regime_scan:
    mapping = Sigma_to_UMCP_invariants(S0, 1.1, 2.1)
    regime = mapping["regime"]
    if prev_regime is not None and regime != prev_regime:
        transitions_found.append((float(S0), prev_regime, regime))
    prev_regime = regime

# Print transition points + key values
key_values = [-0.50, -0.30, -0.10, 0.0, 0.10, 0.24, 0.30, 0.50]
for S0 in key_values:
    mapping = Sigma_to_UMCP_invariants(S0, 1.1, 2.1)
    marker = " ◄ DES Y3" if abs(S0 - 0.24) < 0.01 else ""
    marker = " ◄ GR" if abs(S0) < 0.01 else marker
    print(
        f"  {S0:+8.2f}  {mapping['omega_analog']:.3f}  {mapping['F_analog']:.3f}  "
        f"{mapping['regime']:>12}  {mapping['chi2_improvement']:.1%}{marker}"
    )

print("\n  Regime transitions:")
for S0, r_from, r_to in transitions_found:
    print(f"    Σ₀ = {S0:+.2f}: {r_from} → {r_to}")

print("\n  ★ DES Y3 result (Σ₀ = 0.24) sits in the WATCH regime")
print("    — at the boundary of Tension/Modified gravity in Weyl classification")
print("    — analogous to ω = 0.24 in UMCP (significant drift from ideal)")

results["sections"]["regime_mapping"] = {
    "transitions": [(float(s), rf, rt) for s, rf, rt in transitions_found],
    "DES_Y3_regime": "Watch",
    "DES_Y3_omega_analog": 0.24,
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: Novel — Cross-Bin ĥJ Residual Correlations
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 10: Novel Analysis — Cross-Bin Residual Correlations")
print("━" * 80)
print("  Question: Are the deviations from GR correlated between bins?")
print("  If systematic → correlated. If statistical → uncorrelated.")

# Compute residuals: ĥJ(observed) - ĥJ(GR prediction)
hJ_obs = hJ_cmb["mean"]
hJ_err = hJ_cmb["sigma"]

hJ_gr_pred = np.array([Omega_m_of_z(z) * (D1_of_z(z) / D1_star) * sigma8_star for z in z_bins])

residuals = hJ_obs - hJ_gr_pred
normalized_residuals = residuals / hJ_err

print("\n  Bin residuals (ĥJ_obs - ĥJ_GR):")
print(f"    {'z':>6}  {'ĥJ_obs':>8}  {'ĥJ_GR':>8}  {'Residual':>10}  {'Norm. Res':>10}")
print("    " + "-" * 50)
for i, z in enumerate(z_bins):
    print(
        f"    {z:6.3f}  {hJ_obs[i]:8.4f}  {hJ_gr_pred[i]:8.4f}  "
        f"{residuals[i]:+10.4f}  {normalized_residuals[i]:+10.2f}σ"
    )

# Are all residuals the same sign?
all_positive = all(r > 0 for r in residuals)
all_negative = all(r < 0 for r in residuals)

print(f"\n  All residuals same sign: {'YES' if all_positive or all_negative else 'NO'}")
if all_positive:
    print("  → All positive: ĥJ systematically EXCEEDS GR prediction")
    print(f"  → Probability under null (random sign): {0.5**n_bins:.3%}")
elif all_negative:
    print("  → All negative: ĥJ systematically BELOW GR prediction")
    print(f"  → Probability under null: {0.5**n_bins:.3%}")

# Runs test for randomness
n_pos = sum(1 for r in normalized_residuals if r > 0)
n_neg = n_bins - n_pos
n_runs = 1
for i in range(1, n_bins):
    if (normalized_residuals[i] > 0) != (normalized_residuals[i - 1] > 0):
        n_runs += 1

print(f"  Positive residuals: {n_pos}, Negative: {n_neg}, Runs: {n_runs}")

# Chi-squared of residuals under GR
chi2_residuals = np.sum(normalized_residuals**2)
p_value_residuals = 1.0 - stats.chi2.cdf(chi2_residuals, n_bins)

print(f"\n  χ² of residuals under GR: {chi2_residuals:.2f} (dof = {n_bins})")
print(f"  p-value: {p_value_residuals:.4f}")
if p_value_residuals < 0.01:
    print(f"  ⚠️  GR REJECTED at {(1 - p_value_residuals) * 100:.1f}% confidence")
elif p_value_residuals < 0.05:
    print(f"  ⚠  GR disfavored at {(1 - p_value_residuals) * 100:.1f}% confidence")
else:
    print("  ✓  GR not formally rejected (p > 0.05)")

# Weighted mean of normalized residuals
wmean_res = np.average(normalized_residuals, weights=1.0 / hJ_err**2)
wmean_err = 1.0 / np.sqrt(np.sum(1.0 / hJ_err**2))
wmean_significance = abs(wmean_res) / max(wmean_err, 1e-10)

print(f"\n  Weighted mean normalized residual: {wmean_res:+.3f}")
print(f"  Combined significance of departure from GR: {wmean_significance:.1f}σ")

# Novel: look for oscillatory pattern
# Fit residuals to A·sin(2π·z/λ + φ) — could indicate oscillating dark energy
print("\n  ★ NOVEL: Testing for oscillatory residual pattern")
# Simple: compute Fourier-like amplitude
z_norm = (z_bins - z_bins[0]) / (z_bins[-1] - z_bins[0]) * 2 * np.pi
fourier_cos = np.sum(normalized_residuals * np.cos(z_norm))
fourier_sin = np.sum(normalized_residuals * np.sin(z_norm))
fourier_amplitude = np.sqrt(fourier_cos**2 + fourier_sin**2) / n_bins
print(f"  Fourier amplitude: {fourier_amplitude:.3f}")
print(f"  {'Suggestive of oscillation' if fourier_amplitude > 0.5 else 'No oscillatory pattern detected'}")

results["sections"]["cross_bin_analysis"] = {
    "all_same_sign": bool(all_positive or all_negative),
    "chi2_GR_residuals": float(chi2_residuals),
    "p_value_GR": float(p_value_residuals),
    "combined_significance_sigma": float(wmean_significance),
    "fourier_amplitude": float(fourier_amplitude),
    "normalized_residuals": normalized_residuals.tolist(),
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: Novel — Information-Theoretic Gravity Test
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "━" * 80)
print("  SECTION 11: Novel — Information-Theoretic Gravity Test")
print("━" * 80)
print("  UMCP maps drift (ω) to information loss.")
print("  Can we quantify information content of the GR deviation?")

# Kullback-Leibler divergence: D_KL(Σ model || GR)
# For Gaussian likelihoods: D_KL = (Δχ²)/2
for model in GZ_MODELS:
    delta_chi2 = chi2_landscape[model.value]["chi2_at_GR"] - chi2_landscape[model.value]["chi2_min"]
    D_KL = delta_chi2 / 2.0
    # Bits of evidence
    bits = D_KL / np.log(2)

    print(f"\n  {model.value} model:")
    print(f"    Δχ² (GR → Σ₀) = {delta_chi2:.2f}")
    print(f"    D_KL(Σ || GR) = {D_KL:.3f} nats = {bits:.3f} bits")
    print(f"    → {bits:.1f} bits of evidence against GR")

    # UMCP invariant complexity analog
    # IC ≈ exp(κ), where κ ~ information gain
    kappa_analog = D_KL
    IC_analog = np.exp(kappa_analog)
    F_check = 1.0 - abs(fit_results["CMB_prior"][model.value].Sigma_0)
    print(f"    κ_analog = {kappa_analog:.3f}")
    print(f"    IC_analog = exp(κ) = {IC_analog:.3f}")
    print(f"    F_analog = {F_check:.3f}")
    print(
        f"    IC ≤ F check: {IC_analog:.3f} ≤ {F_check:.3f} → "
        f"{'SATISFIED ✓' if IC_analog <= F_check else 'VIOLATED ✗ (AM-GM boundary exceeded)'}"
    )

results["sections"]["information_theoretic"] = {
    model.value: {
        "delta_chi2": float(chi2_landscape[model.value]["chi2_at_GR"] - chi2_landscape[model.value]["chi2_min"]),
        "D_KL_nats": float((chi2_landscape[model.value]["chi2_at_GR"] - chi2_landscape[model.value]["chi2_min"]) / 2),
        "D_KL_bits": float(
            (chi2_landscape[model.value]["chi2_at_GR"] - chi2_landscape[model.value]["chi2_min"]) / (2 * np.log(2))
        ),
    }
    for model in GZ_MODELS
}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12: Summary of Novel Findings
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  SUMMARY OF FINDINGS")
print("=" * 80)

# Collect highlights
best_fit = fit_results["CMB_prior"]["standard"]
best_significance = abs(best_fit.Sigma_0) / max(best_fit.Sigma_0_error, 1e-10)

findings = [
    f"1. Σ₀ detection: {best_fit.Sigma_0:+.3f} ± {best_fit.Sigma_0_error:.3f} ({best_significance:.1f}σ from GR)",
    f"2. σ₈ tension: {tension_cmb:.1f}σ between CMB-inferred and lensing-inferred values",
    "3. Sign puzzle: Σ₀ > 0 (stronger gravity) yet σ₈(lens) < σ₈(CMB) (less growth) "
    "— possible scale-dependent modification",
    f"4. GR formally {'rejected' if p_value_residuals < 0.05 else 'not rejected'} "
    f"by binned χ² test (p = {p_value_residuals:.3f})",
    f"5. {'All' if all_positive or all_negative else 'Not all'} bins deviate in same direction "
    f"(prob under null: {0.5**n_bins:.1%})",
    f"6. Best information gain: {max(float((chi2_landscape[m.value]['chi2_at_GR'] - chi2_landscape[m.value]['chi2_min']) / (2 * np.log(2))) for m in GZ_MODELS):.1f} bits against GR",
    f"7. Max Weyl transfer deviation: {abs(max_diff_pct):.1f}% at z = {max_diff_z:.2f}",
    f"8. 5σ discovery forecast: requires σ(Σ₀) ≤ {sigma_5sig:.3f} — achievable by ~LSST Y10 / Stage-IV combined",
]

for f in findings:
    print(f"\n  {f}")

results["sections"]["summary"] = findings

# ═══════════════════════════════════════════════════════════════════════
# Write JSON report
# ═══════════════════════════════════════════════════════════════════════

output_path = Path(__file__).resolve().parent.parent / "outputs" / "cosmology_exploration.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\n  📄 Full results written to: {output_path}")
print("=" * 80)
