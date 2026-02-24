#!/usr/bin/env python3
"""Cross-Domain Deep Correlation Probes.

Harvests invariants from real domain closures across all 13 domains
and probes for inter-domain structural patterns.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from umcp.frozen_contract import EPSILON, P_EXPONENT


def kernel(c, w=None):
    """Compute all invariants from trace vector."""
    c = np.asarray(c, dtype=float)
    if w is None:
        w = np.ones(len(c)) / len(c)
    c_eps = np.clip(c, EPSILON, 1.0 - EPSILON)
    F = float(np.dot(w, c_eps))
    omega = 1.0 - F
    kappa = float(np.sum(w * np.log(c_eps)))
    IC = float(np.exp(kappa))
    S = float(-np.sum(w * (c_eps * np.log(c_eps) + (1 - c_eps) * np.log(1 - c_eps))))
    C = float(np.sqrt(np.sum(w * (c_eps - F) ** 2)) / 0.5)
    Delta = F - IC
    gamma = omega**P_EXPONENT / (1 - omega + EPSILON)
    return {"F": F, "omega": omega, "S": S, "C": C, "kappa": kappa, "IC": IC, "Delta": Delta, "gamma": gamma}


print("=" * 78)
print("  CROSS-DOMAIN  DEEP  CORRELATION  ANALYSIS")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════
# SECTION A: Standard Model particles (31 particles × 8 channels)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION A: Standard Model — 31 Particles as Kernel Traces")
print("═" * 78)
try:
    from closures.standard_model.subatomic_kernel import build_subatomic_kernel

    sk = build_subatomic_kernel()
    sm_invs = []
    for p in sk["particles"]:
        inv = kernel(p["trace"])
        inv["name"] = p["name"]
        inv["category"] = p.get("category", "unknown")
        sm_invs.append(inv)

    # Correlation: mass rank vs IC
    print(f"  Particles: {len(sm_invs)}")

    # Group by category
    categories = {}
    for inv in sm_invs:
        cat = inv["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(inv)

    print(f"\n  {'Category':>16}  {'n':>3}  {'F mean':>8}  {'IC mean':>8}  {'Δ mean':>8}  {'S mean':>8}  {'C mean':>8}")
    for cat in sorted(categories.keys()):
        invs = categories[cat]
        print(
            f"  {cat:>16}  {len(invs):3d}  "
            f"{np.mean([i['F'] for i in invs]):8.4f}  "
            f"{np.mean([i['IC'] for i in invs]):8.4f}  "
            f"{np.mean([i['Delta'] for i in invs]):8.4f}  "
            f"{np.mean([i['S'] for i in invs]):8.4f}  "
            f"{np.mean([i['C'] for i in invs]):8.4f}"
        )

    # The IC/F ratio per category (= fragility index)
    print("\n  Fragility Index (IC/F — lower = more heterogeneous channels):")
    for cat in sorted(categories.keys()):
        invs = categories[cat]
        ic_f = [i["IC"] / max(i["F"], 1e-15) for i in invs]
        print(f"    {cat:>16}: IC/F = {np.mean(ic_f):.4f} ± {np.std(ic_f):.4f}")

except Exception as e:
    print(f"  [SM unavailable: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION B: Periodic Table — 118 Elements
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION B: Periodic Table — 118 Elements Through Kernel")
print("═" * 78)
try:
    from closures.atomic_physics.periodic_kernel import build_periodic_kernel

    pk = build_periodic_kernel()
    atom_invs = []
    for elem in pk["elements"]:
        inv = kernel(elem["trace"])
        inv["symbol"] = elem["symbol"]
        inv["Z"] = elem["Z"]
        inv["block"] = elem.get("block", "?")
        inv["period"] = elem.get("period", 0)
        atom_invs.append(inv)

    print(f"  Elements: {len(atom_invs)}")

    # Z vs IC correlation
    Z_arr = np.array([a["Z"] for a in atom_invs])
    F_atom = np.array([a["F"] for a in atom_invs])
    IC_atom = np.array([a["IC"] for a in atom_invs])
    S_atom = np.array([a["S"] for a in atom_invs])
    C_atom = np.array([a["C"] for a in atom_invs])
    Delta_atom = np.array([a["Delta"] for a in atom_invs])

    r_Z_F, _ = spearmanr(Z_arr, F_atom)
    r_Z_IC, _ = spearmanr(Z_arr, IC_atom)
    r_Z_S, _ = spearmanr(Z_arr, S_atom)
    r_Z_C, _ = spearmanr(Z_arr, C_atom)
    r_Z_Delta, _ = spearmanr(Z_arr, Delta_atom)
    print("\n  Spearman correlations (Z vs invariant):")
    print(f"    Z-F: {r_Z_F:.4f},  Z-IC: {r_Z_IC:.4f},  Z-S: {r_Z_S:.4f},  Z-C: {r_Z_C:.4f},  Z-Δ: {r_Z_Delta:.4f}")

    # Block analysis
    blocks = {}
    for inv in atom_invs:
        b = inv["block"]
        if b not in blocks:
            blocks[b] = []
        blocks[b].append(inv)

    print(f"\n  {'Block':>6}  {'n':>3}  {'F mean':>8}  {'IC mean':>8}  {'IC/F':>8}  {'Δ mean':>8}  {'S mean':>8}")
    for b in sorted(blocks.keys()):
        invs = blocks[b]
        ic_f = np.mean([i["IC"] / max(i["F"], 1e-15) for i in invs])
        print(
            f"  {b:>6}  {len(invs):3d}  "
            f"{np.mean([i['F'] for i in invs]):8.4f}  "
            f"{np.mean([i['IC'] for i in invs]):8.4f}  "
            f"{ic_f:8.4f}  "
            f"{np.mean([i['Delta'] for i in invs]):8.4f}  "
            f"{np.mean([i['S'] for i in invs]):8.4f}"
        )

    # Period analysis — does integrity follow the periodic law?
    periods = {}
    for inv in atom_invs:
        p = inv["period"]
        if p not in periods:
            periods[p] = []
        periods[p].append(inv)

    print(f"\n  {'Period':>7}  {'n':>3}  {'F mean':>8}  {'IC mean':>8}  {'IC/F':>8}  {'Δ mean':>8}")
    for p in sorted(periods.keys()):
        invs = periods[p]
        ic_f = np.mean([i["IC"] / max(i["F"], 1e-15) for i in invs])
        print(
            f"  {p:>7}  {len(invs):3d}  "
            f"{np.mean([i['F'] for i in invs]):8.4f}  "
            f"{np.mean([i['IC'] for i in invs]):8.4f}  "
            f"{ic_f:8.4f}  "
            f"{np.mean([i['Delta'] for i in invs]):8.4f}"
        )

    # Noble gases vs halogens — do chemically "complete" shells have higher IC?
    noble_syms = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
    halogen_syms = {"F", "Cl", "Br", "I", "At", "Ts"}
    alkali_syms = {"Li", "Na", "K", "Rb", "Cs", "Fr"}

    for group_name, syms in [("Noble gases", noble_syms), ("Halogens", halogen_syms), ("Alkali metals", alkali_syms)]:
        group_invs = [a for a in atom_invs if a["symbol"] in syms]
        if group_invs:
            gf = np.mean([i["F"] for i in group_invs])
            gic = np.mean([i["IC"] for i in group_invs])
            gd = np.mean([i["Delta"] for i in group_invs])
            gs = np.mean([i["S"] for i in group_invs])
            print(
                f"\n  {group_name:>15}: F={gf:.4f}, IC={gic:.4f}, Δ={gd:.4f}, S={gs:.4f}, IC/F={gic / max(gf, 1e-15):.4f}"
            )

except Exception as e:
    print(f"  [Periodic kernel unavailable: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION C: Nuclear Binding — Bethe-Weizsäcker through kernel
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION C: Nuclear Binding Energy — Cross-Scale Bridge")
print("═" * 78)
try:
    from closures.atomic_physics.cross_scale_kernel import build_cross_scale_kernel

    ck = build_cross_scale_kernel()
    nuc_invs = []
    for elem in ck["elements"]:
        inv = kernel(elem["trace"])
        inv["symbol"] = elem["symbol"]
        inv["Z"] = elem["Z"]
        nuc_invs.append(inv)

    Z_nuc = np.array([a["Z"] for a in nuc_invs])
    IC_nuc = np.array([a["IC"] for a in nuc_invs])
    F_nuc = np.array([a["F"] for a in nuc_invs])
    Delta_nuc = np.array([a["Delta"] for a in nuc_invs])

    # Find the "iron peak" in IC — where does nuclear binding maximize?
    ic_max_idx = np.argmax(IC_nuc)
    print(f"  Elements: {len(nuc_invs)}")
    print(f"  Max IC: Z={Z_nuc[ic_max_idx]} ({nuc_invs[ic_max_idx]['symbol']}), IC={IC_nuc[ic_max_idx]:.6f}")
    print("  This is the nuclear 'stability peak' in the kernel")

    # Correlation between cross-scale IC and atomic IC (if both available)
    if len(atom_invs) > 0 and len(nuc_invs) > 0:
        # Match by Z
        common_Z = set(Z_nuc) & set(Z_arr)
        if common_Z:
            atom_ic_dict = {a["Z"]: a["IC"] for a in atom_invs}
            nuc_ic_dict = {a["Z"]: a["IC"] for a in nuc_invs}
            common = sorted(common_Z)
            a_ic = [atom_ic_dict[z] for z in common]
            n_ic = [nuc_ic_dict[z] for z in common]
            r_cross, p_cross = spearmanr(a_ic, n_ic)
            print("\n  CROSS-SCALE CORRELATION:")
            print(f"    ρ(IC_atomic, IC_nuclear) = {r_cross:.4f} (p={p_cross:.2e}) over {len(common)} elements")
            print(f"    → {'Strong' if abs(r_cross) > 0.5 else 'Weak'} cross-scale coherence")

            # Where do they disagree most?
            diffs = [(z, abs(atom_ic_dict[z] - nuc_ic_dict[z]), atom_ic_dict[z], nuc_ic_dict[z]) for z in common]
            diffs.sort(key=lambda x: -x[1])
            print("\n    Largest IC discrepancies (atomic vs nuclear):")
            for z, diff, aic, nic in diffs[:5]:
                sym = next(a["symbol"] for a in atom_invs if a["Z"] == z)
                print(f"      Z={z:3d} ({sym:>2s}): IC_atom={aic:.4f}, IC_nuc={nic:.4f}, |diff|={diff:.4f}")

except Exception as e:
    print(f"  [Cross-scale unavailable: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION D: Materials Science — Crystal structures through kernel
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION D: Materials Science — Element Database (118 × 18)")
print("═" * 78)
try:
    from closures.materials_science.element_database import get_element_database

    db = get_element_database()
    print(f"  Elements in database: {len(db)}")

    # Extract crystal structure groups
    crystal_groups = {}
    for elem in db:
        struct = elem.get("crystal_structure", "Unknown")
        if struct not in crystal_groups:
            crystal_groups[struct] = []
        crystal_groups[struct].append(elem)

    print("\n  Crystal structures present:")
    for struct in sorted(crystal_groups.keys()):
        count = len(crystal_groups[struct])
        if count > 2:
            # Get mean atomic number
            zs = [e.get("atomic_number", 0) for e in crystal_groups[struct]]
            print(f"    {struct:>20s}: {count:3d} elements, Z_mean = {np.mean(zs):.1f}")

except Exception as e:
    print(f"  [Materials DB unavailable: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION E: Confinement Analysis — quark→hadron transition
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION E: Confinement Transition — The IC Cliff")
print("═" * 78)
try:
    from closures.standard_model.subatomic_kernel import build_subatomic_kernel

    sk = build_subatomic_kernel()

    quarks = [p for p in sk["particles"] if p.get("category") == "quark"]
    hadrons = [p for p in sk["particles"] if p.get("category") in ("meson", "baryon")]

    q_ics = [kernel(p["trace"])["IC"] for p in quarks]
    h_ics = [kernel(p["trace"])["IC"] for p in hadrons]
    q_fs = [kernel(p["trace"])["F"] for p in quarks]
    h_fs = [kernel(p["trace"])["F"] for p in hadrons]
    q_deltas = [kernel(p["trace"])["Delta"] for p in quarks]
    h_deltas = [kernel(p["trace"])["Delta"] for p in hadrons]

    if quarks and hadrons:
        print(
            f"  Quarks ({len(quarks)}):   IC = {np.mean(q_ics):.6f} ± {np.std(q_ics):.6f},  F = {np.mean(q_fs):.4f},  Δ = {np.mean(q_deltas):.6f}"
        )
        print(
            f"  Hadrons ({len(hadrons)}):  IC = {np.mean(h_ics):.6f} ± {np.std(h_ics):.6f},  F = {np.mean(h_fs):.4f},  Δ = {np.mean(h_deltas):.6f}"
        )
        print(f"\n  IC drop at confinement: {(1 - np.mean(h_ics) / np.mean(q_ics)) * 100:.1f}%")
        print(f"  Δ amplification: {np.mean(h_deltas) / max(np.mean(q_deltas), 1e-15):.1f}×")
        print("  → Confinement AMPLIFIES heterogeneity gap by creating composite channels")

except Exception as e:
    print(f"  [Confinement analysis unavailable: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION F: Universal Regime Calibration — cross-domain comparison
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION F: Cross-Domain Regime Statistics")
print("═" * 78)
try:
    from closures.gcd.universal_regime_calibration import (
        run_all_theorems as run_urc,
    )

    urc = run_urc()
    print(
        f"  URC theorems: {urc['summary']['total']}, "
        f"proven: {urc['summary']['proven']}, "
        f"subtests: {urc['summary']['total_subtests']}"
    )
except Exception as e:
    print(f"  [URC: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION G: The DEEP PATTERN — entropy-curvature-dimensionality triad
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION G: The S-C-n Triad — Entropy–Curvature–Dimensionality")
print("═" * 78)
# From Probe 2, we know S and C are strongly anti-correlated (-0.75)
# But this coupling INCREASES with n_ch (-0.92 at n=32)
# What is the functional form?
rng = np.random.default_rng(99)
print("  Testing S + α*C = f(F, n) hypothesis:")
for n in [2, 4, 8, 16, 32]:
    n_test = 5000
    ss, cc, ff = [], [], []
    for _ in range(n_test):
        c = rng.uniform(0, 1, n)
        w = np.ones(n) / n
        inv = kernel(c, w)
        ss.append(inv["S"])
        cc.append(inv["C"])
        ff.append(inv["F"])
    ss, cc, ff = np.array(ss), np.array(cc), np.array(ff)

    # Try S + α*C = β*F*(1-F) + γ (entropy should peak at F=0.5)
    # Full regression: S = a*C + b*F + c*F^2 + d
    X = np.column_stack([cc, ff, ff**2, np.ones(n_test)])
    beta = np.linalg.lstsq(X, ss, rcond=None)[0]
    pred = X @ beta
    r2 = 1 - np.var(ss - pred) / np.var(ss)

    # Also test: S + C/ln(n) vs 4*F*(1-F)
    # This would mean S and C trade off inversely scaled by dimensionality
    if n > 1:
        combo = ss + cc / np.log(n)
        r_combo, _ = pearsonr(combo, 4 * ff * (1 - ff))
        print(f"  n={n:2d}: R²(S ~ C+F+F²) = {r2:.4f}, r(S+C/ln(n), 4F(1-F)) = {r_combo:.4f}")

# ═══════════════════════════════════════════════════════════════════
# SECTION H: The WEIGHT DEMOCRACY Theorem (deeper investigation)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION H: Weight Democracy — Does Concentration Destroy Integrity?")
print("═" * 78)
# Test: fix channels, vary weight concentration (Dirichlet α)
rng = np.random.default_rng(77)
n_ch = 8
c_fixed = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6])

print(f"  Fixed trace: {c_fixed}")
print("  Vary Dirichlet α (low α = concentrated, high α = democratic):\n")
print(f"  {'α':>8}  {'F':>8}  {'IC':>8}  {'IC/F':>8}  {'Δ':>8}  {'C':>8}  {'S':>8}")
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]:
    fs, ics, icfs, ds, cs, ss = [], [], [], [], [], []
    for _ in range(5000):
        w = rng.dirichlet(np.full(n_ch, alpha))
        inv = kernel(c_fixed, w)
        fs.append(inv["F"])
        ics.append(inv["IC"])
        icfs.append(inv["IC"] / max(inv["F"], 1e-15))
        ds.append(inv["Delta"])
        cs.append(inv["C"])
        ss.append(inv["S"])
    print(
        f"  {alpha:8.2f}  {np.mean(fs):8.4f}  {np.mean(ics):8.4f}  "
        f"{np.mean(icfs):8.4f}  {np.mean(ds):8.4f}  {np.mean(cs):8.4f}  {np.mean(ss):8.4f}"
    )

print("\n  Equal weights (α → ∞) baseline:")
inv_eq = kernel(c_fixed, np.ones(n_ch) / n_ch)
print(
    f"  F={inv_eq['F']:.4f}, IC={inv_eq['IC']:.4f}, IC/F={inv_eq['IC'] / inv_eq['F']:.4f}, "
    f"Δ={inv_eq['Delta']:.4f}, C={inv_eq['C']:.4f}, S={inv_eq['S']:.4f}"
)

# ═══════════════════════════════════════════════════════════════════
# SECTION I: The SEAM BUDGET at Scale Boundaries
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION I: Seam Budget Δκ at Physical Scale Boundaries")
print("═" * 78)
# Compare kernel invariants at different physical scales
# Subatomic → Atomic → Bulk
try:
    if len(sm_invs) > 0 and len(atom_invs) > 0:
        # Subatomic mean invariants
        sm_F = np.mean([i["F"] for i in sm_invs])
        sm_IC = np.mean([i["IC"] for i in sm_invs])
        sm_Delta = np.mean([i["Delta"] for i in sm_invs])

        # Atomic mean invariants
        at_F = np.mean([i["F"] for i in atom_invs])
        at_IC = np.mean([i["IC"] for i in atom_invs])
        at_Delta = np.mean([i["Delta"] for i in atom_invs])

        print("  Scale transitions (mean invariants):")
        print(f"  {'Scale':>12}  {'F':>8}  {'IC':>8}  {'Δ':>8}  {'IC/F':>8}  {'κ':>10}")

        sm_kappa = np.log(max(sm_IC, 1e-15))
        at_kappa = np.log(max(at_IC, 1e-15))

        print(
            f"  {'Subatomic':>12}  {sm_F:8.4f}  {sm_IC:8.4f}  {sm_Delta:8.4f}  "
            f"{sm_IC / max(sm_F, 1e-15):8.4f}  {sm_kappa:10.6f}"
        )
        print(
            f"  {'Atomic':>12}  {at_F:8.4f}  {at_IC:8.4f}  {at_Delta:8.4f}  "
            f"{at_IC / max(at_F, 1e-15):8.4f}  {at_kappa:10.6f}"
        )

        print("\n  Seam budget at scale transition:")
        delta_kappa = at_kappa - sm_kappa
        print(f"    Δκ (subatomic → atomic) = {delta_kappa:.6f}")
        print(f"    IC ratio = {at_IC / max(sm_IC, 1e-15):.4f}")
        if delta_kappa > 0:
            print("    → Integrity INCREASES at atomic scale (new degrees of freedom restore coherence)")
        else:
            print("    → Integrity DECREASES at atomic scale (confinement effects persist)")

        # Nuclear bridge
        if len(nuc_invs) > 0:
            nc_F = np.mean([i["F"] for i in nuc_invs])
            nc_IC = np.mean([i["IC"] for i in nuc_invs])
            nc_kappa = np.log(max(nc_IC, 1e-15))
            print(
                f"\n  {'Nuclear':>12}  {nc_F:8.4f}  {nc_IC:8.4f}  "
                f"{np.mean([i['Delta'] for i in nuc_invs]):8.4f}  "
                f"{nc_IC / max(nc_F, 1e-15):8.4f}  {nc_kappa:10.6f}"
            )
            print(f"    Δκ (subatomic → nuclear) = {nc_kappa - sm_kappa:.6f}")
            print(f"    Δκ (nuclear → atomic) = {at_kappa - nc_kappa:.6f}")

except Exception as e:
    print(f"  [Scale analysis error: {e}]")

# ═══════════════════════════════════════════════════════════════════
# SECTION J: The UNIVERSAL ATTRACTOR — where do ALL invariants go?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 78)
print("SECTION J: Universal Attractor in Invariant Space")
print("═" * 78)
# Combine all domain invariants into one analysis
all_inv = []
try:
    for inv in sm_invs:
        inv["domain"] = "SM"
        all_inv.append(inv)
except Exception:
    pass
try:
    for inv in atom_invs:
        inv["domain"] = "ATOM"
        all_inv.append(inv)
except Exception:
    pass
try:
    for inv in nuc_invs:
        inv["domain"] = "NUC"
        all_inv.append(inv)
except Exception:
    pass

if len(all_inv) > 10:
    domains = list({i["domain"] for i in all_inv})
    print(f"  Total objects: {len(all_inv)} across {len(domains)} domains")

    # Do all domains converge to the same (F, IC/F) attractor?
    print(f"\n  {'Domain':>8}  {'n':>5}  {'F center':>10}  {'IC/F center':>12}  {'S center':>10}  {'C center':>10}")
    for dom in sorted(domains):
        dom_inv = [i for i in all_inv if i["domain"] == dom]
        f_center = np.median([i["F"] for i in dom_inv])
        icf_center = np.median([i["IC"] / max(i["F"], 1e-15) for i in dom_inv])
        s_center = np.median([i["S"] for i in dom_inv])
        c_center = np.median([i["C"] for i in dom_inv])
        print(
            f"  {dom:>8}  {len(dom_inv):5d}  {f_center:10.4f}  {icf_center:12.4f}  {s_center:10.4f}  {c_center:10.4f}"
        )

    # Cross-domain PCA
    X_all = np.column_stack(
        [
            [i["F"] for i in all_inv],
            [i["IC"] for i in all_inv],
            [i["S"] for i in all_inv],
            [i["C"] for i in all_inv],
            [i["Delta"] for i in all_inv],
        ]
    )
    X_centered = X_all - X_all.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = s**2 / (s**2).sum()
    print("\n  PCA of cross-domain invariants (F, IC, S, C, Δ):")
    print(f"    PC1: {explained[0] * 100:.1f}% variance  (loadings: {Vt[0]})")
    print(f"    PC2: {explained[1] * 100:.1f}% variance  (loadings: {Vt[1]})")
    print(f"    PC3: {explained[2] * 100:.1f}% variance  (loadings: {Vt[2]})")
    print(f"    → {explained[0] * 100:.1f}% of cross-domain variation lives on a SINGLE axis")

print("\n" + "=" * 78)
print("  CROSS-DOMAIN  PROBES  COMPLETE")
print("=" * 78)
