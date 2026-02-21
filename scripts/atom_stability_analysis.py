"""Cross-scale analysis: Atom size vs nuclear stability vs particle distances.

Investigates correlations between atomic radius, islands of stability,
magic numbers, and inter-particle distances in the GCD kernel space.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
from scipy.stats import pearsonr, spearmanr

from closures.atomic_physics.cross_scale_kernel import (
    EnhancedKernelResult,
    compute_all_enhanced,
    magic_proximity,
)
from closures.atomic_physics.periodic_kernel import PropertyKernelResult, batch_compute_all

# === Imports ===
from closures.materials_science.element_database import ELEMENTS as MAT_ELEMENTS
from closures.nuclear_physics.element_data import ELEMENTS as NUC_ELEMENTS
from closures.nuclear_physics.shell_structure import MAGIC_N, MAGIC_Z, compute_shell
from closures.standard_model.subatomic_kernel import (
    compute_all_composite,
    compute_all_fundamental,
)


def section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main() -> None:
    # =========================================================
    # SECTION 1: Atom size vs nuclear stability + magic numbers
    # =========================================================
    section("1. ATOM SIZE vs NUCLEAR STABILITY")

    mat_by_z = {el.Z: el for el in MAT_ELEMENTS}
    nuc_by_z = {el.Z: el for el in NUC_ELEMENTS}

    print(f"Magic Z: {MAGIC_Z}")
    print(f"Magic N: {MAGIC_N}\n")

    # Collect data
    radii, be_per_a, magic_prox_vals, half_lives = [], [], [], []
    z_vals, symbols_list = [], []
    is_magic_z_list, is_magic_n_list = [], []

    header = f"{'Z':>3} {'Sym':>3} {'r(pm)':>7} {'BE/A':>6} {'half_life':>12} {'magZ':>5} {'magN':>5} {'N/Z':>5} {'mag_prox':>8}"
    print(header)
    print("-" * len(header))

    for z in range(1, 119):
        mel = mat_by_z.get(z)
        nel = nuc_by_z.get(z)
        if not mel or not nel:
            continue

        radius = mel.atomic_radius_pm
        bea = nel.BE_per_A
        hl = nel.half_life_s
        A = nel.A
        N = nel.N

        shell = compute_shell(z, A)
        mp = magic_proximity(z, A)

        radii.append(radius if radius else None)
        be_per_a.append(bea)
        magic_prox_vals.append(mp)
        half_lives.append(hl)
        z_vals.append(z)
        is_magic_z_list.append(shell.magic_proton)
        is_magic_n_list.append(shell.magic_neutron)
        symbols_list.append(mel.symbol)

        # Print selected elements
        show = z <= 36 or z in [50, 56, 79, 82, 83, 92, 114, 118] or shell.magic_proton or shell.doubly_magic
        if show:
            hl_str = "stable" if hl == 0 else f"{hl:.2e}"
            r_str = f"{radius:.0f}" if radius else "N/A"
            nz = N / z if z > 0 else 0
            print(
                f"{z:3d} {mel.symbol:>3} {r_str:>7} {bea:6.3f} {hl_str:>12}"
                f" {shell.magic_proton!s:>5} {shell.magic_neutron!s:>5}"
                f" {nz:5.2f} {mp:8.3f}"
            )

    # Doubly magic nuclei
    print("\n--- DOUBLY MAGIC NUCLEI ---")
    for i, z in enumerate(z_vals):
        nel = nuc_by_z[z]
        shell = compute_shell(z, nel.A)
        if shell.doubly_magic:
            r = radii[i]
            r_str = f"{r:.0f}" if r else "N/A"
            print(
                f"  Z={z:3d} ({symbols_list[i]:>2}), A={nel.A}, "
                f"r={r_str}pm, BE/A={nel.BE_per_A:.3f}, magic_prox={magic_prox_vals[i]:.3f}"
            )

    # Correlations
    valid = [
        (z_vals[i], radii[i], be_per_a[i], magic_prox_vals[i], half_lives[i])
        for i in range(len(z_vals))
        if radii[i] is not None
    ]

    z_v = np.array([v[0] for v in valid])
    r_v = np.array([v[1] for v in valid])
    bea_v = np.array([v[2] for v in valid])
    mp_v = np.array([v[3] for v in valid])
    hl_v = np.array([v[4] for v in valid])
    hl_log = np.array([math.log10(h + 1) if h > 0 else 30.0 for h in hl_v])

    print(f"\n--- CORRELATIONS (n={len(valid)} elements with known radius) ---")
    corrs = [
        ("Radius vs BE/A", r_v, bea_v),
        ("Radius vs magic_proximity", r_v, mp_v),
        ("Radius vs log(stability)", r_v, hl_log),
        ("Radius vs Z", r_v, z_v),
        ("BE/A vs magic_proximity", bea_v, mp_v),
        ("BE/A vs log(stability)", bea_v, hl_log),
        ("magic_proximity vs log(stability)", mp_v, hl_log),
    ]
    for name, x, y in corrs:
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        print(f"  {name:40s}: Pearson r={r_p:+.4f} (p={p_p:.2e}), Spearman rho={r_s:+.4f} (p={p_s:.2e})")

    # =========================================================
    # SECTION 2: Islands of stability — radius anomalies near magic numbers
    # =========================================================
    section("2. ISLANDS OF STABILITY — Radius Anomalies Near Magic Z")

    # For each magic Z, look at elements near it and compare radii
    for mz in MAGIC_Z:
        window = range(max(1, mz - 3), min(119, mz + 4))
        print(f"\n  Near magic Z={mz}:")
        for z in window:
            idx = None
            for i, zz in enumerate(z_vals):
                if zz == z:
                    idx = i
                    break
            if idx is None:
                continue
            radius = radii[idx]
            bea = be_per_a[idx]
            mp = magic_prox_vals[idx]
            marker = " *** MAGIC" if z == mz else ""
            r_str = f"{radius:.0f}" if radius else "N/A"
            print(f"    Z={z:3d} ({symbols_list[idx]:>2}) r={r_str:>5}pm BE/A={bea:.3f} magic_prox={mp:.3f}{marker}")

    # =========================================================
    # SECTION 3: Kernel-space analysis — GCD invariants by atom size
    # =========================================================
    section("3. KERNEL-SPACE: Atom Size vs GCD Invariants")

    # Use the periodic kernel (8-channel)
    pk_results = batch_compute_all()
    pk_by_z = {r.Z: r for r in pk_results}

    # Size bins: small (<120pm), medium (120-180pm), large (>180pm)
    small: list[PropertyKernelResult] = []
    medium: list[PropertyKernelResult] = []
    large: list[PropertyKernelResult] = []
    unknown: list[PropertyKernelResult] = []
    for r in pk_results:
        mel = mat_by_z.get(r.Z)
        if not mel or mel.atomic_radius_pm is None:
            unknown.append(r)
        elif mel.atomic_radius_pm < 120:
            small.append(r)
        elif mel.atomic_radius_pm < 180:
            medium.append(r)
        else:
            large.append(r)

    size_buckets: list[tuple[str, list[PropertyKernelResult]]] = [
        ("Small (<120pm)", small),
        ("Medium (120-180pm)", medium),
        ("Large (>180pm)", large),
    ]
    for label, bucket_pk in size_buckets:
        if not bucket_pk:
            continue
        f_avg = np.mean([rr.F for rr in bucket_pk])
        ic_avg = np.mean([rr.IC for rr in bucket_pk])
        gap_avg = np.mean([rr.heterogeneity_gap for rr in bucket_pk])
        s_avg = np.mean([rr.S for rr in bucket_pk])
        c_avg = np.mean([rr.C for rr in bucket_pk])
        regimes: dict[str, int] = {}
        for rr in bucket_pk:
            regimes[rr.regime] = regimes.get(rr.regime, 0) + 1
        print(
            f"  {label:20s} (n={len(bucket_pk):2d}): "
            f"<F>={f_avg:.3f}  <IC>={ic_avg:.3f}  <gap>={gap_avg:.3f}  "
            f"<S>={s_avg:.3f}  <C>={c_avg:.3f}  regimes={regimes}"
        )

    # =========================================================
    # SECTION 4: Cross-scale kernel — 12-channel with nuclear channels
    # =========================================================
    section("4. CROSS-SCALE KERNEL: Nuclear + Atomic Channels (12-ch)")

    cs_results = compute_all_enhanced()

    # Compare magic vs non-magic elements
    magic_els = [r for r in cs_results if r.is_magic]
    non_magic = [r for r in cs_results if not r.is_magic]

    print(f"  Magic elements: {len(magic_els)}")
    print(f"  Non-magic elements: {len(non_magic)}")
    print()

    magic_buckets: list[tuple[str, list[EnhancedKernelResult]]] = [
        ("Magic", magic_els),
        ("Non-magic", non_magic),
    ]
    for label, bucket_cs in magic_buckets:
        f_avg = np.mean([rr.F for rr in bucket_cs])
        ic_avg = np.mean([rr.IC for rr in bucket_cs])
        gap_avg = np.mean([rr.heterogeneity_gap for rr in bucket_cs])
        mp_avg = np.mean([rr.magic_proximity for rr in bucket_cs])
        bea_avg = np.mean([rr.BE_per_A for rr in bucket_cs])
        print(
            f"  {label:12s}: <F>={f_avg:.4f}  <IC>={ic_avg:.4f}  "
            f"<gap>={gap_avg:.4f}  <magic_prox>={mp_avg:.3f}  <BE/A>={bea_avg:.3f}"
        )

    # Per-block analysis
    print("\n  --- By electron block ---")
    block_groups: dict[str, list[EnhancedKernelResult]] = {}
    for rr in cs_results:
        b = rr.block
        if b not in block_groups:
            block_groups[b] = []
        block_groups[b].append(rr)

    for b in ["s", "p", "d", "f"]:
        if b not in block_groups:
            continue
        bucket_blk = block_groups[b]
        f_avg = np.mean([rr.F for rr in bucket_blk])
        ic_avg = np.mean([rr.IC for rr in bucket_blk])
        gap_avg = np.mean([rr.heterogeneity_gap for rr in bucket_blk])
        radii_b: list[float] = [
            mat_by_z[rr.Z].atomic_radius_pm
            for rr in bucket_blk
            if mat_by_z.get(rr.Z) and mat_by_z[rr.Z].atomic_radius_pm is not None
        ]
        r_avg = np.mean(radii_b) if radii_b else 0.0
        print(
            f"    block-{b}: n={len(bucket_blk):2d}  <F>={f_avg:.3f}  <IC>={ic_avg:.3f}  "
            f"<gap>={gap_avg:.3f}  <radius>={r_avg:.0f}pm"
        )

    # =========================================================
    # SECTION 5: INTER-PARTICLE DISTANCES in kernel space
    # =========================================================
    section("5. INTER-PARTICLE DISTANCES in Kernel Space (Standard Model)")

    fund_results = compute_all_fundamental()
    comp_results = compute_all_composite()

    # Compute pairwise Euclidean distances between fundamental particles
    fund_traces: list[np.ndarray] = []
    fund_names: list[str] = []
    for rr in fund_results:
        fund_traces.append(np.array(rr.trace_vector))
        fund_names.append(rr.symbol)

    n_fund = len(fund_traces)
    fund_dist = np.zeros((n_fund, n_fund))
    for i in range(n_fund):
        for j in range(n_fund):
            fund_dist[i, j] = np.linalg.norm(fund_traces[i] - fund_traces[j])

    # Print condensed distance matrix for key particles
    print("  --- Fundamental particle distance matrix (selected) ---")
    key_indices = list(range(min(n_fund, 17)))  # All 17 fundamentals
    print(f"  {'':>8}", end="")
    for j in key_indices:
        print(f" {fund_names[j]:>6}", end="")
    print()
    for i in key_indices:
        print(f"  {fund_names[i]:>8}", end="")
        for j in key_indices:
            print(f" {fund_dist[i, j]:6.3f}", end="")
        print()

    # Nearest neighbors for each fundamental
    print("\n  --- Nearest neighbors (fundamental) ---")
    for i in range(n_fund):
        dists = [(fund_dist[i, j], fund_names[j]) for j in range(n_fund) if i != j]
        dists.sort()
        nn1, nn2, nn3 = dists[0], dists[1], dists[2]
        print(
            f"    {fund_names[i]:>8}: "
            f"1st={nn1[1]}({nn1[0]:.3f})  "
            f"2nd={nn2[1]}({nn2[0]:.3f})  "
            f"3rd={nn3[1]}({nn3[0]:.3f})"
        )

    # Composite particles — distances from composite to fundamental
    print("\n  --- Composite-to-Fundamental distances ---")
    comp_traces: list[np.ndarray] = []
    comp_names: list[str] = []
    for rr in comp_results:
        comp_traces.append(np.array(rr.trace_vector))
        comp_names.append(rr.symbol)

    for i, ct in enumerate(comp_traces):
        dists = [(np.linalg.norm(ct - ft), fn) for ft, fn in zip(fund_traces, fund_names, strict=False)]
        dists.sort()
        nn1, nn2 = dists[0], dists[1]
        mass_str = (
            f"{comp_results[i].mass_GeV:.3f}" if comp_results[i].mass_GeV < 10 else f"{comp_results[i].mass_GeV:.1f}"
        )
        print(
            f"    {comp_names[i]:>8} ({mass_str:>6} GeV): "
            f"nearest_fund={nn1[1]}({nn1[0]:.3f}), "
            f"2nd={nn2[1]}({nn2[0]:.3f})"
        )

    # =========================================================
    # SECTION 6: CROSS-SCALE — Atom radius vs kernel-space position
    # =========================================================
    section("6. ATOM SIZE vs KERNEL-SPACE POSITION (118 elements)")

    # For each element, compute distance to every other in 12-channel space
    cs_traces: list[np.ndarray] = []
    cs_radii: list[float | None] = []
    cs_syms: list[str] = []
    cs_n_channels: list[int] = []
    for el_r in cs_results:
        cs_traces.append(np.array(el_r.trace_vector))
        mel = mat_by_z.get(el_r.Z)
        cs_radii.append(mel.atomic_radius_pm if mel and mel.atomic_radius_pm else None)
        cs_syms.append(el_r.symbol)
        cs_n_channels.append(el_r.n_channels)

    def safe_dist(a: np.ndarray, b: np.ndarray) -> float:
        """Distance using shared channels only."""
        min_len = min(len(a), len(b))
        return float(np.linalg.norm(a[:min_len] - b[:min_len]))

    # Compute average distance to all others (mean kernel distance)
    n = len(cs_traces)
    mean_dist = np.zeros(n)
    for i in range(n):
        dists = [safe_dist(cs_traces[i], cs_traces[j]) for j in range(n) if j != i]
        mean_dist[i] = np.mean(dists)

    # Correlate mean kernel distance with radius
    valid_idx = [i for i in range(n) if cs_radii[i] is not None]
    r_arr = np.array([cs_radii[i] for i in valid_idx])
    d_arr = np.array([mean_dist[i] for i in valid_idx])
    f_arr = np.array([cs_results[i].F for i in valid_idx])
    ic_arr = np.array([cs_results[i].IC for i in valid_idx])
    mp_arr = np.array([cs_results[i].magic_proximity for i in valid_idx])
    bea_arr = np.array([cs_results[i].BE_per_A for i in valid_idx])

    print("  --- Correlations: Radius vs Kernel-Space Position ---")
    for name, x, y in [
        ("Radius vs mean_kernel_distance", r_arr, d_arr),
        ("Radius vs F (12-ch)", r_arr, f_arr),
        ("Radius vs IC (12-ch)", r_arr, ic_arr),
        ("Radius vs magic_proximity", r_arr, mp_arr),
        ("Radius vs BE/A", r_arr, bea_arr),
        ("mean_kernel_dist vs F", d_arr, f_arr),
        ("mean_kernel_dist vs IC", d_arr, ic_arr),
        ("mean_kernel_dist vs magic_prox", d_arr, mp_arr),
        ("F vs IC (12-ch)", f_arr, ic_arr),
    ]:
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        print(f"    {name:40s}: r={rp:+.4f} (p={pp:.2e}) {sig:>3}  rho={rs:+.4f} (p={ps:.2e})")

    # =========================================================
    # SECTION 7: Islands of stability — kernel clustering
    # =========================================================
    section("7. KERNEL CLUSTERING: Islands of Stability in 12-ch Space")

    # Find elements with highest IC (kernel stability)
    sorted_by_ic = sorted(
        [(cs_results[i], cs_radii[i]) for i in range(n)],
        key=lambda x: x[0].IC,
        reverse=True,
    )

    print("  --- Top 20 by IC (islands of kernel stability) ---")
    print(
        f"  {'Z':>3} {'Sym':>3} {'r(pm)':>7} {'F':>6} {'IC':>6} {'gap':>6} {'BE/A':>6} {'mag_p':>6} {'regime':>8} {'magic':>5}"
    )
    for el_r, rad in sorted_by_ic[:20]:
        r_str = f"{rad:.0f}" if rad else "N/A"
        print(
            f"  {el_r.Z:3d} {el_r.symbol:>3} {r_str:>7} {el_r.F:6.3f} {el_r.IC:6.4f} "
            f"{el_r.heterogeneity_gap:6.4f} {el_r.BE_per_A:6.3f} {el_r.magic_proximity:6.3f} "
            f"{el_r.regime:>8} {'Y' if el_r.is_magic else 'N':>5}"
        )

    print("\n  --- Bottom 10 by IC (kernel instability) ---")
    for el_r, rad in sorted_by_ic[-10:]:
        r_str = f"{rad:.0f}" if rad else "N/A"
        print(
            f"  {el_r.Z:3d} {el_r.symbol:>3} {r_str:>7} {el_r.F:6.3f} {el_r.IC:6.4f} "
            f"{el_r.heterogeneity_gap:6.4f} {el_r.BE_per_A:6.3f} {el_r.magic_proximity:6.3f} "
            f"{el_r.regime:>8} {'Y' if el_r.is_magic else 'N':>5}"
        )

    # Cluster elements by regime and compare radii
    print("\n  --- Radius by regime (12-ch kernel) ---")
    regime_groups: dict[str, list[tuple[EnhancedKernelResult, float]]] = {}
    for i in range(n):
        reg = cs_results[i].regime
        if reg not in regime_groups:
            regime_groups[reg] = []
        if cs_radii[i] is not None:
            regime_groups[reg].append((cs_results[i], cs_radii[i]))

    for reg in ["Stable", "Watch", "Collapse"]:
        if reg not in regime_groups or not regime_groups[reg]:
            continue
        rads = [x[1] for x in regime_groups[reg]]
        ics = [x[0].IC for x in regime_groups[reg]]
        print(
            f"    {reg:>8}: n={len(rads):2d}  "
            f"<r>={np.mean(rads):.0f}pm (std={np.std(rads):.0f})  "
            f"<IC>={np.mean(ics):.4f}  "
            f"r_range=[{min(rads):.0f}, {max(rads):.0f}]"
        )

    # =========================================================
    # SECTION 8: Particle-to-Atom bridge — distances across scales
    # =========================================================
    section("8. CROSS-SCALE DISTANCES: Subatomic-to-Atomic Bridge")

    # For selected atoms, compute a "composite trace" that includes both
    # nuclear (from NUC_ELEMENTS) and atomic (from periodic kernel) info
    # Then compare with SM particles

    # Use the 8-ch periodic kernel traces
    pk_by_z = {pkr.Z: pkr for pkr in pk_results}

    # Key atoms near magic numbers
    key_atoms = [2, 4, 6, 8, 14, 20, 26, 28, 50, 56, 79, 82, 92]
    print("  Atom trace (8-ch) distances to nearest SM fundamental particle:")
    print(f"  {'Z':>3} {'Sym':>3} {'r(pm)':>7} {'near_particle':>14} {'dist':>6} {'2nd_near':>14} {'dist2':>6}")

    for z in key_atoms:
        pk = pk_by_z.get(z)
        mel = mat_by_z.get(z)
        if not pk or not mel:
            continue
        atom_trace = np.array(pk.trace_vector)
        # Compare with fundamental particles (8 channels each)
        dists = []
        for fi, ft in enumerate(fund_traces):
            # Pad or truncate to match dimensions
            min_len = min(len(atom_trace), len(ft))
            d = np.linalg.norm(atom_trace[:min_len] - ft[:min_len])
            dists.append((d, fund_names[fi]))
        dists.sort()
        r_str = f"{mel.atomic_radius_pm:.0f}" if mel.atomic_radius_pm else "N/A"
        print(
            f"  {z:3d} {mel.symbol:>3} {r_str:>7} "
            f"{dists[0][1]:>14}({dists[0][0]:.3f}) "
            f"{dists[1][1]:>14}({dists[1][0]:.3f})"
        )

    # =========================================================
    # SECTION 9: Grand synthesis
    # =========================================================
    section("9. GRAND SYNTHESIS")

    print("  KEY FINDINGS:")
    print()

    # Compute some summary statistics
    stable_count = sum(1 for el_r in cs_results if el_r.regime == "Stable")
    watch_count = sum(1 for el_r in cs_results if el_r.regime == "Watch")
    collapse_count = sum(1 for el_r in cs_results if el_r.regime == "Collapse")

    magic_ic = np.mean([el_r.IC for el_r in magic_els])
    nonmagic_ic = np.mean([el_r.IC for el_r in non_magic])
    ic_ratio = magic_ic / nonmagic_ic if nonmagic_ic > 0 else float("inf")

    # Average fund-fund distance
    fund_fund_avg = np.mean([fund_dist[i, j] for i in range(n_fund) for j in range(i + 1, n_fund)])

    # Average comp-fund min distance
    comp_fund_min = []
    for ct in comp_traces:
        dists = [np.linalg.norm(ct - ft) for ft in fund_traces]
        comp_fund_min.append(min(dists))
    comp_fund_avg_min = np.mean(comp_fund_min)

    print(f"  1. Regime distribution (12-ch): Stable={stable_count}, Watch={watch_count}, Collapse={collapse_count}")
    print(f"  2. Magic vs non-magic IC: {magic_ic:.4f} vs {nonmagic_ic:.4f} (ratio={ic_ratio:.2f}x)")
    print(f"  3. Avg fundamental-fundamental distance: {fund_fund_avg:.3f}")
    print(f"  4. Avg composite→nearest fundamental distance: {comp_fund_avg_min:.3f}")
    print(f"  5. Total elements analyzed: {n}")
    print(f"  6. Elements with known radius: {len(valid_idx)}")

    # Radius vs stability final insight
    stable_r = [cs_radii[i] for i in range(n) if cs_results[i].regime == "Stable" and cs_radii[i]]
    watch_r = [cs_radii[i] for i in range(n) if cs_results[i].regime == "Watch" and cs_radii[i]]
    collapse_r = [cs_radii[i] for i in range(n) if cs_results[i].regime == "Collapse" and cs_radii[i]]

    if stable_r:
        print("\n  RADIUS vs REGIME:")
        print(f"    Stable:   <r>={np.mean(stable_r):.0f}pm  (range {min(stable_r):.0f}–{max(stable_r):.0f})")
    if watch_r:
        print(f"    Watch:    <r>={np.mean(watch_r):.0f}pm  (range {min(watch_r):.0f}–{max(watch_r):.0f})")
    if collapse_r:
        print(f"    Collapse: <r>={np.mean(collapse_r):.0f}pm  (range {min(collapse_r):.0f}–{max(collapse_r):.0f})")


if __name__ == "__main__":
    main()
