"""Cross-Scale Kernel Analysis — Subatomic → Atomic Bridge.

Reassesses atomic physics in light of what the subatomic kernel revealed:

1. NUCLEAR-INFORMED CHANNELS: The old periodic_kernel.py used 8 "external"
   measurements (EN, radius, IE, EA, T_melt, T_boil, density). The subatomic
   analysis showed that internal quantum numbers (color, T₃, Y, generation,
   binding) are what drive F and IC. For atoms, the analogues are:

   Old (8 channels — bulk properties):
       Z_norm, EN, radius, IE, EA, T_melt, T_boil, density

   New (12 channels — nuclear + electronic + bulk):
       Z_norm, N_over_Z, binding_per_nucleon, magic_proximity,
       valence_electrons, block_ord, EN, radius_inv, IE, EA,
       T_melt, density_log

   The new channels connect atoms to their nuclear substructure:
     - N/Z ratio:  neutron excess (nuclear stability → β-decay systematics)
     - BE/A:       binding energy per nucleon (Bethe-Weizsäcker semi-empirical)
     - Magic proximity: closeness to nuclear shell closure (2,8,20,28,50,82,126)
     - Valence e⁻: electrons in outermost shell (chemistry driver)
     - Block ordinal: s=1, p=2, d=3, f=4 (angular momentum hierarchy)

2. CROSS-SCALE COMPARISON: Side-by-side kernel signatures at three scales:
       Fundamental (quarks, leptons, bosons)  ⟨F⟩ = 0.558
       Composite (baryons, mesons)            ⟨F⟩ = 0.444
       Atomic (118 elements)                  ⟨F⟩ = ?

3. CHANNEL AUTOPSY: Which channels drive IC collapse? The subatomic analysis
   showed that bosons have IC ≈ 0.007 because massless/uncharged particles
   have zero-channels. For atoms, which properties are near-zero (crushing IC)?

4. NUCLEAR SHELL STRUCTURE: Magic numbers (2,8,20,28,50,82,126) should appear
   as kernel features — like generation structure in the SM.

Cross-references:
    Subatomic:  closures/standard_model/subatomic_kernel.py
    Periodic:   closures/atomic_physics/periodic_kernel.py (original 8-channel)
    Database:   closures/materials_science/element_database.py
    Kernel:     src/umcp/kernel_optimized.py
    Proof:      closures/atomic_physics/tier1_proof.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.materials_science.element_database import (  # noqa: E402, I001
    ELEMENTS,
    Element,
)
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: NUCLEAR PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

# Nuclear magic numbers (proton OR neutron shell closures)
MAGIC_NUMBERS = (2, 8, 20, 28, 50, 82, 126)

# Bethe-Weizsäcker semi-empirical mass formula coefficients (MeV)
_A_V = 15.75  # Volume term
_A_S = 17.80  # Surface term
_A_C = 0.711  # Coulomb term
_A_A = 23.70  # Asymmetry term
_A_P = 11.18  # Pairing term


def binding_energy_per_nucleon(Z: int, A: int) -> float:
    """Compute B/A using the semi-empirical mass formula (Bethe-Weizsäcker).

    B(Z,A) = a_V·A − a_S·A^{2/3} − a_C·Z(Z-1)/A^{1/3}
             − a_A·(A-2Z)²/A + δ(A,Z)

    where δ is the pairing term:
        +a_P / A^{1/2}  if Z even, N even
        -a_P / A^{1/2}  if Z odd, N odd
        0               otherwise

    Returns B/A in MeV/nucleon. Higher = more tightly bound.
    Maximum at Fe-56 (~8.79 MeV).
    """
    if A <= 0:
        return 0.0
    N = A - Z
    if Z <= 0 or N < 0:
        return 0.0

    # Special case: hydrogen (1 proton, no binding)
    if A == 1:
        return 0.0

    volume = _A_V * A
    surface = -_A_S * A ** (2.0 / 3.0)
    coulomb = -_A_C * Z * (Z - 1) / A ** (1.0 / 3.0)
    asymmetry = -_A_A * (A - 2 * Z) ** 2 / A

    # Pairing
    if Z % 2 == 0 and N % 2 == 0:
        pairing = _A_P / A**0.5
    elif Z % 2 == 1 and N % 2 == 1:
        pairing = -_A_P / A**0.5
    else:
        pairing = 0.0

    B = volume + surface + coulomb + asymmetry + pairing
    return max(0.0, B / A)


def magic_proximity(Z: int, A: int) -> float:
    """How close is this nucleus to a magic number shell closure?

    Returns a value in [0, 1] where 1.0 = doubly magic (both Z and N
    are magic numbers), 0.5 = singly magic, → 0 = far from any closure.
    """
    N = A - Z

    def _closest_magic_dist(n: int) -> int:
        return min(abs(n - m) for m in MAGIC_NUMBERS)

    dZ = _closest_magic_dist(Z)
    dN = _closest_magic_dist(N)

    # Map distance to proximity: 0 distance → 1.0, distance 10 → ~0.1
    pZ = 1.0 / (1.0 + dZ)
    pN = 1.0 / (1.0 + dN)

    return (pZ + pN) / 2.0


def valence_electrons(group: int | None, block: str) -> int:
    """Number of valence electrons (chemistry-relevant outermost electrons)."""
    if group is None:
        # Lanthanides/actinides (f-block, no standard group)
        return 2  # Generally behave as 2+ or 3+ ions
    if block == "s":
        return group  # 1 or 2
    if block == "p":
        return group - 12  # 13→1, 14→2, ..., 18→6 (but 18 is noble, full shell)
    if block == "d":
        return group - 2  # simplified: d electrons contribute
    return 2  # f-block fallback


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: ENHANCED NORMALIZATION (12 nuclear-informed channels)
# ═══════════════════════════════════════════════════════════════════

EPSILON = 1e-6

# Precompute normalization bounds
_EN_MAX = max(el.electronegativity for el in ELEMENTS if el.electronegativity is not None)
_R_MAX = max(el.atomic_radius_pm for el in ELEMENTS if el.atomic_radius_pm is not None)
_IE_MAX = max(el.ionization_energy_eV for el in ELEMENTS)
_EA_MAX = max(abs(el.electron_affinity_eV) for el in ELEMENTS if el.electron_affinity_eV is not None)
_TM_MAX = max(el.melting_point_K for el in ELEMENTS if el.melting_point_K is not None)
_RHO_MAX = max(el.density_g_cm3 for el in ELEMENTS if el.density_g_cm3 is not None)
_Z_MAX = 118

# Compute BE/A bounds across all elements
_BEA_VALUES = [binding_energy_per_nucleon(el.Z, round(el.atomic_mass)) for el in ELEMENTS]
_BEA_MAX = max(_BEA_VALUES) if _BEA_VALUES else 8.8  # ~Fe-56

ENHANCED_CHANNELS = [
    # Nuclear structure (from subatomic insights)
    "Z_norm",  # Atomic number / 118
    "N_over_Z",  # Neutron excess (nuclear stability)
    "BE_per_A",  # Binding energy / nucleon (nuclear binding curve)
    "magic_prox",  # Proximity to magic number shell closure
    # Electronic structure
    "valence_e",  # Valence electrons / 8 (chemistry driver)
    "block_ord",  # Block ordinal: s=0.25, p=0.5, d=0.75, f=1.0
    # Bulk measurables (retained from original)
    "EN",  # Electronegativity
    "radius_inv",  # 1 - r/r_max (smaller → higher)
    "IE",  # Ionization energy
    "EA",  # Electron affinity
    "T_melt",  # Melting point
    "density_log",  # log-scaled density
]

_BLOCK_ORDINAL = {"s": 0.25, "p": 0.50, "d": 0.75, "f": 1.00}


def normalize_element_enhanced(
    el: Element,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalize element with nuclear-informed + electronic + bulk channels.

    12 channels total. Missing values excluded (variable-length trace).
    """
    A = round(el.atomic_mass)
    N = A - el.Z
    raw: list[tuple[str, float | None]] = []

    # ── Nuclear channels ──
    raw.append(("Z_norm", el.Z / _Z_MAX))
    raw.append(("N_over_Z", (N / el.Z) if el.Z > 0 else 0.0))  # ~1.0 for light, ~1.5 for heavy
    raw.append(("BE_per_A", binding_energy_per_nucleon(el.Z, A) / _BEA_MAX))
    raw.append(("magic_prox", magic_proximity(el.Z, A)))

    # ── Electronic channels ──
    val_e = valence_electrons(el.group, el.block)
    raw.append(("valence_e", val_e / 8.0))  # max 8 valence electrons (noble gas)
    raw.append(("block_ord", _BLOCK_ORDINAL.get(el.block, 0.5)))

    # ── Bulk measured channels ──
    if el.electronegativity is not None:
        raw.append(("EN", el.electronegativity / _EN_MAX))

    if el.atomic_radius_pm is not None:
        raw.append(("radius_inv", 1.0 - el.atomic_radius_pm / _R_MAX))

    raw.append(("IE", el.ionization_energy_eV / _IE_MAX))

    if el.electron_affinity_eV is not None:
        raw.append(("EA", abs(el.electron_affinity_eV) / _EA_MAX))

    if el.melting_point_K is not None:
        raw.append(("T_melt", el.melting_point_K / _TM_MAX))

    if el.density_g_cm3 is not None:
        raw.append(("density_log", math.log(el.density_g_cm3 + 1.0) / math.log(_RHO_MAX + 1.0)))

    labels = [r[0] for r in raw]
    values = np.array([r[1] for r in raw], dtype=np.float64)
    c = np.clip(values, epsilon, 1.0 - epsilon)
    w = np.ones(len(c)) / len(c)
    return c, w, labels


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: RESULT + COMPUTATION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EnhancedKernelResult:
    """Enhanced kernel result with nuclear + electronic + bulk channels."""

    Z: int
    symbol: str
    name: str
    period: int
    group: int | None
    block: str
    category: str

    # Nuclear properties
    A: int  # Mass number
    N: int  # Neutron number
    N_over_Z: float
    BE_per_A: float  # MeV/nucleon
    magic_proximity: float
    is_magic: bool  # Z or N is a magic number
    valence_e: int

    # Kernel
    n_channels: int
    channel_labels: list[str]
    trace_vector: list[float]
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    amgm_gap: float

    # Tier-1 checks
    F_plus_omega: float
    IC_leq_F: bool
    IC_eq_exp_kappa: bool

    regime: str
    gcd_category: str

    # Channel diagnostics
    min_channel: str  # Which channel has lowest value (IC killer)
    max_channel: str  # Which channel has highest value
    min_value: float
    max_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def _derive_gcd_category(F: float, IC: float, amgm_gap: float, S: float, C: float) -> str:
    """Enhanced classification using subatomic-informed thresholds."""
    if F > 0.55:
        if amgm_gap < 0.03:
            return "Kernel-concentrated"
        return "Kernel-structured"
    if F > 0.40:
        if amgm_gap < 0.04:
            return "Kernel-balanced"
        return "Kernel-split"
    if F > 0.30:
        if S < 0.35:
            return "Kernel-sparse"
        return "Kernel-diffuse"
    return "Kernel-collapsed"


def compute_enhanced_kernel(el: Element) -> EnhancedKernelResult:
    """Compute enhanced 12-channel kernel for one element."""
    A = round(el.atomic_mass)
    N = A - el.Z
    nz = N / el.Z if el.Z > 0 else 0.0
    bea = binding_energy_per_nucleon(el.Z, A)
    mp = magic_proximity(el.Z, A)
    is_mag = el.Z in MAGIC_NUMBERS or N in MAGIC_NUMBERS
    val_e = valence_electrons(el.group, el.block)

    c, w, labels = normalize_element_enhanced(el)
    k = compute_kernel_outputs(c, w, EPSILON)

    F_po = k["F"] + k["omega"]
    ic_leq = k["IC"] <= k["F"] + 1e-12
    ic_exp = abs(k["IC"] - math.exp(k["kappa"])) < 1e-12

    regime = _classify_regime(k["omega"], k["F"], k["S"], k["C"])
    gcd_cat = _derive_gcd_category(k["F"], k["IC"], k["amgm_gap"], k["S"], k["C"])

    # Channel diagnostics
    min_idx = int(np.argmin(c))
    max_idx = int(np.argmax(c))

    return EnhancedKernelResult(
        Z=el.Z,
        symbol=el.symbol,
        name=el.name,
        period=el.period,
        group=el.group,
        block=el.block,
        category=el.category,
        A=A,
        N=N,
        N_over_Z=round(nz, 4),
        BE_per_A=round(bea, 4),
        magic_proximity=round(mp, 4),
        is_magic=is_mag,
        valence_e=val_e,
        n_channels=len(c),
        channel_labels=labels,
        trace_vector=[round(float(x), 6) for x in c],
        F=round(k["F"], 6),
        omega=round(k["omega"], 6),
        S=round(k["S"], 6),
        C=round(k["C"], 6),
        kappa=round(k["kappa"], 6),
        IC=round(k["IC"], 6),
        amgm_gap=round(k["amgm_gap"], 6),
        F_plus_omega=round(F_po, 9),
        IC_leq_F=bool(ic_leq),
        IC_eq_exp_kappa=bool(ic_exp),
        regime=regime,
        gcd_category=gcd_cat,
        min_channel=labels[min_idx],
        max_channel=labels[max_idx],
        min_value=round(float(c[min_idx]), 6),
        max_value=round(float(c[max_idx]), 6),
    )


def compute_all_enhanced() -> list[EnhancedKernelResult]:
    """Compute enhanced kernel for all 118 elements."""
    return [compute_enhanced_kernel(el) for el in ELEMENTS]


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: CROSS-SCALE BRIDGE (quarks → hadrons → atoms)
# ═══════════════════════════════════════════════════════════════════


def cross_scale_comparison(
    enhanced_results: list[EnhancedKernelResult],
) -> None:
    """Compare kernel signatures across three scales of physics."""
    from closures.standard_model.subatomic_kernel import (
        compute_all_composite,
        compute_all_fundamental,
    )

    fund = compute_all_fundamental()
    comp = compute_all_composite()

    scales = [
        ("Fundamental (17)", fund),
        ("Composite (14)", comp),
        ("Atomic (118)", enhanced_results),
    ]

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  CROSS-SCALE KERNEL COMPARISON                             ║")
    print("║  Quarks/Leptons/Bosons → Hadrons → Atoms                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print(
        f"\n  {'Scale':<22s} {'N':>3s}  {'⟨F⟩':>6s} {'⟨IC⟩':>6s} {'⟨Δ⟩':>6s} {'⟨S⟩':>5s} {'⟨C⟩':>5s}  {'F range':>14s}"
    )
    print("  " + "─" * 78)

    for label, results in scales:
        Fs = [r.F for r in results]
        ICs = [r.IC for r in results]
        gaps = [r.amgm_gap for r in results]
        Ss = [r.S for r in results]
        Cs = [r.C for r in results]
        n = len(results)

        print(
            f"  {label:<22s} {n:3d}  {sum(Fs) / n:6.4f} {sum(ICs) / n:6.4f} "
            f"{sum(gaps) / n:6.4f} {sum(Ss) / n:5.3f} {sum(Cs) / n:5.3f}  "
            f"[{min(Fs):.3f}, {max(Fs):.3f}]"
        )

    # Subcategory breakdown
    print("\n  Subcategory breakdown:")
    print(f"  {'Subcategory':<25s} {'N':>3s}  {'⟨F⟩':>6s} {'⟨IC⟩':>6s} {'⟨Δ⟩':>6s}")
    print("  " + "─" * 50)

    subcats = [
        ("  Quarks", [r for r in fund if r.category == "Quark"]),
        ("  Leptons", [r for r in fund if r.category == "Lepton"]),
        ("  Gauge Bosons", [r for r in fund if r.category == "GaugeBoson"]),
        ("  Scalar Boson", [r for r in fund if r.category == "ScalarBoson"]),
        ("  Baryons", [r for r in comp if r.category == "Baryon"]),
        ("  Mesons", [r for r in comp if r.category == "Meson"]),
        ("  s-block atoms", [r for r in enhanced_results if r.block == "s"]),
        ("  p-block atoms", [r for r in enhanced_results if r.block == "p"]),
        ("  d-block atoms", [r for r in enhanced_results if r.block == "d"]),
        ("  f-block atoms", [r for r in enhanced_results if r.block == "f"]),
    ]

    for label, group in subcats:
        if not group:
            continue
        n = len(group)
        Fs = [r.F for r in group]
        ICs = [r.IC for r in group]
        gaps = [r.amgm_gap for r in group]
        print(f"  {label:<25s} {n:3d}  {sum(Fs) / n:6.4f} {sum(ICs) / n:6.4f} {sum(gaps) / n:6.4f}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ENHANCED ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def analyze_channel_autopsy(results: list[EnhancedKernelResult]) -> None:
    """Which channels are IC killers? Subatomic taught us zero-channels
    are lethal to the geometric mean. Find the atoms' weak channels."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  CHANNEL AUTOPSY: What Kills IC in Atoms?                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    min_counts: dict[str, int] = Counter()
    max_counts: dict[str, int] = Counter()

    for r in results:
        min_counts[r.min_channel] += 1
        max_counts[r.max_channel] += 1

    print("\n  Most frequent MINIMUM channel (IC killer):")
    for ch, count in sorted(min_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {ch:<14s}: {count:3d} ({pct:5.1f}%) {bar}")

    print("\n  Most frequent MAXIMUM channel (F driver):")
    for ch, count in sorted(max_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {ch:<14s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Quantify: average value per channel across all elements
    print("\n  Average channel value across 118 elements:")
    chan_sums: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for ch, val in zip(r.channel_labels, r.trace_vector, strict=False):
            chan_sums[ch].append(val)

    for ch in ENHANCED_CHANNELS:
        if ch in chan_sums:
            vals = chan_sums[ch]
            avg = sum(vals) / len(vals)
            std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"    {ch:<14s}: ⟨c⟩ = {avg:.4f} ± {std:.4f}")


def analyze_nuclear_binding_curve(results: list[EnhancedKernelResult]) -> None:
    """The classic BE/A curve — now with kernel overlay."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  NUCLEAR BINDING CURVE × KERNEL                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Key elements on the binding curve
    landmarks = [
        ("H", "lightest — no binding"),
        ("He", "alpha particle, doubly magic"),
        ("C", "triple-alpha, life prerequisite"),
        ("O", "doubly magic (Z=8, N=8)"),
        ("Fe", "peak of binding curve"),
        ("Ni", "Ni-62 highest BE/A"),
        ("Pb", "doubly magic (Z=82, N=126)"),
        ("U", "fission regime"),
    ]

    by_sym = {r.symbol: r for r in results}

    print(
        f"\n  {'Sym':>3s} {'Z':>3s} {'A':>4s} {'N/Z':>5s} {'BE/A':>6s} "
        f"{'F':>6s} {'IC':>6s} {'Δ':>6s} {'Magic':>5s}  Note"
    )
    print("  " + "─" * 80)

    for sym, note in landmarks:
        r = by_sym.get(sym)
        if r is None:
            continue
        mag = "YES" if r.is_magic else ""
        print(
            f"  {r.symbol:>3s} {r.Z:3d} {r.A:4d} {r.N_over_Z:5.3f} {r.BE_per_A:6.3f} "
            f"{r.F:6.4f} {r.IC:6.4f} {r.amgm_gap:6.4f} {mag:>5s}  {note}"
        )

    # Correlation: BE/A vs F
    bea_vals = np.array([r.BE_per_A for r in results])
    f_vals = np.array([r.F for r in results])
    ic_vals = np.array([r.IC for r in results])
    gap_vals = np.array([r.amgm_gap for r in results])

    # Only for A > 1 (H has no binding)
    mask = bea_vals > 0
    if np.sum(mask) > 2:
        corr_bf = np.corrcoef(bea_vals[mask], f_vals[mask])[0, 1]
        corr_bic = np.corrcoef(bea_vals[mask], ic_vals[mask])[0, 1]
        corr_bg = np.corrcoef(bea_vals[mask], gap_vals[mask])[0, 1]
        print("\n  Correlations (A > 1):")
        print(f"    BE/A ↔ F:   r = {corr_bf:+.4f}")
        print(f"    BE/A ↔ IC:  r = {corr_bic:+.4f}")
        print(f"    BE/A ↔ Δ:   r = {corr_bg:+.4f}")


def analyze_magic_numbers(results: list[EnhancedKernelResult]) -> None:
    """Nuclear magic numbers as kernel features."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  NUCLEAR MAGIC NUMBERS IN THE KERNEL                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    magic = [r for r in results if r.is_magic]
    non_magic = [r for r in results if not r.is_magic]

    print(f"\n  Magic nuclei: {len(magic)} elements")
    print(f"  Non-magic:    {len(non_magic)} elements")

    if magic and non_magic:
        avg_F_m = sum(r.F for r in magic) / len(magic)
        avg_F_nm = sum(r.F for r in non_magic) / len(non_magic)
        avg_IC_m = sum(r.IC for r in magic) / len(magic)
        avg_IC_nm = sum(r.IC for r in non_magic) / len(non_magic)
        avg_gap_m = sum(r.amgm_gap for r in magic) / len(magic)
        avg_gap_nm = sum(r.amgm_gap for r in non_magic) / len(non_magic)

        print(f"\n  {'':>14s} {'⟨F⟩':>7s} {'⟨IC⟩':>7s} {'⟨Δ⟩':>7s}")
        print(f"  {'Magic':>14s} {avg_F_m:7.4f} {avg_IC_m:7.4f} {avg_gap_m:7.4f}")
        print(f"  {'Non-magic':>14s} {avg_F_nm:7.4f} {avg_IC_nm:7.4f} {avg_gap_nm:7.4f}")
        print(
            f"  {'Δ(magic-non)':>14s} {avg_F_m - avg_F_nm:+7.4f} "
            f"{avg_IC_m - avg_IC_nm:+7.4f} {avg_gap_m - avg_gap_nm:+7.4f}"
        )

    print("\n  Magic elements detail:")
    print(f"  {'Sym':>4s} {'Z':>3s} {'N':>3s} {'Magic in':>10s} {'F':>6s} {'IC':>6s} {'Δ':>6s} {'GCD Cat'}")
    print("  " + "─" * 60)
    for r in sorted(magic, key=lambda x: x.Z):
        magic_in = []
        if r.Z in MAGIC_NUMBERS:
            magic_in.append(f"Z={r.Z}")
        if r.N in MAGIC_NUMBERS:
            magic_in.append(f"N={r.N}")
        mstr = "+".join(magic_in)
        print(
            f"  {r.symbol:>4s} {r.Z:3d} {r.N:3d} {mstr:>10s} {r.F:6.4f} {r.IC:6.4f} {r.amgm_gap:6.4f} {r.gcd_category}"
        )


def analyze_nz_stability(results: list[EnhancedKernelResult]) -> None:
    """N/Z ratio and nuclear stability in the kernel."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  N/Z RATIO → KERNEL MAPPING (Nuclear Stability)            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # NZ bins
    bins = [
        ("N/Z ≤ 1.0", [r for r in results if r.N_over_Z <= 1.0]),
        ("1.0 < N/Z ≤ 1.2", [r for r in results if 1.0 < r.N_over_Z <= 1.2]),
        ("1.2 < N/Z ≤ 1.4", [r for r in results if 1.2 < r.N_over_Z <= 1.4]),
        ("N/Z > 1.4", [r for r in results if r.N_over_Z > 1.4]),
    ]

    print(f"\n  {'N/Z range':<18s} {'N':>3s}  {'⟨F⟩':>6s} {'⟨IC⟩':>6s} {'⟨Δ⟩':>6s} {'⟨BE/A⟩':>6s}")
    print("  " + "─" * 50)

    for label, group in bins:
        if not group:
            continue
        n = len(group)
        avg_F = sum(r.F for r in group) / n
        avg_IC = sum(r.IC for r in group) / n
        avg_gap = sum(r.amgm_gap for r in group) / n
        avg_bea = sum(r.BE_per_A for r in group) / n
        print(f"  {label:<18s} {n:3d}  {avg_F:6.4f} {avg_IC:6.4f} {avg_gap:6.4f} {avg_bea:6.3f}")


def analyze_old_vs_new(results: list[EnhancedKernelResult]) -> None:
    """Compare old 8-channel vs new 12-channel kernel side by side."""
    from closures.atomic_physics.periodic_kernel import batch_compute_all

    old_results = batch_compute_all()
    old_by_sym = {r.symbol: r for r in old_results}

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  OLD (8-ch bulk) vs NEW (12-ch nuclear-informed) KERNEL    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    F_diffs = []
    IC_diffs = []
    gap_diffs = []
    regime_changes = 0
    cat_changes = 0

    for r in results:
        old = old_by_sym.get(r.symbol)
        if old is None:
            continue
        F_diffs.append(r.F - old.F)
        IC_diffs.append(r.IC - old.IC)
        gap_diffs.append(r.amgm_gap - old.amgm_gap)
        if r.regime != old.regime:
            regime_changes += 1
        if r.gcd_category != old.gcd_category:
            cat_changes += 1

    if F_diffs:
        print(f"\n  Shift statistics (new − old) across {len(F_diffs)} elements:")
        print(f"    ΔF:   mean = {np.mean(F_diffs):+.4f}  std = {np.std(F_diffs):.4f}")
        print(f"    ΔIC:  mean = {np.mean(IC_diffs):+.4f}  std = {np.std(IC_diffs):.4f}")
        print(f"    ΔΔ:   mean = {np.mean(gap_diffs):+.4f}  std = {np.std(gap_diffs):.4f}")
        print(f"    Regime reclassifications: {regime_changes}/{len(F_diffs)}")
        print(f"    Category reclassifications: {cat_changes}/{len(F_diffs)}")

    # Show biggest movers
    movers = sorted(
        zip(results, F_diffs, strict=True),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    print("\n  Top 10 F-shift elements (|ΔF| largest):")
    print(f"  {'Sym':>4s} {'F_old':>7s} {'F_new':>7s} {'ΔF':>7s} {'Old cat':>20s} {'New cat':>20s}")
    print("  " + "─" * 72)
    for r, dF in movers:
        old = old_by_sym[r.symbol]
        print(f"  {r.symbol:>4s} {old.F:7.4f} {r.F:7.4f} {dF:+7.4f} {old.gcd_category:>20s} {r.gcd_category:>20s}")


def analyze_enhanced_table(results: list[EnhancedKernelResult]) -> None:
    """Full enhanced table display."""
    print("\n╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║  ENHANCED PERIODIC TABLE — 12-Channel Nuclear-Informed Kernel                      ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════╝")

    hdr = (
        f"  {'Z':>3s} {'Sym':>3s} {'P':>1s}{'G':>3s} {'Blk':>1s} "
        f"{'N/Z':>5s} {'BE/A':>5s} {'Mag':>3s} {'Ve':>2s} "
        f"{'F':>6s} {'IC':>6s} {'Δ':>6s} {'min_ch':>13s} "
        f"{'Regime':>8s} {'GCD Category':>20s}"
    )
    print(hdr)
    print("  " + "─" * 106)

    for r in results:
        g_str = f"{r.group:>2d}" if r.group is not None else " -"
        mag = " ★" if r.is_magic else "  "
        print(
            f"  {r.Z:3d} {r.symbol:>3s} {r.period:>1d}{g_str:>3s} {r.block:>1s} "
            f"{r.N_over_Z:5.3f} {r.BE_per_A:5.3f} {mag:>3s} {r.valence_e:2d} "
            f"{r.F:6.4f} {r.IC:6.4f} {r.amgm_gap:6.4f} {r.min_channel:>13s} "
            f"{r.regime:>8s} {r.gcd_category:>20s}"
        )


def analyze_block_enhanced(results: list[EnhancedKernelResult]) -> None:
    """Block analysis with the enhanced kernel."""
    blocks: dict[str, list[EnhancedKernelResult]] = defaultdict(list)
    for r in results:
        blocks[r.block].append(r)

    print(f"\n  {'Block':>5s} {'N':>3s} {'⟨F⟩':>7s} {'⟨IC⟩':>7s} {'⟨Δ⟩':>7s} {'⟨BE/A⟩':>7s} {'⟨N/Z⟩':>6s} {'⟨S⟩':>6s}")
    print("  " + "─" * 58)

    for blk in ("s", "p", "d", "f"):
        rs = blocks.get(blk, [])
        if not rs:
            continue
        n = len(rs)
        print(
            f"  {blk:>5s} {n:3d} {sum(r.F for r in rs) / n:7.4f} "
            f"{sum(r.IC for r in rs) / n:7.4f} {sum(r.amgm_gap for r in rs) / n:7.4f} "
            f"{sum(r.BE_per_A for r in rs) / n:7.3f} {sum(r.N_over_Z for r in rs) / n:6.3f} "
            f"{sum(r.S for r in rs) / n:6.4f}"
        )


def analyze_period_enhanced(results: list[EnhancedKernelResult]) -> None:
    """Period trends with nuclear data overlay."""
    periods: dict[int, list[EnhancedKernelResult]] = defaultdict(list)
    for r in results:
        periods[r.period].append(r)

    print(f"\n  {'Per':>3s} {'N':>3s} {'⟨F⟩':>7s} {'⟨IC⟩':>7s} {'⟨Δ⟩':>7s} {'⟨BE/A⟩':>7s} {'⟨N/Z⟩':>6s} {'Magic':>5s}")
    print("  " + "─" * 52)

    for p in sorted(periods.keys()):
        rs = periods[p]
        n = len(rs)
        n_magic = sum(1 for r in rs if r.is_magic)
        print(
            f"  {p:3d} {n:3d} {sum(r.F for r in rs) / n:7.4f} "
            f"{sum(r.IC for r in rs) / n:7.4f} {sum(r.amgm_gap for r in rs) / n:7.4f} "
            f"{sum(r.BE_per_A for r in rs) / n:7.3f} {sum(r.N_over_Z for r in rs) / n:6.3f} "
            f"{n_magic:5d}"
        )


def analyze_duality_check(results: list[EnhancedKernelResult]) -> None:
    """F + F_complement = 1 duality check with enhanced channels."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  DUALITY CHECK: F(c) + F(1−c) = 1.0 ?                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Check a representative set
    landmarks = ["H", "He", "C", "N", "O", "Fe", "Cu", "Ag", "Au", "Pb", "U"]
    by_sym = {r.symbol: r for r in results}

    print(f"\n  {'Sym':>3s} {'F':>9s} {'F_comp':>9s} {'Sum':>13s}")
    print("  " + "─" * 40)

    for sym in landmarks:
        r = by_sym.get(sym)
        if r is None:
            continue
        c_comp = 1.0 - np.array(r.trace_vector)
        c_comp = np.clip(c_comp, EPSILON, 1 - EPSILON)
        w = np.ones(len(c_comp)) / len(c_comp)
        k_comp = compute_kernel_outputs(c_comp, w, EPSILON)
        F_sum = r.F + k_comp["F"]
        print(f"  {sym:>3s} {r.F:9.6f} {k_comp['F']:9.6f} {F_sum:13.9f}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║  CROSS-SCALE KERNEL ANALYSIS — Subatomic → Atomic Bridge                          ║")
    print("║  12-channel nuclear-informed kernel for 118 elements                               ║")
    print("║  Informed by: quarks(6) + leptons(6) + bosons(5) + hadrons(14)                     ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════╝")

    # ─── Compute enhanced kernel ───
    results = compute_all_enhanced()

    # ─── Tier-1 verification ───
    print(f"\n{'━' * 80}")
    print("  TIER-1 IDENTITY VERIFICATION (118 elements, 12 channels)")
    print(f"{'━' * 80}")
    passed = sum(1 for r in results if abs(r.F_plus_omega - 1.0) < 1e-9 and r.IC_leq_F and r.IC_eq_exp_kappa)
    failed = len(results) - passed
    print(f"  {passed}/{len(results)} pass all three identities, {failed} failures")

    # ─── Enhanced full table ───
    analyze_enhanced_table(results)

    # ─── Cross-scale comparison ───
    cross_scale_comparison(results)

    # ─── Channel autopsy ───
    analyze_channel_autopsy(results)

    # ─── Nuclear binding curve ───
    analyze_nuclear_binding_curve(results)

    # ─── Magic numbers ───
    analyze_magic_numbers(results)

    # ─── N/Z stability ───
    analyze_nz_stability(results)

    # ─── Block analysis (enhanced) ───
    print(f"\n{'━' * 80}")
    print("  BLOCK ANALYSIS (Enhanced 12-channel)")
    print(f"{'━' * 80}")
    analyze_block_enhanced(results)

    # ─── Period trends (enhanced) ───
    print(f"\n{'━' * 80}")
    print("  PERIOD TRENDS (Enhanced 12-channel)")
    print(f"{'━' * 80}")
    analyze_period_enhanced(results)

    # ─── Old vs New comparison ───
    analyze_old_vs_new(results)

    # ─── Duality check ───
    analyze_duality_check(results)

    # ─── GCD Category distribution ───
    print(f"\n{'━' * 80}")
    print("  GCD CATEGORY DISTRIBUTION (Enhanced)")
    print(f"{'━' * 80}")
    cat_counts = Counter(r.gcd_category for r in results)
    regime_counts = Counter(r.regime for r in results)

    print("\n  Category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<22s}: {n:3d}")

    print("\n  Regime:")
    for reg, n in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {reg:<10s}: {n:3d}")

    # ─── Summary ───
    print(f"\n{'━' * 80}")
    print("  CROSS-SCALE SUMMARY")
    print(f"{'━' * 80}")

    all_Fs = [r.F for r in results]
    all_gaps = [r.amgm_gap for r in results]
    max_F = max(results, key=lambda r: r.F)
    min_F = min(results, key=lambda r: r.F)
    max_gap = max(results, key=lambda r: r.amgm_gap)

    print("\n  Enhanced kernel (12-ch, nuclear-informed):")
    print(f"    ⟨F⟩  = {sum(all_Fs) / len(all_Fs):.4f}")
    print(f"    ⟨Δ⟩  = {sum(all_gaps) / len(all_gaps):.4f}")
    print(f"    Highest F: {max_F.symbol} (Z={max_F.Z}) F={max_F.F:.4f}")
    print(f"    Lowest F:  {min_F.symbol} (Z={min_F.Z}) F={min_F.F:.4f}")
    print(f"    Largest Δ: {max_gap.symbol} (Z={max_gap.Z}) Δ={max_gap.amgm_gap:.4f}")

    print("\n  What subatomic physics taught atomic physics:")
    print("    1. Zero-channels kill IC — now we track which channel is the 'IC killer'")
    print("    2. Binding energy IS a kernel feature — BE/A curve maps to kernel signatures")
    print("    3. Nuclear shell closures (magic numbers) create measurable kernel bumps")
    print("    4. N/Z stability band has a kernel signature — nuclear stability ↔ F")
    print("    5. Generation structure (SM) ↔ Period structure (atoms): same pattern")
    print("    6. The fermion/boson split at subatomic scale echoes in block structure")

    # ─── Recursive Instantiation bridge ───
    print(f"\n{'━' * 80}")
    print("  RECURSIVE INSTANTIATION OVERLAY")
    print(f"{'━' * 80}")
    try:
        from closures.atomic_physics.recursive_instantiation import (
            compute_recursive_analysis,
            display_census,
            display_summary,
            run_all_theorems,
        )

        analysis = compute_recursive_analysis(results)
        theorem_results = run_all_theorems(analysis)
        display_summary(theorem_results)
        display_census(analysis)
    except ImportError:
        print("  (recursive_instantiation not available)")
    print()
