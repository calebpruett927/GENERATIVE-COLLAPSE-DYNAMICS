"""Periodic Table Kernel Analysis — Tier-1 Rederivation from Atomic Properties.

Takes each element's fundamental measurable properties, normalizes them into
a GCD kernel trace vector c ∈ [ε, 1−ε]^n, and derives Tier-1 invariants:

    F + ω = 1            (Definition 4)
    IC ≤ F               (AM-GM inequality, Lemma 4)
    IC = exp(κ)          (Log-space identity, Lemma 2)
    amgm_gap = F − IC    (Heterogeneity diagnostic, Lemma 34)
    S = h_w(c)           (Weighted entropy, Definition 6)
    C = curvature(c)     (Dispersion proxy, Definition 7)

The traditional periodic table classifies elements by electron configuration
(block), position (period/group), and chemical behavior (category). The
kernel rederivation asks: *what structure does the GCD kernel find in the
raw measurements alone, without knowing the labels?*

Normalization channels (8 measurable properties → trace vector):
    1. atomic_mass:         c₁ = Z / 118             (linear in Z, proxy for mass)
    2. electronegativity:   c₂ = EN / 4.0            (Pauling scale, max ≈ 4.0)
    3. atomic_radius:       c₃ = 1 − r / r_max       (inverted: smaller → higher fidelity)
    4. ionization_energy:   c₄ = IE / IE_max          (normalized to max observed)
    5. electron_affinity:   c₅ = |EA| / EA_max        (magnitude, normalized)
    6. melting_point:       c₆ = T_m / T_m_max        (normalized to max)
    7. boiling_point:       c₇ = T_b / T_b_max        (normalized to max)
    8. density:             c₈ = log(ρ+1) / log(ρ_max+1)  (log-scaled density)

Each channel maps a physical observable into [0, 1]. The ε-clamp ensures
c_i ∈ [ε, 1−ε] for kernel stability. Missing values (None) are excluded
from the trace vector — the kernel adapts to variable-length inputs.

This gives every element an honest Tier-1 measurement from its own physics.

Cross-references:
    Kernel:     src/umcp/kernel_optimized.py (compute_kernel_outputs)
    Database:   closures/materials_science/element_database.py (Element, ELEMENTS)
    Gap-capture: closures/materials_science/gap_capture_ss1m.py (SS1M)
    Spec:       KERNEL_SPECIFICATION.md (Tier-1 identities, Lemmas 1-34)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure workspace root is on path for closures import
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.materials_science.element_database import (  # noqa: E402
    ELEMENTS,
    Element,
    get_element,
)
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ═══════════════════════════════════════════════════════════════════
# Normalization constants (derived from the full periodic table)
# ═══════════════════════════════════════════════════════════════════

# Compute global normalization bounds from the database
_EN_MAX = max(el.electronegativity for el in ELEMENTS if el.electronegativity is not None)
_R_MAX = max(el.atomic_radius_pm for el in ELEMENTS if el.atomic_radius_pm is not None)
_IE_MAX = max(el.ionization_energy_eV for el in ELEMENTS)
_EA_MAX = max(abs(el.electron_affinity_eV) for el in ELEMENTS if el.electron_affinity_eV is not None)
_TM_MAX = max(el.melting_point_K for el in ELEMENTS if el.melting_point_K is not None)
_TB_MAX = max(el.boiling_point_K for el in ELEMENTS if el.boiling_point_K is not None)
_RHO_MAX = max(el.density_g_cm3 for el in ELEMENTS if el.density_g_cm3 is not None)
_Z_MAX = 118

# Channel labels for the property kernel
PROPERTY_CHANNELS = [
    "Z_norm",  # Atomic number (position in table)
    "EN",  # Electronegativity
    "radius",  # Atomic radius (inverted)
    "IE",  # Ionization energy
    "EA",  # Electron affinity
    "T_melt",  # Melting point
    "T_boil",  # Boiling point
    "density",  # Density (log-scaled)
]


# ═══════════════════════════════════════════════════════════════════
# Normalization: physical property → kernel coordinate c ∈ [ε, 1−ε]
# ═══════════════════════════════════════════════════════════════════


def _normalize_element(el: Element, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalize an element's properties into a kernel trace vector.

    Returns:
        (c, w, labels) where:
        - c: trace vector of available channels, ε-clamped
        - w: equal weights summing to 1.0
        - labels: channel labels for available channels
    """
    raw: list[tuple[str, float | None]] = []

    # Channel 1: Z / Z_max (always available)
    raw.append(("Z_norm", el.Z / _Z_MAX))

    # Channel 2: Electronegativity / EN_max
    if el.electronegativity is not None:
        raw.append(("EN", el.electronegativity / _EN_MAX))

    # Channel 3: Radius — inverted (smaller atom → higher c)
    if el.atomic_radius_pm is not None:
        raw.append(("radius", 1.0 - el.atomic_radius_pm / _R_MAX))

    # Channel 4: IE / IE_max
    raw.append(("IE", el.ionization_energy_eV / _IE_MAX))

    # Channel 5: |EA| / EA_max
    if el.electron_affinity_eV is not None:
        raw.append(("EA", abs(el.electron_affinity_eV) / _EA_MAX))

    # Channel 6: T_melt / T_melt_max
    if el.melting_point_K is not None:
        raw.append(("T_melt", el.melting_point_K / _TM_MAX))

    # Channel 7: T_boil / T_boil_max
    if el.boiling_point_K is not None:
        raw.append(("T_boil", el.boiling_point_K / _TB_MAX))

    # Channel 8: log(ρ+1) / log(ρ_max+1)
    if el.density_g_cm3 is not None:
        raw.append(("density", math.log(el.density_g_cm3 + 1.0) / math.log(_RHO_MAX + 1.0)))

    labels = [r[0] for r in raw]
    values = np.array([r[1] for r in raw], dtype=np.float64)

    # ε-clamp to [ε, 1−ε]
    c = np.clip(values, epsilon, 1.0 - epsilon)
    w = np.ones(len(c)) / len(c)

    return c, w, labels


# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PropertyKernelResult:
    """Full Tier-1 kernel analysis of one element from its atomic properties."""

    # Identity
    Z: int
    symbol: str
    name: str

    # Traditional labels
    period: int
    group: int | None
    block: str
    category: str

    # Kernel input
    n_channels: int
    channel_labels: list[str]
    trace_vector: list[float]

    # Tier-1 invariants
    F: float  # Fidelity (arithmetic mean of trace)
    omega: float  # Degradation = 1 − F
    S: float  # Shannon entropy of trace profile
    C: float  # Curvature (dispersion proxy)
    kappa: float  # Log-space fidelity = Σ w_i ln(c_i)
    IC: float  # Geometric mean = exp(κ)
    amgm_gap: float  # F − IC (always ≥ 0)

    # Tier-1 identity checks
    F_plus_omega: float  # Should be exactly 1.0
    IC_leq_F: bool  # AM-GM: IC ≤ F
    IC_eq_exp_kappa: bool  # IC = exp(κ)

    # Regime classification (from kernel)
    regime: str

    # GCD-derived classification
    gcd_category: str  # Derived from kernel invariants

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# Regime / GCD category derivation
# ═══════════════════════════════════════════════════════════════════


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Standard Tier-0 regime classification."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def _derive_gcd_category(
    F: float,
    omega: float,
    S: float,
    C: float,
    IC: float,
    amgm_gap: float,
    block: str,
    el: Element,
) -> str:
    """Derive a GCD-based element classification from Tier-1 invariants.

    The traditional periodic table uses electron configuration as the
    organizing principle. The GCD kernel uses the SHAPE of the property
    profile — how uniform vs. dispersed the measurable properties are.

    Classification logic (derived, not assumed):
        - High F (> 0.55) + Low C (< 0.1):    "Kernel-stable"    (uniform properties)
        - High F + High C (> 0.1):             "Kernel-structured" (high but dispersed)
        - Mid F (0.35-0.55) + Low amgm:        "Kernel-balanced"  (moderate, homogeneous)
        - Mid F + High amgm (> 0.05):          "Kernel-split"     (moderate, heterogeneous)
        - Low F (< 0.35) + Low S:              "Kernel-sparse"    (weak, low entropy)
        - Low F + High S:                      "Kernel-diffuse"   (weak, high entropy)
    """
    if F > 0.55:
        if C < 0.10:
            return "Kernel-stable"
        return "Kernel-structured"
    if F > 0.35:
        if amgm_gap < 0.05:
            return "Kernel-balanced"
        return "Kernel-split"
    if S < 0.40:
        return "Kernel-sparse"
    return "Kernel-diffuse"


# ═══════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════


def compute_element_kernel(
    symbol: str,
    epsilon: float = 1e-6,
) -> PropertyKernelResult:
    """Compute Tier-1 kernel analysis for one element.

    Takes the element's measurable atomic properties, normalizes them,
    feeds through the GCD kernel, and returns the full invariant set
    alongside traditional and GCD-derived classifications.
    """
    el = get_element(symbol)
    if el is None:
        raise ValueError(f"Unknown element: {symbol}")

    # Normalize properties → trace vector
    c, w, labels = _normalize_element(el, epsilon)

    # Feed through GCD kernel
    k = compute_kernel_outputs(c, w, epsilon)

    # Tier-1 identity checks
    F_plus_omega = k["F"] + k["omega"]
    IC_leq_F = k["IC"] <= k["F"] + 1e-12  # tolerance for float
    IC_eq_exp_kappa = abs(k["IC"] - math.exp(k["kappa"])) < 1e-12

    # Regime
    regime = _classify_regime(k["omega"], k["F"], k["S"], k["C"])

    # GCD-derived category
    gcd_cat = _derive_gcd_category(
        k["F"],
        k["omega"],
        k["S"],
        k["C"],
        k["IC"],
        k["amgm_gap"],
        el.block,
        el,
    )

    return PropertyKernelResult(
        Z=el.Z,
        symbol=el.symbol,
        name=el.name,
        period=el.period,
        group=el.group,
        block=el.block,
        category=el.category,
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
        F_plus_omega=round(F_plus_omega, 9),
        IC_leq_F=bool(IC_leq_F),
        IC_eq_exp_kappa=bool(IC_eq_exp_kappa),
        regime=regime,
        gcd_category=gcd_cat,
    )


def batch_compute_all(
    epsilon: float = 1e-6,
) -> list[PropertyKernelResult]:
    """Compute property kernel for all 118 elements."""
    results: list[PropertyKernelResult] = []
    for el in ELEMENTS:
        r = compute_element_kernel(el.symbol, epsilon)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# Display functions
# ═══════════════════════════════════════════════════════════════════


def print_full_table(results: list[PropertyKernelResult]) -> None:
    """Print full periodic table with traditional + GCD labels."""
    header = (
        f"{'Z':>3} {'Sym':>3} {'P':>1}{'G':>3} {'Blk':>1} "
        f"{'Ch':>2} {'F':>7} {'ω':>7} {'IC':>7} {'Gap':>7} "
        f"{'S':>7} {'C':>7} {'Regime':>8} "
        f"{'Traditional':>22} {'GCD-derived':>18}"
    )
    print(header)
    print("─" * len(header))
    for r in results:
        g_str = f"{r.group:>2d}" if r.group is not None else " -"
        print(
            f"{r.Z:>3} {r.symbol:>3} {r.period:>1}{g_str:>3} {r.block:>1} "
            f"{r.n_channels:>2} {r.F:>7.4f} {r.omega:>7.4f} {r.IC:>7.4f} {r.amgm_gap:>7.4f} "
            f"{r.S:>7.4f} {r.C:>7.4f} {r.regime:>8} "
            f"{r.category:>22} {r.gcd_category:>18}"
        )


def print_identity_audit(results: list[PropertyKernelResult]) -> None:
    """Verify all Tier-1 identities hold for every element."""
    violations = []
    for r in results:
        if abs(r.F_plus_omega - 1.0) > 1e-9:
            violations.append(f"{r.symbol}: F+ω = {r.F_plus_omega}")
        if not r.IC_leq_F:
            violations.append(f"{r.symbol}: IC > F ({r.IC} > {r.F})")
        if not r.IC_eq_exp_kappa:
            violations.append(f"{r.symbol}: IC ≠ exp(κ)")

    print(f"Tier-1 Identity Audit: {len(results)} elements")
    print(f"  F + ω = 1:    {'✓ ALL PASS' if not any('F+ω' in v for v in violations) else 'VIOLATIONS'}")
    print(f"  IC ≤ F:       {'✓ ALL PASS' if not any('IC > F' in v for v in violations) else 'VIOLATIONS'}")
    print(f"  IC = exp(κ):  {'✓ ALL PASS' if not any('IC ≠' in v for v in violations) else 'VIOLATIONS'}")
    if violations:
        for v in violations:
            print(f"  ✗ {v}")


def print_category_comparison(results: list[PropertyKernelResult]) -> None:
    """Compare traditional categories against GCD-derived categories."""
    from collections import Counter

    print("\n═══ Traditional Category Distribution ═══")
    trad = Counter(r.category for r in results)
    for cat, count in sorted(trad.items(), key=lambda x: -x[1]):
        print(f"  {cat:>25s}: {count:>3}")

    print("\n═══ GCD-Derived Category Distribution ═══")
    gcd = Counter(r.gcd_category for r in results)
    for cat, count in sorted(gcd.items(), key=lambda x: -x[1]):
        print(f"  {cat:>25s}: {count:>3}")

    print("\n═══ Cross-tabulation: Traditional → GCD ═══")
    # For each traditional category, show GCD breakdown
    for trad_cat in sorted(trad.keys()):
        elems = [r for r in results if r.category == trad_cat]
        gcd_sub = Counter(r.gcd_category for r in elems)
        mapping = ", ".join(f"{g}({c})" for g, c in sorted(gcd_sub.items(), key=lambda x: -x[1]))
        print(f"  {trad_cat:>25s} [{len(elems):>2}] → {mapping}")


def print_block_analysis(results: list[PropertyKernelResult]) -> None:
    """Analyze kernel invariants by electron-configuration block."""
    blocks = {"s": [], "p": [], "d": [], "f": []}
    for r in results:
        blocks[r.block].append(r)

    print("\n═══ Block Analysis (mean Tier-1 invariants) ═══")
    print(f"{'Block':>5} {'N':>3} {'⟨F⟩':>7} {'⟨ω⟩':>7} {'⟨IC⟩':>7} {'⟨Gap⟩':>7} {'⟨S⟩':>7} {'⟨C⟩':>7}")
    print("─" * 55)
    for blk in ("s", "p", "d", "f"):
        rs = blocks[blk]
        if not rs:
            continue
        n = len(rs)
        mean_F = sum(r.F for r in rs) / n
        mean_w = sum(r.omega for r in rs) / n
        mean_IC = sum(r.IC for r in rs) / n
        mean_gap = sum(r.amgm_gap for r in rs) / n
        mean_S = sum(r.S for r in rs) / n
        mean_C = sum(r.C for r in rs) / n
        print(
            f"{blk:>5} {n:>3} {mean_F:>7.4f} {mean_w:>7.4f} "
            f"{mean_IC:>7.4f} {mean_gap:>7.4f} {mean_S:>7.4f} {mean_C:>7.4f}"
        )


def print_period_trend(results: list[PropertyKernelResult]) -> None:
    """Show how kernel invariants trend across periods."""
    periods: dict[int, list[PropertyKernelResult]] = {}
    for r in results:
        periods.setdefault(r.period, []).append(r)

    print("\n═══ Period Trends (mean Tier-1 invariants) ═══")
    print(f"{'Period':>6} {'N':>3} {'⟨F⟩':>7} {'⟨ω⟩':>7} {'⟨IC⟩':>7} {'⟨Gap⟩':>7} {'⟨S⟩':>7} {'⟨C⟩':>7}")
    print("─" * 57)
    for p in sorted(periods.keys()):
        rs = periods[p]
        n = len(rs)
        mean_F = sum(r.F for r in rs) / n
        mean_w = sum(r.omega for r in rs) / n
        mean_IC = sum(r.IC for r in rs) / n
        mean_gap = sum(r.amgm_gap for r in rs) / n
        mean_S = sum(r.S for r in rs) / n
        mean_C = sum(r.C for r in rs) / n
        print(
            f"{p:>6} {n:>3} {mean_F:>7.4f} {mean_w:>7.4f} "
            f"{mean_IC:>7.4f} {mean_gap:>7.4f} {mean_S:>7.4f} {mean_C:>7.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PERIODIC TABLE KERNEL ANALYSIS — Tier-1 Rederivation      ║")
    print("║  118 elements × 8 property channels → GCD kernel           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Compute all 118
    results = batch_compute_all()

    # 1. Identity audit
    print_identity_audit(results)

    # 2. Full table
    print()
    print_full_table(results)

    # 3. Block analysis
    print_block_analysis(results)

    # 4. Period trends
    print_period_trend(results)

    # 5. Category comparison
    print_category_comparison(results)

    # 6. Archive
    archive = [r.to_dict() for r in results]
    out_path = "outputs/periodic_kernel_118.json"
    with open(out_path, "w") as f:
        json.dump(archive, f, indent=2)
    print(f"\nArchived {len(archive)} results → {out_path}")
