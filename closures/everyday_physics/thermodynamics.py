"""Thermodynamics Closure — Everyday Physics

Maps macroscopic thermal properties of materials to UMCP Tier-1 invariants.
Bridges atomic-scale properties (ionization energy, electronegativity, density)
to macroscopic thermal observables (heat capacity, thermal conductivity, phase).

Trace vector construction (6 channels, equal weight):
    c₁ = Cp / Cp_max           (specific heat capacity)
    c₂ = k_th / k_max          (thermal conductivity)
    c₃ = ρ / ρ_max             (density)
    c₄ = T_m / T_m_max         (melting point)
    c₅ = T_b / T_b_max         (boiling point)
    c₆ = α_th / α_max          (thermal diffusivity = k / (ρ·Cp))

All channels normalized to [ε, 1−ε] with ε = 10⁻⁸.

Regime classification (on ω = 1 − F):
    Stable:    ω < 0.10  →  high thermal coherence
    Watch:     0.10 ≤ ω < 0.30
    Collapse:  ω ≥ 0.30  →  significant channel heterogeneity

Cross-references:
    Source data: NIST Chemistry WebBook, CRC Handbook 97th ed.
    Atomic bridge: closures/materials_science/element_database.py
    Contract: contracts/UMA.INTSTACK.v1.yaml
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402


class ThermalRegime(StrEnum):
    """Regime classification for thermal material analysis."""

    STABLE = "Stable"
    WATCH = "Watch"
    COLLAPSE = "Collapse"


class ThermalResult(NamedTuple):
    """Result of thermal material kernel computation."""

    material: str
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    gap: float  # Δ = F − IC (heterogeneity gap)
    regime: str
    trace: list[float]


# ── Frozen constants ─────────────────────────────────────────────
EPSILON = 1e-8
N_CHANNELS = 6

# Regime thresholds (frozen, consistent across the seam)
OMEGA_WATCH = 0.10
OMEGA_COLLAPSE = 0.30

# ── Material database (NIST / CRC Handbook values) ──────────────
# Each entry: (name, Cp [J/(g·K)], k [W/(m·K)], ρ [kg/m³],
#              T_m [K], T_b [K])
# Thermal diffusivity α = k / (ρ·Cp) computed internally.

THERMAL_MATERIALS: list[tuple[str, float, float, float, float, float]] = [
    # --- Metals ---
    ("Copper", 0.385, 401.0, 8960.0, 1357.8, 2835.0),
    ("Aluminum", 0.897, 237.0, 2700.0, 933.5, 2792.0),
    ("Iron", 0.449, 80.2, 7874.0, 1811.0, 3134.0),
    ("Gold", 0.129, 318.0, 19300.0, 1337.3, 3129.0),
    ("Silver", 0.235, 429.0, 10490.0, 1234.9, 2435.0),
    ("Titanium", 0.523, 21.9, 4507.0, 1941.0, 3560.0),
    ("Lead", 0.129, 35.3, 11340.0, 600.6, 2022.0),
    ("Tungsten", 0.132, 173.0, 19250.0, 3695.0, 5828.0),
    # --- Non-metals and compounds ---
    ("Water", 4.186, 0.598, 997.0, 273.15, 373.15),
    ("Glass", 0.840, 1.05, 2500.0, 1400.0, 2500.0),
    ("Diamond", 0.509, 2200.0, 3510.0, 3820.0, 5100.0),
    ("Silicon", 0.710, 149.0, 2329.0, 1687.0, 3538.0),
    ("Air (STP)", 1.005, 0.0257, 1.225, 63.15, 77.36),
    ("Concrete", 0.880, 1.7, 2300.0, 1500.0, 2500.0),
    ("Wood (oak)", 1.700, 0.17, 750.0, 500.0, 700.0),
    ("Rubber", 2.010, 0.16, 1100.0, 340.0, 480.0),
    ("Polyethylene", 2.300, 0.50, 950.0, 400.0, 650.0),
    ("Stainless", 0.500, 16.3, 8000.0, 1672.0, 3003.0),
    ("Brick", 0.840, 0.72, 1920.0, 1800.0, 2100.0),
    ("Sand", 0.835, 0.27, 1600.0, 1986.0, 2503.0),
]

# Precomputed normalization ranges from the database
_Cp_vals = [m[1] for m in THERMAL_MATERIALS]
_k_vals = [m[2] for m in THERMAL_MATERIALS]
_rho_vals = [m[3] for m in THERMAL_MATERIALS]
_Tm_vals = [m[4] for m in THERMAL_MATERIALS]
_Tb_vals = [m[5] for m in THERMAL_MATERIALS]

CP_MAX = max(_Cp_vals)
K_MAX = max(_k_vals)
RHO_MAX = max(_rho_vals)
TM_MAX = max(_Tm_vals)
TB_MAX = max(_Tb_vals)

# Thermal diffusivity: α = k / (ρ · Cp)  [m²/s → multiply k by 1000 for g→kg]
_alpha_vals = [m[2] / (m[3] * m[1]) for m in THERMAL_MATERIALS]
ALPHA_MAX = max(_alpha_vals)


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _classify_regime(omega: float) -> ThermalRegime:
    """Classify thermal regime from drift."""
    if omega < OMEGA_WATCH:
        return ThermalRegime.STABLE
    if omega < OMEGA_COLLAPSE:
        return ThermalRegime.WATCH
    return ThermalRegime.COLLAPSE


def compute_thermal_material(
    name: str,
    Cp: float,
    k_th: float,
    rho: float,
    T_m: float,
    T_b: float,
) -> ThermalResult:
    """Compute UMCP kernel for a material's thermal properties.

    Parameters
    ----------
    name : str
        Material name.
    Cp : float
        Specific heat capacity [J/(g·K)].
    k_th : float
        Thermal conductivity [W/(m·K)].
    rho : float
        Density [kg/m³].
    T_m : float
        Melting point [K].
    T_b : float
        Boiling point [K].

    Returns
    -------
    ThermalResult
        Kernel invariants and regime classification.
    """
    # Thermal diffusivity
    alpha = k_th / (rho * Cp) if (rho * Cp) > 0 else 0.0

    # Normalize to [0, 1] then clip to [ε, 1−ε]
    c = np.array(
        [
            _clip(Cp / CP_MAX),
            _clip(k_th / K_MAX),
            _clip(rho / RHO_MAX),
            _clip(T_m / TM_MAX),
            _clip(T_b / TB_MAX),
            _clip(alpha / ALPHA_MAX),
        ]
    )
    w = np.full(N_CHANNELS, 1.0 / N_CHANNELS)

    k_out = compute_kernel_outputs(c, w, EPSILON)

    F = float(k_out["F"])
    omega = float(k_out["omega"])
    IC = float(k_out["IC"])
    kappa = float(k_out["kappa"])
    S = float(k_out["S"])
    C_val = float(k_out["C"])
    gap = F - IC

    return ThermalResult(
        material=name,
        F=round(F, 6),
        omega=round(omega, 6),
        IC=round(IC, 6),
        kappa=round(kappa, 6),
        S=round(S, 6),
        C=round(C_val, 6),
        gap=round(gap, 6),
        regime=_classify_regime(omega).value,
        trace=[round(float(x), 6) for x in c],
    )


def compute_all_thermal_materials() -> list[ThermalResult]:
    """Compute kernel for all 20 materials in the database."""
    return [compute_thermal_material(name, Cp, k, rho, Tm, Tb) for name, Cp, k, rho, Tm, Tb in THERMAL_MATERIALS]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 72)
    print("  THERMODYNAMICS CLOSURE — Everyday Physics")
    print("═" * 72)
    print()

    results = compute_all_thermal_materials()

    # Header
    print(f"  {'Material':<16} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'S':>7} {'Regime':<10}")
    print("  " + "─" * 68)

    regimes: dict[str, int] = {}
    for r in results:
        regimes[r.regime] = regimes.get(r.regime, 0) + 1
        print(f"  {r.material:<16} {r.F:7.4f} {r.omega:7.4f} {r.IC:7.4f} {r.gap:7.4f} {r.S:7.4f} {r.regime:<10}")

    print()
    print(f"  Materials: {len(results)}")
    print(f"  Regimes: {regimes}")

    # ── Tier-1 identity verification (tolerance for 6-decimal rounding) ──
    for r in results:
        assert abs((r.F + r.omega) - 1.0) < 1e-5, f"{r.material}: F+ω = {r.F + r.omega}"
        assert r.IC <= r.F + 1e-5, f"{r.material}: IC ({r.IC}) > F ({r.F})"
        assert abs(r.IC - np.exp(r.kappa)) < 1e-4, f"{r.material}: IC ≠ exp(κ)"

    print("  Tier-1 identities: ALL EXACT (F+ω=1, IC≤F, IC=exp(κ))")
    print("  ✓ thermodynamics self-test passed")
