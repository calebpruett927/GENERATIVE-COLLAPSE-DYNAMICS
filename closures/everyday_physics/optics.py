"""Optics Closure — Everyday Physics

Maps optical material properties to UMCP Tier-1 invariants.
Bridges atomic electron transitions (spectral lines, electron affinity)
to macroscopic light behavior (refraction, dispersion, transparency).

Trace vector construction (6 channels, equal weight):
    c₁ = n / n_max              (refractive index)
    c₂ = V_d / V_max            (Abbe number — dispersion measure)
    c₃ = T_vis                  (visible-band transmittance, 0–1)
    c₄ = R_vis                  (visible-band reflectance, 0–1)
    c₅ = E_gap / E_gap_max     (optical band gap)
    c₆ = n_g / n_g_max         (group index — pulse spread)

All channels normalized to [ε, 1−ε].

Cross-scale connections:
    subatomic → photon (massless boson, λ = c/f)
    atomic    → electron transitions determine absorption/emission
    everyday  → lenses, fiber optics, cameras, vision, color

Cross-references:
    Source data: Schott glass catalog, CRC Handbook, Hecht "Optics"
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


class OpticalRegime(StrEnum):
    """Optical material regime."""

    STABLE = "Stable"
    WATCH = "Watch"
    COLLAPSE = "Collapse"


class OpticalResult(NamedTuple):
    """Result of optical material kernel computation."""

    material: str
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    gap: float
    regime: str
    trace: list[float]


# ── Frozen constants ─────────────────────────────────────────────
EPSILON = 1e-8
N_CHANNELS = 6
OMEGA_WATCH = 0.10
OMEGA_COLLAPSE = 0.30

# Speed of light (exact)
C_LIGHT = 299_792_458.0  # m/s

# ── Optical material database ───────────────────────────────────
# Each entry: (name, n_d, V_d, T_vis, R_vis, E_gap_eV, n_group)
# n_d: refractive index at 587.6 nm (sodium D line)
# V_d: Abbe number (dispersion) — higher = less dispersion
# T_vis: visible transmittance (0–1) for typical thickness
# R_vis: visible reflectance (0–1) at normal incidence
# E_gap: optical band gap [eV]
# n_group: group refractive index at 587.6 nm

OPTICAL_MATERIALS: list[tuple[str, float, float, float, float, float, float]] = [
    # --- Glasses ---
    ("BK7 (crown)", 1.5168, 64.17, 0.92, 0.04, 4.20, 1.525),
    ("SF11 (flint)", 1.7847, 25.76, 0.88, 0.08, 3.50, 1.840),
    ("Fused silica", 1.4585, 67.82, 0.93, 0.035, 9.00, 1.463),
    ("Soda-lime", 1.5100, 59.0, 0.85, 0.04, 3.50, 1.520),
    # --- Crystals ---
    ("Diamond", 2.4190, 55.3, 0.70, 0.17, 5.47, 2.465),
    ("Sapphire", 1.7700, 72.2, 0.85, 0.07, 8.80, 1.780),
    ("Calcite (o)", 1.6584, 50.0, 0.88, 0.06, 6.00, 1.670),
    ("Quartz", 1.5443, 69.9, 0.90, 0.04, 9.00, 1.550),
    # --- Liquids ---
    ("Water", 1.3330, 55.0, 0.98, 0.02, 6.50, 1.340),
    ("Ethanol", 1.3611, 60.0, 0.97, 0.02, 6.00, 1.368),
    ("Glycerol", 1.4730, 52.0, 0.95, 0.04, 5.50, 1.480),
    # --- Gases ---
    ("Air (STP)", 1.000293, 89.0, 0.999, 0.0001, 12.0, 1.000294),
    ("CO₂", 1.000449, 85.0, 0.998, 0.0001, 10.0, 1.000450),
    # --- Everyday ---
    ("Polycarbonate", 1.5860, 30.0, 0.88, 0.05, 3.80, 1.600),
    ("PMMA (acrylic)", 1.4917, 57.4, 0.92, 0.04, 4.50, 1.500),
    ("Window glass", 1.5200, 58.0, 0.87, 0.04, 3.60, 1.530),
    # --- Fiber optics ---
    ("SMF-28 fiber", 1.4682, 68.0, 0.95, 0.035, 9.00, 1.472),
    ("Chalcogenide", 2.8000, 15.0, 0.60, 0.22, 2.00, 2.900),
    # --- Metals (opaque) ---
    ("Aluminum film", 1.3700, 10.0, 0.05, 0.92, 0.00, 1.380),
    ("Gold film", 0.4700, 10.0, 0.01, 0.95, 0.00, 0.500),
]

# Normalization ranges
_n_vals = [m[1] for m in OPTICAL_MATERIALS]
_V_vals = [m[2] for m in OPTICAL_MATERIALS]
_Eg_vals = [m[5] for m in OPTICAL_MATERIALS]
_ng_vals = [m[6] for m in OPTICAL_MATERIALS]

N_MAX = max(_n_vals)
V_MAX = max(_V_vals)
EG_MAX = max(_Eg_vals)
NG_MAX = max(_ng_vals)


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _classify_regime(omega: float) -> OpticalRegime:
    if omega < OMEGA_WATCH:
        return OpticalRegime.STABLE
    if omega < OMEGA_COLLAPSE:
        return OpticalRegime.WATCH
    return OpticalRegime.COLLAPSE


def compute_optical_material(
    name: str,
    n_d: float,
    V_d: float,
    T_vis: float,
    R_vis: float,
    E_gap: float,
    n_group: float,
) -> OpticalResult:
    """Compute UMCP kernel for optical material properties.

    Parameters
    ----------
    name : str
        Material name.
    n_d : float
        Refractive index at Na D line (587.6 nm).
    V_d : float
        Abbe number (dispersion).
    T_vis : float
        Visible transmittance (0–1).
    R_vis : float
        Visible reflectance (0–1).
    E_gap : float
        Optical band gap [eV]. 0 for metals.
    n_group : float
        Group refractive index.
    """
    c = np.array(
        [
            _clip(n_d / N_MAX),
            _clip(V_d / V_MAX),
            _clip(T_vis),
            _clip(R_vis),
            _clip(E_gap / EG_MAX) if EG_MAX > 0 else EPSILON,
            _clip(n_group / NG_MAX),
        ]
    )
    w = np.full(N_CHANNELS, 1.0 / N_CHANNELS)

    k_out = compute_kernel_outputs(c, w, EPSILON)

    F = float(k_out["F"])
    omega = float(k_out["omega"])
    IC = float(k_out["IC"])
    kappa = float(k_out["kappa"])
    S_val = float(k_out["S"])
    C_val = float(k_out["C"])
    gap_val = F - IC

    return OpticalResult(
        material=name,
        F=round(F, 6),
        omega=round(omega, 6),
        IC=round(IC, 6),
        kappa=round(kappa, 6),
        S=round(S_val, 6),
        C=round(C_val, 6),
        gap=round(gap_val, 6),
        regime=_classify_regime(omega).value,
        trace=[round(float(x), 6) for x in c],
    )


def compute_all_optical_materials() -> list[OpticalResult]:
    """Compute kernel for all materials in the optical database."""
    return [compute_optical_material(name, n, v, t, r, eg, ng) for name, n, v, t, r, eg, ng in OPTICAL_MATERIALS]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 72)
    print("  OPTICS CLOSURE — Everyday Physics")
    print("═" * 72)
    print()

    results = compute_all_optical_materials()

    print(f"  {'Material':<16} {'n_d':>6} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'Regime':<10}")
    print("  " + "─" * 68)

    for r in results:
        # Extract n_d from trace (first channel × N_MAX)
        n_d = r.trace[0] * N_MAX
        print(f"  {r.material:<16} {n_d:6.3f} {r.F:7.4f} {r.omega:7.4f} {r.IC:7.4f} {r.gap:7.4f} {r.regime:<10}")

    # Tier-1 verification
    for r in results:
        assert abs((r.F + r.omega) - 1.0) < 1e-5
        assert r.IC <= r.F + 1e-5
        assert abs(r.IC - np.exp(r.kappa)) < 1e-4

    print()
    print(f"  Materials: {len(results)}")
    mean_F = np.mean([r.F for r in results])
    mean_gap = np.mean([r.gap for r in results])
    print(f"  ⟨F⟩ = {mean_F:.4f}, ⟨Δ⟩ = {mean_gap:.4f}")
    print(f"  Tier-1 identities: ALL EXACT across {len(results)} materials")
    print("  ✓ optics self-test passed")
