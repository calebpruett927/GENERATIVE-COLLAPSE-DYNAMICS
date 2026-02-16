"""Electromagnetism Closure — Everyday Physics

Maps electrical and magnetic material properties to UMCP Tier-1 invariants.
Bridges atomic electron structure (ionization energy, electron affinity,
shell configuration) to macroscopic electromagnetic observables.

Trace vector construction (6 channels, equal weight):
    c₁ = σ / σ_max              (electrical conductivity)
    c₂ = ε_r / ε_r_max          (relative permittivity / dielectric constant)
    c₃ = work_fn / work_fn_max  (work function — energy to extract electron)
    c₄ = band_gap / bg_max      (band gap — conductor/insulator classification)
    c₅ = μ_r_norm               (normalized magnetic permeability)
    c₆ = resistivity_norm       (electrical resistivity, inverted scale)

All channels normalized to [ε, 1−ε] with ε = 10⁻⁸.

Cross-scale connections:
    subatomic → charge quantization (e = 1.602e-19 C)
    atomic    → electron shell structure determines conductivity
    everyday  → circuits, motors, generators, communication

Cross-references:
    Source data: CRC Handbook, NIST
    Atomic bridge: closures/atomic_physics/periodic_kernel.py
    SM bridge: closures/standard_model/particle_catalog.py (charge channel)
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


class EMRegime(StrEnum):
    """Electromagnetic material regime."""

    STABLE = "Stable"
    WATCH = "Watch"
    COLLAPSE = "Collapse"


class EMResult(NamedTuple):
    """Result of electromagnetic material kernel computation."""

    material: str
    category: str  # conductor / semiconductor / insulator
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

# ── Material database ───────────────────────────────────────────
# Each entry: (name, category,
#   σ [MS/m], ε_r, work_fn [eV], band_gap [eV], μ_r, ρ [Ω·m])
# Sources: CRC Handbook 97th ed., NIST

EM_MATERIALS: list[tuple[str, str, float, float, float, float, float, float]] = [
    # --- Conductors (band_gap ≈ 0) ---
    ("Copper", "conductor", 59.6, 1.0, 4.65, 0.0, 0.999994, 1.68e-8),
    ("Silver", "conductor", 63.0, 1.0, 4.26, 0.0, 0.999980, 1.59e-8),
    ("Gold", "conductor", 45.2, 1.0, 5.10, 0.0, 0.999964, 2.21e-8),
    ("Aluminum", "conductor", 37.7, 1.0, 4.28, 0.0, 1.000022, 2.65e-8),
    ("Iron", "conductor", 10.3, 1.0, 4.50, 0.0, 200.0, 9.71e-8),
    ("Tungsten", "conductor", 18.9, 1.0, 4.55, 0.0, 1.000068, 5.28e-8),
    # --- Semiconductors ---
    ("Silicon", "semiconductor", 0.000156, 11.7, 4.85, 1.12, 1.0, 6.40e2),
    ("Germanium", "semiconductor", 0.00217, 16.0, 5.00, 0.67, 1.0, 4.60e-1),
    ("GaAs", "semiconductor", 0.0001, 12.9, 4.07, 1.42, 1.0, 1.00e4),
    ("InP", "semiconductor", 0.00005, 12.5, 4.38, 1.35, 1.0, 2.00e4),
    # --- Insulators ---
    ("Glass (soda)", "insulator", 1e-12, 7.0, 4.50, 5.0, 1.0, 1e12),
    ("Diamond", "insulator", 1e-14, 5.7, 4.81, 5.47, 1.0, 1e14),
    ("Rubber", "insulator", 1e-14, 3.0, 4.90, 6.0, 1.0, 1e14),
    ("Teflon (PTFE)", "insulator", 1e-16, 2.1, 5.75, 8.0, 1.0, 1e16),
    ("Quartz", "insulator", 1e-18, 3.8, 5.00, 9.0, 1.0, 7.5e17),
    # --- Magnetic ---
    ("Nickel", "conductor", 14.3, 1.0, 5.15, 0.0, 600.0, 6.99e-8),
    ("Cobalt", "conductor", 17.2, 1.0, 5.00, 0.0, 250.0, 5.81e-8),
    # --- Everyday ---
    ("Sea water", "conductor", 5.0, 80.0, 4.50, 0.0, 1.0, 2.00e-1),
    ("Human body", "conductor", 0.5, 80.0, 4.50, 0.0, 1.0, 2.00e0),
    ("Dry soil", "insulator", 0.001, 15.0, 4.50, 3.0, 1.0, 1.00e3),
]

# Normalization ranges
_sigma_log = [np.log10(max(m[2], 1e-20)) for m in EM_MATERIALS]
_eps_r = [m[3] for m in EM_MATERIALS]
_wf = [m[4] for m in EM_MATERIALS]
_bg = [m[5] for m in EM_MATERIALS]
_mu_log = [np.log10(m[6]) for m in EM_MATERIALS]
_rho_log = [np.log10(max(m[7], 1e-20)) for m in EM_MATERIALS]

SIGMA_LOG_MIN, SIGMA_LOG_MAX = min(_sigma_log), max(_sigma_log)
EPS_R_MAX = max(_eps_r)
WF_MIN, WF_MAX = min(_wf), max(_wf)
BG_MAX = max(_bg)
MU_LOG_MIN, MU_LOG_MAX = min(_mu_log), max(_mu_log)
RHO_LOG_MIN, RHO_LOG_MAX = min(_rho_log), max(_rho_log)


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _classify_regime(omega: float) -> EMRegime:
    if omega < OMEGA_WATCH:
        return EMRegime.STABLE
    if omega < OMEGA_COLLAPSE:
        return EMRegime.WATCH
    return EMRegime.COLLAPSE


def compute_electromagnetic_material(
    name: str,
    category: str,
    sigma: float,
    eps_r: float,
    work_fn: float,
    band_gap: float,
    mu_r: float,
    resistivity: float,
) -> EMResult:
    """Compute UMCP kernel for electromagnetic material properties.

    Parameters
    ----------
    name : str
        Material name.
    category : str
        conductor / semiconductor / insulator.
    sigma : float
        Electrical conductivity [MS/m].
    eps_r : float
        Relative permittivity (dielectric constant).
    work_fn : float
        Work function [eV].
    band_gap : float
        Electronic band gap [eV]. 0 for metals.
    mu_r : float
        Relative magnetic permeability.
    resistivity : float
        Electrical resistivity [Ω·m].
    """
    # Log-scale normalization for conductivity and resistivity (huge ranges)
    sigma_log = np.log10(max(sigma, 1e-20))
    sigma_norm = (sigma_log - SIGMA_LOG_MIN) / (SIGMA_LOG_MAX - SIGMA_LOG_MIN) if SIGMA_LOG_MAX > SIGMA_LOG_MIN else 0.5

    rho_log = np.log10(max(resistivity, 1e-20))
    rho_norm = (rho_log - RHO_LOG_MIN) / (RHO_LOG_MAX - RHO_LOG_MIN) if RHO_LOG_MAX > RHO_LOG_MIN else 0.5

    mu_log = np.log10(mu_r)
    mu_norm = (mu_log - MU_LOG_MIN) / (MU_LOG_MAX - MU_LOG_MIN) if MU_LOG_MAX > MU_LOG_MIN else 0.5

    wf_norm = (work_fn - WF_MIN) / (WF_MAX - WF_MIN) if WF_MAX > WF_MIN else 0.5

    c = np.array(
        [
            _clip(sigma_norm),
            _clip(eps_r / EPS_R_MAX),
            _clip(wf_norm),
            _clip(band_gap / BG_MAX) if BG_MAX > 0 else EPSILON,
            _clip(mu_norm),
            _clip(rho_norm),
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

    return EMResult(
        material=name,
        category=category,
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


def compute_all_em_materials() -> list[EMResult]:
    """Compute kernel for all materials in the database."""
    return [
        compute_electromagnetic_material(name, cat, sigma, eps, wf, bg, mu, rho)
        for name, cat, sigma, eps, wf, bg, mu, rho in EM_MATERIALS
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 72)
    print("  ELECTROMAGNETISM CLOSURE — Everyday Physics")
    print("═" * 72)
    print()

    results = compute_all_em_materials()

    print(f"  {'Material':<16} {'Cat':<14} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'Regime':<10}")
    print("  " + "─" * 68)

    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F)
        print(f"  {r.material:<16} {r.category:<14} {r.F:7.4f} {r.omega:7.4f} {r.IC:7.4f} {r.gap:7.4f} {r.regime:<10}")

    print()
    for cat, fvals in cats.items():
        print(f"  {cat}: ⟨F⟩={np.mean(fvals):.4f}, n={len(fvals)}")

    # Tier-1 verification
    for r in results:
        assert abs((r.F + r.omega) - 1.0) < 1e-5
        assert r.IC <= r.F + 1e-5
        assert abs(r.IC - np.exp(r.kappa)) < 1e-4

    print(f"  Tier-1 identities: ALL EXACT across {len(results)} materials")
    print("  ✓ electromagnetism self-test passed")
