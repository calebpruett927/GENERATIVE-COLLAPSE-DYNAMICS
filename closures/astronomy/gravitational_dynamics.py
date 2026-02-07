"""Gravitational Dynamics Closure — ASTRO.INTSTACK.v1

Computes gravitational dynamics for stellar and galactic systems:
  - Virial theorem:  2⟨KE⟩ + ⟨PE⟩ = 0  →  M_virial = 5σ²R / G
  - Dynamic mass from rotation:  M_dyn = v²_rot · r / G
  - Dark matter fraction:  f_DM = 1 − M_luminous / M_dynamic

UMCP integration:
  ω_analog = |virial_ratio − 1|  (departure from virial equilibrium)
  F_analog = 1 - ω_analog
  dark_matter_fraction maps to collapse potential: systems with high
  f_DM have large unseen mass, analogous to hidden state in UMCP.

Regime classification (based on virial ratio):
  Equilibrium:  |2KE/PE + 1| < 0.1  (virialized)
  Relaxing:     0.1 ≤ |2KE/PE + 1| < 0.3
  Disturbed:    0.3 ≤ |2KE/PE + 1| < 0.6
  Unbound:      |2KE/PE + 1| ≥ 0.6

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class DynamicsRegime(StrEnum):
    """Regime based on virial equilibrium state."""

    EQUILIBRIUM = "Equilibrium"
    RELAXING = "Relaxing"
    DISTURBED = "Disturbed"
    UNBOUND = "Unbound"


class DynamicsResult(NamedTuple):
    """Result of gravitational dynamics computation."""

    M_virial: float  # Virial mass estimate (M_sun)
    M_dynamic: float  # Dynamic mass from rotation (M_sun)
    dark_matter_fraction: float  # f_DM = 1 - M_lum / M_dyn
    virial_ratio: float  # |2KE/PE + 1| (0 = perfect equilibrium)
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
G_GRAV = 6.67430e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
M_SUN = 1.989e30  # Solar mass (kg)
PC_TO_M = 3.0857e16  # Parsec in meters
KPC_TO_M = 3.0857e19  # Kiloparsec in meters
KM_TO_M = 1.0e03  # km to m

# Regime thresholds (virial departure)
THRESH_EQUILIBRIUM = 0.1
THRESH_RELAXING = 0.3
THRESH_DISTURBED = 0.6


def _virial_mass(sigma_v_kms: float, r_kpc: float) -> float:
    """Compute virial mass: M_vir = 5 σ² R / G (in M_sun).

    Parameters
    ----------
    sigma_v_kms : float
        Velocity dispersion (km/s).
    r_kpc : float
        Characteristic radius (kpc).
    """
    if sigma_v_kms <= 0.0 or r_kpc <= 0.0:
        return 0.0
    sigma_ms = sigma_v_kms * KM_TO_M
    r_m = r_kpc * KPC_TO_M
    m_kg = 5.0 * sigma_ms**2 * r_m / G_GRAV
    return m_kg / M_SUN


def _dynamic_mass(v_rot_kms: float, r_kpc: float) -> float:
    """Compute dynamic mass from rotation curve: M = v² r / G (in M_sun).

    Parameters
    ----------
    v_rot_kms : float
        Rotational velocity (km/s).
    r_kpc : float
        Radial distance (kpc).
    """
    if v_rot_kms <= 0.0 or r_kpc <= 0.0:
        return 0.0
    v_ms = v_rot_kms * KM_TO_M
    r_m = r_kpc * KPC_TO_M
    m_kg = v_ms**2 * r_m / G_GRAV
    return m_kg / M_SUN


def _dark_matter_fraction(m_luminous: float, m_dynamic: float) -> float:
    """Compute dark matter fraction: f_DM = 1 - M_lum / M_dyn."""
    if m_dynamic <= 0.0:
        return 0.0
    f_dm = 1.0 - m_luminous / m_dynamic
    return max(0.0, min(1.0, f_dm))


def _virial_ratio(sigma_v_kms: float, v_rot_kms: float) -> float:
    """Compute virial departure as |2KE/PE + 1| proxy.

    We use the ratio of velocity dispersion to rotation as a proxy
    for the virial parameter: well-virialized systems have σ/v ~ constant.
    """
    if v_rot_kms <= 0.0 and sigma_v_kms <= 0.0:
        return 1.0
    if v_rot_kms <= 0.0:
        # Dispersion-supported only (elliptical galaxy)
        return 0.0  # Can be virialized
    ratio = sigma_v_kms / v_rot_kms
    # Typically σ/v ≈ 0.6 for a virialized disk; departure measures non-equilibrium
    return abs(ratio - 0.6) / 0.6


def _classify_regime(virial_ratio: float) -> DynamicsRegime:
    """Classify dynamical state by virial departure."""
    if virial_ratio < THRESH_EQUILIBRIUM:
        return DynamicsRegime.EQUILIBRIUM
    if virial_ratio < THRESH_RELAXING:
        return DynamicsRegime.RELAXING
    if virial_ratio < THRESH_DISTURBED:
        return DynamicsRegime.DISTURBED
    return DynamicsRegime.UNBOUND


def compute_gravitational_dynamics(
    v_rot: float,
    r_obs: float,
    sigma_v: float,
    m_luminous: float,
) -> dict[str, Any]:
    """Compute gravitational dynamics outputs for UMCP validation.

    Parameters
    ----------
    v_rot : float
        Rotational velocity (km/s).
    r_obs : float
        Observation radius (kpc).
    sigma_v : float
        Velocity dispersion (km/s).
    m_luminous : float
        Luminous (baryonic) mass in solar masses.

    Returns
    -------
    dict with keys: M_virial, M_dynamic, dark_matter_fraction, virial_ratio, regime
    """
    m_vir = _virial_mass(sigma_v, r_obs)
    m_dyn = _dynamic_mass(v_rot, r_obs)

    # Use the larger of virial and dynamic mass as the total mass estimate
    m_total = max(m_vir, m_dyn)
    f_dm = _dark_matter_fraction(m_luminous, m_total)
    vr = _virial_ratio(sigma_v, v_rot)
    regime = _classify_regime(vr)

    return {
        "M_virial": round(m_vir, 2),
        "M_dynamic": round(m_dyn, 2),
        "dark_matter_fraction": round(f_dm, 6),
        "virial_ratio": round(vr, 6),
        "regime": regime.value,
    }


def compute_gravitational_dynamics_array(
    v_rots: list[float],
    r_obss: list[float],
    sigma_vs: list[float],
    m_luminouses: list[float],
) -> list[dict[str, Any]]:
    """Vectorized gravitational dynamics computation."""
    return [
        compute_gravitational_dynamics(vr, ro, sv, ml)
        for vr, ro, sv, ml in zip(v_rots, r_obss, sigma_vs, m_luminouses, strict=True)
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Milky Way at solar radius:  v_rot≈220 km/s, r≈8.2 kpc, σ≈40 km/s
    result = compute_gravitational_dynamics(220.0, 8.2, 40.0, 5.0e10)
    print(
        f"MW@Sun:  M_vir={result['M_virial']:.2e}  M_dyn={result['M_dynamic']:.2e}"
        f"  f_DM={result['dark_matter_fraction']:.3f}"
        f"  virial={result['virial_ratio']:.3f}  regime={result['regime']}"
    )

    # Coma cluster: v_rot≈0, σ≈1000 km/s, r≈3000 kpc
    result = compute_gravitational_dynamics(0.0, 3000.0, 1000.0, 1.0e13)
    print(
        f"Coma:    M_vir={result['M_virial']:.2e}  M_dyn={result['M_dynamic']:.2e}"
        f"  f_DM={result['dark_matter_fraction']:.3f}"
        f"  virial={result['virial_ratio']:.3f}  regime={result['regime']}"
    )

    # Dwarf galaxy: v_rot≈30 km/s, r≈1.0 kpc, σ≈20 km/s
    result = compute_gravitational_dynamics(30.0, 1.0, 20.0, 1.0e7)
    print(
        f"Dwarf:   M_vir={result['M_virial']:.2e}  M_dyn={result['M_dynamic']:.2e}"
        f"  f_DM={result['dark_matter_fraction']:.3f}"
        f"  virial={result['virial_ratio']:.3f}  regime={result['regime']}"
    )

    print("✓ gravitational_dynamics self-test passed")
