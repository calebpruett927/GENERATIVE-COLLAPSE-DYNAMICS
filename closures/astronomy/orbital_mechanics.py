"""Orbital Mechanics Closure — ASTRO.INTSTACK.v1

Validates Kepler's third law and computes orbital parameters:
  - Kepler III:  P² = (4π²/GM)·a³
  - Orbital velocity:  v = 2πa/P  (circular approximation)
  - Specific orbital energy:  E = -GM/(2a)

UMCP integration:
  ω_analog = kepler_residual = |P²_obs - P²_predicted| / P²_predicted
  F_analog = 1 - kepler_residual
  Orbital eccentricity maps to regime classification.

Regime classification (orbital_stability):
  Stable:    e < 0.3
  Eccentric: 0.3 ≤ e < 0.7
  Unstable:  0.7 ≤ e < 0.95
  Escape:    e ≥ 0.95

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class OrbitalRegime(StrEnum):
    """Regime based on orbital eccentricity."""

    STABLE = "Stable"
    ECCENTRIC = "Eccentric"
    UNSTABLE = "Unstable"
    ESCAPE = "Escape"


class OrbitalResult(NamedTuple):
    """Result of orbital mechanics computation."""

    P_predicted: float       # Predicted period from Kepler III (s)
    kepler_residual: float   # Fractional Kepler III residual
    v_orb: float             # Orbital velocity (m/s)
    E_orbital: float         # Specific orbital energy (J/kg)
    regime: str              # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
G_GRAV = 6.67430e-11   # Gravitational constant (m³ kg⁻¹ s⁻²)
M_SUN = 1.989e+30      # Solar mass (kg)
AU_TO_M = 1.496e+11    # AU in meters

# Regime thresholds (eccentricity)
THRESH_STABLE = 0.3
THRESH_ECCENTRIC = 0.7
THRESH_UNSTABLE = 0.95


def _kepler_period(a_m: float, m_total_kg: float) -> float:
    """Compute orbital period from Kepler's third law.

    P = 2π √(a³ / (G·M))

    Parameters
    ----------
    a_m : float
        Semi-major axis in meters.
    m_total_kg : float
        Total system mass in kg.

    Returns
    -------
    float : Period in seconds.
    """
    if a_m <= 0.0 or m_total_kg <= 0.0:
        return 0.0
    return 2.0 * math.pi * math.sqrt(a_m ** 3 / (G_GRAV * m_total_kg))


def _orbital_velocity(a_m: float, p_s: float) -> float:
    """Mean orbital velocity for circular approximation: v = 2πa/P."""
    if p_s <= 0.0:
        return 0.0
    return 2.0 * math.pi * a_m / p_s


def _specific_orbital_energy(a_m: float, m_total_kg: float) -> float:
    """Specific orbital energy: E = -GM/(2a)."""
    if a_m <= 0.0:
        return 0.0
    return -G_GRAV * m_total_kg / (2.0 * a_m)


def _classify_regime(e_orb: float) -> OrbitalRegime:
    """Classify orbital stability regime by eccentricity."""
    if e_orb < THRESH_STABLE:
        return OrbitalRegime.STABLE
    if e_orb < THRESH_ECCENTRIC:
        return OrbitalRegime.ECCENTRIC
    if e_orb < THRESH_UNSTABLE:
        return OrbitalRegime.UNSTABLE
    return OrbitalRegime.ESCAPE


def compute_orbital_mechanics(
    p_orb: float,
    a_semi: float,
    m_total: float,
    e_orb: float,
) -> dict[str, Any]:
    """Compute orbital mechanics outputs for UMCP validation.

    Parameters
    ----------
    p_orb : float
        Observed orbital period in seconds.
    a_semi : float
        Semi-major axis in AU.
    m_total : float
        Total system mass in solar masses.
    e_orb : float
        Orbital eccentricity (0 = circular, 1 = parabolic).

    Returns
    -------
    dict with keys: P_predicted, kepler_residual, v_orb, E_orbital, regime
    """
    a_m = a_semi * AU_TO_M
    m_kg = m_total * M_SUN

    # Kepler's third law prediction
    p_predicted = _kepler_period(a_m, m_kg)

    # Kepler residual (drift analog)
    kepler_residual = abs(p_orb ** 2 - p_predicted ** 2) / (p_predicted ** 2) if p_predicted > 0.0 else 1.0
    # Orbital velocity (using predicted period for consistency)
    v_orb = _orbital_velocity(a_m, p_predicted)

    # Specific orbital energy
    e_orbital = _specific_orbital_energy(a_m, m_kg)

    regime = _classify_regime(e_orb)

    return {
        "P_predicted": round(p_predicted, 4),
        "kepler_residual": round(kepler_residual, 9),
        "v_orb": round(v_orb, 2),
        "E_orbital": round(e_orbital, 2),
        "regime": regime.value,
    }


def compute_orbital_mechanics_array(
    periods: list[float],
    semi_axes: list[float],
    masses: list[float],
    eccentricities: list[float],
) -> list[dict[str, Any]]:
    """Vectorized orbital mechanics computation."""
    return [
        compute_orbital_mechanics(p, a, m, e)
        for p, a, m, e in zip(periods, semi_axes, masses, eccentricities, strict=True)
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Earth: P = 1 year, a = 1 AU, M = 1 M_sun, e = 0.0167
    year_s = 365.25 * 24 * 3600
    result = compute_orbital_mechanics(year_s, 1.0, 1.0, 0.0167)
    print(f"Earth:   P_pred={result['P_predicted']:.0f}s  "
          f"residual={result['kepler_residual']:.6e}  "
          f"v_orb={result['v_orb']:.0f} m/s  "
          f"regime={result['regime']}")
    assert result["regime"] == "Stable"
    assert result["kepler_residual"] < 0.01, "Kepler residual should be tiny for Earth"

    # Jupiter: P ≈ 11.86 yr, a = 5.203 AU, M ≈ 1.001 M_sun, e = 0.0489
    p_jup = 11.862 * year_s
    result = compute_orbital_mechanics(p_jup, 5.203, 1.001, 0.0489)
    print(f"Jupiter: P_pred={result['P_predicted']:.0f}s  "
          f"residual={result['kepler_residual']:.6e}  "
          f"v_orb={result['v_orb']:.0f} m/s  "
          f"regime={result['regime']}")

    # Halley's comet: P ≈ 75.3 yr, a ≈ 17.8 AU, M ≈ 1.0, e = 0.967
    p_halley = 75.3 * year_s
    result = compute_orbital_mechanics(p_halley, 17.8, 1.0, 0.967)
    print(f"Halley:  P_pred={result['P_predicted']:.0f}s  "
          f"residual={result['kepler_residual']:.6e}  "
          f"v_orb={result['v_orb']:.0f} m/s  "
          f"regime={result['regime']}")
    assert result["regime"] == "Escape"

    print("✓ orbital_mechanics self-test passed")
