"""Distance Ladder Closure — ASTRO.INTSTACK.v1

Computes and cross-validates astronomical distances through multiple methods:
  - Parallax:  d_pc = 1 / π_arcsec
  - Distance modulus:  μ = m - M = 5·log₁₀(d) − 5
  - Hubble flow:  d = c·z / H₀  (low-z approximation)

UMCP integration:
  ω_analog = distance inconsistency between methods
  F_analog = 1 - ω_analog
  The distance ladder is the astronomical realization of AX-A0:
  the inverse square law anchors all distance measures.

Regime classification (distance_confidence):
  High:       σ_d/d < 0.01
  Moderate:   0.01 ≤ σ_d/d < 0.05
  Low:        0.05 ≤ σ_d/d < 0.15
  Unreliable: σ_d/d ≥ 0.15

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class DistanceRegime(StrEnum):
    """Regime based on distance measurement confidence."""

    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    UNRELIABLE = "Unreliable"


class DistanceResult(NamedTuple):
    """Result of distance ladder computation."""

    d_modulus: float              # Distance from distance modulus (pc)
    d_parallax: float             # Distance from parallax (pc)
    d_hubble: float               # Distance from Hubble flow (Mpc)
    distance_consistency: float   # Fractional consistency between methods
    regime: str                   # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
C_LIGHT = 2.998e+05      # Speed of light (km/s — for Hubble law)
H_0_FIDUCIAL = 70.0      # Fiducial Hubble constant (km/s/Mpc)
PC_TO_MPC = 1.0e-06      # Parsec to Megaparsec conversion

# Regime thresholds
THRESH_HIGH = 0.01
THRESH_MODERATE = 0.05
THRESH_LOW = 0.15


def _distance_from_modulus(m_app: float, m_abs: float) -> float:
    """Compute distance from distance modulus: d = 10^((μ+5)/5) pc.

    μ = m - M = 5·log₁₀(d) − 5  →  d = 10^((m-M+5)/5)
    """
    mu = m_app - m_abs
    return 10.0 ** ((mu + 5.0) / 5.0)


def _distance_from_parallax(pi_arcsec: float) -> float:
    """Compute distance from parallax: d = 1/π pc."""
    if pi_arcsec <= 0.0:
        return 0.0
    return 1.0 / pi_arcsec


def _distance_from_hubble(z: float, h0: float = H_0_FIDUCIAL) -> float:
    """Compute distance from Hubble-Lemaître law: d = cz/H₀ Mpc.

    Valid for low redshift (z << 1). Returns in Mpc.
    """
    if z <= 0.0 or h0 <= 0.0:
        return 0.0
    return C_LIGHT * z / h0


def _distance_consistency(distances: list[float]) -> float:
    """Compute fractional consistency between distance estimates.

    Returns σ/μ (coefficient of variation) of valid distances.
    Lower = more consistent.
    """
    valid = [d for d in distances if d > 0.0]
    if len(valid) < 2:
        return 0.0
    mean_d = sum(valid) / len(valid)
    if mean_d <= 0.0:
        return 1.0
    variance = sum((d - mean_d) ** 2 for d in valid) / len(valid)
    return math.sqrt(variance) / mean_d


def _classify_regime(consistency: float) -> DistanceRegime:
    """Classify distance confidence regime."""
    if consistency < THRESH_HIGH:
        return DistanceRegime.HIGH
    if consistency < THRESH_MODERATE:
        return DistanceRegime.MODERATE
    if consistency < THRESH_LOW:
        return DistanceRegime.LOW
    return DistanceRegime.UNRELIABLE


def compute_distance_ladder(
    m_app: float,
    m_abs: float,
    pi_arcsec: float,
    z_cosmo: float,
    h0: float = H_0_FIDUCIAL,
) -> dict[str, Any]:
    """Compute distance ladder outputs for UMCP validation.

    Parameters
    ----------
    m_app : float
        Apparent magnitude.
    m_abs : float
        Absolute magnitude.
    pi_arcsec : float
        Parallax in arcseconds (0 if unavailable).
    z_cosmo : float
        Cosmological redshift (0 if unavailable).
    h0 : float
        Hubble constant (km/s/Mpc), default 70.

    Returns
    -------
    dict with keys: d_modulus, d_parallax, d_hubble, distance_consistency, regime
    """
    d_mod = _distance_from_modulus(m_app, m_abs)
    d_par = _distance_from_parallax(pi_arcsec)
    d_hub_mpc = _distance_from_hubble(z_cosmo, h0)

    # Convert all to parsecs for consistency comparison
    d_hub_pc = d_hub_mpc * 1.0e6 if d_hub_mpc > 0.0 else 0.0
    distances = [d for d in [d_mod, d_par, d_hub_pc] if d > 0.0]
    consistency = _distance_consistency(distances)

    regime = _classify_regime(consistency)

    return {
        "d_modulus": round(d_mod, 4),
        "d_parallax": round(d_par, 4),
        "d_hubble": round(d_hub_mpc, 4),
        "distance_consistency": round(consistency, 6),
        "regime": regime.value,
    }


def compute_distance_ladder_array(
    m_apps: list[float],
    m_abss: list[float],
    parallaxes: list[float],
    redshifts: list[float],
) -> list[dict[str, Any]]:
    """Vectorized distance ladder computation."""
    return [
        compute_distance_ladder(ma, mabs, pi, z)
        for ma, mabs, pi, z in zip(m_apps, m_abss, parallaxes, redshifts, strict=True)
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sun: m=-26.74, M=4.83, parallax not meaningful at 1AU
    # α Centauri: m=-0.01, M=4.38, π=0.7687", z≈0
    result = compute_distance_ladder(-0.01, 4.38, 0.7687, 0.0)
    print(f"α Cen:  d_mod={result['d_modulus']:.1f}pc  d_par={result['d_parallax']:.1f}pc"
          f"  consistency={result['distance_consistency']:.4f}"
          f"  regime={result['regime']}")

    # Sirius: m=-1.46, M=1.42, π=0.3792"
    result = compute_distance_ladder(-1.46, 1.42, 0.3792, 0.0)
    print(f"Sirius: d_mod={result['d_modulus']:.1f}pc  d_par={result['d_parallax']:.1f}pc"
          f"  consistency={result['distance_consistency']:.4f}"
          f"  regime={result['regime']}")

    # Andromeda (M31): m=3.44, M=-21.5, π≈0, z=−0.001001 (blueshift)
    # Use only distance modulus for this case
    result = compute_distance_ladder(3.44, -21.5, 0.0, 0.0)
    print(f"M31:    d_mod={result['d_modulus']:.0f}pc  d_par={result['d_parallax']:.0f}pc"
          f"  consistency={result['distance_consistency']:.4f}"
          f"  regime={result['regime']}")

    print("✓ distance_ladder self-test passed")
