"""Stellar Luminosity Closure — ASTRO.INTSTACK.v1

Computes stellar luminosity from fundamental parameters using:
  - Stefan-Boltzmann law:  L = 4π R² σ_SB T_eff⁴
  - Mass-luminosity relation:  L ∝ M^α  (piecewise by mass range)
  - Wien displacement law:  λ_peak = b / T_eff

UMCP integration:
  ω_analog = delta_L = |L_obs - L_predicted| / L_predicted   (drift from ML relation)
  F_analog = 1 - delta_L                                     (fidelity to prediction)

Regime classification (luminosity_deviation):
  Consistent:  delta_L < 0.05
  Mild:        0.05 ≤ delta_L < 0.15
  Significant: 0.15 ≤ delta_L < 0.30
  Anomalous:   delta_L ≥ 0.30

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
  Registry: closures/registry.yaml (extensions.astronomy)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class LuminosityRegime(StrEnum):
    """Regime based on deviation from mass-luminosity relation."""

    CONSISTENT = "Consistent"
    MILD = "Mild"
    SIGNIFICANT = "Significant"
    ANOMALOUS = "Anomalous"


class LuminosityResult(NamedTuple):
    """Result of stellar luminosity computation."""

    L_predicted: float  # Mass-luminosity predicted luminosity (L_sun)
    L_SB: float  # Stefan-Boltzmann luminosity (L_sun)
    delta_L: float  # Fractional deviation |L_SB - L_predicted| / L_predicted
    lambda_peak: float  # Wien peak wavelength (nm)
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
SIGMA_SB = 5.670374419e-08  # Stefan-Boltzmann constant (W m⁻² K⁻⁴)
B_WIEN = 2_897_771.955  # Wien displacement constant (nm·K)
L_SUN = 3.828e26  # Solar luminosity (W)
R_SUN = 6.957e08  # Solar radius (m)
T_SUN = 5778.0  # Solar effective temperature (K)

# Mass-luminosity exponents (piecewise)
ML_ALPHA_LOW = 2.3  # M < 0.43 M_sun
ML_ALPHA_MID = 4.0  # 0.43 ≤ M < 2 M_sun
ML_ALPHA_HIGH = 3.5  # 2 ≤ M < 55 M_sun
ML_ALPHA_MASSIVE = 1.0  # M ≥ 55 M_sun

# Regime thresholds
THRESH_CONSISTENT = 0.05
THRESH_MILD = 0.15
THRESH_SIGNIFICANT = 0.30


def _mass_luminosity(m_star: float) -> float:
    """Compute predicted luminosity from mass-luminosity relation (solar units).

    Piecewise power-law:
      L/L_sun = (M/M_sun)^α
    where α depends on mass range.
    """
    if m_star <= 0.0:
        return 0.0
    if m_star < 0.43:
        return m_star**ML_ALPHA_LOW
    if m_star < 2.0:
        return m_star**ML_ALPHA_MID
    if m_star < 55.0:
        return m_star**ML_ALPHA_HIGH
    return m_star**ML_ALPHA_MASSIVE


def _stefan_boltzmann_luminosity(r_star: float, t_eff: float) -> float:
    """Compute luminosity via Stefan-Boltzmann law in solar units.

    L = 4π R² σ T⁴ → L/L_sun = (R/R_sun)² · (T/T_sun)⁴
    """
    if r_star <= 0.0 or t_eff <= 0.0:
        return 0.0
    return (r_star**2) * (t_eff / T_SUN) ** 4


def _classify_regime(delta_l: float) -> LuminosityRegime:
    """Classify luminosity deviation regime."""
    if delta_l < THRESH_CONSISTENT:
        return LuminosityRegime.CONSISTENT
    if delta_l < THRESH_MILD:
        return LuminosityRegime.MILD
    if delta_l < THRESH_SIGNIFICANT:
        return LuminosityRegime.SIGNIFICANT
    return LuminosityRegime.ANOMALOUS


def compute_stellar_luminosity(
    m_star: float,
    t_eff: float,
    r_star: float,
) -> dict[str, Any]:
    """Compute stellar luminosity outputs for UMCP validation.

    Parameters
    ----------
    m_star : float
        Stellar mass in solar masses (M/M_sun).
    t_eff : float
        Effective temperature (K).
    r_star : float
        Stellar radius in solar radii (R/R_sun).

    Returns
    -------
    dict with keys: L_predicted, L_SB, delta_L, lambda_peak, regime
    """
    # Mass-luminosity prediction
    l_predicted = _mass_luminosity(m_star)

    # Stefan-Boltzmann observed luminosity
    l_sb = _stefan_boltzmann_luminosity(r_star, t_eff)

    # Fractional deviation
    delta_l = abs(l_sb - l_predicted) / l_predicted if l_predicted > 0.0 else 1.0
    # Wien peak wavelength
    lambda_peak = B_WIEN / t_eff if t_eff > 0.0 else 0.0

    regime = _classify_regime(delta_l)

    return {
        "L_predicted": round(l_predicted, 6),
        "L_SB": round(l_sb, 6),
        "delta_L": round(delta_l, 6),
        "lambda_peak": round(lambda_peak, 2),
        "regime": regime.value,
    }


def compute_stellar_luminosity_array(
    masses: list[float],
    temperatures: list[float],
    radii: list[float],
) -> list[dict[str, Any]]:
    """Vectorized stellar luminosity computation over multiple stars."""
    return [compute_stellar_luminosity(m, t, r) for m, t, r in zip(masses, temperatures, radii, strict=True)]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sun: M=1, T=5778, R=1 → L≈1, delta_L≈0
    result = compute_stellar_luminosity(1.0, 5778.0, 1.0)
    print(
        f"Sun:    L_pred={result['L_predicted']:.4f}  L_SB={result['L_SB']:.4f}"
        f"  delta_L={result['delta_L']:.4f}  λ_peak={result['lambda_peak']:.1f} nm"
        f"  regime={result['regime']}"
    )
    assert result["regime"] == "Consistent", f"Expected Consistent, got {result['regime']}"
    assert abs(result["L_SB"] - 1.0) < 0.01, f"Sun L_SB should be ~1, got {result['L_SB']}"

    # Sirius A: M≈2.06, T≈9940, R≈1.71
    result = compute_stellar_luminosity(2.06, 9940.0, 1.71)
    print(
        f"Sirius: L_pred={result['L_predicted']:.4f}  L_SB={result['L_SB']:.4f}"
        f"  delta_L={result['delta_L']:.4f}  λ_peak={result['lambda_peak']:.1f} nm"
        f"  regime={result['regime']}"
    )

    # Betelgeuse: M≈18, T≈3600, R≈764
    result = compute_stellar_luminosity(18.0, 3600.0, 764.0)
    print(
        f"Betel:  L_pred={result['L_predicted']:.4f}  L_SB={result['L_SB']:.4f}"
        f"  delta_L={result['delta_L']:.4f}  λ_peak={result['lambda_peak']:.1f} nm"
        f"  regime={result['regime']}"
    )

    print("✓ stellar_luminosity self-test passed")
