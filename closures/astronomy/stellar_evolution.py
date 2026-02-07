"""Stellar Evolution Closure — ASTRO.INTSTACK.v1

Computes stellar evolution parameters:
  - Main sequence lifetime:  t_MS ≈ (M/L) · t_sun  (in Gyr)
  - Evolutionary phase classification (MS, subgiant, giant, WD, etc.)
  - ZAMS (Zero-Age Main Sequence) properties from mass
  - Luminosity evolution proxy

UMCP integration:
  ω_analog = |age / t_MS|  (fractional evolution through main sequence)
  F_analog = 1 - ω_analog  (remaining evolutionary capacity)
  Stars near end of MS → high ω → approaching collapse phase.

Regime classification (evolutionary phase):
  Pre-MS:     age < 0.01 · t_MS
  Main-Seq:   0.01 ≤ age/t_MS < 0.9  (hydrogen burning)
  Subgiant:   0.9 ≤ age/t_MS < 1.0   (hydrogen shell)
  Giant:      1.0 ≤ age/t_MS < 1.5   (helium flash/burn)
  Post-AGB:   age/t_MS ≥ 1.5         (white dwarf / remnant)

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class EvolutionRegime(StrEnum):
    """Regime based on evolutionary phase."""

    PRE_MS = "Pre-MS"
    MAIN_SEQ = "Main-Seq"
    SUBGIANT = "Subgiant"
    GIANT = "Giant"
    POST_AGB = "Post-AGB"


class EvolutionResult(NamedTuple):
    """Result of stellar evolution computation."""

    t_MS: float  # Main sequence lifetime (Gyr)
    evolutionary_phase: str  # Phase classification
    L_ZAMS: float  # ZAMS luminosity (L_sun)
    T_ZAMS: float  # ZAMS temperature (K)
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
T_SUN_MS = 10.0  # Solar main sequence lifetime (Gyr)
T_SUN = 5778.0  # Solar temperature (K)
L_SUN_ZAMS = 0.7  # Sun ZAMS luminosity (70% of current)

# Mass-luminosity exponents
ML_ALPHA_LOW = 2.3
ML_ALPHA_MID = 4.0
ML_ALPHA_HIGH = 3.5
ML_ALPHA_MASSIVE = 1.0

# Evolutionary phase boundaries (fraction of t_MS)
PHASE_PRE_MS = 0.01
PHASE_MAIN_SEQ = 0.90
PHASE_SUBGIANT = 1.00
PHASE_GIANT = 1.50


def _mass_luminosity(m: float) -> float:
    """Compute luminosity from mass-luminosity relation (solar units)."""
    if m <= 0.0:
        return 0.0
    if m < 0.43:
        return m**ML_ALPHA_LOW
    if m < 2.0:
        return m**ML_ALPHA_MID
    if m < 55.0:
        return m**ML_ALPHA_HIGH
    return m**ML_ALPHA_MASSIVE


def _main_sequence_lifetime(m_star: float) -> float:
    """Compute main sequence lifetime in Gyr.

    t_MS ≈ t_sun · (M/M_sun) / (L/L_sun) = t_sun · M / M^α = t_sun · M^(1-α)
    """
    if m_star <= 0.0:
        return 0.0
    l_star = _mass_luminosity(m_star)
    if l_star <= 0.0:
        return 0.0
    return T_SUN_MS * m_star / l_star


def _zams_luminosity(m_star: float) -> float:
    """Compute ZAMS luminosity (approximately 70% of MS luminosity for solar-type)."""
    l_ms = _mass_luminosity(m_star)
    # ZAMS fraction varies with mass, but ~0.7 is representative
    return l_ms * 0.7


def _zams_temperature(m_star: float) -> float:
    """Estimate ZAMS effective temperature from mass.

    T_ZAMS ≈ T_sun · (M/M_sun)^0.57  (empirical MS relation for solar-type)
    """
    if m_star <= 0.0:
        return 0.0
    return T_SUN * m_star**0.57


def _classify_phase(age_gyr: float, t_ms: float) -> EvolutionRegime:
    """Classify evolutionary phase from age/t_MS ratio."""
    if t_ms <= 0.0:
        return EvolutionRegime.POST_AGB
    frac = age_gyr / t_ms
    if frac < PHASE_PRE_MS:
        return EvolutionRegime.PRE_MS
    if frac < PHASE_MAIN_SEQ:
        return EvolutionRegime.MAIN_SEQ
    if frac < PHASE_SUBGIANT:
        return EvolutionRegime.SUBGIANT
    if frac < PHASE_GIANT:
        return EvolutionRegime.GIANT
    return EvolutionRegime.POST_AGB


def compute_stellar_evolution(
    m_star: float,
    l_obs: float,
    t_eff: float,
    age_gyr: float,
) -> dict[str, Any]:
    """Compute stellar evolution outputs for UMCP validation.

    Parameters
    ----------
    m_star : float
        Stellar mass in solar masses.
    l_obs : float
        Observed luminosity in solar luminosities.
    t_eff : float
        Observed effective temperature (K).
    age_gyr : float
        Estimated age in Gyr.

    Returns
    -------
    dict with keys: t_MS, evolutionary_phase, L_ZAMS, T_ZAMS, regime
    """
    t_ms = _main_sequence_lifetime(m_star)
    phase = _classify_phase(age_gyr, t_ms)
    l_zams = _zams_luminosity(m_star)
    t_zams = _zams_temperature(m_star)

    return {
        "t_MS": round(t_ms, 4),
        "evolutionary_phase": phase.value,
        "L_ZAMS": round(l_zams, 6),
        "T_ZAMS": round(t_zams, 1),
        "regime": phase.value,
    }


def compute_stellar_evolution_array(
    masses: list[float],
    luminosities: list[float],
    temperatures: list[float],
    ages: list[float],
) -> list[dict[str, Any]]:
    """Vectorized stellar evolution computation."""
    return [
        compute_stellar_evolution(m, lum, t, a)
        for m, lum, t, a in zip(masses, luminosities, temperatures, ages, strict=True)
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sun: M=1, age=4.6 Gyr, t_MS≈10 Gyr → Main-Seq
    result = compute_stellar_evolution(1.0, 1.0, 5778.0, 4.6)
    print(
        f"Sun:      t_MS={result['t_MS']:.2f} Gyr  phase={result['evolutionary_phase']}"
        f"  L_ZAMS={result['L_ZAMS']:.3f}  T_ZAMS={result['T_ZAMS']:.0f} K"
    )
    assert result["evolutionary_phase"] == "Main-Seq"

    # Sirius A: M=2.06, age≈0.25 Gyr
    result = compute_stellar_evolution(2.06, 25.4, 9940.0, 0.25)
    print(
        f"Sirius:   t_MS={result['t_MS']:.2f} Gyr  phase={result['evolutionary_phase']}"
        f"  L_ZAMS={result['L_ZAMS']:.3f}  T_ZAMS={result['T_ZAMS']:.0f} K"
    )

    # Proxima Centauri: M=0.12, age≈4.85 Gyr → long-lived, Main-Seq
    result = compute_stellar_evolution(0.12, 0.0017, 3042.0, 4.85)
    print(
        f"Proxima:  t_MS={result['t_MS']:.2f} Gyr  phase={result['evolutionary_phase']}"
        f"  L_ZAMS={result['L_ZAMS']:.6f}  T_ZAMS={result['T_ZAMS']:.0f} K"
    )
    assert result["evolutionary_phase"] == "Pre-MS" or result["evolutionary_phase"] == "Main-Seq"

    # Betelgeuse: M≈18, age≈10 Myr → evolved, possibly Giant/Post-AGB
    result = compute_stellar_evolution(18.0, 126000.0, 3600.0, 0.01)
    print(
        f"Betel:    t_MS={result['t_MS']:.4f} Gyr  phase={result['evolutionary_phase']}"
        f"  L_ZAMS={result['L_ZAMS']:.1f}  T_ZAMS={result['T_ZAMS']:.0f} K"
    )

    # White dwarf remnant: M=0.6, age=12 Gyr
    result = compute_stellar_evolution(0.6, 0.001, 8000.0, 12.0)
    print(
        f"WD:       t_MS={result['t_MS']:.2f} Gyr  phase={result['evolutionary_phase']}"
        f"  L_ZAMS={result['L_ZAMS']:.4f}  T_ZAMS={result['T_ZAMS']:.0f} K"
    )

    print("✓ stellar_evolution self-test passed")
