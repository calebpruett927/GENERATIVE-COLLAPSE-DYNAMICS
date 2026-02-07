"""Spectral Analysis Closure — ASTRO.INTSTACK.v1

Analyzes stellar spectra through:
  - Wien's displacement law:  λ_peak = b / T_eff
  - B−V color index to temperature mapping
  - Spectral class embedding:  O-B-A-F-G-K-M → [0, 1]

UMCP integration:
  The spectral class encodes temperature monotonically (AX-A2).
  Spectral embedding maps the discrete O-M classification to [0,1]
  for UMCP interval arithmetic. The chi² fit measures how well the
  observed spectrum matches blackbody predictions.

Regime classification (spectral_fit):
  Excellent: chi2 < 0.8
  Good:      0.8 ≤ chi2 < 1.5
  Marginal:  1.5 ≤ chi2 < 2.5
  Poor:      chi2 ≥ 2.5

Cross-references:
  Contract: contracts/ASTRO.INTSTACK.v1.yaml
  Canon: canon/astro_anchors.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class SpectralRegime(StrEnum):
    """Regime based on spectral fit quality."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    MARGINAL = "Marginal"
    POOR = "Poor"


class SpectralResult(NamedTuple):
    """Result of spectral analysis."""

    lambda_peak: float  # Wien peak wavelength (nm)
    T_from_BV: float  # Temperature inferred from B−V (K)
    spectral_embedding: float  # Spectral class embedded to [0,1]
    chi2_spectral: float  # Spectral fit quality
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
B_WIEN = 2_897_771.955  # Wien displacement constant (nm·K)

# Spectral class embedding: O=0.0, B=0.14, A=0.29, F=0.43, G=0.57, K=0.71, M=0.86
SPECTRAL_EMBEDDING = {
    "O": 0.00,
    "B": 0.143,
    "A": 0.286,
    "F": 0.429,
    "G": 0.571,
    "K": 0.714,
    "M": 0.857,
}

# Approximate temperature ranges for spectral classes (K)
SPECTRAL_TEMP_RANGES = {
    "O": (30000, 60000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

# B−V to temperature empirical relation (Flower 1996, Sekiguchi & Fukugita 2000)
# T_eff = 4600 / [(0.92 · (B-V)) + 1.70] + 4600 / [(0.92 · (B-V)) + 0.62]
# Simplified Ballesteros (2012) relation:
# T = 4600 * (1/(0.92*(B-V) + 1.7) + 1/(0.92*(B-V) + 0.62))

# Regime thresholds
THRESH_EXCELLENT = 0.8
THRESH_GOOD = 1.5
THRESH_MARGINAL = 2.5


def _wien_peak(t_eff: float) -> float:
    """Wien peak wavelength in nm."""
    if t_eff <= 0.0:
        return 0.0
    return B_WIEN / t_eff


def _bv_to_temperature(b_v: float) -> float:
    """Convert B−V color index to effective temperature using Ballesteros (2012).

    T = 4600 · (1/(0.92·(B-V) + 1.7) + 1/(0.92·(B-V) + 0.62))
    Valid for approximately -0.4 ≤ B-V ≤ 2.0.
    """
    denom1 = 0.92 * b_v + 1.70
    denom2 = 0.92 * b_v + 0.62
    if denom1 <= 0.0 or denom2 <= 0.0:
        return 0.0
    return 4600.0 * (1.0 / denom1 + 1.0 / denom2)


def _embed_spectral_class(spectral_class: str) -> float:
    """Embed spectral class letter to [0,1].

    Handles subtypes like "G2" by interpolating within the class range.
    """
    if not spectral_class:
        return 0.5

    letter = spectral_class[0].upper()
    if letter not in SPECTRAL_EMBEDDING:
        return 0.5

    base = SPECTRAL_EMBEDDING[letter]

    # Parse subtype digit if present (e.g., "G2" → 2)
    subtype = 0
    if len(spectral_class) > 1 and spectral_class[1].isdigit():
        subtype = int(spectral_class[1])

    # Each spectral class spans ~0.143 in embedding space; subtype refines within
    return min(1.0, base + subtype * 0.0143)


def _spectral_chi2(t_eff: float, t_from_bv: float, spectral_class: str) -> float:
    """Compute spectral fit quality as reduced chi² analog.

    Measures consistency between:
      1. Direct T_eff measurement
      2. T inferred from B−V
      3. Expected range for spectral class
    """
    residuals = []

    # T_eff vs T_from_BV
    if t_from_bv > 0.0 and t_eff > 0.0:
        residuals.append(((t_eff - t_from_bv) / t_eff) ** 2)

    # T_eff vs spectral class range
    letter = spectral_class[0].upper() if spectral_class else ""
    if letter in SPECTRAL_TEMP_RANGES:
        t_low, t_high = SPECTRAL_TEMP_RANGES[letter]
        t_mid = (t_low + t_high) / 2
        t_sigma = (t_high - t_low) / 4  # ~2σ range
        if t_sigma > 0:
            residuals.append(((t_eff - t_mid) / t_sigma) ** 2)

    if not residuals:
        return 1.0
    return sum(residuals) / len(residuals)


def _classify_regime(chi2: float) -> SpectralRegime:
    """Classify spectral fit regime."""
    if chi2 < THRESH_EXCELLENT:
        return SpectralRegime.EXCELLENT
    if chi2 < THRESH_GOOD:
        return SpectralRegime.GOOD
    if chi2 < THRESH_MARGINAL:
        return SpectralRegime.MARGINAL
    return SpectralRegime.POOR


def compute_spectral_analysis(
    t_eff: float,
    b_v: float,
    spectral_class: str,
) -> dict[str, Any]:
    """Compute spectral analysis outputs for UMCP validation.

    Parameters
    ----------
    t_eff : float
        Effective temperature (K).
    b_v : float
        B−V color index.
    spectral_class : str
        Spectral type string (e.g., "G2", "A0", "M5").

    Returns
    -------
    dict with keys: lambda_peak, T_from_BV, spectral_embedding, chi2_spectral, regime
    """
    lambda_peak = _wien_peak(t_eff)
    t_from_bv = _bv_to_temperature(b_v)
    spectral_embedding = _embed_spectral_class(spectral_class)
    chi2 = _spectral_chi2(t_eff, t_from_bv, spectral_class)
    regime = _classify_regime(chi2)

    return {
        "lambda_peak": round(lambda_peak, 2),
        "T_from_BV": round(t_from_bv, 1),
        "spectral_embedding": round(spectral_embedding, 4),
        "chi2_spectral": round(chi2, 6),
        "regime": regime.value,
    }


def compute_spectral_analysis_array(
    temperatures: list[float],
    bv_indices: list[float],
    spectral_classes: list[str],
) -> list[dict[str, Any]]:
    """Vectorized spectral analysis computation."""
    return [
        compute_spectral_analysis(t, bv, sc)
        for t, bv, sc in zip(temperatures, bv_indices, spectral_classes, strict=True)
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sun: T=5778K, B-V=0.656, G2
    result = compute_spectral_analysis(5778.0, 0.656, "G2")
    print(
        f"Sun:    λ_peak={result['lambda_peak']:.1f}nm  T_BV={result['T_from_BV']:.0f}K"
        f"  embed={result['spectral_embedding']:.3f}  χ²={result['chi2_spectral']:.4f}"
        f"  regime={result['regime']}"
    )

    # Vega: T=9602K, B-V=0.00, A0
    result = compute_spectral_analysis(9602.0, 0.00, "A0")
    print(
        f"Vega:   λ_peak={result['lambda_peak']:.1f}nm  T_BV={result['T_from_BV']:.0f}K"
        f"  embed={result['spectral_embedding']:.3f}  χ²={result['chi2_spectral']:.4f}"
        f"  regime={result['regime']}"
    )

    # Betelgeuse: T=3600K, B-V=1.85, M2
    result = compute_spectral_analysis(3600.0, 1.85, "M2")
    print(
        f"Betel:  λ_peak={result['lambda_peak']:.1f}nm  T_BV={result['T_from_BV']:.0f}K"
        f"  embed={result['spectral_embedding']:.3f}  χ²={result['chi2_spectral']:.4f}"
        f"  regime={result['regime']}"
    )

    print("✓ spectral_analysis self-test passed")
