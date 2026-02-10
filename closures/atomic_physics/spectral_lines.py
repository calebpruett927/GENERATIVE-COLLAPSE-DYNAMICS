"""Spectral Lines Closure — ATOM.INTSTACK.v1

Computes atomic spectral line wavelengths using the Rydberg formula
and maps spectral precision to GCD invariants.

Physics:
  1/λ = R_∞ · Z² · (1/n₁² − 1/n₂²)     (Rydberg formula)
  R_∞ = 1.0973731568539 × 10⁷ m⁻¹

  Named series:
    Lyman:   n₁=1 (UV)        Balmer:  n₁=2 (visible)
    Paschen: n₁=3 (IR)        Brackett: n₁=4 (IR)

UMCP integration:
  ω_eff = |λ_measured − λ_predicted| / λ_measured   (spectral drift)
  F_eff = 1 − ω_eff
  Ψ_λ = λ_predicted / λ_measured  clipped to [0,1]

Regime classification:
  Resolved:     ω_eff < 0.001
  Broadened:    0.001 ≤ ω_eff < 0.01
  Blended:      0.01 ≤ ω_eff < 0.05
  Unresolved:   ω_eff ≥ 0.05

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Sources:   NIST ASD, Kramida et al. (2023)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class SpectralRegime(StrEnum):
    """Regime based on spectral line resolution quality."""

    RESOLVED = "Resolved"
    BROADENED = "Broadened"
    BLENDED = "Blended"
    UNRESOLVED = "Unresolved"


class SpectralResult(NamedTuple):
    """Result of spectral line computation."""

    lambda_predicted_nm: float  # Predicted wavelength (nm)
    lambda_measured_nm: float  # Measured wavelength (nm)
    energy_eV: float  # Transition energy (eV)
    series_name: str  # Spectral series name
    n_lower: int  # Lower principal quantum number
    n_upper: int  # Upper principal quantum number
    omega_eff: float  # Spectral drift
    F_eff: float  # Spectral fidelity
    regime: str  # Classification


# ── Frozen constants ─────────────────────────────────────────────
R_INF = 1.0973731568539e7  # Rydberg constant (m⁻¹)
HC_EV_NM = 1239.8419843  # hc in eV·nm
SERIES_NAMES = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund"}

# Hydrogen Balmer series reference wavelengths (nm) — NIST
H_BALMER_REF: dict[int, float] = {
    3: 656.281,
    4: 486.135,
    5: 434.047,
    6: 410.174,
    7: 397.007,
}

THRESH_RESOLVED = 0.001
THRESH_BROADENED = 0.01
THRESH_BLENDED = 0.05


def _classify_regime(omega_eff: float) -> SpectralRegime:
    if omega_eff < THRESH_RESOLVED:
        return SpectralRegime.RESOLVED
    if omega_eff < THRESH_BROADENED:
        return SpectralRegime.BROADENED
    if omega_eff < THRESH_BLENDED:
        return SpectralRegime.BLENDED
    return SpectralRegime.UNRESOLVED


def compute_spectral_lines(
    Z: int,
    n_lower: int,
    n_upper: int,
    *,
    lambda_measured_nm: float | None = None,
) -> SpectralResult:
    """Compute spectral line wavelength and GCD mapping.

    Parameters
    ----------
    Z : int
        Atomic number (1 = hydrogen, 2 = He⁺, etc.)
    n_lower, n_upper : int
        Principal quantum numbers for the transition (n_upper > n_lower).
    lambda_measured_nm : float | None
        Measured wavelength (nm).  Falls back to prediction.

    Returns
    -------
    SpectralResult
    """
    if n_upper <= n_lower:
        msg = f"n_upper ({n_upper}) must be > n_lower ({n_lower})"
        raise ValueError(msg)
    if Z < 1:
        msg = f"Z must be ≥ 1, got {Z}"
        raise ValueError(msg)

    # Rydberg formula
    inv_lambda = R_INF * Z**2 * (1.0 / n_lower**2 - 1.0 / n_upper**2)

    if inv_lambda > 0:
        lambda_pred_m = 1.0 / inv_lambda
        lambda_pred_nm = lambda_pred_m * 1e9
    else:
        lambda_pred_nm = 0.0

    energy_ev = HC_EV_NM / lambda_pred_nm if lambda_pred_nm > 0 else 0.0
    series_name = SERIES_NAMES.get(n_lower, f"n₁={n_lower}")

    # Reference value
    if lambda_measured_nm is not None:
        lam_meas = lambda_measured_nm
    elif Z == 1 and n_lower == 2 and n_upper in H_BALMER_REF:
        lam_meas = H_BALMER_REF[n_upper]
    else:
        lam_meas = lambda_pred_nm

    # GCD mapping
    omega_eff = abs(lam_meas - lambda_pred_nm) / lam_meas if lam_meas > 0 else 1.0

    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff
    regime = _classify_regime(omega_eff)

    return SpectralResult(
        lambda_predicted_nm=round(lambda_pred_nm, 4),
        lambda_measured_nm=round(lam_meas, 4),
        energy_eV=round(energy_ev, 6),
        series_name=series_name,
        n_lower=n_lower,
        n_upper=n_upper,
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


def compute_series(Z: int, n_lower: int, n_upper_max: int = 7) -> list[dict[str, Any]]:
    """Compute full spectral series from n_lower+1 to n_upper_max."""
    results = []
    for n_up in range(n_lower + 1, n_upper_max + 1):
        r = compute_spectral_lines(Z, n_lower, n_up)
        results.append(r._asdict())
    return results


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Hydrogen Balmer Series:")
    for n_up in range(3, 8):
        r = compute_spectral_lines(1, 2, n_up)
        print(f"  n={n_up}→2  λ={r.lambda_predicted_nm:.3f} nm  E={r.energy_eV:.4f} eV  {r.regime}")
