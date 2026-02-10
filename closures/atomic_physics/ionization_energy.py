"""Ionization Energy Closure — ATOM.INTSTACK.v1

Computes ionization energies using the hydrogen-like model with quantum
defect corrections, maps to GCD invariants via binding deficit.

Physics:
  E_n = -13.6 eV · Z_eff² / n_eff²   (hydrogen-like with quantum defect)
  n_eff = n - δ_l                       (effective quantum number)
  Z_eff = Z - σ                         (Slater screening)
  IE₁ reference: H = 13.598 eV

UMCP integration:
  ω_eff = |IE_measured - IE_predicted| / IE_measured  (ionization drift)
  F_eff = 1 - ω_eff                                    (ionization fidelity)
  Ψ_IE = IE₁ / 13.598  clipped to [0,1]               (normalized trace)

Regime classification:
  Precise:       ω_eff < 0.01
  Approximate:   0.01 ≤ ω_eff < 0.05
  Screened:      0.05 ≤ ω_eff < 0.15
  Anomalous:     ω_eff ≥ 0.15

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Canon:     canon/atom_anchors.yaml
  Sources:   NIST ASD, Kramida et al. (2023)
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class IonizationRegime(StrEnum):
    """Regime based on ionization energy prediction accuracy."""

    PRECISE = "Precise"
    APPROXIMATE = "Approximate"
    SCREENED = "Screened"
    ANOMALOUS = "Anomalous"


class IonizationResult(NamedTuple):
    """Result of ionization energy computation."""

    IE_predicted_eV: float  # Predicted ionization energy (eV)
    IE_measured_eV: float  # Measured/reference ionization energy (eV)
    Z_eff: float  # Effective nuclear charge (Slater)
    n_eff: float  # Effective principal quantum number
    omega_eff: float  # Drift from prediction
    F_eff: float  # Fidelity
    Psi_IE: float  # Normalized IE trace channel
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
RYDBERG_EV = 13.5984  # Hydrogen ground-state IE (eV)

# Slater screening constants (approximate, by shell)
SLATER_RULES: dict[str, float] = {
    "1s": 0.30,
    "2s": 0.85,
    "2p": 0.85,
    "3s": 1.70,
    "3p": 1.70,
    "3d": 1.00,
    "4s": 2.00,
    "4p": 2.00,
    "4d": 1.00,
    "4f": 1.00,
}

# NIST reference first ionization energies (eV) for Z=1..36
NIST_IE1: dict[int, float] = {
    1: 13.598,
    2: 24.587,
    3: 5.392,
    4: 9.323,
    5: 8.298,
    6: 11.260,
    7: 14.534,
    8: 13.618,
    9: 17.423,
    10: 21.565,
    11: 5.139,
    12: 7.646,
    13: 5.986,
    14: 8.152,
    15: 10.487,
    16: 10.360,
    17: 12.968,
    18: 15.760,
    19: 4.341,
    20: 6.113,
    21: 6.562,
    22: 6.828,
    23: 6.746,
    24: 6.767,
    25: 7.434,
    26: 7.902,
    27: 7.881,
    28: 7.640,
    29: 7.726,
    30: 9.394,
    31: 5.999,
    32: 7.900,
    33: 9.789,
    34: 9.752,
    35: 11.814,
    36: 14.000,
}

# Regime thresholds
THRESH_PRECISE = 0.01
THRESH_APPROXIMATE = 0.05
THRESH_SCREENED = 0.15


def _classify_regime(omega_eff: float) -> IonizationRegime:
    """Classify ionization deviation regime."""
    if omega_eff < THRESH_PRECISE:
        return IonizationRegime.PRECISE
    if omega_eff < THRESH_APPROXIMATE:
        return IonizationRegime.APPROXIMATE
    if omega_eff < THRESH_SCREENED:
        return IonizationRegime.SCREENED
    return IonizationRegime.ANOMALOUS


def _slater_screening(Z: int, n: int) -> float:
    """Approximate Slater screening constant σ for outermost electron."""
    # Simplified: σ ≈ 0.30 per 1s electron + 0.85 per inner shell electron
    if n == 1:
        return 0.30 * max(0, min(Z - 1, 1))
    elif n == 2:
        return 2 * 0.85 + max(0, Z - 3) * 0.35
    elif n == 3:
        return 2 * 1.00 + 8 * 0.85 + max(0, Z - 11) * 0.35
    elif n == 4:
        return 2 * 1.00 + 8 * 1.00 + 8 * 0.85 + max(0, Z - 19) * 0.35
    else:
        # General approximation
        return Z * 0.65


def compute_ionization(
    Z: int,
    n: int = 0,
    *,
    IE_measured_eV: float | None = None,
    quantum_defect: float = 0.0,
) -> IonizationResult:
    """Compute ionization energy and GCD mapping.

    Parameters
    ----------
    Z : int
        Atomic number.
    n : int
        Principal quantum number of valence electron.  If 0, auto-assigned
        from periodic table period.
    IE_measured_eV : float | None
        Measured IE (eV).  Falls back to NIST reference or prediction.
    quantum_defect : float
        Quantum defect δ_l for the valence orbital.

    Returns
    -------
    IonizationResult
    """
    if Z < 1:
        msg = f"Atomic number Z must be ≥ 1, got {Z}"
        raise ValueError(msg)

    # Auto-assign principal quantum number from period
    if n <= 0:
        if Z <= 2:
            n = 1
        elif Z <= 10:
            n = 2
        elif Z <= 18:
            n = 3
        elif Z <= 36:
            n = 4
        elif Z <= 54:
            n = 5
        elif Z <= 86:
            n = 6
        else:
            n = 7

    # Effective quantum number
    n_eff = n - quantum_defect

    # Slater screening
    sigma = _slater_screening(Z, n)
    z_eff = max(1.0, Z - sigma)

    # Hydrogen-like prediction
    ie_predicted = RYDBERG_EV * z_eff**2 / n_eff**2

    # Measured/reference value
    if IE_measured_eV is not None:
        ie_measured = IE_measured_eV
    elif Z in NIST_IE1:
        ie_measured = NIST_IE1[Z]
    else:
        ie_measured = ie_predicted  # fallback

    # GCD mapping
    omega_eff = abs(ie_measured - ie_predicted) / ie_measured if ie_measured > 0 else 1.0

    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff
    psi_ie = max(0.0, min(1.0, ie_measured / RYDBERG_EV))

    regime = _classify_regime(omega_eff)

    return IonizationResult(
        IE_predicted_eV=round(ie_predicted, 4),
        IE_measured_eV=round(ie_measured, 4),
        Z_eff=round(z_eff, 4),
        n_eff=round(n_eff, 4),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        Psi_IE=round(psi_ie, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (1, "Hydrogen"),
        (2, "Helium"),
        (6, "Carbon"),
        (11, "Sodium"),
        (26, "Iron"),
        (29, "Copper"),
    ]
    for z, name in tests:
        r = compute_ionization(z)
        print(
            f"{name:12s} (Z={z:2d})  IE_pred={r.IE_predicted_eV:8.3f}  "
            f"IE_meas={r.IE_measured_eV:8.3f}  ω={r.omega_eff:.4f}  {r.regime}"
        )
