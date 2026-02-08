"""Nuclide Binding Closure — NUC.INTSTACK.v1

Computes binding energy per nucleon using the Bethe-Weizsäcker
semi-empirical mass formula (SEMF), normalizes to the Ni-62 peak,
and maps to GCD effective drift (ω_eff) and fidelity (F_eff).

Physics:
  BE = a_V·A − a_S·A^(2/3) − a_C·Z(Z−1)/A^(1/3) − a_A·(A−2Z)²/A ± a_P/A^(1/2)
  Frozenref: Ni-62 at BE/A = 8.7945 MeV/nucleon.

UMCP integration:
  ω_eff = 1 − BE_per_A / BE_peak_ref   (binding deficit = drift)
  F_eff = BE_per_A / BE_peak_ref        (binding fraction = fidelity)
  Ψ_BE = F_eff = BE_per_A / 8.7945     (normalized trace channel)

Regime classification (binding_deficit):
  Peak:     ω_eff < 0.01  (within 1% of Ni-62)
  Plateau:  0.01 ≤ ω_eff < 0.05
  Slope:    0.05 ≤ ω_eff < 0.15
  Deficit:  ω_eff ≥ 0.15

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Sources:   von Weizsäcker 1935; Bethe & Bacher 1936
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class BindingRegime(StrEnum):
    """Regime based on binding deficit from Ni-62 peak."""

    PEAK = "Peak"
    PLATEAU = "Plateau"
    SLOPE = "Slope"
    DEFICIT = "Deficit"


class BindingResult(NamedTuple):
    """Result of nuclide binding computation."""

    BE_total: float  # Total binding energy (MeV)
    BE_per_A: float  # Binding energy per nucleon (MeV/nucleon)
    BE_per_A_norm: float  # Normalized to Ni-62 peak
    Psi_BE: float  # Ψ_BE trace channel (= BE_per_A_norm, clipped to [0,1])
    omega_eff: float  # Effective drift from peak
    F_eff: float  # Effective fidelity (= Ψ_BE)
    regime: str  # Binding deficit regime


# ── Frozen constants (SEMF coefficients) ─────────────────────────
A_V = 15.67  # Volume term (MeV)
A_S = 17.23  # Surface term (MeV)
A_C = 0.714  # Coulomb term (MeV)
A_A = 23.29  # Asymmetry term (MeV)
A_P = 11.2  # Pairing term (MeV)
BE_PEAK_REF = 8.7945  # Ni-62 BE/A peak (MeV/nucleon)

# Regime thresholds
THRESH_PEAK = 0.01
THRESH_PLATEAU = 0.05
THRESH_SLOPE = 0.15


def _classify_regime(omega_eff: float) -> BindingRegime:
    """Classify binding deficit regime."""
    if omega_eff < THRESH_PEAK:
        return BindingRegime.PEAK
    if omega_eff < THRESH_PLATEAU:
        return BindingRegime.PLATEAU
    if omega_eff < THRESH_SLOPE:
        return BindingRegime.SLOPE
    return BindingRegime.DEFICIT


def compute_binding(
    Z: int,
    A: int,
    *,
    BE_per_A_measured: float | None = None,
) -> BindingResult:
    """Compute binding energy and GCD mapping for a nuclide.

    Parameters
    ----------
    Z : int
        Atomic number (proton count).
    A : int
        Mass number (proton + neutron count).
    BE_per_A_measured : float | None
        If provided, use measured BE/A instead of SEMF estimate.
        Measured values from AME2020 are more accurate than SEMF.

    Returns
    -------
    BindingResult
        Named tuple with BE, normalization, and GCD mapping.
    """
    if A < 1:
        msg = f"Mass number A must be ≥ 1, got {A}"
        raise ValueError(msg)
    if Z < 0 or Z > A:
        msg = f"Atomic number Z must be in [0, A], got Z={Z}, A={A}"
        raise ValueError(msg)

    N = A - Z

    if BE_per_A_measured is not None:
        be_per_a = BE_per_A_measured
        be_total = be_per_a * A
    else:
        # Semi-Empirical Mass Formula (Bethe-Weizsäcker)
        vol = A_V * A
        surf = A_S * A ** (2.0 / 3.0)
        coul = A_C * Z * (Z - 1) / A ** (1.0 / 3.0) if A > 0 else 0.0
        asym = A_A * (A - 2 * Z) ** 2 / A if A > 0 else 0.0

        # Pairing term
        if Z % 2 == 0 and N % 2 == 0:
            pair = A_P / A**0.5  # even-even: extra binding
        elif Z % 2 == 1 and N % 2 == 1:
            pair = -A_P / A**0.5  # odd-odd: less binding
        else:
            pair = 0.0  # odd-even or even-odd

        be_total = vol - surf - coul - asym + pair
        be_per_a = be_total / A if A > 0 else 0.0

    # Normalization to Ni-62 peak
    be_norm = be_per_a / BE_PEAK_REF
    psi_be = max(0.0, min(1.0, be_norm))  # clip to [0,1]

    # GCD mapping
    omega_eff = max(0.0, 1.0 - be_norm)
    f_eff = psi_be

    regime = _classify_regime(omega_eff)

    return BindingResult(
        BE_total=round(be_total, 4),
        BE_per_A=round(be_per_a, 4),
        BE_per_A_norm=round(be_norm, 6),
        Psi_BE=round(psi_be, 6),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with known values
    tests = [
        (1, 1, 0.000, "Hydrogen-1"),
        (2, 4, 7.074, "Helium-4"),
        (26, 56, 8.7903, "Iron-56"),
        (28, 62, 8.7945, "Nickel-62 (peak)"),
        (37, 85, 8.6970, "Rubidium-85"),
        (82, 208, 7.8675, "Lead-208"),
        (92, 238, 7.5701, "Uranium-238"),
    ]

    for z, a, be_measured, name in tests:
        r = compute_binding(z, a, BE_per_A_measured=be_measured)
        print(
            f"{name:25s}  BE/A={r.BE_per_A:7.4f}  Ψ_BE={r.Psi_BE:.4f}  "
            f"ω_eff={r.omega_eff:.4f}  F_eff={r.F_eff:.4f}  {r.regime}"
        )
