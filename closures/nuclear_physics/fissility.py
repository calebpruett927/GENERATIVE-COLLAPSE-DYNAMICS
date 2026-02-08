"""Fissility Closure — NUC.INTSTACK.v1

Computes the fissility parameter x = (Z²/A) / (Z²/A)_crit and
the competition between Coulomb repulsion and surface tension that
governs spontaneous fission probability.

Physics:
  Liquid drop model (Bohr & Wheeler 1939):
    (Z²/A)_crit = 2·a_S / a_C ≈ 2 × 17.23 / 0.714 ≈ 48.26
  Fissility parameter:
    x = (Z²/A) / (Z²/A)_crit
  Coulomb energy:   E_C = a_C · Z(Z−1) / A^(1/3)
  Surface energy:   E_S = a_S · A^(2/3)
  Ψ_fiss = min(1, x)   [trace channel, clipped to [0,1]]

  Note: The precise critical value depends on the SEMF coefficient
  set chosen.  With a_S=17.23, a_C=0.714: (Z²/A)_crit = 48.26.
  Some references use 50.88 (Royer & Remaud 1985 parameterization).
  We freeze (Z²/A)_crit = 48.26 for consistency with our SEMF set.

Regime classification (fissility):
  Subfissile:    x < 0.70  (far from fission instability)
  Transitional:  0.70 ≤ x < 0.90
  Fissile:       0.90 ≤ x < 1.00
  Supercritical: x ≥ 1.00

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Sources:   Bohr & Wheeler 1939; Myers & Swiatecki 1966
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class FissilityRegime(StrEnum):
    """Regime based on fissility parameter."""

    SUBFISSILE = "Subfissile"
    TRANSITIONAL = "Transitional"
    FISSILE = "Fissile"
    SUPERCRITICAL = "Supercritical"


class FissilityResult(NamedTuple):
    """Result of fissility computation."""

    Z_squared_over_A: float  # Z²/A
    fissility_x: float  # x = (Z²/A) / (Z²/A)_crit
    coulomb_energy: float  # E_C = a_C · Z(Z−1)/A^(1/3) (MeV)
    surface_energy: float  # E_S = a_S · A^(2/3) (MeV)
    Psi_fiss: float  # Fissility trace channel (clipped)
    regime: str


# ── Frozen constants ─────────────────────────────────────────────
A_S = 17.23  # Surface term (MeV) — frozen with SEMF set
A_C = 0.714  # Coulomb term (MeV) — frozen with SEMF set
Z2A_CRIT = 2.0 * A_S / A_C  # ≈ 48.26 (Bohr-Wheeler critical fissility)

# Regime thresholds
THRESH_TRANSITIONAL = 0.70
THRESH_FISSILE = 0.90
THRESH_SUPERCRITICAL = 1.00


def _classify_regime(x: float) -> FissilityRegime:
    """Classify fissility regime."""
    if x < THRESH_TRANSITIONAL:
        return FissilityRegime.SUBFISSILE
    if x < THRESH_FISSILE:
        return FissilityRegime.TRANSITIONAL
    if x < THRESH_SUPERCRITICAL:
        return FissilityRegime.FISSILE
    return FissilityRegime.SUPERCRITICAL


def compute_fissility(
    Z: int,
    A: int,
) -> FissilityResult:
    """Compute fissility parameter and Coulomb/surface competition.

    Parameters
    ----------
    Z : int
        Atomic number.
    A : int
        Mass number.

    Returns
    -------
    FissilityResult
    """
    if A < 1:
        msg = f"Mass number A must be ≥ 1, got {A}"
        raise ValueError(msg)
    if Z < 0 or Z > A:
        msg = f"Z must be in [0, A], got Z={Z}, A={A}"
        raise ValueError(msg)

    z2_a = Z**2 / A
    x = z2_a / Z2A_CRIT

    coul = A_C * Z * (Z - 1) / A ** (1.0 / 3.0) if A > 0 else 0.0
    surf = A_S * A ** (2.0 / 3.0)

    psi_fiss = min(1.0, max(0.0, x))
    regime = _classify_regime(x)

    return FissilityResult(
        Z_squared_over_A=round(z2_a, 4),
        fissility_x=round(x, 6),
        coulomb_energy=round(coul, 4),
        surface_energy=round(surf, 4),
        Psi_fiss=round(psi_fiss, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (2, 4, "He-4"),
        (26, 56, "Fe-56"),
        (28, 62, "Ni-62"),
        (37, 85, "Rb-85"),
        (82, 208, "Pb-208"),
        (92, 235, "U-235"),
        (92, 238, "U-238"),
        (94, 239, "Pu-239"),
        (114, 298, "Fl-298 (SHE)"),
    ]

    print(f"{'Nuclide':15s} {'Z²/A':>8s} {'x':>8s} {'E_C':>8s} {'E_S':>8s} {'Ψ_fiss':>8s} {'Regime':>15s}")
    print("-" * 80)
    for z, a, name in tests:
        r = compute_fissility(z, a)
        print(
            f"{name:15s} {r.Z_squared_over_A:8.3f} {r.fissility_x:8.4f} "
            f"{r.coulomb_energy:8.2f} {r.surface_energy:8.2f} "
            f"{r.Psi_fiss:8.4f} {r.regime:>15s}"
        )

    print(f"\n(Z²/A)_crit = {Z2A_CRIT:.2f}")
