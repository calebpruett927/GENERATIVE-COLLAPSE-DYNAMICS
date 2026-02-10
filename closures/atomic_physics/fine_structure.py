"""Fine Structure Closure — ATOM.INTSTACK.v1

Computes relativistic fine structure corrections to hydrogen-like energy
levels using the Dirac formula and Lamb shift estimates.

Physics:
  E_nj = E_n · [1 + (Zα)²/n · (1/(j+½) − 3/(4n))]         (fine structure)
  Lamb shift ≈ α⁵ mc² / (4π n³) · δ(l,0)                   (QED correction)
  α = 1/137.036  (fine structure constant)

UMCP integration:
  ω_eff = |ΔE_fine| / |E_n|  (fine-structure fraction of total energy)
  F_eff = 1 − ω_eff

Regime:
  NonRelativistic: (Zα)² < 0.01
  Relativistic:    0.01 ≤ (Zα)² < 0.1
  HeavyAtom:       (Zα)² ≥ 0.1

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Sources:   Dirac (1928), Lamb & Retherford (1947)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class FineStructureRegime(StrEnum):
    NON_RELATIVISTIC = "NonRelativistic"
    RELATIVISTIC = "Relativistic"
    HEAVY_ATOM = "HeavyAtom"


class FineStructureResult(NamedTuple):
    """Result of fine structure computation."""

    E_n_eV: float  # Gross energy level (eV)
    E_fine_eV: float  # Fine structure correction (eV)
    E_lamb_eV: float  # Lamb shift estimate (eV)
    E_total_eV: float  # Total corrected energy (eV)
    Z_alpha_squared: float  # Relativistic parameter (Zα)²
    splitting_eV: float  # j-splitting magnitude
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
ALPHA_FINE = 1.0 / 137.035999084  # Fine structure constant
RYDBERG_EV = 13.5984  # eV
M_E_C2_EV = 510998.95  # electron rest energy (eV)


def _classify(z_alpha_sq: float) -> FineStructureRegime:
    if z_alpha_sq < 0.01:
        return FineStructureRegime.NON_RELATIVISTIC
    if z_alpha_sq < 0.1:
        return FineStructureRegime.RELATIVISTIC
    return FineStructureRegime.HEAVY_ATOM


def compute_fine_structure(
    Z: int,
    n: int,
    l_val: int,
    j: float,
) -> FineStructureResult:
    """Compute fine structure energy correction for hydrogen-like atom.

    Parameters
    ----------
    Z : int
        Atomic number.
    n : int
        Principal quantum number.
    l_val : int
        Orbital angular momentum quantum number.
    j : float
        Total angular momentum quantum number (l ± ½).

    Returns
    -------
    FineStructureResult
    """
    if n < 1:
        msg = f"n must be ≥ 1, got {n}"
        raise ValueError(msg)
    if l_val < 0 or l_val >= n:
        msg = f"l must be in [0, n-1], got l={l_val}, n={n}"
        raise ValueError(msg)
    if not (abs(j - l_val) <= 0.5 + 1e-9):
        msg = f"j must be l ± 1/2, got j={j}, l={l_val}"
        raise ValueError(msg)

    # Gross energy (non-relativistic)
    e_n = -RYDBERG_EV * Z**2 / n**2

    # Fine structure correction (first-order perturbation)
    z_alpha = Z * ALPHA_FINE
    z_alpha_sq = z_alpha**2

    # ΔE_fine = E_n · (Zα)²/n · [1/(j + 1/2) − 3/(4n)]
    delta_fine = e_n * z_alpha_sq / n * (1.0 / (j + 0.5) - 3.0 / (4.0 * n))

    # Lamb shift estimate (s-states only, leading order)
    lamb_shift = ALPHA_FINE**5 * M_E_C2_EV * Z**4 / (4.0 * math.pi * n**3) if l_val == 0 else 0.0

    e_total = e_n + delta_fine + lamb_shift

    # j-splitting: difference between j=l+1/2 and j=l-1/2 levels
    if l_val > 0:
        delta_j_plus = e_n * z_alpha_sq / n * (1.0 / (l_val + 1) - 3.0 / (4.0 * n))
        delta_j_minus = e_n * z_alpha_sq / n * (1.0 / l_val - 3.0 / (4.0 * n))
        splitting = abs(delta_j_plus - delta_j_minus)
    else:
        splitting = 0.0

    # GCD mapping
    omega_eff = abs(delta_fine / e_n) if e_n != 0 else 0.0
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(z_alpha_sq)

    return FineStructureResult(
        E_n_eV=round(e_n, 6),
        E_fine_eV=round(delta_fine, 8),
        E_lamb_eV=round(lamb_shift, 10),
        E_total_eV=round(e_total, 6),
        Z_alpha_squared=round(z_alpha_sq, 8),
        splitting_eV=round(splitting, 8),
        omega_eff=round(omega_eff, 8),
        F_eff=round(f_eff, 8),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hydrogen n=2 fine structure
    for l_val in [0, 1]:
        j_vals = [l_val + 0.5] if l_val == 0 else [l_val - 0.5, l_val + 0.5]
        for j_val in j_vals:
            r = compute_fine_structure(1, 2, l_val, j_val)
            print(
                f"H n=2 l={l_val} j={j_val:.1f}:  E_n={r.E_n_eV:.4f}  "
                f"ΔE_fine={r.E_fine_eV:.8f}  Lamb={r.E_lamb_eV:.10f}  {r.regime}"
            )
