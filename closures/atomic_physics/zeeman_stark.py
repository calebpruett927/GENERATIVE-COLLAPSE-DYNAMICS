"""Zeeman & Stark Effect Closure — ATOM.INTSTACK.v1

Computes energy level splitting in external electromagnetic fields.

Physics:
  Zeeman effect (magnetic field B):
    Normal:     ΔE = m_l · μ_B · B                (singlet states)
    Anomalous:  ΔE = m_j · g_J · μ_B · B          (spin-orbit coupled)
    g_J = 1 + [j(j+1) + s(s+1) - l(l+1)] / [2j(j+1)]   (Landé g-factor)

  Stark effect (electric field E):
    Linear:     ΔE = 3eEa₀n(n₁-n₂)/2  (degenerate, hydrogen)
    Quadratic:  ΔE = -½α_D E²          (non-degenerate)
    α_D ≈ (9/2)a₀³n⁷(n²-1) / Z⁴       (hydrogen-like polarizability)

UMCP integration:
  ω_eff = |ΔE_field| / |E_n|   (field perturbation fraction)

Regime:
  Weak:     ΔE_field < 0.01 |E_n|
  Moderate: 0.01 ≤ ratio < 0.1
  Strong:   ratio ≥ 0.1  (Paschen-Back / complete mixing)

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Sources:   Zeeman (1896), Stark (1913)
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class FieldRegime(StrEnum):
    WEAK = "Weak"
    MODERATE = "Moderate"
    STRONG = "Strong"


class ZeemanStarkResult(NamedTuple):
    """Result of Zeeman/Stark effect computation."""

    E_n_eV: float  # Unperturbed energy (eV)
    delta_E_zeeman_eV: float  # Zeeman splitting (eV)
    delta_E_stark_eV: float  # Stark shift (eV)
    delta_E_total_eV: float  # Combined perturbation
    g_lande: float  # Landé g-factor
    n_zeeman_levels: int  # Number of Zeeman sub-levels
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
MU_B_EV_PER_T = 5.7883818060e-5  # Bohr magneton (eV/T)
A_BOHR_M = 5.29177210903e-11  # Bohr radius (m)
E_CHARGE = 1.602176634e-19  # elementary charge (C)
EPSILON_0 = 8.8541878128e-12  # vacuum permittivity (F/m)
FOUR_PI_EPS0 = 4.0 * 3.141592653589793 * EPSILON_0  # 4πε₀
RYDBERG_EV = 13.5984


def _lande_g(j: float, ell: int, s: float) -> float:
    """Compute Landé g-factor."""
    if j == 0:
        return 0.0
    jj = j * (j + 1)
    ll = ell * (ell + 1)
    ss = s * (s + 1)
    return 1.0 + (jj + ss - ll) / (2.0 * jj)


def _classify(ratio: float) -> FieldRegime:
    if ratio < 0.01:
        return FieldRegime.WEAK
    if ratio < 0.1:
        return FieldRegime.MODERATE
    return FieldRegime.STRONG


def compute_zeeman_stark(
    Z: int,
    n: int,
    ell: int,
    s: float,
    j: float,
    m_j: float,
    B_tesla: float = 0.0,
    E_field_Vm: float = 0.0,
) -> ZeemanStarkResult:
    """Compute Zeeman + Stark energy perturbations.

    Parameters
    ----------
    Z : int
        Atomic number.
    n, ell, s, j, m_j : quantum numbers
    B_tesla : float
        External magnetic field (Tesla).
    E_field_Vm : float
        External electric field (V/m).

    Returns
    -------
    ZeemanStarkResult
    """
    # Unperturbed energy
    e_n = -RYDBERG_EV * Z**2 / n**2

    # Landé g-factor
    g_j = _lande_g(j, ell, s)

    # Zeeman splitting: ΔE = m_j · g_J · μ_B · B
    delta_zeeman = m_j * g_j * MU_B_EV_PER_T * B_tesla

    # Number of Zeeman sub-levels
    n_levels = int(2 * j + 1)

    # Stark shift (quadratic, with l-dependent polarizability)
    # Polarizability volume: α_vol = (a₀³/2) · n⁴(5n² + 1 − 3l(l+1)) / Z⁴
    #   (Dalgarno 1962; Sobel'man 1972)
    # SI polarizability: α_SI = 4πε₀ · α_vol  (conversion from CGS volume)
    # Energy shift: ΔE = −½ · α_SI · E²
    #
    # This replaces the old n⁷(n²−1)/Z⁴ formula which:
    #   - Gave zero for n=1 (ground state)
    #   - Was missing 4πε₀ conversion (wrong by ~10¹⁰)
    #   - Ignored l-dependence entirely
    if Z > 0 and E_field_Vm != 0.0:
        alpha_factor = n**4 * (5 * n**2 + 1 - 3 * ell * (ell + 1))
        alpha_vol = 0.5 * A_BOHR_M**3 * alpha_factor / Z**4
        alpha_si = FOUR_PI_EPS0 * alpha_vol
        delta_stark_j = -0.5 * alpha_si * E_field_Vm**2
        delta_stark_ev = delta_stark_j / E_CHARGE  # J → eV
    else:
        delta_stark_ev = 0.0

    delta_total = delta_zeeman + delta_stark_ev

    # GCD mapping
    omega_eff = abs(delta_total / e_n) if abs(e_n) > 0 else 0.0
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(omega_eff)

    return ZeemanStarkResult(
        E_n_eV=round(e_n, 6),
        delta_E_zeeman_eV=round(delta_zeeman, 10),
        delta_E_stark_eV=round(delta_stark_ev, 10),
        delta_E_total_eV=round(delta_total, 10),
        g_lande=round(g_j, 6),
        n_zeeman_levels=n_levels,
        omega_eff=round(omega_eff, 8),
        F_eff=round(f_eff, 8),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # H n=2, l=1, j=3/2 in 1 Tesla
    r = compute_zeeman_stark(1, 2, 1, 0.5, 1.5, 0.5, B_tesla=1.0)
    print(f"H 2p₃/₂  B=1T: ΔE_Z={r.delta_E_zeeman_eV:.8f} eV  g={r.g_lande:.4f}  {r.regime}")
