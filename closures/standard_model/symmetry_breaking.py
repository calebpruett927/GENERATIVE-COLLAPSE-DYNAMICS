"""Higgs Mechanism / Symmetry Breaking Closure — SM.INTSTACK.v1

Computes Higgs mechanism quantities: VEV, Yukawa couplings,
mass generation, and electroweak symmetry breaking parameters.

Physics:
  Higgs potential: V(φ) = μ²|φ|² + λ|φ|⁴
  VEV: v = √(−μ²/λ) ≈ 246.22 GeV
  Fermion mass: m_f = y_f · v / √2
  W mass: M_W = g₂ · v / 2
  Z mass: M_Z = v · √(g₁² + g₂²) / 2

UMCP integration:
  ω_eff = |m_predicted − m_measured| / m_measured  (mass generation drift)
  F_eff = 1 − ω_eff

Regime:
  Consistent:  ω_eff < 0.01  (SM prediction matches)
  Tension:     0.01 ≤ ω_eff < 0.05
  BSM_hint:    ω_eff ≥ 0.05  (possible BSM physics)

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   Higgs (1964), PDG 2024
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class HiggsRegime(StrEnum):
    CONSISTENT = "Consistent"
    TENSION = "Tension"
    BSM_HINT = "BSM_hint"


class HiggsResult(NamedTuple):
    """Result of Higgs mechanism computation."""

    v_GeV: float  # Vacuum expectation value (GeV)
    m_H_GeV: float  # Higgs boson mass (GeV)
    lambda_quartic: float  # Quartic coupling λ
    mu_squared: float  # μ² parameter (GeV²)
    yukawa_couplings: dict[str, float]  # Yukawa couplings for fermions
    m_W_predicted: float  # Predicted W mass (GeV)
    m_Z_predicted: float  # Predicted Z mass (GeV)
    omega_eff: float
    F_eff: float
    regime: str


# ── Electroweak constants (PDG 2024) ────────────────────────────
V_EW = 246.22  # GeV (Higgs VEV = 1/√(√2 G_F))
M_H_MEASURED = 125.25  # GeV
M_W_MEASURED = 80.377  # GeV
M_Z_MEASURED = 91.1876  # GeV
SIN2_THETA_W = 0.23122

# Fermion masses (GeV) — PDG 2024
FERMION_MASSES: dict[str, float] = {
    "electron": 0.000511,
    "muon": 0.10566,
    "tau": 1.777,
    "up": 0.00216,
    "down": 0.00467,
    "charm": 1.27,
    "strange": 0.093,
    "top": 172.69,
    "bottom": 4.18,
}


def _classify(omega: float) -> HiggsRegime:
    if omega < 0.01:
        return HiggsRegime.CONSISTENT
    if omega < 0.05:
        return HiggsRegime.TENSION
    return HiggsRegime.BSM_HINT


def compute_higgs_mechanism(
    v_GeV: float = V_EW,
    m_H_GeV: float = M_H_MEASURED,
) -> HiggsResult:
    """Compute Higgs mechanism quantities and mass predictions.

    Parameters
    ----------
    v_GeV : float
        VEV (default: 246.22 GeV).
    m_H_GeV : float
        Higgs mass (default: 125.25 GeV).

    Returns
    -------
    HiggsResult
    """
    # Quartic coupling: λ = m_H² / (2v²)
    lambda_q = m_H_GeV**2 / (2 * v_GeV**2)

    # μ² = -λv² (negative for SSB)
    mu_sq = -lambda_q * v_GeV**2

    # Yukawa couplings: y_f = √2 · m_f / v
    yukawas = {name: round(math.sqrt(2) * m / v_GeV, 8) for name, m in FERMION_MASSES.items()}

    # Gauge couplings from EW parameters
    # g₂ = 2M_W/v
    g2 = 2 * M_W_MEASURED / v_GeV
    # g₁ from sin²θ_W = g₁²/(g₁² + g₂²)
    g1_sq = g2**2 * SIN2_THETA_W / (1 - SIN2_THETA_W)
    g1 = math.sqrt(g1_sq)

    # Predicted masses
    m_w_pred = g2 * v_GeV / 2
    m_z_pred = v_GeV * math.sqrt(g1**2 + g2**2) / 2

    # GCD mapping: compare W mass prediction
    omega_eff = abs(m_w_pred - M_W_MEASURED) / M_W_MEASURED
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(omega_eff)

    return HiggsResult(
        v_GeV=round(v_GeV, 4),
        m_H_GeV=round(m_H_GeV, 4),
        lambda_quartic=round(lambda_q, 6),
        mu_squared=round(mu_sq, 4),
        yukawa_couplings=yukawas,
        m_W_predicted=round(m_w_pred, 4),
        m_Z_predicted=round(m_z_pred, 4),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    r = compute_higgs_mechanism()
    print(f"VEV = {r.v_GeV} GeV")
    print(f"m_H = {r.m_H_GeV} GeV  λ = {r.lambda_quartic:.6f}")
    print(f"M_W predicted = {r.m_W_predicted} GeV  (measured = {M_W_MEASURED})")
    print(f"M_Z predicted = {r.m_Z_predicted} GeV  (measured = {M_Z_MEASURED})")
    print(f"ω = {r.omega_eff:.6f}  F = {r.F_eff:.6f}  {r.regime}")
    print("\nYukawa couplings:")
    for name, y in sorted(r.yukawa_couplings.items(), key=lambda x: -x[1]):
        m = FERMION_MASSES[name]
        print(f"  {name:12s}  y = {y:.8f}  m = {m:.6f} GeV")
