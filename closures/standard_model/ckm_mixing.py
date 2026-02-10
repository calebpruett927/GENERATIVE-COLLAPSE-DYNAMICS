"""CKM Mixing Matrix Closure — SM.INTSTACK.v1

Computes the CKM (Cabibbo-Kobayashi-Maskawa) quark mixing matrix
and unitarity triangle parameters.

Physics:
  V_CKM parametrized by 3 angles (θ₁₂, θ₂₃, θ₁₃) and 1 CP phase (δ):
    |V_CKM| ≈ [[V_ud  V_us  V_ub]
                [V_cd  V_cs  V_cb]
                [V_td  V_ts  V_tb]]

  Wolfenstein parametrization:
    λ = sin θ_C ≈ 0.22650
    A = sin θ₂₃ / λ² ≈ 0.790
    ρ̄ ≈ 0.141,  η̄ ≈ 0.357

  Unitarity triangle: V_ud V_ub* + V_cd V_cb* + V_td V_tb* = 0
  Jarlskog invariant: J ≈ 3.08 × 10⁻⁵

UMCP integration:
  ω_eff = |1 − row_unitarity|   (unitarity deficit)
  F_eff = 1 − ω_eff

Regime:
  Unitary:     ω_eff < 0.001
  Tension:     0.001 ≤ ω_eff < 0.01
  BSM_hint:    ω_eff ≥ 0.01

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   PDG 2024 (CKMfitter), Wolfenstein (1983)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class CKMRegime(StrEnum):
    UNITARY = "Unitary"
    TENSION = "Tension"
    BSM_HINT = "BSM_hint"


class CKMResult(NamedTuple):
    """Result of CKM mixing computation."""

    V_matrix: list[list[float]]  # |V_ij| magnitudes (3×3)
    lambda_wolf: float  # Wolfenstein λ
    A_wolf: float  # Wolfenstein A
    rho_bar: float  # Wolfenstein ρ̄
    eta_bar: float  # Wolfenstein η̄
    J_CP: float  # Jarlskog invariant
    unitarity_row1: float  # |V_ud|² + |V_us|² + |V_ub|² (should = 1)
    unitarity_row2: float
    unitarity_row3: float
    triangle_angles: dict[str, float]  # α, β, γ in degrees
    omega_eff: float
    F_eff: float
    regime: str


# ── PDG 2024 CKM elements (magnitudes) ──────────────────────────
V_UD = 0.97373
V_US = 0.2243
V_UB = 0.00382
V_CD = 0.221
V_CS = 0.975
V_CB = 0.0408
V_TD = 0.0086
V_TS = 0.0415
V_TB = 1.014  # Note: can be > 1 within uncertainties, usually quoted as ~0.999

# Wolfenstein parameters (PDG 2024)
LAMBDA_W = 0.22650
A_W = 0.790
RHO_BAR = 0.141
ETA_BAR = 0.357

# Jarlskog invariant
J_PDG = 3.08e-5


def _classify(omega: float) -> CKMRegime:
    if omega < 0.001:
        return CKMRegime.UNITARY
    if omega < 0.01:
        return CKMRegime.TENSION
    return CKMRegime.BSM_HINT


def compute_ckm_mixing(
    lambda_w: float = LAMBDA_W,
    A: float = A_W,
    rho_bar: float = RHO_BAR,
    eta_bar: float = ETA_BAR,
) -> CKMResult:
    """Compute CKM matrix from Wolfenstein parameters.

    Parameters
    ----------
    lambda_w, A, rho_bar, eta_bar : Wolfenstein parameters

    Returns
    -------
    CKMResult
    """
    lam = lambda_w
    lam2 = lam**2
    lam3 = lam**3

    # Wolfenstein parametrization (to O(λ³))
    V = [
        [1 - lam2 / 2, lam, A * lam3 * math.sqrt(rho_bar**2 + eta_bar**2)],
        [lam, 1 - lam2 / 2, A * lam2],
        [A * lam3 * (1 - rho_bar), A * lam2, 1.0],
    ]

    # Row unitarity checks
    u1 = sum(x**2 for x in V[0])
    u2 = sum(x**2 for x in V[1])
    u3 = sum(x**2 for x in V[2])

    # Jarlskog invariant
    J = A**2 * lam**6 * eta_bar

    # Unitarity triangle angles
    # α (phi_2), β (phi_1), γ (phi_3)
    beta = math.atan2(eta_bar, rho_bar) if rho_bar != 0 else math.pi / 2
    gamma = math.atan2(eta_bar, rho_bar)
    alpha = math.pi - beta - gamma  # unitarity constraint

    # GCD mapping: max unitarity deficit
    omega_eff = max(abs(1 - u1), abs(1 - u2), abs(1 - u3))
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(omega_eff)

    return CKMResult(
        V_matrix=[[round(x, 6) for x in row] for row in V],
        lambda_wolf=round(lam, 6),
        A_wolf=round(A, 6),
        rho_bar=round(rho_bar, 6),
        eta_bar=round(eta_bar, 6),
        J_CP=round(J, 8),
        unitarity_row1=round(u1, 8),
        unitarity_row2=round(u2, 8),
        unitarity_row3=round(u3, 8),
        triangle_angles={
            "alpha_deg": round(math.degrees(alpha), 2),
            "beta_deg": round(math.degrees(beta), 2),
            "gamma_deg": round(math.degrees(gamma), 2),
        },
        omega_eff=round(omega_eff, 8),
        F_eff=round(f_eff, 8),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    r = compute_ckm_mixing()
    print("CKM Matrix |V_ij|:")
    labels_r = ["u", "c", "t"]
    labels_c = ["d", "s", "b"]
    for i, row in enumerate(r.V_matrix):
        print(f"  {labels_r[i]}: " + "  ".join(f"{v:.6f}" for v in row))
    print(f"\nWolfenstein: λ={r.lambda_wolf} A={r.A_wolf} ρ̄={r.rho_bar} η̄={r.eta_bar}")
    print(f"Jarlskog J = {r.J_CP:.8f}")
    print(f"Unitarity: row1={r.unitarity_row1:.8f}  row2={r.unitarity_row2:.8f}  row3={r.unitarity_row3:.8f}")
    print(f"Triangle: {r.triangle_angles}")
    print(f"ω = {r.omega_eff:.8f}  {r.regime}")
