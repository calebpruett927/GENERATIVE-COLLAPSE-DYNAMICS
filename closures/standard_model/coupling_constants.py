"""Running Coupling Constants Closure — SM.INTSTACK.v1

Computes the running of Standard Model gauge coupling constants
with energy scale Q using 1-loop RGE.

Physics:
  Strong coupling α_s(Q²):
    α_s(Q) = α_s(M_Z) / [1 + (α_s(M_Z)·b₃/2π)·ln(Q²/M_Z²)]
    b₃ = 11 − 2n_f/3  (n_f = active quark flavors)

  Electromagnetic coupling α_em(Q²):
    α_em(Q) = α_em(0) / [1 − (α_em(0)/3π)·Σ_f Q_f² · ln(Q²/m_f²)]

  Weak coupling G_F:
    G_F = π α_em / (√2 · M_W² · sin²θ_W)

UMCP integration:
  ω_eff = |α(Q) − α(M_Z)| / α(M_Z)   (coupling drift from reference)
  F_eff = 1 − ω_eff
  Regime: Perturbative / Transitional / NonPerturbative

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   PDG 2024, Bethke (2009)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class CouplingRegime(StrEnum):
    PERTURBATIVE = "Perturbative"
    TRANSITIONAL = "Transitional"
    NON_PERTURBATIVE = "NonPerturbative"


class CouplingResult(NamedTuple):
    """Result of running coupling computation."""

    alpha_s: float  # Strong coupling at Q
    alpha_em: float  # EM coupling at Q
    sin2_theta_W: float  # Weak mixing angle
    G_F: float  # Fermi constant (GeV⁻²)
    Q_GeV: float  # Energy scale
    n_flavors: int  # Active quark flavors
    omega_eff: float  # Coupling drift from M_Z reference
    F_eff: float
    regime: str
    unification_proximity: float  # How close couplings are to unification


# ── PDG 2024 reference values ────────────────────────────────────
ALPHA_S_MZ = 0.1180  # α_s(M_Z)
ALPHA_EM_0 = 1.0 / 137.036  # α_em(0)
ALPHA_EM_MZ = 1.0 / 127.952  # α_em(M_Z)
M_Z_GEV = 91.1876  # Z boson mass (GeV)
M_W_GEV = 80.377  # W boson mass (GeV)
SIN2_THETA_W = 0.23122  # sin²θ_W at M_Z
G_FERMI = 1.1663788e-5  # Fermi constant (GeV⁻²)

# Quark mass thresholds (GeV)
QUARK_THRESHOLDS = [0.00216, 0.00467, 0.093, 1.27, 4.18, 172.69]


def _n_active_flavors(Q_gev: float) -> int:
    """Number of active quark flavors at energy scale Q."""
    return sum(1 for m_q in QUARK_THRESHOLDS if Q_gev > 2 * m_q)


def _alpha_s_running(Q_gev: float) -> float:
    """1-loop running of α_s."""
    if Q_gev <= 0.3:
        return 1.0  # Non-perturbative (Λ_QCD ~ 0.3 GeV)

    n_f = _n_active_flavors(Q_gev)
    b3 = 11.0 - 2.0 * n_f / 3.0

    if b3 <= 0:
        return ALPHA_S_MZ  # safety

    log_ratio = math.log(Q_gev**2 / M_Z_GEV**2)
    denom = 1.0 + ALPHA_S_MZ * b3 / (2.0 * math.pi) * log_ratio

    if denom <= 0:
        return 1.0  # Landau pole

    return ALPHA_S_MZ / denom


def _alpha_em_running(Q_gev: float) -> float:
    """1-loop running of α_em."""
    if Q_gev <= 0:
        return ALPHA_EM_0

    # Simplified: use known α_em(0) → α_em(M_Z) interpolation
    if Q_gev < 1.0:
        return ALPHA_EM_0
    if Q_gev <= M_Z_GEV:
        # Linear interpolation in log scale
        t = math.log(Q_gev) / math.log(M_Z_GEV)
        return ALPHA_EM_0 + t * (ALPHA_EM_MZ - ALPHA_EM_0)

    # Above M_Z: 1-loop QED running
    log_ratio = math.log(Q_gev**2 / M_Z_GEV**2)
    n_f = _n_active_flavors(Q_gev)
    # Sum of Q_f² for active quarks (u,d,c,s,t,b charges: 2/3, -1/3, ...)
    charge_sq_sum = 0.0
    charges = [2 / 3, -1 / 3, 2 / 3, -1 / 3, 2 / 3, -1 / 3]
    for i in range(min(n_f, 6)):
        charge_sq_sum += charges[i] ** 2
    # Add leptons (e, μ, τ)
    charge_sq_sum += 3.0  # 3 charged leptons

    denom = 1.0 - ALPHA_EM_MZ / (3.0 * math.pi) * charge_sq_sum * log_ratio

    if denom <= 0:
        return 1.0 / 100.0  # safety
    return ALPHA_EM_MZ / denom


def _classify(alpha_s: float) -> CouplingRegime:
    if alpha_s < 0.3:
        return CouplingRegime.PERTURBATIVE
    if alpha_s < 0.6:
        return CouplingRegime.TRANSITIONAL
    return CouplingRegime.NON_PERTURBATIVE


def compute_running_coupling(Q_GeV: float) -> CouplingResult:
    """Compute SM running couplings at energy scale Q.

    Parameters
    ----------
    Q_GeV : float
        Energy scale in GeV.

    Returns
    -------
    CouplingResult
    """
    if Q_GeV <= 0:
        msg = f"Q must be > 0, got {Q_GeV}"
        raise ValueError(msg)

    alpha_s = _alpha_s_running(Q_GeV)
    alpha_em = _alpha_em_running(Q_GeV)
    n_f = _n_active_flavors(Q_GeV)

    # Coupling drift relative to M_Z reference
    omega_eff = abs(alpha_s - ALPHA_S_MZ) / ALPHA_S_MZ
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    # Unification proximity: how close are the 3 couplings?
    # At GUT scale (~10¹⁶ GeV) they should converge
    g1 = math.sqrt(5 / 3) * math.sqrt(4 * math.pi * alpha_em) / math.sqrt(1 - SIN2_THETA_W)
    g2 = math.sqrt(4 * math.pi * alpha_em) / math.sqrt(SIN2_THETA_W)
    g3 = math.sqrt(4 * math.pi * alpha_s)
    mean_g = (g1 + g2 + g3) / 3
    if mean_g > 0:
        spread = ((g1 - mean_g) ** 2 + (g2 - mean_g) ** 2 + (g3 - mean_g) ** 2) / (3 * mean_g**2)
        unification_prox = max(0.0, 1.0 - math.sqrt(spread))
    else:
        unification_prox = 0.0

    regime = _classify(alpha_s)

    return CouplingResult(
        alpha_s=round(alpha_s, 6),
        alpha_em=round(alpha_em, 8),
        sin2_theta_W=round(SIN2_THETA_W, 6),
        G_F=G_FERMI,
        Q_GeV=round(Q_GeV, 4),
        n_flavors=n_f,
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
        unification_proximity=round(unification_prox, 6),
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    scales = [0.5, 1.0, 2.0, 10.0, 91.2, 500.0, 1000.0, 14000.0]
    for q in scales:
        r = compute_running_coupling(q)
        print(
            f"Q={q:8.1f} GeV  α_s={r.alpha_s:.4f}  α_em={r.alpha_em:.6f}  "
            f"n_f={r.n_flavors}  ω={r.omega_eff:.4f}  {r.regime}"
        )
