"""Alpha Decay Closure — NUC.INTSTACK.v1

Computes Geiger-Nuttall/Gamow scaling for alpha decay, converting
half-life to mean lifetime and mapping to GCD trace channel Ψ_logτ.

Physics:
  log₁₀(T½) = a(Z_daughter) / √Q_α + b(Z_daughter)
  τ_mean = T½ / ln(2)                     [frozen lifetime choice]
  Q_α = M_parent − M_daughter − M_He4     [MeV, from AME tables]

  When Q_α is not directly available from AME, approximate as:
    Q_α ≈ BE(daughter) + BE(He-4) − BE(parent)
  using total binding energies.

UMCP integration:
  Ψ_logτ = log₁₀(τ_mean) / τ_scale   (normalized, τ_scale ~ 25 for
           geological timescales, or per-study frozen choice)
  Ψ_Qα  = Q_α / Q_α_scale            (normalized, Q_α_scale = 10 MeV)

Regime classification (decay_urgency):
  Eternal:      τ_mean > 10^17 yr   (age of universe)
  Geological:   10^3 yr ≤ τ_mean ≤ 10^17 yr
  Laboratory:   10^−3 s ≤ τ_mean < 10^3 yr (~3.15×10^10 s)
  Ephemeral:    τ_mean < 10^−3 s

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml  (AX-N1, AX-N2)
  Sources:   Geiger & Nuttall 1911; Gamow 1928; Gurney & Condon 1929
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class DecayRegime(StrEnum):
    """Regime based on mean lifetime magnitude."""

    ETERNAL = "Eternal"
    GEOLOGICAL = "Geological"
    LABORATORY = "Laboratory"
    EPHEMERAL = "Ephemeral"
    STABLE = "Stable"


class AlphaDecayResult(NamedTuple):
    """Result of alpha decay computation."""

    Q_alpha: float  # Alpha decay energy (MeV)
    half_life_s: float  # Half-life in seconds
    mean_lifetime_s: float  # Mean lifetime τ = T½/ln(2) in seconds
    log10_half_life_s: float  # log₁₀(T½/s)
    log10_tau_s: float  # log₁₀(τ/s)
    Psi_log_tau: float  # Normalized log₁₀(τ) trace channel
    Psi_Q_alpha: float  # Normalized Q_α trace channel
    regime: str  # Decay urgency regime


# ── Frozen constants ─────────────────────────────────────────────
LN2 = math.log(2)  # 0.693147...
Q_ALPHA_SCALE = 10.0  # MeV (normalization)
TAU_SCALE = 25.0  # log₁₀ decades for Ψ_logτ normalization
SEC_PER_YEAR = 3.15576e7  # s/yr (Julian year)

# Regime thresholds
ETERNAL_THRESHOLD_YR = 1e17  # > age of universe
GEOLOGICAL_LOWER_YR = 1e3  # 1000 years
LABORATORY_LOWER_S = 1e-3  # 1 ms

# Geiger-Nuttall a,b coefficients for selected Z_daughter values
# Empirical fits: log₁₀(T½/s) = a/√(Q_α/MeV) + b
# Source: Viola & Seaborg 1966, updated Royer 2000
GEIGER_NUTTALL_COEFFS: dict[int, tuple[float, float]] = {
    # Z_daughter: (a, b)
    78: (60.37, -41.17),  # Pt
    80: (61.68, -42.27),  # Hg
    82: (63.26, -43.56),  # Pb
    84: (64.11, -44.18),  # Po
    86: (65.42, -45.23),  # Rn
    88: (66.67, -46.21),  # Ra
    90: (67.88, -47.15),  # Th
    92: (69.07, -48.07),  # U
    94: (70.23, -48.96),  # Pu
    96: (71.36, -49.82),  # Cm
    98: (72.47, -50.66),  # Cf
}

# Fallback Viola-Seaborg universal fit
VS_A = 1.66175
VS_B = -8.5166
VS_C = -0.20228
VS_D = -33.9069


def _classify_regime(tau_s: float) -> DecayRegime:
    """Classify decay urgency regime by mean lifetime."""
    if tau_s == float("inf"):
        return DecayRegime.STABLE
    tau_yr = tau_s / SEC_PER_YEAR
    if tau_yr > ETERNAL_THRESHOLD_YR:
        return DecayRegime.ETERNAL
    if tau_yr >= GEOLOGICAL_LOWER_YR:
        return DecayRegime.GEOLOGICAL
    if tau_s >= LABORATORY_LOWER_S:
        return DecayRegime.LABORATORY
    return DecayRegime.EPHEMERAL


def _geiger_nuttall_log10_halflife(
    Z_daughter: int,
    Q_alpha_MeV: float,
) -> float:
    """Compute log₁₀(T½/s) using Geiger-Nuttall relation.

    Uses tabulated a,b coefficients if available, otherwise falls
    back to the Viola-Seaborg universal parameterization.
    """
    if Q_alpha_MeV <= 0:
        return float("inf")

    sqrt_Q = math.sqrt(Q_alpha_MeV)

    if Z_daughter in GEIGER_NUTTALL_COEFFS:
        a, b = GEIGER_NUTTALL_COEFFS[Z_daughter]
        return a / sqrt_Q + b

    # Viola-Seaborg universal: log₁₀(T½) = (aZ+b)/√Q + (cZ+d)
    a_vs = VS_A * Z_daughter + VS_B
    b_vs = VS_C * Z_daughter + VS_D
    return a_vs / sqrt_Q + b_vs


def compute_alpha_decay(
    Z_parent: int,
    A_parent: int,
    Q_alpha_MeV: float,
    *,
    half_life_s_measured: float | None = None,
) -> AlphaDecayResult:
    """Compute alpha decay properties and GCD mapping.

    Parameters
    ----------
    Z_parent : int
        Parent atomic number.
    A_parent : int
        Parent mass number.
    Q_alpha_MeV : float
        Alpha decay energy in MeV. Use 0.0 for stable nuclei
        (energetically forbidden alpha decay).
    half_life_s_measured : float | None
        If provided, use measured half-life instead of Geiger-Nuttall
        estimate. Preferred when AME/NUBASE data is available.

    Returns
    -------
    AlphaDecayResult
        Named tuple with decay metrics and GCD mapping.
    """
    Z_daughter = Z_parent - 2
    # A_daughter = A_parent - 4   # implies daughter identity (unused)

    # Determine half-life
    if half_life_s_measured is not None:
        t_half_s = half_life_s_measured
    elif Q_alpha_MeV <= 0:
        t_half_s = float("inf")
    else:
        log10_t = _geiger_nuttall_log10_halflife(Z_daughter, Q_alpha_MeV)
        t_half_s = 10.0**log10_t

    # Mean lifetime: τ = T½/ln(2) [frozen choice per AX-N1]
    if t_half_s == float("inf"):
        tau_s = float("inf")
        log10_t_half = float("inf")
        log10_tau = float("inf")
    else:
        tau_s = t_half_s / LN2
        log10_t_half = math.log10(t_half_s) if t_half_s > 0 else float("-inf")
        log10_tau = math.log10(tau_s) if tau_s > 0 else float("-inf")

    # Normalized channels
    if log10_tau == float("inf") or log10_tau == float("-inf"):
        psi_log_tau = 1.0 if log10_tau == float("inf") else 0.0
    else:
        psi_log_tau = max(0.0, min(1.0, log10_tau / TAU_SCALE))

    psi_q_alpha = max(0.0, min(1.0, Q_alpha_MeV / Q_ALPHA_SCALE))

    regime = _classify_regime(tau_s)

    return AlphaDecayResult(
        Q_alpha=round(Q_alpha_MeV, 4),
        half_life_s=t_half_s,
        mean_lifetime_s=tau_s,
        log10_half_life_s=round(log10_t_half, 4) if log10_t_half != float("inf") else float("inf"),
        log10_tau_s=round(log10_tau, 4) if log10_tau != float("inf") else float("inf"),
        Psi_log_tau=round(psi_log_tau, 6),
        Psi_Q_alpha=round(psi_q_alpha, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # (Z, A, Q_α MeV, measured T½ s, name)
        (92, 238, 4.270, 1.41e17, "U-238"),
        (90, 234, 4.857, 7.54e12, "Th-234 (β-emitter, Q_α shown for context)"),
        (88, 226, 4.871, 5.05e10, "Ra-226"),
        (86, 222, 5.590, 3.30e5, "Rn-222"),
        (84, 218, 6.115, 1.86e2, "Po-218"),
        (84, 214, 7.833, 1.64e-4, "Po-214"),
        (84, 210, 5.407, 1.20e7, "Po-210"),
        (83, 209, 3.137, 6.01e26, "Bi-209 (quasi-stable)"),
    ]

    for z, a, q, t_half, name in tests:
        r = compute_alpha_decay(z, a, q, half_life_s_measured=t_half)
        print(
            f"{name:40s}  Q_α={r.Q_alpha:6.3f}  "
            f"log₁₀(τ/s)={r.log10_tau_s!s:>8s}  "
            f"Ψ_logτ={r.Psi_log_tau:.4f}  "
            f"Ψ_Qα={r.Psi_Q_alpha:.4f}  {r.regime}"
        )
