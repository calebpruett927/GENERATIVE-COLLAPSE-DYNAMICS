"""Cross Sections Closure — SM.INTSTACK.v1

Computes Standard Model cross sections and the R-ratio for
e⁺e⁻ → hadrons, validating QCD predictions.

Physics:
  R-ratio: R(s) = σ(e⁺e⁻→hadrons) / σ(e⁺e⁻→μ⁺μ⁻)
  Point cross section: σ_pt = 4πα²/(3s)  (at tree level)
  R_QCD = N_c · Σ_f Q_f² · (1 + α_s/π + ...)

UMCP integration:
  ω_eff = |R_measured − R_predicted| / R_measured
  F_eff = 1 − ω_eff

Regime:
  Validated:     ω_eff < 0.02
  Tension:       0.02 ≤ ω_eff < 0.10
  Anomalous:     ω_eff ≥ 0.10

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   PDG 2024, Bethke (2009)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class CrossSectionRegime(StrEnum):
    VALIDATED = "Validated"
    TENSION = "Tension"
    ANOMALOUS = "Anomalous"


class CrossSectionResult(NamedTuple):
    """Result of cross section computation."""

    sqrt_s_GeV: float
    sigma_point_pb: float  # Point cross section (pb)
    R_predicted: float  # R-ratio prediction
    R_QCD_corrected: float  # With QCD correction
    n_colors: int  # N_c = 3
    n_active_flavors: int
    sum_Qf_squared: float  # Σ Q_f²
    alpha_s_at_s: float
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
NC = 3  # Number of colors
ALPHA_EM = 1.0 / 137.036
HBARC2_GEV2_PB = 3.8937966e8  # (ℏc)² in GeV²·pb

QUARK_CHARGES = [2 / 3, -1 / 3, 2 / 3, -1 / 3, 2 / 3, -1 / 3]  # u, d, c, s, t, b
QUARK_MASSES_GEV = [0.00216, 0.00467, 0.093, 1.27, 4.18, 172.69]

# Experimental R-ratio reference points (PDG)
R_EXPERIMENTAL: dict[str, tuple[float, float]] = {
    "below_charm": (2.0, 2.17),  # √s < 3 GeV: R ≈ 2
    "charm_region": (4.0, 3.56),  # √s ~ 4 GeV: R ≈ 3.5
    "bottom_region": (12.0, 3.85),  # √s ~ 12 GeV: R ≈ 3.9
    "Z_peak": (91.2, 20.79),  # Z resonance
    "above_Z": (200.0, 4.0),  # √s > M_Z
}


def _classify(omega: float) -> CrossSectionRegime:
    if omega < 0.02:
        return CrossSectionRegime.VALIDATED
    if omega < 0.10:
        return CrossSectionRegime.TENSION
    return CrossSectionRegime.ANOMALOUS


def compute_cross_section(
    sqrt_s_GeV: float,
    *,
    R_measured: float | None = None,
    alpha_s: float | None = None,
) -> CrossSectionResult:
    """Compute e⁺e⁻ → hadrons cross section and R-ratio.

    Parameters
    ----------
    sqrt_s_GeV : float
        Center-of-mass energy (GeV).
    R_measured : float | None
        Measured R-ratio for comparison.
    alpha_s : float | None
        Strong coupling.  If None, uses 1-loop running.

    Returns
    -------
    CrossSectionResult
    """
    if sqrt_s_GeV <= 0:
        msg = f"√s must be > 0, got {sqrt_s_GeV}"
        raise ValueError(msg)

    s = sqrt_s_GeV**2

    # Point cross section: σ_pt = 4πα²ℏc² / (3s)
    sigma_pt = 4 * math.pi * ALPHA_EM**2 * HBARC2_GEV2_PB / (3 * s)

    # Active flavors
    n_f = sum(1 for m_q in QUARK_MASSES_GEV if sqrt_s_GeV > 2 * m_q)

    # Sum of squared charges
    sum_qf2 = sum(QUARK_CHARGES[i] ** 2 for i in range(n_f))

    # R-ratio at tree level
    R_tree = NC * sum_qf2

    # QCD correction
    if alpha_s is None:
        # Simple 1-loop running
        from closures.standard_model.coupling_constants import compute_running_coupling

        coupling = compute_running_coupling(sqrt_s_GeV)
        a_s = coupling.alpha_s
    else:
        a_s = alpha_s

    R_qcd = R_tree * (1 + a_s / math.pi)

    # Compare with measurement
    r_ref = R_measured if R_measured is not None else R_qcd

    omega_eff = abs(R_qcd - r_ref) / r_ref if r_ref > 0 else 0.0

    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff
    regime = _classify(omega_eff)

    return CrossSectionResult(
        sqrt_s_GeV=round(sqrt_s_GeV, 4),
        sigma_point_pb=round(sigma_pt, 4),
        R_predicted=round(R_tree, 6),
        R_QCD_corrected=round(R_qcd, 6),
        n_colors=NC,
        n_active_flavors=n_f,
        sum_Qf_squared=round(sum_qf2, 6),
        alpha_s_at_s=round(a_s, 6),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    for e_name, (sqrt_s, r_exp) in R_EXPERIMENTAL.items():
        r = compute_cross_section(sqrt_s, R_measured=r_exp)
        print(
            f"{e_name:18s} √s={sqrt_s:6.1f}  R_tree={r.R_predicted:.3f}  "
            f"R_QCD={r.R_QCD_corrected:.3f}  R_exp={r_exp:.2f}  ω={r.omega_eff:.4f}  {r.regime}"
        )
