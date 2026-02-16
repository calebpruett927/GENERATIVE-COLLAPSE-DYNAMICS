"""PMNS Neutrino Mixing Matrix Closure — SM.INTSTACK.v1

Computes the PMNS (Pontecorvo-Maki-Nakagawa-Sakata) lepton mixing matrix
and neutrino oscillation parameters.

Physics:
  U_PMNS parametrized by 3 angles (θ₁₂, θ₂₃, θ₁₃) and 1 Dirac CP phase (δ_CP):
    |U_PMNS| ≈ [[U_e1   U_e2   U_e3 ]
                 [U_μ1   U_μ2   U_μ3 ]
                 [U_τ1   U_τ2   U_τ3 ]]

  NuFIT 5.3 (2024) global fit (normal ordering):
    sin²θ₁₂ = 0.303 ± 0.012
    sin²θ₂₃ = 0.572 ± 0.018
    sin²θ₁₃ = 0.02203 ± 0.00056
    δ_CP     = 197° +27°/-24°

  Mass-squared differences:
    Δm²₂₁ = (7.41 ± 0.21) × 10⁻⁵ eV²
    Δm²₃₂ = (2.507 ± 0.027) × 10⁻³ eV²  (normal ordering)

  Jarlskog invariant: J_CP ≈ −0.033 sin(δ_CP)

Structural comparison with CKM:
  CKM (quarks):  Small mixing — |V_ub| ≈ 0.004 → large heterogeneity gap
  PMNS (leptons): Large mixing — sin²θ₁₂ ≈ 0.30  → small heterogeneity gap
  This is Theorem T11 (Mixing Complementarity): quark and lepton sectors
  exhibit complementary patterns when viewed through the GCD kernel.

UMCP integration:
  ω_eff = |1 − row_unitarity|   (unitarity deficit)
  F_eff = 1 − ω_eff
  IC computed from 4-channel trace per row: [|U_i1|², |U_i2|², |U_i3|², unitarity]

Regime:
  Unitary:     ω_eff < 0.001
  Tension:     0.001 ≤ ω_eff < 0.01
  BSM_hint:    ω_eff ≥ 0.01   (sterile neutrino hints)

Cross-references:
  Contract:  contracts/SM.INTSTACK.v1.yaml
  Sources:   NuFIT 5.3 (2024), PDG 2024, Pontecorvo (1957), MNS (1962)
  Sibling:   closures/standard_model/ckm_mixing.py (quark sector)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class PMNSRegime(StrEnum):
    UNITARY = "Unitary"
    TENSION = "Tension"
    BSM_HINT = "BSM_hint"


class PMNSResult(NamedTuple):
    """Result of PMNS mixing computation."""

    U_matrix: list[list[float]]  # |U_αi| magnitudes (3×3)
    sin2_theta12: float  # sin²θ₁₂ (solar angle)
    sin2_theta23: float  # sin²θ₂₃ (atmospheric angle)
    sin2_theta13: float  # sin²θ₁₃ (reactor angle)
    delta_CP_deg: float  # Dirac CP phase (degrees)
    J_CP: float  # Jarlskog invariant
    dm2_21: float  # Δm²₂₁ (eV²)
    dm2_32: float  # Δm²₃₂ (eV²)
    unitarity_row_e: float  # |U_e1|² + |U_e2|² + |U_e3|² (should = 1)
    unitarity_row_mu: float
    unitarity_row_tau: float
    mixing_entropy: float  # −Σ |U_αi|² ln |U_αi|² per row (avg)
    omega_eff: float
    F_eff: float
    regime: str


class MixingComparison(NamedTuple):
    """Comparison between CKM (quark) and PMNS (lepton) mixing."""

    ckm_max_mixing: float  # max off-diagonal |V_ij|² in CKM
    pmns_max_mixing: float  # max off-diagonal |U_αi|² in PMNS
    ckm_entropy: float  # mixing entropy of CKM
    pmns_entropy: float  # mixing entropy of PMNS
    complementarity_12: float  # θ₁₂_CKM + θ₁₂_PMNS (should ≈ π/4)
    complementarity_deficit: float  # |θ₁₂_CKM + θ₁₂_PMNS − π/4|
    ckm_heterogeneity: float  # F − IC for CKM row
    pmns_heterogeneity: float  # F − IC for PMNS row
    verdict: str  # "Complementary" or "Non-complementary"


# ── NuFIT 5.3 (2024) — Normal Ordering ──────────────────────────
SIN2_THETA12 = 0.303  # ± 0.012
SIN2_THETA23 = 0.572  # ± 0.018
SIN2_THETA13 = 0.02203  # ± 0.00056
DELTA_CP_DEG = 197.0  # +27/-24 degrees

# Mass-squared differences (eV²)
DM2_21 = 7.41e-5  # ± 0.21e-5 (solar)
DM2_32 = 2.507e-3  # ± 0.027e-3 (atmospheric, NO)

# Guard band
EPSILON = 1e-8


def _classify(omega: float) -> PMNSRegime:
    if omega < 0.001:
        return PMNSRegime.UNITARY
    if omega < 0.01:
        return PMNSRegime.TENSION
    return PMNSRegime.BSM_HINT


def _mixing_entropy(row: list[float]) -> float:
    """Bernoulli field entropy of a mixing matrix row.

    Each |U_αi|² is a collapse channel — the entropy measures
    how evenly distributed the flavor content is across mass
    eigenstates. Maximal mixing → max entropy → democratic collapse.
    """
    s = 0.0
    for p in row:
        p2 = p**2
        if p2 > EPSILON:
            s -= p2 * math.log(p2)
    return s


def compute_pmns_mixing(
    sin2_theta12: float = SIN2_THETA12,
    sin2_theta23: float = SIN2_THETA23,
    sin2_theta13: float = SIN2_THETA13,
    delta_CP_deg: float = DELTA_CP_DEG,
    dm2_21: float = DM2_21,
    dm2_32: float = DM2_32,
) -> PMNSResult:
    """Compute PMNS matrix from mixing angles and CP phase.

    Parameters
    ----------
    sin2_theta12 : Solar mixing angle sin²θ₁₂
    sin2_theta23 : Atmospheric mixing angle sin²θ₂₃
    sin2_theta13 : Reactor mixing angle sin²θ₁₃
    delta_CP_deg : Dirac CP phase in degrees
    dm2_21       : Solar mass-squared difference (eV²)
    dm2_32       : Atmospheric mass-squared difference (eV²)

    Returns
    -------
    PMNSResult with full PMNS matrix, unitarity checks, and GCD mapping
    """
    # Extract angles
    s12 = math.sqrt(sin2_theta12)
    c12 = math.sqrt(1 - sin2_theta12)
    s23 = math.sqrt(sin2_theta23)
    c23 = math.sqrt(1 - sin2_theta23)
    s13 = math.sqrt(sin2_theta13)
    c13 = math.sqrt(1 - sin2_theta13)
    delta = math.radians(delta_CP_deg)

    # Standard PDG parametrization (magnitudes only for |U_αi|)
    # Full complex matrix for Jarlskog, but we store magnitudes
    # U = R₂₃ · U_δ · R₁₃ · U_δ† · R₁₂
    #
    # |U_e1|  = c12 * c13
    # |U_e2|  = s12 * c13
    # |U_e3|  = s13
    # |U_μ1|  = |−s12*c23 − c12*s23*s13*e^(iδ)|
    # |U_μ2|  = |c12*c23 − s12*s23*s13*e^(iδ)|
    # |U_μ3|  = s23 * c13
    # |U_τ1|  = |s12*s23 − c12*c23*s13*e^(iδ)|
    # |U_τ2|  = |−c12*s23 − s12*c23*s13*e^(iδ)|
    # |U_τ3|  = c23 * c13

    cos_d = math.cos(delta)
    sin_d = math.sin(delta)

    u_e1 = c12 * c13
    u_e2 = s12 * c13
    u_e3 = s13

    # μ row (complex magnitudes)
    u_mu1_re = -s12 * c23 - c12 * s23 * s13 * cos_d
    u_mu1_im = -c12 * s23 * s13 * sin_d
    u_mu1 = math.sqrt(u_mu1_re**2 + u_mu1_im**2)

    u_mu2_re = c12 * c23 - s12 * s23 * s13 * cos_d
    u_mu2_im = -s12 * s23 * s13 * sin_d
    u_mu2 = math.sqrt(u_mu2_re**2 + u_mu2_im**2)

    u_mu3 = s23 * c13

    # τ row (complex magnitudes)
    u_tau1_re = s12 * s23 - c12 * c23 * s13 * cos_d
    u_tau1_im = -c12 * c23 * s13 * sin_d
    u_tau1 = math.sqrt(u_tau1_re**2 + u_tau1_im**2)

    u_tau2_re = -c12 * s23 - s12 * c23 * s13 * cos_d
    u_tau2_im = -s12 * c23 * s13 * sin_d
    u_tau2 = math.sqrt(u_tau2_re**2 + u_tau2_im**2)

    u_tau3 = c23 * c13

    U = [
        [u_e1, u_e2, u_e3],
        [u_mu1, u_mu2, u_mu3],
        [u_tau1, u_tau2, u_tau3],
    ]

    # Row unitarity
    u_e = sum(x**2 for x in U[0])
    u_mu = sum(x**2 for x in U[1])
    u_tau = sum(x**2 for x in U[2])

    # Jarlskog invariant: J = c12 s12 c23 s23 c13² s13 sin(δ)
    J = c12 * s12 * c23 * s23 * c13**2 * s13 * sin_d

    # Mixing entropy (averaged over rows)
    ent = sum(_mixing_entropy(row) for row in U) / 3.0

    # GCD mapping
    omega_eff = max(abs(1 - u_e), abs(1 - u_mu), abs(1 - u_tau))
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(omega_eff)

    return PMNSResult(
        U_matrix=[[round(x, 6) for x in row] for row in U],
        sin2_theta12=round(sin2_theta12, 6),
        sin2_theta23=round(sin2_theta23, 6),
        sin2_theta13=round(sin2_theta13, 6),
        delta_CP_deg=round(delta_CP_deg, 2),
        J_CP=round(J, 8),
        dm2_21=dm2_21,
        dm2_32=dm2_32,
        unitarity_row_e=round(u_e, 8),
        unitarity_row_mu=round(u_mu, 8),
        unitarity_row_tau=round(u_tau, 8),
        mixing_entropy=round(ent, 6),
        omega_eff=round(omega_eff, 8),
        F_eff=round(f_eff, 8),
        regime=regime.value,
    )


def compute_mixing_comparison() -> MixingComparison:
    """Compare CKM (quark) and PMNS (lepton) mixing patterns.

    This function computes the quark-lepton complementarity hypothesis
    (θ₁₂_CKM + θ₁₂_PMNS ≈ π/4) and compares the heterogeneity gap
    Δ = F − IC across sectors.

    Returns
    -------
    MixingComparison with sector comparison and complementarity test
    """
    # PMNS mixing
    pmns = compute_pmns_mixing()

    # CKM row 1 (from PDG values)
    ckm_row1 = [0.97373, 0.2243, 0.00382]

    # Mixing angles for complementarity
    theta12_ckm = math.asin(0.2243)  # ≈ Cabibbo angle
    theta12_pmns = math.asin(math.sqrt(SIN2_THETA12))  # solar angle

    # Complementarity: θ₁₂_CKM + θ₁₂_PMNS ≈ π/4
    comp_sum = theta12_ckm + theta12_pmns
    comp_deficit = abs(comp_sum - math.pi / 4)

    # Mixing entropy comparison
    ckm_ent = _mixing_entropy(ckm_row1)
    pmns_ent = _mixing_entropy(pmns.U_matrix[0])

    # Max off-diagonal mixing
    ckm_max = max(0.2243**2, 0.00382**2)
    pmns_max = max(pmns.U_matrix[0][1] ** 2, pmns.U_matrix[0][2] ** 2)

    # Heterogeneity gap Δ = F − IC per row
    # For CKM row 1: channels are |V_ui|²
    ckm_channels = [x**2 for x in ckm_row1]
    ckm_f = sum(ckm_channels) / len(ckm_channels)
    ckm_ic = 1.0
    for c in ckm_channels:
        ckm_ic *= max(c, EPSILON)
    ckm_ic = ckm_ic ** (1.0 / len(ckm_channels))
    ckm_het = ckm_f - ckm_ic

    # For PMNS row e: channels are |U_ei|²
    pmns_channels = [x**2 for x in pmns.U_matrix[0]]
    pmns_f = sum(pmns_channels) / len(pmns_channels)
    pmns_ic = 1.0
    for c in pmns_channels:
        pmns_ic *= max(c, EPSILON)
    pmns_ic = pmns_ic ** (1.0 / len(pmns_channels))
    pmns_het = pmns_f - pmns_ic

    # Verdict: complementary if deficit < 5°
    verdict = "Complementary" if comp_deficit < math.radians(5.0) else "Non-complementary"

    return MixingComparison(
        ckm_max_mixing=round(ckm_max, 6),
        pmns_max_mixing=round(pmns_max, 6),
        ckm_entropy=round(ckm_ent, 6),
        pmns_entropy=round(pmns_ent, 6),
        complementarity_12=round(math.degrees(comp_sum), 2),
        complementarity_deficit=round(math.degrees(comp_deficit), 2),
        ckm_heterogeneity=round(ckm_het, 6),
        pmns_heterogeneity=round(pmns_het, 6),
        verdict=verdict,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    r = compute_pmns_mixing()
    print("PMNS Matrix |U_αi|:")
    labels_r = ["e", "μ", "τ"]
    labels_c = ["1", "2", "3"]
    for i, row in enumerate(r.U_matrix):
        print(f"  ν_{labels_r[i]}: " + "  ".join(f"{v:.6f}" for v in row))
    print(f"\nAngles: sin²θ₁₂={r.sin2_theta12}  sin²θ₂₃={r.sin2_theta23}  sin²θ₁₃={r.sin2_theta13}")
    print(f"δ_CP = {r.delta_CP_deg}°")
    print(f"Jarlskog J = {r.J_CP:.8f}")
    print(f"Δm²₂₁ = {r.dm2_21:.2e} eV²  Δm²₃₂ = {r.dm2_32:.2e} eV²")
    print(f"Unitarity: row_e={r.unitarity_row_e:.8f}  row_μ={r.unitarity_row_mu:.8f}  row_τ={r.unitarity_row_tau:.8f}")
    print(f"Mixing entropy = {r.mixing_entropy:.6f}")
    print(f"ω = {r.omega_eff:.8f}  {r.regime}")

    print("\n── Quark-Lepton Complementarity ──")
    c = compute_mixing_comparison()
    print(f"CKM max mixing:  {c.ckm_max_mixing:.6f}  PMNS max mixing: {c.pmns_max_mixing:.6f}")
    print(f"CKM entropy:     {c.ckm_entropy:.6f}  PMNS entropy:    {c.pmns_entropy:.6f}")
    print(f"θ₁₂_CKM + θ₁₂_PMNS = {c.complementarity_12:.2f}° (deficit = {c.complementarity_deficit:.2f}°)")
    print(f"CKM Δ(F−IC):     {c.ckm_heterogeneity:.6f}  PMNS Δ(F−IC):  {c.pmns_heterogeneity:.6f}")
    print(f"Verdict: {c.verdict}")
