"""Surface & Catalysis Closure — MATL.INTSTACK.v1

Derives surface energetics and catalytic activity from bulk cohesive properties,
coordination geometry, and d-band electronic structure.

Physics — From Bulk Collapse to Surface Return:
  A surface is a TRUNCATED collapse — atoms at the surface have fewer neighbors
  than bulk atoms, so their collapse-return cycle is incomplete.  The missing
  bonds manifest as surface energy γ_s.

  Broken-bond model (Mackenzie 1962):
    γ_s = (E_coh / A_atom) · (1 − z_s/z_b) · f_relax
    where z_s = surface coordination, z_b = bulk coordination,
    f_relax ≈ 0.85-0.95 accounts for surface relaxation.

  Work of adhesion:
    W_ad = γ₁ + γ₂ − γ₁₂  (Dupré equation)

  Vacancy formation:
    E_vac ≈ E_coh · (1 − z_vac/z_b)  (broken bonds at vacancy site)

  d-band center theory (Hammer & Nørskov 1995, 2000):
    ε_d = Σ εᵢ · DOS(εᵢ) / Σ DOS(εᵢ)
    Adsorption: E_ads ≈ α·ε_d + β  (linear scaling, Sabatier volcano)
    BEP relation: E_a = γ·E_ads + δ  (Brønsted-Evans-Polanyi)

  RCFT interpretation:
    Surface energy = the cost of an INCOMPLETE return cycle.
    Catalysis = facilitating a collapse-return at the surface that
    wouldn't happen in isolation.  The optimal catalyst balances:
    - Too weak binding (high ω → reactants don't stick)
    - Too strong binding (low ω → products don't desorb)
    This is the Sabatier principle expressed as an RCFT attractor
    depth optimization.

UMCP integration:
  ω_eff = |γ_predicted − γ_measured| / max(γ_measured, ε)
  F_eff = 1 − ω_eff

Regime:
  Noble:    ω_eff < 0.10 (inert surface, good agreement)
  Reactive: 0.10 ≤ ω_eff < 0.30
  Active:   ω_eff ≥ 0.30 (strongly catalytic, model challenged)

Cross-references:
  Cohesive:  closures/materials_science/cohesive_energy.py (E_coh, bondtype)
  Band:      closures/materials_science/band_structure.py (d-band character)
  Elastic:   closures/materials_science/elastic_moduli.py (surface stress)
  RCFT:      closures/rcft/attractor_basin.py (attractor depth → catalysis)
  Contract:  contracts/MATL.INTSTACK.v1.yaml
  Canon:     canon/matl_anchors.yaml (A-MT-SURF-*)
  Sources:   Tyson & Miller (1977), Hammer & Nørskov (2000), Kittel (2005)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class SurfaceRegime(StrEnum):
    """GCD regime for surface/catalysis closure."""

    NOBLE = "Noble"  # ω < 0.10
    REACTIVE = "Reactive"  # 0.10 ≤ ω < 0.30
    ACTIVE = "Active"  # 0.30 ≤ ω < 0.60
    ANOMALOUS = "Anomalous"  # ω ≥ 0.60


class CatalyticClass(StrEnum):
    """Catalytic activity classification."""

    INERT = "Inert"  # Noble metals, weak adsorption
    MODERATE = "Moderate"  # Balanced adsorption/desorption
    STRONG = "Strong"  # Strong binding, poisoning risk
    VOLCANO_PEAK = "VolcanoPeak"  # Near Sabatier optimum


class SurfaceCatalysisResult(NamedTuple):
    """Result of surface/catalysis computation."""

    gamma_J_m2: float  # Surface energy (J/m²)
    gamma_measured_J_m2: float  # Reference surface energy
    W_adhesion_J_m2: float  # Work of adhesion (self-adhesion)
    E_vacancy_eV: float  # Vacancy formation energy (eV)
    broken_bond_fraction: float  # (1 − z_s/z_b)
    A_atom_A2: float  # Atomic area on surface (ų)
    d_band_center_eV: float  # d-band center estimate (eV)
    E_ads_eV: float  # Adsorption energy (eV)
    BEP_barrier_eV: float  # Brønsted-Evans-Polanyi barrier (eV)
    catalytic_class: str  # Inert / Moderate / Strong / VolcanoPeak
    sabatier_distance: float  # Distance from Sabatier optimum
    rcft_return_deficit: float  # How incomplete the surface return is
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
EV_PER_J_M2 = 16.0218  # 1 eV/ų = 16.0218 J/m²
RELAX_FACTOR = 0.90  # f_relax typical surface relaxation

# Regime thresholds
THRESH_NOBLE = 0.10
THRESH_REACTIVE = 0.30
THRESH_ACTIVE = 0.60

# ── Reference Data ───────────────────────────────────────────────
# γ (J/m²), z_b (bulk coord), d_band_center (eV relative to Fermi),
# catalytic_class
REFERENCE_SURFACE: dict[str, dict[str, float | str]] = {
    # Transition metals (close-packed surfaces)
    "Fe": {"gamma": 2.42, "z_b": 8, "z_s": 4, "d_center": -1.2, "E_coh": 4.28, "r0": 2.48, "cat": "Strong"},
    "Co": {"gamma": 2.55, "z_b": 12, "z_s": 9, "d_center": -1.3, "E_coh": 4.39, "r0": 2.50, "cat": "Strong"},
    "Ni": {"gamma": 2.38, "z_b": 12, "z_s": 9, "d_center": -1.5, "E_coh": 4.44, "r0": 2.49, "cat": "Moderate"},
    "Cu": {"gamma": 1.79, "z_b": 12, "z_s": 9, "d_center": -2.7, "E_coh": 3.49, "r0": 2.56, "cat": "Inert"},
    "Ru": {"gamma": 3.04, "z_b": 12, "z_s": 9, "d_center": -1.4, "E_coh": 6.74, "r0": 2.65, "cat": "Strong"},
    "Rh": {"gamma": 2.66, "z_b": 12, "z_s": 9, "d_center": -1.7, "E_coh": 5.75, "r0": 2.69, "cat": "Moderate"},
    "Pd": {"gamma": 2.00, "z_b": 12, "z_s": 9, "d_center": -1.8, "E_coh": 3.89, "r0": 2.75, "cat": "Moderate"},
    "Ag": {"gamma": 1.25, "z_b": 12, "z_s": 9, "d_center": -4.0, "E_coh": 2.95, "r0": 2.89, "cat": "Inert"},
    "Ir": {"gamma": 3.00, "z_b": 12, "z_s": 9, "d_center": -2.1, "E_coh": 6.94, "r0": 2.71, "cat": "Moderate"},
    "Pt": {"gamma": 2.49, "z_b": 12, "z_s": 9, "d_center": -2.3, "E_coh": 5.84, "r0": 2.77, "cat": "VolcanoPeak"},
    "Au": {"gamma": 1.51, "z_b": 12, "z_s": 9, "d_center": -3.6, "E_coh": 3.81, "r0": 2.88, "cat": "Inert"},
    "W": {"gamma": 3.27, "z_b": 8, "z_s": 4, "d_center": -0.8, "E_coh": 8.90, "r0": 2.74, "cat": "Strong"},
    "Mo": {"gamma": 2.95, "z_b": 8, "z_s": 4, "d_center": -1.0, "E_coh": 6.82, "r0": 2.73, "cat": "Strong"},
    "Ti": {"gamma": 2.10, "z_b": 12, "z_s": 9, "d_center": -0.5, "E_coh": 4.85, "r0": 2.89, "cat": "Strong"},
    # s/p metals
    "Al": {"gamma": 1.14, "z_b": 12, "z_s": 9, "d_center": 0.0, "E_coh": 3.39, "r0": 2.86, "cat": "Inert"},
    "Zn": {"gamma": 0.99, "z_b": 12, "z_s": 9, "d_center": -7.0, "E_coh": 1.35, "r0": 2.66, "cat": "Inert"},
}

# d-band scaling parameters for adsorption (CO on transition metals)
# E_ads ≈ ALPHA_D * ε_d + BETA_D  (linear scaling relation)
ALPHA_D_ADS = -0.50  # eV per eV of d-band center
BETA_D_ADS = -1.20  # eV intercept

# BEP parameters: E_a = GAMMA_BEP * E_ads + DELTA_BEP
GAMMA_BEP = 0.85
DELTA_BEP = 1.50  # eV

# Sabatier optimum (CO oxidation): E_ads ≈ −1.5 eV
SABATIER_OPTIMUM_EV = -1.5


def _classify_regime(omega: float) -> SurfaceRegime:
    if omega < THRESH_NOBLE:
        return SurfaceRegime.NOBLE
    if omega < THRESH_REACTIVE:
        return SurfaceRegime.REACTIVE
    if omega < THRESH_ACTIVE:
        return SurfaceRegime.ACTIVE
    return SurfaceRegime.ANOMALOUS


def _classify_catalytic(E_ads: float, d_center: float) -> CatalyticClass:
    """Classify catalytic activity based on adsorption energy."""
    sabatier_dist = abs(E_ads - SABATIER_OPTIMUM_EV)
    if sabatier_dist < 0.3:
        return CatalyticClass.VOLCANO_PEAK
    if E_ads > -0.5:  # Weak binding
        return CatalyticClass.INERT
    if E_ads < -2.5:  # Too strong binding
        return CatalyticClass.STRONG
    return CatalyticClass.MODERATE


def compute_surface_catalysis(
    E_coh_eV: float,
    *,
    Z: int = 0,
    symbol: str = "",
    r0_A: float = 0.0,
    z_bulk: int = 12,
    z_surface: int = 9,
    d_band_center_eV: float | None = None,
    gamma_measured_J_m2: float | None = None,
) -> SurfaceCatalysisResult:
    """Compute surface energy and catalytic properties.

    Parameters
    ----------
    E_coh_eV : float
        Cohesive energy per atom (eV).
    Z : int
        Atomic number (for reference lookup).
    symbol : str
        Element symbol for reference lookup.
    r0_A : float
        Nearest-neighbor distance (Å).
    z_bulk : int
        Bulk coordination number (FCC=12, BCC=8).
    z_surface : int
        Surface coordination number (FCC(111)=9, BCC(110)=4).
    d_band_center_eV : float | None
        d-band center relative to Fermi level (eV).
    gamma_measured_J_m2 : float | None
        Measured surface energy for ω_eff calculation.

    Returns
    -------
    SurfaceCatalysisResult
    """
    # ── Reference lookup ──
    ref = REFERENCE_SURFACE.get(symbol, {})
    if gamma_measured_J_m2 is None:
        gamma_measured_J_m2 = float(ref.get("gamma", 0.0))
    if E_coh_eV <= 0 and ref:
        E_coh_eV = float(ref.get("E_coh", 0.0))
    if r0_A <= 0 and ref:
        r0_A = float(ref.get("r0", 2.5))
    if d_band_center_eV is None:
        d_band_center_eV = float(ref.get("d_center", 0.0))
    if ref:
        z_bulk = int(ref.get("z_b", z_bulk))
        z_surface = int(ref.get("z_s", z_surface))

    # ── Broken bond fraction ──
    broken_fraction = 1.0 - z_surface / z_bulk if z_bulk > 0 else 0.0

    # ── Atomic area on surface ──
    # For close-packed: A ≈ (√3/2) · r₀²
    A_atom_A2 = math.sqrt(3) / 2.0 * r0_A**2 if r0_A > 0 else 10.0

    # ── Surface energy (broken-bond model) ──
    # γ = E_coh · broken_fraction · f_relax / A_atom
    # Convert from eV/ų to J/m²
    if A_atom_A2 > 0 and E_coh_eV > 0:
        gamma_eV_A2 = E_coh_eV * broken_fraction * RELAX_FACTOR / A_atom_A2
        gamma_J_m2 = gamma_eV_A2 * EV_PER_J_M2
    else:
        gamma_J_m2 = 0.0

    # ── Work of adhesion (self-adhesion) ──
    W_ad = 2.0 * gamma_J_m2

    # ── Vacancy formation energy ──
    # E_vac ≈ E_coh × broken_fraction_vacancy
    # Vacancy has ~z_bulk − 1 neighbors removed from 1 site
    vac_fraction = 1.0 / z_bulk if z_bulk > 0 else 0.0
    E_vacancy = E_coh_eV * (1.0 - vac_fraction) * 0.5  # empirical factor

    # ── d-band center → adsorption ──
    E_ads = ALPHA_D_ADS * d_band_center_eV + BETA_D_ADS

    # ── BEP barrier ──
    BEP_barrier = max(0.0, GAMMA_BEP * E_ads + DELTA_BEP)

    # ── Catalytic classification ──
    cat_class = _classify_catalytic(E_ads, d_band_center_eV)
    sabatier_dist = abs(E_ads - SABATIER_OPTIMUM_EV)

    # ── RCFT return deficit ──
    # How incomplete is the surface return cycle?
    rcft_deficit = broken_fraction

    # ── GCD mapping ──
    eps = 1e-12
    if gamma_measured_J_m2 > eps:
        omega_eff = abs(gamma_J_m2 - gamma_measured_J_m2) / gamma_measured_J_m2
    elif gamma_J_m2 > eps:
        omega_eff = 0.0
    else:
        omega_eff = 0.0

    omega_eff = min(1.0, omega_eff)
    F_eff = 1.0 - omega_eff
    regime = _classify_regime(omega_eff)

    return SurfaceCatalysisResult(
        gamma_J_m2=round(gamma_J_m2, 6),
        gamma_measured_J_m2=round(gamma_measured_J_m2, 6),
        W_adhesion_J_m2=round(W_ad, 6),
        E_vacancy_eV=round(E_vacancy, 6),
        broken_bond_fraction=round(broken_fraction, 6),
        A_atom_A2=round(A_atom_A2, 6),
        d_band_center_eV=round(d_band_center_eV, 4),
        E_ads_eV=round(E_ads, 6),
        BEP_barrier_eV=round(BEP_barrier, 6),
        catalytic_class=cat_class.value,
        sabatier_distance=round(sabatier_dist, 6),
        rcft_return_deficit=round(rcft_deficit, 6),
        omega_eff=round(omega_eff, 8),
        F_eff=round(F_eff, 8),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Surface & Catalysis Self-Test ===\n")

    for sym in ["Fe", "Cu", "Pt", "Au", "W", "Ni", "Pd", "Rh", "Al"]:
        r = compute_surface_catalysis(0, symbol=sym)
        print(
            f"  {sym:4s}  γ={r.gamma_J_m2:6.3f} J/m²  (ref {r.gamma_measured_J_m2:.2f})  "
            f"ε_d={r.d_band_center_eV:5.2f}eV  E_ads={r.E_ads_eV:6.3f}eV  "
            f"{r.catalytic_class:12s}  ω={r.omega_eff:.4f}  {r.regime}"
        )
