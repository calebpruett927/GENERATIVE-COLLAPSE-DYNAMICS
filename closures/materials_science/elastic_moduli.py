"""Elastic Moduli Closure — MATL.INTSTACK.v1

Derives bulk, shear, and Young's moduli from interatomic potential curvature,
bridging atomic-scale bonding to macroscopic mechanical response through
RCFT scaling relations.

Physics:
  The elastic modulus is the second derivative of the interatomic potential
  at equilibrium, scaled to bulk:

    K (bulk modulus) = V₀ · (d²E/dV²) = (r₀/9V₀) · (d²U/dr²)|_{r₀}
    G (shear modulus) ≈ K · (3 − 6ν_P) / (3 + 6ν_P)   (isotropic approx)
    E (Young's modulus) = 9KG / (3K + G)

  The interatomic potential U(r) is derived from:
    - Pair potential curvature (Lennard-Jones, Morse, Born-Mayer)
    - Cohesive energy & equilibrium distance from cohesive_energy.py

RCFT connection:
  Near a structural phase transition, elastic moduli exhibit RCFT scaling:
    K ~ |T − T_c|^{d_eff·ν − 2}  (from hyperscaling + compressibility)
    G ~ |T − T_c|^{(d_eff − 2)·ν}  (shear modulus softening)

  For GCD universality (p = 3):
    K ~ |t|^0  (marginal — logarithmic correction)
    G ~ |t|^{4/3}  (shear softening near structural transitions)

  This explains why bulk modulus is robust near T_c while shear modulus
  collapses — the underlying RCFT structure preserves volumetric coherence
  while allowing shear deformation as the attractor basin reshapes.

UMCP integration:
  ω_eff = |K_predicted − K_measured| / K_measured  (elastic drift)
  F_eff = 1 − ω_eff

Regime:
  Stiff:       ω_eff < 0.10  (prediction within 10%)
  Moderate:    0.10 ≤ ω_eff < 0.25
  Compliant:   0.25 ≤ ω_eff < 0.50
  Anomalous:   ω_eff ≥ 0.50

Cross-references:
  Materials:  closures/materials_science/cohesive_energy.py (E_coh, r₀)
  Materials:  closures/materials_science/phase_transition.py (softening near T_c)
  RCFT:       closures/rcft/universality_class.py (d_eff, ν for scaling)
  Atomic:     closures/atomic_physics/ionization_energy.py (Z_eff for potential)
  Sources:    Born & Huang (1954), Grimvall (1999)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class ElasticRegime(StrEnum):
    STIFF = "Stiff"
    MODERATE = "Moderate"
    COMPLIANT = "Compliant"
    ANOMALOUS = "Anomalous"


class ElasticResult(NamedTuple):
    """Result of elastic moduli computation."""

    K_GPa: float  # Bulk modulus (GPa)
    G_GPa: float  # Shear modulus (GPa)
    E_GPa: float  # Young's modulus (GPa)
    nu_poisson: float  # Poisson's ratio
    K_measured_GPa: float  # Reference value (GPa)
    potential_curvature_eV_A2: float  # d²U/dr² at equilibrium
    debye_stiffness: float  # √(K/ρ) proxy for Debye velocity
    # RCFT softening predictions
    rcft_K_exponent: float  # Bulk modulus critical exponent
    rcft_G_exponent: float  # Shear modulus critical exponent
    omega_eff: float
    F_eff: float
    regime: str


# ── Conversion constants ─────────────────────────────────────────
EV_PER_A3_TO_GPA = 160.2176634  # 1 eV/Å³ = 160.22 GPa

# Reference bulk moduli (GPa) — selected materials
REFERENCE_K: dict[str, float] = {
    # Alkali metals
    "Li": 11.0,
    "Na": 6.3,
    "K": 3.1,
    "Rb": 2.5,
    "Cs": 1.6,
    # Alkaline earth
    "Be": 130.0,
    "Mg": 45.0,
    "Ca": 17.0,
    "Sr": 12.0,
    "Ba": 9.6,
    # Transition metals
    "Ti": 110.0,
    "V": 162.0,
    "Cr": 160.0,
    "Fe": 170.0,
    "Co": 180.0,
    "Ni": 180.0,
    "Cu": 140.0,
    "Zn": 70.0,
    "Nb": 170.0,
    "Mo": 230.0,
    "Ru": 220.0,
    "Rh": 270.0,
    "Pd": 180.0,
    "Ag": 100.0,
    "Ta": 200.0,
    "W": 310.0,
    "Re": 370.0,
    "Os": 462.0,
    "Ir": 320.0,
    "Pt": 230.0,
    "Au": 220.0,
    # Post-transition / covalent
    "Al": 76.0,
    "Si": 98.0,
    "Ge": 75.0,
    "C_diamond": 442.0,
    # Compounds (selected)
    "NaCl": 24.0,
    "MgO": 160.0,
    "Al2O3": 252.0,
    "SiC": 225.0,
    "TiC": 242.0,
    "WC": 439.0,
}

# Typical equilibrium atomic volumes (Å³/atom)
ATOMIC_VOLUMES: dict[str, float] = {
    "Li": 21.7,
    "Na": 37.7,
    "K": 73.6,
    "Fe": 11.8,
    "Co": 11.1,
    "Ni": 10.9,
    "Cu": 11.8,
    "Al": 16.6,
    "Si": 20.0,
    "W": 15.8,
    "Au": 17.0,
    "Ag": 17.1,
    "Pt": 15.1,
    "Mo": 15.6,
    "Os": 14.0,
    "C_diamond": 5.67,
}

# Regime thresholds
THRESH_STIFF = 0.10
THRESH_MODERATE = 0.25
THRESH_COMPLIANT = 0.50


def _classify(omega: float) -> ElasticRegime:
    if omega < THRESH_STIFF:
        return ElasticRegime.STIFF
    if omega < THRESH_MODERATE:
        return ElasticRegime.MODERATE
    if omega < THRESH_COMPLIANT:
        return ElasticRegime.COMPLIANT
    return ElasticRegime.ANOMALOUS


def _pair_potential_curvature(
    E_coh_eV: float,
    r0_A: float,
    n_neighbors: int = 12,
) -> float:
    """Estimate d²U/dr² at equilibrium from cohesive energy.

    For a Lennard-Jones-like potential: U(r) = ε[(r₀/r)^12 − 2(r₀/r)^6]
      U''(r₀) = 72ε / r₀²

    For real materials, the curvature scales with E_coh and r₀ but
    depends on potential form. We use the generalized scaling:
      U''(r₀) ≈ (2 · n_nn · E_coh / n_neighbors) · k / r₀²
    where k ≈ 36 for LJ (half of 72 because pair counting).
    """
    if r0_A <= 0 or E_coh_eV <= 0:
        return 0.0
    # Per-bond energy
    e_bond = E_coh_eV / (n_neighbors / 2.0)  # each bond shared by 2 atoms
    # Curvature ≈ 36 · e_bond / r₀² (from LJ scaling)
    return 36.0 * e_bond / r0_A**2


def compute_elastic_moduli(
    E_coh_eV: float,
    r0_A: float,
    *,
    symbol: str = "",
    V0_A3: float = 0.0,
    n_neighbors: int = 12,
    nu_poisson: float = 0.30,
    K_measured_GPa: float | None = None,
    p: int = 3,
) -> ElasticResult:
    """Compute elastic moduli from interatomic potential with RCFT scaling.

    Parameters
    ----------
    E_coh_eV : float
        Cohesive energy per atom (eV), from cohesive_energy.py.
    r0_A : float
        Equilibrium interatomic distance (Å).
    symbol : str
        Element symbol for reference lookup.
    V0_A3 : float
        Atomic volume (Å³). If 0, estimated from r₀.
    n_neighbors : int
        Coordination number (FCC/HCP = 12, BCC = 8, diamond = 4).
    nu_poisson : float
        Poisson's ratio (default 0.30 for metals).
    K_measured_GPa : float | None
        Measured bulk modulus for validation.
    p : int
        GCD drift exponent for RCFT scaling predictions.

    Returns
    -------
    ElasticResult
    """
    if E_coh_eV < 0 or r0_A <= 0:
        msg = f"E_coh must be ≥ 0 and r₀ > 0, got E_coh={E_coh_eV}, r₀={r0_A}"
        raise ValueError(msg)

    # Atomic volume estimate
    if V0_A3 <= 0:
        V0_A3 = ATOMIC_VOLUMES[symbol] if symbol in ATOMIC_VOLUMES else (4.0 / 3.0) * math.pi * (r0_A / 2.0) ** 3

    # Potential curvature
    curv = _pair_potential_curvature(E_coh_eV, r0_A, n_neighbors)

    # Bulk modulus: K = (r₀ / 9V₀) · d²U/dr² · n_nn (Cauchy-Born)
    # Simplified: K ≈ n_nn · curv · r₀ / (9 · V₀)  in eV/Å³ → GPa
    K_ev_a3 = n_neighbors * curv * r0_A / (9.0 * V0_A3)
    K_gpa = K_ev_a3 * EV_PER_A3_TO_GPA

    # Shear modulus from Poisson's ratio
    # G = 3K(1 − 2ν) / (2(1 + ν))
    if nu_poisson >= 0.5:
        nu_poisson = 0.49  # incompressible limit guard
    G_gpa = 3.0 * K_gpa * (1.0 - 2.0 * nu_poisson) / (2.0 * (1.0 + nu_poisson))

    # Young's modulus: E = 9KG / (3K + G)
    denom = 3.0 * K_gpa + G_gpa
    E_gpa = 9.0 * K_gpa * G_gpa / denom if denom > 0 else 0.0

    # Debye stiffness proxy: √(K/ρ) ∝ √(K_eV_A3 / mass)
    # We report the volumetric stiffness directly
    debye_stiffness = math.sqrt(K_ev_a3) if K_ev_a3 > 0 else 0.0

    # RCFT critical exponents for elastic softening near phase transitions
    nu_rcft = 1.0 / p
    d_eff = 2.0 * p
    # Bulk modulus exponent: from compressibility χ_T ~ |t|^{−γ}
    # K ~ 1/χ_T ~ |t|^γ  (γ = (p-2)/p)
    rcft_K_exp = (p - 2.0) / p
    # Shear modulus exponent: G ~ |t|^{(d_eff - 2)ν}
    rcft_G_exp = (d_eff - 2.0) * nu_rcft

    # Reference lookup
    if K_measured_GPa is not None:
        k_ref = K_measured_GPa
    elif symbol in REFERENCE_K:
        k_ref = REFERENCE_K[symbol]
    else:
        k_ref = K_gpa

    # GCD mapping
    omega_eff = abs(K_gpa - k_ref) / k_ref if k_ref > 0 else 0.0
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(omega_eff)

    return ElasticResult(
        K_GPa=round(K_gpa, 2),
        G_GPa=round(G_gpa, 2),
        E_GPa=round(E_gpa, 2),
        nu_poisson=round(nu_poisson, 4),
        K_measured_GPa=round(k_ref, 2),
        potential_curvature_eV_A2=round(curv, 4),
        debye_stiffness=round(debye_stiffness, 6),
        rcft_K_exponent=round(rcft_K_exp, 4),
        rcft_G_exponent=round(rcft_G_exp, 4),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Fe", 4.28, 2.48, 8, 0.29),
        ("Cu", 3.49, 2.56, 12, 0.34),
        ("Al", 3.39, 2.86, 12, 0.35),
        ("W", 8.90, 2.74, 8, 0.28),
        ("Au", 3.81, 2.88, 12, 0.44),
        ("Si", 4.63, 2.35, 4, 0.22),
    ]
    print("Elastic Moduli from Atomic Potentials + RCFT Scaling")
    print("=" * 85)
    for sym, e_coh, r0, nn, nu_p in tests:
        r = compute_elastic_moduli(e_coh, r0, symbol=sym, n_neighbors=nn, nu_poisson=nu_p)
        print(
            f"{sym:3s}  K={r.K_GPa:7.1f}  G={r.G_GPa:7.1f}  E={r.E_GPa:7.1f} GPa  "
            f"K_ref={r.K_measured_GPa:7.1f}  ω={r.omega_eff:.4f}  "
            f"RCFT: K~|t|^{r.rcft_K_exponent:.2f}  G~|t|^{r.rcft_G_exponent:.2f}  {r.regime}"
        )
