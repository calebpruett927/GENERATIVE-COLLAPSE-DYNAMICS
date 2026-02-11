"""Cohesive Energy Closure — MATL.INTSTACK.v1

Derives bulk cohesive energy from atomic-level ionization energies, effective
nuclear charges, and inter-atomic potential models (Madelung, Born-Mayer,
Lennard-Jones), mapping results to GCD Tier-1 invariants.

Physics:
  The cohesive energy E_coh is the energy gained when isolated atoms condense
  into a solid.  It bridges the atomic scale (ionization, electron affinity)
  and the material scale (structural stability, melting point).

  Ionic crystals (Madelung-Born-Mayer):
    E_coh = -M · e² / (4πε₀ r₀) · (1 - ρ/r₀)     per ion pair
    M = Madelung constant (NaCl: 1.748, CsCl: 1.763, ZnS: 1.638)

  Metallic bonding (Wigner-Seitz / embedded atom):
    E_coh ≈ IE₁ · f_metallic(Z_eff, n_eff)
    f_metallic captures band-structure screening

  Covalent bonding:
    E_coh ≈ β · √(n_bonds) · S_overlap(Z₁, Z₂, r₀)

UMCP integration:
  ω_eff = |E_coh_predicted - E_coh_measured| / |E_coh_measured|  (cohesion drift)
  F_eff = 1 - ω_eff
  Ψ_coh = E_coh / E_coh_ref  clipped to [0,1]

Regime:
  StrongBond:    ω_eff < 0.05   (prediction within 5%)
  ModerateBond:  0.05 ≤ ω_eff < 0.15
  WeakBond:      0.15 ≤ ω_eff < 0.30
  Anomalous:     ω_eff ≥ 0.30

Cross-references:
  Atomic physics:  closures/atomic_physics/ionization_energy.py (IE₁, Z_eff)
  RCFT:            closures/rcft/universality_class.py (partition function, phase structure)
  Nuclear physics: closures/nuclear_physics/nuclide_binding.py (analogous binding deficit pattern)
  Sources:         Kittel (2005), Ashcroft & Mermin (1976)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class BondType(StrEnum):
    """Primary bonding mechanism."""

    IONIC = "Ionic"
    METALLIC = "Metallic"
    COVALENT = "Covalent"
    VAN_DER_WAALS = "VanDerWaals"
    MIXED = "Mixed"


class CohesionRegime(StrEnum):
    """Regime based on cohesive energy prediction accuracy."""

    STRONG_BOND = "StrongBond"
    MODERATE_BOND = "ModerateBond"
    WEAK_BOND = "WeakBond"
    ANOMALOUS = "Anomalous"


class CohesiveResult(NamedTuple):
    """Result of cohesive energy computation."""

    E_coh_eV: float  # Cohesive energy per atom (eV)
    E_coh_measured_eV: float  # Measured/reference value (eV)
    bond_type: str  # Primary bonding mechanism
    madelung_constant: float  # Effective Madelung constant (ionic)
    born_mayer_rho_A: float  # Born-Mayer range parameter (Å)
    r0_A: float  # Equilibrium interatomic distance (Å)
    Z_eff_avg: float  # Average effective nuclear charge
    omega_eff: float  # Cohesion drift
    F_eff: float  # Cohesion fidelity
    Psi_coh: float  # Normalized cohesion channel
    regime: str


# ── Frozen constants ─────────────────────────────────────────────
KE2_EV_A = 14.3996  # e²/(4πε₀) in eV·Å (Coulomb constant × e²)
E_COH_REF = 8.94  # Reference: highest cohesive energy (W, ~8.9 eV/atom)

# Reference cohesive energies (eV/atom) — Kittel / CRC Handbook
REFERENCE_COHESIVE: dict[str, float] = {
    # Alkali metals
    "Li": 1.63,
    "Na": 1.11,
    "K": 0.93,
    "Rb": 0.85,
    "Cs": 0.80,
    # Alkaline earth
    "Be": 3.32,
    "Mg": 1.51,
    "Ca": 1.84,
    "Sr": 1.72,
    "Ba": 1.90,
    # Transition metals (selected)
    "Ti": 4.85,
    "V": 5.31,
    "Cr": 4.10,
    "Mn": 2.92,
    "Fe": 4.28,
    "Co": 4.39,
    "Ni": 4.44,
    "Cu": 3.49,
    "Zn": 1.35,
    "Zr": 6.25,
    "Nb": 7.57,
    "Mo": 6.82,
    "Ru": 6.74,
    "Rh": 5.75,
    "Pd": 3.89,
    "Ag": 2.95,
    "Cd": 1.16,
    "Hf": 6.44,
    "Ta": 8.10,
    "W": 8.90,
    "Re": 8.03,
    "Os": 8.17,
    "Ir": 6.94,
    "Pt": 5.84,
    "Au": 3.81,
    # Post-transition / metalloids
    "Al": 3.39,
    "Ga": 2.81,
    "In": 2.52,
    "Sn": 3.14,
    "Pb": 2.03,
    # Nonmetals
    "C": 7.37,
    "Si": 4.63,
    "Ge": 3.85,
    # Noble gases
    "Ne": 0.020,
    "Ar": 0.080,
    "Kr": 0.116,
    "Xe": 0.170,
    # Ionic (given as per formula unit / 2 for per-atom approximation)
    "NaCl": 3.28,
    "KCl": 3.40,
    "MgO": 5.20,
    "CaO": 5.46,
}

# Madelung constants for common crystal structures
MADELUNG: dict[str, float] = {
    "NaCl": 1.74756,
    "CsCl": 1.76267,
    "ZnS_zinc_blende": 1.63806,
    "ZnS_wurtzite": 1.64132,
    "CaF2_fluorite": 2.51939,
    "TiO2_rutile": 2.408,
}

# Regime thresholds
THRESH_STRONG = 0.05
THRESH_MODERATE = 0.15
THRESH_WEAK = 0.30


def _classify_regime(omega_eff: float) -> CohesionRegime:
    if omega_eff < THRESH_STRONG:
        return CohesionRegime.STRONG_BOND
    if omega_eff < THRESH_MODERATE:
        return CohesionRegime.MODERATE_BOND
    if omega_eff < THRESH_WEAK:
        return CohesionRegime.WEAK_BOND
    return CohesionRegime.ANOMALOUS


def _identify_bond_type(
    IE1_eV: float,
    electronegativity_diff: float,
    n_valence: int,
) -> BondType:
    """Classify dominant bonding mechanism from atomic properties.

    The electronegativity difference (Pauling scale) determines character:
      |Δχ| > 1.7  → Ionic
      |Δχ| < 0.5  → Covalent or Metallic (IE determines which)
      else        → Mixed

    Low IE₁ with small Δχ → Metallic (delocalized electrons).
    """
    if electronegativity_diff > 1.7:
        return BondType.IONIC
    if electronegativity_diff < 0.5:
        if IE1_eV < 8.0:  # Low ionization → metallic
            return BondType.METALLIC
        if n_valence >= 3:
            return BondType.COVALENT
        return BondType.METALLIC
    return BondType.MIXED


def _ionic_cohesive(
    madelung: float,
    r0_angstrom: float,
    rho_angstrom: float = 0.30,
) -> float:
    """Madelung-Born-Mayer cohesive energy for ionic crystal (eV per ion pair).

    E = -M · (e²/4πε₀r₀) · (1 - ρ/r₀)
    """
    if r0_angstrom <= 0:
        return 0.0
    return madelung * KE2_EV_A / r0_angstrom * (1.0 - rho_angstrom / r0_angstrom)


def _metallic_cohesive(
    IE1_eV: float,
    Z_eff: float,
    n_eff: float,
    n_valence: int,
) -> float:
    """Estimate metallic cohesive energy using Friedel d-band + EAM correction.

    Model: E_coh = E_friedel + E_embed (many-body EAM correction)

    For transition metals (n_d electrons in d-band):
      Friedel:   E_d ∝ W · n_d(10 − n_d) / 20
      EAM embed: E_embed ∝ −√(n_d · z) · β_eam
        where z is coordination and β_eam captures many-body embedding.
        The square-root dependence is the hallmark of EAM: the embedding
        energy is a concave function of host electron density, giving
        the many-body character that pair potentials miss.

      Magnetic exchange correction (for Fe, Co, Ni):
        E_mag ≈ k_B · T_c · n_unp / 2 (approximate Heisenberg exchange)

    For s/p metals:
      E_coh ∝ IE₁ · √(n_valence) / n_eff² (jellium-like)

    The key physics: metallic bonding has THREE contributions:
      1. Band energy (Friedel) — delocalization gain
      2. Embedding energy (EAM) — many-body electron density
      3. Magnetic exchange (ferromagnets) — spin ordering stabilization

    OPT-EAM: The EAM embedding function F(ρ̄) ≈ −A·√ρ̄ is the
    simplest physically correct many-body term.  It ensures that
    adding one atom to a surface gives less energy than adding
    to bulk — the concavity drives surface relaxation naturally.

    Cross-references:
      Daw & Baskes (1984) — Original EAM
      Foiles, Baskes & Daw (1986) — EAM potentials for FCC metals
      Friedel (1969) — d-band filling model
    """
    if n_eff <= 0 or IE1_eV <= 0:
        return 0.0

    if n_valence > 2 and n_eff > 3.0:
        # Transition metal: Friedel + EAM + magnetic exchange
        n_d = min(n_valence - 2, 10)  # d-electrons (subtract s²)

        # 1. Friedel d-band energy
        friedel_factor = n_d * (10 - n_d) / 20.0
        bandwidth_proxy = IE1_eV * 0.50  # d-bandwidth ≈ 50% of IE₁
        E_friedel = bandwidth_proxy * friedel_factor

        # 2. EAM many-body embedding correction
        # E_embed ≈ −β_eam · √(n_d · z_eff / z_ref)
        # β_eam calibrated: W (n_d≈4, E_coh=8.90), Cu (n_d≈9, E_coh=3.49)
        z_nn = 8 if n_d <= 5 else 12  # BCC vs FCC typical
        beta_eam = 0.50  # eV, many-body embedding scale
        E_embed = beta_eam * math.sqrt(n_d * z_nn / 12.0)

        # 3. Magnetic exchange correction (ferromagnets)
        # Hund's rule: n_unpaired = n_d if n_d ≤ 5, else 10 − n_d
        n_unpaired = n_d if n_d <= 5 else 10 - n_d
        # Exchange energy ~ k_B * Tc * n_unpaired / 2
        # Empirical scaling: ~0.12 eV per unpaired electron for d-block
        E_mag = 0.12 * n_unpaired if 1 <= n_d <= 9 else 0.0

        return E_friedel + E_embed + E_mag

    else:
        # s/p metal: jellium + structure factor
        correction = 0.18
        return IE1_eV * math.sqrt(n_valence) / n_eff**2 * Z_eff**0.5 * correction * n_eff


def _covalent_cohesive(
    IE1_eV: float,
    n_valence: int,
    Z_eff: float,
) -> float:
    """Estimate covalent cohesive energy from orbital overlap.

    Model: E_coh ≈ β · √(n_bonds) where β ∝ IE₁ · Z_eff / n²
    """
    n_bonds = min(n_valence, 8 - n_valence)  # maximum covalent bonds
    if n_bonds <= 0 or IE1_eV <= 0:
        return 0.0
    beta = IE1_eV * 0.15 * Z_eff**0.25
    return beta * math.sqrt(n_bonds)


def compute_cohesive_energy(
    Z: int,
    *,
    symbol: str = "",
    IE1_eV: float = 0.0,
    Z_eff: float = 0.0,
    n_eff: float = 0.0,
    n_valence: int = 0,
    electronegativity_diff: float = 0.0,
    r0_angstrom: float = 0.0,
    madelung_constant: float = 1.748,
    born_mayer_rho: float = 0.30,
    E_coh_measured_eV: float | None = None,
) -> CohesiveResult:
    """Compute cohesive energy from atomic properties with GCD mapping.

    Parameters
    ----------
    Z : int
        Atomic number of primary element.
    symbol : str
        Element symbol (for reference lookup).
    IE1_eV : float
        First ionization energy (eV). If 0, looked up from NIST table.
    Z_eff : float
        Effective nuclear charge (Slater). If 0, estimated.
    n_eff : float
        Effective quantum number. If 0, estimated from Z.
    n_valence : int
        Number of valence electrons. If 0, estimated.
    electronegativity_diff : float
        Pauling electronegativity difference (for compounds).
    r0_angstrom : float
        Equilibrium interatomic distance (Å). 0 = estimate.
    madelung_constant : float
        Madelung constant (default: NaCl).
    born_mayer_rho : float
        Born-Mayer repulsion range (Å).
    E_coh_measured_eV : float | None
        Measured cohesive energy for validation.

    Returns
    -------
    CohesiveResult
    """
    # Import atomic physics closure for IE if not provided
    if IE1_eV <= 0:
        try:
            from closures.atomic_physics.ionization_energy import NIST_IE1

            IE1_eV = NIST_IE1.get(Z, 7.0)  # fallback to median
        except ImportError:
            IE1_eV = 7.0

    # Estimate effective charge and quantum number if not provided
    if Z_eff <= 0:
        # Simple Slater estimate
        if Z <= 2:
            Z_eff = Z - 0.30 * max(0, Z - 1)
        elif Z <= 10:
            Z_eff = Z - 2 * 0.85 - max(0, Z - 3) * 0.35
        else:
            Z_eff = max(1.0, Z * 0.35)

    if n_eff <= 0:
        if Z <= 2:
            n_eff = 1.0
        elif Z <= 10:
            n_eff = 2.0
        elif Z <= 18:
            n_eff = 3.0
        elif Z <= 36:
            n_eff = 3.7  # effective for 3d/4s
        else:
            n_eff = 4.0

    if n_valence <= 0:
        # Rough estimate from group position
        if Z <= 2:
            n_valence = Z
        elif Z <= 10:
            n_valence = Z - 2
        elif Z <= 12 or Z <= 18:
            n_valence = Z - 10
        elif Z <= 20:
            n_valence = Z - 18
        elif Z <= 30:
            n_valence = min(Z - 18, 12)  # d-block
        else:
            n_valence = min(Z - 28, 8)

    # Identify bonding character
    bond_type = _identify_bond_type(IE1_eV, electronegativity_diff, n_valence)

    # Estimate equilibrium distance if not given
    if r0_angstrom <= 0:
        # Approximate: r₀ ≈ 1.5 · n_eff / Z_eff^(1/3)  (very rough)
        r0_angstrom = 1.5 * n_eff / max(Z_eff ** (1.0 / 3.0), 0.5)

    # Compute cohesive energy based on bond type
    if bond_type == BondType.IONIC:
        e_coh = _ionic_cohesive(madelung_constant, r0_angstrom, born_mayer_rho) / 2.0
    elif bond_type == BondType.METALLIC:
        e_coh = _metallic_cohesive(IE1_eV, Z_eff, n_eff, n_valence)
    elif bond_type == BondType.COVALENT:
        e_coh = _covalent_cohesive(IE1_eV, n_valence, Z_eff)
    else:
        # Mixed: weighted average
        e_ionic = _ionic_cohesive(madelung_constant, r0_angstrom, born_mayer_rho) / 2.0
        e_cov = _covalent_cohesive(IE1_eV, n_valence, Z_eff)
        ionicity = min(1.0, electronegativity_diff / 2.0)
        e_coh = ionicity * e_ionic + (1.0 - ionicity) * e_cov

    # Measured reference
    if E_coh_measured_eV is not None:
        e_measured = E_coh_measured_eV
    elif symbol in REFERENCE_COHESIVE:
        e_measured = REFERENCE_COHESIVE[symbol]
    else:
        e_measured = e_coh  # no reference → prediction is "truth"

    # GCD mapping
    omega_eff = abs(e_coh - e_measured) / e_measured if e_measured > 0 else 0.0
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff
    psi_coh = max(0.0, min(1.0, e_coh / E_COH_REF))

    regime = _classify_regime(omega_eff)

    return CohesiveResult(
        E_coh_eV=round(e_coh, 4),
        E_coh_measured_eV=round(e_measured, 4),
        bond_type=bond_type.value,
        madelung_constant=round(madelung_constant, 5),
        born_mayer_rho_A=round(born_mayer_rho, 4),
        r0_A=round(r0_angstrom, 4),
        Z_eff_avg=round(Z_eff, 4),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        Psi_coh=round(psi_coh, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (11, "Na", "Alkali metal"),
        (13, "Al", "s/p metal"),
        (14, "Si", "Semiconductor"),
        (26, "Fe", "Transition metal"),
        (29, "Cu", "Noble metal"),
        (74, "W", "Refractory"),
        (6, "C", "Covalent (diamond)"),
    ]
    for z, sym, desc in tests:
        r = compute_cohesive_energy(z, symbol=sym)
        print(
            f"{desc:20s} ({sym:2s})  E_coh={r.E_coh_eV:6.3f} eV  "
            f"ref={r.E_coh_measured_eV:6.3f}  ω={r.omega_eff:.4f}  "
            f"F={r.F_eff:.4f}  {r.bond_type:10s}  {r.regime}"
        )
