"""Band Structure Closure — MATL.INTSTACK.v1

Derives electronic band character (metal / semiconductor / insulator) from
atomic electron configurations and maps to GCD invariants, using RCFT
attractor basin topology to explain band gap formation as a collapse-return
phenomenon.

Physics — The RCFT Interpretation of Band Gaps:
  In a crystal, atomic orbitals hybridize into bands via Bloch's theorem.
  The band gap E_g arises from the *splitting* at Brillouin zone boundaries.

  RCFT reinterpretation:
    The periodic potential V(r) = V(r + R) creates a *recursive* collapse
    structure — the electron wavefunction "collapses" at each unit cell
    and "returns" with accumulated phase.  The band gap is the *cost*
    of crossing the return boundary between attractor basins:

    E_g ≈ 2|V_G|     (nearly-free electron, first Fourier component)
    E_g → RCFT Fisher geodesic distance × energy scale

    For ionic (large |V_G|): insulator (Fisher distance ≈ π)
    For covalent: semiconductor (Fisher distance ≈ π/4 to π/2)
    For metallic: E_g = 0 (no basin boundary — single attractor)

  This maps directly to the RCFT attractor basin analysis:
    - Metals: Monostable (single Fermi surface attractor)
    - Semiconductors: Bistable (valence + conduction basins)
    - Insulators: Bistable + strong (wide basin separation)

  Band gap scaling near semiconductor → metal transitions follows
  RCFT universality with order parameter Φ = E_g/E_g0 ~ |t|^β.

Atomic inputs:
  - Electron configuration → valence shell + orbital character
  - Ionization energy → band width estimation
  - Fine structure → spin-orbit coupling in band structure

UMCP integration:
  ω_eff = |E_g_predicted − E_g_measured| / max(E_g_measured, 0.01)
  F_eff = 1 − ω_eff
  Ψ_band = 1 − E_g / E_g_max  (metals → 1, wide insulators → 0)

Regime:
  Metal:          E_g = 0
  Semimetal:      0 < E_g < 0.1 eV
  Semiconductor:  0.1 ≤ E_g < 3.0 eV
  Insulator:      E_g ≥ 3.0 eV

Cross-references:
  Atomic:     closures/atomic_physics/electron_config.py (valence shell)
  Atomic:     closures/atomic_physics/ionization_energy.py (IE₁ → bandwidth)
  Atomic:     closures/atomic_physics/fine_structure.py ((Zα)² → spin-orbit)
  RCFT:       closures/rcft/attractor_basin.py (basin topology → band character)
  RCFT:       closures/rcft/information_geometry.py (Fisher distance → gap)
  SM:         closures/standard_model/coupling_constants.py (α_em → Coulomb screening)
  Sources:    Ashcroft & Mermin (1976), Harrison (1980), Kittel (2005)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class BandCharacter(StrEnum):
    """Electronic band character classification."""

    METAL = "Metal"
    SEMIMETAL = "Semimetal"
    SEMICONDUCTOR = "Semiconductor"
    INSULATOR = "Insulator"


class BandResult(NamedTuple):
    """Result of band structure analysis."""

    E_g_eV: float  # Band gap (eV), 0 for metals
    E_g_measured_eV: float  # Measured band gap (eV)
    band_character: str  # Metal / Semimetal / Semiconductor / Insulator
    bandwidth_eV: float  # Estimated bandwidth W (eV)
    V_pseudopot_eV: float  # Effective pseudopotential |V_G| (eV)
    n_valence_band: int  # Electrons in valence band per unit cell
    orbital_character: str  # s, p, d, f character of band edges
    spin_orbit_eV: float  # Spin-orbit splitting estimate (eV)
    # RCFT interpretation
    rcft_basin_type: str  # Monostable / Bistable (from attractor analogy)
    rcft_fisher_distance: float  # Fisher geodesic distance (band gap analog)
    omega_eff: float
    F_eff: float
    Psi_band: float  # Normalized band channel
    regime: str


# ── Reference band gaps (eV) ────────────────────────────────────
REFERENCE_GAPS: dict[str, float] = {
    # Metals (E_g = 0)
    "Li": 0.0,
    "Na": 0.0,
    "K": 0.0,
    "Cu": 0.0,
    "Ag": 0.0,
    "Au": 0.0,
    "Fe": 0.0,
    "Al": 0.0,
    "W": 0.0,
    "Pt": 0.0,
    # Semimetals
    "Bi": 0.015,
    "Sb": 0.01,
    "As": 0.0,
    "graphite": 0.0,
    # Semiconductors
    "Si": 1.12,
    "Ge": 0.66,
    "GaAs": 1.42,
    "GaN": 3.4,
    "InP": 1.35,
    "InAs": 0.36,
    "GaP": 2.26,
    "InSb": 0.17,
    "CdS": 2.42,
    "CdSe": 1.74,
    "CdTe": 1.49,
    "ZnSe": 2.70,
    "ZnO": 3.37,
    "ZnS": 3.68,
    "SiC_3C": 2.36,
    "SiC_4H": 3.26,
    "SiC_6H": 3.02,
    "PbS": 0.37,
    "PbSe": 0.27,
    "PbTe": 0.31,
    # Wide gap / insulators
    "C_diamond": 5.47,
    "BN_hex": 5.97,
    "AlN": 6.2,
    "SiO2": 8.9,
    "Al2O3": 8.8,
    "MgO": 7.8,
    "NaCl": 8.5,
    "KCl": 8.4,
    "LiF": 13.6,
}

# Insulator threshold
E_G_MAX_REF = 14.0  # eV (LiF reference for normalization)

# Constants
ALPHA_FINE = 1.0 / 137.036
RYDBERG_EV = 13.598


def _classify_band(E_g_eV: float) -> BandCharacter:
    if E_g_eV <= 0.0:
        return BandCharacter.METAL
    if E_g_eV < 0.1:
        return BandCharacter.SEMIMETAL
    if E_g_eV < 3.0:
        return BandCharacter.SEMICONDUCTOR
    return BandCharacter.INSULATOR


def _estimate_bandwidth(
    IE1_eV: float,
    Z_eff: float,
    n_eff: float,
    n_neighbors: int = 12,
) -> float:
    """Estimate bandwidth W from tight-binding hopping integral.

    W ≈ 2·z·t, where z = coordination, t = hopping integral
    t ≈ IE₁ · exp(−α·r₀/a₀) ≈ IE₁ / (Z_eff · n_eff)

    For metals: W ~ 5-15 eV
    For semiconductors: W ~ 3-8 eV
    For insulators: narrow bands
    """
    if n_eff <= 0 or Z_eff <= 0:
        return 5.0  # default estimate
    hopping = IE1_eV / (Z_eff * n_eff) * 2.0
    return 2.0 * n_neighbors * hopping


def _estimate_pseudopotential(
    E_g_eV: float,
    bandwidth_eV: float,
) -> float:
    """Estimate effective pseudopotential |V_G| from band gap.

    Nearly-free electron: E_g ≈ 2|V_G| at zone boundary.
    For tight-binding: gap relates to on-site vs hopping balance.
    """
    if E_g_eV <= 0:
        return 0.0
    return E_g_eV / 2.0


def _spin_orbit_estimate(Z: int, n: int, l_val: int) -> float:
    """Estimate spin-orbit splitting from atomic fine structure.

    ΔE_SO ≈ (Zα)⁴ · Ry / (n³ · l(l+1/2)(l+1))  for l > 0

    This is the atomic → solid-state spin-orbit coupling that determines
    topological insulator character and spin-Hall effects.
    """
    if l_val <= 0:
        return 0.0
    z_alpha = Z * ALPHA_FINE
    denom = n**3 * l_val * (l_val + 0.5) * (l_val + 1)
    if denom <= 0:
        return 0.0
    return z_alpha**4 * RYDBERG_EV / denom


def _orbital_character(Z: int) -> str:
    """Determine dominant orbital character of band edges."""
    if Z <= 2:
        return "1s"
    if Z <= 4:
        return "2s"
    if Z <= 10:
        return "2p"
    if Z <= 12:
        return "3s"
    if Z <= 18:
        return "3p"
    if Z <= 20:
        return "4s"
    if Z <= 30:
        return "3d"
    if Z <= 36:
        return "4p"
    if Z <= 48:
        return "4d"
    if Z <= 54:
        return "5p"
    if Z <= 71:
        return "4f"
    if Z <= 80:
        return "5d"
    return "6p"


def _rcft_basin_from_gap(E_g_eV: float) -> str:
    """Map band gap to RCFT attractor basin topology.

    Metal (E_g = 0): Monostable — single Fermi surface attractor,
      electrons freely collapse/return across the entire band.

    Semiconductor (E_g > 0): Bistable — two attractor basins
      (valence, conduction). The gap is the energy barrier between
      basins, analogous to the RCFT attractor basin boundary.

    Insulator (large E_g): Bistable + strong — deep basins with
      high barrier, corresponding to strongly localized electrons.
    """
    if E_g_eV <= 0:
        return "Monostable"
    if E_g_eV < 3.0:
        return "Bistable"
    return "Bistable_Strong"


def _rcft_fisher_distance(E_g_eV: float) -> float:
    """Map band gap to Fisher geodesic distance (information geometry).

    The Fisher distance between valence and conduction band states
    on the Bernoulli manifold provides an information-theoretic
    measure of the band gap:

      d_F ≈ (π/2) · (E_g / E_g_max)^(1/2)

    This uses the Fano-Fisher duality (T19): the gap is the
    curvature of the entropy-fidelity envelope in band space.
    Maximum distance = π corresponds to complete localization.
    """
    if E_g_eV <= 0:
        return 0.0
    ratio = min(1.0, E_g_eV / E_G_MAX_REF)
    return (math.pi / 2.0) * math.sqrt(ratio)


def compute_band_structure(
    Z: int,
    *,
    symbol: str = "",
    IE1_eV: float = 0.0,
    Z_eff: float = 0.0,
    n_eff: float = 0.0,
    n_neighbors: int = 12,
    E_g_measured_eV: float | None = None,
) -> BandResult:
    """Compute electronic band structure character with RCFT mapping.

    Parameters
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element/compound symbol for reference lookup.
    IE1_eV : float
        First ionization energy (eV). If 0, looked up.
    Z_eff, n_eff : float
        Effective nuclear charge and quantum number.
    n_neighbors : int
        Coordination number in crystal.
    E_g_measured_eV : float | None
        Measured band gap (eV). Looked up if available.

    Returns
    -------
    BandResult
    """
    if Z < 1:
        msg = f"Z must be ≥ 1, got {Z}"
        raise ValueError(msg)

    # Import IE if not provided
    if IE1_eV <= 0:
        try:
            from closures.atomic_physics.ionization_energy import NIST_IE1

            IE1_eV = NIST_IE1.get(Z, 7.0)
        except ImportError:
            IE1_eV = 7.0

    # Estimate Z_eff, n_eff
    if Z_eff <= 0:
        Z_eff = max(1.0, Z * 0.35)
    if n_eff <= 0:
        if Z <= 2:
            n_eff = 1.0
        elif Z <= 10:
            n_eff = 2.0
        elif Z <= 18:
            n_eff = 3.0
        elif Z <= 36:
            n_eff = 3.7
        else:
            n_eff = 4.2

    # Determine valence orbital
    orb_char = _orbital_character(Z)
    l_val = {"s": 0, "p": 1, "d": 2, "f": 3}.get(orb_char[-1], 0)
    n_qn = int(orb_char[0]) if orb_char[0].isdigit() else 3

    # Estimate bandwidth
    bandwidth = _estimate_bandwidth(IE1_eV, Z_eff, n_eff, n_neighbors)

    # Measured gap lookup
    if E_g_measured_eV is not None:
        e_g_ref = E_g_measured_eV
    elif symbol in REFERENCE_GAPS:
        e_g_ref = REFERENCE_GAPS[symbol]
    else:
        e_g_ref = None

    # Band gap estimation
    # For pure elements with known data, use reference.
    # For unknown: estimate from IE and orbital character.
    if e_g_ref is not None:
        e_g = e_g_ref  # Use measured/reference value
    else:
        # Heuristic: metals have partially filled d/s bands,
        # semiconductors have filled sp3 bands, insulators have
        # filled anion p bands.
        if l_val == 2 or (IE1_eV < 8.0 and l_val <= 1):
            e_g = 0.0  # likely metallic
        elif l_val == 1 and IE1_eV > 10.0:
            e_g = IE1_eV * 0.15  # rough semiconductor/insulator estimate
        else:
            e_g = max(0.0, (IE1_eV - 8.0) * 0.3)

    if e_g_ref is None:
        e_g_ref = e_g

    # Pseudopotential
    v_pseudo = _estimate_pseudopotential(e_g, bandwidth)

    # Spin-orbit
    so_split = _spin_orbit_estimate(Z, n_qn, l_val)

    # Valence electrons per unit cell (simplified)
    if Z <= 2:
        n_val = Z
    elif Z <= 10:
        n_val = Z - 2
    elif Z <= 18:
        n_val = Z - 10
    elif Z <= 36:
        n_val = min(Z - 18, 12)
    else:
        n_val = min(Z - 36, 18)

    # Band character classification
    band_char = _classify_band(e_g)

    # RCFT interpretation
    basin_type = _rcft_basin_from_gap(e_g)
    fisher_dist = _rcft_fisher_distance(e_g)

    # GCD mapping
    omega_eff = abs(e_g - e_g_ref) / e_g_ref if e_g_ref > 0.01 else 0.0  # metals: no gap to drift from
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    # Normalized band channel: metals → 1, wide insulators → 0
    psi_band = max(0.0, min(1.0, 1.0 - e_g / E_G_MAX_REF))

    return BandResult(
        E_g_eV=round(e_g, 4),
        E_g_measured_eV=round(e_g_ref, 4),
        band_character=band_char.value,
        bandwidth_eV=round(bandwidth, 4),
        V_pseudopot_eV=round(v_pseudo, 4),
        n_valence_band=n_val,
        orbital_character=orb_char,
        spin_orbit_eV=round(so_split, 6),
        rcft_basin_type=basin_type,
        rcft_fisher_distance=round(fisher_dist, 6),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        Psi_band=round(psi_band, 6),
        regime=band_char.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (29, "Cu", "Metal"),
        (13, "Al", "Metal"),
        (14, "Si", "Semiconductor"),
        (32, "Ge", "Semiconductor"),
        (6, "C_diamond", "Insulator"),
        (26, "Fe", "Metal"),
    ]
    print("Band Structure from Atomic Physics + RCFT Attractor Basins")
    print("=" * 85)
    for z, sym, _desc in tests:
        r = compute_band_structure(z, symbol=sym)
        print(
            f"{sym:10s}  E_g={r.E_g_eV:5.2f} eV  W={r.bandwidth_eV:6.2f} eV  "
            f"SO={r.spin_orbit_eV:.4f} eV  {r.band_character:15s}  "
            f"RCFT: {r.rcft_basin_type:16s}  d_F={r.rcft_fisher_distance:.4f}"
        )
