"""Magnetic Properties Closure — MATL.INTSTACK.v1

Derives bulk magnetic behavior from atomic-level Zeeman/Stark splitting,
unpaired electron counts, and exchange interactions, mapping through the
RCFT universality framework to predict material magnetism.

Physics — From Atomic Moments to Bulk Magnetism:
  Individual atoms: Zeeman splitting gives μ_eff = g_J · √(J(J+1)) · μ_B
  Bulk material:    Weiss molecular field theory scales atomic moments
                    to cooperative magnetic ordering.

  Diamagnetic:        χ < 0, all electrons paired (Langevin)
    χ_dia ≈ −(Z·e²)/(6m_e·c²) · <r²>  (Larmor precession)

  Paramagnetic:       χ > 0, T > T_ordering (Curie law)
    χ_para = C / T       where C = n·μ_eff²·μ₀/(3·k_B)

  Ferromagnetic:      T < T_c, J_exchange > 0 (Weiss + RCFT)
    M(T) = M_sat · |1 − T/T_c|^β   (β = 5/6 from RCFT, p=3)
    χ(T>T_c) = C/(T − T_c)         (Curie-Weiss law)

  Antiferromagnetic:  T < T_N, J_exchange < 0
    χ(T>T_N) = C/(T + θ)           (negative Weiss constant)

RCFT interpretation:
  Magnetic ordering IS collapse: above T_c, spins are disordered (high ω);
  below T_c, they freeze into ordered alignment (low ω).  The RCFT critical
  exponents (β=5/6, γ=1/3, ν=1/3) govern the transition identically to
  structural phase transitions because they share the same Γ(ω) landscape.

UMCP integration:
  ω_eff = |M_predicted − M_measured| / max(M_measured, ε)
  F_eff = 1 − ω_eff

Regime:
  Diamagnetic:       χ < 0 (no unpaired electrons)
  Paramagnetic:      χ > 0 and T > T_ordering
  Ferromagnetic:     T < T_c and n_unpaired > 0 and J > 0
  Antiferromagnetic: T < T_N and n_unpaired > 0 and J < 0

Cross-references:
  Atomic:   closures/atomic_physics/zeeman_stark.py (μ_eff, g_J)
  Phase:    closures/materials_science/phase_transition.py (RCFT exponents)
  RCFT:     closures/rcft/universality_class.py (c_eff, β, γ, ν)
  Contract: contracts/MATL.INTSTACK.v1.yaml
  Canon:    canon/matl_anchors.yaml (A-MT-MAG-*)
  Sources:  Kittel (2005), Ashcroft & Mermin (1976), Blundell (2001)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class MagneticClass(StrEnum):
    """Bulk magnetic classification."""

    DIAMAGNETIC = "Diamagnetic"
    PARAMAGNETIC = "Paramagnetic"
    FERROMAGNETIC = "Ferromagnetic"
    ANTIFERROMAGNETIC = "Antiferromagnetic"
    FERRIMAGNETIC = "Ferrimagnetic"


class MagneticRegime(StrEnum):
    """GCD regime for magnetic closure."""

    ORDERED = "Ordered"  # ω < 0.10
    MODERATE = "Moderate"  # 0.10 ≤ ω < 0.30
    DISORDERED = "Disordered"  # 0.30 ≤ ω < 0.60
    ANOMALOUS = "Anomalous"  # ω ≥ 0.60


class MagneticResult(NamedTuple):
    """Result of magnetic properties computation."""

    mu_eff_B: float  # Effective moment per atom (μ_B)
    M_total_B: float  # Total magnetization at T (μ_B/atom)
    M_sat_B: float  # Saturation magnetization (μ_B/atom)
    chi_SI: float  # Magnetic susceptibility (SI, dimensionless)
    T_ordering_K: float  # Ordering temperature (K)
    T_K: float  # Temperature (K)
    magnetic_class: str  # Dia / Para / Ferro / AF / Ferri
    J_exchange_meV: float  # Mean-field exchange energy (meV)
    g_lande: float  # Effective Landé g-factor
    Curie_const: float  # Curie constant (K)
    n_unpaired: int  # Number of unpaired electrons
    rcft_beta: float  # RCFT critical exponent β
    rcft_gamma: float  # RCFT critical exponent γ
    omega_eff: float
    F_eff: float
    regime: str


# ── Physical Constants ────────────────────────────────────────────
MU_B_EV = 5.7883818060e-5  # Bohr magneton (eV/T)
MU_B_JT = 9.2740100783e-24  # Bohr magneton (J/T)
K_B_EV = 8.617333262e-5  # Boltzmann constant (eV/K)
K_B_J = 1.380649e-23  # Boltzmann constant (J/K)
MU_0 = 1.25663706212e-6  # Vacuum permeability (H/m)
N_A = 6.02214076e23  # Avogadro's number

# RCFT exponents for p=3
BETA_RCFT = 5 / 6  # Order parameter exponent
GAMMA_RCFT = 1 / 3  # Susceptibility exponent
NU_RCFT = 1 / 3  # Correlation length exponent

# Regime thresholds
THRESH_ORDERED = 0.10
THRESH_MODERATE = 0.30
THRESH_DISORDERED = 0.60

# ── Reference Data ───────────────────────────────────────────────
#  M_sat (μ_B/atom), T_c or T_N (K), class, n_unpaired
REFERENCE_MAGNETIC: dict[str, dict[str, float | str | int]] = {
    # Ferromagnets
    "Fe": {"M_sat": 2.22, "T_c": 1043.0, "class": "Ferromagnetic", "n_unpaired": 4},
    "Co": {"M_sat": 1.72, "T_c": 1394.0, "class": "Ferromagnetic", "n_unpaired": 3},
    "Ni": {"M_sat": 0.606, "T_c": 631.0, "class": "Ferromagnetic", "n_unpaired": 2},
    "Gd": {"M_sat": 7.63, "T_c": 292.0, "class": "Ferromagnetic", "n_unpaired": 7},
    "Dy": {"M_sat": 10.2, "T_c": 88.0, "class": "Ferromagnetic", "n_unpaired": 5},
    # Antiferromagnets
    "Cr": {"M_sat": 0.0, "T_N": 311.0, "class": "Antiferromagnetic", "n_unpaired": 5},
    "Mn": {"M_sat": 0.0, "T_N": 100.0, "class": "Antiferromagnetic", "n_unpaired": 5},
    "MnO": {"M_sat": 0.0, "T_N": 116.0, "class": "Antiferromagnetic", "n_unpaired": 5},
    "NiO": {"M_sat": 0.0, "T_N": 523.0, "class": "Antiferromagnetic", "n_unpaired": 2},
    "FeO": {"M_sat": 0.0, "T_N": 198.0, "class": "Antiferromagnetic", "n_unpaired": 4},
    "CoO": {"M_sat": 0.0, "T_N": 291.0, "class": "Antiferromagnetic", "n_unpaired": 3},
    # Ferrimagnets
    "Fe3O4": {"M_sat": 4.1, "T_c": 858.0, "class": "Ferrimagnetic", "n_unpaired": 4},
    # Diamagnets
    "Cu": {"M_sat": 0.0, "T_c": 0.0, "class": "Diamagnetic", "n_unpaired": 0},
    "Au": {"M_sat": 0.0, "T_c": 0.0, "class": "Diamagnetic", "n_unpaired": 0},
    "Bi": {"M_sat": 0.0, "T_c": 0.0, "class": "Diamagnetic", "n_unpaired": 0},
    # Paramagnets (above T_c/T_N)
    "Al": {"M_sat": 0.0, "T_c": 0.0, "class": "Paramagnetic", "n_unpaired": 1},
    "Pt": {"M_sat": 0.0, "T_c": 0.0, "class": "Paramagnetic", "n_unpaired": 0},
}


def _effective_moment(n_unpaired: int, g: float = 2.0) -> float:
    """Spin-only effective moment: μ_eff = g·√(S(S+1)) μ_B."""
    s = n_unpaired / 2.0
    return g * math.sqrt(s * (s + 1))


def _curie_constant(mu_eff_B: float) -> float:
    """Curie constant C = μ₀·N_A·μ_eff²·μ_B² / (3·k_B) in K."""
    return MU_0 * N_A * (mu_eff_B * MU_B_JT) ** 2 / (3.0 * K_B_J)


def _exchange_from_Tc(T_c: float, S: float) -> float:
    """Mean-field exchange J from T_c: k_B·T_c = (2/3)·z·J·S(S+1).
    Returns J in meV (assuming z=8 neighbors)."""
    if S == 0 or T_c <= 0:
        return 0.0
    z = 8  # typical coordination
    J_eV = 3 * K_B_EV * T_c / (2 * z * S * (S + 1))
    return J_eV * 1000  # meV


def _classify_regime(omega: float) -> MagneticRegime:
    if omega < THRESH_ORDERED:
        return MagneticRegime.ORDERED
    if omega < THRESH_MODERATE:
        return MagneticRegime.MODERATE
    if omega < THRESH_DISORDERED:
        return MagneticRegime.DISORDERED
    return MagneticRegime.ANOMALOUS


def compute_magnetic_properties(
    Z: int,
    *,
    symbol: str = "",
    T_K: float = 300.0,
    B_tesla: float = 0.0,
    n_unpaired: int | None = None,
    mu_eff_B: float = 0.0,
    T_ordering_K: float | None = None,
    J_exchange_meV: float = 0.0,
    g_lande: float = 2.0,
    magnetic_class_hint: str = "",
    M_measured_B: float | None = None,
) -> MagneticResult:
    """Compute bulk magnetic properties from atomic-level inputs.

    Parameters
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element/compound symbol for reference lookup.
    T_K : float
        Temperature (K).
    B_tesla : float
        Applied magnetic field (T).
    n_unpaired : int | None
        Number of unpaired electrons. Auto-detected from reference if None.
    mu_eff_B : float
        Effective magnetic moment (μ_B). Computed from n_unpaired if 0.
    T_ordering_K : float | None
        Ordering temperature (T_c or T_N). Looked up from reference if None.
    J_exchange_meV : float
        Exchange energy (meV). Derived from T_ordering if 0.
    g_lande : float
        Landé g-factor (default 2.0 = spin-only).
    magnetic_class_hint : str
        Override automatic classification.
    M_measured_B : float | None
        Measured magnetization for ω_eff calculation.

    Returns
    -------
    MagneticResult
    """
    # ── Reference lookup ──
    ref = REFERENCE_MAGNETIC.get(symbol, {})
    if n_unpaired is None:
        n_unpaired = int(ref.get("n_unpaired", 0))
    if T_ordering_K is None:
        T_ordering_K = float(ref.get("T_c", ref.get("T_N", 0.0)))
    if M_measured_B is None:
        M_measured_B = float(ref.get("M_sat", 0.0))
    ref_class = str(ref.get("class", ""))

    # ── Effective moment ──
    if mu_eff_B <= 0:
        mu_eff_B = _effective_moment(n_unpaired, g_lande)
    S = n_unpaired / 2.0

    # ── Curie constant ──
    C = _curie_constant(mu_eff_B)

    # ── Exchange energy ──
    if J_exchange_meV == 0 and T_ordering_K > 0:
        J_exchange_meV = _exchange_from_Tc(T_ordering_K, S)

    # ── Classify magnetic type ──
    if magnetic_class_hint:
        mag_class = MagneticClass(magnetic_class_hint)
    elif ref_class:
        mag_class = MagneticClass(ref_class)
    elif n_unpaired == 0:
        mag_class = MagneticClass.DIAMAGNETIC
    elif T_ordering_K <= 0:
        mag_class = MagneticClass.PARAMAGNETIC
    elif J_exchange_meV > 0:
        mag_class = MagneticClass.FERROMAGNETIC
    elif J_exchange_meV < 0:
        mag_class = MagneticClass.ANTIFERROMAGNETIC
    else:
        mag_class = MagneticClass.PARAMAGNETIC

    # ── Magnetization & susceptibility ──
    M_sat = float(ref.get("M_sat", mu_eff_B))  # saturation in μ_B/atom
    M_total = 0.0
    chi = 0.0
    eps = 1e-12

    if mag_class == MagneticClass.DIAMAGNETIC:
        # Larmor diamagnetic susceptibility (approximate)
        chi = -Z * 1e-6  # rough scaling −Z × 10⁻⁶
        M_total = chi * B_tesla / MU_0 * 1e-6  # tiny

    elif mag_class == MagneticClass.PARAMAGNETIC:
        # Curie law: χ = C/T
        chi = C / T_K if eps < T_K else C / eps
        M_total = chi * B_tesla / MU_0 if B_tesla > 0 else 0.0
        # Normalize to μ_B scale
        if abs(M_total) > mu_eff_B:
            M_total = mu_eff_B

    elif mag_class in (MagneticClass.FERROMAGNETIC, MagneticClass.FERRIMAGNETIC):
        if T_ordering_K > 0 and T_ordering_K > T_K:
            # Below T_c: M(T) = M_sat · |1 − T/T_c|^β  (RCFT)
            t_red = abs(1.0 - T_K / T_ordering_K)
            M_total = M_sat * t_red**BETA_RCFT
        else:
            # Above T_c: Curie-Weiss: χ = C/(T − T_c)
            M_total = 0.0
            denom = T_K - T_ordering_K
            chi = C / denom if abs(denom) > eps else C / eps  # diverges at T_c

    elif mag_class == MagneticClass.ANTIFERROMAGNETIC:
        T_N = T_ordering_K
        if T_N > 0 and T_K < T_N:
            # Below T_N: staggered order, net M ≈ 0
            M_total = 0.0
            chi = C / (T_K + T_N)  # Modified Curie-Weiss
        else:
            # Above T_N: χ = C/(T + θ)
            if eps < T_K + T_N:
                chi = C / (T_K + T_N)
            M_total = 0.0

    # ── GCD mapping ──
    if M_measured_B and M_measured_B > eps:
        omega_eff = abs(M_total - M_measured_B) / M_measured_B
    elif mu_eff_B > eps:
        omega_eff = abs(M_total - mu_eff_B) / mu_eff_B
    else:
        omega_eff = 0.0

    omega_eff = min(1.0, omega_eff)
    F_eff = 1.0 - omega_eff
    regime = _classify_regime(omega_eff)

    return MagneticResult(
        mu_eff_B=round(mu_eff_B, 6),
        M_total_B=round(M_total, 6),
        M_sat_B=round(M_sat, 6),
        chi_SI=chi,
        T_ordering_K=T_ordering_K,
        T_K=T_K,
        magnetic_class=mag_class.value,
        J_exchange_meV=round(J_exchange_meV, 6),
        g_lande=round(g_lande, 6),
        Curie_const=round(C, 8),
        n_unpaired=n_unpaired,
        rcft_beta=round(BETA_RCFT, 6),
        rcft_gamma=round(GAMMA_RCFT, 6),
        omega_eff=round(omega_eff, 8),
        F_eff=round(F_eff, 8),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Magnetic Properties Self-Test ===\n")

    tests = [
        ("Fe", 300.0, "Below T_c"),
        ("Fe", 1043.0, "At T_c"),
        ("Fe", 1200.0, "Above T_c"),
        ("Ni", 300.0, "Ferromagnet"),
        ("Co", 300.0, "Ferromagnet"),
        ("Gd", 300.0, "Above T_c (292K)"),
        ("Cr", 300.0, "Paramagnetic (T_N=311K)"),
        ("Cu", 300.0, "Diamagnetic"),
        ("Al", 300.0, "Paramagnetic"),
    ]

    for sym, T, desc in tests:
        r = compute_magnetic_properties(0, symbol=sym, T_K=T)
        print(
            f"  {sym:5s} T={T:6.0f}K  M={r.M_total_B:8.4f}μ_B  "
            f"χ={r.chi_SI:12.6e}  {r.magnetic_class:18s}  "
            f"ω={r.omega_eff:.4f}  {r.regime}  | {desc}"
        )
