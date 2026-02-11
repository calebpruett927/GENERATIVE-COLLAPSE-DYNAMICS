"""Debye Thermal Closure — MATL.INTSTACK.v1

Derives thermal properties (specific heat, Debye temperature, thermal
conductivity) from the RCFT partition function framework applied to
lattice vibrations.

Physics — The RCFT Bridge to Debye Model:
  The RCFT partition function Z(β) = ∫ exp(−βΓ(ω)) dω from
  universality_class.py has *exactly* the structure of the phonon
  partition function when we identify:

    ω (GCD drift) ↔ ω_D (Debye cutoff, normalized frequency)
    Γ(ω) = ω^p/(1−ω+ε) ↔ phonon density of states × ℏω
    β ↔ 1/k_BT (inverse temperature)

  The Debye model partition function:
    Z_Debye(T) = ∏_k [2sinh(ℏω_k/2k_BT)]^{−1}
    ⟹ ln Z = −3N Σ_k ln[2sinh(x_k/2)]  where x_k = ℏω_k/k_BT

  The RCFT central charge c_eff = 1/p governs the UV cutoff:
    Θ_D = c_eff × (ℏ/k_B) × (6π²n/V)^{1/3} × v_s

  RCFT prediction for specific heat:
    C_V(T) → 3Nk_B   (Dulong-Petit, T >> Θ_D)
    C_V(T) → (12π⁴/5)Nk_B(T/Θ_D)³  (Debye T³ law, T << Θ_D)

  The c_eff = 1/3 (p=3) central charge means 1/3 of lattice degrees
  of freedom are thermally "active" at any temperature — exactly
  matching the 3 acoustic phonon branches out of 3N total modes.

Atomic inputs:
  - Elastic moduli → sound velocity → Θ_D
  - Atomic mass → molar specific heat
  - Coordination → Debye cutoff frequency

UMCP integration:
  ω_eff = |C_V(T) − C_V_measured| / C_V_Dulong_Petit
  F_eff = 1 − ω_eff

Regime:
  Quantum:       T < 0.1 Θ_D  (T³ law)
  Intermediate:  0.1 ≤ T/Θ_D < 0.8
  Classical:     T/Θ_D ≥ 0.8  (Dulong-Petit)

Cross-references:
  Materials:  closures/materials_science/elastic_moduli.py (K, G → v_s)
  RCFT:       closures/rcft/universality_class.py (partition function, c_eff)
  RCFT:       closures/rcft/information_geometry.py (thermodynamic efficiency)
  Atomic:     closures/atomic_physics/ (mass, coordination)
  Sources:    Debye (1912), Kittel (2005), Grimvall (1999)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple

import numpy as np


class ThermalRegime(StrEnum):
    """Regime based on T/Θ_D ratio."""

    QUANTUM = "Quantum"
    INTERMEDIATE = "Intermediate"
    CLASSICAL = "Classical"


class DebyeResult(NamedTuple):
    """Result of Debye thermal computation."""

    T_K: float  # Temperature (K)
    Theta_D_K: float  # Debye temperature (K)
    T_over_Theta: float  # Reduced temperature
    C_V_J_mol_K: float  # Molar heat capacity (J/mol·K)
    C_V_normalized: float  # C_V / 3R (approaches 1 at high T)
    U_J_mol: float  # Internal energy (J/mol)
    S_J_mol_K: float  # Entropy (J/mol·K)
    v_sound_ms: float  # Average sound velocity (m/s)
    kappa_thermal_W_mK: float  # Estimated thermal conductivity (W/m·K)
    mean_free_path_nm: float  # Phonon mean free path estimate (nm)
    # RCFT quantities
    rcft_c_eff: float  # Central charge (= 1/p, fraction of active modes)
    rcft_partition_analogy: float  # Z_phonon / Z_RCFT ratio
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
K_B = 1.380649e-23  # Boltzmann constant (J/K)
R_GAS = 8.31446  # Gas constant (J/mol·K)
HBAR = 1.054571817e-34  # ℏ (J·s)
N_A = 6.02214076e23  # Avogadro's number
DULONG_PETIT = 3.0 * R_GAS  # 24.94 J/mol·K

# Reference Debye temperatures (K)
REFERENCE_THETA_D: dict[str, float] = {
    # Metals
    "Li": 344,
    "Na": 158,
    "K": 91,
    "Rb": 56,
    "Cs": 38,
    "Be": 1440,
    "Mg": 400,
    "Ca": 230,
    "Al": 428,
    "Cu": 343,
    "Ag": 225,
    "Au": 165,
    "Fe": 470,
    "Co": 445,
    "Ni": 450,
    "Cr": 630,
    "Ti": 420,
    "V": 380,
    "Nb": 275,
    "Mo": 450,
    "W": 400,
    "Pt": 240,
    "Pd": 274,
    "Rh": 480,
    "Ir": 420,
    "Zn": 327,
    "Cd": 209,
    "Pb": 105,
    "Sn": 200,
    # Semiconductors / insulators
    "Si": 645,
    "Ge": 374,
    "C_diamond": 2230,
    "GaAs": 344,
    "InSb": 202,
    "SiC": 1200,
    "NaCl": 321,
    "KCl": 235,
    "MgO": 946,
}


def _classify(t_ratio: float) -> ThermalRegime:
    if t_ratio < 0.1:
        return ThermalRegime.QUANTUM
    if t_ratio < 0.8:
        return ThermalRegime.INTERMEDIATE
    return ThermalRegime.CLASSICAL


def _debye_function_3(x: float, n_points: int = 200) -> float:
    """Debye function D₃(x) = (3/x³) ∫₀ˣ t³/(e^t − 1) dt.

    This is the key integral that maps the RCFT partition function
    structure to the phonon heat capacity.
    """
    if x <= 0:
        return 1.0  # High-T limit: D₃(0) = 1

    if x > 100:
        # Very low T: D₃(x) → (4π⁴/5)/x³ (Debye T³ law)
        return (4.0 * math.pi**4 / 5.0) / x**3

    t = np.linspace(1e-10, x, n_points)
    integrand = t**3 / (np.exp(t) - 1.0)
    integral = float(np.trapezoid(integrand, t))
    return 3.0 * integral / x**3


def _debye_energy_function(x: float, n_points: int = 200) -> float:
    """Internal energy integral: (3/x³) ∫₀ˣ t⁴ e^t / (e^t − 1)² dt.

    Used for internal energy: U = 9Nk_BT · D₃(Θ_D/T) + U_zero_point
    """
    if x <= 0:
        return 1.0
    if x > 100:
        return (4.0 * math.pi**4 / 5.0) / x**3

    t = np.linspace(1e-10, x, n_points)
    et = np.exp(t)
    integrand = t**4 * et / (et - 1.0) ** 2
    integral = float(np.trapezoid(integrand, t))
    return 3.0 * integral / x**3


def _debye_entropy_function(x: float, n_points: int = 200) -> float:
    """Entropy function: 4D₃(x) − 3 ln(1 − e^{−x})."""
    d3 = _debye_function_3(x, n_points)
    log_term = -3.0 * math.log(1.0 - math.exp(-x)) if x > 1e-10 else 3.0 * math.log(x)
    return 4.0 * d3 + log_term


def _sound_velocity_from_K_G(K_GPa: float, G_GPa: float, rho_kg_m3: float) -> float:
    """Average sound velocity from elastic moduli.

    v_s = [(1/3)(2/v_t³ + 1/v_l³)]^{−1/3}
    v_l = √((K + 4G/3)/ρ)
    v_t = √(G/ρ)
    """
    if rho_kg_m3 <= 0 or K_GPa <= 0:
        return 3000.0  # typical metal estimate

    K_pa = K_GPa * 1e9
    G_pa = G_GPa * 1e9

    v_l = math.sqrt((K_pa + 4.0 * G_pa / 3.0) / rho_kg_m3)
    v_t = math.sqrt(G_pa / rho_kg_m3) if G_pa > 0 else v_l * 0.55

    # Average (Debye):
    avg_inv_cube = (2.0 / v_t**3 + 1.0 / v_l**3) / 3.0
    return avg_inv_cube ** (-1.0 / 3.0)


def compute_debye_thermal(
    T_K: float,
    *,
    symbol: str = "",
    Theta_D_K: float = 0.0,
    atomic_mass_amu: float = 0.0,
    K_GPa: float = 0.0,
    G_GPa: float = 0.0,
    rho_kg_m3: float = 0.0,
    p: int = 3,
) -> DebyeResult:
    """Compute thermal properties using Debye model with RCFT interpretation.

    Parameters
    ----------
    T_K : float
        Temperature (K).
    symbol : str
        Element symbol for reference lookup.
    Theta_D_K : float
        Debye temperature (K). If 0, looked up or estimated.
    atomic_mass_amu : float
        Atomic mass (amu). Used for density and molar quantities.
    K_GPa, G_GPa : float
        Elastic moduli (GPa). Used to estimate sound velocity.
    rho_kg_m3 : float
        Mass density (kg/m³). Used for sound velocity.
    p : int
        GCD drift exponent for RCFT central charge.

    Returns
    -------
    DebyeResult
    """
    if T_K < 0:
        msg = f"Temperature must be ≥ 0, got {T_K}"
        raise ValueError(msg)
    if T_K == 0:
        T_K = 0.01  # avoid division by zero

    # Debye temperature lookup/estimation
    if Theta_D_K <= 0 and symbol in REFERENCE_THETA_D:
        Theta_D_K = REFERENCE_THETA_D[symbol]

    # Sound velocity
    v_sound = _sound_velocity_from_K_G(K_GPa, G_GPa, rho_kg_m3)

    if Theta_D_K <= 0:
        # Estimate from sound velocity if elastic data available
        if rho_kg_m3 > 0 and atomic_mass_amu > 0:
            # Θ_D = (ℏ/k_B)(6π²N_A ρ / M)^{1/3} · v_s
            n_density = N_A * rho_kg_m3 / (atomic_mass_amu * 1e-3)
            Theta_D_K = (HBAR / K_B) * (6.0 * math.pi**2 * n_density) ** (1.0 / 3.0) * v_sound
        else:
            Theta_D_K = 300.0  # generic fallback

    # Reduced temperature
    t_ratio = T_K / Theta_D_K
    x = Theta_D_K / T_K  # Debye argument

    # Specific heat: C_V = 9Nk_B · (T/Θ_D)³ · ∫₀^{Θ_D/T} t⁴e^t/(e^t-1)² dt
    # = 3R · D₃'(x)  where D₃' involves the derivative integral
    cv_normalized = _debye_energy_function(x)
    C_V = DULONG_PETIT * cv_normalized  # J/mol·K

    # Internal energy: U = 9Nk_BT · D₃(Θ_D/T) + (9/8)Nk_BΘ_D
    d3 = _debye_function_3(x)
    U_thermal = 3.0 * R_GAS * T_K * d3  # J/mol (thermal part)
    U_zero_point = (9.0 / 8.0) * R_GAS * Theta_D_K  # zero-point energy
    U_total = U_thermal + U_zero_point

    # Entropy
    if t_ratio > 0.001:
        S_debye = 3.0 * R_GAS * (4.0 * d3 - math.log(1.0 - math.exp(-x)) if x < 500 else 4.0 * d3)
    else:
        S_debye = 0.0

    # Thermal conductivity estimate (kinetic theory)
    # κ = (1/3) C_V v_s λ_mfp
    # Mean free path: λ ~ a₀/T at high T (Umklapp), λ ~ exp(Θ_D/bT) at low T
    if t_ratio > 0.5:
        # High T: Umklapp dominated
        mfp_m = 3e-10 / t_ratio  # ~3Å at Θ_D, decreasing
    else:
        # Low T: boundary/defect limited
        mfp_m = 1e-6 * math.exp(min(20.0, 1.0 / (3.0 * max(t_ratio, 0.01))))
        mfp_m = min(mfp_m, 1e-3)  # cap at 1 mm

    mfp_nm = mfp_m * 1e9

    # κ in W/m·K (using volumetric specific heat approximation)
    C_v_vol = C_V * 1e3  # rough J/m³·K for typical solid
    if rho_kg_m3 > 0 and atomic_mass_amu > 0:
        C_v_vol = C_V * rho_kg_m3 / (atomic_mass_amu * 1e-3)
    kappa_thermal = (1.0 / 3.0) * C_v_vol * v_sound * mfp_m

    # RCFT central charge
    c_eff = 1.0 / p
    # The partition function analogy: ratio of Debye Z to RCFT Z
    # At high T, both approach power-law → ratio → 1
    rcft_z_ratio = cv_normalized  # C_V/3R as proxy

    # GCD mapping
    omega_eff = abs(1.0 - cv_normalized)  # deviation from Dulong-Petit
    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff

    regime = _classify(t_ratio)

    return DebyeResult(
        T_K=round(T_K, 2),
        Theta_D_K=round(Theta_D_K, 1),
        T_over_Theta=round(t_ratio, 4),
        C_V_J_mol_K=round(C_V, 4),
        C_V_normalized=round(cv_normalized, 6),
        U_J_mol=round(U_total, 2),
        S_J_mol_K=round(S_debye, 4),
        v_sound_ms=round(v_sound, 1),
        kappa_thermal_W_mK=round(kappa_thermal, 2),
        mean_free_path_nm=round(mfp_nm, 2),
        rcft_c_eff=round(c_eff, 6),
        rcft_partition_analogy=round(rcft_z_ratio, 6),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Debye Thermal Model via RCFT Partition Function Bridge")
    print("=" * 80)

    # Cu at various temperatures
    for T in [10, 30, 100, 200, 300, 500, 1000]:
        r = compute_debye_thermal(T, symbol="Cu")
        print(
            f"Cu T={T:5d}K  T/Θ_D={r.T_over_Theta:.3f}  "
            f"C_V={r.C_V_J_mol_K:6.2f} J/mol·K  "
            f"C_V/3R={r.C_V_normalized:.4f}  "
            f"κ={r.kappa_thermal_W_mK:8.1f} W/m·K  {r.regime}"
        )

    print("\nRCFT Bridge — c_eff = 1/3 (p=3):")
    print("  The central charge c_eff = 1/3 corresponds to the 3 acoustic")
    print("  phonon branches being the 'active' collapse channels out of")
    print("  3N total modes — exactly the Debye model assumption.")
