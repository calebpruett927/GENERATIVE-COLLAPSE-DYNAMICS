"""Phase Transition Closure — MATL.INTSTACK.v1

Derives material phase transition behavior from RCFT universality classes.
This is the critical bridge: the GCD kernel's drift potential Γ(ω) = ω^p/(1−ω+ε)
defines a universality class with critical exponents that *exactly* govern
material phase transitions (structural, magnetic, superconducting).

Key RCFT → Materials Derivation:
  The partition function Z(β) = ∫ exp(−βΓ(ω)) dω from RCFT universality_class.py
  governs how a material's order parameter Φ vanishes near T_c:

    Φ ~ (T_c − T)^β_exp              order parameter
    χ ~ |T − T_c|^{−γ}               susceptibility divergence
    C_V ~ |T − T_c|^{−α}             specific heat
    ξ ~ |T − T_c|^{−ν}               correlation length

  For the GCD kernel with p = 3:
    (ν, γ, α, β_exp) = (1/3, 1/3, 0, 5/6)
    d_eff = 6  (upper critical dimension)

  This means materials governed by GCD collapse dynamics exhibit:
    - Logarithmic specific heat (α = 0) near T_c
    - Weak susceptibility divergence (γ = 1/3 < mean-field γ = 1)
    - Strong order parameter persistence (β = 5/6 > mean-field β = 1/2)

Physics:
  Structural transitions:   Crystal structure changes (BCC ↔ FCC ↔ HCP)
  Magnetic transitions:     Ferromagnetic ↔ paramagnetic (Curie point)
  Superconducting:          Normal → superconducting (T_c)
  Displacive:               Soft-mode driven (ferroelectric T_c)

UMCP integration:
  ω_eff = |T − T_c| / T_c    (thermal drift from critical point)
  F_eff = 1 − ω_eff
  Regime classification uses RCFT attractor basin topology

Cross-references:
  RCFT:   closures/rcft/universality_class.py (T20, T21 — critical exponents)
  RCFT:   closures/rcft/attractor_basin.py (basin topology → polymorphism)
  RCFT:   closures/rcft/resonance_pattern.py (soft modes near T_c)
  Atomic: closures/atomic_physics/fine_structure.py (relativistic corrections to magnetic coupling)
  SM:     closures/standard_model/symmetry_breaking.py (SSB as paradigm for phase transitions)
  Sources: Landau & Lifshitz (1980), Wilson & Kogut (1974), Goldenfeld (1992)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class TransitionType(StrEnum):
    """Classification of phase transition."""

    STRUCTURAL = "Structural"
    MAGNETIC = "Magnetic"
    SUPERCONDUCTING = "Superconducting"
    DISPLACIVE = "Displacive"
    ORDER_DISORDER = "OrderDisorder"


class TransitionOrder(StrEnum):
    """Ehrenfest classification."""

    FIRST = "First"  # Latent heat, discontinuous order parameter
    SECOND = "Second"  # Continuous order parameter, divergent susceptibility
    CROSSOVER = "Crossover"  # No true singularity


class PhaseRegime(StrEnum):
    """Regime relative to critical point."""

    ORDERED = "Ordered"  # T << T_c, deep in ordered phase
    CRITICAL = "Critical"  # |T − T_c| / T_c < 0.05
    FLUCTUATION = "Fluctuation"  # Near T_c but above
    DISORDERED = "Disordered"  # T >> T_c


class PhaseTransitionResult(NamedTuple):
    """Result of phase transition analysis using RCFT universality."""

    T_K: float  # Current temperature (K)
    T_c_K: float  # Critical temperature (K)
    reduced_t: float  # Reduced temperature t = (T − T_c) / T_c
    transition_type: str
    transition_order: str
    # RCFT-derived quantities
    order_parameter: float  # Φ ~ |t|^β below T_c
    susceptibility: float  # χ ~ |t|^{−γ}
    correlation_length_ratio: float  # ξ/ξ₀ ~ |t|^{−ν}
    specific_heat_singular: float  # C_sing ~ |t|^{−α} (logarithmic for α=0)
    # Critical exponents (from RCFT universality class)
    nu: float
    gamma: float
    alpha: float
    beta_exp: float
    c_eff: float  # Central charge
    d_eff: float  # Effective dimension
    # GCD mapping
    omega_eff: float
    F_eff: float
    regime: str


# ── RCFT-derived critical exponents ─────────────────────────────
# From universality_class.py T21 with p = 3 (GCD kernel default)
P_DEFAULT = 3
NU_GCD = 1.0 / P_DEFAULT  # 1/3
GAMMA_GCD = (P_DEFAULT - 2.0) / P_DEFAULT  # 1/3
ALPHA_GCD = 0.0
BETA_EXP_GCD = (P_DEFAULT + 2.0) / (2.0 * P_DEFAULT)  # 5/6
DELTA_GCD = (3.0 * P_DEFAULT - 2.0) / (P_DEFAULT + 2.0)  # 7/5
D_EFF_GCD = 2.0 * P_DEFAULT  # 6
C_EFF_GCD = 1.0 / P_DEFAULT  # 1/3

# Reference material critical temperatures (K)
REFERENCE_TC: dict[str, dict[str, Any]] = {
    # Magnetic transitions (Curie temperature)
    "Fe": {"T_c": 1043.0, "type": "Magnetic", "order": "Second"},
    "Co": {"T_c": 1388.0, "type": "Magnetic", "order": "Second"},
    "Ni": {"T_c": 627.0, "type": "Magnetic", "order": "Second"},
    "Gd": {"T_c": 292.0, "type": "Magnetic", "order": "Second"},
    "MnO": {"T_c": 116.0, "type": "Magnetic", "order": "Second"},  # Néel
    # Structural transitions
    "Fe_bcc_fcc": {"T_c": 1185.0, "type": "Structural", "order": "First"},
    "Ti_hcp_bcc": {"T_c": 1155.0, "type": "Structural", "order": "First"},
    "Zr_hcp_bcc": {"T_c": 1136.0, "type": "Structural", "order": "First"},
    "Sn_grey_white": {"T_c": 286.0, "type": "Structural", "order": "First"},
    # Superconducting (conventional BCS)
    "Nb": {"T_c": 9.25, "type": "Superconducting", "order": "Second"},
    "Pb": {"T_c": 7.19, "type": "Superconducting", "order": "Second"},
    "Al_sc": {"T_c": 1.18, "type": "Superconducting", "order": "Second"},
    "Sn_sc": {"T_c": 3.72, "type": "Superconducting", "order": "Second"},
    "MgB2": {"T_c": 39.0, "type": "Superconducting", "order": "Second"},
    # Displacive / ferroelectric
    "BaTiO3": {"T_c": 393.0, "type": "Displacive", "order": "First"},
    "SrTiO3": {"T_c": 105.0, "type": "Displacive", "order": "Second"},
    "PbTiO3": {"T_c": 763.0, "type": "Displacive", "order": "First"},
}

# Regime thresholds (in reduced temperature)
THRESH_CRITICAL = 0.05
THRESH_FLUCTUATION = 0.30


def _classify_regime(reduced_t: float) -> PhaseRegime:
    """Classify phase regime from reduced temperature."""
    abs_t = abs(reduced_t)
    if abs_t < THRESH_CRITICAL:
        return PhaseRegime.CRITICAL
    if reduced_t < 0:
        return PhaseRegime.ORDERED
    if abs_t < THRESH_FLUCTUATION:
        return PhaseRegime.FLUCTUATION
    return PhaseRegime.DISORDERED


def _rcft_order_parameter(
    reduced_t: float,
    beta_exp: float,
    *,
    phi_0: float = 1.0,
) -> float:
    """Order parameter Φ = Φ₀ · |t|^β for T < T_c (t < 0).

    From RCFT universality class T21:
      β = (p+2)/(2p) = 5/6 for p = 3

    Physical meaning: The order parameter (magnetization, lattice distortion,
    superconducting gap) persists more strongly near T_c than mean-field
    theory predicts (β_MF = 1/2 vs β_GCD = 5/6). This implies that
    GCD-governed materials have *more robust ordered phases*.
    """
    if reduced_t >= 0:
        return 0.0  # Disordered above T_c
    return phi_0 * abs(reduced_t) ** beta_exp


def _rcft_susceptibility(
    reduced_t: float,
    gamma_exp: float,
    *,
    chi_0: float = 1.0,
    regularize: float = 1e-6,
) -> float:
    """Susceptibility χ = χ₀ · |t|^{−γ}.

    From RCFT T21: γ = (p−2)/p = 1/3 for p = 3

    Compared to Ising 3D (γ ≈ 1.24) or mean-field (γ = 1),
    the GCD universality has *weak* susceptibility divergence.
    This reflects the kernel's self-stabilizing nature:
    even near collapse, the drift potential Γ(ω) prevents
    runaway fluctuations.
    """
    abs_t = max(abs(reduced_t), regularize)
    return chi_0 / abs_t**gamma_exp


def _rcft_correlation_length(
    reduced_t: float,
    nu_exp: float,
    *,
    regularize: float = 1e-6,
) -> float:
    """Correlation length ratio ξ/ξ₀ = |t|^{−ν}.

    From RCFT T21: ν = 1/p = 1/3 for p = 3

    Short correlation length divergence means interactions
    remain relatively local even near T_c — consistent with
    the GCD kernel's one-pass (no back-edges) constraint.
    """
    abs_t = max(abs(reduced_t), regularize)
    return 1.0 / abs_t**nu_exp


def _rcft_specific_heat(
    reduced_t: float,
    alpha_exp: float,
    *,
    regularize: float = 1e-6,
) -> float:
    """Singular specific heat C_sing.

    For α = 0 (GCD universality): C_sing ~ −ln|t| (logarithmic).
    This is the *marginal* case — the specific heat has a cusp
    but no divergence, exactly as the partition function analysis
    (T20 central charge) predicts: C_V → 1/p = const.
    """
    abs_t = max(abs(reduced_t), regularize)
    if abs(alpha_exp) < 1e-10:
        # α = 0: logarithmic singularity
        return -math.log(abs_t)
    return 1.0 / abs_t**alpha_exp


def compute_phase_transition(
    T_K: float,
    T_c_K: float,
    *,
    material_key: str = "",
    transition_type: str = "Magnetic",
    transition_order: str = "Second",
    p: int = P_DEFAULT,
    phi_0: float = 1.0,
    chi_0: float = 1.0,
) -> PhaseTransitionResult:
    """Compute phase transition behavior using RCFT universality class.

    This is the core RCFT → materials science bridge. The critical exponents
    derived from the GCD kernel's drift potential (closures/rcft/universality_class.py)
    predict how material order parameters, susceptibilities, and correlation
    lengths behave near phase transitions.

    Parameters
    ----------
    T_K : float
        Current temperature (K).
    T_c_K : float
        Critical temperature (K). If material_key given, looked up.
    material_key : str
        Key into REFERENCE_TC dictionary.
    transition_type : str
        Structural, Magnetic, Superconducting, Displacive.
    transition_order : str
        First, Second, or Crossover.
    p : int
        GCD drift exponent (determines universality class).
    phi_0 : float
        Order parameter amplitude.
    chi_0 : float
        Susceptibility amplitude.

    Returns
    -------
    PhaseTransitionResult
    """
    if T_K < 0:
        msg = f"Temperature must be ≥ 0, got {T_K}"
        raise ValueError(msg)

    # Look up reference if available
    if material_key and material_key in REFERENCE_TC:
        ref = REFERENCE_TC[material_key]
        T_c_K = ref["T_c"]
        transition_type = ref["type"]
        transition_order = ref["order"]

    if T_c_K <= 0:
        msg = f"Critical temperature must be > 0, got {T_c_K}"
        raise ValueError(msg)

    # RCFT critical exponents from p
    nu = 1.0 / p
    gamma_exp = (p - 2.0) / p
    alpha_exp = 0.0
    beta_exp = (p + 2.0) / (2.0 * p)
    c_eff = 1.0 / p
    d_eff = 2.0 * p

    # Reduced temperature
    reduced_t = (T_K - T_c_K) / T_c_K

    # Compute RCFT-predicted quantities
    order_param = _rcft_order_parameter(reduced_t, beta_exp, phi_0=phi_0)
    suscept = _rcft_susceptibility(reduced_t, gamma_exp, chi_0=chi_0)
    corr_len = _rcft_correlation_length(reduced_t, nu)
    c_sing = _rcft_specific_heat(reduced_t, alpha_exp)

    # GCD mapping: thermal drift from critical point
    omega_eff = min(1.0, abs(reduced_t))
    f_eff = 1.0 - omega_eff

    regime = _classify_regime(reduced_t)

    return PhaseTransitionResult(
        T_K=round(T_K, 2),
        T_c_K=round(T_c_K, 2),
        reduced_t=round(reduced_t, 6),
        transition_type=transition_type,
        transition_order=transition_order,
        order_parameter=round(order_param, 6),
        susceptibility=round(suscept, 6),
        correlation_length_ratio=round(corr_len, 4),
        specific_heat_singular=round(c_sing, 4),
        nu=round(nu, 6),
        gamma=round(gamma_exp, 6),
        alpha=round(alpha_exp, 6),
        beta_exp=round(beta_exp, 6),
        c_eff=round(c_eff, 6),
        d_eff=round(d_eff, 1),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        regime=regime.value,
    )


def scan_phase_diagram(
    T_c_K: float,
    *,
    T_min_K: float = 0.0,
    T_max_K: float = 0.0,
    n_points: int = 100,
    p: int = P_DEFAULT,
    transition_type: str = "Magnetic",
) -> list[PhaseTransitionResult]:
    """Scan temperature range to build a phase diagram.

    Returns a list of PhaseTransitionResult across the temperature range,
    useful for plotting order parameter, susceptibility, and correlation
    length as functions of temperature.
    """
    if T_max_K <= 0:
        T_max_K = 2.0 * T_c_K
    if T_min_K <= 0:
        T_min_K = 0.01 * T_c_K

    temperatures = [T_min_K + (T_max_K - T_min_K) * i / (n_points - 1) for i in range(n_points)]

    return [compute_phase_transition(T, T_c_K, p=p, transition_type=transition_type) for T in temperatures]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RCFT → Materials Phase Transition Bridge")
    print("=" * 65)

    # Iron Curie transition
    for t_frac in [0.5, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.5]:
        T = t_frac * 1043.0
        r = compute_phase_transition(T, 1043.0, material_key="Fe")
        print(
            f"T={T:7.1f}K  t={r.reduced_t:+.4f}  Φ={r.order_parameter:.4f}  "
            f"χ={r.susceptibility:8.2f}  ξ/ξ₀={r.correlation_length_ratio:8.2f}  "
            f"C={r.specific_heat_singular:6.2f}  {r.regime}"
        )

    print("\nRCFT Critical Exponents (p=3, GCD universality):")
    print(
        f"  ν = {NU_GCD:.4f}  γ = {GAMMA_GCD:.4f}  α = {ALPHA_GCD:.4f}  "
        f"β = {BETA_EXP_GCD:.4f}  c_eff = {C_EFF_GCD:.4f}  d_eff = {D_EFF_GCD:.1f}"
    )
