"""BCS Superconductivity Closure — MATL.INTSTACK.v1

Derives superconducting properties from Debye temperature and electron-phonon
coupling via BCS theory, mapping through the RCFT partition function framework.

Physics — BCS as RCFT Condensation:
  BCS theory (Bardeen-Cooper-Schrieffer 1957):
    Cooper pairs form when attractive electron-phonon interaction V
    overcomes Coulomb repulsion μ*:
      T_c = (Θ_D / 1.45) · exp(−1.04(1+λ) / (λ − μ*(1+0.62λ)))   [McMillan]
      Δ(0) = 1.764 · k_B · T_c                                     [weak coupling]
      2Δ(0)/(k_B·T_c) = 3.528                                      [BCS ratio]

  RCFT interpretation:
    The BCS gap Δ(0) is the RCFT collapse-return boundary energy.
    The partition function Z(β) = ∫ exp(−βΓ(ω))dω develops a NEW minimum
    at the Cooper pair condensation — this is exactly a partition function
    phase transition.  The return time τ_R determines the gap scale:
      Δ ∝ ℏ/τ_R  (uncertainty principle → gap ↔ return time)

    The electron-phonon coupling λ_ep maps to the RCFT attractor depth:
      Strong coupling (λ > 1) → deep attractor, robust superconductor
      Weak coupling (λ < 0.5) → shallow attractor, fragile pairing

  Key observables:
    T_c   — Critical temperature (McMillan equation)
    Δ(0)  — Zero-temperature gap energy
    C_jump — Specific heat jump ratio ΔC/(γT_c) = 1.43 (BCS)
    ξ₀    — BCS coherence length = ℏv_F/(πΔ(0))
    λ_L   — London penetration depth
    κ_GL  — Ginzburg-Landau parameter = λ_L/ξ₀
            κ < 1/√2 → Type I,  κ > 1/√2 → Type II

UMCP integration:
  ω_eff = |T_c_predicted − T_c_measured| / max(T_c_measured, ε)
  F_eff = 1 − ω_eff

Regime:
  Normal:  T > T_c (no superconductivity)
  TypeI:   T < T_c and κ_GL < 1/√2
  TypeII:  T < T_c and κ_GL ≥ 1/√2

Cross-references:
  Debye:    closures/materials_science/debye_thermal.py (Θ_D)
  RCFT:     closures/rcft/universality_class.py (partition function)
  Phase:    closures/materials_science/phase_transition.py (critical exponents)
  Contract: contracts/MATL.INTSTACK.v1.yaml
  Canon:    canon/matl_anchors.yaml (A-MT-BCS-*)
  Sources:  BCS (1957), McMillan (1968), Tinkham (2004), Allen & Dynes (1975)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class SCType(StrEnum):
    """Superconductor type classification."""

    NORMAL = "Normal"
    TYPE_I = "TypeI"
    TYPE_II = "TypeII"


class SCRegime(StrEnum):
    """GCD regime for superconductivity closure."""

    PRECISE = "Precise"  # ω < 0.10
    MODERATE = "Moderate"  # 0.10 ≤ ω < 0.30
    APPROXIMATE = "Approximate"  # 0.30 ≤ ω < 0.60
    ANOMALOUS = "Anomalous"  # ω ≥ 0.60


class BCSResult(NamedTuple):
    """Result of BCS superconductivity computation."""

    T_c_K: float  # Critical temperature (K)
    T_c_measured_K: float  # Measured T_c for comparison
    Delta_0_meV: float  # Zero-temperature gap (meV)
    BCS_ratio: float  # 2Δ(0)/(k_B·T_c)
    C_jump_ratio: float  # ΔC/(γT_c)
    xi_0_nm: float  # BCS coherence length (nm)
    lambda_L_nm: float  # London penetration depth (nm)
    kappa_GL: float  # Ginzburg-Landau parameter λ/ξ
    sc_type: str  # TypeI / TypeII / Normal
    Theta_D_K: float  # Debye temperature used
    lambda_ep: float  # Electron-phonon coupling
    mu_star: float  # Coulomb pseudopotential
    rcft_attractor_depth: float  # Depth of RCFT condensation minimum
    rcft_tau_R_ratio: float  # τ_R / τ_R_critical
    omega_eff: float
    F_eff: float
    regime: str


# ── Constants ────────────────────────────────────────────────────
K_B_EV = 8.617333262e-5  # Boltzmann constant (eV/K)
K_B_J = 1.380649e-23
HBAR = 1.054571817e-34  # ℏ (J·s)
E_CHARGE = 1.602176634e-19  # elementary charge (C)
M_ELECTRON = 9.1093837015e-31  # electron mass (kg)
EPS_0 = 8.8541878128e-12  # vacuum permittivity

# BCS universal ratios (weak coupling)
BCS_RATIO_WEAK = 3.528  # 2Δ(0)/(k_B·T_c)
BCS_C_JUMP = 1.43  # ΔC/(γTc)
GAMMA_EULER = 0.5772156649  # Euler-Mascheroni constant

# Regime thresholds
THRESH_PRECISE = 0.10
THRESH_MODERATE = 0.30
THRESH_APPROXIMATE = 0.60

# ── Reference Data ───────────────────────────────────────────────
# T_c (K), Θ_D (K), λ_ep, μ*, sc_type, Δ(0) (meV)
REFERENCE_SC: dict[str, dict[str, float | str]] = {
    "Nb": {"T_c": 9.26, "Theta_D": 275, "lambda_ep": 1.04, "mu_star": 0.13, "type": "TypeII", "Delta_0": 1.55},
    "Pb": {"T_c": 7.19, "Theta_D": 105, "lambda_ep": 1.55, "mu_star": 0.13, "type": "TypeI", "Delta_0": 1.35},
    "Sn": {"T_c": 3.72, "Theta_D": 200, "lambda_ep": 0.72, "mu_star": 0.12, "type": "TypeI", "Delta_0": 0.58},
    "Al": {"T_c": 1.18, "Theta_D": 428, "lambda_ep": 0.43, "mu_star": 0.10, "type": "TypeI", "Delta_0": 0.18},
    "In": {"T_c": 3.41, "Theta_D": 112, "lambda_ep": 0.81, "mu_star": 0.13, "type": "TypeI", "Delta_0": 0.54},
    "V": {"T_c": 5.40, "Theta_D": 380, "lambda_ep": 0.80, "mu_star": 0.13, "type": "TypeII", "Delta_0": 0.80},
    "Ta": {"T_c": 4.47, "Theta_D": 258, "lambda_ep": 0.73, "mu_star": 0.13, "type": "TypeII", "Delta_0": 0.70},
    "Hg": {"T_c": 4.15, "Theta_D": 72, "lambda_ep": 1.60, "mu_star": 0.13, "type": "TypeI", "Delta_0": 0.82},
    "Zn": {"T_c": 0.85, "Theta_D": 327, "lambda_ep": 0.38, "mu_star": 0.10, "type": "TypeI", "Delta_0": 0.13},
    "MgB2": {"T_c": 39.0, "Theta_D": 750, "lambda_ep": 1.01, "mu_star": 0.10, "type": "TypeII", "Delta_0": 7.1},
    "NbTi": {"T_c": 10.0, "Theta_D": 300, "lambda_ep": 0.90, "mu_star": 0.13, "type": "TypeII", "Delta_0": 1.6},
    "Nb3Sn": {"T_c": 18.0, "Theta_D": 230, "lambda_ep": 1.80, "mu_star": 0.13, "type": "TypeII", "Delta_0": 3.4},
}


def _mcmillan_tc(theta_d: float, lambda_ep: float, mu_star: float) -> float:
    """McMillan equation for T_c.

    T_c = (Θ_D/1.45) · exp(−1.04(1+λ)/(λ − μ*(1+0.62λ)))
    """
    if lambda_ep <= 0:
        return 0.0
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if denom <= 0:
        return 0.0  # No superconductivity
    exponent = -1.04 * (1.0 + lambda_ep) / denom
    if exponent < -50:
        return 0.0  # Numerically zero
    return (theta_d / 1.45) * math.exp(exponent)


def _bcs_gap(T_c: float) -> float:
    """Weak-coupling BCS gap: Δ(0) = 1.764 · k_B · T_c (in meV)."""
    return 1.764 * K_B_EV * T_c * 1000  # eV → meV


def _coherence_length_nm(v_F_ms: float, Delta_J: float) -> float:
    """BCS coherence length: ξ₀ = ℏv_F/(πΔ(0)) in nm."""
    if Delta_J <= 0:
        return float("inf")
    return HBAR * v_F_ms / (math.pi * Delta_J) * 1e9  # m → nm


def _classify_regime(omega: float) -> SCRegime:
    if omega < THRESH_PRECISE:
        return SCRegime.PRECISE
    if omega < THRESH_MODERATE:
        return SCRegime.MODERATE
    if omega < THRESH_APPROXIMATE:
        return SCRegime.APPROXIMATE
    return SCRegime.ANOMALOUS


def compute_bcs_superconductivity(
    Theta_D_K: float,
    lambda_ep: float,
    *,
    T_K: float = 0.0,
    mu_star: float = 0.13,
    symbol: str = "",
    v_F_ms: float = 1.0e6,  # Typical Fermi velocity (m/s)
    n_density_m3: float = 8.5e28,  # Electron density (m⁻³), ~Cu
    T_c_measured_K: float | None = None,
) -> BCSResult:
    """Compute BCS superconducting properties.

    Parameters
    ----------
    Theta_D_K : float
        Debye temperature (K).
    lambda_ep : float
        Electron-phonon coupling constant.
    T_K : float
        Temperature for state evaluation (K).
    mu_star : float
        Coulomb pseudopotential (typically 0.10-0.15).
    symbol : str
        Material symbol for reference lookup.
    v_F_ms : float
        Fermi velocity (m/s) for coherence length.
    n_density_m3 : float
        Conduction electron density (m⁻³) for penetration depth.
    T_c_measured_K : float | None
        Measured T_c for ω_eff calculation.

    Returns
    -------
    BCSResult
    """
    # ── Reference lookup ──
    ref = REFERENCE_SC.get(symbol, {})
    if Theta_D_K <= 0 and ref:
        Theta_D_K = float(ref.get("Theta_D", 0.0))
    if lambda_ep <= 0 and ref:
        lambda_ep = float(ref.get("lambda_ep", 0.0))
    if T_c_measured_K is None:
        T_c_measured_K = float(ref.get("T_c", 0.0))

    # ── McMillan T_c ──
    T_c = _mcmillan_tc(Theta_D_K, lambda_ep, mu_star)

    # ── BCS gap ──
    Delta_0_meV = _bcs_gap(T_c)
    Delta_0_J = Delta_0_meV * 1e-3 * E_CHARGE  # meV → J

    # ── BCS ratio ──
    bcs_ratio = BCS_RATIO_WEAK  # 3.528 for weak coupling
    # Strong coupling correction: ratio increases with λ
    if lambda_ep > 1.0:
        bcs_ratio = BCS_RATIO_WEAK * (1.0 + 0.3 * (lambda_ep - 1.0))

    # ── Specific heat jump ──
    c_jump = BCS_C_JUMP  # 1.43 weak coupling
    if lambda_ep > 1.0:
        c_jump = BCS_C_JUMP * (1.0 + 0.4 * (lambda_ep - 1.0))

    # ── Coherence length ──
    xi_0 = _coherence_length_nm(v_F_ms, Delta_0_J)

    # ── London penetration depth ──
    # λ_L = √(m_e/(μ₀·n·e²))
    if n_density_m3 > 0:
        lambda_L_m = math.sqrt(M_ELECTRON / (MU_0 * n_density_m3 * E_CHARGE**2))
        lambda_L_nm = lambda_L_m * 1e9
    else:
        lambda_L_nm = 50.0  # typical default

    # ── GL parameter ──
    kappa_GL = lambda_L_nm / xi_0 if xi_0 > 0 and xi_0 != float("inf") else 0.0

    # ── Type classification ──
    if T_c <= 0 or T_c < T_K:
        sc_type = SCType.NORMAL
    elif kappa_GL < 1.0 / math.sqrt(2):
        sc_type = SCType.TYPE_I
    else:
        sc_type = SCType.TYPE_II

    # ── RCFT analogies ──
    # Attractor depth: deeper λ → more stable condensate
    rcft_depth = lambda_ep / (1.0 + lambda_ep) if lambda_ep > 0 else 0.0
    # τ_R ratio: T_c/Θ_D measures how deep the condensation is
    rcft_tau_R = T_c / Theta_D_K if Theta_D_K > 0 else 0.0

    # ── GCD mapping ──
    eps = 1e-12
    if T_c_measured_K and T_c_measured_K > eps:
        omega_eff = abs(T_c - T_c_measured_K) / T_c_measured_K
    elif T_c > eps:
        omega_eff = 0.0  # No reference, assume perfect
    else:
        omega_eff = 0.0

    omega_eff = min(1.0, omega_eff)
    F_eff = 1.0 - omega_eff
    regime = _classify_regime(omega_eff)

    return BCSResult(
        T_c_K=round(T_c, 6),
        T_c_measured_K=T_c_measured_K,
        Delta_0_meV=round(Delta_0_meV, 6),
        BCS_ratio=round(bcs_ratio, 4),
        C_jump_ratio=round(c_jump, 4),
        xi_0_nm=round(xi_0, 4),
        lambda_L_nm=round(lambda_L_nm, 4),
        kappa_GL=round(kappa_GL, 6),
        sc_type=sc_type.value,
        Theta_D_K=Theta_D_K,
        lambda_ep=lambda_ep,
        mu_star=mu_star,
        rcft_attractor_depth=round(rcft_depth, 6),
        rcft_tau_R_ratio=round(rcft_tau_R, 8),
        omega_eff=round(omega_eff, 8),
        F_eff=round(F_eff, 8),
        regime=regime.value,
    )


MU_0 = 1.25663706212e-6  # (defined at module level for _coherence_length_nm)


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== BCS Superconductivity Self-Test ===\n")

    for sym in ["Nb", "Pb", "Al", "Sn", "V", "MgB2", "Hg"]:
        r = compute_bcs_superconductivity(0.0, 0.0, symbol=sym)
        print(
            f"  {sym:6s}  T_c={r.T_c_K:7.3f}K  (ref {r.T_c_measured_K:.2f})  "
            f"Δ₀={r.Delta_0_meV:6.3f}meV  ξ₀={r.xi_0_nm:8.2f}nm  "
            f"κ={r.kappa_GL:.4f}  {r.sc_type:6s}  "
            f"ω={r.omega_eff:.4f}  {r.regime}"
        )
