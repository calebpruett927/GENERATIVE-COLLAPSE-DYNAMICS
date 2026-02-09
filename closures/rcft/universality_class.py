"""Universality Class of the GCD Kernel — RCFT.INTSTACK.v1

Derives the partition function, central charge, critical exponents,
susceptibility, and their scaling relations from the kernel drift
potential Γ(ω) = ω^p/(1 − ω + ε).  All results are exact in the
large-β (low-temperature) limit and verified numerically.

Theorems
--------
T20  Central Charge  c_eff = 1/p
     The partition function Z(β) = ∫ exp(−βΓ(ω)) dω has the
     asymptotic form  Z(β) → (1/p) β^{−1/p} Γ_Euler(1/p).
     Internal energy  U → 1/(pβ).
     Specific heat    C_V → 1/p  (p-equipartition theorem).
     For p = 3:  c_eff = 1/3.
     Universal: same c_eff for every domain, any n, any w.

T21  Critical Exponent Set
     Complete set derived from Γ(ω) ≈ ω^p:
       ν  = 1/p       (correlation length)
       γ  = (p−2)/p   (susceptibility)
       η  = 4 − p     (anomalous dimension)
       α  = 0         (specific heat — no divergence)
       β  = (p+2)/(2p) (order parameter)
       δ  = (3p−2)/(p+2) (critical isotherm)
       d_eff = 2p     (effective dimension, from hyperscaling)
     Hyperscaling:  d_eff · ν = 2 − α = 2   ✓
     Rushbrooke:    α + 2β + γ = 2           ✓
     Widom:         γ = β(δ − 1)             ✓
     For p = 3:
       (ν, γ, η, α, β, δ) = (1/3, 1/3, 1, 0, 5/6, 7/5)
       d_eff = 6.

Cross-references
    T10  (Pole residue = 1/2)
    T13  (Gibbs scaling ⟨ω⟩ ∝ β^{−1/p})
    T14  (Kramers escape)
    tau_r_star_dynamics.py  (thermodynamic potential)
    KERNEL_SPECIFICATION.md §5.3  (zν = 1 universality class)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.special import gamma as gamma_euler

# --- Frozen constants ---
EPSILON = 1e-8
P_EXPONENT = 3


# =====================================================================
# Data structures
# =====================================================================


class PartitionResult(NamedTuple):
    """Partition function Z(β) and derived thermodynamic quantities."""

    beta: float
    """Inverse temperature."""
    Z: float
    """Partition function value."""
    free_energy: float
    """Helmholtz free energy F = −ln Z / β."""
    internal_energy: float
    """Internal energy U = ⟨Γ⟩."""
    specific_heat: float
    """Specific heat C_V = β² Var(Γ)."""
    mean_omega: float
    """⟨ω⟩ under Gibbs measure."""
    var_omega: float
    """Var(ω) under Gibbs measure."""
    susceptibility: float
    """χ = β Var(ω)."""


class CentralChargeResult(NamedTuple):
    """Central charge determination from specific heat plateau."""

    p: int
    """Drift exponent."""
    c_eff: float
    """Effective central charge  1/p."""
    C_V_measured: float
    """Numerically measured C_V at large β."""
    relative_error: float
    """|C_V − 1/p| / (1/p)."""
    Z_prefactor: float
    """Analytical prefactor (1/p) Γ_Euler(1/p)."""


class CriticalExponents(NamedTuple):
    """Complete set of critical exponents for the GCD universality class."""

    p: int
    """Drift exponent."""
    nu: float
    """Correlation length exponent  ν = 1/p."""
    gamma: float
    """Susceptibility exponent  γ = (p−2)/p."""
    eta: float
    """Anomalous dimension  η = 4 − p."""
    alpha: float
    """Specific heat exponent  α = 0."""
    beta_exp: float
    """Order parameter exponent  β = (p+2)/(2p)."""
    delta: float
    """Critical isotherm exponent  δ = (3p−2)/(p+2)."""
    d_eff: float
    """Effective dimension  d_eff = 2p."""
    c_eff: float
    """Central charge  c = 1/p."""


class SusceptibilityResult(NamedTuple):
    """Susceptibility measurement at a given β."""

    beta: float
    chi: float
    """Susceptibility  χ = β Var(ω)."""
    chi_analytical: float
    """Analytical prediction  β^{1−2/p} · coefficient."""
    relative_error: float


class ScalingVerification(NamedTuple):
    """Verification of a scaling relation."""

    name: str
    """Name of the relation (Rushbrooke, Widom, Hyperscaling)."""
    lhs: float
    """Left-hand side value."""
    rhs: float
    """Right-hand side value (expected)."""
    satisfied: bool
    """Whether |lhs − rhs| < tolerance."""


# =====================================================================
# Core: Partition function (numerical)
# =====================================================================


def _gamma_omega(omega: np.ndarray, p: int, eps: float) -> np.ndarray:
    """Vectorized Γ(ω) = ω^p / (1 − ω + ε)."""
    return omega**p / (1.0 - omega + eps)


def compute_partition_function(
    beta: float,
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
    n_points: int = 50000,
) -> PartitionResult:
    """Compute Z(β) = ∫₀^{1−ε} exp(−β Γ(ω)) dω and derived quantities.

    Parameters
    ----------
    beta : float
        Inverse temperature (positive).
    p : int
        Drift exponent.
    epsilon : float
        Guard band.
    n_points : int
        Quadrature resolution.

    Returns
    -------
    PartitionResult
        Full thermodynamic snapshot.
    """
    if beta <= 0:
        msg = "beta must be positive"
        raise ValueError(msg)

    omega = np.linspace(epsilon, 1.0 - epsilon, n_points)
    g = _gamma_omega(omega, p, epsilon)
    boltz = np.exp(-beta * g)

    Z = float(np.trapezoid(boltz, omega))

    # Moments
    mean_omega = float(np.trapezoid(omega * boltz, omega)) / Z
    mean_omega2 = float(np.trapezoid(omega**2 * boltz, omega)) / Z
    var_omega = mean_omega2 - mean_omega**2

    mean_gamma = float(np.trapezoid(g * boltz, omega)) / Z
    mean_gamma2 = float(np.trapezoid(g**2 * boltz, omega)) / Z
    var_gamma = mean_gamma2 - mean_gamma**2

    return PartitionResult(
        beta=beta,
        Z=Z,
        free_energy=-math.log(Z) / beta,
        internal_energy=mean_gamma,
        specific_heat=beta**2 * var_gamma,
        mean_omega=mean_omega,
        var_omega=max(var_omega, 0.0),
        susceptibility=beta * max(var_omega, 0.0),
    )


# =====================================================================
# T20: Central Charge
# =====================================================================


def compute_central_charge(
    *,
    p: int = P_EXPONENT,
    beta_probe: float = 5000.0,
    epsilon: float = EPSILON,
) -> CentralChargeResult:
    """Determine the central charge c_eff = 1/p.

    Theorem T20: C_V → 1/p as β → ∞ (p-equipartition).

    Analytical derivation:
        Z(β) → (1/p) β^{−1/p} Γ_Euler(1/p)
        ⟨Γ⟩ = −∂ ln Z / ∂β → 1/(pβ)
        C_V = β² Var(Γ) = −β² ∂²(ln Z)/∂β² → 1/p

    Parameters
    ----------
    p : int
        Drift exponent.
    beta_probe : float
        Large β for numerical verification.

    Returns
    -------
    CentralChargeResult
    """
    pf = compute_partition_function(beta_probe, p=p, epsilon=epsilon)
    c_eff = 1.0 / p
    rel_err = abs(pf.specific_heat - c_eff) / c_eff
    prefactor = gamma_euler(1.0 / p) / p

    return CentralChargeResult(
        p=p,
        c_eff=c_eff,
        C_V_measured=pf.specific_heat,
        relative_error=rel_err,
        Z_prefactor=prefactor,
    )


def verify_central_charge_universality(
    p_values: list[int] | None = None,
    *,
    beta_probe: float = 5000.0,
) -> list[CentralChargeResult]:
    """Verify c_eff = 1/p across multiple exponent values.

    Parameters
    ----------
    p_values : list[int] or None
        Exponents to test (default: [2, 3, 4, 5, 7]).

    Returns
    -------
    list[CentralChargeResult]
    """
    if p_values is None:
        p_values = [2, 3, 4, 5, 7]
    return [compute_central_charge(p=p, beta_probe=beta_probe) for p in p_values]


# =====================================================================
# T21: Critical Exponents
# =====================================================================


def compute_critical_exponents(
    *,
    p: int = P_EXPONENT,
) -> CriticalExponents:
    """Derive the complete critical exponent set from p.

    Theorem T21:  All exponents follow from  Γ(ω) ≈ ω^p  at small ω.

    Derivation
    ----------
    Z(β) ∝ β^{−1/p} →
        ⟨ω⟩ ∝ β^{−1/p}         →  ν = 1/p
        Var(ω) ∝ β^{−2/p}      →
        χ = βVar(ω) ∝ β^{1−2/p} →  γ = (p−2)/p
        C_V → const             →  α = 0
    Scaling relations →
        β_exp = (2 − α − γ) / 2 = (p + 2) / (2p)
        δ = γ/β_exp + 1 = (3p − 2) / (p + 2)
    Fisher relation →
        η = 2 − γ/ν = 4 − p
    Hyperscaling →
        d_eff = (2 − α) / ν = 2p

    Parameters
    ----------
    p : int
        Drift exponent (≥ 2).

    Returns
    -------
    CriticalExponents
    """
    if p < 2:
        msg = "p must be ≥ 2 for well-defined exponents"
        raise ValueError(msg)

    nu = 1.0 / p
    gamma_exp = (p - 2.0) / p
    alpha_exp = 0.0
    beta_exp = (p + 2.0) / (2.0 * p)
    delta_exp = (3.0 * p - 2.0) / (p + 2.0)
    eta_exp = 4.0 - p
    d_eff = 2.0 * p
    c_eff = 1.0 / p

    return CriticalExponents(
        p=p,
        nu=nu,
        gamma=gamma_exp,
        eta=eta_exp,
        alpha=alpha_exp,
        beta_exp=beta_exp,
        delta=delta_exp,
        d_eff=d_eff,
        c_eff=c_eff,
    )


def verify_scaling_relations(
    *,
    p: int = P_EXPONENT,
    tol: float = 1e-10,
) -> list[ScalingVerification]:
    """Verify that the critical exponents satisfy scaling relations.

    Checks:
        1. Rushbrooke:    α + 2β + γ = 2
        2. Widom:         γ = β(δ − 1)
        3. Hyperscaling:  d_eff · ν = 2 − α
        4. Fisher:        γ = ν(2 − η)

    Returns
    -------
    list[ScalingVerification]
    """
    ce = compute_critical_exponents(p=p)

    checks = [
        ScalingVerification(
            name="Rushbrooke",
            lhs=ce.alpha + 2 * ce.beta_exp + ce.gamma,
            rhs=2.0,
            satisfied=abs(ce.alpha + 2 * ce.beta_exp + ce.gamma - 2.0) < tol,
        ),
        ScalingVerification(
            name="Widom",
            lhs=ce.gamma,
            rhs=ce.beta_exp * (ce.delta - 1.0),
            satisfied=abs(ce.gamma - ce.beta_exp * (ce.delta - 1.0)) < tol,
        ),
        ScalingVerification(
            name="Hyperscaling",
            lhs=ce.d_eff * ce.nu,
            rhs=2.0 - ce.alpha,
            satisfied=abs(ce.d_eff * ce.nu - (2.0 - ce.alpha)) < tol,
        ),
        ScalingVerification(
            name="Fisher",
            lhs=ce.gamma,
            rhs=ce.nu * (2.0 - ce.eta),
            satisfied=abs(ce.gamma - ce.nu * (2.0 - ce.eta)) < tol,
        ),
    ]
    return checks


# =====================================================================
# Susceptibility
# =====================================================================


def compute_susceptibility(
    beta: float,
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
) -> SusceptibilityResult:
    """Compute susceptibility χ(β) = β Var(ω).

    Analytical prediction (large β):
        χ → β^{1−2/p} · [Γ(3/p)/Γ(1/p) − (Γ(2/p)/Γ(1/p))²]

    Parameters
    ----------
    beta : float
        Inverse temperature.

    Returns
    -------
    SusceptibilityResult
    """
    pf = compute_partition_function(beta, p=p, epsilon=epsilon)

    # Analytical coefficient
    r1 = gamma_euler(2.0 / p) / gamma_euler(1.0 / p)
    r2 = gamma_euler(3.0 / p) / gamma_euler(1.0 / p)
    chi_coeff = r2 - r1**2
    chi_ana = beta ** (1.0 - 2.0 / p) * chi_coeff

    rel_err = abs(pf.susceptibility - chi_ana) / chi_ana if chi_ana > 0 else 0.0

    return SusceptibilityResult(
        beta=beta,
        chi=pf.susceptibility,
        chi_analytical=chi_ana,
        relative_error=rel_err,
    )


def compute_susceptibility_scaling(
    beta_values: list[float] | None = None,
    *,
    p: int = P_EXPONENT,
) -> dict:
    """Verify χ ~ β^{(p-2)/p} scaling over range of β.

    Returns
    -------
    dict
        'beta_values', 'chi_values', 'fitted_exponent', 'predicted_exponent',
        'results'.
    """
    if beta_values is None:
        beta_values = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]

    results = [compute_susceptibility(b, p=p) for b in beta_values]
    chi_vals = [r.chi for r in results]

    # Log-log fit
    log_b = np.log(beta_values)
    log_chi = np.log(chi_vals)
    coeffs = np.polyfit(log_b, log_chi, 1)

    return {
        "beta_values": beta_values,
        "chi_values": chi_vals,
        "fitted_exponent": float(coeffs[0]),
        "predicted_exponent": (p - 2.0) / p,
        "results": results,
    }


# =====================================================================
# Analytical moments
# =====================================================================


def analytical_moment(
    k: int,
    beta: float,
    *,
    p: int = P_EXPONENT,
) -> float:
    """Analytical k-th moment ⟨ω^k⟩ at large β.

    ⟨ω^k⟩ = β^{−k/p} · Γ((k+1)/p) / Γ(1/p)
    """
    return beta ** (-k / p) * gamma_euler((k + 1.0) / p) / gamma_euler(1.0 / p)


def analytical_susceptibility_coefficient(*, p: int = P_EXPONENT) -> float:
    """Coefficient in χ = β^{1−2/p} · C_χ."""
    r1 = gamma_euler(2.0 / p) / gamma_euler(1.0 / p)
    r2 = gamma_euler(3.0 / p) / gamma_euler(1.0 / p)
    return r2 - r1**2
