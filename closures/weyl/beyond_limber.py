"""
Beyond-Limber Correction for Low Multipoles

Implements the exact (non-Limber) computation for ℓ < 200 from:
"Model-independent test of gravity with a network of galaxy surveys"
(Nature Communications 15:9295, 2024)

Mathematical Basis (Eq. 7):
    For ℓ < 200, they split P_nl = P_lin + (P_nl - P_lin), using Limber
    only for the nonlinear residual, and computing the linear term exactly:

    C_ℓ^{ΔΔ}(z_i, z_j)|_lin = (2/π) ∫ dχ₁ n_i(χ₁)(1+z(χ₁))H(χ₁)ĥb_i(χ₁)
                               × ∫ dχ₂ n_j(χ₂)(1+z(χ₂))H(χ₂)ĥb_j(χ₂)
                               × ∫_0^∞ dk k² [P_lin(k,z*)/σ8²(z*)] j_ℓ(kχ₁) j_ℓ(kχ₂)

Where j_ℓ(x) are spherical Bessel functions.

UMCP Integration:
    - Regime-dependent closure: switches at ℓ = 200
    - Beyond-Limber = higher precision, analogous to strict-mode validation
    - Spherical Bessel functions = harmonic expansion of return structure

Reference: Eq. 7 of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import special


class BeyondLimberRegime(str, Enum):
    """Beyond-Limber computation regime."""

    REQUIRED = "Beyond_Limber_required"  # ℓ < 100
    RECOMMENDED = "Beyond_Limber_recommended"  # 100 ≤ ℓ < 200
    UNNECESSARY = "Limber_sufficient"  # ℓ ≥ 200


class BeyondLimberResult(NamedTuple):
    """Result of beyond-Limber computation."""

    C_ell_lin: float  # Linear C_ℓ (exact)
    C_ell_nl_residual: float  # Nonlinear residual (Limber)
    C_ell_total: float  # Total C_ℓ
    ell: float  # Multipole
    regime: str  # BeyondLimberRegime
    n_bessel_evals: int  # Number of Bessel function evaluations


@dataclass
class BeyondLimberConfig:
    """Configuration for beyond-Limber computation."""

    # Integration limits
    chi_min: float = 1.0  # Mpc/h
    chi_max: float = 5000.0  # Mpc/h
    k_min: float = 1e-4  # h/Mpc
    k_max: float = 1.0  # h/Mpc (linear regime)

    # Discretization
    n_chi_points: int = 50
    n_k_points: int = 100

    # Regime thresholds
    ell_beyond_required: int = 100
    ell_beyond_recommended: int = 200

    # Numerical parameters
    epsilon: float = 1e-10
    rel_tol: float = 1e-5


def spherical_bessel_j(ell: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute spherical Bessel function j_ℓ(x).

    Uses scipy's spherical_jn for accurate computation.

    Args:
        ell: Order ℓ (non-negative integer)
        x: Argument array

    Returns:
        j_ℓ(x) array
    """
    # Handle x=0 case
    x_safe = np.where(x == 0, 1e-10, x)
    return special.spherical_jn(ell, x_safe)


def classify_beyond_limber_regime(ell: float, config: BeyondLimberConfig | None = None) -> str:
    """Classify whether beyond-Limber correction is needed."""
    if config is None:
        config = BeyondLimberConfig()

    if ell < config.ell_beyond_required:
        return BeyondLimberRegime.REQUIRED.value
    elif ell < config.ell_beyond_recommended:
        return BeyondLimberRegime.RECOMMENDED.value
    else:
        return BeyondLimberRegime.UNNECESSARY.value


def compute_bessel_integral(
    ell: int,
    chi_1: float,
    chi_2: float,
    k_values: NDArray[np.floating],
    P_lin_values: NDArray[np.floating],
) -> float:
    """
    Compute the Bessel integral:
        ∫ dk k² P_lin(k) j_ℓ(kχ₁) j_ℓ(kχ₂)

    Args:
        ell: Multipole (integer for spherical Bessel)
        chi_1, chi_2: Comoving distances
        k_values: Wavenumber array
        P_lin_values: P_lin(k,z*)/σ8²(z*) array

    Returns:
        Integral value
    """
    # Compute j_ℓ(kχ) for both distances
    x_1 = k_values * chi_1
    x_2 = k_values * chi_2

    j_ell_1 = spherical_bessel_j(ell, x_1)
    j_ell_2 = spherical_bessel_j(ell, x_2)

    # Integrand: k² P(k) j_ℓ(kχ₁) j_ℓ(kχ₂)
    integrand = k_values**2 * P_lin_values * j_ell_1 * j_ell_2

    # Trapezoidal integration (use trapezoid, trapz is deprecated)
    return float(np.trapezoid(integrand, k_values))


def compute_beyond_limber_c_ell(
    ell: int,
    n_i: Callable[[float], float],
    n_j: Callable[[float], float],
    H_chi: Callable[[float], float],
    z_of_chi: Callable[[float], float],
    hb_i: Callable[[float], float],
    hb_j: Callable[[float], float],
    P_lin_over_sigma8sq: Callable[[float], float],
    config: BeyondLimberConfig | None = None,
) -> BeyondLimberResult:
    """
    Compute exact (beyond-Limber) linear C_ℓ for low multipoles.

    Implements Eq. 7 from Nature Comms 15:9295 (2024):
        C_ℓ^{ΔΔ}|_lin = (2/π) ∫∫ dχ₁ dχ₂ W_i(χ₁) W_j(χ₂)
                        × ∫ dk k² P_lin(k) j_ℓ(kχ₁) j_ℓ(kχ₂)

    Where W_i(χ) = n_i(χ) (1+z) H(χ) ĥb_i(χ)

    Args:
        ell: Multipole (must be integer for spherical Bessel)
        n_i, n_j: Redshift distributions as functions of χ
        H_chi: Conformal Hubble as function of χ
        z_of_chi: Redshift as function of χ
        hb_i, hb_j: Bias-growth combinations
        P_lin_over_sigma8sq: Linear power spectrum / σ8²
        config: Optional configuration

    Returns:
        BeyondLimberResult with exact C_ℓ
    """
    if config is None:
        config = BeyondLimberConfig()

    regime = classify_beyond_limber_regime(ell, config)

    # If Limber is sufficient, return early with flag
    if regime == BeyondLimberRegime.UNNECESSARY.value:
        return BeyondLimberResult(
            C_ell_lin=0.0,
            C_ell_nl_residual=0.0,
            C_ell_total=0.0,
            ell=float(ell),
            regime=regime,
            n_bessel_evals=0,
        )

    # Set up grids
    chi_grid = np.linspace(config.chi_min, config.chi_max, config.n_chi_points)
    k_grid = np.logspace(np.log10(config.k_min), np.log10(config.k_max), config.n_k_points)

    # Precompute P_lin
    P_lin_grid = np.array([P_lin_over_sigma8sq(k) for k in k_grid])

    # Compute window functions W_i, W_j on chi grid
    def W_i(chi: float) -> float:
        z = z_of_chi(chi)
        return n_i(chi) * (1 + z) * H_chi(chi) * hb_i(chi)

    def W_j(chi: float) -> float:
        z = z_of_chi(chi)
        return n_j(chi) * (1 + z) * H_chi(chi) * hb_j(chi)

    W_i_grid = np.array([W_i(chi) for chi in chi_grid])
    W_j_grid = np.array([W_j(chi) for chi in chi_grid])

    # Double integral over χ₁, χ₂
    n_bessel_evals = 0

    def outer_integrand(i1: int, i2: int) -> float:
        nonlocal n_bessel_evals
        chi_1 = chi_grid[i1]
        chi_2 = chi_grid[i2]

        # Bessel integral
        bessel_int = compute_bessel_integral(ell, chi_1, chi_2, k_grid, P_lin_grid)
        n_bessel_evals += 2 * len(k_grid)

        return W_i_grid[i1] * W_j_grid[i2] * bessel_int

    # Double sum approximation (discrete integration)
    dchi = chi_grid[1] - chi_grid[0]
    C_ell_lin = 0.0

    for i1 in range(len(chi_grid)):
        for i2 in range(len(chi_grid)):
            C_ell_lin += outer_integrand(i1, i2) * dchi**2

    # Prefactor 2/π
    C_ell_lin *= 2.0 / np.pi

    return BeyondLimberResult(
        C_ell_lin=float(C_ell_lin),
        C_ell_nl_residual=0.0,  # Would need nonlinear P(k) to compute
        C_ell_total=float(C_ell_lin),
        ell=float(ell),
        regime=regime,
        n_bessel_evals=n_bessel_evals,
    )


def compute_hybrid_c_ell(
    ell: int,
    c_ell_limber: float,
    c_ell_beyond_limber_lin: float,
    c_ell_limber_lin: float,
) -> float:
    """
    Compute hybrid C_ℓ using beyond-Limber for linear, Limber for nonlinear.

    The hybrid approach from Eq. 7:
        C_ℓ^{total} = C_ℓ^{lin}|_{exact} + (C_ℓ^{nl}|_{Limber} - C_ℓ^{lin}|_{Limber})

    Args:
        ell: Multipole
        c_ell_limber: Full C_ℓ from Limber approximation
        c_ell_beyond_limber_lin: Linear C_ℓ from exact computation
        c_ell_limber_lin: Linear C_ℓ from Limber approximation

    Returns:
        Hybrid C_ℓ
    """
    # Nonlinear residual from Limber
    nl_residual = c_ell_limber - c_ell_limber_lin

    # Hybrid: exact linear + Limber nonlinear residual
    return c_ell_beyond_limber_lin + nl_residual


# =============================================================================
# UMCP INTEGRATION: Precision Regime Interpretation
# =============================================================================


def beyond_limber_as_strict_mode(ell: float) -> dict:
    """
    Interpret beyond-Limber as UMCP strict mode analog.

    UMCP Mapping:
        - Limber (ℓ ≥ 200) = default validation (faster, approximate)
        - Beyond-Limber (ℓ < 200) = strict validation (exact, slower)

    This is analogous to UMCP's --strict flag: publication-quality
    requires the exact computation at low multipoles.

    Args:
        ell: Multipole

    Returns:
        Interpretation dictionary
    """
    regime = classify_beyond_limber_regime(ell)

    return {
        "ell": ell,
        "regime": regime,
        "umcp_analog": ("strict_mode" if regime != BeyondLimberRegime.UNNECESSARY.value else "default_mode"),
        "interpretation": (
            f"At ℓ={ell}, {'exact computation required (strict mode)' if regime == BeyondLimberRegime.REQUIRED.value else 'Limber approximation is sufficient (default mode)' if regime == BeyondLimberRegime.UNNECESSARY.value else 'exact computation recommended'}"
        ),
        "computation_cost": (
            "HIGH - O(n² × n_k) Bessel evaluations"
            if regime != BeyondLimberRegime.UNNECESSARY.value
            else "LOW - single integral"
        ),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Beyond-Limber Self-Test ===\n")

    # Test spherical Bessel functions
    x_test = np.array([0.1, 1.0, 10.0])
    for ell in [0, 2, 10]:
        j_ell = spherical_bessel_j(ell, x_test)
        print(f"j_{ell}([0.1, 1, 10]) = {j_ell}")

    # Test regime classification
    print("\nRegime classification:")
    for ell in [50, 150, 250]:
        regime = classify_beyond_limber_regime(ell)
        interp = beyond_limber_as_strict_mode(ell)
        print(f"  ℓ = {ell}: {regime} ({interp['umcp_analog']})")

    # Test Bessel integral (simplified)
    k_test = np.linspace(0.001, 0.1, 50)
    P_test = k_test ** (-2)  # Simple power law

    result = compute_bessel_integral(
        ell=10,
        chi_1=100.0,
        chi_2=100.0,
        k_values=k_test,
        P_lin_values=P_test,
    )
    print(f"\nBessel integral test (ℓ=10, χ=100): {result:.6e}")

    print("\n✓ Self-test complete")
