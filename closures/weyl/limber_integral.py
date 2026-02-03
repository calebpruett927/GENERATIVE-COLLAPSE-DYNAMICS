"""
Limber Integral Computation for Angular Power Spectra

Implements the Limber approximation for galaxy-galaxy lensing C_ℓ^{Δκ}
and galaxy clustering C_ℓ^{ΔΔ} from:
"Model-independent test of gravity with a network of galaxy surveys"
(Nature Communications 15:9295, 2024)

Mathematical Basis:

Galaxy-Galaxy Lensing (Eq. 3):
    C_ℓ^{Δκ}(z_i, z_j) = (3/2) ∫ dz n_i(z) H²(z) ĥb_i(z) ĥJ(z) B(k_ℓ,χ)
                          × [P_lin^{δδ}(k_ℓ, z*)/σ8²(z*)]
                          × ∫ dz' n_j(z') [χ'(z')-χ(z)]/[χ(z)χ'(z')]

Galaxy Clustering (Eq. 6, ℓ ≥ 200):
    C_ℓ^{ΔΔ}(z_i, z_j) = ∫ dz n_i(z) n_j(z) [H(z)(1+z)/χ²(z)]
                          × ĥb_i(z) ĥb_j(z) B(k_ℓ,χ)
                          × [P_lin^{δδ}(k_ℓ,z*)/σ8²(z*)]

Where:
    k_ℓ = (ℓ + 1/2)/χ

UMCP Integration:
    - Tier-1 closure for WEYL.INTSTACK.v1
    - C_ℓ → invariant projection (Layer 2 in infrastructure geometry)
    - ĥJ, ĥb are the primary "coordinates" being projected

Reference: Eq. 3, 6 of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate


class LimberRegime(str, Enum):
    """Limber approximation validity regime."""

    VALID = "Limber_valid"  # ℓ ≥ 200
    MARGINAL = "Limber_marginal"  # 100 ≤ ℓ < 200
    INVALID = "Limber_invalid"  # ℓ < 100, need beyond-Limber
    UNDEFINED = "Undefined"


class GGLResult(NamedTuple):
    """Galaxy-Galaxy Lensing C_ℓ result."""

    C_ell: float  # C_ℓ^{Δκ}(z_i, z_j)
    ell: float  # Multipole
    z_lens: float  # Effective lens redshift
    z_source: float  # Effective source redshift
    regime: str  # LimberRegime
    integrand_max: float  # Maximum of integrand (diagnostic)


class GalaxyClusteringResult(NamedTuple):
    """Galaxy Clustering C_ℓ result."""

    C_ell: float  # C_ℓ^{ΔΔ}(z_i, z_j)
    ell: float  # Multipole
    z_eff: float  # Effective redshift
    regime: str  # LimberRegime
    integrand_max: float  # Maximum of integrand


@dataclass
class LimberConfig:
    """Configuration for Limber integral computation."""

    # Integration limits
    z_min: float = 0.01
    z_max: float = 3.0
    n_z_points: int = 100

    # Limber validity thresholds
    ell_limber_valid: int = 200
    ell_limber_marginal: int = 100

    # Numerical parameters
    epsilon: float = 1e-10
    rel_tol: float = 1e-6
    abs_tol: float = 1e-10


def k_ell(ell: float, chi: float, epsilon: float = 1e-10) -> float:
    """
    Compute k_ℓ = (ℓ + 1/2)/χ.

    Args:
        ell: Multipole ℓ
        chi: Comoving distance χ(z) in Mpc/h
        epsilon: Safety floor

    Returns:
        Wavenumber k in h/Mpc
    """
    return (ell + 0.5) / max(chi, epsilon)


def classify_limber_regime(ell: float, config: LimberConfig | None = None) -> str:
    """Classify Limber approximation validity based on ℓ."""
    if config is None:
        config = LimberConfig()

    if ell >= config.ell_limber_valid:
        return LimberRegime.VALID.value
    elif ell >= config.ell_limber_marginal:
        return LimberRegime.MARGINAL.value
    else:
        return LimberRegime.INVALID.value


def compute_ggl_c_ell(
    ell: float,
    z_lens_eff: float,
    z_source_eff: float,
    n_lens: Callable[[float], float],
    n_source: Callable[[float], float],
    H_z: Callable[[float], float],
    chi_z: Callable[[float], float],
    hJ: Callable[[float], float],
    hb: Callable[[float], float],
    B_boost: Callable[[float, float], float],
    P_lin_over_sigma8sq: Callable[[float], float],
    config: LimberConfig | None = None,
) -> GGLResult:
    """
    Compute galaxy-galaxy lensing angular power spectrum C_ℓ^{Δκ}.

    Implements Eq. 3 from Nature Comms 15:9295 (2024):
        C_ℓ^{Δκ}(z_i, z_j) = (3/2) ∫ dz n_i(z) H²(z) ĥb_i(z) ĥJ(z) B(k_ℓ,χ)
                              × [P_lin(k_ℓ,z*)/σ8²(z*)]
                              × ∫ dz' n_j(z') [χ'-χ]/[χ·χ']

    Args:
        ell: Multipole ℓ
        z_lens_eff: Effective lens redshift (for reporting)
        z_source_eff: Effective source redshift (for reporting)
        n_lens: Lens redshift distribution n_i(z)
        n_source: Source redshift distribution n_j(z)
        H_z: Conformal Hubble H(z) in km/s/Mpc
        chi_z: Comoving distance χ(z) in Mpc/h
        hJ: Weyl evolution proxy ĥJ(z)
        hb: Bias-growth combination ĥb(z)
        B_boost: Nonlinear boost B(k, z)
        P_lin_over_sigma8sq: P_lin(k,z*)/σ8²(z*) at anchor
        config: Optional configuration

    Returns:
        GGLResult with C_ℓ and diagnostics
    """
    if config is None:
        config = LimberConfig()

    eps = config.epsilon
    regime = classify_limber_regime(ell, config)

    # If Limber is invalid, return with warning regime
    if regime == LimberRegime.INVALID.value:
        return GGLResult(
            C_ell=0.0,
            ell=ell,
            z_lens=z_lens_eff,
            z_source=z_source_eff,
            regime=regime,
            integrand_max=0.0,
        )

    # Track integrand maximum for diagnostics
    integrand_values = []

    def integrand(z: float) -> float:
        """Outer integral over lens redshift."""
        chi = chi_z(z)
        if chi < eps:
            return 0.0

        k = k_ell(ell, chi, eps)

        # Lensing kernel (inner integral over sources)
        def source_kernel(z_prime: float) -> float:
            chi_prime = chi_z(z_prime)
            if chi_prime <= chi + eps:
                return 0.0
            return n_source(z_prime) * (chi_prime - chi) / (chi * chi_prime + eps)

        # Integrate source kernel from z to z_max
        lensing_kernel, _ = integrate.quad(source_kernel, z, config.z_max, limit=50, epsrel=config.rel_tol)

        # Full integrand
        result = n_lens(z) * H_z(z) ** 2 * hb(z) * hJ(z) * B_boost(k, z) * P_lin_over_sigma8sq(k) * lensing_kernel

        integrand_values.append(abs(result))
        return result

    # Outer integral
    C_ell_raw, _ = integrate.quad(integrand, config.z_min, config.z_max, limit=100, epsrel=config.rel_tol)

    # Prefactor 3/2
    C_ell = 1.5 * C_ell_raw

    integrand_max = max(integrand_values) if integrand_values else 0.0

    return GGLResult(
        C_ell=float(C_ell),
        ell=float(ell),
        z_lens=z_lens_eff,
        z_source=z_source_eff,
        regime=regime,
        integrand_max=float(integrand_max),
    )


def compute_clustering_c_ell(
    ell: float,
    z_eff: float,
    n_i: Callable[[float], float],
    n_j: Callable[[float], float],
    H_z: Callable[[float], float],
    chi_z: Callable[[float], float],
    hb_i: Callable[[float], float],
    hb_j: Callable[[float], float],
    B_boost: Callable[[float, float], float],
    P_lin_over_sigma8sq: Callable[[float], float],
    config: LimberConfig | None = None,
) -> GalaxyClusteringResult:
    """
    Compute galaxy clustering angular power spectrum C_ℓ^{ΔΔ}.

    Implements Eq. 6 from Nature Comms 15:9295 (2024):
        C_ℓ^{ΔΔ}(z_i, z_j) = ∫ dz n_i(z) n_j(z) [H(z)(1+z)/χ²(z)]
                              × ĥb_i(z) ĥb_j(z) B(k_ℓ,χ)
                              × [P_lin(k_ℓ,z*)/σ8²(z*)]

    Args:
        ell: Multipole ℓ
        z_eff: Effective redshift (for reporting)
        n_i, n_j: Redshift distributions
        H_z: Conformal Hubble H(z)
        chi_z: Comoving distance χ(z)
        hb_i, hb_j: Bias-growth combinations
        B_boost: Nonlinear boost B(k, z)
        P_lin_over_sigma8sq: P_lin(k,z*)/σ8²(z*)
        config: Optional configuration

    Returns:
        GalaxyClusteringResult with C_ℓ and diagnostics
    """
    if config is None:
        config = LimberConfig()

    eps = config.epsilon
    regime = classify_limber_regime(ell, config)

    if regime == LimberRegime.INVALID.value:
        return GalaxyClusteringResult(
            C_ell=0.0,
            ell=ell,
            z_eff=z_eff,
            regime=regime,
            integrand_max=0.0,
        )

    integrand_values = []

    def integrand(z: float) -> float:
        chi = chi_z(z)
        if chi < eps:
            return 0.0

        k = k_ell(ell, chi, eps)

        result = (
            n_i(z)
            * n_j(z)
            * H_z(z)
            * (1 + z)
            / (chi**2 + eps)
            * hb_i(z)
            * hb_j(z)
            * B_boost(k, z)
            * P_lin_over_sigma8sq(k)
        )

        integrand_values.append(abs(result))
        return result

    C_ell, _ = integrate.quad(integrand, config.z_min, config.z_max, limit=100, epsrel=config.rel_tol)

    integrand_max = max(integrand_values) if integrand_values else 0.0

    return GalaxyClusteringResult(
        C_ell=float(C_ell),
        ell=float(ell),
        z_eff=z_eff,
        regime=regime,
        integrand_max=float(integrand_max),
    )


def compute_c_ell_array(
    ell_values: NDArray[np.floating],
    compute_func: Callable[[float], float],
) -> NDArray[np.floating]:
    """
    Vectorized C_ℓ computation over array of multipoles.

    Args:
        ell_values: Array of ℓ values
        compute_func: Function that computes C_ℓ for single ℓ

    Returns:
        Array of C_ℓ values
    """
    return np.array([compute_func(ell) for ell in ell_values])


# =============================================================================
# UMCP INTEGRATION: Projection Interpretation
# =============================================================================


def limber_as_projection(
    hJ_values: NDArray[np.floating],
    hb_values: NDArray[np.floating],
    z_bins: NDArray[np.floating],
) -> dict:
    """
    Interpret Limber integrals as UMCP projections.

    UMCP Mapping:
        - ĥJ, ĥb are coordinates in Ψ-space
        - C_ℓ is a projection onto multipole basis
        - This is Layer 2 (invariant coordinates) in infrastructure geometry

    Args:
        hJ_values: ĥJ values per redshift bin
        hb_values: ĥb values per redshift bin
        z_bins: Effective redshift per bin

    Returns:
        Projection interpretation dictionary
    """
    # Compute simple statistics
    hJ_mean = float(np.mean(hJ_values))
    hJ_std = float(np.std(hJ_values))
    hb_mean = float(np.mean(hb_values))

    # Weighted "fidelity" analog
    F_analog = hJ_mean  # ĥJ ≈ 0.34 in DES → F ≈ 0.34
    omega_analog = 1.0 - F_analog

    # Curvature analog (variation across bins)
    C_analog = hJ_std / 0.5 if hJ_std < 0.5 else 1.0

    return {
        "hJ_mean": hJ_mean,
        "hJ_std": hJ_std,
        "hb_mean": hb_mean,
        "F_analog": F_analog,
        "omega_analog": omega_analog,
        "C_analog": C_analog,
        "n_bins": len(z_bins),
        "interpretation": (
            "The Limber integral projects {ĥJ, ĥb} coordinates "
            "onto angular multipole basis C_ℓ, analogous to "
            "UMCP's Layer-2 invariant projection."
        ),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Limber Integral Self-Test ===\n")

    # Test regime classification
    for ell in [50, 150, 250]:
        regime = classify_limber_regime(ell)
        print(f"ℓ = {ell}: {regime}")

    # Test k_ℓ computation
    chi_test = 1000.0  # Mpc/h
    for ell in [100, 200, 500]:
        k = k_ell(ell, chi_test)
        print(f"k(ℓ={ell}, χ={chi_test}) = {k:.4f} h/Mpc")

    # Test projection interpretation with DES Y3 data
    hJ_des = np.array([0.326, 0.332, 0.387, 0.354])
    hb_des = np.array([0.5, 0.6, 0.7, 0.8])  # Approximate
    z_bins = np.array([0.295, 0.467, 0.626, 0.771])

    proj = limber_as_projection(hJ_des, hb_des, z_bins)
    print(f"\nDES Y3 Projection Analysis:")
    print(f"  ĥJ mean = {proj['hJ_mean']:.3f}")
    print(f"  ĥJ std = {proj['hJ_std']:.3f}")
    print(f"  F_analog = {proj['F_analog']:.3f}")
    print(f"  ω_analog = {proj['omega_analog']:.3f}")
    print(f"  C_analog = {proj['C_analog']:.3f}")

    print("\n✓ Self-test complete")
