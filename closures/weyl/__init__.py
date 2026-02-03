"""
WEYL Closures Module

Provides closures for the Weyl evolution / modified gravity framework
based on Nature Communications 15:9295 (2024).

This module implements the mathematical framework for testing modified
gravity using weak gravitational lensing, mapped to UMCP's contract
and closure discipline.

Available Closures:
    - weyl_transfer: Weyl transfer function T_{Ψ_W}(k,z)
    - limber_integral: Limber approximation for C_ℓ (ℓ ≥ 200)
    - beyond_limber: Exact computation for low ℓ (ℓ < 200)
    - sigma_evolution: Σ(z) parametrization and fitting
    - boost_factor: Nonlinear boost B(k,z)
    - cosmology_background: H(z), χ(z), D₁(z), σ8(z)

UMCP Integration:
    - Contract: WEYL.INTSTACK.v1 (extends GCD.INTSTACK.v1)
    - Canon anchors: canon/weyl_anchors.yaml
    - Reference casepack: casepacks/weyl_des_y3/

Core Axiom Realization:
    AX-W0 (Weyl Axiom): "Reference anchor defines deviation"

    The high-z anchor z* where GR holds defines the meaning of
    gravitational deviation at lower redshift. This is the
    cosmological realization of AX-1 ("Boundary defines interior").
"""

from closures.weyl.beyond_limber import (
    BeyondLimberConfig,
    BeyondLimberRegime,
    BeyondLimberResult,
    beyond_limber_as_strict_mode,
    compute_beyond_limber_c_ell,
    compute_hybrid_c_ell,
    spherical_bessel_j,
)
from closures.weyl.boost_factor import (
    BoostConfig,
    BoostResult,
    NonlinearRegime,
    boost_as_curvature,
    classify_scale_regime,
    compute_boost_array,
    compute_boost_factor,
    halofit_boost,
)
from closures.weyl.cosmology_background import (
    PLANCK_2018,
    BackgroundResult,
    CosmologyParams,
    D1_of_z,
    H_of_z,
    Omega_Lambda_of_z,
    Omega_m_of_z,
    chi_of_z,
    compute_background,
    compute_background_array,
    compute_des_y3_background,
    cosmology_as_embedding,
    sigma8_of_z,
)
from closures.weyl.limber_integral import (
    GalaxyClusteringResult,
    GGLResult,
    LimberConfig,
    LimberRegime,
    classify_limber_regime,
    compute_clustering_c_ell,
    compute_ggl_c_ell,
    k_ell,
    limber_as_projection,
)
from closures.weyl.sigma_evolution import (
    DES_Y3_DATA,
    GzModel,
    Sigma_to_UMCP_invariants,
    SigmaConfig,
    SigmaFitResult,
    SigmaRegime,
    SigmaResult,
    compute_Sigma,
    compute_Sigma_from_hJ,
    fit_Sigma_0,
    g_z_constant,
    g_z_exponential,
    g_z_standard,
)
from closures.weyl.weyl_transfer import (
    WeylRegime,
    WeylTransferConfig,
    WeylTransferResult,
    compute_weyl_transfer,
    compute_weyl_transfer_array,
    weyl_return_domain,
)

__all__ = [
    "DES_Y3_DATA",
    "PLANCK_2018",
    "BackgroundResult",
    "BeyondLimberConfig",
    # beyond_limber
    "BeyondLimberRegime",
    "BeyondLimberResult",
    "BoostConfig",
    "BoostResult",
    # cosmology_background
    "CosmologyParams",
    "D1_of_z",
    "GGLResult",
    "GalaxyClusteringResult",
    "GzModel",
    "H_of_z",
    "LimberConfig",
    # limber_integral
    "LimberRegime",
    # boost_factor
    "NonlinearRegime",
    "Omega_Lambda_of_z",
    "Omega_m_of_z",
    "SigmaConfig",
    "SigmaFitResult",
    # sigma_evolution
    "SigmaRegime",
    "SigmaResult",
    "Sigma_to_UMCP_invariants",
    # weyl_transfer
    "WeylRegime",
    "WeylTransferConfig",
    "WeylTransferResult",
    "beyond_limber_as_strict_mode",
    "boost_as_curvature",
    "chi_of_z",
    "classify_limber_regime",
    "classify_scale_regime",
    "compute_Sigma",
    "compute_Sigma_from_hJ",
    "compute_background",
    "compute_background_array",
    "compute_beyond_limber_c_ell",
    "compute_boost_array",
    "compute_boost_factor",
    "compute_clustering_c_ell",
    "compute_des_y3_background",
    "compute_ggl_c_ell",
    "compute_hybrid_c_ell",
    "compute_weyl_transfer",
    "compute_weyl_transfer_array",
    "cosmology_as_embedding",
    "fit_Sigma_0",
    "g_z_constant",
    "g_z_exponential",
    "g_z_standard",
    "halofit_boost",
    "k_ell",
    "limber_as_projection",
    "sigma8_of_z",
    "spherical_bessel_j",
    "weyl_return_domain",
]
