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

from closures.weyl.weyl_transfer import (
    WeylRegime,
    WeylTransferConfig,
    WeylTransferResult,
    compute_weyl_transfer,
    compute_weyl_transfer_array,
    weyl_return_domain,
)

from closures.weyl.limber_integral import (
    LimberRegime,
    LimberConfig,
    GGLResult,
    GalaxyClusteringResult,
    compute_ggl_c_ell,
    compute_clustering_c_ell,
    k_ell,
    classify_limber_regime,
    limber_as_projection,
)

from closures.weyl.beyond_limber import (
    BeyondLimberRegime,
    BeyondLimberConfig,
    BeyondLimberResult,
    compute_beyond_limber_c_ell,
    compute_hybrid_c_ell,
    spherical_bessel_j,
    beyond_limber_as_strict_mode,
)

from closures.weyl.sigma_evolution import (
    SigmaRegime,
    GzModel,
    SigmaConfig,
    SigmaResult,
    SigmaFitResult,
    compute_Sigma,
    compute_Sigma_from_hJ,
    fit_Sigma_0,
    g_z_standard,
    g_z_constant,
    g_z_exponential,
    DES_Y3_DATA,
    Sigma_to_UMCP_invariants,
)

from closures.weyl.boost_factor import (
    NonlinearRegime,
    BoostConfig,
    BoostResult,
    compute_boost_factor,
    compute_boost_array,
    halofit_boost,
    classify_scale_regime,
    boost_as_curvature,
)

from closures.weyl.cosmology_background import (
    CosmologyParams,
    BackgroundResult,
    PLANCK_2018,
    H_of_z,
    chi_of_z,
    D1_of_z,
    sigma8_of_z,
    Omega_m_of_z,
    Omega_Lambda_of_z,
    compute_background,
    compute_background_array,
    compute_des_y3_background,
    cosmology_as_embedding,
)


__all__ = [
    # weyl_transfer
    "WeylRegime",
    "WeylTransferConfig",
    "WeylTransferResult",
    "compute_weyl_transfer",
    "compute_weyl_transfer_array",
    "weyl_return_domain",
    # limber_integral
    "LimberRegime",
    "LimberConfig",
    "GGLResult",
    "GalaxyClusteringResult",
    "compute_ggl_c_ell",
    "compute_clustering_c_ell",
    "k_ell",
    "classify_limber_regime",
    "limber_as_projection",
    # beyond_limber
    "BeyondLimberRegime",
    "BeyondLimberConfig",
    "BeyondLimberResult",
    "compute_beyond_limber_c_ell",
    "compute_hybrid_c_ell",
    "spherical_bessel_j",
    "beyond_limber_as_strict_mode",
    # sigma_evolution
    "SigmaRegime",
    "GzModel",
    "SigmaConfig",
    "SigmaResult",
    "SigmaFitResult",
    "compute_Sigma",
    "compute_Sigma_from_hJ",
    "fit_Sigma_0",
    "g_z_standard",
    "g_z_constant",
    "g_z_exponential",
    "DES_Y3_DATA",
    "Sigma_to_UMCP_invariants",
    # boost_factor
    "NonlinearRegime",
    "BoostConfig",
    "BoostResult",
    "compute_boost_factor",
    "compute_boost_array",
    "halofit_boost",
    "classify_scale_regime",
    "boost_as_curvature",
    # cosmology_background
    "CosmologyParams",
    "BackgroundResult",
    "PLANCK_2018",
    "H_of_z",
    "chi_of_z",
    "D1_of_z",
    "sigma8_of_z",
    "Omega_m_of_z",
    "Omega_Lambda_of_z",
    "compute_background",
    "compute_background_array",
    "compute_des_y3_background",
    "cosmology_as_embedding",
]
