"""
Background Cosmology Functions

Computes the background cosmological quantities needed for Weyl evolution:
- Conformal Hubble H(z)
- Comoving distance χ(z)
- Linear growth function D₁(z)
- Matter density Ω_m(z)
- σ8(z) amplitude evolution

These are the foundation for all Weyl closures, analogous to UMCP's
coordinate embedding (Tier-0) that maps from observables to Ψ-space.

UMCP Integration:
    - Cosmological background = embedding specification
    - H(z), χ(z), D₁(z) = coordinate transformations
    - Frozen cosmological parameters = frozen contract parameters
    - z* anchor = reference state for return computation

Reference: Methods section of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate


class BackgroundResult(NamedTuple):
    """Result of background cosmology computation."""
    
    H_z: float  # Conformal Hubble H(z) in km/s/Mpc
    chi: float  # Comoving distance χ(z) in Mpc/h
    D1: float  # Linear growth function D₁(z)
    Omega_m_z: float  # Matter density Ω_m(z)
    Omega_Lambda_z: float  # Dark energy density Ω_Λ(z)
    sigma8_z: float  # σ8(z) amplitude
    a: float  # Scale factor a = 1/(1+z)


@dataclass
class CosmologyParams:
    """Fiducial cosmological parameters (Planck 2018)."""
    
    # Matter density today
    Omega_m_0: float = 0.315
    
    # Dark energy density today
    Omega_Lambda_0: float = 0.685
    
    # Hubble constant in km/s/Mpc
    H_0: float = 67.4
    
    # σ8 at z=0
    sigma8_0: float = 0.811
    
    # Scalar spectral index
    n_s: float = 0.965
    
    # Speed of light for distance computation
    c_km_s: float = 299792.458  # km/s


# Default Planck 2018 cosmology
PLANCK_2018 = CosmologyParams()


def H_of_z(z: float, params: CosmologyParams | None = None) -> float:
    """
    Compute Hubble parameter H(z) in km/s/Mpc.
    
    For flat ΛCDM:
        H(z) = H₀ √[Ω_m(1+z)³ + Ω_Λ]
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        H(z) in km/s/Mpc
    """
    if params is None:
        params = PLANCK_2018
    
    E_z = np.sqrt(
        params.Omega_m_0 * (1 + z) ** 3 + params.Omega_Lambda_0
    )
    return params.H_0 * E_z


def Omega_m_of_z(z: float, params: CosmologyParams | None = None) -> float:
    """
    Compute matter density Ω_m(z).
    
    Ω_m(z) = Ω_m,0 (1+z)³ / E²(z)
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        Ω_m(z)
    """
    if params is None:
        params = PLANCK_2018
    
    E_z_sq = params.Omega_m_0 * (1 + z) ** 3 + params.Omega_Lambda_0
    return params.Omega_m_0 * (1 + z) ** 3 / E_z_sq


def Omega_Lambda_of_z(z: float, params: CosmologyParams | None = None) -> float:
    """
    Compute dark energy density Ω_Λ(z).
    
    Ω_Λ(z) = Ω_Λ,0 / E²(z)
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        Ω_Λ(z)
    """
    if params is None:
        params = PLANCK_2018
    
    E_z_sq = params.Omega_m_0 * (1 + z) ** 3 + params.Omega_Lambda_0
    return params.Omega_Lambda_0 / E_z_sq


def chi_of_z(z: float, params: CosmologyParams | None = None, n_points: int = 100) -> float:
    """
    Compute comoving distance χ(z) in Mpc/h.
    
    χ(z) = c ∫₀ᶻ dz' / H(z')
    
    Args:
        z: Redshift
        params: Cosmological parameters
        n_points: Integration points
    
    Returns:
        χ(z) in Mpc/h
    """
    if params is None:
        params = PLANCK_2018
    
    if z <= 0:
        return 0.0
    
    # Integrate c/H(z) from 0 to z
    def integrand(z_prime: float) -> float:
        return params.c_km_s / H_of_z(z_prime, params)
    
    result, _ = integrate.quad(integrand, 0, z)
    
    # Convert to Mpc/h
    h = params.H_0 / 100.0
    return result * h


def D1_of_z(z: float, params: CosmologyParams | None = None) -> float:
    """
    Compute linear growth function D₁(z), normalized to D₁(0) = 1.
    
    Uses approximate form valid for flat ΛCDM:
        D₁(z) ≈ (5/2) Ω_m(z) g(z) / (1+z)
    
    Where g(z) is the growth suppression factor.
    
    For more accurate computation, use CAMB or CLASS.
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        D₁(z) normalized to D₁(0) = 1
    """
    if params is None:
        params = PLANCK_2018
    
    # Growth suppression factor (Carroll, Press & Turner 1992)
    Omega_m_z_val = Omega_m_of_z(z, params)
    Omega_Lambda_z_val = Omega_Lambda_of_z(z, params)
    
    g_z = (5.0 / 2.0) * Omega_m_z_val / (
        Omega_m_z_val ** (4.0 / 7.0)
        - Omega_Lambda_z_val
        + (1 + Omega_m_z_val / 2.0) * (1 + Omega_Lambda_z_val / 70.0)
    )
    
    # Normalize to z=0
    g_0 = (5.0 / 2.0) * params.Omega_m_0 / (
        params.Omega_m_0 ** (4.0 / 7.0)
        - params.Omega_Lambda_0
        + (1 + params.Omega_m_0 / 2.0) * (1 + params.Omega_Lambda_0 / 70.0)
    )
    
    # Include (1+z) scaling
    D1_unnorm = g_z / (1 + z)
    D1_0 = g_0
    
    return D1_unnorm / D1_0


def sigma8_of_z(z: float, params: CosmologyParams | None = None) -> float:
    """
    Compute σ8(z) = σ8(0) × D₁(z).
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        σ8(z)
    """
    if params is None:
        params = PLANCK_2018
    
    return params.sigma8_0 * D1_of_z(z, params)


def compute_background(z: float, params: CosmologyParams | None = None) -> BackgroundResult:
    """
    Compute all background cosmology quantities at redshift z.
    
    Args:
        z: Redshift
        params: Cosmological parameters
    
    Returns:
        BackgroundResult with all quantities
    """
    if params is None:
        params = PLANCK_2018
    
    return BackgroundResult(
        H_z=H_of_z(z, params),
        chi=chi_of_z(z, params),
        D1=D1_of_z(z, params),
        Omega_m_z=Omega_m_of_z(z, params),
        Omega_Lambda_z=Omega_Lambda_of_z(z, params),
        sigma8_z=sigma8_of_z(z, params),
        a=1.0 / (1 + z),
    )


def compute_background_array(
    z_values: NDArray[np.floating],
    params: CosmologyParams | None = None,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute background quantities over array of redshifts.
    
    Args:
        z_values: Redshift array
        params: Cosmological parameters
    
    Returns:
        Dictionary of arrays for each quantity
    """
    if params is None:
        params = PLANCK_2018
    
    n = len(z_values)
    
    return {
        "z": z_values,
        "H_z": np.array([H_of_z(z, params) for z in z_values]),
        "chi": np.array([chi_of_z(z, params) for z in z_values]),
        "D1": np.array([D1_of_z(z, params) for z in z_values]),
        "Omega_m_z": np.array([Omega_m_of_z(z, params) for z in z_values]),
        "Omega_Lambda_z": np.array([Omega_Lambda_of_z(z, params) for z in z_values]),
        "sigma8_z": np.array([sigma8_of_z(z, params) for z in z_values]),
        "a": 1.0 / (1 + z_values),
    }


# =============================================================================
# UMCP INTEGRATION: Embedding Interpretation
# =============================================================================

def cosmology_as_embedding(params: CosmologyParams | None = None) -> dict:
    """
    Interpret cosmological parameters as UMCP embedding specification.
    
    UMCP Mapping:
        - Cosmological parameters → frozen contract parameters
        - z → t (time/evolution axis)
        - H(z), χ(z), D₁(z) → coordinate transformations
        - z* = 10 (matter era) → return domain anchor
        - Observable embedding: (ĥJ, ĥb) → Ψ-coordinates
    
    This is the Tier-0 analog: the interface between physical
    observables and the abstract Ψ-space where UMCP operates.
    
    Args:
        params: Cosmological parameters
    
    Returns:
        Embedding interpretation dictionary
    """
    if params is None:
        params = PLANCK_2018
    
    return {
        "contract_parameters": {
            "Omega_m_0": params.Omega_m_0,
            "Omega_Lambda_0": params.Omega_Lambda_0,
            "H_0": params.H_0,
            "sigma8_0": params.sigma8_0,
            "n_s": params.n_s,
        },
        "embedding_specification": {
            "domain": "z ∈ [0, ∞)",
            "codomain": "[H(z), χ(z), D₁(z), Ω_m(z), σ8(z)]",
            "anchor": "z* = 10 (matter era, GR recovery)",
        },
        "umcp_mapping": {
            "z → t": "Redshift as evolution parameter",
            "D₁(z) → Ψ-coordinate": "Growth as bounded state",
            "σ8(z) → fidelity proxy": "Amplitude normalization",
            "z* → return domain": "High-z anchor for GR baseline",
        },
        "frozen_assumption": (
            "Flat ΛCDM cosmology with Planck 2018 parameters. "
            "Deviations from this constitute the 'modified gravity' signal."
        ),
    }


def compute_z_of_chi(chi_target: float, params: CosmologyParams | None = None) -> float:
    """
    Invert χ(z) to get z(χ).
    
    Uses root-finding to solve χ(z) = χ_target.
    
    Args:
        chi_target: Comoving distance in Mpc/h
        params: Cosmological parameters
    
    Returns:
        Redshift z such that χ(z) = χ_target
    """
    from scipy import optimize
    
    if chi_target <= 0:
        return 0.0
    
    def objective(z: float) -> float:
        return chi_of_z(z, params) - chi_target
    
    # Find z in [0, 10] (covers most of observable universe)
    try:
        z_result: float = optimize.brentq(objective, 0, 10)  # type: ignore[assignment]
        return z_result
    except ValueError:
        # chi_target outside range
        return 10.0 if chi_target > chi_of_z(10, params) else 0.0


# =============================================================================
# DES Y3 LENS BIN BACKGROUND VALUES
# =============================================================================

def compute_des_y3_background() -> dict:
    """
    Compute background quantities at DES Y3 lens bin redshifts.
    
    z_bins = [0.295, 0.467, 0.626, 0.771]
    
    Returns:
        Dictionary with background values at each bin
    """
    z_bins = np.array([0.295, 0.467, 0.626, 0.771])
    
    bg = compute_background_array(z_bins)
    
    # Also compute at anchor z*=10
    bg_star = compute_background(10.0)
    
    return {
        "z_bins": z_bins,
        "H_z": bg["H_z"],
        "chi": bg["chi"],
        "D1": bg["D1"],
        "Omega_m_z": bg["Omega_m_z"],
        "sigma8_z": bg["sigma8_z"],
        "anchor_z_star": {
            "z": 10.0,
            "D1_star": bg_star.D1,
            "sigma8_star": bg_star.sigma8_z,
            "Omega_m_star": bg_star.Omega_m_z,
        },
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Background Cosmology Self-Test ===\n")
    
    # Print fiducial parameters
    print("Planck 2018 Parameters:")
    print(f"  Ω_m,0 = {PLANCK_2018.Omega_m_0}")
    print(f"  Ω_Λ,0 = {PLANCK_2018.Omega_Lambda_0}")
    print(f"  H_0 = {PLANCK_2018.H_0} km/s/Mpc")
    print(f"  σ8,0 = {PLANCK_2018.sigma8_0}")
    
    # Test at different redshifts
    print("\nBackground quantities:")
    z_test = [0.0, 0.5, 1.0, 2.0, 10.0]
    for z in z_test:
        bg = compute_background(z)
        print(f"  z={z:.1f}: H={bg.H_z:.1f}, χ={bg.chi:.0f} Mpc/h, D₁={bg.D1:.3f}, σ8={bg.sigma8_z:.3f}")
    
    # Test DES Y3 bins
    print("\nDES Y3 Lens Bin Background:")
    des_bg = compute_des_y3_background()
    for i, z in enumerate(des_bg["z_bins"]):
        print(f"  z={z:.3f}: D₁={des_bg['D1'][i]:.3f}, σ8={des_bg['sigma8_z'][i]:.3f}")
    print(f"  z*={des_bg['anchor_z_star']['z']}: D₁*={des_bg['anchor_z_star']['D1_star']:.3f}")
    
    # Test embedding interpretation
    print("\nUMCP Embedding Interpretation:")
    emb = cosmology_as_embedding()
    print(f"  Domain: {emb['embedding_specification']['domain']}")
    print(f"  Anchor: {emb['embedding_specification']['anchor']}")
    
    print("\n✓ Self-test complete")
