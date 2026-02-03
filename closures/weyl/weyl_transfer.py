"""
Weyl Transfer Function Computation

Implements the Weyl potential transfer function T_{Ψ_W}(k,z) from:
"Model-independent test of gravity with a network of galaxy surveys"
(Nature Communications 15:9295, 2024)

Mathematical Basis:
    T_{Ψ_W}(k, z) = [H²(z) J(k,z) / (H²(z*) D₁(z*))] · √[B(k,z)/B(k,z*)] · T_{Ψ_W}(k, z*)
    
    Where:
    - H(z) is the conformal-time Hubble rate
    - J(k,z) is the Weyl deviation function (J=1 for GR)
    - D₁(z) is the linear growth function
    - B(k,z) is the nonlinear boost factor
    - z* is the high-z anchor where GR+ΛCDM is assumed

UMCP Integration:
    - Tier-1 closure for WEYL.INTSTACK.v1
    - Maps cosmological evolution to return-domain geometry
    - z* anchor = reference state for deviation measurement

Reference: Eq. 2 of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class WeylRegime(str, Enum):
    """Weyl transfer function regime classification."""
    
    GR_CONSISTENT = "GR_consistent"  # J ≈ 1, no deviation
    MILD_DEVIATION = "Mild_deviation"  # Small J departure
    STRONG_DEVIATION = "Strong_deviation"  # Significant J departure
    UNDEFINED = "Undefined"  # Cannot compute


class WeylTransferResult(NamedTuple):
    """Result of Weyl transfer function computation."""
    
    T_Weyl: float  # Transfer function T_{Ψ_W}(k,z)
    T_ratio: float  # Ratio T(z)/T(z*)
    J_effective: float  # Effective J(z) value
    regime: str  # WeylRegime classification
    H_ratio: float  # H²(z)/H²(z*)
    D1_star: float  # D₁(z*) normalization
    B_ratio: float  # √[B(k,z)/B(k,z*)]


@dataclass
class WeylTransferConfig:
    """Configuration for Weyl transfer computation."""
    
    # Tolerance for GR consistency
    J_tol_gr: float = 0.05  # |J-1| < this → GR consistent
    J_tol_mild: float = 0.15  # |J-1| < this → mild deviation
    
    # Numerical safety
    epsilon: float = 1e-10
    
    # Default anchor
    z_star: float = 10.0  # Matter era anchor


def compute_weyl_transfer(
    H_z: float,
    H_z_star: float,
    J_z: float,
    D1_z_star: float,
    B_ratio: float,
    T_Weyl_star: float,
    config: WeylTransferConfig | None = None,
) -> WeylTransferResult:
    """
    Compute Weyl transfer function T_{Ψ_W}(k, z).
    
    Implements Eq. 2 from Nature Comms 15:9295 (2024):
        T_{Ψ_W}(k, z) = [H²(z) J(k,z) / (H²(z*) D₁(z*))] 
                        · √[B(k,z)/B(k,z*)] · T_{Ψ_W}(k, z*)
    
    Args:
        H_z: Conformal Hubble rate at redshift z
        H_z_star: Conformal Hubble rate at anchor z*
        J_z: Weyl deviation function J(z) (J=1 for GR)
        D1_z_star: Linear growth function D₁(z*)
        B_ratio: √[B(k,z)/B(k,z*)] boost ratio
        T_Weyl_star: Transfer function at anchor T_{Ψ_W}(k, z*)
        config: Optional configuration parameters
    
    Returns:
        WeylTransferResult with T_Weyl, T_ratio, regime
    
    Reference:
        The transfer function encodes how the Weyl potential Ψ_W = (Φ+Ψ)/2
        evolves from the high-z anchor (where GR holds) to lower redshift.
        J(z) ≠ 1 indicates modified gravity.
    """
    if config is None:
        config = WeylTransferConfig()
    
    eps = config.epsilon
    
    # Validate inputs
    if H_z_star < eps or D1_z_star < eps or T_Weyl_star < eps:
        return WeylTransferResult(
            T_Weyl=0.0,
            T_ratio=0.0,
            J_effective=J_z,
            regime=WeylRegime.UNDEFINED.value,
            H_ratio=0.0,
            D1_star=D1_z_star,
            B_ratio=B_ratio,
        )
    
    # Compute H² ratio
    H_ratio = (H_z ** 2) / (H_z_star ** 2 + eps)
    
    # Compute transfer function (Eq. 2)
    prefactor = H_ratio * J_z / (D1_z_star + eps)
    T_Weyl = prefactor * B_ratio * T_Weyl_star
    
    # Compute ratio to anchor
    T_ratio = T_Weyl / (T_Weyl_star + eps)
    
    # Classify regime based on J deviation
    J_deviation = abs(J_z - 1.0)
    if J_deviation < config.J_tol_gr:
        regime = WeylRegime.GR_CONSISTENT.value
    elif J_deviation < config.J_tol_mild:
        regime = WeylRegime.MILD_DEVIATION.value
    else:
        regime = WeylRegime.STRONG_DEVIATION.value
    
    return WeylTransferResult(
        T_Weyl=float(T_Weyl),
        T_ratio=float(T_ratio),
        J_effective=float(J_z),
        regime=regime,
        H_ratio=float(H_ratio),
        D1_star=float(D1_z_star),
        B_ratio=float(B_ratio),
    )


def compute_weyl_transfer_array(
    H_z: NDArray[np.floating],
    H_z_star: float,
    J_z: NDArray[np.floating],
    D1_z_star: float,
    B_ratio: NDArray[np.floating],
    T_Weyl_star: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Vectorized Weyl transfer function computation.
    
    Args:
        H_z: Array of H(z) values
        H_z_star: H(z*) anchor value
        J_z: Array of J(z) values
        D1_z_star: D₁(z*) anchor value
        B_ratio: Array of √[B(k,z)/B(k,z*)] values
        T_Weyl_star: T_{Ψ_W}(k, z*) anchor value
    
    Returns:
        Tuple of (T_Weyl array, T_ratio array)
    """
    eps = 1e-10
    
    H_ratio = (H_z ** 2) / (H_z_star ** 2 + eps)
    prefactor = H_ratio * J_z / (D1_z_star + eps)
    T_Weyl = prefactor * B_ratio * T_Weyl_star
    T_ratio = T_Weyl / (T_Weyl_star + eps)
    
    return T_Weyl, T_ratio


# =============================================================================
# UMCP INTEGRATION: Return-Domain Interpretation
# =============================================================================

def weyl_return_domain(
    z_values: NDArray[np.floating],
    J_values: NDArray[np.floating],
    z_star: float,
    eta_J: float = 0.05,
) -> dict:
    """
    Interpret Weyl deviation as return-domain geometry.
    
    UMCP Mapping:
        - z* anchor = reference state for return
        - J(z) ≈ 1 = "returned" to GR-consistent state
        - |J(z) - 1| > η = deviation from GR (no return)
    
    This is the cosmological analog of τ_R computation:
        τ_R = "when did we return to GR?" in redshift space
    
    Args:
        z_values: Redshift array (descending, z=0 is present)
        J_values: J(z) array at each redshift
        z_star: High-z anchor where J=1
        eta_J: Tolerance for J ≈ 1 (GR-consistent)
    
    Returns:
        Dictionary with return analysis
    """
    # Find indices where J is GR-consistent
    gr_consistent = np.abs(J_values - 1.0) < eta_J
    
    # Return domain: indices eligible as "return" states
    return_domain = np.where(gr_consistent)[0]
    
    # First deviation from GR (going from z* toward z=0)
    deviation_idx = np.where(~gr_consistent)[0]
    first_deviation_z = float(z_values[deviation_idx[0]]) if len(deviation_idx) > 0 else None
    
    # Return rate: fraction of redshifts with J ≈ 1
    return_rate = float(np.mean(gr_consistent))
    
    return {
        "return_domain_size": len(return_domain),
        "total_points": len(z_values),
        "return_rate": return_rate,
        "first_deviation_z": first_deviation_z,
        "gr_consistent_fraction": return_rate,
        "interpretation": (
            "GR_HOLDS" if return_rate > 0.9 else
            "MILD_DEVIATION" if return_rate > 0.5 else
            "STRONG_DEVIATION"
        ),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Weyl Transfer Function Self-Test ===\n")
    
    # Test case: GR-consistent evolution (J=1)
    result_gr = compute_weyl_transfer(
        H_z=70.0,  # H(z=0) ~ 70 km/s/Mpc (simplified)
        H_z_star=700.0,  # H(z*=10) ~ 10x higher
        J_z=1.0,  # GR: J=1
        D1_z_star=0.1,  # Growth at z*=10
        B_ratio=1.0,  # No nonlinear boost at linear scales
        T_Weyl_star=1.0,  # Normalized
    )
    print(f"GR-consistent (J=1):")
    print(f"  T_Weyl = {result_gr.T_Weyl:.4f}")
    print(f"  regime = {result_gr.regime}")
    print(f"  H_ratio = {result_gr.H_ratio:.4f}")
    
    # Test case: Modified gravity (J=1.24, Σ₀=0.24)
    result_mg = compute_weyl_transfer(
        H_z=70.0,
        H_z_star=700.0,
        J_z=1.24,  # Σ₀=0.24 → J deviation
        D1_z_star=0.1,
        B_ratio=1.0,
        T_Weyl_star=1.0,
    )
    print(f"\nModified gravity (J=1.24):")
    print(f"  T_Weyl = {result_mg.T_Weyl:.4f}")
    print(f"  regime = {result_mg.regime}")
    print(f"  J_effective = {result_mg.J_effective}")
    
    # Test return-domain interpretation
    z_test = np.linspace(0, 2, 21)
    J_test = 1.0 + 0.24 * np.exp(-z_test)  # Deviation at low z
    
    return_analysis = weyl_return_domain(z_test, J_test, z_star=10.0, eta_J=0.05)
    print(f"\nReturn-domain analysis:")
    print(f"  GR-consistent fraction = {return_analysis['gr_consistent_fraction']:.2%}")
    print(f"  First deviation at z = {return_analysis['first_deviation_z']}")
    print(f"  Interpretation: {return_analysis['interpretation']}")
    
    print("\n✓ Self-test complete")
