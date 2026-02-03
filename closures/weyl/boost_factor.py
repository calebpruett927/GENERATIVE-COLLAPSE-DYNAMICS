"""
Nonlinear Boost Factor B(k,z)

Implements the nonlinear boost factor from:
"Model-independent test of gravity with a network of galaxy surveys"
(Nature Communications 15:9295, 2024)

Mathematical Basis:
    The boost factor B(k,z) encodes the ratio of nonlinear to linear
    power spectrum at scale k and redshift z:

    B(k,z) = P_nl(k,z) / P_lin(k,z)

    This appears in the Weyl transfer function (Eq. 2) as:
    √[B(k,z) / B(k,z*)]

    And in the Limber integrals (Eq. 3, 6) as a multiplicative correction.

UMCP Integration:
    - Boost factor → closure that modifies linear prediction
    - B ≠ 1 → nonlinear regime, analogous to curvature C
    - Scale-dependent: k enters as parameter, like ℓ in regime gates

Reference: Implicit in Eq. 2, 3, 6 of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class NonlinearRegime(str, Enum):
    """Nonlinear regime classification."""

    LINEAR = "Linear"  # B ≈ 1
    QUASI_LINEAR = "Quasi_linear"  # 1 < B < 2
    NONLINEAR = "Nonlinear"  # 2 ≤ B < 10
    HIGHLY_NONLINEAR = "Highly_nonlinear"  # B ≥ 10


class BoostResult(NamedTuple):
    """Result of boost factor computation."""

    B_boost: float  # B(k,z)
    B_ratio: float  # √[B(k,z)/B(k,z*)]
    k: float  # Wavenumber
    z: float  # Redshift
    regime: str  # NonlinearRegime
    nl_correction: float  # B - 1 (excess over linear)


@dataclass
class BoostConfig:
    """Configuration for boost factor computation."""

    # Regime thresholds
    B_linear_threshold: float = 1.05
    B_quasi_linear_threshold: float = 2.0
    B_nonlinear_threshold: float = 10.0

    # Scale thresholds (h/Mpc)
    k_linear_max: float = 0.1
    k_nonlinear_min: float = 0.5

    # z* anchor (matter era)
    z_star: float = 10.0

    # Numerical parameters
    epsilon: float = 1e-10


def halofit_boost(k: float, z: float, sigma8: float = 0.811) -> float:
    """
    Simplified Halofit-inspired boost factor.

    This is a simplified model. Real implementations use:
    - CAMB/CLASS Halofit
    - HMcode
    - Euclid Emulator

    Approximate form:
        B(k,z) ≈ 1 + (k/k_nl)^α × f(z)

    Where k_nl ~ 0.2 h/Mpc at z=0.

    Args:
        k: Wavenumber in h/Mpc
        z: Redshift
        sigma8: σ8 normalization

    Returns:
        B(k,z) boost factor
    """
    # Nonlinear scale (evolves with redshift)
    k_nl = 0.2 * (1 + z) ** 0.5  # Approximate

    # Growth suppression at high z
    growth_factor = 1.0 / (1 + z)

    # Boost amplitude
    alpha = 2.0
    B_amplitude = (sigma8 / 0.811) ** 2

    if k < k_nl:
        # Linear regime
        return 1.0 + 0.1 * (k / k_nl) ** alpha * B_amplitude * growth_factor**2
    else:
        # Nonlinear regime
        return 1.0 + (k / k_nl) ** alpha * B_amplitude * growth_factor**2


def compute_boost_factor(
    k: float,
    z: float,
    z_star: float = 10.0,
    P_lin: Callable[[float, float], float] | None = None,
    P_nl: Callable[[float, float], float] | None = None,
    config: BoostConfig | None = None,
) -> BoostResult:
    """
    Compute nonlinear boost factor B(k,z).

    If P_lin and P_nl are provided, computes:
        B(k,z) = P_nl(k,z) / P_lin(k,z)

    Otherwise uses simplified Halofit model.

    Args:
        k: Wavenumber in h/Mpc
        z: Redshift
        z_star: High-z anchor
        P_lin: Linear power spectrum function
        P_nl: Nonlinear power spectrum function
        config: Optional configuration

    Returns:
        BoostResult with B(k,z) and diagnostics
    """
    if config is None:
        config = BoostConfig()

    eps = config.epsilon

    # Compute B(k,z)
    if P_lin is not None and P_nl is not None:
        P_lin_val = P_lin(k, z)
        P_nl_val = P_nl(k, z)
        B_z = P_nl_val / (P_lin_val + eps)

        # B at anchor
        P_lin_star = P_lin(k, z_star)
        P_nl_star = P_nl(k, z_star)
        B_star = P_nl_star / (P_lin_star + eps)
    else:
        # Use simplified model
        B_z = halofit_boost(k, z)
        B_star = halofit_boost(k, z_star)

    # Boost ratio for transfer function
    B_ratio = np.sqrt(B_z / (B_star + eps))

    # Nonlinear correction
    nl_correction = B_z - 1.0

    # Classify regime
    if B_z < config.B_linear_threshold:
        regime = NonlinearRegime.LINEAR.value
    elif B_z < config.B_quasi_linear_threshold:
        regime = NonlinearRegime.QUASI_LINEAR.value
    elif B_z < config.B_nonlinear_threshold:
        regime = NonlinearRegime.NONLINEAR.value
    else:
        regime = NonlinearRegime.HIGHLY_NONLINEAR.value

    return BoostResult(
        B_boost=float(B_z),
        B_ratio=float(B_ratio),
        k=float(k),
        z=float(z),
        regime=regime,
        nl_correction=float(nl_correction),
    )


def compute_boost_array(
    k_values: NDArray[np.floating],
    z: float,
    z_star: float = 10.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Vectorized boost factor computation.

    Args:
        k_values: Array of wavenumbers
        z: Redshift
        z_star: Anchor redshift

    Returns:
        Tuple of (B(k,z) array, B_ratio array)
    """
    B_z = np.array([halofit_boost(k, z) for k in k_values])
    B_star = np.array([halofit_boost(k, z_star) for k in k_values])
    B_ratio = np.sqrt(B_z / np.maximum(B_star, 1e-10))

    return B_z, B_ratio


def classify_scale_regime(k: float) -> str:
    """
    Classify scale regime based on wavenumber.

    Args:
        k: Wavenumber in h/Mpc

    Returns:
        Scale regime string
    """
    if k < 0.01:
        return "Ultra_large_scale"
    elif k < 0.1:
        return "Linear_scale"
    elif k < 0.5:
        return "Quasi_linear_scale"
    elif k < 2.0:
        return "Nonlinear_scale"
    else:
        return "Highly_nonlinear_scale"


# =============================================================================
# UMCP INTEGRATION: Curvature-like Interpretation
# =============================================================================


def boost_as_curvature(
    k_values: NDArray[np.floating],
    z: float,
) -> dict:
    """
    Interpret boost factor as UMCP curvature analog.

    UMCP Mapping:
        - B(k,z) = 1 → linear (low curvature, uniform)
        - B(k,z) > 1 → nonlinear (high curvature, non-uniform)
        - C ~ (B - 1) / B_max is a normalized curvature proxy

    This is analogous to C(t) measuring coordinate non-uniformity.
    In cosmology, nonlinear growth creates "non-uniformity" at small scales.

    Args:
        k_values: Wavenumbers to evaluate
        z: Redshift

    Returns:
        Curvature interpretation dictionary
    """
    B_values = np.array([halofit_boost(k, z) for k in k_values])

    # Max boost for normalization
    B_max = np.max(B_values)

    # Curvature proxy: normalized excess over linear
    C_proxy = (B_values - 1.0) / max(B_max - 1.0, 0.01)

    # Statistics
    C_mean = float(np.mean(C_proxy))
    C_max = float(np.max(C_proxy))

    # k at which nonlinearity kicks in
    k_nl = k_values[np.argmax(B_values > 1.5)] if np.any(B_values > 1.5) else np.max(k_values)

    return {
        "C_proxy_mean": C_mean,
        "C_proxy_max": C_max,
        "B_max": float(B_max),
        "k_nonlinear": float(k_nl),
        "n_scales": len(k_values),
        "interpretation": (
            "Boost factor B(k,z) is the cosmological analog of curvature C(t). "
            f"At z={z}, nonlinearity begins at k~{k_nl:.2f} h/Mpc, "
            f"reaching max B={B_max:.1f}."
        ),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Boost Factor Self-Test ===\n")

    # Test simplified Halofit
    k_test = [0.01, 0.1, 0.5, 1.0, 2.0]
    print("Halofit boost B(k, z=0.5):")
    for k in k_test:
        B = halofit_boost(k, z=0.5)
        regime = classify_scale_regime(k)
        print(f"  k={k:.2f} h/Mpc: B={B:.3f} ({regime})")

    # Test full computation
    print("\nBoost factor results:")
    for k in [0.05, 0.2, 1.0]:
        result = compute_boost_factor(k, z=0.5, z_star=10.0)
        print(f"  k={k}: B={result.B_boost:.3f}, ratio={result.B_ratio:.3f}, {result.regime}")

    # Test curvature interpretation
    k_array = np.logspace(-2, 0.5, 20)
    curvature = boost_as_curvature(k_array, z=0.5)
    print("\nCurvature interpretation at z=0.5:")
    print(f"  C_proxy mean = {curvature['C_proxy_mean']:.3f}")
    print(f"  k_nonlinear = {curvature['k_nonlinear']:.3f} h/Mpc")
    print(f"  B_max = {curvature['B_max']:.2f}")

    # Redshift evolution
    print("\nB(k=0.5, z) vs redshift:")
    for z in [0.0, 0.5, 1.0, 2.0]:
        B = halofit_boost(0.5, z)
        print(f"  z={z}: B={B:.3f}")

    print("\n✓ Self-test complete")
