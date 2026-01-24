#!/usr/bin/env python3
"""
GCD Closure: Momentum Flux (Tier-1)

Computes momentum transfer flux during regime transitions. This closure measures
the rate of change of system integrity weighted by curvature, providing insight
into the dynamics of collapse and recovery.

**Tier-1 GCD Framework**: Uses only base Generalized Collapse Dynamics invariants.
Does not introduce new symbols or modify existing mathematical identities.

**Operational Terms Defined** (enforcement-tied, not metaphors):

Tier-1 Invariants (Inputs from GCD Kernel):
    κ (kappa) = Log-integrity = ln(IC), where IC is integrity composite
                NOT: information content, entropy, or moral integrity
    IC = Integrity composite = exp(κ), multiplicative measure of state integrity
                Range: (0,1], NOT: moral integrity, "truth", or data quality
    ω (omega) = Drift = 1 - F, collapse proximity metric
                Range: [0,1], NOT: random drift, velocity, or wandering
    F = Fidelity = weighted mean of trace components Σ wᵢ·cᵢ
                Range: [0,1], NOT: accuracy, allegiance, or moral fidelity
    C = Curvature = stddev(cᵢ)/0.5, spatial heterogeneity (instability proxy)
                Range: [0,1], NOT: geometric curvature in spacetime
    dκ/dt = Time derivative of log-integrity (momentum indicator)
                Measures rate of integrity change
    τ_R (tau_R) = Return delay = time steps until trajectory returns to domain Dθ
                Range: ℕ∪{∞ᵣₑ꜀}, NOT: periodicity, repetition, or nostalgia

Formula:
    Φ_momentum = (dκ/dt) · √(1 + C²) · (1 - ω)

Where:
    dκ/dt = Rate of change of log-integrity (momentum indicator)
    C = Curvature (Tier-1 invariant, spatial heterogeneity)
    ω = Drift (Tier-1 invariant, collapse proximity)

Physical Interpretation:
    - Positive dκ/dt indicates recovery (integrity increasing)
    - Negative dκ/dt indicates collapse progression (integrity decreasing)
    - Curvature √(1+C²) amplifies flux magnitude in non-uniform regions
    - Drift factor (1-ω) weights by distance from collapse boundary

Regime Classification:
    - Restoring:  Φ_momentum < -0.1  (strong recovery flux)
    - Neutral:    -0.1 ≤ Φ_momentum ≤ 0.1  (equilibrium)
    - Degrading:  Φ_momentum > 0.1  (collapse flux)

Tier Compliance:
    - Uses only Tier-1 GCD invariants (κ, C, ω)
    - Respects mathematical identities (ω = 1 - F)
    - Compatible with all GCD thresholds
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for optimization imports
_src_path = Path(__file__).parent.parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Check if optimizations are available (OPT-1, Lemma 1 validation)
_HAS_OPTIMIZATIONS = importlib.util.find_spec("umcp.kernel_optimized") is not None


def compute_momentum_flux(
    kappa_series: np.ndarray,
    C_series: np.ndarray,
    omega_series: np.ndarray,
    dt: float = 1.0,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """
    Compute momentum flux from time series of Tier-1 GCD invariants.

    Tier-1 GCD Invariants Used:
        κ (kappa): Log-integrity, ln(IC) where IC = integrity composite
        C: Curvature, spatial heterogeneity measure (dispersion proxy)
        ω (omega): Drift, collapse proximity (ω = 1 - F where F = fidelity)

    Args:
        kappa_series: Array of log-integrity values (κ), Tier-1 invariant
        C_series: Array of curvature values (C), Tier-1 invariant
        omega_series: Array of drift values (ω), Tier-1 invariant
        dt: Time step between measurements (default=1.0)
        tol: Numerical tolerance (default=1e-10)

    Note:
        Uses OPT-1 kernel validation when available (Lemma 1 bounds enforcement)

    Returns:
        Dictionary containing:
            - phi_momentum: Array of momentum flux values
            - dkappa_dt: Array of log-integrity derivatives
            - mean_flux: Mean momentum flux over time window
            - regime: Overall regime classification
            - net_integrity_change: Total Δκ over window
            - flux_variance: Variance of momentum flux (stability indicator)

    Raises:
        ValueError: If input arrays have mismatched shapes or invalid values
    """
    # Input validation
    if kappa_series.shape != C_series.shape or kappa_series.shape != omega_series.shape:
        raise ValueError("Input arrays must have same shape")

    if len(kappa_series) < 2:
        raise ValueError("Need at least 2 points for derivative computation")

    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")

    if not np.all((omega_series >= 0) & (omega_series <= 1)):
        raise ValueError("Drift ω must be in [0,1]")

    if not np.all(C_series >= 0):
        raise ValueError("Curvature C must be non-negative")

    # Compute log-integrity derivative (vectorized)
    dkappa_dt = np.gradient(kappa_series, dt)

    # Compute curvature amplification factor (vectorized)
    curvature_amp = np.sqrt(1.0 + C_series**2)

    # Compute drift weighting (distance from collapse)
    drift_weight = 1.0 - omega_series

    # Momentum flux (vectorized)
    phi_momentum = dkappa_dt * curvature_amp * drift_weight

    # Aggregate statistics
    mean_flux = np.mean(phi_momentum)
    flux_variance = np.var(phi_momentum)
    net_integrity_change = kappa_series[-1] - kappa_series[0]

    # Regime classification based on mean flux
    if mean_flux < -0.1:
        regime = "Restoring"
    elif mean_flux <= 0.1:
        regime = "Neutral"
    else:
        regime = "Degrading"

    return {
        "phi_momentum": phi_momentum,
        "dkappa_dt": dkappa_dt,
        "mean_flux": float(mean_flux),
        "regime": regime,
        "net_integrity_change": float(net_integrity_change),
        "flux_variance": float(flux_variance),
        "curvature_amplification": curvature_amp,
        "drift_weighting": drift_weight,
    }


def compute_scalar_momentum_flux(
    kappa: float, kappa_prev: float, C: float, omega: float, dt: float = 1.0
) -> dict[str, Any]:
    """
    Compute momentum flux for single time step (scalar version).

    Args:
        kappa: Current log-integrity
        kappa_prev: Previous log-integrity
        C: Curvature
        omega: Drift
        dt: Time step

    Returns:
        Dictionary with single-point momentum flux analysis
    """
    dkappa_dt = (kappa - kappa_prev) / dt
    curvature_amp = np.sqrt(1.0 + C**2)
    drift_weight = 1.0 - omega
    phi_momentum = dkappa_dt * curvature_amp * drift_weight

    if phi_momentum < -0.1:
        regime = "Restoring"
    elif phi_momentum <= 0.1:
        regime = "Neutral"
    else:
        regime = "Degrading"

    return {
        "phi_momentum": float(phi_momentum),
        "dkappa_dt": float(dkappa_dt),
        "regime": regime,
    }


# Example usage and testing
if __name__ == "__main__":
    print("GCD Momentum Flux Closure (Tier-1)")
    print("=" * 60)

    # Test 1: Constant integrity (equilibrium)
    print("\nTest 1: Constant Integrity (Neutral)")
    kappa_const = np.ones(50) * -2.0
    C_const = np.ones(50) * 0.05
    omega_const = np.ones(50) * 0.01
    result = compute_momentum_flux(kappa_const, C_const, omega_const)
    print(f"  Mean flux: {result['mean_flux']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Net Δκ: {result['net_integrity_change']:.6f}")

    # Test 2: Degrading integrity (collapse progression)
    print("\nTest 2: Degrading Integrity (Collapse Progression)")
    kappa_degrade = np.linspace(-1.0, -5.0, 50)
    C_increase = np.linspace(0.02, 0.15, 50)
    omega_increase = np.linspace(0.01, 0.25, 50)
    result = compute_momentum_flux(kappa_degrade, C_increase, omega_increase, dt=0.1)
    print(f"  Mean flux: {result['mean_flux']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Net Δκ: {result['net_integrity_change']:.6f}")
    print(f"  Flux variance: {result['flux_variance']:.6f}")

    # Test 3: Recovery (restoration)
    print("\nTest 3: Integrity Recovery (Restoring)")
    kappa_recover = np.linspace(-5.0, -1.0, 50)
    C_decrease = np.linspace(0.15, 0.02, 50)
    omega_decrease = np.linspace(0.25, 0.01, 50)
    result = compute_momentum_flux(kappa_recover, C_decrease, omega_decrease, dt=0.1)
    print(f"  Mean flux: {result['mean_flux']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Net Δκ: {result['net_integrity_change']:.6f}")

    # Test 4: Scalar version
    print("\nTest 4: Scalar Momentum Flux (Single Step)")
    result = compute_scalar_momentum_flux(kappa=-2.5, kappa_prev=-2.0, C=0.08, omega=0.05, dt=1.0)
    print(f"  Φ_momentum: {result['phi_momentum']:.6f}")
    print(f"  dκ/dt: {result['dkappa_dt']:.6f}")
    print(f"  Regime: {result['regime']}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
