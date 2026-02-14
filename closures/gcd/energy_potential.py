"""
GCD Energy Potential Closure

Computes the energy potential of a system state based on Tier-1 invariants.
Energy potential represents the system's capacity for generative collapse.

Formula: E = ω² + α·S + β·C²
where α and β are configurable damping factors.
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for optimization imports
_src_path = Path(__file__).parent.parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import optimizations if available (OPT-1, Lemma 1 validation)
try:
    from umcp.kernel_optimized import validate_kernel_bounds

    _has_optimizations = True
except ImportError:
    _has_optimizations = False
    validate_kernel_bounds = None  # type: ignore[assignment]


def compute_energy_potential(
    omega: float, S: float, C: float, alpha: float = 1.0, beta: float = 0.5, **kwargs: Any
) -> dict[str, Any]:
    """
    Compute energy potential from Tier-1 invariants.

    Parameters:
    -----------
    omega : float
        Drift/collapse metric (ω)
    S : float
        Bernoulli field entropy (Shannon entropy is the degenerate limit)
    C : float
        Curvature
    alpha : float, optional
        Entropy damping factor (default: 1.0)
    beta : float, optional
        Curvature damping factor (default: 0.5)

    Returns:
    --------
    dict :
        {
            'E_potential': float,  # Total energy potential
            'E_collapse': float,   # Collapse component (ω²)
            'E_entropy': float,    # Entropy component (α·S)
            'E_curvature': float,  # Curvature component (β·C²)
            'regime': str          # Energy-based regime
        }

    Notes:
    ------
    - E_collapse dominates near collapse boundary
    - E_entropy represents uncertainty contribution
    - E_curvature penalizes non-uniformity
    - Energy regimes: Low (E<0.01), Medium (0.01≤E<0.05), High (E≥0.05)
    """
    # OPT-1: Use optimized kernel validation if available (Lemma 1 bounds)
    if _has_optimizations and validate_kernel_bounds is not None:
        # Approximate κ and IC from ω for validation
        import math

        IC = max(1 - omega, 1e-10)
        kappa = math.log(IC)
        F = 1 - omega
        validate_kernel_bounds(F, omega, C, IC, kappa)

    # Validate inputs
    if not (0 <= omega <= 1):
        raise ValueError(f"omega must be in [0,1], got {omega}")
    if not (S >= 0):
        raise ValueError(f"S must be non-negative, got {S}")
    if not (0 <= C <= 1):
        raise ValueError(f"C must be in [0,1], got {C}")

    # Compute energy components
    E_collapse = omega**2
    E_entropy = alpha * S
    E_curvature = beta * (C**2)

    # Total energy potential
    E_potential = E_collapse + E_entropy + E_curvature

    # Classify energy regime
    if E_potential < 0.01:
        regime = "Low"
    elif E_potential < 0.05:
        regime = "Medium"
    else:
        regime = "High"

    return {
        "E_potential": E_potential,
        "E_collapse": E_collapse,
        "E_entropy": E_entropy,
        "E_curvature": E_curvature,
        "regime": regime,
        "alpha": alpha,
        "beta": beta,
    }


def main():
    """Example usage and validation."""
    # Zero entropy state
    print("Zero Entropy State (S=0):")
    result = compute_energy_potential(omega=0.0, S=0.0, C=0.0)
    print(f"  E_potential = {result['E_potential']:.6f}")
    print(f"  Regime: {result['regime']}")
    print()

    # Stable regime
    print("Stable Regime:")
    result = compute_energy_potential(omega=0.01, S=0.056, C=0.0)
    print(f"  E_potential = {result['E_potential']:.6f}")
    print(f"  E_collapse = {result['E_collapse']:.6f}")
    print(f"  E_entropy = {result['E_entropy']:.6f}")
    print(f"  Regime: {result['regime']}")
    print()

    # Watch regime
    print("Watch Regime:")
    result = compute_energy_potential(omega=0.15, S=0.25, C=0.05)
    print(f"  E_potential = {result['E_potential']:.6f}")
    print(f"  Regime: {result['regime']}")
    print()

    # Collapse boundary
    print("Collapse Boundary:")
    result = compute_energy_potential(omega=0.30, S=0.50, C=0.10)
    print(f"  E_potential = {result['E_potential']:.6f}")
    print(f"  Regime: {result['regime']}")


if __name__ == "__main__":
    main()
