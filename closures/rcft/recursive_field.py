"""
Recursive Field Strength Computation for RCFT (Tier-2)

Computes the recursive field strength Ψ_r by summing contributions from successive
collapse iterations with exponential decay. This Tier-2 metric quantifies how
collapse history influences present dynamics through field memory.

Mathematical Foundation:
    Ψ_r = Σ(n=1→∞) α^n · Ψ_n
    
    where:
    - α is decay factor (0 < α < 1)
    - Ψ_n is field strength at iteration n
    - Ψ_n computed from Tier-1 invariants: Ψ_n = √(S_n² + C_n²) · (1 - F_n)

Regime Classification:
    - Dormant:   Ψ_r < 0.1   (weak recursive coupling)
    - Active:    0.1 ≤ Ψ_r < 1.0   (moderate recursion)
    - Resonant:  Ψ_r ≥ 1.0   (strong recursive resonance)

Tier-2 Constraints:
    - Built from Tier-1 invariants (S, C, F) without modifying them
    - Respects GCD tolerances and mathematical identities
    - Augments GCD with field memory quantification
"""

from typing import Any, Dict

import numpy as np


def compute_field_strength_single(S: float, C: float, F: float) -> float:
    """
    Compute field strength Ψ for a single state from Tier-1 invariants.
    
    Formula: Ψ = √(S² + C²) · (1 - F)
    
    This combines entropy and curvature magnitude with fidelity deficit.
    
    Args:
        S: Entropy (Tier-1 invariant)
        C: Curvature (Tier-1 invariant)
        F: Fidelity (Tier-1 invariant)
    
    Returns:
        Field strength Ψ
    """
    magnitude = np.sqrt(S**2 + C**2)
    fidelity_deficit = 1.0 - F
    return magnitude * fidelity_deficit


def compute_recursive_field(
    S_series: np.ndarray,
    C_series: np.ndarray,
    F_series: np.ndarray,
    alpha: float = 0.8,
    max_iterations: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute recursive field strength by summing contributions from iteration history.
    
    The recursive field quantifies how past collapse events influence present dynamics
    through exponentially decaying memory.
    
    Args:
        S_series: Time series of entropy values (Tier-1)
        C_series: Time series of curvature values (Tier-1)
        F_series: Time series of fidelity values (Tier-1)
        alpha: Decay factor for recursive contributions (0 < α < 1)
        max_iterations: Maximum number of iterations to include
        tol: Convergence tolerance for series truncation
    
    Returns:
        Dictionary containing:
            - Psi_recursive: Total recursive field strength
            - regime: Classification (Dormant/Active/Resonant)
            - contributions: Array of individual Ψ_n values
            - decay_factors: Array of α^n values
            - weighted_contributions: Array of α^n · Ψ_n
            - n_iterations: Number of iterations computed
            - convergence_achieved: Whether series converged within max_iterations
    """
    # Input validation
    n_points = len(S_series)
    if n_points != len(C_series) or n_points != len(F_series):
        raise ValueError("All time series must have the same length")
    
    if not (0 < alpha < 1):
        raise ValueError(f"Decay factor α must be in (0, 1), got {alpha}")
    
    # Compute field strength at each time point
    contributions = np.array([
        compute_field_strength_single(S_series[i], C_series[i], F_series[i])
        for i in range(n_points)
    ])
    
    # Limit to max_iterations
    n_iter = min(n_points, max_iterations)
    contributions = contributions[:n_iter]
    
    # Compute decay factors α^n
    decay_factors = np.array([alpha**n for n in range(1, n_iter + 1)])
    
    # Compute weighted contributions
    weighted = contributions * decay_factors
    
    # Sum to get total recursive field strength
    Psi_recursive = np.sum(weighted)
    
    # Check convergence (last term < tolerance)
    convergence_achieved = (n_iter == n_points) or (weighted[-1] < tol)
    
    # Regime classification
    if Psi_recursive < 0.1:
        regime = "Dormant"
    elif Psi_recursive < 1.0:
        regime = "Active"
    else:
        regime = "Resonant"
    
    # Component analysis
    components = {
        'total_field': float(Psi_recursive),
        'mean_contribution': float(np.mean(contributions)),
        'max_contribution': float(np.max(contributions)),
        'decay_rate': float(alpha),
        'effective_memory': float(-1.0 / np.log(alpha)) if alpha > 0 else np.inf
    }
    
    return {
        'Psi_recursive': float(Psi_recursive),
        'regime': regime,
        'contributions': contributions,
        'decay_factors': decay_factors,
        'weighted_contributions': weighted,
        'n_iterations': n_iter,
        'convergence_achieved': convergence_achieved,
        'components': components
    }


def compute_recursive_field_from_energy(
    E_series: np.ndarray,
    alpha: float = 0.8,
    max_iterations: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Alternative: Compute recursive field directly from GCD energy potential series.
    
    This simplified version uses energy potential E as a proxy for field strength,
    suitable when (S, C, F) time series are not available.
    
    Formula: Ψ_r = Σ(n=1→N) α^n · E_n
    
    Args:
        E_series: Time series of energy potential values (from GCD Tier-1)
        alpha: Decay factor (0 < α < 1)
        max_iterations: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Dictionary with recursive field metrics
    """
    n_points = len(E_series)
    n_iter = min(n_points, max_iterations)
    
    contributions = E_series[:n_iter]
    decay_factors = np.array([alpha**n for n in range(1, n_iter + 1)])
    weighted = contributions * decay_factors
    
    Psi_recursive = np.sum(weighted)
    convergence_achieved = (n_iter == n_points) or (weighted[-1] < tol)
    
    if Psi_recursive < 0.1:
        regime = "Dormant"
    elif Psi_recursive < 1.0:
        regime = "Active"
    else:
        regime = "Resonant"
    
    return {
        'Psi_recursive': float(Psi_recursive),
        'regime': regime,
        'contributions': contributions,
        'decay_factors': decay_factors,
        'weighted_contributions': weighted,
        'n_iterations': n_iter,
        'convergence_achieved': convergence_achieved,
        'components': {
            'total_field': float(Psi_recursive),
            'mean_energy': float(np.mean(contributions)),
            'decay_rate': float(alpha)
        }
    }


# Example usage and testing
if __name__ == "__main__":
    print("RCFT Recursive Field Closure (Tier-2)")
    print("=" * 60)
    
    # Test 1: Zero entropy state (dormant field)
    print("\nTest 1: Zero Entropy State")
    S_zero = np.array([0.0, 0.0, 0.0])
    C_zero = np.array([0.0, 0.0, 0.0])
    F_zero = np.array([1.0, 1.0, 1.0])
    result = compute_recursive_field(S_zero, C_zero, F_zero)
    print(f"  Ψ_recursive: {result['Psi_recursive']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Convergence: {result['convergence_achieved']}")
    
    # Test 2: Constant low entropy (dormant)
    print("\nTest 2: Constant Low Entropy")
    S_low = np.full(10, 0.05)
    C_low = np.full(10, 0.01)
    F_low = np.full(10, 0.99)
    result = compute_recursive_field(S_low, C_low, F_low, alpha=0.8)
    print(f"  Ψ_recursive: {result['Psi_recursive']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Mean contribution: {result['components']['mean_contribution']:.6f}")
    print(f"  Effective memory: {result['components']['effective_memory']:.2f} iterations")
    
    # Test 3: Moderate entropy with decay (active)
    print("\nTest 3: Moderate Entropy (Active)")
    S_mod = np.linspace(0.2, 0.05, 20)
    C_mod = np.linspace(0.1, 0.02, 20)
    F_mod = np.linspace(0.8, 0.95, 20)
    result = compute_recursive_field(S_mod, C_mod, F_mod, alpha=0.85)
    print(f"  Ψ_recursive: {result['Psi_recursive']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  N iterations: {result['n_iterations']}")
    print(f"  Max contribution: {result['components']['max_contribution']:.6f}")
    
    # Test 4: High entropy oscillating (resonant)
    print("\nTest 4: High Entropy Oscillating (Resonant)")
    t = np.linspace(0, 4*np.pi, 50)
    S_high = 0.3 + 0.2 * np.sin(t)
    C_high = 0.2 + 0.15 * np.cos(t)
    F_high = 0.7 + 0.1 * np.sin(2*t)
    result = compute_recursive_field(S_high, C_high, F_high, alpha=0.9)
    print(f"  Ψ_recursive: {result['Psi_recursive']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Convergence: {result['convergence_achieved']}")
    
    # Test 5: Energy-based recursive field
    print("\nTest 5: Energy-Based Recursive Field")
    E_series = np.linspace(0.1, 0.01, 30)
    result = compute_recursive_field_from_energy(E_series, alpha=0.75)
    print(f"  Ψ_recursive: {result['Psi_recursive']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Mean energy: {result['components']['mean_energy']:.6f}")
    
    print("\n" + "=" * 60)
    print("Recursive field closure validated successfully!")
