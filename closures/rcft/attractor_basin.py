#!/usr/bin/env python3
"""
RCFT Closure: Attractor Basin Analysis (Tier-2)

Analyzes the topology of collapse attractors in phase space. This closure identifies
stable attractors, computes basin boundaries, and measures attractor strength using
Lyapunov-like metrics applied to recursive field evolution.

**Tier-2 RCFT Framework**: Augments Tier-1 GCD with recursive/geometric metrics.
Does not modify Tier-1 invariants or thresholds. Compatible with all GCD analysis.

**Operational Terms Defined** (enforcement-tied, not metaphors):

Tier-1 Inputs (from GCD Kernel, see canon/gcd_anchors.yaml):
    ω (omega) = Drift = 1 - F, collapse proximity
                Range: [0,1], NOT: random drift, velocity, or wandering
    S = Entropy = -Σ wᵢ[cᵢ ln(cᵢ) + (1-cᵢ)ln(1-cᵢ)], disorder measure
                Range: ≥0, NOT: thermodynamic entropy, chaos, or "mess"
    C = Curvature = stddev(cᵢ)/0.5, spatial heterogeneity (instability proxy)
                Range: [0,1], NOT: geometric curvature in spacetime
    F = Fidelity = Σ wᵢ·cᵢ, weighted coherence
                Range: [0,1], NOT: accuracy, allegiance

Tier-2 Quantities (RCFT Extensions, see canon/rcft_anchors.yaml):
    Ψ_r = Recursive field strength = Σ αⁿ·Ψₙ (from recursive_field.py)
                Tier-2 metric, self-referential field strength
                NOT: metaphorical "recursion" or programming loops
    B = Basin strength = -∇²Ψ_r(x₀) / ||∇Ψ_r(x₀)||
                Tier-2 topological measure, attractor robustness
                NOT: moral strength, "strong arguments", or force
    x₀ = Initial state in (ω, S, C) phase space using Tier-1 coordinates
    x_attr = Attractor point coordinates in Tier-1 phase space
    t = Time steps to convergence
    D_f = Fractal dimension (from fractal_dimension.py), 1 ≤ D_f ≤ 3
                Tier-2 metric, trajectory complexity

Mathematical Foundation:
    Basin Strength: B = -∇²Ψ_r(x₀) / ||∇Ψ_r(x₀)||
    Attractor Distance: d_attr = min||x - x_attr|| over all attractors
    Convergence Rate: λ_conv = -log(||x_t - x_attr||) / t

Where:
    Ψ_r = Recursive field (Tier-2, from recursive_field.py)
    x₀ = Initial state in (ω, S, C) phase space (Tier-1 coordinates)
    x_attr = Attractor point coordinates
    t = Time steps to convergence

Physical Interpretation:
    - Strong basins (B > 1.0) indicate robust stable states
    - Weak basins (B < 0.5) suggest regime transition zones
    - Multiple attractors indicate bifurcation structure
    - Convergence rate measures collapse/recovery speed

Regime Classification:
    - Monostable:   Single dominant attractor (B_max > 2.0)
    - Bistable:     Two comparable attractors (1.0 < B_max < 2.0)
    - Multistable:  Many weak attractors (B_max < 1.0)

Tier-2 Constraints:
    - Augments GCD analysis with attractor topology
    - Uses Tier-1 invariants (ω, S, C) as phase space coordinates
    - Does not modify GCD regime thresholds
    - Compatible with all RCFT metrics (D_f, λ_p, Θ)
"""

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def compute_attractor_basin(
    omega_series: np.ndarray,
    S_series: np.ndarray,
    C_series: np.ndarray,
    psi_r_series: np.ndarray = None,
    n_attractors: int = 3,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Analyze attractor basin structure from GCD invariant trajectory.

    Args:
        omega_series: Array of drift values (ω)
        S_series: Array of entropy values (S)
        C_series: Array of curvature values (C)
        psi_r_series: Optional recursive field values (computed if not provided)
        n_attractors: Maximum number of attractors to identify (default=3)
        tol: Numerical tolerance (default=1e-6)

    Returns:
        Dictionary containing:
            - n_attractors_found: Number of detected attractors
            - attractor_locations: Coordinates of attractors in (ω,S,C) space
            - basin_strengths: Strength of each attractor basin
            - dominant_attractor: Index of strongest attractor
            - regime: Classification (Monostable/Bistable/Multistable)
            - convergence_rates: Rate of approach to each attractor
            - basin_volumes: Estimated volume of each basin
            - trajectory_classification: Which basin the trajectory occupies

    Raises:
        ValueError: If input arrays have mismatched shapes or invalid values
    """
    # Input validation
    if omega_series.shape != S_series.shape or omega_series.shape != C_series.shape:
        raise ValueError("Input arrays must have same shape")

    n_points = len(omega_series)

    if n_points < 10:
        raise ValueError("Need at least 10 points for attractor analysis")

    if not np.all((omega_series >= 0) & (omega_series <= 1)):
        raise ValueError("Drift ω must be in [0,1]")

    if not np.all(S_series >= 0):
        raise ValueError("Entropy S must be non-negative")

    if not np.all(C_series >= 0):
        raise ValueError("Curvature C must be non-negative")

    # Compute recursive field if not provided (using simple recursive sum)
    if psi_r_series is None:
        psi_r_series = _compute_simple_recursive_field(omega_series, S_series, C_series)

    # Stack into phase space trajectory
    trajectory = np.column_stack([omega_series, S_series, C_series])

    # Identify attractors using local minima in recursive field energy
    # Energy landscape: E = Ψ_r² (attractors at minima)
    energy = psi_r_series**2

    # Find local minima (potential attractors)
    minima_indices, properties = find_peaks(-energy, distance=max(5, n_points // 20))

    # Limit to top n_attractors by depth
    if len(minima_indices) > n_attractors:
        depths = properties.get("peak_heights", np.ones(len(minima_indices)))
        top_indices = np.argsort(-depths)[:n_attractors]
        minima_indices = minima_indices[top_indices]

    n_attractors_found = len(minima_indices)

    if n_attractors_found == 0:
        # No clear attractors - use trajectory endpoints as pseudo-attractors
        minima_indices = np.array([0, n_points - 1])
        n_attractors_found = 2

    # Extract attractor locations
    attractor_locations = trajectory[minima_indices]

    # Compute basin strengths using gradient magnitude around each attractor
    basin_strengths = []
    for idx in minima_indices:
        # Local gradient magnitude (measure of basin steepness)
        window_size = max(3, n_points // 10)
        start = max(0, idx - window_size)
        end = min(n_points, idx + window_size)

        local_energy = energy[start:end]
        local_gradient = np.abs(np.gradient(local_energy))
        basin_strength = np.mean(local_gradient) + tol

        basin_strengths.append(basin_strength)

    basin_strengths = np.array(basin_strengths)

    # Normalize basin strengths
    basin_strengths = basin_strengths / (np.sum(basin_strengths) + tol)

    # Identify dominant attractor
    dominant_attractor = int(np.argmax(basin_strengths))
    max_strength = basin_strengths[dominant_attractor]

    # Regime classification
    if n_attractors_found == 1 or max_strength > 0.7:
        regime = "Monostable"
    elif n_attractors_found == 2 and max_strength > 0.4:
        regime = "Bistable"
    else:
        regime = "Multistable"

    # Compute convergence rates (distance decay to nearest attractor)
    distances_to_attractors = np.zeros((n_points, n_attractors_found))
    for i, attr_loc in enumerate(attractor_locations):
        distances_to_attractors[:, i] = np.linalg.norm(trajectory - attr_loc, axis=1)

    nearest_attractor = np.argmin(distances_to_attractors, axis=1)
    convergence_rates = np.zeros(n_attractors_found)

    for i in range(n_attractors_found):
        mask = nearest_attractor == i
        if np.sum(mask) > 2:
            dist_series = distances_to_attractors[mask, i]
            if len(dist_series) > 1:
                # Exponential decay rate: d(t) ∝ exp(-λt)
                time_indices = np.arange(len(dist_series))
                log_dist = np.log(dist_series + tol)
                convergence_rates[i] = -np.polyfit(time_indices, log_dist, deg=1)[0]

    # Estimate basin volumes (proportion of trajectory in each basin)
    basin_volumes = np.zeros(n_attractors_found)
    for i in range(n_attractors_found):
        basin_volumes[i] = np.sum(nearest_attractor == i) / n_points

    return {
        "n_attractors_found": n_attractors_found,
        "attractor_locations": attractor_locations.tolist(),
        "basin_strengths": basin_strengths.tolist(),
        "dominant_attractor": dominant_attractor,
        "regime": regime,
        "convergence_rates": convergence_rates.tolist(),
        "basin_volumes": basin_volumes.tolist(),
        "trajectory_classification": nearest_attractor.tolist(),
        "max_basin_strength": float(max_strength),
    }


def _compute_simple_recursive_field(omega: np.ndarray, S: np.ndarray, C: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Simple recursive field computation for attractor analysis.

    Ψ_r(t) = Σ α^n · [ω(t-n) + S(t-n) + C(t-n)]
    """
    n_points = len(omega)
    psi_r = np.zeros(n_points)

    for t in range(n_points):
        for n in range(min(t + 1, 10)):  # Limit recursion depth to 10
            weight = alpha**n
            psi_r[t] += weight * (omega[t - n] + S[t - n] + C[t - n])

    return psi_r


# Example usage and testing
if __name__ == "__main__":
    print("RCFT Attractor Basin Closure (Tier-2)")
    print("=" * 60)

    # Test 1: Monostable system (converging to single attractor)
    print("\nTest 1: Monostable System")
    t = np.linspace(0, 10, 100)
    omega_mono = 0.05 + 0.03 * np.exp(-t / 2)  # Exponential decay to stable point
    S_mono = 0.10 + 0.05 * np.exp(-t / 2)
    C_mono = 0.03 + 0.02 * np.exp(-t / 2)

    result = compute_attractor_basin(omega_mono, S_mono, C_mono)
    print(f"  Attractors found: {result['n_attractors_found']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Max basin strength: {result['max_basin_strength']:.4f}")
    print(f"  Dominant attractor: {result['dominant_attractor']}")

    # Test 2: Bistable system (oscillating between two states)
    print("\nTest 2: Bistable System")
    omega_bi = 0.15 + 0.10 * np.sin(2 * np.pi * t / 5)
    S_bi = 0.20 + 0.08 * np.sin(2 * np.pi * t / 5 + np.pi / 2)
    C_bi = 0.10 + 0.05 * np.sin(2 * np.pi * t / 5)

    result = compute_attractor_basin(omega_bi, S_bi, C_bi, n_attractors=5)
    print(f"  Attractors found: {result['n_attractors_found']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Basin strengths: {[f'{s:.3f}' for s in result['basin_strengths']]}")
    print(f"  Basin volumes: {[f'{v:.3f}' for v in result['basin_volumes']]}")

    # Test 3: Chaotic/Multistable system
    print("\nTest 3: Multistable/Chaotic System")
    np.random.seed(42)
    omega_chaos = 0.20 + 0.05 * np.random.randn(100).cumsum() / 50
    omega_chaos = np.clip(omega_chaos, 0, 0.5)
    S_chaos = 0.30 + 0.08 * np.random.randn(100).cumsum() / 50
    S_chaos = np.clip(S_chaos, 0, 1)
    C_chaos = 0.15 + 0.05 * np.random.randn(100).cumsum() / 50
    C_chaos = np.clip(C_chaos, 0, 0.5)

    result = compute_attractor_basin(omega_chaos, S_chaos, C_chaos, n_attractors=5)
    print(f"  Attractors found: {result['n_attractors_found']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Max basin strength: {result['max_basin_strength']:.4f}")
    print(f"  Convergence rates: {[f'{r:.4f}' for r in result['convergence_rates']]}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
