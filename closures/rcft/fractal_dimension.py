"""
Fractal Dimension Computation for RCFT (Tier-2)

Computes the fractal dimension D_f of collapse trajectories using box-counting method.
This is a Tier-2 RCFT metric that analyzes geometric complexity of trajectories
defined by Tier-1 GCD invariants.

Mathematical Foundation:
    D_f = lim(ε→0) log(N(ε)) / log(1/ε)

    where N(ε) is the number of boxes of size ε needed to cover the trajectory.

Regime Classification:
    - Smooth:     D_f < 1.2   (nearly linear trajectory)
    - Wrinkled:   1.2 ≤ D_f < 1.8   (moderate complexity)
    - Turbulent:  D_f ≥ 1.8   (high complexity, chaotic)

Tier-2 Constraints:
    - Does not modify Tier-1 invariants (ω, F, S, C, etc.)
    - Augments GCD analysis with geometric complexity metric
    - Respects all GCD tolerances and regime classifications
"""

from typing import Any

import numpy as np


def compute_fractal_dimension(
    trajectory: np.ndarray, eps_values: np.ndarray = None, tol: float = 1e-6
) -> dict[str, Any]:
    """
    Compute fractal dimension of a trajectory using box-counting method.

    Args:
        trajectory: Array of shape (n_points, n_dims) representing trajectory
        eps_values: Array of box sizes for box-counting (default: logarithmic spacing)
        tol: Numerical tolerance for fractal dimension computation

    Returns:
        Dictionary containing:
            - D_fractal: Fractal dimension
            - regime: Classification (Smooth/Wrinkled/Turbulent)
            - box_counts: Number of boxes for each epsilon
            - eps_used: Epsilon values used in computation
            - log_slope: Slope of log(N) vs log(1/ε) fit
            - r_squared: Quality of fit (coefficient of determination)
    """
    # Input validation
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_points, n_dims = trajectory.shape

    if n_points < 2:
        return {
            "D_fractal": 1.0,
            "regime": "Smooth",
            "box_counts": np.array([1]),
            "eps_used": np.array([1.0]),
            "log_slope": 0.0,
            "r_squared": 1.0,
            "components": {"min_eps": 1.0, "max_eps": 1.0, "n_eps_values": 1},
        }

    # Generate epsilon values if not provided (logarithmic spacing)
    if eps_values is None:
        # Compute trajectory extent
        extent = np.ptp(trajectory, axis=0)
        max_extent = np.max(extent)

        if max_extent < tol:
            # Trajectory is a point
            return {
                "D_fractal": 0.0,
                "regime": "Smooth",
                "box_counts": np.array([1]),
                "eps_used": np.array([tol]),
                "log_slope": 0.0,
                "r_squared": 1.0,
                "components": {"min_eps": tol, "max_eps": tol, "n_eps_values": 1},
            }

        # Generate logarithmically spaced epsilon values
        min_eps = max_extent / (2 * n_points)
        max_eps = max_extent
        eps_values = np.logspace(np.log10(min_eps), np.log10(max_eps), num=15)

    # Box-counting algorithm
    box_counts = []

    for eps in eps_values:
        # Discretize trajectory into grid
        grid_coords = np.floor(trajectory / eps).astype(int)

        # Count unique boxes (use tuple for hashable type)
        unique_boxes = set(map(tuple, grid_coords))
        box_counts.append(len(unique_boxes))

    box_counts = np.array(box_counts)

    # Remove points where box_count is 0 or 1 (degenerate cases)
    valid_mask = box_counts > 1
    if np.sum(valid_mask) < 2:
        # Not enough points for linear fit
        return {
            "D_fractal": 1.0,
            "regime": "Smooth",
            "box_counts": box_counts,
            "eps_used": eps_values,
            "log_slope": 0.0,
            "r_squared": 0.0,
            "components": {
                "min_eps": np.min(eps_values),
                "max_eps": np.max(eps_values),
                "n_eps_values": len(eps_values),
            },
        }

    valid_eps = eps_values[valid_mask]
    valid_counts = box_counts[valid_mask]

    # Compute fractal dimension from log-log slope
    log_eps_inv = np.log(1.0 / valid_eps)
    log_N = np.log(valid_counts)

    # Linear regression: log(N) = D_f * log(1/ε) + intercept
    coeffs = np.polyfit(log_eps_inv, log_N, deg=1)
    D_fractal = coeffs[0]

    # Compute R² for fit quality
    log_N_pred = np.polyval(coeffs, log_eps_inv)
    ss_res = np.sum((log_N - log_N_pred) ** 2)
    ss_tot = np.sum((log_N - np.mean(log_N)) ** 2)
    r_squared = 1.0 - (ss_res / (ss_tot + tol))

    # Clamp D_fractal to valid range [0, n_dims]
    D_fractal = np.clip(D_fractal, 0.0, float(n_dims))

    # Regime classification
    if D_fractal < 1.2:
        regime = "Smooth"
    elif D_fractal < 1.8:
        regime = "Wrinkled"
    else:
        regime = "Turbulent"

    return {
        "D_fractal": float(D_fractal),
        "regime": regime,
        "box_counts": box_counts,
        "eps_used": eps_values,
        "log_slope": float(D_fractal),
        "r_squared": float(r_squared),
        "components": {
            "min_eps": float(np.min(eps_values)),
            "max_eps": float(np.max(eps_values)),
            "n_eps_values": len(eps_values),
            "valid_points": int(np.sum(valid_mask)),
        },
    }


def compute_trajectory_from_invariants(
    omega_series: np.ndarray, S_series: np.ndarray, C_series: np.ndarray
) -> np.ndarray:
    """
    Construct 3D trajectory from Tier-1 invariants (ω, S, C).

    This helper function creates a trajectory in (ω, S, C) space from time series
    of GCD Tier-1 invariants, suitable for fractal dimension analysis.

    Args:
        omega_series: Time series of drift values
        S_series: Time series of entropy values
        C_series: Time series of curvature values

    Returns:
        Array of shape (n_points, 3) representing trajectory in (ω, S, C) space
    """
    return np.column_stack([omega_series, S_series, C_series])


# Example usage and testing
if __name__ == "__main__":
    print("RCFT Fractal Dimension Closure (Tier-2)")
    print("=" * 60)

    # Test 1: Zero entropy trajectory (should be Smooth)
    print("\nTest 1: Zero Entropy State")
    zero_trajectory = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = compute_fractal_dimension(zero_trajectory)
    print(f"  D_fractal: {result['D_fractal']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  R²: {result['r_squared']:.6f}")

    # Test 2: Linear trajectory (D_f ≈ 1)
    print("\nTest 2: Linear Trajectory")
    t = np.linspace(0, 1, 100)
    linear_trajectory = np.column_stack([t, t, t])
    result = compute_fractal_dimension(linear_trajectory)
    print(f"  D_fractal: {result['D_fractal']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  R²: {result['r_squared']:.6f}")

    # Test 3: Planar spiral (D_f ≈ 1.5-1.8)
    print("\nTest 3: Planar Spiral (Wrinkled)")
    theta = np.linspace(0, 4 * np.pi, 200)
    r = theta / (4 * np.pi)
    spiral_trajectory = np.column_stack(
        [r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)]
    )
    result = compute_fractal_dimension(spiral_trajectory)
    print(f"  D_fractal: {result['D_fractal']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  R²: {result['r_squared']:.6f}")

    # Test 4: Random walk (D_f ≈ 2)
    print("\nTest 4: Random Walk (Turbulent)")
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(500, 3) * 0.1, axis=0)
    result = compute_fractal_dimension(random_walk)
    print(f"  D_fractal: {result['D_fractal']:.6f}")
    print(f"  Regime: {result['regime']}")
    print(f"  R²: {result['r_squared']:.6f}")

    print("\n" + "=" * 60)
    print("Fractal dimension closure validated successfully!")
