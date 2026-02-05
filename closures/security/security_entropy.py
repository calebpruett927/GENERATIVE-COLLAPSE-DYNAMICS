"""
UMCP Security Domain - Tier-1 Kernel: Security Entropy

Computes H (Security Entropy) from security signals.
Analogous to S (Entropy) in UMCP core.

Formula:
    H = -Σ w_i [s_i ln(s_i) + (1-s_i) ln(1-s_i)]

Interpretation:
    - Low H: High certainty in classification (signals near 0 or 1)
    - High H: High uncertainty (signals near 0.5)

Tier-1 rules:
    - Pure function of frozen Tier-0 inputs
    - No back-edges, no retroactive tuning
    - Deterministic: same inputs → same outputs
"""

import numpy as np


def binary_entropy(p: float, epsilon: float = 1e-8) -> float:
    """
    Compute binary entropy h(p) = -[p ln(p) + (1-p) ln(1-p)].

    Args:
        p: Probability in (0, 1)
        epsilon: Numerical safety for log

    Returns:
        Binary entropy value ≥ 0
    """
    p_safe = np.clip(p, epsilon, 1 - epsilon)
    return -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))


def compute_security_entropy(signals: np.ndarray, weights: np.ndarray, epsilon: float = 1e-8) -> dict[str, float]:
    """
    Compute Security Entropy (H).

    Args:
        signals: Security signals s_i ∈ [0, 1], shape (n,)
        weights: Weights w_i ≥ 0, Σw_i = 1, shape (n,)
        epsilon: Numerical safety for log

    Returns:
        Dict with H and per-signal entropy

    Raises:
        ValueError: If inputs violate Tier-0 constraints
    """
    # Validate Tier-0 constraints
    if not np.allclose(np.sum(weights), 1.0, atol=1e-9):
        raise ValueError(f"Weights must sum to 1, got {np.sum(weights)}")

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    if np.any(signals < 0) or np.any(signals > 1):
        raise ValueError("Signals must be in [0, 1]")

    if len(signals) != len(weights):
        raise ValueError(f"Signal/weight length mismatch: {len(signals)} vs {len(weights)}")

    # Compute per-signal binary entropy
    per_signal_H = np.array([binary_entropy(s, epsilon) for s in signals])

    # H = Σ w_i · h(s_i)
    weighted_H = weights * per_signal_H
    H = float(np.sum(weighted_H))

    # Maximum possible entropy (all signals at 0.5)
    H_max = np.log(2)  # binary entropy at p=0.5

    # Normalized entropy (0 = certain, 1 = maximum uncertainty)
    H_normalized = H / H_max if H_max > 0 else 0.0

    return {
        "H": H,
        "H_normalized": H_normalized,
        "H_max": H_max,
        "per_signal_entropy": per_signal_H.tolist(),
        "weighted_entropy": weighted_H.tolist(),
    }


def compute_security_entropy_series(
    signal_series: np.ndarray, weights: np.ndarray, epsilon: float = 1e-8
) -> list[dict[str, float]]:
    """
    Compute Security Entropy over time series.

    Args:
        signal_series: Shape (T, n) - T timesteps, n signals
        weights: Shape (n,) - weights for each signal
        epsilon: Numerical safety

    Returns:
        List of T dictionaries with H per timestep
    """
    results = []
    for t in range(signal_series.shape[0]):
        result = compute_security_entropy(signal_series[t], weights, epsilon)
        result["t"] = t + 1  # 1-indexed time
        results.append(result)
    return results


def compute_signal_dispersion(signals: np.ndarray) -> dict[str, float]:
    """
    Compute signal dispersion D (curvature proxy).

    Formula:
        D = std_pop(signals) / 0.5

    Interpretation:
        - D = 0: All signals identical
        - D = 1: Maximum dispersion (signals at 0 and 1)

    Args:
        signals: Security signals s_i ∈ [0, 1], shape (n,)

    Returns:
        Dict with D (dispersion)
    """
    if len(signals) < 2:
        return {"D": 0.0, "std": 0.0}

    # Population standard deviation
    std = float(np.std(signals, ddof=0))

    # Normalized by max possible std for [0,1] signals
    # Max std = 0.5 when half signals at 0, half at 1
    D = std / 0.5
    D = min(D, 1.0)  # Clip to [0, 1]

    return {
        "D": D,
        "std": std,
        "mean": float(np.mean(signals)),
        "min": float(np.min(signals)),
        "max": float(np.max(signals)),
    }


if __name__ == "__main__":
    # Example: compute security entropy
    signals = np.array([0.95, 0.88, 0.92, 0.90])
    weights = np.array([0.4, 0.2, 0.25, 0.15])

    result = compute_security_entropy(signals, weights)
    print(f"Security Entropy (H): {result['H']:.4f}")
    print(f"Normalized Entropy: {result['H_normalized']:.4f}")

    dispersion = compute_signal_dispersion(signals)
    print(f"Signal Dispersion (D): {dispersion['D']:.4f}")
