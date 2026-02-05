"""
UMCP Security Domain - Tier-1 Kernel: Trust Fidelity

Computes T (Trust Fidelity) and θ (Threat Drift) from security signals.
Analogous to F (Fidelity) and ω (Drift) in UMCP core.

Formula:
    T = Σ w_i · s_i     (weighted sum of security signals)
    θ = 1 - T           (threat drift)

Tier-1 rules:
    - Pure function of frozen Tier-0 inputs
    - No back-edges, no retroactive tuning
    - Deterministic: same inputs → same outputs
"""

import numpy as np


def compute_trust_fidelity(signals: np.ndarray, weights: np.ndarray, epsilon: float = 1e-8) -> dict[str, float]:
    """
    Compute Trust Fidelity (T) and Threat Drift (θ).

    Args:
        signals: Security signals s_i ∈ [0, 1], shape (n,)
        weights: Weights w_i ≥ 0, Σw_i = 1, shape (n,)
        epsilon: Numerical safety for boundary

    Returns:
        Dict with T, theta, and per-signal contributions

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

    # Clip for numerical safety (OOR handling)
    signals_safe = np.clip(signals, epsilon, 1 - epsilon)

    # T = Σ w_i · s_i
    contributions = weights * signals_safe
    T = float(np.sum(contributions))

    # θ = 1 - T
    theta = 1.0 - T

    return {
        "T": T,
        "theta": theta,
        "signal_contributions": contributions.tolist(),
        "signals_used": signals_safe.tolist(),
        "weights_used": weights.tolist(),
    }


def compute_trust_fidelity_series(
    signal_series: np.ndarray, weights: np.ndarray, epsilon: float = 1e-8
) -> list[dict[str, float]]:
    """
    Compute Trust Fidelity over time series.

    Args:
        signal_series: Shape (T, n) - T timesteps, n signals
        weights: Shape (n,) - weights for each signal
        epsilon: Numerical safety

    Returns:
        List of T dictionaries with T, theta per timestep
    """
    results = []
    for t in range(signal_series.shape[0]):
        result = compute_trust_fidelity(signal_series[t], weights, epsilon)
        result["t"] = t + 1  # 1-indexed time
        results.append(result)
    return results


def classify_trust_status(T: float, tau_A: int | str | None, thresholds: dict[str, float] | None = None) -> str:
    """
    Classify trust status based on T and τ_A.

    Args:
        T: Trust Fidelity
        tau_A: Anomaly return time (None or int for finite, "INF_ANOMALY" for infinite)
        thresholds: Classification thresholds

    Returns:
        Status string: TRUSTED, SUSPICIOUS, BLOCKED, or NON_EVALUABLE
    """
    if thresholds is None:
        thresholds = {"trusted": 0.8, "suspicious": 0.4, "max_tau_A": 32}

    # Handle non-evaluable case
    if T is None or tau_A == "UNIDENTIFIABLE":
        return "NON_EVALUABLE"

    # No return = no trust credit
    if tau_A == "INF_ANOMALY":
        return "BLOCKED"

    # Low trust = blocked
    if thresholds["suspicious"] > T:
        return "BLOCKED"

    # High trust with timely return = trusted
    if thresholds["trusted"] <= T and (
        tau_A is None or (isinstance(tau_A, (int, float)) and tau_A <= thresholds["max_tau_A"])
    ):
        return "TRUSTED"

    # Everything else is suspicious
    return "SUSPICIOUS"


if __name__ == "__main__":
    # Example: compute trust fidelity for sample signals
    signals = np.array([0.95, 0.88, 0.92, 0.90])
    weights = np.array([0.4, 0.2, 0.25, 0.15])  # From threat_patterns.v1.yaml

    result = compute_trust_fidelity(signals, weights)
    print(f"Trust Fidelity (T): {result['T']:.4f}")
    print(f"Threat Drift (θ): {result['theta']:.4f}")
    print(f"Status: {classify_trust_status(result['T'], 1)}")
