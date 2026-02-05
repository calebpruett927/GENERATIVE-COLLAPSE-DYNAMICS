"""
UMCP Security Domain - Tier-1 Kernel: Anomaly Return Time

Computes τ_A (Anomaly Return Time) - the security analog of τ_R (Return Time).
Measures how long until security state returns to baseline.

Formula:
    τ_A(t) = min{t - u : u ∈ D_θ(t), ||S(t) - S(u)|| ≤ η}

Where:
    - D_θ(t): Return domain generator (eligible baseline states)
    - S(t): Security signal vector at time t
    - η: Baseline proximity tolerance (eta_baseline)
    - ||·||: L2 norm (from closure)

Typed outcomes (mandatory):
    - τ_A(t) = integer: Return detected, finite delay
    - τ_A(t) = INF_ANOMALY: No return within horizon
    - τ_A(t) = UNIDENTIFIABLE: Cannot compute (insufficient data)

Tier-1 rules:
    - Pure function of frozen Tier-0 inputs
    - No back-edges, no retroactive tuning
    - Deterministic: same inputs → same outputs
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum


class TauAType(Enum):
    """Typed outcomes for τ_A."""
    FINITE = "finite"
    INF_ANOMALY = "INF_ANOMALY"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"


def compute_norm(
    v1: np.ndarray,
    v2: np.ndarray,
    norm_type: str = "L2"
) -> float:
    """
    Compute distance between two vectors.
    
    Args:
        v1, v2: Vectors to compare
        norm_type: "L1", "L2", or "Linf"
        
    Returns:
        Distance value
    """
    diff = v1 - v2
    
    if norm_type == "L1":
        return float(np.sum(np.abs(diff)))
    elif norm_type == "L2":
        return float(np.sqrt(np.sum(diff ** 2)))
    elif norm_type == "Linf":
        return float(np.max(np.abs(diff)))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def build_return_domain(
    t: int,
    signal_history: np.ndarray,
    horizon: int = 64,
    exclude_anomalies: bool = True,
    anomaly_threshold: float = 0.4,
    weights: Optional[np.ndarray] = None
) -> List[int]:
    """
    Build return domain D_θ(t) - eligible baseline indices.
    
    Args:
        t: Current time index (0-based)
        signal_history: Shape (T, n) - full signal history
        horizon: Maximum lookback (H_rec)
        exclude_anomalies: Whether to exclude known anomalous states
        anomaly_threshold: Trust threshold below which state is anomalous
        weights: Weights for computing trust (if exclude_anomalies)
        
    Returns:
        List of eligible indices u ∈ D_θ(t)
    """
    if weights is None:
        weights = np.ones(signal_history.shape[1]) / signal_history.shape[1]
    
    # Start with all indices in horizon
    start_idx = max(0, t - horizon)
    candidates = list(range(start_idx, t))  # Exclude current time
    
    if not exclude_anomalies:
        return candidates
    
    # Filter out anomalous states (low trust)
    eligible = []
    for u in candidates:
        T_u = float(np.sum(weights * signal_history[u]))
        if T_u >= anomaly_threshold:
            eligible.append(u)
    
    return eligible


def compute_anomaly_return(
    t: int,
    signal_history: np.ndarray,
    eta: float = 0.01,
    horizon: int = 64,
    norm_type: str = "L2",
    exclude_anomalies: bool = True,
    anomaly_threshold: float = 0.4,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute Anomaly Return Time τ_A(t).
    
    Args:
        t: Current time index (0-based)
        signal_history: Shape (T, n) - full signal history
        eta: Baseline proximity tolerance (η)
        horizon: Maximum lookback (H_rec)
        norm_type: Norm for distance computation
        exclude_anomalies: Whether to exclude anomalous states from domain
        anomaly_threshold: Trust threshold for anomaly detection
        weights: Weights for trust computation
        
    Returns:
        Dict with tau_A, type, and diagnostics
    """
    # Validate inputs
    if t < 0 or t >= signal_history.shape[0]:
        return {
            "tau_A": "UNIDENTIFIABLE",
            "type": TauAType.UNIDENTIFIABLE.value,
            "reason": f"Invalid time index: {t}"
        }
    
    if t == 0:
        # No history to return to
        return {
            "tau_A": None,
            "type": TauAType.FINITE.value,
            "reason": "First sample, no return computation",
            "t": t + 1
        }
    
    current_signals = signal_history[t]
    
    # Build return domain
    D_theta = build_return_domain(
        t, signal_history, horizon,
        exclude_anomalies, anomaly_threshold, weights
    )
    
    if len(D_theta) == 0:
        return {
            "tau_A": "INF_ANOMALY",
            "type": TauAType.INF_ANOMALY.value,
            "reason": "Empty return domain (no eligible baseline states)",
            "t": t + 1,
            "horizon": horizon
        }
    
    # Find return candidates: U_θ(t) = {u ∈ D_θ(t) : ||S(t) - S(u)|| ≤ η}
    return_candidates = []
    distances = {}
    
    for u in D_theta:
        dist = compute_norm(current_signals, signal_history[u], norm_type)
        distances[u] = dist
        
        if dist <= eta:
            return_candidates.append(u)
    
    if len(return_candidates) == 0:
        # No return within tolerance
        closest_u = min(D_theta, key=lambda u: distances[u])
        closest_dist = distances[closest_u]
        
        return {
            "tau_A": "INF_ANOMALY",
            "type": TauAType.INF_ANOMALY.value,
            "reason": f"No return within η={eta} (closest: {closest_dist:.6f})",
            "t": t + 1,
            "closest_distance": closest_dist,
            "closest_candidate": closest_u + 1,  # 1-indexed
            "eta": eta,
            "horizon": horizon,
            "domain_size": len(D_theta)
        }
    
    # τ_A(t) = min{t - u : u ∈ return_candidates}
    tau_A = min(t - u for u in return_candidates)
    best_u = t - tau_A
    
    return {
        "tau_A": tau_A,
        "type": TauAType.FINITE.value,
        "t": t + 1,  # 1-indexed
        "return_to": best_u + 1,  # 1-indexed
        "distance": distances[best_u],
        "eta": eta,
        "num_candidates": len(return_candidates),
        "domain_size": len(D_theta)
    }


def compute_anomaly_return_series(
    signal_history: np.ndarray,
    eta: float = 0.01,
    horizon: int = 64,
    norm_type: str = "L2",
    exclude_anomalies: bool = True,
    anomaly_threshold: float = 0.4,
    weights: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Compute Anomaly Return Time for entire series.
    
    Args:
        signal_history: Shape (T, n) - full signal history
        eta: Baseline proximity tolerance
        horizon: Maximum lookback
        norm_type: Norm type
        exclude_anomalies: Whether to exclude anomalous states
        anomaly_threshold: Trust threshold for anomaly detection
        weights: Weights for trust computation
        
    Returns:
        List of τ_A results per timestep
    """
    results = []
    for t in range(signal_history.shape[0]):
        result = compute_anomaly_return(
            t, signal_history, eta, horizon,
            norm_type, exclude_anomalies, anomaly_threshold, weights
        )
        results.append(result)
    return results


def detect_anomaly_events(
    tau_A_series: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect anomaly events (transitions to/from INF_ANOMALY).
    
    Args:
        tau_A_series: List of τ_A results
        
    Returns:
        List of anomaly events with start, end, duration
    """
    events = []
    in_anomaly = False
    event_start = None
    
    for i, result in enumerate(tau_A_series):
        is_anomaly = result.get("type") == TauAType.INF_ANOMALY.value
        
        if is_anomaly and not in_anomaly:
            # Anomaly starts
            in_anomaly = True
            event_start = i + 1  # 1-indexed
            
        elif not is_anomaly and in_anomaly:
            # Anomaly ends (return detected)
            in_anomaly = False
            if event_start is not None:
                events.append({
                    "type": "anomaly",
                    "start": event_start,
                    "end": i + 1,  # 1-indexed
                    "duration": (i + 1) - event_start,
                    "recovery_tau_A": result.get("tau_A")
                })
    
    # Handle ongoing anomaly at end of series
    if in_anomaly and event_start is not None:
        events.append({
            "type": "anomaly_ongoing",
            "start": event_start,
            "end": None,
            "duration": len(tau_A_series) - event_start + 1,
            "recovery_tau_A": None
        })
    
    return events


if __name__ == "__main__":
    # Example: compute anomaly return time
    # Simulate: normal → anomaly → recovery
    np.random.seed(42)
    
    # 25 timesteps, 4 signals
    signal_history = np.zeros((25, 4))
    
    # Normal: t=0-8 (high signals)
    for t in range(9):
        signal_history[t] = [0.95 - 0.01*t, 0.88 - 0.01*t, 0.92 - 0.01*t, 0.90 - 0.01*t]
    
    # Anomaly: t=9-18 (low signals)
    for t in range(9, 19):
        signal_history[t] = [0.40, 0.30, 0.25, 0.35]
    
    # Recovery: t=19-24 (signals increase)
    for t in range(19, 25):
        recovery = (t - 18) * 0.1
        signal_history[t] = [0.40 + recovery, 0.30 + recovery, 0.25 + recovery, 0.35 + recovery]
    
    weights = np.array([0.4, 0.2, 0.25, 0.15])
    
    # Compute τ_A series
    tau_A_series = compute_anomaly_return_series(
        signal_history,
        eta=0.01,
        horizon=64,
        weights=weights
    )
    
    print("Anomaly Return Time (τ_A) Series:")
    for result in tau_A_series:
        t = result.get("t", "?")
        tau_A = result.get("tau_A", "?")
        type_str = result.get("type", "?")
        print(f"  t={t}: τ_A={tau_A} ({type_str})")
    
    # Detect anomaly events
    events = detect_anomaly_events(tau_A_series)
    print("\nAnomaly Events:")
    for event in events:
        print(f"  {event}")
