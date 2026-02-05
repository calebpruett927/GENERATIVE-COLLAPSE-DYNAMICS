"""
UMCP Security Domain - Tier-2 Overlay: Behavior Profiler

Builds and analyzes behavior profiles for anomaly detection.
This is a DIAGNOSTIC overlay - it reads signal history but results
are Tier-2 diagnostics that inform threat classification.

Tier-2 rules:
    - May read Tier-1 invariant history
    - Builds statistical profiles from frozen historical data
    - Results are DIAGNOSTIC - they inform but don't alter Tier-1
    - Profile updates between runs require seam accounting

Behavior Analysis:
    - Baseline computation (rolling statistics)
    - Deviation detection (from baseline)
    - Trend analysis (improving, stable, declining)
    - Anomaly scoring (how far from normal)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class BehaviorTrend(Enum):
    """Behavior trend types."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class AnomalyLevel(Enum):
    """Anomaly severity levels."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class BehaviorProfile:
    """Baseline behavior profile."""
    mean: np.ndarray
    std: np.ndarray
    min_vals: np.ndarray
    max_vals: np.ndarray
    samples: int
    window: int


@dataclass
class DeviationResult:
    """Deviation analysis result."""
    deviation_score: float
    z_scores: np.ndarray
    max_deviation: float
    anomaly_level: AnomalyLevel
    deviating_signals: List[int]


def compute_baseline_profile(
    signal_history: np.ndarray,
    window: Optional[int] = None,
    exclude_anomalies: bool = True,
    anomaly_threshold: float = 0.4,
    weights: Optional[np.ndarray] = None
) -> BehaviorProfile:
    """
    Compute baseline behavior profile from signal history.
    
    Args:
        signal_history: Shape (T, n) - signal history
        window: Lookback window (None = use all)
        exclude_anomalies: Whether to exclude low-trust samples
        anomaly_threshold: Trust threshold for exclusion
        weights: Weights for trust computation
        
    Returns:
        BehaviorProfile with baseline statistics
    """
    T, n = signal_history.shape
    
    if weights is None:
        weights = np.ones(n) / n
    
    # Apply window
    if window is not None:
        signal_history = signal_history[-window:]
    
    # Optionally exclude anomalies
    if exclude_anomalies:
        trust_values = np.sum(signal_history * weights, axis=1)
        mask = trust_values >= anomaly_threshold
        if np.sum(mask) < 3:
            # Not enough good samples, use all
            mask = np.ones(len(signal_history), dtype=bool)
        signal_history = signal_history[mask]
    
    return BehaviorProfile(
        mean=np.mean(signal_history, axis=0),
        std=np.std(signal_history, axis=0, ddof=1),
        min_vals=np.min(signal_history, axis=0),
        max_vals=np.max(signal_history, axis=0),
        samples=len(signal_history),
        window=window or T
    )


def compute_deviation(
    current_signals: np.ndarray,
    baseline: BehaviorProfile,
    weights: Optional[np.ndarray] = None
) -> DeviationResult:
    """
    Compute deviation from baseline.
    
    Args:
        current_signals: Current signal values, shape (n,)
        baseline: Baseline profile
        weights: Signal weights for aggregation
        
    Returns:
        DeviationResult with deviation metrics
    """
    n = len(current_signals)
    
    if weights is None:
        weights = np.ones(n) / n
    
    # Compute z-scores (how many std from mean)
    # Handle zero std (constant signals)
    std_safe = np.where(baseline.std > 1e-8, baseline.std, 1e-8)
    z_scores = (current_signals - baseline.mean) / std_safe
    
    # Weighted absolute deviation score
    abs_z = np.abs(z_scores)
    deviation_score = float(np.sum(weights * abs_z))
    
    # Maximum deviation
    max_deviation = float(np.max(abs_z))
    
    # Find signals with significant deviation (|z| > 2)
    deviating_signals = list(np.where(abs_z > 2.0)[0])
    
    # Determine anomaly level
    if deviation_score < 1.0:
        level = AnomalyLevel.NONE
    elif deviation_score < 2.0:
        level = AnomalyLevel.LOW
    elif deviation_score < 3.0:
        level = AnomalyLevel.MEDIUM
    elif deviation_score < 4.0:
        level = AnomalyLevel.HIGH
    else:
        level = AnomalyLevel.CRITICAL
    
    return DeviationResult(
        deviation_score=deviation_score,
        z_scores=z_scores,
        max_deviation=max_deviation,
        anomaly_level=level,
        deviating_signals=deviating_signals
    )


def analyze_trend(
    value_history: np.ndarray,
    window: int = 5
) -> Tuple[BehaviorTrend, Dict[str, float]]:
    """
    Analyze trend in a value series.
    
    Args:
        value_history: 1D array of values over time
        window: Window for trend analysis
        
    Returns:
        Tuple of (BehaviorTrend, statistics dict)
    """
    if len(value_history) < 3:
        return BehaviorTrend.UNKNOWN, {"samples": len(value_history)}
    
    # Use recent window
    recent = value_history[-min(window, len(value_history)):]
    
    # Compute statistics
    mean_val = float(np.mean(recent))
    std_val = float(np.std(recent))
    
    # Linear regression for trend
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    
    # Trend classification
    slope_threshold = std_val / len(recent) if std_val > 0 else 0.01
    
    if slope > slope_threshold:
        trend = BehaviorTrend.IMPROVING
    elif slope < -slope_threshold:
        trend = BehaviorTrend.DECLINING
    else:
        # Check volatility
        if std_val > mean_val * 0.3:
            trend = BehaviorTrend.VOLATILE
        else:
            trend = BehaviorTrend.STABLE
    
    stats = {
        "slope": float(slope),
        "intercept": float(intercept),
        "mean": mean_val,
        "std": std_val,
        "samples": len(recent)
    }
    
    return trend, stats


def profile_behavior_series(
    signal_history: np.ndarray,
    baseline_window: int = 64,
    trend_window: int = 5,
    weights: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Profile behavior over entire signal series.
    
    Args:
        signal_history: Shape (T, n) - full signal history
        baseline_window: Window for baseline computation
        trend_window: Window for trend analysis
        weights: Signal weights
        
    Returns:
        List of behavior profile results per timestep
    """
    T, n = signal_history.shape
    
    if weights is None:
        weights = np.ones(n) / n
    
    results = []
    
    for t in range(T):
        # Build baseline from history up to t
        if t < 3:
            # Not enough history
            results.append({
                "t": t + 1,
                "deviation_score": 0.0,
                "anomaly_level": AnomalyLevel.NONE.value,
                "trend": BehaviorTrend.UNKNOWN.value,
                "note": "Insufficient history"
            })
            continue
        
        # Compute baseline from historical data
        history = signal_history[:t]
        start_idx = max(0, t - baseline_window)
        baseline_data = signal_history[start_idx:t]
        
        baseline = compute_baseline_profile(
            baseline_data,
            window=None,
            exclude_anomalies=True,
            weights=weights
        )
        
        # Compute deviation
        deviation = compute_deviation(
            signal_history[t],
            baseline,
            weights
        )
        
        # Analyze trust trend
        trust_history = np.sum(history * weights, axis=1)
        trend, trend_stats = analyze_trend(trust_history, trend_window)
        
        results.append({
            "t": t + 1,
            "deviation_score": deviation.deviation_score,
            "max_deviation": deviation.max_deviation,
            "anomaly_level": deviation.anomaly_level.value,
            "deviating_signals": deviation.deviating_signals,
            "trend": trend.value,
            "trend_slope": trend_stats.get("slope", 0.0),
            "baseline_samples": baseline.samples
        })
    
    return results


def detect_behavior_anomalies(
    profile_results: List[Dict[str, Any]],
    min_duration: int = 3
) -> List[Dict[str, Any]]:
    """
    Detect anomaly events from behavior profile series.
    
    Args:
        profile_results: List of behavior profile results
        min_duration: Minimum duration for anomaly event
        
    Returns:
        List of detected anomaly events
    """
    events = []
    current_event = None
    
    anomaly_levels = ["MEDIUM", "HIGH", "CRITICAL"]
    
    for result in profile_results:
        level = result.get("anomaly_level", "NONE")
        is_anomaly = level in anomaly_levels
        
        if is_anomaly:
            if current_event is None:
                current_event = {
                    "start": result["t"],
                    "end": result["t"],
                    "max_level": level,
                    "max_deviation": result.get("deviation_score", 0)
                }
            else:
                current_event["end"] = result["t"]
                if anomaly_levels.index(level) > anomaly_levels.index(current_event["max_level"]):
                    current_event["max_level"] = level
                current_event["max_deviation"] = max(
                    current_event["max_deviation"],
                    result.get("deviation_score", 0)
                )
        else:
            if current_event is not None:
                duration = current_event["end"] - current_event["start"] + 1
                if duration >= min_duration:
                    current_event["duration"] = duration
                    events.append(current_event)
                current_event = None
    
    # Handle ongoing anomaly
    if current_event is not None:
        duration = current_event["end"] - current_event["start"] + 1
        if duration >= min_duration:
            current_event["duration"] = duration
            current_event["ongoing"] = True
            events.append(current_event)
    
    return events


if __name__ == "__main__":
    # Example: profile behavior
    np.random.seed(42)
    
    # Simulate signal history with anomaly
    T, n = 25, 4
    signal_history = np.zeros((T, n))
    
    # Normal period
    for t in range(9):
        signal_history[t] = [0.9, 0.85, 0.88, 0.87] + np.random.randn(n) * 0.02
    
    # Anomaly period
    for t in range(9, 19):
        signal_history[t] = [0.35, 0.30, 0.28, 0.32] + np.random.randn(n) * 0.03
    
    # Recovery
    for t in range(19, 25):
        recovery = (t - 18) * 0.08
        signal_history[t] = [0.35 + recovery, 0.30 + recovery, 0.28 + recovery, 0.32 + recovery]
    
    signal_history = np.clip(signal_history, 0, 1)
    
    # Profile behavior
    weights = np.array([0.4, 0.2, 0.25, 0.15])
    results = profile_behavior_series(signal_history, weights=weights)
    
    print("Behavior Profile Analysis:")
    for r in results:
        print(f"  t={r['t']}: deviation={r['deviation_score']:.2f}, "
              f"level={r['anomaly_level']}, trend={r['trend']}")
    
    # Detect anomaly events
    events = detect_behavior_anomalies(results)
    print(f"\nDetected Anomaly Events: {len(events)}")
    for e in events:
        print(f"  {e}")
