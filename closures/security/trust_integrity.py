"""
UMCP Security Domain - Tier-1 Kernel: Trust Integrity

Computes σ (Log-Integrity) and TIC (Trust Integrity Composite).
Analogous to κ (Log-Integrity) and IC (Integrity Composite) in UMCP core.

Formula:
    σ = Σ w_i ln(s_i,ε)     (log-additive integrity ledger)
    TIC = exp(σ)            (geometric mean composite)

Interpretation:
    - σ turns multiplicative trust into additive accounting
    - TIC is the geometric mean of trust signals
    - Used for seam accounting across validation events

Tier-1 rules:
    - Pure function of frozen Tier-0 inputs
    - No back-edges, no retroactive tuning
    - Deterministic: same inputs → same outputs
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def compute_trust_integrity(
    signals: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    Compute Trust Integrity (σ and TIC).
    
    Args:
        signals: Security signals s_i ∈ [0, 1], shape (n,)
        weights: Weights w_i ≥ 0, Σw_i = 1, shape (n,)
        epsilon: Numerical safety for log (prevents -inf)
        
    Returns:
        Dict with sigma (σ), TIC, and per-signal contributions
        
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
    
    # Apply epsilon clipping for log safety
    signals_safe = np.clip(signals, epsilon, 1.0)
    
    # Compute log of each signal
    log_signals = np.log(signals_safe)
    
    # σ = Σ w_i ln(s_i,ε)
    weighted_log = weights * log_signals
    sigma = float(np.sum(weighted_log))
    
    # TIC = exp(σ)
    TIC = float(np.exp(sigma))
    
    # Verify TIC is in (0, 1]
    TIC = min(max(TIC, 0.0), 1.0)
    
    return {
        "sigma": sigma,
        "TIC": TIC,
        "log_signals": log_signals.tolist(),
        "weighted_log": weighted_log.tolist(),
        "signals_safe": signals_safe.tolist()
    }


def compute_trust_integrity_series(
    signal_series: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1e-8
) -> List[Dict[str, float]]:
    """
    Compute Trust Integrity over time series.
    
    Args:
        signal_series: Shape (T, n) - T timesteps, n signals
        weights: Shape (n,) - weights for each signal
        epsilon: Numerical safety
        
    Returns:
        List of T dictionaries with sigma, TIC per timestep
    """
    results = []
    for t in range(signal_series.shape[0]):
        result = compute_trust_integrity(signal_series[t], weights, epsilon)
        result["t"] = t + 1  # 1-indexed time
        results.append(result)
    return results


def compute_trust_seam(
    sigma_t0: float,
    sigma_t1: float,
    TIC_t0: float,
    TIC_t1: float
) -> Dict[str, float]:
    """
    Compute trust seam between two validation events.
    
    Analogous to seam accounting in UMCP core.
    
    Args:
        sigma_t0: Log-integrity at t0
        sigma_t1: Log-integrity at t1
        TIC_t0: Trust integrity composite at t0
        TIC_t1: Trust integrity composite at t1
        
    Returns:
        Seam metrics: delta_sigma (ledger change), trust_ratio
    """
    # Ledger change: Δσ = σ(t1) - σ(t0) = ln(TIC(t1)/TIC(t0))
    delta_sigma_ledger = sigma_t1 - sigma_t0
    
    # Trust ratio: i_r = TIC(t1) / TIC(t0)
    if TIC_t0 > 0:
        trust_ratio = TIC_t1 / TIC_t0
    else:
        trust_ratio = float('inf') if TIC_t1 > 0 else 1.0
    
    # Verify consistency: exp(Δσ) should equal trust_ratio
    expected_ratio = np.exp(delta_sigma_ledger)
    ratio_consistent = np.isclose(expected_ratio, trust_ratio, rtol=1e-6)
    
    return {
        "delta_sigma_ledger": delta_sigma_ledger,
        "trust_ratio": trust_ratio,
        "sigma_t0": sigma_t0,
        "sigma_t1": sigma_t1,
        "TIC_t0": TIC_t0,
        "TIC_t1": TIC_t1,
        "ratio_consistent": ratio_consistent
    }


def compute_seam_residual(
    delta_sigma_ledger: float,
    delta_sigma_budget: float,
    tol_seam: float = 0.005
) -> Dict[str, Any]:
    """
    Compute seam residual and PASS/FAIL status.
    
    Args:
        delta_sigma_ledger: Measured ledger change
        delta_sigma_budget: Modeled budget change
        tol_seam: Tolerance for pass/fail (from contract)
        
    Returns:
        Seam residual and status
    """
    residual = delta_sigma_budget - delta_sigma_ledger
    
    passed = abs(residual) <= tol_seam
    
    return {
        "residual": residual,
        "abs_residual": abs(residual),
        "tol_seam": tol_seam,
        "passed": passed,
        "status": "PASS" if passed else "FAIL"
    }


if __name__ == "__main__":
    # Example: compute trust integrity
    signals = np.array([0.95, 0.88, 0.92, 0.90])
    weights = np.array([0.4, 0.2, 0.25, 0.15])
    
    result = compute_trust_integrity(signals, weights)
    print(f"Log-Integrity (σ): {result['sigma']:.4f}")
    print(f"Trust Integrity Composite (TIC): {result['TIC']:.4f}")
    
    # Example: seam computation
    # Anomaly case: signals drop
    signals_anomaly = np.array([0.40, 0.30, 0.25, 0.35])
    result_anomaly = compute_trust_integrity(signals_anomaly, weights)
    
    seam = compute_trust_seam(
        result["sigma"], result_anomaly["sigma"],
        result["TIC"], result_anomaly["TIC"]
    )
    print(f"\nSeam (normal → anomaly):")
    print(f"  Δσ: {seam['delta_sigma_ledger']:.4f}")
    print(f"  Trust ratio: {seam['trust_ratio']:.4f}")
