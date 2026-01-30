#!/usr/bin/env python3
"""
Kinematics Closure: Kinematic Stability (KIN-Domain Closure B - Stability Overlay)

Computes stability indices, regime classification, and kinematic budgets.
Maps motion characteristics to UMCP stability framework.

Axiom-0 Enforcement (no_return_no_credit):
  - K_stability credit is only granted if τ_kin < ∞
  - Non-returning motion receives R_kin_effective = 0 (no credit)
  - This embodies "What Returns Is Real"

BUDGET EQUATION DISCLAIMER:
  ΔK_kin = S_kin - C_kin + R_kin is a KINEMATICS OVERLAY ONLY.
  This is NOT the UMCP weld/continuity budget law (Δκ).
  It CANNOT be used to certify continuity across seams.
  Symbols S_kin, C_kin, R_kin are KIN-namespace only, not kernel S, C.

NOTE: This is a DOMAIN CLOSURE, not a UMCP Tier-1 kernel invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_kinematic_stability(
    x_series: np.ndarray,
    v_series: np.ndarray,
    sigma_max: float = 1.0,
    v_max: float = 1.0,
) -> dict[str, Any]:
    """
    Compute kinematic stability index K_stability ∈ [0,1].

    Higher values indicate more stable (less chaotic) motion.

    Args:
        x_series: Time series of positions
        v_series: Time series of velocities
        sigma_max: Maximum position variance for normalization
        v_max: Maximum velocity for normalization

    Returns:
        Dictionary with stability metrics.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    if len(x_series) < 2:
        return {"K_stability": 1.0, "regime": "Stable"}

    # Position variance
    sigma_x = float(np.std(x_series))

    # Mean velocity
    v_mean = float(np.mean(v_series))

    # Velocity variance
    sigma_v = float(np.std(v_series))

    # Stability components
    position_stability = max(0.0, 1.0 - sigma_x / max(sigma_max, 1e-10))
    velocity_stability = max(0.0, 1.0 - v_mean / max(v_max, 1e-10))
    smoothness = max(0.0, 1.0 - sigma_v / max(v_max, 1e-10))

    # Combined stability index
    K_stability = (position_stability + velocity_stability + smoothness) / 3.0
    K_stability = max(0.0, min(1.0, K_stability))

    # Regime classification
    if K_stability > 0.7:
        regime = "Stable"
    elif K_stability > 0.4:
        regime = "Watch"
    else:
        regime = "Unstable"

    return {
        "K_stability": K_stability,
        "position_stability": position_stability,
        "velocity_stability": velocity_stability,
        "smoothness": smoothness,
        "sigma_x": sigma_x,
        "sigma_v": sigma_v,
        "v_mean": v_mean,
        "regime": regime,
    }


def compute_stability_margin(
    K_stability: float,
    stable_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Compute stability margin above/below threshold.

    Args:
        K_stability: Current stability index
        stable_threshold: Threshold for "Stable" regime

    Returns:
        Dictionary with margin info.
    """
    margin = K_stability - stable_threshold

    if margin > 0:
        margin_status = "Positive"
    elif margin > -0.1:
        margin_status = "Marginal"
    else:
        margin_status = "Negative"

    return {
        "margin": margin,
        "margin_status": margin_status,
        "K_stability": K_stability,
        "stable_threshold": stable_threshold,
    }


def compute_stability_trend(
    K_series: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze trend in stability over time.

    Args:
        K_series: Time series of stability indices

    Returns:
        Dictionary with trend analysis.
    """
    K_series = np.asarray(K_series, dtype=float)

    if len(K_series) < 2:
        return {"trend_direction": "Unknown", "trend_slope": 0.0}

    # Linear regression for trend
    n = len(K_series)
    t = np.arange(n)
    slope = float(np.polyfit(t, K_series, 1)[0])

    if slope > 0.01:
        trend_direction = "Improving"
    elif slope < -0.01:
        trend_direction = "Degrading"
    else:
        trend_direction = "Stable"

    return {
        "trend_direction": trend_direction,
        "trend_slope": slope,
        "K_initial": float(K_series[0]),
        "K_final": float(K_series[-1]),
        "K_mean": float(np.mean(K_series)),
    }


def classify_motion_regime(
    v_mean: float,
    a_mean: float,
    K_stability: float,
    tau_kin: float,
) -> dict[str, Any]:
    """
    Classify motion regime based on kinematic properties.

    Args:
        v_mean: Mean velocity
        a_mean: Mean acceleration
        K_stability: Stability index
        tau_kin: Kinematic return time

    Returns:
        Dictionary with motion regime classification.
    """
    # Classify based on velocity
    if v_mean < 0.05:
        motion_regime = "Static"
    elif a_mean < 0.05:
        motion_regime = "Uniform"
    elif tau_kin < float("inf") and tau_kin < 100:
        motion_regime = "Oscillatory"
    elif K_stability < 0.4:
        motion_regime = "Chaotic"
    else:
        motion_regime = "Transient"

    return {
        "motion_regime": motion_regime,
        "v_mean": v_mean,
        "a_mean": a_mean,
        "K_stability": K_stability,
        "tau_kin": tau_kin,
    }


def compute_kinematic_budget(
    K_stability_t0: float,
    K_stability_t1: float,
    S_kin: float,
    C_kin: float,
    tau_kin: float,
    R_kin: float = 0.0,
) -> dict[str, Any]:
    """
    Compute kinematic budget equation (UMCP-style).

    ΔK = S_kin - C_kin + R_kin (sources - consumption + returns)

    AXIOM-0 ENFORCEMENT (no_return_no_credit):
    If τ_kin = INF_KIN, R_kin is forced to 0 regardless of input.
    Non-returning motion receives no return credit.

    Args:
        K_stability_t0: Initial stability
        K_stability_t1: Final stability
        S_kin: Kinematic sources (energy input)
        C_kin: Kinematic consumption (dissipation)
        tau_kin: Return time
        R_kin: Return contribution

    Returns:
        Dictionary with budget analysis.
    """
    # AXIOM-0: no_return_no_credit
    # If tau_kin is infinite, motion does not return → no credit
    if tau_kin == float("inf") or tau_kin < 0:
        R_kin_effective = 0.0
        return_credited = False
    else:
        R_kin_effective = R_kin
        return_credited = True

    delta_K_ledger = K_stability_t1 - K_stability_t0
    delta_K_budget = S_kin - C_kin + R_kin_effective
    residual = delta_K_ledger - delta_K_budget

    budget_balanced = abs(residual) < 0.01

    return {
        "delta_K_ledger": delta_K_ledger,
        "delta_K_budget": delta_K_budget,
        "residual": residual,
        "budget_balanced": budget_balanced,
        "S_kin": S_kin,
        "C_kin": C_kin,
        "R_kin": R_kin,
        "R_kin_effective": R_kin_effective,
        "return_credited": return_credited,
        "tau_kin": tau_kin,
    }
