#!/usr/bin/env python3
"""
Kinematics Closure: Linear Kinematics (KIN-Domain Closure A)

Computes linear kinematic quantities: position, velocity, acceleration.
All quantities are normalized to [0,1] via bounded embedding with OOR policy.

Bounded Embedding (UMCP Tier-0 Compliant):
    q̃(t) = clip([0,1], q(t) / q_ref), emit OOR flag if clipped
    q̃_ε(t) = clip([ε, 1-ε], q̃(t)) for log-safe channels

NOTE: This is a DOMAIN CLOSURE, not a UMCP Tier-1 kernel invariant.
UMCP Tier-1 invariants (ω, F, S, C, τ_R, κ, IC) are computed by the kernel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_linear_kinematics(
    x: float,
    v: float,
    a: float,
    dt: float = 1.0,
) -> dict[str, Any]:
    """
    Compute linear kinematic invariants.

    Args:
        x: Normalized position ∈ [0,1]
        v: Normalized velocity ∈ [0,1]
        a: Normalized acceleration ∈ [0,1]
        dt: Time step

    Returns:
        Dictionary containing kinematic invariants.
    """
    # Clip to valid range
    x_oor = x < 0.0 or x > 1.0
    v_oor = v < 0.0 or v > 1.0
    a_oor = a < 0.0 or a > 1.0
    
    x = max(0.0, min(1.0, x))
    v = max(0.0, min(1.0, v))
    a = max(0.0, min(1.0, a))
    
    # Phase space magnitude: ||(x,v)||
    phase_magnitude = math.sqrt(x**2 + v**2)
    
    # Predict next state
    predicted_v_next = max(0.0, min(1.0, v + a * dt))
    predicted_x_next = max(0.0, min(1.0, x + v * dt + 0.5 * a * dt**2))
    
    # Regime classification
    if v < 0.3 and a < 0.2:
        regime = "Stable"
    elif v < 0.6 and a < 0.5:
        regime = "Watch"
    else:
        regime = "Critical"
    
    return {
        "position": x,
        "velocity": v,
        "acceleration": a,
        "phase_magnitude": phase_magnitude,
        "predicted_x_next": predicted_x_next,
        "predicted_v_next": predicted_v_next,
        "regime": regime,
        "oor_flags": {
            "position": x_oor,
            "velocity": v_oor,
            "acceleration": a_oor,
        },
    }


def compute_trajectory(
    x_series: np.ndarray,
    v_series: np.ndarray,
    a_series: np.ndarray,
    dt: float = 1.0,
) -> dict[str, Any]:
    """
    Compute trajectory-level statistics.

    Args:
        x_series: Array of positions
        v_series: Array of velocities
        a_series: Array of accelerations
        dt: Time step

    Returns:
        Dictionary with trajectory statistics.
    """
    x_series = np.clip(np.asarray(x_series), 0.0, 1.0)
    v_series = np.clip(np.asarray(v_series), 0.0, 1.0)
    a_series = np.clip(np.asarray(a_series), 0.0, 1.0)
    
    n = len(x_series)
    if n < 2:
        return {"error": "Need at least 2 points"}
    
    total_displacement = float(x_series[-1] - x_series[0])
    mean_velocity = float(np.mean(v_series))
    mean_acceleration = float(np.mean(a_series))
    
    if mean_velocity < 0.3 and mean_acceleration < 0.2:
        regime = "Stable"
    elif mean_velocity < 0.6:
        regime = "Watch"
    else:
        regime = "Critical"
    
    return {
        "total_displacement": total_displacement,
        "mean_velocity": mean_velocity,
        "mean_acceleration": mean_acceleration,
        "n_points": n,
        "trajectory_regime": regime,
    }


def verify_kinematic_consistency(
    x_series: np.ndarray,
    v_series: np.ndarray,
    a_series: np.ndarray,
    dt: float = 1.0,
    tol: float = 0.1,
) -> dict[str, Any]:
    """
    Verify v ≈ dx/dt and a ≈ dv/dt.

    Args:
        x_series: Array of positions
        v_series: Array of velocities
        a_series: Array of accelerations
        dt: Time step
        tol: Tolerance

    Returns:
        Dictionary with consistency results.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)
    a_series = np.asarray(a_series, dtype=float)
    
    if len(x_series) < 2:
        return {"velocity_consistent": False, "acceleration_consistent": False}
    
    dx_dt = np.diff(x_series) / dt
    dv_dt = np.diff(v_series) / dt
    
    v_error = np.abs(dx_dt - v_series[:-1])
    a_error = np.abs(dv_dt - a_series[:-1])
    
    return {
        "velocity_consistent": bool(np.max(v_error) < tol),
        "acceleration_consistent": bool(np.max(a_error) < tol),
        "max_v_error": float(np.max(v_error)),
        "max_a_error": float(np.max(a_error)),
    }
