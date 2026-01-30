#!/usr/bin/env python3
"""
Kinematics Closure: Rotational Kinematics (KIN-Domain Closure A)

Computes angular kinematic quantities: angular position, velocity, acceleration,
angular momentum, and torque. All quantities normalized to [0,1].

SYMBOL NAMESPACE (avoids kernel collision):
    Ω (omega_rot) - angular velocity (NOT ω which is UMCP drift)
    T (torque)    - torque (NOT τ which is UMCP return time)
    θ (theta)     - angular position
    α (alpha)     - angular acceleration
    L             - angular momentum

NOTE: This is a DOMAIN CLOSURE, not a UMCP Tier-1 kernel invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rotational_kinematics(
    theta: float,
    omega_rot: float,
    alpha: float,
    I_normalized: float = 1.0,
    dt: float = 1.0,
) -> dict[str, Any]:
    """
    Compute rotational kinematic invariants.

    Args:
        theta: Normalized angular position ∈ [0,1] (maps to [0, 2π])
        omega_rot: Normalized angular velocity ∈ [0,1]
        alpha: Normalized angular acceleration ∈ [0,1]
        I_normalized: Normalized moment of inertia ∈ [0,1]
        dt: Time step

    Returns:
        Dictionary containing rotational kinematic invariants.
    """
    # Clip to valid range
    theta = max(0.0, min(1.0, theta))
    omega_rot = max(0.0, min(1.0, omega_rot))
    alpha = max(0.0, min(1.0, alpha))
    I_normalized = max(0.0, min(1.0, I_normalized))

    # Angular momentum: L = I * omega
    angular_momentum = I_normalized * omega_rot

    # Torque: T = I * alpha
    torque = I_normalized * alpha

    # Rotational kinetic energy: E_rot = 0.5 * I * omega^2
    E_rotational = 0.5 * I_normalized * omega_rot**2

    # Predicted next state
    omega_next = max(0.0, min(1.0, omega_rot + alpha * dt))
    theta_next = (theta + omega_rot * dt + 0.5 * alpha * dt**2) % 1.0

    # Regime classification
    if omega_rot < 0.3 and alpha < 0.2:
        regime = "Stable"
    elif omega_rot < 0.6 and alpha < 0.5:
        regime = "Watch"
    else:
        regime = "Critical"

    return {
        "theta": theta,
        "omega_rot": omega_rot,
        "alpha": alpha,
        "I_normalized": I_normalized,
        "angular_momentum": angular_momentum,
        "torque": torque,
        "E_rotational": E_rotational,
        "predicted_theta_next": theta_next,
        "predicted_omega_next": omega_next,
        "regime": regime,
    }


def compute_centripetal(
    omega_rot: float,
    r_normalized: float = 1.0,
    m_normalized: float = 1.0,
) -> dict[str, Any]:
    """
    Compute centripetal acceleration and force.

    Args:
        omega_rot: Normalized angular velocity
        r_normalized: Normalized radius
        m_normalized: Normalized mass

    Returns:
        Dictionary with centripetal quantities.
    """
    omega_rot = max(0.0, min(1.0, omega_rot))
    r_normalized = max(0.0, min(1.0, r_normalized))
    m_normalized = max(0.0, min(1.0, m_normalized))

    # Centripetal acceleration: a_c = omega^2 * r
    a_centripetal = omega_rot**2 * r_normalized

    # Centripetal force: F_c = m * omega^2 * r
    F_centripetal = m_normalized * omega_rot**2 * r_normalized

    # Tangential velocity: v = omega * r
    v_tangential = omega_rot * r_normalized

    return {
        "a_centripetal": a_centripetal,
        "F_centripetal": F_centripetal,
        "v_tangential": v_tangential,
        "omega_rot": omega_rot,
        "r_normalized": r_normalized,
        "m_normalized": m_normalized,
    }


def compute_rotational_trajectory(
    theta_series: np.ndarray,
    omega_series: np.ndarray,
    alpha_series: np.ndarray,
    I_normalized: float = 1.0,
    dt: float = 1.0,
) -> dict[str, Any]:
    """
    Compute trajectory statistics for rotational motion.

    Args:
        theta_series: Array of angular positions
        omega_series: Array of angular velocities
        alpha_series: Array of angular accelerations
        I_normalized: Moment of inertia
        dt: Time step

    Returns:
        Dictionary with trajectory statistics.
    """
    theta_series = np.asarray(theta_series, dtype=float)
    omega_series = np.clip(np.asarray(omega_series), 0.0, 1.0)
    alpha_series = np.clip(np.asarray(alpha_series), 0.0, 1.0)

    n = len(theta_series)
    if n < 2:
        return {"error": "Need at least 2 points"}

    # Total angular displacement
    total_rotation = float(theta_series[-1] - theta_series[0])

    # Number of complete rotations
    n_rotations = int(abs(total_rotation))

    # Mean angular velocity
    mean_omega = float(np.mean(omega_series))

    # Mean angular momentum
    mean_L = float(I_normalized * np.mean(omega_series))

    # Total rotational energy over time
    E_rot_series = 0.5 * I_normalized * omega_series**2
    mean_E_rot = float(np.mean(E_rot_series))

    return {
        "total_rotation": total_rotation,
        "n_rotations": n_rotations,
        "mean_omega": mean_omega,
        "mean_angular_momentum": mean_L,
        "mean_E_rotational": mean_E_rot,
        "n_points": n,
    }
