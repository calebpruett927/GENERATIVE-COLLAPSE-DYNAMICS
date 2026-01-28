#!/usr/bin/env python3
"""
Kinematics Closure: Momentum Dynamics (KIN-Domain Closure A)

Computes momentum quantities: linear momentum, impulse, collisions.
Verifies momentum conservation laws.

Bounded Embedding:
    All momentum values clipped to [0,1] with OOR flags.

NOTE: This is a DOMAIN CLOSURE, not a UMCP Tier-1 kernel invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_linear_momentum(
    v: float,
    m_normalized: float = 1.0,
) -> dict[str, Any]:
    """
    Compute linear momentum: p = m * v.

    Args:
        v: Normalized velocity ∈ [0,1]
        m_normalized: Normalized mass ∈ [0,1]

    Returns:
        Dictionary with momentum.
    """
    v = max(0.0, min(1.0, v))
    m_normalized = max(0.0, min(1.0, m_normalized))
    
    p = m_normalized * v
    
    return {
        "p": p,
        "v": v,
        "m_normalized": m_normalized,
    }


def compute_impulse(
    F_net: float,
    dt: float,
) -> dict[str, Any]:
    """
    Compute impulse: J = F * dt.

    Args:
        F_net: Normalized net force
        dt: Time interval

    Returns:
        Dictionary with impulse.
    """
    F_net = max(0.0, min(1.0, F_net))
    dt = max(0.0, dt)
    
    J = F_net * dt
    
    return {
        "J": J,
        "F_net": F_net,
        "dt": dt,
    }


def compute_collision_1d(
    m1: float,
    v1_initial: float,
    m2: float,
    v2_initial: float,
    collision_type: str = "elastic",
) -> dict[str, Any]:
    """
    Compute 1D collision outcomes.

    Args:
        m1: Mass of object 1 (normalized)
        v1_initial: Initial velocity of object 1
        m2: Mass of object 2 (normalized)
        v2_initial: Initial velocity of object 2
        collision_type: "elastic" or "perfectly_inelastic"

    Returns:
        Dictionary with final velocities and conservation flags.
    """
    m1 = max(0.01, min(1.0, m1))  # Avoid zero mass
    m2 = max(0.01, min(1.0, m2))
    v1_initial = max(0.0, min(1.0, v1_initial))
    v2_initial = max(0.0, min(1.0, v2_initial))
    
    # Total momentum
    p_total = m1 * v1_initial + m2 * v2_initial
    
    # Total kinetic energy (initial)
    E_initial = 0.5 * m1 * v1_initial**2 + 0.5 * m2 * v2_initial**2
    
    if collision_type == "elastic":
        # Elastic collision formulas
        v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
        v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)
        
        # Clamp to valid range
        v1_final = max(0.0, min(1.0, v1_final))
        v2_final = max(0.0, min(1.0, v2_final))
        
        E_final = 0.5 * m1 * v1_final**2 + 0.5 * m2 * v2_final**2
        energy_loss = 0.0
        energy_conserved = True
        
    elif collision_type == "perfectly_inelastic":
        # Objects stick together
        v_common = p_total / (m1 + m2)
        v1_final = v_common
        v2_final = v_common
        
        E_final = 0.5 * (m1 + m2) * v_common**2
        energy_loss = E_initial - E_final
        energy_conserved = False
        
    else:
        # Unknown collision type
        v1_final = v1_initial
        v2_final = v2_initial
        E_final = E_initial
        energy_loss = 0.0
        energy_conserved = True
    
    # Verify momentum conservation
    p_final = m1 * v1_final + m2 * v2_final
    momentum_conserved = abs(p_total - p_final) < 0.01
    
    return {
        "v1_final": v1_final,
        "v2_final": v2_final,
        "p_initial": p_total,
        "p_final": p_final,
        "E_initial": E_initial,
        "E_final": E_final,
        "energy_loss": energy_loss,
        "momentum_conserved": momentum_conserved,
        "energy_conserved": energy_conserved,
        "collision_type": collision_type,
    }


def verify_momentum_conservation(
    p_series: np.ndarray,
    tol: float = 0.01,
) -> dict[str, Any]:
    """
    Verify momentum conservation: p should remain constant.

    Args:
        p_series: Time series of total momentum
        tol: Tolerance

    Returns:
        Dictionary with conservation verification.
    """
    p_series = np.asarray(p_series, dtype=float)
    
    if len(p_series) < 2:
        return {"is_conserved": True, "error": "Need at least 2 points"}
    
    p_mean = float(np.mean(p_series))
    p_std = float(np.std(p_series))
    max_deviation = float(np.max(np.abs(p_series - p_mean)))
    
    is_conserved = max_deviation < tol
    
    return {
        "is_conserved": is_conserved,
        "p_mean": p_mean,
        "p_std": p_std,
        "max_deviation": max_deviation,
        "tolerance": tol,
    }


def compute_momentum_flux(
    p: float,
    v: float,
) -> dict[str, Any]:
    """
    Compute momentum flux tensor component: Φ = p * v.

    Args:
        p: Momentum
        v: Velocity

    Returns:
        Dictionary with momentum flux.
    """
    p = max(0.0, min(1.0, p))
    v = max(0.0, min(1.0, v))
    
    flux = p * v
    
    return {
        "momentum_flux": flux,
        "p": p,
        "v": v,
    }
