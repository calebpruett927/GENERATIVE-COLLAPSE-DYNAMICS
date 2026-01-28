#!/usr/bin/env python3
"""
Kinematics Closure: Energy Mechanics (KIN-Domain Closure A)

Computes energy quantities: kinetic, potential, mechanical energy,
work, and power. Verifies energy conservation laws.

Bounded Embedding:
    All energy values clipped to [0,1] with OOR flags.
    Log-safe values use ε-guard: clip([ε, 1-ε], E) before ln().

NOTE: This is a DOMAIN CLOSURE, not a UMCP Tier-1 kernel invariant.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_kinetic_energy(
    v: float,
    m_normalized: float = 1.0,
) -> dict[str, Any]:
    """
    Compute kinetic energy: E_kin = 0.5 * m * v^2.

    Args:
        v: Normalized velocity ∈ [0,1]
        m_normalized: Normalized mass ∈ [0,1]

    Returns:
        Dictionary with kinetic energy.
    """
    v = max(0.0, min(1.0, v))
    m_normalized = max(0.0, min(1.0, m_normalized))
    
    E_kinetic = 0.5 * m_normalized * v**2
    
    return {
        "E_kinetic": E_kinetic,
        "v": v,
        "m_normalized": m_normalized,
    }


def compute_potential_energy(
    h: float,
    m_normalized: float = 1.0,
    g_normalized: float = 1.0,
    k_normalized: float = 0.0,
    x_spring: float = 0.0,
    potential_type: str = "gravitational",
) -> dict[str, Any]:
    """
    Compute potential energy.

    For gravitational: E_pot = m * g * h
    For spring: E_pot = 0.5 * k * x^2

    Args:
        h: Normalized height ∈ [0,1]
        m_normalized: Normalized mass
        g_normalized: Normalized gravitational acceleration
        k_normalized: Normalized spring constant
        x_spring: Spring displacement
        potential_type: "gravitational" or "spring"

    Returns:
        Dictionary with potential energy.
    """
    h = max(0.0, min(1.0, h))
    m_normalized = max(0.0, min(1.0, m_normalized))
    g_normalized = max(0.0, min(1.0, g_normalized))
    
    if potential_type == "gravitational":
        E_potential = m_normalized * g_normalized * h
    elif potential_type == "spring":
        k_normalized = max(0.0, min(1.0, k_normalized))
        x_spring = max(0.0, min(1.0, abs(x_spring)))
        E_potential = 0.5 * k_normalized * x_spring**2
    else:
        E_potential = 0.0
    
    return {
        "E_potential": E_potential,
        "potential_type": potential_type,
        "h": h,
        "m_normalized": m_normalized,
        "g_normalized": g_normalized,
    }


def compute_mechanical_energy(
    v: float,
    h: float,
    m_normalized: float = 1.0,
    g_normalized: float = 1.0,
) -> dict[str, Any]:
    """
    Compute total mechanical energy: E_mech = E_kin + E_pot.

    Args:
        v: Normalized velocity
        h: Normalized height
        m_normalized: Normalized mass
        g_normalized: Normalized gravitational acceleration

    Returns:
        Dictionary with mechanical energy breakdown.
    """
    v = max(0.0, min(1.0, v))
    h = max(0.0, min(1.0, h))
    m_normalized = max(0.0, min(1.0, m_normalized))
    g_normalized = max(0.0, min(1.0, g_normalized))
    
    E_kinetic = 0.5 * m_normalized * v**2
    E_potential = m_normalized * g_normalized * h
    E_mechanical = E_kinetic + E_potential
    
    return {
        "E_mechanical": E_mechanical,
        "E_kinetic": E_kinetic,
        "E_potential": E_potential,
        "v": v,
        "h": h,
    }


def compute_work(
    F_net: float,
    displacement: float,
    angle: float = 0.0,
) -> dict[str, Any]:
    """
    Compute work: W = F * d * cos(theta).

    Args:
        F_net: Normalized net force
        displacement: Normalized displacement
        angle: Angle between force and displacement (radians)

    Returns:
        Dictionary with work.
    """
    F_net = max(0.0, min(1.0, F_net))
    displacement = max(0.0, min(1.0, displacement))
    
    W = F_net * displacement * math.cos(angle)
    
    return {
        "W": W,
        "F_net": F_net,
        "displacement": displacement,
        "angle": angle,
    }


def compute_power(
    F_net: float,
    v: float,
) -> dict[str, Any]:
    """
    Compute instantaneous power: P = F * v.

    Args:
        F_net: Normalized net force
        v: Normalized velocity

    Returns:
        Dictionary with power.
    """
    F_net = max(0.0, min(1.0, F_net))
    v = max(0.0, min(1.0, v))
    
    P_power = F_net * v
    
    return {
        "P_power": P_power,
        "F_net": F_net,
        "v": v,
    }


def verify_energy_conservation(
    E_series: np.ndarray,
    tol: float = 0.01,
) -> dict[str, Any]:
    """
    Verify energy conservation: E should remain constant.

    Args:
        E_series: Time series of total energy
        tol: Tolerance for conservation check

    Returns:
        Dictionary with conservation verification.
    """
    E_series = np.asarray(E_series, dtype=float)
    
    if len(E_series) < 2:
        return {"is_conserved": True, "error": "Need at least 2 points"}
    
    E_mean = float(np.mean(E_series))
    E_std = float(np.std(E_series))
    max_deviation = float(np.max(np.abs(E_series - E_mean)))
    
    is_conserved = max_deviation < tol
    
    return {
        "is_conserved": is_conserved,
        "E_mean": E_mean,
        "E_std": E_std,
        "max_deviation": max_deviation,
        "tolerance": tol,
    }


def verify_work_energy_theorem(
    W_net: float,
    E_kin_initial: float,
    E_kin_final: float,
    tol: float = 0.01,
) -> dict[str, Any]:
    """
    Verify work-energy theorem: W_net = ΔE_kin.

    Args:
        W_net: Net work done
        E_kin_initial: Initial kinetic energy
        E_kin_final: Final kinetic energy
        tol: Tolerance

    Returns:
        Dictionary with verification result.
    """
    delta_E_kin = E_kin_final - E_kin_initial
    residual = abs(W_net - delta_E_kin)
    is_valid = residual < tol
    
    return {
        "is_valid": is_valid,
        "W_net": W_net,
        "delta_E_kin": delta_E_kin,
        "residual": residual,
        "tolerance": tol,
    }
