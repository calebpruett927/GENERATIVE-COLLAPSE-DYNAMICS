#!/usr/bin/env python3
"""
Kinematics Closure: Phase Space Return (KIN-Domain Closure B - Return Overlay)

Implements UMCP return axiom in (x,v) phase space.
Detects oscillations, computes return times, and classifies dynamics.

METRIC SPECIFICATION (FROZEN):
  Metric: squared-L2 (Euclidean squared distance)
  d² = (Δx)² + (Δv)²
  η_phase is a SQUARED distance tolerance

AXIOM-0 ENFORCEMENT (Core Protocol):
  "Collapse is generative; only what returns is real."

  This closure is the kinematics embodiment of Axiom-0:
  - τ_kin < ∞ → motion returns → kinematic credit granted
  - τ_kin = INF_KIN → motion does not return → NO credit (no_return_no_credit)

  Non-returning motion (drifting, divergent, chaotic) receives τ_kin = INF_KIN
  and is classified as "Non_Returning" with kinematic_credit = 0.

NOTE: This is a DOMAIN CLOSURE (return overlay), not UMCP Tier-1.5 weld.
τ_kin is NOT τ_R (kernel return time). They use distinct symbols.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# TYPED SENTINELS (KIN-AX-1: IEEE Inf/NaN forbidden)
# =============================================================================


class KinSpecialValue(Enum):
    """Typed sentinel values for τ_kin (replaces IEEE Inf/NaN)."""

    INF_KIN = "INF_KIN"  # No return detected
    UNIDENTIFIABLE_KIN = "UNIDENTIFIABLE"  # Cannot determine return


# Type alias for τ_kin: positive integer OR typed sentinel
TauKin = int | KinSpecialValue


# Legacy compatibility (deprecated - use KinSpecialValue instead)
INF_KIN = KinSpecialValue.INF_KIN
UNIDENTIFIABLE_KIN = KinSpecialValue.UNIDENTIFIABLE_KIN


# =============================================================================
# FROZEN CONSTANTS (from spec)
# =============================================================================

W_DEFAULT = 64  # Window size in samples
DELTA_DEFAULT = 3  # Debounce lag in samples
T_CRIT = 10.0  # Critical return time threshold
ETA_PHASE_DEFAULT = 0.01  # Phase space tolerance (squared-L2)


def compute_phase_distance(
    x1: float,
    v1: float,
    x2: float,
    v2: float,
) -> float:
    """
    Compute Euclidean distance in phase space.

    Args:
        x1, v1: First phase point
        x2, v2: Second phase point

    Returns:
        Distance between points.
    """
    return math.sqrt((x2 - x1) ** 2 + (v2 - v1) ** 2)


def compute_kinematic_return(
    x_series: np.ndarray,
    v_series: np.ndarray,
    eta_phase: float = ETA_PHASE_DEFAULT,
    debounce: int = DELTA_DEFAULT,
    window: int = W_DEFAULT,
) -> dict[str, Any]:
    """
    Compute kinematic return time in phase space.

    The return time τ_kin is the minimum delay to a previous state within
    η_phase tolerance. Uses squared-L2 distance with strict inequality.

    Args:
        x_series: Time series of normalized positions (x̃)
        v_series: Time series of normalized velocities (ṽ)
        eta_phase: Phase space squared-distance tolerance (strict <)
        debounce: Minimum lag δ (samples) before counting as return
        window: Window size W (samples)

    Returns:
        Dictionary with return statistics.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    n = len(x_series)
    t = n - 1  # Current time index (last point)

    # =================================================================
    # Patch 3: Three-case domain size formula
    # |D_{W,δ}(t)| = 0 if t < δ
    #             = t - δ + 1 if δ ≤ t < W
    #             = W - δ + 1 if t ≥ W
    # =================================================================
    if t < debounce:
        domain_size = 0
    elif t < window:
        domain_size = t - debounce + 1
    else:
        domain_size = window - debounce + 1

    # =================================================================
    # Patch 4: Empty domain → return_rate = 0 (no divide-by-zero)
    # =================================================================
    if domain_size == 0:
        return {
            "tau_kin": KinSpecialValue.INF_KIN,
            "return_count": 0,
            "return_rate": 0.0,
            "dynamics_regime": "Non_Returning",
            "kinematic_credit": 0.0,
            "domain_size": 0,
            "startup": True,
        }

    # Build effective domain D_{W,δ}(t)
    lower_bound = max(0, t - window)
    D_W_delta = [u for u in range(lower_bound, t) if (t - u) >= debounce]

    # =================================================================
    # Patch 1: Phase metric uses tilde variables (already normalized)
    # d²(γ1, γ2) = (x̃2 - x̃1)² + (ṽ2 - ṽ1)²
    # =================================================================
    gamma_t = (x_series[t], v_series[t])

    # Find valid returns U(t) with STRICT inequality d² < η_phase
    valid_returns: list[int] = []
    for u in D_W_delta:
        if u < len(x_series):
            gamma_u = (x_series[u], v_series[u])
            d_squared = (gamma_t[0] - gamma_u[0]) ** 2 + (gamma_t[1] - gamma_u[1]) ** 2
            if d_squared < eta_phase:  # STRICT inequality per spec
                valid_returns.append(u)

    return_count = len(valid_returns)

    # =================================================================
    # Patch 5: τ_kin is positive INTEGER or typed sentinel
    # =================================================================
    tau_kin: TauKin
    if return_count > 0:
        # τ_kin = min delay (most recent return)
        delays: list[int] = [t - u for u in valid_returns]
        tau_kin = min(delays)  # Integer
    else:
        tau_kin = KinSpecialValue.INF_KIN  # Typed sentinel

    # Return rate = |U| / |D|
    return_rate = return_count / domain_size

    # Classify dynamics
    if return_rate > 0.5:
        dynamics_regime = "Returning"
    elif return_rate > 0.3:
        dynamics_regime = "Partially_Returning"
    elif return_rate > 0.1:
        dynamics_regime = "Weakly_Returning"
    else:
        dynamics_regime = "Non_Returning"

    # AXIOM-0 ENFORCEMENT: no_return_no_credit
    kinematic_credit = _compute_credit(tau_kin, return_rate)

    return {
        "tau_kin": tau_kin,
        "return_count": return_count,
        "return_rate": return_rate,
        "dynamics_regime": dynamics_regime,
        "kinematic_credit": kinematic_credit,
        "eta_phase": eta_phase,
        "domain_size": domain_size,
        "n_points": n,
        "startup": False,
    }


def _compute_credit(tau_kin: TauKin, return_rate: float) -> float:
    """Internal credit computation with Axiom-0 enforcement."""
    if isinstance(tau_kin, KinSpecialValue):
        return 0.0
    if return_rate <= 0.1:
        return 0.0
    return (1.0 / (1.0 + tau_kin / T_CRIT)) * return_rate


def compute_kinematic_credit(
    tau_kin: TauKin,
    return_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Compute kinematic credit based on Axiom-0.

    AXIOM-0: "Only what returns is real"
    - Finite τ_kin → motion returns → credit granted
    - τ_kin = INF_KIN → motion does not return → NO credit

    Args:
        tau_kin: Kinematic return time (positive int or KinSpecialValue)
        return_rate: Fraction of trajectory that returns

    Returns:
        Dictionary with credit computation.
    """
    # no_return_no_credit enforcement
    if isinstance(tau_kin, KinSpecialValue):
        kinematic_credit = 0.0
        credit_status = "NO_CREDIT"
        reason = f"no_return_no_credit: τ_kin = {tau_kin.value}"
    elif tau_kin <= 0:
        kinematic_credit = 0.0
        credit_status = "NO_CREDIT"
        reason = "no_return_no_credit: invalid τ_kin"
    elif return_rate <= 0.1:
        kinematic_credit = 0.0
        credit_status = "NO_CREDIT"
        reason = "no_return_no_credit: return_rate ≤ 0.1"
    else:
        # Credit formula: faster returns and higher return rate = more credit
        time_factor = 1.0 / (1.0 + tau_kin / T_CRIT)
        rate_factor = return_rate
        kinematic_credit = time_factor * rate_factor
        credit_status = "CREDITED"
        reason = f"τ_kin={tau_kin}, return_rate={return_rate:.2f}"

    return {
        "kinematic_credit": kinematic_credit,
        "credit_status": credit_status,
        "reason": reason,
        "tau_kin": tau_kin,
        "return_rate": return_rate,
    }


def compute_phase_trajectory(
    x_series: np.ndarray,
    v_series: np.ndarray,
) -> dict[str, Any]:
    """
    Compute phase trajectory properties.

    Args:
        x_series: Time series of positions
        v_series: Time series of velocities

    Returns:
        Dictionary with trajectory properties.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    n = len(x_series)
    if n < 2:
        return {"error": "Need at least 2 points"}

    # Path length in phase space
    dx = np.diff(x_series)
    dv = np.diff(v_series)
    segment_lengths = np.sqrt(dx**2 + dv**2)
    path_length = float(np.sum(segment_lengths))

    # Approximate enclosed area using shoelace formula
    # For non-closed curves, this gives signed area
    enclosed_area = 0.5 * abs(float(np.sum(x_series[:-1] * v_series[1:] - x_series[1:] * v_series[:-1])))

    # Centroid
    centroid_x = float(np.mean(x_series))
    centroid_v = float(np.mean(v_series))

    # Bounding box
    x_range = float(np.max(x_series) - np.min(x_series))
    v_range = float(np.max(v_series) - np.min(v_series))

    return {
        "path_length": path_length,
        "enclosed_area": enclosed_area,
        "centroid": (centroid_x, centroid_v),
        "x_range": x_range,
        "v_range": v_range,
        "n_points": n,
    }


def detect_oscillation(
    x_series: np.ndarray,
    v_series: np.ndarray,
    tol: float = 0.05,
) -> dict[str, Any]:
    """
    Detect oscillatory behavior in phase space.

    Args:
        x_series: Time series of positions
        v_series: Time series of velocities
        tol: Tolerance for period detection

    Returns:
        Dictionary with oscillation properties.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    n = len(x_series)
    if n < 4:
        return {"oscillation_type": "Unknown", "sign_changes": 0}

    # Deviation from mean
    x_centered = x_series - np.mean(x_series)
    v_centered = v_series - np.mean(v_series)

    # Count sign changes (zero crossings)
    x_sign_changes = int(np.sum(np.diff(np.sign(x_centered)) != 0))
    v_sign_changes = int(np.sum(np.diff(np.sign(v_centered)) != 0))
    sign_changes = x_sign_changes + v_sign_changes

    # Estimate period from autocorrelation
    if n > 10:
        # Simple autocorrelation
        x_autocorr = np.correlate(x_centered, x_centered, mode="full")
        x_autocorr = x_autocorr[n - 1 :]  # Keep positive lags

        # Find first peak after lag 0
        peaks: list[int] = []
        for i in range(1, len(x_autocorr) - 1):
            if x_autocorr[i] > x_autocorr[i - 1] and x_autocorr[i] > x_autocorr[i + 1]:
                peaks.append(i)

        period_estimate = float(peaks[0]) if peaks else float(n)
    else:
        period_estimate = float(n)

    # Classify oscillation type
    if sign_changes > n // 2:
        oscillation_type = "Periodic"
    elif sign_changes > n // 4:
        oscillation_type = "Quasi_Periodic"
    elif sign_changes > 2:
        oscillation_type = "Damped"
    else:
        oscillation_type = "Non_Oscillatory"

    return {
        "oscillation_type": oscillation_type,
        "sign_changes": sign_changes,
        "x_sign_changes": x_sign_changes,
        "v_sign_changes": v_sign_changes,
        "period_estimate": period_estimate,
    }


def compute_lyapunov_estimate(
    x_series: np.ndarray,
    v_series: np.ndarray,
    eps: float = 0.01,
) -> dict[str, Any]:
    """
    Estimate largest Lyapunov exponent (crude method).

    Args:
        x_series: Time series of positions
        v_series: Time series of velocities
        eps: Initial separation

    Returns:
        Dictionary with Lyapunov estimate.
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    n = len(x_series)
    if n < 10:
        return {"lambda_max": 0.0, "stable": True, "n_pairs": 0}

    # Find nearby trajectory pairs and track divergence
    divergences: list[float] = []

    for i in range(n // 2):
        x0, v0 = x_series[i], v_series[i]

        # Find nearest neighbor after debounce
        min_dist = float("inf")
        j_min = -1
        for j in range(i + 5, min(i + n // 2, n)):
            d = compute_phase_distance(x0, v0, x_series[j], v_series[j])
            if eps < d < min_dist:
                min_dist = d
                j_min = j

        if j_min > 0 and i + 5 < n and j_min + 5 < n:
            # Track how separation evolves
            d_later = compute_phase_distance(x_series[i + 5], v_series[i + 5], x_series[j_min + 5], v_series[j_min + 5])
            if min_dist > 0:
                divergences.append(math.log(max(d_later, eps) / min_dist) / 5)

    lambda_max = float(np.mean(divergences)) if divergences else 0.0

    stable = lambda_max < 0.1

    return {
        "lambda_max": lambda_max,
        "stable": stable,
        "n_pairs": len(divergences),
    }
