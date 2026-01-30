#!/usr/bin/env python3
"""
KIN.REF.PHASE Closure: Deterministic Phase-Anchor Selection

Implements canonical phase-anchor selection for oscillatory motion.
This is a Tier-0/Tier-2 kinematics closure, NOT Tier-1 kernel.

FROZEN PARAMETERS:
  δφ_max = π/6 ≈ 0.5235987756 (30° maximum phase mismatch)
  W = 20 (window size in samples)
  δ = 3 (debounce lag in samples)

PHASE MAPPING (φ):
  Given (x, v) ∈ [0,1]², compute:
    x' = 2x - 1 (center to [-1,1])
    v' = 2v - 1 (center to [-1,1])
    φ = atan2(v', x') wrapped to [0, 2π)

CIRCULAR DISTANCE (Δφ):
  Δφ(a, b) = min(|a - b|, 2π - |a - b|)
  Result is in [0, π]

SELECTOR RULES:
  1. Build eligible set ℰ(t) = D_{W,δ}(t)
  2. If ℰ(t) empty → undefined (EMPTY_ELIGIBLE)
  3. Find minimum Δφ among eligible
  4. If min Δφ > δφ_max → undefined (PHASE_MISMATCH)
  5. Tie-breakers:
     (i) minimize Δφ
     (ii) if tied: choose most recent u (largest u)
     (iii) if still tied: minimize d(Ψ(t), Ψ(u))

DOES NOT REDEFINE TIER-1 SYMBOLS: {ω, F, S, C, τ_R, IC, κ}
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, NamedTuple

import numpy as np

# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

# Maximum phase mismatch (π/6 radians = 30°)
DELTA_PHI_MAX = math.pi / 6  # ≈ 0.5235987756

# Window size in samples
WINDOW = 20

# Debounce lag in samples
DEBOUNCE = 3

# Two pi constant
TWO_PI = 2.0 * math.pi


# =============================================================================
# TYPES
# =============================================================================


class UndefinedReason(Enum):
    """Reasons for undefined reference anchor."""

    NONE = "NONE"
    EMPTY_ELIGIBLE = "EMPTY_ELIGIBLE"
    PHASE_MISMATCH = "PHASE_MISMATCH"


class PhaseAnchorResult(NamedTuple):
    """Result of phase-anchor selection."""

    idx: int  # Current time index
    phi: float  # Phase at current index
    eligible_count: int  # Number of eligible anchors
    anchor_u: int | None  # Selected anchor index (None if undefined)
    delta_phi: float | None  # Phase distance to anchor (None if undefined)
    undefined_reason: UndefinedReason  # Reason for undefined


class CensorEvent(NamedTuple):
    """Typed censor event for undefined reference."""

    idx: int
    phi: float
    reason: str
    eligible_count: int


# =============================================================================
# PHASE COMPUTATION (FROZEN)
# =============================================================================


def compute_phase(x: float, v: float) -> float:
    """
    Compute phase angle from normalized (x, v) coordinates.

    Mapping (FROZEN):
        x' = 2x - 1  (center [0,1] to [-1,1])
        v' = 2v - 1  (center [0,1] to [-1,1])
        φ = atan2(v', x') wrapped to [0, 2π)

    Args:
        x: Normalized position in [0, 1]
        v: Normalized velocity in [0, 1]

    Returns:
        Phase angle φ ∈ [0, 2π)
    """
    # Center to [-1, 1]
    x_centered = 2.0 * x - 1.0
    v_centered = 2.0 * v - 1.0

    # Compute atan2 (returns value in [-π, π])
    phi_raw = math.atan2(v_centered, x_centered)

    # Wrap to [0, 2π)
    if phi_raw < 0:
        phi_raw += TWO_PI

    return phi_raw


def circular_distance(a: float, b: float) -> float:
    """
    Compute circular distance between two phase angles.

    Formula (FROZEN):
        Δφ(a, b) = min(|a - b|, 2π - |a - b|)

    Args:
        a: First phase angle in [0, 2π)
        b: Second phase angle in [0, 2π)

    Returns:
        Circular distance in [0, π]
    """
    diff = abs(a - b)
    return min(diff, TWO_PI - diff)


def compute_psi_distance(x1: float, v1: float, x2: float, v2: float) -> float:
    """
    Compute Euclidean distance in phase space (for tie-breaker iii).

    Args:
        x1, v1: First point
        x2, v2: Second point

    Returns:
        Euclidean distance
    """
    return math.sqrt((x2 - x1) ** 2 + (v2 - v1) ** 2)


# =============================================================================
# RETURN-DOMAIN GENERATOR (FROZEN)
# =============================================================================


def build_eligible_set(t: int, window: int = WINDOW, debounce: int = DEBOUNCE) -> list[int]:
    """
    Build the eligible set ℰ(t) = D_{W,δ}(t).

    D_W(t) = {u : max(0, t - W) ≤ u ≤ t - 1}
    D_{W,δ}(t) = {u ∈ D_W(t) : (t - u) ≥ δ}

    Args:
        t: Current time index
        window: Window size W
        debounce: Debounce lag δ

    Returns:
        List of eligible indices
    """
    if t < debounce:
        return []

    lower_bound = max(0, t - window)
    upper_bound = t - debounce  # Inclusive: t - u >= δ means u <= t - δ

    return list(range(lower_bound, upper_bound + 1))


# =============================================================================
# PHASE-ANCHOR SELECTOR (MAIN CLOSURE)
# =============================================================================


def select_phase_anchor(
    x_series: np.ndarray | list[float],
    v_series: np.ndarray | list[float],
    t: int,
    delta_phi_max: float = DELTA_PHI_MAX,
    window: int = WINDOW,
    debounce: int = DEBOUNCE,
) -> PhaseAnchorResult:
    """
    Select reference phase anchor for oscillatory motion.

    This is the main KIN.REF.PHASE closure function.

    Args:
        x_series: Time series of normalized positions
        v_series: Time series of normalized velocities
        t: Current time index
        delta_phi_max: Maximum phase mismatch threshold
        window: Return-domain window size
        debounce: Debounce lag

    Returns:
        PhaseAnchorResult with selection details
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    # Compute phase at current index
    phi_t = compute_phase(x_series[t], v_series[t])

    # Build eligible set
    eligible = build_eligible_set(t, window, debounce)
    eligible_count = len(eligible)

    # Case 1: Empty eligible set
    if eligible_count == 0:
        return PhaseAnchorResult(
            idx=t,
            phi=phi_t,
            eligible_count=0,
            anchor_u=None,
            delta_phi=None,
            undefined_reason=UndefinedReason.EMPTY_ELIGIBLE,
        )

    # Compute phase distances for all eligible anchors
    candidates: list[tuple[int, float, float]] = []  # (u, delta_phi, psi_distance)

    for u in eligible:
        phi_u = compute_phase(x_series[u], v_series[u])
        d_phi = circular_distance(phi_t, phi_u)
        d_psi = compute_psi_distance(x_series[t], v_series[t], x_series[u], v_series[u])
        candidates.append((u, d_phi, d_psi))

    # Find minimum delta_phi
    min_delta_phi = min(c[1] for c in candidates)

    # Case 2: Phase mismatch (all candidates exceed threshold)
    if min_delta_phi > delta_phi_max:
        return PhaseAnchorResult(
            idx=t,
            phi=phi_t,
            eligible_count=eligible_count,
            anchor_u=None,
            delta_phi=None,
            undefined_reason=UndefinedReason.PHASE_MISMATCH,
        )

    # Case 3: Valid selection with tie-breakers
    # Filter to candidates with minimum delta_phi
    best_candidates = [c for c in candidates if abs(c[1] - min_delta_phi) < 1e-12]

    if len(best_candidates) == 1:
        # No tie
        anchor_u = best_candidates[0][0]
    else:
        # Tie-breaker (ii): choose most recent u (largest u)
        best_candidates.sort(key=lambda c: -c[0])  # Sort by u descending
        max_u = best_candidates[0][0]

        # Check if still tied on u (shouldn't happen, but handle it)
        same_u_candidates = [c for c in best_candidates if c[0] == max_u]

        if len(same_u_candidates) == 1:
            anchor_u = same_u_candidates[0][0]
        else:
            # Tie-breaker (iii): minimize psi distance
            same_u_candidates.sort(key=lambda c: c[2])  # Sort by psi_distance ascending
            anchor_u = same_u_candidates[0][0]

    # Get the delta_phi for the selected anchor
    selected = next(c for c in candidates if c[0] == anchor_u)

    return PhaseAnchorResult(
        idx=t,
        phi=phi_t,
        eligible_count=eligible_count,
        anchor_u=anchor_u,
        delta_phi=selected[1],
        undefined_reason=UndefinedReason.NONE,
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_trajectory(
    x_series: np.ndarray | list[float],
    v_series: np.ndarray | list[float],
    delta_phi_max: float = DELTA_PHI_MAX,
    window: int = WINDOW,
    debounce: int = DEBOUNCE,
) -> tuple[list[PhaseAnchorResult], list[CensorEvent]]:
    """
    Process entire trajectory and return results + censor events.

    Args:
        x_series: Time series of normalized positions
        v_series: Time series of normalized velocities
        delta_phi_max: Maximum phase mismatch threshold
        window: Return-domain window size
        debounce: Debounce lag

    Returns:
        Tuple of (results list, censor events list)
    """
    x_series = np.asarray(x_series, dtype=float)
    v_series = np.asarray(v_series, dtype=float)

    n = len(x_series)
    results: list[PhaseAnchorResult] = []
    censors: list[CensorEvent] = []

    for t in range(n):
        result = select_phase_anchor(
            x_series,
            v_series,
            t,
            delta_phi_max=delta_phi_max,
            window=window,
            debounce=debounce,
        )
        results.append(result)

        # Record censor event if undefined
        if result.undefined_reason != UndefinedReason.NONE:
            censors.append(
                CensorEvent(
                    idx=result.idx,
                    phi=result.phi,
                    reason=result.undefined_reason.value,
                    eligible_count=result.eligible_count,
                )
            )

    return results, censors


def results_to_csv_rows(results: list[PhaseAnchorResult]) -> list[dict[str, Any]]:
    """
    Convert results to CSV-compatible rows.

    Args:
        results: List of PhaseAnchorResult

    Returns:
        List of dicts with CSV columns
    """
    rows: list[dict[str, Any]] = []

    for r in results:
        rows.append(
            {
                "idx": r.idx,
                "phi": f"{r.phi:.10f}",
                "eligible_count": r.eligible_count,
                "anchor_u": r.anchor_u if r.anchor_u is not None else "",
                "delta_phi": f"{r.delta_phi:.10f}" if r.delta_phi is not None else "",
                "undefined_reason": r.undefined_reason.value,
            }
        )

    return rows


def censors_to_json(
    censors: list[CensorEvent],
    delta_phi_max: float = DELTA_PHI_MAX,
    window: int = WINDOW,
    debounce: int = DEBOUNCE,
) -> dict[str, Any]:
    """
    Convert censor events to JSON structure.

    Args:
        censors: List of CensorEvent
        delta_phi_max: Frozen parameter
        window: Frozen parameter
        debounce: Frozen parameter

    Returns:
        JSON-serializable dict
    """
    return {
        "schema": "KIN.REF.PHASE.CENSOR.v1",
        "description": "Typed censor events for undefined reference anchors",
        "parameters": {
            "delta_phi_max": delta_phi_max,
            "window": window,
            "debounce": debounce,
        },
        "censors": [
            {
                "idx": c.idx,
                "phi": float(c.phi),
                "reason": c.reason,
                "eligible_count": c.eligible_count,
            }
            for c in censors
        ],
        "total_censored": len(censors),
    }


# =============================================================================
# FROZEN CONFIGURATION HASH
# =============================================================================


def get_frozen_config_sha256() -> str:
    """
    Get SHA256 hash of frozen configuration.

    This provides a deterministic fingerprint of the closure parameters.

    Returns:
        SHA256 hex digest
    """
    import hashlib
    import json

    config: dict[str, float | int | str | list[str]] = {
        "closure_id": "KIN.REF.PHASE.v1",
        "delta_phi_max": DELTA_PHI_MAX,
        "window": WINDOW,
        "debounce": DEBOUNCE,
        "phi_mapping": "atan2_centered",
        "circular_distance": "min(|a-b|, 2π-|a-b|)",
        "tie_breakers": ["min_delta_phi", "most_recent_u", "min_psi_distance"],
    }

    # Deterministic JSON serialization
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == "__main__":
    # Print frozen configuration
    print("KIN.REF.PHASE Closure Configuration")
    print("=" * 40)
    print(f"delta_phi_max: {DELTA_PHI_MAX:.10f} ({math.degrees(DELTA_PHI_MAX):.1f}°)")
    print(f"window: {WINDOW}")
    print(f"debounce: {DEBOUNCE}")
    print(f"frozen_config_sha256: {get_frozen_config_sha256()}")
