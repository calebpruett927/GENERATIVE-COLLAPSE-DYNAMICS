"""
UMCP Computational Utilities

Shared utilities for computational optimizations across the UMCP system.

Key features:
- OPT-17: Zero-weight dimension pruning
- Weight normalization and validation
- Coordinate clipping with diagnostics
- Vectorized batch operations (OPT-20)

Interconnections:
- Used by: kernel_optimized.py, validator.py, seam_optimized.py
- Implements: KERNEL_SPECIFICATION.md input preprocessing
- Documentation: docs/COMPUTATIONAL_OPTIMIZATIONS.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PruningResult:
    """Result of weight-based dimension pruning."""

    c_active: np.ndarray  # Pruned coordinate array
    w_active: np.ndarray  # Pruned and renormalized weights
    active_mask: np.ndarray  # Boolean mask of active dimensions
    n_original: int  # Original dimension count
    n_active: int  # Active dimension count
    pruned_indices: list[int]  # Indices that were pruned


@dataclass
class ClippingResult:
    """Result of coordinate clipping with diagnostics."""

    c_clipped: np.ndarray  # Clipped coordinates
    clip_count: int  # Number of coordinates clipped
    clip_perturbation: float  # Total perturbation from clipping
    max_perturbation: float  # Maximum single-coordinate perturbation
    oor_indices: list[int]  # Indices of out-of-range coordinates


def prune_zero_weights(
    c: np.ndarray,
    w: np.ndarray,
    threshold: float = 1e-15,
) -> PruningResult:
    """
    OPT-17: Prune zero-weight dimensions before kernel computation.

    Lemma 31: Zero-weight coordinates don't affect kernel outputs.
    Pruning them provides ~N/N_active speedup.

    Args:
        c: Coordinate array (n,)
        w: Weight array (n,)
        threshold: Minimum weight to consider active

    Returns:
        PruningResult with pruned arrays and diagnostics
    """
    active_mask = w > threshold
    n_original = len(c)
    n_active = int(np.sum(active_mask))

    if n_active == 0:
        raise ValueError("All weights are zero or below threshold")

    if n_active == n_original:
        # No pruning needed
        return PruningResult(
            c_active=c,
            w_active=w,
            active_mask=active_mask,
            n_original=n_original,
            n_active=n_active,
            pruned_indices=[],
        )

    # Prune and renormalize
    c_active = c[active_mask]
    w_active = w[active_mask]
    w_active = w_active / w_active.sum()  # Renormalize to sum=1

    pruned_indices = list(np.where(~active_mask)[0])

    return PruningResult(
        c_active=c_active,
        w_active=w_active,
        active_mask=active_mask,
        n_original=n_original,
        n_active=n_active,
        pruned_indices=pruned_indices,
    )


def clip_coordinates(
    c: np.ndarray,
    epsilon: float = 1e-6,
    w: np.ndarray | None = None,
) -> ClippingResult:
    """
    Clip coordinates to [ε, 1-ε] with diagnostics.

    Lemma 17: Clipping perturbation bounded by clip magnitude.

    Args:
        c: Coordinate array (may be outside [0,1])
        epsilon: Clipping tolerance
        w: Optional weights for perturbation weighting

    Returns:
        ClippingResult with clipped coordinates and diagnostics
    """
    c_clipped = np.clip(c, epsilon, 1 - epsilon)
    perturbations = np.abs(c - c_clipped)

    clip_count = int(np.sum(perturbations > 0))
    max_perturbation = float(np.max(perturbations)) if clip_count > 0 else 0.0

    clip_perturbation = float(np.sum(w * perturbations)) if w is not None else float(np.sum(perturbations))

    oor_indices = list(np.where(perturbations > 0)[0])

    return ClippingResult(
        c_clipped=c_clipped,
        clip_count=clip_count,
        clip_perturbation=clip_perturbation,
        max_perturbation=max_perturbation,
        oor_indices=oor_indices,
    )


def normalize_weights(w: np.ndarray, validate: bool = True) -> np.ndarray:
    """
    Normalize weights to sum to 1.0.

    Args:
        w: Weight array
        validate: Whether to validate non-negativity

    Returns:
        Normalized weight array

    Raises:
        ValueError: If weights are all zero or contain negatives
    """
    if validate and np.any(w < 0):
        raise ValueError("Weights must be non-negative")

    w_sum = w.sum()
    if w_sum == 0:
        raise ValueError("Weights sum to zero")

    result: np.ndarray = w / w_sum
    return result


def validate_inputs(
    c: np.ndarray,
    w: np.ndarray,
    epsilon: float = 1e-6,
) -> dict[str, bool | str]:
    """
    Validate kernel inputs against contract requirements.

    Checks:
    - Dimensions match
    - Weights sum to 1.0
    - Coordinates in [ε, 1-ε]
    - No NaN/Inf values

    Returns:
        Dict with validation status and any error messages
    """
    errors: list[str] = []

    # Dimension check
    if len(c) != len(w):
        errors.append(f"Dimension mismatch: c has {len(c)}, w has {len(w)}")

    # Weight sum check
    if not np.allclose(w.sum(), 1.0, atol=1e-9):
        errors.append(f"Weights must sum to 1.0, got {w.sum()}")

    # Non-negativity check
    if np.any(w < 0):
        errors.append("Weights contain negative values")

    # Coordinate range check
    if np.any(c < epsilon) or np.any(c > 1 - epsilon):
        oor_count = int(np.sum((c < epsilon) | (c > 1 - epsilon)))
        errors.append(f"{oor_count} coordinates outside [{epsilon}, {1 - epsilon}]")

    # NaN/Inf check
    if np.any(~np.isfinite(c)):
        errors.append("Coordinates contain NaN or Inf")
    if np.any(~np.isfinite(w)):
        errors.append("Weights contain NaN or Inf")

    return {
        "valid": len(errors) == 0,
        "errors": "; ".join(errors) if errors else "",
    }


def batch_validate_outputs(
    outputs_array: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    OPT-20: Vectorized range checking for batch outputs.

    Args:
        outputs_array: Array of shape (T, 5) with [F, omega, C, IC, kappa]
        epsilon: Clipping tolerance for IC bounds

    Returns:
        Boolean array of shape (T,) indicating valid timesteps
    """
    F = outputs_array[:, 0]
    omega = outputs_array[:, 1]
    C = outputs_array[:, 2]
    IC = outputs_array[:, 3]
    kappa = outputs_array[:, 4]

    # Vectorized range checks
    valid = (
        (F >= 0)
        & (F <= 1)
        & (omega >= 0)
        & (omega <= 1)
        & (C >= 0)
        & (C <= 1)
        & (epsilon <= IC)
        & (1 - epsilon >= IC)
        & np.isfinite(kappa)
    )

    result: np.ndarray = valid
    return result


def preprocess_trace_row(
    c: np.ndarray,
    w: np.ndarray,
    epsilon: float = 1e-6,
    prune_weights: bool = True,
    clip_coords: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, PruningResult | ClippingResult | None]]:
    """
    Complete preprocessing for a trace row.

    Applies:
    - OPT-17: Zero-weight pruning
    - Clipping to [ε, 1-ε]
    - Weight normalization

    Args:
        c: Coordinate array
        w: Weight array
        epsilon: Clipping tolerance
        prune_weights: Whether to apply OPT-17
        clip_coords: Whether to clip coordinates

    Returns:
        (processed_c, processed_w, diagnostics_dict)
    """
    diagnostics: dict[str, PruningResult | ClippingResult | None] = {
        "pruning": None,
        "clipping": None,
    }

    c_proc = c.copy()
    w_proc = w.copy()

    # OPT-17: Weight pruning
    if prune_weights:
        pruning_result = prune_zero_weights(c_proc, w_proc)
        c_proc = pruning_result.c_active
        w_proc = pruning_result.w_active
        diagnostics["pruning"] = pruning_result

    # Clipping
    if clip_coords:
        clipping_result = clip_coordinates(c_proc, epsilon, w_proc)
        c_proc = clipping_result.c_clipped
        diagnostics["clipping"] = clipping_result

    # Final normalization (should be no-op if pruning did its job)
    w_proc = normalize_weights(w_proc, validate=False)

    return c_proc, w_proc, diagnostics


class BatchProcessor:
    """
    Batch processing utilities for trace data.

    Optimized for processing multiple timesteps efficiently.
    """

    def __init__(self, epsilon: float = 1e-6):
        """Initialize batch processor."""
        self.epsilon = epsilon

    def preprocess_trace(
        self,
        trace: np.ndarray,
        weights: np.ndarray,
        prune_weights: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
        """
        Preprocess entire trace.

        Args:
            trace: Trace array (T x n)
            weights: Weight array (n,) or (T x n)
            prune_weights: Whether to apply OPT-17

        Returns:
            (processed_trace, processed_weights, per_row_diagnostics)
        """
        T, n_dim = trace.shape

        # Handle weight broadcasting
        if weights.ndim == 1:
            weights_2d = np.tile(weights, (T, 1))
            assert len(weights) == n_dim, "Weight dimension mismatch"
        else:
            weights_2d = weights

        # Clip entire trace at once (vectorized)
        trace_clipped = np.clip(trace, self.epsilon, 1 - self.epsilon)

        # Count clipping per row
        clip_counts = np.sum(trace != trace_clipped, axis=1)

        diagnostics = [{"row": t, "clip_count": int(clip_counts[t])} for t in range(T)]

        return trace_clipped, weights_2d, diagnostics

    def compute_batch_statistics(self, trace: np.ndarray) -> dict[str, float]:
        """
        Compute summary statistics for trace.

        Useful for adaptive parameter selection.
        """
        return {
            "mean": float(np.mean(trace)),
            "std": float(np.std(trace)),
            "min": float(np.min(trace)),
            "max": float(np.max(trace)),
            "homogeneity": float(1 - np.std(trace.mean(axis=1))),
        }
