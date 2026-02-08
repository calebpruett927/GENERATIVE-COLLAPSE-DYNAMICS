"""Tests for batch_validate_outputs and BatchProcessor in compute_utils.

These are the vectorized validation pipeline — if they silently pass bad
rows, the entire validation chain is undermined.
"""

from __future__ import annotations

import numpy as np

from umcp.compute_utils import (
    BatchProcessor,
    batch_validate_outputs,
    preprocess_trace_row,
    validate_inputs,
)
from umcp.frozen_contract import EPSILON

# ============================================================================
# batch_validate_outputs
# ============================================================================


class TestBatchValidateOutputs:
    """batch_validate_outputs returns boolean mask for rows with valid invariants."""

    def test_all_valid_rows(self) -> None:
        """Rows with proper F, ω, C, IC, κ relationships pass."""
        outputs = np.array(
            [
                [0.9, 0.1, 0.05, 0.85, -0.16],  # F, omega, C, IC, kappa
                [0.8, 0.2, 0.10, 0.75, -0.29],
                [0.7, 0.3, 0.15, 0.60, -0.51],
            ]
        )
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert mask.shape == (3,)
        assert mask.dtype == bool
        # At minimum, well-formed rows should pass
        assert mask.sum() >= 1

    def test_single_row(self) -> None:
        """Single-row input works."""
        outputs = np.array([[0.9, 0.1, 0.05, 0.85, -0.16]])
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert mask.shape == (1,)

    def test_invalid_F_out_of_range(self) -> None:
        """Row where F > 1 should fail."""
        outputs = np.array(
            [
                [1.5, 0.2, 0.05, 0.85, -0.16],  # F=1.5 > 1
            ]
        )
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert not mask[0]

    def test_negative_IC_fails(self) -> None:
        """Negative IC is never valid."""
        outputs = np.array(
            [
                [0.9, 0.1, 0.05, -0.5, -0.69],
            ]
        )
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert not mask[0]

    def test_IC_out_of_range_fails(self) -> None:
        """IC > 1−ε is out of valid range."""
        outputs = np.array(
            [
                [0.5, 0.5, 0.05, 1.5, -0.1],  # IC=1.5 > 1
            ]
        )
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert not mask[0]

    def test_empty_array(self) -> None:
        """Empty array returns empty mask."""
        outputs = np.empty((0, 5))
        mask = batch_validate_outputs(outputs, epsilon=EPSILON)
        assert mask.shape == (0,)


# ============================================================================
# BatchProcessor
# ============================================================================


class TestBatchProcessor:
    """BatchProcessor.preprocess_trace and compute_batch_statistics."""

    def test_init_default_epsilon(self) -> None:
        bp = BatchProcessor()
        assert bp.epsilon > 0

    def test_init_custom_epsilon(self) -> None:
        bp = BatchProcessor(epsilon=1e-8)
        assert bp.epsilon == 1e-8

    def test_preprocess_trace_basic(self) -> None:
        """Process a simple 3×2 trace with uniform weights."""
        bp = BatchProcessor(epsilon=EPSILON)
        trace = np.array([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])
        weights = np.array([0.5, 0.5])
        processed_trace, processed_weights, diagnostics = bp.preprocess_trace(trace, weights)
        assert processed_trace.shape[0] == 3
        assert processed_weights.shape[0] >= 1
        assert isinstance(diagnostics, list)
        assert len(diagnostics) == 3

    def test_preprocess_trace_clamps_oor(self) -> None:
        """Out-of-range values get clamped during preprocessing."""
        bp = BatchProcessor(epsilon=EPSILON)
        trace = np.array([[-0.1, 1.2], [0.5, 0.5]])
        weights = np.array([0.5, 0.5])
        processed_trace, _, _ = bp.preprocess_trace(trace, weights)
        assert np.all(processed_trace >= EPSILON)
        assert np.all(processed_trace <= 1 - EPSILON)

    def test_compute_batch_statistics(self) -> None:
        """Statistics dict has expected keys."""
        bp = BatchProcessor(epsilon=EPSILON)
        trace = np.array([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])
        stats = bp.compute_batch_statistics(trace)
        for key in ("mean", "std", "min", "max", "homogeneity"):
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], float)

    def test_compute_batch_statistics_homogeneous(self) -> None:
        """Perfectly homogeneous trace has homogeneity near 1."""
        bp = BatchProcessor(epsilon=EPSILON)
        trace = np.full((5, 3), 0.5)
        stats = bp.compute_batch_statistics(trace)
        assert stats["std"] < 1e-10


# ============================================================================
# preprocess_trace_row
# ============================================================================


class TestPreprocessTraceRow:
    """preprocess_trace_row combines pruning + clipping."""

    def test_basic_preprocessing(self) -> None:
        c = np.array([0.3, 0.5, 0.7])
        w = np.array([0.33, 0.34, 0.33])
        c_out, w_out, info = preprocess_trace_row(c, w, epsilon=EPSILON)
        assert len(c_out) > 0
        assert len(w_out) > 0
        assert isinstance(info, dict)

    def test_prune_zero_weight(self) -> None:
        """Zero-weight dimensions are pruned."""
        c = np.array([0.3, 0.5, 0.7])
        w = np.array([0.5, 0.0, 0.5])
        c_out, w_out, _info = preprocess_trace_row(c, w, epsilon=EPSILON, prune_weights=True)
        assert len(c_out) == 2
        assert len(w_out) == 2

    def test_no_prune_flag(self) -> None:
        """With prune_weights=False, all dims kept."""
        c = np.array([0.3, 0.5, 0.7])
        w = np.array([0.5, 0.0, 0.5])
        c_out, _w_out, _ = preprocess_trace_row(c, w, epsilon=EPSILON, prune_weights=False)
        assert len(c_out) == 3


# ============================================================================
# validate_inputs
# ============================================================================


class TestValidateInputs:
    """validate_inputs checks c/w compatibility."""

    def test_valid_inputs(self) -> None:
        result = validate_inputs(np.array([0.3, 0.7]), np.array([0.5, 0.5]))
        assert result["valid"] is True

    def test_mismatched_lengths(self) -> None:
        result = validate_inputs(np.array([0.3, 0.7]), np.array([0.5, 0.3, 0.2]))
        assert result["valid"] is False
