"""
Tests for compute_utils.py - Kernel computation utilities.

These tests verify that the compute utilities correctly implement
the preprocessing and validation as specified in KERNEL_SPECIFICATION.md.
"""

import numpy as np
import pytest

from umcp.compute_utils import (
    ClippingResult,
    PruningResult,
    clip_coordinates,
    normalize_weights,
    prune_zero_weights,
)


class TestClipCoordinates:
    """Tests for the coordinate clipping function."""

    def test_clip_respects_epsilon(self):
        """Values are clipped to [ε, 1-ε]."""
        c = np.array([0.0, 0.5, 1.0])
        eps = 1e-6
        result = clip_coordinates(c, epsilon=eps)
        assert isinstance(result, ClippingResult)
        assert result.c_clipped[0] >= eps
        assert result.c_clipped[2] <= 1 - eps
        assert result.c_clipped[1] == 0.5  # Interior unchanged

    def test_clip_preserves_interior(self):
        """Interior values unchanged."""
        c = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = clip_coordinates(c, epsilon=1e-6)
        np.testing.assert_array_almost_equal(result.c_clipped, c)

    def test_clip_tracks_modifications(self):
        """Clipping result tracks what was modified."""
        c = np.array([0.0, 0.5, 1.0])
        result = clip_coordinates(c, epsilon=1e-6)
        # First and last values should be flagged as clipped
        assert result.clip_count >= 2

    def test_clip_perturbation_computed(self):
        """Clipping records perturbation magnitude."""
        c = np.array([0.0, 0.5, 1.0])
        result = clip_coordinates(c, epsilon=1e-6)
        assert result.clip_perturbation > 0
        assert result.max_perturbation > 0

    def test_clip_oor_indices_tracked(self):
        """Out of range indices are tracked."""
        c = np.array([0.0, 0.5, 1.0])
        result = clip_coordinates(c, epsilon=1e-6)
        assert 0 in result.oor_indices
        assert 2 in result.oor_indices


class TestNormalizeWeights:
    """Tests for weight normalization."""

    def test_weights_sum_to_one(self):
        """Normalized weights sum to 1."""
        w = np.array([1.0, 2.0, 3.0])
        normalized = normalize_weights(w)
        assert abs(normalized.sum() - 1.0) < 1e-10

    def test_normalize_preserves_ratios(self):
        """Normalization preserves relative ratios."""
        w = np.array([1.0, 2.0, 4.0])
        normalized = normalize_weights(w)
        assert abs(normalized[1] / normalized[0] - 2.0) < 1e-10
        assert abs(normalized[2] / normalized[0] - 4.0) < 1e-10

    def test_normalize_uniform(self):
        """Equal weights normalize to uniform."""
        w = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = normalize_weights(w)
        np.testing.assert_array_almost_equal(normalized, [0.25, 0.25, 0.25, 0.25])

    def test_normalize_handles_zero_sum(self):
        """Zero sum raises ValueError."""
        w = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            normalize_weights(w)


class TestPruneZeroWeights:
    """Tests for zero weight pruning."""

    def test_prune_removes_zeros(self):
        """Zero weights are removed."""
        c = np.array([0.5, 0.6, 0.7, 0.8])
        w = np.array([0.5, 0.0, 0.3, 0.2])
        result = prune_zero_weights(c, w)
        assert isinstance(result, PruningResult)
        assert result.n_active == 3
        assert len(result.c_active) == 3
        assert len(result.w_active) == 3

    def test_prune_preserves_nonzero(self):
        """Non-zero weights preserved."""
        c = np.array([0.5, 0.6, 0.7])
        w = np.array([0.3, 0.3, 0.4])
        result = prune_zero_weights(c, w)
        np.testing.assert_array_almost_equal(result.c_active, c)

    def test_prune_renormalizes_weights(self):
        """Pruned weights are renormalized to sum to 1."""
        c = np.array([0.5, 0.6, 0.7, 0.8])
        w = np.array([0.4, 0.0, 0.3, 0.3])
        result = prune_zero_weights(c, w)
        assert abs(result.w_active.sum() - 1.0) < 1e-10

    def test_prune_tracks_indices(self):
        """Pruned indices are tracked."""
        c = np.array([0.5, 0.6, 0.7, 0.8])
        w = np.array([0.5, 0.0, 0.3, 0.2])
        result = prune_zero_weights(c, w)
        assert 1 in result.pruned_indices

    def test_prune_all_zero_raises(self):
        """All zero weights raises ValueError."""
        c = np.array([0.5, 0.6, 0.7])
        w = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            prune_zero_weights(c, w)
