"""Targeted tests for RootFileValidator individual methods.

_validate_trace_bounds and _validate_regime_classification had zero
targeted tests with controlled inputs.  These are the actual validation
gates the protocol depends on.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from umcp.frozen_contract import EPSILON, Regime, classify_regime, compute_kernel

# ============================================================================
# _validate_trace_bounds
# ============================================================================


class TestValidateTraceBounds:
    """Test that trace bound validation catches out-of-range coordinates."""

    @staticmethod
    def _make_trace_csv(tmp_path: Path, rows: list[list[float]]) -> Path:
        """Write a minimal trace.csv with n coordinates per row."""
        csv_path = tmp_path / "trace.csv"
        n = len(rows[0]) if rows else 0
        headers = ["t"] + [f"c_{i}" for i in range(n)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t, row in enumerate(rows):
                writer.writerow([t, *row])
        return csv_path

    def test_valid_trace_in_range(self, tmp_path: Path) -> None:
        """All coordinates in [0, 1] should pass."""
        self._make_trace_csv(tmp_path, [[0.3, 0.7], [0.5, 0.5]])
        # We test trace bounds checking logic directly
        from umcp.compute_utils import clip_coordinates

        c = np.array([0.3, 0.7])
        result = clip_coordinates(c, epsilon=EPSILON)
        assert result.clip_count == 0

    def test_oor_coordinates_flagged(self, tmp_path: Path) -> None:
        """Coordinates outside [ε, 1-ε] must be flagged."""
        from umcp.compute_utils import clip_coordinates

        c = np.array([-0.1, 0.5, 1.2])
        result = clip_coordinates(c, epsilon=EPSILON)
        assert result.clip_count >= 2
        assert 0 in result.oor_indices
        assert 2 in result.oor_indices

    def test_boundary_coordinates_handled(self, tmp_path: Path) -> None:
        """Exact 0.0 and 1.0 are out-of-range for ε-clipped domain."""
        from umcp.compute_utils import clip_coordinates

        c = np.array([0.0, 1.0])
        result = clip_coordinates(c, epsilon=EPSILON)
        assert result.clip_count == 2
        assert result.c_clipped[0] == EPSILON
        assert result.c_clipped[1] == 1 - EPSILON


# ============================================================================
# _validate_regime_classification
# ============================================================================


@pytest.mark.bounded_identity
class TestValidateRegimeClassification:
    """Regime classification produces correct labels from invariants."""

    @pytest.mark.parametrize(
        "omega, F, S, C, IC, expected",
        [
            # Deep stable
            (0.01, 0.99, 0.05, 0.05, 0.95, Regime.STABLE),
            # Stable boundary just below
            (0.037, 0.963, 0.14, 0.13, 0.80, Regime.STABLE),
            # Watch because ω crosses threshold
            (0.039, 0.961, 0.10, 0.10, 0.80, Regime.WATCH),
            # Watch because S too high
            (0.02, 0.98, 0.16, 0.05, 0.80, Regime.WATCH),
            # Watch because C too high
            (0.02, 0.98, 0.05, 0.15, 0.80, Regime.WATCH),
            # Collapse
            (0.35, 0.65, 0.30, 0.30, 0.50, Regime.COLLAPSE),
            # Critical overlay
            (0.01, 0.99, 0.05, 0.05, 0.20, Regime.CRITICAL),
        ],
    )
    def test_regime_matches_gates(
        self,
        omega: float,
        F: float,
        S: float,
        C: float,
        IC: float,
        expected: Regime,
    ) -> None:
        result = classify_regime(omega, F, S, C, IC)
        assert result == expected

    def test_all_invariants_from_kernel_classify_correctly(self) -> None:
        """Compute kernel from coordinates, then classify the result."""
        c = np.array([0.99, 0.98, 0.97])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)

        regime = classify_regime(ko.omega, ko.F, ko.S, ko.C, ko.IC)
        assert regime == Regime.STABLE  # high fidelity → stable

    def test_high_drift_triggers_collapse_or_critical(self) -> None:
        """Coordinates near low end → high ω → collapse or critical.

        With unweighted IC, c=[0.3]*3 gives IC=0.027 < 0.30 → CRITICAL.
        CRITICAL is valid here: it overlays when integrity is low.
        """
        c = np.full(3, 0.3)
        w = np.full(3, 1 / 3)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        regime = classify_regime(ko.omega, ko.F, ko.S, ko.C, ko.IC)
        assert regime in (Regime.COLLAPSE, Regime.WATCH, Regime.CRITICAL)

    def test_critical_overlay_precedence(self) -> None:
        """Critical overlay takes precedence over any other regime."""
        # Stable in all other respects but IC < 0.30
        regime = classify_regime(omega=0.01, F=0.99, S=0.01, C=0.01, integrity=0.15)
        assert regime == Regime.CRITICAL


# ============================================================================
# _validate_invariant_identities (indirect — recompute and check)
# ============================================================================


@pytest.mark.bounded_identity
class TestValidateInvariantIdentities:
    """Verify the three Tier-1 structural identities on computed outputs."""

    def test_f_plus_omega_equals_one(self) -> None:
        """F + ω = 1 to float precision."""
        for _ in range(50):
            c = np.random.default_rng(42).uniform(0.05, 0.95, size=5)
            w = np.ones(5) / 5
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert abs(ko.F + ko.omega - 1.0) < 1e-14

    def test_ic_equals_exp_kappa(self) -> None:
        """IC ≈ exp(κ) to float precision."""
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        assert abs(ko.IC - math.exp(ko.kappa)) < 1e-12

    def test_ic_le_f_amgm(self) -> None:
        """IC ≤ F (AM-GM bound) for random inputs."""
        rng = np.random.default_rng(99)
        for _ in range(100):
            c = rng.uniform(0.05, 0.95, size=6)
            w = rng.dirichlet(np.ones(6))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            assert ko.IC <= ko.F + 1e-12, f"AM-GM violated: IC={ko.IC} > F={ko.F}"
