"""Tests for Return Rope — Adaptive return cycle protocol.

Validates the mathematical properties of the return rope:
- Contraction bound: IC · ω ≤ 1/4
- Grip monotonicity: grip increases with each parse
- Rope monotonicity: τ_R decreases with each parse
- Tier-1 identity preservation: F + ω = 1, IC ≤ F in every parse
- Gesture detection: IC < ε snaps the rope (τ_R = ∞_rec)
- Convergence: rope stabilizes when |Δτ| < tol_seam
- Contraction bound proof: empirical sweep over 10K traces
- Grip convergence: geometric series analysis
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from umcp.return_rope import (
    ParseResult,
    ReturnRope,
    contraction_bound_proof,
    grip_convergence_analysis,
)

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def rope() -> ReturnRope:
    """Fresh rope with default settings."""
    return ReturnRope()


@pytest.fixture()
def healthy_trace() -> list[float]:
    """High-fidelity 8-channel trace (Stable regime)."""
    return [0.95, 0.92, 0.90, 0.93, 0.91, 0.94, 0.92, 0.93]


@pytest.fixture()
def mixed_trace() -> list[float]:
    """Mixed-channel trace (Watch regime)."""
    return [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85]


@pytest.fixture()
def dead_channel_trace() -> list[float]:
    """Trace with multiple near-zero channels (triggers gesture)."""
    return [0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10]


# ─────────────────────────────────────────────────────────────────
# Basic construction
# ─────────────────────────────────────────────────────────────────


class TestConstruction:
    """ReturnRope initialization."""

    def test_default_initial_tau(self) -> None:
        rope = ReturnRope()
        assert rope.initial_tau == 1.0
        assert rope.state.tau_r == 1.0

    def test_custom_initial_tau(self) -> None:
        rope = ReturnRope(initial_tau=5.0)
        assert rope.initial_tau == 5.0
        assert rope.state.tau_r == 5.0

    def test_initial_grip_is_zero(self) -> None:
        rope = ReturnRope()
        assert rope.state.grip == 0.0

    def test_initial_iteration_is_zero(self) -> None:
        rope = ReturnRope()
        assert rope.state.iteration == 0

    def test_initial_not_converged(self) -> None:
        rope = ReturnRope()
        assert not rope.state.converged

    def test_initial_not_snapped(self) -> None:
        rope = ReturnRope()
        assert not rope.state.snapped

    def test_initial_empty_parses(self) -> None:
        rope = ReturnRope()
        assert rope.state.parses == []

    def test_empty_summary(self) -> None:
        rope = ReturnRope()
        s = rope.summary()
        assert s["parses"] == 0
        assert s["grip"] == 0.0


# ─────────────────────────────────────────────────────────────────
# Single parse
# ─────────────────────────────────────────────────────────────────


class TestSingleParse:
    """Single parse behavior."""

    def test_parse_returns_result(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace, label="first")
        assert isinstance(result, ParseResult)

    def test_parse_increments_iteration(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert result.iteration == 1
        assert rope.state.iteration == 1

    def test_parse_label(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace, label="test_parse")
        assert result.label == "test_parse"

    def test_parse_stores_trace(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert len(result.trace) == 8

    def test_parse_uniform_weights(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert all(abs(w - 0.125) < 1e-12 for w in result.weights)

    def test_parse_contracts_rope(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert result.tau_r < 1.0

    def test_parse_increases_grip(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert result.grip > 0.0

    def test_parse_not_gesture(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        result = rope.parse(healthy_trace)
        assert not result.is_gesture


# ─────────────────────────────────────────────────────────────────
# Tier-1 identity preservation
# ─────────────────────────────────────────────────────────────────


class TestTier1Identities:
    """Every parse must satisfy Tier-1 kernel identities."""

    @pytest.mark.parametrize(
        "trace",
        [
            [0.95, 0.92, 0.90, 0.93, 0.91, 0.94, 0.92, 0.93],
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
            [0.99, 0.99, 0.99, 0.01, 0.99, 0.99, 0.99, 0.99],
        ],
        ids=["healthy", "mixed", "equator", "alternating", "one_weak"],
    )
    def test_duality_identity(self, trace: list[float]) -> None:
        """F + ω = 1 exactly."""
        rope = ReturnRope()
        result = rope.parse(trace)
        assert abs(result.F + result.omega - 1.0) < 1e-14

    @pytest.mark.parametrize(
        "trace",
        [
            [0.95, 0.92, 0.90, 0.93, 0.91, 0.94, 0.92, 0.93],
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
        ],
        ids=["healthy", "mixed", "equator", "alternating"],
    )
    def test_integrity_bound(self, trace: list[float]) -> None:
        """IC ≤ F (integrity cannot exceed fidelity)."""
        rope = ReturnRope()
        result = rope.parse(trace)
        assert result.IC <= result.F + 1e-14

    @pytest.mark.parametrize(
        "trace",
        [
            [0.95, 0.92, 0.90, 0.93, 0.91, 0.94, 0.92, 0.93],
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        ids=["healthy", "mixed", "equator"],
    )
    def test_log_integrity_relation(self, trace: list[float]) -> None:
        """IC = exp(κ)."""
        rope = ReturnRope()
        result = rope.parse(trace)
        assert abs(result.IC - math.exp(result.kappa)) < 1e-10


# ─────────────────────────────────────────────────────────────────
# Contraction bound
# ─────────────────────────────────────────────────────────────────


class TestContractionBound:
    """IC · ω ≤ 1/4 — the contraction is bounded."""

    @pytest.mark.parametrize(
        "trace",
        [
            [0.95, 0.92, 0.90, 0.93, 0.91, 0.94, 0.92, 0.93],
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
            [0.01, 0.99, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        ids=["healthy", "mixed", "equator", "alternating", "extreme"],
    )
    def test_contraction_bounded(self, trace: list[float]) -> None:
        """Each parse: IC · ω ≤ 0.25."""
        rope = ReturnRope()
        result = rope.parse(trace)
        assert result.contraction <= 0.25 + 1e-12

    def test_contraction_bound_proof_empirical(self) -> None:
        """Sweep 10K random traces — bound must hold everywhere."""
        proof = contraction_bound_proof(n_samples=10_000)
        assert proof["bound_holds"]
        assert proof["max_product"] <= 0.25 + 1e-12
        assert proof["margin"] >= 0.0

    def test_equator_maximizes_contraction(self) -> None:
        """The equator (all channels = 0.5) approaches max contraction."""
        rope = ReturnRope()
        # Homogeneous at equator: F = IC = 0.5, ω = 0.5, IC·ω = 0.25
        result = rope.parse([0.5] * 8)
        # Should be very close to 0.25 (max)
        assert result.contraction > 0.24


# ─────────────────────────────────────────────────────────────────
# Rope monotonicity
# ─────────────────────────────────────────────────────────────────


class TestRopeMonotonicity:
    """Rope length monotonically decreases; grip monotonically increases."""

    def test_tau_decreases_across_parses(self) -> None:
        rope = ReturnRope()
        traces = [
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.85, 0.75, 0.55, 0.35, 0.82, 0.65, 0.45, 0.88],
            [0.88, 0.78, 0.58, 0.38, 0.84, 0.68, 0.48, 0.90],
            [0.90, 0.80, 0.60, 0.40, 0.86, 0.70, 0.50, 0.91],
        ]
        results = rope.explore(traces, labels=[f"parse_{i}" for i in range(4)])
        taus = [r.tau_r for r in results]
        for i in range(1, len(taus)):
            assert taus[i] < taus[i - 1], f"τ not decreasing at parse {i}"

    def test_grip_increases_across_parses(self) -> None:
        rope = ReturnRope()
        traces = [
            [0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.85],
            [0.85, 0.75, 0.55, 0.35, 0.82, 0.65, 0.45, 0.88],
            [0.88, 0.78, 0.58, 0.38, 0.84, 0.68, 0.48, 0.90],
        ]
        results = rope.explore(traces)
        grips = [r.grip for r in results]
        for i in range(1, len(grips)):
            assert grips[i] > grips[i - 1], f"Grip not increasing at parse {i}"

    def test_grip_bounded_below_one(self) -> None:
        rope = ReturnRope()
        for _ in range(20):
            rope.parse([0.5] * 8)
        assert rope.state.grip < 1.0


# ─────────────────────────────────────────────────────────────────
# Gesture detection (rope snap)
# ─────────────────────────────────────────────────────────────────


class TestGestureDetection:
    """IC < ε → rope snaps — territory is a gesture, not a weld."""

    def test_dead_channel_triggers_gesture(self, rope: ReturnRope) -> None:
        """Multiple near-zero channels should kill IC via geometric slaughter."""
        result = rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        assert result.is_gesture
        assert rope.state.snapped

    def test_gesture_sets_inf_tau(self, rope: ReturnRope) -> None:
        result = rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        assert result.tau_r == float("inf")

    def test_gesture_freezes_grip(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        # First parse builds some grip
        r1 = rope.parse(healthy_trace)
        grip_before = r1.grip
        # Gesture parse should freeze grip at current value
        r2 = rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        assert r2.grip == grip_before  # Grip frozen at pre-gesture value

    def test_cannot_parse_after_snap(self, rope: ReturnRope) -> None:
        rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        with pytest.raises(RuntimeError, match="snapped"):
            rope.parse([0.9] * 8)

    def test_gesture_contraction_is_zero(self, rope: ReturnRope) -> None:
        result = rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        assert result.contraction == 0.0


# ─────────────────────────────────────────────────────────────────
# Convergence
# ─────────────────────────────────────────────────────────────────


class TestConvergence:
    """Rope converges when |Δτ| < tol_seam."""

    def test_high_fidelity_converges_quickly(self) -> None:
        """Near-perfect traces have tiny IC·ω, so Δτ is small."""
        rope = ReturnRope()
        # Very high fidelity → small ω → small IC·ω → small Δτ
        trace = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
        # With F≈0.99, ω≈0.01, IC≈0.99 → IC·ω ≈ 0.0099
        # Δτ_k = τ_k · 0.0099; converges when τ_k < 0.505
        # That takes ~70 parses
        for _ in range(80):
            rope.parse(trace)
        assert rope.state.converged

    def test_not_converged_after_one_parse(self, rope: ReturnRope, mixed_trace: list[float]) -> None:
        rope.parse(mixed_trace)
        assert not rope.state.converged


# ─────────────────────────────────────────────────────────────────
# Explore (multi-parse)
# ─────────────────────────────────────────────────────────────────


class TestExplore:
    """Multi-parse exploration convenience method."""

    def test_explore_returns_all_results(self, rope: ReturnRope) -> None:
        traces = [[0.5] * 8, [0.6] * 8, [0.7] * 8]
        results = rope.explore(traces)
        assert len(results) == 3

    def test_explore_default_labels(self, rope: ReturnRope) -> None:
        traces = [[0.5] * 8, [0.6] * 8]
        results = rope.explore(traces)
        assert results[0].label == "parse_1"
        assert results[1].label == "parse_2"

    def test_explore_custom_labels(self, rope: ReturnRope) -> None:
        traces = [[0.5] * 8, [0.6] * 8]
        results = rope.explore(traces, labels=["alpha", "beta"])
        assert results[0].label == "alpha"
        assert results[1].label == "beta"

    def test_explore_stops_on_gesture(self, rope: ReturnRope) -> None:
        traces = [
            [0.9] * 8,
            [0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10],  # Gesture
            [0.9] * 8,  # Should not be reached
        ]
        results = rope.explore(traces)
        assert len(results) == 2
        assert results[-1].is_gesture


# ─────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────


class TestReset:
    """Reset returns rope to initial state."""

    def test_reset_restores_tau(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        rope.parse(healthy_trace)
        rope.reset()
        assert rope.state.tau_r == 1.0

    def test_reset_clears_grip(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        rope.parse(healthy_trace)
        rope.reset()
        assert rope.state.grip == 0.0

    def test_reset_clears_parses(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        rope.parse(healthy_trace)
        rope.reset()
        assert rope.state.parses == []

    def test_reset_clears_snapped(self, rope: ReturnRope) -> None:
        rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        rope.reset()
        assert not rope.state.snapped

    def test_can_parse_after_reset(self, rope: ReturnRope, healthy_trace: list[float]) -> None:
        rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        rope.reset()
        result = rope.parse(healthy_trace)
        assert not result.is_gesture


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────


class TestSummary:
    """Summary statistics."""

    def test_summary_after_parses(self) -> None:
        rope = ReturnRope()
        rope.parse([0.5] * 8)
        rope.parse([0.6] * 8)
        s = rope.summary()
        assert s["parses"] == 2
        assert s["grip"] > 0
        assert s["final_tau"] < s["initial_tau"]
        assert s["contraction_ratio"] < 1.0
        assert 0 < s["mean_ic"] <= 1
        assert 0 <= s["mean_omega"] <= 1
        assert s["gesture_count"] == 0

    def test_summary_gesture_count(self) -> None:
        rope = ReturnRope()
        rope.parse([0.9, 0.9, 0.9, 0.9, 0.9, 1e-10, 1e-10, 1e-10])
        s = rope.summary()
        assert s["gesture_count"] == 1
        assert s["snapped"]


# ─────────────────────────────────────────────────────────────────
# RopeState sequence properties
# ─────────────────────────────────────────────────────────────────


class TestStateSequences:
    """RopeState property sequences."""

    def test_ic_sequence(self) -> None:
        rope = ReturnRope()
        rope.parse([0.5] * 8)
        rope.parse([0.7] * 8)
        assert len(rope.state.ic_sequence) == 2
        assert all(0 < ic <= 1 for ic in rope.state.ic_sequence)

    def test_tau_sequence(self) -> None:
        rope = ReturnRope()
        rope.parse([0.5] * 8)
        rope.parse([0.7] * 8)
        taus = rope.state.tau_sequence
        assert len(taus) == 2
        assert taus[0] > taus[1]  # Monotonically decreasing

    def test_grip_sequence(self) -> None:
        rope = ReturnRope()
        rope.parse([0.5] * 8)
        rope.parse([0.7] * 8)
        grips = rope.state.grip_sequence
        assert len(grips) == 2
        assert grips[1] > grips[0]  # Monotonically increasing

    def test_contraction_sequence(self) -> None:
        rope = ReturnRope()
        rope.parse([0.5] * 8)
        rope.parse([0.7] * 8)
        contractions = rope.state.contraction_sequence
        assert len(contractions) == 2
        assert all(0 < c <= 0.25 + 1e-12 for c in contractions)


# ─────────────────────────────────────────────────────────────────
# Contraction bound proof (standalone function)
# ─────────────────────────────────────────────────────────────────


class TestContractionBoundProof:
    """Standalone proof function."""

    def test_bound_holds(self) -> None:
        proof = contraction_bound_proof(n_samples=5_000)
        assert proof["bound_holds"]

    def test_margin_positive(self) -> None:
        proof = contraction_bound_proof(n_samples=5_000)
        assert proof["margin"] >= 0.0


# ─────────────────────────────────────────────────────────────────
# Grip convergence analysis (standalone function)
# ─────────────────────────────────────────────────────────────────


class TestGripConvergence:
    """Standalone grip convergence analysis."""

    def test_grip_approaches_one(self) -> None:
        result = grip_convergence_analysis(ic_omega_constant=0.1, n_parses=100)
        assert result["final_grip"] > 0.999

    def test_half_life(self) -> None:
        result = grip_convergence_analysis(ic_omega_constant=0.1)
        # ln(0.5)/ln(0.9) ≈ 6.58 → ceil = 7
        assert result["half_life"] == 7

    def test_ninety_life(self) -> None:
        result = grip_convergence_analysis(ic_omega_constant=0.1)
        # ln(0.1)/ln(0.9) ≈ 21.85 → ceil = 22
        assert result["ninety_life"] == 22

    def test_small_contraction_slower(self) -> None:
        fast = grip_convergence_analysis(ic_omega_constant=0.2)
        slow = grip_convergence_analysis(ic_omega_constant=0.05)
        assert fast["half_life"] < slow["half_life"]


# ─────────────────────────────────────────────────────────────────
# Hubble tension demonstration
# ─────────────────────────────────────────────────────────────────


class TestHubbleTensionExploration:
    """Run the return rope on H₀ measurement traces as a demonstration.

    This shows the rope in action: exploring the Hubble tension
    landscape measurement-by-measurement.
    """

    @pytest.fixture()
    def h0_traces(self) -> list[list[float]]:
        """Simplified H₀ measurement traces (8-channel)."""
        return [
            # Early universe (CMB-like: high systemic control, model-dependent)
            [0.99, 0.5, 0.99, 0.80, 0.95, 0.70, 0.90, 0.85],  # Planck
            [0.98, 0.5, 0.98, 0.75, 0.92, 0.65, 0.88, 0.80],  # ACT DR6
            # Late universe (distance ladder: lower model dependence, cross-checks)
            [0.90, 0.7, 0.85, 0.50, 0.80, 0.95, 0.75, 0.90],  # SH0ES
            [0.88, 0.7, 0.82, 0.55, 0.78, 0.90, 0.72, 0.88],  # TRGB
            # Geometric methods (fewer systematics, independent)
            [0.85, 0.8, 0.90, 0.60, 0.85, 0.92, 0.80, 0.95],  # TDCOSMO
            [0.80, 0.85, 0.88, 0.65, 0.82, 0.88, 0.78, 0.92],  # Megamasers
        ]

    def test_rope_contracts_through_h0_landscape(self, h0_traces: list[list[float]]) -> None:
        rope = ReturnRope()
        results = rope.explore(
            h0_traces,
            labels=["Planck", "ACT_DR6", "SH0ES", "TRGB", "TDCOSMO", "Megamasers"],
        )
        assert len(results) == 6
        # Rope should have contracted
        assert rope.state.tau_r < 1.0
        # Grip should be positive and growing
        assert rope.state.grip > 0
        # All parses should satisfy Tier-1 identities
        for r in results:
            assert abs(r.F + r.omega - 1.0) < 1e-14
            assert r.IC <= r.F + 1e-14
            assert not r.is_gesture

    def test_early_vs_late_heterogeneity(self, h0_traces: list[list[float]]) -> None:
        """Combined exploration should build grip across the landscape."""
        rope = ReturnRope()
        results = rope.explore(h0_traces, labels=["Planck", "ACT", "SH0ES", "TRGB", "TDCOSMO", "Mega"])
        # Grip should accumulate across all 6 parses
        assert results[-1].grip > results[0].grip
        # All should satisfy the contraction bound
        for r in results:
            assert r.contraction <= 0.25 + 1e-12


# ─────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_two_channel_trace(self) -> None:
        rope = ReturnRope()
        result = rope.parse([0.8, 0.6])
        assert abs(result.F + result.omega - 1.0) < 1e-14

    def test_single_channel_trace(self) -> None:
        rope = ReturnRope()
        result = rope.parse([0.7])
        assert abs(result.F + result.omega - 1.0) < 1e-14

    def test_many_channels(self) -> None:
        rope = ReturnRope()
        result = rope.parse([0.5 + 0.01 * i for i in range(32)])
        assert result.F > 0

    def test_custom_weights(self) -> None:
        rope = ReturnRope()
        result = rope.parse([0.8, 0.6, 0.4], weights=[0.5, 0.3, 0.2])
        assert abs(result.F + result.omega - 1.0) < 1e-14

    def test_numpy_input(self) -> None:
        rope = ReturnRope()
        result = rope.parse(np.array([0.8, 0.6, 0.4, 0.7]))
        assert abs(result.F + result.omega - 1.0) < 1e-14

    def test_large_initial_tau(self) -> None:
        rope = ReturnRope(initial_tau=1000.0)
        result = rope.parse([0.5] * 8)
        assert result.tau_r < 1000.0

    def test_very_small_convergence_tol(self) -> None:
        rope = ReturnRope(convergence_tol=1e-15)
        # Use high-contraction traces to reach convergence faster
        # At equator: IC·ω ≈ 0.25, so τ shrinks rapidly
        for _ in range(200):
            rope.parse([0.5] * 8)
        # After 200 parses at max contraction, τ ≈ (0.75)^200 ≈ 1.3e-25
        # Δτ ≈ τ · 0.25 ≈ 3e-26 < 1e-15
        assert rope.state.converged


# ─────────────────────────────────────────────────────────────────
# Grip mathematical properties
# ─────────────────────────────────────────────────────────────────


class TestGripProperties:
    """Mathematical properties of the grip function."""

    def test_grip_formula_consistency(self) -> None:
        """Verify grip = 1 - ∏(1 - IC_i · ω_i) by manual computation."""
        rope = ReturnRope()
        traces = [[0.5] * 8, [0.6] * 8, [0.7] * 8, [0.8] * 8]
        results = rope.explore(traces)

        # Manual computation
        gap_product = 1.0
        for r in results:
            gap_product *= 1.0 - r.contraction
        expected_grip = 1.0 - gap_product

        assert abs(rope.state.grip - expected_grip) < 1e-14

    def test_grip_is_subadditive(self) -> None:
        """Grip from combined exploration ≤ sum of individual grips."""
        rope1 = ReturnRope()
        rope1.parse([0.5] * 8)
        rope1.parse([0.6] * 8)
        combined_grip = rope1.state.grip

        rope_a = ReturnRope()
        rope_a.parse([0.5] * 8)
        rope_b = ReturnRope()
        rope_b.parse([0.6] * 8)
        sum_grip = rope_a.state.grip + rope_b.state.grip
        assert combined_grip <= sum_grip + 1e-12
