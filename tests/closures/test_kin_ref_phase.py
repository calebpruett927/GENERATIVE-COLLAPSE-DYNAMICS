#!/usr/bin/env python3
"""
Test suite for KIN.REF.PHASE closure.

Validates:
- Phase-anchor selection correctness
- Tie-breaker behavior (most recent u selected when Δφ ties)
- Empty eligible case produces EMPTY_ELIGIBLE censor
- Phase mismatch case produces PHASE_MISMATCH censor
- Determinism (two runs produce identical output)

CasePack: casepacks/kin_ref_phase_oscillator
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent

# Load the KIN.REF.PHASE closure module dynamically
closure_path = REPO_ROOT / "casepacks" / "kin_ref_phase_oscillator" / "closures" / "kin_ref_phase.py"
spec = importlib.util.spec_from_file_location("kin_ref_phase", closure_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from {closure_path}")
kin_ref_phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kin_ref_phase)

# Import the required items from the module
DEBOUNCE = kin_ref_phase.DEBOUNCE
DELTA_PHI_MAX = kin_ref_phase.DELTA_PHI_MAX
WINDOW = kin_ref_phase.WINDOW
UndefinedReason = kin_ref_phase.UndefinedReason
build_eligible_set = kin_ref_phase.build_eligible_set
circular_distance = kin_ref_phase.circular_distance
compute_phase = kin_ref_phase.compute_phase
get_frozen_config_sha256 = kin_ref_phase.get_frozen_config_sha256
process_trajectory = kin_ref_phase.process_trajectory
select_phase_anchor = kin_ref_phase.select_phase_anchor


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def casepack_dir() -> Path:
    """Get the KIN.REF.PHASE casepack directory."""
    return REPO_ROOT / "casepacks" / "kin_ref_phase_oscillator"


@pytest.fixture
def raw_measurements(casepack_dir: Path) -> tuple[list[float], list[float]]:
    """Load raw measurements from CSV."""
    csv_path = casepack_dir / "raw_measurements.csv"
    x_series: list[float] = []
    v_series: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_series.append(float(row["x"]))
            v_series.append(float(row["v"]))

    return x_series, v_series


@pytest.fixture
def expected_results(casepack_dir: Path) -> list[dict[str, Any]]:
    """Load expected results from CSV."""
    csv_path = casepack_dir / "expected" / "ref_phase_expected.csv"
    results: list[dict[str, Any]] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "idx": int(row["idx"]),
                    "phi": float(row["phi"]),
                    "eligible_count": int(row["eligible_count"]),
                    "anchor_u": int(row["anchor_u"]) if row["anchor_u"] else None,
                    "delta_phi": float(row["delta_phi"]) if row["delta_phi"] else None,
                    "undefined_reason": row["undefined_reason"],
                }
            )

    return results


@pytest.fixture
def expected_censors(casepack_dir: Path) -> list[dict[str, Any]]:
    """Load expected censor events from JSON."""
    json_path = casepack_dir / "expected" / "censor_expected.json"

    with open(json_path) as f:
        data = json.load(f)

    return data["censors"]


# =============================================================================
# UNIT TESTS: Core Functions
# =============================================================================


class TestComputePhase:
    """Tests for compute_phase function."""

    def test_phase_at_origin(self) -> None:
        """Test phase at centered origin (x=0.5, v=0.5 -> x'=0, v'=0)."""
        # At centered origin, atan2(0, 0) is undefined but returns 0
        phi = compute_phase(0.5, 0.5)
        assert phi == 0.0

    def test_phase_at_right(self) -> None:
        """Test phase at right (x=1, v=0.5 -> x'=1, v'=0)."""
        phi = compute_phase(1.0, 0.5)
        assert abs(phi - 0.0) < 1e-9  # atan2(0, 1) = 0

    def test_phase_at_top(self) -> None:
        """Test phase at top (x=0.5, v=1 -> x'=0, v'=1)."""
        phi = compute_phase(0.5, 1.0)
        assert abs(phi - np.pi / 2) < 1e-9  # atan2(1, 0) = π/2

    def test_phase_at_left(self) -> None:
        """Test phase at left (x=0, v=0.5 -> x'=-1, v'=0)."""
        phi = compute_phase(0.0, 0.5)
        assert abs(phi - np.pi) < 1e-9  # atan2(0, -1) = π

    def test_phase_at_bottom(self) -> None:
        """Test phase at bottom (x=0.5, v=0 -> x'=0, v'=-1)."""
        phi = compute_phase(0.5, 0.0)
        assert abs(phi - 3 * np.pi / 2) < 1e-9  # atan2(-1, 0) = -π/2 → 3π/2

    def test_phase_wrapping_positive(self) -> None:
        """Test phase is always in [0, 2π)."""
        for x in np.linspace(0, 1, 10):
            for v in np.linspace(0, 1, 10):
                phi = compute_phase(x, v)
                assert 0 <= phi < 2 * np.pi, f"Phase {phi} out of range for ({x}, {v})"


class TestCircularDistance:
    """Tests for circular_distance function."""

    def test_identical_phases(self) -> None:
        """Test distance between identical phases."""
        assert circular_distance(0.0, 0.0) == 0.0
        assert circular_distance(np.pi, np.pi) == 0.0

    def test_opposite_phases(self) -> None:
        """Test distance between opposite phases (π apart)."""
        d = circular_distance(0.0, np.pi)
        assert abs(d - np.pi) < 1e-9

    def test_wrap_around(self) -> None:
        """Test wrap-around distance calculation."""
        # Distance from 0 to 2π - 0.1 should be 0.1 (wrap around)
        d = circular_distance(0.0, 2 * np.pi - 0.1)
        assert abs(d - 0.1) < 1e-9

    def test_symmetry(self) -> None:
        """Test that circular distance is symmetric."""
        for a in np.linspace(0, 2 * np.pi, 10):
            for b in np.linspace(0, 2 * np.pi, 10):
                assert abs(circular_distance(a, b) - circular_distance(b, a)) < 1e-12


class TestBuildEligibleSet:
    """Tests for build_eligible_set function."""

    def test_startup_phase(self) -> None:
        """Test that startup phase has empty eligible set."""
        assert build_eligible_set(0) == []
        assert build_eligible_set(1) == []
        assert build_eligible_set(2) == []

    def test_first_eligible(self) -> None:
        """Test first time index with eligible anchors."""
        eligible = build_eligible_set(3)
        assert eligible == [0]

    def test_growing_window(self) -> None:
        """Test eligible set grows with t."""
        e3 = build_eligible_set(3)
        e4 = build_eligible_set(4)
        assert len(e4) == len(e3) + 1

    def test_window_limit(self) -> None:
        """Test eligible set is bounded by window size."""
        # At t >= WINDOW + DEBOUNCE, eligible set should have WINDOW - DEBOUNCE + 1 elements
        e_large = build_eligible_set(100)
        expected_size = WINDOW - DEBOUNCE + 1  # 20 - 3 + 1 = 18
        assert len(e_large) == expected_size


# =============================================================================
# INTEGRATION TESTS: Full Selection
# =============================================================================


class TestSelectPhaseAnchor:
    """Tests for select_phase_anchor function."""

    def test_empty_eligible_returns_censor(self, raw_measurements: tuple[list[float], list[float]]) -> None:
        """Test that empty eligible set produces EMPTY_ELIGIBLE."""
        x_series, v_series = raw_measurements

        for t in [0, 1, 2]:
            result = select_phase_anchor(x_series, v_series, t)
            assert result.undefined_reason == UndefinedReason.EMPTY_ELIGIBLE
            assert result.anchor_u is None
            assert result.eligible_count == 0

    def test_defined_anchor_exists(
        self, raw_measurements: tuple[list[float], list[float]], expected_results: list[dict[str, Any]]
    ) -> None:
        """Test that defined anchors match expected."""
        x_series, v_series = raw_measurements

        for expected in expected_results:
            if expected["anchor_u"] is not None:
                result = select_phase_anchor(x_series, v_series, expected["idx"])
                assert (
                    result.anchor_u == expected["anchor_u"]
                ), f"Anchor mismatch at idx={expected['idx']}: got {result.anchor_u}, expected {expected['anchor_u']}"


class TestTieBreaker:
    """Tests for tie-breaker behavior."""

    def test_most_recent_u_selected_on_tie(self, raw_measurements: tuple[list[float], list[float]]) -> None:
        """Test that when Δφ ties, the most recent u (largest u) is selected."""
        x_series, v_series = raw_measurements

        # Find rows where phi is exactly the same
        # From expected: rows 12, 13 both have phi = 0.0000000000
        # and both select anchor_u = 3 (because that's the best match)

        # We need to verify the tie-breaker logic specifically
        # Let's check consecutive rows with same phi
        result_12 = select_phase_anchor(x_series, v_series, 12)
        result_13 = select_phase_anchor(x_series, v_series, 13)

        # Both should find anchor with matching phase
        assert result_12.anchor_u is not None
        assert result_13.anchor_u is not None

        # The key test: when there are multiple eligible anchors with same delta_phi,
        # the most recent one (largest u) should be chosen
        # Row 13 should select anchor >= row 12's anchor (if tie-breaker applies)

        # Let's verify by checking if there were multiple candidates with same delta_phi
        # If delta_phi is 0 for both, then the most recent should be selected
        if abs(result_12.delta_phi) < 1e-9 and abs(result_13.delta_phi) < 1e-9:
            # Both have perfect matches - verify most recent selection
            assert (
                result_13.anchor_u >= result_12.anchor_u
            ), f"Tie-breaker failed: row 13 selected u={result_13.anchor_u}, row 12 selected u={result_12.anchor_u}"

    def test_tie_breaker_explicit(self) -> None:
        """Test tie-breaker with explicit data designed to create a tie."""
        # Create data with exact phase matches
        x_series = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # All same
        v_series = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # All same

        # At t=6, eligible set is {0, 1, 2, 3}
        # All have phi = 0.0 (same as t=6)
        # All have delta_phi = 0.0
        # Tie-breaker (ii): choose most recent u = 3

        result = select_phase_anchor(x_series, v_series, 6)

        assert result.anchor_u == 3, f"Tie-breaker should select most recent u=3, got {result.anchor_u}"


class TestPhaseMismatch:
    """Tests for phase mismatch censor case."""

    def test_phase_mismatch_returns_censor(
        self, raw_measurements: tuple[list[float], list[float]], expected_censors: list[dict[str, Any]]
    ) -> None:
        """Test that phase mismatch produces PHASE_MISMATCH censor."""
        x_series, v_series = raw_measurements

        # Get phase mismatch indices from expected censors
        mismatch_indices: list[int] = [c["idx"] for c in expected_censors if c["reason"] == "PHASE_MISMATCH"]

        assert len(mismatch_indices) > 0, "Test data should include phase mismatch cases"

        for idx in mismatch_indices:
            result = select_phase_anchor(x_series, v_series, idx)
            assert (
                result.undefined_reason == UndefinedReason.PHASE_MISMATCH
            ), f"Expected PHASE_MISMATCH at idx={idx}, got {result.undefined_reason}"
            assert result.anchor_u is None
            assert result.eligible_count > 0  # Has eligible anchors, but none within threshold


# =============================================================================
# END-TO-END TESTS
# =============================================================================


class TestEndToEnd:
    """End-to-end tests for full trajectory processing."""

    def test_all_results_match_expected(
        self, raw_measurements: tuple[list[float], list[float]], expected_results: list[dict[str, Any]]
    ) -> None:
        """Test that all computed results match expected outputs."""
        x_series, v_series = raw_measurements
        results, _censors = process_trajectory(x_series, v_series)

        assert len(results) == len(
            expected_results
        ), f"Result count mismatch: {len(results)} vs {len(expected_results)}"

        for i, (result, expected) in enumerate(zip(results, expected_results, strict=False)):
            assert result.idx == expected["idx"]
            assert (
                abs(result.phi - expected["phi"]) < 1e-8
            ), f"Phi mismatch at idx={i}: {result.phi} vs {expected['phi']}"
            assert result.eligible_count == expected["eligible_count"]
            assert (
                result.anchor_u == expected["anchor_u"]
            ), f"Anchor mismatch at idx={i}: {result.anchor_u} vs {expected['anchor_u']}"

            if result.delta_phi is not None and expected["delta_phi"] is not None:
                assert (
                    abs(result.delta_phi - expected["delta_phi"]) < 1e-8
                ), f"Delta_phi mismatch at idx={i}: {result.delta_phi} vs {expected['delta_phi']}"

            assert result.undefined_reason.value == expected["undefined_reason"]

    def test_censor_events_match_expected(
        self, raw_measurements: tuple[list[float], list[float]], expected_censors: list[dict[str, Any]]
    ) -> None:
        """Test that all censor events match expected."""
        x_series, v_series = raw_measurements
        _results, censors = process_trajectory(x_series, v_series)

        assert len(censors) == len(
            expected_censors
        ), f"Censor count mismatch: {len(censors)} vs {len(expected_censors)}"

        for censor, expected in zip(censors, expected_censors, strict=False):
            assert censor.idx == expected["idx"]
            assert abs(censor.phi - expected["phi"]) < 1e-8
            assert censor.reason == expected["reason"]
            assert censor.eligible_count == expected["eligible_count"]


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_two_runs_identical(self, raw_measurements: tuple[list[float], list[float]]) -> None:
        """Test that two runs produce identical output."""
        x_series, v_series = raw_measurements

        results1, censors1 = process_trajectory(x_series, v_series)
        results2, censors2 = process_trajectory(x_series, v_series)

        assert len(results1) == len(results2)
        assert len(censors1) == len(censors2)

        for r1, r2 in zip(results1, results2, strict=False):
            assert r1 == r2, f"Determinism failure: {r1} != {r2}"

    def test_frozen_config_hash_stable(self) -> None:
        """Test that frozen config hash is stable."""
        hash1 = get_frozen_config_sha256()
        hash2 = get_frozen_config_sha256()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_generator_output_matches_committed(self, casepack_dir: Path) -> None:
        """Test that regenerating expected outputs produces no diff."""
        csv_path = casepack_dir / "expected" / "ref_phase_expected.csv"
        json_path = casepack_dir / "expected" / "censor_expected.json"

        # Compute hashes of committed files
        with open(csv_path, "rb") as f:
            csv_hash_before = hashlib.sha256(f.read()).hexdigest()
        with open(json_path, "rb") as f:
            _json_hash_before = hashlib.sha256(f.read()).hexdigest()

        # Load data and regenerate
        x_series: list[float] = []
        v_series: list[float] = []
        raw_csv = casepack_dir / "raw_measurements.csv"

        with open(raw_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x_series.append(float(row["x"]))
                v_series.append(float(row["v"]))

        results, _censors = process_trajectory(x_series, v_series)

        # Write to temp file and compare
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            fieldnames = ["idx", "phi", "eligible_count", "anchor_u", "delta_phi", "undefined_reason"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                writer.writerow(
                    {
                        "idx": r.idx,
                        "phi": f"{r.phi:.10f}",
                        "eligible_count": r.eligible_count,
                        "anchor_u": r.anchor_u if r.anchor_u is not None else "",
                        "delta_phi": f"{r.delta_phi:.10f}" if r.delta_phi is not None else "",
                        "undefined_reason": r.undefined_reason.value,
                    }
                )
            temp_csv_path = f.name

        with open(temp_csv_path, "rb") as f:
            csv_hash_after = hashlib.sha256(f.read()).hexdigest()

        Path(temp_csv_path).unlink()

        assert (
            csv_hash_before == csv_hash_after
        ), "CSV regeneration produced different output - expected outputs may be stale"


# =============================================================================
# FROZEN PARAMETERS TESTS
# =============================================================================


class TestFrozenParameters:
    """Tests for frozen parameter values."""

    def test_delta_phi_max_value(self) -> None:
        """Test that δφ_max is π/6 (30°)."""
        expected = np.pi / 6
        assert abs(DELTA_PHI_MAX - expected) < 1e-10

    def test_window_value(self) -> None:
        """Test that window is 20."""
        assert WINDOW == 20

    def test_debounce_value(self) -> None:
        """Test that debounce is 3."""
        assert DEBOUNCE == 3
