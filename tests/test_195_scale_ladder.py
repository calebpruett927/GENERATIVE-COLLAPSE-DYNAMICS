"""Tests for closures/scale_ladder.py — Scale Ladder from Minimal to Universal.

Covers:
  - All 11 rungs build without error
  - Tier-1 identities (Three Pillars) hold at every rung
  - Object counts match expectations
  - Bridge chain is contiguous (no gaps)
  - Fidelity compression: F ∈ [0.01, 0.90] across 61 OOM
  - Heterogeneity gap is non-trivial at every rung
  - Regime classification is consistent
  - Display functions run without error
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ── Path setup ──────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.scale_ladder import (  # noqa: E402
    Bridge,
    ScaleLadder,
    ScaleObject,
    build_scale_ladder,
    display_ladder,
    get_bridge_map,
)

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ladder() -> ScaleLadder:
    """Build the complete ladder once for all tests."""
    return build_scale_ladder()


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL TESTS
# ═══════════════════════════════════════════════════════════════════


class TestLadderStructure:
    """Verify the ladder has the right shape."""

    def test_eleven_rungs(self, ladder: ScaleLadder) -> None:
        assert len(ladder.rungs) == 11

    def test_rung_numbering(self, ladder: ScaleLadder) -> None:
        for i, rung in enumerate(ladder.rungs):
            assert rung.number == i

    def test_rung_names(self, ladder: ScaleLadder) -> None:
        expected = [
            "Planck",
            "Subatomic",
            "Nuclear",
            "Atomic",
            "Molecular",
            "Cellular",
            "Everyday",
            "Geological",
            "Stellar",
            "Galactic",
            "Cosmological",
        ]
        actual = [r.name for r in ladder.rungs]
        assert actual == expected

    def test_scale_monotonic(self, ladder: ScaleLadder) -> None:
        """Scales must increase monotonically from rung 0 to rung 10."""
        scales = [r.scale_meters for r in ladder.rungs]
        for i in range(1, len(scales)):
            assert scales[i] > scales[i - 1], f"Scale not monotonic at rung {i}: {scales[i - 1]:.1e} → {scales[i]:.1e}"

    def test_total_objects_positive(self, ladder: ScaleLadder) -> None:
        assert ladder.total_objects > 300  # should be ~406

    def test_every_rung_has_objects(self, ladder: ScaleLadder) -> None:
        for rung in ladder.rungs:
            assert rung.n_objects > 0, f"Rung {rung.name} has no objects"

    def test_every_rung_has_channels(self, ladder: ScaleLadder) -> None:
        for rung in ladder.rungs:
            assert rung.n_channels >= 6, f"Rung {rung.name} has {rung.n_channels} channels"


# ═══════════════════════════════════════════════════════════════════
# TIER-1 IDENTITY TESTS (THREE PILLARS)
# ═══════════════════════════════════════════════════════════════════


class TestThreePillars:
    """Verify the three pillars hold at every object on every rung."""

    def test_zero_violations(self, ladder: ScaleLadder) -> None:
        assert ladder.total_tier1_violations == 0

    def test_pillar_1_duality_per_object(self, ladder: ScaleLadder) -> None:
        """F + ω = 1 at every object."""
        for rung in ladder.rungs:
            for obj in rung.objects:
                assert abs(obj.F + obj.omega - 1.0) < 1e-5, (
                    f"Pillar 1 fails at {rung.name}/{obj.name}: F={obj.F}, ω={obj.omega}, sum={obj.F + obj.omega}"
                )

    def test_pillar_2_integrity_bound_per_object(self, ladder: ScaleLadder) -> None:
        """IC ≤ F at every object."""
        for rung in ladder.rungs:
            for obj in rung.objects:
                assert obj.IC <= obj.F + 1e-10, f"Pillar 2 fails at {rung.name}/{obj.name}: IC={obj.IC} > F={obj.F}"

    def test_pillar_3_exponential_bridge_per_object(self, ladder: ScaleLadder) -> None:
        """IC ≈ exp(κ) at every object."""
        for rung in ladder.rungs:
            for obj in rung.objects:
                if obj.kappa > -500:
                    expected_IC = math.exp(obj.kappa)
                    assert abs(obj.IC - expected_IC) < 1e-4, (
                        f"Pillar 3 fails at {rung.name}/{obj.name}: IC={obj.IC}, exp(κ)={expected_IC}"
                    )

    def test_gap_nonnegative(self, ladder: ScaleLadder) -> None:
        """Δ = F - IC ≥ 0 (consequence of Pillar 2)."""
        for rung in ladder.rungs:
            for obj in rung.objects:
                assert obj.gap >= -1e-10, f"Negative gap at {rung.name}/{obj.name}: Δ={obj.gap}"


# ═══════════════════════════════════════════════════════════════════
# BRIDGE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestBridges:
    """Verify the bridge chain is contiguous."""

    def test_bridge_chain_complete(self, ladder: ScaleLadder) -> None:
        """Every rung except the last should have a bridge_up."""
        for rung in ladder.rungs[:-1]:
            assert len(rung.bridges_up) > 0, f"Rung {rung.name} missing bridge_up"

    def test_bridge_map_length(self, ladder: ScaleLadder) -> None:
        bridges = get_bridge_map(ladder)
        assert len(bridges) >= 10  # at least one per gap

    def test_bridge_from_to_consistency(self, ladder: ScaleLadder) -> None:
        """Bridge from_rung/to_rung should match adjacent rungs."""
        for i, rung in enumerate(ladder.rungs):
            for bridge in rung.bridges_up:
                assert bridge.from_rung == rung.name
                if i + 1 < len(ladder.rungs):
                    assert bridge.to_rung == ladder.rungs[i + 1].name

    def test_bridges_have_content(self, ladder: ScaleLadder) -> None:
        """Every bridge should have non-empty mechanism and example."""
        for rung in ladder.rungs:
            for bridge in rung.bridges_up + rung.bridges_down:
                assert len(bridge.mechanism) > 10, f"Short mechanism on {rung.name}"
                assert len(bridge.example) > 10, f"Short example on {rung.name}"


# ═══════════════════════════════════════════════════════════════════
# CROSS-SCALE INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCrossScaleInvariants:
    """Verify properties that should hold across rung boundaries."""

    def test_fidelity_compression(self, ladder: ScaleLadder) -> None:
        """Mean F should be bounded ∈ [0.01, 0.90] at every rung."""
        for rung in ladder.rungs:
            mean_F = rung.mean_F
            assert 0.01 <= mean_F <= 0.90, f"Fidelity compression violated at {rung.name}: ⟨F⟩={mean_F:.4f}"

    def test_heterogeneity_gap_nontrivial(self, ladder: ScaleLadder) -> None:
        """Every rung should have a nonzero mean heterogeneity gap."""
        for rung in ladder.rungs:
            mean_gap = rung.mean_gap
            assert mean_gap > 0.001, f"Gap trivial at {rung.name}: ⟨Δ⟩={mean_gap:.6f}"

    def test_regime_counts_add_up(self, ladder: ScaleLadder) -> None:
        """Regime counts at each rung should sum to n_objects."""
        for rung in ladder.rungs:
            total = sum(rung.regime_counts.values())
            assert total == rung.n_objects

    def test_dynamic_range_61_oom(self, ladder: ScaleLadder) -> None:
        """Scale range should span ~61 OOM."""
        smallest = ladder.rungs[0].scale_meters
        largest = ladder.rungs[-1].scale_meters
        oom = math.log10(largest / smallest)
        assert oom > 58  # allow slight margin
        assert oom < 65


# ═══════════════════════════════════════════════════════════════════
# PER-RUNG OBJECT COUNT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestRungCounts:
    """Verify each rung has the expected number of objects."""

    def test_planck(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[0].n_objects == 1

    def test_subatomic(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[1].n_objects == 31

    def test_nuclear(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[2].n_objects == 92

    def test_atomic(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[3].n_objects == 118

    def test_molecular(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[4].n_objects == 20

    def test_cellular(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[5].n_objects == 12

    def test_everyday(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[6].n_objects == 84

    def test_geological(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[7].n_objects == 14

    def test_stellar(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[8].n_objects == 16

    def test_galactic(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[9].n_objects == 12

    def test_cosmological(self, ladder: ScaleLadder) -> None:
        assert ladder.rungs[10].n_objects == 6


# ═══════════════════════════════════════════════════════════════════
# DISPLAY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDisplay:
    """Verify display functions run without error."""

    def test_display_normal(self, ladder: ScaleLadder, capsys: pytest.CaptureFixture[str]) -> None:
        display_ladder(ladder, verbose=False)
        captured = capsys.readouterr()
        assert "SCALE LADDER" in captured.out
        assert "VERDICT" in captured.out
        assert "Solum quod redit" in captured.out

    def test_display_verbose(self, ladder: ScaleLadder, capsys: pytest.CaptureFixture[str]) -> None:
        display_ladder(ladder, verbose=True)
        captured = capsys.readouterr()
        assert "RUNG 0: Planck" in captured.out
        assert "RUNG 10: Cosmological" in captured.out


# ═══════════════════════════════════════════════════════════════════
# DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDataclasses:
    """Verify dataclass construction and properties."""

    def test_scale_object_frozen(self) -> None:
        obj = ScaleObject(
            name="test",
            F=0.5,
            omega=0.5,
            IC=0.3,
            kappa=-1.2,
            S=0.6,
            C=0.3,
            gap=0.2,
            regime="Stable",
        )
        with pytest.raises(AttributeError):
            obj.F = 0.9  # type: ignore[misc]

    def test_bridge_frozen(self) -> None:
        b = Bridge(
            from_rung="A",
            to_rung="B",
            mechanism="test",
            example="test",
            channel_mapping="test",
        )
        with pytest.raises(AttributeError):
            b.from_rung = "C"  # type: ignore[misc]

    def test_rung_by_name(self, ladder: ScaleLadder) -> None:
        r = ladder.rung_by_name("Stellar")
        assert r is not None
        assert r.name == "Stellar"

    def test_rung_by_name_case_insensitive(self, ladder: ScaleLadder) -> None:
        r = ladder.rung_by_name("stellar")
        assert r is not None

    def test_rung_by_name_missing(self, ladder: ScaleLadder) -> None:
        r = ladder.rung_by_name("nonexistent")
        assert r is None


# ═══════════════════════════════════════════════════════════════════
# PILLAR RESULTS DICTIONARY
# ═══════════════════════════════════════════════════════════════════


class TestPillarResults:
    """Verify the pillar_results dictionary is well-formed."""

    def test_pillar_1_exact(self, ladder: ScaleLadder) -> None:
        assert ladder.pillar_results["pillar_1_duality"]["status"] == "EXACT"

    def test_pillar_2_proven(self, ladder: ScaleLadder) -> None:
        assert ladder.pillar_results["pillar_2_integrity_bound"]["status"] == "PROVEN"

    def test_pillar_3_exact(self, ladder: ScaleLadder) -> None:
        assert ladder.pillar_results["pillar_3_exponential_bridge"]["status"] == "EXACT"

    def test_fidelity_range(self, ladder: ScaleLadder) -> None:
        f_min, f_max = ladder.pillar_results["fidelity_range"]
        assert f_min > 0.0
        assert f_max < 1.0
        assert f_min < f_max

    def test_dynamic_range_oom(self, ladder: ScaleLadder) -> None:
        assert ladder.pillar_results["dynamic_range_OOM"] == 61
