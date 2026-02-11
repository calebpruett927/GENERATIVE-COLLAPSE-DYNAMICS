"""Tests for umcp.insights — InsightEntry, PatternDatabase, InsightEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from umcp.insights import (
    InsightEngine,
    InsightEntry,
    InsightSeverity,
    PatternDatabase,
    PatternType,
    _daily_seed,
    _hash_short,
    _pearson,
)

# ── InsightEntry ──────────────────────────────────────────────────


class TestInsightEntry:
    """Unit tests for InsightEntry dataclass."""

    @pytest.fixture()
    def sample_entry(self) -> InsightEntry:
        return InsightEntry(
            id="TEST-001",
            domain="Testing",
            pattern="Sample pattern",
            lesson="Sample lesson",
            implication="Sample implication",
            severity=InsightSeverity.EMPIRICAL,
            pattern_type=PatternType.PHYSICAL_INSIGHT,
            source="canon",
            elements=["Fe", "Cu"],
            omega_range=(0.0, 0.5),
        )

    def test_to_dict(self, sample_entry: InsightEntry) -> None:
        d = sample_entry.to_dict()
        assert d["id"] == "TEST-001"
        assert d["domain"] == "Testing"
        assert d["pattern"] == "Sample pattern"
        assert d["lesson"] == "Sample lesson"
        assert d["implication"] == "Sample implication"
        assert d["severity"] == "Empirical"
        assert d["pattern_type"] == "PhysicalInsight"
        assert d["source"] == "canon"
        assert d["elements"] == ["Fe", "Cu"]
        assert d["omega_range"] == [0.0, 0.5]

    def test_from_dict(self) -> None:
        d: dict[str, Any] = {
            "id": "FROM-DICT-001",
            "domain": "Cross-Domain",
            "pattern": "p",
            "lesson": "l",
            "implication": "i",
            "severity": "Fundamental",
            "pattern_type": "RegimeBoundary",
            "source": "discovered",
            "elements": ["Al"],
            "omega_range": [0.1, 0.9],
        }
        entry = InsightEntry.from_dict(d)
        assert entry.id == "FROM-DICT-001"
        assert entry.severity == InsightSeverity.FUNDAMENTAL
        assert entry.pattern_type == PatternType.REGIME_BOUNDARY
        assert entry.elements == ["Al"]
        assert entry.omega_range == (0.1, 0.9)

    def test_from_dict_defaults(self) -> None:
        d: dict[str, Any] = {"id": "MINIMAL"}
        entry = InsightEntry.from_dict(d)
        assert entry.domain == "General"
        assert entry.severity == InsightSeverity.EMPIRICAL
        assert entry.source == "canon"
        assert entry.omega_range == (0.0, 1.0)

    def test_roundtrip(self, sample_entry: InsightEntry) -> None:
        d = sample_entry.to_dict()
        restored = InsightEntry.from_dict(d)
        assert restored.id == sample_entry.id
        assert restored.domain == sample_entry.domain
        assert restored.severity == sample_entry.severity


# ── PatternDatabase ───────────────────────────────────────────────


class TestPatternDatabase:
    """Unit tests for PatternDatabase."""

    @pytest.fixture()
    def db(self) -> PatternDatabase:
        return PatternDatabase()

    def _make_entry(self, id: str, domain: str = "Test", **kwargs: Any) -> InsightEntry:
        return InsightEntry(
            id=id,
            domain=domain,
            pattern=kwargs.get("pattern", "p"),
            lesson=kwargs.get("lesson", "l"),
            implication=kwargs.get("implication", "i"),
            severity=kwargs.get("severity", InsightSeverity.EMPIRICAL),
            pattern_type=kwargs.get("pattern_type", PatternType.PHYSICAL_INSIGHT),
            source=kwargs.get("source", "canon"),
        )

    def test_add_unique(self, db: PatternDatabase) -> None:
        assert db.add(self._make_entry("A"))
        assert db.count() == 1

    def test_add_duplicate(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A"))
        assert not db.add(self._make_entry("A"))
        assert db.count() == 1

    def test_query_by_domain(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A", domain="Cohesive"))
        db.add(self._make_entry("B", domain="Magnetic"))
        db.add(self._make_entry("C", domain="Cohesive"))
        assert len(db.query(domain="Cohesive")) == 2
        assert len(db.query(domain="Magnetic")) == 1
        assert len(db.query(domain="Nonexistent")) == 0

    def test_query_by_severity(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A", severity=InsightSeverity.FUNDAMENTAL))
        db.add(self._make_entry("B", severity=InsightSeverity.EMPIRICAL))
        assert len(db.query(severity=InsightSeverity.FUNDAMENTAL)) == 1

    def test_query_by_pattern_type(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A", pattern_type=PatternType.REGIME_BOUNDARY))
        db.add(self._make_entry("B", pattern_type=PatternType.PERIODIC_TREND))
        assert len(db.query(pattern_type=PatternType.REGIME_BOUNDARY)) == 1

    def test_query_by_source(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A", source="canon"))
        db.add(self._make_entry("B", source="discovered"))
        assert len(db.query(source="discovered")) == 1

    def test_domains(self, db: PatternDatabase) -> None:
        db.add(self._make_entry("A", domain="Z"))
        db.add(self._make_entry("B", domain="A"))
        db.add(self._make_entry("C", domain="M"))
        assert db.domains() == ["A", "M", "Z"]

    def test_load_yaml_no_yaml(self, db: PatternDatabase) -> None:
        """Loading from a non-existent path returns 0."""
        count = db.load_yaml(Path("/tmp/nonexistent_file.yaml"))
        # Returns 0 if file doesn't exist or yaml not available
        assert count == 0

    def test_load_canon_no_file(self, db: PatternDatabase) -> None:
        count = db.load_canon(Path("/tmp/nonexistent_canon.yaml"))
        assert count == 0

    def test_save_yaml(self, db: PatternDatabase, tmp_path: Path) -> None:
        db.add(self._make_entry("SAVE-1", domain="SaveTest"))
        out = tmp_path / "insights.yaml"
        db.save_yaml(out)
        # File should exist and be loadable
        assert out.exists()
        text = out.read_text()
        assert "SAVE-1" in text

    def test_load_yaml_roundtrip(self, tmp_path: Path) -> None:
        db1 = PatternDatabase()
        db1.add(
            InsightEntry(
                id="RT-1",
                domain="RT",
                pattern="roundtrip",
                lesson="test",
                implication="test",
            )
        )
        out = tmp_path / "rt.yaml"
        db1.save_yaml(out)

        db2 = PatternDatabase()
        loaded = db2.load_yaml(out)
        assert loaded == 1
        assert db2.count() == 1
        assert db2.entries[0].id == "RT-1"

    def test_load_canon_real(self, db: PatternDatabase) -> None:
        """Load from actual canon/matl_anchors.yaml if it exists."""
        count = db.load_canon()
        # May or may not have lessons_learned; at least it shouldn't crash
        assert count >= 0


# ── InsightEngine ─────────────────────────────────────────────────


class TestInsightEngine:
    """Tests for InsightEngine orchestrator."""

    def test_init_default(self) -> None:
        engine = InsightEngine()
        # Should initialize without errors
        assert engine.db is not None

    def test_init_no_canon(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        assert engine.db.count() == 0

    def test_discover_periodic_trends(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_periodic_trends()
        # May find trends (closures available in repo) or empty list
        assert isinstance(results, list)

    def test_discover_regime_boundaries(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_regime_boundaries()
        assert isinstance(results, list)

    def test_discover_cross_correlations(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_cross_correlations()
        assert isinstance(results, list)

    def test_discover_universality_signatures(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_universality_signatures()
        assert isinstance(results, list)

    def test_discover_all(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_all()
        assert isinstance(results, list)

    def test_show_startup_insight_empty(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        assert engine.show_startup_insight() == ""

    def test_show_startup_insight_with_entries(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(
            InsightEntry(
                id="SHOW-1",
                domain="Test",
                pattern="Test pattern with sufficient text",
                lesson="This is a test lesson for the startup insight display.",
                implication="Works correctly",
            )
        )
        text = engine.show_startup_insight(seed=42)
        assert "Test pattern" in text
        assert "SHOW-1" in text

    def test_show_startup_insight_deterministic(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        for i in range(3):
            engine.db.add(
                InsightEntry(id=f"DET-{i}", domain="Test", pattern=f"p{i}", lesson=f"l{i}", implication=f"i{i}")
            )
        result1 = engine.show_startup_insight(seed=99)
        result2 = engine.show_startup_insight(seed=99)
        assert result1 == result2

    def test_full_report_empty(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        assert engine.full_report() == "No insights in database."

    def test_full_report_with_entries(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(
            InsightEntry(
                id="REPORT-1",
                domain="Cohesive Energy",
                pattern="Pattern A",
                lesson="Lesson A",
                implication="Impl A",
                elements=["Fe", "Cu"],
            )
        )
        engine.db.add(
            InsightEntry(
                id="REPORT-2",
                domain="Magnetic",
                pattern="Pattern B",
                lesson="Lesson B",
                implication="Impl B",
                severity=InsightSeverity.FUNDAMENTAL,
            )
        )
        report = engine.full_report()
        assert "LESSONS-LEARNED REPORT" in report
        assert "2 insights" in report
        assert "Cohesive Energy" in report
        assert "Magnetic" in report
        assert "REPORT-1" in report
        assert "REPORT-2" in report
        assert "Fe, Cu" in report
        assert "What returns through collapse is real" in report

    def test_summary_stats(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(
            InsightEntry(
                id="STAT-1",
                domain="D1",
                pattern="p",
                lesson="l",
                implication="i",
                severity=InsightSeverity.FUNDAMENTAL,
                source="canon",
            )
        )
        engine.db.add(
            InsightEntry(
                id="STAT-2",
                domain="D2",
                pattern="p",
                lesson="l",
                implication="i",
                severity=InsightSeverity.EMPIRICAL,
                source="discovered",
            )
        )
        stats = engine.summary_stats()
        assert stats["total_insights"] == 2
        assert stats["domains"]["D1"] == 1
        assert stats["domains"]["D2"] == 1
        assert stats["by_severity"]["Fundamental"] == 1
        assert stats["by_severity"]["Empirical"] == 1
        assert stats["by_source"]["canon"] == 1
        assert stats["by_source"]["discovered"] == 1
        assert sorted(stats["domain_list"]) == ["D1", "D2"]

    def test_save(self, tmp_path: Path) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(InsightEntry(id="SAVE-E", domain="D", pattern="p", lesson="l", implication="i"))
        out = tmp_path / "save_test.yaml"
        engine.save(out)
        assert out.exists()

    def test_philosophy(self) -> None:
        text = InsightEngine.philosophy()
        assert "collapse" in text.lower()
        assert "ω_eff" in text


# ── Utility Functions ─────────────────────────────────────────────


class TestUtilities:
    """Tests for module-level utility functions."""

    def test_hash_short_deterministic(self) -> None:
        h1 = _hash_short("test input")
        h2 = _hash_short("test input")
        assert h1 == h2
        assert len(h1) == 8

    def test_hash_short_different_inputs(self) -> None:
        assert _hash_short("a") != _hash_short("b")

    def test_daily_seed_type(self) -> None:
        seed = _daily_seed()
        assert isinstance(seed, int)
        assert seed > 20000000  # YYYYMMDD format

    def test_pearson_perfect_positive(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(x, y) - 1.0) < 1e-10

    def test_pearson_perfect_negative(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson(x, y) - (-1.0)) < 1e-10

    def test_pearson_zero(self) -> None:
        # Uncorrelated: constant y
        x = [1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0]
        assert _pearson(x, y) == 0.0

    def test_pearson_short_list(self) -> None:
        assert _pearson([1.0], [2.0]) == 0.0
        assert _pearson([], []) == 0.0


# ── Enum coverage ────────────────────────────────────────────────


def test_severity_values() -> None:
    assert InsightSeverity.FUNDAMENTAL.value == "Fundamental"
    assert InsightSeverity.STRUCTURAL.value == "Structural"
    assert InsightSeverity.CURIOUS.value == "Curious"


def test_pattern_type_values() -> None:
    assert PatternType.PERIODIC_TREND.value == "PeriodicTrend"
    assert PatternType.CROSS_CORRELATION.value == "CrossCorrelation"
    assert PatternType.UNIVERSALITY.value == "Universality"
    assert PatternType.MODEL_LIMITATION.value == "ModelLimitation"
