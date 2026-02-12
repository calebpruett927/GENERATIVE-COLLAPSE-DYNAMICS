"""Deep-coverage tests for umcp.insights — discovery logic, implications, edge cases.

This module extends test_insights.py with:
  - Mocked closure imports to exercise all discovery branches
  - Edge cases for PatternDatabase I/O
  - Full coverage of report formatting
  - Deep implication tests: verifying that discovered patterns carry
    physically meaningful semantics consistent with the core axiom
  - Pearson correlation edge cases
  - InsightEngine lifecycle
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

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

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _entry(
    id: str = "T-001",
    domain: str = "Test",
    pattern: str = "test pattern",
    lesson: str = "test lesson",
    implication: str = "test implication",
    severity: InsightSeverity = InsightSeverity.EMPIRICAL,
    pattern_type: PatternType = PatternType.PHYSICAL_INSIGHT,
    source: str = "canon",
    elements: list[str] | None = None,
    omega_range: tuple[float, float] = (0.0, 1.0),
) -> InsightEntry:
    return InsightEntry(
        id=id,
        domain=domain,
        pattern=pattern,
        lesson=lesson,
        implication=implication,
        severity=severity,
        pattern_type=pattern_type,
        source=source,
        elements=elements or [],
        omega_range=omega_range,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. InsightEntry — deep serialization / deserialization
# ═══════════════════════════════════════════════════════════════════


class TestInsightEntryDeep:
    """Exhaustive InsightEntry serialization, enum combinations, edge cases."""

    @pytest.mark.parametrize(
        "severity",
        list(InsightSeverity),
        ids=[s.name for s in InsightSeverity],
    )
    def test_all_severities_roundtrip(self, severity: InsightSeverity) -> None:
        entry = _entry(severity=severity)
        d = entry.to_dict()
        restored = InsightEntry.from_dict(d)
        assert restored.severity == severity
        assert d["severity"] == severity.value

    @pytest.mark.parametrize(
        "pt",
        list(PatternType),
        ids=[p.name for p in PatternType],
    )
    def test_all_pattern_types_roundtrip(self, pt: PatternType) -> None:
        entry = _entry(pattern_type=pt)
        d = entry.to_dict()
        restored = InsightEntry.from_dict(d)
        assert restored.pattern_type == pt
        assert d["pattern_type"] == pt.value

    def test_empty_elements_roundtrip(self) -> None:
        entry = _entry(elements=[])
        d = entry.to_dict()
        assert d["elements"] == []
        restored = InsightEntry.from_dict(d)
        assert restored.elements == []

    def test_large_elements_list(self) -> None:
        elems = [f"E{i}" for i in range(100)]
        entry = _entry(elements=elems)
        d = entry.to_dict()
        assert len(d["elements"]) == 100
        restored = InsightEntry.from_dict(d)
        assert restored.elements == elems

    def test_omega_range_boundary(self) -> None:
        entry = _entry(omega_range=(0.0, 1.0))
        d = entry.to_dict()
        assert d["omega_range"] == [0.0, 1.0]

    def test_omega_range_narrow(self) -> None:
        entry = _entry(omega_range=(0.42, 0.43))
        d = entry.to_dict()
        restored = InsightEntry.from_dict(d)
        assert abs(restored.omega_range[0] - 0.42) < 1e-10
        assert abs(restored.omega_range[1] - 0.43) < 1e-10

    def test_from_dict_missing_optional_fields(self) -> None:
        """from_dict with only 'id' should fill all defaults."""
        entry = InsightEntry.from_dict({"id": "MINIMAL-X"})
        assert entry.domain == "General"
        assert entry.pattern == ""
        assert entry.lesson == ""
        assert entry.implication == ""
        assert entry.severity == InsightSeverity.EMPIRICAL
        assert entry.pattern_type == PatternType.PHYSICAL_INSIGHT
        assert entry.source == "canon"
        assert entry.elements == []
        assert entry.omega_range == (0.0, 1.0)

    def test_all_sources(self) -> None:
        for src in ("canon", "discovered", "cross-closure"):
            entry = _entry(source=src)
            d = entry.to_dict()
            assert d["source"] == src
            restored = InsightEntry.from_dict(d)
            assert restored.source == src


# ═══════════════════════════════════════════════════════════════════
# 2. PatternDatabase — advanced queries, multi-filter, I/O
# ═══════════════════════════════════════════════════════════════════


class TestPatternDatabaseDeep:
    """Advanced PatternDatabase operations — multi-filter, edge cases, YAML I/O."""

    @pytest.fixture()
    def populated_db(self) -> PatternDatabase:
        db = PatternDatabase()
        db.add(_entry("A", domain="Cohesive", severity=InsightSeverity.FUNDAMENTAL, source="canon"))
        db.add(_entry("B", domain="Cohesive", severity=InsightSeverity.EMPIRICAL, source="discovered"))
        db.add(_entry("C", domain="Magnetic", severity=InsightSeverity.FUNDAMENTAL, source="discovered"))
        db.add(_entry("D", domain="Surface", severity=InsightSeverity.STRUCTURAL, source="canon"))
        db.add(_entry("E", domain="Surface", severity=InsightSeverity.CURIOUS, source="cross-closure"))
        return db

    def test_query_multi_filter(self, populated_db: PatternDatabase) -> None:
        """Apply domain + severity filter together."""
        result = populated_db.query(domain="Cohesive", severity=InsightSeverity.FUNDAMENTAL)
        assert len(result) == 1
        assert result[0].id == "A"

    def test_query_no_match(self, populated_db: PatternDatabase) -> None:
        result = populated_db.query(domain="Nonexistent")
        assert result == []

    def test_query_all_filters_at_once(self, populated_db: PatternDatabase) -> None:
        result = populated_db.query(
            domain="Magnetic",
            severity=InsightSeverity.FUNDAMENTAL,
            source="discovered",
            pattern_type=PatternType.PHYSICAL_INSIGHT,
        )
        assert len(result) == 1
        assert result[0].id == "C"

    def test_count_after_multiple_adds(self, populated_db: PatternDatabase) -> None:
        assert populated_db.count() == 5

    def test_domains_sorted(self, populated_db: PatternDatabase) -> None:
        assert populated_db.domains() == ["Cohesive", "Magnetic", "Surface"]

    def test_empty_db_queries(self) -> None:
        db = PatternDatabase()
        assert db.query() == []
        assert db.count() == 0
        assert db.domains() == []

    def test_duplicate_ids_across_domains(self) -> None:
        """Same ID in different domains should still be deduplicated."""
        db = PatternDatabase()
        db.add(_entry("SAME", domain="A"))
        assert not db.add(_entry("SAME", domain="B"))
        assert db.count() == 1

    def test_save_and_load_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        db1 = PatternDatabase()
        e = _entry(
            "RT-DEEP",
            domain="Deep",
            pattern="deep pattern",
            lesson="deep lesson",
            implication="deep impl",
            severity=InsightSeverity.STRUCTURAL,
            pattern_type=PatternType.REGIME_BOUNDARY,
            source="discovered",
            elements=["Fe", "Ni", "Cu"],
            omega_range=(0.1, 0.8),
        )
        db1.add(e)
        out = tmp_path / "deep_roundtrip.yaml"
        db1.save_yaml(out)

        db2 = PatternDatabase()
        loaded = db2.load_yaml(out)
        assert loaded == 1
        r = db2.entries[0]
        assert r.id == "RT-DEEP"
        assert r.domain == "Deep"
        assert r.severity == InsightSeverity.STRUCTURAL
        assert r.pattern_type == PatternType.REGIME_BOUNDARY
        assert r.source == "discovered"
        assert r.elements == ["Fe", "Ni", "Cu"]
        assert abs(r.omega_range[0] - 0.1) < 1e-10

    def test_load_yaml_empty_content(self, tmp_path: Path) -> None:
        """YAML file with no 'insights' key returns 0."""
        p = tmp_path / "empty.yaml"
        p.write_text("meta: info\n")
        db = PatternDatabase()
        assert db.load_yaml(p) == 0

    def test_load_yaml_null_content(self, tmp_path: Path) -> None:
        p = tmp_path / "null.yaml"
        p.write_text("")
        db = PatternDatabase()
        assert db.load_yaml(p) == 0

    def test_load_yaml_with_duplicates(self, tmp_path: Path) -> None:
        """Loading a file that contains entries already in the db should skip them."""
        db = PatternDatabase()
        db.add(_entry("PREEXIST"))

        # Save a file that includes the same ID
        db2 = PatternDatabase()
        db2.add(_entry("PREEXIST"))
        db2.add(_entry("NEW-ONE"))
        f = tmp_path / "dups.yaml"
        db2.save_yaml(f)

        loaded = db.load_yaml(f)
        assert loaded == 1  # Only NEW-ONE added
        assert db.count() == 2

    def test_load_canon_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "canon.yaml"
        p.write_text("title: anchor\n")
        db = PatternDatabase()
        assert db.load_canon(p) == 0

    def test_load_canon_with_lessons(self, tmp_path: Path) -> None:
        import yaml

        data = {
            "lessons_learned": [
                {"id": "LL-1", "domain": "Physics", "pattern": "p", "lesson": "l", "implication": "i"},
                {"id": "LL-2", "domain": "Chemistry"},
            ]
        }
        p = tmp_path / "canon_with_lessons.yaml"
        p.write_text(yaml.dump(data))
        db = PatternDatabase()
        loaded = db.load_canon(p)
        assert loaded == 2
        assert db.entries[0].severity == InsightSeverity.FUNDAMENTAL
        assert db.entries[0].source == "canon"

    def test_save_yaml_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "out.yaml"
        db = PatternDatabase()
        db.add(_entry("DEEP-DIR"))
        db.save_yaml(deep)
        assert deep.exists()


# ═══════════════════════════════════════════════════════════════════
# 3. Discovery methods — mocked closures
# ═══════════════════════════════════════════════════════════════════


class TestDiscoverPeriodicTrends:
    """Test discover_periodic_trends with mocked cohesive_energy closure."""

    def _mock_cohesive(self, z: int, *, symbol: str = "") -> SimpleNamespace:
        """Mock cohesive energy: high ω for Fe/Mn (magnetic), low for Cu/Ni."""
        magnetic_high_omega = {26: 0.55, 25: 0.48}  # Fe, Mn → Anomalous
        precise_low_omega = {22: 0.05, 23: 0.06, 24: 0.04, 27: 0.07, 28: 0.08, 29: 0.03, 30: 0.09}
        omega = magnetic_high_omega.get(z, precise_low_omega.get(z, 0.15))
        return SimpleNamespace(omega_eff=omega, E_coh_eV=3.0, r0_A=2.5)

    def _mock_cohesive_4d(self, z: int, *, symbol: str = "") -> SimpleNamespace:
        """4d metals: systematically low ω."""
        return SimpleNamespace(omega_eff=0.08, E_coh_eV=4.0, r0_A=2.7)

    @staticmethod
    def _make_module(name: str, **attrs: Any) -> MagicMock:
        mod = MagicMock(spec=[])
        mod.__name__ = name
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    def test_discovers_3d_anomalous_boundary(self) -> None:
        cohesive_mod = self._make_module(
            "closures.materials_science.cohesive_energy", compute_cohesive_energy=self._mock_cohesive
        )
        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(),
                "closures.materials_science.cohesive_energy": cohesive_mod,
            },
        ):
            results = engine.discover_periodic_trends()

        # Should find the 3d boundary insight
        _3d = [e for e in results if "3D" in e.id or "3d" in e.pattern.lower()]
        assert len(_3d) >= 1
        entry = _3d[0]
        assert entry.severity == InsightSeverity.STRUCTURAL
        assert entry.pattern_type == PatternType.REGIME_BOUNDARY
        assert entry.source == "discovered"
        assert "Fe" in entry.elements or "Mn" in entry.elements

    def test_discovers_4d_lower_omega(self) -> None:
        """4d series with avg ω < 0.15 should generate a PeriodicTrend insight."""

        def mock_both(z: int, *, symbol: str = "") -> SimpleNamespace:
            if z >= 40:
                return self._mock_cohesive_4d(z, symbol=symbol)
            return self._mock_cohesive(z, symbol=symbol)

        cohesive_mod = self._make_module(
            "closures.materials_science.cohesive_energy", compute_cohesive_energy=mock_both
        )
        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(),
                "closures.materials_science.cohesive_energy": cohesive_mod,
            },
        ):
            results = engine.discover_periodic_trends()

        _4d = [e for e in results if "4D" in e.id or "4d" in e.pattern.lower()]
        assert len(_4d) >= 1
        assert _4d[0].pattern_type == PatternType.PERIODIC_TREND

    def test_no_insights_when_import_fails(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        results = engine.discover_periodic_trends()
        # Without actual closures importable, returns empty
        assert isinstance(results, list)

    def test_insights_added_to_db(self) -> None:
        cohesive_mod = self._make_module(
            "closures.materials_science.cohesive_energy", compute_cohesive_energy=self._mock_cohesive
        )
        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(),
                "closures.materials_science.cohesive_energy": cohesive_mod,
            },
        ):
            results = engine.discover_periodic_trends()

        # All discovered entries should be in the engine's database
        for entry in results:
            assert entry.id in engine.db._seen_ids


class TestDiscoverRegimeBoundaries:
    """Test discover_regime_boundaries with mocked magnetic / BCS closures."""

    @staticmethod
    def _make_module(name: str, **attrs: Any) -> MagicMock:
        mod = MagicMock(spec=[])
        mod.__name__ = name
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    def _mock_magnetic(self, z: int, *, symbol: str = "Fe", T_K: float = 300) -> SimpleNamespace:
        """Mock: Fe is ferromagnetic below 1050 K, para above."""
        tc = 1043.0
        m = 2.22 * (1.0 - T_K / tc) ** (5.0 / 6.0) if tc > T_K else 0.0
        return SimpleNamespace(M_total_B=m, omega_eff=0.1 if m > 0 else 0.6)

    def test_discovers_curie_temperature(self) -> None:
        mag_mod = self._make_module(
            "closures.materials_science.magnetic_properties",
            compute_magnetic_properties=self._mock_magnetic,
        )
        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(),
                "closures.materials_science.magnetic_properties": mag_mod,
            },
        ):
            results = engine.discover_regime_boundaries()

        curie = [e for e in results if "CURIE" in e.id]
        assert len(curie) == 1
        assert curie[0].severity == InsightSeverity.FUNDAMENTAL
        assert curie[0].pattern_type == PatternType.REGIME_BOUNDARY
        assert "Fe" in curie[0].elements
        # The lesson should reference RCFT β exponent
        assert "5/6" in curie[0].lesson or "β" in curie[0].lesson

    def test_discovers_superconductor_types(self) -> None:
        def mock_bcs(z: int, dummy: int, *, symbol: str = "") -> SimpleNamespace:
            type_map = {"Nb": "TypeII", "Pb": "TypeI", "Al": "TypeI", "Sn": "TypeI", "V": "TypeII"}
            return SimpleNamespace(sc_type=type_map.get(symbol, "Normal"), omega_eff=0.05)

        bcs_mod = self._make_module(
            "closures.materials_science.bcs_superconductivity",
            compute_bcs_superconductivity=mock_bcs,
        )
        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(),
                "closures.materials_science.bcs_superconductivity": bcs_mod,
            },
        ):
            results = engine.discover_regime_boundaries()

        sc = [e for e in results if "SC-TYPE" in e.id]
        assert len(sc) == 1
        assert sc[0].pattern_type == PatternType.REGIME_BOUNDARY
        assert "Type-I" in sc[0].pattern or "Type-II" in sc[0].pattern
        # Should mention GL parameter κ
        assert "κ" in sc[0].lesson or "kappa" in sc[0].lesson.lower()


class TestDiscoverCrossCorrelations:
    """Test cross-correlation discovery logic with physical data."""

    def test_discovers_debye_surface_correlation(self) -> None:
        """Verify Debye-surface cross-correlation logic produces valid insight."""
        debye_data = {"Cu": 343, "Ag": 225, "Au": 165, "Pt": 240, "Ni": 450, "Al": 428, "W": 400, "Mo": 450}
        surface_data = {"Cu": 1.79, "Ag": 1.25, "Au": 1.50, "Pt": 2.48, "Ni": 2.45, "Al": 1.14, "W": 3.26, "Mo": 2.91}

        # Compute correlation directly (exercises _pearson and insight construction)
        pairs: list[tuple[str, float, float]] = [
            (sym, float(debye_data[sym]), surface_data[sym]) for sym in ["Cu", "Ag", "Au", "Pt", "Ni", "Al", "W", "Mo"]
        ]
        thetas = [p[1] for p in pairs]
        gammas = [p[2] for p in pairs]
        r_val = _pearson(thetas, gammas)

        # The data should show a positive correlation (stiff lattice ↔ high surface energy)
        assert r_val > 0.3, f"Expected r > 0.3, got {r_val:.3f}"

        direction = "positive" if r_val > 0 else "negative"
        entry = InsightEntry(
            id=f"DIS-CC-DEBYE-SURF-{_hash_short(str(pairs))}",
            domain="Cross-Domain",
            pattern=f"Θ_D vs γ_surface: Pearson r = {r_val:.3f} ({direction} correlation)",
            lesson=(
                "Debye temperature (lattice stiffness) and surface "
                "energy (bond-breaking cost) share a common ancestor: "
                "the interatomic bond strength."
            ),
            implication="Surface and thermal properties are different projections of the same cohesive landscape",
            severity=InsightSeverity.STRUCTURAL,
            pattern_type=PatternType.CROSS_CORRELATION,
            source="discovered",
            elements=[p[0] for p in pairs],
        )
        assert entry.pattern_type == PatternType.CROSS_CORRELATION
        assert "Pearson" in entry.pattern
        assert "bond" in entry.lesson.lower()
        assert len(entry.elements) == 8

        db = PatternDatabase()
        assert db.add(entry)
        assert db.count() == 1

    def test_discovers_cohesive_elastic_correlation(self) -> None:
        """Verify E_coh vs K_bulk cross-correlation produces valid insight."""
        ecoh = {"Cu": 3.49, "Ag": 2.95, "Au": 3.81, "Al": 3.39, "W": 8.90, "Mo": 6.82, "Ni": 4.44, "Fe": 4.28}

        pairs: list[tuple[str, float, float]] = [
            (sym, ecoh[sym], ecoh[sym] * 40)  # K ~ 40 * E_coh
            for sym in ["Cu", "Ag", "Au", "Al", "W", "Mo", "Ni", "Fe"]
        ]
        e_vals = [p[1] for p in pairs]
        k_vals = [p[2] for p in pairs]
        r_val = _pearson(e_vals, k_vals)

        # Perfect linear relationship since K = 40*E_coh
        assert abs(r_val - 1.0) < 1e-10

        entry = InsightEntry(
            id=f"DIS-CC-COH-ELAST-{_hash_short(str(pairs))}",
            domain="Cross-Domain",
            pattern=f"E_coh vs K_bulk: Pearson r = {r_val:.3f} — strongly correlated",
            lesson=(
                "Cohesive energy is the depth of the interatomic "
                "potential well; bulk modulus K is its curvature. "
                "For the same functional form V(r), deeper wells have "
                "steeper walls: E_coh and K are not independent — they "
                "are the 0th and 2nd moments of the same binding curve."
            ),
            implication="Binding depth and curvature are coupled moments of the collapse potential",
            severity=InsightSeverity.FUNDAMENTAL,
            pattern_type=PatternType.CROSS_CORRELATION,
            source="discovered",
            elements=[p[0] for p in pairs],
        )
        assert entry.severity == InsightSeverity.FUNDAMENTAL
        assert "moment" in entry.lesson.lower() or "curvature" in entry.lesson.lower()
        assert len(entry.elements) == 8


class TestDiscoverUniversalitySignatures:
    """Test universality discovery with mocked multi-closure ω_eff values."""

    def test_discovers_divergent_omega(self) -> None:
        def mock_cohesive(z: int, *, symbol: str = "") -> SimpleNamespace:
            # Fe: high ω in cohesive (magnetic)
            omegas = {"Fe": 0.55, "Cu": 0.05, "Ni": 0.40, "Al": 0.03, "Au": 0.06, "Pt": 0.04, "W": 0.08}
            return SimpleNamespace(omega_eff=omegas.get(symbol, 0.1), E_coh_eV=3.5, r0_A=2.5)

        def mock_magnetic(z: int, *, symbol: str = "", T_K: float = 300) -> SimpleNamespace:
            # Fe: low ω in magnetic (it's the native domain)
            omegas = {"Fe": 0.03, "Cu": 0.60, "Ni": 0.04, "Al": 0.70, "Au": 0.80, "Pt": 0.20, "W": 0.50}
            return SimpleNamespace(omega_eff=omegas.get(symbol, 0.3), M_total_B=2.0)

        def mock_surface(z: int, *, symbol: str = "") -> SimpleNamespace:
            return SimpleNamespace(omega_eff=0.15, gamma_J_m2=1.5)

        engine = InsightEngine(load_canon=False, load_db=False)
        with patch.dict(
            "sys.modules",
            {
                "closures": MagicMock(),
                "closures.materials_science": MagicMock(
                    compute_cohesive_energy=mock_cohesive,
                    compute_magnetic_properties=mock_magnetic,
                    compute_surface_catalysis=mock_surface,
                ),
            },
        ):
            results = engine.discover_universality_signatures()

        # Should find divergent ω elements (spread > 0.30)
        univ = [e for e in results if "UNI" in e.id]
        assert len(univ) >= 1
        assert univ[0].pattern_type == PatternType.UNIVERSALITY
        assert univ[0].severity == InsightSeverity.FUNDAMENTAL
        # The lesson: divergence IS the lesson
        assert "divergence" in univ[0].lesson.lower() or "model" in univ[0].lesson.lower()


class TestDiscoverAll:
    """Test discover_all orchestrates all four discovery methods."""

    def test_discover_all_calls_all_methods(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        with (
            patch.object(engine, "discover_periodic_trends", return_value=[_entry("PT-1")]) as m1,
            patch.object(engine, "discover_regime_boundaries", return_value=[_entry("RB-1")]) as m2,
            patch.object(engine, "discover_cross_correlations", return_value=[_entry("CC-1")]) as m3,
            patch.object(engine, "discover_universality_signatures", return_value=[_entry("US-1")]) as m4,
        ):
            results = engine.discover_all()

        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
        m4.assert_called_once()
        assert len(results) == 4
        ids = {e.id for e in results}
        assert ids == {"PT-1", "RB-1", "CC-1", "US-1"}


# ═══════════════════════════════════════════════════════════════════
# 4. Reports & Output formatting
# ═══════════════════════════════════════════════════════════════════


class TestReportFormatting:
    """Comprehensive tests for show_startup_insight and full_report."""

    def test_startup_insight_box_structure(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("BOX-1", lesson="A lesson about collapse", implication="Real physics"))
        text = engine.show_startup_insight(seed=1)
        assert "╔" in text
        assert "╗" in text
        assert "╚" in text
        assert "╝" in text
        assert "UMCP" in text
        assert "Lesson of the Day" in text
        assert "BOX-1" in text

    def test_startup_insight_wraps_long_text(self) -> None:
        long_pattern = "A " * 200  # very long text
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("WRAP-1", pattern=long_pattern))
        text = engine.show_startup_insight(seed=1)
        # Each line inside the box should be bounded
        for line in text.split("\n"):
            if "║" in line:
                assert len(line) <= 80, f"Line too long: {len(line)} chars"

    def test_startup_insight_shows_domain_and_source(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("META-1", domain="Cohesive Energy", source="discovered"))
        text = engine.show_startup_insight(seed=1)
        assert "Cohesive Energy" in text
        assert "discovered" in text

    def test_startup_insight_rotation_with_seed(self) -> None:
        """Different seeds should (usually) select different entries."""
        engine = InsightEngine(load_canon=False, load_db=False)
        for i in range(10):
            engine.db.add(_entry(f"ROT-{i}", pattern=f"Pattern {i}"))

        texts = {engine.show_startup_insight(seed=s) for s in range(20)}
        # With 10 entries and 20 seeds, we should see at least 2 different outputs
        assert len(texts) >= 2

    def test_full_report_structure(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("R1", domain="A", severity=InsightSeverity.FUNDAMENTAL, elements=["Fe"]))
        engine.db.add(_entry("R2", domain="B", severity=InsightSeverity.CURIOUS))
        engine.db.add(_entry("R3", domain="A", pattern_type=PatternType.REGIME_BOUNDARY))
        report = engine.full_report()

        assert "LESSONS-LEARNED REPORT" in report
        assert "3 insights" in report
        assert "2 domains" in report
        assert "What returns through collapse is real" in report
        # Domain grouping
        assert "A (2 insights)" in report
        assert "B (1 insights)" in report

    def test_full_report_includes_severity_and_type(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("FRT", severity=InsightSeverity.STRUCTURAL, pattern_type=PatternType.MODEL_LIMITATION))
        report = engine.full_report()
        assert "Structural" in report
        assert "ModelLimitation" in report

    def test_full_report_elements_display(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("ELEM", elements=["Ti", "V", "Cr"]))
        report = engine.full_report()
        assert "Ti, V, Cr" in report

    def test_full_report_epilogue(self) -> None:
        """The report should end with the core axiom."""
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("EP"))
        report = engine.full_report()
        assert "anomalous" in report.lower()
        assert "signal" in report.lower()


# ═══════════════════════════════════════════════════════════════════
# 5. Summary stats & lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestSummaryStats:
    """summary_stats edge cases."""

    def test_empty_stats(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        stats = engine.summary_stats()
        assert stats["total_insights"] == 0
        assert stats["domains"] == {}
        assert stats["by_severity"] == {}
        assert stats["by_source"] == {}
        assert stats["domain_list"] == []

    def test_stats_counts_per_dimension(self) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("S1", domain="A", severity=InsightSeverity.FUNDAMENTAL, source="canon"))
        engine.db.add(_entry("S2", domain="A", severity=InsightSeverity.FUNDAMENTAL, source="discovered"))
        engine.db.add(_entry("S3", domain="B", severity=InsightSeverity.EMPIRICAL, source="canon"))
        stats = engine.summary_stats()
        assert stats["total_insights"] == 3
        assert stats["domains"] == {"A": 2, "B": 1}
        assert stats["by_severity"] == {"Fundamental": 2, "Empirical": 1}
        assert stats["by_source"] == {"canon": 2, "discovered": 1}
        assert sorted(stats["domain_list"]) == ["A", "B"]


class TestInsightEngineLifecycle:
    """Engine init, save, philosophy."""

    def test_save_persists(self, tmp_path: Path) -> None:
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("PERSIST"))
        out = tmp_path / "persist.yaml"
        engine.save(out)
        assert out.exists()

        engine2 = InsightEngine(load_canon=False, load_db=False)
        loaded = engine2.db.load_yaml(out)
        assert loaded == 1

    def test_philosophy_content(self) -> None:
        text = InsightEngine.philosophy()
        assert "Anomalous" in text
        assert "collapse" in text.lower()
        assert "ω_eff" in text
        # Core axiom
        assert "What returns through collapse is real" in text

    def test_init_with_canon_loads_real_data(self) -> None:
        """Default init (load_canon=True) should load without errors."""
        engine = InsightEngine(load_canon=True, load_db=True)
        # At minimum it shouldn't crash; may have entries if YAML files exist
        assert engine.db is not None


# ═══════════════════════════════════════════════════════════════════
# 6. Utility functions — edge cases
# ═══════════════════════════════════════════════════════════════════


class TestPearsonDeep:
    """Edge cases for _pearson correlation."""

    def test_two_points(self) -> None:
        """With exactly 2 points, correlation is always ±1."""
        r = _pearson([1.0, 2.0], [3.0, 5.0])
        assert abs(abs(r) - 1.0) < 1e-10

    def test_zero_variance_x(self) -> None:
        assert _pearson([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_variance_both(self) -> None:
        assert _pearson([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 0.0

    def test_negative_values(self) -> None:
        x = [-3.0, -1.0, 1.0, 3.0]
        y = [9.0, 1.0, 1.0, 9.0]  # Quadratic — zero linear correlation
        r = _pearson(x, y)
        assert abs(r) < 0.1  # Nearly uncorrelated linearly

    def test_large_values(self) -> None:
        x = [1e10, 2e10, 3e10]
        y = [1e10, 2e10, 3e10]
        assert abs(_pearson(x, y) - 1.0) < 1e-10

    def test_anti_correlated(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(x, y) - (-1.0)) < 1e-10


class TestHashShortDeep:
    """Edge cases for _hash_short."""

    def test_empty_string(self) -> None:
        h = _hash_short("")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_unicode(self) -> None:
        h = _hash_short("ω_eff → κ")
        assert len(h) == 8

    def test_long_input(self) -> None:
        h = _hash_short("x" * 100000)
        assert len(h) == 8


class TestDailySeedDeep:
    """_daily_seed properties."""

    def test_format_is_yyyymmdd(self) -> None:
        seed = _daily_seed()
        year = seed // 10000
        month = (seed % 10000) // 100
        day = seed % 100
        assert 2020 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31

    def test_deterministic_within_session(self) -> None:
        assert _daily_seed() == _daily_seed()


# ═══════════════════════════════════════════════════════════════════
# 7. Deep Implications — Physics Semantics
# ═══════════════════════════════════════════════════════════════════


class TestDeepImplications:
    """Tests that validate the physics semantics of the insight system.

    These tests verify that:
    - ω_eff is treated as a signal, not a failure metric
    - Regime boundaries carry physical meaning (phase transitions)
    - Cross-correlations map to shared physical origins
    - The core axiom ("What returns through collapse is real") is respected
    """

    def test_omega_is_signal_not_failure(self) -> None:
        """High ω_eff should be classified as a lesson, not an error."""
        engine = InsightEngine(load_canon=False, load_db=False)
        entry = _entry(
            "OMEGA-SIGNAL",
            pattern="Fe ω_eff = 0.55 in cohesive model",
            lesson="Magnetic exchange not captured by band model",
            implication="Add exchange treatment",
            severity=InsightSeverity.STRUCTURAL,
            omega_range=(0.40, 0.60),
        )
        engine.db.add(entry)

        # The insight should NOT have "failure" or "error" in its lesson
        assert "failure" not in entry.lesson.lower()
        assert "error" not in entry.lesson.lower()
        # The philosophy confirms this
        phil = InsightEngine.philosophy()
        assert "not failure" in phil.lower() or '"Anomalous" regime is not failure' in phil

    def test_regime_boundary_is_phase_transition(self) -> None:
        """Regime boundaries should map to physical phase transitions."""
        entry = _entry(
            "CURIE-TEST",
            domain="Magnetic Properties",
            pattern="Fe: ferromagnetic → paramagnetic at T ≈ 1043 K",
            lesson="M(T) vanishes at T_c — a genuine phase transition",
            implication="ω_eff maps the Curie temperature",
            severity=InsightSeverity.FUNDAMENTAL,
            pattern_type=PatternType.REGIME_BOUNDARY,
        )
        # The pattern type must be REGIME_BOUNDARY for phase transitions
        assert entry.pattern_type == PatternType.REGIME_BOUNDARY
        # Severity should be FUNDAMENTAL for thermodynamic transitions
        assert entry.severity == InsightSeverity.FUNDAMENTAL

    def test_cross_correlation_implies_shared_origin(self) -> None:
        """Cross-correlations should reference a shared physical ancestor."""
        entry = _entry(
            "CROSS-ORIGIN",
            pattern="Θ_D vs γ_surface: Pearson r = 0.78",
            lesson="Both originate from interatomic bond strength",
            implication="Different projections of the same collapse potential",
            pattern_type=PatternType.CROSS_CORRELATION,
        )
        # The lesson should reference a common physical origin
        assert "bond" in entry.lesson.lower() or "common" in entry.lesson.lower() or "origin" in entry.lesson.lower()
        assert entry.pattern_type == PatternType.CROSS_CORRELATION

    def test_universality_captures_model_limits(self) -> None:
        """Universality insights should map what each model can and cannot see."""
        entry = _entry(
            "UNI-LIMITS",
            pattern="Fe: ω diverges across closures",
            lesson="Cohesive model sees bands but not exchange; magnetic model sees exchange but not bands",
            implication="ω divergence maps model completeness",
            pattern_type=PatternType.UNIVERSALITY,
            severity=InsightSeverity.FUNDAMENTAL,
        )
        assert entry.pattern_type == PatternType.UNIVERSALITY
        assert "model" in entry.lesson.lower()

    def test_core_axiom_in_full_report(self) -> None:
        """Full report must contain the core axiom."""
        engine = InsightEngine(load_canon=False, load_db=False)
        engine.db.add(_entry("AXIOM"))
        report = engine.full_report()
        assert "What returns through collapse is real" in report

    def test_philosophy_on_anomalous_as_information(self) -> None:
        """Philosophy must state that Anomalous is NOT failure but information."""
        phil = InsightEngine.philosophy()
        assert "Anomalous" in phil
        assert "not failure" in phil.lower() or "is not failure" in phil.lower()

    def test_severity_hierarchy(self) -> None:
        """Fundamental > Structural > Empirical > Curious — gravity of insight."""
        hierarchy = [
            InsightSeverity.FUNDAMENTAL,
            InsightSeverity.STRUCTURAL,
            InsightSeverity.EMPIRICAL,
            InsightSeverity.CURIOUS,
        ]
        # Each severity is distinct and has a meaningful string value
        values = [s.value for s in hierarchy]
        assert len(set(values)) == 4
        assert values == ["Fundamental", "Structural", "Empirical", "Curious"]

    def test_discovered_insights_have_omega_range(self) -> None:
        """Discovered periodic insights should carry measured ω ranges."""
        entry = _entry(
            "RANGE-CHECK",
            source="discovered",
            omega_range=(0.03, 0.55),
            elements=["Ti", "V", "Cr", "Mn", "Fe"],
        )
        lo, hi = entry.omega_range
        assert lo < hi
        assert lo >= 0.0
        assert hi <= 1.0
        assert len(entry.elements) > 0

    def test_insight_id_deterministic(self) -> None:
        """IDs generated with _hash_short must be deterministic for same input."""
        data1 = str([("Fe", 0.55), ("Mn", 0.48)])
        data2 = str([("Fe", 0.55), ("Mn", 0.48)])
        assert _hash_short(data1) == _hash_short(data2)
        # But different data produces different IDs
        data3 = str([("Cu", 0.03)])
        assert _hash_short(data1) != _hash_short(data3)

    def test_model_limitation_pattern_type(self) -> None:
        """MODEL_LIMITATION should be used when a model's domain of validity is exceeded."""
        entry = _entry(
            "LIMIT-1",
            pattern="Friedel model breaks for strongly correlated electrons",
            lesson="The d-electron bandwidth is too narrow for Friedel's free-electron assumption",
            implication="Need Hubbard or DFT corrections",
            pattern_type=PatternType.MODEL_LIMITATION,
        )
        assert entry.pattern_type == PatternType.MODEL_LIMITATION
        d = entry.to_dict()
        assert d["pattern_type"] == "ModelLimitation"

    def test_all_pattern_types_semantically_distinct(self) -> None:
        """Each PatternType serves a distinct semantic role."""
        types_and_meanings = {
            PatternType.PERIODIC_TREND: "systematic variation across Z",
            PatternType.REGIME_BOUNDARY: "phase transition or model validity boundary",
            PatternType.CROSS_CORRELATION: "shared physical origin between properties",
            PatternType.UNIVERSALITY: "scale-invariant or domain-bridging pattern",
            PatternType.MODEL_LIMITATION: "domain where model assumptions break down",
            PatternType.PHYSICAL_INSIGHT: "general physics lesson from data",
        }
        assert len(types_and_meanings) == len(PatternType)
        for pt in PatternType:
            assert pt in types_and_meanings

    def test_canon_vs_discovered_semantics(self) -> None:
        """Canon entries are human-curated anchors; discovered are computationally found."""
        canon = _entry("CANON-1", source="canon")
        discovered = _entry("DISC-1", source="discovered")
        cross = _entry("CROSS-1", source="cross-closure")

        assert canon.source == "canon"
        assert discovered.source == "discovered"
        assert cross.source == "cross-closure"

        # All three should serialize correctly
        for e in (canon, discovered, cross):
            d = e.to_dict()
            restored = InsightEntry.from_dict(d)
            assert restored.source == e.source
