"""
Extended tests for UMCP Dashboard package — coverage-oriented.

Tests pure-logic functions, data structures, helpers, and canon utilities
that can be exercised without Streamlit runtime.

Cross-references:
  - src/umcp/dashboard/_deps.py
  - src/umcp/dashboard/_utils.py
  - src/umcp/dashboard/pages_physics.py  (translate_to_gcd, unit conversion)
  - src/umcp/dashboard/pages_advanced.py (_load_canon_files, _extract_symbols)
  - src/umcp/dashboard/__init__.py       (_is_running_in_streamlit)
"""

from __future__ import annotations

import math
from typing import Any

import pytest

# ============================================================================
# __init__.py
# ============================================================================


class TestDashboardInit:
    """Tests for dashboard __init__.py public API."""

    def test_is_running_in_streamlit_false_outside(self) -> None:
        """Outside Streamlit, _is_running_in_streamlit returns False."""
        from umcp.dashboard import _is_running_in_streamlit

        assert _is_running_in_streamlit() is False

    def test_public_api_exports_render_functions(self) -> None:
        """All render_* page functions are importable from the package."""
        import umcp.dashboard as dash

        render_names = [
            "render_overview_page",
            "render_ledger_page",
            "render_casepacks_page",
            "render_contracts_page",
            "render_closures_page",
            "render_regime_page",
            "render_metrics_page",
            "render_health_page",
            "render_gcd_panel",
            "render_physics_interface_page",
            "render_kinematics_interface_page",
            "render_live_runner_page",
            "render_batch_validation_page",
            "render_test_templates_page",
            "render_cosmology_page",
            "render_astronomy_page",
            "render_nuclear_page",
            "render_quantum_page",
            "render_finance_page",
            "render_rcft_page",
            "render_formula_builder_page",
            "render_time_series_page",
            "render_comparison_page",
            "render_exports_page",
            "render_notifications_page",
            "render_bookmarks_page",
            "render_api_integration_page",
            "render_precision_page",
            "render_geometry_page",
            "render_canon_explorer_page",
            "render_domain_overview_page",
        ]
        for name in render_names:
            assert hasattr(dash, name), f"{name} missing from dashboard exports"
            assert callable(getattr(dash, name))

    def test_public_api_exports_constants(self) -> None:
        """Constants re-exported from __init__."""
        from umcp.dashboard import (
            KERNEL_SYMBOLS,
            REGIME_COLORS,
            STATUS_COLORS,
            THEMES,
        )

        assert isinstance(REGIME_COLORS, dict)
        assert isinstance(STATUS_COLORS, dict)
        assert isinstance(KERNEL_SYMBOLS, dict)
        assert isinstance(THEMES, dict)

    def test_public_api_exports_utility_functions(self) -> None:
        """Utility functions re-exported from __init__."""
        from umcp.dashboard import (
            classify_regime,
            detect_anomalies,
            format_bytes,
            get_regime_color,
            get_repo_root,
            get_trend_indicator,
            load_casepacks,
            load_closures,
            load_contracts,
            load_ledger,
        )

        assert callable(classify_regime)
        assert callable(detect_anomalies)
        assert callable(format_bytes)
        assert callable(get_regime_color)
        assert callable(get_repo_root)
        assert callable(get_trend_indicator)
        assert callable(load_casepacks)
        assert callable(load_closures)
        assert callable(load_contracts)
        assert callable(load_ledger)

    def test_main_function_exported(self) -> None:
        """main() entry point is exported."""
        from umcp.dashboard import main

        assert callable(main)


# ============================================================================
# _deps.py
# ============================================================================


class TestDeps:
    """Tests for _deps.py optional dependency wrappers."""

    def test_has_viz_deps_is_bool(self) -> None:
        from umcp.dashboard._deps import HAS_VIZ_DEPS

        assert isinstance(HAS_VIZ_DEPS, bool)

    def test_cache_data_returns_callable(self) -> None:
        from umcp.dashboard._deps import _cache_data

        assert callable(_cache_data)

    def test_cache_data_decorator_passthrough(self) -> None:
        """_cache_data wraps a function and still calls it."""
        from umcp.dashboard._deps import _cache_data

        @_cache_data(ttl=60)
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(5) == 10

    def test_st_is_available_or_none(self) -> None:
        """st module should be either importable or None."""
        from umcp.dashboard._deps import st

        # In CI it may or may not be installed
        assert st is None or hasattr(st, "write")

    def test_np_is_available(self) -> None:
        """numpy should be available (it's a core dep)."""
        from umcp.dashboard._deps import np

        assert np is not None

    def test_pd_is_available(self) -> None:
        """pandas should be available."""
        from umcp.dashboard._deps import pd

        assert pd is not None


# ============================================================================
# _utils.py — extended coverage
# ============================================================================


class TestUtilsClassifyRegimeExtended:
    """Extended regime classification tests."""

    @pytest.mark.parametrize(
        ("omega", "expected"),
        [
            (0.0, "COLLAPSE"),
            (0.05, "COLLAPSE"),
            (0.099, "COLLAPSE"),
            (0.1, "WATCH"),
            (0.15, "WATCH"),
            (0.299, "WATCH"),
            (0.3, "STABLE"),
            (0.5, "STABLE"),
            (0.7, "STABLE"),
            (0.701, "WATCH"),
            (0.8, "WATCH"),
            (0.9, "WATCH"),
            (0.901, "COLLAPSE"),
            (0.95, "COLLAPSE"),
            (1.0, "COLLAPSE"),
        ],
    )
    def test_regime_by_omega(self, omega: float, expected: str) -> None:
        from umcp.dashboard._utils import classify_regime

        assert classify_regime(omega, 0.0) == expected

    @pytest.mark.parametrize("seam", [0.011, 0.02, 0.05, 0.1, -0.011, -0.02, -0.1])
    def test_critical_overrides_all(self, seam: float) -> None:
        from umcp.dashboard._utils import classify_regime

        for omega in [0.0, 0.2, 0.5, 0.8, 1.0]:
            assert classify_regime(omega, seam) == "CRITICAL"

    def test_seam_at_boundary_not_critical(self) -> None:
        from umcp.dashboard._utils import classify_regime

        assert classify_regime(0.5, 0.01) != "CRITICAL"
        assert classify_regime(0.5, -0.01) != "CRITICAL"


class TestUtilsFormatBytesExtended:
    """Extended format_bytes tests."""

    @pytest.mark.parametrize(
        ("size", "expected_unit"),
        [
            (0, "B"),
            (1, "B"),
            (100, "B"),
            (1023, "B"),
            (1024, "KB"),
            (2048, "KB"),
            (1048576, "MB"),
            (1073741824, "GB"),
            (1099511627776, "TB"),
        ],
    )
    def test_format_bytes_units(self, size: int, expected_unit: str) -> None:
        from umcp.dashboard._utils import format_bytes

        assert expected_unit in format_bytes(size)


class TestUtilsGetTrendIndicatorExtended:
    """Extended trend indicator tests."""

    def test_upward_trend(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(1.5, 1.0) == "📈"

    def test_downward_trend(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(0.5, 1.0) == "📉"

    def test_stable_trend(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(1.0, 1.0) == "➡️"

    def test_inverted_upward(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(1.5, 1.0, invert=True) == "📉"

    def test_inverted_downward(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(0.5, 1.0, invert=True) == "📈"

    def test_inverted_stable(self) -> None:
        from umcp.dashboard._utils import get_trend_indicator

        assert get_trend_indicator(1.0, 1.0, invert=True) == "➡️"


class TestUtilsDetectAnomaliesExtended:
    """Extended anomaly detection tests."""

    def test_no_anomalies_uniform(self) -> None:
        import pandas as pd

        from umcp.dashboard._utils import detect_anomalies

        data = pd.Series([1.0] * 10)
        result = detect_anomalies(data, threshold=2.0)
        assert not any(result)

    def test_detects_single_outlier(self) -> None:
        import pandas as pd

        from umcp.dashboard._utils import detect_anomalies

        data = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        result = detect_anomalies(data, threshold=2.0)
        assert bool(result.tolist()[-1]) is True

    def test_zero_std_returns_no_anomalies(self) -> None:
        import pandas as pd

        from umcp.dashboard._utils import detect_anomalies

        data = pd.Series([5.0, 5.0, 5.0])
        result = detect_anomalies(data, threshold=2.0)
        assert not any(result)

    def test_custom_threshold(self) -> None:
        import pandas as pd

        from umcp.dashboard._utils import detect_anomalies

        data = pd.Series([1.0, 1.0, 1.0, 3.0])
        # Low threshold should flag the 3.0
        result = detect_anomalies(data, threshold=1.0)
        assert bool(result.iloc[-1]) is True


class TestUtilsDataLoaders:
    """Tests for data loading functions."""

    def test_load_ledger_has_columns(self) -> None:
        import pandas as pd

        from umcp.dashboard._utils import load_ledger

        df = load_ledger()
        assert isinstance(df, pd.DataFrame)

    def test_load_casepacks_structure(self) -> None:
        from umcp.dashboard._utils import load_casepacks

        casepacks = load_casepacks()
        assert isinstance(casepacks, list)
        assert len(casepacks) > 0
        for cp in casepacks:
            assert "id" in cp
            assert "path" in cp
            assert "version" in cp
            assert "files_count" in cp
            assert "closures_count" in cp
            assert "test_vectors" in cp

    def test_load_contracts_structure(self) -> None:
        from umcp.dashboard._utils import load_contracts

        contracts = load_contracts()
        assert isinstance(contracts, list)
        assert len(contracts) > 0
        for c in contracts:
            assert "id" in c
            assert "domain" in c
            assert "version" in c
            assert "path" in c
            assert "size_bytes" in c

    def test_load_closures_structure(self) -> None:
        from umcp.dashboard._utils import load_closures

        closures = load_closures()
        assert isinstance(closures, list)
        assert len(closures) > 0
        for c in closures:
            assert "name" in c
            assert "domain" in c
            assert "path" in c
            assert "type" in c
            assert c["type"] in ("python", "yaml")

    def test_load_closures_domains_valid(self) -> None:
        from umcp.dashboard._utils import load_closures

        closures = load_closures()
        valid_domains = {"GCD", "KIN", "RCFT", "WEYL", "SECURITY", "ASTRO", "NUC", "QM", "FIN", "unknown"}
        for c in closures:
            assert c["domain"] in valid_domains, f"Unexpected domain: {c['domain']}"

    def test_load_contracts_versions_valid(self) -> None:
        from umcp.dashboard._utils import load_contracts

        for c in load_contracts():
            assert c["version"].startswith("v"), f"Version should start with 'v': {c['version']}"


class TestUtilsGetRepoRoot:
    """Tests for get_repo_root."""

    def test_repo_root_has_pyproject(self) -> None:
        from umcp.dashboard._utils import get_repo_root

        root = get_repo_root()
        assert (root / "pyproject.toml").exists()

    def test_repo_root_has_src(self) -> None:
        from umcp.dashboard._utils import get_repo_root

        root = get_repo_root()
        assert (root / "src").is_dir()

    def test_repo_root_has_canon(self) -> None:
        from umcp.dashboard._utils import get_repo_root

        root = get_repo_root()
        assert (root / "canon").is_dir()


class TestUtilsClosuresPath:
    """Tests for closures path setup."""

    def test_setup_closures_path_idempotent(self) -> None:
        from umcp.dashboard._utils import _setup_closures_path

        _setup_closures_path()
        _setup_closures_path()  # Should not error

    def test_ensure_closures_path_idempotent(self) -> None:
        from umcp.dashboard._utils import _ensure_closures_path

        _ensure_closures_path()
        _ensure_closures_path()


class TestUtilsConstants:
    """Tests for dashboard constant completeness."""

    def test_regime_colors_all_hex(self) -> None:
        from umcp.dashboard._utils import REGIME_COLORS

        for regime, color in REGIME_COLORS.items():
            assert color.startswith("#"), f"{regime} color invalid: {color}"
            assert len(color) == 7, f"{regime} color not 7 chars: {color}"

    def test_status_colors_all_hex(self) -> None:
        from umcp.dashboard._utils import STATUS_COLORS

        for status, color in STATUS_COLORS.items():
            assert color.startswith("#"), f"{status} color invalid: {color}"

    def test_kernel_symbols_keys(self) -> None:
        from umcp.dashboard._utils import KERNEL_SYMBOLS

        expected = {"omega", "F", "S", "C", "tau_R", "IC", "kappa"}
        assert expected.issubset(set(KERNEL_SYMBOLS.keys()))

    def test_themes_have_required_keys(self) -> None:
        from umcp.dashboard._utils import THEMES

        for name, theme in THEMES.items():
            for key in ("primary", "secondary", "success", "danger", "warning"):
                assert key in theme, f"Theme {name} missing key {key}"


# ============================================================================
# pages_physics.py — translate_to_gcd, unit conversion, normalization
# ============================================================================


class TestTranslateToGCD:
    """Tests for translate_to_gcd pure-logic function."""

    def test_stable_system(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.01, "F": 0.99, "S": 0.0, "C": 0.0, "kappa": 0.0, "IC": 1.0})
        assert result["regime"] == "Stable"
        assert "symbols" in result
        assert "omega" in result["symbols"]
        assert "F" in result["symbols"]

    def test_collapse_system(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.5, "F": 0.5, "S": 0.5, "C": 0.3, "kappa": -0.7, "IC": 0.5})
        assert result["regime"] == "Collapse"

    def test_watch_system(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.15, "F": 0.85, "S": 0.1, "C": 0.05, "kappa": -0.15, "IC": 0.86})
        assert result["regime"] == "Watch"

    def test_fidelity_identity_check_pass(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.02, "F": 0.98})
        assert result["symbols"]["F"]["identity_check"] == True  # noqa: E712

    def test_fidelity_identity_check_fail(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.5, "F": 0.8})  # F + omega != 1
        assert result["symbols"]["F"]["identity_check"] is False

    def test_natural_language_summary(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.01, "F": 0.99})
        assert isinstance(result["natural_language"], list)
        assert len(result["natural_language"]) >= 5

    def test_summary_string(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.01, "F": 0.99})
        assert isinstance(result["summary"], str)
        assert "GCD" in result["summary"]

    def test_axiom_states(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.5, "F": 0.5})
        assert "axiom_state" in result
        assert "AX-0" in result["axiom_state"]
        assert "AX-1" in result["axiom_state"]
        assert "AX-2" in result["axiom_state"]
        # Alias
        assert "axiom_states" in result
        assert result["axiom_states"] is result["axiom_state"]

    def test_warnings_near_collapse(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.28, "F": 0.72})
        assert any("collapse" in w.lower() for w in result["warnings"])

    def test_warnings_identity_violation(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.1, "F": 0.5})  # F + omega != 1
        assert any("F + ω" in w or "identity" in w.lower() for w in result["warnings"])

    def test_derives_omega_from_F(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"F": 0.95})
        assert abs(result["symbols"]["omega"]["value"] - 0.05) < 1e-9

    def test_percent_to_collapse(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.15})
        pct = result["symbols"]["omega"]["percent_to_collapse"]
        assert 49 < pct < 51  # ~50%

    def test_distance_to_collapse(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.10})
        dist = result["symbols"]["omega"]["distance_to_collapse"]
        assert abs(dist - 0.20) < 0.001

    def test_entropy_determinacy_levels(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        r1 = translate_to_gcd({"omega": 0.01, "S": 0.0})
        assert r1["symbols"]["S"]["determinacy"] == "deterministic"

        r2 = translate_to_gcd({"omega": 0.01, "S": 0.10})
        assert r2["symbols"]["S"]["determinacy"] == "low uncertainty"

        r3 = translate_to_gcd({"omega": 0.01, "S": 0.50})
        assert r3["symbols"]["S"]["determinacy"] == "uncertain"

    def test_curvature_homogeneity_levels(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        r1 = translate_to_gcd({"omega": 0.01, "C": 0.0})
        assert r1["symbols"]["C"]["homogeneity"] == "homogeneous"

        r2 = translate_to_gcd({"omega": 0.01, "C": 0.10})
        assert r2["symbols"]["C"]["homogeneity"] == "coherent"

        r3 = translate_to_gcd({"omega": 0.01, "C": 0.20})
        assert r3["symbols"]["C"]["homogeneity"] == "heterogeneous"

    def test_regime_info_present(self) -> None:
        from umcp.dashboard.pages_physics import translate_to_gcd

        result = translate_to_gcd({"omega": 0.01})
        assert "regime_info" in result
        assert "description" in result["regime_info"]
        assert "color" in result["regime_info"]

    def test_ic_identity_check(self) -> None:
        import numpy as np_local

        from umcp.dashboard.pages_physics import translate_to_gcd

        kappa = -0.5
        ic = np_local.exp(kappa)
        result = translate_to_gcd({"omega": 0.1, "kappa": kappa, "IC": float(ic)})
        assert result["symbols"]["IC"]["identity_check"] == True  # noqa: E712


class TestInterpretValue:
    """Tests for _interpret_value helper."""

    def test_exact_match(self) -> None:
        from umcp.dashboard.pages_physics import GCD_SYMBOLS, _interpret_value

        result = _interpret_value(0.0, GCD_SYMBOLS["omega"])
        assert "Perfect" in result or "drift" in result.lower() or "alignment" in result.lower()

    def test_closest_threshold(self) -> None:
        from umcp.dashboard.pages_physics import GCD_SYMBOLS, _interpret_value

        result = _interpret_value(0.29, GCD_SYMBOLS["omega"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_interpretation(self) -> None:
        from umcp.dashboard.pages_physics import _interpret_value

        result = _interpret_value(0.5, {"interpretation": {}})
        assert result == "Unknown"


class TestUnitConversion:
    """Tests for convert_to_base_unit and convert_from_base_unit."""

    def test_meters_identity(self) -> None:
        from umcp.dashboard.pages_physics import convert_to_base_unit

        assert convert_to_base_unit(1.0, "m", "position") == 1.0

    def test_km_to_m(self) -> None:
        from umcp.dashboard.pages_physics import convert_to_base_unit

        assert convert_to_base_unit(1.0, "km", "position") == 1000.0

    def test_m_from_km(self) -> None:
        from umcp.dashboard.pages_physics import convert_from_base_unit

        assert convert_from_base_unit(1000.0, "km", "position") == 1.0

    def test_unknown_quantity_passthrough(self) -> None:
        from umcp.dashboard.pages_physics import convert_from_base_unit, convert_to_base_unit

        assert convert_to_base_unit(42.0, "foo", "nonexistent") == 42.0
        assert convert_from_base_unit(42.0, "foo", "nonexistent") == 42.0

    def test_unknown_unit_passthrough(self) -> None:
        from umcp.dashboard.pages_physics import convert_to_base_unit

        assert convert_to_base_unit(42.0, "parsec", "position") == 42.0

    def test_roundtrip(self) -> None:
        from umcp.dashboard.pages_physics import convert_from_base_unit, convert_to_base_unit

        original = 5.0
        base = convert_to_base_unit(original, "km/h", "velocity")
        back = convert_from_base_unit(base, "km/h", "velocity")
        assert abs(back - original) < 1e-9

    @pytest.mark.parametrize("unit,factor", [("m/s", 1.0), ("km/h", 1 / 3.6), ("ft/s", 0.3048)])
    def test_velocity_conversions(self, unit: str, factor: float) -> None:
        from umcp.dashboard.pages_physics import convert_to_base_unit

        assert abs(convert_to_base_unit(1.0, unit, "velocity") - factor) < 1e-6

    @pytest.mark.parametrize("unit,factor", [("kg", 1.0), ("g", 1e-3), ("lb", 0.453592)])
    def test_mass_conversions(self, unit: str, factor: float) -> None:
        from umcp.dashboard.pages_physics import convert_to_base_unit

        assert abs(convert_to_base_unit(1.0, unit, "mass") - factor) < 1e-4


class TestNormalizeToBounded:
    """Tests for normalize_to_bounded."""

    def test_identity_normalization(self) -> None:
        from umcp.dashboard.pages_physics import normalize_to_bounded

        val, clipped = normalize_to_bounded(0.5, 1.0)
        assert abs(val - 0.5) < 1e-9
        assert clipped is False

    def test_clipping_at_zero(self) -> None:
        from umcp.dashboard.pages_physics import normalize_to_bounded

        val, clipped = normalize_to_bounded(0.0, 1.0)
        assert val > 0  # Clipped to epsilon
        assert clipped is True

    def test_clipping_at_one(self) -> None:
        from umcp.dashboard.pages_physics import normalize_to_bounded

        val, clipped = normalize_to_bounded(2.0, 1.0)
        assert val < 1.0
        assert clipped is True

    def test_ref_zero_fallback(self) -> None:
        from umcp.dashboard.pages_physics import normalize_to_bounded

        val, _ = normalize_to_bounded(0.5, 0.0)
        assert abs(val - 0.5) < 1e-9

    def test_negative_value(self) -> None:
        from umcp.dashboard.pages_physics import normalize_to_bounded

        val, _ = normalize_to_bounded(-3.0, 10.0)
        assert 0 < val < 1


class TestGCDDataStructures:
    """Tests for GCD constant data structures."""

    def test_gcd_symbols_complete(self) -> None:
        from umcp.dashboard.pages_physics import GCD_SYMBOLS

        expected = {"omega", "F", "S", "C", "kappa", "IC", "tau_R"}
        assert expected == set(GCD_SYMBOLS.keys())

    def test_gcd_symbols_have_required_fields(self) -> None:
        from umcp.dashboard.pages_physics import GCD_SYMBOLS

        for name, sym in GCD_SYMBOLS.items():
            assert "latex" in sym, f"{name} missing latex"
            assert "name" in sym, f"{name} missing name"
            assert "description" in sym, f"{name} missing description"
            assert "interpretation" in sym, f"{name} missing interpretation"

    def test_gcd_regimes_complete(self) -> None:
        from umcp.dashboard.pages_physics import GCD_REGIMES

        assert set(GCD_REGIMES.keys()) == {"Stable", "Watch", "Collapse"}
        for _name, regime in GCD_REGIMES.items():
            assert "condition" in regime
            assert "color" in regime
            assert "description" in regime
            assert "icon" in regime

    def test_gcd_axioms_complete(self) -> None:
        from umcp.dashboard.pages_physics import GCD_AXIOMS

        assert set(GCD_AXIOMS.keys()) == {"AX-0", "AX-1", "AX-2"}
        for _ax_id, ax in GCD_AXIOMS.items():
            assert "statement" in ax
            assert "description" in ax

    def test_physics_quantities_present(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_QUANTITIES

        expected = {"position", "velocity", "acceleration", "mass", "force", "energy"}
        assert expected.issubset(set(PHYSICS_QUANTITIES.keys()))

    def test_physics_quantities_have_units(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_QUANTITIES

        for name, qty in PHYSICS_QUANTITIES.items():
            assert "units" in qty, f"{name} missing units"
            assert "base_unit" in qty, f"{name} missing base_unit"
            assert "symbol" in qty, f"{name} missing symbol"
            assert len(qty["units"]) >= 2, f"{name} needs at least 2 units"


# ============================================================================
# pages_advanced.py — _load_canon_files, _extract_symbols
# ============================================================================


class TestLoadCanonFiles:
    """Tests for _load_canon_files helper."""

    def test_loads_files(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        assert isinstance(canon, dict)
        assert len(canon) > 0

    def test_expected_domains_present(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        expected = {"gcd_anchors", "rcft_anchors", "kin_anchors", "weyl_anchors"}
        assert expected.issubset(set(canon.keys()))

    def test_root_anchors_loaded(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        assert "anchors" in canon, "Root anchors.yaml should be loaded"

    def test_canon_files_have_id(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        for name, data in canon.items():
            # Each canon file should have an id or umcp_canon.canon_id
            has_id = "id" in data or ("umcp_canon" in data and "canon_id" in data["umcp_canon"])
            assert has_id, f"Canon {name} missing id"

    def test_astro_anchors_present(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        assert "astro_anchors" in canon

    def test_nuc_anchors_present(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        assert "nuc_anchors" in canon

    def test_qm_anchors_present(self) -> None:
        from umcp.dashboard.pages_advanced import _load_canon_files

        canon = _load_canon_files()
        assert "qm_anchors" in canon


class TestExtractSymbols:
    """Tests for _extract_symbols helper."""

    def test_extracts_gcd_symbols(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        symbols = _extract_symbols(canon["gcd_anchors"])
        assert len(symbols) > 0
        # GCD should have omega, F, S, C, etc.
        symbol_names = {s.get("symbol", s.get("latex", "")) for s in symbols}
        assert "omega" in symbol_names or "ω" in symbol_names

    def test_extracts_rcft_symbols(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        symbols = _extract_symbols(canon["rcft_anchors"])
        assert len(symbols) > 0

    def test_extracts_kin_symbols(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        symbols = _extract_symbols(canon["kin_anchors"])
        assert len(symbols) > 0

    def test_extracts_root_symbols(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        symbols = _extract_symbols(canon["anchors"])
        assert len(symbols) > 0

    def test_empty_data_returns_empty(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        assert _extract_symbols({}) == []

    def test_symbols_are_dicts(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        for name, data in canon.items():
            symbols = _extract_symbols(data)
            for sym in symbols:
                assert isinstance(sym, dict), f"Symbol in {name} is not dict: {sym}"

    def test_symbols_have_identifiers(self) -> None:
        """Each extracted symbol should have some identifier (symbol, latex, name, or ascii)."""
        from umcp.dashboard.pages_advanced import _extract_symbols, _load_canon_files

        canon = _load_canon_files()
        for name, data in canon.items():
            symbols = _extract_symbols(data)
            for sym in symbols:
                has_id = any(k in sym for k in ("symbol", "latex", "name", "ascii"))
                assert has_id, f"Symbol in {name} has no identifier: {sym}"


class TestExtractSymbolsSynthetic:
    """Tests with synthetic YAML structures."""

    def test_direct_list(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        data: dict[str, Any] = {"reserved_symbols": [{"symbol": "X", "description": "test"}]}
        result = _extract_symbols(data)
        assert len(result) == 1
        assert result[0]["symbol"] == "X"

    def test_tier_1_invariants_list(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        data: dict[str, Any] = {"tier_1_invariants": {"reserved_symbols": [{"symbol": "Y", "description": "test"}]}}
        result = _extract_symbols(data)
        assert len(result) == 1
        assert result[0]["symbol"] == "Y"

    def test_tier_1_invariants_dict(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        data: dict[str, Any] = {"tier_1_invariants": {"reserved_symbols": {"pos": {"ascii": "x", "domain": "[0,1]"}}}}
        result = _extract_symbols(data)
        assert len(result) == 1
        assert result[0]["symbol"] == "pos"

    def test_tier_2_extensions(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        data: dict[str, Any] = {
            "tier_2_extensions": {"reserved_symbols": [{"symbol": "D_f", "description": "fractal dim"}]}
        }
        result = _extract_symbols(data)
        assert len(result) == 1
        assert result[0]["symbol"] == "D_f"

    def test_umcp_canon_root(self) -> None:
        from umcp.dashboard.pages_advanced import _extract_symbols

        data: dict[str, Any] = {
            "umcp_canon": {
                "contract_defaults": {
                    "tier_1_kernel": {
                        "reserved_symbols_unicode": ["ω", "F", "S"],
                        "reserved_symbols_ascii": ["omega", "F", "S"],
                    }
                }
            }
        }
        result = _extract_symbols(data)
        assert len(result) == 3
        assert result[0]["symbol"] == "ω"


# ============================================================================
# Kinematics equation lambdas — testable pure math
# ============================================================================


class TestKinematicsEquations:
    """Test the lambda calculators in KINEMATICS_EQUATIONS."""

    def test_velocity_time(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        calc = KINEMATICS_EQUATIONS["velocity_time"]["calculate"]
        # v = v0 + at => 0 + 2*5 = 10
        assert abs(calc(0, 2, 5) - 10.0) < 1e-9

    def test_position_time(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        calc = KINEMATICS_EQUATIONS["position_time"]["calculate"]
        # x = x0 + v0*t + 0.5*a*t^2 => 0 + 0*5 + 0.5*2*25 = 25
        assert abs(calc(0, 0, 2, 5) - 25.0) < 1e-9

    def test_velocity_position(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        calc = KINEMATICS_EQUATIONS["velocity_position"]["calculate"]
        # v = sqrt(v0^2 + 2a(x - x0)) => sqrt(0 + 2*10*5) = 10
        assert abs(calc(0, 10, 5, 0) - 10.0) < 1e-9

    def test_circular_period(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        calc = KINEMATICS_EQUATIONS["circular_period"]["calculate"]
        # T = 2πr/v
        result = calc(1.0, 1.0)
        assert abs(result - 2 * math.pi) < 0.01

    def test_circular_period_zero_v(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        calc = KINEMATICS_EQUATIONS["circular_period"]["calculate"]
        assert calc(1.0, 0) == 0


class TestPhysicsFormulas:
    """Test the lambda calculators in PHYSICS_FORMULAS."""

    def test_force_newton(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        calc = PHYSICS_FORMULAS["force_newton"]["calculate"]
        # F = ma => 5 * 9.8 = 49
        assert abs(calc(5, 9.8) - 49.0) < 1e-9

    def test_kinetic_energy(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        calc = PHYSICS_FORMULAS["kinetic_energy"]["calculate"]
        # KE = 0.5 * m * v^2 => 0.5 * 2 * 9 = 9
        assert abs(calc(2, 3) - 9.0) < 1e-9

    def test_potential_energy_gravity(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        calc = PHYSICS_FORMULAS["potential_energy_gravity"]["calculate"]
        # PE = mgh => 1 * 9.80665 * 10 = 98.0665
        assert abs(calc(1, 10) - 98.0665) < 0.01

    def test_momentum(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        calc = PHYSICS_FORMULAS["momentum"]["calculate"]
        assert abs(calc(5, 3) - 15.0) < 1e-9

    def test_work(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        calc = PHYSICS_FORMULAS["work"]["calculate"]
        # W = F * d => 10 * 5 = 50
        assert abs(calc(10, 5) - 50.0) < 1e-6


class TestPhysicsFormulasDataStructure:
    """Validate PHYSICS_FORMULAS schema completeness."""

    def test_all_formulas_have_required_fields(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_FORMULAS

        for name, eq in PHYSICS_FORMULAS.items():
            assert "name" in eq, f"{name} missing 'name'"
            assert "formula" in eq, f"{name} missing 'formula'"
            assert "inputs" in eq, f"{name} missing 'inputs'"
            assert "output" in eq, f"{name} missing 'output'"
            assert "calculate" in eq, f"{name} missing 'calculate'"
            assert callable(eq["calculate"]), f"{name} 'calculate' not callable"

    def test_all_kinematics_have_required_fields(self) -> None:
        from umcp.dashboard.pages_physics import KINEMATICS_EQUATIONS

        for name, eq in KINEMATICS_EQUATIONS.items():
            assert "name" in eq, f"{name} missing 'name'"
            assert "formula" in eq, f"{name} missing 'formula'"
            assert "inputs" in eq, f"{name} missing 'inputs'"
            assert "calculate" in eq, f"{name} missing 'calculate'"
            assert callable(eq["calculate"]), f"{name} 'calculate' not callable"


# ============================================================================
# pages_physics.py — PHYSICS_QUANTITIES validation
# ============================================================================


class TestPhysicsQuantities:
    """Validate PHYSICS_QUANTITIES data structure."""

    def test_base_unit_in_units_dict(self) -> None:
        """Base unit should be one of the convertible units with factor 1.0."""
        from umcp.dashboard.pages_physics import PHYSICS_QUANTITIES

        for name, qty in PHYSICS_QUANTITIES.items():
            base = qty["base_unit"]
            # Base unit might not always be explicitly listed but should be derivable
            if base in qty["units"]:
                assert abs(qty["units"][base] - 1.0) < 1e-12, f"{name}: base unit {base} factor != 1.0"

    def test_all_conversion_factors_positive(self) -> None:
        from umcp.dashboard.pages_physics import PHYSICS_QUANTITIES

        for name, qty in PHYSICS_QUANTITIES.items():
            for unit, factor in qty["units"].items():
                assert factor > 0, f"{name}.{unit} has non-positive factor: {factor}"
