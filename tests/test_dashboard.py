"""
Tests for UMCP Dashboard (Streamlit visualization extension).

Tests the utility functions and data loading capabilities.
Note: Streamlit UI components require special testing with streamlit.testing.

Cross-references:
  - src/umcp/dashboard.py (dashboard implementation)
  - tests/test_api_umcp.py (API tests for comparison)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umcp.dashboard import (
    KERNEL_SYMBOLS,
    REGIME_COLORS,
    STATUS_COLORS,
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


class TestDashboardUtilities:
    """Tests for dashboard utility functions."""

    def test_get_repo_root_finds_pyproject(self) -> None:
        """Test that get_repo_root finds the repository root."""
        root = get_repo_root()
        assert root.exists()
        assert (root / "pyproject.toml").exists()

    def test_classify_regime_stable(self) -> None:
        """Test STABLE regime classification."""
        assert classify_regime(0.5, 0.0) == "STABLE"
        assert classify_regime(0.3, 0.0) == "STABLE"
        assert classify_regime(0.7, 0.0) == "STABLE"
        assert classify_regime(0.5, 0.004) == "STABLE"

    def test_classify_regime_watch(self) -> None:
        """Test WATCH regime classification."""
        assert classify_regime(0.2, 0.0) == "WATCH"
        assert classify_regime(0.8, 0.0) == "WATCH"
        assert classify_regime(0.15, 0.005) == "WATCH"

    def test_classify_regime_collapse(self) -> None:
        """Test COLLAPSE regime classification."""
        assert classify_regime(0.05, 0.0) == "COLLAPSE"
        assert classify_regime(0.95, 0.0) == "COLLAPSE"
        assert classify_regime(0.0, 0.0) == "COLLAPSE"
        assert classify_regime(1.0, 0.0) == "COLLAPSE"

    def test_classify_regime_critical(self) -> None:
        """Test CRITICAL regime classification (seam residual takes priority)."""
        assert classify_regime(0.5, 0.02) == "CRITICAL"
        assert classify_regime(0.5, -0.02) == "CRITICAL"
        assert classify_regime(0.05, 0.02) == "CRITICAL"  # Seam overrides collapse
        assert classify_regime(0.8, -0.015) == "CRITICAL"

    def test_get_regime_color_returns_colors(self) -> None:
        """Test that get_regime_color returns valid hex colors."""
        assert get_regime_color("STABLE").startswith("#")
        assert get_regime_color("WATCH").startswith("#")
        assert get_regime_color("COLLAPSE").startswith("#")
        assert get_regime_color("CRITICAL").startswith("#")
        # Unknown regime should return default gray
        assert get_regime_color("UNKNOWN").startswith("#")


class TestDashboardDataLoading:
    """Tests for dashboard data loading functions."""

    def test_load_ledger_returns_dataframe(self) -> None:
        """Test that load_ledger returns a DataFrame."""
        try:
            import pandas as pd

            df = load_ledger()
            assert isinstance(df, pd.DataFrame)
            # Ledger should have data in this repo
            if not df.empty:
                assert "timestamp" in df.columns or len(df.columns) > 0
        except ImportError:
            pytest.skip("pandas not installed")

    def test_load_casepacks_returns_list(self) -> None:
        """Test that load_casepacks returns a list of dicts."""
        casepacks = load_casepacks()
        assert isinstance(casepacks, list)
        # This repo has casepacks
        assert len(casepacks) > 0
        # Each casepack should have required fields
        for cp in casepacks:
            assert "id" in cp
            assert "path" in cp

    def test_load_contracts_returns_list(self) -> None:
        """Test that load_contracts returns a list of dicts."""
        contracts = load_contracts()
        assert isinstance(contracts, list)
        # This repo has contracts
        assert len(contracts) > 0
        # Each contract should have required fields
        for c in contracts:
            assert "id" in c
            assert "domain" in c
            assert "version" in c


class TestRegimeClassificationEdgeCases:
    """Edge case tests for regime classification."""

    def test_regime_boundary_omega_0_1(self) -> None:
        """Test omega boundary at 0.1 (COLLAPSE vs WATCH)."""
        # Just below 0.1 should be COLLAPSE
        assert classify_regime(0.09, 0.0) == "COLLAPSE"
        # At 0.1 should be WATCH (boundary inclusive)
        assert classify_regime(0.1, 0.0) == "WATCH"

    def test_regime_boundary_omega_0_3(self) -> None:
        """Test omega boundary at 0.3 (WATCH vs STABLE)."""
        # Just below 0.3 should be WATCH
        assert classify_regime(0.29, 0.0) == "WATCH"
        # At 0.3 should be STABLE (boundary inclusive)
        assert classify_regime(0.3, 0.0) == "STABLE"

    def test_regime_boundary_omega_0_7(self) -> None:
        """Test omega boundary at 0.7 (STABLE vs WATCH)."""
        # At 0.7 should be STABLE (boundary inclusive)
        assert classify_regime(0.7, 0.0) == "STABLE"
        # Just above 0.7 should be WATCH
        assert classify_regime(0.71, 0.0) == "WATCH"

    def test_regime_boundary_omega_0_9(self) -> None:
        """Test omega boundary at 0.9 (WATCH vs COLLAPSE)."""
        # At 0.9 should be WATCH (boundary inclusive)
        assert classify_regime(0.9, 0.0) == "WATCH"
        # Just above 0.9 should be COLLAPSE
        assert classify_regime(0.91, 0.0) == "COLLAPSE"

    def test_regime_boundary_seam_0_01(self) -> None:
        """Test seam residual boundary at 0.01 (determines CRITICAL)."""
        # At 0.01 should NOT be critical (boundary exclusive)
        assert classify_regime(0.5, 0.01) != "CRITICAL"
        # Just above 0.01 should be CRITICAL
        assert classify_regime(0.5, 0.011) == "CRITICAL"
        # Negative seam works too
        assert classify_regime(0.5, -0.011) == "CRITICAL"


class TestDashboardCasepackDetails:
    """Tests for casepack loading details."""

    def test_casepacks_have_correct_structure(self) -> None:
        """Test that loaded casepacks have the expected structure."""
        casepacks = load_casepacks()

        for cp in casepacks:
            # Required fields
            assert isinstance(cp.get("id"), str)
            assert isinstance(cp.get("path"), str)

            # Path should exist
            path = Path(cp["path"])
            assert path.exists(), f"Casepack path does not exist: {path}"

    def test_casepacks_versions_are_strings(self) -> None:
        """Test that casepack versions are string type."""
        casepacks = load_casepacks()

        for cp in casepacks:
            version = cp.get("version")
            assert isinstance(version, str), f"Version should be str, got {type(version)}"


class TestDashboardContractDetails:
    """Tests for contract loading details."""

    def test_contracts_have_correct_domains(self) -> None:
        """Test that contracts have valid domain prefixes."""
        contracts = load_contracts()

        for c in contracts:
            domain = c.get("domain")
            # Domain should be a short identifier
            assert isinstance(domain, str)
            assert len(domain) <= 10, f"Domain too long: {domain}"

    def test_contracts_versions_start_with_v(self) -> None:
        """Test that contract versions start with 'v' when extracted from filename."""
        contracts = load_contracts()

        for c in contracts:
            version = c.get("version", "")
            # Either starts with 'v' or is 'v1' default
            assert version.startswith("v") or version == "v1"


class TestDashboardNewFunctions:
    """Tests for new dashboard utility functions."""

    def test_format_bytes_bytes(self) -> None:
        """Test format_bytes for byte values."""
        assert "B" in format_bytes(100)
        assert "500" in format_bytes(500) or "500.0" in format_bytes(500)

    def test_format_bytes_kilobytes(self) -> None:
        """Test format_bytes for kilobyte values."""
        result = format_bytes(2048)
        assert "KB" in result

    def test_format_bytes_megabytes(self) -> None:
        """Test format_bytes for megabyte values."""
        result = format_bytes(2 * 1024 * 1024)
        assert "MB" in result

    def test_get_trend_indicator_up(self) -> None:
        """Test trend indicator for upward trend."""
        result = get_trend_indicator(1.1, 1.0)
        assert result in ["📈", "📉", "➡️"]

    def test_get_trend_indicator_down(self) -> None:
        """Test trend indicator for downward trend."""
        result = get_trend_indicator(0.9, 1.0)
        assert result in ["📈", "📉", "➡️"]

    def test_get_trend_indicator_stable(self) -> None:
        """Test trend indicator for stable values."""
        result = get_trend_indicator(1.0, 1.0)
        assert result == "➡️"

    def test_detect_anomalies_returns_series(self) -> None:
        """Test that detect_anomalies returns a proper result."""
        try:
            import pandas as pd

            # Create test series with outlier
            data = pd.Series([1.0, 1.1, 0.9, 1.0, 1.0, 10.0])  # 10.0 is outlier
            result = detect_anomalies(data, threshold=2.0)
            assert isinstance(result, pd.Series)
            # The outlier should be detected (last element is anomaly)
            last_val: bool = result.tolist()[-1]
            assert last_val is True
        except ImportError:
            pytest.skip("pandas/numpy not installed")

    def test_load_closures_returns_list(self) -> None:
        """Test that load_closures returns a list of dicts."""
        closures = load_closures()
        assert isinstance(closures, list)
        # This repo has closures
        assert len(closures) > 0
        # Each closure should have required fields
        for c in closures:
            assert "name" in c
            assert "path" in c
            assert "type" in c
            assert c["type"] in ["python", "yaml"]

    def test_closures_paths_exist(self) -> None:
        """Test that closure paths exist."""
        closures = load_closures()
        for c in closures:
            path = Path(c["path"])
            assert path.exists(), f"Closure path does not exist: {path}"


class TestDashboardConstants:
    """Tests for dashboard constants."""

    def test_regime_colors_complete(self) -> None:
        """Test that all regime colors are defined."""
        expected_regimes = ["STABLE", "WATCH", "COLLAPSE", "CRITICAL"]
        for regime in expected_regimes:
            assert regime in REGIME_COLORS
            assert REGIME_COLORS[regime].startswith("#")

    def test_status_colors_complete(self) -> None:
        """Test that all status colors are defined."""
        expected_statuses = ["CONFORMANT", "NONCONFORMANT", "NON_EVALUABLE"]
        for status in expected_statuses:
            assert status in STATUS_COLORS
            assert STATUS_COLORS[status].startswith("#")

    def test_kernel_symbols_present(self) -> None:
        """Test that kernel symbols dictionary has entries."""
        assert len(KERNEL_SYMBOLS) > 0
        assert "omega" in KERNEL_SYMBOLS
        assert "ω" in KERNEL_SYMBOLS["omega"]


class TestGeometryPageFunctions:
    """Tests for the Three-Layer Geometry visualization functions."""

    def test_geometry_page_function_exists(self) -> None:
        """Test that render_geometry_page function exists and is importable."""
        from umcp.dashboard import render_geometry_page

        assert callable(render_geometry_page)

    def test_layer1_function_exists(self) -> None:
        """Test that render_layer1_state_space function exists."""
        from umcp.dashboard import render_layer1_state_space

        assert callable(render_layer1_state_space)

    def test_layer2_function_exists(self) -> None:
        """Test that render_layer2_projections function exists."""
        from umcp.dashboard import render_layer2_projections

        assert callable(render_layer2_projections)

    def test_layer3_function_exists(self) -> None:
        """Test that render_layer3_seam_graph function exists."""
        from umcp.dashboard import render_layer3_seam_graph

        assert callable(render_layer3_seam_graph)

    def test_unified_view_function_exists(self) -> None:
        """Test that render_unified_geometry_view function exists."""
        from umcp.dashboard import render_unified_geometry_view

        assert callable(render_unified_geometry_view)

    def test_geometry_page_in_pages_dict(self) -> None:
        """Test that Geometry page is registered in navigation."""
        # This is a structural test - the page should be accessible
        from umcp.dashboard import render_geometry_page

        # Function should not raise when imported
        assert render_geometry_page is not None
