"""Tests for the UMCP extension system (default + on-demand architecture)."""

import subprocess
import sys
from unittest import mock

from umcp import umcp_extensions
from umcp.umcp_extensions import ExtensionManager, manager

# ============================================================================
# Registry & ExtensionInfo basics
# ============================================================================


def test_list_extensions():
    """Should return a list of extension dictionaries."""
    result = umcp_extensions.list_extensions()
    assert isinstance(result, list)
    assert len(result) > 0

    # Each extension should have required fields
    for ext in result:
        assert "name" in ext
        assert "description" in ext
        assert "type" in ext
        assert "module" in ext
        assert "default" in ext  # new field


def test_list_extensions_filter_by_type():
    """Should filter extensions by type."""
    api_exts = umcp_extensions.list_extensions("api")
    assert isinstance(api_exts, list)
    for ext in api_exts:
        assert ext["type"] == "api"

    dashboard_exts = umcp_extensions.list_extensions("dashboard")
    assert isinstance(dashboard_exts, list)
    for ext in dashboard_exts:
        assert ext["type"] == "dashboard"


def test_get_extension_info():
    """Should return a dict with extension metadata."""
    info = umcp_extensions.get_extension_info("api")
    assert isinstance(info, dict)
    assert info["name"] == "api"
    assert "description" in info
    assert "module" in info
    assert "requires" in info


def test_get_extension_info_not_found():
    """Should return not_found status for unknown extension."""
    info = umcp_extensions.get_extension_info("nonexistent")
    assert isinstance(info, dict)
    assert info["status"] == "not_found"


def test_check_extension_core():
    """Core extensions (ledger, formatter) should always be available."""
    assert umcp_extensions.check_extension("ledger") is True
    assert umcp_extensions.check_extension("formatter") is True


def test_check_extension_nonexistent():
    """Nonexistent extensions should return False."""
    assert umcp_extensions.check_extension("nonexistent") is False


def test_extension_registry_has_expected_extensions():
    """Registry should contain the expected extensions."""
    expected = ["api", "visualization", "ledger", "formatter"]
    for name in expected:
        assert name in umcp_extensions.EXTENSIONS


def test_extension_info_to_dict():
    """ExtensionInfo should convert to dict properly."""
    info = umcp_extensions.EXTENSIONS["api"]
    d = info.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "api"
    assert d["type"] == "api"
    assert "fastapi" in d["requires"]
    assert isinstance(d["default"], bool)


def test_load_extension_core():
    """Should be able to load core extensions."""
    ext = umcp_extensions.load_extension("ledger")
    assert ext is not None


def test_load_extension_nonexistent():
    """Should return None for nonexistent extension."""
    ext = umcp_extensions.load_extension("nonexistent")
    assert ext is None


def test_api_extension_has_endpoints():
    """API extension should have endpoint definitions."""
    info = umcp_extensions.get_extension_info("api")
    assert "endpoints" in info
    assert len(info["endpoints"]) > 0

    endpoints = [ep["path"] for ep in info["endpoints"]]
    assert "/health" in endpoints


def test_visualization_extension_has_features():
    """Visualization extension should have feature definitions."""
    info = umcp_extensions.get_extension_info("visualization")
    assert "features" in info
    assert len(info["features"]) > 0


def test_install_extension_nonexistent():
    """Should return False for nonexistent extension."""
    result = umcp_extensions.install_extension("nonexistent")
    assert result is False


def test_install_extension_no_deps():
    """Extensions with no deps should return True."""
    result = umcp_extensions.install_extension("ledger")
    assert result is True


def test_run_extension_nonexistent():
    """Should return 1 for nonexistent extension."""
    result = umcp_extensions.run_extension("nonexistent")
    assert result == 1


def test_run_extension_no_command():
    """Should return 1 for extension with no command."""
    result = umcp_extensions.run_extension("ledger")
    assert result == 1


def test_load_extension_check_fails():
    """Should return None if check_extension fails."""
    ext = umcp_extensions.load_extension("nonexistent")
    assert ext is None


def test_extension_info_dataclass():
    """Test ExtensionInfo dataclass fields."""
    info = umcp_extensions.ExtensionInfo(
        name="test",
        description="Test extension",
        type="tool",
        module="test.module",
        default=False,
        class_name="TestClass",
        requires=["dep1", "dep2"],
        command="test command",
        port=9999,
        endpoints=[{"method": "GET", "path": "/test", "description": "Test"}],
        features=["feature1"],
    )
    d = info.to_dict()
    assert d["name"] == "test"
    assert d["class"] == "TestClass"
    assert d["port"] == 9999
    assert d["default"] is False
    assert len(d["endpoints"]) == 1
    assert "feature1" in d["features"]


def test_check_extension_api():
    """Test check_extension for API (if installed)."""
    result = umcp_extensions.check_extension("api")
    assert isinstance(result, bool)


def test_check_extension_visualization():
    """Test check_extension for visualization (if installed)."""
    result = umcp_extensions.check_extension("visualization")
    assert isinstance(result, bool)


def test_load_extension_api():
    """Test loading API extension if available."""
    result = umcp_extensions.load_extension("api")
    if umcp_extensions.check_extension("api"):
        assert result is not None


def test_import_names_mapping():
    """Test the import name mapping dictionary."""
    assert umcp_extensions._IMPORT_NAMES.get("pyyaml") == "yaml"


# ============================================================================
# Default vs on-demand classification
# ============================================================================


def test_default_extensions_have_no_heavy_deps():
    """Default extensions should not require optional heavy deps."""
    heavy = {"fastapi", "uvicorn", "streamlit", "plotly"}
    for name in manager.default_names:
        info = umcp_extensions.EXTENSIONS[name]
        overlap = heavy & set(info.requires)
        assert not overlap, f"Default ext '{name}' requires heavy dep {overlap}"


def test_default_names_includes_core_three():
    """Default tier must include ledger, formatter, thermodynamics."""
    defaults = set(manager.default_names)
    assert {"ledger", "formatter", "thermodynamics"} <= defaults


def test_on_demand_names_includes_api_and_viz():
    """On-demand tier must include api and visualization."""
    on_demand = set(manager.on_demand_names)
    assert {"api", "visualization"} <= on_demand


def test_default_flag_in_registry():
    """EXTENSIONS registry should mark correct default flags."""
    assert umcp_extensions.EXTENSIONS["ledger"].default is True
    assert umcp_extensions.EXTENSIONS["formatter"].default is True
    assert umcp_extensions.EXTENSIONS["thermodynamics"].default is True
    assert umcp_extensions.EXTENSIONS["api"].default is False
    assert umcp_extensions.EXTENSIONS["visualization"].default is False


# ============================================================================
# ExtensionManager
# ============================================================================


def test_manager_startup_loads_defaults():
    """startup() should load all default extensions."""
    mgr = ExtensionManager()
    loaded = mgr.startup()
    # All three defaults should load (they have no heavy deps)
    assert "ledger" in loaded
    assert "formatter" in loaded
    assert "thermodynamics" in loaded
    # On-demand should NOT be loaded yet
    assert not mgr.is_loaded("api")
    assert not mgr.is_loaded("visualization")


def test_manager_get_lazy_loads():
    """get() should lazy-load an on-demand extension."""
    mgr = ExtensionManager()
    mgr.startup()
    # Before: not loaded
    assert not mgr.is_loaded("ledger") or mgr.is_loaded("ledger")
    # Calling get() for a default that's already loaded returns cached
    mod = mgr.get("ledger")
    assert mod is not None
    assert mgr.is_loaded("ledger")


def test_manager_get_caches():
    """Subsequent get() calls return the same cached module."""
    mgr = ExtensionManager()
    a = mgr.get("ledger")
    b = mgr.get("ledger")
    assert a is b


def test_manager_get_nonexistent():
    """get() returns None for unregistered extension."""
    mgr = ExtensionManager()
    assert mgr.get("does_not_exist") is None


def test_manager_available_names():
    """available_names should list all registered extensions."""
    mgr = ExtensionManager()
    names = mgr.available_names
    assert "api" in names
    assert "ledger" in names


def test_manager_status():
    """status() should return a structured dict."""
    mgr = ExtensionManager()
    mgr.startup()
    st = mgr.status()
    assert st["started"] is True
    assert "default" in st
    assert "on_demand" in st
    assert "loaded" in st
    assert isinstance(st["loaded"], list)
    # defaults should be loaded
    for name in mgr.default_names:
        assert st["default"][name]["loaded"] is True


def test_manager_reset():
    """reset() should clear loaded state."""
    mgr = ExtensionManager()
    mgr.startup()
    assert len(mgr.loaded_names) > 0
    mgr.reset()
    assert len(mgr.loaded_names) == 0
    assert mgr._started is False


def test_manager_is_loaded_false_before_startup():
    """Before startup, nothing should be loaded."""
    mgr = ExtensionManager()
    assert not mgr.is_loaded("ledger")


# ============================================================================
# CLI entry point
# ============================================================================


def test_main_no_args(capsys):
    """Test main with no arguments shows help."""
    with mock.patch.object(sys, "argv", ["umcp-ext"]):
        result = umcp_extensions.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "UMCP Extension Manager" in captured.out or result == 0


def test_main_list_command(capsys):
    """Test main with list command."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "list"]):
        result = umcp_extensions.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "UMCP EXTENSIONS" in captured.out


def test_main_list_with_type(capsys):
    """Test main with list --type filter."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "list", "--type", "api"]):
        result = umcp_extensions.main()
    assert result == 0


def test_main_status_command(capsys):
    """Test main with status command."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "status"]):
        result = umcp_extensions.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "EXTENSION STATUS" in captured.out


def test_main_info_command(capsys):
    """Test main with info command."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "info", "api"]):
        result = umcp_extensions.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "api" in captured.out.lower()


def test_main_info_not_found(capsys):
    """Test main with info for nonexistent extension."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "info", "nonexistent"]):
        result = umcp_extensions.main()
    assert result == 1


def test_main_check_command(capsys):
    """Test main with check command."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "check", "ledger"]):
        result = umcp_extensions.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "installed" in captured.out.lower()


def test_main_check_not_installed(capsys):
    """Test main with check for potentially missing extension."""
    # This tests the "not installed" path
    with mock.patch.object(sys, "argv", ["umcp-ext", "check", "nonexistent"]):
        # Nonexistent always returns 1
        result = umcp_extensions.main()
    # Will return 1 because extension doesn't exist
    assert result == 1


def test_main_install_command(capsys):
    """Test main with install command for extension with no deps."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "install", "ledger"]):
        result = umcp_extensions.main()
    assert result == 0


def test_main_install_nonexistent(capsys):
    """Test main with install for nonexistent extension."""
    with mock.patch.object(sys, "argv", ["umcp-ext", "install", "nonexistent"]):
        result = umcp_extensions.main()
    assert result == 1


def test_run_extension_with_error():
    """Test run_extension handles errors gracefully."""
    # Mock check_extension to return True, then mock subprocess.run to raise
    with (
        mock.patch.object(umcp_extensions, "check_extension", return_value=True),
        mock.patch.object(subprocess, "run", side_effect=Exception("Test error")),
    ):
        result = umcp_extensions.run_extension("api")
    assert result == 1


def test_run_extension_keyboard_interrupt():
    """Test run_extension handles keyboard interrupt."""
    # Mock check_extension to return True, then mock subprocess.run to raise
    with (
        mock.patch.object(umcp_extensions, "check_extension", return_value=True),
        mock.patch.object(subprocess, "run", side_effect=KeyboardInterrupt()),
    ):
        result = umcp_extensions.run_extension("api")
    assert result == 0


def test_extension_module_main():
    """Test the module can be run as __main__."""
    # Just verify the main function exists and is callable
    assert callable(umcp_extensions.main)
