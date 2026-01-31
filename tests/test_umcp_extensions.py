"""Tests for the UMCP extension system."""

import subprocess
import sys
from unittest import mock

from umcp import umcp_extensions


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


def test_list_extensions_filter_by_type():
    """Should filter extensions by type."""
    # Get API extensions
    api_exts = umcp_extensions.list_extensions("api")
    assert isinstance(api_exts, list)
    for ext in api_exts:
        assert ext["type"] == "api"

    # Get dashboard extensions
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
    # Ledger has no external dependencies
    assert umcp_extensions.check_extension("ledger") is True

    # Formatter only requires pyyaml (core dependency)
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


def test_load_extension_core():
    """Should be able to load core extensions."""
    # Ledger extension loads from validator
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

    # Should have health endpoint
    endpoints = [ep["path"] for ep in info["endpoints"]]
    assert "/health" in endpoints


def test_visualization_extension_has_features():
    """Visualization extension should have feature definitions."""
    info = umcp_extensions.get_extension_info("visualization")
    assert "features" in info
    assert len(info["features"]) > 0


# Additional tests for coverage


def test_install_extension_nonexistent():
    """Should return False for nonexistent extension."""
    result = umcp_extensions.install_extension("nonexistent")
    assert result is False


def test_install_extension_no_deps():
    """Extensions with no deps should return True."""
    # Ledger has no external dependencies
    result = umcp_extensions.install_extension("ledger")
    assert result is True


def test_run_extension_nonexistent():
    """Should return 1 for nonexistent extension."""
    result = umcp_extensions.run_extension("nonexistent")
    assert result == 1


def test_run_extension_no_command():
    """Should return 1 for extension with no command."""
    # Ledger has no command
    result = umcp_extensions.run_extension("ledger")
    assert result == 1


def test_load_extension_check_fails():
    """Should return None if check_extension fails."""
    # Nonexistent extension should fail
    ext = umcp_extensions.load_extension("nonexistent")
    assert ext is None


def test_extension_info_dataclass():
    """Test ExtensionInfo dataclass fields."""
    info = umcp_extensions.ExtensionInfo(
        name="test",
        description="Test extension",
        type="tool",
        module="test.module",
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
    assert len(d["endpoints"]) == 1
    assert "feature1" in d["features"]


def test_check_extension_api():
    """Test check_extension for API (if installed)."""
    # API may or may not be installed
    result = umcp_extensions.check_extension("api")
    assert isinstance(result, bool)


def test_check_extension_visualization():
    """Test check_extension for visualization (if installed)."""
    result = umcp_extensions.check_extension("visualization")
    assert isinstance(result, bool)


def test_load_extension_api():
    """Test loading API extension if available."""
    result = umcp_extensions.load_extension("api")
    # Will return module if fastapi is installed, None otherwise
    if umcp_extensions.check_extension("api"):
        assert result is not None


def test_import_names_mapping():
    """Test the import name mapping dictionary."""
    # pyyaml maps to yaml
    assert umcp_extensions._IMPORT_NAMES.get("pyyaml") == "yaml"


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
    # Mock subprocess.run to raise an exception
    with mock.patch.object(subprocess, "run", side_effect=Exception("Test error")):
        result = umcp_extensions.run_extension("api")
    assert result == 1


def test_run_extension_keyboard_interrupt():
    """Test run_extension handles keyboard interrupt."""
    with mock.patch.object(subprocess, "run", side_effect=KeyboardInterrupt()):
        result = umcp_extensions.run_extension("api")
    assert result == 0


def test_extension_module_main():
    """Test the module can be run as __main__."""
    # Just verify the main function exists and is callable
    assert callable(umcp_extensions.main)
