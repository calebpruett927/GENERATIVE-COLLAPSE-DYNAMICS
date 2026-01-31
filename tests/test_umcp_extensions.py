"""Tests for the UMCP extension system."""

import pytest

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
