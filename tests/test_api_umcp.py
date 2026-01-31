"""
Tests for UMCP REST API.

Tests the FastAPI endpoints and utility functions.
Uses TestClient for endpoint testing.
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from src.umcp import api_umcp

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    api_umcp = None  # type: ignore
    TestClient = None  # type: ignore

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi not installed (optional dependency)")


# ============================================================================
# Utility Function Tests
# ============================================================================


def test_verify_api_key():
    """Test verify_api_key returns True for valid key, False for invalid."""
    # Default dev key
    assert api_umcp.verify_api_key("umcp-dev-key") is True
    # Invalid keys
    assert api_umcp.verify_api_key("wrong_key") is False
    assert api_umcp.verify_api_key(None) is False


def test_get_repo_root():
    """Test get_repo_root returns a valid Path."""
    root = api_umcp.get_repo_root()
    assert hasattr(root, "exists")
    assert root.is_dir()
    # Should find pyproject.toml
    assert (root / "pyproject.toml").exists()


def test_classify_regime_stable():
    """Test classify_regime returns STABLE for middle omega values."""
    # omega in [0.3, 0.7] with small seam = STABLE
    assert api_umcp.classify_regime(0.5, 0.5, 0.001, 1.0) == "STABLE"
    assert api_umcp.classify_regime(0.3, 0.7, 0.0, 1.0) == "STABLE"
    assert api_umcp.classify_regime(0.7, 0.3, 0.005, 1.0) == "STABLE"


def test_classify_regime_watch():
    """Test classify_regime returns WATCH for edge omega values."""
    # omega in [0.1, 0.3) or (0.7, 0.9] = WATCH
    assert api_umcp.classify_regime(0.2, 0.8, 0.001, 1.0) == "WATCH"
    assert api_umcp.classify_regime(0.8, 0.2, 0.001, 1.0) == "WATCH"


def test_classify_regime_collapse():
    """Test classify_regime returns COLLAPSE for extreme omega values."""
    # omega < 0.1 or omega > 0.9 = COLLAPSE
    assert api_umcp.classify_regime(0.05, 0.95, 0.001, 1.0) == "COLLAPSE"
    assert api_umcp.classify_regime(0.95, 0.05, 0.001, 1.0) == "COLLAPSE"


def test_classify_regime_critical():
    """Test classify_regime returns CRITICAL for large seam residuals."""
    # |S| > 0.01 = CRITICAL (overrides omega-based)
    assert api_umcp.classify_regime(0.5, 0.5, 0.02, 1.0) == "CRITICAL"
    assert api_umcp.classify_regime(0.5, 0.5, -0.02, 1.0) == "CRITICAL"


def test_get_current_time():
    """Test get_current_time returns ISO format timestamp."""
    result = api_umcp.get_current_time()
    assert isinstance(result, str)
    assert "T" in result  # ISO format has T separator
    assert ":" in result  # Has time component


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(api_umcp.app)


@pytest.fixture
def auth_headers():
    """Authentication headers with valid API key."""
    return {"X-API-Key": "umcp-dev-key"}


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    """Test health endpoint returns status without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy", "degraded"]
    assert "timestamp" in data
    assert "checks" in data


def test_version_endpoint(client):
    """Test version endpoint returns version info."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data
    assert "validator_version" in data
    assert "python_version" in data


def test_casepacks_endpoint_requires_auth(client):
    """Test casepacks endpoint requires authentication."""
    response = client.get("/casepacks")
    assert response.status_code == 401


def test_casepacks_endpoint_with_auth(client, auth_headers):
    """Test casepacks endpoint with valid auth."""
    response = client.get("/casepacks", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should have at least hello_world casepack
    if data:
        assert "id" in data[0]
        assert "version" in data[0]


def test_contracts_endpoint(client, auth_headers):
    """Test contracts endpoint returns contract list."""
    response = client.get("/contracts", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_closures_endpoint(client, auth_headers):
    """Test closures endpoint returns closure list."""
    response = client.get("/closures", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_ledger_endpoint(client, auth_headers):
    """Test ledger endpoint returns ledger entries."""
    response = client.get("/ledger", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_entries" in data
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_casepack_not_found(client, auth_headers):
    """Test casepack detail returns 404 for nonexistent casepack."""
    response = client.get("/casepacks/nonexistent_casepack", headers=auth_headers)
    assert response.status_code == 404


def test_invalid_api_key(client):
    """Test invalid API key returns 401."""
    headers = {"X-API-Key": "invalid-key"}
    response = client.get("/casepacks", headers=headers)
    assert response.status_code == 401
