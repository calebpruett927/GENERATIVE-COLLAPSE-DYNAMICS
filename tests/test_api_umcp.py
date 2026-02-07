"""
Tests for UMCP REST API.

Tests the FastAPI endpoints and utility functions.
Uses TestClient for endpoint testing.
"""

import pytest

try:
    from fastapi.testclient import TestClient

    from umcp import api_umcp

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


# ============================================================================
# Measurement Conversion Endpoint Tests
# ============================================================================


def test_convert_measurements_zscore(client, auth_headers):
    """Test z-score normalization conversion."""
    response = client.post(
        "/convert/measurements",
        json={"values": [1.0, 2.0, 3.0, 4.0, 5.0], "source_unit": "raw", "target_unit": "normalized"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["converted"]) == 5
    # Mean should be ~0 for z-scores
    import numpy as np

    assert abs(np.mean(data["converted"])) < 1e-10


def test_convert_measurements_scaling(client, auth_headers):
    """Test 0-1 scaling conversion."""
    response = client.post(
        "/convert/measurements",
        json={"values": [0.0, 50.0, 100.0], "source_unit": "raw", "target_unit": "scaled"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["converted"] == [0.0, 0.5, 1.0]


def test_convert_measurements_percentage(client, auth_headers):
    """Test percentage to fraction conversion."""
    response = client.post(
        "/convert/measurements",
        json={"values": [25.0, 50.0, 100.0], "source_unit": "percentage", "target_unit": "fraction"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["converted"] == [0.25, 0.5, 1.0]
    assert data["conversion_factor"] == 0.01


def test_embed_coordinates_minmax(client, auth_headers):
    """Test minmax coordinate embedding."""
    response = client.post(
        "/convert/embed",
        json={"values": [0.0, 0.5, 1.0], "method": "minmax", "epsilon": 1e-6},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["embedded"]) == 3
    # Check ε-clipping was applied
    assert all(v >= data["epsilon"] for v in data["embedded"])
    assert all(v <= 1 - data["epsilon"] for v in data["embedded"])


def test_embed_coordinates_sigmoid(client, auth_headers):
    """Test sigmoid coordinate embedding."""
    response = client.post(
        "/convert/embed",
        json={"values": [-2.0, 0.0, 2.0], "method": "sigmoid", "epsilon": 1e-8},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["embedded"]) == 3
    # Sigmoid of 0 should be 0.5
    assert abs(data["embedded"][1] - 0.5) < 0.01


# ============================================================================
# Kernel Computation Endpoint Tests
# ============================================================================


def test_kernel_compute_uniform(client, auth_headers):
    """Test kernel computation with uniform coordinates."""
    response = client.post(
        "/kernel/compute",
        json={"coordinates": [0.5, 0.5, 0.5, 0.5], "epsilon": 1e-8},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["F"] == 0.5
    assert data["omega"] == 0.5
    assert data["is_homogeneous"] is True
    assert data["C"] == 0.0  # No curvature for homogeneous


def test_kernel_compute_heterogeneous(client, auth_headers):
    """Test kernel computation with heterogeneous coordinates."""
    response = client.post(
        "/kernel/compute",
        json={"coordinates": [0.9, 0.8, 0.7, 0.6], "weights": [0.25, 0.25, 0.25, 0.25], "epsilon": 1e-8},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert abs(data["F"] - 0.75) < 1e-10  # Mean of [0.9, 0.8, 0.7, 0.6]
    assert abs(data["omega"] - 0.25) < 1e-10
    assert data["is_homogeneous"] is False
    assert data["amgm_gap"] >= 0  # AM-GM inequality


def test_kernel_budget_pass(client, auth_headers):
    """Test budget identity verification - passing case."""
    response = client.post(
        "/kernel/budget",
        params={"R": 1.0, "tau_R": 0.5, "D_omega": 0.2, "D_C": 0.1, "delta_kappa": 0.2},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["lhs"] == 0.5
    assert data["rhs"] == 0.5
    assert data["seam_pass"] is True


def test_kernel_budget_fail(client, auth_headers):
    """Test budget identity verification - failing case."""
    response = client.post(
        "/kernel/budget",
        params={"R": 1.0, "tau_R": 1.0, "D_omega": 0.1, "D_C": 0.1, "delta_kappa": 0.1},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["seam_residual"] > 0.005
    assert data["seam_pass"] is False


# ============================================================================
# Uncertainty Propagation Endpoint Tests
# ============================================================================


def test_uncertainty_propagate(client, auth_headers):
    """Test uncertainty propagation through kernel."""
    response = client.post(
        "/uncertainty/propagate",
        json={
            "coordinates": [0.9, 0.8, 0.7, 0.6],
            "coordinate_variances": [0.01, 0.01, 0.01, 0.01],
            "epsilon": 1e-8,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    # Should have kernel outputs
    assert "F" in data
    assert "omega" in data
    # Should have standard deviations
    assert "std_F" in data
    assert data["std_F"] >= 0
    # Should have confidence intervals
    assert "ci_F" in data
    assert len(data["ci_F"]) == 2


# ============================================================================
# Time Series Analysis Endpoint Tests
# ============================================================================


def test_timeseries_analysis(client, auth_headers):
    """Test time series analysis."""
    response = client.post(
        "/analysis/timeseries",
        json={
            "omega_series": [0.5, 0.55, 0.52, 0.48, 0.51, 0.53, 0.49, 0.50],
            "timestamps": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
                "2026-01-01T04:00:00Z",
                "2026-01-01T05:00:00Z",
                "2026-01-01T06:00:00Z",
                "2026-01-01T07:00:00Z",
            ],
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_points"] == 8
    assert "omega_mean" in data
    assert "regime_counts" in data
    assert "stability_score" in data
    # All values in STABLE regime
    assert data["regime_counts"]["STABLE"] == 8


def test_timeseries_regime_transitions(client, auth_headers):
    """Test time series analysis with regime transitions."""
    response = client.post(
        "/analysis/timeseries",
        json={
            "omega_series": [0.5, 0.6, 0.75, 0.85, 0.5],  # STABLE -> WATCH -> WATCH -> STABLE
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["regime_transitions"] >= 2


# ============================================================================
# Data Analysis Endpoint Tests
# ============================================================================


def test_statistics_endpoint(client, auth_headers):
    """Test descriptive statistics computation."""
    response = client.post(
        "/analysis/statistics",
        json={"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n"] == 10
    assert data["mean"] == 5.5
    assert data["min"] == 1.0
    assert data["max"] == 10.0
    assert data["median"] == 5.5


def test_correlation_endpoint(client, auth_headers):
    """Test correlation analysis."""
    response = client.post(
        "/analysis/correlation",
        json={"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n"] == 5
    # Perfect positive correlation
    assert abs(data["pearson_r"] - 1.0) < 0.001
    assert data["r_squared"] > 0.99


def test_ledger_analysis_endpoint(client, auth_headers):
    """Test ledger analysis."""
    response = client.get("/analysis/ledger", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_entries" in data
    assert "conformant_rate" in data
    assert "regime_distribution" in data
    assert "stability_index" in data
