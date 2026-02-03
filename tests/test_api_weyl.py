"""
Tests for WEYL API endpoints.

Tests cover:
- /weyl/background: Background cosmology computation
- /weyl/sigma: Σ(z) modified gravity parameter
- /weyl/des-y3: DES Y3 reference data
- /weyl/umcp-mapping: WEYL to UMCP invariant mapping
"""

import pytest

# Skip if API dependencies not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.umcp.api_umcp import app


@pytest.fixture
def client():
    """Test client without API key."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers with valid API key."""
    return {"X-API-Key": "umcp-dev-key"}


class TestWeylBackgroundEndpoint:
    """Test /weyl/background endpoint."""

    def test_background_at_z0(self, client, auth_headers):
        """Test background cosmology at z=0."""
        response = client.get("/weyl/background", params={"z": 0.0}, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["z"] == 0.0
        assert data["H_z"] == pytest.approx(67.4, rel=0.01)  # H_0
        assert data["chi"] == pytest.approx(0.0, abs=1.0)  # χ(0) ≈ 0
        assert data["D1"] == pytest.approx(1.0, rel=0.01)  # D₁(0) = 1
        assert data["sigma8_z"] == pytest.approx(0.811, rel=0.01)  # σ8,0

    def test_background_at_z1(self, client, auth_headers):
        """Test background cosmology at z=1."""
        response = client.get("/weyl/background", params={"z": 1.0}, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["z"] == 1.0
        assert data["H_z"] > 67.4  # H(z>0) > H_0
        assert data["chi"] > 0  # χ(z>0) > 0
        assert 0 < data["D1"] < 1  # 0 < D₁(z>0) < 1
        assert data["sigma8_z"] < 0.811  # σ8(z>0) < σ8,0

    def test_background_monotonic(self, client, auth_headers):
        """Test H(z) and χ(z) are monotonic."""
        z_values = [0.0, 0.5, 1.0, 2.0]
        results = []

        for z in z_values:
            response = client.get("/weyl/background", params={"z": z}, headers=auth_headers)
            assert response.status_code == 200
            results.append(response.json())

        # H(z) increases with z
        H_values = [r["H_z"] for r in results]
        assert H_values == sorted(H_values)

        # χ(z) increases with z
        chi_values = [r["chi"] for r in results]
        assert chi_values == sorted(chi_values)

    def test_background_invalid_z(self, client, auth_headers):
        """Test invalid redshift handling."""
        response = client.get("/weyl/background", params={"z": -1.0}, headers=auth_headers)
        assert response.status_code == 422  # Validation error

        response = client.get("/weyl/background", params={"z": 100.0}, headers=auth_headers)
        assert response.status_code == 422


class TestWeylSigmaEndpoint:
    """Test /weyl/sigma endpoint."""

    def test_sigma_gr_case(self, client, auth_headers):
        """Test Σ₀=0 gives Σ=1 (GR)."""
        response = client.get(
            "/weyl/sigma", params={"z": 0.5, "Sigma_0": 0.0, "g_model": "constant"}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert data["Sigma"] == pytest.approx(1.0, rel=1e-6)
        assert data["regime"] == "GR_consistent"
        assert data["deviation_from_GR"] == pytest.approx(0.0, abs=1e-6)

    def test_sigma_des_y3_value(self, client, auth_headers):
        """Test DES Y3 Σ₀ = 0.24 case."""
        response = client.get(
            "/weyl/sigma", params={"z": 0.5, "Sigma_0": 0.24, "g_model": "constant"}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert data["Sigma"] == pytest.approx(1.24, rel=0.01)
        assert data["regime"] == "Tension"  # 0.1 ≤ |Σ₀| < 0.3

    def test_sigma_modified_gravity_regime(self, client, auth_headers):
        """Test large Σ₀ gives modified gravity regime."""
        response = client.get(
            "/weyl/sigma", params={"z": 0.5, "Sigma_0": 0.4, "g_model": "constant"}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert data["regime"] == "Modified_gravity"

    def test_sigma_all_models(self, client, auth_headers):
        """Test constant and exponential g(z) models."""
        # Note: 'standard' model requires Omega_Lambda_z parameter
        for model in ["constant", "exponential"]:
            response = client.get(
                "/weyl/sigma", params={"z": 0.5, "Sigma_0": 0.2, "g_model": model}, headers=auth_headers
            )
            assert response.status_code == 200
            assert response.json()["Sigma_0"] == 0.2

    def test_sigma_invalid_model(self, client, auth_headers):
        """Test invalid g(z) model."""
        response = client.get(
            "/weyl/sigma", params={"z": 0.5, "Sigma_0": 0.2, "g_model": "invalid"}, headers=auth_headers
        )
        assert response.status_code == 400


class TestDESY3Endpoint:
    """Test /weyl/des-y3 endpoint."""

    def test_des_y3_structure(self, client, auth_headers):
        """Test DES Y3 data structure."""
        response = client.get("/weyl/des-y3", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert len(data["z_bins"]) == 4
        assert len(data["hJ_mean"]) == 4
        assert len(data["hJ_sigma"]) == 4
        assert "mean" in data["Sigma_0_standard"]
        assert "sigma" in data["Sigma_0_standard"]
        assert "mean" in data["Sigma_0_constant"]
        assert "sigma" in data["Sigma_0_constant"]

    def test_des_y3_values(self, client, auth_headers):
        """Test DES Y3 reference values."""
        response = client.get("/weyl/des-y3", headers=auth_headers)
        data = response.json()

        # Check z_bins match paper values
        assert data["z_bins"][0] == pytest.approx(0.295, rel=0.01)
        assert data["z_bins"][1] == pytest.approx(0.467, rel=0.01)

        # Check ĥJ values are in expected range
        for hJ in data["hJ_mean"]:
            assert 0.2 < hJ < 0.5

        # Check Σ₀ fits - constant model has 0.13, standard has 0.24
        assert data["Sigma_0_constant"]["mean"] == pytest.approx(0.13, rel=0.1)
        assert data["Sigma_0_standard"]["mean"] == pytest.approx(0.24, rel=0.1)


class TestUMCPMappingEndpoint:
    """Test /weyl/umcp-mapping endpoint."""

    def test_mapping_structure(self, client, auth_headers):
        """Test UMCP mapping structure."""
        response = client.get(
            "/weyl/umcp-mapping", params={"Sigma_0": 0.24, "chi2_Sigma": 1.1, "chi2_LCDM": 2.1}, headers=auth_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert "omega_analog" in data
        assert "F_analog" in data
        assert "regime" in data
        assert "chi2_improvement" in data

    def test_mapping_conservation(self, client, auth_headers):
        """Test ω + F = 1 conservation."""
        response = client.get(
            "/weyl/umcp-mapping", params={"Sigma_0": 0.24, "chi2_Sigma": 1.1, "chi2_LCDM": 2.1}, headers=auth_headers
        )
        data = response.json()

        assert data["omega_analog"] + data["F_analog"] == pytest.approx(1.0, rel=1e-6)

    def test_mapping_chi2_improvement(self, client, auth_headers):
        """Test χ² improvement calculation."""
        response = client.get(
            "/weyl/umcp-mapping", params={"Sigma_0": 0.24, "chi2_Sigma": 1.0, "chi2_LCDM": 2.0}, headers=auth_headers
        )
        data = response.json()

        # χ² improved from 2.0 to 1.0 = 50% improvement
        assert data["chi2_improvement"] == pytest.approx(0.5, rel=0.1)

    def test_mapping_regimes(self, client, auth_headers):
        """Test regime classification for different Σ₀ values."""
        # Small Σ₀ → Stable (UMCP analog)
        response = client.get(
            "/weyl/umcp-mapping", params={"Sigma_0": 0.05, "chi2_Sigma": 1.0, "chi2_LCDM": 1.0}, headers=auth_headers
        )
        assert response.json()["regime"] == "Stable"

        # Medium Σ₀ → Watch (UMCP analog)
        response = client.get(
            "/weyl/umcp-mapping", params={"Sigma_0": 0.2, "chi2_Sigma": 1.0, "chi2_LCDM": 1.0}, headers=auth_headers
        )
        assert response.json()["regime"] == "Watch"


class TestWeylAPIAuth:
    """Test WEYL API authentication."""

    def test_no_api_key(self):
        """Test endpoints require API key."""
        client_no_auth = TestClient(app)  # No API key header

        response = client_no_auth.get("/weyl/background", params={"z": 0.5})
        assert response.status_code == 401

        response = client_no_auth.get("/weyl/sigma", params={"z": 0.5})
        assert response.status_code == 401

        response = client_no_auth.get("/weyl/des-y3")
        assert response.status_code == 401

        response = client_no_auth.get("/weyl/umcp-mapping")
        assert response.status_code == 401
