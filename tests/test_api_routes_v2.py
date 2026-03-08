"""
Tests for UMCP REST API — Extended Routes v2.

Tests all 44 new endpoints added in api_routes_v2.py:
  - Seam chain (3 routes)
  - τ_R* thermodynamic (4 routes)
  - Epistemic cost (3 routes)
  - Insights engine (4 routes)
  - Standard Model extensions (10 routes)
  - Dynamic semiotics (4 routes)
  - Consciousness coherence (3 routes)
  - Materials science (2 routes)
  - Atomic cross-scale (4 routes)
  - Rosetta translation (2 routes)
  - Orientation (1 route)
  - Kernel comparison (1 route)
  - System info (3 routes)
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from umcp.api_umcp import app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    app = None  # type: ignore
    TestClient = None  # type: ignore

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi not installed")


@pytest.fixture
def client():
    """Create test client for API."""
    assert app is not None and TestClient is not None
    return TestClient(app)


@pytest.fixture
def headers():
    """Authentication headers with valid API key."""
    return {"X-API-Key": "umcp-dev-key"}


# ============================================================================
# Seam Chain Endpoints (3 routes)
# ============================================================================


class TestSeamEndpoints:
    """Tests for /seam/* endpoints."""

    def test_seam_reset(self, client, headers):
        """POST /seam/reset clears the accumulator."""
        resp = client.post("/seam/reset", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reset"

    def test_seam_compute(self, client, headers):
        """POST /seam/compute adds a seam record."""
        # Reset first
        client.post("/seam/reset", headers=headers)
        resp = client.post(
            "/seam/compute",
            json={
                "t0": 0,
                "t1": 10,
                "kappa_t0": -0.1,
                "kappa_t1": -0.2,
                "tau_R": 1.0,
                "R": 0.01,
                "D_omega": 0.005,
                "D_C": 0.003,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["t0"] == 0
        assert data["t1"] == 10
        assert "delta_kappa_ledger" in data
        assert "delta_kappa_budget" in data
        assert "residual" in data
        assert "cumulative_residual" in data

    def test_seam_metrics(self, client, headers):
        """GET /seam/metrics returns chain metrics."""
        # Reset and add a record
        client.post("/seam/reset", headers=headers)
        client.post(
            "/seam/compute",
            json={
                "t0": 0,
                "t1": 10,
                "kappa_t0": -0.1,
                "kappa_t1": -0.2,
                "tau_R": 1.0,
                "R": 0.01,
                "D_omega": 0.005,
                "D_C": 0.003,
            },
            headers=headers,
        )
        resp = client.get("/seam/metrics", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_seams"] >= 1
        assert "total_delta_kappa" in data
        assert "max_residual" in data
        assert isinstance(data["is_returning"], bool)
        assert isinstance(data["failure_detected"], bool)

    def test_seam_requires_auth(self, client):
        """Seam endpoints require authentication."""
        resp = client.get("/seam/metrics")
        assert resp.status_code == 401


# ============================================================================
# τ_R* Thermodynamic Endpoints (4 routes)
# ============================================================================


class TestTauRStarEndpoints:
    """Tests for /tau-r-star/* endpoints."""

    def test_tau_r_star_compute_stable(self, client, headers):
        """POST /tau-r-star/compute with stable parameters."""
        resp = client.post(
            "/tau-r-star/compute",
            json={"omega": 0.02, "C": 0.05, "R": 0.01, "delta_kappa": 0.0},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tau_R_star" in data
        assert "gamma" in data
        assert "D_C" in data
        assert data["phase"] in ("SURPLUS", "DEFICIT", "FREE_RETURN", "TRAPPED", "POLE")
        assert data["dominance"] in ("DRIFT", "CURVATURE", "MEMORY")
        assert "R_critical" in data
        assert "c_trap" in data
        assert isinstance(data["is_trapped"], bool)

    def test_tau_r_star_compute_collapse(self, client, headers):
        """POST /tau-r-star/compute with collapse parameters."""
        resp = client.post(
            "/tau-r-star/compute",
            json={"omega": 0.8, "C": 0.5, "R": 0.001, "delta_kappa": -1.0},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tau_R_star"] >= 0
        assert data["gamma"] >= 0

    def test_r_critical(self, client, headers):
        """POST /tau-r-star/r-critical returns R_critical."""
        resp = client.post(
            "/tau-r-star/r-critical",
            json={"omega": 0.1, "C": 0.1, "delta_kappa": 0.0},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "R_critical" in data
        assert data["omega"] == 0.1
        assert data["C"] == 0.1

    def test_r_min(self, client, headers):
        """POST /tau-r-star/r-min returns R_min."""
        resp = client.post(
            "/tau-r-star/r-min",
            json={"omega": 0.1, "C": 0.1, "tau_R_target": 5.0, "delta_kappa": 0.0},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "R_min" in data
        assert data["tau_R_target"] == 5.0

    def test_trapping_threshold(self, client, headers):
        """GET /tau-r-star/trapping-threshold returns c_trap."""
        resp = client.get("/tau-r-star/trapping-threshold", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "c_trap" in data
        assert "omega_trap" in data
        # c_trap ≈ 0.3178 (Cardano root)
        assert 0.31 < data["c_trap"] < 0.33
        # omega_trap = 1 - c_trap
        assert abs(data["c_trap"] + data["omega_trap"] - 1.0) < 1e-10


# ============================================================================
# Epistemic Cost Endpoints (3 routes)
# ============================================================================


class TestEpistemicEndpoints:
    """Tests for /epistemic/* endpoints."""

    def test_classify_return(self, client, headers):
        """POST /epistemic/classify for a RETURN act (seam passes, finite τ_R)."""
        resp = client.post(
            "/epistemic/classify",
            json={"seam_pass": True, "tau_R": 1.0, "regime": "STABLE"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict" in data
        assert "reasons" in data
        assert isinstance(data["reasons"], list)
        assert data["verdict"].upper() == "RETURN"

    def test_classify_gesture(self, client, headers):
        """POST /epistemic/classify for a GESTURE (seam fails)."""
        resp = client.post(
            "/epistemic/classify",
            json={
                "seam_pass": False,
                "tau_R": 1.0,
                "regime": "WATCH",
                "seam_failures": ["residual_exceeded"],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"].upper() == "GESTURE"

    def test_classify_dissolution(self, client, headers):
        """POST /epistemic/classify for DISSOLUTION (collapse + infinite τ_R)."""
        resp = client.post(
            "/epistemic/classify",
            json={"seam_pass": False, "tau_R": 1e31, "regime": "COLLAPSE"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"].upper() in ("DISSOLUTION", "GESTURE")

    def test_classify_invalid_regime(self, client, headers):
        """POST /epistemic/classify rejects invalid regime."""
        resp = client.post(
            "/epistemic/classify",
            json={"seam_pass": True, "tau_R": 1.0, "regime": "INVALID"},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_positional_illusion(self, client, headers):
        """POST /epistemic/positional-illusion quantifies observation cost."""
        resp = client.post(
            "/epistemic/positional-illusion",
            json={"omega": 0.1, "n_observations": 5},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "gamma" in data
        assert data["n_observations"] == 5
        assert "total_cost" in data
        assert "budget_fraction" in data
        assert "illusion_severity" in data

    def test_trace_assessment(self, client, headers):
        """POST /epistemic/trace-assessment returns conformance verdict."""
        resp = client.post(
            "/epistemic/trace-assessment",
            json={
                "n_components": 8,
                "n_timesteps": 100,
                "n_clipped": 0,
                "is_degenerate": False,
                "seam_pass": True,
                "tau_R": 1.0,
                "regime": "STABLE",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_components"] == 8
        assert data["n_timesteps"] == 100
        assert "epsilon_floor" in data
        assert "verdict" in data


# ============================================================================
# Insights Engine Endpoints (4 routes)
# ============================================================================


class TestInsightsEndpoints:
    """Tests for /insights/* endpoints."""

    def test_insights_summary(self, client, headers):
        """GET /insights/summary returns summary statistics."""
        resp = client.get("/insights/summary", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_insights" in data
        assert isinstance(data["total_insights"], int)
        assert "domains" in data
        assert isinstance(data["domains"], dict)

    def test_insights_discover(self, client, headers):
        """GET /insights/discover runs pattern discovery."""
        resp = client.get("/insights/discover", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_insights_random(self, client, headers):
        """GET /insights/random returns a random insight."""
        resp = client.get("/insights/random?seed=42", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "insight" in data
        assert isinstance(data["insight"], str)

    def test_insights_query(self, client, headers):
        """GET /insights/query with optional filters."""
        resp = client.get("/insights/query", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_insights_query_filter(self, client, headers):
        """GET /insights/query with domain filter."""
        resp = client.get("/insights/query?severity=Fundamental", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


# ============================================================================
# Standard Model Extension Endpoints (10 routes)
# ============================================================================


class TestSMEndpoints:
    """Tests for /sm/* endpoints."""

    def test_ckm(self, client, headers):
        """GET /sm/ckm returns CKM matrix and invariants."""
        resp = client.get("/sm/ckm", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "V_matrix" in data
        assert len(data["V_matrix"]) == 3
        assert len(data["V_matrix"][0]) == 3
        assert "J_CP" in data
        assert data["J_CP"] > 0
        assert "omega_eff" in data
        assert "F_eff" in data
        assert "regime" in data
        # Unitarity checks
        assert "unitarity_row1" in data
        assert "unitarity_row2" in data
        assert "unitarity_row3" in data

    def test_ckm_custom_params(self, client, headers):
        """GET /sm/ckm with custom Wolfenstein parameters."""
        resp = client.get(
            "/sm/ckm?lambda_w=0.2250&A=0.800&rho_bar=0.140&eta_bar=0.350",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "V_matrix" in data

    def test_coupling(self, client, headers):
        """GET /sm/coupling at Z-pole energy."""
        resp = client.get("/sm/coupling?Q_GeV=91.2", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "alpha_s" in data
        assert "alpha_em" in data
        # α_s(M_Z) ≈ 0.118
        assert 0.10 < data["alpha_s"] < 0.13
        assert "sin2_theta_W" in data
        assert "n_flavors" in data
        assert "regime" in data

    def test_coupling_high_energy(self, client, headers):
        """GET /sm/coupling at high energy (GUT scale proximity)."""
        resp = client.get("/sm/coupling?Q_GeV=1000.0", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        # α_s should be smaller at higher energy (asymptotic freedom)
        assert data["alpha_s"] < 0.118

    def test_cross_section(self, client, headers):
        """GET /sm/cross-section at Z-pole."""
        resp = client.get("/sm/cross-section?sqrt_s_GeV=91.2", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "sigma_point_pb" in data
        assert "R_predicted" in data
        assert "R_QCD_corrected" in data
        assert "n_colors" in data
        assert data["n_colors"] == 3
        assert "regime" in data

    def test_higgs(self, client, headers):
        """GET /sm/higgs returns Higgs mechanism results."""
        resp = client.get("/sm/higgs", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["v_GeV"] == 246.22
        assert data["m_H_GeV"] == 125.25
        assert "lambda_quartic" in data
        assert "mu_squared" in data
        assert "yukawa_couplings" in data
        assert isinstance(data["yukawa_couplings"], dict)
        assert "m_W_predicted" in data
        assert "m_Z_predicted" in data
        # W mass ≈ 80.4 GeV
        assert 79 < data["m_W_predicted"] < 82

    def test_neutrino_probability(self, client, headers):
        """GET /sm/neutrino/probability for ν_μ → ν_e oscillation."""
        resp = client.get(
            "/sm/neutrino/probability?alpha=1&beta=0&L_km=1285&E_GeV=2.5",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["L_km"] == 1285.0
        assert data["E_GeV"] == 2.5
        assert "channel" in data

    def test_neutrino_survival(self, client, headers):
        """GET /sm/neutrino/probability for survival probability (α=β)."""
        resp = client.get(
            "/sm/neutrino/probability?alpha=1&beta=1&L_km=1285&E_GeV=2.5",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Survival probability P(ν_μ → ν_μ) is between 0 and 1
        assert 0.0 <= data["probability"] <= 1.0

    def test_dune_prediction(self, client, headers):
        """GET /sm/neutrino/dune returns DUNE experiment predictions."""
        resp = client.get("/sm/neutrino/dune", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "P_mue_vacuum" in data
        assert "P_mue_matter" in data or "matter_enhancement" in data

    def test_matter_genesis(self, client, headers):
        """GET /sm/matter-genesis runs full analysis."""
        resp = client.get("/sm/matter-genesis", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entities" in data
        assert data["total_entities"] >= 90
        assert "total_acts" in data
        assert "acts" in data
        assert isinstance(data["acts"], dict)
        assert "n_theorems" in data

    def test_mass_origins(self, client, headers):
        """GET /sm/matter-genesis/mass-origins returns mass origin analysis."""
        resp = client.get("/sm/matter-genesis/mass-origins", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        first = data[0]
        assert "entity_name" in first
        assert "total_mass_GeV" in first
        assert "higgs_fraction" in first

    def test_matter_map(self, client, headers):
        """GET /sm/matter-map returns full 6-scale map."""
        resp = client.get("/sm/matter-map", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entities" in data
        assert data["total_entities"] >= 100
        assert "scales" in data
        assert "transitions" in data
        assert isinstance(data["transitions"], list)

    def test_matter_map_scale(self, client, headers):
        """GET /sm/matter-map/scale/fundamental returns fundamental entities."""
        resp = client.get("/sm/matter-map/scale/fundamental", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["scale"] == "fundamental"
        assert "n_entities" in data
        assert "entities" in data

    def test_matter_map_scale_atomic(self, client, headers):
        """GET /sm/matter-map/scale/atomic returns atomic entities."""
        resp = client.get("/sm/matter-map/scale/atomic", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["scale"] == "atomic"

    def test_matter_map_scale_invalid(self, client, headers):
        """GET /sm/matter-map/scale/invalid returns 404."""
        resp = client.get("/sm/matter-map/scale/invalid", headers=headers)
        assert resp.status_code == 404


# ============================================================================
# Dynamic Semiotics Endpoints (4 routes)
# ============================================================================


class TestSemioticsEndpoints:
    """Tests for /semiotics/* endpoints."""

    def test_list_semiotic_systems(self, client, headers):
        """GET /semiotics/systems returns all 30 sign systems."""
        resp = client.get("/semiotics/systems", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 30
        first = data[0]
        assert "name" in first
        assert "F" in first

    def test_get_semiotic_system(self, client, headers):
        """GET /semiotics/system/{name} returns a specific system."""
        resp = client.get("/semiotics/system/Modern English", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Modern English"
        assert "F" in data

    def test_get_semiotic_system_not_found(self, client, headers):
        """GET /semiotics/system/{name} returns 404 for unknown system."""
        resp = client.get("/semiotics/system/Nonexistent Language", headers=headers)
        assert resp.status_code == 404

    def test_semiotic_structure(self, client, headers):
        """GET /semiotics/structure returns structural analysis."""
        resp = client.get("/semiotics/structure", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_semiotic_brain_bridge(self, client, headers):
        """GET /semiotics/brain-bridge returns bridge analysis."""
        resp = client.get("/semiotics/brain-bridge", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================================
# Consciousness Coherence Endpoints (3 routes)
# ============================================================================


class TestConsciousnessEndpoints:
    """Tests for /consciousness/* endpoints."""

    def test_list_coherence_systems(self, client, headers):
        """GET /consciousness/systems returns all 20 systems."""
        resp = client.get("/consciousness/systems", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 20

    def test_get_coherence_system(self, client, headers):
        """GET /consciousness/system/{name} returns a specific system."""
        resp = client.get(
            "/consciousness/system/human_waking_consciousness",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data

    def test_get_coherence_system_not_found(self, client, headers):
        """GET /consciousness/system/{name} returns 404 for unknown system."""
        resp = client.get("/consciousness/system/nonexistent", headers=headers)
        assert resp.status_code == 404

    def test_coherence_structure(self, client, headers):
        """GET /consciousness/structure returns structural analysis."""
        resp = client.get("/consciousness/structure", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================================
# Materials Science Endpoints (2 routes)
# ============================================================================


class TestMaterialsEndpoints:
    """Tests for /materials/* endpoints."""

    def test_list_elements(self, client, headers):
        """GET /materials/elements returns all 118 elements."""
        resp = client.get("/materials/elements", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 118

    def test_list_elements_filter_block(self, client, headers):
        """GET /materials/elements?block=d filters by d-block."""
        resp = client.get("/materials/elements?block=d", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        for el in data:
            assert el["block"] == "d"

    def test_list_elements_filter_period(self, client, headers):
        """GET /materials/elements?period=1 filters by period."""
        resp = client.get("/materials/elements?period=1", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        # Period 1: H, He
        assert len(data) == 2

    def test_get_element_by_z(self, client, headers):
        """GET /materials/element/26 returns Iron by Z."""
        resp = client.get("/materials/element/26", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "Fe"
        assert data["Z"] == 26

    def test_get_element_by_symbol(self, client, headers):
        """GET /materials/element/Fe returns Iron by symbol."""
        resp = client.get("/materials/element/Fe", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["Z"] == 26

    def test_get_element_not_found(self, client, headers):
        """GET /materials/element/999 returns 404."""
        resp = client.get("/materials/element/999", headers=headers)
        assert resp.status_code == 404


# ============================================================================
# Atomic Cross-Scale Endpoints (4 routes)
# ============================================================================


class TestAtomicEndpoints:
    """Tests for /atomic/* endpoints."""

    def test_cross_scale_all(self, client, headers):
        """GET /atomic/cross-scale returns 12-channel kernel for elements."""
        resp = client.get("/atomic/cross-scale?limit=5", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 5

    def test_cross_scale_element(self, client, headers):
        """GET /atomic/cross-scale/26 returns kernel for Iron."""
        resp = client.get("/atomic/cross-scale/26", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_cross_scale_element_not_found(self, client, headers):
        """GET /atomic/cross-scale/999 returns 404."""
        resp = client.get("/atomic/cross-scale/999", headers=headers)
        assert resp.status_code == 404

    def test_binding_energy(self, client, headers):
        """GET /atomic/binding-energy for Fe-56."""
        resp = client.get("/atomic/binding-energy?Z=26&A=56", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["Z"] == 26
        assert data["A"] == 56
        assert "BE_per_A_MeV" in data
        # Fe-56 BE/A ≈ 8.8 MeV
        assert 8.0 < data["BE_per_A_MeV"] < 9.5

    def test_magic_proximity(self, client, headers):
        """GET /atomic/magic-proximity for O-16."""
        resp = client.get("/atomic/magic-proximity?Z=8&A=16", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["Z"] == 8
        assert data["A"] == 16
        assert "magic_proximity" in data
        # O-16: Z=8 is magic → high proximity
        assert data["magic_proximity"] >= 0.5


# ============================================================================
# Rosetta Translation Endpoints (2 routes)
# ============================================================================


class TestRosettaEndpoints:
    """Tests for /rosetta/* endpoints."""

    def test_list_lenses(self, client, headers):
        """GET /rosetta/lenses returns all 6 lenses."""
        resp = client.get("/rosetta/lenses", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert len(data) == 6
        expected = {"Epistemology", "Ontology", "Phenomenology", "History", "Policy", "Semiotics"}
        assert set(data.keys()) == expected

    def test_translate(self, client, headers):
        """POST /rosetta/translate maps across lenses."""
        resp = client.post(
            "/rosetta/translate",
            json={
                "drift": "The model weights shifted after fine-tuning",
                "fidelity": "Core reasoning capabilities preserved",
                "roughness": "Training instability in early epochs",
                "return_narrative": "Validated on held-out benchmark",
                "source_lens": "Epistemology",
                "target_lens": "Ontology",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_lens"] == "Epistemology"
        assert data["target_lens"] == "Ontology"
        assert "translations" in data
        assert "Drift" in data["translations"]
        assert "Fidelity" in data["translations"]
        assert "lens_mappings" in data

    def test_translate_invalid_lens(self, client, headers):
        """POST /rosetta/translate rejects unknown lenses."""
        resp = client.post(
            "/rosetta/translate",
            json={
                "drift": "x",
                "fidelity": "x",
                "roughness": "x",
                "return_narrative": "x",
                "source_lens": "InvalidLens",
                "target_lens": "Ontology",
            },
            headers=headers,
        )
        assert resp.status_code == 400


# ============================================================================
# Orientation Endpoint (1 route)
# ============================================================================


class TestOrientationEndpoint:
    """Tests for /orientation endpoint."""

    def test_orientation_all(self, client, headers):
        """GET /orientation runs all 7 sections."""
        resp = client.get("/orientation", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Should have section keys
        assert len(data) >= 1

    def test_orientation_section(self, client, headers):
        """GET /orientation?section=1 runs just section 1 (Duality)."""
        resp = client.get("/orientation?section=1", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================================
# Kernel Comparison Endpoint (1 route)
# ============================================================================


class TestKernelCompare:
    """Tests for /kernel/compare endpoint."""

    def test_compare_two_traces(self, client, headers):
        """POST /kernel/compare compares multiple traces."""
        resp = client.post(
            "/kernel/compare",
            json={
                "traces": [[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2


# ============================================================================
# System Info Endpoints (3 routes)
# ============================================================================


class TestSystemEndpoints:
    """Tests for system info endpoints."""

    def test_frozen_contract(self, client, headers):
        """GET /frozen-contract returns frozen parameters."""
        resp = client.get("/frozen-contract", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "EPSILON" in data
        assert data["EPSILON"] == 1e-8
        assert "P_EXPONENT" in data
        assert data["P_EXPONENT"] == 3
        assert "TOL_SEAM" in data
        assert data["TOL_SEAM"] == 0.005

    def test_schemas(self, client, headers):
        """GET /schemas returns schema list."""
        resp = client.get("/schemas", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_integrity(self, client, headers):
        """GET /integrity returns SHA-256 checksums."""
        resp = client.get("/integrity", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "total_files" in data


# ============================================================================
# Auth boundary tests
# ============================================================================


class TestAuthBoundary:
    """Verify auth is required on all v2 routes."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/seam/metrics"),
            ("GET", "/tau-r-star/trapping-threshold"),
            ("GET", "/insights/summary"),
            ("GET", "/sm/ckm"),
            ("GET", "/sm/coupling"),
            ("GET", "/sm/cross-section"),
            ("GET", "/sm/higgs"),
            ("GET", "/sm/neutrino/dune"),
            ("GET", "/sm/matter-genesis"),
            ("GET", "/sm/matter-genesis/mass-origins"),
            ("GET", "/sm/matter-map"),
            ("GET", "/semiotics/systems"),
            ("GET", "/semiotics/structure"),
            ("GET", "/semiotics/brain-bridge"),
            ("GET", "/consciousness/systems"),
            ("GET", "/consciousness/structure"),
            ("GET", "/materials/elements"),
            ("GET", "/rosetta/lenses"),
            ("GET", "/orientation"),
            ("GET", "/frozen-contract"),
            ("GET", "/schemas"),
            ("GET", "/integrity"),
        ],
    )
    def test_auth_required(self, client, method, path):
        """All v2 endpoints require authentication."""
        resp = getattr(client, method.lower())(path)
        assert resp.status_code == 401, f"{method} {path} should require auth"
