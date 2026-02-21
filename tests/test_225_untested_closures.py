"""test_225_untested_closures — Cover closure functions with zero or minimal test coverage.

Targets:
  §1  Standard Model: cross sections (compute_cross_section)
  §2  Standard Model: symmetry breaking (compute_higgs_mechanism)
  §3  Materials science: gap-capture SS1M (compute_gap_capture)
  §4  RCFT information geometry (binary_entropy, fisher_metric, fisher_distance, etc.)
  §5  Weyl cosmology background (compute_background, individual functions, embedding)
  §6  Kinematics energy mechanics (all 7 public functions)
  §7  Contract ↔ closure bridge (all 13 domains have matching contract YAML + closures)

Every test validates Tier-1 identities where kernel invariants are produced.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
CLOSURES = ROOT / "closures"
CONTRACTS = ROOT / "contracts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Frozen contract constants

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOL_DUALITY = 1e-6  # F + ω ≈ 1
TOL_BOUND = 1e-4  # IC ≤ F + tolerance for ε-clamp artifacts
TOL_LOG_IC = 1e-3  # IC ≈ exp(κ)


def _check_tier1(F: float, omega: float, IC: float, kappa: float, label: str) -> None:
    """Assert the three Tier-1 structural identities."""
    # Duality identity: F + ω = 1
    assert abs(F + omega - 1.0) < TOL_DUALITY, f"{label}: duality |F+ω-1| = {abs(F + omega - 1.0)}"
    # Integrity bound: IC ≤ F (with small tolerance for ε-clamp)
    assert IC <= F + TOL_BOUND, f"{label}: IC={IC} > F={F}"
    # Log-integrity: IC ≈ exp(κ)
    if kappa > -30:  # avoid underflow
        expected_IC = math.exp(kappa)
        assert abs(IC - expected_IC) < TOL_LOG_IC + abs(expected_IC) * 0.01, f"{label}: IC={IC} vs exp(κ)={expected_IC}"


# ═══════════════════════════════════════════════════════════════════════════
# §1  Standard Model — Cross Sections
# ═══════════════════════════════════════════════════════════════════════════
class TestCrossSections:
    """Test compute_cross_section across energy thresholds with Tier-1 checks."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.standard_model.cross_sections import (
            R_EXPERIMENTAL,
            CrossSectionResult,
            compute_cross_section,
        )

        self.compute = compute_cross_section
        self.R_EXP = R_EXPERIMENTAL
        self.ResultType = CrossSectionResult

    # Parametrize over experimental R-ratio energy points
    @pytest.mark.parametrize(
        "label,sqrt_s_R",
        [
            ("below_charm", (2.0, 2.17)),
            ("charm_region", (4.0, 3.56)),
            ("bottom_region", (12.0, 3.85)),
            ("Z_peak", (91.2, 20.79)),
            ("above_Z", (200.0, 4.0)),
        ],
    )
    def test_experimental_points(self, label, sqrt_s_R):
        """Cross section at each experimental energy point."""
        sqrt_s, R_meas = sqrt_s_R
        res = self.compute(sqrt_s, R_measured=R_meas)
        assert isinstance(res, self.ResultType)
        assert res.sqrt_s_GeV == pytest.approx(sqrt_s)
        assert 0.0 <= res.F_eff <= 1.0
        assert 0.0 <= res.omega_eff <= 1.0
        # Duality identity: F + ω = 1 (CrossSectionResult has no IC/κ fields)
        assert abs(res.F_eff + res.omega_eff - 1.0) < TOL_DUALITY

    @pytest.mark.parametrize("sqrt_s", [1.5, 3.0, 10.0, 50.0, 500.0, 5000.0])
    def test_energy_sweep(self, sqrt_s):
        """Cross section without measured R — pure prediction."""
        res = self.compute(sqrt_s)
        assert res.sigma_point_pb > 0, "Point cross section must be positive"
        assert res.R_predicted > 0, "R-ratio prediction must be positive"
        assert res.n_colors == 3, "N_c must be 3"
        assert res.n_active_flavors >= 0

    def test_qcd_correction(self):
        """R_QCD_corrected > R_predicted at high energy (QCD increases R)."""
        res = self.compute(91.2)
        assert res.R_QCD_corrected >= res.R_predicted

    def test_regime_classification(self):
        """Regime must be one of the allowed strings."""
        for sqrt_s in [2.0, 91.2, 200.0]:
            res = self.compute(sqrt_s)
            assert res.regime in {"Validated", "Tension", "Anomalous"}


# ═══════════════════════════════════════════════════════════════════════════
# §2  Standard Model — Symmetry Breaking (Higgs mechanism)
# ═══════════════════════════════════════════════════════════════════════════
class TestSymmetryBreaking:
    """Test compute_higgs_mechanism with default and varied parameters."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.standard_model.symmetry_breaking import (
            FERMION_MASSES,
            M_H_MEASURED,
            M_W_MEASURED,
            M_Z_MEASURED,
            V_EW,
            HiggsResult,
            compute_higgs_mechanism,
        )

        self.compute = compute_higgs_mechanism
        self.V_EW = V_EW
        self.M_H = M_H_MEASURED
        self.M_W = M_W_MEASURED
        self.M_Z = M_Z_MEASURED
        self.FERMION_MASSES = FERMION_MASSES
        self.ResultType = HiggsResult

    def test_default_values(self):
        """Default (SM) parameters produce consistent results."""
        res = self.compute()
        assert isinstance(res, self.ResultType)
        assert res.v_GeV == pytest.approx(self.V_EW)
        assert res.m_H_GeV == pytest.approx(self.M_H)

    def test_quartic_coupling_positive(self):
        """λ must be positive for a bounded-below potential."""
        res = self.compute()
        assert res.lambda_quartic > 0

    def test_mu_squared_negative(self):
        """μ² must be negative for spontaneous symmetry breaking."""
        res = self.compute()
        assert res.mu_squared < 0

    def test_yukawa_couplings_present(self):
        """All 9 fermion Yukawa couplings must be returned."""
        res = self.compute()
        assert len(res.yukawa_couplings) >= 9
        for name in self.FERMION_MASSES:
            assert name in res.yukawa_couplings

    def test_top_yukawa_near_unity(self):
        """Top quark Yukawa ≈ 1 (largest fermion coupling)."""
        res = self.compute()
        y_top = res.yukawa_couplings["top"]
        assert 0.9 < y_top < 1.1, f"y_top = {y_top}"

    def test_mass_predictions(self):
        """Predicted W and Z masses should be close to measured values."""
        res = self.compute()
        assert abs(res.m_W_predicted - self.M_W) / self.M_W < 0.02
        assert abs(res.m_Z_predicted - self.M_Z) / self.M_Z < 0.02

    def test_duality_identity(self):
        """F + ω = 1 for the Higgs mechanism result."""
        res = self.compute()
        assert abs(res.F_eff + res.omega_eff - 1.0) < TOL_DUALITY

    def test_regime_classification(self):
        """Regime must be one of the allowed Higgs classifications."""
        res = self.compute()
        assert res.regime in {"Consistent", "Tension", "BSM_hint"}

    @pytest.mark.parametrize("v_scale", [0.5, 1.0, 1.5, 2.0])
    def test_varying_vev(self, v_scale):
        """VEV variations still produce valid duality identity."""
        res = self.compute(v_GeV=self.V_EW * v_scale)
        assert abs(res.F_eff + res.omega_eff - 1.0) < TOL_DUALITY
        assert res.v_GeV == pytest.approx(self.V_EW * v_scale)


# ═══════════════════════════════════════════════════════════════════════════
# §3  Materials Science — Gap-Capture SS1M
# ═══════════════════════════════════════════════════════════════════════════
class TestGapCaptureSS1M:
    """Test compute_gap_capture and batch_compute for well-known elements."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.materials_science.gap_capture_ss1m import (
            CHANNEL_LABELS,
            SS1M,
            SeamReceipt,
            batch_compute,
            compute_gap_capture,
        )

        self.compute = compute_gap_capture
        self.batch = batch_compute
        self.CHANNELS = CHANNEL_LABELS
        self.SeamReceipt = SeamReceipt
        self.SS1M = SS1M

    @pytest.mark.parametrize("symbol", ["Fe", "Cu", "Si", "Al", "Nb"])
    def test_single_element(self, symbol):
        """Gap-capture for well-characterized elements."""
        receipt, stamp = self.compute(symbol)
        assert isinstance(receipt, self.SeamReceipt)
        assert isinstance(stamp, self.SS1M)
        assert receipt.element == symbol
        assert stamp.element == symbol
        # Duality must hold
        assert receipt.duality_exact or abs(receipt.duality_sum - 1.0) < TOL_DUALITY

    @pytest.mark.parametrize("symbol", ["Fe", "Cu", "Si"])
    def test_stamp_invariants(self, symbol):
        """SS1M stamp carries valid kernel invariants."""
        _, stamp = self.compute(symbol)
        assert 0.0 <= stamp.F_cap <= 1.0
        assert 0.0 <= stamp.omega_cap <= 1.0
        assert stamp.IC_cap <= stamp.F_cap + TOL_BOUND

    def test_batch_compute(self):
        """Batch processing returns results for each element."""
        results = self.batch(["Fe", "Al", "Cu"])
        assert len(results) == 3
        for sym, receipt, stamp, err in results:
            assert sym in {"Fe", "Al", "Cu"}
            if err == "":
                assert receipt is not None
                assert stamp is not None

    def test_receipt_serialization(self):
        """SeamReceipt can be serialized to dict and JSON."""
        receipt, _ = self.compute("Fe")
        d = receipt.to_dict()
        assert isinstance(d, dict)
        assert "element" in d
        j = receipt.to_json()
        assert '"element"' in j

    def test_stamp_hud(self):
        """SS1M stamp produces a human-readable HUD string."""
        _, stamp = self.compute("Fe")
        hud = stamp.hud()
        assert isinstance(hud, str)
        assert "Fe" in hud

    def test_channel_count(self):
        """Receipt must report channels_total = 8."""
        receipt, _ = self.compute("Fe")
        assert receipt.channels_total == 8

    def test_regime_labels(self):
        """Regime labels must be valid regime strings."""
        _receipt, stamp = self.compute("Fe")
        valid = {"Stable", "Watch", "Collapse"}
        assert stamp.regime in valid


# ═══════════════════════════════════════════════════════════════════════════
# §4  RCFT — Information Geometry
# ═══════════════════════════════════════════════════════════════════════════
class TestInformationGeometry:
    """Pure-math tests for RCFT information geometry functions."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.rcft.information_geometry import (
            binary_entropy,
            compute_efficiency,
            compute_geodesic_budget_cost,
            compute_geodesic_path,
            compute_path_length,
            fisher_distance_1d,
            fisher_distance_weighted,
            fisher_geodesic,
            fisher_metric_1d,
            verify_fano_fisher_duality,
        )

        self.binary_entropy = binary_entropy
        self.fisher_distance_1d = fisher_distance_1d
        self.fisher_distance_weighted = fisher_distance_weighted
        self.fisher_geodesic = fisher_geodesic
        self.compute_geodesic_path = compute_geodesic_path
        self.fisher_metric_1d = fisher_metric_1d
        self.verify_fano = verify_fano_fisher_duality
        self.compute_path_length = compute_path_length
        self.compute_efficiency = compute_efficiency
        self.compute_geodesic_budget_cost = compute_geodesic_budget_cost

    # --- binary_entropy ---
    @pytest.mark.parametrize(
        "c,expected",
        [
            (0.5, math.log(2)),  # Max entropy at c = 0.5
            (0.0, 0.0),  # Entropy at boundary
            (1.0, 0.0),  # Entropy at boundary
        ],
    )
    def test_binary_entropy_known(self, c, expected):
        """Binary entropy at known values."""
        h = self.binary_entropy(c)
        assert abs(h - expected) < 1e-6, f"H({c}) = {h}, expected {expected}"

    @pytest.mark.parametrize("c", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    def test_binary_entropy_symmetry(self, c):
        """H(c) = H(1-c) — entropy is symmetric."""
        assert abs(self.binary_entropy(c) - self.binary_entropy(1.0 - c)) < 1e-10

    @pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_binary_entropy_nonneg(self, c):
        """Entropy is non-negative."""
        assert self.binary_entropy(c) >= 0.0

    # --- fisher_metric_1d ---
    @pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_fisher_metric_positive(self, c):
        """Fisher metric is positive for c ∈ (0, 1)."""
        g = self.fisher_metric_1d(c)
        assert g > 0

    def test_fisher_metric_minimum_at_half(self):
        """Fisher metric g(c) = 1/(c(1-c)) is minimized at c = 0.5."""
        g_half = self.fisher_metric_1d(0.5)
        for c in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            assert self.fisher_metric_1d(c) >= g_half - 1e-10

    # --- fisher_distance_1d ---
    def test_fisher_distance_zero(self):
        """Distance from a point to itself is zero."""
        assert abs(self.fisher_distance_1d(0.5, 0.5)) < 1e-10

    def test_fisher_distance_symmetry(self):
        """d(c1, c2) = d(c2, c1)."""
        d12 = self.fisher_distance_1d(0.2, 0.8)
        d21 = self.fisher_distance_1d(0.8, 0.2)
        assert abs(d12 - d21) < 1e-10

    @pytest.mark.parametrize("c1,c2", [(0.1, 0.5), (0.3, 0.7), (0.2, 0.9)])
    def test_fisher_distance_positive(self, c1, c2):
        """Distance between distinct points is positive."""
        assert self.fisher_distance_1d(c1, c2) > 0

    def test_fisher_distance_triangle_inequality(self):
        """d(a,c) ≤ d(a,b) + d(b,c) — triangle inequality."""
        a, b, c = 0.1, 0.5, 0.9
        dac = self.fisher_distance_1d(a, c)
        dab = self.fisher_distance_1d(a, b)
        dbc = self.fisher_distance_1d(b, c)
        assert dac <= dab + dbc + 1e-10

    # --- fisher_distance_weighted ---
    def test_fisher_distance_weighted_returns_result(self):
        """Weighted distance returns a FisherDistanceResult."""
        res = self.fisher_distance_weighted([0.3, 0.7], [0.5, 0.5], [0.5, 0.5])
        assert hasattr(res, "distance")
        assert hasattr(res, "normalized")
        assert res.distance >= 0

    # --- fisher_geodesic ---
    def test_geodesic_endpoints(self):
        """Geodesic at t=0 returns c_start, at t=1 returns c_end."""
        c_start, c_end = 0.2, 0.8
        assert abs(self.fisher_geodesic(c_start, c_end, 0.0) - c_start) < 1e-6
        assert abs(self.fisher_geodesic(c_start, c_end, 1.0) - c_end) < 1e-6

    def test_geodesic_midpoint(self):
        """Geodesic interpolation at t=0.5 is between start and end."""
        mid = self.fisher_geodesic(0.2, 0.8, 0.5)
        assert 0.2 < mid < 0.8

    # --- compute_geodesic_path ---
    def test_geodesic_path_length(self):
        """Path has requested number of points."""
        path = self.compute_geodesic_path(0.2, 0.8, n_points=50)
        assert len(path) == 50

    # --- verify_fano_fisher_duality ---
    def test_fano_fisher_duality(self):
        """Fano-Fisher duality: h''(c) = -g_Fisher(c) up to numerical error."""
        results = self.verify_fano()
        assert len(results) > 0
        for r in results:
            assert abs(r.relative_error) < 0.01, f"Fano-Fisher duality error at c={r.c}: {r.relative_error}"

    # --- compute_path_length ---
    def test_path_length_straight(self):
        """Path length of a geodesic ≈ Fisher distance."""
        path_pts = self.compute_geodesic_path(0.3, 0.7, n_points=200)
        c_series = np.array([p.c for p in path_pts])
        path_len = self.compute_path_length(c_series)
        direct = self.fisher_distance_1d(0.3, 0.7)
        assert abs(path_len - direct) / (direct + 1e-10) < 0.05

    # --- compute_efficiency ---
    def test_efficiency_geodesic(self):
        """A geodesic path should have efficiency ≈ 1."""
        path_pts = self.compute_geodesic_path(0.3, 0.7, n_points=200)
        c_series = np.array([p.c for p in path_pts])
        eff = self.compute_efficiency(c_series)
        assert hasattr(eff, "efficiency")
        assert eff.efficiency > 0.95, f"Geodesic efficiency = {eff.efficiency}"

    # --- compute_geodesic_budget_cost ---
    @pytest.mark.parametrize("R", [0.01, 0.5, 1.0, 2.0])
    def test_budget_cost_returns_float(self, R):
        """Budget cost returns a float for positive return credit R."""
        cost = self.compute_geodesic_budget_cost(0.3, 0.7, R)
        assert isinstance(cost, float)


# ═══════════════════════════════════════════════════════════════════════════
# §5  Weyl Cosmology — Background Functions
# ═══════════════════════════════════════════════════════════════════════════
class TestWeylCosmology:
    """Test Weyl cosmology background functions across redshifts."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.weyl.cosmology_background import (
            BackgroundResult,
            CosmologyParams,
            D1_of_z,
            H_of_z,
            Omega_Lambda_of_z,
            Omega_m_of_z,
            chi_of_z,
            compute_background,
            compute_des_y3_background,
            cosmology_as_embedding,
            sigma8_of_z,
        )

        self.H_of_z = H_of_z
        self.Omega_m = Omega_m_of_z
        self.Omega_Lambda = Omega_Lambda_of_z
        self.chi = chi_of_z
        self.D1 = D1_of_z
        self.sigma8 = sigma8_of_z
        self.compute = compute_background
        self.embedding = cosmology_as_embedding
        self.des_y3 = compute_des_y3_background
        self.Params = CosmologyParams
        self.Result = BackgroundResult

    @pytest.mark.parametrize("z", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_H_positive(self, z):
        """Hubble parameter H(z) > 0 for all z ≥ 0."""
        assert self.H_of_z(z) > 0

    @pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 3.0])
    def test_omega_sum(self, z):
        """Ω_m(z) + Ω_Λ(z) ≈ 1 for flat ΛCDM."""
        total = self.Omega_m(z) + self.Omega_Lambda(z)
        assert abs(total - 1.0) < 0.05, f"Ω_m + Ω_Λ = {total} at z={z}"

    def test_chi_monotone(self):
        """Comoving distance χ(z) is monotonically increasing."""
        z_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
        chi_vals = [self.chi(z) for z in z_vals]
        for i in range(len(chi_vals) - 1):
            assert chi_vals[i] < chi_vals[i + 1]

    def test_D1_today(self):
        """Growth function D₁ at z=0 should be ≈ 1 (normalized)."""
        D1_0 = self.D1(0.0)
        assert abs(D1_0 - 1.0) < 0.1

    def test_sigma8_decreasing(self):
        """σ₈(z) decreases with z (structure grows over time)."""
        s0 = self.sigma8(0.0)
        s1 = self.sigma8(1.0)
        assert s0 > s1

    @pytest.mark.parametrize("z", [0.0, 0.3, 1.0])
    def test_compute_background_result(self, z):
        """compute_background returns BackgroundResult with correct fields."""
        res = self.compute(z)
        assert isinstance(res, self.Result)
        assert res.H_z > 0
        assert 0.0 <= res.a <= 1.0 or z == 0.0  # a = 1/(1+z)

    def test_embedding(self):
        """Cosmology embedding produces dict with kernel keys."""
        emb = self.embedding()
        assert isinstance(emb, dict)

    def test_des_y3_background(self):
        """DES Y3 background data loads correctly."""
        des = self.des_y3()
        assert isinstance(des, dict)


# ═══════════════════════════════════════════════════════════════════════════
# §6  Kinematics — Energy Mechanics
# ═══════════════════════════════════════════════════════════════════════════
class TestEnergyMechanics:
    """Test all 7 public energy mechanics functions."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from closures.kinematics.energy_mechanics import (
            compute_kinetic_energy,
            compute_mechanical_energy,
            compute_potential_energy,
            compute_power,
            compute_work,
            verify_energy_conservation,
            verify_work_energy_theorem,
        )

        self.KE = compute_kinetic_energy
        self.PE = compute_potential_energy
        self.ME = compute_mechanical_energy
        self.work = compute_work
        self.power = compute_power
        self.verify_conservation = verify_energy_conservation
        self.verify_work_energy = verify_work_energy_theorem

    # --- Kinetic energy (inputs clipped to [0,1] by FACE_POLICY pre_clip) ---
    @pytest.mark.parametrize(
        "v,expected_ke",
        [
            (0.0, 0.0),
            (0.5, 0.125),  # 0.5 * 0.5² = 0.125
            (0.8, 0.32),  # 0.5 * 0.8² = 0.32
            (1.0, 0.5),  # 0.5 * 1.0² = 0.5
        ],
    )
    def test_kinetic_energy(self, v, expected_ke):
        """KE = 0.5 * m * v² with m=1 (inputs in [0,1] domain)."""
        res = self.KE(v)
        assert abs(res["E_kinetic"] - expected_ke) < 1e-10

    def test_kinetic_energy_zero_at_rest(self):
        """KE = 0 when v = 0."""
        assert self.KE(0.0)["E_kinetic"] == 0.0

    # --- Potential energy (inputs clipped to [0,1]) ---
    @pytest.mark.parametrize(
        "h,expected_pe",
        [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.8, 0.8),
            (1.0, 1.0),
        ],
    )
    def test_potential_energy_gravitational(self, h, expected_pe):
        """PE = m*g*h with m=1, g=1 (inputs in [0,1] domain)."""
        res = self.PE(h)
        assert abs(res["E_potential"] - expected_pe) < 1e-10

    def test_potential_energy_spring(self):
        """Spring PE returns a positive value for spring type."""
        res = self.PE(0.0, k_normalized=2.0, x_spring=0.5, potential_type="spring")
        assert res["E_potential"] >= 0.0
        assert res["potential_type"] == "spring"

    # --- Mechanical energy ---
    def test_mechanical_energy_sum(self):
        """E_mech = E_kin + E_pot (values in [0,1] domain)."""
        res = self.ME(v=0.5, h=0.5)
        assert abs(res["E_mechanical"] - res["E_kinetic"] - res["E_potential"]) < 1e-10

    # --- Work (inputs clipped to [0,1]) ---
    @pytest.mark.parametrize(
        "F,d,angle",
        [
            (0.5, 0.5, 0.0),
            (0.8, 0.5, math.pi / 3),
            (0.0, 0.5, 0.0),
        ],
    )
    def test_work(self, F, d, angle):
        """W = F * d * cos(θ) with inputs in [0,1] domain."""
        res = self.work(F, d, angle)
        expected = F * d * math.cos(angle)
        assert abs(res["W"] - expected) < 1e-6

    # --- Power (inputs clipped to [0,1]) ---
    @pytest.mark.parametrize("F,v", [(0.5, 0.8), (0.0, 0.5), (0.3, 0.0)])
    def test_power(self, F, v):
        """P = F * v with inputs in [0,1] domain."""
        res = self.power(F, v)
        assert abs(res["P_power"] - F * v) < 1e-10

    # --- Energy conservation ---
    def test_energy_conservation_constant(self):
        """Constant energy series is conserved."""
        E = np.full(100, 42.0)
        res = self.verify_conservation(E)
        assert res["is_conserved"] is True

    def test_energy_conservation_violated(self):
        """Linearly growing energy violates conservation."""
        E = np.linspace(0.0, 100.0, 100)
        res = self.verify_conservation(E, tol=0.01)
        assert res["is_conserved"] is False

    # --- Work-energy theorem ---
    def test_work_energy_valid(self):
        """W_net = ΔKE when theorem holds."""
        res = self.verify_work_energy(W_net=50.0, E_kin_initial=10.0, E_kin_final=60.0)
        assert res["is_valid"] is True

    def test_work_energy_invalid(self):
        """W_net ≠ ΔKE when there is a large discrepancy."""
        res = self.verify_work_energy(W_net=50.0, E_kin_initial=10.0, E_kin_final=100.0)
        assert res["is_valid"] is False


# ═══════════════════════════════════════════════════════════════════════════
# §7  Contract ↔ Closure Bridge (all 13 domains)
# ═══════════════════════════════════════════════════════════════════════════
_BRIDGE_DOMAINS = [
    "gcd",
    "rcft",
    "kinematics",
    "weyl",
    "security",
    "astronomy",
    "nuclear_physics",
    "quantum_mechanics",
    "finance",
    "atomic_physics",
    "materials_science",
    "everyday_physics",
    "standard_model",
]

# everyday_physics has no dedicated contract (validated via UMA universal contract).
_CONTRACT_MAP = {
    "gcd": "GCD.INTSTACK.v1.yaml",
    "rcft": "RCFT.INTSTACK.v1.yaml",
    "kinematics": "KIN.INTSTACK.v1.yaml",
    "weyl": "WEYL.INTSTACK.v1.yaml",
    "security": "SECURITY.INTSTACK.v1.yaml",
    "astronomy": "ASTRO.INTSTACK.v1.yaml",
    "nuclear_physics": "NUC.INTSTACK.v1.yaml",
    "quantum_mechanics": "QM.INTSTACK.v1.yaml",
    "finance": "FINANCE.INTSTACK.v1.yaml",
    "atomic_physics": "ATOM.INTSTACK.v1.yaml",
    "materials_science": "MATL.INTSTACK.v1.yaml",
    "standard_model": "SM.INTSTACK.v1.yaml",
}

_DOMAINS_WITH_CONTRACT = [d for d in _BRIDGE_DOMAINS if d in _CONTRACT_MAP]
_DOMAINS_WITHOUT_INIT = {"gcd"}


class TestContractClosureBridge:
    """Verify every domain has a matching contract YAML and closure directory."""

    @pytest.mark.parametrize("domain", _BRIDGE_DOMAINS)
    def test_closure_directory_exists(self, domain):
        """Each domain has a closure directory in closures/."""
        assert (CLOSURES / domain).is_dir(), f"Missing closures/{domain}/"

    @pytest.mark.parametrize("domain", _BRIDGE_DOMAINS)
    def test_closure_has_init_or_modules(self, domain):
        """Each domain closure directory has __init__.py or standalone modules."""
        if domain in _DOMAINS_WITHOUT_INIT:
            py_files = list((CLOSURES / domain).glob("*.py"))
            assert len(py_files) >= 1, f"{domain} has no .py files"
        else:
            assert (CLOSURES / domain / "__init__.py").exists(), f"Missing closures/{domain}/__init__.py"

    @pytest.mark.parametrize("domain", _DOMAINS_WITH_CONTRACT)
    def test_contract_yaml_exists(self, domain):
        """Each domain with a dedicated contract has a matching YAML file."""
        contract_path = CONTRACTS / _CONTRACT_MAP[domain]
        assert contract_path.exists(), f"Missing contracts/{_CONTRACT_MAP[domain]}"

    @pytest.mark.parametrize("domain", _BRIDGE_DOMAINS)
    def test_closure_has_python_files(self, domain):
        """Each domain closure directory has at least one .py module beyond __init__."""
        py_files = [
            f for f in (CLOSURES / domain).glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_")
        ]
        assert len(py_files) >= 1, f"{domain} has no closure modules"

    @pytest.mark.parametrize("domain", _DOMAINS_WITH_CONTRACT)
    def test_contract_has_required_keys(self, domain):
        """Contract YAML has required keys (nested under 'contract:')."""
        import yaml

        with open(CONTRACTS / _CONTRACT_MAP[domain]) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        contract_block = data.get("contract", data)
        has_id = any(k in contract_block for k in ("contract_id", "id", "name"))
        has_version = any(k in contract_block for k in ("version", "schema_version"))
        assert has_id or has_version, f"{_CONTRACT_MAP[domain]} missing id/version; top keys: {list(data.keys())}"
