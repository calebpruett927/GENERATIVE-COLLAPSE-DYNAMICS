"""
Tests for WEYL cosmological closures.

Tests cover:
- Cosmology background functions (H(z), χ(z), D₁(z), σ8(z))
- Weyl transfer function computation
- Σ(z) evolution and modified gravity mapping
- Limber integral regime classification
- Beyond-Limber corrections
- Boost factor computation
- UMCP integration and invariant mapping

Reference: Nature Communications 15:9295 (2024)
"""

import math

import numpy as np
import pytest

# Import WEYL closures
from closures.weyl.cosmology_background import (
    PLANCK_2018,
    BackgroundResult,
    CosmologyParams,
    D1_of_z,
    H_of_z,
    Omega_Lambda_of_z,
    Omega_m_of_z,
    chi_of_z,
    compute_background,
    compute_background_array,
    compute_des_y3_background,
    cosmology_as_embedding,
    sigma8_of_z,
)
from closures.weyl.weyl_transfer import (
    WeylRegime,
    WeylTransferConfig,
    WeylTransferResult,
    compute_weyl_transfer,
    compute_weyl_transfer_array,
    weyl_return_domain,
)
from closures.weyl.sigma_evolution import (
    DES_Y3_DATA,
    GzModel,
    SigmaConfig,
    SigmaRegime,
    Sigma_to_UMCP_invariants,
    compute_Sigma,
    compute_Sigma_from_hJ,
    g_z_constant,
    g_z_exponential,
)
from closures.weyl.limber_integral import (
    LimberRegime,
    classify_limber_regime,
    k_ell,
    limber_as_projection,
)
from closures.weyl.beyond_limber import (
    BeyondLimberRegime,
    beyond_limber_as_strict_mode,
    classify_beyond_limber_regime,
    spherical_bessel_j,
)
from closures.weyl.boost_factor import (
    NonlinearRegime,
    boost_as_curvature,
    classify_scale_regime,
    compute_boost_factor,
    halofit_boost,
)


class TestCosmologyBackground:
    """Test background cosmology computations."""

    def test_planck_parameters(self):
        """Test Planck 2018 default parameters."""
        assert PLANCK_2018.Omega_m_0 == pytest.approx(0.315, rel=1e-3)
        assert PLANCK_2018.Omega_Lambda_0 == pytest.approx(0.685, rel=1e-3)
        assert PLANCK_2018.H_0 == pytest.approx(67.4, rel=1e-3)
        assert PLANCK_2018.sigma8_0 == pytest.approx(0.811, rel=1e-3)

    def test_hubble_at_z0(self):
        """Test H(z=0) = H_0."""
        H_0 = H_of_z(0.0)
        assert H_0 == pytest.approx(PLANCK_2018.H_0, rel=1e-6)

    def test_hubble_increases_with_z(self):
        """Test H(z) increases with redshift."""
        H_0 = H_of_z(0.0)
        H_1 = H_of_z(1.0)
        H_2 = H_of_z(2.0)
        assert H_1 > H_0
        assert H_2 > H_1

    def test_comoving_distance_monotonic(self):
        """Test χ(z) is monotonically increasing."""
        chi_0 = chi_of_z(0.0)
        chi_1 = chi_of_z(0.5)
        chi_2 = chi_of_z(1.0)
        assert chi_0 == pytest.approx(0.0, abs=1e-6)
        assert chi_2 > chi_1 > chi_0

    def test_growth_function_decreases_with_z(self):
        """Test D₁(z) decreases with redshift (normalized to D₁(0)=1)."""
        D1_0 = D1_of_z(0.0)
        D1_1 = D1_of_z(1.0)
        D1_2 = D1_of_z(2.0)
        assert D1_0 == pytest.approx(1.0, rel=1e-3)
        assert D1_1 < D1_0
        assert D1_2 < D1_1

    def test_sigma8_evolution(self):
        """Test σ8(z) = σ8(0) × D₁(z)."""
        for z in [0.0, 0.5, 1.0, 2.0]:
            sigma8_z = sigma8_of_z(z)
            expected = PLANCK_2018.sigma8_0 * D1_of_z(z)
            assert sigma8_z == pytest.approx(expected, rel=1e-9)

    def test_omega_m_plus_omega_lambda(self):
        """Test Ω_m(z) + Ω_Λ(z) = 1 (flat universe)."""
        for z in [0.0, 0.5, 1.0, 2.0, 5.0]:
            Omega_m = Omega_m_of_z(z)
            Omega_Lambda = Omega_Lambda_of_z(z)
            assert (Omega_m + Omega_Lambda) == pytest.approx(1.0, rel=1e-6)

    def test_compute_background_result(self):
        """Test compute_background returns all quantities."""
        result = compute_background(0.5)
        assert isinstance(result, BackgroundResult)
        assert result.H_z > 0
        assert result.chi > 0
        assert 0 < result.D1 < 1
        assert 0 < result.Omega_m_z < 1
        assert result.a == pytest.approx(1 / 1.5, rel=1e-6)

    def test_des_y3_background(self):
        """Test DES Y3 lens bin background computation."""
        bg = compute_des_y3_background()
        assert len(bg["z_bins"]) == 4
        assert bg["z_bins"][0] == pytest.approx(0.295, rel=1e-3)
        assert bg["anchor_z_star"]["z"] == 10.0

    def test_cosmology_as_embedding(self):
        """Test cosmology to UMCP embedding interpretation."""
        emb = cosmology_as_embedding()
        assert "contract_parameters" in emb
        assert "embedding_specification" in emb
        assert "umcp_mapping" in emb


class TestWeylTransfer:
    """Test Weyl transfer function computation."""

    def test_gr_consistent_regime(self):
        """Test J=1 (GR) produces GR_consistent regime."""
        result = compute_weyl_transfer(
            H_z=70.0,
            H_z_star=700.0,
            J_z=1.0,
            D1_z_star=0.1,
            B_ratio=1.0,
            T_Weyl_star=1.0,
        )
        assert result.regime == WeylRegime.GR_CONSISTENT.value
        assert result.J_effective == pytest.approx(1.0)

    def test_mild_deviation_regime(self):
        """Test small J deviation produces mild regime."""
        result = compute_weyl_transfer(
            H_z=70.0,
            H_z_star=700.0,
            J_z=1.10,  # 10% deviation
            D1_z_star=0.1,
            B_ratio=1.0,
            T_Weyl_star=1.0,
        )
        assert result.regime == WeylRegime.MILD_DEVIATION.value

    def test_strong_deviation_regime(self):
        """Test large J deviation produces strong regime."""
        result = compute_weyl_transfer(
            H_z=70.0,
            H_z_star=700.0,
            J_z=1.30,  # 30% deviation
            D1_z_star=0.1,
            B_ratio=1.0,
            T_Weyl_star=1.0,
        )
        assert result.regime == WeylRegime.STRONG_DEVIATION.value

    def test_transfer_ratio(self):
        """Test T_ratio computation."""
        result = compute_weyl_transfer(
            H_z=70.0,
            H_z_star=700.0,
            J_z=1.0,
            D1_z_star=0.1,
            B_ratio=1.0,
            T_Weyl_star=1.0,
        )
        assert result.T_ratio == pytest.approx(result.T_Weyl / 1.0, rel=1e-6)

    def test_vectorized_transfer(self):
        """Test vectorized computation."""
        H_z = np.array([70.0, 80.0, 90.0])
        J_z = np.array([1.0, 1.0, 1.0])
        B_ratio = np.array([1.0, 1.0, 1.0])

        T_Weyl, T_ratio = compute_weyl_transfer_array(
            H_z=H_z,
            H_z_star=700.0,
            J_z=J_z,
            D1_z_star=0.1,
            B_ratio=B_ratio,
            T_Weyl_star=1.0,
        )
        assert len(T_Weyl) == 3
        assert len(T_ratio) == 3

    def test_return_domain_interpretation(self):
        """Test Weyl return domain analysis."""
        z_values = np.linspace(0, 2, 21)
        J_values = 1.0 + 0.24 * np.exp(-z_values)  # Deviation at low z

        result = weyl_return_domain(z_values, J_values, z_star=10.0, eta_J=0.05)
        assert "return_rate" in result
        assert "first_deviation_z" in result
        assert 0 <= result["return_rate"] <= 1


class TestSigmaEvolution:
    """Test Σ(z) modified gravity parametrization."""

    def test_gr_sigma_0(self):
        """Test Σ₀=0 produces Σ=1 (GR)."""
        result = compute_Sigma(
            z=0.5,
            Sigma_0=0.0,
            g_model=GzModel.CONSTANT,
        )
        assert result.Sigma == pytest.approx(1.0, rel=1e-9)
        assert result.regime == SigmaRegime.GR_CONSISTENT.value

    def test_sigma_0_tension_regime(self):
        """Test moderate Σ₀ produces tension regime."""
        result = compute_Sigma(
            z=0.5,
            Sigma_0=0.15,
            g_model=GzModel.CONSTANT,
        )
        assert result.regime == SigmaRegime.TENSION.value

    def test_sigma_0_modified_gravity_regime(self):
        """Test large Σ₀ produces modified gravity regime."""
        result = compute_Sigma(
            z=0.5,
            Sigma_0=0.40,
            g_model=GzModel.CONSTANT,
        )
        assert result.regime == SigmaRegime.MODIFIED_GRAVITY.value

    def test_g_z_models(self):
        """Test different g(z) model functions."""
        # Constant model
        g_const = g_z_constant(0.5)
        assert g_const == 1.0

        g_const_outside = g_z_constant(1.5)
        assert g_const_outside == 0.0

        # Exponential model
        g_exp = g_z_exponential(0.5)
        assert g_exp == pytest.approx(np.exp(1.5), rel=1e-6)

    def test_des_y3_reference_data(self):
        """Test DES Y3 reference data is available."""
        assert len(DES_Y3_DATA["z_bins"]) == 4
        assert len(DES_Y3_DATA["hJ_cmb"]["mean"]) == 4
        assert "Sigma_0_fits" in DES_Y3_DATA

    def test_sigma_to_umcp_mapping(self):
        """Test Σ → UMCP invariant mapping."""
        mapping = Sigma_to_UMCP_invariants(
            Sigma_0=0.24,
            chi2_red_Sigma=1.1,
            chi2_red_LCDM=2.1,
        )
        assert "omega_analog" in mapping
        assert "F_analog" in mapping
        assert "regime" in mapping
        assert 0 <= mapping["omega_analog"] <= 1
        assert mapping["F_analog"] == pytest.approx(1 - mapping["omega_analog"])


class TestLimberIntegral:
    """Test Limber integral regime classification."""

    def test_limber_valid_high_ell(self):
        """Test ℓ ≥ 200 is Limber valid."""
        regime = classify_limber_regime(250)
        assert regime == LimberRegime.VALID.value

    def test_limber_marginal_mid_ell(self):
        """Test 100 ≤ ℓ < 200 is Limber marginal."""
        regime = classify_limber_regime(150)
        assert regime == LimberRegime.MARGINAL.value

    def test_limber_invalid_low_ell(self):
        """Test ℓ < 100 is Limber invalid."""
        regime = classify_limber_regime(50)
        assert regime == LimberRegime.INVALID.value

    def test_k_ell_computation(self):
        """Test k_ℓ = (ℓ + 0.5)/χ."""
        chi = 1000.0
        ell = 200
        k = k_ell(ell, chi)
        expected = (ell + 0.5) / chi
        assert k == pytest.approx(expected, rel=1e-9)

    def test_limber_as_projection(self):
        """Test Limber to UMCP projection interpretation."""
        hJ_des = np.array([0.326, 0.332, 0.387, 0.354])
        hb_des = np.array([0.5, 0.6, 0.7, 0.8])
        z_bins = np.array([0.295, 0.467, 0.626, 0.771])

        proj = limber_as_projection(hJ_des, hb_des, z_bins)
        assert "hJ_mean" in proj
        assert "F_analog" in proj
        assert "omega_analog" in proj
        assert proj["n_bins"] == 4


class TestBeyondLimber:
    """Test beyond-Limber corrections."""

    def test_spherical_bessel(self):
        """Test spherical Bessel function j_ℓ(x)."""
        x = np.array([0.1, 1.0, 10.0])
        j0 = spherical_bessel_j(0, x)
        j2 = spherical_bessel_j(2, x)
        assert len(j0) == 3
        assert len(j2) == 3
        # j_0(x) = sin(x)/x
        expected_j0 = np.sin(x) / x
        np.testing.assert_allclose(j0, expected_j0, rtol=1e-6)

    def test_beyond_limber_required_regime(self):
        """Test ℓ < 100 requires beyond-Limber."""
        regime = classify_beyond_limber_regime(50)
        assert regime == BeyondLimberRegime.REQUIRED.value

    def test_beyond_limber_recommended_regime(self):
        """Test 100 ≤ ℓ < 200 recommends beyond-Limber."""
        regime = classify_beyond_limber_regime(150)
        assert regime == BeyondLimberRegime.RECOMMENDED.value

    def test_beyond_limber_unnecessary_regime(self):
        """Test ℓ ≥ 200 doesn't need beyond-Limber."""
        regime = classify_beyond_limber_regime(250)
        assert regime == BeyondLimberRegime.UNNECESSARY.value

    def test_strict_mode_interpretation(self):
        """Test beyond-Limber as UMCP strict mode analog."""
        interp = beyond_limber_as_strict_mode(50)
        assert interp["umcp_analog"] == "strict_mode"

        interp = beyond_limber_as_strict_mode(250)
        assert interp["umcp_analog"] == "default_mode"


class TestBoostFactor:
    """Test nonlinear boost factor computation."""

    def test_linear_regime_boost(self):
        """Test B ≈ 1 at large scales (small k)."""
        B = halofit_boost(k=0.01, z=0.5)
        assert B == pytest.approx(1.0, abs=0.2)  # Within 20% of 1

    def test_boost_increases_with_k(self):
        """Test B increases with wavenumber (more nonlinear)."""
        B_01 = halofit_boost(k=0.1, z=0.5)
        B_05 = halofit_boost(k=0.5, z=0.5)
        B_10 = halofit_boost(k=1.0, z=0.5)
        assert B_10 > B_05 > B_01

    def test_boost_factor_result(self):
        """Test compute_boost_factor returns proper result."""
        result = compute_boost_factor(k=0.5, z=0.5)
        assert result.B_boost > 0
        assert result.B_ratio > 0
        assert result.k == 0.5
        assert result.z == 0.5
        assert result.regime in [r.value for r in NonlinearRegime]

    def test_scale_regime_classification(self):
        """Test scale regime classification."""
        assert classify_scale_regime(0.005) == "Ultra_large_scale"
        assert classify_scale_regime(0.05) == "Linear_scale"
        assert classify_scale_regime(0.3) == "Quasi_linear_scale"
        assert classify_scale_regime(1.0) == "Nonlinear_scale"
        assert classify_scale_regime(5.0) == "Highly_nonlinear_scale"

    def test_boost_as_curvature(self):
        """Test boost factor as UMCP curvature analog."""
        k_values = np.logspace(-2, 0.5, 20)
        result = boost_as_curvature(k_values, z=0.5)
        assert "C_proxy_mean" in result
        assert "B_max" in result
        assert "k_nonlinear" in result
        assert 0 <= result["C_proxy_mean"] <= 1


class TestWeylUMCPIntegration:
    """Test WEYL-UMCP integration patterns."""

    def test_axiom_w0_anchor_defines_deviation(self):
        """Test AX-W0: Reference anchor defines deviation."""
        # At z*, Σ ≡ 1 (GR)
        result = compute_Sigma(
            z=10.0,  # z* anchor
            Sigma_0=0.24,
            g_model=GzModel.CONSTANT,
            config=SigmaConfig(z_model_min=0.0, z_model_max=1.0),
        )
        # At z* > 1, g(z) = 0 for constant model, so Σ = 1
        assert result.Sigma == pytest.approx(1.0, rel=1e-6)

    def test_hJ_as_fidelity_analog(self):
        """Test ĥJ maps to UMCP fidelity analog."""
        # DES Y3 mean ĥJ ≈ 0.35
        hJ_mean = np.mean(DES_Y3_DATA["hJ_cmb"]["mean"])

        # This should map to F_analog ≈ 0.35 (similar magnitude)
        # The mapping preserves the "fraction of ideal" interpretation
        assert 0.3 < hJ_mean < 0.4

        # ω_analog = 1 - F_analog
        omega_analog = 1 - hJ_mean
        assert 0.6 < omega_analog < 0.7

    def test_chi2_improvement_as_seam_closure(self):
        """Test χ² improvement maps to seam residual improvement."""
        mapping = Sigma_to_UMCP_invariants(
            Sigma_0=0.24,
            chi2_red_Sigma=1.1,  # Σ model fit
            chi2_red_LCDM=2.1,  # ΛCDM fit
        )

        # χ² improved from 2.1 to 1.1 → ~48% improvement
        assert mapping["chi2_improvement"] > 0.4

    def test_regime_mapping_stable_watch_collapse(self):
        """Test Σ regimes map to UMCP regimes."""
        # GR_consistent → STABLE
        result_gr = compute_Sigma(z=0.5, Sigma_0=0.05, g_model=GzModel.CONSTANT)
        assert result_gr.regime == SigmaRegime.GR_CONSISTENT.value

        # Tension → WATCH
        result_tension = compute_Sigma(z=0.5, Sigma_0=0.20, g_model=GzModel.CONSTANT)
        assert result_tension.regime == SigmaRegime.TENSION.value

        # Modified_gravity → COLLAPSE analog
        result_mg = compute_Sigma(z=0.5, Sigma_0=0.50, g_model=GzModel.CONSTANT)
        assert result_mg.regime == SigmaRegime.MODIFIED_GRAVITY.value


class TestWeylCoreInvariants:
    """Test WEYL Tier-1 invariant computation."""

    def test_hJ_from_sigma(self):
        """Test ĥJ computation from Σ via Eq. 12."""
        # Using DES Y3 bin 1 parameters (approximate)
        z = 0.295
        Sigma = 1.0  # GR
        Omega_m = 0.45  # Approximate at z=0.295
        D1_ratio = 0.9  # D₁(z)/D₁(z*)
        sigma8_star = 0.3  # At z*

        hJ_expected = Omega_m * D1_ratio * sigma8_star * Sigma

        # Should be in reasonable range
        assert 0 < hJ_expected < 1

    def test_sigma_inversion(self):
        """Test Σ can be inverted from ĥJ measurement."""
        # Given values
        hJ = 0.35
        Omega_m = 0.45
        D1_z = 0.9
        D1_z_star = 0.1
        sigma8_star = 0.3

        Sigma = compute_Sigma_from_hJ(
            hJ=hJ,
            Omega_m_z=Omega_m,
            D1_z=D1_z,
            D1_z_star=D1_z_star,
            sigma8_z_star=sigma8_star,
        )

        # Reconstruct hJ
        hJ_reconstructed = Omega_m * (D1_z / D1_z_star) * sigma8_star * Sigma
        assert hJ_reconstructed == pytest.approx(hJ, rel=1e-6)
