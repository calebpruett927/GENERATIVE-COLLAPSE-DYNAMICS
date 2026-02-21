"""
Tests for astronomy closure modules.

Covers 7 files: stellar_luminosity, orbital_mechanics, spectral_analysis,
gravitational_dynamics, cosmology, distance_ladder, stellar_evolution.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/stellar_luminosity.py
# ═══════════════════════════════════════════════════════════════════


class TestStellarLuminosity:
    """Unit tests for compute_stellar_luminosity."""

    def test_sun_values(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=1.0, t_eff=5778.0, r_star=1.0)
        assert result["L_SB"] == pytest.approx(1.0, abs=0.01)
        assert result["regime"] == "Consistent"

    def test_zero_mass(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=0.0, t_eff=5778.0, r_star=1.0)
        assert result["L_predicted"] == 0.0
        assert result["delta_L"] == 1.0

    def test_negative_mass(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=-1.0, t_eff=5778.0, r_star=1.0)
        assert result["L_predicted"] == 0.0

    def test_zero_temperature(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=1.0, t_eff=0.0, r_star=1.0)
        assert result["L_SB"] == 0.0
        assert result["lambda_peak"] == 0.0

    def test_zero_radius(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=1.0, t_eff=5778.0, r_star=0.0)
        assert result["L_SB"] == 0.0

    @pytest.mark.parametrize(
        "m_star,expected_regime",
        [
            (1.0, "Consistent"),
            (10.0, "Significant"),
        ],
    )
    def test_regime_by_mass(self, m_star, expected_regime):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=m_star, t_eff=5778.0, r_star=1.0)
        assert result["regime"] in ("Consistent", "Mild", "Significant", "Anomalous")

    def test_wien_peak(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=1.0, t_eff=5778.0, r_star=1.0)
        expected_peak = 2_897_771.955 / 5778.0
        assert result["lambda_peak"] == pytest.approx(expected_peak, rel=0.01)

    @pytest.mark.parametrize(
        "m_star",
        [0.3, 0.5, 1.0, 3.0, 20.0, 60.0],
    )
    def test_mass_luminosity_piecewise(self, m_star):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=m_star, t_eff=5778.0, r_star=1.0)
        assert result["L_predicted"] > 0.0

    def test_result_keys(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(m_star=1.0, t_eff=5778.0, r_star=1.0)
        assert set(result.keys()) >= {
            "L_predicted",
            "L_SB",
            "delta_L",
            "lambda_peak",
            "regime",
        }

    def test_array_function(self):
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity_array

        results = compute_stellar_luminosity_array(
            masses=[1.0, 2.0],
            temperatures=[5778.0, 8000.0],
            radii=[1.0, 1.5],
        )
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/orbital_mechanics.py
# ═══════════════════════════════════════════════════════════════════


class TestOrbitalMechanics:
    """Unit tests for compute_orbital_mechanics."""

    def test_earth_orbit(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        p_year_s = 365.25 * 24 * 3600
        result = compute_orbital_mechanics(p_orb=p_year_s, a_semi=1.0, m_total=1.0, e_orb=0.017)
        assert result["kepler_residual"] < 0.01
        assert result["regime"] == "Stable"

    def test_circular_orbit_stable(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb=3.156e7, a_semi=1.0, m_total=1.0, e_orb=0.0)
        assert result["regime"] == "Stable"

    def test_eccentric_orbit(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb=3.156e7, a_semi=1.0, m_total=1.0, e_orb=0.5)
        assert result["regime"] == "Eccentric"

    def test_escape_orbit(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb=3.156e7, a_semi=1.0, m_total=1.0, e_orb=0.97)
        assert result["regime"] == "Escape"

    def test_result_keys(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb=3.156e7, a_semi=1.0, m_total=1.0, e_orb=0.0)
        assert set(result.keys()) >= {
            "P_predicted",
            "kepler_residual",
            "v_orb",
            "E_orbital",
            "regime",
        }

    def test_orbital_velocity_positive(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb=3.156e7, a_semi=1.0, m_total=1.0, e_orb=0.0)
        assert result["v_orb"] > 0

    def test_array_function(self):
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics_array

        results = compute_orbital_mechanics_array(
            periods=[3.156e7, 3.156e7],
            semi_axes=[1.0, 5.2],
            masses=[1.0, 1.0],
            eccentricities=[0.017, 0.048],
        )
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/spectral_analysis.py
# ═══════════════════════════════════════════════════════════════════


class TestSpectralAnalysis:
    """Unit tests for compute_spectral_analysis."""

    def test_sun_g2(self):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=5778.0, b_v=0.656, spectral_class="G2")
        assert result["lambda_peak"] > 0
        assert result["T_from_BV"] > 4000

    def test_wien_peak(self):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=10000.0, b_v=0.0, spectral_class="A0")
        expected_peak = 2_897_771.955 / 10000.0
        assert result["lambda_peak"] == pytest.approx(expected_peak, rel=0.01)

    def test_empty_spectral_class(self):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=5778.0, b_v=0.656, spectral_class="")
        assert result["spectral_embedding"] == 0.5

    @pytest.mark.parametrize(
        "spectral_class",
        ["O5", "B3", "A0", "F5", "G2", "K5", "M0"],
    )
    def test_spectral_classes(self, spectral_class):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=5778.0, b_v=0.656, spectral_class=spectral_class)
        assert 0.0 <= result["spectral_embedding"] <= 1.0

    def test_zero_temperature(self):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=0.0, b_v=0.656, spectral_class="G2")
        assert result["lambda_peak"] == 0.0

    def test_result_keys(self):
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff=5778.0, b_v=0.656, spectral_class="G2")
        assert set(result.keys()) >= {
            "lambda_peak",
            "T_from_BV",
            "spectral_embedding",
            "chi2_spectral",
            "regime",
        }


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/gravitational_dynamics.py
# ═══════════════════════════════════════════════════════════════════


class TestGravitationalDynamics:
    """Unit tests for compute_gravitational_dynamics."""

    def test_basic_computation(self):
        from closures.astronomy.gravitational_dynamics import (
            compute_gravitational_dynamics,
        )

        result = compute_gravitational_dynamics(v_rot=220.0, r_obs=8.5, sigma_v=130.0, m_luminous=5e10)
        assert result["M_virial"] > 0
        assert result["M_dynamic"] > 0
        assert 0.0 <= result["dark_matter_fraction"] <= 1.0

    def test_milky_way_like(self):
        from closures.astronomy.gravitational_dynamics import (
            compute_gravitational_dynamics,
        )

        result = compute_gravitational_dynamics(v_rot=220.0, r_obs=8.5, sigma_v=130.0, m_luminous=5e10)
        assert result["dark_matter_fraction"] > 0.5

    @pytest.mark.parametrize(
        "sigma_v,v_rot,expected_regime",
        [
            (130.0, 220.0, "Equilibrium"),
            (50.0, 220.0, "Relaxing"),
            (220.0, 220.0, "Disturbed"),
        ],
    )
    def test_regime_classification(self, sigma_v, v_rot, expected_regime):
        from closures.astronomy.gravitational_dynamics import (
            compute_gravitational_dynamics,
        )

        result = compute_gravitational_dynamics(v_rot=v_rot, r_obs=8.5, sigma_v=sigma_v, m_luminous=5e10)
        assert result["regime"] in ("Equilibrium", "Relaxing", "Disturbed", "Unbound")

    def test_zero_rotation(self):
        from closures.astronomy.gravitational_dynamics import (
            compute_gravitational_dynamics,
        )

        result = compute_gravitational_dynamics(v_rot=0.0, r_obs=8.5, sigma_v=130.0, m_luminous=5e10)
        assert result["M_dynamic"] == 0.0

    def test_result_keys(self):
        from closures.astronomy.gravitational_dynamics import (
            compute_gravitational_dynamics,
        )

        result = compute_gravitational_dynamics(v_rot=220.0, r_obs=8.5, sigma_v=130.0, m_luminous=5e10)
        assert set(result.keys()) >= {
            "M_virial",
            "M_dynamic",
            "dark_matter_fraction",
            "virial_ratio",
            "regime",
        }


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/cosmology.py
# ═══════════════════════════════════════════════════════════════════


class TestCosmology:
    """Unit tests for compute_cosmological_epoch & compute_all_cosmological_epochs."""

    def test_present_day(self):
        from closures.astronomy.cosmology import compute_cosmological_epoch

        result = compute_cosmological_epoch(
            name="Present",
            redshift=0.0,
            H=67.36,
            Omega_b=0.0493,
            Omega_c=0.264,
            Omega_Lambda=0.6847,
            T_cmb=2.7255,
            n_s=0.9649,
            sigma_8=0.8111,
            tau=0.0544,
        )
        assert result.epoch == "Present"
        assert result.F + result.omega == pytest.approx(1.0, abs=1e-5)
        assert result.IC <= result.F + 1e-5

    def test_all_epochs(self):
        from closures.astronomy.cosmology import compute_all_cosmological_epochs

        results = compute_all_cosmological_epochs()
        assert len(results) == 6
        for r in results:
            assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
            assert r.IC <= r.F + 1e-5

    @pytest.mark.parametrize("idx", range(6))
    def test_epoch_tier1_identities(self, idx):
        from closures.astronomy.cosmology import compute_all_cosmological_epochs

        results = compute_all_cosmological_epochs()
        r = results[idx]
        assert r.F + r.omega == pytest.approx(1.0, abs=1e-5)
        assert r.IC <= r.F + 1e-5

    def test_regime_labels(self):
        from closures.astronomy.cosmology import compute_all_cosmological_epochs

        results = compute_all_cosmological_epochs()
        valid_regimes = {"Stable", "Watch", "Collapse"}
        for r in results:
            assert r.regime in valid_regimes

    def test_result_fields(self):
        from closures.astronomy.cosmology import compute_all_cosmological_epochs

        r = compute_all_cosmological_epochs()[0]
        assert hasattr(r, "epoch")
        assert hasattr(r, "F")
        assert hasattr(r, "omega")
        assert hasattr(r, "IC")
        assert hasattr(r, "kappa")
        assert hasattr(r, "S")
        assert hasattr(r, "C")
        assert hasattr(r, "gap")
        assert hasattr(r, "regime")
        assert hasattr(r, "trace")


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/distance_ladder.py
# ═══════════════════════════════════════════════════════════════════


class TestDistanceLadder:
    """Unit tests for compute_distance_ladder."""

    def test_basic_computation(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(m_app=10.0, m_abs=5.0, pi_arcsec=0.01, z_cosmo=0.001)
        assert result["d_modulus"] > 0
        assert result["regime"] in ("High", "Moderate", "Low", "Unreliable")

    def test_distance_modulus_formula(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(m_app=10.0, m_abs=5.0, pi_arcsec=0.0, z_cosmo=0.0)
        expected = 10 ** ((10.0 - 5.0 + 5) / 5)
        assert result["d_modulus"] == pytest.approx(expected, rel=0.01)

    def test_parallax_distance(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(m_app=10.0, m_abs=5.0, pi_arcsec=0.1, z_cosmo=0.0)
        assert result["d_parallax"] == pytest.approx(10.0, rel=0.01)

    def test_zero_parallax(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(m_app=10.0, m_abs=5.0, pi_arcsec=0.0, z_cosmo=0.001)
        assert result["d_parallax"] == 0.0

    def test_result_keys(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(m_app=10.0, m_abs=5.0, pi_arcsec=0.01, z_cosmo=0.001)
        assert set(result.keys()) >= {
            "d_modulus",
            "d_parallax",
            "d_hubble",
            "distance_consistency",
            "regime",
        }

    def test_array_function(self):
        from closures.astronomy.distance_ladder import compute_distance_ladder_array

        results = compute_distance_ladder_array(
            m_apps=[10.0, 12.0],
            m_abss=[5.0, 5.0],
            parallaxes=[0.01, 0.005],
            redshifts=[0.001, 0.002],
        )
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════
# closures/astronomy/stellar_evolution.py
# ═══════════════════════════════════════════════════════════════════


class TestStellarEvolution:
    """Unit tests for compute_stellar_evolution."""

    def test_sun_main_sequence(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=1.0, l_obs=1.0, t_eff=5778.0, age_gyr=4.6)
        assert result["evolutionary_phase"] == "Main-Seq"
        assert result["t_MS"] > 0

    def test_young_star_pre_ms(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=1.0, l_obs=1.0, t_eff=5778.0, age_gyr=0.01)
        assert result["evolutionary_phase"] == "Pre-MS"

    def test_old_star_post_agb(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=1.0, l_obs=1.0, t_eff=5778.0, age_gyr=20.0)
        assert result["evolutionary_phase"] == "Post-AGB"

    @pytest.mark.parametrize(
        "age_gyr,expected_phase",
        [
            (0.001, "Pre-MS"),
            (5.0, "Main-Seq"),
            (9.5, "Subgiant"),
            (12.0, "Giant"),
            (20.0, "Post-AGB"),
        ],
    )
    def test_phase_progression(self, age_gyr, expected_phase):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=1.0, l_obs=1.0, t_eff=5778.0, age_gyr=age_gyr)
        assert result["evolutionary_phase"] == expected_phase

    def test_zero_mass(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=0.0, l_obs=1.0, t_eff=5778.0, age_gyr=5.0)
        assert result["t_MS"] == 0.0

    def test_result_keys(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=1.0, l_obs=1.0, t_eff=5778.0, age_gyr=5.0)
        assert set(result.keys()) >= {
            "t_MS",
            "evolutionary_phase",
            "L_ZAMS",
            "T_ZAMS",
            "regime",
        }
        assert result["regime"] == result["evolutionary_phase"]

    def test_massive_star_short_lifetime(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star=50.0, l_obs=1e5, t_eff=30000.0, age_gyr=5.0)
        assert result["t_MS"] < 0.1

    def test_array_function(self):
        from closures.astronomy.stellar_evolution import compute_stellar_evolution_array

        results = compute_stellar_evolution_array(
            masses=[1.0, 5.0],
            luminosities=[1.0, 500.0],
            temperatures=[5778.0, 15000.0],
            ages=[5.0, 0.05],
        )
        assert len(results) == 2
