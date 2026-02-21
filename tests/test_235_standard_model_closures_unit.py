"""
Tests for standard_model closure modules.

Covers 3 untested files: ckm_mixing, coupling_constants, particle_catalog.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/standard_model/ckm_mixing.py
# ═══════════════════════════════════════════════════════════════════


class TestCKMMixing:
    """Unit tests for compute_ckm_mixing."""

    def test_default_parameters(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert result.lambda_wolf == pytest.approx(0.22650, abs=0.001)
        assert result.A_wolf == pytest.approx(0.790, abs=0.01)

    def test_duality_identity(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert result.F_eff + result.omega_eff == pytest.approx(1.0, abs=0.01)

    def test_jarlskog_invariant(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert result.J_CP > 0
        assert result.J_CP < 1e-3

    def test_unitarity_rows(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert result.unitarity_row1 == pytest.approx(1.0, abs=0.01)
        assert result.unitarity_row2 == pytest.approx(1.0, abs=0.01)

    def test_triangle_angles(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        angles = result.triangle_angles
        assert "alpha_deg" in angles
        assert "beta_deg" in angles
        assert "gamma_deg" in angles
        total = angles["alpha_deg"] + angles["beta_deg"] + angles["gamma_deg"]
        assert total == pytest.approx(180.0, abs=5.0)

    def test_v_matrix_shape(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert len(result.V_matrix) == 3
        assert all(len(row) == 3 for row in result.V_matrix)

    def test_regime_label(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing()
        assert result.regime in ("Unitary", "Tension", "BSM_hint")

    def test_custom_parameters(self):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing(lambda_w=0.225, A=0.8, rho_bar=0.14, eta_bar=0.36)
        assert result.lambda_wolf == pytest.approx(0.225)
        assert result.A_wolf == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "lambda_w,A,rho,eta",
        [
            (0.22650, 0.790, 0.141, 0.357),
            (0.2257, 0.814, 0.135, 0.349),
            (0.230, 0.800, 0.150, 0.350),
        ],
    )
    def test_multiple_parameterizations(self, lambda_w, A, rho, eta):
        from closures.standard_model.ckm_mixing import compute_ckm_mixing

        result = compute_ckm_mixing(lambda_w=lambda_w, A=A, rho_bar=rho, eta_bar=eta)
        assert len(result.V_matrix) == 3
        assert result.J_CP > 0


# ═══════════════════════════════════════════════════════════════════
# closures/standard_model/coupling_constants.py
# ═══════════════════════════════════════════════════════════════════


class TestCouplingConstants:
    """Unit tests for compute_running_coupling."""

    def test_at_mz(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=91.1876)
        assert result.alpha_s == pytest.approx(0.1180, abs=0.01)
        assert result.regime == "Perturbative"

    def test_high_energy_perturbative(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=1000.0)
        assert result.alpha_s < 0.1180
        assert result.regime == "Perturbative"

    def test_low_energy_non_perturbative(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=0.3)
        assert result.alpha_s == pytest.approx(1.0)
        assert result.regime == "NonPerturbative"

    def test_negative_Q_raises(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        with pytest.raises(ValueError):
            compute_running_coupling(Q_GeV=-1.0)

    def test_zero_Q_raises(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        with pytest.raises(ValueError):
            compute_running_coupling(Q_GeV=0.0)

    def test_result_fields(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=91.0)
        assert hasattr(result, "alpha_s")
        assert hasattr(result, "alpha_em")
        assert hasattr(result, "sin2_theta_W")
        assert hasattr(result, "G_F")
        assert hasattr(result, "n_flavors")
        assert hasattr(result, "regime")

    def test_flavor_counting(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        r1 = compute_running_coupling(Q_GeV=1.0)
        r2 = compute_running_coupling(Q_GeV=100.0)
        assert r1.n_flavors <= r2.n_flavors

    @pytest.mark.parametrize(
        "Q_GeV",
        [1.0, 5.0, 10.0, 50.0, 91.1876, 200.0, 1000.0],
    )
    def test_asymptotic_freedom(self, Q_GeV):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=Q_GeV)
        assert result.alpha_s > 0

    def test_alpha_em_at_mz(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=91.1876)
        assert result.alpha_em == pytest.approx(1.0 / 127.952, rel=0.02)

    def test_weinberg_angle(self):
        from closures.standard_model.coupling_constants import compute_running_coupling

        result = compute_running_coupling(Q_GeV=91.1876)
        assert result.sin2_theta_W == pytest.approx(0.23122, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# closures/standard_model/particle_catalog.py
# ═══════════════════════════════════════════════════════════════════


class TestParticleCatalog:
    """Unit tests for particle_catalog."""

    def test_get_electron(self):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle("electron")
        assert p.name == "electron"
        assert p.charge_e == pytest.approx(-1.0)
        assert p.spin == pytest.approx(0.5)

    def test_get_higgs(self):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle("Higgs boson")
        assert p.mass_GeV == pytest.approx(125.25, abs=1.0)
        assert p.spin == 0.0

        with pytest.raises(KeyError):
            get_particle("tachyon")

    def test_list_all_particles(self):
        from closures.standard_model.particle_catalog import list_particles

        particles = list_particles()
        assert len(particles) == 17

    def test_list_quarks(self):
        from closures.standard_model.particle_catalog import list_particles

        quarks = list_particles(category="quark")
        assert len(quarks) == 6

    def test_list_leptons(self):
        from closures.standard_model.particle_catalog import list_particles

        leptons = list_particles(category="lepton")
        assert len(leptons) == 6

    def test_particle_table(self):
        from closures.standard_model.particle_catalog import particle_table

        table = particle_table()
        assert len(table) == 17
        assert all(isinstance(row, dict) for row in table)

    @pytest.mark.parametrize(
        "name",
        [
            "up",
            "down",
            "charm",
            "strange",
            "top",
            "bottom",
            "electron",
            "muon",
            "tau",
            "photon",
            "gluon",
            "W boson",
            "Z boson",
            "Higgs boson",
        ],
    )
    def test_all_particles_accessible(self, name):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle(name)
        assert p.name == name
        assert p.F + p.omega == pytest.approx(1.0, abs=0.02)

    def test_particle_to_dict(self):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle("electron")
        d = p.to_dict()
        assert isinstance(d, dict)
        assert "name" in d
        assert "mass_GeV" in d
        assert "F" in d
        assert "omega" in d

    def test_photon_massless(self):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle("photon")
        assert p.mass_GeV == 0.0
        assert p.charge_e == 0.0

    def test_top_quark_heavy(self):
        from closures.standard_model.particle_catalog import get_particle

        p = get_particle("top")
        assert p.mass_GeV > 170.0
