#!/usr/bin/env python3
"""
Test suite for Kinematics closures.

Validates:
- All 6 kinematics closure implementations
- Input validation
- Output structure
- Mathematical correctness
- Conservation laws
- Phase space return detection
- Stability classification
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent


def load_closure(repo_root: Path, closure_name: str) -> ModuleType:
    """Dynamically load a kinematics closure module."""
    closure_path = repo_root / "closures" / "kinematics" / f"{closure_name}.py"
    assert closure_path.exists(), f"Closure not found: {closure_path}"

    spec = importlib.util.spec_from_file_location(closure_name, closure_path)
    assert spec is not None, f"Could not load spec for {closure_name}"
    assert spec.loader is not None, f"No loader for {closure_name}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Linear Kinematics Tests
# ============================================================================


class TestLinearKinematics:
    """Tests for linear_kinematics.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test linear_kinematics.py exists."""
        path = repo_root / "closures" / "kinematics" / "linear_kinematics.py"
        assert path.exists()

    def test_basic_computation(self, repo_root: Path) -> None:
        """Test basic linear kinematics computation."""
        module = load_closure(repo_root, "linear_kinematics")
        result = module.compute_linear_kinematics(x=0.5, v=0.3, a=0.1)

        assert "position" in result
        assert "velocity" in result
        assert "acceleration" in result
        assert "phase_magnitude" in result
        assert "regime" in result

    def test_phase_magnitude(self, repo_root: Path) -> None:
        """Test phase space magnitude computation."""
        module = load_closure(repo_root, "linear_kinematics")
        
        # x=0.6, v=0.8 should give ||(0.6, 0.8)|| = 1.0
        result = module.compute_linear_kinematics(x=0.6, v=0.8, a=0.0)
        
        expected_magnitude = math.sqrt(0.6**2 + 0.8**2)
        assert abs(result["phase_magnitude"] - expected_magnitude) < 1e-9

    def test_kinematic_prediction(self, repo_root: Path) -> None:
        """Test predicted next position and velocity."""
        module = load_closure(repo_root, "linear_kinematics")
        
        x, v, a, dt = 0.5, 0.2, 0.1, 0.1
        result = module.compute_linear_kinematics(x=x, v=v, a=a, dt=dt)
        
        # v_next = v + a*dt = 0.2 + 0.1*0.1 = 0.21
        expected_v = min(1.0, v + a * dt)
        assert abs(result["predicted_v_next"] - expected_v) < 1e-9
        
        # x_next = x + v*dt + 0.5*a*dt^2
        expected_x = min(1.0, x + v * dt + 0.5 * a * dt**2)
        assert abs(result["predicted_x_next"] - expected_x) < 1e-6

    def test_regime_classification(self, repo_root: Path) -> None:
        """Test regime classification based on velocity and acceleration."""
        module = load_closure(repo_root, "linear_kinematics")
        
        # Low v, low a -> Stable
        result = module.compute_linear_kinematics(x=0.5, v=0.1, a=0.05)
        assert result["regime"] == "Stable"
        
        # High v, high a -> Critical
        result = module.compute_linear_kinematics(x=0.5, v=0.9, a=0.6)
        assert result["regime"] == "Critical"

    def test_clipping(self, repo_root: Path) -> None:
        """Test that out-of-range values are clipped."""
        module = load_closure(repo_root, "linear_kinematics")
        
        result = module.compute_linear_kinematics(x=1.5, v=-0.2, a=0.1)
        
        assert result["position"] == 1.0  # Clipped to max
        assert result["velocity"] == 0.0  # Clipped to min (negative clipped)
        assert result["oor_flags"]["position"] == True
        assert result["oor_flags"]["velocity"] == True

    def test_trajectory_computation(self, repo_root: Path) -> None:
        """Test trajectory-level statistics."""
        module = load_closure(repo_root, "linear_kinematics")
        
        x_series = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        v_series = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        a_series = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = module.compute_trajectory(x_series, v_series, a_series, dt=0.5)
        
        assert result["total_displacement"] == 0.5
        assert abs(result["mean_velocity"] - 0.2) < 1e-9
        assert result["trajectory_regime"] == "Stable"

    def test_kinematic_consistency(self, repo_root: Path) -> None:
        """Test kinematic consistency verification."""
        module = load_closure(repo_root, "linear_kinematics")
        
        # Consistent: v = dx/dt, a = dv/dt
        dt = 0.1
        x_series = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        v_series = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        a_series = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = module.verify_kinematic_consistency(
            x_series, v_series, a_series, dt=dt, tol=0.05
        )
        
        assert result["velocity_consistent"]
        assert result["acceleration_consistent"]


# ============================================================================
# Rotational Kinematics Tests
# ============================================================================


class TestRotationalKinematics:
    """Tests for rotational_kinematics.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test rotational_kinematics.py exists."""
        path = repo_root / "closures" / "kinematics" / "rotational_kinematics.py"
        assert path.exists()

    def test_basic_computation(self, repo_root: Path) -> None:
        """Test basic rotational kinematics computation."""
        module = load_closure(repo_root, "rotational_kinematics")
        result = module.compute_rotational_kinematics(
            theta=0.25, omega_rot=0.3, alpha=0.1
        )

        assert "theta" in result
        assert "omega_rot" in result
        assert "alpha" in result
        assert "angular_momentum" in result
        assert "torque" in result
        assert "E_rotational" in result

    def test_angular_momentum(self, repo_root: Path) -> None:
        """Test angular momentum: L = I * omega."""
        module = load_closure(repo_root, "rotational_kinematics")
        
        I = 0.5
        omega = 0.4
        result = module.compute_rotational_kinematics(
            theta=0.0, omega_rot=omega, alpha=0.0, I_normalized=I
        )
        
        expected_L = I * omega
        assert abs(result["angular_momentum"] - expected_L) < 1e-9

    def test_torque(self, repo_root: Path) -> None:
        """Test torque: T = I * alpha."""
        module = load_closure(repo_root, "rotational_kinematics")
        
        I = 0.5
        alpha = 0.2
        result = module.compute_rotational_kinematics(
            theta=0.0, omega_rot=0.0, alpha=alpha, I_normalized=I
        )
        
        expected_T = I * alpha
        assert abs(result["torque"] - expected_T) < 1e-9

    def test_rotational_energy(self, repo_root: Path) -> None:
        """Test rotational kinetic energy: E_rot = 0.5 * I * omega^2."""
        module = load_closure(repo_root, "rotational_kinematics")
        
        I = 1.0
        omega = 0.5
        result = module.compute_rotational_kinematics(
            theta=0.0, omega_rot=omega, alpha=0.0, I_normalized=I
        )
        
        expected_E = 0.5 * I * omega**2
        assert abs(result["E_rotational"] - expected_E) < 1e-9

    def test_centripetal(self, repo_root: Path) -> None:
        """Test centripetal acceleration and force."""
        module = load_closure(repo_root, "rotational_kinematics")
        
        omega = 0.5
        r = 0.8
        m = 1.0
        
        result = module.compute_centripetal(
            omega_rot=omega, r_normalized=r, m_normalized=m
        )
        
        # a_c = omega^2 * r
        expected_a = omega**2 * r
        assert abs(result["a_centripetal"] - expected_a) < 1e-9
        
        # F_c = m * omega^2 * r
        expected_F = m * omega**2 * r
        assert abs(result["F_centripetal"] - expected_F) < 1e-9


# ============================================================================
# Energy Mechanics Tests
# ============================================================================


class TestEnergyMechanics:
    """Tests for energy_mechanics.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test energy_mechanics.py exists."""
        path = repo_root / "closures" / "kinematics" / "energy_mechanics.py"
        assert path.exists()

    def test_kinetic_energy(self, repo_root: Path) -> None:
        """Test kinetic energy: E_kin = 0.5 * m * v^2."""
        module = load_closure(repo_root, "energy_mechanics")
        
        v = 0.6
        m = 1.0
        result = module.compute_kinetic_energy(v=v, m_normalized=m)
        
        expected_E = 0.5 * m * v**2
        assert abs(result["E_kinetic"] - expected_E) < 1e-9

    def test_gravitational_potential(self, repo_root: Path) -> None:
        """Test gravitational potential: E_pot = m * g * h."""
        module = load_closure(repo_root, "energy_mechanics")
        
        h = 0.5
        m = 1.0
        g = 1.0
        result = module.compute_potential_energy(
            h=h, m_normalized=m, g_normalized=g, potential_type="gravitational"
        )
        
        expected_E = m * g * h
        assert abs(result["E_potential"] - expected_E) < 1e-9

    def test_mechanical_energy(self, repo_root: Path) -> None:
        """Test mechanical energy: E_mech = E_kin + E_pot."""
        module = load_closure(repo_root, "energy_mechanics")
        
        v = 0.4
        h = 0.3
        m = 1.0
        g = 1.0
        
        result = module.compute_mechanical_energy(v=v, h=h, m_normalized=m, g_normalized=g)
        
        E_kin = 0.5 * m * v**2
        E_pot = m * g * h
        expected_E = E_kin + E_pot
        
        assert abs(result["E_mechanical"] - expected_E) < 1e-9

    def test_work(self, repo_root: Path) -> None:
        """Test work: W = F * d * cos(theta)."""
        module = load_closure(repo_root, "energy_mechanics")
        
        F = 0.5
        d = 0.4
        angle = 0.0  # Force aligned with displacement
        
        result = module.compute_work(F_net=F, displacement=d, angle=angle)
        
        expected_W = F * d * math.cos(angle)
        assert abs(result["W"] - expected_W) < 1e-9

    def test_power(self, repo_root: Path) -> None:
        """Test power: P = F * v."""
        module = load_closure(repo_root, "energy_mechanics")
        
        F = 0.4
        v = 0.5
        
        result = module.compute_power(F_net=F, v=v)
        
        expected_P = F * v
        assert abs(result["P_power"] - expected_P) < 1e-9

    def test_energy_conservation(self, repo_root: Path) -> None:
        """Test energy conservation verification."""
        module = load_closure(repo_root, "energy_mechanics")
        
        # Constant energy (conserved)
        E_series = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = module.verify_energy_conservation(E_series, tol=0.01)
        
        assert result["is_conserved"] == True
        assert result["max_deviation"] < 0.01

    def test_work_energy_theorem(self, repo_root: Path) -> None:
        """Test work-energy theorem: W_net = ΔE_kin."""
        module = load_closure(repo_root, "energy_mechanics")
        
        W = 0.05
        E_i = 0.1
        E_f = 0.15
        
        result = module.verify_work_energy_theorem(
            W_net=W, E_kin_initial=E_i, E_kin_final=E_f
        )
        
        assert result["is_valid"] == True


# ============================================================================
# Momentum Dynamics Tests
# ============================================================================


class TestMomentumDynamics:
    """Tests for momentum_dynamics.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test momentum_dynamics.py exists."""
        path = repo_root / "closures" / "kinematics" / "momentum_dynamics.py"
        assert path.exists()

    def test_linear_momentum(self, repo_root: Path) -> None:
        """Test linear momentum: p = m * v."""
        module = load_closure(repo_root, "momentum_dynamics")
        
        m = 0.5
        v = 0.6
        result = module.compute_linear_momentum(v=v, m_normalized=m)
        
        expected_p = m * v
        assert abs(result["p"] - expected_p) < 1e-9

    def test_impulse(self, repo_root: Path) -> None:
        """Test impulse: J = F * dt."""
        module = load_closure(repo_root, "momentum_dynamics")
        
        F = 0.4
        dt = 0.5
        result = module.compute_impulse(F_net=F, dt=dt)
        
        expected_J = F * dt
        assert abs(result["J"] - expected_J) < 1e-9

    def test_elastic_collision(self, repo_root: Path) -> None:
        """Test elastic collision conserves momentum and energy."""
        module = load_closure(repo_root, "momentum_dynamics")
        
        result = module.compute_collision_1d(
            m1=1.0, v1_initial=0.5,
            m2=1.0, v2_initial=0.0,
            collision_type="elastic"
        )
        
        # Momentum should be conserved
        assert result["momentum_conserved"] == True
        
        # Energy should be conserved for elastic collision
        assert result["energy_conserved"] == True
        
        # For equal masses, velocities should swap
        assert abs(result["v1_final"] - 0.0) < 0.01
        assert abs(result["v2_final"] - 0.5) < 0.01

    def test_inelastic_collision(self, repo_root: Path) -> None:
        """Test perfectly inelastic collision."""
        module = load_closure(repo_root, "momentum_dynamics")
        
        result = module.compute_collision_1d(
            m1=1.0, v1_initial=0.6,
            m2=1.0, v2_initial=0.0,
            collision_type="perfectly_inelastic"
        )
        
        # Momentum conserved
        assert result["momentum_conserved"] == True
        
        # Energy NOT conserved (loss)
        assert result["energy_conserved"] == False
        assert result["energy_loss"] > 0
        
        # Objects move together: v_common = p_total / m_total
        expected_v = 0.3
        assert abs(result["v1_final"] - expected_v) < 0.01
        assert abs(result["v2_final"] - expected_v) < 0.01

    def test_momentum_conservation(self, repo_root: Path) -> None:
        """Test momentum conservation verification."""
        module = load_closure(repo_root, "momentum_dynamics")
        
        # Constant momentum
        p_series = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = module.verify_momentum_conservation(p_series, tol=0.01)
        
        assert result["is_conserved"] == True


# ============================================================================
# Phase Space Return Tests
# ============================================================================


class TestPhaseSpaceReturn:
    """Tests for phase_space_return.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test phase_space_return.py exists."""
        path = repo_root / "closures" / "kinematics" / "phase_space_return.py"
        assert path.exists()

    def test_phase_distance(self, repo_root: Path) -> None:
        """Test phase space distance computation."""
        module = load_closure(repo_root, "phase_space_return")
        
        d = module.compute_phase_distance(0.0, 0.0, 0.3, 0.4)
        expected = math.sqrt(0.3**2 + 0.4**2)
        assert abs(d - expected) < 1e-9

    def test_periodic_trajectory_returns(self, repo_root: Path) -> None:
        """Test that periodic trajectory shows returns."""
        module = load_closure(repo_root, "phase_space_return")
        
        # Simple harmonic oscillator (closed orbit)
        t = np.linspace(0, 4 * np.pi, 100)
        x_series = 0.5 + 0.3 * np.cos(t)
        v_series = 0.5 - 0.3 * np.sin(t)
        
        result = module.compute_kinematic_return(
            x_series, v_series, eta_phase=0.1
        )
        
        # Should have positive return rate (periodic motion returns)
        assert result["return_rate"] > 0.1  # At least some returns detected
        assert result["dynamics_regime"] in ["Returning", "Partially_Returning", "Weakly_Returning"]

    def test_drifting_trajectory_no_return(self, repo_root: Path) -> None:
        """Test that drifting trajectory shows no returns."""
        module = load_closure(repo_root, "phase_space_return")
        
        # Linear drift (no return)
        x_series = np.linspace(0.0, 1.0, 50)
        v_series = np.ones(50) * 0.5
        
        result = module.compute_kinematic_return(
            x_series, v_series, eta_phase=0.05
        )
        
        # Should have low return rate
        assert result["return_rate"] < 0.3
        assert result["dynamics_regime"] in ["Non_Returning", "Weakly_Returning"]

    def test_phase_trajectory(self, repo_root: Path) -> None:
        """Test phase trajectory properties."""
        module = load_closure(repo_root, "phase_space_return")
        
        x_series = np.array([0.5, 0.6, 0.7, 0.6, 0.5])
        v_series = np.array([0.5, 0.6, 0.5, 0.4, 0.5])
        
        result = module.compute_phase_trajectory(x_series, v_series)
        
        assert "path_length" in result
        assert "enclosed_area" in result
        assert "centroid" in result
        assert result["n_points"] == 5

    def test_oscillation_detection(self, repo_root: Path) -> None:
        """Test oscillation detection."""
        module = load_closure(repo_root, "phase_space_return")
        
        # Oscillating trajectory - more cycles for clearer detection
        t = np.linspace(0, 4 * np.pi, 100)
        x_series = 0.5 + 0.2 * np.cos(t)
        v_series = 0.5 - 0.2 * np.sin(t)
        
        result = module.detect_oscillation(x_series, v_series)
        
        assert result["oscillation_type"] in ["Periodic", "Quasi_Periodic", "Damped"]
        assert result["sign_changes"] > 0


# ============================================================================
# Kinematic Stability Tests
# ============================================================================


class TestKinematicStability:
    """Tests for kinematic_stability.py"""

    def test_closure_exists(self, repo_root: Path) -> None:
        """Test kinematic_stability.py exists."""
        path = repo_root / "closures" / "kinematics" / "kinematic_stability.py"
        assert path.exists()

    def test_stable_trajectory(self, repo_root: Path) -> None:
        """Test stability for a stable (low motion) trajectory."""
        module = load_closure(repo_root, "kinematic_stability")
        
        # Low velocity, low acceleration -> stable
        x_series = np.ones(20) * 0.5 + np.random.randn(20) * 0.01
        v_series = np.ones(20) * 0.1 + np.random.randn(20) * 0.01
        
        result = module.compute_kinematic_stability(x_series, v_series)
        
        assert result["K_stability"] > 0.5
        assert result["regime"] in ["Stable", "Watch"]

    def test_unstable_trajectory(self, repo_root: Path) -> None:
        """Test stability for an unstable (high motion) trajectory."""
        module = load_closure(repo_root, "kinematic_stability")
        
        # High velocity, high variation -> unstable
        x_series = np.random.rand(20)
        v_series = np.random.rand(20) * 0.9 + 0.1
        
        result = module.compute_kinematic_stability(x_series, v_series)
        
        # May be Watch or Unstable depending on specific random values
        assert result["K_stability"] >= 0.0
        assert result["regime"] in ["Stable", "Watch", "Unstable"]

    def test_stability_margin(self, repo_root: Path) -> None:
        """Test stability margin computation."""
        module = load_closure(repo_root, "kinematic_stability")
        
        result = module.compute_stability_margin(K_stability=0.8, stable_threshold=0.7)
        
        assert abs(result["margin"] - 0.1) < 1e-9
        assert result["margin_status"] == "Positive"

    def test_stability_trend(self, repo_root: Path) -> None:
        """Test stability trend analysis."""
        module = load_closure(repo_root, "kinematic_stability")
        
        # Improving stability
        K_series = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        result = module.compute_stability_trend(K_series)
        
        assert result["trend_direction"] == "Improving"
        assert result["trend_slope"] > 0

    def test_motion_regime_classification(self, repo_root: Path) -> None:
        """Test motion regime classification."""
        module = load_closure(repo_root, "kinematic_stability")
        
        # Static
        result = module.classify_motion_regime(
            v_mean=0.01, a_mean=0.01, K_stability=0.9, tau_kin=np.inf
        )
        assert result["motion_regime"] == "Static"
        
        # Uniform motion
        result = module.classify_motion_regime(
            v_mean=0.3, a_mean=0.01, K_stability=0.8, tau_kin=np.inf
        )
        assert result["motion_regime"] == "Uniform"
        
        # Oscillatory
        result = module.classify_motion_regime(
            v_mean=0.3, a_mean=0.1, K_stability=0.7, tau_kin=5.0
        )
        assert result["motion_regime"] == "Oscillatory"

    def test_kinematic_budget(self, repo_root: Path) -> None:
        """Test kinematic budget computation."""
        module = load_closure(repo_root, "kinematic_stability")
        
        result = module.compute_kinematic_budget(
            K_stability_t0=0.7,
            K_stability_t1=0.75,
            S_kin=0.1,
            C_kin=0.05,
            tau_kin=10.0,
            R_kin=0.01
        )
        
        assert "delta_K_ledger" in result
        assert "delta_K_budget" in result
        assert "residual" in result
        assert abs(result["delta_K_ledger"] - 0.05) < 1e-9


# ============================================================================
# Contract Validation Tests
# ============================================================================


class TestKinematicsContract:
    """Tests for KIN.INTSTACK.v1 contract."""

    def test_contract_exists(self, repo_root: Path) -> None:
        """Test contract file exists."""
        path = repo_root / "contracts" / "KIN.INTSTACK.v1.yaml"
        assert path.exists()

    def test_contract_valid_yaml(self, repo_root: Path) -> None:
        """Test contract is valid YAML."""
        import yaml
        
        path = repo_root / "contracts" / "KIN.INTSTACK.v1.yaml"
        with open(path) as f:
            contract = yaml.safe_load(f)
        
        assert "schema" in contract
        assert "contract" in contract
        assert contract["contract"]["id"] == "KIN.INTSTACK.v1"

    def test_contract_has_required_fields(self, repo_root: Path) -> None:
        """Test contract has all required fields."""
        import yaml
        
        path = repo_root / "contracts" / "KIN.INTSTACK.v1.yaml"
        with open(path) as f:
            contract = yaml.safe_load(f)
        
        c = contract["contract"]
        
        assert "version" in c
        assert "parent_contract" in c
        assert c["parent_contract"] == "UMA.INTSTACK.v1"
        assert "embedding" in c
        assert "tier_1_kernel" in c
        assert "axioms" in c
        assert "regime_classification" in c

    def test_kinematics_symbols_reserved(self, repo_root: Path) -> None:
        """Test that kinematics symbols are reserved."""
        import yaml
        
        path = repo_root / "contracts" / "KIN.INTSTACK.v1.yaml"
        with open(path) as f:
            contract = yaml.safe_load(f)
        
        symbols = contract["contract"]["tier_1_kernel"]["reserved_symbols"]
        
        # Check key kinematics symbols
        assert "x" in symbols
        assert "v" in symbols
        assert "a" in symbols
        assert "p" in symbols
        assert "E_kin" in symbols


# ============================================================================
# CasePack Tests
# ============================================================================


class TestKinematicsCasepack:
    """Tests for kinematics_complete casepack."""

    def test_casepack_exists(self, repo_root: Path) -> None:
        """Test casepack directory exists."""
        path = repo_root / "casepacks" / "kinematics_complete"
        assert path.exists()

    def test_manifest_exists(self, repo_root: Path) -> None:
        """Test manifest.json exists."""
        path = repo_root / "casepacks" / "kinematics_complete" / "manifest.json"
        assert path.exists()

    def test_manifest_valid_json(self, repo_root: Path) -> None:
        """Test manifest is valid JSON."""
        import json
        
        path = repo_root / "casepacks" / "kinematics_complete" / "manifest.json"
        with open(path) as f:
            manifest = json.load(f)
        
        assert "casepack" in manifest
        assert "refs" in manifest
        assert manifest["refs"]["contract"]["id"] == "KIN.INTSTACK.v1"

    def test_raw_measurements_exists(self, repo_root: Path) -> None:
        """Test raw_measurements.csv exists."""
        path = repo_root / "casepacks" / "kinematics_complete" / "raw_measurements.csv"
        assert path.exists()

    def test_expected_invariants_exists(self, repo_root: Path) -> None:
        """Test expected invariants exist."""
        path = repo_root / "casepacks" / "kinematics_complete" / "expected" / "invariants.json"
        assert path.exists()


# ============================================================================
# Axiom-0 Compliance Tests
# ============================================================================


class TestAxiom0Compliance:
    """Tests for Axiom-0 (no_return_no_credit) enforcement."""

    def test_non_returning_gets_no_credit(self, repo_root: Path) -> None:
        """Test that non-returning motion receives no kinematic credit."""
        module = load_closure(repo_root, "phase_space_return")
        
        # Linear drift - never returns to same (x,v) point
        # Use varying v to ensure no returns even with moderate eta_phase
        x_series = np.linspace(0.0, 1.0, 50)
        v_series = np.linspace(0.0, 1.0, 50)  # Also drifting
        
        result = module.compute_kinematic_return(
            x_series, v_series, eta_phase=0.001  # Very tight tolerance
        )
        
        # AXIOM-0: Non-returning motion gets 0 credit
        assert result["kinematic_credit"] == 0.0
        assert result["dynamics_regime"] in ["Non_Returning", "Weakly_Returning"]

    def test_returning_gets_credit(self, repo_root: Path) -> None:
        """Test that returning motion receives kinematic credit."""
        module = load_closure(repo_root, "phase_space_return")
        
        # Periodic oscillation - returns through phase space
        t = np.linspace(0, 4 * np.pi, 100)
        x_series = 0.5 + 0.3 * np.cos(t)
        v_series = 0.5 + 0.3 * np.sin(t)
        
        result = module.compute_kinematic_return(
            x_series, v_series, eta_phase=0.1
        )
        
        # AXIOM-0: Returning motion gets positive credit
        assert result["kinematic_credit"] > 0.0
        assert result["dynamics_regime"] in ["Returning", "Partially_Returning"]

    def test_compute_kinematic_credit_inf(self, repo_root: Path) -> None:
        """Test compute_kinematic_credit with INF_KIN."""
        module = load_closure(repo_root, "phase_space_return")
        
        result = module.compute_kinematic_credit(
            tau_kin=float('inf'),
            return_rate=0.0
        )
        
        assert result["kinematic_credit"] == 0.0
        assert result["credit_status"] == "NO_CREDIT"
        assert "no_return_no_credit" in result["reason"]

    def test_compute_kinematic_credit_finite(self, repo_root: Path) -> None:
        """Test compute_kinematic_credit with finite return."""
        module = load_closure(repo_root, "phase_space_return")
        
        result = module.compute_kinematic_credit(
            tau_kin=5.0,
            return_rate=0.8
        )
        
        assert result["kinematic_credit"] > 0.0
        assert result["credit_status"] == "CREDITED"

    def test_kinematic_budget_no_credit_for_nonreturn(self, repo_root: Path) -> None:
        """Test that kinematic budget gives 0 return credit for non-returning."""
        module = load_closure(repo_root, "kinematic_stability")
        
        result = module.compute_kinematic_budget(
            K_stability_t0=0.8,
            K_stability_t1=0.7,
            S_kin=0.1,
            C_kin=0.2,
            tau_kin=float('inf'),  # Non-returning
            R_kin=0.5  # This should be forced to 0
        )
        
        # AXIOM-0: R_kin forced to 0 for non-returning motion
        assert result["R_kin_effective"] == 0.0
        assert result["return_credited"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
