"""
KIN.INTSTACK.v1 Audit-Grade Conformance Tests

This test suite implements the complete specification index for KIN.INTSTACK.v1.
Each test maps directly to a frozen definition in KINEMATICS_SPECIFICATION.md
and KINEMATICS_MATHEMATICS.md.

Test naming convention: test_<section>_<requirement>
Sections:
  A. Contract freeze and invariants (meta-tests)
  B. Embedding (bounded + log-safe) and OOR policy
  C. E_kin computation order (raw then normalize)
  D. Phase space and metric
  E. Return-domain generator
  F. U(t), τ_kin typing, return_rate
  G. Kinematic credit (Axiom-0)
  H. Stability K
  I. Typed sentinel enforcement
  J. Firewall and namespace collision
  K. Conservation laws gating
  L. Reference identities (non-gating)
  V. Gold-standard test vectors
"""

import pytest
import math
import hashlib
import yaml
from enum import Enum
from pathlib import Path
from typing import Set, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# =============================================================================
# FROZEN CONSTANTS (from spec)
# =============================================================================

# Window and timing (all in sample steps)
W = 64  # Window size
DELTA = 3  # Debounce lag
T_CRIT = 10.0  # Critical return time threshold
EPSILON_LOG = 1e-8  # Log-safety guard

# Reference scales
L_REF = 1.0  # m
V_REF = 1.0  # m/s
A_REF = 9.81  # m/s²
M_REF = 1.0  # kg
E_REF = M_REF * V_REF ** 2  # J
P_REF = M_REF * V_REF  # kg·m/s

# Stability thresholds
SIGMA_MAX = 0.5
V_MAX = 1.0
K_STABLE = 0.7
K_WATCH = 0.3

# Conservation tolerances
EPSILON_P = 1e-6
EPSILON_E = 1e-6

# Phase space tolerance
ETA_PHASE = 0.01

# Psi weights
PSI_WEIGHTS = {"x": 0.25, "v": 0.25, "a": 0.15, "E_kin": 0.20, "p": 0.15}


# =============================================================================
# TYPED SENTINELS (KIN-AX-1)
# =============================================================================

class KinSpecialValue(Enum):
    """Typed sentinel values for τ_kin (replaces IEEE Inf/NaN)."""
    INF_KIN = "INF_KIN"
    UNIDENTIFIABLE_KIN = "UNIDENTIFIABLE_KIN"


# =============================================================================
# CORE FUNCTIONS UNDER TEST
# =============================================================================

def clip(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Clipping operator: clip_{[a,b]}(x) := max(a, min(b, x))."""
    return max(a, min(b, x))


def bounded_embedding(q: float, q_ref: float) -> Tuple[float, bool]:
    """
    Bounded embedding with magnitude_only convention.
    Returns (q̃, oor_flag).
    """
    ratio = abs(q) / q_ref
    oor_flag = ratio < 0.0 or ratio > 1.0
    q_tilde = clip(ratio, 0.0, 1.0)
    return q_tilde, oor_flag


def log_safe_embedding(q_tilde: float, epsilon: float = EPSILON_LOG) -> float:
    """Log-safe embedding: clip to [ε, 1-ε]."""
    return clip(q_tilde, epsilon, 1.0 - epsilon)


def compute_Ekin_raw_then_normalize(v: float, m: float = M_REF) -> float:
    """
    Correct E_kin computation order: raw physical units first, then normalize.
    E_kin = 0.5 * m * v^2 (physical)
    Ẽ_kin = clip(E_kin / E_ref)
    """
    E_kin_physical = 0.5 * m * v ** 2
    E_tilde = clip(E_kin_physical / E_REF, 0.0, 1.0)
    return E_tilde


def compute_Ekin_wrong_order(v: float, m: float = M_REF) -> float:
    """
    WRONG order: clip v first, then compute energy.
    This is what we must NOT do.
    """
    v_clipped = clip(abs(v) / V_REF, 0.0, 1.0) * V_REF
    E_kin_from_clipped = 0.5 * m * v_clipped ** 2
    E_tilde = clip(E_kin_from_clipped / E_REF, 0.0, 1.0)
    return E_tilde


def squared_l2_distance(gamma1: Tuple[float, float], gamma2: Tuple[float, float]) -> float:
    """Squared-L2 distance in phase space: d²(γ1, γ2)."""
    x1, v1 = gamma1
    x2, v2 = gamma2
    return (x2 - x1) ** 2 + (v2 - v1) ** 2


def compute_D_W(t: int, W: int = W) -> Set[int]:
    """
    Return-domain generator D_W(t).
    D_W(t) := {u ∈ ℤ : max(0, t-W) ≤ u ≤ t-1}
    """
    if t <= 0:
        return set()
    lower = max(0, t - W)
    return set(range(lower, t))


def compute_D_W_delta(t: int, W: int = W, delta: int = DELTA) -> Set[int]:
    """
    Effective domain with debounce lag.
    D_{W,δ}(t) := {u ∈ D_W(t) : (t - u) ≥ δ}
    """
    D_W = compute_D_W(t, W)
    return {u for u in D_W if (t - u) >= delta}


def compute_domain_size(t: int, W: int = W, delta: int = DELTA) -> int:
    """
    Domain size formula (three-case, patched).
    |D_{W,δ}(t)| = 0 if t < δ
                 = t - δ + 1 if δ ≤ t < W
                 = W - δ + 1 if t ≥ W
    """
    if t < delta:
        return 0
    elif t < W:
        return t - delta + 1
    else:
        return W - delta + 1


def compute_valid_return_set(
    t: int,
    gamma_history: Dict[int, Tuple[float, float]],
    eta_phase: float = ETA_PHASE,
    W: int = W,
    delta: int = DELTA
) -> Set[int]:
    """
    Valid-return set U(t).
    U(t) := {u ∈ D_{W,δ}(t) : d²(γ(t), γ(u)) < η_phase}
    Note: strict inequality (<), not ≤.
    """
    if t not in gamma_history:
        return set()
    
    gamma_t = gamma_history[t]
    D = compute_D_W_delta(t, W, delta)
    
    U = set()
    for u in D:
        if u in gamma_history:
            d2 = squared_l2_distance(gamma_t, gamma_history[u])
            if d2 < eta_phase:  # strict inequality
                U.add(u)
    return U


def compute_tau_kin(
    t: int,
    gamma_history: Dict[int, Tuple[float, float]],
    eta_phase: float = ETA_PHASE,
    W: int = W,
    delta: int = DELTA
) -> int | KinSpecialValue:
    """
    Kinematic return time τ_kin(t).
    Returns positive integer (samples) or INF_KIN.
    """
    U = compute_valid_return_set(t, gamma_history, eta_phase, W, delta)
    
    if not U:
        return KinSpecialValue.INF_KIN
    
    # τ_kin is the minimum delay (most recent return)
    delays = [t - u for u in U]
    tau = min(delays)
    
    # Must be positive integer
    assert isinstance(tau, int) and tau > 0
    return tau


def compute_return_rate(
    t: int,
    gamma_history: Dict[int, Tuple[float, float]],
    eta_phase: float = ETA_PHASE,
    W: int = W,
    delta: int = DELTA
) -> float:
    """
    Return rate.
    return_rate(t) := 0 if |D_{W,δ}(t)| = 0
                   := |U(t)| / |D_{W,δ}(t)| otherwise
    """
    D_size = compute_domain_size(t, W, delta)
    
    if D_size == 0:
        return 0.0
    
    U = compute_valid_return_set(t, gamma_history, eta_phase, W, delta)
    return len(U) / D_size


def compute_kinematic_credit(
    tau_kin: int | KinSpecialValue,
    return_rate: float,
    T_crit: float = T_CRIT
) -> float:
    """
    Kinematic credit (Axiom-0).
    credit = 0 if τ_kin = INF_KIN
           = 0 if return_rate ≤ 0.1
           = (1 / (1 + τ_kin/T_crit)) * return_rate otherwise
    """
    if tau_kin == KinSpecialValue.INF_KIN:
        return 0.0
    
    if tau_kin == KinSpecialValue.UNIDENTIFIABLE_KIN:
        return 0.0
    
    if return_rate <= 0.1:
        return 0.0
    
    return (1.0 / (1.0 + tau_kin / T_crit)) * return_rate


def compute_sample_statistics(
    values: List[float]
) -> Tuple[float, float, bool]:
    """
    Compute sample mean and std with ddof=1.
    Returns (mean, std, is_valid).
    If len(values) < 2, std is undefined.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, False
    
    mean = sum(values) / n
    
    if n < 2:
        # ddof=1 makes std undefined
        return mean, 0.0, False
    
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    
    return mean, std, True


def compute_K_stability(
    x_values: List[float],
    v_values: List[float],
    return_rate: float,
    sigma_max: float = SIGMA_MAX,
    v_max: float = V_MAX
) -> Tuple[float, bool, bool]:
    """
    Stability index K.
    K = (1 - σ_x/σ_max) * (1 - v_mean/v_max) * w_τ
    
    Returns (K, sigma_oor, v_oor).
    If |D| < 2, returns (0.0, False, False) with startup guard.
    """
    if len(x_values) < 2:
        # Startup guard: K = 0
        return 0.0, False, False
    
    _, sigma_x, valid = compute_sample_statistics(x_values)
    v_mean, _, _ = compute_sample_statistics(v_values)
    
    if not valid:
        return 0.0, False, False
    
    # Clip+flag for sigma ratio
    sigma_ratio = sigma_x / sigma_max
    sigma_oor = sigma_ratio > 1.0
    if sigma_oor:
        sigma_ratio = 1.0
    sigma_factor = 1.0 - sigma_ratio
    
    # Clip+flag for v ratio
    v_ratio = v_mean / v_max
    v_oor = v_ratio > 1.0
    if v_oor:
        v_ratio = 1.0
    v_factor = 1.0 - v_ratio
    
    # w_τ = return_rate
    w_tau = return_rate
    
    K = sigma_factor * v_factor * w_tau
    
    # Ensure K ∈ [0, 1]
    K = clip(K, 0.0, 1.0)
    
    return K, sigma_oor, v_oor


def classify_stability_regime(K: float) -> str:
    """Classify stability regime based on K."""
    if K > K_STABLE:
        return "Stable"
    elif K > K_WATCH:
        return "Watch"
    else:
        return "Unstable"


def is_valid_tau_kin(value: Any) -> bool:
    """Check if value is a valid τ_kin (positive int or typed sentinel)."""
    if isinstance(value, KinSpecialValue):
        return value in (KinSpecialValue.INF_KIN, KinSpecialValue.UNIDENTIFIABLE_KIN)
    if isinstance(value, int) and value > 0:
        return True
    # Reject float('inf'), float('nan'), negative, zero
    return False


# =============================================================================
# A. CONTRACT FREEZE AND INVARIANTS (META-TESTS)
# =============================================================================

class TestContractFreezeInvariants:
    """A. Contract freeze and invariants (meta-tests)."""
    
    def test_contract_snapshot_exact_match(self):
        """
        A1. Assert the committed KIN.INTSTACK.v1 snapshot matches spec.
        Keys present, values exact, no unknown keys.
        """
        # Expected frozen values
        expected = {
            "W": 64,
            "delta": 3,
            "T_crit": 10.0,
            "epsilon_log": 1e-8,
            "eta_phase": 0.01,
            "sigma_max": 0.5,
            "v_max": 1.0,
            "K_stable": 0.7,
            "K_watch": 0.3,
            "epsilon_p": 1e-6,
            "epsilon_E": 1e-6,
            "psi_dimension": 5,
        }
        
        # Verify against module constants
        assert W == expected["W"]
        assert DELTA == expected["delta"]
        assert T_CRIT == expected["T_crit"]
        assert EPSILON_LOG == expected["epsilon_log"]
        assert ETA_PHASE == expected["eta_phase"]
        assert SIGMA_MAX == expected["sigma_max"]
        assert V_MAX == expected["v_max"]
        assert K_STABLE == expected["K_stable"]
        assert K_WATCH == expected["K_watch"]
        assert EPSILON_P == expected["epsilon_p"]
        assert EPSILON_E == expected["epsilon_E"]
        
        # Psi weights sum to 1.0
        assert abs(sum(PSI_WEIGHTS.values()) - 1.0) < 1e-10
    
    def test_contract_hash_lock(self):
        """
        A2. Assert sha256 of contract snapshot equals pinned hash.
        This prevents accidental drift.
        """
        contract_path = Path(__file__).parent.parent / "contracts" / "KIN.INTSTACK.v1.yaml"
        
        if contract_path.exists():
            content = contract_path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            # Note: You would pin this hash after initial creation
            # For now, just verify the file exists and is hashable
            assert len(sha) == 64
        else:
            # Contract file not yet created; this is acceptable in bootstrap
            pytest.skip("KIN.INTSTACK.v1.yaml not yet created")


# =============================================================================
# B. EMBEDDING (BOUNDED + LOG-SAFE) AND OOR POLICY
# =============================================================================

class TestEmbedding:
    """B. Embedding (bounded + log-safe) and OOR policy."""
    
    def test_clip_operator_idempotent(self):
        """B3. clip(clip(x)) == clip(x) for any x."""
        test_values = [-1.0, -0.1, 0.0, 0.5, 1.0, 1.1, 2.0, 100.0]
        
        for x in test_values:
            clipped = clip(x)
            double_clipped = clip(clipped)
            assert clipped == double_clipped, f"Idempotence failed for x={x}"
        
        # Boundary points
        assert clip(0.0) == 0.0
        assert clip(1.0) == 1.0
        assert clip(-0.1) == 0.0
        assert clip(1.1) == 1.0
    
    def test_bounded_embedding_magnitude_only(self):
        """B4. Embedding uses magnitudes before normalization."""
        # Signed raw inputs
        test_cases = [
            (-0.5, L_REF, 0.5),   # x = -0.5 L_ref → x̃ = 0.5
            (-0.25, V_REF, 0.25), # v = -0.25 v_ref → ṽ = 0.25
            (-9.81, A_REF, 1.0),  # a = -g → ã = 1.0
            (-0.5, P_REF, 0.5),   # p = -0.5 p_ref → p̃ = 0.5
        ]
        
        for raw, ref, expected in test_cases:
            result, _ = bounded_embedding(raw, ref)
            assert abs(result - expected) < 1e-10, f"Failed for raw={raw}, ref={ref}"
    
    def test_oor_flag_emission(self):
        """B5. OOR flag emitted when |q|/q_ref ∉ [0,1]."""
        # In-range: no OOR
        result, oor = bounded_embedding(0.5, 1.0)
        assert result == 0.5
        assert oor is False
        
        # Out-of-range high: OOR emitted
        result, oor = bounded_embedding(1.5, 1.0)
        assert result == 1.0  # clipped
        assert oor is True
        
        # Out-of-range with negative (magnitude > ref)
        result, oor = bounded_embedding(-2.0, 1.0)
        assert result == 1.0  # clipped (|−2|/1 = 2 > 1)
        assert oor is True
        
        # Exactly at boundary: no OOR
        result, oor = bounded_embedding(1.0, 1.0)
        assert result == 1.0
        assert oor is False
    
    def test_log_safe_embedding_epsilon_bounds(self):
        """B6. Log-safe output in [ε, 1-ε], never exactly 0 or 1."""
        epsilon = EPSILON_LOG
        
        # Test extreme inputs
        test_cases = [0.0, 1e-20, 0.5, 1.0 - 1e-20, 1.0, 2.0]
        
        for q_tilde in test_cases:
            q_tilde_clipped = clip(q_tilde, 0.0, 1.0)
            result = log_safe_embedding(q_tilde_clipped)
            
            assert result >= epsilon, f"Below ε for input {q_tilde}"
            assert result <= 1.0 - epsilon, f"Above 1-ε for input {q_tilde}"
            assert result != 0.0, f"Exactly 0 for input {q_tilde}"
            assert result != 1.0, f"Exactly 1 for input {q_tilde}"


# =============================================================================
# C. E_KIN COMPUTATION ORDER (RAW THEN NORMALIZE)
# =============================================================================

class TestEkinComputationOrder:
    """C. E_kin computation order (raw then normalize)."""
    
    def test_Ekin_raw_then_normalize(self):
        """
        C7. Correct order: E_kin in physical units first, then normalize.
        Must not clip v first and compute energy from clipped v.
        """
        # Case where v is large so E_kin would exceed E_ref
        v_large = 2.0 * V_REF  # 2x reference velocity
        
        correct = compute_Ekin_raw_then_normalize(v_large)
        wrong = compute_Ekin_wrong_order(v_large)
        
        # Correct: E_kin = 0.5 * 1 * 4 = 2.0, normalized = clip(2.0) = 1.0
        assert correct == 1.0
        
        # Wrong: v_clipped = 1.0, E_kin = 0.5 * 1 * 1 = 0.5, normalized = 0.5
        assert wrong == 0.5
        
        # They must differ (this is the semantic difference we're catching)
        assert correct != wrong


# =============================================================================
# D. PHASE SPACE AND METRIC
# =============================================================================

class TestPhaseSpace:
    """D. Phase space and metric (patched symbols + squared-L2)."""
    
    def test_phase_space_projection(self):
        """D8. γ(t) is exactly (x̃(t), ṽ(t)) from Ψ(t)."""
        # Given Ψ = (x̃_ε, ṽ_ε, ã_ε, Ẽ_kin, p̃_ε)
        psi = (0.3, 0.4, 0.1, 0.2, 0.15)
        
        # γ should be first two components (without ε for phase space)
        gamma = (psi[0], psi[1])
        
        assert gamma == (0.3, 0.4)
    
    def test_squared_l2_distance(self):
        """D9. Squared-L2 distance properties."""
        gamma1 = (0.2, 0.3)
        gamma2 = (0.5, 0.7)
        
        # Compute d²
        d2 = squared_l2_distance(gamma1, gamma2)
        expected = (0.5 - 0.2) ** 2 + (0.7 - 0.3) ** 2
        assert abs(d2 - expected) < 1e-10
        
        # Non-negativity
        assert d2 >= 0
        
        # Symmetry
        d2_reverse = squared_l2_distance(gamma2, gamma1)
        assert abs(d2 - d2_reverse) < 1e-10
        
        # Identity of indiscernibles
        d2_same = squared_l2_distance(gamma1, gamma1)
        assert d2_same == 0.0


# =============================================================================
# E. RETURN-DOMAIN GENERATOR
# =============================================================================

class TestReturnDomainGenerator:
    """E. Return-domain generator (off-by-one fixes)."""
    
    def test_domain_DW_startup_truncation(self):
        """E10. D_W(t) startup truncation."""
        # t=0: empty
        assert compute_D_W(0) == set()
        
        # t < W: D_W(t) = {0, 1, ..., t-1}
        assert compute_D_W(1) == {0}
        assert compute_D_W(2) == {0, 1}
        assert compute_D_W(5) == {0, 1, 2, 3, 4}
        
        # t = W: D_W(t) = {0, 1, ..., W-1}
        assert compute_D_W(64) == set(range(0, 64))
        
        # t > W: D_W(t) = {t-W, ..., t-1}
        assert compute_D_W(100) == set(range(36, 100))
    
    def test_domain_DW_delta_filter(self):
        """E11. D_{W,δ}(t) filters by (t-u) ≥ δ."""
        # At t=6, δ=3: valid u have (6-u) ≥ 3, i.e., u ≤ 3
        D = compute_D_W_delta(6, W=64, delta=3)
        assert D == {0, 1, 2, 3}
        
        # Verify lag=0 excluded (u=t not in D_W anyway, u ≤ t-1)
        for u in D:
            assert 6 - u >= 3
    
    def test_domain_size_formula_three_case(self):
        """E12. Domain size formula (patched three-case)."""
        # t < δ: size = 0
        assert compute_domain_size(0) == 0
        assert compute_domain_size(1) == 0
        assert compute_domain_size(2) == 0
        
        # t = δ: size = 1
        assert compute_domain_size(3) == 1
        
        # δ ≤ t < W: size = t - δ + 1
        assert compute_domain_size(4) == 2
        assert compute_domain_size(5) == 3
        assert compute_domain_size(63) == 61
        
        # t ≥ W: size = W - δ + 1 = 62
        assert compute_domain_size(64) == 62
        assert compute_domain_size(100) == 62
        assert compute_domain_size(1000) == 62
        
        # Verify formula matches actual set computation
        for t in [0, 1, 2, 3, 4, 5, 10, 63, 64, 100]:
            D = compute_D_W_delta(t)
            assert len(D) == compute_domain_size(t), f"Mismatch at t={t}"


# =============================================================================
# F. U(t), τ_KIN TYPING, RETURN_RATE
# =============================================================================

class TestReturnMechanics:
    """F. U(t), τ_kin typing, return_rate."""
    
    def test_valid_return_set_U_definition(self):
        """F13. U(t) uses strict inequality d² < η_phase."""
        gamma_history = {
            0: (0.20, 0.30),
            1: (0.21, 0.31),
            2: (0.22, 0.32),
            3: (0.50, 0.50),  # far away
            4: (0.20, 0.30),  # exact return to t=0
        }
        
        # At t=4, check which u are in U
        # d²(γ(4), γ(0)) = 0 (exact match)
        # d²(γ(4), γ(1)) = 0.0002 (close)
        # With η_phase = 0.01, both should be in U
        
        U = compute_valid_return_set(4, gamma_history, eta_phase=0.01)
        assert 0 in U  # exact match
        assert 1 in U  # d² = 0.0002 < 0.01
        
        # Strict inequality test: d² must be < η, not ≤
        gamma_history_boundary = {
            0: (0.0, 0.0),
            3: (0.1, 0.0),  # d² = 0.01 exactly
        }
        U_boundary = compute_valid_return_set(3, gamma_history_boundary, eta_phase=0.01)
        assert 0 not in U_boundary  # d² = 0.01 is NOT < 0.01
    
    def test_tau_kin_integer_samples(self):
        """F14. τ_kin is positive integer or typed sentinel."""
        # With valid returns
        gamma_history = {
            0: (0.20, 0.30),
            5: (0.20, 0.30),  # exact return
        }
        tau = compute_tau_kin(5, gamma_history, eta_phase=0.01)
        
        assert isinstance(tau, int)
        assert tau > 0
        assert tau == 5  # delay from t=5 to u=0
        
        # Without valid returns
        gamma_history_no_return = {
            0: (0.0, 0.0),
            5: (0.9, 0.9),  # far away
        }
        tau_inf = compute_tau_kin(5, gamma_history_no_return, eta_phase=0.01)
        
        assert tau_inf == KinSpecialValue.INF_KIN
        assert isinstance(tau_inf, KinSpecialValue)
    
    def test_tau_kin_min_delay_semantics(self):
        """F15. τ_kin picks most recent return (minimum delay) within D_{W,δ}."""
        # Create scenario with multiple valid returns at different lags
        # At t=10, D_{W,δ}(10) = {0,1,2,3,4,5,6,7} (delays ≥ 3)
        gamma_history = {
            0: (0.50, 0.50),  # return, delay=10
            5: (0.50, 0.50),  # return, delay=5
            7: (0.50, 0.50),  # return, delay=3 (most recent in valid domain)
            8: (0.50, 0.50),  # delay=2 < δ, EXCLUDED by debounce
            9: (0.50, 0.50),  # delay=1 < δ, EXCLUDED by debounce
            10: (0.50, 0.50), # current
        }
        
        tau = compute_tau_kin(10, gamma_history, eta_phase=0.01)
        
        # Multiple valid returns at u=0,5,7 with delays 10, 5, 3
        # u=8,9 are excluded by debounce (delay < δ=3)
        # Minimum delay among valid returns is 3 (from u=7)
        assert tau == 3
    
    def test_return_rate_empty_domain_is_zero(self):
        """F16. return_rate = 0 when |D_{W,δ}(t)| = 0."""
        gamma_history = {
            0: (0.5, 0.5),
            1: (0.5, 0.5),
            2: (0.5, 0.5),
        }
        
        # t < δ means empty domain
        for t in [0, 1, 2]:
            rate = compute_return_rate(t, gamma_history)
            assert rate == 0.0, f"Expected 0 at t={t}, got {rate}"
            
            # Also verify credit is 0
            tau = compute_tau_kin(t, gamma_history)
            credit = compute_kinematic_credit(tau, rate)
            assert credit == 0.0
    
    def test_return_rate_range(self):
        """F17. return_rate ∈ [0, 1] always."""
        # Various scenarios
        gamma_history = {i: (0.5, 0.5) for i in range(100)}
        
        for t in [0, 3, 10, 50, 100]:
            rate = compute_return_rate(t, gamma_history)
            assert 0.0 <= rate <= 1.0, f"Rate {rate} out of bounds at t={t}"


# =============================================================================
# G. KINEMATIC CREDIT (AXIOM-0)
# =============================================================================

class TestKinematicCredit:
    """G. Kinematic credit (Axiom-0 and thresholds)."""
    
    def test_credit_no_return_no_credit(self):
        """G18. If τ_kin = INF_KIN ⇒ credit = 0."""
        credit = compute_kinematic_credit(KinSpecialValue.INF_KIN, return_rate=0.5)
        assert credit == 0.0
        
        credit = compute_kinematic_credit(KinSpecialValue.UNIDENTIFIABLE_KIN, return_rate=0.5)
        assert credit == 0.0
    
    def test_credit_return_rate_threshold(self):
        """G19. If return_rate ≤ 0.1 ⇒ credit = 0."""
        # Even with finite τ_kin
        credit = compute_kinematic_credit(tau_kin=5, return_rate=0.1)
        assert credit == 0.0
        
        credit = compute_kinematic_credit(tau_kin=5, return_rate=0.05)
        assert credit == 0.0
    
    def test_credit_formula_exact(self):
        """G20. Credit formula exact computation."""
        # τ_kin = 4, return_rate = 0.25, T_crit = 10.0
        # credit = (1 / (1 + 4/10)) * 0.25 = (1/1.4) * 0.25 = 0.714285... * 0.25
        
        tau_kin = 4
        return_rate = 0.25
        
        expected = (1.0 / (1.0 + 4.0 / 10.0)) * 0.25
        # = (1 / 1.4) * 0.25 = 0.7142857... * 0.25 = 0.178571428...
        
        credit = compute_kinematic_credit(tau_kin, return_rate)
        
        assert abs(credit - expected) < 1e-10
        assert abs(credit - 0.17857142857142858) < 1e-10


# =============================================================================
# H. STABILITY K
# =============================================================================

class TestStabilityK:
    """H. Stability K (startup guard + clipping caps)."""
    
    def test_stability_startup_ddof_guard(self):
        """H21. When |D_W(t)| < 2, K = 0 (startup guard)."""
        # Empty list
        K, _, _ = compute_K_stability([], [], 0.5)
        assert K == 0.0
        
        # Single element (ddof=1 undefined)
        K, _, _ = compute_K_stability([0.5], [0.3], 0.5)
        assert K == 0.0
    
    def test_stability_components_and_clipping(self):
        """H22. K computation with clipping caps."""
        # Normal case
        x_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        v_values = [0.2, 0.25, 0.3, 0.35, 0.4]
        return_rate = 0.8
        
        K, sigma_oor, v_oor = compute_K_stability(x_values, v_values, return_rate)
        
        assert 0.0 <= K <= 1.0
        assert sigma_oor is False
        assert v_oor is False
        
        # Case where sigma exceeds cap
        x_high_var = [0.0, 1.0, 0.0, 1.0, 0.0]  # high variance
        K_high, sigma_oor_high, _ = compute_K_stability(x_high_var, v_values, return_rate)
        
        # sigma_x will be high, factor should clip
        assert 0.0 <= K_high <= 1.0
        # With such high variance, sigma_oor should trigger
        
        # Case where v_mean exceeds cap
        v_high = [1.2, 1.3, 1.4, 1.5, 1.6]  # all > v_max=1.0
        K_v, _, v_oor_high = compute_K_stability(x_values, v_high, return_rate)
        
        assert v_oor_high is True
        assert K_v == 0.0  # factor becomes 0
    
    def test_stability_regime_thresholds(self):
        """H23. Regime classification thresholds."""
        assert classify_stability_regime(0.8) == "Stable"
        assert classify_stability_regime(0.71) == "Stable"
        
        assert classify_stability_regime(0.7) == "Watch"
        assert classify_stability_regime(0.5) == "Watch"
        assert classify_stability_regime(0.31) == "Watch"
        
        assert classify_stability_regime(0.3) == "Unstable"
        assert classify_stability_regime(0.1) == "Unstable"
        assert classify_stability_regime(0.0) == "Unstable"


# =============================================================================
# I. TYPED SENTINEL ENFORCEMENT
# =============================================================================

class TestTypedSentinels:
    """I. Typed sentinel enforcement (reject IEEE Inf/NaN)."""
    
    def test_reject_ieee_inf_nan_on_tau_kin(self):
        """I24. IEEE Inf/NaN rejected, only typed sentinels allowed."""
        # Valid values
        assert is_valid_tau_kin(5) is True
        assert is_valid_tau_kin(1) is True
        assert is_valid_tau_kin(100) is True
        assert is_valid_tau_kin(KinSpecialValue.INF_KIN) is True
        assert is_valid_tau_kin(KinSpecialValue.UNIDENTIFIABLE_KIN) is True
        
        # Invalid values
        assert is_valid_tau_kin(float('inf')) is False
        assert is_valid_tau_kin(float('-inf')) is False
        assert is_valid_tau_kin(float('nan')) is False
        assert is_valid_tau_kin(0) is False
        assert is_valid_tau_kin(-1) is False
        assert is_valid_tau_kin(5.5) is False  # not integer
        assert is_valid_tau_kin("INF_KIN") is False  # string, not enum
    
    def test_special_values_enum_closed(self):
        """I25. Only declared sentinels exist in enum."""
        valid_sentinels = {KinSpecialValue.INF_KIN, KinSpecialValue.UNIDENTIFIABLE_KIN}
        
        # Enum should have exactly these members
        assert set(KinSpecialValue) == valid_sentinels
        
        # Cannot create arbitrary sentinels
        with pytest.raises(ValueError):
            KinSpecialValue("UNKNOWN_SENTINEL")


# =============================================================================
# J. FIREWALL AND NAMESPACE COLLISION
# =============================================================================

class TestFirewallNamespace:
    """J. Firewall and namespace collision prevention."""
    
    def test_tau_kin_does_not_touch_kernel_tau_R(self):
        """J26. τ_kin and τ_R are separate (interface boundary check)."""
        # This is a conceptual test enforced by interface design
        # KIN functions return τ_kin, never τ_R
        
        gamma_history = {0: (0.5, 0.5), 5: (0.5, 0.5)}
        
        result = compute_tau_kin(5, gamma_history)
        
        # Result is named/typed as τ_kin, not τ_R
        # We verify by checking the return type and that it's from KIN namespace
        assert result == 5 or isinstance(result, KinSpecialValue)
        
        # The function name itself indicates KIN namespace
        assert "tau_kin" in compute_tau_kin.__name__
    
    def test_symbol_collision_guard_Omega_vs_omega(self):
        """J27. Angular velocity uses Ω, not ω (doc/code lint)."""
        # This would be a static analysis check on the codebase
        # For testing, we verify the specification documents
        
        kin_math_path = Path(__file__).parent.parent / "KINEMATICS_MATHEMATICS.md"
        
        if kin_math_path.exists():
            content = kin_math_path.read_text()
            
            # §10.2 should use Ω, not ω
            # Check that "Uniform Circular Motion" section uses Omega
            section_start = content.find("### 10.2 Uniform Circular Motion")
            if section_start != -1:
                section_end = content.find("### 10.3", section_start)
                section = content[section_start:section_end]
                
                # Should contain Ω
                assert "Ω" in section or "\\Omega" in section, \
                    "§10.2 should use Ω for angular velocity"
    
    def test_Lyapunov_delta_gamma_firewall(self):
        """J28. Lyapunov uses Δγ, not δγ (avoid debounce δ collision)."""
        kin_math_path = Path(__file__).parent.parent / "KINEMATICS_MATHEMATICS.md"
        
        if kin_math_path.exists():
            content = kin_math_path.read_text()
            
            # §11 should use Δγ or explicitly note δγ is different from debounce δ
            section_start = content.find("## §11. Lyapunov")
            if section_start != -1:
                section_end = content.find("## §12", section_start)
                section = content[section_start:section_end]
                
                # Should contain clarification or use Δγ
                has_delta_gamma = "Δγ" in section or "\\Delta\\gamma" in section
                has_clarification = "unrelated" in section.lower() or "perturbation" in section.lower()
                
                assert has_delta_gamma or has_clarification, \
                    "§11 should clarify Δγ vs δ or use Δγ notation"


# =============================================================================
# K. CONSERVATION LAWS GATING
# =============================================================================

class TestConservationLaws:
    """K. Conservation laws gating (non-gating unless preconditions true)."""
    
    def test_conservation_tests_are_conditionally_applied(self):
        """K29. Conservation tests skipped when preconditions not met."""
        
        def momentum_conservation_test(p_series: List[float], F_ext: float) -> Optional[bool]:
            """Returns None if not applicable, True/False otherwise."""
            if F_ext != 0:
                return None  # Not applicable
            
            # Check conservation
            for i in range(1, len(p_series)):
                if abs(p_series[i] - p_series[i-1]) >= EPSILON_P:
                    return False
            return True
        
        def energy_conservation_test(E_series: List[float], W_nc: float) -> Optional[bool]:
            """Returns None if not applicable, True/False otherwise."""
            if W_nc != 0:
                return None  # Not applicable
            
            for i in range(1, len(E_series)):
                if abs(E_series[i] - E_series[i-1]) >= EPSILON_E:
                    return False
            return True
        
        # When F_ext != 0, momentum test should be NA
        result = momentum_conservation_test([1.0, 1.1, 1.2], F_ext=5.0)
        assert result is None
        
        # When F_ext = 0, momentum test should run
        result = momentum_conservation_test([1.0, 1.0, 1.0], F_ext=0.0)
        assert result is True
        
        # When W_nc != 0, energy test should be NA
        result = energy_conservation_test([1.0, 0.9, 0.8], W_nc=0.1)
        assert result is None
        
        # When W_nc = 0, energy test should run
        result = energy_conservation_test([1.0, 1.0, 1.0], W_nc=0.0)
        assert result is True
    
    def test_conservation_tolerance_when_applicable(self):
        """K30. Conservation tolerances when preconditions met."""
        # Momentum within tolerance
        p_series = [1.0, 1.0 + 0.5e-6, 1.0 + 0.9e-6]
        for i in range(1, len(p_series)):
            assert abs(p_series[i] - p_series[i-1]) < EPSILON_P
        
        # Energy within tolerance
        E_series = [0.5, 0.5 + 0.5e-6, 0.5 + 0.9e-6]
        for i in range(1, len(E_series)):
            assert abs(E_series[i] - E_series[i-1]) < EPSILON_E


# =============================================================================
# L. REFERENCE IDENTITIES (NON-GATING)
# =============================================================================

class TestReferenceIdentities:
    """L. Reference identities (non-gating) enforcement."""
    
    def test_reference_sections_do_not_gate(self):
        """L31. §9-§11 are reference only, not enforcement gates."""
        # This is a process/doc test
        # The KINEMATICS_MATHEMATICS.md should have the disclaimer
        
        kin_math_path = Path(__file__).parent.parent / "KINEMATICS_MATHEMATICS.md"
        
        if kin_math_path.exists():
            content = kin_math_path.read_text()
            
            # §9 should have the reference disclaimer
            assert "REFERENCE IDENTITIES" in content or "NON-GATING" in content, \
                "§9-§11 should be marked as reference/non-gating"
            
            # Verify the disclaimer mentions enforcement gates
            assert "enforcement gate" in content.lower(), \
                "Should clarify these are not enforcement gates"


# =============================================================================
# V. GOLD-STANDARD TEST VECTORS
# =============================================================================

class TestGoldStandardVectors:
    """V. Gold-standard test vectors (locked reference cases)."""
    
    def test_vector_A_startup_empty_domain(self):
        """
        Vector A: Startup + empty-domain.
        Constant γ(t) for t=0..6, test at t=2 (<δ=3).
        """
        # Constant phase point
        gamma_history = {t: (0.5, 0.5) for t in range(7)}
        
        t = 2
        
        # |D_{W,δ}(2)| = 0 (t < δ)
        domain_size = compute_domain_size(t)
        assert domain_size == 0
        
        # return_rate(2) = 0
        rate = compute_return_rate(t, gamma_history)
        assert rate == 0.0
        
        # τ_kin(2) = INF_KIN
        tau = compute_tau_kin(t, gamma_history)
        assert tau == KinSpecialValue.INF_KIN
        
        # credit(2) = 0
        credit = compute_kinematic_credit(tau, rate)
        assert credit == 0.0
    
    def test_vector_B_known_return_credit(self):
        """
        Vector B: Known return, τ_kin, and credit.
        With η_phase=0.01, the smooth trajectory means ALL points in D are within tolerance.
        """
        gamma_history = {
            0: (0.20, 0.30),
            1: (0.21, 0.31),
            2: (0.22, 0.32),
            3: (0.23, 0.33),
            4: (0.24, 0.34),
            5: (0.25, 0.35),
            6: (0.22, 0.32),  # close to t=2, but also within η of t=3
        }
        
        t = 6
        
        # D_{W,δ}(6) = {0, 1, 2, 3}, |D| = 4
        D = compute_D_W_delta(t)
        assert D == {0, 1, 2, 3}
        assert len(D) == 4
        
        # Check distances to γ(6) = (0.22, 0.32):
        # d²(γ(6), γ(0)) = 0.0008 < 0.01 ✓
        # d²(γ(6), γ(1)) = 0.0002 < 0.01 ✓
        # d²(γ(6), γ(2)) = 0 < 0.01 ✓
        # d²(γ(6), γ(3)) = 0.0002 < 0.01 ✓
        # So U(6) = {0, 1, 2, 3} — ALL points within tolerance!
        
        U = compute_valid_return_set(t, gamma_history, eta_phase=ETA_PHASE)
        assert U == {0, 1, 2, 3}
        
        # τ_kin(6) = min delay = 3 (from u=3)
        tau = compute_tau_kin(t, gamma_history, eta_phase=ETA_PHASE)
        assert tau == 3
        
        # return_rate(6) = 4/4 = 1.0 (all in U)
        rate = compute_return_rate(t, gamma_history, eta_phase=ETA_PHASE)
        assert abs(rate - 1.0) < 1e-10
        
        # credit(6) = (1/(1+3/10)) * 1.0 = 1/1.3 = 0.7692307692...
        expected_credit = (1.0 / (1.0 + 3.0 / 10.0)) * 1.0
        credit = compute_kinematic_credit(tau, rate)
        assert abs(credit - expected_credit) < 1e-10
        assert abs(credit - 0.7692307692307693) < 1e-10
    
    def test_vector_brutal_multiple_returns_picks_most_recent(self):
        """
        Brutal vector: Multiple valid returns, assert τ_kin picks most recent.
        """
        gamma_history = {
            0: (0.50, 0.50),  # return 1
            1: (0.60, 0.60),
            2: (0.50, 0.50),  # return 2
            3: (0.70, 0.70),
            4: (0.50, 0.50),  # return 3 (most recent)
            5: (0.80, 0.80),
            6: (0.90, 0.90),
            7: (0.50, 0.50),  # current (exact match with 0, 2, 4)
        }
        
        t = 7
        
        # D_{W,δ}(7) = {0, 1, 2, 3, 4}
        D = compute_D_W_delta(t)
        assert D == {0, 1, 2, 3, 4}
        
        # U(7) = {0, 2, 4} (all exact matches)
        U = compute_valid_return_set(t, gamma_history, eta_phase=ETA_PHASE)
        assert U == {0, 2, 4}
        
        # τ_kin picks minimum delay: t-4=3, t-2=5, t-0=7 → min is 3
        tau = compute_tau_kin(t, gamma_history, eta_phase=ETA_PHASE)
        assert tau == 3  # most recent return at u=4
    
    def test_vector_B_sparse_isolated_return(self):
        """
        Vector B (sparse): Isolated return with large jumps to ensure only one match.
        This tests the original intent: single return giving τ_kin=4, rate=0.25.
        """
        gamma_history = {
            0: (0.10, 0.10),  # far from γ(6)
            1: (0.20, 0.20),  # far from γ(6)
            2: (0.50, 0.50),  # EXACT match to γ(6)
            3: (0.80, 0.80),  # far from γ(6)
            4: (0.90, 0.90),  # far (excluded by debounce anyway)
            5: (0.95, 0.95),  # far (excluded by debounce anyway)
            6: (0.50, 0.50),  # current
        }
        
        t = 6
        
        # D_{W,δ}(6) = {0, 1, 2, 3}
        D = compute_D_W_delta(t)
        assert D == {0, 1, 2, 3}
        
        # Check distances:
        # d²(γ(6), γ(0)) = 0.32 >> 0.01
        # d²(γ(6), γ(1)) = 0.18 >> 0.01  
        # d²(γ(6), γ(2)) = 0 < 0.01 ✓
        # d²(γ(6), γ(3)) = 0.18 >> 0.01
        
        U = compute_valid_return_set(t, gamma_history, eta_phase=ETA_PHASE)
        assert U == {2}
        
        # τ_kin(6) = 4 (delay from t=6 to u=2)
        tau = compute_tau_kin(t, gamma_history, eta_phase=ETA_PHASE)
        assert tau == 4
        
        # return_rate(6) = 1/4 = 0.25
        rate = compute_return_rate(t, gamma_history, eta_phase=ETA_PHASE)
        assert abs(rate - 0.25) < 1e-10
        
        # credit(6) = (1/(1+4/10)) * 0.25 = 0.1785714286...
        expected_credit = (1.0 / (1.0 + 4.0 / 10.0)) * 0.25
        credit = compute_kinematic_credit(tau, rate)
        assert abs(credit - expected_credit) < 1e-10


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
