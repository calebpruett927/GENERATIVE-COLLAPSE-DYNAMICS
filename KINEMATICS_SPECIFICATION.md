# Kinematics Specification

## Overview

The Kinematics Extension (KIN.INTSTACK.v1) provides a physics-based layer for describing motion within the UMCP framework. This specification defines how classical mechanics concepts—position, velocity, acceleration, energy, and momentum—map to UMCP's return axiom and tiered architecture.

**Core Insight**: Phase space (x,v) naturally connects to UMCP's "what returns is real" axiom. Periodic and oscillatory motion exhibits finite return times, while drifting motion has infinite return times. This creates a natural classification of motion regimes.

**IMPORTANT**: Kinematics closures are **DOMAIN CLOSURES**, not UMCP Tier-1 kernel invariants. UMCP Tier-1 invariants (ω, F, S, C, τ_R, κ, IC) are computed only by the kernel. Kinematics closures produce observables and overlays. See TIER_SYSTEM.md for clarification.

## Contract Hierarchy

```
UMA.INTSTACK.v1 (Base UMCP Contract)
    │
    └── KIN.INTSTACK.v1 (Kinematics Extension)
            ├── Reserved Symbols: x, v, a, p, E_kin, E_pot, etc.
            ├── Axioms: KIN-AX-0 through KIN-AX-3
            ├── Regime Classification: Stable/Watch/Unstable
            └── Mathematical Identities: Energy & Momentum Conservation
```

## Symbol Namespace (Kernel Collision Avoidance)

| KIN Symbol | Meaning | NOT Confused With |
|------------|---------|-------------------|
| **Ω** (omega_rot) | Angular velocity | ω (UMCP kernel drift) |
| **T** (torque) | Torque | τ (UMCP return time prefix) |
| **τ_kin** | Kinematic return time | τ_R (UMCP kernel return time) |
| **S_kin** | Kinematic sources | S (UMCP kernel entropy) |
| **C_kin** | Kinematic consumption | C (UMCP kernel curvature) |

## Mathematical Foundation

### 1. Bounded Embedding (Normalization with OOR Policy)

All physical quantities are normalized to [0,1] domain via **bounded embedding**:

$$\tilde{q}(t) = \text{clip}_{[0,1]}\left(\frac{q(t)}{q_{\text{ref}}}\right), \quad \text{emit OOR flag if clipped}$$

For log-safe channels (used in entropy/log computations):

$$\tilde{q}_\epsilon(t) = \text{clip}_{[\epsilon, 1-\epsilon]}(\tilde{q}(t))$$

where ε = 10⁻⁸ is the log-safety guard.

| Quantity | Symbol | Bounded Embedding | Reference Scale |
|----------|--------|-------------------|-----------------|
| Position | x | x̃ = clip([0,1], x / L_ref), OOR flag | L_ref = 1.0 m |
| Velocity | v | ṽ = clip([0,1], v / v_ref), OOR flag | v_ref = 1.0 m/s |
| Acceleration | a | ã = clip([0,1], a / a_ref), OOR flag | a_ref = 9.81 m/s² |
| Time | t | t̃ = clip([0,1], t / T_ref), OOR flag | T_ref = L_ref / v_ref |
| Mass | m | m̃ = clip([0,1], m / m_ref), OOR flag | m_ref = 1.0 kg |
| Energy | E | Ẽ = clip([0,1], E / E_ref), OOR flag | E_ref = m_ref · v_ref² |
| Momentum | p | p̃ = clip([0,1], p / p_ref), OOR flag | p_ref = m_ref · v_ref |

**OOR Policy**: `clip_and_flag` — ALL values outside [0,1] are clipped and an OOR flag is emitted.

**Signed Observable Convention (FROZEN)**: All kinematic channels use **magnitudes** before normalization:

$$x := |x|, \quad v := |v|, \quad a := |a|, \quad p := |p|$$

This convention discards directional information in favor of bounded [0,1] embedding. Signed embeddings (preserving direction) are NOT supported in KIN.INTSTACK.v1. If signed observables are required, define a separate contract extension.

### 1.1 Explicit Ψ(t) Channel Vector (Tier-0)

The kinematic observable vector used by the UMCP kernel is:

$$\Psi(t) = \left( \tilde{x}_\epsilon(t), \tilde{v}_\epsilon(t), \tilde{a}_\epsilon(t), \tilde{E}_{kin,\epsilon}(t), \tilde{p}_\epsilon(t) \right) \in [0,1]^5$$

**Frozen in KIN.INTSTACK.v1**:
- `psi_dimension`: 5
- `psi_channels`: [x, v, a, E_kin, p]
- `psi_weights`: {x: 0.25, v: 0.25, a: 0.15, E_kin: 0.20, p: 0.15} (Σ = 1.0)

All channels use ε-guarded clipping for log-safety.

**Log-Sensitive Kernel Computations**: Kernel computations requiring logarithms (S, κ) use Ψ_ε(t) with ε = 10⁻⁸ already applied componentwise. This guarantees no channel is exactly 0 or 1, preventing log(0) or log(1−1) failures.

**E_kin Computation Order (FROZEN)**: Compute kinetic energy in physical units *before* normalization:

$$E_{kin}(t) = \frac{1}{2} m(t) v(t)^2 \quad \text{(physical units)}$$

$$\tilde{E}_{kin}(t) = \text{clip}_{[0,1]}\left(\frac{E_{kin}(t)}{E_{ref}}\right), \quad E_{ref} = m_{ref} \cdot v_{ref}^2$$

This ensures E_kin reflects actual energy, not an artifact of clipped velocity.

**Mass Assumption (FROZEN for v1)**: $m(t)$ is assumed constant and equal to $m_{ref}$ unless an external measurement pipeline for $m(t)$ is explicitly declared. Variable-mass systems (rockets, conveyor loading, etc.) are **out-of-scope for KIN.INTSTACK.v1**. The Ψ channel vector does not include a mass channel; mass is a frozen reference constant, not a measured observable.

### 2. Phase Space

The kinematic phase space is the 2D space (x, v):

```
Γ_kin = {(x, v) : x ∈ [0,1], v ∈ [0,1]}
```

**Phase Magnitude Squared** (consistent with squared-L2 metric):
$$|\gamma|^2 = x^2 + v^2$$

**Phase Magnitude** (if needed):
$$|\gamma| = \sqrt{x^2 + v^2}$$

**Phase Distance (FROZEN METRIC: squared-L2)**:

Metric = squared-L2. Therefore η_phase is a **squared distance tolerance**.

$$d^2((x_1, v_1), (x_2, v_2)) = (x_2 - x_1)^2 + (v_2 - v_1)^2$$

### 3. Return Axiom Mapping

UMCP's return axiom "What Returns Through Collapse Is Real" maps to phase space dynamics:

#### 3.0 Return-Domain Generator (FROZEN)

**Time Units (FROZEN)**: All temporal quantities in this section are expressed in **sample steps** (discrete index units), NOT seconds:
- $W = 64$ — window size in samples
- $\delta = 3$ — debounce lag in samples
- $\tau_{kin}$ — return time in samples
- $T_{crit} = 10.0$ — critical return threshold in samples

If physical time is needed, define the sample period $\Delta t_{sample}$ (seconds per sample) and convert:
$$\tau_{kin,seconds} = \tau_{kin} \cdot \Delta t_{sample}$$

The return-domain generator defines where to search for returns:

$$D_W(t) = \{u : 0 < t - u \leq W\}$$

where:
- $W = H_{rec,kin} = 64$ (window size in samples, frozen)
- $\delta = 3$ (debounce lag in samples: prevents spurious rapid re-triggers; note that lag=0 is already excluded by $0 < t-u$)

**Effective Domain (with debounce lag, discrete-time inclusive)**:

$$D_{W,\delta}(t) = \{u \in D_W(t) : (t - u) \geq \delta\}$$

For discrete integer-indexed time with both endpoints included:
$$|D_{W,\delta}(t)| = W - \delta + 1 = 62 \quad \text{(for } W=64, \delta=3\text{, when } t \geq W\text{)}$$

**Startup-Truncated Domain (FORMAL DEFINITION)**:

For all $t \geq 0$, including startup ($t < W$):

$$D_W(t) := \{ u \in \mathbb{Z} : \max(0, t-W) \leq u \leq t-1 \}$$

$$D_{W,\delta}(t) := \{ u \in D_W(t) : (t - u) \geq \delta \}$$

**Effective Domain Size**:
$$|D_{W,\delta}(t)| = \begin{cases} \max(0, t - \delta) & \text{if } t < W \\ W - \delta + 1 = 62 & \text{if } t \geq W \end{cases}$$

This makes `return_rate` and `τ_kin` fully well-posed for all $t \geq 0$, including during startup when $|D_{W,\delta}(t)| < 62$.

**Kinematic Return Time (τ_kin)** — Discrete-Time Definition:

Given phase point γ(t), tolerance η_phase, and effective domain $D_{W,\delta}(t)$:

1. **Valid-return set**:
$$U(t) = \{u \in D_{W,\delta}(t) : d^2(\gamma(t), \gamma(u)) < \eta_{phase}\}$$

2. **Return time** (a *delay*, not a timestamp; uses `min` for discrete samples):
$$\tau_{kin}(t) = \begin{cases} \min\{t - u : u \in U(t)\} & \text{if } U(t) \neq \emptyset \\ \text{INF\_KIN} & \text{otherwise} \end{cases}$$

**Note**: τ_kin is the *delay* to the most recent return, not the timestamp of the return event. This is consistent with kernel τ_R semantics. The `min` operation is well-defined for finite discrete sample sets; no measure-theoretic `inf` is needed.

**Return Classification**:

| τ_kin | Classification | Physical Interpretation |
|-------|----------------|-------------------------|
| 0 < τ_kin < T_crit | Returning | Periodic/oscillatory motion |
| T_crit ≤ τ_kin < 2·T_crit | Partially_Returning | Quasi-periodic motion |
| 2·T_crit ≤ τ_kin < ∞ | Weakly_Returning | Damped oscillations |
| τ_kin = INF_KIN | Non_Returning | Drifting/divergent motion |

### 3.1 Axiom-0 Enforcement (no_return_no_credit)

**AXIOM-0**: "Collapse is generative; only what returns is real."

The kinematics layer enforces this axiom through `no_return_no_credit`:

| τ_kin | Credit | Reason |
|-------|--------|--------|
| τ_kin < ∞ | kinematic_credit > 0 | Motion returns → credit granted |
| τ_kin = INF_KIN | kinematic_credit = 0 | Motion does not return → NO credit |

#### 3.1.1 Return Rate Definition (FROZEN)

The `return_rate` is defined as the fraction of the effective domain with valid returns:

$$\text{return\_rate}(t) = \frac{|U(t)|}{|D_{W,\delta}(t)|} \in [0,1]$$

where:
- $U(t)$ is the valid-return set (defined above)
- $D_{W,\delta}(t)$ is the effective domain with debounce lag
- $|D_{W,\delta}(t)| = W - \delta + 1 = 62$ (discrete-time inclusive, for W=64, δ=3)

**Interpretation**: return_rate = 1.0 means every candidate in the effective domain is a valid return. The exclusion lag does not artificially reduce the maximum achievable rate.

**Threshold (FROZEN)**: `return_rate ≤ 0.1` → `kinematic_credit = 0`

**Credit Formula** (for returning motion):
$$\text{kinematic\_credit} = \frac{1}{1 + \tau_{kin}/T_{crit}} \cdot \text{return\_rate}$$

where $T_{crit} = 10.0$ (frozen).

#### 3.1.2 Firewall: τ_kin vs τ_R

**IMPORTANT**: τ_kin is a kinematics overlay diagnostic and does **NOT** define or substitute for kernel τ_R. 

Kernel τ_R is computed only from Ψ using the frozen UMCP return-domain generator (D_θ) and tolerance (η). τ_kin operates in phase space (x,v) using the kinematics-specific D_W and η_phase.

**Explicit Separation**: τ_R is computed only by the UMCP kernel from Ψ using the frozen UMCP contract (‖·‖_p, η, D_θ, H_rec); KIN’s τ_kin is a domain diagnostic computed on γ(t)=(ẍ,ṽ) with squared-L2 and η_phase, and must not be substituted. There is no expectation that τ_R “should match” τ_kin—they measure different things in different spaces.

If a bridge is needed, it must be defined as a Tier-2 mapping with explicit versioning:
$$\tau_R := \Phi(\tau_{kin}, \ldots)$$ (not implemented — reserved for future extension)

**Enforcement Points**:
1. `phase_space_return.py::compute_kinematic_return()` → Sets `kinematic_credit = 0` for INF_KIN
2. `phase_space_return.py::compute_kinematic_credit()` → Explicit Axiom-0 credit computation
3. `kinematic_stability.py::compute_kinematic_budget()` → Forces `R_kin_effective = 0` for non-returning

### 4. Kinematics Equations

**Linear Motion**:
$$x(t) = x_0 + v_0 t + \frac{1}{2} a t^2$$
$$v(t) = v_0 + a t$$

**Rotational Motion** (using Ω to avoid kernel ω collision):
$$\theta(t) = \theta_0 + \Omega_0 t + \frac{1}{2} \alpha t^2$$
$$\Omega(t) = \Omega_0 + \alpha t$$
$$L = I \Omega$$  (Angular Momentum)
$$T = I \alpha$$  (Torque — using T to avoid kernel τ collision)

**Energy**:
$$E_{kin} = \frac{1}{2} m v^2$$
$$E_{pot,grav} = m g h$$
$$E_{pot,spring} = \frac{1}{2} k x^2$$
$$E_{mech} = E_{kin} + E_{pot}$$

**Momentum**:
$$p = m v$$
$$J = F \cdot \Delta t$$  (Impulse)
$$J = \Delta p$$  (Impulse-Momentum Theorem)

## Axioms

The following are **formal enforcement axioms** with testable mathematical constraints. These are distinct from the prose "KIN Principles" in the contract (which describe physical intent).

### Typed Infinity Definition

**INF_KIN** is a typed sentinel value (enum or tagged union) distinct from IEEE `Inf`/`NaN`. Untyped floating-point infinities and NaN values are rejected by KIN-AX-1. This prevents silent coercion bugs.

**Implementation** (reference only — see `closures/kinematics/phase_space_return.py`):

```python
class KinSpecialValue(Enum):
    INF_KIN = "INF_KIN"              # No return detected
    UNIDENTIFIABLE_KIN = "UNIDENT"   # Cannot determine return
```

### KIN-AX-0: Observable Boundedness (ENFORCEMENT)
The kernel input Ψ(t) and derived phase point γ(t) must be bounded:

$$\Psi(t) \in [0,1]^5 \quad \text{with } \texttt{clip\_and\_flag} \text{ OOR policy}$$

$$\gamma(t) = (\tilde{x}(t), \tilde{v}(t)) \text{ is induced from the first two channels of } \Psi(t)$$

**Test**: Any channel value outside [0,1] triggers OOR flag and is clipped.

### KIN-AX-1: Return Time Finiteness (ENFORCEMENT)
The kinematic return time τ_kin is either finite (τ_kin < ∞) or explicitly typed as INF_KIN or UNIDENTIFIABLE_KIN. Untyped IEEE Inf/NaN are forbidden.

$$\tau_{kin} \in \mathbb{R}^+ \cup \{\text{INF\_KIN}, \text{UNIDENTIFIABLE\_KIN}\}$$

**Test**: `isinstance(tau_kin, (float, KinSpecialValue))` and reject raw `float('inf')` or `float('nan')`.

### KIN-AX-2: Conservation Laws (CONDITIONAL ENFORCEMENT)
Conservation laws are **conditional**, not universal:

- **Momentum**: If $F_{ext} = 0$ then $dp/dt = 0$ (within tolerance $\epsilon_p$)
- **Energy**: If $W_{nc} = 0$ (no non-conservative work) then $dE_{mech}/dt = 0$ (within tolerance $\epsilon_E$)

**Frozen Tolerances**:
- $\epsilon_p = 10^{-6}$ (momentum conservation tolerance)
- $\epsilon_E = 10^{-6}$ (energy conservation tolerance)

**Test**: `verify_momentum_conservation(F_ext=0)` and `verify_energy_conservation(W_nc=0)` must pass. If the preconditions ($F_{ext}=0$ or $W_{nc}=0$) are not met, conservation tests are **not applicable** and should not be run as enforcement gates.

### KIN-AX-3: Stability Index Boundedness (ENFORCEMENT)
The kinematic stability index satisfies K_stability ∈ [0,1], where:
- K_stability > K_stable_threshold implies "Stable" regime
- K_watch_threshold < K_stability ≤ K_stable_threshold implies "Watch" regime
- K_stability ≤ K_watch_threshold implies "Unstable" regime

**Test**: K_stability values outside [0,1] are rejected.

---

**KIN Principles (Contract Prose — Non-Enforcement)**:
The contract `KIN.INTSTACK.v1.yaml` contains prose axiom statements that describe physical intent (e.g., "Conservation governs dynamics"). These are NOT the same as the formal enforcement axioms above. The prose statements guide implementation; the formal axioms gate validation.

## Closure Inventory

**NOTE**: All closures are **KIN-Domain Closures**, not UMCP Tier-1 kernel invariants.

### 1. linear_kinematics.py (KIN-Domain Closure A)
**Purpose**: Compute linear motion quantities and predictions.

**Main Functions**:
- `compute_linear_kinematics(x, v, a, dt)` → position, velocity, acceleration, predictions
- `compute_trajectory(x_series, v_series, a_series, dt)` → trajectory statistics
- `verify_kinematic_consistency(x, v, a, dt, tol)` → consistency verification

**Bounded Embedding**: All inputs clipped to [0,1] with OOR flags emitted.

**Regime Classification**:
- Stable: v < 0.3 and a < 0.2
- Watch: v < 0.6 and a < 0.5
- Critical: v ≥ 0.6 or a ≥ 0.5

### 2. rotational_kinematics.py (KIN-Domain Closure A)
**Purpose**: Compute angular motion, torque, angular momentum.

**Symbol Namespace**: Uses Ω (omega_rot) not ω, uses T not τ.

**Main Functions**:
- `compute_rotational_kinematics(theta, omega_rot, alpha, I)` → angular quantities
- `compute_centripetal(omega_rot, r, m)` → centripetal acceleration/force
- `compute_rotational_trajectory(theta_series, omega_series, alpha_series)` → trajectory stats

### 3. energy_mechanics.py (KIN-Domain Closure A)
**Purpose**: Compute energy quantities and verify conservation.

**Main Functions**:
- `compute_kinetic_energy(v, m)` → E_kin
- `compute_potential_energy(h, m, g, type)` → E_pot
- `compute_mechanical_energy(v, h, m, g)` → E_mech
- `compute_work(F, d, angle)` → W
- `compute_power(F, v)` → P
- `verify_energy_conservation(E_series, tol)` → conservation status
- `verify_work_energy_theorem(W, E_i, E_f)` → theorem validation

### 4. momentum_dynamics.py (KIN-Domain Closure A)
**Purpose**: Compute momentum, impulse, collisions.

**Main Functions**:
- `compute_linear_momentum(v, m)` → p
- `compute_impulse(F, dt)` → J
- `compute_collision_1d(m1, v1, m2, v2, type)` → collision results
- `verify_momentum_conservation(p_series, tol)` → conservation status
- `compute_momentum_flux(p, v)` → flux tensor

**Collision Equations Disclaimer (magnitude_only)**: The collision equations (elastic/inelastic) included in this closure are **classical reference identities** that assume signed velocities with directionality. Under `signed_convention=magnitude_only` (frozen in §1), directional information is discarded before normalization. Therefore:
1. Collision resolution is **not physically complete** under magnitude_only
2. Collision equations are **not used as enforcement gates** in KIN.INTSTACK.v1
3. Tests using collision identities are **illustrative only**, not conformance tests

If physically complete collision handling is required, define a separate extension with `signed_convention=signed`.

### 5. phase_space_return.py (KIN-Domain Closure B - Return Overlay)
**Purpose**: Implement UMCP return axiom in (x,v) phase space.

**Metric (FROZEN)**: squared-L2. η_phase is a squared distance tolerance.

**Main Functions**:
- `compute_kinematic_return(x_series, v_series, eta)` → τ_kin, return_rate, regime
- `compute_kinematic_credit(tau_kin, return_rate)` → Axiom-0 credit computation
- `compute_phase_trajectory(x_series, v_series)` → path_length, enclosed_area
- `detect_oscillation(x_series, v_series)` → oscillation_type, period_estimate
- `compute_lyapunov_estimate(x_series, v_series)` → λ_max estimate

**NOTE**: τ_kin is NOT τ_R (kernel return time). They use distinct symbols.

**Return Regimes**:
- Returning: return_rate > 0.5
- Partially_Returning: 0.3 < return_rate ≤ 0.5
- Weakly_Returning: 0.1 < return_rate ≤ 0.3
- Non_Returning: return_rate ≤ 0.1

### 6. kinematic_stability.py (KIN-Domain Closure B - Stability Overlay)
**Purpose**: Compute stability indices and regime classification.

**Main Functions**:
- `compute_kinematic_stability(x_series, v_series)` → K_stability, regime
- `compute_stability_margin(K, threshold)` → margin
- `compute_kinematic_budget(K_t0, K_t1, S_kin, C_kin, tau_kin, R_kin)` → budget equation
- `classify_motion_regime(v_mean, a_mean, K, tau)` → motion regime

**BUDGET EQUATION DISCLAIMER**:
ΔK_kin = S_kin - C_kin + R_kin is a **KINEMATICS OVERLAY ONLY**.
This is NOT the UMCP weld/continuity budget law (Δκ).
It CANNOT be used to certify continuity across seams.
Symbols S_kin, C_kin, R_kin are KIN-namespace only, not kernel S, C.

**K_stability Computation**:
$$K_{stability} = (1 - σ_x/σ_{max}) \cdot (1 - v_{mean}/v_{max}) \cdot w_{τ}$$

**Frozen Definitions for K_stability** (v1):

| Component | Definition | Frozen Value/Policy |
|-----------|------------|--------------------|
| $\sigma_x(t)$ | Sample standard deviation of $\tilde{x}_\epsilon$ over $D_W(t)$ | Population formula with Bessel correction (ddof=1) |
| $\sigma_{max}$ | Fixed constant (not data-derived) | $\sigma_{max} = 0.5$ |
| $v_{mean}(t)$ | Arithmetic mean of $\tilde{v}_\epsilon$ over $D_W(t)$ | Standard mean |
| $v_{max}$ | Fixed constant (not data-derived) | $v_{max} = 1.0$ |
| $w_\tau$ | Return weight mapping | $w_\tau = \text{return\_rate}(t)$ (from §3.1.1) |

**Clip+Flag Policy for K_stability**: If any ratio $(\sigma_x/\sigma_{max})$ or $(v_{mean}/v_{max})$ exceeds 1.0:
1. Clip the ratio to 1.0
2. Emit an OOR flag
3. The corresponding factor becomes 0, forcing $K_{stability} = 0$

This prevents $K_{stability}$ from going negative and ensures bounded output $\in [0,1]$.

### 7. kin_ref_phase.py (KIN-Domain Closure C - Phase-Anchor Selection)

**Purpose**: Deterministic phase-anchor selection for oscillatory motion.

**NOTE**: This is a Tier-0/Tier-2 kinematics closure, NOT Tier-1 kernel. Does not redefine Tier-1 symbols {ω, F, S, C, τ_R, IC, κ}.

**CasePack**: `casepacks/kin_ref_phase_oscillator/`

**Main Functions**:
- `compute_phase(x, v)` → φ (phase angle in [0, 2π))
- `circular_distance(a, b)` → Δφ (circular distance in [0, π])
- `build_eligible_set(t)` → list of eligible anchor indices
- `select_phase_anchor(x_series, v_series, t)` → anchor_u, delta_phi, undefined_reason

#### Phase Mapping φ(u) (FROZEN)

Given normalized phase-space coordinates (x, v) ∈ [0,1]²:

$$x' = 2x - 1 \quad \text{(center to [-1,1])}$$
$$v' = 2v - 1 \quad \text{(center to [-1,1])}$$
$$\phi = \text{atan2}(v', x') \quad \text{(wrapped to [0, 2π))}$$

#### Circular Distance Δφ(a, b) (FROZEN)

$$\Delta\phi(a, b) = \min(|a - b|, 2\pi - |a - b|)$$

Result is in [0, π].

#### Selector Rule

1. Build eligible set: $\mathcal{E}(t) = D_{W,\delta}(t)$
2. If $\mathcal{E}(t) = \emptyset$: **undefined** (EMPTY_ELIGIBLE)
3. For each $u \in \mathcal{E}(t)$, compute $\Delta\phi(\phi(t), \phi(u))$
4. Find $\Delta\phi_{min} = \min\{\Delta\phi(\phi(t), \phi(u)) : u \in \mathcal{E}(t)\}$
5. If $\Delta\phi_{min} > \delta\phi_{max}$: **undefined** (PHASE_MISMATCH)
6. Otherwise, select anchor via tie-breakers:
   - (i) Minimize $\Delta\phi$
   - (ii) If tied: choose most recent $u$ (largest $u$)
   - (iii) If still tied: minimize $d(\Psi(t), \Psi(u))$ (Euclidean distance in phase space)

#### Frozen Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\delta\phi_{max}$ | 0.5235987756 ($\pi/6$) | Maximum phase mismatch (30°) |
| $W$ | 20 | Window size (samples) |
| $\delta$ | 3 | Debounce lag (samples) |

#### Edge Cases

- **EMPTY_ELIGIBLE**: $t < \delta$ (startup phase), no eligible anchors
- **PHASE_MISMATCH**: All eligible anchors exceed $\delta\phi_{max}$ threshold
- **Tie-breaker**: Multiple anchors with identical $\Delta\phi$ → select most recent $u$

## Casepack: kinematics_complete

### Scenarios

1. **Projectile Motion**
   - Initial: x=0, v=0.7, a=-0.098
   - Motion: Parabolic trajectory
   - τ_kin: INF_KIN (non-returning)

2. **Simple Harmonic Oscillator**
   - Initial: x=0.5, v=0, a=-0.5
   - Motion: Periodic oscillation
   - τ_kin: Finite (returning)

3. **Damped Oscillator**
   - Initial: x=0.5, v=0, a=-0.4
   - Motion: Decaying oscillation
   - τ_kin: Finite but increasing (weakly returning)

4. **Circular Motion**
   - Initial: theta=0, omega=0.5, alpha=0
   - Motion: Uniform circular
   - τ_kin: Finite (returning)

5. **Elastic Collision**
   - Two-body collision
   - Momentum conserved
   - Energy conserved

### Observables

```yaml
position:
  symbol: x
  unit: normalized
  domain: [0, 1]
  description: "Normalized position coordinate"

velocity:
  symbol: v
  unit: normalized  
  domain: [0, 1]
  description: "Normalized velocity magnitude"

acceleration:
  symbol: a
  unit: normalized
  domain: [-1, 1]
  description: "Normalized acceleration"

kinetic_energy:
  symbol: E_kin
  unit: normalized
  domain: [0, 1]
  description: "Normalized kinetic energy"

momentum:
  symbol: p
  unit: normalized
  domain: [-1, 1]
  description: "Normalized linear momentum"

stability_index:
  symbol: K_stability
  unit: dimensionless
  domain: [0, 1]
  description: "Kinematic stability index"

return_time:
  symbol: tau_kin
  unit: normalized
  domain: [0, INF_KIN]
  description: "Kinematic return time"
```

## Integration with UMCP Architecture

### Tier Mapping

| UMCP Tier | Kinematics Component |
|-----------|---------------------|
| Tier-0 (Protocol) | Contract KIN.INTSTACK.v1, Phase space return detection |
| Tier-1 (Kernel) | Linear/Rotational kinematics |
| Tier-2 (Expansion) | Stability classification |

### Ledger Integration

Kinematic measurements append to UMCP ledger:

```csv
timestamp,scenario,observable,value,regime,tau_kin,K_stability
2025-01-24T12:00:00Z,harmonic,x,0.5,Stable,3.14,0.85
2025-01-24T12:00:01Z,harmonic,v,0.707,Stable,3.14,0.85
```

### Validation Pipeline

```
Raw Measurements (CSV)
        │
        ▼
Closure Computation
        │
        ▼
Phase Space Mapping
        │
        ▼
Return Detection (τ_kin)
        │
        ▼
Stability Classification
        │
        ▼
Ledger Append
        │
        ▼
Receipt Generation
```

## Mathematical Identities

### Conservation Laws (for closed systems)

**Momentum Conservation**:
$$\frac{dp_{total}}{dt} = 0 \quad \text{when} \quad F_{ext} = 0$$

**Energy Conservation**:
$$\frac{dE_{mech}}{dt} = 0 \quad \text{when} \quad W_{nc} = 0$$

### Work-Energy Theorem

$$W_{net} = \Delta E_{kin} = E_{kin,f} - E_{kin,i}$$

### Impulse-Momentum Theorem

$$J = \int F \, dt = \Delta p$$

### Kinematic Relations

$$v = \frac{dx}{dt}, \quad a = \frac{dv}{dt} = \frac{d^2x}{dt^2}$$

$$ω = \frac{dθ}{dt}, \quad α = \frac{dω}{dt}$$

## Debugging and Validation

### Common Issues

1. **Out-of-Range Values**: Position/velocity outside [0,1]
   - Solution: Check normalization, set `oor_flags`

2. **Infinite Return Time**: τ_kin = INF_KIN
   - May be valid (drifting motion) or error (bad phase detection)
   - Check trajectory for periodicity

3. **Energy Non-Conservation**: ΔE_mech > tolerance
   - Check for non-conservative forces (friction, drag)
   - Verify integration accuracy

4. **Kinematic Inconsistency**: v ≠ dx/dt
   - Check time step (dt) accuracy
   - Verify measurement sampling

### Test Commands

```bash
# Run kinematics tests
pytest tests/test_120_kinematics_closures.py -v

# Run with coverage
pytest tests/test_120_kinematics_closures.py --cov=closures.kinematics

# Run specific test class
pytest tests/test_120_kinematics_closures.py::TestPhaseSpaceReturn -v
```

## References

1. **UMCP Core**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md)
2. **Tier System**: [TIER_SYSTEM.md](TIER_SYSTEM.md)
3. **Infrastructure Geometry**: [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md)
4. **Return Axiom**: [AXIOM.md](AXIOM.md)
5. **Contract Format**: [contracts/KIN.INTSTACK.v1.yaml](contracts/KIN.INTSTACK.v1.yaml)

---

*Kinematics Extension v1.0.0 — Extending UMCP to Classical Mechanics*
