# KIN.INTSTACK.v1 — Complete Kinematics Specification

## Finalized Reference for UMCP Kinematics Extension

**Version**: 1.0.0  
**Status**: FROZEN  
**Extends**: UMA.INTSTACK.v1 (Base UMCP Contract)

This document is the single authoritative reference for KIN.INTSTACK.v1. It consolidates all frozen definitions, formulas, constraints, and implementation requirements. A conforming implementation must satisfy every testable clause herein.

---

## Part I: Foundation

### §1. Contract Position

```
UMA.INTSTACK.v1 (Base UMCP Contract)
    │
    └── KIN.INTSTACK.v1 (Kinematics Extension)
            ├── Axioms: KIN-AX-0 through KIN-AX-3
            ├── Closures: 6 domain closures
            ├── Frozen Parameters: 16 constants
            └── Phase Space Return Overlay
```

**Scope**: KIN.INTSTACK.v1 provides classical mechanics observables to UMCP. All KIN closures are **domain closures**—they produce observables and overlays but do NOT compute UMCP Tier-1 kernel invariants (ω, F, S, C, τ_R, κ, IC).

### §2. Symbol Namespace (Kernel Collision Avoidance)

| KIN Symbol | Meaning | NOT Confused With |
|------------|---------|-------------------|
| **Ω** | Angular velocity | ω (UMCP kernel drift) |
| **T** | Torque | τ (UMCP return time prefix) |
| **τ_kin** | Kinematic return time | τ_R (UMCP kernel return time) |
| **S_kin** | Kinematic sources | S (UMCP kernel entropy) |
| **C_kin** | Kinematic consumption | C (UMCP kernel curvature) |
| **K_stability** | Stability index | κ (UMCP kernel curvature) |

### §3. Frozen Parameters (16 Constants)

| Parameter | Symbol | Value | Units | Purpose |
|-----------|--------|-------|-------|---------|
| Window size | $W$ | 64 | samples | Return-domain lookback |
| Debounce lag | $\delta$ | 3 | samples | Spurious re-trigger guard |
| Critical return time | $T_{crit}$ | 10.0 | samples | Return classification threshold |
| Log-safety guard | $\epsilon$ | $10^{-8}$ | — | Prevents log(0) |
| Phase tolerance | $\eta_{phase}$ | 0.01 | — | Squared-L2 return tolerance |
| Momentum tolerance | $\epsilon_p$ | $10^{-6}$ | — | Conservation test bound |
| Energy tolerance | $\epsilon_E$ | $10^{-6}$ | — | Conservation test bound |
| Position std cap | $\sigma_{max}$ | 0.5 | — | K_stability denominator |
| Velocity cap | $v_{max}$ | 1.0 | — | K_stability denominator |
| Stable threshold | $K_{stable}$ | 0.7 | — | Regime boundary |
| Watch threshold | $K_{watch}$ | 0.3 | — | Regime boundary |
| Return rate minimum | — | 0.1 | — | Credit gate |
| Ψ dimension | — | 5 | channels | Observable vector size |
| Length reference | $L_{ref}$ | 1.0 | m | Normalization scale |
| Velocity reference | $v_{ref}$ | 1.0 | m/s | Normalization scale |
| Acceleration reference | $a_{ref}$ | 9.81 | m/s² | Normalization scale |

**Convention**: All temporal quantities (W, δ, τ_kin, T_crit) are in **sample steps**, not seconds.

---

## Part II: State Space

### §4. Physical State

The full kinematic state at discrete time $t \in \mathbb{Z}^{\geq 0}$:

$$\mathcal{S}(t) = (x(t), v(t), a(t), m, t)$$

**Signed Convention (FROZEN)**: `signed_convention=magnitude_only`

$$x := |x|, \quad v := |v|, \quad a := |a|, \quad p := |p|$$

All observables are magnitudes. Directional information is discarded before normalization.

### §5. Bounded Embedding

**Definition (Clipping Operator)**:
$$\text{clip}_{[a,b]}(x) := \max(a, \min(b, x))$$

**Definition (Bounded Embedding)**:
$$\tilde{q}(t) := \text{clip}_{[0,1]}\left(\frac{|q(t)|}{q_{ref}}\right)$$

**OOR Policy**: `clip_and_flag` — values outside [0,1] are clipped and an OOR flag is emitted.

**Definition (Log-Safe Embedding)** for entropy/curvature computations:
$$\tilde{q}_\epsilon(t) := \text{clip}_{[\epsilon, 1-\epsilon]}(\tilde{q}(t))$$

### §6. Normalized Quantities

| Physical | Normalized | Formula | Reference |
|----------|------------|---------|-----------|
| $x(t)$ | $\tilde{x}(t)$ | $\text{clip}_{[0,1]}(\|x\|/L_{ref})$ | 1.0 m |
| $v(t)$ | $\tilde{v}(t)$ | $\text{clip}_{[0,1]}(\|v\|/v_{ref})$ | 1.0 m/s |
| $a(t)$ | $\tilde{a}(t)$ | $\text{clip}_{[0,1]}(\|a\|/a_{ref})$ | 9.81 m/s² |
| $E_{kin}(t)$ | $\tilde{E}_{kin}(t)$ | $\text{clip}_{[0,1]}(E_{kin}/E_{ref})$ | 1.0 J |
| $p(t)$ | $\tilde{p}(t)$ | $\text{clip}_{[0,1]}(\|p\|/p_{ref})$ | 1.0 kg·m/s |

**E_kin Computation Order (FROZEN)**: Compute kinetic energy in physical units *before* normalization:

$$E_{kin}(t) = \frac{1}{2} m(t) v(t)^2 \quad \text{(physical units first)}$$
$$\tilde{E}_{kin}(t) = \text{clip}_{[0,1]}\left(\frac{E_{kin}(t)}{E_{ref}}\right)$$

**Mass Assumption (FROZEN)**: $m(t) = m_{ref} = 1.0\text{ kg}$ (constant). Variable-mass systems are out-of-scope for v1.

### §7. Observable Vector Ψ(t)

The kernel input vector (Tier-0):

$$\Psi(t) = \begin{pmatrix} \tilde{x}_\epsilon(t) \\ \tilde{v}_\epsilon(t) \\ \tilde{a}_\epsilon(t) \\ \tilde{E}_{kin,\epsilon}(t) \\ \tilde{p}_\epsilon(t) \end{pmatrix} \in [\epsilon, 1-\epsilon]^5 \subset [0,1]^5$$

**Frozen Weights** (for weighted metrics):
$$w = (0.25, 0.25, 0.15, 0.20, 0.15), \quad \sum_i w_i = 1$$

### §8. Phase Space

**Definition**: The kinematic phase space is:

$$\Gamma_{kin} := [0,1] \times [0,1]$$

**Phase Point**:
$$\gamma(t) := (\tilde{x}(t), \tilde{v}(t)) \in \Gamma_{kin}$$

**Metric (FROZEN: squared-L2)**:

$$d^2(\gamma_1, \gamma_2) := (\tilde{x}_2 - \tilde{x}_1)^2 + (\tilde{v}_2 - \tilde{v}_1)^2$$

The tolerance $\eta_{phase}$ is a **squared distance** threshold.

**Phase Magnitude**:
$$|\gamma|^2 := \tilde{x}^2 + \tilde{v}^2$$

---

## Part III: Return Axiom Implementation

### §9. Return-Domain Generator

**Startup-Truncated Domain (FORMAL)**:

For all $t \geq 0$:

$$D_W(t) := \{ u \in \mathbb{Z} : \max(0, t-W) \leq u \leq t-1 \}$$

**Effective Domain with Debounce Lag**:

$$D_{W,\delta}(t) := \{ u \in D_W(t) : (t - u) \geq \delta \}$$

**Domain Size**:

$$|D_{W,\delta}(t)| = \begin{cases} 
0 & \text{if } t < \delta \\
t - \delta + 1 & \text{if } \delta \leq t < W \\
W - \delta + 1 = 62 & \text{if } t \geq W
\end{cases}$$

### §10. Valid-Return Set

Given tolerance $\eta_{phase} = 0.01$:

$$U(t) := \{ u \in D_{W,\delta}(t) : d^2(\gamma(t), \gamma(u)) < \eta_{phase} \}$$

### §11. Kinematic Return Time

$$\tau_{kin}(t) := \begin{cases}
\min\{t - u : u \in U(t)\} & \text{if } U(t) \neq \emptyset \\
\text{INF\_KIN} & \text{otherwise}
\end{cases}$$

**Type (FROZEN)**: 
$$\tau_{kin} \in \mathbb{Z}^+ \cup \{\text{INF\_KIN}, \text{UNIDENTIFIABLE\_KIN}\}$$

τ_kin is a **positive integer** (sample count), not a real. The threshold $T_{crit} = 10.0$ remains a real for comparison.

**Typed Sentinels**: INF_KIN and UNIDENTIFIABLE_KIN are typed enum values, NOT IEEE `float('inf')` or `float('nan')`.

### §12. Return Rate

$$\text{return\_rate}(t) := \begin{cases}
0 & \text{if } |D_{W,\delta}(t)| = 0 \\
\displaystyle\frac{|U(t)|}{|D_{W,\delta}(t)|} & \text{otherwise}
\end{cases} \in [0,1]$$

### §13. Return Classification

| Condition | Classification | Physical Meaning |
|-----------|---------------|------------------|
| $0 < \tau_{kin} < T_{crit}$ | Returning | Periodic/oscillatory |
| $T_{crit} \leq \tau_{kin} < 2T_{crit}$ | Partially_Returning | Quasi-periodic |
| $2T_{crit} \leq \tau_{kin} < \infty$ | Weakly_Returning | Damped oscillations |
| $\tau_{kin} = \text{INF\_KIN}$ | Non_Returning | Drifting/divergent |

### §14. Kinematic Credit (Axiom-0)

**Axiom-0**: "Only what returns is real."

$$\text{kinematic\_credit}(t) := \begin{cases}
0 & \text{if } \tau_{kin}(t) = \text{INF\_KIN} \\
0 & \text{if } \text{return\_rate}(t) \leq 0.1 \\
\displaystyle\frac{1}{1 + \tau_{kin}/T_{crit}} \cdot \text{return\_rate}(t) & \text{otherwise}
\end{cases}$$

### §15. Firewall: τ_kin vs τ_R

| Property | τ_kin | τ_R |
|----------|-------|-----|
| Namespace | KIN | UMCP Kernel |
| Space | Phase space $\Gamma_{kin}$ | Observable space $\Psi$ |
| Metric | Squared-L2 | Frozen kernel norm |
| Tolerance | $\eta_{phase}$ | $\eta$ (kernel) |
| Domain | $D_{W,\delta}$ | $D_\theta$ (kernel) |

**τ_kin does NOT substitute for τ_R**. They measure different quantities in different spaces.

---

## Part IV: Kinematics Equations

### §16. Linear Motion

**Kinematic Relations**:
$$v(t) = \frac{dx}{dt}, \qquad a(t) = \frac{dv}{dt}$$

**Equations of Motion** (constant acceleration):
$$x(t) = x_0 + v_0 t + \frac{1}{2}at^2$$
$$v(t) = v_0 + at$$
$$v^2 = v_0^2 + 2a(x - x_0)$$

**Discrete-Time Approximations**:
$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2}a_t \Delta t^2$$
$$v_{t+1} = v_t + a_t \Delta t$$

### §17. Rotational Motion

**Angular Relations** (using Ω to avoid kernel ω):
$$\Omega(t) = \frac{d\theta}{dt}, \qquad \alpha(t) = \frac{d\Omega}{dt}$$

**Equations of Motion**:
$$\theta(t) = \theta_0 + \Omega_0 t + \frac{1}{2}\alpha t^2$$
$$\Omega(t) = \Omega_0 + \alpha t$$

**Angular Momentum**:
$$L = I\Omega$$

**Torque** (using T to avoid kernel τ):
$$T = I\alpha = \frac{dL}{dt}$$

### §18. Circular Motion

**Centripetal Acceleration**:
$$a_c = \frac{v^2}{r} = \Omega^2 r$$

**Period**:
$$T_{period} = \frac{2\pi r}{v} = \frac{2\pi}{\Omega}$$

### §19. Projectile Motion

**Horizontal** (no acceleration):
$$x(t) = x_0 + v_{0x}t$$

**Vertical** (acceleration $-g$):
$$y(t) = y_0 + v_{0y}t - \frac{1}{2}gt^2$$

**Range** (level ground):
$$R = \frac{v_0^2 \sin(2\theta_0)}{g}$$

---

## Part V: Energy and Momentum

### §20. Kinetic Energy

$$E_{kin}(t) = \frac{1}{2}m v(t)^2$$

**Rotational**:
$$E_{kin,rot} = \frac{1}{2}I\Omega^2$$

### §21. Potential Energy

**Gravitational**:
$$E_{pot,grav} = mgh$$

**Elastic**:
$$E_{pot,spring} = \frac{1}{2}kx^2$$

### §22. Mechanical Energy

$$E_{mech} = E_{kin} + E_{pot}$$

### §23. Work and Power

$$W = \int F \cdot dx$$
$$P = F \cdot v$$

**Work-Energy Theorem**:
$$W_{net} = \Delta E_{kin}$$

### §24. Momentum

$$p = mv$$

**Impulse-Momentum Theorem**:
$$J = \int F \, dt = \Delta p$$

### §25. Collisions (Reference Only)

**DISCLAIMER**: Under `signed_convention=magnitude_only`, collision equations are **not physically complete** and are **not enforcement gates** in v1.

**Elastic (1D)**:
$$v_{1f} = \frac{(m_1 - m_2)v_{1i} + 2m_2 v_{2i}}{m_1 + m_2}$$

**Perfectly Inelastic**:
$$v_f = \frac{m_1 v_{1i} + m_2 v_{2i}}{m_1 + m_2}$$

---

## Part VI: Conservation Laws

### §26. Momentum Conservation (Conditional)

**Precondition**: $F_{ext} = 0$

$$\frac{dp_{total}}{dt} = 0$$

**Discrete Test**:
$$|p_{total}(t+1) - p_{total}(t)| < \epsilon_p = 10^{-6}$$

### §27. Energy Conservation (Conditional)

**Precondition**: $W_{nc} = 0$ (no non-conservative work)

$$\frac{dE_{mech}}{dt} = 0$$

**Discrete Test**:
$$|E_{mech}(t+1) - E_{mech}(t)| < \epsilon_E = 10^{-6}$$

### §28. Angular Momentum Conservation

**Precondition**: $T_{ext} = 0$ (no external torque)

$$\frac{dL_{total}}{dt} = 0$$

### §29. Precondition Gating

Conservation tests are **not applicable** when preconditions are not met. Tests must be gated:
- Momentum test requires $F_{ext} = 0$
- Energy test requires $W_{nc} = 0$
- Angular momentum test requires $T_{ext} = 0$

---

## Part VII: Stability Analysis

### §30. Stability Index K

$$K_{stability}(t) := \left(1 - \frac{\sigma_x(t)}{\sigma_{max}}\right) \cdot \left(1 - \frac{v_{mean}(t)}{v_{max}}\right) \cdot w_\tau(t)$$

**Frozen Definitions**:

| Component | Definition | Frozen Value |
|-----------|------------|--------------|
| $\sigma_x(t)$ | Sample std of $\tilde{x}_\epsilon$ over $D_W(t)$, ddof=1 | computed |
| $\sigma_{max}$ | Fixed constant | 0.5 |
| $v_{mean}(t)$ | Mean of $\tilde{v}_\epsilon$ over $D_W(t)$ | computed |
| $v_{max}$ | Fixed constant | 1.0 |
| $w_\tau(t)$ | Return weight | return_rate(t) |

### §31. Sample Statistics

**Sample Mean**:
$$\bar{q}(t) := \frac{1}{|D_W(t)|} \sum_{u \in D_W(t)} q(u)$$

**Sample Standard Deviation** (Bessel-corrected, ddof=1):
$$\sigma_q(t) := \sqrt{\frac{1}{|D_W(t)| - 1} \sum_{u \in D_W(t)} (q(u) - \bar{q}(t))^2}$$

**Startup Guard**: If $|D_W(t)| < 2$:
- Set $\sigma_x(t) := 0$
- Set $K_{stability}(t) := 0$
- Emit STARTUP flag

### §32. Clip+Flag Policy for K

If any ratio exceeds 1.0:
$$\frac{\sigma_x}{\sigma_{max}} > 1 \Rightarrow \text{clip to 1, emit OOR, factor} = 0$$
$$\frac{v_{mean}}{v_{max}} > 1 \Rightarrow \text{clip to 1, emit OOR, factor} = 0$$

**Result**: $K_{stability} \in [0, 1]$ is guaranteed.

### §33. Regime Classification

| Condition | Regime |
|-----------|--------|
| $K_{stability} > 0.7$ | Stable |
| $0.3 < K_{stability} \leq 0.7$ | Watch |
| $K_{stability} \leq 0.3$ | Unstable |

### §34. Kinematic Budget Equation (Overlay Only)

$$\Delta K_{kin} = S_{kin} - C_{kin} + R_{kin}$$

**DISCLAIMER**: This is a kinematics overlay equation only. It is NOT the UMCP kernel continuity budget law and CANNOT certify seam continuity.

---

## Part VIII: Oscillatory Motion (Reference)

**NOTE**: §35–§37 are classical reference identities, **not enforcement gates** unless explicitly invoked.

### §35. Simple Harmonic Motion

$$\frac{d^2x}{dt^2} + \omega_0^2 x = 0$$

**Solution**:
$$x(t) = A\cos(\omega_0 t + \phi)$$
$$v(t) = -A\omega_0\sin(\omega_0 t + \phi)$$

**Phase Space**: Ellipse in $(x, v)$.

### §36. Damped Harmonic Motion

$$\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2 x = 0$$

**Underdamped** ($\gamma < \omega_0$):
$$x(t) = Ae^{-\gamma t}\cos(\omega_d t + \phi)$$

where $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$.

### §37. Lyapunov Stability Estimate

**Note**: $\Delta\gamma$ here denotes phase perturbation, unrelated to debounce lag $\delta$.

$$\lambda_{max} = \lim_{t \to \infty} \frac{1}{t} \ln\frac{|\Delta\gamma(t)|}{|\Delta\gamma(0)|}$$

| $\lambda_{max}$ | Behavior |
|-----------------|----------|
| $< 0$ | Stable (converging) |
| $= 0$ | Neutral (periodic) |
| $> 0$ | Chaotic |

---

## Part IX: Formal Enforcement Axioms

### KIN-AX-0: Observable Boundedness

$$\Psi(t) \in [0,1]^5 \text{ with clip+flag OOR policy}$$

**Test**: Any channel value outside [0,1] triggers OOR flag and is clipped.

### KIN-AX-1: Return Time Well-Typed

$$\tau_{kin} \in \mathbb{Z}^+ \cup \{\text{INF\_KIN}, \text{UNIDENTIFIABLE\_KIN}\}$$

**Test**: Reject `float('inf')` or `float('nan')`. Only typed sentinels allowed.

### KIN-AX-2: Conservation (Conditional)

$$F_{ext} = 0 \Rightarrow |dp/dt| < \epsilon_p$$
$$W_{nc} = 0 \Rightarrow |dE_{mech}/dt| < \epsilon_E$$

**Test**: Conservation tests only run when preconditions are met.

### KIN-AX-3: Stability Index Bounded

$$K_{stability} \in [0, 1]$$

**Test**: K_stability values outside [0,1] are rejected.

---

## Part X: Closure Inventory

### Closure A: Kinematic Computations

| File | Purpose |
|------|---------|
| `linear_kinematics.py` | Linear motion, trajectory, consistency |
| `rotational_kinematics.py` | Angular motion, torque, centripetal |
| `energy_mechanics.py` | Energy, work, power, conservation |
| `momentum_dynamics.py` | Momentum, impulse, collisions |

### Closure B: Return and Stability Overlays

| File | Purpose |
|------|---------|
| `phase_space_return.py` | τ_kin, return_rate, credit, Lyapunov |
| `kinematic_stability.py` | K_stability, regime, budget |

### Reference Implementation Types

```python
from enum import Enum
from typing import Union

class KinSpecialValue(Enum):
    INF_KIN = "INF_KIN"
    UNIDENTIFIABLE_KIN = "UNIDENT"

TauKin = Union[int, KinSpecialValue]
```

---

## Part XI: Test Suite Summary

The audit-grade test suite (`test_130_kin_audit_spec.py`) validates:

| Category | Tests | Purpose |
|----------|-------|---------|
| Contract Freeze | 3 | Verify frozen constants |
| Embedding | 3 | Magnitude-only, E_kin order, Ψ dimension |
| Phase Space | 2 | Metric, tolerance interpretation |
| Domain Generator | 4 | Size, startup, debounce |
| Return Mechanics | 3 | τ_kin type, classification, rate formula |
| Credit | 2 | Axiom-0 enforcement |
| Stability K | 3 | Formula, ddof, clip+flag |
| Typed Sentinels | 2 | INF_KIN/UNIDENTIFIABLE_KIN handling |
| Firewall | 1 | τ_kin ≠ τ_R |
| Conservation | 2 | Conditional enforcement |
| Reference Identities | 4 | E_kin, momentum, work-energy, kinematics |
| Gold Vectors | 6 | Known-answer tests |

**Total**: 35 tests covering all frozen parameters and formulas.

---

## Part XII: Glossary

| Symbol | Meaning |
|--------|---------|
| $x$ | Position (magnitude) |
| $v$ | Velocity (magnitude) |
| $a$ | Acceleration (magnitude) |
| $m$ | Mass |
| $p$ | Linear momentum |
| $E_{kin}$ | Kinetic energy |
| $E_{pot}$ | Potential energy |
| $E_{mech}$ | Mechanical energy |
| $\Omega$ | Angular velocity (avoiding kernel ω) |
| $\alpha$ | Angular acceleration |
| $L$ | Angular momentum |
| $T$ | Torque (avoiding kernel τ) |
| $I$ | Moment of inertia |
| $\tau_{kin}$ | Kinematic return time (samples) |
| $\tau_R$ | Kernel return time (NOT τ_kin) |
| $\gamma(t)$ | Phase point $(\tilde{x}, \tilde{v})$ |
| $\Gamma_{kin}$ | Phase space $[0,1]^2$ |
| $\Psi(t)$ | Observable vector |
| $K_{stability}$ | Stability index |
| $W$ | Window size (64 samples) |
| $\delta$ | Debounce lag (3 samples) |
| $\eta_{phase}$ | Phase return tolerance (0.01) |
| $T_{crit}$ | Critical return threshold (10.0) |
| $\epsilon$ | Log-safety guard ($10^{-8}$) |

---

## Part XIII: References

| Document | Purpose |
|----------|---------|
| [KINEMATICS_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md) | Original specification with rationale |
| [KINEMATICS_MATHEMATICS.md](KINEMATICS_MATHEMATICS.md) | Extended mathematical formalism |
| [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) | UMCP kernel reference |
| [TIER_SYSTEM.md](TIER_SYSTEM.md) | System tier boundaries |
| [AXIOM.md](AXIOM.md) | UMCP return axiom |
| [contracts/KIN.INTSTACK.v1.yaml](contracts/KIN.INTSTACK.v1.yaml) | YAML contract file |

---

*KIN.INTSTACK.v1 — Complete Kinematics Specification*  
*Frozen for v1 Conformance*
