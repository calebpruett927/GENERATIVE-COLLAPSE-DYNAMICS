# Kinematics Mathematics

## Complete Mathematical Formalism for KIN.INTSTACK.v1

This document provides the full mathematical specification for the UMCP Kinematics Extension. All definitions are frozen for v1 conformance.

---

## §1. Notation and Conventions

### 1.1 Index Sets and Domains

| Symbol | Definition |
|--------|------------|
| $\mathbb{Z}^+$ | Positive integers $\{1, 2, 3, \ldots\}$ |
| $\mathbb{Z}^{\geq 0}$ | Non-negative integers $\{0, 1, 2, \ldots\}$ |
| $\mathbb{R}^+$ | Positive reals $(0, \infty)$ |
| $\mathbb{R}^{\geq 0}$ | Non-negative reals $[0, \infty)$ |
| $[a, b]$ | Closed interval $\{x \in \mathbb{R} : a \leq x \leq b\}$ |
| $t$ | Discrete time index, $t \in \mathbb{Z}^{\geq 0}$ |

### 1.2 Time Units (FROZEN)

All temporal quantities are in **sample steps** (discrete index units):

$$\Delta t_{sample} : \text{physical time per sample (seconds/sample)}$$

| Quantity | Symbol | Units | Frozen Value |
|----------|--------|-------|--------------|
| Window size | $W$ | samples | 64 |
| Debounce lag | $\delta$ | samples | 3 |
| Critical return time | $T_{crit}$ | samples | 10.0 |
| Kinematic return time | $\tau_{kin}$ | samples | computed |

**Physical time conversion**:
$$\tau_{kin,seconds} = \tau_{kin} \cdot \Delta t_{sample}$$

### 1.3 Reference Scales (FROZEN)

| Quantity | Symbol | Reference | Value |
|----------|--------|-----------|-------|
| Length | $L_{ref}$ | Reference length | 1.0 m |
| Velocity | $v_{ref}$ | Reference velocity | 1.0 m/s |
| Acceleration | $a_{ref}$ | Reference acceleration | 9.81 m/s² |
| Mass | $m_{ref}$ | Reference mass | 1.0 kg |
| Time | $T_{ref}$ | Reference time | $L_{ref}/v_{ref}$ = 1.0 s |
| Energy | $E_{ref}$ | Reference energy | $m_{ref} \cdot v_{ref}^2$ = 1.0 J |
| Momentum | $p_{ref}$ | Reference momentum | $m_{ref} \cdot v_{ref}$ = 1.0 kg·m/s |

---

## §2. State Space and Phase Space

### 2.1 Physical State Space

The full kinematic state at time $t$ is:

$$\mathcal{S}(t) = (x(t), v(t), a(t), m, t) \in \mathbb{R}^{\geq 0} \times \mathbb{R}^{\geq 0} \times \mathbb{R}^{\geq 0} \times \mathbb{R}^+ \times \mathbb{Z}^{\geq 0}$$

**Convention**: Under `signed_convention=magnitude_only`, all kinematic quantities are magnitudes:

$$x := |x|, \quad v := |v|, \quad a := |a|, \quad p := |p|$$

### 2.2 Bounded Embedding

**Definition (Clipping Operator)**:
$$\text{clip}_{[a,b]}(x) := \max(a, \min(b, x))$$

**Definition (Bounded Embedding)** (using magnitudes per §2.1 convention):
$$\tilde{q}(t) := \text{clip}_{[0,1]}\left(\frac{|q(t)|}{q_{ref}}\right)$$

with OOR (out-of-range) flag:
$$\text{OOR}(q, t) := \mathbf{1}\left[\frac{|q(t)|}{q_{ref}} \notin [0,1]\right]$$

**Definition (Log-Safe Embedding)**:

For channels used in logarithmic computations, with $\epsilon = 10^{-8}$:

$$\tilde{q}_\epsilon(t) := \text{clip}_{[\epsilon, 1-\epsilon]}(\tilde{q}(t))$$

### 2.3 Normalized State Vector

| Physical | Normalized | Formula |
|----------|------------|---------|
| $x(t)$ | $\tilde{x}(t)$ | $\text{clip}_{[0,1]}(x/L_{ref})$ |
| $v(t)$ | $\tilde{v}(t)$ | $\text{clip}_{[0,1]}(v/v_{ref})$ |
| $a(t)$ | $\tilde{a}(t)$ | $\text{clip}_{[0,1]}(a/a_{ref})$ |
| $E_{kin}(t)$ | $\tilde{E}_{kin}(t)$ | $\text{clip}_{[0,1]}(E_{kin}/E_{ref})$ |
| $p(t)$ | $\tilde{p}(t)$ | $\text{clip}_{[0,1]}(p/p_{ref})$ |

### 2.4 Observable Vector Ψ(t)

The kernel input vector (Tier-0):

$$\Psi(t) = \begin{pmatrix} \tilde{x}_\epsilon(t) \\ \tilde{v}_\epsilon(t) \\ \tilde{a}_\epsilon(t) \\ \tilde{E}_{kin,\epsilon}(t) \\ \tilde{p}_\epsilon(t) \end{pmatrix} \in [\epsilon, 1-\epsilon]^5 \subset [0,1]^5$$

**Frozen weights** (for weighted metrics):
$$w = (w_x, w_v, w_a, w_E, w_p) = (0.25, 0.25, 0.15, 0.20, 0.15), \quad \sum_i w_i = 1$$

### 2.5 Phase Space

**Definition**: The kinematic phase space is the 2D projection:

$$\Gamma_{kin} := [0,1] \times [0,1]$$

**Phase point**:
$$\gamma(t) := (\tilde{x}(t), \tilde{v}(t)) \in \Gamma_{kin}$$

**Metric (FROZEN: squared-L2)**:

Let $\gamma_i = (\tilde{x}_i, \tilde{v}_i) \in \Gamma_{kin}$. The squared distance is:

$$d^2(\gamma_1, \gamma_2) := (\tilde{x}_2 - \tilde{x}_1)^2 + (\tilde{v}_2 - \tilde{v}_1)^2$$

$$d(\gamma_1, \gamma_2) := \sqrt{d^2(\gamma_1, \gamma_2)}$$

(Note: return detection uses $d^2$; the $d$ form is an optional helper.)

**Phase magnitude**:
$$|\gamma|^2 := \tilde{x}^2 + \tilde{v}^2, \qquad |\gamma| := \sqrt{\tilde{x}^2 + \tilde{v}^2}$$

---

## §3. Fundamental Kinematic Equations

### 3.1 Linear Motion

**Kinematic relations** (continuous-time):

$$v(t) = \frac{dx}{dt}, \qquad a(t) = \frac{dv}{dt} = \frac{d^2x}{dt^2}$$

**Equations of motion** (constant acceleration $a$):

$$x(t) = x_0 + v_0 t + \frac{1}{2}at^2$$

$$v(t) = v_0 + at$$

$$v^2 = v_0^2 + 2a(x - x_0)$$

**Discrete-time approximations** (sample period $\Delta t$):

$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2}a_t \Delta t^2$$

$$v_{t+1} = v_t + a_t \Delta t$$

### 3.2 Rotational Motion

**Angular kinematic relations**:

$$\Omega(t) = \frac{d\theta}{dt}, \qquad \alpha(t) = \frac{d\Omega}{dt}$$

where $\Omega$ denotes angular velocity (avoiding collision with kernel $\omega$).

**Equations of motion** (constant angular acceleration $\alpha$):

$$\theta(t) = \theta_0 + \Omega_0 t + \frac{1}{2}\alpha t^2$$

$$\Omega(t) = \Omega_0 + \alpha t$$

$$\Omega^2 = \Omega_0^2 + 2\alpha(\theta - \theta_0)$$

**Angular momentum**:
$$L = I\Omega$$

**Torque** (using $T$ to avoid collision with kernel $\tau$):
$$T = I\alpha = \frac{dL}{dt}$$

### 3.3 Circular Motion

**Centripetal acceleration**:
$$a_c = \frac{v^2}{r} = \Omega^2 r$$

**Centripetal force**:
$$F_c = ma_c = \frac{mv^2}{r} = m\Omega^2 r$$

**Tangential acceleration**:
$$a_t = \alpha r$$

**Total acceleration** (circular motion):
$$a_{total} = \sqrt{a_c^2 + a_t^2}$$

### 3.4 Projectile Motion

For projectile motion under gravity (2D, with $g = a_{ref}$):

**Horizontal** (no acceleration):
$$x(t) = x_0 + v_{0x}t$$

**Vertical** (constant acceleration $-g$):
$$y(t) = y_0 + v_{0y}t - \frac{1}{2}gt^2$$

$$v_y(t) = v_{0y} - gt$$

**Trajectory equation**:
$$y = y_0 + (x - x_0)\tan\theta_0 - \frac{g(x-x_0)^2}{2v_0^2\cos^2\theta_0}$$

**Range** (level ground, $y_0 = 0$):
$$R = \frac{v_0^2 \sin(2\theta_0)}{g}$$

**Maximum height**:
$$h_{max} = \frac{v_0^2 \sin^2\theta_0}{2g}$$

---

## §4. Energy

### 4.1 Kinetic Energy

**Definition**:
$$E_{kin}(t) = \frac{1}{2}m(t)v(t)^2$$

**Mass Assumption (FROZEN for v1)**: $m(t) = m_{ref}$ (constant). Variable-mass systems are out-of-scope.

**Rotational kinetic energy**:
$$E_{kin,rot} = \frac{1}{2}I\Omega^2$$

**Normalized kinetic energy**:
$$\tilde{E}_{kin}(t) = \text{clip}_{[0,1]}\left(\frac{E_{kin}(t)}{E_{ref}}\right), \qquad E_{ref} = m_{ref} \cdot v_{ref}^2$$

**Computation Order (FROZEN)**: Compute $E_{kin}$ in physical units *before* normalization to avoid clipping artifacts.

### 4.2 Potential Energy

**Gravitational potential energy**:
$$E_{pot,grav} = mgh$$

**Elastic potential energy** (spring, stiffness $k$):
$$E_{pot,spring} = \frac{1}{2}kx^2$$

**General conservative force**:
$$F = -\nabla E_{pot}$$

### 4.3 Mechanical Energy

**Total mechanical energy**:
$$E_{mech} = E_{kin} + E_{pot}$$

**Conservation** (conditional—see §7):
$$\frac{dE_{mech}}{dt} = 0 \quad \text{iff} \quad W_{nc} = 0$$

where $W_{nc}$ is work done by non-conservative forces.

### 4.4 Work and Power

**Work** (constant force):
$$W = F \cdot d \cdot \cos\theta$$

**Work** (variable force):
$$W = \int_{x_1}^{x_2} F(x) \, dx$$

**Power**:
$$P = \frac{dW}{dt} = F \cdot v$$

**Average power**:
$$\bar{P} = \frac{W}{\Delta t}$$

### 4.5 Work-Energy Theorem

$$W_{net} = \Delta E_{kin} = E_{kin,f} - E_{kin,i}$$

$$W_{net} = \frac{1}{2}mv_f^2 - \frac{1}{2}mv_i^2$$

---

## §5. Momentum

### 5.1 Linear Momentum

**Definition**:
$$p = mv$$

**Newton's Second Law**:
$$F = \frac{dp}{dt} = ma \quad \text{(constant mass)}$$

**Normalized momentum**:
$$\tilde{p}(t) = \text{clip}_{[0,1]}\left(\frac{p(t)}{p_{ref}}\right), \qquad p_{ref} = m_{ref} \cdot v_{ref}$$

### 5.2 Impulse

**Definition**:
$$J = \int_{t_1}^{t_2} F(t) \, dt$$

**Constant force**:
$$J = F \cdot \Delta t$$

**Impulse-Momentum Theorem**:
$$J = \Delta p = p_f - p_i$$

### 5.3 Angular Momentum

**Definition**:
$$L = I\Omega$$

**For a point mass**:
$$L = r \times p = mrv\sin\theta$$

**Torque-angular momentum relation**:
$$T = \frac{dL}{dt}$$

### 5.4 Collisions (Illustrative Only)

**DISCLAIMER**: Under `signed_convention=magnitude_only`, collision equations are **not physically complete** and are **not enforcement gates** in v1. These are classical reference identities only.

**Elastic collision (1D)**:
$$m_1 v_{1i} + m_2 v_{2i} = m_1 v_{1f} + m_2 v_{2f}$$
$$\frac{1}{2}m_1 v_{1i}^2 + \frac{1}{2}m_2 v_{2i}^2 = \frac{1}{2}m_1 v_{1f}^2 + \frac{1}{2}m_2 v_{2f}^2$$

**Final velocities (elastic, 1D)**:
$$v_{1f} = \frac{(m_1 - m_2)v_{1i} + 2m_2 v_{2i}}{m_1 + m_2}$$
$$v_{2f} = \frac{(m_2 - m_1)v_{2i} + 2m_1 v_{1i}}{m_1 + m_2}$$

**Perfectly inelastic collision**:
$$v_f = \frac{m_1 v_{1i} + m_2 v_{2i}}{m_1 + m_2}$$

**Coefficient of restitution**:
$$e = \frac{v_{2f} - v_{1f}}{v_{1i} - v_{2i}}$$

| Collision Type | $e$ |
|---------------|-----|
| Perfectly elastic | 1 |
| Inelastic | $0 < e < 1$ |
| Perfectly inelastic | 0 |

---

## §6. Return Axiom Mapping

### 6.1 Return-Domain Generator

**Startup-Truncated Domain (FORMAL)**:

For all $t \geq 0$:

$$D_W(t) := \{ u \in \mathbb{Z} : \max(0, t-W) \leq u \leq t-1 \}$$

**Effective domain with debounce lag**:

$$D_{W,\delta}(t) := \{ u \in D_W(t) : (t - u) \geq \delta \}$$

**Domain size**:

$$|D_{W,\delta}(t)| = \begin{cases} 
0 & \text{if } t < \delta \\
t - \delta + 1 & \text{if } \delta \leq t < W \\
W - \delta + 1 = 62 & \text{if } t \geq W
\end{cases}$$

### 6.2 Valid-Return Set

Given tolerance $\eta_{phase} > 0$:

$$U(t) := \{ u \in D_{W,\delta}(t) : d^2(\gamma(t), \gamma(u)) < \eta_{phase} \}$$

where $d^2$ is the squared-L2 distance in phase space.

### 6.3 Kinematic Return Time

$$\tau_{kin}(t) := \begin{cases}
\min\{t - u : u \in U(t)\} & \text{if } U(t) \neq \emptyset \\
\text{INF\_KIN} & \text{otherwise}
\end{cases}$$

**Type**: $\tau_{kin}(t) \in \mathbb{Z}^+ \cup \{\text{INF\_KIN}, \text{UNIDENTIFIABLE\_KIN}\}$ — a positive integer (samples), not a real. $T_{crit}$ remains a real threshold (10.0).

### 6.4 Return Rate

$$\text{return\_rate}(t) := \begin{cases}
0 & \text{if } |D_{W,\delta}(t)| = 0 \\
\displaystyle\frac{|U(t)|}{|D_{W,\delta}(t)|} & \text{otherwise}
\end{cases} \in [0,1]$$

**Interpretation**: Fraction of effective domain with valid returns. If the domain is empty (startup, $t < \delta$), return_rate is defined as 0, yielding no credit (consistent with Axiom-0).

### 6.5 Return Classification

| Condition | Classification | Physical Meaning |
|-----------|---------------|------------------|
| $0 < \tau_{kin} < T_{crit}$ | Returning | Periodic/oscillatory |
| $T_{crit} \leq \tau_{kin} < 2T_{crit}$ | Partially_Returning | Quasi-periodic |
| $2T_{crit} \leq \tau_{kin} < \infty$ | Weakly_Returning | Damped oscillations |
| $\tau_{kin} = \text{INF\_KIN}$ | Non_Returning | Drifting/divergent |

### 6.6 Kinematic Credit (Axiom-0)

**Axiom-0**: "Only what returns is real."

$$\text{kinematic\_credit}(t) := \begin{cases}
0 & \text{if } \tau_{kin}(t) = \text{INF\_KIN} \\
0 & \text{if } \text{return\_rate}(t) \leq 0.1 \\
\displaystyle\frac{1}{1 + \tau_{kin}/T_{crit}} \cdot \text{return\_rate}(t) & \text{otherwise}
\end{cases}$$

### 6.7 Typed Infinity

**Definition**: INF_KIN is a typed sentinel value, distinct from IEEE `Inf`/`NaN`:

```
KinSpecialValue := {INF_KIN, UNIDENTIFIABLE_KIN}
τ_kin ∈ ℤ⁺ ∪ KinSpecialValue
```

Untyped `float('inf')` or `float('nan')` are rejected by KIN-AX-1.

### 6.8 Firewall: τ_kin vs τ_R

| Property | τ_kin | τ_R |
|----------|-------|-----|
| Namespace | KIN | UMCP Kernel |
| Space | Phase space $\Gamma_{kin}$ | Observable space $\Psi$ |
| Metric | Squared-L2 | Frozen kernel norm |
| Tolerance | $\eta_{phase}$ | $\eta$ (kernel) |
| Domain | $D_{W,\delta}$ | $D_\theta$ (kernel) |

**τ_kin does NOT substitute for τ_R**. They measure different quantities in different spaces.

---

## §7. Conservation Laws

### 7.1 Momentum Conservation (Conditional)

**Statement**: If $F_{ext} = 0$, then:

$$\frac{dp_{total}}{dt} = 0$$

**Discrete-time form**:
$$|p_{total}(t+1) - p_{total}(t)| < \epsilon_p$$

**Frozen tolerance**: $\epsilon_p = 10^{-6}$

**Multi-body system**:
$$p_{total} = \sum_{i=1}^{N} m_i v_i$$

### 7.2 Energy Conservation (Conditional)

**Statement**: If $W_{nc} = 0$ (no non-conservative work), then:

$$\frac{dE_{mech}}{dt} = 0$$

**Discrete-time form**:
$$|E_{mech}(t+1) - E_{mech}(t)| < \epsilon_E$$

**Frozen tolerance**: $\epsilon_E = 10^{-6}$

### 7.3 Angular Momentum Conservation

**Statement**: If $T_{ext} = 0$ (no external torque), then:

$$\frac{dL_{total}}{dt} = 0$$

### 7.4 Non-Applicability

**Important**: Conservation tests are **not applicable** when preconditions are not met:
- Momentum test requires $F_{ext} = 0$
- Energy test requires $W_{nc} = 0$
- Angular momentum test requires $T_{ext} = 0$

Tests should be gated on these preconditions.

---

## §8. Stability Analysis

### 8.1 Stability Index K

**Definition**:

$$K_{stability}(t) := \left(1 - \frac{\sigma_x(t)}{\sigma_{max}}\right) \cdot \left(1 - \frac{v_{mean}(t)}{v_{max}}\right) \cdot w_\tau(t)$$

**Frozen Components**:

| Component | Definition | Frozen Value |
|-----------|------------|--------------|
| $\sigma_x(t)$ | Sample std of $\tilde{x}_\epsilon$ over $D_W(t)$, ddof=1 | computed |
| $\sigma_{max}$ | Fixed constant | 0.5 |
| $v_{mean}(t)$ | Mean of $\tilde{v}_\epsilon$ over $D_W(t)$ | computed |
| $v_{max}$ | Fixed constant | 1.0 |
| $w_\tau(t)$ | Return weight | $\text{return\_rate}(t)$ |

### 8.2 Sample Statistics

**Sample mean**:
$$\bar{q}(t) := \frac{1}{|D_W(t)|} \sum_{u \in D_W(t)} q(u)$$

**Sample standard deviation** (Bessel-corrected):
$$\sigma_q(t) := \sqrt{\frac{1}{|D_W(t)| - 1} \sum_{u \in D_W(t)} (q(u) - \bar{q}(t))^2}$$

**Startup Guard (|D_W(t)| < 2)**: If $|D_W(t)| < 2$, the sample standard deviation is undefined (division by zero). In this case:
- Set $\sigma_x(t) := 0$
- Set $K_{stability}(t) := 0$
- Emit STARTUP flag (or UNIDENTIFIABLE_KIN)

This makes startup behavior deterministic across implementations.

### 8.3 Clip+Flag Policy for K

If any ratio exceeds 1.0:

$$\frac{\sigma_x}{\sigma_{max}} > 1 \quad \Rightarrow \quad \text{clip to 1, emit OOR, factor} = 0$$

$$\frac{v_{mean}}{v_{max}} > 1 \quad \Rightarrow \quad \text{clip to 1, emit OOR, factor} = 0$$

**Result**: $K_{stability} \in [0, 1]$ is guaranteed.

### 8.4 Regime Classification

| Condition | Regime |
|-----------|--------|
| $K_{stability} > K_{stable}$ | Stable |
| $K_{watch} < K_{stability} \leq K_{stable}$ | Watch |
| $K_{stability} \leq K_{watch}$ | Unstable |

**Frozen thresholds**:
- $K_{stable} = 0.7$
- $K_{watch} = 0.3$

### 8.5 Kinematic Budget Equation (Overlay Only)

$$\Delta K_{kin} = S_{kin} - C_{kin} + R_{kin}$$

**DISCLAIMER**: This is a **kinematics overlay equation only**. It is NOT the UMCP kernel weld/continuity budget law and CANNOT certify continuity across seams.

---

## §9. Oscillatory Motion

**REFERENCE IDENTITIES (NON-GATING)**: §9–§11 are classical kinematics reference identities provided for completeness. They are **not used as KIN.INTSTACK.v1 enforcement gates** unless explicitly invoked by an overlay. Symbols like $\omega_0$ in oscillator equations are standard physics notation and are distinct from the UMCP kernel drift symbol $\omega$.

### 9.1 Simple Harmonic Motion

**Equation of motion**:
$$\frac{d^2x}{dt^2} + \omega_0^2 x = 0$$

**Solution**:
$$x(t) = A\cos(\omega_0 t + \phi)$$
$$v(t) = -A\omega_0\sin(\omega_0 t + \phi)$$

**Angular frequency**:
$$\omega_0 = \sqrt{\frac{k}{m}} \quad \text{(spring-mass)}$$
$$\omega_0 = \sqrt{\frac{g}{L}} \quad \text{(simple pendulum, small angles)}$$

**Period**:
$$T = \frac{2\pi}{\omega_0}$$

**Energy**:
$$E = \frac{1}{2}kA^2 = \frac{1}{2}m\omega_0^2 A^2$$

### 9.2 Damped Harmonic Motion

**Equation of motion**:
$$\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2 x = 0$$

where $\gamma$ is the damping coefficient.

**Solution** (underdamped, $\gamma < \omega_0$):
$$x(t) = Ae^{-\gamma t}\cos(\omega_d t + \phi)$$

where $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$.

**Damping regimes**:

| Condition | Regime | Behavior |
|-----------|--------|----------|
| $\gamma < \omega_0$ | Underdamped | Oscillates with decay |
| $\gamma = \omega_0$ | Critically damped | Fastest non-oscillatory decay |
| $\gamma > \omega_0$ | Overdamped | Slow exponential decay |

### 9.3 Forced Harmonic Motion

**Equation of motion**:
$$\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2 x = \frac{F_0}{m}\cos(\omega t)$$

**Steady-state amplitude**:
$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + (2\gamma\omega)^2}}$$

**Resonance** occurs at $\omega \approx \omega_0$ (for weak damping).

### 9.4 Phase Space Trajectories

**SHM (conservative)**:
$$\frac{x^2}{A^2} + \frac{v^2}{(\omega_0 A)^2} = 1$$

This is an ellipse in $(x, v)$ phase space.

**Damped oscillator**: Spiral toward origin.

**Limit cycle**: Closed curve (sustained oscillation).

---

## §10. Special Cases and Limits

### 10.1 Free Fall

$$x(t) = x_0 + v_0 t + \frac{1}{2}gt^2$$
$$v(t) = v_0 + gt$$

Terminal velocity (with drag):
$$v_{terminal} = \sqrt{\frac{2mg}{\rho C_D A}}$$

### 10.2 Uniform Circular Motion

$$a_c = \frac{v^2}{r} = \Omega^2 r, \qquad v = \Omega r$$

**Period**:
$$T = \frac{2\pi r}{v} = \frac{2\pi}{\Omega}$$

### 10.3 Central Force Motion

For central force $F(r) \hat{r}$:

**Angular momentum conserved**:
$$L = mr^2\dot{\theta} = \text{const}$$

**Energy**:
$$E = \frac{1}{2}m\dot{r}^2 + \frac{L^2}{2mr^2} + U(r)$$

### 10.4 Two-Body Problem

**Reduced mass**:
$$\mu = \frac{m_1 m_2}{m_1 + m_2}$$

**Center of mass**:
$$\vec{R}_{cm} = \frac{m_1\vec{r}_1 + m_2\vec{r}_2}{m_1 + m_2}$$

---

## §11. Lyapunov Stability Estimate

**Note**: In this section, $\Delta\gamma$ denotes an infinitesimal perturbation of the phase point and is **unrelated** to the debounce lag $\delta$ defined in §1.2.

### 11.1 Definition

The maximal Lyapunov exponent characterizes trajectory divergence:

$$\lambda_{max} = \lim_{t \to \infty} \frac{1}{t} \ln\frac{|\Delta\gamma(t)|}{|\Delta\gamma(0)|}$$

### 11.2 Interpretation

| $\lambda_{max}$ | Behavior |
|-----------------|----------|
| $\lambda_{max} < 0$ | Stable (trajectories converge) |
| $\lambda_{max} = 0$ | Neutral (periodic or quasi-periodic) |
| $\lambda_{max} > 0$ | Chaotic (sensitive to initial conditions) |

### 11.3 Numerical Estimate

For discrete samples with perturbation separation $\Delta\gamma(t)$:

$$\lambda_{max} \approx \frac{1}{N\Delta t} \sum_{i=1}^{N} \ln\frac{|\Delta\gamma(t_i)|}{|\Delta\gamma(t_{i-1})|}$$

---

## §12. Summary of Frozen Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Window size | $W$ | 64 | samples |
| Debounce lag | $\delta$ | 3 | samples |
| Critical return time | $T_{crit}$ | 10.0 | samples |
| Log-safety guard | $\epsilon$ | $10^{-8}$ | — |
| Momentum tolerance | $\epsilon_p$ | $10^{-6}$ | — |
| Energy tolerance | $\epsilon_E$ | $10^{-6}$ | — |
| Position std cap | $\sigma_{max}$ | 0.5 | — |
| Velocity cap | $v_{max}$ | 1.0 | — |
| Stable threshold | $K_{stable}$ | 0.7 | — |
| Watch threshold | $K_{watch}$ | 0.3 | — |
| Return rate minimum | — | 0.1 | — |
| Ψ dimension | — | 5 | — |

---

## §13. Formal Enforcement Axioms

### KIN-AX-0: Observable Boundedness

$$\Psi(t) \in [0,1]^5 \text{ with clip+flag OOR policy}$$

### KIN-AX-1: Return Time Well-Typed

$$\tau_{kin} \in \mathbb{Z}^+ \cup \{\text{INF\_KIN}, \text{UNIDENTIFIABLE\_KIN}\}$$

$\tau_{kin}$ is a positive integer (sample count). IEEE `Inf`/`NaN` are forbidden.

### KIN-AX-2: Conservation (Conditional)

$$F_{ext} = 0 \Rightarrow |dp/dt| < \epsilon_p$$
$$W_{nc} = 0 \Rightarrow |dE_{mech}/dt| < \epsilon_E$$

### KIN-AX-3: Stability Index Bounded

$$K_{stability} \in [0, 1]$$

---

## §14. Glossary

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
| $\Omega$ | Angular velocity (avoiding kernel $\omega$) |
| $\alpha$ | Angular acceleration |
| $L$ | Angular momentum |
| $T$ | Torque (avoiding kernel $\tau$) |
| $I$ | Moment of inertia |
| $\tau_{kin}$ | Kinematic return time |
| $\tau_R$ | Kernel return time (NOT τ_kin) |
| $\gamma(t)$ | Phase point $(x, v)$ |
| $\Gamma_{kin}$ | Phase space |
| $\Psi(t)$ | Observable vector |
| $K_{stability}$ | Stability index |
| $W$ | Window size (samples) |
| $\delta$ | Debounce lag (samples) |
| $\eta_{phase}$ | Phase space return tolerance |

---

*KIN.INTSTACK.v1 — Complete Mathematical Formalism*
