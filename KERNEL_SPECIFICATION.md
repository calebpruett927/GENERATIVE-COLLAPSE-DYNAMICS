# UMCP Kernel Specification

**Manuscript Reference**: UMCP Manuscript v1.0.0, §8  
**Status**: Canonical specification (freeze-controlled)  
**Purpose**: Formal mathematical definitions and lemmas for kernel invariants, return machinery, seam accounting, and implementation bounds

---

## Overview

This document provides the **complete formal specification** of the UMCP kernel. It defines:

1. **Kernel invariants** (F, ω, S, C, κ, IC) with precise mathematical formulas
2. **Return machinery** (Tier-1; typed boundaries for τ_R computation)
3. **Seam accounting** (Tier-1.5; weld interface with residual computation)
4. **Fundamental bounds and sensitivity facts** (34 implementation-critical lemmas)

**Critical principle**: These definitions are **algebraic + geometric + calculus objects**. They are not trend statistics or heuristics. Changes to conventions (normalization, domain, neighborhood) constitute **structural changes** and must be handled via closures and seam accounting.

---

## 1. Kernel Invariants (Tier-1 Outputs)

### Definition 7: Curvature Proxy (Default; Closure if Varied)

**Default curvature proxy**:

```
C(t) := std_pop({c_i(t)}_{i=1}^n) / 0.5
```

If a non-default curvature neighborhood or variance convention is used, it **MUST** be declared as a closure.

**Interpretation**: C(t) is a **normalized dispersion proxy** on [0, 1]. The denominator 0.5 is the maximal population standard deviation attainable by a bounded variable on [0, 1]. Thus C(t) ∈ [0, 1] under the default convention (formalized in Lemma 10 below).

**Closure requirement**: Any change to dispersion convention (sample vs population, normalization constant, neighborhood definition, masked coordinates) **changes the meaning of C** and therefore must be treated as a closure.

---

### Definition 8: Log-Integrity and Integrity Composite

```
κ(t) := Σ_{i=1}^n w_i ln c_i(t)

IC(t) := exp(κ(t))
```

**Interpretation (algebraic and geometric meaning)**:

- **κ is log-additive integrity**: It turns multiplicative aggregation into additive ledger arithmetic.
- **IC is the corresponding geometric-mean composite** (see Lemma 2).

These definitions are the **backbone of seam accounting**: seams compare differences of κ and ratios of IC across a transition.

---

## 2. Return Machinery (Tier-1; Typed Boundaries)

### Definition

Return is a **geometric event** evaluated on a discrete-time trace. It requires:

1. A declared **neighborhood notion** (norm ‖·‖ and tolerance η)
2. A declared **eligible-reference rule** (return-domain generator D_θ(t))
3. A **typing rule** for non-return or non-identifiability (∞_rec vs UNIDENTIFIABLE)

The return machinery makes "did it come back?" **computable, auditable, and comparable**.

---

### Definition 9: Return-Domain Generator and Return Candidates

A **return-domain generator** is a deterministic rule producing eligible historical indices:

```
D_θ(t) ⊆ {0, 1, ..., t-1}
```

bounded by the declared horizon H_rec (or equivalent rule).

**Return-candidate set**:

```
U_θ(t) := {u ∈ D_θ(t) : ‖Ψ_ε(t) - Ψ_ε(u)‖ ≤ η}
```

**Interpretation (what the domain generator prevents)**: D_θ(t) is **not a cosmetic parameter**. It determines which past states are eligible as references. It prevents false returns caused by:

- Stale references
- Forbidden segments
- Missingness-dominated rows
- Out-of-range conditions (as defined by policy)

The norm ‖·‖ and tolerance η define the **neighborhood**; the domain D_θ defines **admissibility**.

---

### Definition 10: Re-Entry Delay / Return Time

If U_θ(t) ≠ ∅, define:

```
τ_R(t) := min{t - u : u ∈ U_θ(t)}
```

**Typed outcomes** (mandatory):

- If U_θ(t) = ∅ within the declared horizon and domain: **τ_R(t) = ∞_rec** (INF_REC)
- If τ_R is not computable under the sampling/missingness regime: **τ_R(t) = UNIDENTIFIABLE**

**Interpretation (calculus object, not a trend statistic)**: τ_R(t) is a **discrete hitting-time object**. It is defined by a minimization over eligible indices, which makes it **inherently discontinuous** under small perturbations (e.g., a one-sample shift can change the minimum). This is a **feature, not a bug**: return time is an event-time.

**Rule**: Typed outcomes are mandatory. ∞_rec means "no return occurred under the declared horizon/domain," while UNIDENTIFIABLE means "the event is not computable under the sampling/missingness regime." These must **not** be collapsed into arbitrary finite surrogates.

---

## 3. Seam Accounting (Tier-1.5; Weld Interface)

### Definition

Seam accounting is the **continuity interface**. It compares a measured ledger change in log-integrity to a modeled budget change under frozen closures. A seam does not "prove continuity" by narrative; it produces a **residual** and applies explicit **PASS/FAIL rules** under a frozen contract and closure registry.

---

### Definition 11: Ledger Change, Budget Change, and Residual

For a candidate seam (t_0 → t_1), define:

**Ledger change**:

```
Δκ_ledger := κ(t_1) - κ(t_0) = ln(IC(t_1) / IC(t_0))

i_r := IC(t_1) / IC(t_0)
```

**Budget model**:

```
Δκ_budget := R · τ_R(t_1) - (D_ω(t_1) + D_C(t_1))
```

**Seam residual**:

```
s := Δκ_budget - Δκ_ledger
```

**Typed censoring**: If τ_R(t_1) = ∞_rec, typed censoring applies and the credit term is defined as R · τ_R := 0 (no return implies no seam credit).

**Interpretation (why this is algebra + geometry + calculus together)**:

- **Algebra**: Δκ_ledger is a difference, and i_r is the ratio form of the same claim.
- **Geometry**: τ_R(t_1) exists only because return neighborhoods and admissible domains are declared.
- **Calculus**: τ_R is a discrete event-time; residuals can change discontinuously when the minimizer changes.

Typed censoring is an **algebraic safeguard**: it prevents "infinite credit" artifacts by enforcing "no return, no credit."

---

### Definition 12: Closures Required for Weld Claims

A weld claim requires a **frozen closure registry** specifying at minimum:

1. A functional form Γ(ω; p, ε) that defines D_ω
2. A definition for D_C (default D_C = α·C) with α frozen
3. A deterministic R estimator (form + inputs + timing)
4. Any aggregation/timing rules used to evaluate these terms

**Rule**: Closures are **Tier-1.5 objects**. They may depend on Tier-1 outputs, but they must be **frozen before seam evaluation**. Changing closure form, timing, or aggregation rules is a **structural change** and must be treated as a seam event if continuity is asserted across it.

---

### Definition 13: Weld Gate (PASS/FAIL)

A seam is **PASS** (welds) only if **all** are satisfied under the frozen contract:

1. τ_R(t_1) is finite (not ∞_rec and not UNIDENTIFIABLE)
2. |s| ≤ tol (the frozen seam tolerance)
3. The identity check for i_r and Δκ_ledger passes to the declared numerical precision

Otherwise it is **FAIL** or **NOT_WELDABLE** (typed).

**Interpretation**: PASS is not "good looking plots." PASS is a **specific conjunction** of:
- (i) return evaluability
- (ii) residual tolerance
- (iii) algebraic identity consistency between ratio and log-difference

Any failure is typed so the reason for non-weldability is explicit.

---

## 4. Fundamental Bounds and Sensitivity Facts

**Purpose**: These 34 lemmas are **guardrails, stability theorems, and operational foundations**. They prevent common implementation errors, enforce well-posedness on the clipped domain, provide simple sanity checks for debugging, and formalize the connection between kernel outputs, return dynamics, seam accounting, and the return axiom. If a computed run violates these bounds, the run is almost certainly nonconformant (wrong clipping, wrong weights, wrong normalization, or a changed convention).

---

### Lemma 1: Range Bounds Under [0,1] Embedding

Assume Ψ(t) = (c_1(t), ..., c_n(t)) ∈ [0,1]^n and Ψ_ε(t) ∈ [ε, 1-ε]^n. Under the default normalization:

```
F(t) ∈ [0, 1]
ω(t) ∈ [0, 1]
C(t) ∈ [0, 1]
IC(t) ∈ [min_i c_i(t), max_i c_i(t)] ⊆ [ε, 1-ε]
```

---

### Lemma 2: IC is a Weighted Geometric Mean

```
IC(t) = exp(Σ_i w_i ln c_i(t)) = ∏_{i=1}^n c_i(t)^{w_i}
```

In particular, **IC is monotone increasing** in each coordinate c_i and is bounded by the min/max of the coordinates.

---

### Lemma 3: κ Sensitivity is ε-Controlled

For c_i ∈ [ε, 1-ε]:

```
∂κ/∂c_i = w_i / c_i
```

so |∂κ/∂c_i| ≤ w_i/ε.

**Interpretation**: ε is **not cosmetic**. It sets a hard bound on log-integrity sensitivity to small coordinate perturbations near the wall.

---

### Remark 1: What These Lemmas Are For

These facts prevent common implementation errors:

- Reporting out-of-range κ due to missing ε-clipping
- Treating IC as an arbitrary score rather than a geometric mean
- Allowing silent changes in C conventions (which must be handled as closures)

---

### Lemma 4: Weighted AM–GM Link (IC(t) ≤ F(t))

Assume c_i(t) ∈ [0,1] and weights w_i ≥ 0 with Σw_i = 1. Then:

```
IC(t) = ∏_{i=1}^n c_i(t)^{w_i} ≤ Σ_{i=1}^n w_i c_i(t) = F(t)
```

Equality holds iff c_i(t) = c_j(t) for all i,j with w_i w_j > 0.

---

### Lemma 5: Entropy Bound

Assume Ψ_ε(t) ∈ [ε, 1-ε]^n and Σw_i = 1. Then:

```
0 ≤ S(t) ≤ ln 2
```

Moreover, S(t) = ln 2 iff c_i(t) = 1/2 for all i with w_i > 0.

---

### Lemma 6: Stability of F and ω

Let Ψ(t) = (c_1, ..., c_n) and Ψ̃(t) = (c̃_1, ..., c̃_n) be in [0,1]^n. Then:

```
|F(t) - F̃(t)| ≤ Σ_{i=1}^n w_i |c_i - c̃_i|
|ω(t) - ω̃(t)| = |F(t) - F̃(t)|
```

In particular, if max_i |c_i - c̃_i| ≤ δ, then |F(t) - F̃(t)| ≤ δ.

---

### Lemma 7: κ is ε-Controlled (Finite Change Bound)

Assume c_i, c̃_i ∈ [ε, 1-ε] for all i. Then:

```
|κ(t) - κ̃(t)| = |Σ_{i=1}^n w_i ln(c̃_i / c_i)| ≤ (1/ε) Σ_{i=1}^n w_i |c_i - c̃_i|
```

---

### Lemma 8: Well-Posedness of τ_R Under Finite Horizon

Assume the return-domain generator produces a finite set D_θ(t) (e.g., via a finite horizon H_rec). If U_θ(t) ≠ ∅, then:

```
τ_R(t) = min{t - u : u ∈ U_θ(t)}
```

exists and is finite. If U_θ(t) = ∅ within the declared domain/horizon, the typed outcome τ_R(t) = ∞_rec is uniquely determined by the contract.

---

### Lemma 9: Permutation Invariance (with Consistent Weights)

Let π be a permutation of {1, ..., n} and define c̃_i(t) = c_π(i)(t) and w̃_i = w_π(i). Then the kernel outputs F(t), ω(t), S(t), κ(t), IC(t) computed from (c, w) equal those computed from (c̃, w̃).

---

### Numeric Compliance (Implementation-Critical)

All logarithms in S(t) and κ(t) **MUST** be evaluated on Ψ_ε(t). Any clipping event **MUST** be recorded under the frozen OOR policy. If a non-default convention for C(t) is used (variance convention, neighborhood, normalization), it **MUST** be declared as a closure.

---

### Lemma 10: Curvature Proxy is Bounded

Assume Ψ(t) ∈ [0,1]^n and define:

```
C(t) = std_pop({c_i(t)}_{i=1}^n) / 0.5
```

Then 0 ≤ C(t) ≤ 1. Moreover, C(t) = 0 iff c_1(t) = ... = c_n(t).

---

### Lemma 11: Log-Integrity Upper Bound via Fidelity

Assume c_i(t) ∈ [0,1] and Σw_i = 1. Then:

```
κ(t) = ln(IC(t)) ≤ ln(F(t))
```

---

### Lemma 12: Kernel Monotonicity Under Componentwise Improvement

Let Ψ(t) = (c_1, ..., c_n) and Ψ̃(t) = (c̃_1, ..., c̃_n) be in [0,1]^n such that c_i ≤ c̃_i for all i. Then:

```
F(t) ≤ F̃(t)
ω(t) ≥ ω̃(t)
κ(t) ≤ κ̃(t)
IC(t) ≤ ĨC(t)
```

---

### Lemma 13: Entropy Stability on the ε-Clipped Domain

Assume c_i, c̃_i ∈ [ε, 1-ε] for all i, and define S(t) on Ψ_ε(t). Then:

```
|S(t) - S̃(t)| ≤ ln((1-ε)/ε) Σ_{i=1}^n w_i |c_i - c̃_i|
```

---

### Lemma 14: Return Monotonicity Under Relaxed Admissibility

Fix t and a norm ‖·‖. Consider two frozen contracts K_1, K_2 that are identical except that η_1 ≤ η_2 and D_θ,1(t) ⊆ D_θ,2(t). Let τ_R^(1)(t) and τ_R^(2)(t) be the re-entry delays computed under K_1 and K_2.

If τ_R^(1)(t) is finite, then:

```
τ_R^(2)(t) ≤ τ_R^(1)(t)
```

If τ_R^(1)(t) = ∞_rec, then τ_R^(2)(t) is either ∞_rec or finite (but cannot be excluded by monotonicity).

---

### Definition 14: Contract Equivalence (Comparability of Kernel Outputs)

Two kernel evaluations are **contract-equivalent** at time t if they share the same frozen contract tuple for:

- The embedding codomain [a, b] (default [0, 1])
- Face policy
- ε (clipping tolerance)
- Dimension n
- Weights w
- Norm ‖·‖
- Return tolerance η
- Horizon H_rec
- Return-domain generator D_θ
- OOR/typed-boundary policies

**Rule**: Contract equivalence is the formal **"same experiment" condition** for kernel comparisons. If contract equivalence does not hold, differences in F, ω, S, C, κ, IC, τ_R cannot be attributed without seam accounting because the **meaning of those quantities changed**.

---

### Lemma 15: Entropy–Fidelity Envelope

Define the Bernoulli entropy function h(u) := -u ln u - (1-u) ln(1-u) for u ∈ (0,1). Assume Ψ_ε(t) ∈ [ε, 1-ε]^n and Σw_i = 1. Then:

```
S(t) = Σ_{i=1}^n w_i h(c_i(t)) ≤ h(Σ_{i=1}^n w_i c_i(t)) = h(F(t))
```

Equality holds iff c_i(t) = F(t) for all i with w_i > 0.

---

### Lemma 16: Drift-Form Envelope

Under the assumptions of Lemma 15:

```
S(t) ≤ h(1 - ω(t))
```

---

### Lemma 17: Clipping Perturbation Bound for F and ω

Let y(t) = (y_1, ..., y_n) ∈ ℝ^n be a pre-clipped row and define Ψ(t) = clip_[0,1](y(t)) componentwise. Then:

```
|F_Ψ(t) - F_y(t)| ≤ Σ_{i=1}^n w_i |clip_[0,1](y_i(t)) - y_i(t)|
|ω_Ψ(t) - ω_y(t)| = |F_Ψ(t) - F_y(t)|
```

where F_Ψ denotes F computed on Ψ(t) and F_y denotes the same affine functional applied to y(t).

---

### Lemma 18: Two-Time Stability of the Ledger Change Δκ_ledger

Let (t_0, t_1) be fixed. Assume c_i(t), c̃_i(t) ∈ [ε, 1-ε] for all i and t ∈ {t_0, t_1}. Let κ(t) and κ̃(t) be computed from c_i(t) and c̃_i(t), respectively. Then:

```
|Δκ_ledger - Δκ̃_ledger| = |(κ(t_1) - κ(t_0)) - (κ̃(t_1) - κ̃(t_0))| 
                           ≤ (1/ε) Σ_{t∈{t_0,t_1}} Σ_{i=1}^n w_i |c_i(t) - c̃_i(t)|
```

---

### Lemma 19: Seam Residual Sensitivity (Closure-Aware)

Fix a seam (t_0 → t_1) with finite τ_R(t_1). Let the budget model be:

```
Δκ_budget = R · τ_R(t_1) - (D_ω(t_1) + D_C(t_1))
s = Δκ_budget - Δκ_ledger
```

For any perturbed evaluation (denoted by tildes) computed under the same frozen closures and contract-equivalent settings, the residual difference satisfies the bound:

```
|s - s̃| ≤ |τ_R(t_1)| |R - R̃| + |R̃| |τ_R(t_1) - τ̃_R(t_1)| 
         + |D_ω(t_1) - D̃_ω(t_1)| + |D_C(t_1) - D̃_C(t_1)| 
         + |Δκ_ledger - Δκ̃_ledger|
```

**Critical observation**: Even when Ψ-perturbations are small, integer-valued changes in τ_R can induce **discontinuous changes** in s. This is an inherent feature of return-time minimization.

---

### ⚠️ Warning: Lemma 19 and Practical Reality

Lemma 19 is the formal statement of a practical reality: **seam residuals inherit discontinuities from τ_R**. This is why typed outcomes, frozen horizons/domains, and explicit windows are **non-negotiable**. Without them, seam claims cannot be audited.

---

### Lemma 20: Seam Composition Law (Multi-Seam Chains)

Let (t_0 → t_1) and (t_1 → t_2) be two consecutive seams with finite τ_R(t_1) and τ_R(t_2). Assume contract equivalence holds across the chain. Then the composed ledger change satisfies:

```
Δκ_ledger(t_0 → t_2) = Δκ_ledger(t_0 → t_1) + Δκ_ledger(t_1 → t_2)
```

and the composed residual satisfies:

```
s(t_0 → t_2) = Δκ_budget(t_0 → t_2) - Δκ_ledger(t_0 → t_2)
```

where Δκ_budget(t_0 → t_2) must be computed from the return time τ_R(t_2) measured from the original reference at t_0 or an updated reference at t_1, depending on the declared return-domain policy.

**Interpretation**: Seam residuals do **not** generally compose additively because return times reset or accumulate based on the return-domain generator. This lemma formalizes why **each seam requires independent validation** and why chained seams cannot be collapsed without explicit accounting.

---

### Lemma 21: Return-Domain Coverage Theorem

Fix a norm ‖·‖, tolerance η, and horizon H_rec. For any trace {Ψ(t)}_{t=0}^T, define the **coverage set** at time t as:

```
C_t := {u ∈ [max(0, t-H_rec), t-1] : ‖Ψ(t) - Ψ(u)‖ ≤ η}
```

If τ_R(t) = ∞_rec under the declared return-domain generator D_θ(t), then:

```
C_t ∩ D_θ(t) = ∅
```

**Corollary**: If D_θ(t) = [max(0, t-H_rec), t-1] (full-window policy) and τ_R(t) = ∞_rec, then the state Ψ(t) is **η-novel** with respect to its entire admissible history.

**Interpretation**: This lemma formalizes what ∞_rec **means geometrically**: the system has exited the η-neighborhood of all eligible historical references. It is the formal statement of "collapse is generative"—only states that return from this novelty are considered real.

---

### Lemma 22: Collapse Gate Monotonicity Under Threshold Relaxation

Fix a trace {Ψ(t)}_{t=0}^T and frozen contract K. Let G_tight and G_relaxed be two collapse gate specifications identical except:

```
ω_thresh,tight ≤ ω_thresh,relaxed
C_thresh,tight ≤ C_thresh,relaxed
S_thresh,tight ≥ S_thresh,relaxed
```

Let T_tight ⊆ {0, ..., T} be the set of timesteps flagged as collapse under G_tight, and T_relaxed under G_relaxed. Then:

```
T_tight ⊇ T_relaxed
```

**Interpretation**: Tightening collapse thresholds increases the collapse regime, making the system "harder to satisfy." This monotonicity is critical for threshold calibration and ensures that collapse detection is **order-preserving** under threshold variation.

---

### Lemma 23: Kernel Lipschitz Continuity on the ε-Clipped Domain

Assume Ψ(t), Ψ̃(t) ∈ [ε, 1-ε]^n with Σw_i = 1 and w_i ≥ 0. Define δ := max_i |c_i(t) - c̃_i(t)|. Then:

```
|F(t) - F̃(t)| ≤ δ
|ω(t) - ω̃(t)| ≤ δ
|κ(t) - κ̃(t)| ≤ (1/ε) δ
|S(t) - S̃(t)| ≤ ln((1-ε)/ε) δ
```

**Interpretation**: All kernel outputs are **Lipschitz continuous** on the ε-clipped domain with explicit Lipschitz constants. The constants depend on ε, formalizing why **ε is not cosmetic**—it controls the sensitivity envelope of all kernel outputs.

---

### Lemma 24: Return Time Stability Under Small Perturbations

Fix t and contract K with norm ‖·‖, tolerance η, and return-domain generator D_θ. Let Ψ(t) yield τ_R(t) = τ < ∞_rec with reference u* = t - τ ∈ D_θ(t). Assume:

```
‖Ψ(t) - Ψ(u*)‖ ≤ η - 2δ   (strict inequality, margin 2δ)
```

and let Ψ̃ be a perturbed trace with max_s |Ψ(s) - Ψ̃(s)| ≤ δ. Then under the same contract:

```
τ_R(t) computed from Ψ̃ satisfies: τ_R(t) ≤ τ
```

**Interpretation**: If a return occurs with **margin** (strict inequality), small perturbations preserve the return event (though possibly with a shorter return time). This lemma formalizes the **stability of return** and explains why typed outcomes near the boundary η are **inherently unstable**.

**Corollary**: States with ‖Ψ(t) - Ψ(u*)‖ ≈ η are on the **return boundary** and exhibit discontinuous behavior under perturbation. This is not a bug—it is the geometric reality of threshold-based return detection.

---

### Lemma 25: Closure Perturbation Bound

Let K and K̃ be two contracts identical except for a single closure parameter (e.g., different ε values: ε vs ε̃, or different curvature normalizations). Let Ψ(t) be a fixed trace and compute κ(t) under both K and K̃. Then:

```
|κ(t) - κ̃(t)| ≤ f_closure(|ε - ε̃|, Ψ(t))
```

where f_closure depends on the specific closure variation and is bounded by lemmas 3, 7, 13, 18.

**Interpretation**: Closure changes induce **bounded perturbations** on kernel outputs, but these perturbations **change the meaning** of the outputs. This lemma justifies why closures must be declared: even though the perturbations are bounded, **the contract changed**, and direct comparisons are invalid without seam accounting.

**Critical distinction**: Small numerical changes ≠ semantic equivalence. A 1% change in ε changes the meaning of κ, even if |Δκ| is small.

---

### Lemma 26: Entropy–Drift Coherence Bound

Assume Ψ_ε(t) ∈ [ε, 1-ε]^n and Σw_i = 1. Define the **coherence proxy**:

```
Θ(t) := 1 - ω(t) + S(t)/ln(2)
```

Then:

```
Θ(t) ∈ [0, 2]
```

Moreover, if Ψ(t) is **homogeneous** (c_i(t) = c for all i), then S(t) = h(c) and Θ(t) = 1 + h(c)/ln(2) ∈ [1, 2].

**Interpretation**: Θ(t) combines fidelity (1 - ω) and normalized entropy into a single **coherence measure**. High Θ indicates low drift and high determinacy. This lemma provides a **single-number check** for system coherence that respects the axiom: coherence requires both low drift (returnability) and low entropy (determinacy).

---

### Lemma 27: Residual Accumulation Bound Over Sequences

Let {(t_k → t_{k+1})}_{k=0}^{K-1} be a sequence of K seams with residuals s_k and finite return times τ_R(t_{k+1}). Assume contract equivalence holds across the sequence and define:

```
Σ_accum := Σ_{k=0}^{K-1} |s_k|
```

If max_k |s_k| ≤ s_max and all return times satisfy τ_R(t_k) ≤ τ_max, then:

```
Σ_accum ≤ K · s_max
```

**Interpretation**: This trivial bound becomes non-trivial when combined with **concentration inequalities** on s_k. If residuals are statistically controlled (e.g., E[s_k] ≈ 0 under correct budget models), then Σ_accum grows sublinearly with probability approaching 1, enabling **long-horizon seam validation**.

**Operational meaning**: A system that "returns" in the sense of the axiom should exhibit **bounded accumulated residuals** over sequences. Unbounded growth signals model failure or non-returning dynamics.

---

### Lemma 28: Minimal Closure Set Theorem

Let K be a frozen contract and let C = {c_1, ..., c_m} be a set of closures. Define K[C] as the contract instantiated with closures C. Two closure sets C and C' are **functionally equivalent** if they produce identical kernel outputs on all admissible traces.

**Theorem**: For any closure set C, there exists a **minimal subset** C_min ⊆ C such that K[C_min] is functionally equivalent to K[C].

**Proof sketch**: Closures either affect kernel computation or do not. Non-affecting closures can be removed without changing outputs. The minimal set is obtained by removing all non-affecting closures. ∎

**Interpretation**: This theorem formalizes **closure minimality**: there is no benefit to declaring redundant closures. The registry should contain only **functionally distinct** closures. This is the formal justification for the closure registry auditing rule in `closures/registry.yaml`.

---

### Lemma 29: Return Probability Under Bounded Random Walk

Assume the trace {Ψ(t)} evolves as a **bounded random walk** on [0,1]^n with step size σ and reflecting boundaries. Fix a norm ‖·‖, tolerance η, and horizon H_rec. Let P_return(t) be the probability that τ_R(t) < ∞_rec.

If H_rec → ∞ and η > 2σ√n, then:

```
lim_{H_rec → ∞} P_return(t) → 1
```

**Interpretation**: Under bounded stochastic dynamics, **return is almost certain** if the tolerance exceeds the typical step size and the horizon is sufficiently long. This lemma provides a **stochastic foundation** for the return axiom: systems with bounded noise **must return** with high probability.

**Corollary**: If τ_R(t) = ∞_rec persistently under reasonable η and H_rec, the system is **not following bounded dynamics**—it is undergoing drift or collapse. This is the statistical signature of **generative collapse**.

---

### Lemma 30: Weight Perturbation Stability Envelope

Let w = (w_1, ..., w_n) and w̃ = (w̃_1, ..., w̃_n) be two weight vectors with Σw_i = Σw̃_i = 1 and w_i, w̃_i ≥ 0. Define δ_w := max_i |w_i - w̃_i|. Then for any Ψ(t) ∈ [ε, 1-ε]^n:

```
|F(t) - F̃(t)| ≤ δ_w
|ω(t) - ω̃(t)| ≤ δ_w
|κ(t) - κ̃(t)| ≤ (1/ε) ln((1-ε)/ε) δ_w
|S(t) - S̃(t)| ≤ 2 ln(2) δ_w
```

**Interpretation**: Kernel outputs are **continuous in the weights** with explicit Lipschitz bounds. This lemma formalizes why **frozen weights are required** for contract equivalence: even small weight changes alter kernel outputs, and direct comparisons require seam accounting if weights vary.

---

### Lemma 31: Embedding Consistency Across Dimensions

Let Ψ^{(n)}(t) ∈ [0,1]^n and Ψ^{(m)}(t) ∈ [0,1]^m be two embeddings of the same underlying observation, with n < m (the m-dimensional embedding includes additional coordinates). Let w^{(n)} and w^{(m)} be weights satisfying:

```
w_i^{(m)} = w_i^{(n)} / Z   for i ≤ n
w_i^{(m)} = 0               for i > n
where Z = Σ_{i=1}^n w_i^{(n)}
```

Then the kernel outputs satisfy:

```
F^{(n)}(t) = F^{(m)}(t)
ω^{(n)}(t) = ω^{(m)}(t)
κ^{(n)}(t) = κ^{(m)}(t)
```

**Interpretation**: Adding zero-weight dimensions does not change kernel outputs. This lemma formalizes **dimension consistency**: the kernel respects **weight-based marginalization**. It justifies why masked coordinates (zero weight) can be included in the embedding without altering results.

---

### Lemma 32: Temporal Coarse-Graining Stability

Let {Ψ(t)}_{t=0}^T be a trace and define a coarse-grained trace {Ψ̄(t')}_{t'=0}^{T'} by:

```
Ψ̄(t') = (1/M) Σ_{k=0}^{M-1} Ψ(Mt' + k)
```

where M is the coarsening factor and T' = ⌊T/M⌋. Assume Ψ(t), Ψ̄(t') ∈ [ε, 1-ε]^n for all t, t'. Then for kernel outputs F, ω, κ computed on the coarse-grained trace:

```
|F̄(t') - (1/M) Σ_{k=0}^{M-1} F(Mt' + k)| ≤ ε_coarse(M)
```

where ε_coarse(M) → 0 as M → 1 and depends on the smoothness of the original trace.

**Interpretation**: Coarse-graining **does not preserve kernel outputs exactly** (F is nonlinear), but the perturbations are bounded. This lemma justifies **multi-scale analysis**: kernel outputs at different timescales can be compared via seam accounting, treating the coarsening as a closure.

---

### Lemma 33: Sufficient Condition for Finite-Time Return

Fix a trace {Ψ(t)}_{t=0}^T, norm ‖·‖, tolerance η, and return-domain generator D_θ. Assume there exists a time t and a reference u ∈ D_θ(t) such that:

```
‖Ψ(t) - Ψ(u)‖ < η
```

Then τ_R(t) < ∞_rec and:

```
τ_R(t) ≤ t - min{u ∈ D_θ(t) : ‖Ψ(t) - Ψ(u)‖ < η}
```

**Interpretation**: This is the **direct verification** of return. If a return candidate exists in the admissible domain with strict inequality, return is certified. This lemma is the operational check performed by `tau_R_compute.py` and is the formal definition of "what returns is real."

---

### Lemma 34: Drift Threshold Calibration via AM–GM Gap

Define the **AM–GM gap** as:

```
Δ_gap(t) := F(t) - IC(t) ≥ 0
```

By Lemma 4, Δ_gap(t) = 0 iff all c_i(t) are equal (homogeneity). Define ω_crit := 1 - IC(t) (drift measured from integrity). Then:

```
ω(t) = 1 - F(t) ≤ 1 - IC(t) + Δ_gap(t)
```

**Interpretation**: The AM–GM gap Δ_gap(t) quantifies **heterogeneity** in the coordinate distribution. Large gaps indicate non-uniform integrity and provide a **calibration signal** for drift thresholds: if Δ_gap is large, ω-based collapse gates may trigger earlier than IC-based gates.

**Operational meaning**: This lemma links **geometric heterogeneity** (AM–GM gap) to **collapse detection** (drift threshold), providing a principled method for threshold calibration based on coordinate dispersion.

---

### ⚠️ Synthesis: Lemmas 20–34 and the Return Axiom

Lemmas 20–34 extend the formal foundation across five domains:

1. **Seam Accounting** (L20, L27): Composition laws and accumulation bounds
2. **Return Machinery** (L21, L24, L29, L33): Coverage, stability, probability, sufficiency
3. **Collapse Detection** (L22, L26, L34): Monotonicity, coherence, calibration
4. **Closure Interactions** (L25, L28, L32): Perturbation bounds, minimality, coarse-graining
5. **Kernel Stability** (L23, L30, L31): Lipschitz continuity, weight perturbations, dimension consistency

**Connection to Axiom-0**: Each lemma reinforces the operational meaning of "only what returns is real":

- **L21**: Formalizes what ∞_rec means (η-novelty, collapse)
- **L24**: Return stability under perturbation (returnability is robust)
- **L29**: Stochastic guarantee of return under bounded dynamics
- **L33**: Sufficient condition for return certification

Together, Lemmas 1–34 provide a **complete formal foundation** for kernel validation, seam accounting, and return-based continuity claims.

---

## 5. Relationship to Other Protocol Documents

This specification document is referenced by and depends on:

- **[TIER_SYSTEM.md](TIER_SYSTEM.md)**: Tier-1 (kernel invariants), Tier-1.5 (seam accounting), Tier-2 (overlays)
- **[AXIOM.md](AXIOM.md)**: Operational definitions for Return, Drift, Integrity, Entropy, Collapse
- **[SYMBOL_INDEX.md](SYMBOL_INDEX.md)**: One-page symbol table preventing namespace collisions
- **[FACE_POLICY.md](FACE_POLICY.md)**: Boundary governance and admissible clipping rules
- **[UHMP.md](UHMP.md)**: Identity governance for manifest roots and ledger registration
- **[PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md)**: Publication row format and weld accounting

---

## 6. Implementation Notes

### Critical Compliance Requirements

1. **All logarithms in S(t) and κ(t) MUST be evaluated on Ψ_ε(t)** (the ε-clipped domain)
2. **Any clipping event MUST be recorded** under the frozen OOR policy
3. **Non-default C(t) conventions MUST be declared as closures**
4. **Typed outcomes for τ_R are mandatory** (∞_rec vs UNIDENTIFIABLE)
5. **Contract equivalence is required** for direct comparison of kernel outputs

### Common Pitfalls

- **Missing ε-clipping**: Results in out-of-range κ and undefined logarithms
- **Treating IC as an arbitrary score**: IC is a weighted geometric mean with specific monotonicity properties
- **Silent changes to C conventions**: Any change to variance/normalization/neighborhood requires closure declaration
- **Collapsing typed outcomes**: ∞_rec and UNIDENTIFIABLE must not be replaced with arbitrary finite values
- **Comparing non-equivalent contracts**: Differences cannot be attributed without seam accounting

### Debugging with Lemmas

If a computed run violates the bounds in Lemmas 1-34, the implementation is almost certainly nonconformant. Check:

- Clipping applied before logarithms (Lemma 1, Lemma 3)
- Weight normalization (Σw_i = 1)
- C(t) normalization (denominator 0.5 for default convention)
- Return domain generator producing finite sets (Lemma 8)
- Typed outcome handling for edge cases
- Seam composition and residual accumulation (Lemmas 20, 27)
- Return stability near boundaries (Lemma 24)
- Closure minimality (Lemma 28)

---

## 7. Version Control and Freeze Requirements

**Status**: This specification is **freeze-controlled** and versioned with the UMCP protocol.

- Changes to definitions or lemmas require a **new protocol version**
- Changes to default conventions (ε, normalization constants, domain generators) require **closure declarations**
- Implementation changes that preserve mathematical definitions do not require versioning (but must pass conformance tests)

**Current Version**: UMCP v1.4.0 (as of UMCP Manuscript v1.0.0, §8)

---

## 8. References

- **UMCP Manuscript v1.0.0**: §8 (Formal Definitions and Lemmas)
- **TIER_SYSTEM.md**: Tier separation and freeze gates
- **AXIOM.md**: Operational term definitions
- **SYMBOL_INDEX.md**: Canonical symbol table
- **PUBLICATION_INFRASTRUCTURE.md**: Weld accounting and publication rows

---

**Document Status**: Complete formal specification from manuscript §8  
**Last Updated**: 2026-01-21  
**Checksum**: (Recorded in integrity/sha256.txt)
