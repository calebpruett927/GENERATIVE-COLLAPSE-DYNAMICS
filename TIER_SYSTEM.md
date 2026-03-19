# UMCP Tier System

**Version**: 2.2.0
**Status**: Protocol Foundation
**Source**: UMCP Manuscript v1.0.0 §3 (revised per cross-domain validation, 146 experiments)
**Last Updated**: 2026-03-11

---

## Overview

The **UMCP tier system** has exactly three tiers. No half-tiers. No confusion.

| Tier | Name | Role |
|------|------|------|
| **1** | **The Kernel** | The mathematical function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC), its six definitions, their provable identities (F + ω = 1, IC ≤ F, IC = exp(κ), S ≈ f(F,C)), and the 46 lemmas, 38 structural identities, and 5 structural constants that follow. 3 effective degrees of freedom (F, κ, C). Domain-independent. Immutable. |
| **0** | **Protocol** | The operational machinery that implements and interprets the Tier-1 kernel: embedding raw data into [0,1]ⁿ, computing the kernel function, applying regime gates, running seam calculus, enforcing contracts/schemas/SHA-256 integrity, and issuing three-valued verdicts. |
| **2** | **Expansion Space** | Domain closures that choose which real-world quantities become the trace vector c ∈ [0,1]ⁿ and weights w ∈ Δⁿ. Channel selection, entity catalogs, normalization, and domain-specific theorems. Validated through Tier-0 against Tier-1. |

**An important distinction**: The kernel has three aspects that must not be conflated:

1. **The function** (Tier-1) — the six formulas (F = Σ wᵢcᵢ, ω = 1−F, S, C, κ, IC = exp(κ)) and everything provable about them (identities, lemmas, structural constants like c* = 0.7822). This is the mathematical object. It is domain-independent and immutable.
2. **The implementation** (Tier-0) — the code in `kernel_optimized.py` that evaluates the Tier-1 formulas, plus the embedding that maps raw data into the kernel's input space, the regime gates that interpret its output, and the seam calculus that tracks continuity. This is the operational machinery.
3. **The inputs** (Tier-2) — the choice of which real-world quantities become channels, how they are normalized, and what entities are measured. This is the domain closure.

The identities (F + ω = 1, IC ≤ F) are not separate objects that sit beside the kernel at Tier-1. They are **theorems about the kernel function** — properties that follow from the six definitions. They serve as test oracles: if the Tier-0 code ever returns F + ω ≠ 1, the code is wrong, not the identity.

---

## The Stack

```
Tier-1   THE KERNEL (mathematical object)
         ┌─────────────────────────────────────────────────┐
         │ 4 PRIMITIVES (independent computations):         │
         │   F = Σ wᵢcᵢ            κ = Σ wᵢ ln cᵢ         │
         │   S = Σ wᵢ h(cᵢ)        C = σ(c)/0.5           │
         │                                                  │
         │ 2 DERIVED (determined by the primitives):        │
         │   ω = 1-F               IC = exp(κ)             │
         │                                                  │
         │ 3 IDENTITIES (theorems about the definitions):   │
         │   F + ω = 1    IC ≤ F    IC = exp(κ)            │
         │                                                  │
         │ STRUCTURAL CONSTANTS (discovered, not chosen):   │
         │   c* = 0.7822   c_trap = 0.3178   ε = 10⁻⁸     │
         │                                                  │
         │ EFFECTIVE RANK (proven for all n = 4..64):      │
         │   3 degrees of freedom: F, κ, C                  │
         │   S ≈ f(F, C) [statistical, tightens with n]     │
         │   corr(C, S) → −1 as n → ∞ (CLT)                │
         │                                                  │
         │ 46 LEMMAS · 44 IDENTITIES · 8 EQUATIONS         │
         │ (all properties of this one function)            │
         └─────────────────────────────────────────────────┘
         Immutable. Domain-independent. Mathematically complete.
         ──────────────────────────────────────────────────────

Tier-0   PROTOCOL (operational machinery)
         ┌─────────────────────────────────────────────────┐
         │ INPUT:  Embedding x(t) → c ∈ [0,1]ⁿ           │
         │         Weight assignment w ∈ Δⁿ                │
         │         ε-clamp, face policy, normalization     │
         │                                                  │
         │ COMPUTE: Run the Tier-1 kernel on (c, w)        │
         │          (kernel_optimized.py implements Tier-1  │
         │           formulas — the code is Tier-0,        │
         │           what it computes is Tier-1)            │
         │                                                  │
         │ INTERPRET: Regime gates on (ω, F, S, C)         │
         │            Seam calculus: Γ, D_C, Δκ, residual  │
         │            Verdict: CONFORMANT / NON / NON_EVAL │
         │            SHA-256, schemas, contracts           │
         └─────────────────────────────────────────────────┘
         Frozen per run. Makes Tier-1 computable and auditable.
         ──────────────────────────────────────────────────────

Tier-2   EXPANSION SPACE (domain closures)
         ┌─────────────────────────────────────────────────┐
         │ CHANNEL SELECTION: Which quantities become c     │
         │ ENTITY CATALOGS: Which objects are measured      │
         │ NORMALIZATION: How raw values map to [0,1]       │
         │ DOMAIN THEOREMS: Interpreting kernel outputs     │
         │   in domain-specific language                    │
         └─────────────────────────────────────────────────┘
         Freely extensible. Validated through Tier-0 against Tier-1.
         ✗ NO FEEDBACK to Tier-1 or Tier-0 within a frozen run
```

**One-way dependency**: Tier-2 reads Tier-1 outputs through Tier-0 but cannot alter them. No back-edges within a frozen run.

**Return-based canonization across runs**: Tier-2 results can be promoted to Tier-1 canon ONLY through formal validation — threshold check → seam weld → new contract version. "The cycle must return or it's not real."

---

## Tier-1: The Kernel

**What it is**: The kernel is the mathematical function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC). It takes a trace vector c ∈ [0,1]ⁿ and a weight vector w ∈ Δⁿ (the probability simplex) and produces six invariants. The six formulas define the function. The identities are theorems about it. The 46 lemmas, 38 structural identities, 8 equations, and 5 structural constants are all properties of this one mathematical object.

Tier-1 is the kernel function *and everything provable about it*. The identities are not separate objects floating above the kernel — they are consequences of the definitions. F + ω = 1 is a theorem about F(c,w) = Σ wᵢcᵢ and ω = 1 − F. IC ≤ F is a theorem about the relationship between the weighted geometric and arithmetic means. They cannot exist without the definitions that produce them.

### The Six Definitions (4 Primitive, 2 Derived)

The kernel outputs six values, but only **four are independent computations** (primitives). The other two are algebraically determined by the primitives. All six remain Tier-1 — the distinction is internal structure, not tier assignment.

| Symbol | Formula | What It Computes | Status |
|--------|---------|------------------|--------|
| **F** | F = Σ wᵢcᵢ | Weighted arithmetic mean of the trace (fidelity) | **Primitive** |
| **κ** | κ = Σ wᵢ ln(cᵢ,ε) | Weighted log-sum, ε-clamped (log-integrity) | **Primitive** |
| **S** | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] | Weighted Bernoulli field entropy | **Primitive** |
| **C** | C = stddev(cᵢ) / 0.5 | Normalized standard deviation (curvature proxy) | **Primitive** |
| **ω** | ω = 1 − F | Complement of fidelity (drift) | Derived from F |
| **IC** | IC = exp(κ) | Weighted geometric mean (integrity composite) | Derived from κ |

**Why ω and IC remain Tier-1** (not diagnostics): They are properties of the kernel function and appear in the immutable identities (F+ω=1, IC≤F, IC=exp(κ)). Moving them out of Tier-1 would break the identity structure. What a "diagnostic" is — heterogeneity regime labels, regime classification, seam PASS/FAIL — are Tier-0 *interpretations* of Tier-1 outputs. IC and ω are Tier-1 *outputs*, not interpretations.

**Why the primitive/derived distinction matters**: The 4 primitives (F, κ, S, C) are the four **independently computed** quantities — each is evaluated from (c, w) by its own formula. But independently computed ≠ informationally independent. The kernel's effective dimensionality is **3**, not 4, because S is nearly determined by F and C (a statistical constraint that tightens as n → ∞). The 4th formula computes a real quantity; it just doesn't add a degree of freedom. The heterogeneity gap Δ = F − IC requires BOTH F (Primitive 1) and IC (from Primitive 2) — their difference carries the heterogeneity signal that neither alone can measure.

**Why S is computed but not free**: Entropy S = −Σ wᵢ h(cᵢ) is a concave function of the channels. By Jensen's inequality, for fixed mean F and standard deviation C, the entropy is bounded. As n grows, the CLT forces S toward its conditional expectation given (F, C). The correlation corr(C, S) → −1 as n → ∞. S is a primitive *computation* (it requires its own formula) but not a primitive *degree of freedom* (its value is asymptotically determined by F and C).

### Identities (theorems about the definitions — 0 violations across 10,162 tests in 18 domains)

| Identity | Type | Why It Holds | Structural Meaning |
|----------|------|-------------|-------------------|
| **F + ω = 1** | Algebraic | By definition: ω := 1 − F | Fidelity and drift are exhaustive complements. No third bucket. |
| **IC ≤ F** | Algebraic | Geometric mean ≤ arithmetic mean (the integrity bound) | Coherence cannot exceed fidelity. Heterogeneity always costs. |
| **IC = exp(κ)** | Algebraic | By definition: κ := Σ wᵢ ln cᵢ, IC := exp(κ) | Log-integrity and multiplicative coherence are the same invariant in different coordinates. |
| **S ≈ f(F, C)** | Statistical | Jensen + concavity + CLT | Entropy is asymptotically determined by fidelity and curvature. |

The first three are exact (hold to machine precision, always). The fourth is statistical (strengthens with n, exact only in the n → ∞ limit). Together they reduce 6 kernel outputs to **3 effective degrees of freedom**.

### The Rank-3 Theorem (effective dimensionality of the kernel)

The kernel K maps n-dimensional trace vectors to 6 outputs. PCA on 10,000 random traces reveals that the first 3 principal components capture >99% of variance for ALL n:

| n (input channels) | PC1 | PC2 | PC3 | Σ(1–3) | Rank at 99% |
|:------------------:|:---:|:---:|:---:|:------:|:-----------:|
| 4 | 65% | 28% | 6% | 99.2% | **3** |
| 8 | 65% | 31% | 3% | 99.2% | **3** |
| 16 | 66% | 32% | 2% | 99.2% | **3** |
| 32 | 66% | 32% | 1% | 99.3% | **3** |
| 64 | 65% | 33% | 1% | 99.6% | **3** |

The rank is **invariant to input dimensionality**. Adding more channels does not add degrees of freedom — it dilutes them (std ~ 1/√n by CLT). The kernel compresses n dimensions to 3.

The three independent quantities and what they measure:

| DOF | Symbol | What It Measures |
|:---:|--------|------------------|
| 1 | **F** (fidelity) | What persists through collapse |
| 2 | **κ** (log-integrity) | How much coherence is lost (logarithmic sensitivity) |
| 3 | **C** (curvature) | How unevenly it is lost (channel heterogeneity) |

Why these three and not others:
- F and κ are not interchangeable: F is the arithmetic mean (what survives *on average*), κ is the log-sum (sensitive to *any channel near zero*). A system with F = 0.8 can have κ = −0.3 (uniform channels) or κ = −5.0 (one dead channel). The heterogeneity gap Δ = F − IC = F − exp(κ) measures exactly this difference.
- C and S share information because both depend on the same central moments. But C carries the heterogeneity signal that modulates the budget cost (z = Γ(ω) + αC), while S is reconstructable from F and C.

This theorem is a provable property of the kernel function (Identity #29). Verification: `python scripts/unified_geometry.py` §1.

### Rank Sub-Classification (Degenerate Cases)

The maximum rank is 3 (the general case). Special trace structures yield degenerate ranks where fewer than 3 DOF are independent. The rank is a **property of the trace vector**, not a parameter — it is measured, not chosen (*gradus non eligitur; mensuratur*).

| Rank | DOF | Condition | IC = F? | C determined by (F, κ)? |
|:----:|:---:|-----------|:-------:|:-----------------------:|
| **1** | 1 | All channels equal (cᵢ = c₀) | Yes | Yes (C = 0) |
| **2** | 2 | Effective 2-channel structure | No | Yes |
| **3** | 3 | General heterogeneous (n ≥ 3, non-trivial) | No | No |

- **Rank-1** (homogeneous): F alone determines everything. IC = F, C = 0, Δ = 0. The *baseline* — thermodynamic equilibrium of the kernel.
- **Rank-2** (binary differentiation): F and κ suffice. C is algebraically determined from them. The 2-channel solvability condition c₁,₂ = F ± √(F² − IC²) applies exactly.
- **Rank-3** (structured heterogeneity): F, κ, and C are mutually independent. The *distribution* of channel values matters, not just their mean and product. This is where confinement cliffs, scale inversions, and cross-domain bridges live.

Rank-1 ⊂ Rank-2 ⊂ Rank-3 (strict hierarchy under constraint count). Almost all real-world systems with n ≥ 3 channels are rank-3. Full definitions with conditions, examples, and transition rules: see [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) §4c.

### The Three Agents (structural reading of the three DOF)

The three degrees of freedom map to three epistemic roles — three "agents" that partition the full space of what the kernel measures:

| Agent | Symbol | Role | Structural Identity |
|-------|--------|------|---------------------|
| **Agent 1 — Measuring** | ω = 1 − F | What is being measured RIGHT NOW. The live signal, the act of observation. | Agent 1 + Agent 2 = 1 (duality) |
| **Agent 2 — Archive** | F | What has been measured BEFORE and persists. The accumulated record. | Complementary to Agent 1 |
| **Agent 3 — Unknown** | Γ(ω) = ω³/(1−ω+ε) | What has NEVER been measured. The cost of engaging new territory. | Nonlinear function of Agent 1, pole at ω = 1 |

C modulates Agent 3 (heterogeneity adds cost: z = Γ(ω) + αC). The regime classification is agent dominance:
- **Stable**: Agent 2 dominates (F > 0.90, ω < 0.038) — the system is mostly archive.
- **Watch**: Agents balanced — measurement is active but not overwhelming.
- **Collapse**: Agent 3 dominates (ω ≥ 0.30, Γ grows nonlinearly) — the cost of unknowns overwhelms.

The three agents are not imposed from outside. They are the structural reading of the rank-3 decomposition.

### Reserved Symbols (Immutable Meanings)

| Symbol | Name | Formula | Range | Structural Role |
|--------|------|---------|-------|-----------------|
| **F** | Fidelity | F = Σ wᵢcᵢ | [0,1] | How much of the signal survives collapse |
| **ω** | Drift | ω = 1 − F | [0,1] | How much is lost to collapse |
| **S** | Entropy | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] | ≥0 | Bernoulli field entropy (Shannon entropy is the degenerate limit) |
| **C** | Curvature | C = stddev(cᵢ)/0.5 | [0,1] | Coupling to uncontrolled external degrees of freedom |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | ≤0 | Logarithmic fidelity (sensitivity-aware) |
| **IC** | Integrity composite | IC = exp(κ) | (0,1] | Multiplicative coherence |
| **τ_R** | Return time | Re-entry delay to D_θ | ℕ∪{∞_rec} | How long until the system returns to its domain |
| **regime** | Regime label | Gates on (ω,F,S,C) | {Stable, Watch, Collapse} | Which structural phase the system occupies |

### Structural Constants (derived from the kernel, not chosen)

| Constant | Value | How It Arises |
|----------|-------|---------------|
| **c*** | 0.7822 | Unique fixed point of σ(1/c) — maximizes S + κ per channel |
| **c_trap** | 0.3178 | Cardano root of x³ + x − 1 = 0 — Γ(ω) = 1 threshold |
| **ε** | 10⁻⁸ | Guard band where the pole at ω = 1 does not affect measurement to machine precision |
| **p** | 3 | Unique integer yielding closed-form ω_trap |
| **tol_seam** | 0.005 | Width where IC ≤ F holds at 100% across all domains |

### What Tier-1 Is and Is NOT

- **IS the kernel function**: Four primitive formulas (F, κ, S, C) and two derived values (ω = 1−F, IC = exp(κ)) define K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC). All six outputs, and all provable properties, are Tier-1.
- **IS the identities**: F + ω = 1, IC ≤ F, IC = exp(κ) are theorems *about* the kernel function. They are part of Tier-1 because they cannot exist without the definitions that produce them.
- **IS the lemmas and structural constants**: The 46 lemmas, 44 identities, c* = 0.7822, c_trap = 0.3178 — all properties of the same function.
- **IS internally structured**: The 4 primitives are independently *computed* (each has its own formula); 2 values are algebraically derived (ω, IC); and 1 statistical constraint (S ≈ f(F,C)) reduces the effective degrees of freedom to 3. The rank-3 structure is not a tier distinction — it is the internal dependency structure of the kernel function itself, proven invariant to input dimensionality (n = 4..64).
- **NOT diagnostics**: ω and IC are derived from primitives but are NOT diagnostics. Diagnostics (heterogeneity regime labels, Stable/Watch/Collapse classification, seam PASS/FAIL) are Tier-0 *interpretations*. ω and IC are Tier-1 *outputs* that appear in the immutable identities.
- **NOT computation**: The *code* that evaluates the formulas is Tier-0. The *formulas themselves* are Tier-1. `kernel_optimized.py` is a Tier-0 implementation of the Tier-1 function.
- **NOT a model of the world**: Tier-1 makes no domain claims. It says "given trace c and weights w, F = Σ wᵢcᵢ." It does not say what those channels *mean* for a nuclide versus a star. Meaning-mapping is Tier-2.
- **NOT "constants we chose"**: The frozen parameters are **consistent** across the seam — the same rules on both sides of collapse-return. They are discovered by the mathematics, not prescribed by convention.

---

## Tier-0: Protocol

**What it is**: Everything that makes the Tier-1 kernel function computable, testable, and auditable. Tier-0 includes three distinct roles:

1. **Implementation of the kernel**: The code in `kernel_optimized.py` that evaluates the 4 Tier-1 primitives (F, κ, S, C) and the 2 derived values (ω, IC). The code is Tier-0 (protocol); what it computes is Tier-1 (the function). If the code ever disagrees with the identities, the code is wrong — the identities are the test oracle.
2. **Interpretation of the output**: Everything that acts on the kernel's output — regime gates (translating continuous Tier-1 invariants into discrete labels), seam calculus (tracking continuity), contracts, schemas, SHA-256 integrity, and three-valued verdicts. **These interpretations are Tier-0 objects, not Tier-1.** Regime labels (Stable/Watch/Collapse), seam verdicts (PASS/FAIL), and heterogeneity regime names are protocol-level classifications that consume Tier-1 outputs. They are not part of the kernel function.
3. **Preparation of the input**: Embedding raw measurements x(t) into the trace vector Ψ(t) ∈ [0,1]ⁿ, assigning weights, and applying clipping/normalization policies.

**The key distinction**: ω and IC are Tier-1 *outputs* (they appear in the immutable identities F+ω=1 and IC≤F). Regime labels and seam verdicts are Tier-0 *interpretations* (they classify those outputs into categories). The outputs exist without the interpretations; the interpretations cannot exist without the outputs.

### Protocol Functions

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| **Regime classification** | (ω, F, S, C) | {Stable, Watch, Collapse} | Four-gate filter on invariant space |
| **Three-valued status** | Validation result | {CONFORMANT, NONCONFORMANT, NON_EVALUABLE} | Verdict |
| **τ_R classification** | Return time | {permanent, finite, retrocausal (forbidden)} | Return type |
| **Schema enforcement** | Artifact structure | {valid, invalid} | Structural conformance |
| **SHA256 integrity** | File contents | {match, mismatch} | Artifact identity |
| **Diagnostics** | Control experiments | {confirmed, failed} | Validates the protocol itself |
| **Seam calculus** | Transition accounting | {PASS, FAIL} | Continuity verification |

### Regime Gates

The four-gate criterion translates continuous Tier-1 invariants into discrete regime labels:

```
Stable:   ω < 0.038  AND  F > 0.90  AND  S < 0.15  AND  C < 0.14
Collapse: ω ≥ 0.30
Watch:    everything else
```

Conjunctive for Stable (all four must pass) because stability requires *all* invariants to be clean simultaneously.

### Diagnostics (Protocol Self-Validation)

Diagnostics are control experiments that confirm the protocol works before any domain expansion result is trusted:

| Diagnostic | N | % Stable | What it confirms |
|------------|---|----------|------------------|
| **Kinematics** (KIN.INTSTACK.v1) | 5 | 100% | When C = 0 and ω is bounded, gates correctly classify all as Stable |
| **PHYS-04** (GCD.INTSTACK.v1) | 1 | 100% | Perfect fixed point — every invariant at ideal value yields expected classification |

If kinematics didn't return 100% Stable, or PHYS-04 didn't hit all zeros, the protocol is broken and no Tier-2 result can be trusted.

### Seam Calculus (Continuity Accounting)

The seam calculus makes continuity claims **falsifiable** by reconciling ledger versus budget under frozen closures:

```
Δκ_ledger = κ₁ − κ₀                    (measured change)
Δκ_budget = R·τ_R − (D_ω + D_C)        (modeled change)
s = Δκ_budget − Δκ_ledger              (residual)

PASS if |e^(Δκ) − IC₁/IC₀| < 10⁻⁹  AND  |s| ≤ tol_seam  AND  τ_R finite
```

If τ_R = ∞_rec, there is no return credit; the seam **cannot pass**. Closures must be frozen before seam evaluation — no tuning to force PASS.

### Allowed Scope

- **Observables**: Raw measurements x(t) with units, sampling, provenance
- **Embedding N_K**: Map x(t) → Ψ(t) ∈ [0,1]ⁿ with explicit clipping/normalization
- **OOR Policy**: Out-of-range handling (e.g., `pre_clip` with ε = 1e-8)
- **Face Policy**: Boundary behavior (see [FACE_POLICY.md](FACE_POLICY.md))
- **Weights**: Declared wᵢ ≥ 0, Σwᵢ = 1
- **Return Settings**: Domain D_θ(t), tolerance η, horizon H_rec
- **Validation**: Contract checking, schema enforcement, integrity verification
- **Deterministic computation**: Compute the Tier-1 invariants from frozen trace and weights

### Prohibitions

| Prohibition | Violation | Consequence |
|-------------|-----------|-------------|
| **No Tier-1 redefinition** | Tier-0 introduces "ω" for angular frequency | Automatic nonconformance (symbol capture) |
| **No adaptive preprocessing** | Interface changes after seeing outputs | New run required (re-freeze, re-hash, re-issue) |
| **No hidden transforms** | Undocumented smoothing or filtering | Not auditable, nonconformant |
| **No post hoc closure tuning** | Changing closures to force seam PASS | Categorical nonconformance |

### Required Artifacts

```yaml
Tier-0 (frozen before compute):
  - contract.yaml         # Active contract ID, frozen params, tolerances
  - observables.yaml      # Units, sampling, pipeline, raw hashes
  - embedding.yaml        # N_K: x(t) → Ψ(t), OOR policy, face, ε
  - trace.csv             # Ψ(t) ∈ [0,1]ⁿ + flags
  - weights.yaml          # w_i, sum=1 validation
  - invariants.csv        # Computed Tier-1 invariants (deterministic output)
  - closures.yaml         # Γ form, R estimator, D_C definition, tolerances (if weld)
  - welds.csv             # Ledger, budget, residual, PASS/FAIL (if continuity claimed)
```

---

## Tier-2: Expansion Space

**What it is**: The space where domain-specific **input choices** are made. The 4 Tier-1 primitive equations take c ∈ [0,1]ⁿ and w ∈ Δⁿ — they do not know or care what those channels represent. Tier-2 is where the decision is made to encode a proton as c = [0.49, 0.75, 0.67, 0.0, 0.75, 0.0, 1.0, 0.33] with channels [mass_log, spin, charge, color, T3, L, B, generation]. That encoding — which quantities become channels, how they are normalized, which entities are measured — is the domain closure.

Tier-2 answers: **What real-world quantities become c and w for the 4 primitives, and what do the 6 kernel outputs mean in domain-specific terms?**

The 42 proven theorems (SM T1–T10, MG T1–T10, KS T1–T7, CC T1–T7, PM T1–T8) are Tier-2 because they depend on domain-specific channel choices. "IC drops 98% at the quark-hadron boundary" is true given the 8-channel SM encoding — but the theorem requires that specific channel selection to state. The primitive computation it relies on (F, κ, S, C) and the derived values (ω, IC) are Tier-1; the channel choice that makes it about confinement is Tier-2.

### Current Domain Expansions

| Domain | Contract | Selector |
|--------|----------|----------|
| Generative Collapse Dynamics | GCD.INTSTACK.v1 | Base kernel computation |
| Recursive Collapse Field Theory | RCFT.INTSTACK.v1 | Multi-scale recursive return |
| Kinematics | KIN.INTSTACK.v1 | Phase-space trajectory analysis |
| Financial markets | FINANCE.INTSTACK.v1 | Bounded drift despite high coupling |
| Security / adversarial | SECURITY.INTSTACK.v1 | Engineered decoupling vs adversarial ω |
| Stellar & cosmological | ASTRO.INTSTACK.v1 | Gravitational equilibrium |
| Nuclear physics | NUC.INTSTACK.v1 | Proximity to iron peak |
| Quantum mechanics | QM.INTSTACK.v1 | Lightest in charge/spin class |
| Cosmology / dark energy | WEYL.INTSTACK.v1 | Irreducible cosmological drift |
| Atomic physics | ATOM.INTSTACK.v1 | 118 elements, periodic kernel |
| Materials science | MATL.INTSTACK.v1 | Element database (118 × 18 fields) |
| Standard Model | SM.INTSTACK.v1 | 31 particles, 10 proven theorems |

### The Validation Path

Every Tier-2 experiment follows this path:

```
raw data → domain closure → (ω, F, S, C, IC, τ_R) → Tier-0 gates → regime verdict
                                     ↑
                              Tier-1 identities
                              must hold
```

### Allowed Scope

1. **Domain closures**: Functions that map domain observables into Tier-1 invariants
2. **Domain models**: Deterministic, stochastic, hybrid; ML models; mechanistic models
3. **Domain diagnostics**: Sensitivity analyses, residual checks (labeled `diagnostic`, never used as gates)
4. **Controllers/heuristics**: Parameter suggestions, experimental design guidance (any proposed Tier-0 change = new run)
5. **Narrative interpretation**: Ontology, philosophy (explicitly labeled as interpretive, never structural)

### Prohibitions

| Prohibition | Violation Example |
|-------------|-------------------|
| **No symbol capture** | Tier-2 redefines ω, F, S, C, τ_R, κ, IC or regime gates |
| **No diagnostic-as-gate** | Tier-2 diagnostics cannot change regime labels or seam PASS/FAIL |
| **No rescue operations** | Tier-2 cannot override τ_R = ∞_rec or seam FAIL |
| **No structural override** | Domain closures cannot modify the Tier-1 identities |

### Required Artifacts

```yaml
Tier-2 (domain expansion; validated through Tier-0 against Tier-1):
  - closures/domain_name/   # Domain-specific closure functions
  - casepacks/domain_name/  # Casepack with raw data, manifest, expected invariants
  - canon/domain_anchors.yaml  # Domain canon anchors
  - contracts/DOMAIN.INTSTACK.v1.yaml  # Domain contract
```

---

## Cross-Tier Rules

### Freeze Gate

**Rule**: `/freeze` is a **hard gate**. Tier-0 declarations (observables, embedding, flags, weights, return settings) and closure registry (if weld is used) must be frozen **before** invariant computation and seam evaluation. If `/freeze` is missing or incomplete, the run is **nonconformant**.

**Why frozen**: The seam demands it. To verify that what returned through collapse is consistent with what went in, the rules of measurement must be identical on both sides of the seam. If ε changes between the outbound run and the return, closures cannot be compared. If tol_seam shifts, "CONFORMANT" on one side means something different than "CONFORMANT" on the other. Frozen does not mean "we decided this forever" — it means "this does not change within a single collapse-return cycle." The word is not "constant" but **consistent**.

### One-Way Dependency Within Runs

```
WITHIN A FROZEN RUN:
Tier-1 (immutable structure) → Tier-0 (protocol) → Tier-2 (expansion)
                                                      ↓
                                                 NO FEEDBACK ✗
```

**Critical**: No back-edges inside a run. Tier-2 may read Tier-1 outputs through Tier-0 but cannot alter them.

### Return-Based Canonization Across Runs

```
ACROSS RUNS:
Run N: Tier-2 domain result
         ↓ threshold validation
         ↓ seam weld computation (Tier-0)
         ↓ IF weld passes ("returns")
Run N+1: New Tier-1 canon (promoted result)
         ✓ FORMAL PROMOTION via contract versioning
         ✗ IF weld fails → remains Tier-2 ("didn't return = not canonical")
```

**"The cycle must return or it's not real."**

### Audit Traceability

**Rule**: Any change between runs must be **attributable to a tier**. If you cannot localize the change, you do not have an audit trail and the comparison is **invalid**.

---

## Nonconformance Criteria

| Criterion | Consequence |
|-----------|-------------|
| **Missing Tier-0 freeze artifacts** | No valid run. Nonconformant |
| **Any instance of symbol capture** | Automatic nonconformance |
| **Using diagnostics as gates** | Automatic nonconformance |
| **Post hoc closure tuning** | Automatic nonconformance |
| **Promoting Tier-2 to Tier-1 without seam weld** | Automatic nonconformance |
| **Claiming continuity without demonstrating return** | Automatic nonconformance |

---

## Quick Reference

### Tier Table

| Tier | Role | Allowed | Forbidden |
|------|------|---------|-----------|
| **1** | Immutable invariants | Structural identities, reserved symbol meanings, dimensionless invariants | Being overridden by anyone; being treated as arbitrary math |
| **0** | Protocol | Regime gates, diagnostics, seam calculus, validation, verdicts, contract enforcement | Redefining invariants; adaptive preprocessing; hidden transforms |
| **2** | Expansion space | Domain closures, casepacks, contracts, canon anchors, domain models, narrative | Symbol capture; overriding regime or seam verdicts; structural override |

### Implementation Notes

#### YAML Contract Key: `tier_1_kernel`

The `tier_1_kernel` key in contract YAML files defines the Tier-1 immutable invariants that the contract inherits and freezes. The field name reflects the structural role: these are the kernel invariants whose identities (F+ω=1, IC≤F) are immutable across all domains. Domain contracts inherit this structure and extend with domain-specific closures at Tier-2.

The field is required by `schemas/contract.schema.json` and `schemas/canon.anchors.schema.json`.

---

## Implementation Status (v2.2.0)

- ✅ **Tier-1**: The kernel function defined, verified, and mathematically complete
  - 4 primitive equations (F, κ, S, C) + 2 derived values (ω = 1−F, IC = exp(κ)) — defining K: [0,1]ⁿ × Δⁿ → ℝ⁶
  - 3 effective degrees of freedom (F, κ, C) — rank invariant to input dimensionality (n = 4..64)
  - 3 algebraic identities + 1 statistical constraint — verified across 10,162 tests in 18 domains
  - 46 lemmas, 38 structural identities, 5 structural constants (c*, c_trap, ε, p, tol_seam)
  - Reserved symbols: F, ω, S, C, κ, IC, τ_R (all six outputs + τ_R are Tier-1)
  - Three-agent structural reading: Measuring (ω), Archive (F), Unknown (Γ(ω))

- ✅ **Tier-0**: Full protocol support
  - Validator: [src/umcp/validator.py](src/umcp/validator.py)
  - CLI: [src/umcp/cli.py](src/umcp/cli.py) (`umcp validate`)
  - Regime gate computation, three-valued verdicts
  - Diagnostics: Kinematics 5/5 Stable, PHYS-04 perfect fixed point
  - Seam structure defined; closures implemented
  - ⚠️ Automated weld PASS/FAIL computation not yet integrated into CLI

- ✅ **Tier-2**: 17 domain expansions active (511 entities, 42 theorems)
  - GCD, RCFT, Kinematics, Finance, Security, Astronomy
  - Nuclear, Quantum, Weyl, Atomic Physics, Materials Science, Standard Model
  - Everyday Physics, Evolution, Dynamic Semiotics, Consciousness Coherence, Continuity Theory

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and operational definitions
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - Formal mathematical specification of Tier-1 invariants
- [UHMP.md](docs/UHMP.md) - Universal Hash Manifest Protocol (identity governance)
- [FACE_POLICY.md](FACE_POLICY.md) - Boundary governance (Tier-0 admissibility)
- [SYMBOL_INDEX.md](docs/SYMBOL_INDEX.md) - Authoritative symbol table (prevents Tier-2 capture)
- [contracts/](contracts/) - Frozen contract specifications
- [closures/](closures/) - Domain expansion closures (Tier-2) and seam closures (Tier-0)
