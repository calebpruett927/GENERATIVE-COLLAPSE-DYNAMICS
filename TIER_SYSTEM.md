# UMCP Tier System

**Version**: 3.0.0
**Status**: Protocol Foundation
**Source**: UMCP Manuscript v1.0.0 §3 (revised per cross-domain validation, 146 experiments)
**Last Updated**: 2026-02-08

---

## Overview

The **UMCP tier system** has exactly three tiers. No half-tiers. No confusion.

| Tier | Name | Role |
|------|------|------|
| **1** | **The Kernel** | The mathematical function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC), its six definitions, their provable identities (F + ω = 1, IC ≤ F, IC = exp(κ)), and the 46 lemmas, 28 structural identities, and 5 structural constants that follow. Domain-independent. Immutable. |
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
         │ DEFINITIONS:                                     │
         │   F = Σ wᵢcᵢ     ω = 1-F     S = Bernoulli(c,w)│
         │   C = σ(c)/0.5   κ = Σ wᵢ ln cᵢ   IC = exp(κ) │
         │                                                  │
         │ IDENTITIES (theorems about the definitions):     │
         │   F + ω = 1    IC ≤ F    IC = exp(κ)            │
         │                                                  │
         │ STRUCTURAL CONSTANTS (derived, not chosen):      │
         │   c* = 0.7822   c_trap = 0.3178   ε = 10⁻⁸     │
         │                                                  │
         │ 46 LEMMAS · 28 IDENTITIES · 8 EQUATIONS         │
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

**What it is**: The kernel is the mathematical function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC). It takes a trace vector c ∈ [0,1]ⁿ and a weight vector w ∈ Δⁿ (the probability simplex) and produces six invariants. The six formulas define the function. The identities are theorems about it. The 46 lemmas, 28 structural identities, 8 equations, and 5 structural constants are all properties of this one mathematical object.

Tier-1 is the kernel function *and everything provable about it*. The identities are not separate objects floating above the kernel — they are consequences of the definitions. F + ω = 1 is a theorem about F(c,w) = Σ wᵢcᵢ and ω = 1 − F. IC ≤ F is a theorem about the relationship between the weighted geometric and arithmetic means. They cannot exist without the definitions that produce them.

### The Six Definitions

| Symbol | Formula | What It Computes |
|--------|---------|------------------|
| **F** | F = Σ wᵢcᵢ | Weighted arithmetic mean of the trace (fidelity) |
| **ω** | ω = 1 − F | Complement of fidelity (drift) |
| **S** | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] | Weighted Bernoulli field entropy |
| **C** | C = stddev(cᵢ) / 0.5 | Normalized standard deviation (curvature proxy) |
| **κ** | κ = Σ wᵢ ln(cᵢ,ε) | Weighted log-sum, ε-clamped (log-integrity) |
| **IC** | IC = exp(κ) | Weighted geometric mean (integrity composite) |

### Identities (theorems about the definitions — 0 violations across 10,162 tests in 17 domains)

| Identity | Why It Holds | Structural Meaning |
|----------|-------------|-------------------|
| **F + ω = 1** | By definition: ω := 1 − F | Fidelity and drift are exhaustive complements. No third bucket. |
| **IC ≤ F** | Geometric mean ≤ arithmetic mean (the integrity bound) | Coherence cannot exceed fidelity. Heterogeneity always costs. |
| **IC = exp(κ)** | By definition: κ := Σ wᵢ ln cᵢ, IC := exp(κ) | Log-integrity and multiplicative coherence are the same invariant in different coordinates. |

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

- **IS the kernel function**: The six formulas define a mathematical function on [0,1]ⁿ × Δⁿ. This function, and all its provable properties, is Tier-1.
- **IS the identities**: F + ω = 1, IC ≤ F, IC = exp(κ) are theorems *about* the kernel function. They are part of Tier-1 because they cannot exist without the definitions that produce them.
- **IS the lemmas and structural constants**: The 46 lemmas, 28 identities, c* = 0.7822, c_trap = 0.3178 — all properties of the same function.
- **NOT computation**: The *code* that evaluates the formulas is Tier-0. The *formulas themselves* are Tier-1. `kernel_optimized.py` is a Tier-0 implementation of the Tier-1 function.
- **NOT a model of the world**: Tier-1 makes no domain claims. It says "given trace c and weights w, F = Σ wᵢcᵢ." It does not say what those channels *mean* for a nuclide versus a star. Meaning-mapping is Tier-2.
- **NOT "constants we chose"**: The frozen parameters are **consistent** across the seam — the same rules on both sides of collapse-return. They are discovered by the mathematics, not prescribed by convention.

---

## Tier-0: Protocol

**What it is**: Everything that makes the Tier-1 kernel function computable, testable, and auditable. Tier-0 includes two distinct roles:

1. **Implementation of the kernel**: The code in `kernel_optimized.py` that evaluates the Tier-1 formulas F = Σ wᵢcᵢ, κ = Σ wᵢ ln cᵢ, etc. The code is Tier-0 (protocol); what it computes is Tier-1 (the function). If the code ever disagrees with the identities, the code is wrong — the identities are the test oracle.
2. **Interpretation of the output**: Everything that acts on the kernel's output — regime gates (translating continuous invariants into discrete labels), seam calculus (tracking continuity), contracts, schemas, SHA-256 integrity, and three-valued verdicts.

Tier-0 also handles the *input side*: embedding raw measurements x(t) into the trace vector Ψ(t) ∈ [0,1]ⁿ, assigning weights, and applying clipping/normalization policies.

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

**What it is**: The space where domain-specific **input choices** are made. The kernel function (Tier-1) takes c ∈ [0,1]ⁿ and w ∈ Δⁿ — it does not know or care what those channels represent. Tier-2 is where the decision is made to encode a proton as c = [0.49, 0.75, 0.67, 0.0, 0.75, 0.0, 1.0, 0.33] with channels [mass_log, spin, charge, color, T3, L, B, generation]. That encoding — which quantities become channels, how they are normalized, which entities are measured — is the domain closure.

Tier-2 answers: **What real-world quantities feed the kernel, and what do the kernel's outputs mean in domain-specific terms?**

The 42 proven theorems (SM T1–T10, MG T1–T10, KS T1–T7, CC T1–T7, PM T1–T8) are Tier-2 because they depend on domain-specific channel choices. "IC drops 98% at the quark-hadron boundary" is true given the 8-channel SM encoding — but the theorem requires that specific channel selection to state. The kernel computation it relies on is Tier-1; the channel choice that makes it about confinement is Tier-2.

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

## Implementation Status (v2.0.0)

- ✅ **Tier-1**: The kernel function defined, verified, and mathematically complete
  - Six definitions: F, ω, S, C, κ, IC — defining the kernel K: [0,1]ⁿ × Δⁿ → ℝ⁶
  - Three identities (theorems): F+ω=1, IC≤F, IC=exp(κ) — verified across 10,162 tests in 17 domains
  - 46 lemmas, 28 structural identities, 5 structural constants (c*, c_trap, ε, p, tol_seam)
  - Reserved symbols: F, ω, S, C, κ, IC, τ_R

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
