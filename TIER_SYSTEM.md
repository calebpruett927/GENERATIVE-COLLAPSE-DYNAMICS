# UMCP Tier System

**Version**: 2.0.0
**Status**: Protocol Foundation
**Source**: UMCP Manuscript v1.0.0 §3 (revised per cross-domain validation, 146 experiments)
**Last Updated**: 2026-02-08

---

## Overview

The **UMCP tier system** establishes strict separation between invariant structure (Tier-1), translation (Tier-0), diagnostics (Tier-1.5), and domain expansion (Tier-2). This separation makes claims auditable, falsifiable, and reproducible without narrative rescue.

**Tier-1** is invariant because it describes the **structure of collapse itself** — not because someone chose convenient equations. The identities hold across 146 experiments spanning 7 physics domains not because they were imposed, but because the structure forces them. F + ω = 1 is not a definition; it is the structural fact that fidelity and drift are complements — what isn't lost to drift *is* fidelity. There is no third option.

**Tier-0** translates that structure into operational verdicts — regime gates, three-valued status, τ_R classification. It is the validator, the filter, the machinery that makes Tier-1 actionable.

**Tier-1.5** confirms that the translation is faithful. Diagnostic casepacks (kinematics, PHYS-04) serve as control experiments — they verify the instrument reads zero when pointed at a known zero.

**Tier-2** is where physics happens. Domain expansion closures map raw measurements (binding energies, particle masses, stellar luminosities, market returns) into the Tier-1 invariants. Every domain expansion is validated *through* Tier-0, *against* Tier-1, *using* Tier-1.5 as reference.

**Core Principle**: One-way dependency flow within a frozen run, with return-based canonization between runs.

**Within-run**: Authority flows in one direction. The frozen interface (ingest + embedding + contract + closures + weights) determines the bounded trace Ψ(t); Tier-1 invariants are computed as functions of that frozen trace; Tier-2 domain closures may read Tier-1 outputs but cannot reach upstream to alter the interface, the trace, or the structural definitions. No back-edges, no retroactive tuning, no "the result changed the rules that produced it."

**Between-run**: Continuity is never presumed. A new run may exist freely, but it is only "canon-continuous" with a prior run if it returns and welds: the seam has admissible return (no continuity credit in ∞_rec segments), and the κ/IC continuity claim closes under the weld tolerances and identity checks. If that closure fails, the new run is still valid as an experiment or variant—but it does not become a canon edge.

**Constitutional Clauses** (equivalent formulations):
- "Within-run: frozen causes only. Between-run: continuity only by return-weld."
- "Runs are deterministic under /freeze; canon is a graph whose edges require returned seam closure."
- "No back-edges inside a run; no canon claims between runs without welded return."

**Formal Statement**: For any run r with frozen config φ_r and bounded trace Ψ_r(t), the Tier-1 structural invariants K_r(t) := K(Ψ_r(t); φ_r) hold regardless of any Tier-2 domain object. For two runs r₀, r₁, the statement "r₁ canonizes r₀" is admissible iff the seam returns (τ_R finite under policy) and the weld closes (ledger–budget residual within tol + identity check). Otherwise, r₁ is non-canon relative to r₀.

```
Tier-1   (invariant structure of collapse)
    ↓
Tier-0   (translation layer: regime gates, validator, verdicts)
    ↓
Tier-1.5 (diagnostics: confirms Tier-0 faithfully implements Tier-1)
         (seam calculus: tests continuity across transitions)
    ↓
Tier-2   (domain expansion closures: maps physics into Tier-1 structure)
    ✗ NO FEEDBACK within frozen run
    ✓ RETURN-BASED CANONIZATION across runs:
      IF Tier-2 meets/surpasses threshold
      THEN seam weld + declare new Tier-1 canon
      ELSE no promotion ("the cycle must return or it's not real")
```

### Return-Based Canonization (Axiom Embodiment)

**Rule**: Tier-2 results can be promoted to Tier-1 canon ONLY through formal validation:

1. **Threshold Validation**: Tier-2 result must meet or surpass declared criteria
2. **Seam Weld Required**: Compute continuity via seam calculus (Δκ, IC ratio, tolerance budget)
3. **New Canon Declaration**: If weld passes, declare new contract version with updated Tier-1 semantics
4. **No Ad-Hoc Promotion**: Results that don't "return" through validation remain Tier-2 domain closures

**Axiom Connection**: This embodies "What Returns Through Collapse Is Real":
- Tier-2 exploration = hypothesis space (domain expansion)
- Return validation = collapse event (testing against Tier-1 structure)
- Seam weld = proof of continuity
- Canon promotion = what survived becomes structural/canonical

**Example Flow**:
```
Run 1: Tier-2 discovers new metric M with improved stability
       → Validate: Does M meet stability threshold?
       → Compute seam: Δκ_M vs Δκ_old, tolerance check
       → IF |residual| ≤ tol_seam: Promote M to Tier-1 in new contract
       → ELSE: M remains Tier-2 domain closure

Run 2: Use new contract with M as Tier-1 invariant
       → Entire system operates on new canon
       → Previous M values are now structural outputs
```

---

## Tier-1: Invariant Structure of Collapse

**Definition**: Tier-1 is the invariant structure of collapse itself. It defines the structural relationships that hold regardless of what is collapsing, where, or when. Tier-1 is **discovered in the data, not imposed on it** — the structure forces the equations, not the other way around.

Tier-1 answers: **What is structurally true about collapse?**

### Structural Identities

These hold across all 146 validated experiments (0 violations):

| Identity | Structural Meaning | Validated |
|----------|-------------------|-----------|
| **F = 1 − ω** | Fidelity and drift are complements; total capacity is conserved | 146/146 |
| **IC ≤ F** | Informational coherence cannot exceed fidelity (AM-GM bound) | 146/146 |
| **IC ≈ exp(κ)** | Coherence tracks log-integrity exponentially | verified |

These are not definitions we chose. They are structural constraints discovered by computing F and ω independently from raw data and finding they sum to 1. A system cannot be more coherent than it is faithful. The bound emerges from how the invariants relate through the kernel — it is the geometry of collapse.

### Reserved Invariants (Structural Meanings)

| Symbol | Name | Formula | Range | Structural Role |
|--------|------|---------|-------|-----------------|
| **F** | Fidelity | F = Σ w_i c_i | [0,1] | How much of the signal survives collapse |
| **ω** | Drift | ω = 1 − F | [0,1] | How much is lost to collapse |
| **S** | Entropy | S = −Σ w_i [c_i ln(c_i) + (1−c_i)ln(1−c_i)] | ≥0 | Internal configurational complexity |
| **C** | Curvature | C = stddev(c_i)/0.5 | [0,1] | Coupling to uncontrolled external degrees of freedom |
| **κ** | Log-integrity | κ = Σ w_i ln(c_i,ε) | ≤0 | Logarithmic fidelity (sensitivity-aware) |
| **IC** | Integrity composite | IC = exp(κ) | (0,1] | Multiplicative coherence |
| **τ_R** | Return time | Re-entry delay to D_θ | ℕ∪{∞_rec} | How long until the system returns to its domain |
| **regime** | Regime label | Kernel gates on (ω,F,S,C) | {Stable, Watch, Collapse} | Which structural phase the system occupies |

### Required Properties

1. **Structural invariance**: The identities hold regardless of domain, scale, or physical substrate. They are not contingent on the choice of closure or measurement.
2. **Reserved meaning integrity**: Tier-1 symbols have exact meanings fixed across the entire protocol. Any alternative measure must be minted as Tier-2 with distinct notation and isolated outputs.
3. **Dimensionless by construction**: Tier-1 outputs are unitless because Ψ(t) ∈ [0,1]ⁿ. If you need unitful interpretation, it must remain in Tier-0 or be introduced as Tier-2 mapping.

### What Tier-1 Is NOT

- **Not "the math we picked"**: The identities describe structure discovered across domains, not equations selected for convenience.
- **Not "kernel computation"**: Computation is Tier-0's job. Tier-1 is the structure that computation must find.
- **Not a model of the world**: Tier-1 makes no domain claims. It says fidelity and drift are complements; it does not say what fidelity *means* for a nuclide versus a star versus a market. That meaning-mapping is Tier-2.

---

## Tier-0: Translation Layer

**Definition**: Tier-0 translates the invariant structure of Tier-1 into operational verdicts. It is the validator — the regime gates, the three-valued status, the τ_R classification, the contract checking, the SHA256 integrity verification. Tier-0 produces the bounded trace Ψ(t) from raw observations, declares units, embedding, clipping policy, weights, and return settings, and then **filters the Tier-1 structure into actionable output**.

Tier-0 answers: **How does Tier-1 get applied to data?**

### Translation Functions

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| **Regime classification** | (ω, F, S, C) | {Stable, Watch, Collapse} | Four-gate filter on invariant space |
| **Three-valued status** | Validation result | {CONFORMANT, NONCONFORMANT, NON_EVALUABLE} | Verdict |
| **τ_R classification** | Return time | {permanent, finite, retrocausal (forbidden)} | Return type |
| **Schema enforcement** | Artifact structure | {valid, invalid} | Structural conformance |
| **SHA256 integrity** | File contents | {match, mismatch} | Artifact identity |

### Regime Gates (Tier-0 Filtering Tier-1)

The four-gate criterion translates the continuous Tier-1 invariants into discrete regime labels:

```
Stable:   ω < 0.038  AND  F > 0.90  AND  S < 0.15  AND  C < 0.14
Collapse: ω ≥ 0.30
Watch:    everything else
```

These gates are the translation layer's primary function. They take the invariant space (ω, F, S, C) and partition it into three structural phases. The gates are conjunctive for Stable (all four must pass) because stability requires *all* invariants to be clean simultaneously.

### Allowed Scope

- **Observables**: Declare raw measurements x(t) with units, sampling resolution, and provenance
- **Embedding N_K**: Map x(t) → Ψ(t) ∈ [0,1]ⁿ with explicit clipping/normalization rules
- **OOR Policy**: Out-of-range handling (e.g., `pre_clip` with ε = 1e-8)
- **Face Policy**: Boundary behavior (see [FACE_POLICY.md](FACE_POLICY.md))
- **Weights**: Declare w_i ≥ 0, Σ w_i = 1
- **Return Settings**: Domain D_θ(t), tolerance η, horizon H_rec
- **Validation**: Contract checking, schema enforcement, integrity verification
- **Deterministic computation**: Compute the reserved Tier-1 invariants from frozen trace and weights

### Trace Row Format (Required)

**Rule**: In files, the mapping is fixed: c₁ ↔ `c1`, c₂ ↔ `c2`. Flags are explicit columns (e.g., `oor_c1`, `oor_c2`). Missingness indicators (if any) must also be explicit columns.

**Example trace.csv**:

```csv
t,c1,c2,oor_c1,oor_c2
1,0.42,0.30,0,0
2,0.55,0.28,0,0
3,0.99,0.15,1,0
```

Where:
- **t**: Time index (integer sample number or timestamp)
- **c1, c2**: Trace components Ψ(t) = (c₁(t), c₂(t)) ∈ [0,1]ⁿ
- **oor_c1, oor_c2**: OOR flags (0 = no clip, 1 = clipped at boundary)

**Warning**: If a reader cannot reconstruct Ψ(t) from stated units, conversions, bounds, and flag rules, the interface is **under-specified**. Under-specification is not "minor"; it **breaks auditability**.

### Prohibitions (Categorical)

| Prohibition | Violation | Consequence |
|-------------|-----------|-------------|
| **No Tier-1 structural redefinition** | Tier-0 introduces "ω" for angular frequency | Automatic nonconformance (symbol capture) |
| **No adaptive preprocessing** | Interface changes after seeing kernel outputs | New run required (re-freeze, re-hash, re-issue) |
| **No hidden transforms** | Undocumented smoothing or filtering | Not auditable, nonconformant |

**Warning**: Tier-0 may not introduce, redefine, or repurpose Tier-1 invariants: ω, F, S, C, τ_R, κ, IC. Tier-0 produces Ψ(t) (and flags) and computes the invariants; Tier-1 defines their structural meaning.

### Required Artifacts

```yaml
Tier-0 (frozen before compute):
  - contract.yaml         # Active contract ID, frozen params, tolerances
  - observables.yaml      # Units, sampling, pipeline, raw hashes
  - embedding.yaml        # N_K: x(t) → Ψ(t), OOR policy, face, ε
  - trace.csv             # Ψ(t) ∈ [0,1]ⁿ + flags
  - weights.yaml          # w_i, sum=1 validation
  - invariants.csv        # Computed Tier-1 invariants (deterministic output)
```

---

## Tier-1.5: Diagnostics and Seam Calculus

**Definition**: Tier-1.5 serves two functions: (a) **diagnostics** — control experiments that confirm the Tier-0 translation faithfully implements Tier-1 structure, and (b) **seam calculus** — continuity accounting across transitions that makes continuity claims falsifiable.

Tier-1.5 answers: **Does the Tier-0 filter faithfully implement Tier-1?** and **Is continuity across transitions real?**

### Diagnostics: The Control Experiments

Diagnostic casepacks confirm the translation layer works before any domain expansion result is trusted:

| Diagnostic | N | % Stable | What it confirms |
|------------|---|----------|------------------|
| **Kinematics** (KIN.INTSTACK.v1) | 5 | 100% | When C = 0 and ω is bounded, four-gate criterion correctly classifies all as Stable |
| **PHYS-04** (GCD.INTSTACK.v1) | 1 | 100% | When every invariant is at its ideal value (ω=0, F=1, S=0, C=0), the framework produces the expected fixed-point classification (IC=1, Stable, τ_R=INF_REC) |

These are *not* physics discoveries. They are **calibration anchors**. If kinematics didn't come back 100% Stable, or if PHYS-04 didn't hit all zeros, the translation layer is broken and no Tier-2 domain expansion result can be trusted.

Additional diagnostic roles:
- GCD reference casepack (gcd_complete): Verifies base contract conformance
- RCFT reference (rcft_complete): Verifies overlay conformance
- Any casepack with known-zero or known-boundary inputs

### Seam Calculus: Continuity Accounting

The seam calculus makes continuity claims **falsifiable** by reconciling ledger versus budget under frozen closures and issuing weld PASS/FAIL. If the seam fails, the correct output is **FAIL**, not narrative repair.

**Rule**: The seam test computes the **identity ledger term**:
```
Δκ_ledger = κ₁ − κ₀   (equivalently ln(IC₁/IC₀))
```

**Rule**: The seam test computes the **budget term**:
```
Δκ_budget = R·τ_R − (D_ω + D_C)
```
using frozen closures:
- Explicit Γ form (e.g., `gamma.default.v1.yaml`)
- R estimator (return credit, e.g., `return_domain.window64.v1.yaml`)
- Curvature neighborhood/variance convention (if non-default)
- D_C definition (e.g., αC)

**Rule**: The seam test computes the **residual**:
```
s = Δκ_budget − Δκ_ledger
```
and applies declared tolerances to issue PASS/FAIL:
```
PASS if |e^(Δκ) − IC₁/IC₀| < 10^-9  AND  |s| ≤ tol_seam
```

**Rule**: The seam may issue weld PASS/FAIL **only if return is finite** under declared horizon. If τ_R = ∞_rec, there is no return credit; the seam **cannot pass**.

### Required Artifacts

```yaml
Tier-1.5 (diagnostics + seam calculus; closures frozen):
  - diagnostic casepacks   # Kinematics, PHYS-04 (control experiments)
  - closures.yaml          # Γ form, R estimator, D_C definition, tolerances
  - welds.csv              # Ledger, budget components, residual, PASS/FAIL
  - ss1m_*.json            # SS1m receipt rows (seam + integrity metadata)
```

### Prohibitions

| Prohibition | Consequence |
|-------------|-------------|
| **No repair by closure** | Closures must be declared **before compute**. Cannot tune closures post hoc to force PASS. Categorical nonconformance. |
| **No narrative substitution** | FAIL is admissible outcome and must remain failed until new frozen run executed |
| **No continuity claim without weld row** | If seam is asserted, weld row is required |

---

## Tier-2: Domain Expansion Closures

**Definition**: Tier-2 is where physics happens. Domain expansion closures map raw measurements (binding energies, particle masses, stellar luminosities, market returns, decay half-lives) into the Tier-1 invariant structure. Every Tier-2 domain is validated *through* Tier-0 (translation), *against* Tier-1 (structure), *using* Tier-1.5 (diagnostics) as reference.

Tier-2 answers: **What fraction of each domain's configuration space has reached the Tier-1 fixed point?**

### Current Domain Expansions

| Domain | Contract | N | % Stable | Selector |
|--------|----------|---|----------|----------|
| Financial markets | FINANCE.INTSTACK.v1 | 12 | 58% | Bounded drift despite high coupling |
| Security / adversarial | SECURITY.INTSTACK.v1 | 25 | 40% | Engineered decoupling vs adversarial ω |
| Stellar & cosmological | ASTRO.INTSTACK.v1 | 28 | 25% | Gravitational equilibrium |
| Nuclear physics | NUC.INTSTACK.v1 | 30 | 23% | Proximity to iron peak |
| Quantum mechanics | QM.INTSTACK.v1 | 30 | 23% | Lightest in charge/spin class |
| Cosmology / dark energy | WEYL.INTSTACK.v1 | 4 | 0% | Irreducible cosmological drift |

### Purpose

**Rule**: Tier-2 exists so domain logic can be added without changing the invariant structure. Tier-2 closures map domain-specific observables into the five Tier-1 invariants, but they are **explicitly subordinate** to the structural identities and the Tier-0 regime verdicts.

### The Validation Path

Every Tier-2 experiment follows this path:

```
raw data → domain closure → (ω, F, S, C, IC, τ_R) → Tier-0 gates → regime verdict
                                     ↑                      ↑
                              Tier-1 identities        Tier-1.5 confirms
                              must hold                this path works
```

### Allowed Scope

1. **Domain closures**: Functions that map domain observables into Tier-1 invariants (e.g., `ω_eff = 1 − BE/A / 8.7945` for nuclear physics)
2. **Domain models**: Deterministic, stochastic, hybrid; ML models; mechanistic models (read lower-tier outputs; assumptions stated explicitly)
3. **Diagnostics**: Domain-specific residual checks, sensitivity analyses (labeled `diagnostic`, never used as gates)
4. **Controllers/heuristics**: Parameter suggestions, experimental design guidance (any proposed Tier-0 change = new run = new freeze)
5. **Narrative interpretation**: Ontology, philosophy (explicitly labeled as interpretive, never presented as structural output)

### Required Labeling Rules

**Rule**: Tier-2 quantities must use **distinct notation** and must **not reuse reserved symbols**. Collision avoidance is mandatory, not optional.

**Rule**: Tier-2 outputs must be **separated from Tier-1 outputs** in files and in manuscript blocks. Do not mix Tier-2 columns into `invariants.csv`.

**Rule**: Tier-2 must **declare assumptions and dependencies** explicitly so readers can bracket them without ambiguity.

### Prohibitions

| Prohibition | Violation Example |
|-------------|-------------------|
| **No symbol capture** | Tier-2 redefines ω, F, S, C, τ_R, κ, IC or regime gates |
| **No diagnostic-as-gate** | Tier-2 diagnostics cannot change regime labels or weld PASS/FAIL |
| **No rescue operations** | Tier-2 cannot override τ_R = ∞_rec or weld FAIL. It may propose new interface/experiment, but that is a new run |
| **No structural override** | Domain closures cannot modify the Tier-1 identities (F + ω = 1, IC ≤ F) |

### Required Artifacts

```yaml
Tier-2 (domain expansion; validated through Tier-0 against Tier-1):
  - closures/domain_name/   # Domain-specific closure functions
  - casepacks/domain_name/  # Casepack with raw data, manifest, expected invariants
  - canon/domain_anchors.yaml  # Domain canon anchors
  - contracts/DOMAIN.INTSTACK.v1.yaml  # Domain contract
```

---

## Cross-Tier Freeze and Flow Rules

### Freeze Gate

**Rule**: `/freeze` is a **hard gate**. Tier-0 declarations (observables, embedding, flags, weights, return settings) and closure registry (if weld is used) must be frozen **before** invariant computation and seam evaluation. If `/freeze` is missing or incomplete, the run is **nonconformant**.

### The Full Stack

```
Tier-1    │  Invariant structure of collapse              ← discovered, not imposed
          │  F + ω = 1,  IC ≤ F,  IC ≈ exp(κ)
          │
Tier-0    │  Translation layer (regime gates, validator)  ← filters structure into verdicts
          │
Tier-1.5  │  Diagnostics (kinematics, PHYS-04)           ← confirms the filter is faithful
          │  + Seam calculus (weld)                       ← tests continuity across transitions
          │
Tier-2    │  Domain expansion closures                    ← maps physics into the structure
          │  FIN 58%  SEC 40%  ASTRO 25%  NUC 23%  QM 23%  WEYL 0%
```

**Tier-1 says**: this is how collapse is structured.
**Tier-0 makes**: that structure operational.
**Tier-1.5 proves**: the operation preserves the structure.
**Tier-2 discovers**: which fraction of each domain's configuration space has reached the structure's fixed point.

### One-Way Dependency Within Runs

**Rule**: Tier-0 implements Tier-1; Tier-1.5 depends on Tier-0 plus closures; Tier-2 may depend on all prior tiers but **cannot feed back** to alter their outcomes for a frozen run.

```
WITHIN A FROZEN RUN:
Tier-1 (structure) → Tier-0 (translation) → Tier-1.5 (diagnostics + weld) → Tier-2 (domain)
                                                                               ↓
                                                                          NO FEEDBACK ✗

ACROSS RUNS (Return-Based Canonization):
Run N: Tier-2 domain result
         ↓ threshold validation
         ↓ seam weld computation
         ↓ IF weld passes ("returns")
Run N+1: New Tier-1 canon (promoted result)
         ✓ FORMAL PROMOTION via contract versioning
```

**Critical Distinction**:
- **Within-run feedback**: PROHIBITED (breaks determinism, enables narrative rescue)
- **Cross-run canonization**: ALLOWED via seam welding (embodies "cycle must return")
- **Requirement**: Return validation is not optional — results that don't return remain non-canonical

### Audit Traceability

**Rule**: Any change between runs must be **attributable to a tier**. If you cannot localize the change, you do not have an audit trail and the comparison is **invalid under the standard**.

---

## Nonconformance Criteria (Tier-Linked)

| Criterion | Consequence |
|-----------|-------------|
| **Missing Tier-0 freeze artifacts** | No valid run. Report without contract.yaml and auditable trace is nonconformant |
| **Missing closure registry** | No valid continuity claim. Seam assertion without closures.yaml and welds.csv is nonconformant |
| **Any instance of symbol capture** | Automatic nonconformance |
| **Using diagnostics as gates** | Automatic nonconformance |
| **Post hoc changes to preprocessing/closures to force outcomes** | Automatic nonconformance |
| **Promoting Tier-2 to Tier-1 without seam weld** | Automatic nonconformance. Canon promotion requires return validation |
| **Claiming continuity without demonstrating return** | Automatic nonconformance. "The cycle must return or it's not real" |

---

## Minimal Examples

### Clean Example (Proper Tier Separation)

**Definition**: This example shows correct separation: Tier-0 declarations → Tier-1.5 diagnostic confirmation → Tier-2 domain expansion. The invariant structure (Tier-1) holds throughout.

```
casepacks/example_clean/
├── contract.yaml                    # Tier-0: Frozen contract
├── observables.yaml                 # Tier-0: Units, sampling, raw hashes
├── embedding.yaml                   # Tier-0: N_K, OOR policy, ε
├── trace.csv                        # Tier-0: Ψ(t) ∈ [0,1]ⁿ + flags
├── weights.yaml                     # Tier-0: w_i, sum=1 check
├── derived/
│   ├── invariants.csv               # Tier-0 output: F, ω, S, C, κ, IC, τ_R, regime
│   ├── closures.yaml                # Tier-1.5: Γ form, R, D_C, tolerances
│   ├── welds.csv                    # Tier-1.5: dk_ledger, dk_budget, s, PASS/FAIL
│   └── ss1m_receipt.json            # Tier-1.5: Seam metadata
└── domain/
    ├── diagnostics.csv              # Tier-2: Domain-specific sensitivity checks
    ├── model_forecast.py            # Tier-2: Domain model (assumptions stated)
    └── interpretation.md            # Tier-2: Narrative (explicitly labeled)
```

### Violation Example (Nonconformant) and Correction

**Violation**: Diagnostic-as-gate + symbol capture

```
❌ NONCONFORMANT:

We measure capacitance C(t) and later compute kernel curvature C.
We then re-label a run as Stable because an equator diagnostic looks good,
even though the kernel gate says Watch.
```

**Why invalid**:
1. **Symbol capture**: C is reserved for kernel curvature proxy (Tier-1 structural invariant). Using C(t) for capacitance breaks auditability
2. **Diagnostic-as-gate**: Tier-2 diagnostics cannot override Tier-0 regime labels

**Correction** (conformant):

```
✅ CONFORMANT:

Tier-0: Measure capacitance C_cap(t) [units: F].
        Declare pipeline and hash raw data.
        Declare embedding N_K and produce Ψ(t) ∈ [0,1]ⁿ with flags.
        Compute kernel curvature C (reserved structural meaning) from Ψ(t).
        Regime label comes from Tier-0 gates implementing Tier-1 structure.

Tier-2: Compute equator diagnostic as report-only.
        Diagnostic is recorded, but does NOT change regime label.
```

---

## Pre-Compute Checklist (Mandatory)

**Before you compute: verify these (mandatory)**

### 1. Tier-0 Freeze Complete

- [ ] `contract.yaml` exists and is the active contract for this run
- [ ] `observables.yaml` exists (units, sampling, pipeline, raw hashes)
- [ ] `embedding.yaml` exists (N_K, OOR policy, face, ε)
- [ ] `trace.csv` exists (bounded Ψ(t) + flags), or deterministic recipe exists to regenerate it
- [ ] Weights w_i are declared, validated (Σ w_i = 1), and frozen

### 2. Tier-1.5 Diagnostics Confirmed

- [ ] At least one diagnostic casepack (kinematics or PHYS-04) validates as expected
- [ ] Regime gates produce expected results on known inputs

### 3. If You Will Claim Continuity (Weld)

- [ ] `closures.yaml` exists (Γ form, R-estimator, D_C definition, tolerances)
- [ ] Closure parameters are pinned and hashed; no "tune-to-pass" permitted

### 4. Tier Discipline Sanity

- [ ] No reserved symbols reused for domain meanings (no symbol capture)
- [ ] No diagnostic will be used as a gate (Tier-0 gates only)
- [ ] No post hoc preprocessing/closure changes after inspecting outputs

---

## Quick Reference Tables

### Tier Table: Role, Allowed, Forbidden

| Tier | Role | Allowed | Forbidden |
|------|------|---------|-----------|
| **1** | Invariant structure of collapse | Structural identities (F+ω=1, IC≤F), reserved symbol meanings, dimensionless invariants | Being treated as arbitrary math; being overridden by domain claims |
| **0** | Translation layer | Observables, embedding, OOR policy, weights, regime gate computation, validation, verdicts | Redefining structural invariants; adaptive preprocessing; hidden transforms |
| **1.5** | Diagnostics + Seam calculus | Control experiments (kinematics, PHYS-04); weld accounting (ledger/budget/residual/PASS/FAIL) | Post hoc tuning to force PASS; continuity claim without weld row; overriding diagnostic results |
| **2** | Domain expansion closures | Domain closure functions, casepacks, contracts, canon anchors, domain models, narrative | Symbol capture; overriding regime or weld verdicts; rescuing τ_R = ∞_rec; modifying structural identities |

### Dependency Flow

```
WITHIN FROZEN RUN:
Tier-1 (structure: F+ω=1, IC≤F)
    ↓
Tier-0 (translation: compute invariants, apply regime gates, issue verdicts)
    ↓ outputs: invariants.csv + regime labels
Tier-1.5 (diagnostics: confirm gates work; seam: compute weld)
    ↓ outputs: diagnostic verdicts, welds.csv + SS1m
Tier-2 (domain expansion: map physics into Tier-1; may read anything above)
    ✗ Tier-2 may NOT feed back to alter Tier-1/0/1.5 outcomes

ACROSS RUNS (Return-Based Canonization):
Tier-2 result → threshold check → seam weld → IF passes → New Tier-1 canon
                                             → IF fails → Remains Tier-2
(Embodies: "The cycle must return or it's not real")
```

---

## Implementation Notes

### YAML Contract Key: `tier_1_kernel`

The `tier_1_kernel` key in contract YAML files (e.g., `contracts/GCD.INTSTACK.v1.yaml`) defines the Tier-1 invariant structure that the contract inherits and freezes. The field name reflects the structural role: these are the kernel invariants whose structural identities (F+ω=1, IC≤F) are invariant across all domains. Domain contracts inherit this structure via "Tier-1 (inherited, frozen)" and extend it with domain-specific closures at Tier-2.

### Schema Validation

The `tier_1_kernel` field is required by `schemas/contract.schema.json` and `schemas/canon.anchors.schema.json`. It contains frozen parameters, tolerances, and identity definitions that embody the Tier-1 invariant structure.

---

## Implementation Status in UMCP

### Current Support (v1.5.0)

- ✅ **Tier-1**: Invariant structure defined and verified
  - Structural identities: F+ω=1, IC≤F — 146/146 hold
  - Reserved symbols: F, ω, S, C, κ, IC, τ_R
  - Regime structure: Stable, Watch, Collapse

- ✅ **Tier-0**: Full translation layer support
  - [src/umcp/validator.py](src/umcp/validator.py): Contract validation, invariant computation
  - [src/umcp/cli.py](src/umcp/cli.py): `umcp validate` command
  - contract.yaml, observables.yaml, embedding.yaml parsing
  - trace.csv with OOR flags
  - weights.yaml validation
  - Regime gate computation
  - Three-valued CONFORMANT/NONCONFORMANT/NON_EVALUABLE verdicts

- ✅ **Tier-1.5**: Diagnostics confirmed + seam structure defined
  - Kinematics casepack: 5/5 Stable (100%) — diagnostic confirmed
  - PHYS-04 casepack: 1/1 perfect fixed point — diagnostic confirmed
  - closures/registry.yaml structure defined
  - Individual closures implemented (gamma, return_domain, etc.)
  - welds.csv structure documented
  - ⚠️ Automated weld PASS/FAIL computation not yet integrated into CLI

- ✅ **Tier-2**: 6 domain expansions active
  - Finance (12 experiments, 58% Stable)
  - Security (25 experiments, 40% Stable)
  - Astronomy (28 experiments, 25% Stable)
  - Nuclear physics (30 experiments, 23% Stable)
  - Quantum mechanics (30 experiments, 23% Stable)
  - Cosmology/Weyl (4 experiments, 0% Stable)
  - Total: 129 domain expansion experiments + 17 calibration/diagnostic = 146 validated

### Future Work

- [ ] Implement automated weld computation in `umcp validate` CLI
- [ ] Add pre-compute checklist validation command
- [ ] Enforce Tier-2 domain closure separation in file structure
- [ ] Add symbol capture detection in validation
- [ ] Create tier violation detection tests

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and operational definitions
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - Formal mathematical specification of Tier-1 invariants
- [UHMP.md](UHMP.md) - Universal Hash Manifest Protocol (identity governance)
- [FACE_POLICY.md](FACE_POLICY.md) - Boundary governance (Tier-0 admissibility)
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - Authoritative symbol table (prevents Tier-2 capture)
- [contracts/](contracts/) - Frozen contract specifications
- [closures/](closures/) - Domain expansion closures (Tier-2) and seam closures (Tier-1.5)
