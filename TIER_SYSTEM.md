# UMCP Tier System

**Version**: 1.0.0  
**Status**: Protocol Foundation  
**Source**: UMCP Manuscript v1.0.0 §3  
**Last Updated**: 2026-01-21

---

## Overview

The **UMCP tier system** establishes strict separation between interface (Tier-0), kernel computation (Tier-1), continuity accounting (Tier-1.5), and domain overlays (Tier-2). This separation makes claims auditable, falsifiable, and reproducible without narrative rescue.

**Core Principle**: One-way dependency flow within a frozen run, with return-based canonization between runs.

**Within-run**: Authority flows in one direction. The frozen interface (ingest + embedding + contract + closures + weights) determines the bounded trace Ψ(t); Tier-1 invariants are computed as functions of that frozen trace; Tier-2 overlays may read Tier-1 outputs but cannot reach upstream to alter the interface, the trace, or the kernel definitions. No back-edges, no retroactive tuning, no "the result changed the rules that produced it."

**Between-run**: Continuity is never presumed. A new run may exist freely, but it is only "canon-continuous" with a prior run if it returns and welds: the seam has admissible return (no continuity credit in ∞_rec segments), and the κ/IC continuity claim closes under the weld tolerances and identity checks. If that closure fails, the new run is still valid as an experiment or variant—but it does not become a canon edge.

**Constitutional Clauses** (equivalent formulations):
- "Within-run: frozen causes only. Between-run: continuity only by return-weld."
- "Runs are deterministic under /freeze; canon is a graph whose edges require returned seam closure."
- "No back-edges inside a run; no canon claims between runs without welded return."

**Formal Statement**: For any run r with frozen config φ_r and bounded trace Ψ_r(t), Tier-1 kernel K_r(t) := K(Ψ_r(t); φ_r) is invariant to any Tier-2 object. For two runs r₀, r₁, the statement "r₁ canonizes r₀" is admissible iff the seam returns (τ_R finite under policy) and the weld closes (ledger–budget residual within tol + identity check). Otherwise, r₁ is non-canon relative to r₀.

```
Tier-0 (declare + freeze) 
    ↓ 
Tier-1 (kernel compute) 
    ↓ 
Tier-1.5 (weld compute) 
    ↓ 
Tier-2 (diagnostics/models/narrative)
    ✗ NO FEEDBACK within frozen run
    ✓ RETURN-BASED CANONIZATION across runs:
      IF Tier-2 meets/surpasses threshold
      THEN seam weld + declare new Tier-1 canon
      ELSE no promotion ("the cycle must return or it's not real")
```

### Return-Based Canonization (Axiom Embodiment)

**Rule**: Tier-2 results can be promoted to Tier-1 canon ONLY through formal validation:

1. **Threshold Validation**: Tier-2 result must meet or surpass declared criteria
2. **Seam Weld Required**: Compute continuity via Tier-1.5 machinery (Δκ, IC ratio, tolerance budget)
3. **New Canon Declaration**: If weld passes, declare new contract version with updated Tier-1 semantics
4. **No Ad-Hoc Promotion**: Results that don't "return" through validation remain Tier-2 overlays

**Axiom Connection**: This embodies "What Returns Through Collapse Is Real":
- Tier-2 exploration = hypothesis space
- Return validation = collapse event (testing against Tier-1 invariants)
- Seam weld = proof of continuity
- Canon promotion = what survived becomes real/canonical

**Example Flow**:
```
Run 1: Tier-2 discovers new metric M with improved stability
       → Validate: Does M meet stability threshold?
       → Compute seam: Δκ_M vs Δκ_old, tolerance check
       → IF |residual| ≤ tol_seam: Promote M to Tier-1 in new contract
       → ELSE: M remains Tier-2 diagnostic

Run 2: Use new contract with M as Tier-1 invariant
       → Entire system operates on new canon
       → Previous M values are now kernel outputs
```

---

## Tier-0: Interface Layer

**Definition**: Tier-0 produces the bounded trace Ψ(t) from raw observations. It declares units, embedding, clipping policy, weights, and return settings. **Tier-0 is frozen before Tier-1 computes**.

### Allowed Scope

- **Observables**: Declare raw measurements x(t) with units, sampling resolution, and provenance
- **Embedding N_K**: Map x(t) → Ψ(t) ∈ [0,1]ⁿ with explicit clipping/normalization rules
- **OOR Policy**: Out-of-range handling (e.g., `pre_clip` with ε = 1e-8)
- **Face Policy**: Boundary behavior (see [FACE_POLICY.md](FACE_POLICY.md))
- **Weights**: Declare w_i ≥ 0, Σ w_i = 1
- **Return Settings**: Domain D_θ(t), tolerance η, horizon H_rec

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
| **No Tier-1 symbol redefinition** | Tier-0 introduces "ω" for angular frequency | Automatic nonconformance (symbol capture) |
| **No adaptive preprocessing** | Interface changes after seeing kernel outputs | New run required (re-freeze, re-hash, re-issue) |
| **No hidden transforms** | Undocumented smoothing or filtering | Not auditable, nonconformant |

**Warning**: Tier-0 may not introduce, redefine, or repurpose Tier-1 invariants: ω, F, S, C, τ_R, κ, IC. Tier-0 produces Ψ(t) (and flags); Tier-1 produces invariants from it.

### Required Artifacts

```yaml
Tier-0 (frozen before compute):
  - contract.yaml         # Active contract ID, frozen params, tolerances
  - observables.yaml      # Units, sampling, pipeline, raw hashes
  - embedding.yaml        # N_K: x(t) → Ψ(t), OOR policy, face, ε
  - trace.csv             # Ψ(t) ∈ [0,1]ⁿ + flags
  - weights.yaml          # w_i, sum=1 validation
```

---

## Tier-1: Kernel Invariants

**Definition**: Tier-1 is the kernel. It computes reserved invariants deterministically from frozen Tier-0 trace and frozen weights. **Tier-1 is not a model of the world; it is an audit computation over a bounded representation.**

### Allowed Scope (Strict Determinism Given Frozen Tier-0)

**Rule**: Tier-1 may compute reserved invariants from Ψ_ε(t) and w_i **only**. If an input is not in frozen Tier-0 artifacts, Tier-1 cannot use it.

**Rule**: Tier-1 may compute return status τ_R using frozen norm/metric, D_θ(t), η, and H_rec. If the declared time basis or sampling resolution makes τ_R non-identifiable, the correct output is a **typed boundary** (∞_rec), not an estimate.

**Rule**: Tier-1 must produce time-indexed outputs (e.g., `invariants.csv`) with explicit **determinism guarantee**: same frozen inputs ⇒ same outputs. If an implementation cannot satisfy this, it is not a Tier-1 implementation under the standard.

### Reserved Invariants (Canonical Meanings)

| Symbol | Name | Formula | Range | File Column |
|--------|------|---------|-------|-------------|
| **F** | Fidelity | F = Σ w_i c_i | [0,1] | `F` |
| **ω** | Drift | ω = 1 - F | [0,1] | `omega` |
| **S** | Entropy | S = -Σ w_i [c_i ln(c_i) + (1-c_i)ln(1-c_i)] | ≥0 | `S` |
| **C** | Curvature | C = stddev(c_i)/0.5 | [0,1] | `C` |
| **κ** | Log-integrity | κ = Σ w_i ln(c_i,ε) | ≤0 | `kappa` |
| **IC** | Integrity composite | IC = exp(κ) | (0,1] | `IC` |
| **τ_R** | Return time | Re-entry delay to D_θ | ℕ∪{∞_rec} | `tau_R` |
| **regime** | Regime label | Kernel gates on (ω,F,S,C) | {Stable, Watch, Collapse} | `regime` |

### Required Properties

1. **Determinism**: No randomness; no adaptive branching; no dependence on external state (system time, nondeterministic floating kernels, unpinned library behavior)
2. **Reserved meaning integrity**: Tier-1 symbols have exact meanings fixed across entire protocol. Any alternative measure must be minted as Tier-2 with distinct notation and isolated outputs
3. **Dimensionless by construction**: Tier-1 outputs are unitless because Ψ(t) ∈ [0,1]ⁿ. If you need unitful interpretation, it must remain in Tier-0 or be introduced as Tier-2 mapping

### Prohibitions

| Prohibition | Explanation |
|-------------|-------------|
| **No probabilistic interpretation by default** | Probability may be introduced as Tier-2 overlay, but Tier-1 does not require it and does not presume it |
| **No domain narrative/model fitting/prediction** | Those are Tier-2 operations and must be labeled as such |
| **No diagnostics as gates** | Regime labels are kernel-only and may not be overridden by Tier-2 |

### Required Artifacts

```yaml
Tier-1 (deterministic kernel):
  - invariants.csv        # F, ω, S, C, κ, IC, τ_R, regime (time-indexed)
```

**Example invariants.csv**:

```csv
t,F,omega,S,C,kappa,IC,tau_R,regime
1,0.92,0.08,0.11,0.10,-0.15,0.86,12,Stable
2,0.88,0.12,0.14,0.12,-0.22,0.80,18,Stable
3,0.35,0.65,0.42,0.28,-2.10,0.12,INF_REC,Collapse
```

---

## Tier-1.5: Seam Calculus (Weld)

**Definition**: Tier-1.5 is the seam calculus: continuity accounting across transitions. It makes continuity claims **falsifiable** by reconciling ledger versus budget under frozen closures and issuing weld PASS/FAIL. If the seam fails, the correct output is **FAIL**, not narrative repair.

### Purpose

**Rule**: Tier-1.5 exists to audit transitions t₀ → t₁ **without interpretive rescue**. The seam test is an accounting identity plus a declared budget model. Continuity is **not asserted**; it is **tested**.

### Allowed Scope (Requires Explicit Closure Registry)

**Rule**: Tier-1.5 may compute the **identity ledger term**:
```
Δκ_ledger = κ₁ - κ₀   (equivalently ln(IC₁/IC₀))
```

**Rule**: Tier-1.5 may compute the **budget term**:
```
Δκ_budget = R·τ_R - (D_ω + D_C)
```
using frozen closures:
- Explicit Γ form (e.g., `gamma.default.v1.yaml`)
- R estimator (return credit, e.g., `return_domain.window64.v1.yaml`)
- Curvature neighborhood/variance convention (if non-default)
- D_C definition (e.g., αC)

**Rule**: Tier-1.5 may compute the **residual**:
```
s = Δκ_budget - Δκ_ledger
```
and apply declared tolerances to issue PASS/FAIL:
```
PASS if |e^(Δκ) - IC₁/IC₀| < 10^-9  AND  |s| ≤ tol_seam
```

**Rule**: Tier-1.5 may issue weld PASS/FAIL **only if return is finite** under declared horizon. If τ_R = ∞_rec, there is no return credit; the seam **cannot pass**.

### Required Artifacts

```yaml
Tier-1.5 (seam calculus; closures frozen):
  - closures.yaml         # Γ form, R estimator, D_C definition, tolerances
  - welds.csv             # Ledger, budget components, residual, PASS/FAIL
  - ss1m_*.json           # SS1m receipt rows (seam + integrity metadata)
```

**Example closures.yaml** (registry):

```yaml
schema: closures.registry.v1
closures:
  gamma:
    id: gamma.default.v1
    form: "Gamma(omega; p, epsilon) = omega^p / (omega^p + epsilon^p)"
    parameters:
      p: 3
      epsilon: 0.001
    file: closures/gamma.default.v1.yaml
    
  return_credit:
    id: return_domain.window64.v1
    estimator: "R = mean(1 - omega[t-64:t]) if tau_R < inf else 0"
    file: closures/return_domain.window64.v1.yaml
    
  curvature_dissipation:
    form: "D_C = alpha * C"
    parameters:
      alpha: 1.0
```

**Example welds.csv**:

```csv
weld_id,t0,t1,dk_ledger,dk_budget,D_omega,D_C,R,tau_R,residual_s,tol_seam,pass
W-2026-01-A,100,200,-0.05,-0.048,0.12,0.10,0.85,12,0.002,0.005,PASS
W-2026-01-B,200,300,-0.15,-0.18,0.25,0.15,0.72,INF_REC,NA,0.005,FAIL
```

### Prohibitions

| Prohibition | Consequence |
|-------------|-------------|
| **No repair by closure** | Closures must be declared **before compute**. Cannot tune closures post hoc to force PASS. Categorical nonconformance. |
| **No Tier-2 narrative substitution** | FAIL is admissible outcome and must remain failed until new frozen run executed |
| **No continuity claim without weld row** | If seam is asserted, weld row is required |

---

## Tier-2: Overlays

**Definition**: Tier-2 enables domain-specific modeling, diagnostics, control, and interpretation **without corrupting the kernel**. Tier-2 may explain or predict, but it remains **subordinate** to Tier-1 and Tier-1.5 outputs and **cannot rewrite them**.

### Purpose

**Rule**: Tier-2 exists so domain logic can be added without changing what the kernel computed. Tier-2 is **explicitly non-authoritative** over regime labels and weld PASS/FAIL.

### Allowed Scope

1. **Diagnostics**: Residual checks, equator relations, sensitivity analyses (labeled `diagnostic`, never used as gates)
2. **Domain models**: Deterministic, stochastic, hybrid; ML models; mechanistic models (read lower-tier outputs; assumptions stated explicitly)
3. **Controllers/heuristics**: Parameter suggestions, experimental design guidance (any proposed Tier-0 change = new run = new freeze)
4. **Narrative interpretation**: Ontology, philosophy (explicitly labeled as interpretive, never presented as kernel output)

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

### Required Artifacts

```yaml
Tier-2 (overlays; explicitly subordinate):
  - diagnostics.csv       # Sensitivity, equator checks, residual plots
  - model_*.py            # Domain models with assumptions declared
  - interpretation.md     # Narrative; explicitly labeled Tier-2
```

---

## Cross-Tier Freeze and Flow Rules

### Freeze Gate

**Rule**: `/freeze` is a **hard gate**. Tier-0 declarations (observables, embedding, flags, weights, return settings) and closure registry (if weld is used) must be frozen **before** Tier-1 and Tier-1.5 compute. If `/freeze` is missing or incomplete, the run is **nonconformant**.

### One-Way Dependency Within Runs

**Rule**: Tier-1 depends on Tier-0; Tier-1.5 depends on Tier-1 plus closures; Tier-2 may depend on all prior tiers but **cannot feed back** to alter their outcomes for a frozen run.

```
WITHIN A FROZEN RUN:
Tier-0 (frozen) → Tier-1 (kernel) → Tier-1.5 (weld) → Tier-2 (overlay)
                                                         ↓
                                                    NO FEEDBACK ✗

ACROSS RUNS (Return-Based Canonization):
Run N: Tier-2 result
         ↓ threshold validation
         ↓ seam weld computation
         ↓ IF weld passes ("returns")
Run N+1: New Tier-1 canon (promoted result)
         ✓ FORMAL PROMOTION via contract versioning
```

**Critical Distinction**:
- **Within-run feedback**: PROHIBITED (breaks determinism, enables narrative rescue)
- **Cross-run canonization**: ALLOWED via seam welding (embodies "cycle must return")
- **Requirement**: Return validation is not optional - results that don't return remain non-canonical

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

**Definition**: This example shows correct separation: Tier-0 declarations → Tier-1 outputs → Tier-1.5 weld row → Tier-2 diagnostic. Nothing feeds back into the kernel.

```
casepacks/example_clean/
├── contract.yaml                    # Tier-0: Frozen contract
├── observables.yaml                 # Tier-0: Units, sampling, raw hashes
├── embedding.yaml                   # Tier-0: N_K, OOR policy, ε
├── trace.csv                        # Tier-0: Ψ(t) ∈ [0,1]ⁿ + flags
├── weights.yaml                     # Tier-0: w_i, sum=1 check
├── derived/
│   ├── invariants.csv               # Tier-1: F, ω, S, C, κ, IC, τ_R, regime
│   ├── closures.yaml                # Tier-1.5: Γ form, R, D_C, tolerances
│   ├── welds.csv                    # Tier-1.5: dk_ledger, dk_budget, s, PASS/FAIL
│   └── ss1m_receipt.json            # Tier-1.5: Seam metadata
└── tier2/
    ├── diagnostics.csv              # Tier-2: Sensitivity, equator checks
    ├── model_forecast.py            # Tier-2: Predictive model (assumptions stated)
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
1. **Symbol capture**: C is reserved for kernel curvature proxy. Using C(t) for capacitance breaks auditability
2. **Diagnostic-as-gate**: Tier-2 diagnostics cannot override Tier-1 regime labels

**Correction** (conformant):

```
✅ CONFORMANT:

Tier-0: Measure capacitance C_cap(t) [units: F]. 
        Declare pipeline and hash raw data.
        Declare embedding N_K and produce Ψ(t) ∈ [0,1]ⁿ with flags.

Tier-1: Compute kernel curvature C (reserved meaning) from Ψ(t).
        Regime label comes from kernel gates only.

Tier-2: Compute equator diagnostic as report-only diagnostic.
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

### 2. If You Will Claim Continuity (Weld)

- [ ] `closures.yaml` exists (Γ form, R-estimator, D_C definition, tolerances)
- [ ] Closure parameters are pinned and hashed; no "tune-to-pass" permitted

### 3. Tier Discipline Sanity

- [ ] No reserved symbols reused for domain meanings (no symbol capture)
- [ ] No diagnostic will be used as a gate (kernel gates only)
- [ ] No post hoc preprocessing/closure changes after inspecting outputs

---

## Quick Reference Tables

### Tier Table: Allowed, Forbidden, Required Artifacts

| Tier | Allowed | Forbidden | Required Artifacts |
|------|---------|-----------|-------------------|
| **0** | Observables, units, pipeline, embedding N_K, OOR/missingness policy, ε clip, w_i, return settings | Defining/redefining kernel invariants; adaptive preprocessing; hidden transforms | contract.yaml, observables.yaml, embedding.yaml, trace.csv, raw hashes, flags |
| **1** | Deterministic computation of kernel invariants and return status from frozen Tier-0 | Model fitting; probabilistic claims by default; diagnostics-as-gates | invariants.csv (+ regime labels) |
| **1.5** | Weld accounting: ledger/budget/residual; PASS/FAIL under frozen closures | Post hoc tuning to force PASS; continuity claim without weld row; narrative repair | closures.yaml, welds.csv, SS1m receipts |
| **2** | Diagnostics, models, controllers, narrative interpretation (explicitly labeled) | Symbol capture; overriding regime or weld PASS/FAIL; rescuing τ_R = ∞_rec | Separate overlay files, assumptions declared, distinct notation |

### Dependency Flow (Within-Run vs Cross-Run)

```
WITHIN FROZEN RUN:
Tier-0 (declare + freeze)
    ↓
Tier-1 (kernel compute)
    ↓ outputs: invariants.csv + regime
Tier-1.5 (weld compute)
    ↓ outputs: welds.csv + SS1m
Tier-2 (diagnostics/models/narrative; may read anything above)
    ✗ Tier-2 may NOT feed back to alter Tier-0/1/1.5 outcomes

ACROSS RUNS (Return-Based Canonization):
Tier-2 result → threshold check → seam weld → IF passes → New Tier-1 canon
                                            → IF fails → Remains Tier-2
(Embodies: "The cycle must return or it's not real")
```

---

## Implementation Status in UMCP

### Current Support (v1.5.0)

- ✅ **Tier-0**: Full support in [src/umcp/validator.py](src/umcp/validator.py)
  - contract.yaml, observables.yaml, embedding.yaml parsing
  - trace.csv with OOR flags
  - weights.yaml validation
  
- ✅ **Tier-1**: Full support in [src/umcp/validator.py](src/umcp/validator.py) and [closures/](closures/)
  - Deterministic invariant computation
  - Reserved symbols: F, ω, S, C, κ, IC, τ_R
  - Regime gates: Stable, Watch, Collapse
  - Output: outputs/invariants.csv
  
- ⚠️ **Tier-1.5**: Partial support
  - closures/registry.yaml structure defined
  - Individual closures implemented (gamma, return_domain, etc.)
  - welds.csv structure documented
  - ⚠️ Automated weld PASS/FAIL computation not yet integrated into CLI
  
- ⚠️ **Tier-2**: Partial support
  - Visualization dashboard (Streamlit) as Tier-2 overlay
  - API endpoints for diagnostics
  - ⚠️ No enforcement of "no diagnostic-as-gate" rule in code

### Future Work

- [ ] Implement automated weld computation in `umcp validate` CLI
- [ ] Add pre-compute checklist validation command
- [ ] Enforce Tier-2 overlay separation in file structure
- [ ] Add symbol capture detection in validation
- [ ] Create tier violation detection tests

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and operational definitions
- [UHMP.md](UHMP.md) - Universal Hash Manifest Protocol (identity governance)
- [FACE_POLICY.md](FACE_POLICY.md) - Boundary governance (Tier-0 admissibility)
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - Authoritative symbol table (prevents Tier-2 capture)
- [contracts/](contracts/) - Frozen contract specifications
- [closures/](closures/) - Tier-1.5 closure registry
