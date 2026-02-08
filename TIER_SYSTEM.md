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
| **0** | **Protocol** | The validation machinery: regime gates, contracts, schemas, diagnostics, seam calculus, SHA256 integrity, three-valued verdicts |
| **1** | **Immutable Invariants** | The structural identities of collapse: F + ω = 1, IC ≤ F, IC ≈ exp(κ). Discovered, not imposed. |
| **2** | **Expansion Space** | Domain closures that map physics into the invariant structure. Validated through Tier-0 against Tier-1. |

**Tier-1** is immutable because it describes the **structure of collapse itself** — not because someone chose convenient equations. F + ω = 1 is the structural fact that fidelity and drift are complements. IC ≤ F is AM-GM. These hold across 146 experiments in 7 domains because the structure forces them.

**Tier-0** is the protocol that makes Tier-1 actionable — regime gates, validator, diagnostics (kinematics, PHYS-04), seam calculus, contract enforcement, integrity verification. Everything that tests, confirms, and enforces. The seam is where consistency is verified: the same frozen rules must govern both sides of collapse-return.

**Tier-2** is the space where expansions live with validity checks. Domain closures map raw measurements into the Tier-1 invariants. Every expansion is validated *through* Tier-0 *against* Tier-1.

---

## The Stack

```
Tier-1   IMMUTABLE INVARIANTS
         F + ω = 1  |  IC ≤ F  |  IC ≈ exp(κ)
         Structure of collapse. Discovered, not imposed.
         ──────────────────────────────────────────────────

Tier-0   PROTOCOL
         Regime gates, validator, diagnostics, seam calculus,
         contracts, schemas, SHA256 integrity, verdicts.
         Makes Tier-1 actionable. Tests everything.
         ──────────────────────────────────────────────────

Tier-2   EXPANSION SPACE
         Domain closures: NUC, QM, ASTRO, FIN, SEC, WEYL, ...
         Map physics into invariant structure.
         Validated through Tier-0 against Tier-1.
         ✗ NO FEEDBACK to Tier-1 or Tier-0 within a frozen run
```

**One-way dependency**: Tier-2 reads Tier-1 outputs through Tier-0 but cannot alter them. No back-edges within a frozen run.

**Return-based canonization across runs**: Tier-2 results can be promoted to Tier-1 canon ONLY through formal validation — threshold check → seam weld → new contract version. "The cycle must return or it's not real."

---

## Tier-1: Immutable Invariants

**What it is**: The structural identities of collapse. These are constraints discovered by computing F and ω independently from raw data across 146 experiments and finding they hold universally. They are not definitions we chose. They are what the structure forces.

### Structural Identities (0 violations across 146 experiments)

| Identity | Structural Meaning |
|----------|-------------------|
| **F = 1 − ω** | Fidelity and drift are complements — what isn't lost to drift IS fidelity. No third option. |
| **IC ≤ F** | Coherence cannot exceed fidelity (AM-GM bound). A system cannot be more coherent than it is faithful. |
| **IC ≈ exp(κ)** | Coherence tracks log-integrity exponentially. |

### Reserved Symbols (Immutable Meanings)

| Symbol | Name | Formula | Range | Structural Role |
|--------|------|---------|-------|-----------------|
| **F** | Fidelity | F = Σ wᵢcᵢ | [0,1] | How much of the signal survives collapse |
| **ω** | Drift | ω = 1 − F | [0,1] | How much is lost to collapse |
| **S** | Entropy | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] | ≥0 | Internal configurational complexity |
| **C** | Curvature | C = stddev(cᵢ)/0.5 | [0,1] | Coupling to uncontrolled external degrees of freedom |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | ≤0 | Logarithmic fidelity (sensitivity-aware) |
| **IC** | Integrity composite | IC = exp(κ) | (0,1] | Multiplicative coherence |
| **τ_R** | Return time | Re-entry delay to D_θ | ℕ∪{∞_rec} | How long until the system returns to its domain |
| **regime** | Regime label | Gates on (ω,F,S,C) | {Stable, Watch, Collapse} | Which structural phase the system occupies |

### What Tier-1 Is NOT

- **Not "the math we picked"**: The identities describe structure discovered across domains, not equations selected for convenience.
- **Not computation**: Computation is Tier-0's job. Tier-1 is the structure that computation must find.
- **Not a model of the world**: Tier-1 makes no domain claims. It says fidelity and drift are complements; it does not say what fidelity *means* for a nuclide versus a star versus a market. That meaning-mapping is Tier-2.
- **Not "constants"**: The frozen parameters (ε, p, α, λ, tol_seam) are not constant because someone chose them arbitrarily. They are **consistent** across the seam — the same rules on both sides of collapse-return. Freezing is a seam demand, not a design preference.

---

## Tier-0: Protocol

**What it is**: Everything that makes Tier-1 actionable and testable. The protocol includes the validator, regime gates, diagnostics, seam calculus, contract enforcement, schema validation, and SHA256 integrity checks. Tier-0 produces the bounded trace Ψ(t), computes the Tier-1 invariants, and issues verdicts.

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

**What it is**: The space where domain expansions live with validity checks. Domain closures map raw measurements (binding energies, particle masses, stellar luminosities, market returns, decay half-lives) into the Tier-1 invariant structure. Every expansion is validated through Tier-0 against Tier-1.

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

- ✅ **Tier-1**: Immutable invariants defined and verified
  - Structural identities: F+ω=1, IC≤F — 146/146 hold
  - Reserved symbols: F, ω, S, C, κ, IC, τ_R

- ✅ **Tier-0**: Full protocol support
  - Validator: [src/umcp/validator.py](src/umcp/validator.py)
  - CLI: [src/umcp/cli.py](src/umcp/cli.py) (`umcp validate`)
  - Regime gate computation, three-valued verdicts
  - Diagnostics: Kinematics 5/5 Stable, PHYS-04 perfect fixed point
  - Seam structure defined; closures implemented
  - ⚠️ Automated weld PASS/FAIL computation not yet integrated into CLI

- ✅ **Tier-2**: 6 domain expansions active (129 domain experiments + 17 calibration = 146 total)
  - Finance (12, 58%), Security (25, 40%), Astronomy (28, 25%)
  - Nuclear (30, 23%), Quantum (30, 23%), Weyl (4, 0%)

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and operational definitions
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - Formal mathematical specification of Tier-1 invariants
- [UHMP.md](UHMP.md) - Universal Hash Manifest Protocol (identity governance)
- [FACE_POLICY.md](FACE_POLICY.md) - Boundary governance (Tier-0 admissibility)
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - Authoritative symbol table (prevents Tier-2 capture)
- [contracts/](contracts/) - Frozen contract specifications
- [closures/](closures/) - Domain expansion closures (Tier-2) and seam closures (Tier-0)
