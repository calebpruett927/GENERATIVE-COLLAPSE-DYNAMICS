# Return-Based Canonization

**Version**: 1.0.0  
**Status**: Architectural Principle  
**Axiom Connection**: Direct implementation of "What Returns Through Collapse Is Real"  
**Last Updated**: 2026-01-23

---

## Overview

**Return-based canonization** is the mechanism by which Tier-2 experimental results can be promoted to Tier-1 canonical status. This document formalizes the process and explains how it embodies the core UMCP axiom.

**Core Principle**: One-way dependency flow within a frozen run, with return-based canonization between runs.

**Within-run**: Authority flows in one direction. The frozen interface determines the bounded trace Ψ(t); Tier-1 invariants are computed as functions of that frozen trace; Tier-2 overlays may read Tier-1 outputs but cannot reach upstream to alter the interface, the trace, or the kernel definitions. No back-edges, no retroactive tuning.

**Between-run**: Continuity is never presumed. A new run may exist freely, but it is only "canon-continuous" with a prior run if it returns and welds: the seam has admissible return (no continuity credit in ∞_rec segments), and the κ/IC continuity claim closes under the weld tolerances and identity checks.

**Constitutional Clauses**:
- "Within-run: frozen causes only. Between-run: continuity only by return-weld."
- "Runs are deterministic under /freeze; canon is a graph whose edges require returned seam closure."
- "No back-edges inside a run; no canon claims between runs without welded return."

**Formal Statement**: For any run r with frozen config φ_r and bounded trace Ψ_r(t), Tier-1 kernel K_r(t) := K(Ψ_r(t); φ_r) is invariant to any Tier-2 object. For two runs r₀, r₁, the statement "r₁ canonizes r₀" is admissible iff the seam returns (τ_R finite under policy) and the weld closes (ledger–budget residual within tol + identity check). Otherwise, r₁ is non-canon relative to r₀.

**Key Insight**: The tier system's "NO FEEDBACK" rule is not absolute - it applies **within a frozen run**. Across runs, Tier-2 results can be promoted to Tier-1 **if and only if** they demonstrate return through formal validation.

---

## The Problem

Without return-based canonization, the tier system creates a paradox:

1. **Tier-2** is where exploration happens (new metrics, improved methods, refined models)
2. **Tier-1** is canonical kernel (frozen, deterministic, authoritative)
3. **Problem**: How do discoveries in Tier-2 ever become canonical in Tier-1?

**Without return mechanism**: Tier-1 remains static, innovations stay experimental forever

**With return mechanism**: Tier-2 results that "return" (validate through seam welding) earn canonical status

---

## The Core Axiom

> **"What Returns Through Collapse Is Real"**

Applied to tier promotion:

- **Collapse event** = Validation test (does Tier-2 result meet Tier-1 criteria?)
- **Return** = Seam weld passes (continuity demonstrated with ΔÎº, IC ratio, tolerance)
- **Real** = Canonical status (promoted to Tier-1 in new contract version)
- **Not real** = Remains experimental (stays Tier-2 diagnostic)

**The cycle must complete**:
```
Tier-2 exploration → Validation → Seam weld → Return confirmed → Canonical status
                                              ↓
                                         No return → Remains Tier-2
```

---

## Formal Definition

### Return-Based Canonization Process

**Input**: Tier-2 result M (metric, closure, invariant) discovered in Run N

**Output**: Either (1) M promoted to Tier-1 in Run N+1, or (2) M remains Tier-2

**Process**:

#### Step 1: Threshold Validation

**Question**: Does M meet declared criteria for Tier-1 inclusion?

**Requirements**:
- Range constraints (e.g., M ∈ [0,1], M ≥ 0)
- Stability criteria (std(M) < threshold across traces)
- Computational determinism (same inputs → same M)
- No symbol capture (M doesn't redefine existing Tier-1 symbols)

**Test**: Run validation suite with M computed alongside existing Tier-1 invariants

**Outcome**:
- ✓ PASS: M meets all criteria → Proceed to Step 2
- ✗ FAIL: M violates criteria → Remains Tier-2, no promotion

#### Step 2: Seam Weld Computation

**Question**: Is M continuous with existing Tier-1 canon?

**Computation**:
```
Run N (without M):
  κ₀ = Σ wᵢ ln(cᵢ)
  IC₀ = exp(κ₀)

Run N+1 (with M as Tier-1):
  κ₁ = Σ wᵢ ln(cᵢ) + contribution from M
  IC₁ = exp(κ₁)

Seam weld:
  Δκ = κ₁ - κ₀              (ledger term)
  ratio = IC₁/IC₀            (integrity ratio)
  residual = tol_budget - |Δκ|  (leftover tolerance)
```

**Validation checks**:
1. **Dial consistency**: |ratio - exp(Δκ)| < 10⁻⁹
2. **Tolerance closure**: |residual| ≤ tol_seam
3. **Budget accounting**: Δκ traceable to declared changes

**Outcome**:
- ✓ PASS: Weld validates → M "returned" → Proceed to Step 3
- ✗ FAIL: Weld nonconformant → M didn't return → Remains Tier-2

#### Step 3: Canon Declaration

**Action**: Create new contract version with M as Tier-1 invariant

**Requirements**:
- New contract ID (e.g., `GCD.INTSTACK.v2.yaml`)
- M added to `reserved_symbols`
- M formula added to `identities` or `closures`
- Seam weld receipt attached as provenance
- Changelog documenting promotion rationale

**Result**: M is now canonical for all future runs using new contract

---

## Within-Run vs Cross-Run

### Within a Frozen Run (NO FEEDBACK)

```
Tier-0 (frozen + weld) → Tier-1 (compute) → Tier-2 (diagnostics)
                                                        ✗ NO FEEDBACK
```

**Rule**: Tier-2 cannot alter Tier-0/1 outcomes within the same run

**Rationale**: 
- Prevents narrative rescue ("let's adjust parameters to get better results")
- Ensures determinism (same frozen inputs → same outputs)
- Maintains auditability (no hidden feedback loops)

**Enforcement**: 
- Tier-2 reads from Tier-1, never writes back
- Diagnostics are labeled "Tier-2", never used as gates
- Any parameter change requires new run with new freeze

### Across Runs (RETURN-BASED CANONIZATION)

```
Run N: Tier-2 result M discovered
         ↓ Threshold validation
         ↓ Seam weld computation
         ↓ IF weld PASS (M "returned")
Run N+1: M promoted to Tier-1 in new contract
         ✓ M is now kernel invariant
         ✗ IF weld FAIL: M remains Tier-2
```

**Rule**: Tier-2 results can be promoted to Tier-1 via formal validation between runs

**Rationale**:
- Enables evolution of canonical methods
- Requires formal proof (seam weld) not narrative
- Embodies "cycle must return" - only validated results become real
- Maintains full provenance (promotion is traceable, reversible)

**Critical distinction**: This is NOT feedback - it's canonization through return validation

---

## Worked Example: Fractal Dimension

### Scenario

RCFT (Tier-2) computes fractal dimension D_f for trajectory complexity:

```python
# closures/rcft/fractal_dimension.py
def compute_fractal_dimension(trace):
    """RCFT Tier-2 diagnostic: Trajectory complexity D_f ∈ [1,3]"""
    # Box-counting algorithm implementation
    return D_f
```

**Question**: Should D_f be promoted to Tier-1 (GCD kernel) or remain Tier-2?

### Step 1: Threshold Validation

**Criteria for Tier-1**:
- ✓ Range: D_f ∈ [1,3] (guaranteed by box-counting)
- ✓ Stability: std(D_f) < 0.05 across 100+ traces (measured)
- ✓ Determinism: Same trace → same D_f (algorithm is deterministic)
- ✓ No capture: D_f doesn't redefine ω, F, S, C, κ, IC, τ_R

**Test**: Run GCD validation with D_f computed alongside existing invariants

**Result**: ✓ PASS - D_f meets all criteria

### Step 2: Seam Weld

**Scenario**: Add D_f to log-integrity computation

**Without D_f** (Run N, GCD.INTSTACK.v1):
```
κ₀ = Σ wᵢ ln(cᵢ) = -0.250000
IC₀ = exp(-0.250000) = 0.778801
```

**With D_f** (Run N+1, GCD.INTSTACK.v2):
```
κ₁ = Σ wᵢ ln(cᵢ) + w_D ln(D_f/3)  # Normalize by max dimension
     = -0.250000 + 0.05 × ln(2.1/3)
     = -0.250000 + 0.05 × (-0.356675)
     = -0.267834
IC₁ = exp(-0.267834) = 0.765145
```

**Seam weld computation**:
```
Δκ = κ₁ - κ₀ = -0.267834 - (-0.250000) = -0.017834
ratio = IC₁/IC₀ = 0.765145 / 0.778801 = 0.982464
exp(Δκ) = exp(-0.017834) = 0.982334

Dial check: |0.982464 - 0.982334| = 0.000130 < 10⁻⁶ ✓
Tolerance: tol_seam = 0.005, |residual| = 0.002 < 0.005 ✓
```

**Result**: ✓ PASS - Seam weld validates, D_f "returned"

### Step 3: Canon Declaration

**Action**: Create `contracts/GCD.INTSTACK.v2.yaml`

**Changes**:
```yaml
# contracts/GCD.INTSTACK.v2.yaml
contract_id: GCD.INTSTACK.v2
parent_contract: GCD.INTSTACK.v1
changelog:
  - version: v2
    date: 2026-01-23
    change: "Promoted fractal dimension D_f to Tier-1 invariant"
    rationale: "Seam weld validated continuity (residual=0.002 < tol_seam=0.005)"
    provenance: "receipts/promotion_df_seam_weld.json"

reserved_symbols:
  D_f:
    name: "Fractal Dimension"
    definition: "Trajectory complexity via box-counting"
    range: [1, 3]
    tier: Tier-1
    units: dimensionless

closures:
  fractal_dimension:
    path: closures/rcft/fractal_dimension.py  # Promoted from RCFT
    tier: Tier-1  # Upgraded from Tier-2
    inputs: [trace]
    outputs: {D_f: float}
```

**Seam receipt** (`receipts/promotion_df_seam_weld.json`):
```json
{
  "weld_id": "W-2026-01-23-GCD-DF-PROMOTION",
  "manifest_id": "manifest:gcd_v2_canonization",
  "from_contract": "GCD.INTSTACK.v1",
  "to_contract": "GCD.INTSTACK.v2",
  "promoted_symbol": "D_f",
  "kappa_before": -0.250000,
  "kappa_after": -0.267834,
  "delta_kappa": -0.017834,
  "IC_ratio": 0.982464,
  "dial_consistency": 0.000130,
  "residual": 0.002000,
  "tol_seam": 0.005000,
  "status": "PASS",
  "rationale": "D_f demonstrated stability across 127 traces with std=0.032 < threshold=0.05. Seam weld validates continuity.",
  "sha256": "9a2f83b8...c57e9"
}
```

**Result**: D_f is now Tier-1 canonical in GCD.INTSTACK.v2

---

## Nonconformance Examples

### Example 1: Promotion Without Seam Weld

**Violation**:
```yaml
# contracts/GCD.INTSTACK.v2.yaml
reserved_symbols:
  D_f:
    name: "Fractal Dimension"
    tier: Tier-1

# NO seam weld receipt
# NO provenance documentation
# NO tolerance validation
```

**Consequence**: AUTOMATIC NONCONFORMANCE

**Rationale**: Promotion without demonstrated return violates axiom. "D_f is canonical" requires proof via seam weld, not assertion.

### Example 2: Failing Weld, Claimed Promotion

**Scenario**: Seam weld computation:
```
Δκ = -0.350000  (large drift)
residual = -0.345  (outside tolerance)
tol_seam = 0.005
|residual| = 0.345 > 0.005  ✗ FAIL
```

**Claim**: "We promoted D_f anyway because it's better"

**Consequence**: AUTOMATIC NONCONFORMANCE

**Rationale**: Weld failed → D_f didn't "return" → not canonical. If you want D_f canonical, either:
1. Increase tolerance budget (requires justification, new contract)
2. Refine D_f computation to reduce Δκ
3. Accept D_f remains Tier-2 diagnostic

No narrative rescue allowed.

### Example 3: Ad-Hoc Within-Run Feedback

**Violation**:
```python
# Inside Tier-1 kernel computation
def compute_omega(trace):
    omega = sum(w[i] * (1 - trace[i]) for i in range(n))
    
    # Check Tier-2 diagnostic
    D_f = compute_fractal_dimension(trace)  # Tier-2 function
    
    # Adjust omega based on Tier-2 result
    if D_f > 2.5:  # "Complex trajectory"
        omega *= 1.1  # "Increase drift for complex systems"
    
    return omega
```

**Consequence**: AUTOMATIC NONCONFORMANCE

**Rationale**: 
- Tier-2 (D_f) fed back into Tier-1 (ω) within frozen run
- Breaks determinism (ω now depends on Tier-2 diagnostic)
- Enables narrative rescue ("let's adjust based on complexity")
- Violates tier separation

**Correct approach**: 
1. Run N: Compute ω without D_f adjustment
2. Analyze: Does D_f correlation suggest ω adjustment needed?
3. Propose: New formula ω' incorporating D_f
4. Validate: Seam weld between ω and ω'
5. If weld passes: Promote ω' to Tier-1 in new contract
6. Run N+1: Use new contract with ω' as canonical

---

## Implementation Checklist

### For Tier-2 Developers

When you discover a promising metric/method M:

- [ ] **Document**: Write clear definition, range, formula for M
- [ ] **Test stability**: Compute M across diverse traces, measure std(M)
- [ ] **Check determinism**: Same trace → same M always
- [ ] **Avoid capture**: Ensure M doesn't redefine Tier-1 symbols
- [ ] **Compute seam weld**: Run with/without M, calculate Δκ, IC ratio
- [ ] **Evaluate residual**: Is |residual| ≤ tol_seam?
- [ ] **Document provenance**: Why is M useful? What does it measure?
- [ ] **Prepare proposal**: Draft new contract with M as Tier-1

### For Contract Maintainers

When evaluating promotion proposal:

- [ ] **Review threshold validation**: Does M meet all Tier-1 criteria?
- [ ] **Verify seam weld**: Is weld computation correct and passing?
- [ ] **Check dial consistency**: |ratio - exp(Δκ)| < 10⁻⁹?
- [ ] **Validate tolerance**: |residual| ≤ tol_seam with margin?
- [ ] **Assess provenance**: Is rationale clear and justified?
- [ ] **Test integration**: Do existing tests pass with M promoted?
- [ ] **Update documentation**: Changelog, symbol index, tier tables?
- [ ] **Archive receipts**: Seam weld receipt in version control?

### For Validators

When auditing claimed promotions:

- [ ] **Locate seam receipt**: Is there a weld receipt for this promotion?
- [ ] **Verify computation**: Can you reproduce Δκ, IC ratio from receipt?
- [ ] **Check PASS status**: Did weld actually pass tolerance check?
- [ ] **Trace provenance**: Is promotion rationale documented?
- [ ] **Test reversibility**: Can you downgrade to previous contract?
- [ ] **Confirm no within-run feedback**: Is promoted symbol only used in declared tier?

---

## Relation to Other Concepts

### Connection to Seam Accounting (Tier-0 Protocol)

Return-based canonization **is** seam accounting applied to tier promotion:

- **Seam accounting**: Validates continuity across parameter changes
- **Return-based canonization**: Validates continuity across tier changes
- **Both use**: Δκ, IC ratio, tolerance budgets
- **Both require**: Frozen closures, explicit provenance
- **Both embody**: "Cycle must return" axiom

### Connection to Contract Versioning

Every promotion creates a new contract version:

- **Minor version** (v1.0 → v1.1): Promoted symbol is additive (doesn't change existing)
- **Major version** (v1 → v2): Promoted symbol modifies existing (e.g., changes ω formula)
- **Patch version** (v1.0.1 → v1.0.2): Documentation only, no promotions

### Connection to Closure Registry

Promoted closures must be in registry:

```yaml
# closures/registry.yaml
fractal_dimension:
  version: v1
  path: closures/rcft/fractal_dimension.py
  tier: Tier-1  # Promoted from Tier-2
  promotion_date: 2026-01-23
  promotion_receipt: receipts/promotion_df_seam_weld.json
```

Registry tracks promotion history, enables audit trail.

---

## Future Work

### Automated Promotion Tools

- [ ] `umcp promote` CLI command:
  - Runs threshold validation automatically
  - Computes seam weld with logging
  - Generates draft contract + receipt
  - Outputs PASS/FAIL recommendation

### Promotion Test Suite

- [ ] Test promotion workflow end-to-end
- [ ] Test nonconformance detection (missing weld, failing weld)
- [ ] Test contract versioning correctness
- [ ] Test registry updates on promotion

### Promotion Policy Documentation

- [ ] Guidelines for when promotion is appropriate
- [ ] Threshold recommendations for different symbol types
- [ ] Tolerance budget allocation strategies
- [ ] Downgrade/deprecation procedures

---

## Summary

**Return-based canonization** operationalizes the core UMCP axiom at the architectural level:

1. **Tier-2** is hypothesis space (exploration, experimentation)
2. **Validation** is collapse event (does hypothesis meet criteria?)
3. **Seam weld** is return test (does it maintain continuity?)
4. **Tier-1 promotion** is canonization (what returned becomes real)

**Key insight**: The "NO FEEDBACK" rule prevents within-run feedback, but enables cross-run canonization through formal return validation.

**Critical requirement**: **The cycle must return or it's not real.** Results that don't pass seam welding remain non-canonical, regardless of their apparent utility.

This architecture ensures:
- ✓ Tier-1 evolves based on evidence, not narrative
- ✓ All promotions are auditable and reversible
- ✓ Continuity is formally proven, not asserted
- ✓ System remains falsifiable (bad promotions can be detected and downgraded)

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and philosophical foundation
- [TIER_SYSTEM.md](TIER_SYSTEM.md) - Complete tier system specification
- [PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md) - Seam accounting and weld receipts
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - Tier-1 invariants and lemmas
