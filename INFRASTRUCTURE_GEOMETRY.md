# Infrastructure Geometry

**Version**: 1.0.0  
**Status**: Architectural Foundation  
**Protocol Infrastructure**: [Tier System](TIER_SYSTEM.md) | [Kernel Specification](KERNEL_SPECIFICATION.md) | [Axiom](AXIOM.md)  
**Last Updated**: 2026-01-23

---

## Preamble: Geometry as Operational Meaning

The "geometry of infrastructure" in UMCP is not a metaphor layered on top of the protocol—**it is the operational meaning of the stack**. The infrastructure is literally constructed from geometric objects: state spaces, trajectories, projections, partitions, and certified transitions. Each tier performs a specific geometric operation, and together they enforce a discipline that makes continuity, comparability, and change-control **auditable rather than asserted**.

**Core Principle**: One-way dependency flow within a frozen run, with return-based canonization between runs.

**What the infrastructure holds** is summarized in one tight formulation:

> A portable notion of **state** (the contract-bound Ψ), a portable notion of **geometry** (declared distance + return tolerance + return domain), a portable **coordinate system** (the Tier-1 ledger as projections), a portable **partitioning** (regime gates as closures unless proven portable), and a portable **continuity test** (weld closure with residual and identity checks). That entire package is designed to make cross-domain transfer an **empirical property** (demonstrated by reproducible seam closure), not an assumption.

**Constitutional Clauses** (geometry-specific formulations):
- **Within-run**: Frozen causes only—the geometry declared at Tier-0 determines all downstream computations; no back-edges.
- **Between-run**: Continuity only by return-weld—transitions are certified only when the seam closes under declared budget and tolerance.

**Formal Statement**: For any run r with frozen config φ_r and bounded trace Ψ_r(t), Tier-1 kernel K_r(t) := K(Ψ_r(t); φ_r) is invariant to any Tier-2 object. For two runs r₀, r₁, the statement "r₁ canonizes r₀" is admissible iff the seam returns (τ_R finite under policy) and the weld closes (ledger–budget residual within tol + identity check). Otherwise, r₁ is non-canon relative to r₀.

---

## 1. Geometric Foundation: The Three-Layer Architecture

The infrastructure's geometry has three coupled layers, each performing a distinct geometric job:

```
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 3: SEAM GRAPH (Continuity Certification)        │
│     Transitions certified only when weld accounting closes      │
│     Residual s = Δκ_budget - Δκ_ledger within tolerance         │
├─────────────────────────────────────────────────────────────────┤
│           LAYER 2: INVARIANT COORDINATES (Projections)          │
│     Ledger {ω, F, S, C, τ_R, κ, IC} as multi-projection         │
│     Regime gates partition the projection space                  │
├─────────────────────────────────────────────────────────────────┤
│           LAYER 1: STATE SPACE (Manifold + Distance)             │
│     Ψ(t) ∈ [0,1]ⁿ with declared norm ‖·‖ and tolerance η       │
│     Return = re-entry to η-balls under declared geometry        │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: The State-Space Layer

**Domain**: `[0,1]ⁿ` (or "manifold then normalize")

**Objects**:
- Trajectory: Ψ(t) as time-evolution in bounded state space
- Distance: Declared norm ‖·‖ (e.g., L² under `norms.l2_eta1e-3.v1`)
- Tolerance: η-neighborhoods defining "closeness"
- Return domain: D_θ(t) specifying admissible reference states

**Geometric Operation**: The return detector is genuinely geometric. Prior states define η-balls under the declared norm, and a return occurs when the trajectory re-enters the union of those balls within the declared return domain:

```
Return-candidate set:
U_θ(t) := {u ∈ D_θ(t) : ‖Ψ_ε(t) - Ψ_ε(u)‖ ≤ η}

Return time (first-hitting):
τ_R(t) := min{t - u : u ∈ U_θ(t)}   if U_θ(t) ≠ ∅
          ∞_rec                       otherwise
```

**Critical Insight**: τ_R is not a narrative flourish—it is a **first-hitting-time computation** defined by:
- The norm you freeze (geometry)
- The tolerance you freeze (neighborhood size)
- The cadence you freeze (sampling resolution)
- The horizon you freeze (H_rec)
- The domain generator you freeze (admissibility rule)

**What This Layer Holds**: The meaning of "closeness" and "return" as computable geometric properties.

---

### Layer 2: The Invariant-Coordinate Layer

**Domain**: Projection space for interpretation and governance

**Objects**:
- Ledger: {ω, F, S, C, τ_R, κ, IC} as multi-dimensional coordinates
- Axes: ω/F (drift–fidelity), S/C (entropy–curvature), τ_R (recurrence timing), κ/IC (integrity)
- Gates: Regime predicates {Stable, Watch, Collapse}

**Geometric Operation**: The ledger is explicitly positioned as **multi-projection**—each invariant is a projection of the full state trajectory into an interpretable coordinate:

| Projection | Geometric Interpretation | Axis Role |
|------------|--------------------------|-----------|
| ω/F | Drift–fidelity axis | Distance from ideal performance |
| S | Entropy proxy | Spread/uncertainty in normalized state |
| C | Curvature proxy | Shape dispersion in state coordinates |
| τ_R | Return time | Recurrence timing under declared geometry |
| κ/IC | Log-integrity / Integrity composite | Additive/multiplicative coordinate for seam accounting |

**The Stability Region as Preimage**:

The "Stable" regime is defined as the **preimage** of the full gate predicate:

```
Stable = {Ψ(t) : ω(t) ≤ ω_stable ∧ C(t) ≤ C_stable ∧ τ_R(t) < ∞_rec ∧ ...}
```

**Critical Discipline**: Any 2D view (e.g., plotting ω vs C) is **illustrative only**. The Stable region is not a rectangle in (ω, C)—it is a preimage of conjunctive gates across all invariants. **Low-dimensional stability never certifies PASS.**

**What This Layer Holds**: Interpretable coordinates for governance, with the discipline that visualization ≠ certification.

---

### Layer 3: The Seam-Graph Layer

**Domain**: Transition graph over time points {t₀ → t₁}

**Objects**:
- Seams: Candidate transitions between time points
- Ledger change: Δκ_ledger = κ(t₁) - κ(t₀) = ln(IC₁/IC₀)
- Budget model: Δκ_budget = R·τ_R - (D_ω + D_C)
- Residual: s = Δκ_budget - Δκ_ledger
- Weld gate: PASS/FAIL under tolerance

**Geometric Operation**: Seam certification is where the infrastructure "holds structure" in the strongest sense. A seam is certified **only when** the weld accounting closes:

```
Weld PASS requires ALL of:
1. τ_R(t₁) is finite (not ∞_rec, not UNIDENTIFIABLE)
2. |s| ≤ tol_seam (residual within tolerance)
3. |exp(Δκ) - IC₁/IC₀| < 10⁻⁹ (identity closure)
```

**Why This Prevents Narrative Rescue**: The residual s is the mechanism that **exposes silent embedding/closure drift** instead of letting it get absorbed into story. If your claimed continuity produces a residual outside tolerance, the infrastructure rejects it—regardless of how compelling the interpretation.

**What This Layer Holds**: The geometric constraint that transitions must close under accounting, not assertion.

---

## 2. Tier-Layer Correspondence

The three geometric layers map directly to the tier system:

| Tier | Layer | Geometric Role | Artifacts |
|------|-------|----------------|-----------|
| **Tier-0** | State-Space Setup | Declare the manifold, embedding, distance, tolerance, domain | `contract.yaml`, `embedding.yaml`, `trace.csv` |
| **Tier-1** | Invariant Computation | Compute projections deterministically from frozen state | `invariants.csv` |
| **Tier-0** | Seam Certification | Verify weld closure under frozen closures | `welds.csv`, `closures.yaml`, `ss1m_receipt.json` |
| **Tier-2** | Overlay (Subordinate) | Interpret without authority to override geometry | `diagnostics.csv`, `interpretation.md` |

**Flow** (geometry propagates downward, never upward within a run):

```
Tier-0: Freeze State-Space Geometry
        (manifold, distance, tolerance, domain, embedding)
              ↓
Tier-1: Compute Invariant Projections
        (ω, F, S, C, κ, IC, τ_R, regime)
              ↓
Tier-0: Evaluate Seam Geometry (protocol)
        (Δκ_ledger, Δκ_budget, residual s, PASS/FAIL)
              ↓
Tier-2: Interpret (subordinate, non-authoritative)
        (diagnostics, models, narrative)
              ↓
        NO FEEDBACK TO UPPER TIERS ✗
```

---

## 3. Geometric Discipline: What Each Declaration Constrains

### 3.1 Embedding Declaration (Tier-0)

**What you freeze**: The map N_K : x(t) → Ψ(t) ∈ [0,1]ⁿ

**Geometric consequence**: You have declared the **manifold** on which trajectories evolve. Changing the embedding changes the geometry—every downstream computation inherits this choice.

**Auditability requirement**: A reader must be able to reconstruct Ψ(t) from stated units, conversions, bounds, and clipping rules. Under-specification breaks geometry.

### 3.2 Norm/Tolerance Declaration (Tier-0)

**What you freeze**: ‖·‖ (norm), η (tolerance), H_rec (horizon)

**Geometric consequence**: You have declared the **neighborhood structure**. The return detector operates on η-balls under this norm. Different norms produce different return times for the same trajectory.

**Example closures**:
- `norms.l2_eta1e-3.v1.yaml`: L² norm with η = 10⁻³
- Custom norms must be frozen as closures with explicit registry

### 3.3 Return Domain Declaration (Tier-0)

**What you freeze**: D_θ(t) ⊆ {0, 1, ..., t-1}

**Geometric consequence**: You have declared which **past states are admissible references**. The domain generator prevents false returns from stale references, forbidden segments, or missingness-dominated rows.

**Example closures**:
- `return_domain.window64.v1.yaml`: Sliding window of 64 prior samples

### 3.4 Closure Registry (Tier-0)

**What you freeze**: Γ form, R estimator, D_C definition, tolerances

**Geometric consequence**: You have declared the **budget model** for seam accounting. The weld predicate evaluates whether the realized ledger change is reconcilable with the declared budget.

---

## 4. The Seam as Geometric Closure Test

### 4.1 Accounting Identity

For a candidate seam (t₀ → t₁):

```
Ledger Change (what happened):
    Δκ_ledger = κ(t₁) - κ(t₀) = ln(IC(t₁)/IC(t₀))
    i_r = IC(t₁)/IC(t₀)

Budget Change (what was modeled):
    Δκ_budget = R · τ_R(t₁) - (D_ω(t₁) + D_C(t₁))

Residual (discrepancy):
    s = Δκ_budget - Δκ_ledger
```

### 4.2 The Three Closure Checks

**Check 1: Return Evaluability**
```
τ_R(t₁) must be finite
τ_R = ∞_rec → seam FAIL (no return, no credit)
τ_R = UNIDENTIFIABLE → seam NOT_WELDABLE (cannot evaluate)
```

**Check 2: Residual Tolerance**
```
|s| ≤ tol_seam → PASS
|s| > tol_seam → FAIL (budget model does not close)
```

**Check 3: Identity Consistency**
```
|exp(Δκ_ledger) - i_r| < 10⁻⁹ → algebraically consistent
violation → numerical or convention error
```

### 4.3 Typed Censoring as Geometric Constraint

The rule `no_return_no_credit: true` is a geometric constraint:

```yaml
typed_censoring:
  no_return_no_credit: true
  # If τ_R = ∞_rec, the return credit term R·τ_R is set to 0
  # Seam cannot pass without demonstrated return
```

This prevents "infinite credit" artifacts and enforces the axiom: **"What returns through collapse is real."**

---

## 5. What the Infrastructure Holds (Summary)

### 5.1 Portable State

**The contract-bound Ψ(t)**: Once the contract is frozen, the studied object is treated as a trajectory on the declared state space `[0,1]ⁿ`. This is portable because:
- Units are declared (Tier-0)
- Embedding is frozen (Tier-0)
- Clipping policy is explicit (Tier-0)
- Trace is reproducible from raw + pipeline (hashes stored)

### 5.2 Portable Geometry

**Declared distance + return tolerance + return domain**: The meaning of "close" and "return" is portable because:
- Norm is frozen (closure)
- Tolerance η is frozen (contract)
- Domain generator D_θ is frozen (closure)
- Horizon H_rec is frozen (contract)

### 5.3 Portable Coordinates

**The Tier-1 ledger as projections**: Invariants {ω, F, S, C, τ_R, κ, IC} are portable coordinates because:
- Formulas are canonical (Kernel Specification)
- Weights are frozen (Tier-0)
- Curvature convention is declared (closure if non-default)
- Determinism guarantee (same inputs → same outputs)

### 5.4 Portable Partitioning

**Regime gates as closure-controlled**: Partitions {Stable, Watch, Collapse} are portable because:
- Gate thresholds are frozen (contract)
- Gate predicates are conjunctive over invariants
- Override requires closure declaration

### 5.5 Portable Continuity Test

**Weld closure with residual and identity checks**: Continuity claims are portable because:
- Budget model is frozen (closures)
- Tolerance is frozen (contract)
- Residual is computed, not asserted
- Identity check prevents numerical drift

---

## 6. Cross-Domain Transfer as Empirical Property

The entire geometric package is designed to make **cross-domain transfer an empirical property**, not an assumption.

### 6.1 The Claim

> Two domains share structure if and only if seam welds close between them under the same frozen contract.

### 6.2 The Test

```
Domain A: Produces trajectory Ψ_A(t), invariants, receipt
Domain B: Produces trajectory Ψ_B(t), invariants, receipt

Cross-domain seam (A → B):
  Δκ_ledger = κ_B - κ_A
  Δκ_budget = (budget under shared closures)
  s = Δκ_budget - Δκ_ledger

PASS ⟺ |s| ≤ tol_seam ∧ τ_R finite ∧ identity check passes
```

### 6.3 The Enforcement

If the seam does not close:
- Continuity claim is **FAIL**
- The infrastructure explicitly rejects the transfer
- This is a **feature**, not a limitation—it prevents unfounded generalization

---

## 7. Schematic: Infrastructure Geometry Flow

```
                      ┌──────────────────────────────┐
                      │      RAW OBSERVATIONS        │
                      │        x(t) with units       │
                      └──────────────┬───────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │         TIER-0: FREEZE          │
                    │  • Embedding N_K → Ψ(t) ∈ [0,1]ⁿ│
                    │  • Norm ‖·‖, tolerance η        │
                    │  • Domain D_θ(t), horizon H_rec │
                    │  • Weights w_i, ε-clipping      │
                    └────────────────┬────────────────┘
                                     │
        ╔════════════════════════════▼════════════════════════════╗
        ║           LAYER 1: STATE-SPACE GEOMETRY                  ║
        ║  Trajectory Ψ(t) evolves on bounded manifold [0,1]ⁿ     ║
        ║  Return = re-entry to η-balls under declared norm        ║
        ║  τ_R = first-hitting-time to neighborhood union          ║
        ╚════════════════════════════╤════════════════════════════╝
                                     │
                    ┌────────────────▼────────────────┐
                    │   TIER-1: KERNEL COMPUTATION    │
                    │  Deterministic projections:     │
                    │  ω, F, S, C, τ_R, κ, IC, regime │
                    └────────────────┬────────────────┘
                                     │
        ╔════════════════════════════▼════════════════════════════╗
        ║         LAYER 2: INVARIANT-COORDINATE GEOMETRY           ║
        ║  Multi-projection: ω/F (drift), S (entropy), C (shape)  ║
        ║  τ_R (recurrence timing), κ/IC (integrity accounting)   ║
        ║  Regime = preimage of gate predicate (not 2D rectangle) ║
        ╚════════════════════════════╤════════════════════════════╝
                                     │
                    ┌────────────────▼────────────────┐
                    │ TIER-0: SEAM CALCULUS (PROTOCOL) │
                    │  Freeze closures (Γ, R, D_C)    │
                    │  Compute Δκ_ledger, Δκ_budget, s│
                    │  Evaluate weld PASS/FAIL        │
                    └────────────────┬────────────────┘
                                     │
        ╔════════════════════════════▼════════════════════════════╗
        ║            LAYER 3: SEAM-GRAPH GEOMETRY                  ║
        ║  Transitions (t₀ → t₁) form a directed graph            ║
        ║  Edge exists ⟺ weld PASS (accounting closes)            ║
        ║  Residual s exposes silent drift (prevents narrative)   ║
        ╚════════════════════════════╤════════════════════════════╝
                                     │
                    ┌────────────────▼────────────────┐
                    │      TIER-2: OVERLAYS           │
                    │  Diagnostics, models, narrative │
                    │  SUBORDINATE: cannot override   │
                    │  geometry or weld outcomes      │
                    └─────────────────────────────────┘
```

---

## 8. Symbol Discipline as Geometric Governance

Symbol discipline is non-negotiable governance because **symbols are coordinates**:

| Symbol | Coordinate Role | Capture Consequence |
|--------|-----------------|---------------------|
| ω | Drift coordinate | Corrupts distance-from-ideal axis |
| F | Fidelity coordinate | Corrupts performance projection |
| S | Entropy coordinate | Corrupts uncertainty measurement |
| C | Curvature coordinate | Corrupts shape dispersion |
| τ_R | Return-time coordinate | Corrupts recurrence geometry |
| κ | Log-integrity coordinate | Breaks seam accounting arithmetic |
| IC | Integrity coordinate | Breaks multiplicative closure |

**Rule**: Tier-1 reserved symbols cannot be repurposed without breaking auditability—i.e., **corrupting the coordinate system**.

---

## 9. Axiom Embodiment in Geometry

The core axiom—**"Collapse is generative; only what returns is real"**—is geometrically embodied:

| Axiom Element | Geometric Realization |
|---------------|----------------------|
| **Collapse** | Trajectory enters Collapse regime (gate predicate satisfied) |
| **Return** | Trajectory re-enters η-neighborhood of prior state (τ_R finite) |
| **Real** | Seam weld PASS (accounting closes, continuity certified) |
| **Not real** | τ_R = ∞_rec or seam FAIL (no certificate issued) |

**The geometry enforces the axiom**: You cannot claim continuity without demonstrating return through the declared neighborhood structure, and you cannot certify continuity without weld closure under the declared budget model.

---

## 10. Conclusion: The Infrastructure as Geometric Discipline

The UMCP infrastructure is not a metaphor—it is a **concrete geometric system** that:

1. **Declares state spaces** with explicit embedding, distance, and tolerance
2. **Computes trajectories** as evolutions in those spaces
3. **Projects to invariant coordinates** for interpretation and governance
4. **Partitions coordinates** into regimes via gate predicates
5. **Certifies transitions** only when seam accounting closes

The "structure it holds" is:
- **Continuity** (seams close or they don't)
- **Comparability** (same geometry → comparable invariants)
- **Change-control** (any change is attributable to a tier)

This is the operational meaning of the stack: **geometry as governance**, not geometry as metaphor.

---

## References

- [TIER_SYSTEM.md](TIER_SYSTEM.md): Tier definitions and dependency rules
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md): Formal invariant definitions and lemmas
- [AXIOM.md](AXIOM.md): The core axiom and its operational implementation
- [RETURN_BASED_CANONIZATION.md](RETURN_BASED_CANONIZATION.md): Promotion mechanism through return
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md): File interconnection map
