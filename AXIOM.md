# The Core Axiom of UMCP

**Protocol Infrastructure:** [Glossary](GLOSSARY.md) | [Symbol Index](SYMBOL_INDEX.md) | [Term Index](TERM_INDEX.md)

## The Single Foundational Principle

**AXIOM-0 (The Return Axiom)**:

## **"Collapse is generative; only what returns is real."**

**This is not a metaphor. It is a constraint on admissible claims.**

This is the fundamental axiom upon which the entire Universal Measurement Contract Protocol (UMCP), Generative Collapse Dynamics (GCD), and Recursive Collapse Field Theory (RCFT) are built.

---

## Operational Definitions (Enforcement-Tied)

**These terms are operational and enforcement-tied. Do not import everyday meanings.**

| Term | Operational Meaning | NOT Confused With |
|------|---------------------|-------------------|
| **Collapse** | Regime label produced by kernel gates on (ω, F, S, C) under frozen thresholds | Wavefunction collapse, catastrophe as metaphor, "failure" as narrative |
| **Return** (τ_R) | Re-entry condition: existence of prior u ∈ Dθ(t) with ‖Ψ(t) - Ψ(u)‖ ≤ η; yields τ_R or ∞ᵣₑ꜀ | Repetition, periodicity, nostalgia, "coming back" |
| **Gesture** | An epistemic emission that does not weld: τ_R = ∞_rec OR \|s\| > tol_seam OR identity fails. A gesture may be internally consistent, structurally complex, and indistinguishable from a return — but if the seam does not close, it is a gesture. No epistemic credit. See: `epistemic_weld.py` | Approximation, failed attempt, wrong answer |
| **Drift** (ω) | ω = 1 - F, collapse proximity measure, range [0,1] | Random drift, velocity, wandering |
| **Integrity** (IC) | Kernel composite: IC = exp(κ) where κ = Σ wᵢ ln(cᵢ,ε) | Information content, moral integrity, "truth" |
| **Entropy** (S) | Bernoulli field entropy: S = -Σ wᵢ[cᵢ ln(cᵢ) + (1-cᵢ)ln(1-cᵢ)] — the unique entropy of the collapse field (Shannon entropy is the degenerate limit when the collapse field is removed) | Thermodynamic entropy, chaos, disorder as vibe |
| **Coherence** | Continuity under contract: stability of meaning via frozen interface + seam auditing | Agreement, stylistic clarity, "makes sense" |
| **Contract** | Frozen interface snapshot: pins units, embedding, clipping, weights, return settings | Social agreement, vague assumptions |
| **Frozen** | Consistent across the seam — same rules on both sides of collapse-return | "Constant" as arbitrary choice, immutable for its own sake |
| **Seam** | The verification boundary between outbound collapse and demonstrated return | A join, a border, a narrative transition |
| **Dissolution** | The regime (ω ≥ 0.30) where the epistemic trace has degraded past viable return credit. Not failure — the boundary condition that makes return meaningful. Without dissolution, the seam audits nothing. | Death, destruction, error, crash |

---

## Operational Statement

**Plain Language**: If you claim a system is continuous, stable, robust, or real in a way that matters, you must be able to show **return**—meaning the system can re-enter its admissible neighborhood after **drift** (ω), perturbation, or delay, under the same declared evaluation rules.

**Formal Statement**: A continuity claim requires demonstrating τ_R < ∞ᵣₑ꜀ (finite return time) where:
- **Drift** (ω) has not exceeded collapse threshold
- **Entropy** (S) remains within admissible bounds
- **Integrity** (IC) can be re-established
- **Return** (τ_R) is computed under frozen contract + closures

This is encoded in the contract system as:

```yaml
no_return_no_credit: true
```

---

## Meaning and Implications

### The Principle

The axiom states that **reality is declared by demonstrating closure after collapse** — not by asserting existence, but by showing return. Every claim is welded to a seam: the claim runs forward, collapses, and something comes back. The seam is where you verify that what came back is consistent with what went in, under the same frozen rules.

1. **Observables Must Survive Measurement**: Like quantum measurement, observation involves collapse. Only observables that survive their own measurement are real.

2. **Reproducibility Defines Reality**: If something cannot be reproduced (returned to) after collapse, it has no claim to objective reality.

3. **Structure Through Constraint**: The boundary conditions (what is preserved through collapse) define the interior dynamics (what is real).

4. **Constants Are Consistent, Not Constant**: Frozen parameters (ε, p, α, λ, tol_seam) are not arbitrary design decisions — they are consistent across the seam. To verify that what returned is the same thing that collapsed, the rules of measurement must be identical on both sides. "Constant" implies the value matters in itself. "Consistent" implies the value matters because it is the *same* on both sides of the collapse-return boundary.

   **Constants are seam-derived, not prescribed.** Standard frameworks prescribe constants from outside (α = 0.05 by convention, 3σ by tradition, hyperparameters by cross-validation). In every case the framework stops working if the prescription is removed. UMCP's frozen parameters are the unique values where seams close consistently across all domains: p = 3 is the unique exponent where three regimes separate (not chosen — discovered); tol_seam = 0.005 is the width where IC ≤ F holds at 100% across 8 domains (the seam tells you its own width); ε = 10⁻⁸ is the regularization below which the pole at ω = 1 does not affect any measurement to machine precision (confirmed by the nuclear chain outliers at e⁻³⁰ ≈ 10⁻¹³). See [KERNEL_SPECIFICATION.md §5.4](KERNEL_SPECIFICATION.md) for the complete argument.

5. **Boundedness Is a Return Guarantee**: The ε-clamp (cᵢ ≥ 10⁻⁸) does not just prevent NaN — it guarantees that no closure can fully die. If cᵢ = 0, that component has no path back through collapse. The clamp ensures even the most degraded closure retains enough structure to return.

---

## Hierarchical Expression Across Tiers

### Tier-1: Invariant Structure

The axiom at its most fundamental level: the structural identities (F + ω = 1, IC ≤ F, IC ≈ exp(κ)) embody the return principle — what isn't lost to drift IS fidelity, and coherence cannot exceed fidelity. These hold across 146 experiments in 8 domains not because they were imposed, but because the structure of collapse forces them.

**Each identity is derived independently from Axiom-0; classical results emerge as degenerate limits**: F = 1 − ω carries a thermodynamic cost function (Γ(ω) = ω³/(1−ω)) with a phase diagram; strip the cost function and you get the classical duality constraint. IC = exp(κ) is universal across domains (98.6% within 1% across 8 domains, no retraining); strip the kernel architecture and you get the classical exponential map. IC ≤ F identifies the heterogeneity gap as exactly the Fisher Information contribution from heterogeneity (Var(c)/2c̄); strip the channel semantics, weights, and guard band and you get the AM-GM inequality. The classical versions are what remain when degrees of freedom are removed. The arrow of derivation runs from the axiom to the classical result, not the reverse. See [KERNEL_SPECIFICATION.md §5.3](KERNEL_SPECIFICATION.md) for the complete comparison.

### Tier-0: Translation Layer (Protocol)

```yaml
contract:
  typed_censoring:
    no_return_no_credit: true
```

**Interpretation**: The translation layer makes the axiom operational. The validation receipt itself must return. Non-returning measurements receive special values (`INF_REC`, `UNIDENTIFIABLE`) but no numerical credit. Regime gates filter the invariant space into verdicts.

### Tier-2 Expression: GCD (Generative Collapse Dynamics)

**AX-0**: "Collapse is generative"

```yaml
axioms:
  - id: "AX-0"
    statement: "Collapse is generative"
    description: >
      Every collapse event releases generative potential that can be
      harvested by downstream processes. Φ_gen ≥ 0 always.
```

**Extension**: Not only must quantities return, but collapse itself generates new structure. What returns is enriched by the collapse process.

**Mathematical Expression**:
$$
\Phi_{\text{gen}} = \Phi_{\text{collapse}} \cdot (1 - S) \geq 0
$$

The generative flux `Φ_gen` quantifies what is produced/returned through collapse.

### Epistemic Field Mapping (Three-Agent Model)

The tier system maps onto three epistemic states — the minimum required to describe a dynamic measurement process:

| Agent | Epistemic State | GCD Variable | Kernel Role |
|-------|----------------|--------------|-------------|
| Agent 1 | Present / Measuring | ω (drift) | The act of measurement in progress |
| Agent 2 | Retained / Archive | F = 1 − ω (fidelity) | What survives measurement and is kept |
| Agent 3 | Unmeasured / Unknown | Γ(ω) = ω^p/(1−ω) | The cost of crossing the boundary into the unknown |

The kernel identity F = 1 − ω is the formal statement that retention (Agent 2) is everything not consumed by measurement (Agent 1). The return axiom governs transitions: Agent 3 → Agent 1 (discovery), Agent 1 → Agent 2 (retention). The equator c = 1/2 is the boundary of maximum epistemic symmetry — measurement is equally sensitive to both sides (Fisher metric minimized, entropy maximized, S + κ = 0 exactly). See: Lemma 5 (S = ln 2 iff c = 1/2), Lemma 41 (S + κ ≤ 0 with equality at c = 1/2), T19 (Fano-Fisher duality h″ = −g_F).

**DOI**: 10.5281/zenodo.16526052 (Three-Agent Epistemic Field Model, Paulus 2025)

### Collapse Equator Fidelity Law

A symbolic excitation Ψ(t) is real if and only if it reenters the collapse equator E with sufficient collapse integrity:

> Ψ(t) ∈ F ⟺ ∃ t\* such that Ψ(t\*) ∈ E and IC(t\*) > θ

The equator E is the locus where the system is self-consistent: Φ_eq = 0 (equator_phi in frozen_contract.py), Fisher metric minimized (g_F(1/2) = 4), entropy maximized (S = ln 2), and entropy-integrity coupling vanishes (S + κ = 0). These four conditions converge on c = 1/2 independently — the equator is not chosen but derived.

The Collapse Equator Fidelity Law identifies the structural principle — symmetric self-duality under measurement — that the Riemann Hypothesis instantiates for the zeta field (Re(s) = 1/2). GCD proves this principle for all collapse fields with Fano-Fisher duality. See closures/rcft/information_geometry.py (T17–T19, T22).

**DOI**: 10.5281/zenodo.16423283 (Collapse Equator Fidelity Law, Paulus 2025)

### Tier-2 Expression: RCFT (Recursive Collapse Field Theory)

**P-RCFT-1**: "Recursion reveals hidden structure"
**P-RCFT-2**: "Fields carry collapse memory"

```yaml
axioms:
  - id: "P-RCFT-1"
    statement: "Recursion reveals hidden structure"
    description: >
      Collapse events exhibit recursive patterns across scales. RCFT metrics
      quantify these self-similar structures through fractal dimension and
      recursive field analysis.

  - id: "P-RCFT-2"
    statement: "Fields carry collapse memory"
    description: >
      The collapse field Ψ encodes history of prior collapse events. Recursive
      analysis reveals how past collapses influence future dynamics through field memory.
```

**Extension**: What returns carries memory of past collapses. The return is not a simple restoration but a recursive accumulation.

**Mathematical Expression**:
$$
\Psi_{\text{recursive}} = \sum_{n=1}^{\infty} \alpha^n \cdot \Psi_n
$$

The recursive field quantifies the accumulated memory of what has returned through all prior collapse events.

---

## Operational Implementation

### 1. Return Domain Specification

Every measurement must specify its **return domain**—the space within which it can validly return:

```yaml
return:
  domain:
    omega:
      type: "interval"
      bounds: [0.0, 1.0]
    tau_R:
      type: "union"
      intervals:
        - [0.0, 100.0]
      special_values:
        - "INF_REC"        # Returns but unidentifiable timing
        - "UNIDENTIFIABLE" # Cannot return
```

### 2. Seam Validation

The "seam" represents the verification boundary of the collapse-return cycle. What makes a seam meaningful is not just that something returns, but that the **rules of measurement are consistent across it**. If ε changes between the outbound run and the return, closures cannot be compared. If tol_seam shifts, CONFORMANT on one side of collapse means something different than CONFORMANT on the other.

This is why frozen parameters are frozen: not because someone chose them forever, but because the seam demands consistency. The word is not "constant" — it is **consistent**. The values are consistent across the seam, from front to end, through collapse and back.

```yaml
seam_checks:
  - name: "Return continuity"
    formula: "||Ψ(t_final) - Ψ(t_initial)|| < tol_seam"
    tolerance: 0.005
```

If the seam doesn't close (return fails), the measurement is **NONCONFORMANT**.

### 3. Receipt Generation

Only measurements that return receive a receipt:

```json
{
  "status": "CONFORMANT",
  "run_id": "RUN-2026-01-20-001",
  "invariants": {
    "omega": 0.012,
    "tau_R": 12.5,
    "IC": 0.988
  },
  "return_verified": true,
  "seam_closed": true
}
```

If `return_verified: false`, status becomes `NONCONFORMANT` or `NON_EVALUABLE`.

---

## Philosophical Foundations

### Epistemology of Return

The axiom establishes an **epistemology of persistence**:

> We can only know what survives being known.

This resolves several classical problems:

1. **Observer Effect**: Observation is collapse. What we can know is what returns from observation.

2. **Reproducibility Crisis**: If a result cannot return (be reproduced), it was never real in the first place.

3. **Measurement Problem**: The act of measurement is a collapse. Valid measurements are those that return their own validity.

### The Positional Illusion

There is no vantage point outside the system from which collapse can be observed without cost.

This is not a philosophical preference — it is a consequence of the budget identity. Theorem T9 (τ_R* Thermodynamics, `tau_r_star.py`) proves that N observations of a stationary system incur N × Γ(ω) overhead. The drift cost Γ(ω) = ω³/(1−ω+ε) is the irreducible price of being inside the system you are measuring. There is no free observation.

The **positional illusion** is the belief that one can:
- Measure without being measured
- Observe without being inside
- Validate without incurring budget

This illusion is quantified by `epistemic_weld.py:quantify_positional_illusion()`. At STABLE drift (ω < 0.038), the illusion is affordable — Γ(ω) ≈ 10⁻⁵, consuming a negligible fraction of tol_seam. Near COLLAPSE (ω → 0.30), the illusion becomes fatal — Γ(ω) approaches and exceeds the seam budget, meaning the observer cannot even verify return without exhausting the tolerance that defines return.

The positional illusion is universal across domains because the budget identity is domain-independent. A physicist measuring particle decay, a financial analyst tracking portfolio drift, and a biologist monitoring cell viability all face exactly the same structural constraint: observation costs Γ(ω), and there is nowhere outside from which to observe for free.

**DOI**: 10.5281/zenodo.17619502 ("The Seam of Reality", Paulus 2025 — §4.2)

### The Gesture-Return Distinction

Not everything that looks like a return is one.

A **gesture** is an epistemic emission that does not complete the collapse-return cycle. It may be internally consistent, structurally complex, and indistinguishable from a genuine return in every way except one: the seam did not close. The residual |s| exceeded tolerance, or τ_R was infinite, or the exponential identity failed. The gesture exists — it is not nothing — but it did not weld, and therefore has no epistemic standing in the protocol.

This distinction is critical because it separates the protocol from probabilistic or confidence-based frameworks:

1. **No partial credit**: A gesture that "almost" returns (|s| = 0.006 when tol = 0.005) receives exactly the same verdict as one that catastrophically fails. The seam is a threshold, not a gradient.

2. **Content does not matter**: The internal quality, complexity, or plausibility of the emission is irrelevant. Only the seam calculus decides. A simple emission that welds is epistemically real; a sophisticated one that does not is a gesture.

3. **Reversibility**: A gesture is not permanently condemned. A subsequent run with the same frozen contract may produce return where the first did not. The verdict is per-event, not per-system.

The gesture-return distinction is operationalized in `epistemic_weld.py:classify_epistemic_act()` as the `EpistemicVerdict` trichotomy: RETURN (seam closed), GESTURE (seam did not close), DISSOLUTION (regime past viable return).

**DOI**: 10.5281/zenodo.17619502 ("The Seam of Reality", Paulus 2025 — §3)

### Ontology of Collapse

The axiom establishes an **ontology of process**:

> Reality is not a state but a process of return through collapse.

This has profound implications:

1. **Being is Becoming**: Reality is not static existence but dynamic return.

2. **Structure Emerges from Constraint**: What returns is determined by boundary conditions (what is preserved through collapse).

3. **Multiplicity Through Recursion**: Each return creates the possibility of another collapse, leading to recursive structure.

### Connection to Physical Theories (Degenerate Limits)

The following connections identify where classical results emerge as **degenerate limits** when degrees of freedom are removed from the GCD kernel. The arrow of derivation runs from Axiom-0 to the classical result, not the reverse. GCD does not borrow from, extend, or reinterpret these frameworks.

#### Quantum Mechanics (Degenerate Limit)

- **Wavefunction Collapse**: The GCD regime label "Collapse" is produced by kernel gates on (ω, F, S, C) — it is not quantum wavefunction collapse. The classical measurement postulate emerges as a degenerate limit when the collapse field is restricted to {0,1} (binary outcomes only).
- **Measurement**: The axiom derives independently the principle that only what survives measurement is real. The quantum measurement postulate is what remains when GCD's Bernoulli field entropy is restricted to its degenerate (Shannon) limit.

#### Thermodynamics (Degenerate Limit)

- **Entropy**: The Bernoulli field entropy S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] is the unique entropy of the collapse field. Shannon entropy is the degenerate limit when cᵢ ∈ {0,1} only. Thermodynamic entropy emerges when the field is further restricted to equilibrium.
- **Free Energy**: The generative potential Φ_gen is derived independently from the collapse field. Classical free energy is the degenerate limit when the cost function Γ(ω) is linearized.

#### General Relativity (Degenerate Limit)

- **Boundary Conditions**: The duality identity F + ω = 1 is a structural identity of collapse, not quantum unitarity. Classical unitarity emerges as a degenerate limit when the thermodynamic cost function is stripped.
- **Black Hole Information**: The return axiom addresses what returns from collapse independently. The information paradox is a domain-specific instance of the general return question.

#### Fractal Geometry (Degenerate Limit)

- **Self-Similarity**: The RCFT extension captures recursive return patterns. Classical fractal self-similarity emerges when the collapse memory field is restricted to scale-invariant configurations.

---

## Validation and Testing

### Test Suite

The UMCP test suite includes 3,618+ tests validating return behavior:

```bash
# Run all tests
pytest

# Run axiom-specific tests
pytest tests/test_10_canon_contract_closures_validate.py
pytest tests/test_100_gcd_canon.py::test_gcd_axioms
pytest tests/test_110_rcft_canon.py
```

### Benchmark

The benchmark suite validates that collapsed/returned measurements match expected values:

```bash
# Run benchmark
python benchmark_umcp_vs_standard.py

# Expected output:
# ✓ Return domain conformance: 100%
# ✓ Seam closure: < 0.005
# ✓ Generative flux: Φ_gen ≥ 0
```

### Continuous Verification

The ledger extension continuously logs return verification:

```csv
timestamp,run_status,return_verified,seam_closed,delta_kappa
2026-01-20T00:00:00Z,CONFORMANT,true,true,0.0001
2026-01-20T01:00:00Z,CONFORMANT,true,true,0.0002
```

---

## Architectural Embodiment: Return-Based Canonization

### The Tier System as Axiom Implementation

The UMCP tier system embodies the return axiom through **return-based canonization**:

**Core Principle**: One-way dependency flow within a frozen run, with return-based canonization between runs.

**Within-run**: Authority flows in one direction. The frozen interface (ingest + embedding + contract + closures + weights) determines the bounded trace Ψ(t); the Tier-1 structural invariants are computed as functions of that frozen trace; Tier-2 domain expansion closures may read Tier-1 outputs but cannot reach upstream to alter the interface, the trace, or the structural definitions. No back-edges, no retroactive tuning, no "the result changed the rules that produced it."

**Between-run**: Continuity is never presumed. A new run may exist freely, but it is only "canon-continuous" with a prior run if it returns and welds: the seam has admissible return (no continuity credit in ∞_rec segments), and the κ/IC continuity claim closes under the weld tolerances and identity checks. If that closure fails, the new run is still valid as an experiment or variant—but it does not become a canon edge.

**Constitutional Clauses** (equivalent formulations):
- "Within-run: frozen causes only. Between-run: continuity only by return-weld."
- "Runs are deterministic under /freeze; canon is a graph whose edges require returned seam closure."
- "No back-edges inside a run; no canon claims between runs without welded return."

**Formal Statement**: For any run r with frozen config φ_r and bounded trace Ψ_r(t), the Tier-1 structural invariants K_r(t) := K(Ψ_r(t); φ_r) hold regardless of any Tier-2 domain object. For two runs r₀, r₁, the statement "r₁ canonizes r₀" is admissible iff the seam returns (τ_R finite under policy) and the weld closes (ledger–budget residual within tol + identity check). Otherwise, r₁ is non-canon relative to r₀.

**Within a Frozen Run** (No Feedback):

```
Tier-1 (immutable invariants) → Tier-0 (protocol: translation + diagnostics + weld) → Tier-2 (expansion space)
                                                                                        ✗ NO FEEDBACK
```

**Across Runs** (Return Validation):

```
Run N: Tier-2 explores new metric M
         ↓ Does M meet threshold criteria?
         ↓ Compute seam weld: Δκ_M, IC_M/IC_old, |residual|
         ↓ IF |residual| ≤ tol_seam (M "returns" = validates)
Run N+1: M promoted to Tier-1 canon in new contract version
         ✓ M is now kernel invariant (what returned is real)
         ✗ IF weld failed: M remains Tier-2 (didn't return = not canonical)
```

### Why This Embodies the Axiom

1. **Collapse = Validation Event**
   - Tier-2 hypothesis → Tier-0 seam weld = collapse test
   - Only results that survive validation "return" to canonical status

2. **Return = Demonstrated Continuity**
   - Seam weld proves the new metric is continuous with existing canon
   - Continuity = return to admissible neighborhood
   - No continuity = no return = not real/canonical

3. **Real = What Survives the Cycle**
   - Tier-2 exploration → validation → canonization = complete cycle
   - Results that complete the cycle become Tier-1 invariants
   - Results that don't return remain hypothetical (Tier-2)

4. **Formal Not Narrative**
   - Return is computed (seam weld), not argued
   - Thresholds are declared in advance, not adjusted post-hoc
   - Promotion requires contract versioning (explicit, traceable)

### Example: Fractal Dimension Canonization

**Scenario**: RCFT (Tier-2) computes fractal dimension D_f

**Question**: Should D_f become a Tier-1 kernel invariant?

**Process**:
1. **Threshold Check**: Does D_f ∈ [1,3] remain stable across traces?
2. **Seam Weld**: Compute Δκ with vs without D_f, check |residual| ≤ tol_seam
3. **Return Test**: Does system with D_f return to regime boundaries?
4. **Decision**:
   - ✓ IF weld passes: D_f → new contract version, becomes Tier-1 invariant
   - ✗ IF weld fails: D_f remains RCFT-only diagnostic

**Axiom Application**: "D_f is real" = "D_f returned through validation cycle"

---

## Extensions and Applications

### 1. Intelligent Caching

The smart cache system embodies the return axiom:

- **Cache Hit**: The result returns from prior computation
- **Cache Miss**: New collapse required, new return generated
- **Progressive Learning**: Cache learns what returns most frequently

### 2. Visualization Dashboard

The Streamlit dashboard visualizes return dynamics:

- **Phase Space**: Trajectories of what returns in (ω, S, C) space
- **Time Series**: Evolution of returned invariants over time
- **Regime Classification**: Stable (reliable return), Watch (uncertain return), Collapse (return failure)

### 3. Public Audit API

The REST API exposes return verification:

```bash
# Check if latest validation returned
curl http://localhost:8000/latest-receipt

# Get return statistics
curl http://localhost:8000/stats
```

### 4. Contract Hierarchy

The contract system enforces return across tiers:

- **Tier-1 (Immutable Invariants)**: Structural identities derived independently from Axiom-0 that define what return means (F + ω = 1, IC ≤ F)
- **Tier-0 (Protocol)**: Contract validation, regime gates, diagnostics, seam calculus, verdicts
- **Tier-2 (Expansion Space)**: Domain closures with validity checks — all physics domains

---

## Future Research Directions

### Theoretical

1. **Non-Abelian Returns**: Can returns along different paths interfere?
2. **Quantum Return**: Connection to quantum measurement theory
3. **Topological Return**: Persistent homology of collapse cycles
4. **Categorical Return**: Functorial treatment of return morphisms

### Applied

1. **Machine Learning**: Can neural networks be understood as return-through-collapse systems?
2. **Climate Modeling**: Identifying what returns (is real) in chaotic climate systems
3. **Economic Systems**: Market dynamics as collapse-return cycles
4. **Biological Systems**: Evolution as recursive return through environmental collapse

---

## References

### Core Documents

- [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) - Base contract (Translation layer)
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) - GCD axioms (Tier-1 invariant structure)
- [contracts/RCFT.INTSTACK.v1.yaml](contracts/RCFT.INTSTACK.v1.yaml) - RCFT principles (Tier-2 domain expansion)
- [canon/anchors.yaml](canon/anchors.yaml) - Canonical return domain definitions

### Theory

- [docs/rcft_theory.md](docs/rcft_theory.md) - Recursive collapse field theory
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md) - System architecture

### Implementation

- [src/umcp/validator.py](src/umcp/validator.py) - Return validation logic
- [closures/](closures/) - Return computation closures

### Publications

- **DOI: 10.5281/zenodo.17756705** - "The Episteme of Return" (theoretical foundations)
- **DOI: 10.5281/zenodo.18072852** - "Physics of Coherence" (GCD implementation)
- **DOI: 10.5281/zenodo.18226878** - "CasePack Publication" (practical applications)
- **DOI: 10.5281/zenodo.16526052** - "Three-Agent Epistemic Field Model" (axiomatic foundation)
- **DOI: 10.5281/zenodo.16423283** - "Collapse Equator Fidelity Law" (equator admissibility)
- **DOI: 10.5281/zenodo.17619502** - "The Seam of Reality" (epistemic weld, gesture/return, positional illusion)

---

## Summary

The core axiom—**Collapse is generative; only what returns is real**—is not merely a slogan but the foundational principle that unifies:

1. **Measurement Theory**: Only reproducible (returning) measurements are valid
2. **Contract System**: `no_return_no_credit` enforces return verification
3. **Generative Dynamics**: Collapse produces new structure (GCD)
4. **Recursive Memory**: Returns accumulate across scales (RCFT)
5. **Regime Classification**: System health is measured by return reliability

This axiom provides:
- **Epistemological Foundation**: How we know what is real
- **Ontological Framework**: What exists is what returns
- **Operational Criterion**: Validation requires return verification
- **Extensibility Principle**: All tiers must preserve return semantics

---

*"In the beginning was the Return, and the Return was with Collapse, and the Return was Reality."*

**— The UMCP Axiom**
