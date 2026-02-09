# UMCP Protocol Glossary

**Version:** 1.0.0  
**Status:** Canonical  
**Last Updated:** 2026-01-20

## Purpose

This glossary provides authoritative definitions for every term and reserved symbol in the UMCP protocol ecosystem. Every entry follows the structured format required for protocol reproducibility and prevents ambiguous interpretations.

**Rule**: If a term or symbol is used in this repository but cannot be located quickly in this glossary and at least one index, it is not acceptable protocol writing.

See also:
- [Symbol Index](SYMBOL_INDEX.md) - Fast Unicode/ASCII symbol lookup
- [Term Index](TERM_INDEX.md) - Alphabetical term cross-reference
- [Canon Anchors](canon/) - Machine-readable symbol definitions

---

## Glossary Structure

Each entry contains:
- **Term** (canonical spelling)
- **Tier tag**: Tier-0 / Tier-1 / Tier-2 / Meta
- **Definition**: Operational, non-narrative
- **Not to be confused with**: Common collisions and misreads
- **Inputs/outputs**: Required if procedural
- **Where defined**: Textual pointer to canonical source
- **Where used**: Primary locations and examples
- **Status**: Canonical / Optional / Deprecated
- **Synonyms/aliases**: Redirects to canonical term

---

## Tier-0: Interface and Measurement

### Observable (x(t))

**Tier tag:** Tier-0

**Definition:** A measured quantity with units and provenance, sampled over an explicit time axis. Observables are not kernel symbols; they become kernel inputs only after embedding into Ψ(t).

**Not to be confused with:**
- Trace components c_i(t) (unitless, post-embedding)
- Model latent variables (Tier-2 constructs)
- Raw sensor readings without provenance metadata

**Inputs/outputs:**
- Consumes: Raw data acquisition with timestamps
- Produces: Unitful time series + provenance metadata

**Where defined:**
- [canon/anchors.yaml](canon/anchors.yaml) (Tier-0 interface)
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md#2-observables--trace--invariants)

**Where used:**
- All CasePacks: `observables.yaml`
- [casepacks/hello_world](casepacks/hello_world/)
- [casepacks/UMCP-REF-E2E-0001](casepacks/UMCP-REF-E2E-0001/)

**Status:** Canonical

**Synonyms/aliases:** measurement channel (allowed in context)

### Bounded Trace (Psi) (Ψ(t))

**Tier tag:** Tier-0 (declared object), consumed by Tier-1

**Definition:** The bounded trace is the embedded representation of the measured system at time index t, written Ψ(t) ∈ [0,1]^n. It is produced by the frozen Tier-0 embedding and is the sole input object for Tier-1 invariant computation.

**Not to be confused with:**
- Raw observables x(t) (unitful)
- Latent states from domain models (Tier-2)
- Wavefunction ψ from quantum mechanics (different context)

**Inputs/outputs:**
- Consumes: Tier-0 observables + embedding parameters
- Produces: Bounded vector Ψ(t) ∈ [0,1]^n used by Tier-1

**Where defined:**
- [canon/anchors.yaml](canon/anchors.yaml) (trace_objects)
- [AXIOM.md](AXIOM.md#core-axiom) (philosophical grounding)

**Where used:**
- All Tier-1 computation
- [derived/trace.csv](derived/trace.csv) (root workspace)
- All GCD and RCFT computations

**Status:** Canonical

**Synonyms/aliases:** trace, state vector (discouraged; use "trace")

### Embedding / Normalization (N_K)

**Tier tag:** Tier-0

**Definition:** A frozen mapping from observables to the bounded trace: N_K : x(t) ↦ Ψ(t) ∈ [0,1]^n. This is the declared interface between units and unitless kernel computation.

**Not to be confused with:**
- Learned embeddings (Tier-2 neural network layers)
- Dimensionality reduction used post hoc
- Feature engineering without frozen specification

**Inputs/outputs:**
- Consumes: observables + parameters (bounds, scales, transforms)
- Produces: bounded Ψ(t) and OOR flags

**Where defined:**
- [canon/anchors.yaml](canon/anchors.yaml) (Tier-0 scope)
- [embedding.yaml](embedding.yaml) (specification template)

**Where used:**
- All CasePacks with `embedding.yaml`
- [docs/python_coding_key.md](docs/python_coding_key.md) (implementation patterns)

**Status:** Canonical

**Synonyms/aliases:** normalization map (allowed)

### Epsilon-clipping / log-safety (ε)

**Tier tag:** Tier-0 frozen parameter; used by Tier-1

**Definition:** A frozen clipping threshold ε ∈ (0, 1/2) applied componentwise before any logarithms are computed. It exists to guarantee numerical stability and reproducibility, not to change meanings post hoc. Default: ε = 1e-10.

**Not to be confused with:**
- Machine epsilon (2.22e-16 for float64)
- Ad hoc "small constants" inserted after seeing results
- Tolerance parameters for optimization

**Inputs/outputs:**
- Consumes: Ψ(t) raw components
- Produces: Ψ_ε(t) (log-safe) + clipping flags when triggered

**Where defined:**
- [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) (epsilon parameter)
- [canon/anchors.yaml](canon/anchors.yaml) (log-safety conventions)

**Where used:**
- Tier-1 entropy S(t) computation
- Tier-1 integrity κ(t) computation
- [src/umcp/validator.py](src/umcp/validator.py) (implementation)

**Status:** Canonical

**Synonyms/aliases:** log-safety clip (allowed)

### Weights (w_i)

**Tier tag:** Tier-0 (declared and frozen), consumed by Tier-1

**Definition:** Nonnegative weights used to aggregate components into scalar invariants, with w_i ≥ 0 and Σ_i w_i = 1. Weights are frozen before compute; any change is an interface change (Tier-0) and must produce a new case/version.

**Not to be confused with:**
- Model parameters or learned coefficients (Tier-2)
- Neural network weights (different context)
- Arbitrary importance rankings without normalization

**Inputs/outputs:**
- Consumes: Contract/weights artifact specification
- Produces: Deterministic weighting in Tier-1 equations

**Where defined:**
- [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) (weights section)
- [weights.csv](weights.csv) (root-level default)

**Where used:**
- All Tier-1 scalar computations (F, ω, S, κ)
- All weld ledger/budget terms indirectly
- [casepacks/*/weights.csv](casepacks/) in all CasePacks

**Status:** Canonical

**Synonyms/aliases:** importance weights (allowed only if explicitly Tier-0)

### Missingness Policy

**Tier tag:** Tier-0

**Definition:** The frozen rules for handling missing observations or undefined preprocessing steps. Missingness must never be silently repaired; either error, or deterministic imputation with explicit flagging is permitted if declared before compute.

**Not to be confused with:**
- Quiet data cleaning without documentation
- Interpolation without recording provenance
- "Best guess" fills that compromise reproducibility

**Inputs/outputs:**
- Consumes: Raw observations + missingness flags
- Produces: Preprocessed values + explicit missingness flags

**Where defined:**
- [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) (missingness_policy)
- CasePack schema conventions

**Where used:**
- Trace generation in all CasePacks
- Nonconformance checks in validation
- "unknown — hint" reporting patterns

**Status:** Canonical

**Synonyms/aliases:** NA handling (avoid informal phrasing)

---

## Tier-1: Reserved Symbols (GCD Framework)

See [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) for complete machine-readable definitions.

### Drift (ω(t))

**Tier tag:** Tier-1 reserved

**Definition:** Kernel drift defined as ω(t) = 1 - F(t). It is a bounded scalar in [0,1] under standard constraints and is used directly in regime gates and (via closures) in weld budgeting.

**Not to be confused with:**
- Angular frequency Ω (must be written as Ω_freq or Omega_freq)
- Oscillation rate in signal processing
- Any domain-specific "omega" without tier declaration

**Inputs/outputs:**
- Consumes: F(t) (Tier-1 fidelity)
- Produces: Scalar in [0,1]

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.omega)
- [docs/rcft_theory.md](docs/rcft_theory.md#theoretical-foundation)

**Where used:**
- Regime gates: Stable (ω < 0.038), Collapse (ω ≥ 0.30)
- Closures: Γ(ω; p, ε) for weld budgeting
- [outputs/invariants.csv](outputs/invariants.csv) (root workspace or casepacks)

**Status:** Canonical

**Synonyms/aliases:** none

### Fidelity (F(t))

**Tier tag:** Tier-1 reserved

**Definition:** Weighted component average F(t) = Σ_i w_i c_i(t). Fidelity is dimensionless by construction and is computed deterministically from the frozen trace and weights.

**Not to be confused with:**
- Likelihood or probability (Tier-2 construct)
- Fitness scores from evolutionary algorithms
- Model accuracy metrics (Tier-2 diagnostic)

**Inputs/outputs:**
- Consumes: w_i (Tier-0 weights), c_i(t) (Tier-0 trace components)
- Produces: Scalar in [0,1]

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.F)
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (identity: F = 1 - ω)

**Where used:**
- Regime gates: Collapse threshold F < 0.75
- [tests/test_100_gcd_canon.py](tests/test_100_gcd_canon.py) (identity validation)
- All CasePack invariants outputs

**Status:** Canonical

**Synonyms/aliases:** none

### Entropy Functional (S(t))

**Tier tag:** Tier-1 reserved

**Definition:** Operational functional on the ε-clipped trace, defined componentwise to avoid log singularities. It is not automatically thermodynamic entropy; any thermodynamic mapping is Tier-2 and must be declared explicitly.

**Not to be confused with:**
- Thermodynamic entropy S_th (requires Tier-2 mapping)
- Shannon entropy of another distribution
- Cross-entropy losses in machine learning (Tier-2)

**Inputs/outputs:**
- Consumes: w_i and Ψ_ε(t) (log-safe trace)
- Produces: Nonnegative scalar (unitless)

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.S)
- [closures/README.md](closures/README.md) (entropy computation patterns)

**Where used:**
- Regime gates: collapse boundary detection
- RCFT Tier-2: fractal dimension embedding space
- [tests/test_101_gcd_closures.py](tests/test_101_gcd_closures.py)

**Status:** Canonical

**Synonyms/aliases:** none

### Curvature Proxy (C(t))

**Tier tag:** Tier-1 reserved (closure-controlled variants allowed)

**Definition:** Operational dispersion/shape proxy over components of Ψ(t). Default: normalized population standard deviation across {c_i(t)} using a fixed normalizer. Variants are permitted only as declared closure choices that preserve the operational meaning (component dispersion/shape proxy).

**Not to be confused with:**
- Capacitance C_cap (must use subscript/suffix)
- Differential-geometric curvature (Riemannian geometry)
- Any domain "C" without tier declaration

**Inputs/outputs:**
- Consumes: Ψ(t) and frozen variance convention
- Produces: Scalar (unitless) in [0,1] typically

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.C)
- [closures/registry.yaml](closures/registry.yaml) (curvature variants)

**Where used:**
- Regime gates: stability thresholds
- Weld budgeting: D_C dissipation term
- [closures/curvature_neighborhood.default.v1.yaml](closures/curvature_neighborhood.default.v1.yaml)

**Status:** Canonical

**Synonyms/aliases:** none

### Re-entry Delay / Return Time (τ_R(t))

**Tier tag:** Tier-1 reserved (typed boundary states permitted)

**Definition:** Return time defined operationally in Ψ-space via a frozen return domain D_θ(t), norm, tolerance η, and horizon H_rec. If no return candidate exists within the frozen domain/horizon, τ_R(t) takes the typed boundary state ∞_rec.

**Not to be confused with:**
- Large sentinel integer (e.g., 99999)
- Fitted parameter from model
- Recurrence intervals in raw x-space unless explicitly declared

**Inputs/outputs:**
- Consumes: Ψ(t), D_θ(t), ||·||, η, H_rec
- Produces: ℕ ∪ {∞_rec} (typed)

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.tau_R)
- [closures/tau_R_compute.py](closures/tau_R_compute.py) (implementation)
- [closures/return_domain.window64.v1.yaml](closures/return_domain.window64.v1.yaml)

**Where used:**
- Weld gate: finite return requirement (no ∞_rec allowed for PASS)
- Seam budgeting: R·τ_R credit term
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** return delay (allowed)

### Log-integrity (κ(t))

**Tier tag:** Tier-1 reserved

**Definition:** Weighted log-integrity computed from ε-clipped components: κ(t) = Σ_i w_i ln(c_i,ε(t)). It is the log of the integrity composite: κ = ln(IC) exactly.

**Not to be confused with:**
- Curvature symbol κ in differential geometry
- Complexity scores from information theory
- Model evidence in Bayesian statistics (Tier-2)

**Inputs/outputs:**
- Consumes: w_i, Ψ_ε(t)
- Produces: Scalar (typically ≤ 0)

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.kappa)
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (kappa/IC identity)

**Where used:**
- Weld ledger identity: Δκ_ledger = κ_1 - κ_0
- Seam accounting in all weld computations
- [tests/test_102_gcd_contract.py](tests/test_102_gcd_contract.py)

**Status:** Canonical

**Synonyms/aliases:** none

### Integrity Composite (IC)

**Tier tag:** Tier-1 reserved

**Definition:** Composite integrity defined as IC(t) = exp(κ(t)), i.e., the weighted geometric aggregate of clipped component values in Ψ_ε(t).

**Not to be confused with:**
- Information content in external literature (use I_info)
- Mutual information
- Entropy-based information metrics

**Inputs/outputs:**
- Consumes: κ(t)
- Produces: Scalar in (0,1] under ε-clipping

**Where defined:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) (reserved_symbols.IC)
- Mathematical identity: IC = exp(κ)

**Where used:**
- Severity markers: Critical tag when min_t IC(t) below threshold
- Weld ledger/budget via Δκ
- [tests/test_102_gcd_contract.py](tests/test_102_gcd_contract.py) (identity checks)

**Status:** Canonical

**Synonyms/aliases:** integrity (allowed only if unambiguous)

### Regime Labels: Stable / Watch / Collapse

**Tier tag:** Tier-1 gates (kernel-only)

**Definition:** Binding categorical labels computed only from the kernel gate thresholds on reserved Tier-1 invariants. They are gates, not diagnostics.

**Not to be confused with:**
- Diagnostic severity tags (Tier-2)
- Narrative judgments about system state
- Model-based classifications (Tier-2 domain expansion closures)

**Inputs/outputs:**
- Consumes: (ω, F, S, C) and frozen thresholds from contract
- Produces: Categorical label {Stable, Watch, Collapse}

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (regime_gates)
- [AXIOM.md](AXIOM.md#regime-classifications)

**Where used:**
- All case summaries
- [outputs/regimes.csv](outputs/regimes.csv) in CasePacks
- [tests/test_100_gcd_canon.py](tests/test_100_gcd_canon.py) (regime validation)

**Status:** Canonical

**Synonyms/aliases:** none (do not rename labels)

---

## Seam and Weld Calculus (Tier-0 Protocol)

### Seam

**Tier tag:** Tier-0 (context object)

**Definition:** A declared transition context t_0 → t_1 across which continuity is evaluated. A seam is not automatically a weld; it becomes weld-claimable only when a weld row is computed under a frozen closure registry.

**Not to be confused with:**
- Formatting changes or document revisions
- EID (artifact structural fingerprints)
- Narrative "phase changes" without accounting

**Inputs/outputs:**
- Consumes: Seam endpoints t_0, t_1 and context rules
- Produces: Weld row when computed with closures

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (seam_accounting)
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md#component-hierarchy)

**Where used:**
- All continuity claims
- [expected/seam_receipt.json](expected/seam_receipt.json)
- Canon evolution: PRE→POST seams

**Status:** Canonical

**Synonyms/aliases:** transition (allowed if defined)

### Weld

**Tier tag:** Tier-0 (seam event with PASS/FAIL gate)

**Definition:** A seam event with explicit continuity accounting: ledger identity plus budget closure, residual computation, and a PASS/FAIL gate under frozen tolerances and finite return.

**Not to be confused with:**
- EID artifact fingerprinting (structural only)
- Narrative linking without accounting
- Post hoc interpretation of similarity

**Inputs/outputs:**
- Consumes: Seam endpoints, Tier-1 invariants, closures, tolerances
- Produces: Weld row + PASS/FAIL status

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (weld_gate)
- [AXIOM.md](AXIOM.md#weld-continuity)

**Where used:**
- All continuity claims
- [outputs/welds.csv](outputs/welds.csv) in CasePacks
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** seam test (allowed if refers to PASS/FAIL gate)

### Ledger Delta (Δκ_ledger)

**Tier tag:** Tier-0 (identity)

**Definition:** The identity-based continuity ledger: Δκ_ledger = κ_1 - κ_0. It is not a model; it is an accounting identity.

**Not to be confused with:**
- Budget delta Δκ_budget (closure-dependent)
- Diagnostic residuals (Tier-2)

**Inputs/outputs:**
- Consumes: κ_0, κ_1 (Tier-1 log-integrity at endpoints)
- Produces: Scalar ledger delta

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (ledger identity)
- Mathematical identity: simple difference

**Where used:**
- Residual computation: s = Δκ_budget - Δκ_ledger
- SS1m receipts
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py) (weld validation)

**Status:** Canonical

**Synonyms/aliases:** ledger (allowed if unambiguous)

### Budget Delta (Δκ_budget)

**Tier tag:** Tier-0 (closure-dependent)

**Definition:** The closure-driven continuity budget, typically Δκ_budget = R·τ_R - (D_ω + D_C) with typed censoring when τ_R = ∞_rec.

**Not to be confused with:**
- Ledger identity (closure-independent)
- Interpretive narrative about "expected change"

**Inputs/outputs:**
- Consumes: τ_R, closures for R, D_ω, D_C
- Produces: Scalar budget delta

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (budget_formula)
- [closures/registry.yaml](closures/registry.yaml) (closure specifications)

**Where used:**
- Residual computation for PASS/FAIL gate
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** budget (allowed if precise)

### Residual (Seam Residual s)

**Tier tag:** Tier-0 (gate term)

**Definition:** The reconciliation residual s = Δκ_budget - Δκ_ledger. PASS/FAIL is determined by whether |s| is within the frozen seam tolerance, alongside other gate conditions (finite return, identity check).

**Not to be confused with:**
- Diagnostic residuals (Tier-2), such as equator-fit residuals
- Model prediction errors
- Regression residuals

**Inputs/outputs:**
- Consumes: Ledger and budget deltas
- Produces: Scalar residual used by weld gate

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (gate_conditions)

**Where used:**
- [outputs/welds.csv](outputs/welds.csv) (residual column)
- PASS/FAIL determination
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** weld residual (allowed)

### PASS / FAIL (Weld Gate)

**Tier tag:** Tier-0 gate

**Definition:** A binding continuity decision computed deterministically under a frozen contract and closure registry. Diagnostics may explain failure but cannot convert FAIL into PASS.

**Not to be confused with:**
- Interpretive approval or acceptance
- Peer-review decision
- Diagnostic threshold interpretation

**Inputs/outputs:**
- Consumes: Finite return check, tolerances, residual, identity check
- Produces: Categorical weld status {PASS, FAIL}

**Where defined:**
- [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) (weld_gate section)
- [AXIOM.md](AXIOM.md#gate-vs-diagnostic)

**Where used:**
- All continuity claims and receipts
- [expected/seam_receipt.json](expected/seam_receipt.json)
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** none

### Gesture

**Tier tag:** Tier-0 (epistemic classification)

**Definition:** An epistemic emission that does not complete the collapse-return cycle. A gesture exists — it is not nothing — but the seam did not close: either τ_R = ∞_rec, |s| > tol_seam, or the exponential identity failed. A gesture may be internally consistent, structurally complex, and indistinguishable from a genuine return in every way except the weld. No epistemic credit is awarded.

**Not to be confused with:**
- A failed test (a gesture is not "wrong" — it simply did not return)
- An approximation (gestures are not close-enough returns; the threshold is absolute)
- A preliminary result (gestures cannot be upgraded to returns by argument)

**Inputs/outputs:**
- Consumes: Seam calculus results (residual, τ_R, identity check)
- Produces: EpistemicVerdict.GESTURE with GestureReason list

**Where defined:**
- [src/umcp/epistemic_weld.py](src/umcp/epistemic_weld.py) (EpistemicVerdict, GestureReason)
- [AXIOM.md](AXIOM.md#the-gesture-return-distinction) (philosophical grounding)
- "The Seam of Reality" (Paulus, 2025; DOI: 10.5281/zenodo.17619502) §3

**Where used:**
- Epistemic verdict classification in epistemic_weld.py
- NonconformanceType.GESTURE in frozen_contract.py
- Gesture diagnostics in epistemic_weld.py (diagnose_gesture)

**Status:** Canonical

**Synonyms/aliases:** non-returning emission (allowed)

### Positional Illusion

**Tier tag:** Tier-0 (epistemic concept)

**Definition:** The belief that one can observe a system without incurring measurement cost — that there exists a vantage point outside the system from which collapse can be observed for free. Theorem T9 (Zeno analog) proves this is impossible: each observation costs Γ(ω) = ω^p/(1−ω+ε) in seam budget. N observations incur N×Γ(ω) overhead. The illusion is quantified by the ratio of observation cost to seam tolerance.

**Not to be confused with:**
- Observer bias (a statistical artifact, not a structural cost)
- Objectivity (the protocol does not deny objectivity — it prices it)
- Measurement error (error is in the value; the illusion is about cost)

**Inputs/outputs:**
- Consumes: ω (drift), N (observation count), frozen constants
- Produces: PositionalIllusion (gamma, total_cost, budget_fraction, severity)

**Where defined:**
- [src/umcp/epistemic_weld.py](src/umcp/epistemic_weld.py) (quantify_positional_illusion)
- [src/umcp/tau_r_star.py](src/umcp/tau_r_star.py) Thm T9
- [AXIOM.md](AXIOM.md#the-positional-illusion) (philosophical grounding)
- "The Seam of Reality" (Paulus, 2025; DOI: 10.5281/zenodo.17619502) §4.2

**Where used:**
- Epistemic assessment in epistemic_weld.py (SeamEpistemology.illusion)
- Budget analysis in tau_r_star.py (observation cost commentary)

**Status:** Canonical

**Synonyms/aliases:** observation cost illusion (allowed)

### Dissolution

**Tier tag:** Tier-0 (epistemic classification)

**Definition:** The epistemic verdict when a system enters Regime.COLLAPSE (ω ≥ 0.30). The trace has degraded past the point of viable return credit. Dissolution is not failure — it is the boundary condition that makes return meaningful. Without the possibility of dissolution, the seam would audit nothing.

**Not to be confused with:**
- System failure (dissolution is structural, not operational)
- Data loss (the trace exists; it simply cannot earn return credit)
- Error (dissolution is a valid epistemic state, not an error condition)

**Inputs/outputs:**
- Consumes: Regime classification
- Produces: EpistemicVerdict.DISSOLUTION

**Where defined:**
- [src/umcp/epistemic_weld.py](src/umcp/epistemic_weld.py) (EpistemicVerdict)
- [AXIOM.md](AXIOM.md#operational-definitions-enforcement-tied) (operational table)
- "The Seam of Reality" (Paulus, 2025; DOI: 10.5281/zenodo.17619502) §3

**Where used:**
- Epistemic verdict classification in epistemic_weld.py
- Regime interpretation throughout codebase

**Status:** Canonical

**Synonyms/aliases:** none

---

## Tier-2: RCFT Overlay Extensions

See [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) for complete machine-readable definitions.

### Fractal Dimension (D_fractal)

**Tier tag:** Tier-2 (RCFT extension)

**Definition:** Box-counting fractal dimension of the collapse trajectory in (ω, S, C) embedding space. Quantifies geometric complexity: D_f ≈ 1 for smooth curves, D_f → 2 for space-filling behavior.

**Not to be confused with:**
- Hausdorff dimension (different mathematical definition)
- Euclidean dimension (topological property)
- Tier-1 reserved symbols (this is an overlay metric)

**Inputs/outputs:**
- Consumes: Time series (ω(t), S(t), C(t)) from Tier-1
- Produces: Scalar in [1, 3] (bounded by embedding dimension)

**Where defined:**
- [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) (tier_2_extensions.D_fractal)
- [docs/rcft_theory.md](docs/rcft_theory.md#fractal-dimension)
- [closures/rcft/fractal_dimension.py](closures/rcft/fractal_dimension.py)

**Where used:**
- RCFT regime classification: Smooth/Wrinkled/Turbulent
- [casepacks/rcft_complete/](casepacks/rcft_complete/)
- [tests/test_110_rcft_canon.py](tests/test_110_rcft_canon.py)

**Status:** Canonical (Tier-2)

**Synonyms/aliases:** D_f (mathematical notation)

### Recursive Field (Ψ_recursive)

**Tier tag:** Tier-2 (RCFT extension)

**Definition:** Exponentially-decayed summation of historical field strengths: Ψ_r = Σ(n=1→∞) α^n · Ψ_n, where Ψ_n = √(S_n² + C_n²) · (1 - F_n). Quantifies cumulative collapse memory.

**Not to be confused with:**
- Bounded trace Ψ(t) (Tier-0/Tier-1 input object)
- Autoregressive model states (Tier-2 modeling)
- Neural network recurrence (different context)

**Inputs/outputs:**
- Consumes: Historical (S, C, F) from Tier-1, decay α parameter
- Produces: Scalar in [0, ∞), typically [0, 2]

**Where defined:**
- [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) (tier_2_extensions.Psi_recursive)
- [docs/rcft_theory.md](docs/rcft_theory.md#recursive-field)
- [closures/rcft/recursive_field.py](closures/rcft/recursive_field.py)

**Where used:**
- RCFT regime: Dormant/Active/Resonant classification
- [casepacks/rcft_complete/](casepacks/rcft_complete/)
- [tests/test_111_rcft_closures.py](tests/test_111_rcft_closures.py)

**Status:** Canonical (Tier-2)

**Synonyms/aliases:** Ψ_r (mathematical notation)

### Resonance Pattern (λ_pattern, Θ_phase)

**Tier tag:** Tier-2 (RCFT extension)

**Definition:** FFT-derived oscillatory structure analysis. λ_pattern = 2π/k_dominant (wavelength), Θ_phase = arctan(Im/Re) (phase angle). Identifies Standing/Mixed/Traveling wave patterns.

**Not to be confused with:**
- Eigenvalue λ from linear algebra (different context)
- Phase transitions in thermodynamics
- Signal processing phase without field theory context

**Inputs/outputs:**
- Consumes: Time series of field quantity (typically Ψ_r or S)
- Produces: λ (wavelength), Θ (phase), pattern classification

**Where defined:**
- [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) (tier_2_extensions.lambda_pattern)
- [docs/rcft_theory.md](docs/rcft_theory.md#resonance-patterns)
- [closures/rcft/resonance_pattern.py](closures/rcft/resonance_pattern.py)

**Where used:**
- RCFT pattern classification
- [casepacks/rcft_complete/](casepacks/rcft_complete/)
- [tests/test_111_rcft_closures.py](tests/test_111_rcft_closures.py)

**Status:** Canonical (Tier-2)

**Synonyms/aliases:** λ_p (mathematical notation for wavelength)

---

## Tier-1 Extensions: WEYL Cosmological Framework

See [canon/weyl_anchors.yaml](canon/weyl_anchors.yaml) for complete machine-readable definitions.

### Weyl Potential (Ψ_W)

**Tier tag:** Tier-1 (WEYL extension)

**Definition:** The geometric sum of metric potentials Ψ_W ≡ (Φ + Ψ)/2, where Φ and Ψ are the Newtonian and curvature potentials in the metric perturbation. This is the primary gravitational observable for weak lensing.

**Not to be confused with:**
- Tier-1 bounded trace Ψ(t) (different object, same Greek letter)
- Wavefunction ψ from quantum mechanics
- Individual metric potentials Φ or Ψ alone

**Inputs/outputs:**
- Consumes: Metric perturbations from cosmological model
- Produces: Lensing observable Ψ_W(k, z)

**Where defined:**
- [canon/weyl_anchors.yaml](canon/weyl_anchors.yaml) (reserved_symbols.Psi_W)
- [contracts/WEYL.INTSTACK.v1.yaml](contracts/WEYL.INTSTACK.v1.yaml)

**Where used:**
- [closures/weyl/weyl_transfer.py](closures/weyl/weyl_transfer.py)
- [casepacks/weyl_des_y3/](casepacks/weyl_des_y3/)

**Status:** Canonical (WEYL extension)

### Gravity Modification Function (Σ)

**Tier tag:** Tier-1 (WEYL extension)

**Definition:** The phenomenological function Σ(z) that modifies the Poisson equation for lensing: ∇²Ψ_W = 4πG Σ a² ρ_m δ_m. General Relativity corresponds to Σ = 1.

**Not to be confused with:**
- Summation symbol Σ (context-dependent)
- Variance σ² (different symbol entirely)
- Tier-1 entropy S (reserved for different purpose)

**Inputs/outputs:**
- Consumes: ĥJ measurements, cosmological background
- Produces: Σ(z) values per redshift bin

**Where defined:**
- [canon/weyl_anchors.yaml](canon/weyl_anchors.yaml) (reserved_symbols.Sigma)
- [contracts/WEYL.INTSTACK.v1.yaml](contracts/WEYL.INTSTACK.v1.yaml) (Eq. 11-13)

**Where used:**
- [closures/weyl/sigma_evolution.py](closures/weyl/sigma_evolution.py)
- [casepacks/weyl_des_y3/](casepacks/weyl_des_y3/)

**Status:** Canonical (WEYL extension)

### Weyl Evolution Proxy (ĥJ)

**Tier tag:** Tier-1 (WEYL extension)

**Definition:** Dimensionless lensing proxy ĥJ(z) = J(z) × σ8(z) / D₁(z), where J(z) is the Weyl deviation function. This is the UMCP "fidelity analog" for cosmological observations.

**Not to be confused with:**
- Tier-1 fidelity F (different invariant, similar role)
- Galaxy bias b (separate quantity)
- Hubble parameter H (different symbol)

**Inputs/outputs:**
- Consumes: Lensing measurements, growth function, σ8 normalization
- Produces: Dimensionless proxy per redshift bin

**Where defined:**
- [canon/weyl_anchors.yaml](canon/weyl_anchors.yaml) (reserved_symbols.hJ)
- [contracts/WEYL.INTSTACK.v1.yaml](contracts/WEYL.INTSTACK.v1.yaml) (Eq. 4)

**Where used:**
- [closures/weyl/sigma_evolution.py](closures/weyl/sigma_evolution.py)
- [casepacks/weyl_des_y3/](casepacks/weyl_des_y3/)

**Status:** Canonical (WEYL extension)

---

## Closure Registry and Governance

### Closure

**Tier tag:** Tier-0 (declared auxiliary specification)

**Definition:** Any additional specification needed to complete seam computation beyond Tier-1 identities, especially the weld budget and neighborhood-dependent computation. Closures are declared as "form + parameters" and frozen before compute.

**Not to be confused with:**
- Changing Tier-1 definitions
- Narrative explanations without formal specification
- Diagnostics (Tier-2 overlays)

**Inputs/outputs:**
- Consumes: Reserved Tier-1 quantities and frozen parameters
- Produces: Auxiliary terms for budgeting or controlled variants

**Where defined:**
- [closures/registry.yaml](closures/registry.yaml) (registry specification)
- [closures/README.md](closures/README.md) (governance rules)

**Where used:**
- All weld budgeting computations
- Curvature variants
- [tests/test_101_gcd_closures.py](tests/test_101_gcd_closures.py)

**Status:** Canonical

**Synonyms/aliases:** none

### Closure Registry (closures.yaml)

**Tier tag:** Tier-0 governance artifact (mandatory for weld claims)

**Definition:** A machine-readable declaration of all closure forms, parameters, timing/aggregation rules, and stable IDs/versions. No weld claim is admissible without a complete closure registry.

**Not to be confused with:**
- Prose descriptions without machine-readable format
- Implicit defaults not recorded
- "Standard closure" claims without ID/version

**Inputs/outputs:**
- Consumes: Closure choices at freeze time
- Produces: Pinned registry used by compute and audit

**Where defined:**
- [closures/registry.yaml](closures/registry.yaml) (root-level registry)
- [closures/README.md](closures/README.md) (schema and rules)
- [schemas/closures_registry.schema.json](schemas/closures_registry.schema.json)

**Where used:**
- All weld rows and receipts
- CasePack closure specifications
- [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Status:** Canonical

**Synonyms/aliases:** registry (allowed if ID/version always included)

### Freeze

**Tier tag:** Procedure (Tier-0 gating step)

**Definition:** The required step that locks the measurement interface and the closure registry (Tier-0) prior to any kernel or seam computation. Without freeze, results are nonconformant.

**Not to be confused with:**
- Software snapshot without artifact hashes
- "We decided later" choices (post hoc)
- Version control commits (insufficient alone)

**Inputs/outputs:**
- Consumes: Contract, embedding, observables, return settings, weights, closures
- Produces: Hash-pinned frozen artifacts

**Where defined:**
- [docs/production_deployment.md](docs/production_deployment.md#freeze-procedure)
- Pipeline: /ingest → /freeze → /compute

**Where used:**
- All admissible runs
- Validation checks: [src/umcp/validator.py](src/umcp/validator.py)
- [tests/test_97_root_integration.py](tests/test_97_root_integration.py)

**Status:** Canonical

**Synonyms/aliases:** lock step (avoid informal phrasing)

---

## Meta: Reporting and Governance

### Nonconformance

**Tier tag:** Meta (protocol violation state)

**Definition:** A structural protocol violation that invalidates admissibility (distinct from a weld FAIL, which is a valid outcome). Nonconformant runs must still export artifacts and a report, but results must be marked "do not interpret."

**Not to be confused with:**
- Weld FAIL (valid outcome within protocol)
- Poor performance or surprising results
- Diagnostic warnings (Tier-2)

**Inputs/outputs:**
- Consumes: Validation checks (freeze, closures, hashes, symbol capture)
- Produces: status=NONCONFORMANT with reason codes

**Where defined:**
- [src/umcp/validator.py](src/umcp/validator.py) (validation logic)
- [AXIOM.md](AXIOM.md#conformance-rules)

**Where used:**
- CasePack validation outcomes
- [validator.result.json](validator.result.json)
- [tests/test_10_canon_contract_closures_validate.py](tests/test_10_canon_contract_closures_validate.py)

**Status:** Canonical

**Synonyms/aliases:** invalid run (discouraged; use nonconformant)

### Diagnostic vs Gate

**Tier tag:** Meta enforcement boundary

**Definition:** A gate produces binding classifications (regimes; weld PASS/FAIL; hard stops). A diagnostic provides insight but has no authority to decide labels or admissibility. This separation preserves falsifiability and prevents narrative repair.

**Not to be confused with:**
- "Important diagnostic" treated as implicit gate
- Heuristic thresholds given authoritative status
- Soft gates (not permitted in protocol)

**Inputs/outputs:**
- Consumes: Computed quantities
- Produces: Categorical outputs only for gates; numeric/context for diagnostics

**Where defined:**
- [AXIOM.md](AXIOM.md#diagnostic-vs-gate)
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md#4-invariants--regimes)

**Where used:**
- All reporting
- Schema separation: invariants.csv vs diagnostics.*
- [tests/test_90_edge_cases.py](tests/test_90_edge_cases.py)

**Status:** Canonical

**Synonyms/aliases:** none

### Manifest (manifest.yaml)

**Tier tag:** Meta (case bundle index)

**Definition:** The authoritative inventory of a CasePack: required files, schema version, IDs, hashes, and code/environment pins. The manifest is the reproducibility entry point.

**Not to be confused with:**
- Directory listing (unstructured)
- Informal notes or README
- Package.json (different context)

**Inputs/outputs:**
- Consumes: All case artifacts
- Produces: Pinned reproducibility bundle record

**Where defined:**
- [manifest.yaml](manifest.yaml) (root-level example)
- [schemas/manifest.schema.json](schemas/manifest.schema.json)
- CasePack schema documentation

**Where used:**
- All CasePacks: `manifest.yaml`
- Validation entry point: `umcp validate`
- [tests/test_20_casepack_hello_world_validates.py](tests/test_20_casepack_hello_world_validates.py)

**Status:** Canonical

**Synonyms/aliases:** bundle index (allowed if precise)

### "unknown — hint"

**Tier tag:** Meta (gap reporting convention)

**Definition:** A controlled format for acknowledging missing information without inventing it: "unknown — <cause>; hint: <actionable next step>." Prevents silent repair and false precision.

**Not to be confused with:**
- Guessing defaults silently
- Footnoted omissions without structured format
- Hiding gaps in prose

**Inputs/outputs:**
- Consumes: Missing/ambiguous requirements
- Produces: Structured gap item in Gaps section of report

**Where defined:**
- [AXIOM.md](AXIOM.md#unknown-hint-convention)
- Reporting standards

**Where used:**
- All CasePack reports: Gaps section
- [outputs/report.txt](outputs/report.txt) patterns
- [tests/test_96_file_references.py](tests/test_96_file_references.py)

**Status:** Canonical

**Synonyms/aliases:** unknown note (discouraged; use exact format)

---

## Kinematics Extension (Tier-1)

The Kinematics Extension (KIN.INTSTACK.v1) provides physics-based motion analysis within UMCP. See [KINEMATICS_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md) for complete mathematical foundations.

### Position (x)

**Tier tag:** Tier-1 (Kinematics)

**Definition:** Normalized position in phase space, x ∈ [0,1]. Maps physical position to bounded domain via x̃ = x / L_ref.

**Not to be confused with:**
- Raw position in meters
- Observable traces (Tier-0)
- Coordinate indices

**Inputs/outputs:**
- Consumes: Physical position, reference length L_ref
- Produces: Normalized position for phase space analysis

**Where defined:**
- [contracts/KIN.INTSTACK.v1.yaml](contracts/KIN.INTSTACK.v1.yaml)
- [canon/kin_anchors.yaml](canon/kin_anchors.yaml)

**Status:** Canonical

### Velocity (v)

**Tier tag:** Tier-1 (Kinematics)

**Definition:** Normalized velocity, v ∈ [0,1]. Maps physical velocity to bounded domain via ṽ = v / v_ref.

**Not to be confused with:**
- Physical velocity in m/s
- Drift ω (different quantity)
- Rate of change of kernel quantities

**Where defined:**
- [contracts/KIN.INTSTACK.v1.yaml](contracts/KIN.INTSTACK.v1.yaml)
- [canon/kin_anchors.yaml](canon/kin_anchors.yaml)

**Status:** Canonical

### Kinematic Return Time (τ_kin)

**Tier tag:** Tier-1 (Kinematics)

**Definition:** The time for a trajectory to return to within η_phase of its initial phase space point: τ_kin = inf{t > δ : d(γ(t), γ₀) < η_phase}. Finite τ_kin indicates periodic/oscillatory motion.

**Not to be confused with:**
- τ_R (return time in abstract state space)
- Physical period T
- Damping time constants

**Inputs/outputs:**
- Consumes: Phase space trajectory γ(t) = (x(t), v(t)), tolerance η_phase
- Produces: Return time or INF_KIN if non-returning

**Where defined:**
- [closures/kinematics/phase_space_return.py](closures/kinematics/phase_space_return.py)
- [KINEMATICS_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md#return-axiom-mapping)

**Status:** Canonical

### Kinematic Stability Index (K_stability)

**Tier tag:** Tier-1 (Kinematics)

**Definition:** A normalized index K_stability ∈ [0,1] measuring dynamic stability. K_stability → 1 indicates convergence to equilibrium; K_stability → 0 indicates instability.

**Not to be confused with:**
- IC (integrity composite from base UMCP)
- Lyapunov exponents (related but different)
- Energy stability

**Inputs/outputs:**
- Consumes: Position, velocity, acceleration series
- Produces: Stability index and regime classification

**Where defined:**
- [closures/kinematics/kinematic_stability.py](closures/kinematics/kinematic_stability.py)
- [canon/kin_anchors.yaml](canon/kin_anchors.yaml)

**Status:** Canonical

### Phase Space (Γ_kin)

**Tier tag:** Tier-1 (Kinematics)

**Definition:** The 2D space of (position, velocity) pairs: Γ_kin = {(x, v) : x ∈ [0,1], v ∈ [0,1]}. Kinematic dynamics trace paths through this space.

**Not to be confused with:**
- Full state space Ψ (higher dimensional)
- Configuration space (position only)
- Momentum space

**Where defined:**
- [KINEMATICS_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md#phase-space)
- [canon/kin_anchors.yaml](canon/kin_anchors.yaml)

**Status:** Canonical

---

## Artifact Integrity

### EID (Artifact Structural Fingerprint)

**Tier tag:** Meta (artifact fingerprinting)

**Definition:** A structural fingerprint of an artifact (counts of elements such as pages/equations/figures/tables/blocks) used to verify document structural stability across edits. EID is not weld identity and cannot stand in for continuity evidence.

**Not to be confused with:**
- Weld ID (seam event identifier)
- Case ID (bundle identifier)
- Cryptographic hashes (content-based)

**Inputs/outputs:**
- Consumes: Artifact structure (element counts)
- Produces: Count vector and optional checksum

**Where defined:**
- [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md#eid-conventions)
- Versioning documentation

**Where used:**
- Artifact metadata tracking
- Document evolution verification
- Publication workflows

**Status:** Canonical

**Synonyms/aliases:** structural fingerprint (allowed)

### sha256 Integrity Ledger

**Tier tag:** Meta (integrity mechanism)

**Definition:** A cryptographic hashing record (SHA-256) for all required artifacts, used to detect drift and support verification. Hashing is part of admissibility: without it, reproducibility is not verifiable.

**Not to be confused with:**
- Non-cryptographic checksums
- EID (structural fingerprints)
- Git commit hashes (code provenance)

**Inputs/outputs:**
- Consumes: Case files
- Produces: Stable digest ledger

**Where defined:**
- [integrity/sha256.txt](integrity/sha256.txt) (root-level ledger)
- [docs/production_deployment.md](docs/production_deployment.md#integrity-verification)

**Where used:**
- Verify procedure for all CasePacks
- Publication citation requirements
- [tests/test_97_root_integration.py](tests/test_97_root_integration.py)

**Status:** Canonical

**Synonyms/aliases:** hash ledger (allowed)

---

## Index Cross-References

See also:
- **[Symbol Index](SYMBOL_INDEX.md)** - Unicode/ASCII symbol lookup table
- **[Term Index](TERM_INDEX.md)** - Alphabetical term index with file locations
- **[Canon Anchors](canon/)** - Machine-readable tier-tagged definitions
- **[Contract Files](contracts/)** - Frozen parameter specifications
- **[Closure Registry](closures/registry.yaml)** - Computational closure governance

---

## Deprecated Terms

### ~~raw_drift~~ (DEPRECATED)

**Status:** Deprecated as of v1.2.0

**Reason:** Ambiguous relationship to Tier-1 ω. Created confusion between raw measurements and normalized kernel quantities.

**Migration:** Use **ω** (Tier-1 reserved drift) for kernel computation. If raw unnormalized values are needed, declare as domain observable x_drift (Tier-0) with explicit units and embedding specification.

**See:** [CHANGELOG.md](CHANGELOG.md#v120-deprecations)

---

## Protocol Compliance

This glossary ensures:
- ✅ Every term has one authoritative meaning
- ✅ Tier boundaries prevent semantic drift
- ✅ "Not to be confused with" prevents common errors
- ✅ Cross-references enable self-service lookup
- ✅ Deprecated terms include migration paths

**Enforcement:** Terms used without glossary entry are protocol violations (NONCONFORMANT).

Last updated: 2026-01-20 | Version: 1.0.0 | Schema: [schemas/glossary.schema.json](schemas/glossary.schema.json)
