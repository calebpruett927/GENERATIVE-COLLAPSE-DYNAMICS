# UMCP Term Index

**Alphabetical term lookup with file locations**  
**Version:** 1.0.0  
**Last Updated:** 2026-01-20

Quick alphabetical index of all protocol terms with definitions and file locations. For symbol-specific lookup, see [Symbol Index](SYMBOL_INDEX.md). For complete structured definitions, see [Glossary](GLOSSARY.md).

---

## A

**Anchors** (Canon) - Tier-tagged authoritative symbol definitions
- Files: [canon/anchors.yaml](canon/anchors.yaml), [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml), [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml)
- Tests: [tests/test_100_gcd_canon.py](tests/test_100_gcd_canon.py)
- Glossary: [GLOSSARY.md](GLOSSARY.md#tier-0-interface-and-measurement)

**Admissibility** - Protocol conformance status (conformant vs nonconformant)
- Definition: [GLOSSARY.md](GLOSSARY.md#nonconformance)
- Validation: [src/umcp/validator.py](src/umcp/validator.py)
- Results: `validator.result.json`

**Augmentation (RCFT Principle)** - Tier-2 extends without override
- Principle: [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) → P-RCFT-0
- Theory: [docs/rcft_theory.md](docs/rcft_theory.md#augmentation-principle)
- Glossary: [GLOSSARY.md](GLOSSARY.md#tier-2-rcft-overlay-extensions)

**Axiom** (Core) - "What Returns Through Collapse Is Real"
- Statement: [AXIOM.md](AXIOM.md#core-axiom)
- no_return_no_credit: true (enforcement)
- Weld gate: no ∞_rec → FAIL

---

## B

**Bounded Trace** (Ψ(t)) - Embedded state vector [0,1]^n
- Symbol: Ψ, Psi
- Definition: [GLOSSARY.md](GLOSSARY.md#bounded-trace-psi-ψt)
- Canon: [canon/anchors.yaml](canon/anchors.yaml)
- Files: `derived/trace.csv` in all CasePacks

**Budget Delta** (Δκ_budget) - Closure-driven seam accounting
- Formula: R·τ_R - (D_ω + D_C)
- Definition: [GLOSSARY.md](GLOSSARY.md#budget-delta-δκ_budget)
- Contract: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → weld_accounting
- Closures: [closures/registry.yaml](closures/registry.yaml)

**Boundary State** (Typed) - ∞_rec, UNIDENTIFIABLE, etc.
- Convention: [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) → typed_boundaries
- Encoding: INF_REC in files
- Theory: [AXIOM.md](AXIOM.md#typed-boundary-conventions)

---

## C

**Canon** - Tier-tagged authoritative specifications
- Root: [canon/](canon/)
- Schemas: [schemas/canon.anchors.schema.json](schemas/canon.anchors.schema.json)
- Versions: UMCP.CANON.v1, UMCP.GCD.v1, UMCP.RCFT.v1

**CasePack** - Reproducible computational bundle
- Structure: manifest + observables + contract + closures + outputs
- Examples: [casepacks/hello_world/](casepacks/hello_world/), [casepacks/UMCP-REF-E2E-0001/](casepacks/UMCP-REF-E2E-0001/)
- Validation: `umcp validate casepacks/<name>`

**Closure** - Auxiliary specification for seam computation
- Definition: [GLOSSARY.md](GLOSSARY.md#closure)
- Registry: [closures/registry.yaml](closures/registry.yaml)
- Docs: [closures/README.md](closures/README.md)
- Implementations: [closures/](closures/) (Python files)

**Collapse** (Regime) - Critical system state (ω ≥ 0.30)
- Gates: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → regime_gates
- Axiom: AX-0 "Collapse is generative"
- Theory: [docs/rcft_theory.md](docs/rcft_theory.md#collapse-dynamics)

**Conformance** - Protocol adherence status
- Conformant: All requirements met
- Nonconformant: Structural violation
- Definition: [GLOSSARY.md](GLOSSARY.md#nonconformance)
- Validation: [src/umcp/validator.py](src/umcp/validator.py)

**Contract** - Frozen parameter and threshold specification
- Root: [contracts/](contracts/)
- Primary: [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml)
- Schema: [schemas/contract.schema.json](schemas/contract.schema.json)
- Tests: [tests/test_102_gcd_contract.py](tests/test_102_gcd_contract.py)

**Curvature** (C) - Component dispersion proxy
- Symbol: C, Tier-1
- Formula: √(Σ w_i (c_i - c̄)²)
- Definition: [GLOSSARY.md](GLOSSARY.md#curvature-proxy-ct)
- Canon: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)
- Not: Capacitance (use C_cap)

---

## D

**Diagnostic** - Insight without gate authority (Tier-2)
- Boundary: [GLOSSARY.md](GLOSSARY.md#diagnostic-vs-gate)
- Rule: Cannot decide labels or admissibility
- Examples: Equator residuals, sensitivity analysis
- Files: Separate from invariants.csv

**Drift** (ω) - Collapse proximity metric
- Symbol: ω (omega), Tier-1
- Formula: ω = 1 - F
- Definition: [GLOSSARY.md](GLOSSARY.md#drift-ωt)
- Not: Angular frequency (use Ω_freq)
- Thresholds: Stable < 0.038, Collapse ≥ 0.30

**Dissipation Terms** (D_ω, D_C) - Weld budget components
- D_ω: Drift dissipation via Γ(ω; p, ε)
- D_C: Curvature dissipation (typically αC)
- Closures: [closures/gamma.default.v1.yaml](closures/gamma.default.v1.yaml)
- Definition: [GLOSSARY.md](GLOSSARY.md#tier-15-seam-and-weld-symbols)

---

## E

**EID** - Artifact structural fingerprint (element counts)
- Definition: [GLOSSARY.md](GLOSSARY.md#eid-artifact-structural-fingerprint)
- Not: Weld ID (seam identity) or Case ID
- Docs: [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md#eid-conventions)

**Embedding** (N_K) - Observable → bounded trace mapping
- Symbol: N_K, Tier-0
- Definition: [GLOSSARY.md](GLOSSARY.md#embedding--normalization-n_k)
- Spec: `embedding.yaml` in CasePacks
- Output: Ψ(t) ∈ [0,1]^n

**Entropy** (S) - Shannon functional on trace
- Symbol: S, Tier-1
- Formula: -Σ w_i [c_i ln(c_i) + (1-c_i) ln(1-c_i)]
- Definition: [GLOSSARY.md](GLOSSARY.md#entropy-functional-st)
- Not: Thermodynamic S_th (requires Tier-2 mapping)

**Epsilon-clipping** (ε) - Log-safety threshold
- Value: Typically 1e-10 (not machine epsilon)
- Definition: [GLOSSARY.md](GLOSSARY.md#epsilon-clipping--log-safety-ε)
- Applied before: κ and S computations
- Generates: Ψ_ε(t) log-safe trace

---

## F

**Fidelity** (F) - Weighted trace average
- Symbol: F, Tier-1
- Formula: F = Σ w_i c_i = 1 - ω
- Definition: [GLOSSARY.md](GLOSSARY.md#fidelity-ft)
- Threshold: Collapse when F < 0.75

**Fractal Dimension** (D_fractal) - RCFT geometric metric
- Symbol: D_f, Tier-2
- Domain: [1, 3]
- Definition: [GLOSSARY.md](GLOSSARY.md#fractal-dimension-d_fractal)
- Implementation: [closures/rcft/fractal_dimension.py](closures/rcft/fractal_dimension.py)
- Regimes: Smooth < 1.2, Wrinkled 1.2-1.8, Turbulent ≥ 1.8

**Freeze** - Lock interface and closures before compute
- Procedure: /ingest → /freeze → /compute
- Definition: [GLOSSARY.md](GLOSSARY.md#freeze)
- Enforcement: Required for conformance
- Docs: [docs/production_deployment.md](docs/production_deployment.md#freeze-procedure)

---

## G

**Gate** - Binding classification authority
- Types: Regime gates, weld gate
- Definition: [GLOSSARY.md](GLOSSARY.md#diagnostic-vs-gate)
- vs Diagnostic: Gates decide, diagnostics inform
- Examples: Stable/Watch/Collapse, PASS/FAIL

**GCD** - Generative Collapse Dynamics (Tier-1 framework)
- Canon: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)
- Contract: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml)
- Axioms: AX-0, AX-1, AX-2
- Reserved: 13 Tier-1 symbols (frozen)

**Glossary** - Structured term definitions (this system)
- Main: [GLOSSARY.md](GLOSSARY.md)
- Symbol index: [SYMBOL_INDEX.md](SYMBOL_INDEX.md)
- Term index: This file
- Required fields: Tier, definition, "not confused with", location

---

## H

**Hash Ledger** - SHA-256 integrity tracking
- File: [integrity/sha256.txt](integrity/sha256.txt) (200 files)
- Definition: [GLOSSARY.md](GLOSSARY.md#sha256-integrity-ledger)
- Verification: `umcp validate` checks hashes
- Required: For conformance and reproducibility

**Horizon** (H_rec) - Return search window limit
- Parameter: Frozen in contract/closures
- Definition: [GLOSSARY.md](GLOSSARY.md#return-machinery)
- Effect: Bounds return candidate search
- Typing: Defines when τ_R = ∞_rec

---

## I

**Integrity Composite** (IC) - Geometric aggregate
- Symbol: IC, Tier-1
- Formula: IC = exp(κ)
- Definition: [GLOSSARY.md](GLOSSARY.md#integrity-composite-ic)
- Identity: κ = ln(IC) exactly
- Critical: When min_t IC(t) below threshold

**Invariants** - Tier-1 reserved quantities
- Count: 13 in GCD (ω, F, S, C, τ_R, κ, IC, etc.)
- Files: `outputs/invariants.csv` in CasePacks
- Canon: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)
- Tests: [tests/test_100_gcd_canon.py](tests/test_100_gcd_canon.py)

---

## K

**Kappa** (κ) - Log-integrity
- Symbol: κ (kappa), Tier-1
- Formula: κ = Σ w_i ln(c_i,ε)
- Definition: [GLOSSARY.md](GLOSSARY.md#log-integrity-κt)
- Identity: κ = ln(IC)
- Weld: Δκ = κ_1 - κ_0 (ledger)

---

## L

**Ledger Delta** (Δκ_ledger) - Identity-based seam accounting
- Formula: κ_1 - κ_0 (exact)
- Definition: [GLOSSARY.md](GLOSSARY.md#ledger-delta-δκ_ledger)
- vs Budget: Ledger is identity, budget uses closures
- Residual: s = Δκ_budget - Δκ_ledger

**Log-safety** - See Epsilon-clipping

---

## M

**Manifest** - CasePack reproducibility index
- File: `manifest.yaml` in all CasePacks
- Definition: [GLOSSARY.md](GLOSSARY.md#manifest-manifestyaml)
- Schema: [schemas/manifest.schema.json](schemas/manifest.schema.json)
- Contents: Files, hashes, versions, IDs

**Missingness Policy** - Handling missing data
- Rule: Never silent repair
- Definition: [GLOSSARY.md](GLOSSARY.md#missingness-policy)
- Options: Error or explicit flagged imputation
- Frozen: Before compute (Tier-0)

---

## N

**Nonconformance** - Protocol violation status
- Definition: [GLOSSARY.md](GLOSSARY.md#nonconformance)
- Not: Weld FAIL (valid outcome)
- Causes: Missing freeze, symbol capture, post-hoc edits
- Output: Marked "do not interpret"

**Norm** (||·||) - Distance metric in Ψ-space
- Default: L2 norm
- Definition: [GLOSSARY.md](GLOSSARY.md#return-machinery)
- Used: Return matching U_θ(t)
- Frozen: Cannot change post-freeze

---

## O

**Observable** (x(t)) - Unitful measurement
- Symbol: x(t), Tier-0
- Definition: [GLOSSARY.md](GLOSSARY.md#observable-xt)
- Files: `observables.yaml` in CasePacks
- Becomes: Ψ(t) after embedding

**Overlay** (Tier-2) - Domain-specific augmentation
- Definition: [GLOSSARY.md](GLOSSARY.md#tier-2-rcft-overlay-extensions)
- Rule: Cannot override Tier-1
- Examples: RCFT metrics, probability models, controllers
- Subordinate: To kernel and seam results

---

## P

**PASS/FAIL** - Weld gate decision
- Type: Categorical, binding
- Definition: [GLOSSARY.md](GLOSSARY.md#pass--fail-weld-gate)
- Conditions: Finite τ_R, |s| ≤ tol, identity check
- Files: `outputs/welds.csv` status column

**Provenance** - Code and environment identity
- Code: [integrity/code_version.txt](integrity/code_version.txt) (git commit)
- Environment: Container/lockfile hash
- Definition: [GLOSSARY.md](GLOSSARY.md#artifact-integrity)
- Required: For reproducibility

---

## R

**RCFT** - Recursive Collapse Field Theory (Tier-2)
- Canon: [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml)
- Contract: [contracts/RCFT.INTSTACK.v1.yaml](contracts/RCFT.INTSTACK.v1.yaml)
- Theory: [docs/rcft_theory.md](docs/rcft_theory.md)
- Metrics: D_fractal, Ψ_recursive, λ_pattern, Θ_phase

**Receipt** - Compact audit row (SS1m format)
- Definition: [GLOSSARY.md](GLOSSARY.md#meta-reporting-and-governance)
- Files: [expected/seam_receipt.json](expected/seam_receipt.json)
- Contents: Identity + closure + integrity metadata

**Recursive Field** (Ψ_recursive) - RCFT memory metric
- Symbol: Ψ_r, Tier-2
- Formula: Σ α^n · Ψ_n (exponential decay)
- Definition: [GLOSSARY.md](GLOSSARY.md#recursive-field-ψ_recursive)
- Implementation: [closures/rcft/recursive_field.py](closures/rcft/recursive_field.py)

**Regime** - System state classification (Stable/Watch/Collapse)
- Labels: Tier-1 gate outputs
- Definition: [GLOSSARY.md](GLOSSARY.md#regime-labels-stable--watch--collapse)
- Thresholds: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml)
- Files: `outputs/regimes.csv`

**Registry** (Closure) - Governance artifact for closures
- File: [closures/registry.yaml](closures/registry.yaml)
- Definition: [GLOSSARY.md](GLOSSARY.md#closure-registry-closuresyaml)
- Required: For weld admissibility
- Schema: [schemas/closures_registry.schema.json](schemas/closures_registry.schema.json)

**Residual** (s) - Weld reconciliation term
- Formula: s = Δκ_budget - Δκ_ledger
- Definition: [GLOSSARY.md](GLOSSARY.md#residual-seam-residual-s)
- Gate: |s| ≤ tol_seam for PASS
- Not: Diagnostic residuals (Tier-2)

**Resonance Pattern** - RCFT oscillation analysis
- Symbols: λ_pattern (wavelength), Θ_phase (phase)
- Definition: [GLOSSARY.md](GLOSSARY.md#resonance-pattern-λ_pattern-θ_phase)
- Method: FFT analysis
- Types: Standing, Mixed, Traveling

**Return Time** (τ_R) - Re-entry delay
- Symbol: τ_R (tau_R), Tier-1
- Definition: [GLOSSARY.md](GLOSSARY.md#re-entry-delay--return-time-τ_rt)
- Domain: ℕ ∪ {∞_rec}
- Typed: ∞_rec when no return in horizon
- Implementation: [closures/tau_R_compute.py](closures/tau_R_compute.py)

---

## S

**Seam** - Transition context for continuity evaluation
- Definition: [GLOSSARY.md](GLOSSARY.md#seam)
- Endpoints: t_0 → t_1
- Becomes weld: When computed with closures
- Examples: PRE→POST canon seams

**Schema** - JSON Schema validation specifications
- Directory: [schemas/](schemas/)
- Count: 10+ schema files
- Examples: contract, manifest, closures, canon
- Validation: `umcp validate` uses schemas

**Stable** (Regime) - Optimal system state
- Gates: ω < 0.038, F > 0.90, S < 0.15, C < 0.14
- Definition: [GLOSSARY.md](GLOSSARY.md#regime-labels-stable--watch--collapse)
- Files: `outputs/regimes.csv`

**Symbol** - Reserved mathematical identifier
- Index: [SYMBOL_INDEX.md](SYMBOL_INDEX.md)
- Canon: [canon/](canon/) anchor files
- Tiers: 0 (interface), 1 (GCD), 1.5 (seam), 2 (RCFT)
- Collision: Must disambiguate (see Symbol Index)

---

## T

**Tier** - Hierarchical classification system
- Tier-0: Measurement interface (observables, embedding)
- Tier-1: GCD reserved symbols (13 invariants)
- Tier-1.5: Seam/weld accounting
- Tier-2: RCFT overlay (augmentation)
- Meta: Governance and reporting

**Tolerance** - Frozen gate threshold
- Types: tol_seam, tol_id, η (return tolerance)
- Definition: [GLOSSARY.md](GLOSSARY.md#seam-and-weld-calculus)
- Contract: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml)
- Frozen: Cannot adjust post-freeze

**Trace** - See Bounded Trace (Ψ(t))

**Typed Boundary** - Explicit non-numeric states
- Examples: ∞_rec, UNIDENTIFIABLE
- Encoding: INF_REC in files
- Convention: [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml)
- Theory: [AXIOM.md](AXIOM.md#typed-boundary-conventions)

---

## U

**UMCP** - Universal Measurement Contract Protocol
- Canon: [canon/anchors.yaml](canon/anchors.yaml)
- Contract: [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml)
- Docs: [README.md](README.md), [AXIOM.md](AXIOM.md)
- Package: [pyproject.toml](pyproject.toml) (v1.3.2)

**Unknown — hint** - Gap reporting convention
- Format: "unknown — <cause>; hint: <action>"
- Definition: [GLOSSARY.md](GLOSSARY.md#unknown--hint)
- Purpose: Acknowledge gaps without inventing data
- Files: Reports, Gaps sections

---

## V

**Validator** - Protocol conformance checker
- CLI: `umcp validate [path]`
- Source: [src/umcp/validator.py](src/umcp/validator.py)
- Output: `validator.result.json`
- Tests: [tests/test_10_canon_contract_closures_validate.py](tests/test_10_canon_contract_closures_validate.py)

**Versioning** - Artifact version tracking
- Code: [integrity/code_version.txt](integrity/code_version.txt)
- Package: [pyproject.toml](pyproject.toml) version field
- Canon: id fields (UMCP.GCD.v1, etc.)
- Docs: [CHANGELOG.md](CHANGELOG.md)

---

## W

**Watch** (Regime) - Intermediate system state
- Gates: Between Stable and Collapse thresholds
- Definition: [GLOSSARY.md](GLOSSARY.md#regime-labels-stable--watch--collapse)
- Purpose: Early warning of instability

**Weights** (w_i) - Component importance
- Symbol: w_i, Tier-0
- Constraint: w_i ≥ 0, Σ w_i = 1
- Definition: [GLOSSARY.md](GLOSSARY.md#weights-w_i)
- Files: `weights.csv` in CasePacks

**Weld** - Seam with PASS/FAIL gate
- Definition: [GLOSSARY.md](GLOSSARY.md#weld)
- Accounting: Ledger + budget + residual
- Output: `outputs/welds.csv`
- Tests: [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

**Weld ID** - Seam event identifier
- Format: String (e.g., "PRE→POST", "W-2025-12-31")
- Definition: [GLOSSARY.md](GLOSSARY.md#seam-and-weld-calculus)
- Not: EID, Case ID, commit hash
- Files: Weld rows, receipts

---

## Quick Navigation

**By Document Type:**
- Definitions: [GLOSSARY.md](GLOSSARY.md)
- Symbols: [SYMBOL_INDEX.md](SYMBOL_INDEX.md)
- Theory: [docs/rcft_theory.md](docs/rcft_theory.md)
- Architecture: [docs/interconnected_architecture.md](docs/interconnected_architecture.md)
- Implementation: [docs/python_coding_key.md](docs/python_coding_key.md)

**By Task:**
- Validation: [src/umcp/validator.py](src/umcp/validator.py)
- CasePack creation: [docs/quickstart.md.](docs/quickstart.md.)
- Extension development: [QUICKSTART_EXTENSIONS.md](QUICKSTART_EXTENSIONS.md)
- Testing: [tests/](tests/)

**By Tier:**
- Tier-0: [canon/anchors.yaml](canon/anchors.yaml)
- Tier-1: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)
- Tier-2: [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml)

---

**Last Updated:** 2026-01-20  
**Version:** 1.0.0  
**Maintenance:** Update when new terms introduced or locations change
