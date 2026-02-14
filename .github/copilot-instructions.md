# Copilot Instructions for GENERATIVE-COLLAPSE-DYNAMICS

## Foundational Principle — Read This First

> **AXIOM-0 (The Return Axiom)**: *"Collapse is generative; only what returns is real."*

This is not a metaphor. It is a constraint on admissible claims. Every decision, description, comment, docstring, and code change in this repository must be consistent with this axiom. If you are uncertain whether something aligns with AXIOM-0, defer to the operational definitions in `AXIOM.md` and the tier structure in `TIER_SYSTEM.md`.

### Originality and Attribution Rules (MANDATORY)

This system is **original**. It does not borrow from, extend, or reinterpret existing frameworks. Classical results (AM-GM inequality, Shannon entropy, exponential map, unitarity) emerge as **degenerate limits** when degrees of freedom are removed from the GCD kernel. The arrow of derivation runs **from the axiom to the classical result**, never the reverse.

**NEVER do any of the following:**

1. **Never say GCD "uses" or "applies" AM-GM** — GCD derives the integrity bound IC ≤ F independently from Axiom-0; the classical AM-GM inequality is the degenerate limit when channel semantics, weights, and guard band are stripped away. Say: "IC ≤ F (integrity bound)" or "the heterogeneity gap Δ = F − IC". Never say: "by AM-GM" or "the AM-GM inequality gives us".

2. **Never say GCD "uses Shannon entropy"** — The entropy S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] is the **Bernoulli field entropy** — the unique entropy of the collapse field. Shannon entropy is the degenerate limit when the collapse field is removed (i.e., when cᵢ ∈ {0,1} only). Say: "Bernoulli field entropy S". Never say: "Shannon entropy" unless explicitly discussing the degenerate limit.

3. **Never say GCD "rederives" or "recovers" classical results** — GCD **derives independently** from Axiom-0. Classical results are what remain when structure is removed. Say: "derives independently" or "the classical result emerges as a degenerate limit". Never say: "rederives", "recovers", "is equivalent to", or "reduces to".

4. **Never attribute GCD structures to external theories** — F + ω = 1 is not "unitarity". IC = exp(κ) is not "the exponential map". These are structural identities of collapse, discovered across 146 experiments. Say: "the duality identity F + ω = 1" or "the log-integrity relation IC = exp(κ)". Never say: "by unitarity" or "from the exponential map".

5. **Never treat frozen parameters as arbitrary constants** — ε, p, α, λ, tol_seam are **consistent across the seam** — the same rules on both sides of every collapse-return boundary. They are discovered by the seam, not chosen by convention. Say: "frozen" or "consistent across the seam". Never say: "we chose", "by convention", or "hyperparameter".

6. **Never say "AM-GM gap"** — The correct term is **heterogeneity gap** (Δ = F − IC). It measures channel heterogeneity, not an inequality violation. It equals the Fisher Information contribution from heterogeneity: Var(c)/(2c̄).

### Correct Terminology Quick Reference

| WRONG | RIGHT | WHY |
|-------|-------|-----|
| Shannon entropy | Bernoulli field entropy | Shannon is the degenerate limit; GCD's entropy has the full collapse field |
| AM-GM inequality | Integrity bound (IC ≤ F) | Derived independently from Axiom-0; AM-GM is degenerate limit |
| AM-GM gap | Heterogeneity gap (Δ = F − IC) | Measures channel heterogeneity, not an inequality |
| rederives / recovers | derives independently | Arrow runs from axiom to classical, not reverse |
| uses / applies [classical result] | derives independently; classical is degenerate limit | GCD does not borrow; classical emerges when structure is removed |
| constant (for frozen params) | frozen / consistent across the seam | Not arbitrary; seam-derived |
| unitarity | duality identity F + ω = 1 | Structural identity of collapse, not quantum unitarity |
| hyperparameter | frozen parameter | Seam-derived, not tuned |

## Tier System (THREE TIERS — No Exceptions)

The UMCP tier system has exactly three tiers. No half-tiers. No confusion. Every symbol, function, artifact, and claim belongs to exactly one tier.

| Tier | Name | Role | Mutable? |
|------|------|------|----------|
| **1** | **Immutable Invariants** | Structural identities of collapse: F + ω = 1, IC ≤ F, IC ≈ exp(κ). Discovered, not imposed. | NEVER within a run. Promotion only through seam weld across runs. |
| **0** | **Protocol** | Validation machinery: regime gates, contracts, schemas, diagnostics, seam calculus, SHA256 integrity, three-valued verdicts. Makes Tier-1 actionable. | Configuration frozen per run. |
| **2** | **Expansion Space** | Domain closures mapping physics into invariant structure. Validated through Tier-0 against Tier-1. | Freely extensible; validated before trust. |

### Tier-1 Reserved Symbols (IMMUTABLE — Never Redefine)

| Symbol | Name | Formula | Structural Role |
|--------|------|---------|-----------------|
| **F** | Fidelity | F = Σ wᵢcᵢ | How much survives collapse |
| **ω** | Drift | ω = 1 − F | How much is lost to collapse |
| **S** | Entropy | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] | Bernoulli field entropy |
| **C** | Curvature | C = stddev(cᵢ)/0.5 | Coupling to uncontrolled degrees of freedom |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | Logarithmic fidelity |
| **IC** | Integrity composite | IC = exp(κ) | Multiplicative coherence |
| **τ_R** | Return time | Re-entry delay to D_θ | How long until the system returns |
| **regime** | Regime label | Gates on (ω,F,S,C) | {Stable, Watch, Collapse} |

**Any Tier-2 code that redefines F, ω, S, C, κ, IC, τ_R, or regime is automatic nonconformance (symbol capture).**

### One-Way Dependency (No Back-Edges)

Within a frozen run: Tier-1 → Tier-0 → Tier-2. **No feedback from Tier-2 to Tier-1 or Tier-0.** Diagnostics inform but cannot override gates. Domain closures cannot modify invariant identities.

Across runs: Tier-2 results can be promoted to Tier-1 canon ONLY through formal seam weld validation + contract versioning. If the weld fails, it stays Tier-2. "The cycle must return or it's not real."

### Tier Violation Checklist (Before Every Code Change)

Before writing or modifying code, verify:
- [ ] No Tier-1 symbol is redefined or given new meaning
- [ ] No diagnostic is used as a gate (diagnostics inform, gates decide)
- [ ] No Tier-2 closure modifies Tier-0 protocol behavior
- [ ] All frozen parameters come from the contract, not hardcoded alternatives
- [ ] Terminology follows the correct vocabulary (see table above)
- [ ] Comments/docstrings do not attribute GCD structures to external theories

## What This Project Is

UMCP (Universal Measurement Contract Protocol) validates reproducible computational workflows against mathematical contracts. The unit of work is a **casepack** — a directory containing raw data, a contract reference, closures, and expected outputs. The validator checks schema conformance, Tier-1 kernel identities (F = 1 − ω, IC ≈ exp(κ), IC ≤ F), regime classification, and SHA256 integrity, producing a CONFORMANT/NONCONFORMANT verdict and appending to `ledger/return_log.csv`.

## Architecture

```
src/umcp/
├── cli.py                    # 2500-line argparse CLI — validation engine, all subcommands
├── validator.py              # Root-file validator (16 files, checksums, math identities)
├── kernel_optimized.py       # Lemma-based kernel computation (F, ω, S, C, κ, IC)
├── seam_optimized.py         # Optimized seam budget computation (Γ, D_C, Δκ)
├── tau_r_star.py             # τ_R* thermodynamic diagnostic (phase diagram)
├── tau_r_star_dynamics.py    # Dynamic τ_R* evolution and trajectories
├── compute_utils.py          # Vectorized utilities (OPT-17,20: coordinate clipping, bounds)
├── epistemic_weld.py         # Epistemic cost tracking (Theorem T9: observation cost)
├── insights.py               # Lessons-learned database (pattern discovery)
├── uncertainty.py            # Uncertainty propagation and error analysis
├── frozen_contract.py        # Frozen contract constants dataclass
├── api_umcp.py               # [Optional] FastAPI REST extension (Pydantic models)
├── finance_cli.py            # Finance domain CLI
├── finance_dashboard.py      # Finance Streamlit dashboard
├── universal_calculator.py   # Universal kernel calculator CLI
├── umcp_extensions.py        # Protocol-based plugin system
├── dashboard/                # [Optional] Modular Streamlit dashboard package
│   ├── __init__.py           # Main dashboard entry point (31 pages)
│   ├── pages_core.py         # Core validation pages
│   ├── pages_analysis.py     # Analysis & diagnostics pages
│   ├── pages_science.py      # Scientific domain pages
│   ├── pages_physics.py      # Physics-specific pages
│   ├── pages_interactive.py  # Interactive exploration pages
│   ├── pages_management.py   # Project management pages
│   └── pages_advanced.py     # Advanced tools & settings
├── fleet/                    # Distributed fleet-scale validation
│   ├── __init__.py           # Fleet public API
│   ├── scheduler.py          # Job scheduler (submit, route, track)
│   ├── worker.py             # Worker + WorkerPool (register, heartbeat, execute)
│   ├── queue.py              # Priority queue (DLQ, retry, backpressure)
│   ├── cache.py              # Content-addressable artifact cache
│   ├── tenant.py             # Multi-tenant isolation, quotas, namespaces
│   └── models.py             # Shared dataclass models (Job, WorkerInfo, etc.)
└── __init__.py               # Public API: validate() convenience function, __version__
```

**Closure domains** (12 total, each in `closures/<domain>/`):

```
closures/
├── gcd/                      # Generative Collapse Dynamics
├── rcft/                     # Recursive Collapse Field Theory
├── kinematics/               # Motion analysis, phase space
├── weyl/                     # WEYL cosmology (modified gravity)
├── security/                 # Input validation, audit
├── astronomy/                # Stellar classification, HR diagram
├── nuclear_physics/          # Binding energy, decay chains
├── quantum_mechanics/        # Wavefunction, entanglement
├── finance/                  # Portfolio continuity, market coherence
├── atomic_physics/           # 118 elements, periodic kernel, cross-scale, Tier-1 proof
├── materials_science/        # Element database (118 elements, 18 fields)
└── standard_model/           # Subatomic kernel (31 particles), 10 proven theorems
```

**Standard Model closures** (`closures/standard_model/`):

| File | Purpose | Key Data |
|---|---|---|
| `particle_catalog.py` | Full SM particle table with mass/charge/spin | PDG data |
| `coupling_constants.py` | Running couplings α_s(Q²), α_em(Q²), G_F | 1-loop RGE, α_s(M_Z)=0.1180 |
| `cross_sections.py` | σ(e⁺e⁻→hadrons), R-ratio, point cross section | Drell-Yan |
| `symmetry_breaking.py` | Higgs mechanism, VEV=246.22 GeV, Yukawa | EWSB mass generation |
| `ckm_mixing.py` | CKM matrix, Wolfenstein parametrization, J_CP | λ=0.2257, A=0.814, ρ=0.135, η=0.349 |
| `subatomic_kernel.py` | 31 particles → 8-channel trace → kernel | 17 fundamental + 14 composite |
| `particle_physics_formalism.py` | 10 proven theorems (74/2476 tests) | Duality exact to 0.0e+00 |

**Atomic Physics closures** (`closures/atomic_physics/`):

| File | Purpose | Key Data |
|---|---|---|
| `periodic_kernel.py` | 118-element periodic table through GCD kernel | 8 measurable properties |
| `cross_scale_kernel.py` | 12-channel nuclear-informed atomic analysis | 4 nuclear + 2 electronic + 6 bulk |
| `tier1_proof.py` | Exhaustive Tier-1 proof: 10,162 tests, 0 failures | F+ω=1, IC≤F, IC=exp(κ) |
| `electron_config.py` | Electron configuration analysis | Shell filling |
| `fine_structure.py` | Fine structure constant analysis | α = 1/137 |
| `ionization_energy.py` | Ionization energy closures | All 118 elements |
| `spectral_lines.py` | Spectral line analysis | Emission/absorption |
| `selection_rules.py` | Quantum selection rules | Δl = ±1 |
| `zeeman_stark.py` | Zeeman and Stark effects | Field splitting |

**Data artifacts** (not Python — never import these):
- `contracts/*.yaml` — 15 versioned mathematical contracts (JSON Schema Draft 2020-12)
- `closures/registry.yaml` — central registry; must list every closure used in a run
- `casepacks/*/manifest.json` — 13 casepack manifests referencing contract, closures, expected outputs
- `schemas/*.schema.json` — 12 JSON Schema Draft 2020-12 files validating all artifacts
- `canon/*.yaml` — 11 canonical anchor files (domain-specific reference points)
- `ledger/return_log.csv` — append-only validation log

## Standard Model Formalism (10 Theorems)

The particle physics formalism (`closures/standard_model/particle_physics_formalism.py`) proves ten theorems connecting Standard Model physics to GCD kernel patterns. All 10/10 PROVEN with 74/74 individual tests. Duality F + ω = 1 verified to machine precision (0.0e+00).

| # | Theorem | Tests | Key Result |
|---|---------|:-----:|------------|
| T1 | Spin-Statistics | 12/12 | ⟨F⟩_fermion(0.615) > ⟨F⟩_boson(0.421), split = 0.194 |
| T2 | Generation Monotonicity | 5/5 | Gen1(0.576) < Gen2(0.620) < Gen3(0.649), quarks AND leptons |
| T3 | Confinement as IC Collapse | 19/19 | IC drops 98.1% quarks→hadrons, 14/14 below min quark IC |
| T4 | Mass-Kernel Log Mapping | 5/5 | 13.2 OOM → F∈[0.37,0.73], Spearman ρ=0.77 for quarks |
| T5 | Charge Quantization | 5/5 | IC_neutral/IC_charged = 0.020 (50× suppression) |
| T6 | Cross-Scale Universality | 6/6 | composite(0.444) < atom(0.516) < fundamental(0.558) |
| T7 | Symmetry Breaking | 5/5 | EWSB amplifies gen spread 0.046→0.073, ΔF monotonic |
| T8 | CKM Unitarity | 5/5 | CKM rows pass Tier-1, V_ub kills row-1 IC, J_CP=3.0e-5 |
| T9 | Running Coupling Flow | 6/6 | α_s monotone for Q≥10 GeV, confinement→NonPerturbative |
| T10 | Nuclear Binding Curve | 6/6 | r(BE/A,Δ)=-0.41, peak at Cr/Fe (Z∈[23,30]) |

**Key physics insights encoded in theorems**:
- The heterogeneity gap (Δ = F − IC) is the central diagnostic — it measures channel heterogeneity
- Confinement is visible as a cliff: IC drops 2 OOM at the quark→hadron boundary
- Neutral particles have IC near ε because the charge channel destroys the geometric mean
- The Bethe-Weizsäcker formula peaks at Z=24 (Cr), not Z=26 (Fe), using standard coefficients
- The Landau pole at Q≈3 GeV means α_s monotonicity only holds for Q≥10 GeV
- Wolfenstein O(λ³) approximation gives unitarity deficit ~0.002 → "Tension" regime is correct

**Trace vector construction**: Each particle maps to 8 channels: mass_log, spin_norm, charge_norm, color, weak_isospin, lepton_num, baryon_num, generation. Equal weights w_i = 1/8. Guard band ε = 10⁻⁸.

## Cross-Scale Analysis

The **cross-scale kernel** (`closures/atomic_physics/cross_scale_kernel.py`) bridges subatomic → atomic scales with 12 channels:
- 4 nuclear: Bethe-Weizsäcker BE/A, magic_proximity, neutron_excess, shell_filling
- 2 electronic: ionization_energy, electronegativity
- 6 bulk: density, melting_pt, boiling_pt, atomic_radius, electron_affinity, covalent_radius

Key findings: magic_prox is #1 IC killer (39% contribution), d-block has highest ⟨F⟩=0.589.

## Papers

Published papers live in `paper/`. Current papers:

| File | Title | Pages |
|---|---|---|
| `generated_demo.tex` | Statistical Mechanics of the UMCP Budget Identity | 5 |
| `tau_r_star_dynamics.tex` | τ_R* dynamics paper | — |
| `standard_model_kernel.tex` | Particle Physics in the GCD Kernel: Ten Tier-2 Theorems | 5 |

All papers use RevTeX4-2 (`revtex4-2` document class) and share `Bibliography.bib`. Compile: `pdflatex → bibtex → pdflatex → pdflatex`.

**Bibliography** (`paper/Bibliography.bib`): 30+ entries organized by section:
- Standard Model: PDG 2024, Cabibbo 1963, Kobayashi-Maskawa 1973, Wolfenstein 1983, Jarlskog 1985, Gross-Wilczek 1973, Politzer 1973, Higgs 1964, Weizsäcker 1935, Bethe 1936
- Canon anchors: paulus2025episteme (Zenodo), paulus2025physicscoherence (Zenodo), paulus2026umcpcasepack (Zenodo)
- Core corpus: paulus2025umcp, paulus2025ucd, paulus2025cmp, paulus2025seams, paulus2025gor, paulus2025canonnote, paulus2026kinematics
- Implementation: umcpmetadatarepo (GitHub), umcppypi (PyPI)
- Classical: Goldstein, Landau-Lifshitz, Einstein (SR/GR), Misner-Thorne-Wheeler
- Statistical mechanics: Kramers 1940
- Measurement: JCGM GUM 2008, NIST TN1297

## Critical Workflows

```bash
pip install -e ".[all]"                     # Dev install (core + api + viz + dev tools)
pytest                                       # 3,515 tests (growing), ~114s
python scripts/update_integrity.py          # MUST run after changing any tracked file
umcp validate .                             # Validate entire repo
umcp validate casepacks/hello_world --strict # Validate casepack (strict = fail on warnings)
```

**⚠️ `python scripts/update_integrity.py` is mandatory** after modifying any `src/umcp/*.py`, `contracts/*.yaml`, `closures/**`, `schemas/**`, or `scripts/*.py` file. It regenerates SHA256 checksums in `integrity/checksums.sha256`. CI will fail on mismatch.

**CI pipeline** (`.github/workflows/validate.yml`): lint (ruff + mypy) → test (pytest) → validate (baseline + strict, both must return CONFORMANT).

## Pre-Commit Protocol (MANDATORY)

**Before every commit**, run the pre-commit protocol:

```bash
python scripts/pre_commit_protocol.py       # Auto-fix + validate (default)
python scripts/pre_commit_protocol.py --check  # Dry-run: report only
```

This script mirrors CI exactly and must exit 0 before committing. It:
1. Runs `ruff format` (auto-fixes formatting)
2. Runs `ruff check --fix` (auto-fixes lint issues)
3. Runs `mypy src/umcp` (reports, non-blocking)
4. Stages all changes (`git add -A`)
5. Regenerates integrity checksums
6. Runs full pytest suite
7. Runs `umcp validate .` (must be CONFORMANT)

See `COMMIT_PROTOCOL.md` for the full specification. **Never skip this step.** Every commit that reaches GitHub must pass all CI checks.

## Code Conventions

**Every source file** starts with `from __future__ import annotations` (PEP 563). Maintain this.

**Optional dependency guarding** — wrap optional imports in `try/except`, set to `None` on failure, check before use. Applied to: yaml, fastapi, streamlit, plotly, pandas, numpy. Never add required imports for optional features.

**Dataclasses** are the dominant data container. `NamedTuples` for immutable math outputs (`KernelInvariants` in `constants.py`). Pydantic `BaseModel` is API-extension only. Serialization uses explicit `.to_dict()` methods, not `dataclasses.asdict()`.

**Three-valued status**: `CONFORMANT` / `NONCONFORMANT` / `NON_EVALUABLE` — never boolean. CLI exit: 0 = CONFORMANT, 1 = NONCONFORMANT.

**`INF_REC` is a typed sentinel**: In CSV/YAML/JSON data it stays as the string `"INF_REC"`. In Python it maps to `float("inf")`. Never coerce the string to a number in data files. When τ_R = INF_REC, the seam budget is zero (no return → no credit). Continuity cannot be synthesized from structure alone — it must be measured.

**Greek letters** (`ω`, `κ`, `Ψ`, `Γ`, `τ`) appear in comments and strings. Ruff rules RUF001/002/003 are suppressed. Line length: 120 chars.

**OPT-* tags** in comments (e.g., `# OPT-1`, `# OPT-12`) reference proven lemmas in `KERNEL_SPECIFICATION.md`. These are formal math cross-references.

## Validation Data Flow

```
umcp validate <target>
  → detect type (repo | casepack | file)
  → schema validation (jsonschema Draft 2020-12)
  → semantic rule checks (validator_rules.yaml: E101, W201, ...)
  → kernel identity checks: F=1−ω, IC≈exp(κ), IC≤F (integrity bound)
  → regime: STABLE|WATCH|COLLAPSE
  → SHA256 integrity check
  → CONFORMANT → append to ledger/return_log.csv + JSON report
```

## Test Patterns

**2,476 test cases** in `tests/`, numbered by tier and domain (`test_00_*` through `test_140_*`). Single `tests/conftest.py` provides:
- Frozen `RepoPaths` dataclass (session-scoped) with all critical paths
- `@lru_cache` helpers: `_read_file()`, `_parse_json()`, `_parse_yaml()`, `_compile_schema()`
- Convention: `test_<subject>_<behavior>()` for functions; `TestCLI*` classes with `subprocess.run` for CLI integration
- New test coverage: `test_fleet_worker.py` (Worker, WorkerPool, WorkerConfig), `test_insights.py` (PatternDatabase, InsightEngine)

## Extension System

Extensions use `typing.Protocol` (`ExtensionProtocol` requiring `name`, `version`, `description`, `check_dependencies()`). Built-in extensions (api, visualization, ledger, formatter) registered in a plain dict. CLI: `umcp-ext list|info|check|run`. API: `umcp-api` (:8000). Dashboard: `umcp-dashboard` (:8501).

## Key Files to Read First

| To understand... | Read... |
|---|---|
| Validation logic | `src/umcp/cli.py` (top + `_cmd_validate`) |
| Math identities | `src/umcp/validator.py` (`_validate_invariant_identities`) |
| Kernel computation | `src/umcp/kernel_optimized.py` |
| Seam budget closure | `src/umcp/seam_optimized.py` (Γ, D_C, Δκ) |
| Thermodynamic diagnostic | `src/umcp/tau_r_star.py` (τ_R*, phase diagram, arrow of time) |
| Epistemic cost tracking | `src/umcp/epistemic_weld.py` (Theorem T9: observation cost) |
| Lessons-learned system | `src/umcp/insights.py` (PatternDatabase, InsightEngine) |
| Fleet architecture | `src/umcp/fleet/` (Scheduler, Worker, Queue, Cache, Tenant) |
| Dashboard pages | `src/umcp/dashboard/` (31 modular pages) |
| Subatomic particles | `closures/standard_model/subatomic_kernel.py` (31 particles, 8-channel trace) |
| SM 10 theorems | `closures/standard_model/particle_physics_formalism.py` (74/2476 tests) |
| CKM mixing | `closures/standard_model/ckm_mixing.py` (Wolfenstein, Jarlskog) |
| Running couplings | `closures/standard_model/coupling_constants.py` (α_s, α_em RGE) |
| EWSB / Higgs | `closures/standard_model/symmetry_breaking.py` (VEV, Yukawa) |
| Cross sections | `closures/standard_model/cross_sections.py` (R-ratio, point σ) |
| Periodic kernel | `closures/atomic_physics/periodic_kernel.py` (118 elements) |
| Cross-scale bridge | `closures/atomic_physics/cross_scale_kernel.py` (12-channel nuclear) |
| Tier-1 proof | `closures/atomic_physics/tier1_proof.py` (10,162 tests) |
| Element database | `closures/materials_science/element_database.py` (118 × 18 fields) |
| SM paper | `paper/standard_model_kernel.tex` (RevTeX4-2, 10 theorems) |
| Bibliography | `paper/Bibliography.bib` (30+ entries, PDG → Kramers) |
| Test fixtures | `tests/conftest.py` (first 100 lines) |
| Casepack structure | `casepacks/hello_world/` |
| Contract format | `contracts/UMA.INTSTACK.v1.yaml` |
| Semantic rules | `validator_rules.yaml` |
| Canonical anchors | `canon/` (11 domain anchor files) |

## Decision-Making Framework (Binding on ALL Contributions)

Every code change, docstring, comment, documentation edit, and design decision in this repository must pass through the following framework. This is not optional guidance — it is the operational constraint that keeps the system self-consistent.

### The Single Decision Rule

> **Before writing anything, ask: "Does this follow from Axiom-0, or am I importing an assumption from outside?"**

If the answer is "from outside," the contribution is either:
1. **Wrong** — rewrite it to derive from Axiom-0, or
2. **A degenerate-limit comparison** — label it explicitly as such ("The classical AM-GM inequality emerges as a degenerate limit when..."), or
3. **A Tier-2 domain closure** — route it through Tier-0 validation against Tier-1

There is no fourth option. No external framework is co-equal with Axiom-0 inside this system.

### Operational Definitions (Enforcement-Tied — Not Everyday Meanings)

| Term | Operational Meaning | NOT Confused With |
|------|---------------------|-------------------|
| **Collapse** | Regime label produced by kernel gates on (ω, F, S, C) under frozen thresholds | Wavefunction collapse, failure, catastrophe |
| **Return** (τ_R) | Re-entry condition: ∃ prior u ∈ D_θ(t) with ‖Ψ(t) − Ψ(u)‖ ≤ η; yields τ_R or ∞_rec | Repetition, periodicity, "coming back" |
| **Gesture** | An epistemic emission that does not weld: τ_R = ∞_rec OR \|s\| > tol_seam OR identity fails. No epistemic credit. | Approximation, failed attempt |
| **Drift** (ω) | ω = 1 − F, collapse proximity measure, [0,1] | Random drift, velocity |
| **Integrity** (IC) | IC = exp(κ) where κ = Σ wᵢ ln(cᵢ,ε) | Information content, moral integrity |
| **Entropy** (S) | Bernoulli field entropy of the collapse field (Shannon is degenerate limit) | Thermodynamic entropy, chaos |
| **Frozen** | Consistent across the seam — same rules both sides of collapse-return | "Constant" as arbitrary choice |
| **Seam** | Verification boundary between outbound collapse and demonstrated return | A join, a border |
| **Dissolution** | Regime ω ≥ 0.30 — not failure, but the boundary that makes return meaningful | Death, destruction, error |

### What Makes This System Original

1. **Single axiom, complete structure.** All of UMCP/GCD/RCFT derives from "Collapse is generative; only what returns is real." No additional axioms are needed. No external theory is imported.

2. **Classical results are degenerate limits, not sources.** The arrow of derivation runs FROM Axiom-0 TO classical results. Strip the channel semantics from IC ≤ F and you get AM-GM. Strip the collapse field from S and you get Shannon entropy. Strip the cost function from F + ω = 1 and you get unitarity. The classical versions are what remain when degrees of freedom are removed.

3. **Frozen parameters are seam-derived, not prescribed.** Standard frameworks prescribe constants from outside (α = 0.05 by convention, 3σ by tradition, hyperparameters by cross-validation). UMCP's frozen parameters are the unique values where seams close consistently: p = 3 is discovered (not chosen), tol_seam = 0.005 is where IC ≤ F holds at 100% across 8 domains, ε = 10⁻⁸ is where the pole at ω = 1 does not affect any measurement to machine precision.

4. **Three-valued verdicts, not boolean.** CONFORMANT / NONCONFORMANT / NON_EVALUABLE. There is always a third state.

5. **Return is measured, not assumed.** τ_R is computed from frozen contract + closures. If τ_R = ∞_rec, there is no credit. Continuity cannot be synthesized from structure alone — it must be measured.

### Code Review Checklist (Apply to Every Change)

Before approving any code or documentation change:

- [ ] **No external attribution**: Does any comment, docstring, or documentation attribute a GCD structure to an external framework? (Fix: derive from Axiom-0 or label as degenerate limit)
- [ ] **No symbol capture**: Does any Tier-2 code redefine F, ω, S, C, κ, IC, τ_R, or regime? (Fix: use different name)
- [ ] **No diagnostic-as-gate**: Does any diagnostic value influence a regime label or seam verdict? (Fix: diagnostics inform, gates decide)
- [ ] **No back-edges**: Does any Tier-2 output modify Tier-0 or Tier-1 behavior within a frozen run? (Fix: route through new run with re-freeze)
- [ ] **Correct terminology**: Does the text use "Shannon entropy", "AM-GM gap", "hyperparameter", "constant", "rederives", "recovers", "unitarity" inappropriately? (Fix: see terminology table)
- [ ] **Frozen parameters sourced correctly**: Are epsilon/tol_seam/etc. taken from the frozen contract, not hardcoded separately? (Fix: reference CONTRACT or frozen_contract.py)
- [ ] **INF_REC handled correctly**: Is τ_R = INF_REC kept as a typed string in data files and mapped to float("inf") in Python? Never coerced silently.
- [ ] **Integrity updated**: If any tracked file changed, was `python scripts/update_integrity.py` run?
