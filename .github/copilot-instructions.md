# Copilot Instructions for GENERATIVE-COLLAPSE-DYNAMICS

## What This Project Is

UMCP (Universal Measurement Contract Protocol) validates reproducible computational workflows against mathematical contracts. The unit of work is a **casepack** — a directory containing raw data, a contract reference, closures, and expected outputs. The validator checks schema conformance, Tier-1 kernel identities (F = 1 − ω, IC ≈ exp(κ), IC ≤ F), regime classification, and SHA256 integrity, producing a CONFORMANT/NONCONFORMANT verdict and appending to `ledger/return_log.csv`.

> **Core Axiom**: *"What Returns Through Collapse Is Real"* — Within-run: frozen causes only (no back-edges). Between-run: continuity only by return-weld (τ_R finite + seam residual within tolerance). Reality is declared by showing closure after collapse: each claim is welded to a seam. Frozen parameters (ε, p, α, λ, tol_seam) are not arbitrary constants — they are **consistent across the seam**, meaning the same rules govern both sides of every collapse-return boundary.

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
- The AM-GM gap (Δ = F − IC) is the central diagnostic — it measures channel heterogeneity
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
pytest                                       # 2,476 tests (growing), ~70s
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
  → kernel identity checks: F=1−ω, IC≈exp(κ), IC≤F (AM-GM)
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
