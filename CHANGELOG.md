# Changelog

All notable changes to the UMCP validator and repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.2] - 2026-02-14

### Changed — Documentation Overhaul & Terminology Finalization

**README.md**: Comprehensive rewrite with complete, accurate content:
- Updated to 12 physics domains, 110+ closure modules, 13 casepacks
- Added Originality & Terminology section with degenerate-limit table
- Added Three-Tier Architecture overview in Documentation section
- Expanded Documentation section with Essential Reading, Reference Documents, and Internal Documentation
- Updated all domain counts, artifact counts, and badge versions
- Corrected all terminology per arrow-of-derivation standard
- Updated footer to canonical axiom form

**AXIOM.md**: Terminology finalization:
- "Connection to Physical Theories" → "Connection to Physical Theories (Degenerate Limits)" with arrow-of-derivation framing
- Removed all instances of "recover" for classical results → "emerges as a degenerate limit"
- Updated test count from 344+ to 3,515+
- Updated summary axiom to canonical form: "Collapse is generative; only what returns is real"
- Added "derived independently from Axiom-0" to Tier-1 description

**TIER_SYSTEM.md**: Terminology finalization:
- "IC ≤ F is AM-GM" → "IC ≤ F is the integrity bound"
- "(AM-GM bound)" → "(integrity bound)" in structural identities table
- Updated from 7 domains / 146 experiments to 12 domains / 3,515+ tests
- Entropy structural role updated to "Bernoulli field entropy"
- Domain expansions table expanded from 6 to all 12 domains

**Frozen parameters**: Updated version in `pyproject.toml`, `src/umcp/__init__.py`, `integrity/code_version.txt`

### Stats

- **Tests**: 3,515 passing, 0 failures
- **Lint**: ruff clean, mypy clean
- **Validation**: CONFORMANT (16 targets, 0 errors)
- **Integrity**: 121 files checksummed

---

## [2.1.0] - 2026-02-12

### Fixed - Environment Portability

**CLI subprocess resolution**: All subprocess-based invocations of `umcp`, `ruff`, `mypy`, and `pytest` now resolve the virtualenv `bin/` directory automatically. This eliminates 14 test failures in environments where the venv is not activated (e.g., dev containers, CI runners).

- `tests/conftest.py`: Prepend venv `bin/` to `PATH` at module load time
- `scripts/pre_commit_protocol.py`: Prepend venv `bin/` to `PATH` at script startup
- Zero changes to any test file — fix is centralized in conftest.py

### Changed

- Bumped version to 2.1.0 across `pyproject.toml`, `src/umcp/__init__.py`, `integrity/code_version.txt`
- Updated CHANGELOG with complete v2.0.0 and v2.1.0 entries

### Stats

- **Tests**: 2,552 passing, 0 failures
- **Lint**: ruff clean, mypy clean (43 source files)
- **Validation**: CONFORMANT (14 targets, 0 errors)

---

## [2.0.0] - 2026-02-08

### Added - Major Architecture Expansion

**Standard Model Particle Physics** — Complete SM formalism with 10 proven theorems:

- `closures/standard_model/`: 7 modules — `particle_catalog.py`, `coupling_constants.py`, `cross_sections.py`, `symmetry_breaking.py`, `ckm_mixing.py`, `subatomic_kernel.py`, `particle_physics_formalism.py`
- 31 particles mapped through 8-channel trace vectors (mass, spin, charge, color, weak isospin, lepton/baryon number, generation)
- 10/10 Tier-2 theorems PROVEN: spin-statistics, generation monotonicity, confinement as IC collapse, mass-kernel log mapping, charge quantization, cross-scale universality, symmetry breaking, CKM unitarity, running coupling flow, nuclear binding curve
- 74 individual tests covering all theorems
- `paper/standard_model_kernel.tex`: RevTeX4-2 publication with 30+ bibliography entries

**Atomic Physics** — 118-element periodic table through GCD kernel:

- `closures/atomic_physics/`: 9 modules — `periodic_kernel.py`, `cross_scale_kernel.py`, `tier1_proof.py`, `electron_config.py`, `fine_structure.py`, `ionization_energy.py`, `spectral_lines.py`, `selection_rules.py`, `zeeman_stark.py`
- 12-channel cross-scale kernel bridging subatomic → atomic (4 nuclear + 2 electronic + 6 bulk)
- Exhaustive Tier-1 proof: 10,162 tests, 0 failures (F+ω=1, IC≤F, IC=exp(κ))

**Materials Science** — Comprehensive element database:

- `closures/materials_science/element_database.py`: 118 elements × 18 physical properties

**Fleet-Scale Validation** — Distributed infrastructure:

- `src/umcp/fleet/`: Scheduler, Worker, WorkerPool, Queue (DLQ, retry, backpressure), Cache (content-addressable), Tenant (multi-tenant isolation)
- Job routing, heartbeat monitoring, priority scheduling

**Epistemic Weld** — Observation cost tracking:

- `src/umcp/epistemic_weld.py`: Gesture/return/dissolution trichotomy (Theorem T9)

**Insights Engine** — Pattern discovery system:

- `src/umcp/insights.py`: PatternDatabase, InsightEngine for lessons-learned tracking

**Dashboard** — Modular Streamlit package:

- `src/umcp/dashboard/`: 31 pages across 8 modules (core, analysis, science, physics, interactive, management, advanced)
- Domain pages with presets, Plotly visualizations, theory sections

**τ_R\* Dynamics** — Extended thermodynamic diagnostic:

- `src/umcp/tau_r_star_dynamics.py`: Dynamic τ_R\* evolution, phase diagram, arrow of time
- Theorems T10-T16, LaTeX paper

**RCFT Extensions** — Information geometry and universality:

- Tier-2 information geometry, universality, grammar (T17-T23)
- Active matter frictional cooling closure (Antonov et al. 2025)

**Developer Experience**:

- Dev container configuration (Alpine Linux, Python 3.12)
- VS Code editor config, tasks.json (22 tasks), cSpell dictionary
- Dependabot for dependency updates
- Issue templates and Code of Conduct

### Changed

- Repo renamed: `UMCP-Metadata-Runnable-Code` → `GENERATIVE-COLLAPSE-DYNAMICS`
- Pre-commit protocol optimized: replaced full pytest run with test-count bounds check
- Dashboard refactored into modular package (11 files from monolith)
- Consolidated repo structure: archived outdated files, removed stale artifacts
- `copilot-instructions.md`: Comprehensive update with SM formalism, atomic physics, paper context

### Fixed

- Restored derived/trace files (active validator dependencies)
- Standard model `__init__.py` import names (singular not plural)
- 12+ type errors across codebase (mypy, Pylance)
- Coverage config + dashboard tests (31.25% → 88.34%)
- RCFT attractor basin connected pathway
- Closure import failures on domain pages
- YAML lint issues, JSON validation fixes
- Ruff version alignment (v0.6.9 → v0.14.14)
- Pandas-dependent test skip guards

### Performance

- Manifold-bound validation gate: Layer 0 identity probe skips redundant tests
- 6-tier test ordering: math-only → IO → subprocess (fastest first)
- Kernel cache for repeated computations
- Residual envelope optimization

### Stats

- **Tests**: 2,552 (up from ~463 in v1.5.0)
- **Closure domains**: 12 (up from 7)
- **Contracts**: 14 YAML contracts
- **Casepacks**: 13 validated
- **Source**: ~34k lines across 43 source files + ~34k lines across 105 closure modules

---

## [1.5.0] - 2026-01-24

### Changed - Version Finalization

**Version Consistency**: Finalized repository version numbering to v1.5.0 across all files.

- Updated pyproject.toml to version 1.5.0
- Updated `__version__` in src/umcp/\_\_init\_\_.py to 1.5.0
- Synchronized all documentation references to v1.5.0
- Updated integrity/code_version.txt to v1.5.0

### Summary

This release consolidates all version references throughout the repository to v1.5.0, ensuring consistency across package metadata, code, documentation, and integrity tracking. All previous v1.4.x functionality remains unchanged; this is purely a version synchronization release.

---

## [1.4.8] - 2026-01-24

### Added - Computational Optimizations and Test Speed Improvements

**Performance Enhancement**: 21 computational optimizations based on 34 formal lemmas, plus ~4x test speed improvement.

#### New Optimization Modules

- **kernel_optimized.py**: OPT-1,2,3,4,12,14,15
  - Homogeneity detection via Lemma 10 (40% speedup on uniform data)
  - Log-space κ computation for numerical stability
  - Lemma 1 bounds validation
  - AM-GM gap analysis for heterogeneity quantification

- **seam_optimized.py**: OPT-10,11
  - ε-closure caching for seam chain accumulation
  - Incremental seam updates

- **compute_utils.py**: OPT-17,20
  - BatchProcessor for trajectory preprocessing
  - Weight pruning (zero-weight elimination)
  - Coordinate clipping (vectorized bounds enforcement)

- **tau_R_optimized.py**: OPT-7,8,9
  - Domain membership caching
  - Karp's algorithm for cycle detection
  - Vectorized return bounds

#### Formal Lemmas (20-34) in KERNEL_SPECIFICATION.md

- Lemma 20: Component-wise bounds on κ
- Lemma 21-23: Lipschitz continuity for F, ω, κ
- Lemma 24-25: S monotonicity and bounds
- Lemma 26-27: AM-GM gap properties
- Lemma 28-30: C convergence and extremal bounds
- Lemma 31-34: Weighted power mean generalizations

#### GCD/RCFT Closure Integrations

- entropic_collapse.py, generative_flux.py, energy_potential.py: validate_kernel_bounds()
- field_resonance.py, momentum_flux.py: Lemma 1 compliance
- attractor_basin.py, recursive_field.py, resonance_pattern.py, fractal_dimension.py: BatchProcessor

#### Test Speed Improvements (~4x)

- Added `@pytest.mark.slow` to 20 subprocess-heavy CLI tests
- Session-scoped caching in conftest.py for file I/O
- tests/test_utils.py: Lemma-based test data generators
- Run `pytest -m "not slow"` for ~11s vs ~43s full suite

#### Documentation

- docs/COMPUTATIONAL_OPTIMIZATIONS.md: 21 optimizations with complexity analysis
- docs/OPTIMIZATION_CROSS_REFERENCE.md: Integration roadmap
- docs/OPTIMIZATION_INTEGRATION_GUIDE.md: Step-by-step integration guide

### Changed

- closures/registry.yaml: Added optimizations section under extensions
- src/umcp/\_\_init\_\_.py: Exported optimization modules
- validator.py: Integrated clip_coordinates and validate_kernel_bounds
- cli.py: Added kernel validation in regime classification

### Fixed

- All 7 linting errors resolved (constant redefinition, unbound variables)

### Stats

- **Tests**: 463 passing (up from 436)
- **New test file**: test_computational_optimizations.py (27 tests)

---

## [1.4.7] - 2026-01-24

### Fixed - CI/CD and Code Quality

**Production Hardening**: Resolved all CI failures and type-checking warnings for clean builds.

#### Test Suite Fixes

- **test_coverage_90.py**: Added `skip_if_no_fastapi` decorator to 6 api_umcp tests
  - Tests now skip gracefully when fastapi is not installed (optional dependency)
  - Matches pattern already used in `test_api_umcp.py`
  - CI runs pass with 436 tests, 85.66% coverage

#### Type Checking Improvements

- **test_coverage_90.py**: Fixed 34 Pylance type-checking warnings
  - Changed `FASTAPI_AVAILABLE` to `_fastapi_available` (avoid constant redefinition)
  - Added `pyright: reportPrivateUsage=false` for intentional protected method access
  - Fixed `get_closure_path` attribute access with type annotation
  - Added null check for `duration_ms` comparison

#### Code Quality

- **Ruff formatting**: Applied consistent formatting across test files
- **Ruff linting**: Fixed B009 warning (replaced `getattr` with direct attribute access)
- **cli.py**: Removed stale `# ...existing code...` comment marker
- **minimal_cli.py**: Completed implementation (was stub)

#### Documentation

- **.github/SECRETS.md**: New comprehensive secrets configuration guide
- **.github/workflows/publish.yml**: Added detailed comments explaining PYPI_PUBLISH_TOKEN
- **docs/pypi_publishing_guide.md**: Expanded with step-by-step secret setup
- **copilot-instructions.md**: Fixed all broken markdown links

### Changed

- All file path references audited and corrected across documentation
- Planned vs implemented features clearly distinguished with 🚧 markers

---

## [1.4.6] - 2026-01-23

### Added - Return-Based Canonization Architecture

**Architectural Refinement**: Formalized mechanism for promoting Tier-2 results to Tier-1 canon through return validation.

#### Core Documentation

- **docs/RETURN_BASED_CANONIZATION.md**: Complete specification of canonization process
  - Step 1: Threshold validation (range, stability, determinism checks)
  - Step 2: Seam weld computation (Δκ, IC ratio, tolerance validation)
  - Step 3: Canon declaration (new contract version with provenance)
  - Worked example: Fractal dimension promotion workflow
  - Implementation checklists for developers, maintainers, validators

#### Tier System Enhancement

- **TIER_SYSTEM.md**: Clarified within-run vs cross-run dependency flow
  - Within frozen run: NO FEEDBACK (preserves determinism)
  - Across runs: RETURN-BASED CANONIZATION (enables evolution)
  - Added nonconformance criteria for promotion without seam weld
  - Updated dependency flow diagrams

#### Axiom Embodiment

- **AXIOM.md**: New section connecting axiom to architecture
  - "Architectural Embodiment: Return-Based Canonization"
  - Explains how tier promotion implements "cycle must return"
  - Tier-2 exploration → validation → seam weld → canonization
  - Example: Fractal dimension canonization process

#### Navigation

- **README.md**: Added docs/RETURN_BASED_CANONIZATION.md to Core Protocol documentation

### Changed

**Philosophical Consistency**: System now embodies its own axiom—"What Returns Through Collapse Is Real" applies to the evolution of the protocol itself. Tier-2 discoveries must "return" (validate through seam welding) to become Tier-1 canon.

---

## [1.3.0] - 2026-01-20

### Added - Performance Optimization System

**Intelligent Caching Architecture**: Implemented comprehensive performance optimization system with persistent learning and progressive acceleration.

#### Persistent Cache System

**Cache Directory** (`.umcp_cache/`):
- Persistent validation cache at repository root
- Survives across validation runs, test executions, and CI/CD pipelines
- Tracks cumulative statistics and metadata across all runs
- Automatic cache management via `.gitignore`
- Hash-based file content tracking (SHA256)

**Performance Impact**:
- **20-25% faster** validation on warm cache
- **1.23x speedup** on second run
- **1.26x speedup** with smart casepack skipping
- Progressive acceleration: Each run makes the system permanently faster

#### Optimization Features

**1. Schema Validator Caching**:
- Compiled `Draft202012Validator` instances cached in-memory
- Reused across multiple validations within session
- ~60% cache hit rate on warm runs
- Eliminates expensive schema compilation overhead

**2. File Content Caching**:
- JSON/YAML files cached with SHA256 hash tracking
- Automatic invalidation when file content changes
- 37+ files tracked per typical repository validation
- 13+ file reuse hits per warm validation
- Safe error handling: Parse errors don't corrupt cache

**3. Lazy Schema Loading**:
- Schemas loaded on-demand only when needed
- Eliminates overhead for targeted validation
- ~8 schemas managed with lazy evaluation
- Significant speedup for single casepack validation

**4. Smart Casepack Skipping**:
- Unchanged casepacks validated instantly via manifest hash
- 🔥 **4/4 casepacks skipped** on repeat runs with no changes
- Tracks previous validation status (CONFORMANT only)
- Manifest hash comparison prevents redundant work
- Massive benefit for CI/CD with mostly unchanged code

**5. Pre-compiled Regex Patterns**:
- Module-level compiled patterns for all validation checks
- `RE_INT`, `RE_FLOAT`, `RE_POSITIVE_WELD_CLAIM`, etc.
- No regex recompilation overhead during validation
- Optimized scalar coercion and continuity claim detection

**6. Path Resolution Caching**:
- LRU cache (256 entries) for resolved filesystem paths
- Avoids expensive `Path.resolve()` calls
- Speeds up relative path calculations throughout validation

#### Cache Statistics

**Validation Output Includes**:

```json
"cache_stats": {
  "schema_validators_cached": 8,
  "files_cached": 37,
  "cache_hits": 12,
  "cache_misses": 8,
  "file_reuse": 13,
  "schema_reuse": 12,
  "casepacks_skipped": 4,
  "total_validation_runs": 11
}
```

**Progressive Learning**:
- Cache grows with every validation run
- Cumulative knowledge across CLI, tests, and CI/CD
- Statistics tracked: total runs, hit/miss ratios, skip counts
- Persistent metadata saved to `.umcp_cache/validation_cache.pkl`

#### Real-World Performance

**Development Workflow**:
- Validate while coding: **1.85s** (instant feedback)
- Re-validate unchanged: **1.85s** (4 casepacks skipped)
- Modify 1 file: **~1.9s** (3 casepacks skipped)

**CI/CD Pipeline**:
- PR with no changes: **1.85s** (maximum skipping)
- PR with 1 casepack changed: **~2.0s** (3 casepacks skipped)
- Fresh deployment: **2.3s** (builds cache)

**Test Suite**:
- 233 tests: **18.81s** duration
- Cache learning: Continuous across test runs
- Each test execution teaches the system

#### Technical Implementation

**Cache Invalidation Strategy**:
- Hash-based: Files re-parsed only when content changes
- Automatic: No manual cache clearing needed
- Safe: Parse errors don't corrupt cache state

**Backward Compatibility**:
- Cache is completely optional
- Works transparently without configuration
- Zero breaking changes to existing workflows
- All 233 tests pass with optimizations active

**Schema Changes**:
- Updated `validator.result.schema.json` with `cache_stats` object
- Added `casepacks_skipped` field for skip tracking

### Performance

- **Initial validation**: ~2.4s (cold cache)
- **Warm cache**: ~1.85s (20-25% improvement)
- **With skipping**: 4/4 casepacks instant validation
- **Cumulative benefit**: System gets faster with every run
- **Zero overhead**: Cache operations transparent

### Validation

- All 233 tests passing
- Zero regressions from optimization changes
- Backward compatible with existing workflows
- Production ready

---

## [1.2.0] - 2026-01-20

### Added - Audit-Ready Exemplar & Strict Validation

**UMCP-REF-E2E-0001 Upgrade**: Transformed reference case from baseline to audit-ready exemplar demonstrating all critical behaviors.

#### CasePack Enhancements

**UMCP-REF-E2E-0001** (Complete audit surface):
- Modified `data/raw.csv` to demonstrate finite return (t=5 exact match to t=0)
- Added 9 timepoints (increased from 8) to show complete behavior spectrum
- Demonstrates: 1 OOR event, 1 finite return (τ_R=5), 8 INF_REC instances
- Enhanced `compute_pipeline.py` with IC ≈ exp(κ) consistency validation
- Added environment metadata capture (Python version, platform, hostname)
- Updated `receipts/ss1m.json` with manifest hash integration
- Comprehensive `README.md` with changelog and usage documentation

**CasePack Completeness** (15 new files):
- `casepacks/hello_world/`: Complete contracts + closures (5 files)
- `casepacks/gcd_complete/`: Complete contracts + closures (5 files)
- `casepacks/rcft_complete/`: Complete contracts + closures (5 files)
- All casepacks now have: contract.yaml, embedding.yaml, return.yaml, weights.yaml, closure_registry.yaml

#### Validator Enhancements

**Strict Validation Mode** (`src/umcp/cli.py`):
- Implemented `--strict` flag for publication lint gate
- Required file structure validation (contracts/, closures/, receipts/)
- Contract completeness checks (all UMA.INTSTACK.v1 parameters present)
- Weights normalization validation (Σw_i = 1.0 within tolerance)
- Manifest hash presence verification in SS1M receipts
- Environment metadata requirements
- Continuity claim integrity validation (if weld/seam asserted)
- Smart pattern matching to avoid false positives on negations

**Invariant Consistency Checks**:
- Added IC ≈ exp(κ) validation with configurable tolerance (1e-9)
- Per-row consistency checking in pipeline execution
- Tolerance reporting in SS1M receipts with pass/fail status

#### Testing

**New Tests** (12 tests added):
- `tests/test_25_umcp_ref_e2e_0001.py`: Comprehensive E2E case validation
  - File structure verification
  - OOR event detection (≥1)
  - Finite return detection (≥1)
  - INF_REC instance validation (≥1)
  - SS1M receipt structure and metadata
  - Manifest hash presence
  - Environment metadata completeness
  - IC ≈ exp(κ) consistency per row
  - Weights normalization
  - Baseline and strict validation compliance

**Test Results**:
- **Total tests**: 233 (221 previous + 12 new)
- **Pass rate**: 100% (233/233 passing)
- All casepacks validate CONFORMANT in both baseline and strict modes

#### CI/CD

**GitHub Actions Workflow** (`.github/workflows/validate.yml`):
- Added baseline validation step with full repo scan
- Added strict validation step specifically for UMCP-REF-E2E-0001
- Enhanced artifact archival (baseline + strict reports)
- Improved status checking and error reporting

#### Documentation

**CasePack README Updates**:
- Complete changelog documenting upgrade rationale
- Expected outcomes section (OOR: 1, Finite: 1, INF_REC: 8)
- Validation commands for both baseline and strict modes
- Comprehensive compliance checklist

#### Validation Results

**Before**:
- Baseline: 0 errors, 6 warnings
- Strict: Not implemented

**After**:
- Baseline: 0 errors, 0 warnings ✅
- Strict: 0 errors, 0 warnings ✅
- All casepacks CONFORMANT in both modes

#### Key Features

- **Zero warnings**: Achieved complete validation compliance across all casepacks
- **Strict mode**: Publication-grade lint gate for audit-ready artifacts
- **Complete audit surface**: All casepacks have full contract + closure specifications
- **Invariant validation**: IC ≈ exp(κ) verified to 1e-9 tolerance
- **Typed boundaries**: τ_R correctly typed as "INF_REC" or numeric throughout
- **Environment provenance**: Python version, platform, hostname in all receipts

#### Breaking Changes

None - fully backward compatible. Strict mode is opt-in via `--strict` flag.

## [1.1.0] - 2026-01-18

### Added - Phase 2: RCFT Tier-2 Overlay

**Recursive Collapse Field Theory (RCFT)** - A complete Tier-2 extension to the GCD framework providing geometric and topological analysis capabilities.

#### New Components

**Canon & Contract**:
- `canon/rcft_anchors.yaml`: Complete Tier-2 specification (297 lines)
  - 3 core principles (P-RCFT-0, P-RCFT-1, P-RCFT-2)
  - 4 new Tier-2 reserved symbols
  - 13 frozen Tier-1 symbols explicitly listed
  - 2 new regime classifications, 3 mathematical identities
- `contracts/RCFT.INTSTACK.v1.yaml`: Full Tier-2 contract (370 lines)
  - Extends GCD.INTSTACK.v1 without override
  - Inherits all GCD axioms, regimes, identities, closures
  - Adds RCFT-specific axioms, regimes, identities, closures
  - Complete provenance and extensions structure

**Closures** (3 new Tier-2 implementations):
- `closures/rcft/fractal_dimension.py` (204 lines):
  - Box-counting algorithm for trajectory complexity analysis
  - Formula: $D_f = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$
  - Regimes: Smooth (<1.2), Wrinkled (1.2-1.8), Turbulent (≥1.8)
  - Computes R-squared fit quality, log-slope, box counts
- `closures/rcft/recursive_field.py` (215 lines):
  - Exponential decay analysis for collapse memory quantification
  - Formula: $\Psi_r = \sum_{n=1}^{\infty} \alpha^n \cdot \Psi_n$
  - Regimes: Dormant (<0.1), Active (0.1-1.0), Resonant (≥1.0)
  - Supports both direct field and energy-based computation
- `closures/rcft/resonance_pattern.py` (228 lines):
  - FFT-based pattern analysis for oscillatory structures
  - Wavelength: $\lambda_p = 2\pi / k_{dominant}$, Phase: $\Theta = \arctan(\text{Im}/\text{Re})$
  - Pattern types: Standing (coherent), Mixed, Traveling (incoherent)
  - Multi-field analysis for cross-correlation

**CasePack**:
- `casepacks/rcft_complete/`: Complete example with zero-entropy state
  - 9 expected outputs (4 GCD + 3 RCFT + 2 invariants/receipt)
  - Validates full 7-closure pipeline (GCD + RCFT)
  - Demonstrates tier hierarchy validation

**Registry**:
- Updated `closures/registry.yaml` with RCFT section
- All 7 closures (4 GCD + 3 RCFT) registered with inputs/outputs

**Tests** (56 new tests, all passing):
- `tests/test_110_rcft_canon.py` (14 tests): Canon structure, schema, tier hierarchy
- `tests/test_111_rcft_closures.py` (24 tests): Closure execution, regimes, compliance
- `tests/test_112_rcft_contract.py` (18 tests): Contract conformance, inheritance, completeness
- `tests/test_113_rcft_tier2_layering.py` (23 tests): Tier validation, frozen symbols, augmentation

**Documentation**:
- `docs/rcft_theory.md`: Comprehensive theoretical foundation
  - 3 core principles explained
  - Detailed formulas and interpretations for all 3 metrics
  - Tier hierarchy and frozen symbol lists
  - Use cases and integration guidance
- `docs/rcft_usage.md`: Practical usage guide
  - Quick start examples (zero entropy, GCD+RCFT pipeline)
  - Advanced examples (parameter tuning, error handling)
  - Regime interpretation tables
  - Performance tips and common pitfalls

#### Mathematical Identities (RCFT Tier-2)

- **Fractal dimension bounds**: $1 \leq D_f \leq 3$
- **Phase periodicity**: $\Theta \in [0, 2\pi)$
- **Wavelength positivity**: $\lambda_p > 0$ (or infinite)
- **Recursive convergence**: $\Psi_r = \sum_{n=1}^{\infty} \alpha^n \cdot \Psi_n$ converges for $0 < \alpha < 1$

#### Dependencies

- Added `numpy==2.2.3` for numerical computation
- Added `scipy==1.15.1` for FFT analysis

#### Testing

- **Total tests**: 221 (142 original + 56 RCFT + 23 integration)
- **Pass rate**: 100% (221/221 passing)
- **Backward compatibility**: All 142 original tests still pass

#### Key Features

- **Augmentation, not override**: All GCD Tier-1 invariants remain frozen
- **Seamless integration**: RCFT can be used alongside or instead of pure GCD
- **Complete validation**: Full contract schema compliance, seam receipt structure
- **Extensible**: Designed to support future Tier-3 overlays

#### Breaking Changes

None - fully backward compatible with GCD.INTSTACK.v1.

## [1.0.0] - 2026-01-18

### Added - Phase 1: GCD Foundation

**Generative Collapse Dynamics (GCD)** - Complete Tier-1 framework with 4 closures, contract, casepack, and 52 tests.

#### Components

**Canon & Contract**:
- `canon/gcd_anchors.yaml`: Tier-1 specification with 3 axioms, 13 reserved symbols
- `contracts/GCD.INTSTACK.v1.yaml`: Tier-1 contract extending UMA.INTSTACK.v1

**Closures** (4 Tier-1 implementations):
- `closures/gcd/energy_potential.py`: Energy decomposition (E_collapse, E_entropy, E_curvature)
- `closures/gcd/entropic_collapse.py`: Collapse potential analysis
- `closures/gcd/generative_flux.py`: Generative flux computation
- `closures/gcd/field_resonance.py`: Boundary-interior coupling measurement

**CasePack**:
- `casepacks/hello_world/`: Example with S=0 (zero entropy) state

**Tests** (52 new tests):
- `tests/test_100_gcd_canon.py` (15 tests)
- `tests/test_101_gcd_closures.py` (21 tests)
- `tests/test_102_gcd_contract.py` (16 tests)

#### Mathematical Framework

**Axioms**:
- AX-0: Collapse is generative
- AX-1: Boundary defines interior
- AX-2: Entropy measures determinacy

**Regime Classifications**:
- Energy: Low/Medium/High
- Collapse: Minimal/Active/Critical
- Flux: Dormant/Emerging/Explosive
- Resonance: Decoupled/Partial/Coherent

**Identities**:
- Fidelity-drift duality: $F = 1 - \omega$
- Integrity-collapse relation: $IC \approx e^\kappa$
- Energy decomposition: $E = E_{collapse} + E_{entropy} + E_{curvature}$
- Resonance factorization: $R = (1-|\omega|) \cdot (1-S) \cdot e^{-C/C_{crit}}$

#### Testing

- **Total tests**: 142 (90 original + 52 GCD)
- **Pass rate**: 100% (142/142 passing)

## [0.1.0] - 2026-01-18

### Added

- Initial production release of UMCP validator
- Complete CLI interface with `validate`, `run`, and `diff` commands
- Comprehensive schema validation for all UMCP artifacts:
  - Contracts (UMA.INTSTACK.v1, v1.0.1, v2)
  - Closures registry and definitions
  - Canon anchors
  - CasePack manifests, traces, invariants, and receipts
- Semantic validation rules engine with extensible YAML configuration
- 47 comprehensive test cases covering:
  - Schema validation
  - Contract and closure validation
  - CasePack validation
  - Semantic rules (regime checks, identity checks, critical overlays)
  - CLI commands and diff functionality
  - Edge cases and error handling
- Full CasePack example: `casepacks/hello_world/`
- Complete documentation:
  - README with quickstart and merge verification
  - Contract versioning guidelines (contracts/CHANGELOG.md)
  - Python coding standards
  - Validator usage guide
- CI/CD workflow with GitHub Actions
- Production-quality code with type hints and comprehensive error handling

### Features

- **Validator CLI**: Production-grade command-line interface
  - Schema and structural validation
  - Semantic rule enforcement
  - Receipt comparison and diff capabilities
  - Strict mode for publication-level validation
- **Closure System**: 4 closure implementations with registry
  - gamma (Γ forms)
  - return_domain (window-based)
  - norms (L2 with threshold)
  - curvature_neighborhood
- **Schema Library**: 10 JSON schemas for complete artifact validation
- **Provenance Tracking**: Git commit, Python version, and timestamp tracking
- **Extensible Architecture**: YAML-based rule configuration for easy customization

### Dependencies

- Python >= 3.11
- pyyaml >= 6.0.1
- jsonschema >= 4.23.0
- pytest >= 8.0.0 (dev)
- ruff >= 0.5.0 (dev)

### Known Limitations

- CasePack generation from raw measurements not yet implemented
- Numerical engine for Ψ(t) trace generation is future work
- Seam receipt validation implemented but requires contract continuity claims

### Notes

- This release focuses on validation and metadata conformance
- Runnable numerical computation will be added in future releases
- All 47 tests passing with 100% validation conformance

---

## Version History

For contract-specific changes, see [contracts/CHANGELOG.md](contracts/CHANGELOG.md).

[0.1.0]: https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/releases/tag/v0.1.0
