# Changelog

All notable changes to the UMCP validator and repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/releases/tag/v0.1.0
