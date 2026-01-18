# GCD/RCFT Implementation Roadmap

**Status**: Foundation Complete | GCD/RCFT Expansion In Progress  
**Current State**: UMCP core at S=0 (zero entropy), 90/90 tests passing  
**Target**: Full GCD framework + RCFT Tier-2 overlay

---

## Current Infrastructure ✓

- [x] UMCP core validator with CLI (`umcp validate`, `umcp health`, `umcp diff`)
- [x] Canon anchors (`canon/anchors.yaml`)
- [x] Contracts: UMA.INTSTACK.v1, v1.0.1, v2
- [x] Closures: 4 implementations + registry
- [x] Data files: trace, invariants, regimes, welds
- [x] Integrity tracking: SHA256, env, code_version
- [x] Test suite: 90 passing tests
- [x] Zero entropy state: S=0, ω=0, F=1.0

---

## Phase 1: GCD Canon & Formalization (Priority: HIGH)

### 1.1 Canon Extension
- [ ] Create `canon/gcd_anchors.yaml` - GCD-specific anchor (UMCP.GCD.v1)
- [ ] Define generative collapse axioms (AX-0: collapse is generative)
- [ ] Formalize Tier-1 invariant skeleton {ω, F, S, C, τ_R, κ, IC, I}
- [ ] Document hierarchy: GCD > UMCP (contract-first)

### 1.2 GCD Closures
- [ ] `closures/gcd/energy_potential.py` - Energy computation
- [ ] `closures/gcd/entropic_collapse.py` - Entropy-driven collapse dynamics
- [ ] `closures/gcd/generative_flux.py` - Generative field computation
- [ ] `closures/gcd/field_resonance.py` - Resonance detection
- [ ] Update `closures/registry.yaml` with GCD entries

### 1.3 GCD Contract
- [ ] `contracts/GCD.INTSTACK.v1.yaml` - GCD-specific thresholds and parameters
- [ ] Freeze Tier-1 invariants (no redefinition)
- [ ] Define GCD-specific tolerance bounds
- [ ] Add patch-level versioning support

### 1.4 GCD Casepack
- [ ] `casepacks/gcd_complete/manifest.json` - GCD manifest
- [ ] `casepacks/gcd_complete/data.csv` - Sample GCD data
- [ ] `casepacks/gcd_complete/expected/` - Expected outputs
- [ ] Documentation: `docs/gcd_reference.md`

---

## Phase 2: RCFT Overlay (Priority: MEDIUM)

### 2.1 RCFT Contracts
- [ ] `contracts/RCFT.INTSTACK.v1.yaml` - RCFT as Tier-2 overlay
- [ ] Field recursion depth parameter
- [ ] Field damping factors
- [ ] Validate no Tier-1 symbol override

### 2.2 RCFT Closures
- [ ] `closures/rcft/fractal_dimension.py` - Fractal analysis
- [ ] `closures/rcft/recursive_field.py` - Field recursion computation
- [ ] `closures/rcft/resonance_pattern.py` - Pattern detection
- [ ] `closures/registry_rcft.yaml` - RCFT closure registry

### 2.3 RCFT Casepack
- [ ] `casepacks/rcft_complete/manifest.json` - RCFT manifest
- [ ] `casepacks/rcft_complete/trace.csv` - RCFT trace data
- [ ] `casepacks/rcft_complete/expected/` - RCFT outputs
- [ ] Documentation: `docs/rcft_reference.md`

### 2.4 RCFT Validator Extensions
- [ ] Detect RCFT overlay from manifest
- [ ] `--overlay rcft` CLI option
- [ ] Validate Tier-2 constraints (no Tier-1 override)
- [ ] RCFT-specific regime classification

---

## Phase 3: Integration & Testing (Priority: HIGH)

### 3.1 Validator Enhancements
- [ ] Load GCD canon automatically
- [ ] Support multiple overlays (UMCP, GCD, RCFT)
- [ ] Validate invariant hierarchy
- [ ] Enhanced error messages for Tier violations

### 3.2 Test Suite Expansion
- [ ] `tests/test_100_gcd_canon.py` - GCD canon validation
- [ ] `tests/test_101_gcd_closures.py` - GCD closure execution
- [ ] `tests/test_102_gcd_casepack.py` - GCD data validation
- [ ] `tests/test_110_rcft_overlay.py` - RCFT overlay validation
- [ ] `tests/test_111_rcft_tier2.py` - Tier-2 constraint checks
- [ ] Target: 120+ tests passing

### 3.3 Documentation
- [ ] Update `docs/file_reference.md` with GCD/RCFT files
- [ ] `docs/gcd_formulas.md` - Complete GCD formula reference
- [ ] `docs/rcft_overlay.md` - RCFT overlay specification
- [ ] Update `docs/production_deployment.md` for GCD/RCFT

---

## Phase 4: Packaging & Release (Priority: LOW)

### 4.1 Version Management
- [ ] Bump to v1.1.0 in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with GCD/RCFT additions
- [ ] Tag release: `v1.1.0-gcd-rcft`

### 4.2 CI/CD
- [ ] Update `.github/workflows/ci.yml` with GCD/RCFT tests
- [ ] Separate test jobs: UMCP core, GCD, RCFT
- [ ] Add PyPI publishing workflow
- [ ] Container image build for RCFT

### 4.3 Distribution
- [ ] Build wheels: `python -m build`
- [ ] Publish to PyPI (optional)
- [ ] Docker images: `umcp-core`, `umcp-gcd`, `umcp-rcft`
- [ ] Update README with installation instructions

---

## Essential Commands Reference

```bash
# Health check
umcp health

# Validate UMCP core
umcp validate . --strict --out validator.result.json

# Validate GCD casepack (after implementation)
umcp validate casepacks/gcd_complete --strict

# Validate RCFT overlay (after implementation)
umcp validate casepacks/rcft_complete --overlay rcft --strict

# Run all tests
pytest -v

# Run GCD-specific tests
pytest tests/test_100_gcd*.py -v

# Run RCFT-specific tests  
pytest tests/test_110_rcft*.py -v

# Generate integrity files
./scripts/generate_integrity_files.sh

# Build package
python -m build
```

---

## Current Priority Actions

**IMMEDIATE (Next Steps)**:
1. Create GCD canon anchor with invariant skeleton
2. Implement core GCD closures (energy, entropy, flux)
3. Add GCD casepack with sample data
4. Write GCD validation tests

**SHORT-TERM (This Week)**:
1. Complete GCD infrastructure
2. Begin RCFT contract definitions
3. Implement RCFT closures
4. Extend validator for overlay detection

**MEDIUM-TERM (Next Sprint)**:
1. Full RCFT overlay implementation
2. Complete test coverage (120+ tests)
3. Documentation updates
4. CI/CD enhancements

---

## Notes

- **Tier-1 Invariants**: {ω, F, S, C, τ_R, κ, IC, I} - FROZEN, cannot be redefined
- **GCD Scope**: GCD > UMCP (contract-first)
- **RCFT Constraint**: Tier-2 overlay only, builds on GCD/UMCP without overriding
- **Current Commit**: 4131d50 (S=0 state, receipt captured)
- **Next Milestone**: GCD canon + 3 core closures + casepack

---

**Last Updated**: 2026-01-18  
**Maintainer**: UMCP Development Team  
**Status**: Roadmap Active, Phase 1 Starting
