# WORKSPACE CONSOLIDATION & PATHWAY REGISTRY

**Last Verified**: April 22, 2026
**Status**: ✓ UNIFIED SYSTEM - All pathway references consolidated and current
**Author**: Workspace Consolidation Protocol v1.0

---

## Executive Summary

The GENERATIVE-COLLAPSE-DYNAMICS repository is operating as a **complete, unified system** with all pathway references tied to current versions. The archive directory serves as an **append-only historical record** with no active code depending on archived resources.

**Key Verification Results:**
- ✓ All active directories present and current
- ✓ No archive/ imports in src/ or closures/
- ✓ Contract system unified: 23 active, 2 archived
- ✓ Script system unified: 106 active, 10 archived
- ✓ Documentation consistent with current structure

---

## Active System Architecture

### Core Tier-0 Protocol
- **Location**: `src/umcp/` (35 files)
- **Components**:
  - `kernel_optimized.py` — Tier-1 kernel computation
  - `validator.py` — Contract & identity validation
  - `seam_optimized.py` — Seam budget & reconciliation
  - `cli.py` — Command-line interface (2,659 lines)
  - `cognitive_equalizer.py` — CE audit framework
  - `frozen_contract.py` — Frozen parameters (Tier-1)
  - `epistemic_weld.py` — Epistemic cost tracking
  - `tau_r_star.py` — τ_R* thermodynamic diagnostic
  - `measurement_engine.py` — Data ingestion pipeline

### C99 Orchestration Core
- **Location**: `src/umcp_c/` (portable C99, ~1,900 lines)
- **Components**:
  - `kernel.h` — Kernel computation (6 invariants)
  - `contract.h` — Frozen contract in C
  - `pipeline.h` — The Spine in C (Contract → Canon → Closures → Ledger → Stance)
  - `regime.h` — Regime classification
  - `trace.h` — Trace vector lifecycle
  - `ledger.h` — Integrity ledger
  - `types.h` — Foundation types
- **Tests**: 326 assertions (166 kernel + 160 orchestration)
- **ABI**: Stable `extern "C"`, callable from any language

### C++ Accelerator
- **Location**: `src/umcp_cpp/` (pybind11 bindings)
- **Features**:
  - 50–80× speedup on kernel computation
  - Zero-copy NumPy bridge
  - Links against C99 core
- **Tests**: 434 Catch2 assertions
- **Integration**: Auto-detected at import; transparent NumPy fallback

### Domain Closures (Tier-2)
- **Location**: `closures/` (37 subdirectories, 245 modules)
- **23 Active Domains**:
  - Physics: `standard_model/`, `quantum_mechanics/`, `nuclear_physics/`, `atomic_physics/`
  - Cosmology: `astronomy/`, `weyl/`
  - Life Sciences: `evolution/`, `clinical_neuroscience/`, `consciousness_coherence/`
  - Materials: `materials_science/`, `everyday_physics/`
  - Mathematics: `kinematics/`, `rcft/`, `continuity_theory/`, `information_theory/`
  - Applied: `finance/`, `security/`, `dynamic_semiotics/`
  - Emerging: `awareness_cognition/`, `spacetime_memory/`, `ecology/`, `immunology/`

### Validation Infrastructure
- **Contracts**: `contracts/` (23 current, all DOMAIN.INTSTACK.v1.yaml format)
- **Schemas**: `schemas/` (17 JSON Schema Draft 2020-12 files)
- **Canon**: `canon/` (22 YAML anchor files, domain-specific reference points)
- **Casepacks**: `casepacks/` (26 self-contained validation bundles)

### Test Suite
- **Location**: `tests/` (232 test files)
- **Scale**: 20,235 tests (from `test_registry_domains.py` parametrization)
- **Range**: test_000 through test_343 (344 test modules)
- **Coverage**: Core infrastructure (1,939 tests) + Domain coverage (18,296 tests)

### Scripts & Tools
- **Location**: `scripts/` (106 Python files)
- **Key Tools**:
  - `ground_truth.py` — Single source of truth for all metrics
  - `sync_ground_truth.py` — Propagates metrics to 14+ files
  - `orientation.py` — Re-derives structural insights (10 sections)
  - `orientation_checkpoint.py` — Validates understanding
  - `deep_diagnostic.py`, `identity_verification.py` — 44 identity proofs
  - `pre_commit_protocol.py` — 11-step validation before commit
  - `verify_workspace_pathways.py` — Pathway consistency audit
  - Domain-specific: `bellcurve_kernel.py`, `cross_domain_bridge.py`, etc.

### Documentation
- **Location**: `docs/` (34 files)
- **Paper Archives**: `paper/` (53 files, RevTeX4-2, 189 bibliography entries)
- **Key References**:
  - `AXIOM.md` — Core axiom & operational definitions
  - `TIER_SYSTEM.md` — Three-tier architecture
  - `KERNEL_SPECIFICATION.md` — Tier-1 formal definitions
  - `KINEMATICS_MATHEMATICS.md` — Kinematic domain math
  - `MANIFESTUM_LATINUM.md` — Latin lexicon & terminology
  - `SEMIOTIC_CONVERGENCE.md` — GCD as semiotic system

---

## Archived System (Historical Records)

### Archive Structure
- **Location**: `archive/` (append-only, never modified)
- **Purpose**: Historical record of superseded versions

### Archived Artifacts
| Path | Content | Status | Reason |
|------|---------|--------|--------|
| `archive/contracts/UMA.INTSTACK.v1.0.1.yaml` | Patch version v1.0.1 | Superseded | Replaced by domain-specific contracts |
| `archive/contracts/UMA.INTSTACK.v2.yaml` | Draft v2 specification | Not adopted | Experimental, never deployed |
| `archive/scripts/generate_kin_runs_v[1-4].py` | Run generators v1–v4 | Superseded | Current: v5 in active scripts/ |
| `archive/runs/KIN.CP.*.RUN[001-003]` | Kinematics validation runs | Historical | Reference trace only |
| `archive/artifacts/*.json` | Old test baselines | Deprecated | Replaced by current schema validation |
| `archive/examples/screenshots/` | UI mockups | Reference | Feature implementations moved to web/ |

### Archive Access Policy
- **Read**: Yes (historical reference)
- **Write**: No (append-only)
- **Execute**: No (dependencies in active code)
- **Reference from active code**: Never
- **Migration**: Content promoted to active only through formal seam weld with full test coverage

---

## Pathway Consolidation Status

### ✓ Contract System (Unified)
**Status**: Consolidated
**Active Contracts**: 23 (current versions)
```
contracts/
├── ASTRO.INTSTACK.v1.yaml
├── ATOM.INTSTACK.v1.yaml
├── AWC.INTSTACK.v1.yaml
├── CLIN.INTSTACK.v1.yaml
├── CONS.INTSTACK.v1.yaml
├── RCFT.INTSTACK.v1.yaml
├── SM.INTSTACK.v1.yaml
└── ... (23 total, all v1 frozen)
```

**Archived Contracts**: 2 (historical)
```
archive/contracts/
├── UMA.INTSTACK.v1.0.1.yaml (patch)
└── UMA.INTSTACK.v2.yaml (draft)
```

**Import Pattern**: All active code uses `contracts/DOMAIN.INTSTACK.v1.yaml`
**No References**: To archive/contracts/ in active codebase ✓

### ✓ Script System (Unified)
**Status**: Consolidated
**Active Scripts**: 106 Python files (comprehensive analysis + orchestration)
**Archived Scripts**: 10 shell + Python files (old run generators v1-v4)

**Key Consolidation**:
- All Tier-0 protocol scripts in `scripts/`
- All domain analysis scripts in `scripts/`
- All validation orchestration in `scripts/`
- Archive keeps v1-v4 as reference; current: v5+ in active

**No References**: To archive/scripts/ in active codebase ✓

### ✓ Documentation System (Unified)
**Status**: Current
- All `.md` files reference current versions
- Archive mentions only in historical context (clearly marked)
- Inline code examples use active imports
- API documentation reflects current structure

**Verification**: `ARCHITECTURE.md`, `README.md`, `.github/copilot-instructions.md` all consistent ✓

### ✓ Test Suite (Unified)
**Status**: Consolidated
**Active Tests**: 20,235 (232 test files, comprehensive coverage)
**Test Baselines**: Archived old profiles in `archive/artifacts/`

**Test Count Truth**: `scripts/ground_truth.py` (computed) → all docs via `sync_ground_truth.py`
**No Breakage**: All tests run against current codebase ✓

---

## Import Path Audit

### src/ Imports
```
✓ from umcp.kernel_optimized import ...      (Tier-0 → Tier-1 kernel)
✓ from umcp.frozen_contract import ...       (Tier-0 frozen parameters)
✓ from closures.* import ...                 (Tier-0 → Tier-2 closures)
✓ from pathlib import Path                   (stdlib)
✓ from dataclasses import dataclass           (stdlib)
✗ NEVER from archive.* or ..archive          (VERIFIED: zero instances)
```

### closures/ Imports
```
✓ from umcp.kernel_optimized import ...      (Tier-2 → Tier-0)
✓ from umcp.frozen_contract import ...       (Tier-2 → frozen params)
✓ from closed.* import ...                   (Tier-2 internal)
✓ from pathlib import Path                   (stdlib)
✗ NEVER from archive.* or ..archive          (VERIFIED: zero instances)
```

### scripts/ Imports
```
✓ from scripts.ground_truth import ...       (metric truth)
✓ from src.umcp import *                     (full access to Tier-0)
✓ from closures.* import ...                 (domain closures)
✓ sys.path.insert(0, repo_root)              (proper repo root setup)
✗ NEVER from archive.* or ..archive          (VERIFIED: zero instances)
```

---

## Consolidation Checklist

- [x] **Contracts**: All current (23), no stale references
- [x] **Scripts**: All current (106), no dangling dependencies
- [x] **Tests**: All consolidated (20,235), proper metrics
- [x] **Documentation**: All references current, archive marked historical
- [x] **Imports**: No archive/ imports in active code
- [x] **Ground Truth**: Unified source with 70+ sync rules
- [x] **Archive Isolation**: Clean (append-only, no back-references)
- [x] **Circular Dependencies**: None detected
- [x] **Stale Paths**: None found
- [x] **Version Consistency**: All active contracts INTSTACK.v1

---

## Verification Reports

### Automated Checks (Run before every commit)
1. **Ground Truth Sync**: `python scripts/sync_ground_truth.py --dry-run`
2. **Pathway Audit**: `python scripts/verify_workspace_pathways.py`
3. **Contract Validation**: `umcp validate .`
4. **Import Scan**: `grep -r 'archive/' src/ closures/ --include=*.py`
5. **Test Count**: `pytest --collect-only -q | grep -oE "tests/[^:]+: \d+" | awk '{sum+=$NF} END {print sum}'`

### Recent Verification (April 22, 2026)
- ✓ Test count: 20,235 (computed from pytest, verified)
- ✓ Domains: 23 (auto-counted from closures/)
- ✓ Closures: 245 (auto-counted)
- ✓ Archive isolation: CLEAN
- ✓ Pathway references: ALL CURRENT
- ✓ Repo state: UNIFIED SYSTEM

---

## Integration Guidelines

### For Developers
1. **Never import from archive/** — Use active versions only
2. **Check ARCHITECTURE.md** — Before adding new paths
3. **Run verify_workspace_pathways.py** — Before pushing
4. **Run pre_commit_protocol.py** — Enforces all checks
5. **Update CHANGELOG.md** — When paths/structure change

### For CI/CD
Run on every commit:
```bash
python scripts/verify_workspace_pathways.py  # Must pass
python scripts/pre_commit_protocol.py       # Must exit 0
python scripts/sync_ground_truth.py         # Automatic sync
pytest --tb=short                           # Must not break
```

### For Documentation
- Document current paths in `docs/`
- Archive mentions only in historical section
- Link to `ARCHITECTURE.md` for structure questions
- Use `TIER_SYSTEM.md` to explain data flows

---

## Key Invariants

1. **Archive is append-only**: No modifications, only additions
2. **Active code never imports archived resources**: Verified by grep audit
3. **Contracts are frozen per run**: Once locked, immutable within run
4. **Tier structure is enforced**: Tier-2 cannot modify Tier-0 or Tier-1
5. **Ground truth is singular**: One source, multiple sinks, auto-propagation
6. **All references are resolvable**: No broken imports or dangling paths

---

## Maintenance Schedule

| Task | Frequency | Owner | Verification |
|------|-----------|-------|--------------|
| Pathway audit | Every commit | CI | `verify_workspace_pathways.py` |
| Ground truth sync | Every commit | CI | `sync_ground_truth.py` |
| Archive review | Quarterly | Maintainer | Manual + checklist |
| Contract versioning | On change | Maintainer | `contracts/CHANGELOG.md` |
| Documentation sync | Per PR | Reviewer | Consistency check |

---

## References

- **TIER_SYSTEM.md** — Complete tier architecture
- **ARCHITECTURE.md** — System design & directory structure
- **AXIOM.md** — Core operational principles
- **scripts/verify_workspace_pathways.py** — Automated verification tool
- **scripts/ground_truth.py** — Metrics truth source
- **scripts/pre_commit_protocol.py** — Full validation pipeline

---

**STATUS**: ✓ WORKSPACE CONSOLIDATION COMPLETE
**NEXT REVIEW**: April 2026 (quarterly schedule)
**APPROVED BY**: Workspace Consolidation Protocol v1.0
