# Casepack Taxonomy — The Migration Contract

> **Status**: This file is the contract for the casepack reorganization.
> Once approved, every entry below is the destination of a `git mv` in
> Phase 3 of the migration. Path references in tests, scripts, and docs
> will be rewritten to match.

> **Frozen classification**: 2026-05-10 (post-approval). Changes after this
> point require a Weld with a named anchor — see `COMMIT_PROTOCOL.md`.

---

## The Four Families

| Family | Path | Members | Entry criterion |
|---|---|:---:|---|
| **Pedagogical** | `pedagogical/` | 2 | First-contact, reference, or teaching example. Not a closure of record. |
| **Ladder rungs** | `ladder/L1_…/` … `ladder/L4_…/` | 7 | Demonstrates exactly one of the four sequential burdens from *The Pass*. |
| **Closures (full)** | `closures/full/<domain>/` | 17 | Tier-2 closure of record for a registered domain. |
| **Closures (mini)** | `closures/mini/` *(reserved)* | 0 | Single-channel or partial closure. None at present. |
| **Total** | | **26** | |

---

## Authoritative Classification

### Pedagogical (`pedagogical/`)

| Casepack | New path | Justification |
|---|---|---|
| `hello_world` | `pedagogical/hello_world/` | Minimal end-to-end validation. The first casepack any reader runs. |
| `UMCP-REF-E2E-0001` | `pedagogical/UMCP-REF-E2E-0001/` | Reference end-to-end example used in glossary and CI smoke. |

### Ladder rungs (`ladder/`)

| Casepack | New path | Burden it demonstrates |
|---|---|---|
| `kin_ref_phase_oscillator` | `ladder/L1_heterogeneous_local/kin_ref_phase_oscillator/` | Heterogeneous channels in a phase oscillator — IC ≠ F under matched arithmetic. |
| `confinement_T3` | `ladder/L1_heterogeneous_local/confinement_T3/` | T3 confinement: a single heterogeneous local pass. |
| `nuclear_chain` | `ladder/L2_finite_return/nuclear_chain/` | Decay chain — finite-return measurement (τ_R is observed, not assumed). |
| `retro_coherent_phys04` | `ladder/L2_finite_return/retro_coherent_phys04/` | Retro-coherence — finite return event with explicit τ_R. |
| `consciousness_kappa_72` | `ladder/L3_welded_seam/consciousness_kappa_72/` | RIP v1 weld at κ=72: the first welded-seam demonstration. |
| `ricp_v2_consciousness` | `ladder/L3_welded_seam/ricp_v2_consciousness/` | RICP v2.0 weld continuity (refined version of the L3 burden). |
| `bell_curve_pgs` | `ladder/L4_seam_chain/bell_curve_pgs/` | Polygenic — chain of welds across cohort boundaries. |

### Closures of record (`closures/full/`)

One full closure per registered Tier-2 domain:

| Casepack | New path | Domain |
|---|---|---|
| `gcd_complete` | `closures/full/gcd/` | gcd |
| `rcft_complete` | `closures/full/rcft/` | rcft |
| `kinematics_complete` | `closures/full/kinematics/` | kinematics |
| `weyl_des_y3` | `closures/full/weyl/` | weyl |
| `astronomy_complete` | `closures/full/astronomy/` | astronomy |
| `quantum_mechanics_complete` | `closures/full/quantum_mechanics/` | quantum_mechanics |
| `finance_continuity` | `closures/full/finance/` | finance |
| `atomic_physics_elements` | `closures/full/atomic_physics/` | atomic_physics |
| `materials_science_elements` | `closures/full/materials_science/` | materials_science |
| `everyday_physics_materials` | `closures/full/everyday_physics/` | everyday_physics |
| `evolution_kernel` | `closures/full/evolution/` | evolution |
| `semiotics_kernel` | `closures/full/dynamic_semiotics/` | dynamic_semiotics |
| `awareness_cognition_kernel` | `closures/full/awareness_cognition/` | awareness_cognition |
| `clinical_neuro_states` | `closures/full/clinical_neuroscience/` | clinical_neuroscience |
| `continuity_theory_systems` | `closures/full/continuity_theory/` | continuity_theory |
| `spacetime_memory_entities` | `closures/full/spacetime_memory/` | spacetime_memory |
| `security_validation` | `closures/full/security/` | security |

### Coverage Gap (incomplete — to finish post-reorg)

The closures directory hosts **23 Tier-2 domains**. The 17 above have a
full casepack of record. Six domains have closure code and tests but **no
casepack**, and therefore no entry in `closures/full/` after the move:

| Domain | `closures/<domain>/` exists | Tests exist | Casepack stub needed |
|---|:---:|:---:|---|
| `nuclear_physics` | ✅ | ✅ (`test_135_*`, `test_250_*`) | ⚠️ flag for post-reorg |
| `consciousness_coherence` | ✅ | ✅ (`test_242_*`, `test_244_*`, `test_257_*`) | ⚠️ flag for post-reorg |
| `standard_model` | ✅ | ✅ (`test_148_*`, `test_149_*`, `test_201_*`, `test_248_*`, `test_302_*`) | ⚠️ flag for post-reorg |
| `ecology` | ✅ | ⚠️ partial | ⚠️ flag for post-reorg |
| `immunology` | ✅ | ⚠️ partial | ⚠️ flag for post-reorg |
| `information_theory` | ✅ | ⚠️ partial | ⚠️ flag for post-reorg |

These domains are functional — the closure code runs through Tier-0
validation and the tests pass — they simply lack a packaged demonstration
unit. Each will receive a casepack stub in a follow-up weld after the
reorganization completes. Tracking issue: see `casepacks/TAXONOMY.md`
section "Coverage Gap" (this section).

> **Note for Phase 3c**: Do **not** create empty `closures/full/<domain>/`
> directories for the six gaps during the move. Empty directories would
> falsely imply a packaged closure exists. The taxonomy table above is
> the authoritative record of which domains have casepacks and which do not.

---

## What Was Moved Out

| Former location | New location | Reason |
|---|---|---|
| `casepacks/JACKSON_CLAIMS_FORMALIZATION.md` | `docs/contributors/jackson/CLAIMS_FORMALIZATION.md` | Engagement document, not a casepack. |
| `casepacks/JACKSON_MATHEMATICAL_REFERENCE.md` | `docs/contributors/jackson/MATHEMATICAL_REFERENCE.md` | Engagement document. |
| `casepacks/JACKSON_TIER2_DIAGNOSTIC.md` | `docs/contributors/jackson/TIER2_DIAGNOSTIC.md` | Engagement document. |

The `docs/contributors/` directory is the new home for all sustained
engagement with external authors. See [`docs/contributors/README.md`](../docs/contributors/README.md).

---

## Migration Phases

The reorganization to this taxonomy follows four phases. Each phase ends
with a green `pre_commit_protocol.py`:

| Phase | Action | Status |
|:---:|---|:---:|
| **1** | Classify (this file) + relocate Jackson docs + introduce `casepacks/README.md` and `docs/contributors/README.md`. No casepack moves. | ✅ this commit |
| **2** | Reference audit — enumerate every file × old-path × new-path tuple. Output: temporary `casepacks/REORG_AUDIT.md`. | ⏳ pending |
| **3a** | Move `pedagogical/` family (2 casepacks, ~12 reference sites). Validates the migration method on the smallest set. | ⏳ pending |
| **3b** | Move `ladder/` family (7 casepacks, ~10 reference sites). | ⏳ pending |
| **3c** | Move `closures/full/` family (17 casepacks, ~60 reference sites). | ⏳ pending |
| **4** | Documentation refresh: update `README.md`, `GLOSSARY.md`, `PROTOCOL_REFERENCE.md`, `QUICKSTART_TUTORIAL.md`, `TIER_SYSTEM.md`, `canon/README.md`, `KINEMATICS_SPECIFICATION.md` to use new canonical paths. | ⏳ pending |

After Phase 4, `casepacks/` will contain only: `README.md`, `TAXONOMY.md`,
the four family directories, and (reserved, possibly empty) `closures/mini/`.

---

> *Spina non negotiabilis est.* — The spine is non-negotiable.
> The taxonomy is published before the moves. Once approved, the moves
> are mechanical applications of this contract.
