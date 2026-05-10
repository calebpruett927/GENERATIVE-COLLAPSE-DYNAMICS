# Casepack Reorg — Reference Audit (Phase 2)

> Generated 2026-05-10. **307 reference lines** across 26 casepacks. Full data in `REORG_AUDIT.json`.

> **TEMPORARY** — delete this file and `REORG_AUDIT.json` after Phase 4 completes.

---

## Triage Summary (by category)

| Category | Description | Files | Sites | Action |
|---|---|---:|---:|---|
| `01_changelog_HISTORICAL` | CHANGELOG.md | 1 | 4 | DO NOT REWRITE — historical record. |
| `02_archive_HISTORICAL` | archive/ (frozen artifacts) | 8 | 30 | DO NOT REWRITE — frozen historical artifacts. |
| `10_tests_LIVE` | tests/ | 14 | 14 | MUST UPDATE — hardcoded path strings in assertions. |
| `11_scripts_LIVE` | scripts/ | 3 | 3 | MUST UPDATE — runtime path references. |
| `12_src_LIVE` | src/ | 3 | 3 | MUST UPDATE — Python source references. |
| `13_casepacks_internal_LIVE` | casepacks/<self>/ | 28 | 33 | MUST UPDATE — manifest.json, generate_expected.py, ss1m_receipt.json self-paths. |
| `14_canon_LIVE` | canon/ | 7 | 11 | MUST UPDATE — anchor file cross-references. |
| `15_contracts_LIVE` | contracts/ | 1 | 1 | MUST UPDATE. |
| `16_closures_LIVE` | closures/ | 3 | 3 | MUST UPDATE — closure code/registry/READMEs. |
| `17_docs_LIVE` | docs/ | 12 | 20 | MUST UPDATE — markdown cross-links. |
| `18_paper_LIVE` | paper/ | 1 | 1 | MUST UPDATE. |
| `19_web_LIVE` | web/ | 56 | 72 | MUST UPDATE — Astro frontend. |
| `20_github_LIVE` | .github/ (CI + templates) | 3 | 3 | MUST UPDATE — CI workflow + issue templates + copilot-instructions. |
| `21_root_md_LIVE` | root *.md | 10 | 18 | MUST UPDATE — README, GLOSSARY, PROTOCOL_REFERENCE, etc. |

**Total LIVE sites to rewrite**: 182 · **Total HISTORICAL/AUTO (preserve)**: 34


---
## Phase 3 Sub-Commit Plan

Each sub-phase: (1) `git mv`, (2) `sed` rewrite of LIVE files only, (3) regenerate integrity, (4) run pre-commit.

| Phase | Casepacks | LIVE files | LIVE sites | Notes |
|---|---:|---:|---:|---|
| **3a_pedagogical** | 2 | 41 | 46 | hello_world dominates — pilot the rewrite recipe here. |
| **3b_ladder** | 7 | 30 | 34 | Smaller; mostly internal self-paths. |
| **3c_closures_full** | 17 | 79 | 102 | Largest — recipe proven by 3a/3b. |

---
## Per-Category File Lists


### `01_changelog_HISTORICAL` — CHANGELOG.md

*Action*: DO NOT REWRITE — historical record.

| File | Casepacks |
|---|---|
| `CHANGELOG.md` | gcd_complete, hello_world, rcft_complete, spacetime_memory_entities |

### `02_archive_HISTORICAL` — archive/ (frozen artifacts)

*Action*: DO NOT REWRITE — frozen historical artifacts.

| File | Casepacks |
|---|---|
| `archive/artifacts/validator.baseline.json` | UMCP-REF-E2E-0001, gcd_complete, hello_world, kin_ref_phase_oscillator, kinematics_complete, rcft_complete, retro_coherent_phys04, weyl_des_y3 |
| `archive/artifacts/validator.negative.json` | UMCP-REF-E2E-0001, gcd_complete, hello_world, kin_ref_phase_oscillator, kinematics_complete, rcft_complete, retro_coherent_phys04, weyl_des_y3 |
| `archive/artifacts/validator.strict.json` | UMCP-REF-E2E-0001, gcd_complete, hello_world, kin_ref_phase_oscillator, kinematics_complete, rcft_complete, retro_coherent_phys04, weyl_des_y3 |
| `archive/dashboard/src/__init__.py` | hello_world |
| `archive/dashboard/src/pages_management.py` | hello_world |
| `archive/reports/ANALYSIS_SUMMARY.txt` | UMCP-REF-E2E-0001 |
| `archive/reports/DEEP_RESEARCH_SWEEP.md` | finance_continuity |
| `archive/reports/REPOSITORY_ANALYSIS_2026-02-10.md` | UMCP-REF-E2E-0001, hello_world |

### `10_tests_LIVE` — tests/

*Action*: MUST UPDATE — hardcoded path strings in assertions.

| File | Casepacks |
|---|---|
| `tests/closures/test_kin_ref_phase.py` | kin_ref_phase_oscillator |
| `tests/test_113_rcft_tier2_layering.py` | rcft_complete |
| `tests/test_173_schema_instances.py` | hello_world |
| `tests/test_177_public_api.py` | hello_world |
| `tests/test_200_fleet.py` | hello_world |
| `tests/test_20_casepack_hello_world_validates.py` | hello_world |
| `tests/test_284_fleet_extended.py` | hello_world |
| `tests/test_288_coverage_final.py` | hello_world |
| `tests/test_289_coverage_push.py` | hello_world |
| `tests/test_290_final_coverage.py` | hello_world |
| `tests/test_291_comprehensive_coverage.py` | hello_world |
| `tests/test_51_cli_diff.py` | hello_world |
| `tests/test_90_edge_cases.py` | hello_world |
| `tests/test_init_api.py` | hello_world |

### `11_scripts_LIVE` — scripts/

*Action*: MUST UPDATE — runtime path references.

| File | Casepacks |
|---|---|
| `scripts/astronomy_analysis.py` | astronomy_complete |
| `scripts/benchmark_umcp_vs_standard.py` | hello_world |
| `scripts/recompute_phys04.py` | retro_coherent_phys04 |

### `12_src_LIVE` — src/

*Action*: MUST UPDATE — Python source references.

| File | Casepacks |
|---|---|
| `src/umcp/__init__.py` | hello_world |
| `src/umcp/cli.py` | hello_world |
| `src/umcp/fleet/__init__.py` | hello_world |

### `13_casepacks_internal_LIVE` — casepacks/<self>/

*Action*: MUST UPDATE — manifest.json, generate_expected.py, ss1m_receipt.json self-paths.

| File | Casepacks |
|---|---|
| `casepacks/UMCP-REF-E2E-0001/README.md` | UMCP-REF-E2E-0001 |
| `casepacks/UMCP-REF-E2E-0001/expected/ss1m_receipt.json` | UMCP-REF-E2E-0001 |
| `casepacks/UMCP-REF-E2E-0001/logs/validator_output.json` | UMCP-REF-E2E-0001, gcd_complete, hello_world, rcft_complete |
| `casepacks/astronomy_complete/README.md` | astronomy_complete |
| `casepacks/bell_curve_pgs/expected/ss1m_receipt.json` | bell_curve_pgs |
| `casepacks/confinement_T3/README.md` | confinement_T3 |
| `casepacks/confinement_T3/run_confinement_T3.py` | confinement_T3 |
| `casepacks/consciousness_kappa_72/README.md` | consciousness_kappa_72 |
| `casepacks/consciousness_kappa_72/expected/ss1m_receipt.json` | consciousness_kappa_72 |
| `casepacks/finance_continuity/README.md` | finance_continuity |
| `casepacks/finance_continuity/expected/invariants.json` | finance_continuity |
| `casepacks/finance_continuity/generate_expected.py` | finance_continuity |
| `casepacks/gcd_complete/README.md` | gcd_complete |
| `casepacks/hello_world/README.md` | hello_world |
| `casepacks/hello_world/expected/ss1m_receipt.json` | hello_world |
| `casepacks/hello_world/generate_expected.py` | hello_world |
| `casepacks/kin_ref_phase_oscillator/README.md` | kin_ref_phase_oscillator, kinematics_complete |
| `casepacks/kin_ref_phase_oscillator/closures/closure_registry.yaml` | kin_ref_phase_oscillator |
| `casepacks/kin_ref_phase_oscillator/manifest.json` | kin_ref_phase_oscillator |
| `casepacks/kinematics_complete/README.md` | kinematics_complete |
| `casepacks/nuclear_chain/README.md` | nuclear_chain |
| `casepacks/quantum_mechanics_complete/README.md` | quantum_mechanics_complete |
| `casepacks/rcft_complete/README.md` | rcft_complete |
| `casepacks/retro_coherent_phys04/README.md` | retro_coherent_phys04 |
| `casepacks/ricp_v2_consciousness/README.md` | consciousness_kappa_72, ricp_v2_consciousness |
| `casepacks/ricp_v2_consciousness/expected/ss1m_receipt.json` | ricp_v2_consciousness |
| `casepacks/ricp_v2_consciousness/one_vector_six_lenses.py` | ricp_v2_consciousness |
| `casepacks/weyl_des_y3/README.md` | weyl_des_y3 |

### `14_canon_LIVE` — canon/

*Action*: MUST UPDATE — anchor file cross-references.

| File | Casepacks |
|---|---|
| `canon/README.md` | astronomy_complete, gcd_complete, kinematics_complete, rcft_complete, weyl_des_y3 |
| `canon/astro_anchors.yaml` | astronomy_complete |
| `canon/docs/validator_usage.md` | hello_world |
| `canon/gcd_anchors.yaml` | gcd_complete |
| `canon/kin_anchors.yaml` | kinematics_complete |
| `canon/rcft_anchors.yaml` | rcft_complete |
| `canon/weyl_anchors.yaml` | weyl_des_y3 |

### `15_contracts_LIVE` — contracts/

*Action*: MUST UPDATE.

| File | Casepacks |
|---|---|
| `contracts/CHANGELOG.md` | kinematics_complete |

### `16_closures_LIVE` — closures/

*Action*: MUST UPDATE — closure code/registry/READMEs.

| File | Casepacks |
|---|---|
| `closures/registry.yaml` | kin_ref_phase_oscillator |
| `closures/security/README.md` | security_validation |
| `closures/weyl/__init__.py` | weyl_des_y3 |

### `17_docs_LIVE` — docs/

*Action*: MUST UPDATE — markdown cross-links.

| File | Casepacks |
|---|---|
| `docs/CASEPACK_REFERENCE.md` | UMCP-REF-E2E-0001 |
| `docs/GLOSSARY_SYSTEM.md` | gcd_complete, hello_world, rcft_complete |
| `docs/PUBLICATION_INFRASTRUCTURE.md` | gcd_complete |
| `docs/RESPONSES_TO_CRITICS.md` | finance_continuity |
| `docs/SYMBOL_INDEX.md` | kinematics_complete, rcft_complete |
| `docs/SYSTEM_ARCHITECTURE.md` | gcd_complete, hello_world, rcft_complete |
| `docs/TERM_INDEX.md` | UMCP-REF-E2E-0001, hello_world |
| `docs/UHMP.md` | UMCP-REF-E2E-0001 |
| `docs/contributors/jackson/MATHEMATICAL_REFERENCE.md` | consciousness_kappa_72, ricp_v2_consciousness |
| `docs/contributors/jackson/TIER2_DIAGNOSTIC.md` | consciousness_kappa_72, ricp_v2_consciousness |
| `docs/rcft_theory.md` | rcft_complete |
| `docs/rcft_usage.md` | rcft_complete |

### `18_paper_LIVE` — paper/

*Action*: MUST UPDATE.

| File | Casepacks |
|---|---|
| `paper/the_first_pass.tex` | hello_world |

### `19_web_LIVE` — web/

*Action*: MUST UPDATE — Astro frontend.

| File | Casepacks |
|---|---|
| `web/src/content/astronomy/casepacks/astronomy_complete.md` | astronomy_complete |
| `web/src/content/astronomy/index.md` | astronomy_complete |
| `web/src/content/atomic_physics/casepacks/atomic_physics_elements.md` | atomic_physics_elements |
| `web/src/content/atomic_physics/index.md` | atomic_physics_elements |
| `web/src/content/awareness_cognition/casepacks/awareness_cognition_kernel.md` | awareness_cognition_kernel |
| `web/src/content/awareness_cognition/index.md` | awareness_cognition_kernel |
| `web/src/content/clinical_neuroscience/casepacks/clinical_neuro_states.md` | clinical_neuro_states |
| `web/src/content/clinical_neuroscience/index.md` | clinical_neuro_states |
| `web/src/content/consciousness_coherence/casepacks/consciousness_kappa_72.md` | consciousness_kappa_72 |
| `web/src/content/consciousness_coherence/casepacks/ricp_v2_consciousness.md` | ricp_v2_consciousness |
| `web/src/content/consciousness_coherence/index.md` | consciousness_kappa_72, ricp_v2_consciousness |
| `web/src/content/continuity_theory/casepacks/continuity_theory_systems.md` | continuity_theory_systems |
| `web/src/content/continuity_theory/index.md` | continuity_theory_systems |
| `web/src/content/dynamic_semiotics/casepacks/semiotics_kernel.md` | semiotics_kernel |
| `web/src/content/dynamic_semiotics/index.md` | semiotics_kernel |
| `web/src/content/everyday_physics/casepacks/everyday_physics_materials.md` | everyday_physics_materials |
| `web/src/content/everyday_physics/index.md` | everyday_physics_materials |
| `web/src/content/evolution/casepacks/astronomy_complete.md` | astronomy_complete |
| `web/src/content/evolution/casepacks/evolution_kernel.md` | evolution_kernel |
| `web/src/content/evolution/casepacks/weyl_des_y3.md` | weyl_des_y3 |
| `web/src/content/evolution/index.md` | astronomy_complete, evolution_kernel, weyl_des_y3 |
| `web/src/content/finance/casepacks/finance_continuity.md` | finance_continuity |
| `web/src/content/finance/index.md` | finance_continuity |
| `web/src/content/gcd/casepacks/atomic_physics_elements.md` | atomic_physics_elements |
| `web/src/content/gcd/casepacks/awareness_cognition_kernel.md` | awareness_cognition_kernel |
| `web/src/content/gcd/casepacks/clinical_neuro_states.md` | clinical_neuro_states |
| `web/src/content/gcd/casepacks/continuity_theory_systems.md` | continuity_theory_systems |
| `web/src/content/gcd/casepacks/everyday_physics_materials.md` | everyday_physics_materials |
| `web/src/content/gcd/casepacks/evolution_kernel.md` | evolution_kernel |
| `web/src/content/gcd/casepacks/gcd_complete.md` | gcd_complete |
| `web/src/content/gcd/casepacks/materials_science_elements.md` | materials_science_elements |
| `web/src/content/gcd/casepacks/quantum_mechanics_complete.md` | quantum_mechanics_complete |
| `web/src/content/gcd/casepacks/rcft_complete.md` | rcft_complete |
| `web/src/content/gcd/casepacks/retro_coherent_phys04.md` | retro_coherent_phys04 |
| `web/src/content/gcd/casepacks/semiotics_kernel.md` | semiotics_kernel |
| `web/src/content/gcd/casepacks/spacetime_memory_entities.md` | spacetime_memory_entities |
| `web/src/content/gcd/index.md` | atomic_physics_elements, awareness_cognition_kernel, clinical_neuro_states, continuity_theory_systems, everyday_physics_materials, evolution_kernel, gcd_complete, materials_science_elements, quantum_mechanics_complete, rcft_complete, retro_coherent_phys04, semiotics_kernel, spacetime_memory_entities |
| `web/src/content/kinematics/casepacks/kin_ref_phase_oscillator.md` | kin_ref_phase_oscillator |
| `web/src/content/kinematics/casepacks/kinematics_complete.md` | kinematics_complete |
| `web/src/content/kinematics/index.md` | kin_ref_phase_oscillator, kinematics_complete |
| `web/src/content/materials_science/casepacks/materials_science_elements.md` | materials_science_elements |
| `web/src/content/materials_science/index.md` | materials_science_elements |
| `web/src/content/nuclear_physics/casepacks/nuclear_chain.md` | nuclear_chain |
| `web/src/content/nuclear_physics/index.md` | nuclear_chain |
| `web/src/content/quantum_mechanics/casepacks/quantum_mechanics_complete.md` | quantum_mechanics_complete |
| `web/src/content/quantum_mechanics/index.md` | quantum_mechanics_complete |
| `web/src/content/rcft/casepacks/rcft_complete.md` | rcft_complete |
| `web/src/content/rcft/index.md` | rcft_complete |
| `web/src/content/security/casepacks/security_validation.md` | security_validation |
| `web/src/content/security/index.md` | security_validation |
| `web/src/content/spacetime_memory/casepacks/spacetime_memory_entities.md` | spacetime_memory_entities |
| `web/src/content/spacetime_memory/index.md` | spacetime_memory_entities |
| `web/src/content/standard_model/casepacks/confinement_T3.md` | confinement_T3 |
| `web/src/content/standard_model/index.md` | confinement_T3 |
| `web/src/content/weyl/casepacks/weyl_des_y3.md` | weyl_des_y3 |
| `web/src/content/weyl/index.md` | weyl_des_y3 |

### `20_github_LIVE` — .github/ (CI + templates)

*Action*: MUST UPDATE — CI workflow + issue templates + copilot-instructions.

| File | Casepacks |
|---|---|
| `.github/ISSUE_TEMPLATE/bug_report.md` | hello_world |
| `.github/copilot-instructions.md` | hello_world |
| `.github/workflows/main.yml` | UMCP-REF-E2E-0001 |

### `21_root_md_LIVE` — root *.md

*Action*: MUST UPDATE — README, GLOSSARY, PROTOCOL_REFERENCE, etc.

| File | Casepacks |
|---|---|
| `AI_WORKFLOW.md` | hello_world |
| `COMMIT_PROTOCOL.md` | UMCP-REF-E2E-0001 |
| `CONTRIBUTING.md` | hello_world |
| `GLOSSARY.md` | UMCP-REF-E2E-0001, hello_world, rcft_complete, weyl_des_y3 |
| `KINEMATICS_SPECIFICATION.md` | kin_ref_phase_oscillator |
| `PROTOCOL_REFERENCE.md` | UMCP-REF-E2E-0001, gcd_complete, hello_world, rcft_complete |
| `QUICKSTART_TUTORIAL.md` | gcd_complete, hello_world |
| `README.md` | hello_world |
| `README_PYPI.md` | hello_world |
| `THE_FIRST_PASS.md` | UMCP-REF-E2E-0001, hello_world |

---
## Mechanical Rewrite Recipe (per casepack)

```bash
OLD=<old_name>
NEW=<new_path>           # e.g. casepacks/pedagogical/hello_world

# 1. Move with history preservation
git mv casepacks/$OLD $NEW

# 2. Rewrite references in LIVE files only
git ls-files \
  | grep -vE '^(CHANGELOG|integrity/|ledger/|derived/|archive/)' \
  | xargs -I{} sed -i "s|casepacks/$OLD/|$NEW/|g" {}

# 3. Regenerate integrity manifest
python scripts/update_integrity.py

# 4. Validate
python scripts/pre_commit_protocol.py --check
```

## Caveats Surfaced by Audit

- **`web/` has 72 LIVE sites** — Astro frontend has cross-links into casepack docs. Will be the biggest rewrite chunk in Phase 3c.
- **`.github/copilot-instructions.md` references `hello_world`** — this is the binding spec for AI agents; rewriting changes path that future agents will read.
- **`closures/registry.yaml` references `kin_ref_phase_oscillator`** — registry is live config, must update.
- **`archive/artifacts/validator.{baseline,negative,strict}.json`** — frozen validator outputs from past runs. DO NOT REWRITE — they document what the validator returned at commit time.
- **`canon/README.md` lists 7 manifest paths** — central index, must update.
- **`casepacks/finance_continuity/generate_expected.py` writes its own path into output JSON** — generator script must be updated AND re-run, or output JSON manually patched.
