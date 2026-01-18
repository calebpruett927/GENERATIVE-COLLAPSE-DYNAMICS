
---

UMCP-Metadata-Runnable-Code/umcp/docs/quickstart.md
```markdown
# Quickstart — run and validate UMCP artifacts

Goal: a third party can clone the repo and mechanically verify:
- pinned schemas/contracts/closures,
- a runnable CasePack,
- emitted invariants and receipts.

UMCP is contract-first: results are meaningful only relative to a pinned contract + pinned closures registry + declared embedding.

## 0) Conventions used in this repo

Terms:
- **Contract**: freezes meanings and typing boundaries (Tier-1 kernel is computed under a frozen contract).
- **Closures registry**: pins non-kernel degrees of freedom (Γ form, return-domain generator, norm/η, curvature neighborhood choices, etc.).
- **CasePack**: runnable publication unit (minimal mechanical example).
- **SS1m receipt**: emitted for every run (minimum audit record).
- **Seam receipt**: continuity-only (required when making continuity/weld claims).

## 1) Repository landmarks (typical)

At repo root (outside `umcp/`), you will usually find:

- `schemas/` — JSON Schemas (hard enforcement boundary)
- `contracts/` — frozen contracts (e.g., `UMA.INTSTACK.v1`)
- `closures/` — closure files + registry files
- `casepacks/` — runnable examples (growth surface)
- `tests/` — regression checks (CI reproducibility)
- `scripts/` — helper scripts (optional)

The `umcp/` folder contains the runtime package and docs.

## 2) Validate a CasePack

From repo root, validate the CasePack directory.

```bash
umcp validate casepacks/<casepack_id> --out validator.result.json
