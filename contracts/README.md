# `contracts/` — frozen UMCP contract boundaries

**Protocol Resources:** [GLOSSARY.md](../GLOSSARY.md) | [SYMBOL_INDEX.md](../SYMBOL_INDEX.md) | [TERM_INDEX.md](../TERM_INDEX.md#c)

This directory contains UMCP **contracts**: normative, versioned boundaries that freeze meanings and typed semantics required for auditable computation.

A contract is not “configuration.” A contract is the execution boundary that makes outputs interpretable and comparable.

## What lives here

- Versioned contract files (YAML), e.g.:
  - `UMA.INTSTACK.v1.yaml`

Contracts are validated against the pinned schema:
- `schemas/contract.schema.json`

## Non-negotiable rules

1) No in-place edits after release
Once a contract is published or referenced by any CasePack/receipt, do not modify it.
Changes require a new versioned contract file.

2) Manifest pinning is mandatory
CasePack manifests must pin:
- `contract_id`
- `contract_version`

1) Kernel symbols are reserved
Contracts define the Tier-1 invariant structure boundary. Domain expansion closures (Tier-2) must not redefine Tier-1 symbols (F, ω, S, C, κ, IC, τ_R, regime). See [TIER_SYSTEM.md](../TIER_SYSTEM.md).

2) Typed semantics are part of correctness
Typed boundary values (e.g. `tau_R = INF_REC`) must remain typed; implementations must not silently coerce them into floats.

## When to create a new contract version

Create a new contract version if you change any of the following:
- Tier-1 definition semantics
- typed-censoring semantics (e.g., special value sets or their meaning)
- log-safety rules (`epsilon` behavior)
- OOR policy semantics or face semantics
- any tolerance/parameter that changes what “PASS/FAIL” means at the seam level

If the change is a “degree of freedom” (choice among alternatives), prefer placing it in **closures** rather than in the contract.

## Suggested workflow

1) Add new contract file (do not overwrite older files)
2) Update `contracts/CHANGELOG.md`
3) Add at least one CasePack that pins the new contract version
4) Add/adjust tests to ensure CI catches drift
5) If continuity is claimed across versions, require seam receipts

## Canonical contract in this repo

- `UMA.INTSTACK.v1.yaml`
