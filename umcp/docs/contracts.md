# Contracts — immutability and version governance

Contracts freeze the meanings and typing boundaries that make UMCP computations auditable. A contract is not “configuration”; it is the normative execution boundary.

## Contract-first rule

A run is interpretable only relative to a pinned contract. If the contract changes, the run meaning changes. Comparability across changes requires governance (identity/variant handling), and continuity claims require seam receipts.

## What a contract must pin (minimum)

A contract should explicitly pin:
- bounded embedding interval (e.g., `[0,1]`)
- face policy (`pre_clip` vs `post_clip`)
- out-of-range policy (e.g., `clip_and_flag`)
- log-safety epsilon (e.g., `1e-8`)
- Tier-1 definitions and any fixed constants used by them
- typed boundaries (e.g., `tau_R` may take `INF_REC` as a typed value)

## Immutability

Published contracts should be immutable:
- do not edit contract files in place after release
- changes require a new versioned contract id/version

## Practical governance guidance

Assume a new contract version is required if you change:
- Tier-1 definition details,
- typed boundary semantics,
- log-safety rules,
- OOR/missingness representation semantics.

If the change is a “degree of freedom” (choice among forms), prefer placing it in closures rather than in the contract.

## Repo convention (typical)

Contracts commonly live in:
- `contracts/` (YAML)
- the contract schema lives in `schemas/`

A CasePack manifest must pin:
- `contract_id`
- `contract_version`

## Checklist: adding a new contract version

1) Add a new versioned contract file (do not overwrite older versions).  
2) Update docs to reference the new version explicitly.  
3) Add at least one CasePack that exercises the new version.  
4) Add validator/tests so CI detects accidental regressions.  
5) If continuity is claimed across versions, require seam receipts.
