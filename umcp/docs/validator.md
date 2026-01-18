
---

UMCP-Metadata-Runnable-Code/umcp/docs/validator.md
```markdown
# Validator — what it checks and how failures are classified

The validator is the executable boundary between “a file exists” and “a claim is auditable.”

UMCP validation has two layers:

1) **Schema validation** (structure): JSON/YAML conforms to pinned schemas.  
2) **Semantic validation** (meaning): pins resolve correctly and governance/typing rules are respected.

## Inputs

Common validation targets:
- a CasePack directory: `casepacks/<casepack_id>/`
- repo root: `.` (if supported by the implementation)

## Output

A validator result should be machine-readable (JSON) and inspection-friendly. Recommended fields:

- `run_status`: `CONFORMANT` | `NONCONFORMANT` | `NON_EVALUABLE`
- `pins`:
  - resolved `contract_id`, `contract_version`
  - resolved `closures_registry_id`
  - any explicit overrides (if allowed)
- `artifacts_checked`: list of paths + schema ids/versions
- `reasons`: structured failure codes with short messages
- optional `diagnostics`: pointers to the exact violating file/field

## High-leverage NONCONFORMANT triggers

These are the common “silent drift” vectors the validator should treat as nonconformance when undeclared:

- clipping performed without asserting OOR flags (`clip_and_flag` violated)
- missingness represented as numeric without `miss=true`
- typed `INF_REC` treated as a finite value
- diagnostic used as a gate (e.g., equator/sensitivity used to assign regime)
- continuity/weld asserted without a seam receipt

## Identity/variant enforcement triggers

These changes typically require identity/variant governance (and versioning), even if the run is otherwise clean:

- weights changed
- embedding changed (bounds or mapping)
- norm/η changed (return metric)
- face policy changed (`pre_clip` vs `post_clip`)
- return-domain generator changed (window size, cycle key, reset rule, etc.)
- curvature neighborhood method changed

## Rule-writing stance

Keep failures inspection-checkable:
- name the violated pin or rule,
- point to the exact file and field,
- avoid “magic” heuristics that cannot be audited.

When in doubt, prefer strictness that prevents silent drift.
