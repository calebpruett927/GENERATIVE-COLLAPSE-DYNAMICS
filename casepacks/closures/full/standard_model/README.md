# standard_model_stub

**Standard Model Demonstration Stub**

> ⚠️ **Stub demonstration unit.** This casepack provides minimal packaging
> coverage for the `standard_model` domain so that every registered Tier-2
> domain has a casepack of record under `closures/full/`. The Tier-1 row is
> a uniform 3-channel Stable-regime demonstration; it does not represent
> domain-specific empirical data.

## Status

- **Casepack ID**: `standard_model_stub`
- **Contract**: `SM.INTSTACK.v1`
- **Canon anchors**: `canon/sm_anchors.yaml`
- **Status**: CONFORMANT (structural stub)
- **Replacement target**: domain-specific raw measurements from `closures/standard_model/`

## What it validates

The stub validates the casepack packaging contract: schema, semantic rules,
Tier-1 identities (F + ω = 1, IC ≤ F, IC = exp(κ)), and CONFORMANT verdict
through `umcp validate`. It does **not** demonstrate the rich domain-specific
phenomena that live in the closure code — those will arrive when the stub
is welded into a full demonstration unit.

## How to validate

```bash
umcp validate casepacks/closures/full/standard_model
```

## Where the real domain lives

Closure code, theorems, and tests for `standard_model` live at:

- Closures: `closures/standard_model/`
- Tests: `tests/test_*standard*` (and related)
- Full description: `casepacks/TAXONOMY.md` (Coverage Gap section)

## Lineage

- Created: 2026-05-10 (Phase 4 reorg follow-up)
- Reason: TAXONOMY.md Coverage Gap — `standard_model` domain had closure
  code and tests but no packaged demonstration unit.
