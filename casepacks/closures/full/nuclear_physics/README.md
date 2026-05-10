# nuclear_physics_stub

**Nuclear Physics Demonstration Stub**

> ⚠️ **Stub demonstration unit.** This casepack provides minimal packaging
> coverage for the `nuclear_physics` domain so that every registered Tier-2
> domain has a casepack of record under `closures/full/`. The Tier-1 row is
> a uniform 3-channel Stable-regime demonstration; it does not represent
> domain-specific empirical data.

## Status

- **Casepack ID**: `nuclear_physics_stub`
- **Contract**: `NUC.INTSTACK.v1`
- **Canon anchors**: `canon/nuc_anchors.yaml`
- **Status**: CONFORMANT (structural stub)
- **Replacement target**: domain-specific raw measurements from `closures/nuclear_physics/`

## What it validates

The stub validates the casepack packaging contract: schema, semantic rules,
Tier-1 identities (F + ω = 1, IC ≤ F, IC = exp(κ)), and CONFORMANT verdict
through `umcp validate`. It does **not** demonstrate the rich domain-specific
phenomena that live in the closure code — those will arrive when the stub
is welded into a full demonstration unit.

## How to validate

```bash
umcp validate casepacks/closures/full/nuclear_physics
```

## Where the real domain lives

Closure code, theorems, and tests for `nuclear_physics` live at:

- Closures: `closures/nuclear_physics/`
- Tests: `tests/test_*nuclear*` (and related)
- Full description: `casepacks/TAXONOMY.md` (Coverage Gap section)

## Lineage

- Created: 2026-05-10 (Phase 4 reorg follow-up)
- Reason: TAXONOMY.md Coverage Gap — `nuclear_physics` domain had closure
  code and tests but no packaged demonstration unit.
