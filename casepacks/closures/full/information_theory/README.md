# information_theory_stub

**Information Theory Demonstration Stub**

> ⚠️ **Stub demonstration unit.** This casepack provides minimal packaging
> coverage for the `information_theory` domain so that every registered Tier-2
> domain has a casepack of record under `closures/full/`. The Tier-1 row is
> a uniform 3-channel Stable-regime demonstration; it does not represent
> domain-specific empirical data.

## Status

- **Casepack ID**: `information_theory_stub`
- **Contract**: `UMA.INTSTACK.v1`
- **Canon anchors**: `canon/anchors.yaml`
- **Status**: CONFORMANT (structural stub)
- **Replacement target**: domain-specific raw measurements from `closures/information_theory/`

## What it validates

The stub validates the casepack packaging contract: schema, semantic rules,
Tier-1 identities (F + ω = 1, IC ≤ F, IC = exp(κ)), and CONFORMANT verdict
through `umcp validate`. It does **not** demonstrate the rich domain-specific
phenomena that live in the closure code — those will arrive when the stub
is welded into a full demonstration unit.

## How to validate

```bash
umcp validate casepacks/closures/full/information_theory
```

## Where the real domain lives

Closure code, theorems, and tests for `information_theory` live at:

- Closures: `closures/information_theory/`
- Tests: `tests/test_*information*` (and related)
- Full description: `casepacks/TAXONOMY.md` (Coverage Gap section)

## Lineage

- Created: 2026-05-10 (Phase 4 reorg follow-up)
- Reason: TAXONOMY.md Coverage Gap — `information_theory` domain had closure
  code and tests but no packaged demonstration unit.
