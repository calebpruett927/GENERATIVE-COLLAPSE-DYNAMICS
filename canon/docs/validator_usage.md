# Validator Usage

This document defines what the UMCP validator checks, how to run it, what "conformant" means in repo terms, and how to integrate validation into CI for publication-grade correctness.

## What the Validator Does

The validator enforces the contract-first discipline:

- **Canon anchors** exist and are well-formed (identifiers + frozen defaults; not claims)
- **Contract files** exist and satisfy schema
- **Closure registry** exists and closure files satisfy schema
- **CasePacks** follow strict layout and include minimum artifacts for auditability
- **Receipts**, when present, are structurally correct and internally consistent

The validator does not "interpret" your science. It enforces reproducible structure so that interpretation can be audited.

## Commands

### Validate Repository

```bash
# Validate everything in the repository
umcp validate .

# Validate specific CasePack
umcp validate casepacks/hello_world

# Strict mode (publication-grade)
umcp validate . --strict

# JSON output
umcp validate . --out result.json
```

### Other Commands

```bash
# Show version
umcp version

# List registered closures
umcp list-closures

# Show repo diff summary
umcp diff

# Health check
umcp health
```

## Run Status

| Status | Meaning |
|--------|---------|
| **CONFORMANT** | All checks pass; repo is structurally valid |
| **NONCONFORMANT** | One or more errors detected; repo needs fixes |
| **NON_EVALUABLE** | Validator could not complete (missing files, parse errors) |

## Validation Levels

### Baseline (default)

- Schema validation for all YAML/JSON files
- Required file existence
- Basic structural consistency

### Strict (`--strict`)

- All baseline checks
- Receipt field completeness
- Mathematical identity verification
- Publication-grade lint

## CI Integration

Add to GitHub Actions:

```yaml
name: Validate UMCP
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install umcp
      - run: umcp validate . --strict
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | CONFORMANT |
| 1 | NONCONFORMANT (errors found) |
| 2 | NON_EVALUABLE (cannot complete) |

## Related Documentation

- [Canon README](../README.md) — Canon standard overview
- [PROTOCOL_REFERENCE.md](../../PROTOCOL_REFERENCE.md) — Full protocol specification
- [CASEPACK_REFERENCE.md](../../CASEPACK_REFERENCE.md) — CasePack structure

---

*Part of the UMCP Canon Standard*
