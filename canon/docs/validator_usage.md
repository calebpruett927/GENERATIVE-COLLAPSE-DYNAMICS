
```markdown
# docs/validator_usage.md

Validator Usage

This document defines what the UMCP validator checks, how to run it, what “conformant” means in repo terms, and how to integrate validation into CI for publication-grade correctness.

What the validator is for

The validator enforces the contract-first discipline:
- Canon anchors exist and are well-formed (identifiers + frozen defaults; not claims)
- The contract file(s) exist and satisfy schema
- The closure registry exists and closure files satisfy schema
- CasePacks follow a strict layout and include the minimum artifacts for auditability
- Receipts, when present, are structurally correct and internally consistent at the field level

The validator does not “interpret” your science. It enforces reproducible structure so that interpretation can be audited.

Recommended commands

Validate everything in the repository:

```bash
umcp validate .
