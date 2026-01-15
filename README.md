![CI](../../actions/workflows/validate.yml/badge.svg)

# UMCP — Universal Measurement Contract Protocol (Metadata + Runnable Validator Surface)

This repository is the metadata + runnable validator surface for UMCP (Universal Measurement Contract Protocol). It is organized as a contract-first, artifact-driven system: what matters is not prose claims, but frozen contracts, pinned closures, formally defined schemas, and receipts that can be revalidated by third parties.

UMCP’s core intent is that a reviewer can verify what was frozen, what was computed, and what is being claimed without relying on hidden defaults or implementation-specific assumptions.

---

## Contents (high-level)

1. Canon anchors (identifiers, not claims)  
2. Frozen contract boundary (UMA.INTSTACK.v1)  
3. Closures (explicit complements required to execute the contract)  
4. Schemas (machine validation for all artifacts)  
5. Validator rules (portable semantic checks)  
6. Python validator CLI (`umcp validate`)  
7. Tests (`pytest`)  
8. CI workflow (GitHub Actions)  
9. Publication receipts (SS1m / seam receipt)  
10. Repo conventions and “how to run” instructions  

---
## CasePacks (runnable/publication units)

A CasePack is a folder under `casepacks/<id>/` with:

- `manifest.json` (declares references + expected outputs)
- `raw_measurements.*` (inputs)
- `expected/psi.csv`
- `expected/invariants.json`
- `expected/ss1m_receipt.json`
- optionally `expected/seam_receipt.json` (only for continuity claims)

Example:
- [`casepacks/hello_world/`](casepacks/hello_world/)

## Start here (how to run)

Everything is run from the repository root: the folder that contains `pyproject.toml`.

### A. Local run (Windows Git Bash / macOS / Linux)

1) Create and activate a virtual environment
      
## Quick links

- Canon anchors (identifiers + defaults): [`canon/anchors.yaml`](canon/anchors.yaml)
- Frozen contract (UMA.INTSTACK.v1): [`contracts/UMA.INTSTACK.v1.yaml`](contracts/UMA.INTSTACK.v1.yaml)
- Closure registry (pins active closure set): [`closures/registry.yaml`](closures/registry.yaml)
- Schemas folder (all formal definitions): [`schemas/`](schemas/)
- Validator rules: [`validator_rules.yaml`](validator_rules.yaml)
- Example CasePack: [`casepacks/hello_world/`](casepacks/hello_world/)
- Test suite: [`tests/`](tests/)
- Validator CLI source: [`src/umcp/cli.py`](src/umcp/cli.py)
- CI workflow: [`.github/workflows/validate.yml`](.github/workflows/validate.yml)
## Repository map

- Canon: [`canon/`](canon/)
- Contracts: [`contracts/`](contracts/)
- Closures: [`closures/`](closures/)
- Schemas: [`schemas/`](schemas/)
- CasePacks: [`casepacks/`](casepacks/)
- Source: [`src/`](src/)
- Tests: [`tests/`](tests/)
- CI: [`.github/workflows/`](.github/workflows/)

## What goes where

| If you are adding… | Put it here | Example |
|---|---|---|
| A new anchor / threshold update | `canon/anchors.yaml` | Update `regimes.*` |
| A new contract version | `contracts/` | `contracts/UMA.INTSTACK.v2.yaml` |
| A new closure set | `closures/` | `closures/gamma.*.yaml` + update `closures/registry.yaml` |
| A new schema | `schemas/` | `schemas/my_artifact.schema.json` |
| A new runnable/publication example | `casepacks/<id>/` | `casepacks/hello_world/` |
| New docs for users | `docs/` | `docs/quickstart.md` |
| Code changes to validator | `src/umcp/` | `src/umcp/cli.py` |
| Tests for new rules | `tests/` | `tests/test_semantic_rules.py` |
## Provenance and change control

- Contracts and closure sets should be treated as immutable once referenced by a published CasePack.
- If semantics change, version the contract/closure registry and create a new CasePack id.
- Canon anchors are identifiers + frozen defaults; they are not claims.

## Validator output (what to read)

`umcp validate` prints JSON. The most important fields are:

- `run_status`: `CONFORMANT` or `NONCONFORMANT`
- `targets[*].issues`: list of issues with `severity`, `code`, `path`, `json_pointer`, and `hint`
- `summary.counts.errors`: quick failure count


## Common tasks
  ```markdown
### If you get stuck
- Confirm you are in the repo root: `ls` should show `pyproject.toml`.
- Confirm venv is active: your prompt shows `(.venv)`.
macOS/Linux:

### Install (one-time per machine)
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install -U pip
pip install -e ".[test]"
  umcp validate .
  pytest
  umcp validate . --out validator.result.json

python -m venv .venv
source .venv/bin/activate


