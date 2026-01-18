[![CI](../../actions/workflows/validate.yml/badge.svg)](../../actions/workflows/validate.yml)

# UMCP — Universal Measurement Contract Protocol (Metadata + Runnable Validator Surface)

This repository contains the metadata and runnable validator for the Universal Measurement Contract Protocol (UMCP).  UMCP is designed as a **contract‑first, artifact‑driven** system.  Instead of prose alone, you’ll find frozen contracts, pinned closure registries, machine‑readable schemas, and receipts that can be re‑validated by third parties.  The goal is for reviewers to verify exactly what was frozen, what was computed, and what claims are made—without hidden defaults or implementation‑specific assumptions.

## Contents

1. **Canon anchors** – Stable identifiers and default numeric thresholds.  
2. **Contracts** – Frozen boundaries defining Tier‑1 kernel semantics (e.g., `UMA.INTSTACK.v1`).  
3. **Closures** – Explicit complements (Γ forms, return‑domain generators, norms) that complete a contract.  
4. **Schemas** – JSON Schema files describing valid structures for all artifacts.  
5. **Validator rules** – Portable semantic checks enforced at runtime.  
6. **Validator CLI** – A Python entrypoint (`umcp validate`) to run schema and semantic validation.  
7. **CasePacks** – Runnable publication units (inputs, invariants, receipts).  
8. **Tests** – Pytest suite for regression.  
9. **CI workflow** – GitHub Actions configuration (`validate.yml`) that runs the validator and tests.  
10. **Quick start** – How to set up and run the validator locally.

---

## Merge Verification

To verify that content has been successfully merged and the repository is in a healthy state, run:

```bash
./scripts/check_merge_status.sh
```

This script checks:
- Git status (clean working tree)
- Merge conflict artifacts
- Test suite (all tests passing)
- UMCP validator (CONFORMANT status)

For a detailed merge verification report, see [`MERGE_VERIFICATION.md`](MERGE_VERIFICATION.md).

---

## CasePacks (runnable publication units)

A CasePack is a self‑contained folder under `casepacks/<id>/` that holds:

- `manifest.json` – Pins the contract ID, version, closure registry ID, and any explicit overrides.  
- `raw_measurements.*` – Inputs used to produce a bounded trace (optional for L0 examples).  
- `expected/psi.csv` – Bounded trace row(s) with out-of-range (OOR) and missingness flags.  
- `expected/invariants.json` – Tier‑1 invariants (`ω`, `F`, `S`, `C`, `τ_R`, `κ`, `IC`) computed on Ψ\_ε(t).  
- `expected/ss1m_receipt.json` – The minimum audit receipt for the run.  
- `expected/seam_receipt.json` – Only when continuity (weld) is claimed.  

Example CasePack: [`casepacks/hello_world/`](casepacks/hello_world/)

---

## Quick start

All commands assume you are in the repository root (the folder containing `pyproject.toml`).  Python 3.8 or later is required.

### Set up a virtual environment

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[test]"
