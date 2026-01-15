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

## Start here (how to run)

Everything is run from the repository root: the folder that contains `pyproject.toml`.

### A. Local run (Windows Git Bash / macOS / Linux)

1) Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate

