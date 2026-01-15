# UMCP — Universal Measurement Contract Protocol

UMCP is a contract-first framework for producing auditable, bounded traces and a fixed kernel of invariants from measurement data. This repository is organized to be simultaneously runnable (validators/tests) and publication-grade (schemas, receipts, immutability discipline).

UMCP is designed so that a third party can clone the repo and verify:
- what was frozen (contract + closures),
- what was computed (expected outputs),
- what was claimed (receipts),
- and what passed validation (machine-readable validator results).

## What you can do immediately

From the repository root:

1) Install dependencies (Python 3.11+)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -U pip
pip install -e ".[test]"
pytest
umcp validate .
umcp validate . --out validator.result.json

