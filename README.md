# UMCP — Universal Measurement Contract Protocol

UMCP is a contract-first framework for producing auditable, bounded traces and a fixed kernel of invariants from measurement data. This repository contains the frozen contract boundary, pinned closures, machine schemas, a validator CLI, and a test suite so a third party can verify conformance from artifacts alone.

The design goal is simple: a reviewer should be able to clone the repo and verify (1) what was frozen, (2) what was computed, and (3) what claims are supported by receipts—without relying on hidden defaults.

## Start here

If you are on Windows, run commands in Git Bash from the repository root (the folder that contains `pyproject.toml`).

To confirm you are in the repo root:

```bash
bash scripts/setup.sh
bash scripts/test.sh
bash scripts/validate.sh
pwd
ls
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -e ".[test]"
pytest
umcp validate .
umcp validate . --out validator.result.json

