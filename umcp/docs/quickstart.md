# docs/quickstart.md

UMCP Quickstart

This quickstart gets you from “fresh clone” to a conformant end-to-end run using the included `casepacks/hello_world` example. It also shows the minimal steps to create a new CasePack that is publication-safe and validator-friendly.

Prerequisites

You need:
- Python 3.11+ (3.10 may work depending on your dependencies)
- Git
- A POSIX shell (macOS/Linux) or PowerShell on Windows

Repository orientation

UMCP repositories separate canon inputs (tracked) from execution outputs (ignored). The core tracked folders are:
- `canon/` (identifiers and frozen defaults; not claims)
- `contracts/` (e.g., UMA.INTSTACK.v1)
- `closures/` (explicit closures and closure registry)
- `schemas/` (machine validation rules)
- `casepacks/` (auditable runs: raw inputs + expected outputs + manifest)

Quickstart: install + validate + run hello world

1) Clone and install editable

```bash
git clone <YOUR_REPO_URL>
cd umcp

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -U pip
pip install -e .
umcp validate .
