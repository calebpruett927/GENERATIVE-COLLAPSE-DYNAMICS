[![CI](../../actions/workflows/validate.yml/badge.svg)](../../actions/workflows/validate.yml)
[![Production Ready](https://img.shields.io/badge/production-ready-brightgreen)](docs/production_deployment.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-221%20passing-success)](tests/)
[![GCD+RCFT](https://img.shields.io/badge/frameworks-GCD%20%2B%20RCFT-blueviolet)](docs/rcft_theory.md)

# UMCP — Universal Measurement Contract Protocol (Metadata + Runnable Validator Surface)

This repository contains the metadata and runnable validator for the Universal Measurement Contract Protocol (UMCP).  UMCP is designed as a **contract‑first, artifact‑driven** system with **production-grade monitoring and observability**.  Instead of prose alone, you'll find frozen contracts, pinned closure registries, machine‑readable schemas, and receipts that can be re‑validated by third parties.  The goal is for reviewers to verify exactly what was frozen, what was computed, and what claims are made—without hidden defaults or implementation‑specific assumptions.

## What's New in v1.1.0 🎉

**Recursive Collapse Field Theory (RCFT)** - A complete Tier-2 framework extending GCD with geometric and topological analysis:

- **3 New Closures**: Fractal dimension, recursive field strength, resonance pattern analysis
- **Complete Integration**: All 221 tests passing (100% success rate), full backward compatibility
- **Production Ready**: Comprehensive documentation ([theory](docs/rcft_theory.md), [usage](docs/rcft_usage.md)), validated with zero-entropy examples
- **Augmentation Philosophy**: RCFT augments GCD without override - all Tier-1 invariants remain frozen

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Contents

1. **Canon anchors** – Stable identifiers and default numeric thresholds (UMCP, GCD, RCFT).  
2. **Contracts** – Frozen boundaries defining Tier‑1 and Tier-2 semantics (`GCD.INTSTACK.v1`, `RCFT.INTSTACK.v1`).  
3. **Closures** – Explicit complements implementing the frameworks:
   - **GCD Tier-1** (4 closures): Energy potential, entropic collapse, generative flux, field resonance
   - **RCFT Tier-2** (3 closures): Fractal dimension, recursive field, resonance pattern
4. **Schemas** – JSON Schema files describing valid structures for all artifacts.  
5. **Validator rules** – Portable semantic checks enforced at runtime.  
6. **Validator CLI** – A Python entrypoint (`umcp validate`, `umcp health`) with structured logging.  
7. **CasePacks** – Runnable publication units (inputs, invariants, receipts) for GCD and RCFT.  
8. **Tests** – Comprehensive pytest suite (221 tests: 142 original + 56 RCFT + 23 integration).  
9. **CI workflow** – GitHub Actions configuration (`validate.yml`) that runs the validator and tests.  
10. **Production deployment** – [Complete guide](docs/production_deployment.md) for enterprise deployment.
11. **Monitoring & Observability** – Structured JSON logging, performance metrics, health checks.
12. **RCFT Documentation** – [Theory](docs/rcft_theory.md) and [Usage Guide](docs/rcft_usage.md) for Tier-2 overlay.

---

## Production Features ⭐

- **🏥 Health Checks**: `umcp health` command for system readiness monitoring
- **📊 Performance Metrics**: Track validation duration, memory usage, CPU utilization
- **📝 Structured Logging**: JSON-formatted logs for ELK, Splunk, CloudWatch integration
- **🐳 Container Ready**: Docker support with health check endpoints
- **☸️ Kubernetes**: Liveness and readiness probe examples
- **🔐 Audit Trail**: Cryptographic SHA256 receipts with git provenance
- **⚡ High Performance**: <5 second validation for typical repositories
- **🎯 Zero Technical Debt**: No TODO/FIXME/HACK markers, production-grade code quality

See the [Production Deployment Guide](docs/production_deployment.md) for details.

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[production]"
```

### Basic Usage

```bash
# Check system health
umcp health

# Validate repository (development mode)
umcp validate .

# Validate repository (production/strict mode)
umcp validate --strict

# Enable performance monitoring
umcp validate --strict --verbose

# Output validation receipt
umcp validate --strict --out validation-result.json

# Compare two receipts
umcp diff old-receipt.json new-receipt.json
```

### JSON Logging for Production

```bash
# Enable structured JSON logs for monitoring systems
export UMCP_JSON_LOGS=1
umcp validate --strict --verbose 2>&1 | tee validation.log
```

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
## Root-Level UMCP Files

In addition to CasePacks, this repository includes root-level UMCP configuration files for direct reference:

**Configuration** (YAML):
- [`manifest.yaml`](manifest.yaml) – Root-level CasePack manifest
- [`contract.yaml`](contract.yaml) – Contract specification
- [`observables.yaml`](observables.yaml) – Observable variable definitions
- [`embedding.yaml`](embedding.yaml) – Embedding transformation config
- [`return.yaml`](return.yaml) – Return domain specifications
- [`closures.yaml`](closures.yaml) – Closure registry references

**Data Files**:
- [`weights.csv`](weights.csv) – Weight coefficients
- [`derived/trace.csv`](derived/trace.csv) – Bounded trace Ψ_ε(t)
- [`derived/trace_meta.yaml`](derived/trace_meta.yaml) – Trace metadata

**Outputs**:
- [`outputs/invariants.csv`](outputs/invariants.csv) – Tier-1 invariants
- [`outputs/regimes.csv`](outputs/regimes.csv) – Regime classifications
- [`outputs/welds.csv`](outputs/welds.csv) – Continuity verification
- [`outputs/report.txt`](outputs/report.txt) – Validation report

**Integrity**:
- [`integrity/sha256.txt`](integrity/sha256.txt) – File checksums
- [`integrity/env.txt`](integrity/env.txt) – Python environment
- [`integrity/code_version.txt`](integrity/code_version.txt) – Git provenance

### Programmatic Access

```python
from umcp import get_umcp_files, get_closure_loader, get_root_validator

# Load any UMCP file
umcp = get_umcp_files()
manifest = umcp.load_manifest()
contract = umcp.load_contract()
invariants = umcp.load_invariants()

# Execute closures
loader = get_closure_loader()
result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)

# Validate system integrity
validator = get_root_validator()
validation_result = validator.validate_all()
print(f"Status: {validation_result['status']}")
```

See [docs/file_reference.md](docs/file_reference.md) and [docs/interconnected_architecture.md](docs/interconnected_architecture.md) for complete documentation.

### Demonstration

Run the interconnected system demonstration:

```bash
python examples/interconnected_demo.py
```

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
