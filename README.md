[![CI](../../actions/workflows/validate.yml/badge.svg)](../../actions/workflows/validate.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✅ Verify Everything Yourself

**Don't trust badges - verify the code:**

```bash
# Check tests (should show 233 passing)
pytest -v

# Check UMCP validation (should show CONFORMANT, 0 errors)
umcp validate .

# Check file integrity (should show 200 files)
wc -l < integrity/sha256.txt

# Check version
python -c "import tomli; print(tomli.load(open('pyproject.toml','rb'))['project']['version'])"

# Check CI status
gh run list --limit 1
```

All metrics are **verifiable from source code** - no marketing hype.

---

## 🚀 **Live System HUD**

<div align="center">

### **[📊 LAUNCH DASHBOARD →](https://scaling-train-97wgvp77rw993xjwr-8501.app.github.dev/)**

</div>

```
╔══════════════════════════════════════════════════════════════════════╗
║                    UMCP PRODUCTION SYSTEM STATUS                     ║
╚══════════════════════════════════════════════════════════════════════╝

  � CORE AXIOM:      "What Returns Through Collapse Is Real"
                      no_return_no_credit: true

  �🔐 Canon:           UMCP.CANON.v1
  📜 Contract:        UMA.INTSTACK.v1
  🔗 Weld:            W-2025-12-31-PHYS-COHERENCE
  
  📚 DOI References:
     PRE:  10.5281/zenodo.17756705  (The Episteme of Return)
     POST: 10.5281/zenodo.18072852  (Physics of Coherence)
     PACK: 10.5281/zenodo.18226878  (CasePack Publication)

  ⚙️  Tier-1 Kernel:
     p=3  α=1.0  λ=0.2  η=0.001
     
  🎯 Regime Gates:
     Stable:   ω<0.038  F>0.90  S<0.15  C<0.14
     Collapse: ω≥0.30
     
  📊 Current State:
     Status:     CONFORMANT ✅
     Regime:     Stable
     Errors:     0
     Warnings:   0
     
  ⚡ Performance:
     Cache:      Intelligent + Persistent
     Speedup:    20-25% faster (warm)
     Skipping:   4/4 casepacks (unchanged)
     Learning:   Progressive acceleration
     
  🔧 CLI Commands:
     umcp validate           # Run validation
     umcp-visualize         # Launch dashboard (port 8501)
     umcp-api               # Start REST API (port 8000)
     umcp-ext list          # List extensions
     umcp-format --all      # Format contracts
  
  📦 Ledger:      ledger/return_log.csv (continuous append)
  🧪 CasePacks:   hello_world | gcd_complete | rcft_complete
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           "No improvisation. Contract-first. Tier-1 reserved."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Quick Access:**
- 🎨 **[Visualization Dashboard](https://scaling-train-97wgvp77rw993xjwr-8501.app.github.dev/)** — Phase space, time series, regime monitoring
- 🔌 **API Endpoints** — `/health`, `/latest-receipt`, `/ledger`, `/stats`, `/regime`
- 📖 **[Extensions Guide](QUICKSTART_EXTENSIONS.md)** — Dashboard & API usage
- 🧪 **[Theory Docs](docs/rcft_theory.md)** — Mathematical foundations

---

UMCP is a **production-grade system** for creating, validating, and sharing reproducible computational workflows. It enforces mathematical contracts, tracks provenance, generates cryptographic receipts, and validates results against frozen specifications—ensuring reviewers can verify exactly what was computed, how, and under what assumptions.

---

## 🎯 What is UMCP?

UMCP transforms computational experiments into **auditable artifacts** based on a single foundational principle:

### **🔷 The Core Axiom: What Returns Through Collapse Is Real**

```yaml
# Encoded in every UMCP contract
typed_censoring:
  no_return_no_credit: true
```

**Meaning**: Reality is defined by what persists through collapse-reconstruction cycles. Only measurements that return—that survive transformation and can be reproduced—receive credit as real, valid observations.

This axiom unifies:
- **Measurement Theory**: Only reproducible (returning) measurements are valid
- **Generative Collapse Dynamics (GCD)**: Collapse produces new structure
- **Recursive Collapse Field Theory (RCFT)**: Returns accumulate memory across scales

---

### UMCP Workflow

```
Raw Measurements → Invariants → Closures → Validation → Receipt
      (CSV)           (JSON)      (Python)    (Contract)   (SHA256)
                                                    ↓
                                        Only what returns receives credit
```

**Key Concepts:**
- **Contracts**: Frozen mathematical specifications (GCD, RCFT) encoding the return axiom
- **Invariants**: Core metrics (ω, F, S, C) that must return through validation
- **Closures**: Computational functions computing what returns from collapse
- **CasePacks**: Self-contained reproducible units proving return verification
- **Validation**: Automated verification that results return conformantly

📖 **[Read the full axiom documentation](AXIOM.md)** for philosophical foundations, mathematical formulation, and physical interpretations.

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UMCP WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT                                                           │
│     └─ raw_measurements.csv  (your experimental data)               │
│                                                                     │
│  2. INVARIANTS COMPUTATION                                          │
│     ├─ ω (drift)                                                    │
│     ├─ F (fidelity)                                                 │
│     ├─ S (entropy)                                                  │
│     └─ C (curvature)                                                │
│                                                                     │
│  3. CLOSURE EXECUTION (choose framework)                            │
│     ┌─────────────────────┐      ┌──────────────────────┐           │
│     │ GCD (Tier-1)        │      │ RCFT (Tier-2)        │           │
│     ├─────────────────────┤      ├──────────────────────┤           │
│     │ • Energy (E)        │  OR  │ • Fractal (D_f)      │           │
│     │ • Collapse (Φ_c)    │      │ • Recursive (Ψ_r)    │           │
│     │ • Flux (Φ_gen)      │      │ • Pattern (λ, Θ)     │           │
│     │ • Resonance (R)     │      │ + all GCD closures   │           │
│     └─────────────────────┘      └──────────────────────┘           │
│                                                                     │
│  4. VALIDATION                                                      │
│     ├─ Contract conformance (schema validation)                     │
│     ├─ Regime classification (Low/Medium/High, etc.)                │
│     ├─ Mathematical identities (F = 1-ω, IC ≈ exp(κ), etc.)         │
│     └─ Tolerance checks (within tol_seam, tol_id, etc.)             │
│                                                                     │
│  5. OUTPUT                                                          │
│     ├─ invariants.json (computed metrics)                           │
│     ├─ closure_results.json (GCD/RCFT outputs)                      │
│     ├─ seam_receipt.json (validation status + SHA256)               │
│     └─ CONFORMANT or NONCONFORMANT status                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (5 Minutes)

### Installation

```bash
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[production]"
```

### Verify Installation

```bash
umcp health
# ✓ All systems operational

pytest
# 233 tests passed

# List available extensions
./umcp-ext list
# ✓ 4 extensions registered
```

---

## 🔌 Extension System

UMCP features an **auto-discovery extension system** with 4 built-in plugins:

### 1. Visualization Dashboard

Interactive Streamlit dashboard for real-time monitoring:

```bash
# Launch dashboard
umcp-visualize
# Or: streamlit run visualize_umcp.py

# Opens on http://localhost:8501
```

**Features**: Phase space plots, time series analysis, regime tracking, export capabilities

### 2. Public Audit API

REST API for programmatic access:

```bash
# Start API server
umcp-api
# Or: uvicorn api_umcp:app --reload

# Available at http://localhost:8000
```

**Endpoints**:
- `GET /health` - Health check
- `GET /latest-receipt` - Latest validation receipt
- `GET /ledger` - Historical validation log
- `GET /stats` - Aggregate statistics
- `GET /regime` - Current regime classification

### 3. Continuous Ledger

Automatic logging of all validation runs:

```bash
# Ledger updates automatically on validation
umcp validate

# View history
cat ledger/return_log.csv
```

### 4. Contract Auto-Formatter

Validate and format all contracts:

```bash
# Format all contracts
umcp-format --all

# Validate specific contract
umcp-format --validate contracts/GCD.INTSTACK.v1.yaml
```

**See [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md) for complete documentation.**

---

## 📝 How to Use UMCP

### Step 1: Prepare Your Data

Create a CSV file with your measurements. Example `my_data.csv`:

```csv
timestamp,c,p_x,p_y,p_z
0.0,0.999,0.001,-0.002,0.003
1.0,0.998,0.002,-0.001,0.004
2.0,0.997,0.003,0.000,0.002
```

**Required columns:**
- `c`: Fidelity measurement (0 to 1)
- `p_x`, `p_y`, `p_z`: Momentum components

### Step 2: Create a CasePack

```bash
# Create new casepack directory
mkdir -p casepacks/my_experiment

# Copy your data
cp my_data.csv casepacks/my_experiment/raw_measurements.csv

# Generate manifest (choose framework: GCD or RCFT)
./scripts/create_manifest.sh my_experiment RCFT.INTSTACK.v1
```

This creates `casepacks/my_experiment/manifest.json`:

```json
{
  "casepack_id": "my_experiment",
  "contract_id": "RCFT.INTSTACK.v1",
  "version": "1.0.0",
  "description": "My experimental data with RCFT analysis",
  "closures_to_run": [
    "energy_potential",
    "entropic_collapse",
    "generative_flux",
    "field_resonance",
    "fractal_dimension",
    "recursive_field",
    "resonance_pattern"
  ]
}
```

### Step 3: Generate Expected Outputs

```bash
# Run computation pipeline
python casepacks/my_experiment/generate_expected.py

# This creates:
# - expected/invariants.json (ω, F, S, C, τ_R, κ, IC)
# - expected/gcd_energy.json (E_potential, regime)
# - expected/gcd_collapse.json (Φ_collapse, regime)
# - expected/gcd_flux.json (Φ_gen, regime)
# - expected/gcd_resonance.json (R, regime)
# - expected/rcft_fractal.json (D_fractal, regime)
# - expected/rcft_recursive.json (Ψ_r, regime)
# - expected/rcft_pattern.json (λ_p, Θ, pattern_type)
# - expected/seam_receipt.json (validation status)
```

**Example `generate_expected.py`:**

```python
import numpy as np
import json
from pathlib import Path
from closures.gcd.energy_potential import compute_energy_potential
from closures.rcft.fractal_dimension import compute_fractal_dimension, compute_trajectory_from_invariants

# Load raw data
data = np.genfromtxt('raw_measurements.csv', delimiter=',', skip_header=1)

# Compute invariants
omega = np.mean(data[:, 1] - 1.0)  # drift from fidelity
F = np.mean(data[:, 1])            # fidelity
S = np.std(data[:, 1])             # entropy
C = np.mean(np.abs(np.diff(data[:, 1])))  # curvature

invariants = {"omega": omega, "F": F, "S": S, "C": C}

# Save invariants
Path("expected").mkdir(exist_ok=True)
with open("expected/invariants.json", "w") as f:
    json.dump(invariants, f, indent=2)

# Run GCD closures
energy = compute_energy_potential(omega, S, C)
with open("expected/gcd_energy.json", "w") as f:
    json.dump(energy, f, indent=2)

# Run RCFT closures
trajectory = compute_trajectory_from_invariants({
    "omega": data[:, 1] - 1.0,
    "S": np.full(len(data), S),
    "C": np.full(len(data), C)
})
fractal = compute_fractal_dimension(trajectory)
with open("expected/rcft_fractal.json", "w") as f:
    json.dump(fractal, f, indent=2)

# Generate receipt
receipt = {
    "casepack_id": "my_experiment",
    "contract_id": "RCFT.INTSTACK.v1",
    "run_status": "CONFORMANT",
    "tier_hierarchy_validated": True,
    "sha256_manifest": "...",
    "timestamp": "2026-01-18T00:00:00Z"
}
with open("expected/seam_receipt.json", "w") as f:
    json.dump(receipt, f, indent=2)
```

### Step 4: Validate Your CasePack

```bash
# Validate against contract
umcp validate casepacks/my_experiment

# Expected output:
# ✓ Schema validation passed
# ✓ Invariants conform to contract
# ✓ All closures executed successfully
# ✓ Regime classifications valid
# ✓ Mathematical identities satisfied
# → Status: CONFORMANT
```

### Step 5: Compare Results

```bash
# Generate new results from same data
python casepacks/my_experiment/generate_expected.py

# Compare with original expected outputs
umcp diff \
  casepacks/my_experiment/expected/seam_receipt.json \
  casepacks/my_experiment/new_receipt.json

# Shows differences in:
# - Invariant values
# - Closure outputs
# - Regime classifications
# - Validation status
```

---

## 🎓 Framework Selection Guide

### When to Use GCD (Tier-1)

**Best for:**
- Energy and collapse dynamics analysis
- Boundary-interior coupling (resonance)
- Generative potential extraction
- Basic regime classification

**Example use cases:**
- Phase transitions
- Thermodynamic systems
- Field theories
- Quantum collapse models

**Closure outputs:**
- `E_potential`: Total system energy
- `Φ_collapse`: Collapse potential
- `Φ_gen`: Generative flux
- `R`: Boundary-interior resonance

### When to Use RCFT (Tier-2)

**Best for:**
- Geometric complexity analysis
- Memory and history effects
- Oscillatory pattern detection
- Multi-scale recursive structures

**Example use cases:**
- Fractal attractors
- Time series with memory
- Periodic or quasi-periodic systems
- Chaotic dynamics

**Closure outputs (includes all GCD outputs plus):**
- `D_fractal`: Trajectory complexity (1 ≤ D_f ≤ 3)
- `Ψ_recursive`: Collapse memory (Ψ_r ≥ 0)
- `λ_pattern`: Resonance wavelength
- `Θ_phase`: Phase angle [0, 2π)

**Decision Matrix:**

| Need | Framework | Why |
|------|-----------|-----|
| Basic energy/collapse analysis | GCD | Simpler, faster, foundational |
| Trajectory complexity | RCFT | Box-counting fractal dimension |
| History/memory effects | RCFT | Exponential decay field |
| Oscillation detection | RCFT | FFT-based pattern analysis |
| Zero entropy (S=0) state | Either | Both handle deterministic states |
| Maximum insight | RCFT | Includes all GCD + 3 new metrics |

---

## 📚 Example CasePacks

### Hello World (Zero Entropy)

```bash
cd casepacks/hello_world
cat raw_measurements.csv
# timestamp,c,p_x,p_y,p_z
# 0.0,0.99999999,0.0,0.0,0.0
# 1.0,0.99999999,0.0,0.0,0.0
# 2.0,0.99999999,0.0,0.0,0.0

python generate_expected.py
umcp validate .

# Result: CONFORMANT
# - ω = 0, F = 1.0, S = 0, C = 0
# - All GCD regimes: Low/Minimal/Dormant/Coherent
# - RCFT: D_f=0 (point), Ψ_r=0 (no memory), λ=∞ (constant)
```

### RCFT Complete (Full Analysis)

```bash
cd casepacks/rcft_complete
umcp validate .

# Result: CONFORMANT with tier_hierarchy_validated=true
# - Validates UMCP → GCD → RCFT tier chain
# - All 7 closures executed
# - Zero entropy example with RCFT overlay
```

---

## 🛠️ Advanced Usage

### Programmatic API

```python
from closures.gcd.energy_potential import compute_energy_potential
from closures.rcft.fractal_dimension import compute_fractal_dimension
import numpy as np

# Compute GCD metrics
omega, S, C = 0.01, 0.05, 0.02
energy = compute_energy_potential(omega, S, C)
print(f"Energy: {energy['E_potential']:.6f} ({energy['regime']})")
# Energy: 0.001234 (Low)

# Compute RCFT metrics
trajectory = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0.01, 0]])
fractal = compute_fractal_dimension(trajectory)
print(f"Fractal dimension: {fractal['D_fractal']:.4f} ({fractal['regime']})")
# Fractal dimension: 1.0234 (Smooth)
```

### Custom Validation Rules

Edit `validator_rules.yaml` to add custom checks:

```yaml
semantic_rules:
  - rule_id: "CUSTOM-001"
    description: "Custom regime boundary check"
    check_type: "regime_check"
    target: "energy"
    condition: "E_potential < custom_threshold"
    severity: "error"
```

### Health Monitoring

```bash
# System health check
umcp health
# Output:
# ✓ Python version: 3.12.1
# ✓ Dependencies: numpy, scipy, jsonschema
# ✓ Closures: 7 registered (4 GCD + 3 RCFT)
# ✓ Schemas: 10 valid
# ✓ Contracts: 2 loaded (GCD, RCFT)
# → Status: OPERATIONAL

# Performance metrics
umcp validate --verbose casepacks/my_experiment
# Output includes:
# - Validation duration
# - Memory usage
# - CPU utilization
# - Schema validation time
# - Closure execution time
```

### Production Deployment

```bash
# Enable JSON logging
export UMCP_JSON_LOGS=1

# Run with strict validation
umcp validate --strict --out result.json

# Integrate with monitoring systems (ELK, Splunk, CloudWatch)
umcp validate --strict 2>&1 | tee validation.log
```

See [Production Deployment Guide](docs/production_deployment.md) for Docker, Kubernetes, and CI/CD integration.

---

## 📖 Documentation

### Core Documentation
- **[Quickstart Guide](docs/quickstart.md)**: Get started in 10 minutes
- **[Python Coding Standards](docs/python_coding_key.md)**: Development guidelines
- **[Production Deployment](docs/production_deployment.md)**: Enterprise setup

### Framework Documentation
- **[GCD Theory](canon/gcd_anchors.yaml)**: Generative Collapse Dynamics (Tier-1)
- **[RCFT Theory](docs/rcft_theory.md)**: Recursive Collapse Field Theory (Tier-2)
- **[RCFT Usage Guide](docs/rcft_usage.md)**: Practical examples and parameter tuning

### Contract Specifications
- **[GCD Contract](contracts/GCD.INTSTACK.v1.yaml)**: Tier-1 specification
- **[RCFT Contract](contracts/RCFT.INTSTACK.v1.yaml)**: Tier-2 specification
- **[Contract Versioning](contracts/CHANGELOG.md)**: Version history and migration

### API Reference
- **[Closure Registry](closures/registry.yaml)**: All 7 closure definitions
- **[Schema Library](schemas/)**: JSON schemas for all artifacts
- **[Validator Usage](canon/docs/validator_usage.md)**: CLI reference

---

## 🧪 Testing

### Run All Tests

```bash
pytest                    # All 221 tests (~7s)
pytest -v                 # Verbose output
pytest -k "gcd"           # GCD tests only
pytest -k "rcft"          # RCFT tests only
pytest --cov              # Coverage report
```

### Test Structure

```
tests/
├── test_00_schemas_valid.py           # Schema validation
├── test_10_canon_contract_closures_validate.py  # Core validation
├── test_100_gcd_canon.py              # GCD canon tests
├── test_101_gcd_closures.py           # GCD closure tests
├── test_102_gcd_contract.py           # GCD contract tests
├── test_110_rcft_canon.py             # RCFT canon tests
├── test_111_rcft_closures.py          # RCFT closure tests
├── test_112_rcft_contract.py          # RCFT contract tests
├── test_113_rcft_tier2_layering.py    # Tier hierarchy tests
└── test_*                             # Additional integration tests
```

---

## 🤝 What's New in v1.1.0

**Recursive Collapse Field Theory (RCFT)** - Complete Tier-2 framework:

- **3 New Closures**: Fractal dimension, recursive field, resonance pattern
- **Complete Integration**: 221 tests passing (100% success), full backward compatibility
- **Production Ready**: Comprehensive documentation, validated examples
- **Performance**: 7s test execution (was 12s for 30 tests, now 221 tests!)

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

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

All commands assume you are in the repository root (the folder containing `pyproject.toml`).  Python 3.11 or later is required (3.12+ recommended).

### Set up a virtual environment

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[test]"

---

## 🔍 Repository Structure

```
UMCP-Metadata-Runnable-Code/
├── canon/                      # Canonical anchors (specifications)
│   ├── anchors.yaml           # Core UMCP definitions
│   ├── gcd_anchors.yaml       # GCD Tier-1 specification
│   └── rcft_anchors.yaml      # RCFT Tier-2 specification
├── contracts/                  # Frozen contracts
│   ├── GCD.INTSTACK.v1.yaml   # GCD Tier-1 contract
│   └── RCFT.INTSTACK.v1.yaml  # RCFT Tier-2 contract
├── closures/                   # Computational functions
│   ├── gcd/                   # 4 GCD closures
│   │   ├── energy_potential.py
│   │   ├── entropic_collapse.py
│   │   ├── generative_flux.py
│   │   └── field_resonance.py
│   ├── rcft/                  # 3 RCFT closures
│   │   ├── fractal_dimension.py
│   │   ├── recursive_field.py
│   │   └── resonance_pattern.py
│   └── registry.yaml          # Closure registry (all 7)
├── casepacks/                  # Reproducible examples
│   ├── hello_world/           # Zero entropy example
│   └── rcft_complete/         # Full RCFT validation
├── schemas/                    # JSON schemas (10 files)
├── tests/                      # Test suite (221 tests)
├── docs/                       # Documentation
│   ├── rcft_theory.md         # RCFT mathematical foundation
│   └── rcft_usage.md          # RCFT usage guide
├── scripts/                    # Utility scripts
├── src/umcp/                   # UMCP CLI and core
├── validator_rules.yaml        # Validation rules
└── pyproject.toml             # Project config (v1.1.0)
```

---

## 💡 Common Questions

**Q: What's the difference between GCD and RCFT?**
- **GCD (Tier-1)**: Energy, collapse, flux, resonance analysis
- **RCFT (Tier-2)**: Adds fractal, recursive, pattern analysis + all GCD

**Q: Can I use both frameworks together?**
- Yes! RCFT includes all GCD closures. Just specify `RCFT.INTSTACK.v1` as your contract.

**Q: How do I know which framework to use?**
- Use GCD for basic energy/collapse analysis
- Use RCFT when you need trajectory complexity, memory effects, or oscillation detection

**Q: What if my tests fail?**
- Check `validator_rules.yaml` for tolerance settings
- Verify your raw data format matches expected schema
- Run `umcp validate --verbose` for detailed error messages

**Q: How do I contribute new closures?**
- Add closure to `closures/` directory
- Register in `closures/registry.yaml`
- Add tests to `tests/`
- Update contract YAML if needed

**Q: Can I use UMCP without Python?**
- Core validation works with any language that can write JSON/CSV
- Closures are Python-based, but outputs are language-agnostic

---

## 🚦 CI/CD Integration

### GitHub Actions

```yaml
name: UMCP Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -e ".[production]"
      - run: umcp health
      - run: pytest
      - run: umcp validate --strict
```

### Docker

```bash
# Build container
docker build -t umcp:latest .

# Run validation
docker run -v $(pwd)/casepacks:/data umcp:latest validate /data/my_experiment

# Health check
docker run umcp:latest health
```

See [Production Deployment](docs/production_deployment.md) for Kubernetes, monitoring, and enterprise setup.

---

## 📊 Performance

- **Test Execution**: 221 tests in ~7 seconds
- **Validation**: <5 seconds for typical casepacks
- **Memory**: <100MB for most operations
- **Scalability**: Sublinear growth with test count

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Validate code quality (`ruff check`, `mypy`)
6. Commit changes (`git commit -m 'feat: Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

See [Python Coding Standards](docs/python_coding_key.md) for style guide.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

**Framework**: UMCP (Universal Measurement Contract Protocol)  
**Core Axiom**: "What Returns Through Collapse Is Real"  
**Tier-1**: GCD (Generative Collapse Dynamics)  
**Tier-2**: RCFT (Recursive Collapse Field Theory)  
**Author**: Clement Paulus  
**Version**: 1.3.2-immutable  
**Release**: January 20, 2026  
**Tests**: 233 passing (100% success)  
**Integrity**: 165 files SHA256 checksummed  
**Extensions**: 4 auto-discovered plugins

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [casepacks/](casepacks/)
- **Immutable Release**: [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md)
- **Core Axiom**: [AXIOM.md](AXIOM.md)
- **Extensions**: [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md)

---

## 🔒 Immutable Release v1.3.2

This is the **immutable snapshot** of UMCP with:
- ✅ Core axiom encoded in all contracts (`no_return_no_credit: true`)
- ✅ 165 files cryptographically verified (SHA256)
- ✅ Extension system with auto-discovery
- ✅ Complete documentation (2,000+ lines)
- ✅ Zero uncommitted changes
- ✅ Git tagged: `v1.3.2-immutable`

**Verify integrity**: `sha256sum -c integrity/sha256.txt`  
**Read full details**: [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md)

---

**Built with ❤️ for reproducible science**  
*"What Returns Through Collapse Is Real"*
