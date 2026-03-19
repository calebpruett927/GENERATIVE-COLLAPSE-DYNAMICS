# UMCP — Universal Measurement Contract Protocol

[![PyPI](https://img.shields.io/pypi/v/umcp)](https://pypi.org/project/umcp/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/blob/main/LICENSE)
[![CI](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/actions/workflows/validate.yml/badge.svg)](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/actions)
[![Tests: 8,397](https://img.shields.io/badge/tests-8%2C397-brightgreen.svg)](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/tree/main/tests)

**A contract-first validation framework for reproducible computational workflows.**

UMCP validates that computational results conform to mathematical contracts — frozen evaluation rules that pin normalization, thresholds, and return conditions *before* any evidence is generated. Every run produces a three-valued verdict: **CONFORMANT**, **NONCONFORMANT**, or **NON\_EVALUABLE**.

Built on **Generative Collapse Dynamics (GCD)**, a measurement theory derived from a single axiom:

> *"Collapse is generative; only what returns is real."*

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Contract-first validation** | Define mathematical contracts before evidence. Frozen parameters ensure reproducibility. |
| **Tier-1 kernel** | Six invariants (F, ω, S, C, κ, IC) computed from any bounded trace vector — domain-independent. |
| **20 scientific domains** | From particle physics and cosmology to neuroscience and finance — all through one kernel. |
| **8,397 tests** | Comprehensive test suite with 174 closure modules and 241 proven theorems. |
| **Three-valued verdicts** | Never boolean. Always CONFORMANT / NONCONFORMANT / NON\_EVALUABLE. |
| **Interactive dashboard** | 46-page Streamlit dashboard for real-time kernel exploration. |
| **CLI + Python API** | Full command-line interface and programmatic access. |
| **Optional C++17 accelerator** | 50–80× speedup via pybind11. Falls back to NumPy transparently. |

---

## Installation

```bash
# Core (validation engine + kernel computation)
pip install umcp

# With interactive dashboard
pip install umcp[viz]

# With REST API server
pip install umcp[api]

# Everything (dev tools + dashboard + API)
pip install umcp[all]
```

**Requires Python ≥ 3.11**. Core dependencies: `numpy`, `scipy`, `pyyaml`, `jsonschema`.

---

## Quick Start

### Python API

```python
import numpy as np
from umcp import compute_kernel, compute_full, validate

# 1. Compute kernel invariants from a trace vector
kernel = compute_kernel(
    c=np.array([0.95, 0.88, 0.92, 0.85]),   # 4-channel trace (values in [0,1])
    w=np.array([0.25, 0.25, 0.25, 0.25]),    # uniform weights (sum to 1.0)
    tau_R=5.0,                                 # return time
)

print(f"Fidelity (F):    {kernel.F:.4f}")      # What survives collapse
print(f"Drift (ω):       {kernel.omega:.4f}")  # What is lost (ω = 1 − F)
print(f"Entropy (S):     {kernel.S:.4f}")      # Bernoulli field entropy
print(f"Curvature (C):   {kernel.C:.4f}")      # Channel heterogeneity
print(f"Log-integrity:   {kernel.kappa:.4f}")   # ln(geometric mean)
print(f"Integrity (IC):  {kernel.IC:.4f}")      # Multiplicative coherence

# 2. Full computation with regime classification
result = compute_full([0.95, 0.88, 0.92, 0.85])
print(f"Regime: {result.regime}")  # STABLE, WATCH, or COLLAPSE

# 3. Validate a casepack against its contract
result = validate("casepacks/hello_world")
print(f"Status: {result.status}")           # CONFORMANT / NONCONFORMANT / NON_EVALUABLE
print(f"Errors: {result.error_count}")
```

### CLI

```bash
# Validate a casepack
umcp validate casepacks/hello_world

# Strict mode (publication-grade)
umcp validate casepacks/hello_world --strict

# Quick kernel computation
umcp-calc -c 0.95,0.88,0.92,0.85

# Launch interactive dashboard (localhost:8501)
umcp-dashboard

# Start REST API server (localhost:8000)
umcp-api
```

---

## The Kernel — Six Invariants

The kernel K maps any bounded trace vector **c** ∈ \[0,1\]ⁿ with weights **w** to six invariants:

| Symbol | Name | Formula | Measures |
|--------|------|---------|----------|
| **F** | Fidelity | F = Σ wᵢcᵢ | What survives collapse |
| **ω** | Drift | ω = 1 − F | What is lost |
| **S** | Entropy | Bernoulli field entropy | Uncertainty of the collapse field |
| **C** | Curvature | stddev(cᵢ) / 0.5 | Channel heterogeneity |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ) | Logarithmic coherence |
| **IC** | Integrity | IC = exp(κ) | Multiplicative coherence |

**Three structural identities** hold by construction:
- **F + ω = 1** — Duality identity (exact to machine precision)
- **IC ≤ F** — Integrity cannot exceed fidelity
- **IC = exp(κ)** — Log-integrity relation

These reduce 6 outputs to **3 effective degrees of freedom** (F, κ, C).

---

## Regime Classification

The kernel maps to three regimes via frozen threshold gates:

| Regime | Condition | Meaning |
|--------|-----------|---------|
| **Stable** | ω < 0.038 ∧ F > 0.90 ∧ S < 0.15 ∧ C < 0.14 | High coherence, minimal drift |
| **Watch** | Intermediate (Stable gates not all met, ω < 0.30) | Partial coherence loss |
| **Collapse** | ω ≥ 0.30 | Significant structural dissolution |
| **+Critical** | IC < 0.30 (overlay on any regime) | Integrity dangerously low |

---

## 20 Scientific Domains

Each domain provides closure modules that map real-world data to trace vectors:

| Domain | What It Measures |
|--------|-----------------|
| **Standard Model** | 31 particles → 8-channel kernel (27 proven theorems) |
| **Nuclear Physics** | Binding energy, decay chains, QGP/RHIC confinement |
| **Quantum Mechanics** | Wavefunction coherence, entanglement, FQHE |
| **Atomic Physics** | 118 elements through periodic kernel |
| **Astronomy** | Stellar classification, HR diagram analysis |
| **Cosmology (Weyl)** | Modified gravity, cosmological coherence |
| **Materials Science** | 118-element database, crystal structures |
| **Finance** | Portfolio continuity, market coherence |
| **Kinematics** | Motion analysis, phase space trajectories |
| **Evolution** | 40 organisms, 10-channel brain kernel |
| **Consciousness** | 20 systems, coherence kernel, 7 theorems |
| **Clinical Neuroscience** | Cortical/structural/metabolic/systemic kernel |
| **Dynamic Semiotics** | 30 sign systems, 8-channel semiotic kernel |
| **Spacetime Memory** | 40 entities, budget-surface kernel |
| ... and 6 more | GCD, RCFT, Security, Everyday Physics, Continuity Theory, Awareness-Cognition |

---

## Validation Pipeline

```
umcp validate <target>
  → Schema validation (JSON Schema Draft 2020-12)
  → Semantic rule checks
  → Kernel identity verification (F + ω = 1, IC ≤ F, IC = exp(κ))
  → Regime classification
  → SHA-256 integrity check
  → Three-valued verdict → ledger append
```

---

## Optional Dependencies

```bash
pip install umcp[viz]            # Streamlit dashboard + Plotly + Pandas
pip install umcp[api]            # FastAPI REST server
pip install umcp[communications] # Dashboard + API combined
pip install umcp[dev]            # pytest, ruff, mypy, pre-commit
pip install umcp[test]           # pytest + coverage
pip install umcp[cpp]            # pybind11 for C++ accelerator build
pip install umcp[all]            # Everything
```

---

## Project Structure

```
src/umcp/               # Core validation engine
├── frozen_contract.py   # Frozen parameters + kernel computation
├── kernel_optimized.py  # Lemma-based kernel with diagnostics
├── validator.py         # 16-file casepack validator
├── seam_optimized.py    # Seam budget computation
├── cli.py               # Full CLI (umcp validate, list, health, ...)
├── dashboard/           # 46-page Streamlit dashboard
├── fleet/               # Distributed validation (scheduler, workers, queue)
└── ...

closures/                # 20 domain closure modules (174 total)
contracts/               # Versioned mathematical contracts (YAML)
casepacks/               # 24 self-contained validation packages
schemas/                 # JSON Schema Draft 2020-12 definitions
```

---

## CLI Entry Points

| Command | Purpose |
|---------|---------|
| `umcp` | Main CLI — validate, list, health, integrity checks |
| `umcp-calc` | Universal kernel calculator |
| `umcp-dashboard` | Launch Streamlit dashboard |
| `umcp-api` | Launch FastAPI REST server |
| `umcp-ext` | Extension management |
| `umcp-finance` | Finance domain CLI |

---

## Links

- **Repository**: [github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS)
- **Documentation**: [README (full)](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/blob/main/README.md)
- **Quick Start Tutorial**: [QUICKSTART_TUTORIAL.md](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/blob/main/QUICKSTART_TUTORIAL.md)
- **Changelog**: [CHANGELOG.md](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/blob/main/CHANGELOG.md)
- **Issues**: [GitHub Issues](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/issues)
- **License**: [MIT](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/blob/main/LICENSE)
