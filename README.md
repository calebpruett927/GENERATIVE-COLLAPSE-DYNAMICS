<p align="center">
  <img src="https://img.shields.io/badge/UMCP-Universal%20Measurement%20Contract%20Protocol-6C63FF?style=for-the-badge" alt="UMCP">
</p>

<h1 align="center">🔬 Universal Measurement Contract Protocol</h1>

<p align="center">
  <strong>Transform computational experiments into auditable, reproducible artifacts with formal mathematical foundations</strong>
</p>

<p align="center">
  <a href="https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/actions/workflows/validate.yml"><img src="https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/actions/workflows/validate.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-1060%20passing-brightgreen?logo=pytest" alt="Tests: 1060 passing"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-1.5.0-blue" alt="Version: 1.5.0"></a>
  <a href="src/umcp/api_umcp.py"><img src="https://img.shields.io/badge/API-37%2B%20endpoints-orange?logo=fastapi" alt="API: 37+ endpoints"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-core-axiom">Core Axiom</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

<div align="center">

> **Core Axiom**: *"What Returns Through Collapse Is Real"*
>
> Reality is defined by what persists through collapse-reconstruction cycles.
> Only measurements that return—that survive transformation and can be reproduced—receive credit as real, valid observations.

</div>

---

## 📋 Table of Contents

<details>
<summary><strong>Click to expand</strong></summary>

- [🚀 Quick Start](#-quick-start)
- [🎯 Core Axiom](#-core-axiom)
- [✨ What Makes UMCP Different](#-what-makes-umcp-different)
- [📊 System Overview](#-system-overview)
- [🏗️ Architecture](#️-architecture)
- [🔧 CLI Commands](#-cli-commands)
- [🌐 REST API](#-rest-api)
- [📈 Visualization Dashboard](#-visualization-dashboard)
- [📦 Frameworks](#-frameworks)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [📂 Repository Map](#-repository-map)
- [🎓 Getting Started Tutorial](#-getting-started-tutorial)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

</details>

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| pip | Latest | Package management |
| git | Any | Version control |

### Installation

```bash
# Clone the repository
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install (choose your level)
pip install -e "."                    # Core only
pip install -e ".[api]"               # + REST API
pip install -e ".[viz]"               # + Dashboard
pip install -e ".[dev]"               # + Dev tools
pip install -e ".[all]"               # Everything
```

### Verify Installation

```bash
umcp health           # System health check
umcp validate .       # Validate repository
pytest                # Run 1002 tests
```

<details>
<summary><strong>📱 One-liner install</strong></summary>

```bash
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git && cd UMCP-Metadata-Runnable-Code && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]" && umcp health
```

</details>

---

## 🎯 Core Axiom

<table>
<tr>
<td width="60%">

### The Foundational Principle

UMCP is built on a single axiom that drives all design decisions:

> **"What Returns Through Collapse Is Real"**

This means:
- ✅ Only measurements that **return** (survive transformation) are valid
- ✅ No credit without **reproducibility**
- ✅ Mathematical contracts are **frozen** artifacts
- ✅ Provenance is **cryptographically verified**

```yaml
# Encoded in every UMCP contract
typed_censoring:
  no_return_no_credit: true
```

</td>
<td width="40%">

### Core Principle

**One-way dependency flow within a frozen run, with return-based canonization between runs.**

| Context | Rule |
|---------|------|
| **Within-run** | Frozen causes only—no back-edges, no retroactive tuning |
| **Between-run** | Continuity only by return-weld—new runs are canon-continuous only if seam returns and closes |

</td>
</tr>
</table>

---

## ✨ What Makes UMCP Different

<table>
<tr>
<th>Traditional Approaches</th>
<th>UMCP Adds</th>
</tr>
<tr>
<td>

| Tool | Purpose |
|------|---------|
| Version control | Tracks code changes |
| Docker | Reproducible environments |
| Unit tests | Validates specific outputs |
| Checksums | File integrity |

</td>
<td>

| Feature | Purpose |
|---------|---------|
| **Return time (τ_R)** | Measures temporal coherence |
| **Budget identity** | Conservation law validation |
| **Frozen contracts** | Immutable mathematical specs |
| **Seam testing** | Budget conservation |
| **Regime classification** | System health monitoring |
| **Uncertainty propagation** | Delta-method through invariants |
| **Human-verifiable checksums** | mod-97 triads |

</td>
</tr>
</table>

---

## 📊 System Overview

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════╗
║                    UMCP SYSTEM AT A GLANCE                        ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   📊 1002 Tests        🔌 37+ API Endpoints    📈 21 Dashboard    ║
║   📦 11 Casepacks      🔧 10 CLI Commands      🧮 46 Lemmas       ║
║   🔬 28 Closures       📜 5 Frameworks         🔒 SHA256 Verified ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   FRAMEWORKS:                                                     ║
║   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           ║
║   │ GCD (Tier-2) │  │ KIN (Tier-0) │  │ RCFT (Tier-2)│           ║
║   │ Energy/      │  │ Phase Space  │  │ Fractal/     │           ║
║   │ Collapse     │  │ Return       │  │ Recursive    │           ║
║   └──────────────┘  └──────────────┘  └──────────────┘           ║
║                                                                   ║
║   ┌──────────────┐  ┌──────────────┐                             ║
║   │ WEYL         │  │ Security     │                             ║
║   │ Cosmological │  │ Validation   │                             ║
║   │ Analysis     │  │ Framework    │                             ║
║   └──────────────┘  └──────────────┘                             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

</div>

### Kernel Invariants (The Seven Core Metrics)

| Symbol | Name | Definition | Range | Purpose |
|:------:|------|------------|:-----:|---------|
| **ω** | Drift | ω = 1 - F | [0,1] | Collapse proximity |
| **F** | Fidelity | F = Σ wᵢ·cᵢ | [0,1] | Weighted coherence |
| **S** | Entropy | S = -Σ wᵢ[cᵢ ln(cᵢ) + (1-cᵢ)ln(1-cᵢ)] | ≥0 | Disorder measure |
| **C** | Curvature | C = stddev(cᵢ)/0.5 | [0,1] | Instability proxy |
| **τ_R** | Return time | Re-entry delay to domain Dθ | ℕ∪{∞} | Recovery measure |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | ≤0 | Composite stability |
| **IC** | Integrity | IC = exp(κ) | (0,1] | System stability |

### Regime Classification

| Regime | Conditions | 🚦 |
|--------|-----------|:--:|
| **STABLE** | ω < 0.038, F > 0.90, S < 0.15, C < 0.14 | 🟢 |
| **WATCH** | 0.038 ≤ ω < 0.30 | 🟡 |
| **COLLAPSE** | ω ≥ 0.30 | 🔴 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UMCP WORKFLOW (v1.5.0)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │ INPUT   │───▶│ KERNEL      │───▶│ CLOSURES    │───▶│ OUTPUT   │ │
│  │ Ψ(t)    │    │ ω,F,S,C,τ_R │    │ Γ(ω),D_C    │    │ Receipts │ │
│  │ [0,1]ⁿ  │    │ κ, IC       │    │ Budget      │    │ Ledger   │ │
│  └─────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│       │                │                 │                 │        │
│       ▼                ▼                 ▼                 ▼        │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │ Tier-0  │    │ Tier-1      │    │ Tier-0 Seam │    │ SHA256   │ │
│  │ Protocol│    │ Invariants  │    │ |s| ≤ 0.005 │    │ Verified │ │
│  └─────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ COST CLOSURES (Frozen Contract v1.5.0)                      │   │
│  │ ────────────────────────────────────────────────────────── │   │
│  │ Γ(ω) = ω³/(1-ω+ε)           [Drift cost - cubic barrier]   │   │
│  │ D_C = α·C                    [Curvature cost]               │   │
│  │ Budget: R·τ_R = D_ω + D_C + Δκ  [Conservation law]          │   │
│  │ Seam: |s| ≤ tol_seam         [PASS condition]               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Canonical Constants

| Symbol | Name | Value | Purpose |
|:------:|------|:-----:|---------|
| ε | Guard band | 10⁻⁸ | Numerical stability |
| p | Power exponent | 3 | Γ(ω) cubic exponent |
| α | Curvature scale | 1.0 | D_C cost closure |
| λ | Damping | 0.2 | Reserved |
| tol_seam | Seam tolerance | 0.005 | Budget residual threshold |

---

## 🔧 CLI Commands

UMCP provides **10 built-in CLI commands**:

```bash
# Core validation
umcp validate [path]        # Validate artifacts, CasePacks, schemas
umcp health                 # System health check
umcp preflight              # Pre-validation checks

# Testing
umcp test                   # Run pytest (supports --coverage, -k, -m)
umcp casepack <name>        # Run specific casepack

# Discovery
umcp list <type>            # List casepacks|closures|contracts|schemas
umcp integrity <path>       # Verify SHA256 hashes

# Analysis
umcp diff file1 file2       # Compare validation receipts
umcp report [path]          # Generate audit reports
umcp run [path]             # Operational validation
```

<details>
<summary><strong>📋 Command Examples</strong></summary>

```bash
# Validate hello_world casepack
umcp validate casepacks/hello_world

# Run tests with coverage
umcp test --coverage

# List all casepacks
umcp list casepacks

# Check system health
umcp health

# Generate report
umcp report casepacks/gcd_complete
```

</details>

---

## 🌐 REST API

UMCP includes a production-ready REST API with **37+ endpoints**:

```bash
pip install -e ".[api]"     # Install API dependencies
umcp-api                    # Start server (port 8000)
```

### Endpoint Categories

| Category | Endpoints | Description |
|----------|:---------:|-------------|
| **System** | 3 | `/`, `/health`, `/version` |
| **Validation** | 1 | `/validate` |
| **Casepacks** | 3 | Browse and execute |
| **Ledger** | 2 | Query validation history |
| **Contracts** | 1 | List available contracts |
| **Closures** | 1 | List closure functions |
| **Analysis** | 4 | Statistics, correlation, timeseries |
| **Kernel** | 3 | Compute invariants, budget, uncertainty |
| **Conversion** | 2 | Unit conversion, embedding |
| **Output** | 10+ | SVG, Markdown, HTML, LaTeX, JUnit, JSON-LD |

<details>
<summary><strong>📋 API Examples</strong></summary>

```bash
# Health check (no auth)
curl http://localhost:8000/health

# List casepacks
curl -H "X-API-Key: umcp-dev-key" http://localhost:8000/casepacks

# Compute kernel
curl -X POST -H "X-API-Key: umcp-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"coordinates": [0.9, 0.85, 0.92], "weights": [0.5, 0.3, 0.2]}' \
  http://localhost:8000/kernel/compute
```

</details>

📖 **Interactive docs**: http://localhost:8000/docs (Swagger UI)

---

## 📈 Visualization Dashboard

UMCP includes an interactive Streamlit dashboard with **21 pages**:

```bash
pip install -e ".[viz]"     # Install visualization dependencies
umcp-dashboard              # Start dashboard (port 8501)
```

### Dashboard Pages

| Category | Pages | Description |
|----------|-------|-------------|
| **Core** | Overview, Geometry, Ledger, Casepacks, Contracts, Closures, Regime, Metrics, Health | System monitoring |
| **Interactive** | Live Runner, Batch Validation, Test Templates | Run validations |
| **Scientific** | Physics, Kinematics, Formula Builder, Cosmology | Domain-specific |
| **Analysis** | Time Series, Comparison | Data analysis |
| **Management** | Exports, Bookmarks, Notifications, API Integration | System management |

📖 **Dashboard URL**: http://localhost:8501

---

## 📦 Frameworks

### Framework Selection Guide

| Framework | Tier | Best For | Closures |
|-----------|:----:|----------|:--------:|
| **GCD** | 2 | Energy/collapse analysis, phase transitions | 5 |
| **Kinematics** | 0 | Physics-based motion, phase space return (diagnostic) | 6 |
| **RCFT** | 2 | Trajectory complexity, memory effects | 4 |
| **WEYL** | 2 | Cosmological analysis, modified gravity | 5 |
| **Security** | 2 | Validation security, input sanitization | 8 |

<details>
<summary><strong>🔬 GCD (Generative Collapse Dynamics)</strong></summary>

**Closures**: `energy_potential`, `entropic_collapse`, `generative_flux`, `field_resonance`, `boundary_detection`

```bash
umcp validate casepacks/gcd_complete
```

</details>

<details>
<summary><strong>⚙️ Kinematics (KIN)</strong></summary>

**Closures**: `linear_kinematics`, `rotational_kinematics`, `energy_mechanics`, `momentum_dynamics`, `phase_space_return`, `kinematic_stability`

```bash
umcp validate casepacks/kinematics_complete
umcp casepack kin_ref_phase_oscillator
```

</details>

<details>
<summary><strong>🌀 RCFT (Recursive Collapse Field Theory)</strong></summary>

**Closures**: All GCD + `fractal_dimension`, `recursive_field`, `resonance_pattern`

```bash
umcp validate casepacks/rcft_complete
```

</details>

<details>
<summary><strong>🌌 WEYL (Cosmological Framework)</strong></summary>

**Purpose**: Modified gravity analysis, DES Y3 data integration

```bash
umcp validate casepacks/weyl_des_y3
```

</details>

---

## 🧪 Testing

```bash
pytest                      # Run all 1002 tests
pytest -v                   # Verbose output
pytest --cov                # With coverage
pytest -k "gcd"             # Pattern matching
pytest -m "not slow"        # Skip slow tests
```

### Test Distribution

| Category | Tests | Description |
|----------|------:|-------------|
| Schema validation | 50 | JSON/YAML schema tests |
| Kernel invariants | 84 | Core metric tests |
| GCD framework | 92 | Energy/collapse tests |
| Kinematics | 133 | Motion analysis tests |
| RCFT framework | 78 | Fractal/recursive tests |
| WEYL framework | 43 | Cosmology tests |
| Extended Lemmas | 53 | Lemmas 35-46 tests |
| Frozen contract | 36 | Canonical constants |
| SS1m triads | 35 | Checksum tests |
| Uncertainty | 23 | Delta-method tests |
| API | 32 | REST endpoint tests |
| Dashboard | 30 | UI component tests |
| Security | 45 | Input validation tests |
| Integration | 150+ | End-to-end tests |

---

## 📚 Documentation

### 📖 Core References

| Document | Description |
|----------|-------------|
| [AXIOM.md](AXIOM.md) | Core axiom: "What returns is real" |
| [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) | Formal definitions (46 lemmas) |
| [MATHEMATICAL_ARCHITECTURE.md](MATHEMATICAL_ARCHITECTURE.md) | Complete mathematical framework |
| [TIER_SYSTEM.md](TIER_SYSTEM.md) | Tier-0/1/2 architecture (v3.0.0) |
| [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md) | Three-layer geometric architecture |

### 🔧 Developer Guides

| Document | Description |
|----------|-------------|
| [QUICKSTART_TUTORIAL.md](QUICKSTART_TUTORIAL.md) | 10-minute hands-on tutorial |
| [docs/quickstart.md](docs/quickstart.md) | Getting started guide |
| [docs/python_coding_key.md](docs/python_coding_key.md) | Development standards |
| [docs/production_deployment.md](docs/production_deployment.md) | Enterprise deployment |

### 📐 Framework Documentation

| Document | Description |
|----------|-------------|
| [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) | GCD specification |
| [canon/kin_anchors.yaml](canon/kin_anchors.yaml) | Kinematics specification |
| [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) | RCFT specification |
| [KINEMATICS_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md) | Kinematics layer docs |

### 📋 Reference

| Document | Description |
|----------|-------------|
| [GLOSSARY.md](GLOSSARY.md) | Authoritative term definitions |
| [SYMBOL_INDEX.md](SYMBOL_INDEX.md) | Symbol table |
| [CASEPACK_REFERENCE.md](CASEPACK_REFERENCE.md) | CasePack structure |
| [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md) | Extension system |

---

## 📂 Repository Map

```
UMCP-Metadata-Runnable-Code/
│
├── 📁 src/umcp/                   # Core Python implementation
│   ├── frozen_contract.py         # Canonical constants & closures
│   ├── validator.py               # Core validation engine
│   ├── cli.py                     # CLI (10 commands)
│   ├── api_umcp.py                # REST API (37+ endpoints)
│   ├── dashboard.py               # Streamlit (21 pages)
│   ├── uncertainty.py             # Delta-method propagation
│   ├── ss1m_triad.py              # Mod-97 checksums
│   └── umcp_extensions.py         # Extension registry
│
├── 📁 tests/                      # Test suite (1002 tests)
│   ├── test_frozen_contract.py    # Frozen contract tests
│   ├── test_extended_lemmas.py    # Lemmas 35-46 tests
│   ├── test_api_umcp.py           # API tests
│   ├── closures/                  # Closure-specific tests
│   └── ...
│
├── 📁 casepacks/                  # Reproducible examples (11)
│   ├── hello_world/               # Zero entropy baseline
│   ├── gcd_complete/              # GCD validation
│   ├── kinematics_complete/       # Kinematics validation
│   ├── rcft_complete/             # RCFT validation
│   ├── weyl_des_y3/               # WEYL cosmology
│   ├── security_validation/       # Security framework
│   └── ...
│
├── 📁 closures/                   # Computational functions (28+)
│   ├── gcd/                       # GCD closures
│   ├── kinematics/                # Kinematics closures
│   ├── rcft/                      # RCFT closures
│   ├── weyl/                      # WEYL closures
│   ├── security/                  # Security closures
│   └── registry.yaml              # Closure registry
│
├── 📁 contracts/                  # Frozen mathematical contracts
│   ├── UMA.INTSTACK.v1.yaml       # Primary contract
│   ├── GCD.INTSTACK.v1.yaml       # GCD framework
│   └── RCFT.INTSTACK.v1.yaml      # RCFT framework
│
├── 📁 canon/                      # Canonical anchors
│   ├── gcd_anchors.yaml           # GCD specification
│   ├── kin_anchors.yaml           # Kinematics specification
│   └── rcft_anchors.yaml          # RCFT specification
│
├── 📁 schemas/                    # JSON schemas (12+)
├── 📁 ledger/                     # Validation log
├── 📁 integrity/                  # SHA256 checksums
├── 📁 docs/                       # Documentation
├── 📁 data/                       # Physics observations
│   └── physics_observations_complete.csv  # 38 observations
│
└── 📄 pyproject.toml              # Project configuration
```

---

## 🎓 Getting Started Tutorial

### Step 1: Understand the Core Concept

UMCP validates computational experiments as **auditable artifacts**. Every claim must have:

1. ✅ **Declared inputs** (raw measurements)
2. ✅ **Frozen rules** (mathematical contracts)  
3. ✅ **Computed outputs** (invariants, closures)
4. ✅ **Cryptographic receipts** (SHA256 verification)

### Step 2: Run Your First Validation

```bash
# Validate the hello_world casepack
umcp validate casepacks/hello_world

# Expected output:
# ✓ CONFORMANT
# Errors: 0, Warnings: 0
```

### Step 3: Explore the Python API

```python
import umcp
from umcp.frozen_contract import compute_kernel, classify_regime
import numpy as np

# Validate a casepack
result = umcp.validate("casepacks/hello_world")
print(f"Status: {'CONFORMANT' if result else 'NONCONFORMANT'}")

# Compute kernel invariants
c = np.array([0.9, 0.85, 0.92])  # Coherence values
w = np.array([0.5, 0.3, 0.2])    # Weights
kernel = compute_kernel(c, w, tau_R=5.0)

print(f"Drift (ω): {kernel.omega:.4f}")
print(f"Fidelity (F): {kernel.F:.4f}")
print(f"Integrity (IC): {kernel.IC:.4f}")

# Classify regime
regime = classify_regime(
    omega=kernel.omega,
    F=kernel.F,
    S=kernel.S,
    C=kernel.C,
    integrity=kernel.IC
)
print(f"Regime: {regime.name}")  # STABLE, WATCH, or COLLAPSE
```

### Step 4: Create Your Own CasePack

```bash
# Copy the hello_world template
cp -r casepacks/hello_world casepacks/my_experiment

# Edit the manifest
nano casepacks/my_experiment/manifest.yaml

# Validate your casepack
umcp validate casepacks/my_experiment
```

### Step 5: Explore Extensions

```bash
# Start the REST API
pip install -e ".[api]"
umcp-api
# Open http://localhost:8000/docs

# Start the Dashboard
pip install -e ".[viz]"
umcp-dashboard
# Open http://localhost:8501
```

### Step 6: Run the Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=umcp --cov-report=html

# Run specific framework tests
pytest -k "gcd"
pytest -k "kinematics"
pytest -k "rcft"
```

### Step 7: Understand the Mathematics

Read these documents in order:
1. [AXIOM.md](AXIOM.md) - The foundational principle
2. [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - 46 formal lemmas
3. [MATHEMATICAL_ARCHITECTURE.md](MATHEMATICAL_ARCHITECTURE.md) - Complete framework

---

## 🤝 Contributing

We welcome contributions! Please read our comprehensive [CONTRIBUTING.md](CONTRIBUTING.md) guide.

### Quick Contribution Workflow

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Create feature branch
git checkout -b feat/your-feature

# 4. Make changes and test
pytest
ruff check .
ruff format .
mypy src/umcp

# 5. Commit and push
git commit -m "feat: your feature description"
git push origin feat/your-feature

# 6. Open Pull Request
```

### Contribution Areas

| Area | Description | Difficulty |
|------|-------------|:----------:|
| 📖 Documentation | Improve docs, fix typos | 🟢 Easy |
| 🧪 Tests | Add test coverage | 🟢 Easy |
| 🐛 Bug fixes | Fix reported issues | 🟡 Medium |
| ✨ Features | New closures, endpoints | 🟡 Medium |
| 🔬 Research | New frameworks, lemmas | 🔴 Hard |

### Code Quality Standards

- ✅ All tests must pass (`pytest`)
- ✅ 80%+ code coverage
- ✅ Zero ruff errors (`ruff check .`)
- ✅ Zero type errors (`mypy src/umcp`)
- ✅ Proper formatting (`ruff format .`)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Clement Paulus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📞 Support & Resources

<table>
<tr>
<td width="50%">

### 🔗 Links

- 📖 [Documentation](docs/)
- 📦 [Examples](casepacks/)
- 🐛 [Issues](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/issues)
- 📋 [Changelog](CHANGELOG.md)

</td>
<td width="50%">

### 📚 Key Files

- [AXIOM.md](AXIOM.md) - Core principle
- [GLOSSARY.md](GLOSSARY.md) - Term definitions
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - 46 lemmas
- [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md) - Release info

</td>
</tr>
</table>

---

<div align="center">

## 🏆 System Status

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    UMCP PRODUCTION SYSTEM STATUS                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   🎯 Core Axiom:    "What Returns Through Collapse Is Real"           ║
║   📜 Contract:      UMA.INTSTACK.v1 + Frozen Contract v1.5.0          ║
║   🔐 Canon:         UMCP.CANON.v1                                     ║
║                                                                       ║
║   ⚙️  Frozen:       ε=10⁻⁸  p=3  α=1.0  λ=0.2  tol=0.005              ║
║                                                                       ║
║   📊 Status:        CONFORMANT ✅                                     ║
║   🧪 Tests:         1002 passing                                      ║
║   📦 Casepacks:     11 validated                                      ║
║   🔧 CLI:           10 commands                                       ║
║   🌐 API:           37+ endpoints                                     ║
║   📈 Dashboard:     21 pages                                          ║
║   🧮 Lemmas:        46 formal proofs                                  ║
║   🔬 Closures:      28+ functions                                     ║
║   🔒 Integrity:     SHA256 verified                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

**Built with ❤️ for reproducible science**

*"What Returns Through Collapse Is Real"*

</div>
