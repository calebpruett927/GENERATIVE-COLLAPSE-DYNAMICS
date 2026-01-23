# UMCP: Universal Measurement Contract Protocol

[![CI](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/actions/workflows/validate.yml/badge.svg)](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/actions/workflows/validate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 344 passing](https://img.shields.io/badge/tests-344%20passing-brightgreen.svg)](tests/)
[![Version: 1.4.0](https://img.shields.io/badge/version-1.4.0-blue.svg)](CHANGELOG.md)

**UMCP transforms computational experiments into auditable artifacts** based on a foundational principle:

> **Core Axiom**: *"What Returns Through Collapse Is Real"*
>
> Reality is defined by what persists through collapse-reconstruction cycles. Only measurements that return—that survive transformation and can be reproduced—receive credit as real, valid observations.

```yaml
# Encoded in every UMCP contract
typed_censoring:
  no_return_no_credit: true
```

UMCP is a **production-grade system** for creating, validating, and sharing reproducible computational workflows. It enforces mathematical contracts, tracks provenance, generates cryptographic receipts, and validates results against frozen specifications.

---

## 📊 Quick Start (5 Minutes)

### Prerequisites

- **Python 3.11+** (3.12+ recommended)
- **pip** (Python package installer)
- **git** (version control)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install production dependencies (includes numpy, scipy, pyyaml, jsonschema)
pip install -e ".[production]"
```

**Optional installations:**

```bash
# Install test dependencies (adds pytest, coverage tools)
pip install -e ".[test]"

# Install extension dependencies (adds streamlit, fastapi, uvicorn)
pip install -e ".[extensions]"

# Install everything (production + test + extensions)
pip install -e ".[test,extensions]"
```

### Verify Installation

```bash
# System health check (should show HEALTHY status)
umcp health

# Run test suite (should show 344 tests passing)
pytest

# Quick validation test
umcp validate casepacks/hello_world

# Check installed version
python -c "import umcp; print(f'UMCP v{umcp.__version__}')"
```

**Expected output:**
```
Status: HEALTHY
Schemas: 11
344 passed in ~13s
```

### Launch Interactive Tools

```bash
# Visualization dashboard (port 8501)
umcp-visualize

# REST API server (port 8000)
umcp-api

# List extensions
umcp-ext list
```

---

## 🎯 What is UMCP?

UMCP is a **measurement discipline for computational claims**. It requires that every serious claim be published as a reproducible record (a **row**) with:

- ✅ **Declared inputs** (raw measurements)
- ✅ **Frozen rules** (mathematical contracts)
- ✅ **Computed outputs** (invariants, closures)
- ✅ **Cryptographic receipts** (SHA256 verification)

### Operational Terms

**Core Invariants** (Tier-1: GCD Framework):

| Symbol | Name | Definition | Range | Purpose |
|--------|------|------------|-------|---------|
| **ω** | Drift | ω = 1 - F | [0,1] | Collapse proximity |
| **F** | Fidelity | F = Σ wᵢ·cᵢ | [0,1] | Weighted coherence |
| **S** | Entropy | S = -Σ wᵢ[cᵢ ln(cᵢ) + (1-cᵢ)ln(1-cᵢ)] | ≥0 | Disorder measure |
| **C** | Curvature | C = stddev(cᵢ)/0.5 | [0,1] | Instability proxy |
| **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | ≤0 | Composite stability |
| **IC** | Integrity | IC = exp(κ) | (0,1] | System stability |
| **τ_R** | Return time | Re-entry delay to domain Dθ | ℕ∪{∞} | Recovery measure |

**Extended Metrics** (Tier-2: RCFT Framework):

| Symbol | Name | Range | Purpose |
|--------|------|-------|---------|
| **Dꜰ** | Fractal dimension | [1,3] | Trajectory complexity |
| **Ψᵣ** | Recursive field | ≥0 | Self-referential strength |
| **B** | Basin strength | [0,1] | Attractor robustness |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UMCP WORKFLOW                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INPUT                                                   │
│     └─ raw_measurements.csv  (experimental data)            │
│                                                             │
│  2. INVARIANTS COMPUTATION                                  │
│     ├─ ω (drift)         ├─ F (fidelity)                    │
│     ├─ S (entropy)       └─ C (curvature)                   │
│                                                             │
│  3. FRAMEWORK SELECTION                                     │
│     ┌─────────────────┐      ┌──────────────────┐          │
│     │ GCD (Tier-1)    │  OR  │ RCFT (Tier-2)    │          │
│     ├─────────────────┤      ├──────────────────┤          │
│     │ • Energy (E)    │      │ • Fractal (Dꜰ)   │          │
│     │ • Collapse (Φ)  │      │ • Recursive (Ψᵣ) │          │
│     │ • Flux (Φ_gen)  │      │ • Pattern (λ, Θ) │          │
│     │ • Resonance (R) │      │ + all GCD        │          │
│     └─────────────────┘      └──────────────────┘          │
│                                                             │
│  4. VALIDATION                                              │
│     ├─ Contract conformance (schema validation)             │
│     ├─ Regime classification (Stable/Collapse/Watch)        │
│     ├─ Mathematical identities (F = 1-ω, IC ≈ exp(κ))       │
│     └─ Tolerance checks (within tol_seam, tol_id)           │
│                                                             │
│  5. OUTPUT                                                  │
│     ├─ invariants.json (computed metrics)                   │
│     ├─ closure_results.json (GCD/RCFT outputs)              │
│     ├─ seam_receipt.json (validation status + SHA256)       │
│     └─ CONFORMANT or NONCONFORMANT status                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Framework Selection Guide

### GCD (Generative Collapse Dynamics) - Tier-1

**Best for**: Energy/collapse analysis, phase transitions, basic regime classification

**Closures** (4):
- `energy_potential`: Total system energy
- `entropic_collapse`: Collapse potential
- `generative_flux`: Generative flux
- `field_resonance`: Boundary-interior resonance

**Example**:
```bash
umcp validate casepacks/gcd_complete
```

### RCFT (Recursive Collapse Field Theory) - Tier-2

**Best for**: Trajectory complexity, memory effects, oscillatory patterns, multi-scale analysis

**Closures** (7 = 4 GCD + 3 RCFT):
- All GCD closures +
- `fractal_dimension`: Trajectory complexity (Dꜰ ∈ [1,3])
- `recursive_field`: Collapse memory (Ψᵣ ≥ 0)
- `resonance_pattern`: Oscillation detection (λ, Θ)

**Example**:
```bash
umcp validate casepacks/rcft_complete
```

### Decision Matrix

| Need | Framework | Why |
|------|-----------|-----|
| Basic energy/collapse | GCD | Simpler, faster, foundational |
| Trajectory complexity | RCFT | Box-counting fractal dimension |
| History/memory | RCFT | Exponential decay field |
| Oscillation detection | RCFT | FFT-based pattern analysis |
| Maximum insight | RCFT | All GCD metrics + 3 new |

---

## 🔌 Extension System

UMCP features **auto-discovery extensions** with 4 built-in plugins:

### 1. Visualization Dashboard
```bash
umcp-visualize
# Opens http://localhost:8501
# Features: Phase space plots, time series, regime tracking
```

### 2. REST API
```bash
umcp-api
# Opens http://localhost:8000
# Endpoints: /health, /latest-receipt, /ledger, /stats, /regime
```

### 3. Continuous Ledger
```bash
# Auto-logs every validation run
cat ledger/return_log.csv
```

### 4. Contract Auto-Formatter
```bash
umcp-format --all
```

📖 **See**: [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md) | [QUICKSTART_EXTENSIONS.md](QUICKSTART_EXTENSIONS.md)

---

## 📚 Documentation

### Core Protocol
- **[AXIOM.md](AXIOM.md)** — Core axiom: "What returns is real"
- **[TIER_SYSTEM.md](TIER_SYSTEM.md)** — Tier-0/1/1.5/2 boundaries, freeze gates
- **[RETURN_BASED_CANONIZATION.md](RETURN_BASED_CANONIZATION.md)** — How Tier-2 results become Tier-1 canon
- **[KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md)** — Formal definitions (19 lemmas)
- **[PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md)** — Publication standards
- **[CASEPACK_REFERENCE.md](CASEPACK_REFERENCE.md)** — CasePack structure

### Indexing & Reference
- **[GLOSSARY.md](GLOSSARY.md)** — Authoritative term definitions
- **[SYMBOL_INDEX.md](SYMBOL_INDEX.md)** — Symbol table (collision prevention)
- **[TERM_INDEX.md](TERM_INDEX.md)** — Alphabetical cross-reference

### Framework Documentation
- **[GCD Theory](canon/gcd_anchors.yaml)** — Tier-1 specification
- **[RCFT Theory](docs/rcft_theory.md)** — Tier-2 mathematical foundations
- **[RCFT Usage](docs/rcft_usage.md)** — Practical examples

### Governance
- **[UHMP.md](UHMP.md)** — Universal Hash Manifest Protocol
- **[FACE_POLICY.md](FACE_POLICY.md)** — Boundary governance
- **[PROTOCOL_REFERENCE.md](PROTOCOL_REFERENCE.md)** — Master navigation

### Developer Guides
- **[Quickstart](docs/quickstart.md)** — Get started in 10 minutes
- **[Python Standards](docs/python_coding_key.md)** — Development guidelines
- **[Production Deployment](docs/production_deployment.md)** — Enterprise setup
- **[PyPI Publishing](docs/pypi_publishing_guide.md)** — Release workflow

---

## 📂 Repository Structure

```
UMCP-Metadata-Runnable-Code/
├── src/umcp/              # All Python code (API, CLI, extensions)
├── tests/                 # Test suite (344 tests)
├── scripts/               # Utility scripts
├── contracts/             # Frozen contracts (GCD, RCFT)
├── closures/              # Computational functions (7 closures)
│   ├── gcd/              # 4 GCD closures
│   └── rcft/             # 3 RCFT closures
├── casepacks/             # Reproducible examples
│   ├── hello_world/      # Zero entropy example
│   ├── gcd_complete/     # GCD validation
│   └── rcft_complete/    # RCFT validation
├── schemas/               # JSON schemas
├── canon/                 # Canonical anchors
├── ledger/                # Validation log (continuous append)
├── integrity/             # Integrity metadata (SHA256)
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration (v1.4.0)
```

---

## 🧪 Testing

```bash
# All tests (344 total)
pytest

# Verbose output
pytest -v

# Specific framework
pytest -k "gcd"    # GCD tests
pytest -k "rcft"   # RCFT tests

# Coverage report
pytest --cov

# Fast subset
pytest tests/test_00_schemas_valid.py
```

**Test Structure**: 344 tests = 142 original + 56 RCFT + 146 integration/coverage

---

## 🚀 Production Features

- ✅ **344 tests** passing (100% success rate)
- ✅ **Health checks**: `umcp health` for system monitoring
- ✅ **Structured logging**: JSON output for ELK/Splunk/CloudWatch
- ✅ **Performance metrics**: Duration, memory, CPU tracking
- ✅ **Container ready**: Docker + Kubernetes support
- ✅ **Cryptographic receipts**: SHA256 verification
- ✅ **Zero technical debt**: No TODO/FIXME/HACK markers
- ✅ **<5s validation**: Fast for typical repositories

📖 **See**: [Production Deployment Guide](docs/production_deployment.md)

---

## 🔒 Integrity & Automation

```bash
# Verify file integrity
sha256sum -c integrity/sha256.txt

# Update after changes
python scripts/update_integrity.py

# Check merge status
./scripts/check_merge_status.sh
```

**Automated**:
- ✅ 344 tests on every commit (CI/CD)
- ✅ Code formatting (ruff, black)
- ✅ Type checking (mypy)
- ✅ SHA256 tracking (12 files)

---

## 📊 What's New in v1.4.0

**Complete Protocol Infrastructure**:
- ✅ 8 major protocol documents (~5,500 lines)
- ✅ Formal specification (19 lemmas, kernel definitions)
- ✅ Publication standards (CasePack structure)
- ✅ 344 tests passing (GCD + RCFT frameworks)
- ✅ Production ready (manuscript-aligned)

📖 **See**: [CHANGELOG.md](CHANGELOG.md) | [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Validate code quality (`ruff check`, `mypy`)
6. Commit changes (`git commit -m 'feat: Description'`)
7. Push to branch (`git push origin feature/name`)
8. Open Pull Request

📖 **See**: [Python Coding Standards](docs/python_coding_key.md) | [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📞 Support & Resources

- **Issues**: [GitHub Issues](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [casepacks/](casepacks/)
- **Immutable Release**: [IMMUTABLE_RELEASE.md](IMMUTABLE_RELEASE.md)

---

## 🏆 System Status

```
╔═══════════════════════════════════════════════════════════╗
║           UMCP PRODUCTION SYSTEM STATUS                   ║
╚═══════════════════════════════════════════════════════════╝

  🎯 Core Axiom:   "What Returns Through Collapse Is Real"
  🔐 Canon:        UMCP.CANON.v1
  📜 Contract:     UMA.INTSTACK.v1
  📚 DOI:          10.5281/zenodo.17756705 (PRE)
                   10.5281/zenodo.18072852 (POST)
                   10.5281/zenodo.18226878 (PACK)
  
  ⚙️  Tier-1:      p=3  α=1.0  λ=0.2  η=0.001
  🎯 Regimes:      Stable: ω<0.038  F>0.90
                   Collapse: ω≥0.30
  
  📊 Status:       CONFORMANT ✅
  🧪 Tests:        344 passing
  📦 Casepacks:    3 validated
  🔒 Integrity:    12 files checksummed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     "No improvisation. Contract-first. Tier-1 reserved."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🎓 Citation

**Framework**: UMCP (Universal Measurement Contract Protocol)  
**Author**: Clement Paulus  
**Version**: 1.4.0  
**Release**: January 23, 2026  
**Tests**: 344 passing  
**Integrity**: SHA256 verified  

**Frameworks**:
- **Tier-1**: GCD (Generative Collapse Dynamics) - 4 closures
- **Tier-2**: RCFT (Recursive Collapse Field Theory) - 7 closures

---

**Built with ❤️ for reproducible science**  
*"What Returns Through Collapse Is Real"*
