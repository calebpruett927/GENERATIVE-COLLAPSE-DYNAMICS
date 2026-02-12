<p align="center">
  <img src="https://img.shields.io/badge/UMCP-Universal%20Measurement%20Contract%20Protocol-6C63FF?style=for-the-badge" alt="UMCP">
</p>

<h1 align="center">Universal Measurement Contract Protocol</h1>

<p align="center">
  <strong>Contract-first validation framework for reproducible computational workflows with formal mathematical foundations</strong>
</p>

<p align="center">
  <a href="https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/actions/workflows/validate.yml"><img src="https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/actions/workflows/validate.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-2476%20passing-brightgreen?logo=pytest" alt="Tests: 2476 passing"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-2.0.0-blue" alt="Version: 2.0.0"></a>
  <a href="src/umcp/api_umcp.py"><img src="https://img.shields.io/badge/API-57%20endpoints-orange?logo=fastapi" alt="API: 57 endpoints"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#core-axiom">Core Axiom</a> &bull;
  <a href="#system-overview">Overview</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#cli-reference">CLI</a> &bull;
  <a href="#rest-api">API</a> &bull;
  <a href="#dashboard">Dashboard</a> &bull;
  <a href="#python-api">Python API</a> &bull;
  <a href="#testing">Testing</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

> **Core Axiom**: *"What Returns Through Collapse Is Real"*
>
> Within-run: frozen causes only (no back-edges). Between-run: continuity only by return-weld
> (tau\_R finite + seam residual within tolerance). Reality is declared by showing closure
> after collapse: each claim is welded to a seam. Frozen parameters (epsilon, p, alpha, lambda, tol\_seam)
> are not arbitrary constants -- they are **consistent across the seam**.

---

## Table of Contents

* [Quick Start](#quick-start)
* [Core Axiom](#core-axiom)
* [Principles to Translation](#principles-to-translation)
* [Getting the Most Out of UMCP](#getting-the-most-out-of-umcp)
* [Translation Workflow](#translation-workflow)
* [System Overview](#system-overview)
* [Architecture](#architecture)
* [Installation](#installation)
* [CLI Reference](#cli-reference)
* [REST API](#rest-api)
* [Dashboard](#dashboard)
* [Python API](#python-api)
* [Frameworks & Domains](#frameworks--domains)
* [Casepacks](#casepacks)
* [Testing](#testing)
* [Pre-Commit Protocol](#pre-commit-protocol)
* [Extension System](#extension-system)
* [Documentation Map](#documentation-map)
* [Repository Structure](#repository-structure)
* [Contributing](#contributing)
* [License](#license)
---

## Principles to Translation

UMCP is built on the principle that **mathematical contracts and epistemic validation** are the foundation for reproducible science. The core axiom, "What Returns Through Collapse Is Real," governs every workflow:

- **Collapse**: Every claim, measurement, or computation must be validated against a contract, with frozen parameters and no retroactive tuning.
- **Return**: Only what survives the seam test (return-weld) is declared real. Continuity is earned, not assumed.
- **Translation**: To translate principles into new domains, map the contract identities, kernel invariants, and epistemic verdicts to the domain's semantics.
- **Epistemic Credit**: Reality is not a boolean; it is earned by closing the seam and passing all identity checks.
- **Deep Implications**: Use the insights engine to discover patterns, regime boundaries, and cross-domain correlations, then translate these into actionable domain knowledge.

---

## Getting the Most Out of UMCP

UMCP is designed for both newcomers and advanced users. To maximize its value:

1. **Start with the Core Axiom**: Understand that validation is contract-first, and epistemic credit is earned by return.
2. **Explore Casepacks**: Run `umcp validate casepacks/hello_world` to see a zero-entropy baseline. Use other casepacks for domain-specific workflows.
3. **Study Kernel Invariants**: Learn the meaning of omega, F, S, C, tau_R, kappa, IC. These are universal metrics for collapse and return.
4. **Use the Insights Engine**: Run deep pattern discovery (`src/umcp/insights.py`) to extract lessons-learned, regime boundaries, and universality signatures. See [tests/test_insights_deep.py](tests/test_insights_deep.py) for advanced coverage.
5. **Translate Principles**: When approaching a new domain, map the kernel identities and epistemic verdicts to the domain's language. Use canonical anchors and contracts as your reference points.
6. **Leverage Extensions**: Use the API, dashboard, and extension system to integrate UMCP into your workflow. See [docs/EXTENSION_INTEGRATION.md](docs/EXTENSION_INTEGRATION.md).
7. **Validate Everything**: Always run `umcp validate .` and the pre-commit protocol before pushing changes. Integrity and reproducibility are paramount.
8. **Contribute and Expand**: Add new closures, contracts, and casepacks by following the extension and contribution guides. Every new domain expands the reach of UMCP.

---

## Translation Workflow

To translate UMCP principles into any domain or workflow:

1. **Identify the Contract**: Choose or create a mathematical contract (YAML in `contracts/`) that defines the invariants and rules for your domain.
2. **Design Closures**: Implement closure functions in `closures/<domain>/` that compute domain-specific metrics, always referencing kernel invariants.
3. **Map Canonical Anchors**: Use `canon/<domain>_anchors.yaml` to define reference values and anchor points for your domain.
4. **Validate with Casepacks**: Create a casepack (`casepacks/<your_casepack>/`) with manifest, raw data, and expected outputs. Run `umcp validate <casepack>`.
5. **Epistemic Validation**: Use the seam test and epistemic weld to ensure your results earn credit. Only what returns through collapse is real.
6. **Discover Deep Insights**: Use the insights engine to find periodic trends, regime boundaries, cross-correlations, and universality signatures. Translate these into domain knowledge and actionable lessons.
7. **Document and Share**: Add your findings to the lessons-learned database, update canonical anchors, and contribute new contracts or closures.

**Key Files for Translation:**
- [src/umcp/validator.py](src/umcp/validator.py): Identity checks, regime classification
- [src/umcp/insights.py](src/umcp/insights.py): Pattern discovery, deep implications
- [tests/test_insights_deep.py](tests/test_insights_deep.py): Advanced coverage and translation examples
- [contracts/](contracts/): Mathematical contracts
- [canon/](canon/): Domain anchors
- [closures/](closures/): Domain-specific computation
- [casepacks/](casepacks/): Reproducible experiments
- [docs/EXTENSION_INTEGRATION.md](docs/EXTENSION_INTEGRATION.md): Extension and translation guide

---

---

## Quick Start

```bash
# Clone
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install everything
pip install -e ".[all]"

# Verify
umcp health                  # System health check
umcp validate .              # Validate entire repository (must be CONFORMANT)
pytest                       # Run 2,476 tests
```

One-liner:

```bash
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git && cd UMCP-Metadata-Runnable-Code && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]" && umcp health
```

---

## Core Axiom

UMCP is built on a single axiom:

> **"What Returns Through Collapse Is Real"**

| Context | Rule |
|---------|------|
| **Within-run** | Frozen causes only -- no back-edges, no retroactive tuning |
| **Between-run** | Continuity only by return-weld -- new runs are canon-continuous only if the seam returns and closes |
| **No return = no credit** | tau\_R = INF\_REC means zero budget; you cannot synthesize continuity from structure alone |
| **Frozen contract** | epsilon=1e-8, p=3, alpha=1.0, lambda=0.2, tol\_seam=0.005 are consistent across the seam |

### Three-Valued Verdict

Every validation produces one of three outcomes -- never a boolean:

| Verdict | Exit Code | Meaning |
|---------|:---------:|---------|
| **CONFORMANT** | 0 | All identities, schemas, and integrity checks pass |
| **NONCONFORMANT** | 1 | At least one check failed |
| **NON\_EVALUABLE** | 1 | Cannot determine (missing data, schema error) |

### Tier System

| Tier | Name | Scope | Mutability |
|------|------|-------|------------|
| **Tier-1** | Kernel Invariants | F + omega = 1, IC <= F (AM-GM), IC approx exp(kappa) | Immutable -- mathematical identities |
| **Tier-0** | Protocol | Regime gates, seam accounting, identity verification | Immutable -- operational rules |
| **Tier-2** | Domain Expansion | Closures, frameworks, diagnostics | Extensible -- community-contributed |

---

## System Overview

| Metric | Value |
|--------|-------|
| **Tests** | 2,476 passing (80 files) |
| **API Endpoints** | 57 (25 GET, 32 POST) |
| **Dashboard Pages** | 31 (refactored into modular package) |
| **CLI Commands** | 11 subcommands, 6 entry points |
| **Casepacks** | 13 validated |
| **Closures** | 100+ Python files, 57 registered |
| **Contracts** | 15 domain contracts |
| **Schemas** | 12 JSON Schema Draft 2020-12 |
| **Domains** | 12 (GCD, KIN, RCFT, WEYL, Security, Astronomy, Nuclear, QM, Finance, Atomic, Materials, **Standard Model**) |
| **Lemmas** | 46 formal proofs |
| **SM Theorems** | 10 proven (74/2476 tests, duality exact) |
| **Atomic Elements** | 118 (all pass Tier-1, 10,162 identity tests) |
| **Subatomic Particles** | 31 (17 fundamental + 14 composite, all pass Tier-1) |
| **Canonical Anchors** | 11 domain anchor files |
| **Source Modules** | 43 Python files (includes fleet/, dashboard/ packages) |
| **Extensions** | 5 built-in |
| **Integrity** | SHA256 verified |
| **Status** | CONFORMANT |

### Kernel Invariants (Seven Core Metrics)

| Symbol | Name | Definition | Range | Purpose |
|:------:|------|------------|:-----:|---------|
| **omega** | Drift | omega = 1 - F | [0,1] | Collapse proximity |
| **F** | Fidelity | F = sum(w\_i * c\_i) | [0,1] | Weighted coherence |
| **S** | Entropy | S = -sum(w\_i [c\_i ln(c\_i) + (1-c\_i) ln(1-c\_i)]) | >= 0 | Disorder measure |
| **C** | Curvature | C = stddev(c\_i) / 0.5 | [0,1] | Instability proxy |
| **tau\_R** | Return time | Re-entry delay to domain | N union {inf} | Recovery measure |
| **kappa** | Log-integrity | kappa = sum(w\_i ln(c\_i + epsilon)) | <= 0 | Composite stability |
| **IC** | Integrity | IC = exp(kappa) | (0,1] | System stability |

### Regime Classification

| Regime | Conditions | Signal |
|--------|-----------|:------:|
| **STABLE** | omega < 0.038, F > 0.90, S < 0.15, C < 0.14 | Green |
| **WATCH** | 0.038 <= omega < 0.30 | Yellow |
| **COLLAPSE** | omega >= 0.30 | Red |

### Canonical Constants

| Symbol | Name | Value | Purpose |
|:------:|------|:-----:|---------|
| epsilon | Guard band | 1e-8 | Numerical stability |
| p | Power exponent | 3 | Gamma(omega) cubic exponent |
| alpha | Curvature scale | 1.0 | D\_C cost closure |
| lambda | Damping | 0.2 | Reserved |
| tol\_seam | Seam tolerance | 0.005 | Budget residual threshold |

### Epistemic Framework

Every emission is classified into exactly one of three **epistemic verdicts**:

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| **RETURN** | Seam closes, tau\_R finite, identity holds | What came back is consistent with what collapsed -- epistemic credit earned |
| **GESTURE** | Emission exists but seam did not close | Internally consistent, but did not complete the collapse-return cycle -- no credit |
| **DISSOLUTION** | omega >= 0.30 (COLLAPSE regime) | Epistemic trace degraded past viable return -- not failure, but the boundary that makes return meaningful |

The **positional illusion** is the belief that observation is free. Theorem T9 proves each observation costs Gamma(omega) in seam budget -- there is no external vantage point. This is quantified by `epistemic_weld.py`.

**Reference**: "The Seam of Reality" (Paulus, 2025; DOI: 10.5281/zenodo.17619502)

---

## Architecture

```
INPUT (raw measurements)
  |
  v
KERNEL COMPUTATION (omega, F, S, C, tau_R, kappa, IC)
  |
  v
TIER-1 IDENTITY CHECKS
  |  F + omega = 1
  |  IC <= F (AM-GM bound)
  |  IC approx exp(kappa)
  |
  v
COST CLOSURES (frozen contract)
  |  Gamma(omega) = omega^3 / (1 - omega + epsilon)
  |  D_C = alpha * C
  |  Budget: R * tau_R = D_omega + D_C + Delta_kappa
  |
  v
SEAM TEST: |seam_residual| <= tol_seam (0.005)
  |
  v
REGIME CLASSIFICATION
  |  STABLE / WATCH / COLLAPSE
  |
  v
SHA256 INTEGRITY CHECK
  |
  v
VERDICT: CONFORMANT / NONCONFORMANT / NON_EVALUABLE
  |
  v
OUTPUT: JSON receipt + ledger/return_log.csv append
```

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| pip | Latest | Package management |
| git | Any | Version control |

### Install Options

```bash
# Core only -- validation engine, CLI, schemas
pip install -e "."

# + REST API (FastAPI + Uvicorn)
pip install -e ".[api]"

# + Visualization dashboard (Streamlit + Plotly + Pandas)
pip install -e ".[viz]"

# + Development tools (pytest, ruff, mypy, pre-commit)
pip install -e ".[dev]"

# + Production monitoring (psutil)
pip install -e ".[production]"

# + All communication extensions (API + dashboard)
pip install -e ".[communications]"

# Everything (dev + production + communications)
pip install -e ".[all]"
```

### Core Dependencies

These are installed with every install option:

| Package | Version | Purpose |
|---------|---------|---------|
| pyyaml | >= 6.0.1 | YAML parsing (contracts, closures, canon) |
| jsonschema | >= 4.23.0 | JSON Schema Draft 2020-12 validation |
| numpy | >= 1.24.0 | Kernel computation, array operations |
| scipy | >= 1.10.0 | Statistical tests, optimization |

### Verify Installation

```bash
# Check all 6 entry points
umcp health                  # Main CLI health check
umcp-ext list                # Extension registry
umcp-calc --help             # Universal calculator
umcp-finance --help          # Finance CLI
umcp-api --help              # REST API server (requires [api])
umcp-dashboard --help        # Streamlit dashboard (requires [viz])

# Full validation
umcp validate .
```

---

## CLI Reference

UMCP provides **11 subcommands** via the `umcp` entry point, plus 5 additional entry points.

### Entry Points

| Command | Target | Install Extra |
|---------|--------|:-------------:|
| `umcp` | Main CLI (11 subcommands) | Core |
| `umcp-ext` | Extension manager | Core |
| `umcp-calc` | Universal calculator | Core |
| `umcp-finance` | Finance CLI | Core |
| `umcp-api` | REST API server (port 8000) | `[api]` |
| `umcp-dashboard` | Streamlit dashboard (port 8501) | `[viz]` |

### `umcp validate` -- Validate Artifacts

```bash
# Validate entire repository
umcp validate .

# Validate a specific casepack
umcp validate casepacks/hello_world

# Strict mode (warnings become errors)
umcp validate casepacks/hello_world --strict

# Validate a single file
umcp validate contracts/UMA.INTSTACK.v1.yaml
```

**What it checks**:
1. Schema conformance (JSON Schema Draft 2020-12)
2. Semantic rules (validator\_rules.yaml: E101, W201, etc.)
3. Tier-1 kernel identities: F = 1 - omega, IC approx exp(kappa), IC <= F
4. Regime classification: STABLE / WATCH / COLLAPSE
5. SHA256 integrity checksums
6. Seam residual: |s| <= 0.005

**Output**: JSON receipt + append to `ledger/return_log.csv`

### `umcp health` -- System Health

```bash
umcp health
```

Checks Python version, dependencies, file integrity, schema validity, contract checksums, and extension availability.

### `umcp preflight` -- Pre-Commit Readiness

```bash
umcp preflight
```

Runs all checks that CI will run: lint, type-check, test, validate.

### `umcp test` -- Run Tests

```bash
umcp test                     # Run all 2,476 tests
umcp test --coverage          # With coverage report
umcp test -k "gcd"            # Pattern matching
umcp test -m "not slow"       # Skip slow markers
```

### `umcp casepack` -- Run Specific Casepack

```bash
umcp casepack hello_world
umcp casepack gcd_complete
umcp casepack kinematics_complete
```

### `umcp list` -- List Artifacts

```bash
umcp list casepacks            # 13 casepacks
umcp list closures             # 57 registered closures
umcp list contracts            # 12 domain contracts
umcp list schemas              # 12 JSON schemas
```

### `umcp integrity` -- Verify SHA256

```bash
umcp integrity .               # Check all tracked files
umcp integrity src/umcp/       # Check source directory
```

### `umcp diff` -- Compare Receipts

```bash
umcp diff receipt1.json receipt2.json
```

Compares two validation receipts field-by-field, highlighting changed invariants.

### `umcp report` -- Generate Audit Report

```bash
umcp report                    # Full repository report
umcp report casepacks/gcd_complete  # Casepack-specific report
```

### `umcp run` -- Operational Validation

```bash
umcp run .                     # Alias for validate with operational context
```

### `umcp engine` -- Measurement Engine

```bash
umcp engine raw_measurements.csv   # Generate Psi(t) trace and invariants
```

Reads raw CSV measurements, computes kernel invariants, produces a full trace.

### Extension Manager (`umcp-ext`)

```bash
umcp-ext list                  # List all 5 extensions
umcp-ext info api              # Show extension details
umcp-ext check api             # Check dependencies
umcp-ext run api               # Launch extension
```

### Universal Calculator (`umcp-calc`)

```bash
umcp-calc                      # Interactive kernel calculator
```

---

## REST API

**57 endpoints** via FastAPI. Install and start:

```bash
pip install -e ".[api]"
umcp-api                       # Starts on http://localhost:8000
```

Interactive documentation: **http://localhost:8000/docs** (Swagger UI)

### Authentication

All endpoints except `/`, `/health`, and `/version` require an API key:

```bash
curl -H "X-API-Key: umcp-dev-key" http://localhost:8000/casepacks
```

### Endpoint Reference

#### System (3 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/version` | Version info |

#### Validation (1 endpoint)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/validate` | Validate a target path |

```bash
curl -X POST -H "X-API-Key: umcp-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"path": "casepacks/hello_world"}' \
  http://localhost:8000/validate
```

#### Casepacks (3 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/casepacks` | List all 13 casepacks |
| GET | `/casepacks/{id}` | Get casepack details |
| POST | `/casepacks/{id}/run` | Execute casepack validation |

```bash
curl -H "X-API-Key: umcp-dev-key" http://localhost:8000/casepacks
curl -X POST -H "X-API-Key: umcp-dev-key" http://localhost:8000/casepacks/hello_world/run
```

#### Ledger (2 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ledger` | Query validation history |
| GET | `/analysis/ledger` | Ledger analysis and statistics |

#### Discovery (3 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/contracts` | List 12 domain contracts |
| GET | `/closures` | List 57 registered closures |
| GET | `/domains` | List 9 domains |

#### Canon (3 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/canon` | List all canonical anchors |
| GET | `/canon/{domain}` | Domain-specific anchors |
| POST | `/regime/classify` | Classify regime from invariants |

```bash
curl -X POST -H "X-API-Key: umcp-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"omega": 0.02, "F": 0.98, "S": 0.05, "C": 0.03, "IC": 0.95}' \
  http://localhost:8000/regime/classify
```

#### Kernel Computation (3 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/kernel/compute` | Compute all 7 kernel invariants |
| POST | `/kernel/budget` | Compute budget identity |
| POST | `/calculate` | General calculation |

```bash
curl -X POST -H "X-API-Key: umcp-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"coordinates": [0.9, 0.85, 0.92], "weights": [0.5, 0.3, 0.2]}' \
  http://localhost:8000/kernel/compute
```

#### Uncertainty (1 endpoint)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/uncertainty/propagate` | Delta-method uncertainty propagation |

#### Analysis (4 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analysis/timeseries` | Time series analysis |
| POST | `/analysis/statistics` | Statistical summary |
| POST | `/analysis/correlation` | Correlation analysis |
| GET | `/analysis/ledger` | Ledger analytics |

#### Conversion (2 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/convert/measurements` | Unit conversion |
| POST | `/convert/embed` | Measurement embedding |

#### Output Formats (10 endpoints)

| Method | Path | Format |
|--------|------|--------|
| GET | `/badge/status.svg` | SVG status badge |
| GET | `/badge/regime.svg` | SVG regime badge |
| GET | `/output/ascii/gauge` | ASCII gauge |
| GET | `/output/ascii/sparkline` | ASCII sparkline |
| GET | `/output/markdown/report` | Markdown report |
| GET | `/output/mermaid/regime` | Mermaid diagram |
| GET | `/output/html/card` | HTML card |
| GET | `/output/latex/invariants` | LaTeX equations |
| GET | `/output/junit` | JUnit XML |
| GET | `/output/jsonld` | JSON-LD linked data |

#### Astronomy (6 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/astro/luminosity` | Stellar luminosity analysis |
| POST | `/astro/distance` | Distance modulus |
| POST | `/astro/spectral` | Spectral type analysis |
| POST | `/astro/evolution` | Stellar evolution |
| POST | `/astro/orbital` | Orbital mechanics |
| POST | `/astro/dynamics` | Stellar dynamics |

#### Nuclear Physics (6 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/nuclear/binding` | Binding energy stability |
| POST | `/nuclear/alpha-decay` | Alpha decay chain |
| POST | `/nuclear/shell` | Nuclear shell model |
| POST | `/nuclear/fissility` | Fissility parameter |
| POST | `/nuclear/decay-chain` | Full decay chain analysis |
| POST | `/nuclear/double-sided` | Double-sided collapse |

#### Quantum Mechanics (6 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/qm/collapse` | Wavefunction collapse |
| POST | `/qm/entanglement` | Bell state entanglement |
| POST | `/qm/tunneling` | Tunneling transmission |
| POST | `/qm/harmonic-oscillator` | Harmonic oscillator fidelity |
| POST | `/qm/spin` | Spin measurement stability |
| POST | `/qm/uncertainty` | Uncertainty principle |

#### Finance (1 endpoint)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/finance/embed` | Financial data embedding |

#### WEYL Cosmology (4 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/weyl/background` | Background cosmology |
| GET | `/weyl/sigma` | Sigma computation |
| GET | `/weyl/des-y3` | DES Y3 data integration |
| GET | `/weyl/umcp-mapping` | UMCP-WEYL mapping |

---

## Dashboard

**31 interactive pages** via Streamlit. Install and start:

```bash
pip install -e ".[viz]"
umcp-dashboard                 # Starts on http://localhost:8501
```

### Page Reference

| Category | Page | Description |
|----------|------|-------------|
| **Core** | Overview | System summary, test status, regime gauge |
| | Domain Overview | Cross-domain comparison |
| | Canon Explorer | Browse canonical anchor values |
| | Precision | Numerical precision analysis |
| | Geometry | Infrastructure geometry visualization |
| | Ledger | Browse validation history |
| | Casepacks | Explore and run casepacks |
| | Contracts | View frozen contracts |
| | Closures | Browse registered closures |
| | Regime | Real-time regime classification |
| | Metrics | Kernel invariant trends |
| | Health | System health dashboard |
| **Interactive** | Live Runner | Interactive single validation |
| | Batch Validation | Run multiple casepacks |
| | Test Templates | Generate test templates |
| **Domain** | Astronomy | Stellar classification, HR diagram |
| | Nuclear | Binding energy, decay chains |
| | Quantum | Wavefunction, entanglement |
| | Finance | Portfolio continuity |
| | RCFT | Fractal dimension, recursion |
| | Physics | GCD energy/collapse |
| | Kinematics | Phase space, motion |
| | Cosmology | WEYL modified gravity |
| **Analysis** | Formula Builder | Build custom formulas |
| | Time Series | Temporal trend analysis |
| | Comparison | Side-by-side receipt comparison |
| **Management** | Exports | Export data (CSV, JSON, LaTeX) |
| | Bookmarks | Save and recall views |
| | Notifications | Alert configuration |
| | API Integration | REST API control panel |

---

## Fleet -- Distributed Validation

UMCP Fleet provides **distributed, parallel validation** at scale via:
- **Scheduler**: Job submission, routing, and tracking
- **Worker**: Register workers, heartbeat, execute validations
- **Queue**: Priority queue with DLQ, retry logic, backpressure
- **Cache**: Content-addressable artifact cache (SHA256)
- **Tenant**: Multi-tenant isolation, quotas, namespaces

### Quick Start

```python
from umcp.fleet import Scheduler, Worker, WorkerPool, Tenant

# Create tenant and scheduler
tenant = Tenant(tenant_id="acme")
scheduler = Scheduler()

# Start worker pool (4 workers by default)
pool = WorkerPool(pool_size=4)
pool.start()

# Submit validation job
job = scheduler.submit("casepacks/hello_world", tenant=tenant)

# Wait for result
result = scheduler.wait(job.job_id, timeout=60.0)
print(f"Verdict: {result.verdict}")

# Clean up
pool.stop()
```

### Architecture

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Scheduler** | Job orchestration | Priority routing, status tracking, result aggregation |
| **Worker** | Task execution | Subprocess isolation, heartbeat, drain/stop lifecycle |
| **WorkerPool** | Fleet management | Scale up/down, graceful drain, health monitoring |
| **Queue** | Job buffering | Priority lanes, DLQ for failures, retry with backoff |
| **Cache** | Artifact storage | SHA256 identity, multi-node sharing, LRU eviction |
| **Tenant** | Isolation | Per-tenant quotas, namespaces, resource limits |

### Scaling Example

```python
# Start with 2 workers
pool = WorkerPool(pool_size=2)
pool.start()

# Scale up to handle load
new_workers = pool.scale_up(3)  # Now 5 total
print(f"Active workers: {pool.active}")

# Drain for maintenance (stop accepting new work)
pool.drain()

# Wait for in-flight jobs to complete
time.sleep(10)

# Scale down by 2 workers
removed = pool.scale_down(2)  # Now 3 workers
```

### Configuration

```python
from umcp.fleet.worker import WorkerConfig

config = WorkerConfig(
    poll_interval_s=1.0,        # Job polling frequency
    heartbeat_interval_s=5.0,   # Heartbeat to scheduler
    validate_timeout_s=300.0,   # Max validation time
    capacity=1,                 # Concurrent jobs per worker
    tags={"env": "prod"}        # Worker metadata
)

worker = Worker("worker-1", config=config)
```

**Fleet is production-ready** with graceful shutdown, backpressure, and telemetry hooks for monitoring.

---

## Python API

### Basic Validation

```python
import umcp

# Validate a casepack (returns True for CONFORMANT)
result = umcp.validate("casepacks/hello_world")
print(f"Status: {'CONFORMANT' if result else 'NONCONFORMANT'}")
```

### Compute Kernel Invariants

```python
from umcp.frozen_contract import compute_kernel, classify_regime
import numpy as np

c = np.array([0.9, 0.85, 0.92])   # Coherence values in [0,1]
w = np.array([0.5, 0.3, 0.2])     # Weights (must sum to 1)
kernel = compute_kernel(c, w, tau_R=5.0)

print(f"Drift (omega):      {kernel.omega:.6f}")
print(f"Fidelity (F):       {kernel.F:.6f}")
print(f"Entropy (S):        {kernel.S:.6f}")
print(f"Curvature (C):      {kernel.C:.6f}")
print(f"Log-integrity (k):  {kernel.kappa:.6f}")
print(f"Integrity (IC):     {kernel.IC:.6f}")
```

### Classify Regime

```python
from umcp.frozen_contract import classify_regime

regime = classify_regime(
    omega=0.02,
    F=0.98,
    S=0.05,
    C=0.03,
    integrity=0.95
)
print(f"Regime: {regime.name}")  # STABLE, WATCH, or COLLAPSE
```

### Optimized Kernel (Lemma-Based)

```python
from umcp.kernel_optimized import compute_kernel_optimized

result = compute_kernel_optimized(
    coordinates=[0.9, 0.85, 0.92],
    weights=[0.5, 0.3, 0.2]
)
# Returns KernelInvariants NamedTuple with all 7 metrics
```

### Budget Identity

```python
from umcp.frozen_contract import compute_budget

budget = compute_budget(omega=0.05, C=0.1, tau_R=5.0, R=1.0)
print(f"Gamma(omega): {budget.D_omega:.6f}")
print(f"D_C:          {budget.D_C:.6f}")
print(f"Seam:         {budget.seam:.6f}")
```

### Thermodynamic Diagnostics (tau\_R\*)

```python
from umcp.tau_r_star import diagnose, diagnose_batch, classify_phase

# Single diagnosis
diag = diagnose(omega=0.05, C=0.1, tau_R=5.0, R=1.0)
print(f"tau_R*:  {diag.tau_R_star}")
print(f"Phase:   {diag.phase}")
print(f"Regime:  {diag.regime}")

# Phase classification
phase = classify_phase(omega=0.05, C=0.1, R=1.0)
print(f"Phase: {phase.name}")  # SURPLUS, BALANCED, DEFICIT, TRAPPED

# Batch diagnostics
results = diagnose_batch([
    {"omega": 0.02, "C": 0.05, "tau_R": 3.0, "R": 1.0},
    {"omega": 0.15, "C": 0.20, "tau_R": 8.0, "R": 0.5},
])
```

### Uncertainty Propagation

```python
from umcp.uncertainty import propagate_uncertainty

result = propagate_uncertainty(
    coordinates=[0.9, 0.85, 0.92],
    weights=[0.5, 0.3, 0.2],
    coord_uncertainties=[0.01, 0.02, 0.01]
)
print(f"omega uncertainty: {result.sigma_omega:.6f}")
print(f"F uncertainty:     {result.sigma_F:.6f}")
```

### SS1M Triads (Human-Verifiable Checksums)

```python
from umcp.ss1m_triad import compute_triad

triad = compute_triad(omega=0.05, F=0.95, IC=0.90)
print(f"Triad: {triad}")  # mod-97 human-checkable
```

### Seam Chain Accumulation

```python
from umcp.seam_optimized import accumulate_seam_chain

chain = accumulate_seam_chain(
    omega_sequence=[0.02, 0.03, 0.04, 0.05],
    C_sequence=[0.05, 0.06, 0.07, 0.08],
    tau_R_sequence=[3, 4, 5, 6],
    R=1.0
)
# Returns cumulative seam residuals for the chain
```

### Epistemic Weld (Gesture / Return / Dissolution)

```python
from umcp.epistemic_weld import (
    classify_epistemic_act,
    quantify_positional_illusion,
    assess_seam_epistemology,
    diagnose_gesture,
)
from umcp.frozen_contract import Regime

# Classify an epistemic emission
verdict, reasons = classify_epistemic_act(
    seam_pass=True, tau_R=1.85, regime=Regime.STABLE
)
print(f"Verdict: {verdict.value}")  # "return"

# Quantify the positional illusion (observation cost)
illusion = quantify_positional_illusion(omega=0.031, n_observations=10)
print(f"Cost per observation: {illusion.gamma:.2e}")
print(f"Illusion severity: {illusion.illusion_severity:.4f}")

# Complete seam epistemology assessment
epi = assess_seam_epistemology(
    seam_pass=True, seam_failures=[], seam_residual=0.0,
    seam_budget=1.697, tau_R=1.85, omega=0.031, regime=Regime.STABLE
)
print(f"Is real: {epi.is_real}")       # True — earned epistemic credit
print(f"Verdict: {epi.verdict.value}")  # "return"

# Diagnose why a gesture did not return
diag = diagnose_gesture(
    seam_residual=0.01, tau_R=float("inf"), omega=0.15, regime=Regime.WATCH
)
print(f"Return status: {diag['return_status']}")  # "INF_REC"
```

---

## Frameworks & Domains

UMCP validates computational workflows across **12 scientific domains**. Each domain has its own closures, contracts, canonical anchors, and casepacks.

| Domain | Code | Tier | Closures | Contract | Casepack |
|--------|:----:|:----:|:--------:|----------|----------|
| **GCD** (Generative Collapse Dynamics) | gcd | 2 | 5 | GCD.INTSTACK.v1 | `gcd_complete` |
| **Kinematics** | kin | 0 | 7 | KIN.INTSTACK.v1 | `kinematics_complete` |
| **RCFT** (Recursive Collapse Field Theory) | rcft | 2 | 5 | RCFT.INTSTACK.v1 | `rcft_complete` |
| **WEYL** (Cosmology) | weyl | 2 | 6 | WEYL.INTSTACK.v1 | `weyl_des_y3` |
| **Security** | security | 2 | 8 | SECURITY.INTSTACK.v1 | `security_validation` |
| **Astronomy** | astronomy | 2 | 6 | ASTRO.INTSTACK.v1 | `astronomy_complete` |
| **Nuclear Physics** | nuclear | 2 | 6 | NUC.INTSTACK.v1 | `nuclear_chain` |
| **Quantum Mechanics** | qm | 2 | 6 | QM.INTSTACK.v1 | `quantum_mechanics_complete` |
| **Finance** | finance | 2 | 6 | FINANCE.INTSTACK.v1 | `finance_continuity` |
| **Atomic Physics** | atom | 2 | 9 | ATOM.INTSTACK.v1 | — |
| **Materials Science** | matl | 2 | 1 | MATL.INTSTACK.v1 | — |
| **Standard Model** | sm | 2 | 7 | SM.INTSTACK.v1 | — |

### GCD Closures

`energy_potential`, `entropic_collapse`, `generative_flux`, `field_resonance`, `boundary_detection`

```bash
umcp validate casepacks/gcd_complete
```

### Kinematics Closures

`linear_kinematics`, `rotational_kinematics`, `energy_mechanics`, `momentum_dynamics`, `phase_space_return`, `kinematic_stability`, `kinematic_trajectory`

```bash
umcp validate casepacks/kinematics_complete
umcp casepack kin_ref_phase_oscillator
```

### RCFT Closures

All GCD closures + `fractal_dimension`, `recursive_field`, `resonance_pattern`, `information_geometry`, `universality_class`, `collapse_grammar`, `active_matter`

```bash
umcp validate casepacks/rcft_complete
```

### Astronomy Closures

`stellar_luminosity`, `main_sequence_stability`, `hr_diagram_classification`, `spectral_type_analysis`, `metallicity_evolution`, `stellar_age_coherence`

```bash
umcp validate casepacks/astronomy_complete
```

### Nuclear Physics Closures

`binding_energy_stability`, `alpha_decay_chain`, `fissility_parameter`, `nuclear_shell_model`, `decay_chain_analysis`, `double_sided_collapse`

```bash
umcp validate casepacks/nuclear_chain
```

### Quantum Mechanics Closures

`wavefunction_coherence`, `density_matrix_stability`, `bell_state_entanglement`, `tunneling_transmission`, `harmonic_oscillator_fidelity`, `spin_measurement_stability`

```bash
umcp validate casepacks/quantum_mechanics_complete
```

### Finance Closures

`portfolio_continuity`, `market_coherence`, `volatility_regime`, `correlation_stability`, `drawdown_analysis`, `return_fidelity`

```bash
umcp validate casepacks/finance_continuity
```

### Security Closures

`input_sanitizer`, `schema_firewall`, `rate_limiter`, `integrity_monitor`, `access_controller`, `audit_logger`, `encryption_validator`, `boundary_enforcer`

```bash
umcp validate casepacks/security_validation
```

### WEYL Closures

DES Y3 integration, modified gravity analysis, sigma computation, background cosmology, UMCP mapping, distance modulus

```bash
umcp validate casepacks/weyl_des_y3
```

### Standard Model Closures

`particle_catalog`, `coupling_constants`, `cross_sections`, `symmetry_breaking`, `ckm_mixing`, `subatomic_kernel`, `particle_physics_formalism`

The **subatomic kernel** maps 31 particles (17 fundamental + 14 composite) to 8-channel trace vectors encoding mass, spin, charge, color, weak isospin, lepton/baryon number, and generation. All 2,476 pass Tier-1 identities.

The **particle physics formalism** proves 10 theorems connecting Standard Model physics to GCD kernel patterns:

| # | Theorem | Tests | Key Result |
|---|---------|:-----:|------------|
| T1 | Spin-Statistics | 12/12 | F\_fermion(0.615) > F\_boson(0.421), split = 0.194 |
| T2 | Generation Monotonicity | 5/5 | Gen1 < Gen2 < Gen3 (quarks AND leptons) |
| T3 | Confinement as IC Collapse | 19/19 | IC drops 98.1% from quarks to hadrons |
| T4 | Mass-Kernel Log Mapping | 5/5 | 13.2 OOM mass range maps to F in [0.37, 0.73] |
| T5 | Charge Quantization | 5/5 | IC\_neutral/IC\_charged = 0.020 (50x suppression) |
| T6 | Cross-Scale Universality | 6/6 | composite < atom < fundamental (same kernel) |
| T7 | Symmetry Breaking | 5/5 | EWSB amplifies generation spread 0.046 to 0.073 |
| T8 | CKM Unitarity | 5/5 | CKM rows pass Tier-1; Jarlskog J = 3.0e-5 |
| T9 | Running Coupling Flow | 6/6 | Asymptotic freedom monotonic for Q >= 10 GeV |
| T10 | Nuclear Binding Curve | 6/6 | r(BE/A, gap) = -0.41; peak at Cr/Fe |

```bash
python closures/standard_model/subatomic_kernel.py
python closures/standard_model/particle_physics_formalism.py
```

### Atomic Physics Closures

`electron_config`, `fine_structure`, `ionization_energy`, `spectral_lines`, `selection_rules`, `zeeman_stark`, `periodic_kernel`, `cross_scale_kernel`, `tier1_proof`

The **periodic kernel** maps all 118 elements to GCD trace vectors using 8 measurable atomic properties (Z, mass, ionization energy, electronegativity, density, melting/boiling points, atomic radius). All 2,476 pass Tier-1.

The **cross-scale kernel** uses 12 nuclear-informed channels (4 nuclear + 2 electronic + 6 bulk) bridging subatomic and atomic scales. Magic number proximity is the #1 IC driver (39%).

The **Tier-1 proof** exhaustively tests 10,162 identity checks across all 118 elements: F + omega = 1 (definitional), IC <= F (Jensen's inequality), IC = exp(kappa) (definitional). Zero failures.

```bash
python closures/atomic_physics/periodic_kernel.py
python closures/atomic_physics/cross_scale_kernel.py
python closures/atomic_physics/tier1_proof.py
```

### Materials Science Closures

`element_database` -- Complete database of all 118 elements with 18 fields each (Z, symbol, name, mass, period, group, block, category, ionization energy, electron affinity, electronegativity, density, melting/boiling points, atomic/covalent radius, electron configuration).

---

## Casepacks

A **casepack** is a self-contained, reproducible computational experiment. Each casepack directory contains:

| File | Purpose |
|------|---------|
| `manifest.yaml` or `manifest.json` | References contract, closures, expected outputs |
| `raw_measurements.csv` | Input data |
| `expected/` | Expected output files |
| `closures.yaml` | Closure parameters |
| `contract.yaml` | Contract reference |

### Available Casepacks (13)

| Casepack | Domain | Description |
|----------|--------|-------------|
| `hello_world` | Core | Zero-entropy baseline, smoke test |
| `gcd_complete` | GCD | Full energy/collapse validation |
| `kinematics_complete` | KIN | Complete kinematics validation |
| `kin_ref_phase_oscillator` | KIN | Phase oscillator reference |
| `rcft_complete` | RCFT | Fractal/recursive validation |
| `weyl_des_y3` | WEYL | DES Y3 cosmological data |
| `astronomy_complete` | ASTRO | Stellar classification |
| `nuclear_chain` | NUC | Decay chain analysis |
| `quantum_mechanics_complete` | QM | Quantum state validation |
| `finance_continuity` | FIN | Portfolio continuity |
| `security_validation` | SEC | Input validation security |
| `retro_coherent_phys04` | Core | Retro-coherent physics |
| `UMCP-REF-E2E-0001` | Core | End-to-end reference |

### Create Your Own Casepack

```bash
# 1. Copy the template
cp -r casepacks/hello_world casepacks/my_experiment

# 2. Edit the manifest
nano casepacks/my_experiment/manifest.yaml

# 3. Add your raw measurements
nano casepacks/my_experiment/raw_measurements.csv

# 4. Validate
umcp validate casepacks/my_experiment
```

### Casepack Manifest Format

```yaml
version: "1.0"
contract: "UMA.INTSTACK.v1"
description: "My experiment description"
closures:
  - gamma
  - return_domain
  - norms
expected_outputs:
  - expected/invariants.json
```

---

## Testing

### Run Tests

```bash
# All 2,476 tests
pytest

# Verbose
pytest -v

# With coverage
pytest --cov=umcp --cov-report=html

# Pattern matching
pytest -k "gcd"
pytest -k "kinematics"
pytest -k "rcft"
pytest -k "tau_r_star"

# Specific file
pytest tests/test_145_tau_r_star.py -v

# Parallel execution
pytest -n auto

# Via CLI
umcp test
umcp test --coverage
```

### Test Distribution (80 files, 2,476 tests)

| Category | Tests | Description |
|----------|------:|-------------|
| Schema validation | 73 | JSON/YAML schema conformance |
| Kernel invariants | 84 | Core metric computation |
| GCD framework | 92 | Energy/collapse closures |
| Kinematics | 133 | Motion analysis, phase space |
| RCFT framework | 140 | Fractal/recursive/universality/active matter closures |
| WEYL framework | 43 | Cosmology closures |
| Extended lemmas | 75 | Lemmas 24-46 |
| Frozen contract | 113 | Contract claims, constants |
| SS1M triads | 35 | Mod-97 checksums |
| Uncertainty | 23 | Delta-method propagation |
| API | 32 | REST endpoint tests |
| Dashboard | 30 | UI component tests |
| Security | 45 | Input validation |
| CLI subcommands | 13 | CLI integration |
| Batch / compute | 39 | Vectorized pipeline |
| Public API | 11 | validate() + ValidationResult |
| Finance | 48 | Domain coverage |
| Astronomy | 28 | Stellar closures |
| Nuclear | 24 | Binding/decay |
| Quantum | 24 | Wavefunction/entanglement |
| tau\_R\* thermodynamics | 79 | Budget, phase, predictions |
| Epistemic weld | 55 | Gesture/return/dissolution trichotomy, positional illusion |
| Active matter | 62 | Frictional cooling, Antonov et al. predictions |
| Integration | 150+ | End-to-end workflows |

---

## Pre-Commit Protocol

**Before every commit**, run the pre-commit protocol. This mirrors CI exactly and must exit 0.

```bash
python scripts/pre_commit_protocol.py         # Auto-fix + validate (default)
python scripts/pre_commit_protocol.py --check  # Dry-run: report only
```

What it does:

1. `ruff format` -- auto-fix formatting
2. `ruff check --fix` -- auto-fix lint issues
3. `mypy src/umcp` -- type checking (reports, non-blocking)
4. `git add -A` -- stage all changes
5. `python scripts/update_integrity.py` -- regenerate SHA256 checksums
6. `pytest` -- run full test suite
7. `umcp validate .` -- must be CONFORMANT

### Manual Equivalent

```bash
ruff format src/umcp tests
ruff check src/umcp tests --fix
mypy src/umcp
python scripts/update_integrity.py
pytest
umcp validate .
```

### Integrity Checksums

After modifying any tracked file (`src/umcp/*.py`, `contracts/*.yaml`, `closures/**`, `schemas/**`, `scripts/*.py`), you **must** regenerate checksums:

```bash
python scripts/update_integrity.py
```

This updates `integrity/sha256.txt`. CI will fail on mismatch.

---

## Extension System

UMCP uses a `typing.Protocol`-based extension system with **5 built-in extensions**.

### Registered Extensions

| Extension | Type | Module | Port | Description |
|-----------|------|--------|:----:|-------------|
| `api` | API | `umcp.api_umcp` | 8000 | 57 REST endpoints, Swagger UI |
| `visualization` | Dashboard | `umcp.dashboard` | 8501 | 31 interactive pages |
| `ledger` | Logging | `umcp.validator` | -- | Append-only return log |
| `formatter` | Tool | `umcp.validator` | -- | Multi-format output |
| `thermodynamics` | Validator | `umcp.tau_r_star` | -- | tau\_R\* diagnostics, phase diagram |

### Extension Commands

```bash
# List all extensions
umcp-ext list

# Show extension details
umcp-ext info api
umcp-ext info visualization
umcp-ext info thermodynamics

# Check dependencies
umcp-ext check api
umcp-ext check visualization

# Launch extension
umcp-ext run api                # Start REST API
umcp-ext run visualization      # Start dashboard
```

### Extension Protocol

To write a custom extension, implement `ExtensionProtocol`:

```python
from typing import Protocol

class ExtensionProtocol(Protocol):
    name: str
    version: str
    description: str

    def check_dependencies(self) -> bool: ...
```

---

## Documentation Map

### Core References

| Document | Description |
|----------|-------------|
| [AXIOM.md](AXIOM.md) | Core axiom: "What Returns Through Collapse Is Real" |
| [KERNEL\_SPECIFICATION.md](KERNEL_SPECIFICATION.md) | 46 formal lemmas, prediction scorecard, constants |
| [MATHEMATICAL\_ARCHITECTURE.md](docs/MATHEMATICAL_ARCHITECTURE.md) | Complete mathematical framework |
| [TIER\_SYSTEM.md](TIER_SYSTEM.md) | Tier-0/1/2 architecture (v3.0.0) |
| [INFRASTRUCTURE\_GEOMETRY.md](docs/INFRASTRUCTURE_GEOMETRY.md) | Three-layer geometric architecture |
| [GLOSSARY.md](GLOSSARY.md) | Authoritative term definitions |
| [SYMBOL\_INDEX.md](docs/SYMBOL_INDEX.md) | Symbol reference table |

### Epistemic & Foundational

| Document | Description |
|----------|-------------|
| [src/umcp/epistemic\_weld.py](src/umcp/epistemic_weld.py) | Seam epistemology module (gesture/return/dissolution, positional illusion) |
| [src/umcp/tau\_r\_star.py](src/umcp/tau_r_star.py) | tau\_R\* thermodynamics, Thm T9 (measurement cost / positional illusion) |
| [src/umcp/frozen\_contract.py](src/umcp/frozen_contract.py) | Frozen constants, seam PASS/FAIL, NonconformanceType (incl. GESTURE) |

### Developer Guides

| Document | Description |
|----------|-------------|
| [QUICKSTART\_TUTORIAL.md](QUICKSTART_TUTORIAL.md) | 10-minute hands-on tutorial |
| [docs/quickstart.md](docs/quickstart.md) | Getting started guide |
| [docs/python\_coding\_key.md](docs/python_coding_key.md) | Development standards |
| [docs/production\_deployment.md](docs/production_deployment.md) | Enterprise deployment |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |

### Framework Documentation

| Document | Description |
|----------|-------------|
| [canon/gcd\_anchors.yaml](canon/gcd_anchors.yaml) | GCD canonical anchors |
| [canon/kin\_anchors.yaml](canon/kin_anchors.yaml) | Kinematics anchors |
| [canon/rcft\_anchors.yaml](canon/rcft_anchors.yaml) | RCFT anchors |
| [canon/weyl\_anchors.yaml](canon/weyl_anchors.yaml) | WEYL anchors |
| [canon/astro\_anchors.yaml](canon/astro_anchors.yaml) | Astronomy anchors |
| [canon/nuc\_anchors.yaml](canon/nuc_anchors.yaml) | Nuclear physics anchors |
| [canon/qm\_anchors.yaml](canon/qm_anchors.yaml) | Quantum mechanics anchors |
| [KINEMATICS\_SPECIFICATION.md](KINEMATICS_SPECIFICATION.md) | Kinematics layer specification |
| [CASEPACK\_REFERENCE.md](docs/CASEPACK_REFERENCE.md) | CasePack structure reference |

### Reference

| Document | Description |
|----------|-------------|
| [EXTENSION\_INTEGRATION.md](docs/EXTENSION_INTEGRATION.md) | Extension system |
| [COMPUTATIONAL\_OPTIMIZATIONS.md](docs/COMPUTATIONAL_OPTIMIZATIONS.md) | Optimization cross-references |
| [validator\_rules.yaml](validator_rules.yaml) | Semantic validation rules (E/W codes) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [IMMUTABLE\_RELEASE.md](IMMUTABLE_RELEASE.md) | Release information |

---

## Repository Structure

```
UMCP-Metadata-Runnable-Code/
|
|-- src/umcp/                     # Core Python implementation (25 modules)
|   |-- __init__.py               # Public API, __version__
|   |-- __main__.py               # python -m umcp
|   |-- cli.py                    # CLI engine (11 subcommands)
|   |-- validator.py              # Root-file validator, Tier-0 checks
|   |-- frozen_contract.py        # Canonical constants, cost closures
|   |-- kernel_optimized.py       # Lemma-based kernel computation
|   |-- tau_r_star.py             # tau_R* thermodynamic diagnostics
|   |-- constants.py              # Regime enum, thresholds
|   |-- uncertainty.py            # Delta-method propagation
|   |-- ss1m_triad.py             # Mod-97 checksums
|   |-- seam_optimized.py         # Seam chain accumulation
|   |-- compute_utils.py          # Computational utilities
|   |-- measurement_engine.py     # Raw CSV to epistemic trace production
|   |-- epistemic_weld.py         # Seam epistemology (gesture/return/dissolution)
|   |-- closures.py               # Closure utilities
|   |-- outputs.py                # Multi-format output
|   |-- preflight.py              # Pre-commit checks
|   |-- logging_utils.py          # Logging utilities
|   |-- file_refs.py              # File reference tracking
|   |-- universal_calculator.py   # Interactive calculator
|   |-- finance_cli.py            # Finance CLI
|   |-- finance_dashboard.py      # Finance dashboard
|   |-- minimal_cli.py            # Minimal CLI
|   |-- api_umcp.py               # FastAPI REST API (57 endpoints)
|   |-- dashboard.py              # Streamlit dashboard (31 pages)
|   '-- umcp_extensions.py        # Extension registry (5 extensions)
|
|-- tests/                        # Test suite (80 files, 2,476 tests)
|   |-- conftest.py               # Fixtures: RepoPaths, caching helpers
|   |-- test_00_* .. test_145_*   # Numbered test groups
|   '-- closures/                 # Closure-specific tests
|
|-- casepacks/                    # Reproducible experiments (13)
|   |-- hello_world/              # Zero entropy baseline
|   |-- gcd_complete/             # GCD validation
|   |-- kinematics_complete/      # Kinematics validation
|   |-- rcft_complete/            # RCFT validation
|   |-- weyl_des_y3/              # WEYL cosmology
|   |-- astronomy_complete/       # Stellar classification
|   |-- nuclear_chain/            # Decay chains
|   |-- quantum_mechanics_complete/ # Quantum states
|   |-- finance_continuity/       # Portfolio analysis
|   |-- security_validation/      # Input validation
|   '-- ...
|
|-- closures/                     # Computational functions (69 files, 9 domains)
|   |-- gcd/                      # GCD closures
|   |-- kinematics/               # Kinematics closures
|   |-- rcft/                     # RCFT closures
|   |-- weyl/                     # WEYL closures
|   |-- security/                 # Security closures
|   |-- astronomy/                # Astronomy closures
|   |-- nuclear_physics/          # Nuclear closures
|   |-- quantum_mechanics/        # QM closures
|   |-- finance/                  # Finance closures
|   '-- registry.yaml             # Central closure registry (57 entries)
|
|-- contracts/                    # Frozen mathematical contracts (12 YAML)
|-- schemas/                      # JSON Schema Draft 2020-12 (12 schemas)
|-- canon/                        # Canonical anchor values (8 domain files)
|-- integrity/                    # SHA256 checksums (auto-generated)
|-- ledger/                       # Append-only validation log
|-- scripts/                      # Maintenance scripts
|   |-- update_integrity.py       # Regenerate checksums
|   '-- pre_commit_protocol.py    # CI-mirror pre-commit
|-- data/                         # Physics observations
|-- docs/                         # Additional documentation
|-- artifacts/                    # CI validation artifacts
'-- pyproject.toml                # Project configuration
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

### Quick Workflow

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# 2. Set up development environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# 3. Create feature branch
git checkout -b feat/your-feature

# 4. Make changes, then run pre-commit protocol
python scripts/pre_commit_protocol.py

# 5. Commit and push
git commit -m "feat: your feature description"
git push origin feat/your-feature

# 6. Open Pull Request
```

### Code Quality Standards

| Check | Command | Requirement |
|-------|---------|-------------|
| Tests | `pytest` | All 2,476 pass |
| Lint | `ruff check src/umcp tests` | Zero errors |
| Format | `ruff format src/umcp tests` | Formatted |
| Types | `mypy src/umcp` | Clean |
| Validation | `umcp validate .` | CONFORMANT |
| Integrity | `python scripts/update_integrity.py` | Checksums match |

---

## License

MIT License. See [LICENSE](LICENSE) for full text.

Copyright (c) 2026 Clement Paulus

---

<div align="center">

**UMCP v2.0.0** &bull; 2,476 tests &bull; 57 API endpoints &bull; 31 dashboard pages &bull; 11 CLI commands &bull; 13 casepacks &bull; 9 domains &bull; 46 lemmas &bull; CONFORMANT

*"What Returns Through Collapse Is Real"*

</div>
