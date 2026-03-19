"""
UMCP REST API — FastAPI Communication Extension (v2.2.3)

Provides HTTP endpoints for remote validation, kernel computation,
ledger access, cross-domain analysis, and domain-specific closure
computations across all 17 UMCP domains.

The Spine governs every response:
  CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE

Core Endpoints:
  GET  /health              - System health + metrics
  GET  /version             - API and validator version info
  GET  /metrics             - Operational metrics and request stats
  POST /validate            - Validate a casepack or repository
  GET  /casepacks           - List available casepacks
  GET  /casepacks/{id}      - Get casepack details
  POST /casepacks/{id}/run  - Run a casepack
  GET  /ledger              - Query the return log ledger

Kernel Endpoints:
  POST /kernel/compute      - Compute kernel invariants (F, ω, S, C, κ, IC)
  POST /kernel/batch        - Batch kernel computation (multiple traces)
  POST /kernel/budget       - Verify seam budget identity
  POST /kernel/spine        - Full spine evaluation (Contract → Stance)

Domain Endpoints (18 domains):
  GET  /domains             - List all 18 UMCP domains with metadata
  POST /astro/*             - Astronomy closures (6 endpoints)
  POST /nuclear/*           - Nuclear physics closures (6 endpoints)
  POST /qm/*                - Quantum mechanics closures (6 endpoints)
  POST /finance/embed       - Finance embedding
  GET  /weyl/*              - WEYL cosmology endpoints (4 endpoints)
  GET  /sm/particles        - Standard Model particle catalog
  GET  /sm/theorems         - Standard Model 10 proven theorems
  GET  /evolution/organisms - Evolution kernel organisms
  GET  /atomic/elements     - Periodic table kernel data

Usage:
  uvicorn umcp.api_umcp:app --reload --host 0.0.0.0 --port 8000

Cross-references:
  - src/umcp/frozen_contract.py (frozen parameters — seam-derived)
  - src/umcp/kernel_optimized.py (kernel computation)
  - src/umcp/validator.py (validation engine)
  - src/umcp/seam_optimized.py (seam budget calculus)
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Query, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Import UMCP core modules
try:
    from . import __version__
    from .frozen_contract import (
        ALPHA,
        DEFAULT_THRESHOLDS,
        EPSILON,
        P_EXPONENT,
        TOL_SEAM,
        Regime,
    )
    from .frozen_contract import classify_regime as _fc_classify_regime
except ImportError:
    # Fallback for direct execution
    __version__ = "2.2.3"
    EPSILON = 1e-8
    P_EXPONENT = 3
    ALPHA = 1.0
    TOL_SEAM = 0.005
    DEFAULT_THRESHOLDS = None  # type: ignore[assignment]
    Regime = None  # type: ignore[assignment, misc]
    _fc_classify_regime = None  # type: ignore[assignment]

# ============================================================================
# Configuration
# ============================================================================

API_VERSION = "3.0.0"
API_TITLE = "UMCP REST API"
API_DESCRIPTION = """
**Universal Measurement Contract Protocol** — REST API v3.0.0

*"Collapse is generative; only what returns is real."* — Axiom-0

**100+ endpoints** across kernel computation, validation, ledger analysis,
thermodynamic diagnostics, epistemic cost tracking, insight discovery,
cross-domain Rosetta translation, live orientation, and domain-specific
closure computations across **18 domains** from particle physics to
consciousness coherence.

**The Spine**: CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE

**Kernel**: K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC)

**Engines**: Seam Chain · τ_R* Diagnostic · Epistemic Weld · Insight Engine · Rosetta

**Standard Model**: Particles · Theorems · CKM · Couplings · Cross Sections · Higgs ·
Neutrino Oscillation · Matter Genesis (99 entities) · Matter Map (485 entities, 6 scales)

**Domains**: GCD · KIN · RCFT · WEYL · Security · ASTRO · NUC · QM · FIN ·
Atomic · Materials · Everyday · Evolution · Semiotics · Consciousness ·
Continuity · Standard Model

**Access Control**: All read and compute endpoints are publicly accessible —
no API key needed.  State-mutating admin operations (e.g. seam reset)
require an admin key via the `X-API-Key` header (set `UMCP_ADMIN_KEY` env var).
Set `UMCP_DEV_MODE=1` to promote all callers to admin during development.

**Three-valued verdicts**: CONFORMANT / NONCONFORMANT / NON_EVALUABLE — never boolean.
"""

# ============================================================================
# Metrics Tracking
# ============================================================================

_METRICS: dict[str, Any] = {
    "start_time": time.monotonic(),
    "start_utc": datetime.now(UTC).isoformat(),
    "total_requests": 0,
    "total_errors": 0,
    "kernel_computations": 0,
    "batch_computations": 0,
    "validations": 0,
    "endpoints_hit": {},
}

# ── Auth (single source of truth in auth.py) ──
try:
    from .auth import ADMIN_KEY, DEV_MODE, require_admin, require_public
except ImportError:  # direct-run fallback
    DEV_MODE = os.environ.get("UMCP_DEV_MODE", "0").lower() in ("1", "true", "yes")
    ADMIN_KEY = os.environ.get("UMCP_ADMIN_KEY", os.environ.get("UMCP_API_KEY", "umcp-dev-key"))

    def require_public(api_key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> str:  # type: ignore[misc]
        return api_key or "public"

    def require_admin(api_key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> str:  # type: ignore[misc]
        if DEV_MODE:
            return "dev-mode-admin"
        if not api_key or api_key != ADMIN_KEY:
            raise HTTPException(status_code=401, detail="Admin API key required")
        return api_key


# Keep legacy aliases so existing tests that reference them still work
API_KEY = ADMIN_KEY
API_KEY_NAME = "X-API-Key"


def get_repo_root() -> Path:
    """Find the repository root (contains pyproject.toml)."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


# Legacy wrappers — kept for backward compatibility with existing tests
def verify_api_key(api_key: str | None = None) -> bool:
    """Legacy: verify API key (use require_public / require_admin instead)."""
    if DEV_MODE:
        return True
    if api_key is None:
        return False
    return api_key == API_KEY


def validate_api_key(api_key: str | None = None) -> str:
    """Legacy: validate API key (use require_public / require_admin instead)."""
    if DEV_MODE:
        return "dev-mode-enabled"
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ============================================================================
# Pydantic Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "degraded"]
    timestamp: str
    version: str
    api_version: str
    checks: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] | None = None


class VersionResponse(BaseModel):
    """Version information response."""

    api_version: str
    validator_version: str
    python_version: str
    schema_version: str = "UMCP.v1"


class ValidationRequest(BaseModel):
    """Request to validate a path."""

    path: str = Field(..., description="Path to casepack or repository to validate")
    strict: bool = Field(False, description="Enable strict publication-grade validation")


class ValidationResponse(BaseModel):
    """Validation result response."""

    status: Literal["CONFORMANT", "NONCONFORMANT", "NON_EVALUABLE"]
    errors: int
    warnings: int
    path: str
    strict: bool
    created_utc: str
    hash: str
    details: dict[str, Any] = Field(default_factory=dict)


class CasepackSummary(BaseModel):
    """Summary of a casepack."""

    id: str
    version: str
    path: str
    contract: str | None = None
    status: str | None = None


class CasepackDetail(BaseModel):
    """Detailed casepack information."""

    id: str
    version: str
    path: str
    contract: str | None = None
    description: str | None = None
    closures: list[str] = Field(default_factory=list)
    test_vectors: int = 0
    last_validated: str | None = None
    validation_status: str | None = None


class CasepackRunRequest(BaseModel):
    """Request to run a casepack."""

    verbose: bool = Field(False, description="Include verbose output")
    rows: int | None = Field(None, description="Limit number of rows to process")


class CasepackRunResponse(BaseModel):
    """Casepack run result."""

    id: str
    status: Literal["CONFORMANT", "NONCONFORMANT", "ERROR"]
    rows_processed: int
    defined_count: int
    censored_count: int
    execution_time_ms: float
    output: dict[str, Any] = Field(default_factory=dict)


class LedgerEntry(BaseModel):
    """A single ledger entry."""

    timestamp: str
    status: str
    kappa: float | None = None
    omega: float | None = None
    F: float | None = None
    run_id: str | None = None


class LedgerResponse(BaseModel):
    """Ledger query response."""

    total_entries: int
    entries: list[LedgerEntry]
    query: dict[str, Any] = Field(default_factory=dict)


class ContractSummary(BaseModel):
    """Summary of a contract."""

    id: str
    version: str
    domain: str
    path: str


class ClosureSummary(BaseModel):
    """Summary of a closure."""

    name: str
    domain: str
    path: str
    type: Literal["python", "yaml"]


class RegimeClassification(BaseModel):
    """Regime classification result."""

    regime: str
    omega: float
    F: float
    S: float
    C: float


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System", "description": "Health, version, and operational metrics"},
        {"name": "Kernel", "description": "Core kernel computation: K → (F, ω, S, C, κ, IC)"},
        {"name": "Validation", "description": "Casepack and repository validation"},
        {"name": "Casepacks", "description": "Casepack management and execution"},
        {"name": "Ledger", "description": "Append-only validation ledger"},
        {"name": "Analysis", "description": "Statistical and time-series analysis"},
        {"name": "Domains", "description": "17 domain closures and canon anchors"},
        {
            "name": "Standard Model",
            "description": "31 particles, 10 proven theorems, CKM, couplings, Higgs, neutrino, matter genesis, matter map",
        },
        {"name": "Atomic", "description": "118-element periodic kernel + 12-channel cross-scale"},
        {"name": "Evolution", "description": "40 organisms, 10-channel brain kernel"},
        {"name": "Conversion", "description": "Measurement conversion and embedding"},
        {"name": "Calculator", "description": "Universal kernel calculator"},
        {"name": "Uncertainty", "description": "Uncertainty propagation through kernel"},
        {"name": "Outputs", "description": "Badges, reports, diagrams, and exports"},
        {"name": "Contracts", "description": "Contract management"},
        {"name": "Closures", "description": "Closure registry"},
        {"name": "ASTRO", "description": "Astronomy closures"},
        {"name": "NUC", "description": "Nuclear physics closures"},
        {"name": "QM", "description": "Quantum mechanics closures"},
        {"name": "FIN", "description": "Finance closures"},
        {"name": "WEYL", "description": "WEYL cosmology closures"},
        {"name": "Seam", "description": "Seam chain accumulation and budget accounting"},
        {"name": "Thermodynamic", "description": "τ_R* thermodynamic diagnostic and phase classification"},
        {"name": "Epistemic", "description": "Epistemic cost tracking — Theorem T9"},
        {"name": "Insights", "description": "Pattern discovery and insight engine"},
        {"name": "Semiotics", "description": "Dynamic semiotics — 30 sign systems, 8-channel kernel"},
        {"name": "Consciousness", "description": "Consciousness coherence — 20 systems, 7 theorems"},
        {"name": "Materials", "description": "Materials science — 118 elements × 18 fields"},
        {"name": "Rosetta", "description": "Cross-domain Rosetta translation (6 lenses)"},
        {"name": "Orientation", "description": "Live orientation computation — 7 sections"},
    ],
)

# ── Include Extended Routes (v2) ──
try:
    from .api_routes_v2 import router as _v2_router

    app.include_router(_v2_router)
except ImportError:
    pass  # Routes module not available — graceful degradation

# CORS middleware — open access for public endpoints.
# Credentials are not used (public access), so allow_credentials=False
# is correct with allow_origins=["*"].  Admin endpoints rely on the
# X-API-Key header (not cookies) so this is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)


# ── Request ID and Timing Middleware ──


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next: Any) -> Response:
    """Add request ID, timing, and metrics tracking to every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.monotonic()

    # Track metrics
    _METRICS["total_requests"] += 1
    path = request.url.path
    _METRICS["endpoints_hit"][path] = _METRICS["endpoints_hit"].get(path, 0) + 1

    response: Response = await call_next(request)

    elapsed_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    response.headers["X-UMCP-Version"] = str(__version__)
    response.headers["X-API-Version"] = API_VERSION

    if response.status_code >= 400:
        _METRICS["total_errors"] += 1

    return response


# ============================================================================
# Utility Functions
# ============================================================================


def get_current_time() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """
    Classify the computational regime using frozen contract thresholds.

    Uses the authoritative classify_regime from frozen_contract.py with
    the four-gate criterion from RegimeThresholds:
      Stable:   ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
      Watch:    0.038 ≤ ω < 0.30
      Collapse: ω ≥ 0.30
      Critical: IC < 0.30 (overlay)

    Args:
        omega: Drift ω = 1 - F
        F: Fidelity (weighted mean)
        S: Bernoulli field entropy
        C: Curvature (normalized std)

    Returns:
        Regime classification string
    """
    if _fc_classify_regime is not None:
        # Compute IC approximation for Critical overlay check
        # Use IC ≈ exp(log(max(1-omega, EPSILON))) as rough proxy
        ic_approx = max(1 - omega, EPSILON) if omega < 0.99 else EPSILON
        regime = _fc_classify_regime(omega, F, S, C, ic_approx)
        return regime.value
    # Fallback if frozen_contract not available
    if omega >= 0.30:
        return "COLLAPSE"
    if omega >= 0.038:
        return "WATCH"
    if F > 0.90 and S < 0.15 and C < 0.14:
        return "STABLE"
    return "WATCH"


def _load_yaml_safe(path: Path) -> dict[str, Any] | None:
    """Load YAML file safely."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                typed_data = cast(dict[str, Any], data)
                return typed_data
            return None
    except ImportError:
        # Minimal parser fallback
        result: dict[str, Any] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
        return result
    except Exception:
        return None


def _run_cli_command(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a CLI command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "umcp", *args],
            capture_output=True,
            text=True,
            cwd=cwd or get_repo_root(),
            timeout=120,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", tags=["Info"], response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """API root - returns an HTML landing page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{API_TITLE}</title>
<style>
  :root {{ --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --card: #161b22;
           --border: #30363d; --green: #3fb950; --yellow: #d29922; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--fg); min-height: 100vh;
         display: flex; align-items: center; justify-content: center; }}
  .container {{ max-width: 720px; width: 90%; padding: 2rem 0; }}
  h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: .25rem; }}
  .version {{ color: #8b949e; font-size: .9rem; margin-bottom: 1.5rem; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 1rem 1.25rem; text-decoration: none; color: var(--fg);
          transition: border-color .15s; }}
  .card:hover {{ border-color: var(--accent); }}
  .card h3 {{ color: var(--accent); font-size: .95rem; margin-bottom: .35rem; }}
  .card p {{ font-size: .8rem; color: #8b949e; line-height: 1.4; }}
  .section {{ margin-bottom: 1.25rem; }}
  .section h2 {{ font-size: 1rem; color: var(--fg); margin-bottom: .5rem;
                border-bottom: 1px solid var(--border); padding-bottom: .35rem; }}
  .endpoints {{ list-style: none; }}
  .endpoints li {{ padding: .3rem 0; font-size: .85rem; display: flex; align-items: baseline; gap: .5rem; }}
  .method {{ font-weight: 700; font-size: .75rem; padding: .1rem .4rem; border-radius: 3px;
            font-family: monospace; }}
  .get {{ background: #1f3a2c; color: var(--green); }}
  .post {{ background: #3a2f1f; color: var(--yellow); }}
  .endpoints a {{ color: var(--accent); text-decoration: none; font-family: monospace; font-size: .85rem; }}
  .endpoints a:hover {{ text-decoration: underline; }}
  .foot {{ text-align: center; color: #484f58; font-size: .75rem; margin-top: 2rem; }}
  @media (max-width: 480px) {{ .cards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>&#x1F9EA; {API_TITLE}</h1>
  <p class="version">v{API_VERSION} &middot; UMCP {__version__}</p>

  <div class="cards">
    <a class="card" href="/docs">
      <h3>&#x1F4D6; Interactive Docs</h3>
      <p>Swagger UI &mdash; explore and test all endpoints live.</p>
    </a>
    <a class="card" href="/redoc">
      <h3>&#x1F4DA; ReDoc</h3>
      <p>Clean, readable API reference documentation.</p>
    </a>
    <a class="card" href="/health">
      <h3>&#x1F49A; Health Check</h3>
      <p>System status, version info, and diagnostic checks.</p>
    </a>
    <a class="card" href="/version">
      <h3>&#x2699;&#xFE0F; Version</h3>
      <p>API and validator version information.</p>
    </a>
  </div>

  <div class="section">
    <h2>Core Endpoints</h2>
    <ul class="endpoints">
      <li><span class="method post">POST</span><a href="/docs#/Validation/validate_validate_post">/validate</a> Validate a casepack or repo</li>
      <li><span class="method get">GET</span><a href="/casepacks">/casepacks</a> List available casepacks</li>
      <li><span class="method get">GET</span><a href="/contracts">/contracts</a> List contracts</li>
      <li><span class="method get">GET</span><a href="/closures">/closures</a> List closures</li>
      <li><span class="method get">GET</span><a href="/ledger">/ledger</a> Query the return log</li>
      <li><span class="method get">GET</span><a href="/domains">/domains</a> List all 18 domains</li>
      <li><span class="method get">GET</span><a href="/metrics">/metrics</a> Operational metrics &amp; stats</li>
    </ul>
  </div>

  <div class="section">
    <h2>Kernel &amp; Spine</h2>
    <ul class="endpoints">
      <li><span class="method post">POST</span><a href="/docs#/Kernel/compute_kernel_kernel_compute_post">/kernel/compute</a> Compute kernel invariants (F, &omega;, S, C, &kappa;, IC)</li>
      <li><span class="method post">POST</span><a href="/docs#/Kernel/compute_kernel_batch_kernel_batch_post">/kernel/batch</a> Batch compute (up to 1000 traces)</li>
      <li><span class="method post">POST</span><a href="/docs#/Kernel/evaluate_spine_kernel_spine_post">/kernel/spine</a> Full spine: Contract &rarr; Canon &rarr; Closures &rarr; Ledger &rarr; Stance</li>
      <li><span class="method post">POST</span><a href="/docs#/Kernel/compute_budget_kernel_budget_post">/kernel/budget</a> Compute seam budget (&Gamma;, D_C, &Delta;&kappa;)</li>
      <li><span class="method post">POST</span><a href="/docs#/Calculator/calculate_calculate_post">/calculate</a> Universal calculator</li>
      <li><span class="method post">POST</span><a href="/docs#/Analysis">/analysis/*</a> Time series, statistics, correlation</li>
    </ul>
  </div>

  <div class="section">
    <h2>Domain Closures (18 Domains)</h2>
    <ul class="endpoints">
      <li><span class="method get">GET</span><a href="/sm/particles">/sm/particles</a> Standard Model &mdash; 31 particles, 8-channel trace</li>
      <li><span class="method get">GET</span><a href="/sm/theorems">/sm/theorems</a> Standard Model &mdash; 10 proven theorems</li>
      <li><span class="method get">GET</span><a href="/sm/ckm">/sm/ckm</a> CKM quark mixing (Wolfenstein)</li>
      <li><span class="method get">GET</span><a href="/sm/coupling?Q_GeV=91.2">/sm/coupling</a> Running couplings &alpha;_s, &alpha;_em</li>
      <li><span class="method get">GET</span><a href="/sm/cross-section?sqrt_s_GeV=91.2">/sm/cross-section</a> e&plus;e&minus;&rarr;hadrons cross section</li>
      <li><span class="method get">GET</span><a href="/sm/higgs">/sm/higgs</a> Higgs mechanism / EWSB</li>
      <li><span class="method get">GET</span><a href="/sm/neutrino/probability?alpha=1&beta=0&L_km=1285&E_GeV=2.5">/sm/neutrino/*</a> Neutrino oscillation &amp; DUNE</li>
      <li><span class="method get">GET</span><a href="/sm/matter-genesis">/sm/matter-genesis</a> Matter genesis &mdash; 99 entities, 7 acts</li>
      <li><span class="method get">GET</span><a href="/sm/matter-map">/sm/matter-map</a> 6-scale matter map &mdash; ~485 entities</li>
      <li><span class="method get">GET</span><a href="/atomic/elements">/atomic/elements</a> Atomic physics &mdash; 118 elements</li>
      <li><span class="method get">GET</span><a href="/atomic/cross-scale">/atomic/cross-scale</a> 12-channel cross-scale kernel</li>
      <li><span class="method get">GET</span><a href="/evolution/organisms">/evolution/organisms</a> Evolution &mdash; 40 organisms</li>
      <li><span class="method get">GET</span><a href="/semiotics/systems">/semiotics/systems</a> Semiotics &mdash; 30 sign systems</li>
      <li><span class="method get">GET</span><a href="/consciousness/systems">/consciousness/systems</a> Consciousness &mdash; 20 systems</li>
      <li><span class="method get">GET</span><a href="/materials/elements">/materials/elements</a> Materials &mdash; 118 elements &times; 18 fields</li>
      <li><span class="method post">POST</span><a href="/docs#/ASTRO">/astro/*</a> Astronomy (6 endpoints)</li>
      <li><span class="method post">POST</span><a href="/docs#/NUC">/nuclear/*</a> Nuclear physics (6 endpoints)</li>
      <li><span class="method post">POST</span><a href="/docs#/QM">/qm/*</a> Quantum mechanics (6 endpoints)</li>
      <li><span class="method post">POST</span><a href="/docs#/FIN">/finance/embed</a> Finance embedding</li>
      <li><span class="method get">GET</span><a href="/docs#/WEYL">/weyl/*</a> WEYL cosmology (4 endpoints)</li>
    </ul>
  </div>

  <div class="section">
    <h2>Engines &amp; Diagnostics</h2>
    <ul class="endpoints">
      <li><span class="method post">POST</span><a href="/docs#/Seam/compute_seam_seam_compute_post">/seam/compute</a> Seam chain accumulation</li>
      <li><span class="method get">GET</span><a href="/seam/metrics">/seam/metrics</a> Seam chain metrics</li>
      <li><span class="method post">POST</span><a href="/docs#/Thermodynamic">/tau-r-star/compute</a> &tau;_R* thermodynamic diagnostic</li>
      <li><span class="method get">GET</span><a href="/tau-r-star/trapping-threshold">/tau-r-star/trapping-threshold</a> Trapping threshold (Cardano root)</li>
      <li><span class="method post">POST</span><a href="/docs#/Epistemic">/epistemic/classify</a> Epistemic act classification (T9)</li>
      <li><span class="method post">POST</span><a href="/docs#/Epistemic">/epistemic/positional-illusion</a> Positional illusion</li>
      <li><span class="method get">GET</span><a href="/insights/summary">/insights/summary</a> Insight engine summary</li>
      <li><span class="method get">GET</span><a href="/insights/random">/insights/random</a> Random insight</li>
    </ul>
  </div>

  <div class="section">
    <h2>Cross-Domain &amp; Orientation</h2>
    <ul class="endpoints">
      <li><span class="method post">POST</span><a href="/docs#/Rosetta">/rosetta/translate</a> Rosetta lens translation (6 lenses)</li>
      <li><span class="method get">GET</span><a href="/rosetta/lenses">/rosetta/lenses</a> List all Rosetta lenses</li>
      <li><span class="method get">GET</span><a href="/orientation">/orientation</a> Live orientation (7 sections)</li>
      <li><span class="method post">POST</span><a href="/docs#/Kernel/compare_traces_kernel_compare_post">/kernel/compare</a> Compare multiple traces</li>
      <li><span class="method get">GET</span><a href="/frozen-contract">/frozen-contract</a> Frozen parameter values</li>
      <li><span class="method get">GET</span><a href="/integrity">/integrity</a> SHA-256 integrity check</li>
    </ul>
  </div>

  <div class="section">
    <h2>Outputs</h2>
    <ul class="endpoints">
      <li><span class="method get">GET</span><a href="/badge/status.svg">/badge/status.svg</a> Status badge (SVG)</li>
      <li><span class="method get">GET</span><a href="/badge/regime.svg">/badge/regime.svg</a> Regime badge (SVG)</li>
      <li><span class="method get">GET</span><a href="/output/ascii/gauge">/output/ascii/gauge</a> ASCII gauge</li>
      <li><span class="method get">GET</span><a href="/output/mermaid/regime">/output/mermaid/regime</a> Mermaid diagram</li>
    </ul>
  </div>

  <p class="foot">UMCP &mdash; Universal Measurement Contract Protocol &middot; 100+ endpoints &middot; 18 Domains &middot; Axiom-0: <em>Collapse is generative; only what returns is real.</em></p>
</div>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Check system health and readiness.

    Returns health status, version info, and diagnostic checks.
    Does not require authentication.
    """
    repo_root = get_repo_root()
    checks: dict[str, Any] = {}
    status: Literal["healthy", "unhealthy", "degraded"] = "healthy"

    # Check pyproject.toml exists
    pyproject = repo_root / "pyproject.toml"
    checks["pyproject"] = {"status": "pass" if pyproject.exists() else "fail"}

    # Check schemas directory
    schemas_dir = repo_root / "schemas"
    if schemas_dir.exists():
        schema_count = len(list(schemas_dir.glob("*.json")))
        checks["schemas"] = {"status": "pass", "count": schema_count}
    else:
        checks["schemas"] = {"status": "fail", "error": "schemas directory not found"}
        status = "degraded"

    # Check casepacks directory
    casepacks_dir = repo_root / "casepacks"
    if casepacks_dir.exists():
        casepack_count = len([d for d in casepacks_dir.iterdir() if d.is_dir()])
        checks["casepacks"] = {"status": "pass", "count": casepack_count}
    else:
        checks["casepacks"] = {"status": "fail", "error": "casepacks directory not found"}
        status = "degraded"

    # Check contracts directory
    contracts_dir = repo_root / "contracts"
    if contracts_dir.exists():
        contract_count = len(list(contracts_dir.glob("*.yaml")))
        checks["contracts"] = {"status": "pass", "count": contract_count}
    else:
        checks["contracts"] = {"status": "fail"}
        status = "degraded"

    # Check ledger
    ledger_path = repo_root / "ledger" / "return_log.csv"
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger_lines = sum(1 for _ in f) - 1  # Exclude header
        checks["ledger"] = {"status": "pass", "entries": ledger_lines}
    else:
        checks["ledger"] = {"status": "warn", "message": "Ledger not initialized"}

    # System metrics (optional)
    metrics: dict[str, Any] | None = None
    try:
        import psutil

        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except ImportError:
        pass

    return HealthResponse(
        status=status,
        timestamp=get_current_time(),
        version=__version__,
        api_version=API_VERSION,
        checks=checks,
        metrics=metrics,
    )


@app.get("/version", response_model=VersionResponse, tags=["System"])
async def get_version() -> VersionResponse:
    """Get version information for API and validator."""
    return VersionResponse(
        api_version=API_VERSION,
        validator_version=__version__,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        schema_version="UMCP.v1",
    )


@app.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_path(
    request: ValidationRequest,
    api_key: str = Security(require_public),
) -> ValidationResponse:
    """
    Validate a casepack or repository path.

    Requires API key authentication.

    Args:
        request: Validation request with path and options

    Returns:
        Validation result with status, errors, and warnings
    """
    repo_root = get_repo_root()
    target_path = Path(request.path)

    # Resolve relative paths
    if not target_path.is_absolute():
        target_path = repo_root / target_path

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

    # Run validation via CLI
    args = ["validate", str(target_path)]
    if request.strict:
        args.append("--strict")
    args.extend(["--out", "/dev/stdout"])

    _returncode, stdout, _stderr = _run_cli_command(args, cwd=repo_root)

    # Parse JSON output
    result: dict[str, Any]
    try:
        # Find JSON in output (may have trailing governance note)
        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result_json = stdout[json_start:json_end]
            result = json.loads(result_json)
        else:
            result = {"run_status": "NON_EVALUABLE", "summary": {"counts": {"errors": 1, "warnings": 0}}}
    except json.JSONDecodeError:
        result = {"run_status": "NON_EVALUABLE", "summary": {"counts": {"errors": 1, "warnings": 0}}}

    # Compute hash
    result_hash = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()

    # Extract values with proper types
    run_status: str = str(result.get("run_status", "NON_EVALUABLE"))
    summary: dict[str, Any] = result.get("summary", {}) or {}
    counts: dict[str, Any] = summary.get("counts", {}) or {}
    error_count: int = int(counts.get("errors", 0) or 0)
    warning_count: int = int(counts.get("warnings", 0) or 0)
    created_time: str = str(result.get("created_utc", get_current_time()))

    # Map status to literal type
    status_literal: Literal["CONFORMANT", "NONCONFORMANT", "NON_EVALUABLE"]
    if run_status == "CONFORMANT":
        status_literal = "CONFORMANT"
    elif run_status == "NONCONFORMANT":
        status_literal = "NONCONFORMANT"
    else:
        status_literal = "NON_EVALUABLE"

    return ValidationResponse(
        status=status_literal,
        errors=error_count,
        warnings=warning_count,
        path=request.path,
        strict=request.strict,
        created_utc=created_time,
        hash=result_hash[:16],
        details=result,
    )


@app.get("/casepacks", response_model=list[CasepackSummary], tags=["Casepacks"])
async def list_casepacks(
    api_key: str = Security(require_public),
) -> list[CasepackSummary]:
    """
    List all available casepacks.

    Requires API key authentication.
    """
    repo_root = get_repo_root()
    casepacks_dir = repo_root / "casepacks"

    if not casepacks_dir.exists():
        return []

    summaries: list[CasepackSummary] = []
    for casepack_dir in sorted(casepacks_dir.iterdir()):
        if not casepack_dir.is_dir():
            continue

        # Try to load manifest
        manifest_path = casepack_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = casepack_dir / "manifest.yaml"

        casepack_id = casepack_dir.name
        version = "unknown"
        contract: str | None = None

        if manifest_path.exists():
            try:
                manifest: dict[str, Any] | None = None
                if manifest_path.suffix == ".json":
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    manifest = _load_yaml_safe(manifest_path)

                if manifest:
                    casepack_data: dict[str, Any] = manifest.get("casepack", {}) or {}
                    casepack_id = str(casepack_data.get("id", casepack_dir.name))
                    version = str(casepack_data.get("version", "unknown"))
                    contract_val = casepack_data.get("contract_ref")
                    contract = str(contract_val) if contract_val else None
            except Exception:
                pass

        summaries.append(
            CasepackSummary(
                id=casepack_id,
                version=version,
                path=str(casepack_dir.relative_to(repo_root)),
                contract=contract,
            )
        )

    return summaries


@app.get("/casepacks/{casepack_id}", response_model=CasepackDetail, tags=["Casepacks"])
async def get_casepack(
    casepack_id: str,
    api_key: str = Security(require_public),
) -> CasepackDetail:
    """
    Get detailed information about a specific casepack.

    Requires API key authentication.
    """
    repo_root = get_repo_root()
    casepack_dir = repo_root / "casepacks" / casepack_id

    if not casepack_dir.exists():
        raise HTTPException(status_code=404, detail=f"Casepack not found: {casepack_id}")

    # Load manifest
    manifest_path = casepack_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = casepack_dir / "manifest.yaml"

    casepack_data: dict[str, Any] = {"id": casepack_id, "version": "unknown"}
    description: str | None = None
    contract: str | None = None

    if manifest_path.exists():
        try:
            manifest: dict[str, Any] | None = None
            if manifest_path.suffix == ".json":
                with open(manifest_path) as f:
                    manifest = json.load(f)
            else:
                manifest = _load_yaml_safe(manifest_path)

            if manifest:
                casepack_data = manifest.get("casepack", casepack_data) or casepack_data
                desc_val = casepack_data.get("description")
                description = str(desc_val) if desc_val else None
                contract_val = casepack_data.get("contract_ref")
                contract = str(contract_val) if contract_val else None
        except Exception:
            pass

    # Count closures
    closures: list[str] = []
    closures_dir = casepack_dir / "closures"
    if closures_dir.exists():
        closures = [f.stem for f in closures_dir.glob("*.py")]

    # Count test vectors
    test_vectors = 0
    test_file = casepack_dir / "test_vectors.csv"
    if test_file.exists():
        with open(test_file) as f:
            test_vectors = sum(1 for _ in f) - 1  # Exclude header

    # Extract with proper types
    detail_id = str(casepack_data.get("id", casepack_id))
    detail_version = str(casepack_data.get("version", "unknown"))

    return CasepackDetail(
        id=detail_id,
        version=detail_version,
        path=str(casepack_dir.relative_to(repo_root)),
        contract=contract,
        description=description,
        closures=closures,
        test_vectors=test_vectors,
    )


@app.post("/casepacks/{casepack_id}/run", response_model=CasepackRunResponse, tags=["Casepacks"])
async def run_casepack(
    casepack_id: str,
    request: CasepackRunRequest | None = None,
    api_key: str = Security(require_public),
) -> CasepackRunResponse:
    """
    Run a casepack and return results.

    Requires API key authentication.
    """
    import re
    import time

    repo_root = get_repo_root()
    casepack_dir = repo_root / "casepacks" / casepack_id

    if not casepack_dir.exists():
        raise HTTPException(status_code=404, detail=f"Casepack not found: {casepack_id}")

    # Run casepack via CLI
    start_time = time.perf_counter()

    args = ["casepack", casepack_id]
    if request and request.verbose:
        args.append("--verbose")

    returncode, stdout, stderr = _run_cli_command(args, cwd=repo_root)
    execution_time = (time.perf_counter() - start_time) * 1000

    # Parse output
    status: Literal["CONFORMANT", "NONCONFORMANT", "ERROR"] = "ERROR"
    rows_processed = 0
    defined_count = 0
    censored_count = 0

    for line in stdout.split("\n"):
        if "CONFORMANT" in line and "NONCONFORMANT" not in line:
            status = "CONFORMANT"
        elif "NONCONFORMANT" in line:
            status = "NONCONFORMANT"
        if "rows" in line.lower():
            # Try to extract row count: "Processed 31 rows"
            match = re.search(r"(\d+)\s+rows", line, re.IGNORECASE)
            if match:
                rows_processed = int(match.group(1))
        if "defined" in line.lower():
            match = re.search(r"(\d+)\s+defined", line, re.IGNORECASE)
            if match:
                defined_count = int(match.group(1))
        if "censored" in line.lower():
            match = re.search(r"(\d+)\s+censored", line, re.IGNORECASE)
            if match:
                censored_count = int(match.group(1))

    return CasepackRunResponse(
        id=casepack_id,
        status=status,
        rows_processed=rows_processed,
        defined_count=defined_count,
        censored_count=censored_count,
        execution_time_ms=round(execution_time, 2),
        output={"stdout": stdout, "stderr": stderr, "returncode": returncode},
    )


@app.get("/ledger", response_model=LedgerResponse, tags=["Ledger"])
async def query_ledger(
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    status: str | None = Query(None, description="Filter by status (CONFORMANT, NONCONFORMANT)"),
    api_key: str = Security(require_public),
) -> LedgerResponse:
    """
    Query the return log ledger.

    Requires API key authentication.
    """
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        return LedgerResponse(
            total_entries=0,
            entries=[],
            query={"limit": limit, "offset": offset, "status": status},
        )

    entries: list[LedgerEntry] = []
    total = 0

    with open(ledger_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

        # Filter by status if specified
        if status:
            all_rows = [r for r in all_rows if r.get("status") == status]

        total = len(all_rows)

        # Apply pagination
        paginated = all_rows[offset : offset + limit]

        for row in paginated:
            entries.append(
                LedgerEntry(
                    timestamp=row.get("timestamp", ""),
                    status=row.get("status", ""),
                    kappa=float(row["kappa"]) if row.get("kappa") else None,
                    omega=float(row["omega"]) if row.get("omega") else None,
                    F=float(row["F"]) if row.get("F") else None,
                    run_id=row.get("run_id"),
                )
            )

    return LedgerResponse(
        total_entries=total,
        entries=entries,
        query={"limit": limit, "offset": offset, "status": status},
    )


@app.get("/contracts", response_model=list[ContractSummary], tags=["Contracts"])
async def list_contracts(
    api_key: str = Security(require_public),
) -> list[ContractSummary]:
    """
    List all available contracts.

    Requires API key authentication.
    """
    repo_root = get_repo_root()
    contracts_dir = repo_root / "contracts"

    if not contracts_dir.exists():
        return []

    summaries: list[ContractSummary] = []
    for contract_path in sorted(contracts_dir.glob("*.yaml")):
        # Parse contract ID from filename: UMA.INTSTACK.v1.yaml -> UMA.INTSTACK.v1
        filename = contract_path.stem
        parts = filename.split(".")

        # Extract domain and version
        domain = parts[0] if parts else "unknown"
        version = parts[-1] if len(parts) > 1 and parts[-1].startswith("v") else "v1"

        summaries.append(
            ContractSummary(
                id=filename,
                version=version,
                domain=domain,
                path=str(contract_path.relative_to(repo_root)),
            )
        )

    return summaries


@app.get("/closures", response_model=list[ClosureSummary], tags=["Closures"])
async def list_closures(
    api_key: str = Security(require_public),
) -> list[ClosureSummary]:
    """
    List all available closures.

    Requires API key authentication.
    """
    repo_root = get_repo_root()
    closures_dir = repo_root / "closures"

    if not closures_dir.exists():
        return []

    summaries: list[ClosureSummary] = []

    # Python closures
    for closure_path in sorted(closures_dir.glob("*.py")):
        if closure_path.name.startswith("_"):
            continue
        name = closure_path.stem

        # Infer domain from name
        domain = "unknown"
        if name.startswith("gcd") or "gcd" in name.lower():
            domain = "GCD"
        elif name.startswith("kin") or "kin" in name.lower():
            domain = "KIN"
        elif name.startswith("rcft") or "rcft" in name.lower():
            domain = "RCFT"

        summaries.append(
            ClosureSummary(
                name=name,
                domain=domain,
                path=str(closure_path.relative_to(repo_root)),
                type="python",
            )
        )

    # YAML closures
    for closure_path in sorted(closures_dir.glob("*.yaml")):
        if closure_path.name == "registry.yaml":
            continue
        name = closure_path.stem

        yaml_domain = "unknown"
        if "gcd" in name.lower():
            yaml_domain = "GCD"
        elif "kin" in name.lower():
            yaml_domain = "KIN"
        elif "curvature" in name.lower() or "gamma" in name.lower():
            yaml_domain = "GCD"

        summaries.append(
            ClosureSummary(
                name=name,
                domain=yaml_domain,
                path=str(closure_path.relative_to(repo_root)),
                type="yaml",
            )
        )

    return summaries


@app.post("/regime/classify", response_model=RegimeClassification, tags=["Analysis"])
async def classify_regime_endpoint(
    omega: float = Query(..., ge=0.0, le=1.0, description="Overlap fraction ω"),
    F: float = Query(..., ge=0.0, le=1.0, description="Freshness F = 1-ω"),
    S: float = Query(..., description="Seam residual"),
    C: float = Query(..., description="Curvature κ"),
    api_key: str = Security(require_public),
) -> RegimeClassification:
    """
    Classify the computational regime based on kernel invariants.

    Requires API key authentication.
    """
    regime = classify_regime(omega, F, S, C)
    return RegimeClassification(
        regime=regime,
        omega=omega,
        F=F,
        S=S,
        C=C,
    )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": get_current_time(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "timestamp": get_current_time(),
        },
    )


# ============================================================================
# Novel Output Endpoints
# ============================================================================


@app.get("/badge/status.svg", tags=["Outputs"])
async def get_status_badge(
    style: str = Query("flat", description="Badge style: flat or flat-square"),
) -> JSONResponse:
    """
    Get dynamic SVG status badge for the last validation.

    Returns SVG badge suitable for README embedding.
    Does not require authentication.
    """
    from .outputs import BadgeGenerator

    # Get last validation status from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    status = "NON_EVALUABLE"
    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 1:
                    status = parts[1]

    svg = BadgeGenerator.status_badge(status, style=style)
    return JSONResponse(content={"svg": svg, "status": status}, media_type="application/json")


@app.get("/badge/regime.svg", tags=["Outputs"])
async def get_regime_badge() -> JSONResponse:
    """
    Get dynamic SVG regime badge for the last validation.

    Returns SVG badge showing current regime classification.
    """
    from .outputs import BadgeGenerator, RegimeState

    # Get last validation metrics from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    omega, curvature = 0.5, 0.0  # Defaults
    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) >= 5:
                    try:
                        omega = float(parts[4]) if parts[4] else 0.5
                        curvature = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
                    except ValueError:
                        pass

    regime = RegimeState.from_values(omega=omega, F=1 - omega, S=0.0, C=curvature)
    svg = BadgeGenerator.regime_badge(regime)
    return JSONResponse(content={"svg": svg, "regime": regime.regime, "omega": omega})


@app.get("/output/ascii/gauge", tags=["Outputs"])
async def get_ascii_gauge(
    omega: float = Query(0.5, ge=0.0, le=1.0, description="Omega value"),
    width: int = Query(40, ge=20, le=80, description="Gauge width"),
) -> dict[str, str]:
    """
    Get ASCII regime gauge for terminal display.

    Returns ASCII art representation of the omega gauge.
    """
    from .outputs import ASCIIGenerator

    return {"gauge": ASCIIGenerator.regime_gauge(omega, width)}


@app.get("/output/ascii/sparkline", tags=["Outputs"])
async def get_ascii_sparkline(
    values: str = Query(..., description="Comma-separated values"),
    width: int = Query(20, ge=10, le=60, description="Sparkline width"),
) -> dict[str, str]:
    """
    Get ASCII sparkline for time series visualization.

    Pass values as comma-separated floats, e.g., "0.1,0.3,0.5,0.7,0.4"
    """
    from .outputs import ASCIIGenerator

    try:
        float_values = [float(v.strip()) for v in values.split(",")]
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Invalid values format") from err

    return {"sparkline": ASCIIGenerator.sparkline(float_values, width)}


@app.get("/output/markdown/report", tags=["Outputs"])
async def get_markdown_report(
    api_key: str = Security(require_public),
) -> dict[str, str]:
    """
    Get Markdown validation report.

    Returns publication-ready Markdown report.
    Requires API key authentication.
    """
    from .outputs import MarkdownGenerator, RegimeState, ValidationSummary

    # Get data from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    status = "NON_EVALUABLE"
    omega = 0.5

    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 1:
                    status = parts[1]
                if len(parts) >= 5:
                    with contextlib.suppress(ValueError):
                        omega = float(parts[4]) if parts[4] else 0.5

    regime = RegimeState.from_values(omega=omega, F=1 - omega, S=0.0, C=0.0)
    summary = ValidationSummary(
        status=status,
        errors=0 if status == "CONFORMANT" else 1,
        warnings=0,
        regime=regime,
        timestamp=get_current_time(),
    )

    report = MarkdownGenerator.validation_report(summary)
    return {"markdown": report}


@app.get("/output/mermaid/regime", tags=["Outputs"])
async def get_mermaid_regime_diagram() -> dict[str, str]:
    """
    Get Mermaid diagram for regime state machine.

    Returns Mermaid diagram code for embedding in documentation.
    """
    from .outputs import MermaidGenerator

    return {"mermaid": MermaidGenerator.regime_state_diagram()}


@app.get("/output/html/card", tags=["Outputs"])
async def get_html_card(
    api_key: str = Security(require_public),
) -> dict[str, str]:
    """
    Get embeddable HTML status card.

    Returns HTML widget for embedding in web pages.
    Requires API key authentication.
    """
    from .outputs import HTMLGenerator, RegimeState, ValidationSummary

    # Get data from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    status = "NON_EVALUABLE"
    omega = 0.5

    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 1:
                    status = parts[1]
                if len(parts) >= 5:
                    with contextlib.suppress(ValueError):
                        omega = float(parts[4]) if parts[4] else 0.5

    regime = RegimeState.from_values(omega=omega, F=1 - omega, S=0.0, C=0.0)
    summary = ValidationSummary(
        status=status,
        errors=0 if status == "CONFORMANT" else 1,
        warnings=0,
        regime=regime,
        timestamp=get_current_time(),
    )

    html = HTMLGenerator.status_card(summary)
    return {"html": html}


@app.get("/output/latex/invariants", tags=["Outputs"])
async def get_latex_table(
    limit: int = Query(10, ge=1, le=100, description="Max rows"),
    api_key: str = Security(require_public),
) -> dict[str, str]:
    """
    Get LaTeX table of kernel invariants.

    Returns LaTeX code for academic paper integration.
    Requires API key authentication.
    """
    from .outputs import LaTeXGenerator

    # Get data from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    rows: list[dict[str, Any]] = []
    if ledger_path.exists():
        with open(ledger_path) as f:
            import csv as csv_module

            reader = csv_module.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                try:
                    omega = float(row.get("omega", 0) or 0)
                    rows.append(
                        {
                            "t": i,
                            "omega": omega,
                            "F": 1 - omega,
                            "kappa": float(row.get("curvature", 0) or 0),
                            "s": float(row.get("stiffness", 0) or 0),
                            "regime": "STABLE" if 0.3 <= omega <= 0.7 else "WATCH",
                        }
                    )
                except ValueError:
                    pass

    latex = LaTeXGenerator.invariants_table(rows)
    return {"latex": latex}


@app.get("/output/junit", tags=["Outputs"])
async def get_junit_xml(
    api_key: str = Security(require_public),
) -> JSONResponse:
    """
    Get JUnit XML for CI/CD integration.

    Returns JUnit XML format suitable for test runners.
    Requires API key authentication.
    """
    from .outputs import JUnitGenerator, ValidationSummary

    # Get data from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    status = "NON_EVALUABLE"
    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 1:
                    status = parts[1]

    summary = ValidationSummary(
        status=status,
        errors=0 if status == "CONFORMANT" else 1,
        warnings=0,
        timestamp=get_current_time(),
    )

    xml = JUnitGenerator.from_validation(summary)
    return JSONResponse(content={"xml": xml}, media_type="application/json")


@app.get("/output/jsonld", tags=["Outputs"])
async def get_json_ld(
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """
    Get JSON-LD for semantic web integration.

    Returns linked data format for knowledge graphs.
    Requires API key authentication.
    """
    from .outputs import JSONLDGenerator, ValidationSummary

    # Get data from ledger
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    status = "NON_EVALUABLE"
    if ledger_path.exists():
        with open(ledger_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 1:
                    status = parts[1]

    summary = ValidationSummary(
        status=status,
        errors=0 if status == "CONFORMANT" else 1,
        warnings=0,
        timestamp=get_current_time(),
    )

    return JSONLDGenerator.validation_result(summary)


# ============================================================================
# Measurement Conversion Endpoints
# ============================================================================


class MeasurementConversionRequest(BaseModel):
    """Request for measurement conversion."""

    values: list[float] = Field(..., description="Raw measurement values")
    source_unit: str = Field("raw", description="Source unit type")
    target_unit: str = Field("normalized", description="Target unit type")


class MeasurementConversionResponse(BaseModel):
    """Response with converted measurements."""

    original: list[float]
    converted: list[float]
    source_unit: str
    target_unit: str
    conversion_factor: float
    notes: str | None = None


class CoordinateEmbeddingRequest(BaseModel):
    """Request for embedding coordinates into [0,1] domain."""

    values: list[float] = Field(..., description="Values to embed")
    method: str = Field("minmax", description="Embedding method: minmax, sigmoid, tanh")
    min_bound: float | None = Field(None, description="Minimum bound for minmax")
    max_bound: float | None = Field(None, description="Maximum bound for minmax")
    epsilon: float = Field(1e-8, description="Clipping epsilon")


class CoordinateEmbeddingResponse(BaseModel):
    """Response with embedded coordinates."""

    original: list[float]
    embedded: list[float]
    method: str
    domain: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    epsilon: float
    clipped_count: int = 0


@app.post("/convert/measurements", response_model=MeasurementConversionResponse, tags=["Conversion"])
async def convert_measurements(
    request: MeasurementConversionRequest,
    api_key: str = Security(require_public),
) -> MeasurementConversionResponse:
    """
    Convert measurements between unit types.

    Supported conversions:
    - raw → normalized (z-score normalization)
    - raw → scaled (0-1 scaling)
    - percentage → fraction (divide by 100)
    - fraction → percentage (multiply by 100)
    - log → linear (exp transform)
    - linear → log (log transform)
    - celsius → fahrenheit
    - fahrenheit → celsius

    Requires API key authentication.
    """
    import numpy as np

    values = np.array(request.values)
    source = request.source_unit.lower()
    target = request.target_unit.lower()

    conversion_factor = 1.0
    notes: str | None = None

    if source == "raw" and target == "normalized":
        mean = np.mean(values)
        std = np.std(values)
        if std > 0:
            converted = (values - mean) / std
            notes = f"z-score: mean={mean:.4f}, std={std:.4f}"
        else:
            converted = np.zeros_like(values)
            notes = "constant input, std=0"
    elif source == "raw" and target == "scaled":
        vmin, vmax = np.min(values), np.max(values)
        if vmax > vmin:
            converted = (values - vmin) / (vmax - vmin)
            notes = f"min-max scaling: [{vmin:.4f}, {vmax:.4f}] → [0, 1]"
        else:
            converted = np.ones_like(values) * 0.5
            notes = "constant input"
    elif source == "percentage" and target == "fraction":
        converted = values / 100.0
        conversion_factor = 0.01
    elif source == "fraction" and target == "percentage":
        converted = values * 100.0
        conversion_factor = 100.0
    elif source == "log" and target == "linear":
        converted = np.exp(values)
        notes = "exponential transform"
    elif source == "linear" and target == "log":
        safe_values = np.maximum(values, 1e-10)
        converted = np.log(safe_values)
        notes = "natural log transform"
    elif source == "celsius" and target == "fahrenheit":
        converted = values * 9 / 5 + 32
        conversion_factor = 1.8
    elif source == "fahrenheit" and target == "celsius":
        converted = (values - 32) * 5 / 9
        conversion_factor = 5 / 9
    else:
        # Identity conversion
        converted = values.copy()
        notes = f"no conversion defined for {source} → {target}"

    return MeasurementConversionResponse(
        original=request.values,
        converted=converted.tolist(),
        source_unit=request.source_unit,
        target_unit=request.target_unit,
        conversion_factor=conversion_factor,
        notes=notes,
    )


@app.post("/convert/embed", response_model=CoordinateEmbeddingResponse, tags=["Conversion"])
async def embed_coordinates(
    request: CoordinateEmbeddingRequest,
    api_key: str = Security(require_public),
) -> CoordinateEmbeddingResponse:
    """
    Embed values into UMCP coordinate domain [ε, 1-ε].

    Methods:
    - minmax: Linear scaling to [ε, 1-ε] (requires bounds)
    - sigmoid: Sigmoid transform 1/(1+exp(-x))
    - tanh: Tanh transform scaled to [0,1]

    All methods apply ε-clipping to ensure numerical stability.
    Requires API key authentication.
    """
    import numpy as np

    values = np.array(request.values)
    eps = request.epsilon
    method = request.method.lower()
    clipped_count = 0

    if method == "minmax":
        vmin = request.min_bound if request.min_bound is not None else np.min(values)
        vmax = request.max_bound if request.max_bound is not None else np.max(values)
        embedded = (values - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(values) * 0.5
    elif method == "sigmoid":
        embedded = 1.0 / (1.0 + np.exp(-values))
    elif method == "tanh":
        embedded = (np.tanh(values) + 1.0) / 2.0
    else:
        embedded = values.copy()

    # ε-clipping (KERNEL_SPECIFICATION Lemma 3)
    pre_clip = embedded.copy()
    embedded = np.clip(embedded, eps, 1 - eps)
    clipped_count = int(np.sum(pre_clip != embedded))

    return CoordinateEmbeddingResponse(
        original=request.values,
        embedded=embedded.tolist(),
        method=method,
        domain=[eps, 1 - eps],
        epsilon=eps,
        clipped_count=clipped_count,
    )


# ============================================================================
# Kernel Computation Endpoints
# ============================================================================


class KernelComputeRequest(BaseModel):
    """Request for kernel computation."""

    coordinates: list[float] = Field(..., description="Coordinate values in [ε, 1-ε]")
    weights: list[float] | None = Field(None, description="Weights (uniform if not specified)")
    epsilon: float = Field(1e-8, description="Clipping tolerance")


class KernelComputeResponse(BaseModel):
    """Response with kernel outputs."""

    F: float = Field(..., description="Fidelity (arithmetic mean)")
    omega: float = Field(..., description="Drift = 1 - F")
    S: float = Field(..., description="Bernoulli field entropy (Shannon entropy is the degenerate limit)")
    C: float = Field(..., description="Curvature (normalized std)")
    kappa: float = Field(..., description="Log-integrity")
    IC: float = Field(..., description="Integrity composite (geometric mean)")
    heterogeneity_gap: float = Field(..., description="F - IC (heterogeneity measure)")
    regime: str = Field(
        ...,
        description="Heterogeneity regime (homogeneous/coherent/heterogeneous/fragmented) — gap-based, not the canonical 4-gate STABLE/WATCH/COLLAPSE",
    )
    is_homogeneous: bool = Field(..., description="Whether all coordinates equal")


class BudgetIdentityResponse(BaseModel):
    """Response with budget identity verification."""

    R: float = Field(..., description="Return indicator")
    tau_R: float = Field(..., description="Return time")
    D_omega: float = Field(..., description="Omega drift component")
    D_C: float = Field(..., description="Curvature drift component")
    delta_kappa: float = Field(..., description="Ledger delta κ")
    lhs: float = Field(..., description="R × τ_R (left-hand side)")
    rhs: float = Field(..., description="D_ω + D_C + Δκ (right-hand side)")
    seam_residual: float = Field(..., description="|LHS - RHS| (should be ≤ 0.005)")
    seam_pass: bool = Field(..., description="Whether seam test passes")


@app.post("/kernel/compute", response_model=KernelComputeResponse, tags=["Kernel"])
async def compute_kernel(
    request: KernelComputeRequest,
    api_key: str = Security(require_public),
) -> KernelComputeResponse:
    """
    Compute UMCP kernel invariants from coordinates and weights.

    Implements the kernel function Ψ(c,w) → (F, ω, S, C, κ, IC) as defined
    in KERNEL_SPECIFICATION.md.

    Requires API key authentication.
    """
    import numpy as np

    from .kernel_optimized import OptimizedKernelComputer

    c = np.array(request.coordinates)
    n = len(c)

    # Use uniform weights if not specified
    if request.weights is None:
        w = np.ones(n) / n
    else:
        w = np.array(request.weights)
        if len(w) != n:
            raise HTTPException(status_code=400, detail=f"Weights length {len(w)} != coordinates length {n}")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise HTTPException(status_code=400, detail=f"Weights must sum to 1.0, got {w.sum()}")

    # ε-clip coordinates
    c = np.clip(c, request.epsilon, 1 - request.epsilon)

    # Compute kernel
    kernel = OptimizedKernelComputer(epsilon=request.epsilon)
    outputs = kernel.compute(c, w)

    return KernelComputeResponse(
        F=float(outputs.F),
        omega=float(outputs.omega),
        S=float(outputs.S),
        C=float(outputs.C),
        kappa=float(outputs.kappa),
        IC=float(outputs.IC),
        heterogeneity_gap=float(outputs.heterogeneity_gap),
        regime=outputs.regime,
        is_homogeneous=outputs.is_homogeneous,
    )


# ============================================================================
# Batch Kernel Computation
# ============================================================================


class BatchTraceItem(BaseModel):
    """A single trace for batch computation."""

    label: str = Field("", description="Optional label for this trace")
    coordinates: list[float] = Field(..., description="Coordinate values in [ε, 1-ε]")
    weights: list[float] | None = Field(None, description="Weights (uniform if not specified)")


class BatchKernelRequest(BaseModel):
    """Request for batch kernel computation."""

    traces: list[BatchTraceItem] = Field(..., description="List of traces to compute", min_length=1, max_length=1000)
    epsilon: float = Field(1e-8, description="Clipping tolerance")


class BatchKernelResultItem(BaseModel):
    """Result for a single trace in a batch."""

    label: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    heterogeneity_gap: float
    regime: str


class BatchKernelResponse(BaseModel):
    """Response with batch kernel results."""

    count: int
    results: list[BatchKernelResultItem]
    summary: dict[str, Any] = Field(default_factory=dict)
    computation_time_ms: float


@app.post("/kernel/batch", response_model=BatchKernelResponse, tags=["Kernel"])
async def compute_kernel_batch(
    request: BatchKernelRequest,
    api_key: str = Security(require_public),
) -> BatchKernelResponse:
    """
    Compute kernel invariants for multiple traces in a single request.

    Returns individual results plus summary statistics across all traces.
    """
    import numpy as np

    from .kernel_optimized import OptimizedKernelComputer

    _METRICS["batch_computations"] += 1
    start = time.monotonic()
    kernel = OptimizedKernelComputer(epsilon=request.epsilon)
    results: list[BatchKernelResultItem] = []

    for trace in request.traces:
        c = np.clip(np.array(trace.coordinates), request.epsilon, 1 - request.epsilon)
        n = len(c)
        w = np.ones(n) / n if trace.weights is None else np.array(trace.weights)
        outputs = kernel.compute(c, w)
        results.append(
            BatchKernelResultItem(
                label=trace.label,
                F=float(outputs.F),
                omega=float(outputs.omega),
                S=float(outputs.S),
                C=float(outputs.C),
                kappa=float(outputs.kappa),
                IC=float(outputs.IC),
                heterogeneity_gap=float(outputs.heterogeneity_gap),
                regime=outputs.regime,
            )
        )

    elapsed = (time.monotonic() - start) * 1000

    # Summary statistics
    f_vals = [r.F for r in results]
    ic_vals = [r.IC for r in results]
    regimes = [r.regime for r in results]
    regime_counts = {}
    for rg in regimes:
        regime_counts[rg] = regime_counts.get(rg, 0) + 1

    summary = {
        "mean_F": float(np.mean(f_vals)),
        "mean_IC": float(np.mean(ic_vals)),
        "mean_gap": float(np.mean(f_vals) - np.mean(ic_vals)),
        "ic_le_f_violations": sum(1 for f, ic in zip(f_vals, ic_vals, strict=True) if ic > f + 1e-10),
        "regime_distribution": regime_counts,
    }

    return BatchKernelResponse(
        count=len(results),
        results=results,
        summary=summary,
        computation_time_ms=elapsed,
    )


# ============================================================================
# Spine Evaluation (Contract → Canon → Closures → Ledger → Stance)
# ============================================================================


class SpineRequest(BaseModel):
    """Request for full spine evaluation."""

    coordinates: list[float] = Field(..., description="Coordinate values in [ε, 1-ε]")
    weights: list[float] | None = Field(None, description="Weights (uniform if not specified)")
    epsilon: float = Field(1e-8, description="Guard band ε")
    R_credit: float = Field(0.1, description="Return credit R")
    tau_R: float | None = Field(None, description="Return time (∞ if not specified)")
    label: str = Field("spine_eval", description="Label for this evaluation")


class SpineResponse(BaseModel):
    """Full spine evaluation response."""

    label: str

    # Stop 1: Contract (frozen parameters)
    contract: dict[str, Any]

    # Stop 2: Canon (kernel invariants — the five words)
    canon: dict[str, Any]

    # Stop 3: Closures (regime gates)
    closures: dict[str, Any]

    # Stop 4: Integrity Ledger (budget)
    ledger: dict[str, Any]

    # Stop 5: Stance (verdict)
    stance: dict[str, Any]

    computation_time_ms: float


@app.post("/kernel/spine", response_model=SpineResponse, tags=["Kernel"])
async def evaluate_spine(
    request: SpineRequest,
    api_key: str = Security(require_public),
) -> SpineResponse:
    """
    Full spine evaluation: Contract → Canon → Closures → Integrity Ledger → Stance.

    This is the canonical GCD pipeline in a single endpoint. Every claim
    passes through exactly five stops, in order.
    """
    import math

    import numpy as np

    from .kernel_optimized import OptimizedKernelComputer

    start = time.monotonic()
    eps = request.epsilon

    c = np.clip(np.array(request.coordinates), eps, 1 - eps)
    n = len(c)
    w = np.ones(n) / n if request.weights is None else np.array(request.weights)

    # Stop 1: Contract — freeze parameters
    contract = {
        "epsilon": eps,
        "p_exponent": P_EXPONENT,
        "alpha": ALPHA,
        "tol_seam": TOL_SEAM,
        "n_channels": n,
        "weight_sum": float(np.sum(w)),
    }

    # Stop 2: Canon — compute kernel (the five words emerge here)
    kernel = OptimizedKernelComputer(epsilon=eps)
    outputs = kernel.compute(c, w)
    canon = {
        "fidelity": {"F": float(outputs.F), "narrative": "What persisted through collapse"},
        "drift": {"omega": float(outputs.omega), "narrative": "What moved — ω = 1 − F"},
        "roughness": {"C": float(outputs.C), "S": float(outputs.S), "narrative": "Where/why it was bumpy"},
        "return": {
            "IC": float(outputs.IC),
            "kappa": float(outputs.kappa),
            "narrative": "Multiplicative coherence — exp(κ)",
        },
        "integrity": {
            "heterogeneity_gap": float(outputs.heterogeneity_gap),
            "ic_le_f": float(outputs.IC) <= float(outputs.F) + 1e-10,
            "f_plus_omega": float(outputs.F) + float(outputs.omega),
            "narrative": "Derived from the reconciled ledger, never asserted",
        },
    }

    # Stop 3: Closures — regime gates
    regime = classify_regime(float(outputs.omega), float(outputs.F), float(outputs.S), float(outputs.C))
    closures = {
        "regime": regime,
        "gates": {
            "omega_threshold": 0.038,
            "F_threshold": 0.90,
            "S_threshold": 0.15,
            "C_threshold": 0.14,
            "collapse_threshold": 0.30,
        },
        "gate_results": {
            "omega_below_stable": float(outputs.omega) < 0.038,
            "F_above_stable": float(outputs.F) > 0.90,
            "S_below_stable": float(outputs.S) < 0.15,
            "C_below_stable": float(outputs.C) < 0.14,
            "omega_below_collapse": float(outputs.omega) < 0.30,
        },
    }

    # Stop 4: Integrity Ledger — debit/credit budget
    tau_r = request.tau_R if request.tau_R is not None else float("inf")
    R = request.R_credit
    omega_val = float(outputs.omega)
    c_val = float(outputs.C)

    # Compute Γ(ω) = ω^p / (1 - ω + ε)
    gamma = omega_val**P_EXPONENT / (1 - omega_val + eps)
    D_omega = gamma  # Drift debit
    D_C = ALPHA * c_val  # Curvature debit

    # Budget: Δκ = R·τ_R − (D_ω + D_C)
    if math.isinf(tau_r):
        lhs = 0.0  # No return → no credit
        tau_r_display = "INF_REC"
    else:
        lhs = R * tau_r
        tau_r_display = str(tau_r)

    rhs = D_omega + D_C
    delta_kappa = lhs - rhs
    residual = abs(delta_kappa)

    ledger_data = {
        "debits": {"D_omega": D_omega, "D_C": D_C, "total_debit": rhs},
        "credits": {"R": R, "tau_R": tau_r_display, "R_times_tau_R": lhs},
        "delta_kappa": delta_kappa,
        "residual": residual,
        "residual_within_tol": residual <= TOL_SEAM,
    }

    # Stop 5: Stance — derived verdict
    if residual <= TOL_SEAM:
        verdict = "CONFORMANT"
    elif math.isinf(tau_r):
        verdict = "NON_EVALUABLE"
    else:
        verdict = "NONCONFORMANT"

    stance = {
        "verdict": verdict,
        "regime": regime,
        "critical_overlay": float(outputs.IC) < 0.30,
        "identities_verified": {
            "F_plus_omega_eq_1": abs(float(outputs.F) + float(outputs.omega) - 1.0) < 1e-12,
            "IC_le_F": float(outputs.IC) <= float(outputs.F) + 1e-10,
        },
    }

    elapsed = (time.monotonic() - start) * 1000

    return SpineResponse(
        label=request.label,
        contract=contract,
        canon=canon,
        closures=closures,
        ledger=ledger_data,
        stance=stance,
        computation_time_ms=elapsed,
    )


# ============================================================================
# Metrics Endpoint
# ============================================================================


@app.get("/metrics", tags=["System"])
async def get_metrics() -> dict[str, Any]:
    """
    Operational metrics and request statistics.

    Includes uptime, request counts, endpoint hit distribution, and
    kernel computation stats.
    """
    uptime_s = time.monotonic() - _METRICS["start_time"]
    return {
        "uptime_seconds": round(uptime_s, 1),
        "start_utc": _METRICS["start_utc"],
        "total_requests": _METRICS["total_requests"],
        "total_errors": _METRICS["total_errors"],
        "error_rate": (_METRICS["total_errors"] / max(_METRICS["total_requests"], 1)),
        "kernel_computations": _METRICS["kernel_computations"],
        "batch_computations": _METRICS["batch_computations"],
        "validations": _METRICS["validations"],
        "top_endpoints": dict(sorted(_METRICS["endpoints_hit"].items(), key=lambda x: x[1], reverse=True)[:20]),
        "frozen_contract": {
            "epsilon": EPSILON,
            "p_exponent": P_EXPONENT,
            "alpha": ALPHA,
            "tol_seam": TOL_SEAM,
        },
        "umcp_version": str(__version__),
        "api_version": API_VERSION,
    }


# ============================================================================
# Standard Model Endpoints
# ============================================================================


@app.get("/sm/particles", tags=["Standard Model"])
async def get_sm_particles(
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """
    Get the Standard Model particle catalog with kernel data.

    Returns 31 particles (17 fundamental + 14 composite) with their
    8-channel trace vectors and kernel invariants (F, ω, S, C, κ, IC).
    """
    _METRICS["kernel_computations"] += 1
    try:
        import numpy as np

        from closures.standard_model.subatomic_kernel import (
            COMPOSITE_PARTICLES,
            PARTICLES,
            build_trace,
        )

        from .kernel_optimized import OptimizedKernelComputer

        kernel = OptimizedKernelComputer(epsilon=EPSILON)
        results = []
        for name, pdata in {**PARTICLES, **COMPOSITE_PARTICLES}.items():
            c, w = build_trace(pdata)
            c_arr = np.clip(np.array(c), EPSILON, 1 - EPSILON)
            w_arr = np.array(w)
            out = kernel.compute(c_arr, w_arr)
            results.append(
                {
                    "name": name,
                    "type": "fundamental" if name in PARTICLES else "composite",
                    "mass_gev": pdata.get("mass_gev", pdata.get("mass_GeV")),
                    "spin": pdata.get("spin"),
                    "charge": pdata.get("charge"),
                    "F": round(float(out.F), 6),
                    "omega": round(float(out.omega), 6),
                    "IC": round(float(out.IC), 6),
                    "heterogeneity_gap": round(float(out.heterogeneity_gap), 6),
                    "regime": out.regime,
                }
            )

        return {
            "total_particles": len(results),
            "fundamental": sum(1 for r in results if r["type"] == "fundamental"),
            "composite": sum(1 for r in results if r["type"] == "composite"),
            "particles": results,
            "channels": [
                "mass_log",
                "spin_norm",
                "charge_norm",
                "color",
                "weak_isospin",
                "lepton_num",
                "baryon_num",
                "generation",
            ],
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Standard Model closure not available: {e}") from e


@app.get("/sm/theorems", tags=["Standard Model"])
async def get_sm_theorems(
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """
    Get the 10 Standard Model theorems with verification status.

    All 10/10 PROVEN with 74/74 subtests. Duality identity F + ω = 1
    verified to machine precision (0.0e+00).
    """
    theorems = [
        {
            "id": "T1",
            "name": "Spin-Statistics Kernel Ordering",
            "result": "⟨F⟩_fermion(0.615) > ⟨F⟩_boson(0.421), split = 0.194",
            "subtests": 12,
            "status": "PROVEN",
        },
        {
            "id": "T2",
            "name": "Generation Monotonicity",
            "result": "Gen1(0.576) < Gen2(0.620) < Gen3(0.649)",
            "subtests": 5,
            "status": "PROVEN",
        },
        {
            "id": "T3",
            "name": "Confinement as IC Collapse",
            "result": "IC drops 98.1% quarks→hadrons; 14/14 below min quark IC",
            "subtests": 19,
            "status": "PROVEN",
        },
        {
            "id": "T4",
            "name": "Mass-Kernel Log Mapping",
            "result": "13.2 OOM → F∈[0.37,0.73]; Spearman ρ=0.77 for quarks",
            "subtests": 5,
            "status": "PROVEN",
        },
        {
            "id": "T5",
            "name": "Charge Quantization",
            "result": "IC_neutral/IC_charged = 0.020 (50× suppression)",
            "subtests": 5,
            "status": "PROVEN",
        },
        {
            "id": "T6",
            "name": "Cross-Scale Universality",
            "result": "composite(0.444) < atom(0.516) < fundamental(0.558)",
            "subtests": 6,
            "status": "PROVEN",
        },
        {
            "id": "T7",
            "name": "Symmetry Breaking Amplification",
            "result": "EWSB amplifies gen spread 0.046→0.073, ΔF monotonic",
            "subtests": 5,
            "status": "PROVEN",
        },
        {
            "id": "T8",
            "name": "CKM Unitarity",
            "result": "CKM rows pass Tier-1; V_ub kills row-1 IC; J_CP=3.0e-5",
            "subtests": 5,
            "status": "PROVEN",
        },
        {
            "id": "T9",
            "name": "Running Coupling Flow",
            "result": "α_s monotone for Q≥10 GeV; confinement→NonPerturbative",
            "subtests": 6,
            "status": "PROVEN",
        },
        {
            "id": "T10",
            "name": "Nuclear Binding Curve",
            "result": "r(BE/A,Δ)=-0.41; peak at Cr/Fe (Z∈[23,30])",
            "subtests": 6,
            "status": "PROVEN",
        },
    ]
    return {
        "total_theorems": 10,
        "total_proven": 10,
        "total_subtests": 74,
        "duality_residual": "0.0e+00",
        "theorems": theorems,
    }


# ============================================================================
# Atomic Physics Endpoints
# ============================================================================


@app.get("/atomic/elements", tags=["Atomic"])
async def get_atomic_elements(
    Z_min: int = Query(1, ge=1, le=118, description="Minimum atomic number"),
    Z_max: int = Query(118, ge=1, le=118, description="Maximum atomic number"),
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """
    Get periodic table kernel data for elements Z_min through Z_max.

    Each element is mapped through the GCD kernel with 8 measurable
    properties. Tier-1 proof: 10,162 tests, 0 failures.
    """
    try:
        import numpy as np

        from closures.atomic_physics.periodic_kernel import ELEMENT_DATA, build_element_trace

        from .kernel_optimized import OptimizedKernelComputer

        kernel = OptimizedKernelComputer(epsilon=EPSILON)
        elements = []
        for z in range(Z_min, min(Z_max + 1, 119)):
            if z not in ELEMENT_DATA:
                continue
            edata = ELEMENT_DATA[z]
            c, w = build_element_trace(edata)
            c_arr = np.clip(np.array(c), EPSILON, 1 - EPSILON)
            w_arr = np.array(w)
            out = kernel.compute(c_arr, w_arr)
            elements.append(
                {
                    "Z": z,
                    "symbol": edata.get("symbol", f"E{z}"),
                    "name": edata.get("name", f"Element {z}"),
                    "F": round(float(out.F), 6),
                    "omega": round(float(out.omega), 6),
                    "IC": round(float(out.IC), 6),
                    "heterogeneity_gap": round(float(out.heterogeneity_gap), 6),
                    "regime": out.regime,
                }
            )

        return {
            "total_elements": len(elements),
            "Z_range": [Z_min, min(Z_max, 118)],
            "elements": elements,
            "tier1_proof": {"tests": 10162, "failures": 0},
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Atomic physics closure not available: {e}") from e


# ============================================================================
# Evolution Endpoints
# ============================================================================


@app.get("/evolution/organisms", tags=["Evolution"])
async def get_evolution_organisms(
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """
    Get evolution kernel data for 40 organisms spanning bacteria to mammals.

    Each organism is mapped through a 10-channel brain/neural kernel
    measuring neural complexity, sensory range, learning capacity, etc.
    """
    try:
        import numpy as np

        from closures.evolution.evolution_kernel import ORGANISMS, build_organism_trace

        from .kernel_optimized import OptimizedKernelComputer

        kernel = OptimizedKernelComputer(epsilon=EPSILON)
        results = []
        for name, odata in ORGANISMS.items():
            c, w = build_organism_trace(odata)
            c_arr = np.clip(np.array(c), EPSILON, 1 - EPSILON)
            w_arr = np.array(w)
            out = kernel.compute(c_arr, w_arr)
            results.append(
                {
                    "name": name,
                    "kingdom": odata.get("kingdom", "unknown"),
                    "F": round(float(out.F), 6),
                    "omega": round(float(out.omega), 6),
                    "IC": round(float(out.IC), 6),
                    "heterogeneity_gap": round(float(out.heterogeneity_gap), 6),
                    "regime": out.regime,
                }
            )

        return {
            "total_organisms": len(results),
            "organisms": results,
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Evolution closure not available: {e}") from e


# ============================================================================
# Universal Calculator Endpoint
# ============================================================================


class UniversalCalcRequest(BaseModel):
    """Request for universal calculator."""

    coordinates: list[float] = Field(..., description="Bounded coordinates c_i ∈ [0,1]")
    weights: list[float] | None = Field(None, description="Channel weights (uniform if not specified)")
    tau_R: float | None = Field(None, description="Return time (INF if not specified)")
    prior_kappa: float | None = Field(None, description="Prior κ for seam accounting")
    prior_IC: float | None = Field(None, description="Prior IC for seam accounting")
    R_credit: float = Field(0.1, description="Return credit for budget identity")
    coord_variances: list[float] | None = Field(None, description="Coordinate variances for uncertainty")
    mode: str = Field("standard", description="Mode: minimal, standard, full, rcft")


class UniversalCalcResponse(BaseModel):
    """Response from universal calculator."""

    timestamp: str
    computation_mode: str
    input_hash: str
    kernel: dict[str, Any]
    regime: str
    costs: dict[str, Any] | None = None
    seam: dict[str, Any] | None = None
    gcd: dict[str, Any] | None = None
    rcft: dict[str, Any] | None = None
    uncertainty: dict[str, Any] | None = None
    ss1m: dict[str, str] | None = None
    diagnostics: dict[str, Any] | None = None


@app.post("/calculate", response_model=UniversalCalcResponse, tags=["Calculator"])
async def universal_calculate(
    request: UniversalCalcRequest,
    api_key: str = Security(require_public),
) -> UniversalCalcResponse:
    """
    Universal UMCP Calculator - compute all metrics in one call.

    This endpoint integrates all UMCP concepts:
    - Tier-1 Kernel Invariants (ω, F, S, C, τ_R, κ, IC)
    - Cost Closures (Γ(ω), D_C, budget identity)
    - Regime Classification (STABLE/WATCH/COLLAPSE/CRITICAL)
    - Seam Accounting (if prior state provided)
    - GCD Metrics (energy, collapse, flux, resonance)
    - RCFT Metrics (fractal dimension, recursive field)
    - Uncertainty Propagation (if variances provided)
    - Human-Verifiable Checksums (SS1M triads)

    Modes:
    - minimal: Tier-1 kernel only
    - standard: Kernel + costs + regime (default)
    - full: All metrics including GCD and RCFT
    - rcft: Focus on RCFT metrics

    Requires API key authentication.
    """
    from .universal_calculator import ComputationMode, UniversalCalculator

    # Map mode string to enum
    mode_map = {
        "minimal": ComputationMode.MINIMAL,
        "standard": ComputationMode.STANDARD,
        "full": ComputationMode.FULL,
        "rcft": ComputationMode.RCFT,
    }

    mode = mode_map.get(request.mode.lower(), ComputationMode.STANDARD)

    # Compute
    calc = UniversalCalculator()
    result = calc.compute_all(
        coordinates=request.coordinates,
        weights=request.weights,
        tau_R=request.tau_R,
        prior_kappa=request.prior_kappa,
        prior_IC=request.prior_IC,
        R_credit=request.R_credit,
        coord_variances=request.coord_variances,
        mode=mode,
    )

    # Convert to response
    return UniversalCalcResponse(
        timestamp=result.timestamp,
        computation_mode=result.computation_mode,
        input_hash=result.input_hash,
        kernel=result.kernel.to_dict(),
        regime=result.regime,
        costs=result.costs.to_dict() if result.costs else None,
        seam=result.seam.to_dict() if result.seam else None,
        gcd=result.gcd.to_dict() if result.gcd else None,
        rcft=result.rcft.to_dict() if result.rcft else None,
        uncertainty=result.uncertainty.to_dict() if result.uncertainty else None,
        ss1m=result.ss1m.to_dict() if result.ss1m else None,
        diagnostics=result.diagnostics if result.diagnostics else None,
    )


@app.post("/kernel/budget", response_model=BudgetIdentityResponse, tags=["Kernel"])
async def verify_budget_identity(
    R: float = Query(..., description="Return indicator (0 or 1)"),
    tau_R: float = Query(..., description="Return time"),
    D_omega: float = Query(..., description="Omega drift component"),
    D_C: float = Query(..., description="Curvature drift component"),
    delta_kappa: float = Query(..., description="Ledger delta κ"),
    api_key: str = Security(require_public),
) -> BudgetIdentityResponse:
    """
    Verify the UMCP budget identity: R·τ_R = D_ω + D_C + Δκ


    The budget identity is the core conservation law of UMCP.
    The seam residual |LHS - RHS| should be ≤ 0.005 for conformance.

    Requires API key authentication.
    """
    lhs = R * tau_R
    rhs = D_omega + D_C + delta_kappa
    seam_residual = abs(lhs - rhs)
    seam_pass = seam_residual <= 0.005

    return BudgetIdentityResponse(
        R=R,
        tau_R=tau_R,
        D_omega=D_omega,
        D_C=D_C,
        delta_kappa=delta_kappa,
        lhs=lhs,
        rhs=rhs,
        seam_residual=seam_residual,
        seam_pass=seam_pass,
    )


# ============================================================================
# Uncertainty Propagation Endpoints
# ============================================================================


class UncertaintyRequest(BaseModel):
    """Request for uncertainty propagation."""

    coordinates: list[float] = Field(..., description="Coordinate values")
    weights: list[float] | None = Field(None, description="Weights (uniform if not specified)")
    coordinate_variances: list[float] = Field(..., description="Variance for each coordinate")
    epsilon: float = Field(1e-8, description="Clipping tolerance")


class UncertaintyResponse(BaseModel):
    """Response with propagated uncertainties."""

    # Kernel outputs
    F: float
    omega: float
    S: float
    kappa: float
    C: float

    # Standard deviations (1σ)
    std_F: float
    std_omega: float
    std_S: float
    std_kappa: float
    std_C: float

    # Confidence intervals (95%)
    ci_F: list[float]
    ci_omega: list[float]
    ci_S: list[float]
    ci_kappa: list[float]
    ci_C: list[float]


@app.post("/uncertainty/propagate", response_model=UncertaintyResponse, tags=["Uncertainty"])
async def propagate_uncertainty(
    request: UncertaintyRequest,
    api_key: str = Security(require_public),
) -> UncertaintyResponse:
    """
    Propagate measurement uncertainty through the kernel.

    Uses delta-method (first-order Taylor expansion) to compute
    output variances from input coordinate variances.

    Reference: KERNEL_SPECIFICATION.md Lemmas 3, 11, 12, 13, 17, 18

    Requires API key authentication.
    """
    import numpy as np

    from .kernel_optimized import OptimizedKernelComputer
    from .uncertainty import propagate_independent_uncertainty

    c = np.array(request.coordinates)
    var_c = np.array(request.coordinate_variances)
    n = len(c)

    if len(var_c) != n:
        raise HTTPException(
            status_code=400,
            detail=f"Variance length {len(var_c)} != coordinates length {n}",
        )

    # Use uniform weights if not specified
    w = np.ones(n) / n if request.weights is None else np.array(request.weights)

    # ε-clip
    eps = request.epsilon
    c = np.clip(c, eps, 1 - eps)

    # Compute kernel outputs
    kernel = OptimizedKernelComputer(epsilon=eps)
    outputs = kernel.compute(c, w)

    # Propagate uncertainty
    bounds = propagate_independent_uncertainty(c, w, var_c, eps)

    # 95% confidence intervals (±1.96σ)
    z = 1.96

    return UncertaintyResponse(
        F=float(outputs.F),
        omega=float(outputs.omega),
        S=float(outputs.S),
        kappa=float(outputs.kappa),
        C=float(outputs.C),
        std_F=bounds.std_F,
        std_omega=bounds.std_omega,
        std_S=bounds.std_S,
        std_kappa=bounds.std_kappa,
        std_C=bounds.std_C,
        ci_F=[float(outputs.F) - z * bounds.std_F, float(outputs.F) + z * bounds.std_F],
        ci_omega=[float(outputs.omega) - z * bounds.std_omega, float(outputs.omega) + z * bounds.std_omega],
        ci_S=[float(outputs.S) - z * bounds.std_S, float(outputs.S) + z * bounds.std_S],
        ci_kappa=[float(outputs.kappa) - z * bounds.std_kappa, float(outputs.kappa) + z * bounds.std_kappa],
        ci_C=[float(outputs.C) - z * bounds.std_C, float(outputs.C) + z * bounds.std_C],
    )


# ============================================================================
# Time Series Analysis Endpoints
# ============================================================================


class TimeSeriesRequest(BaseModel):
    """Request for time series analysis."""

    timestamps: list[str] | None = Field(None, description="ISO timestamps")
    omega_series: list[float] = Field(..., description="Omega values over time")
    F_series: list[float] | None = Field(None, description="F values (computed from omega if missing)")
    kappa_series: list[float] | None = Field(None, description="Kappa values")


class TimeSeriesResponse(BaseModel):
    """Response with time series analysis."""

    n_points: int
    duration_hours: float | None

    # Summary statistics
    omega_mean: float
    omega_std: float
    omega_min: float
    omega_max: float
    omega_trend: float  # Linear trend coefficient

    # Regime analysis
    regime_counts: dict[str, int]
    regime_transitions: int
    stability_score: float  # Fraction in STABLE regime

    # Anomaly detection
    anomalies: list[dict[str, Any]]

    # Return time analysis (if kappa provided)
    avg_return_time: float | None = None
    return_rate: float | None = None


@app.post("/analysis/timeseries", response_model=TimeSeriesResponse, tags=["Analysis"])
async def analyze_timeseries(
    request: TimeSeriesRequest,
    api_key: str = Security(require_public),
) -> TimeSeriesResponse:
    """
    Analyze a time series of kernel invariants.

    Computes:
    - Summary statistics (mean, std, min, max, trend)
    - Regime distribution and transitions
    - Stability score
    - Anomaly detection (3σ outliers)

    Requires API key authentication.
    """
    from datetime import datetime as dt

    import numpy as np

    omega = np.array(request.omega_series)
    n = len(omega)

    # Parse timestamps if provided
    duration_hours: float | None = None
    if request.timestamps and len(request.timestamps) >= 2:
        try:
            t0 = dt.fromisoformat(request.timestamps[0].replace("Z", "+00:00"))
            t1 = dt.fromisoformat(request.timestamps[-1].replace("Z", "+00:00"))
            duration_hours = (t1 - t0).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

    # Summary statistics
    omega_mean = float(np.mean(omega))
    omega_std = float(np.std(omega))
    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))

    # Linear trend (slope of linear regression)
    t = np.arange(n)
    if n > 1:
        coeffs = np.polyfit(t, omega, 1)
        omega_trend = float(coeffs[0])
    else:
        omega_trend = 0.0

    # Regime classification (using frozen contract thresholds)
    regimes = [classify_regime(w, 1.0 - w, 0.0, 0.0) for w in omega]
    regime_counts = {"STABLE": 0, "WATCH": 0, "COLLAPSE": 0}
    for r in regimes:
        regime_counts[r] += 1

    # Count transitions
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])

    # Stability score
    stability_score = regime_counts["STABLE"] / n if n > 0 else 0.0

    # Anomaly detection (3σ)
    anomalies: list[dict[str, Any]] = []
    if omega_std > 0:
        z_scores = (omega - omega_mean) / omega_std
        for i, z in enumerate(z_scores):
            if abs(z) > 3:
                anomalies.append(
                    {
                        "index": i,
                        "omega": float(omega[i]),
                        "z_score": float(z),
                        "type": "high" if z > 0 else "low",
                    }
                )

    # Return time analysis
    avg_return_time: float | None = None
    return_rate: float | None = None
    if request.kappa_series and len(request.kappa_series) > 1:
        kappa = np.array(request.kappa_series)
        # Estimate return times from kappa changes
        dk = np.abs(np.diff(kappa))
        avg_return_time = float(np.mean(dk)) if len(dk) > 0 else None
        return_rate = stability_score  # Use stability as proxy

    return TimeSeriesResponse(
        n_points=n,
        duration_hours=duration_hours,
        omega_mean=omega_mean,
        omega_std=omega_std,
        omega_min=omega_min,
        omega_max=omega_max,
        omega_trend=omega_trend,
        regime_counts=regime_counts,
        regime_transitions=transitions,
        stability_score=stability_score,
        anomalies=anomalies,
        avg_return_time=avg_return_time,
        return_rate=return_rate,
    )


# ============================================================================
# Data Analysis Endpoints
# ============================================================================


class StatisticsRequest(BaseModel):
    """Request for statistical analysis."""

    data: list[float] = Field(..., description="Data values")
    weights: list[float] | None = Field(None, description="Observation weights")


class StatisticsResponse(BaseModel):
    """Response with statistical analysis."""

    n: int
    mean: float
    std: float
    var: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation


class CorrelationRequest(BaseModel):
    """Request for correlation analysis."""

    x: list[float] = Field(..., description="First variable")
    y: list[float] = Field(..., description="Second variable")


class CorrelationResponse(BaseModel):
    """Response with correlation analysis."""

    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    regression_slope: float
    regression_intercept: float
    r_squared: float


@app.post("/analysis/statistics", response_model=StatisticsResponse, tags=["Analysis"])
async def compute_statistics(
    request: StatisticsRequest,
    api_key: str = Security(require_public),
) -> StatisticsResponse:
    """
    Compute comprehensive descriptive statistics.

    Includes moments, quantiles, and distribution shape measures.
    Uses pure numpy implementations to minimize dependencies.
    Requires API key authentication.
    """
    import numpy as np

    data = np.array(request.data, dtype=np.float64)
    n = len(data)

    if n == 0:
        raise HTTPException(status_code=400, detail="Empty data array")

    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n > 1 else 0.0
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0

    # Quantiles
    q25, median, q75 = float(np.percentile(data, 25)), float(np.percentile(data, 50)), float(np.percentile(data, 75))
    iqr = q75 - q25

    # Shape measures (pure numpy implementation)
    # Skewness: E[(X-μ)³] / σ³
    if n > 2 and std > 0:
        m3 = float(np.mean((data - mean) ** 3))
        skewness = m3 / (std**3)
    else:
        skewness = 0.0

    # Kurtosis (excess): E[(X-μ)⁴] / σ⁴ - 3
    if n > 3 and std > 0:
        m4 = float(np.mean((data - mean) ** 4))
        kurtosis = m4 / (std**4) - 3.0
    else:
        kurtosis = 0.0

    # Coefficient of variation
    cv = std / abs(mean) if mean != 0 else 0.0

    return StatisticsResponse(
        n=n,
        mean=mean,
        std=std,
        var=var,
        min=float(np.min(data)),
        max=float(np.max(data)),
        median=median,
        q25=q25,
        q75=q75,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        cv=cv,
    )


@app.post("/analysis/correlation", response_model=CorrelationResponse, tags=["Analysis"])
async def compute_correlation(
    request: CorrelationRequest,
    api_key: str = Security(require_public),
) -> CorrelationResponse:
    """
    Compute correlation and regression between two variables.

    Returns Pearson and Spearman correlations, plus linear regression.
    Uses pure numpy implementations to minimize dependencies.
    Requires API key authentication.
    """
    import numpy as np

    x = np.array(request.x, dtype=np.float64)
    y = np.array(request.y, dtype=np.float64)

    if len(x) != len(y):
        raise HTTPException(status_code=400, detail="x and y must have same length")

    n = len(x)
    if n < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 points")

    # Pearson correlation (pure numpy)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    cov_xy = float(np.sum(x_centered * y_centered))
    std_x = float(np.sqrt(np.sum(x_centered**2)))
    std_y = float(np.sqrt(np.sum(y_centered**2)))
    pearson_r = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else 0.0

    # Approximate p-value using t-distribution approximation
    if abs(pearson_r) < 1.0 and n > 2:
        t_stat = pearson_r * np.sqrt((n - 2) / (1 - pearson_r**2))
        # Approximate p-value (two-tailed) - simplified
        pearson_p = 2.0 * (1.0 - min(0.5 + 0.5 * np.tanh(abs(t_stat) / 1.5), 0.9999))
    else:
        pearson_p = 0.0 if abs(pearson_r) >= 0.9999 else 1.0

    # Spearman correlation (rank-based)
    x_ranks = np.argsort(np.argsort(x)).astype(np.float64) + 1
    y_ranks = np.argsort(np.argsort(y)).astype(np.float64) + 1
    x_rank_centered = x_ranks - np.mean(x_ranks)
    y_rank_centered = y_ranks - np.mean(y_ranks)
    cov_ranks = float(np.sum(x_rank_centered * y_rank_centered))
    std_x_ranks = float(np.sqrt(np.sum(x_rank_centered**2)))
    std_y_ranks = float(np.sqrt(np.sum(y_rank_centered**2)))
    spearman_r = cov_ranks / (std_x_ranks * std_y_ranks) if std_x_ranks > 0 and std_y_ranks > 0 else 0.0

    # Approximate p-value for Spearman
    if abs(spearman_r) < 1.0 and n > 2:
        t_stat_sp = spearman_r * np.sqrt((n - 2) / (1 - spearman_r**2))
        spearman_p = 2.0 * (1.0 - min(0.5 + 0.5 * np.tanh(abs(t_stat_sp) / 1.5), 0.9999))
    else:
        spearman_p = 0.0 if abs(spearman_r) >= 0.9999 else 1.0

    # Linear regression (least squares)
    slope = cov_xy / (std_x**2) if std_x > 0 else 0.0
    intercept = float(np.mean(y)) - slope * float(np.mean(x))
    r_squared = pearson_r**2

    return CorrelationResponse(
        n=n,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        regression_slope=slope,
        regression_intercept=intercept,
        r_squared=r_squared,
    )


class LedgerAnalysisResponse(BaseModel):
    """Comprehensive ledger analysis response."""

    total_entries: int
    date_range: dict[str, str | None]

    # Overall statistics
    conformant_rate: float
    avg_omega: float
    avg_kappa: float
    omega_trend: float

    # Regime distribution
    regime_distribution: dict[str, float]

    # Health indicators
    stability_index: float  # 0-1, higher is better
    drift_indicator: float  # Absolute trend magnitude
    anomaly_count: int

    # Recent performance (last 10 entries)
    recent_conformant_rate: float
    recent_avg_omega: float


@app.get("/analysis/ledger", response_model=LedgerAnalysisResponse, tags=["Analysis"])
async def analyze_ledger(
    api_key: str = Security(require_public),
) -> LedgerAnalysisResponse:
    """
    Comprehensive analysis of the validation ledger.

    Provides health indicators, regime distribution, and trend analysis.
    Requires API key authentication.
    """
    import numpy as np

    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        raise HTTPException(status_code=404, detail="Ledger not found")

    # Read ledger
    entries: list[dict[str, Any]] = []
    with open(ledger_path) as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    n = len(entries)
    if n == 0:
        raise HTTPException(status_code=404, detail="Ledger is empty")

    # Extract values
    statuses: list[str] = [e.get("run_status", e.get("status", "")) for e in entries]
    omegas: list[float] = []
    kappas: list[float] = []
    timestamps: list[str] = []

    import contextlib

    for e in entries:
        if e.get("omega"):
            with contextlib.suppress(ValueError):
                omegas.append(float(e["omega"]))
        if e.get("curvature") or e.get("delta_kappa"):
            with contextlib.suppress(ValueError):
                kappas.append(float(e.get("curvature", e.get("delta_kappa", 0))))
        if e.get("timestamp"):
            timestamps.append(str(e["timestamp"]))

    # Conformant rate
    conformant_count = sum(1 for s in statuses if s == "CONFORMANT")
    conformant_rate = conformant_count / n

    # Statistics
    omega_arr = np.array(omegas) if omegas else np.array([0.5])
    kappa_arr = np.array(kappas) if kappas else np.array([0.0])

    avg_omega = float(np.mean(omega_arr))
    avg_kappa = float(np.mean(kappa_arr))

    # Trend
    if len(omega_arr) > 1:
        t = np.arange(len(omega_arr))
        coeffs = np.polyfit(t, omega_arr, 1)
        omega_trend = float(coeffs[0])
    else:
        omega_trend = 0.0

    # Regime distribution (using frozen contract thresholds)
    regimes = [classify_regime(w, 1.0 - w, 0.0, 0.0) for w in omega_arr]
    regime_dist = {
        "STABLE": regimes.count("STABLE") / len(regimes) if regimes else 0,
        "WATCH": regimes.count("WATCH") / len(regimes) if regimes else 0,
        "COLLAPSE": regimes.count("COLLAPSE") / len(regimes) if regimes else 0,
    }

    # Health indicators
    stability_index = regime_dist["STABLE"]
    drift_indicator = abs(omega_trend)

    # Anomalies (3σ outliers)
    anomaly_count = 0
    if len(omega_arr) > 2:
        std = float(np.std(omega_arr))
        mean = float(np.mean(omega_arr))
        if std > 0:
            z_scores = np.abs((omega_arr - mean) / std)
            anomaly_count = int(np.sum(z_scores > 3))

    # Recent performance
    recent_n = min(10, n)
    recent_statuses = statuses[-recent_n:]
    recent_omegas = omega_arr[-recent_n:] if len(omega_arr) >= recent_n else omega_arr

    recent_conformant = sum(1 for s in recent_statuses if s == "CONFORMANT")
    recent_conformant_rate = recent_conformant / recent_n
    recent_avg_omega = float(np.mean(recent_omegas))

    return LedgerAnalysisResponse(
        total_entries=n,
        date_range={
            "first": timestamps[0] if timestamps else None,
            "last": timestamps[-1] if timestamps else None,
        },
        conformant_rate=conformant_rate,
        avg_omega=avg_omega,
        avg_kappa=avg_kappa,
        omega_trend=omega_trend,
        regime_distribution=regime_dist,
        stability_index=stability_index,
        drift_indicator=drift_indicator,
        anomaly_count=anomaly_count,
        recent_conformant_rate=recent_conformant_rate,
        recent_avg_omega=recent_avg_omega,
    )


# ============================================================================
# WEYL Cosmology Endpoints
# ============================================================================


# ============================================================================
# Domain Summary & Canon Endpoints
# ============================================================================


class DomainInfo(BaseModel):
    """Information about a UMCP domain."""

    name: str = Field(description="Domain short name")
    description: str = Field(description="Domain description")
    tier: str = Field(description="Tier classification")
    closures: int = Field(description="Number of closures")
    contract: str = Field(description="Contract identifier")
    canon: str = Field(description="Canon anchor file")
    casepack: str = Field(description="Primary casepack")


class CanonAnchor(BaseModel):
    """Summary of a canon anchor file."""

    id: str = Field(description="Canon identifier")
    version: str = Field(description="Canon version")
    domain: str = Field(description="Domain name")
    axiom_count: int = Field(description="Number of axioms")
    symbol_count: int = Field(description="Number of reserved symbols")
    gate_count: int = Field(description="Number of regime gates")


class AstroLuminosityRequest(BaseModel):
    """Request for stellar luminosity computation."""

    m_star: float = Field(description="Stellar mass in solar masses")
    t_eff: float = Field(description="Effective temperature in K")
    r_star: float = Field(description="Stellar radius in solar radii")


class AstroDistanceRequest(BaseModel):
    """Request for distance ladder computation."""

    m_app: float = Field(description="Apparent magnitude")
    m_abs: float = Field(description="Absolute magnitude")
    pi_arcsec: float = Field(description="Parallax in arcseconds")
    z_cosmo: float = Field(description="Cosmological redshift")
    h0: float = Field(default=70.0, description="Hubble constant km/s/Mpc")


class NucBindingRequest(BaseModel):
    """Request for nuclear binding energy computation."""

    Z: int = Field(description="Proton number", ge=1, le=120)
    A: int = Field(description="Mass number", ge=1, le=300)


class NucAlphaRequest(BaseModel):
    """Request for alpha decay computation."""

    Z: int = Field(description="Parent proton number", ge=2, le=120)
    A: int = Field(description="Parent mass number", ge=4, le=300)
    Q_alpha_MeV: float = Field(default=0.0, description="Q-value for alpha decay (MeV), auto-computed if 0")


class QMCollapseRequest(BaseModel):
    """Request for wavefunction collapse computation."""

    psi_amplitudes: list[float] = Field(description="Wavefunction amplitudes")
    measurement_probs: list[float] = Field(description="Measured probabilities")
    observed_outcome_idx: int = Field(default=0, description="Observed outcome index")


class QMTunnelingRequest(BaseModel):
    """Request for tunneling computation."""

    e_particle: float = Field(description="Particle energy (eV)")
    v_barrier: float = Field(description="Barrier height (eV)")
    barrier_width: float = Field(description="Barrier width (nm)")
    particle_mass: float | None = Field(default=None, description="Particle mass (kg), default electron")


class QMEntanglementRequest(BaseModel):
    """Request for entanglement computation."""

    rho_eigenvalues: list[float] = Field(description="Density matrix eigenvalues")
    bell_correlations: list[float] | None = Field(default=None, description="Bell correlation values (4)")


class ClosureResult(BaseModel):
    """Generic result from a domain closure computation."""

    result: dict[str, Any] = Field(description="Computation results")
    domain: str = Field(description="Domain name")
    closure: str = Field(description="Closure name")


@app.get("/domains", response_model=list[DomainInfo], tags=["Domains"])
async def list_domains(
    api_key: str = Security(require_public),
) -> list[DomainInfo]:
    """List all 18 UMCP domains with metadata."""
    return [
        DomainInfo(
            name="GCD",
            description="Generative Collapse Dynamics — foundational kernel",
            tier="Tier-1",
            closures=5,
            contract="UMA.INTSTACK.v1",
            canon="gcd_anchors.yaml",
            casepack="gcd_complete",
        ),
        DomainInfo(
            name="KIN",
            description="Kinematics — motion analysis, phase space",
            tier="Tier-0",
            closures=6,
            contract="KIN.INTSTACK.v1",
            canon="kin_anchors.yaml",
            casepack="kinematics_complete",
        ),
        DomainInfo(
            name="RCFT",
            description="Recursive Collapse Field Theory — fractal overlays",
            tier="Tier-2",
            closures=4,
            contract="RCFT.INTSTACK.v1",
            canon="rcft_anchors.yaml",
            casepack="rcft_complete",
        ),
        DomainInfo(
            name="WEYL",
            description="WEYL cosmology — modified gravity, DES Y3 analysis",
            tier="Tier-2",
            closures=6,
            contract="WEYL.INTSTACK.v1",
            canon="weyl_anchors.yaml",
            casepack="weyl_des_y3",
        ),
        DomainInfo(
            name="Security",
            description="Security validation, audit, and integrity checking",
            tier="Tier-2",
            closures=15,
            contract="SECURITY.INTSTACK.v1",
            canon="anchors.yaml",
            casepack="security_validation",
        ),
        DomainInfo(
            name="ASTRO",
            description="Astronomy — stellar classification, HR diagram, spectral analysis",
            tier="Tier-2",
            closures=6,
            contract="ASTRO.INTSTACK.v1",
            canon="astro_anchors.yaml",
            casepack="astronomy_complete",
        ),
        DomainInfo(
            name="NUC",
            description="Nuclear physics — binding energy, decay chains, QGP/RHIC",
            tier="Tier-2",
            closures=6,
            contract="NUC.INTSTACK.v1",
            canon="nuc_anchors.yaml",
            casepack="nuclear_chain",
        ),
        DomainInfo(
            name="QM",
            description="Quantum mechanics — wavefunction, entanglement, QDM, FQHE",
            tier="Tier-2",
            closures=6,
            contract="QM.INTSTACK.v1",
            canon="qm_anchors.yaml",
            casepack="quantum_mechanics_complete",
        ),
        DomainInfo(
            name="FIN",
            description="Finance — portfolio continuity, market coherence",
            tier="Tier-2",
            closures=1,
            contract="FINANCE.INTSTACK.v1",
            canon="—",
            casepack="finance_continuity",
        ),
        DomainInfo(
            name="SM",
            description="Standard Model — 31 particles, 10 proven theorems, 74/74 subtests",
            tier="Tier-2",
            closures=11,
            contract="SM.INTSTACK.v1",
            canon="sm_anchors.yaml",
            casepack="—",
        ),
        DomainInfo(
            name="ATOM",
            description="Atomic physics — 118 elements, periodic kernel, cross-scale analysis",
            tier="Tier-2",
            closures=10,
            contract="ATOM.INTSTACK.v1",
            canon="atom_anchors.yaml",
            casepack="—",
        ),
        DomainInfo(
            name="MATL",
            description="Materials science — element database (118 elements, 18 fields)",
            tier="Tier-2",
            closures=4,
            contract="MATL.INTSTACK.v1",
            canon="matl_anchors.yaml",
            casepack="—",
        ),
        DomainInfo(
            name="EVERY",
            description="Everyday physics — thermodynamics, optics, electromagnetism",
            tier="Tier-2",
            closures=4,
            contract="EVERY.INTSTACK.v1",
            canon="—",
            casepack="—",
        ),
        DomainInfo(
            name="EVO",
            description="Evolution — 40 organisms, 10-channel brain kernel, comparative neuroscience",
            tier="Tier-2",
            closures=3,
            contract="EVO.INTSTACK.v1",
            canon="evo_anchors.yaml",
            casepack="evolution_kernel",
        ),
        DomainInfo(
            name="SEM",
            description="Dynamic semiotics — 30 sign systems, 8-channel semiotic kernel",
            tier="Tier-2",
            closures=3,
            contract="SEM.INTSTACK.v1",
            canon="semiotics_anchors.yaml",
            casepack="—",
        ),
        DomainInfo(
            name="CONSC",
            description="Consciousness coherence — 20 systems, 7 theorems (T-CC-1 to T-CC-7)",
            tier="Tier-2",
            closures=3,
            contract="CONSC.INTSTACK.v1",
            canon="—",
            casepack="—",
        ),
        DomainInfo(
            name="CT",
            description="Continuity theory — continuity law closures",
            tier="Tier-2",
            closures=2,
            contract="CT.INTSTACK.v1",
            canon="ct_anchors.yaml",
            casepack="—",
        ),
        DomainInfo(
            name="AWR",
            description="Awareness cognition — 5+5 channel awareness-aptitude kernel, 10 theorems",
            tier="Tier-2",
            closures=3,
            contract="AWR.INTSTACK.v1",
            canon="—",
            casepack="—",
        ),
    ]


@app.get("/canon", response_model=list[CanonAnchor], tags=["Domains"])
async def list_canon_anchors(
    api_key: str = Security(require_public),
) -> list[CanonAnchor]:
    """List all canon anchor files with summary metadata."""
    repo_root = get_repo_root()
    canon_dir = repo_root / "canon"
    results: list[CanonAnchor] = []
    if not canon_dir.exists():
        return results

    yaml_mod = None
    try:
        import yaml as yaml_mod  # type: ignore[no-redef]
    except ImportError:
        return results

    for yf in sorted(canon_dir.glob("*_anchors.yaml")):
        try:
            with open(yf) as f:
                data = yaml_mod.safe_load(f)
            if not data:
                continue
            meta = data.get("metadata", data.get("canon", {}))
            results.append(
                CanonAnchor(
                    id=meta.get("id", meta.get("canon_id", yf.stem)),
                    version=str(meta.get("version", "1.0.0")),
                    domain=yf.stem.replace("_anchors", "").upper(),
                    axiom_count=len(data.get("axioms", [])),
                    symbol_count=len(data.get("tier_1_symbols", data.get("reserved_symbols", data.get("symbols", [])))),
                    gate_count=len(data.get("regime_gates", data.get("gates", []))),
                )
            )
        except Exception:
            pass

    return results


@app.get("/canon/{domain}", tags=["Domains"])
async def get_canon_detail(
    domain: str,
    api_key: str = Security(require_public),
) -> dict[str, Any]:
    """Get full canon anchor data for a specific domain."""
    repo_root = get_repo_root()
    canon_dir = repo_root / "canon"
    target = canon_dir / f"{domain.lower()}_anchors.yaml"
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Canon not found for domain: {domain}")

    try:
        import yaml

        with open(target) as f:
            data = yaml.safe_load(f)
        return cast(dict[str, Any], data or {})
    except ImportError as e:
        raise HTTPException(status_code=500, detail="PyYAML not available") from e


# ============================================================================
# Astronomy Endpoints
# ============================================================================


@app.post("/astro/luminosity", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_luminosity(
    req: AstroLuminosityRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute stellar luminosity via Stefan-Boltzmann and mass-luminosity relations."""
    try:
        from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

        result = compute_stellar_luminosity(req.m_star, req.t_eff, req.r_star)
        return ClosureResult(result=result, domain="ASTRO", closure="stellar_luminosity")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


@app.post("/astro/distance", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_distance(
    req: AstroDistanceRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Cross-validate distances via modulus, parallax, and Hubble flow."""
    try:
        from closures.astronomy.distance_ladder import compute_distance_ladder

        result = compute_distance_ladder(req.m_app, req.m_abs, req.pi_arcsec, req.z_cosmo, req.h0)
        return ClosureResult(result=result, domain="ASTRO", closure="distance_ladder")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


@app.post("/astro/spectral", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_spectral(
    t_eff: float = Query(..., description="Effective temperature (K)"),
    b_v: float = Query(..., description="B-V color index"),
    spectral_class: str = Query(default="G", description="Spectral class (O,B,A,F,G,K,M)"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Analyze spectrum: Wien peak, color-temperature, spectral embedding."""
    try:
        from closures.astronomy.spectral_analysis import compute_spectral_analysis

        result = compute_spectral_analysis(t_eff, b_v, spectral_class)
        return ClosureResult(result=result, domain="ASTRO", closure="spectral_analysis")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


@app.post("/astro/evolution", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_evolution(
    m_star: float = Query(..., description="Stellar mass (solar masses)"),
    l_obs: float = Query(..., description="Observed luminosity (L☉)"),
    t_eff: float = Query(..., description="Effective temperature (K)"),
    age_gyr: float = Query(..., description="Age in Gyr"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute stellar evolution: MS lifetime, phase, ZAMS properties."""
    try:
        from closures.astronomy.stellar_evolution import compute_stellar_evolution

        result = compute_stellar_evolution(m_star, l_obs, t_eff, age_gyr)
        return ClosureResult(result=result, domain="ASTRO", closure="stellar_evolution")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


@app.post("/astro/orbital", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_orbital(
    p_orb: float = Query(..., description="Orbital period (years)"),
    a_semi: float = Query(..., description="Semi-major axis (AU)"),
    m_total: float = Query(..., description="Total mass (solar masses)"),
    e_orb: float = Query(..., ge=0.0, lt=1.0, description="Eccentricity"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Validate Kepler III, compute orbital velocity and energy."""
    try:
        from closures.astronomy.orbital_mechanics import compute_orbital_mechanics

        result = compute_orbital_mechanics(p_orb, a_semi, m_total, e_orb)
        return ClosureResult(result=result, domain="ASTRO", closure="orbital_mechanics")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


@app.post("/astro/dynamics", response_model=ClosureResult, tags=["ASTRO"])
async def compute_astro_dynamics(
    v_rot: float = Query(..., description="Rotation velocity (km/s)"),
    r_obs: float = Query(..., description="Observed radius (kpc)"),
    sigma_v: float = Query(..., description="Velocity dispersion (km/s)"),
    m_luminous: float = Query(..., description="Luminous mass (solar masses)"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute virial mass, dark matter fraction, virial ratio."""
    try:
        from closures.astronomy.gravitational_dynamics import compute_gravitational_dynamics

        result = compute_gravitational_dynamics(v_rot, r_obs, sigma_v, m_luminous)
        return ClosureResult(result=result, domain="ASTRO", closure="gravitational_dynamics")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Astronomy closures not available") from e


# ============================================================================
# Nuclear Physics Endpoints
# ============================================================================


@app.post("/nuclear/binding", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_binding(
    req: NucBindingRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute SEMF binding energy per nucleon and regime classification."""
    try:
        from closures.nuclear_physics import compute_binding

        result = compute_binding(req.Z, req.A)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="nuclide_binding")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


@app.post("/nuclear/alpha-decay", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_alpha(
    req: NucAlphaRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute alpha decay Q-value and Geiger-Nuttall half-life estimate."""
    try:
        from closures.nuclear_physics import compute_alpha_decay

        result = compute_alpha_decay(req.Z, req.A, req.Q_alpha_MeV)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="alpha_decay")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


@app.post("/nuclear/shell", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_shell(
    req: NucBindingRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute shell structure: magic number proximity and shell closure strength."""
    try:
        from closures.nuclear_physics import compute_shell

        result = compute_shell(req.Z, req.A)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="shell_structure")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


@app.post("/nuclear/fissility", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_fissility(
    req: NucBindingRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute fissility parameter Z²/A vs critical value."""
    try:
        from closures.nuclear_physics import compute_fissility

        result = compute_fissility(req.Z, req.A)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="fissility")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


@app.post("/nuclear/decay-chain", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_decay_chain(
    Z: int = Query(..., ge=2, le=120, description="Starting proton number"),
    A: int = Query(..., ge=4, le=300, description="Starting mass number"),
    max_steps: int = Query(default=14, ge=1, le=50, description="Maximum chain length"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute sequential decay chain from parent nuclide."""
    try:
        from closures.nuclear_physics.decay_chain import ChainStep, compute_decay_chain

        # Build a minimal single-step chain input for the closure
        steps: list[ChainStep] = [
            ChainStep(isotope=f"Z{Z}-A{A}", Z=Z, A=A, decay_mode="alpha", half_life_s=1.0, Q_MeV=0.0)
        ]
        result = compute_decay_chain(steps)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="decay_chain")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


@app.post("/nuclear/double-sided", response_model=ClosureResult, tags=["NUC"])
async def compute_nuclear_double_sided(
    req: NucBindingRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute double-sided collapse: fusion vs fission convergence on iron peak."""
    try:
        from closures.nuclear_physics import compute_binding, compute_double_sided

        # Compute BE/A first, then pass to double-sided
        binding = compute_binding(req.Z, req.A)
        result = compute_double_sided(req.Z, req.A, binding.BE_per_A)
        return ClosureResult(result=result._asdict(), domain="NUC", closure="double_sided_collapse")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Nuclear physics closures not available") from e


# ============================================================================
# Quantum Mechanics Endpoints
# ============================================================================


@app.post("/qm/collapse", response_model=ClosureResult, tags=["QM"])
async def compute_qm_collapse(
    req: QMCollapseRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute Born-rule wavefunction collapse: deviation, fidelity, purity."""
    try:
        from closures.quantum_mechanics.wavefunction_collapse import compute_wavefunction_collapse

        result = compute_wavefunction_collapse(req.psi_amplitudes, req.measurement_probs, req.observed_outcome_idx)
        return ClosureResult(result=result, domain="QM", closure="wavefunction_collapse")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


@app.post("/qm/entanglement", response_model=ClosureResult, tags=["QM"])
async def compute_qm_entanglement(
    req: QMEntanglementRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute entanglement metrics: concurrence, von Neumann entropy, Bell-CHSH parameter."""
    try:
        from closures.quantum_mechanics.entanglement import compute_entanglement

        result = compute_entanglement(req.rho_eigenvalues, req.bell_correlations)
        return ClosureResult(result=result, domain="QM", closure="entanglement")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


@app.post("/qm/tunneling", response_model=ClosureResult, tags=["QM"])
async def compute_qm_tunneling(
    req: QMTunnelingRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute quantum tunneling: WKB transmission coefficient."""
    try:
        from closures.quantum_mechanics.tunneling import compute_tunneling

        result = compute_tunneling(req.e_particle, req.v_barrier, req.barrier_width, req.particle_mass)
        return ClosureResult(result=result, domain="QM", closure="tunneling")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


@app.post("/qm/harmonic-oscillator", response_model=ClosureResult, tags=["QM"])
async def compute_qm_oscillator(
    n_quanta: int = Query(..., ge=0, description="Quantum number"),
    omega_freq: float = Query(..., description="Angular frequency (rad/s)"),
    e_observed: float = Query(..., description="Observed energy (eV)"),
    coherent_alpha: float = Query(default=0.0, description="Coherent state amplitude"),
    squeeze_r: float = Query(default=0.0, description="Squeeze parameter"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute quantized energy levels E_n = ℏω(n+½)."""
    try:
        from closures.quantum_mechanics.harmonic_oscillator import compute_harmonic_oscillator

        result = compute_harmonic_oscillator(n_quanta, omega_freq, e_observed, coherent_alpha, squeeze_r)
        return ClosureResult(result=result, domain="QM", closure="harmonic_oscillator")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


@app.post("/qm/spin", response_model=ClosureResult, tags=["QM"])
async def compute_qm_spin(
    s_total: float = Query(..., description="Total spin quantum number"),
    s_z_observed: float = Query(..., description="Observed S_z component"),
    b_field: float = Query(..., description="Magnetic field (T)"),
    g_factor: float | None = Query(default=None, description="Landé g-factor"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Compute spin measurement: Stern-Gerlach, Larmor frequency, Zeeman splitting."""
    try:
        from closures.quantum_mechanics.spin_measurement import compute_spin_measurement

        result = compute_spin_measurement(s_total, s_z_observed, b_field, g_factor)
        return ClosureResult(result=result, domain="QM", closure="spin_measurement")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


@app.post("/qm/uncertainty", response_model=ClosureResult, tags=["QM"])
async def compute_qm_uncertainty(
    delta_x: float = Query(..., description="Position uncertainty (nm)"),
    delta_p: float = Query(..., description="Momentum uncertainty"),
    units: str = Query(default="SI", description="Unit system: SI or natural"),
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Check Heisenberg uncertainty: Δx·Δp ≥ ℏ/2."""
    try:
        from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty

        result = compute_uncertainty(delta_x, delta_p, units)
        return ClosureResult(result=result, domain="QM", closure="uncertainty_principle")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="QM closures not available") from e


# ============================================================================
# Finance Endpoints
# ============================================================================


class FinanceEmbedRequest(BaseModel):
    """Request for finance embedding computation."""

    month: str = Field(description="Period identifier (e.g. 2026-01)")
    revenue: float = Field(description="Revenue ($)")
    expenses: float = Field(description="Expenses ($)")
    cogs: float = Field(description="Cost of goods sold ($)")
    cashflow: float = Field(description="Net cashflow ($)")
    revenue_target: float = Field(description="Revenue target ($)")
    expense_budget: float = Field(description="Expense budget ($)")
    cashflow_target: float = Field(description="Cashflow target ($)")


@app.post("/finance/embed", response_model=ClosureResult, tags=["FIN"])
async def compute_finance_embed(
    req: FinanceEmbedRequest,
    api_key: str = Security(require_public),
) -> ClosureResult:
    """Embed financial data into UMCP [0,1]⁴ coordinates and compute invariants."""
    try:
        from closures.finance.finance_embedding import FinanceRecord, FinanceTargets, embed_finance

        record = FinanceRecord(
            month=req.month,
            revenue=req.revenue,
            expenses=req.expenses,
            cogs=req.cogs,
            cashflow=req.cashflow,
        )
        targets = FinanceTargets(
            revenue_target=req.revenue_target,
            expense_budget=req.expense_budget,
            cashflow_target=req.cashflow_target,
        )
        embedded = embed_finance(record, targets)
        import math

        coords = list(embedded.c)
        weights = [0.30, 0.25, 0.25, 0.20]
        epsilon = EPSILON
        kappa = sum(w * math.log(max(c, epsilon)) for w, c in zip(weights, coords, strict=False))
        omega = 1.0 - sum(w * c for w, c in zip(weights, coords, strict=False))
        result: dict[str, Any] = {
            "coordinates": coords,
            "oor_flags": embedded.oor_flags,
            "omega": omega,
            "F": 1.0 - omega,
            "kappa": kappa,
            "IC": math.exp(kappa),
            "regime": "STABLE" if 0.3 <= omega <= 0.7 else "WATCH" if 0.1 <= omega <= 0.9 else "COLLAPSE",
        }
        return ClosureResult(result=result, domain="FIN", closure="finance_embedding")
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Finance closures not available") from e


# ============================================================================
# WEYL Cosmology Endpoints
# ============================================================================


class CosmologyParams(BaseModel):
    """Cosmological parameters for WEYL computations."""

    Omega_m_0: float = Field(default=0.315, description="Matter density today")
    Omega_Lambda_0: float = Field(default=0.685, description="Dark energy density today")
    H_0: float = Field(default=67.4, description="Hubble constant in km/s/Mpc")
    sigma8_0: float = Field(default=0.811, description="σ8 amplitude at z=0")


class SigmaResult(BaseModel):
    """Result of Σ(z) computation."""

    z: float = Field(description="Redshift")
    Sigma: float = Field(description="Σ(z) value")
    Sigma_0: float = Field(description="Σ₀ amplitude")
    regime: str = Field(description="Classification regime")
    deviation_from_GR: float = Field(description="|Σ - 1|")


class BackgroundResult(BaseModel):
    """Result of background cosmology computation."""

    z: float = Field(description="Redshift")
    H_z: float = Field(description="Hubble parameter H(z) in km/s/Mpc")
    chi: float = Field(description="Comoving distance χ(z) in Mpc/h")
    D1: float = Field(description="Linear growth function D₁(z)")
    sigma8_z: float = Field(description="σ8(z) amplitude")
    Omega_m_z: float = Field(description="Matter density Ω_m(z)")


class DESY3Data(BaseModel):
    """DES Y3 reference data."""

    z_bins: list[float] = Field(description="Effective redshifts for 4 lens bins")
    hJ_mean: list[float] = Field(description="ĥJ measurements (CMB prior)")
    hJ_sigma: list[float] = Field(description="ĥJ uncertainties")
    Sigma_0_standard: dict[str, float] = Field(description="Σ₀ fit with standard g(z)")
    Sigma_0_constant: dict[str, float] = Field(description="Σ₀ fit with constant g(z)")


class UMCPMapping(BaseModel):
    """WEYL to UMCP invariant mapping."""

    omega_analog: float = Field(description="Drift analog (1 - F)")
    F_analog: float = Field(description="Fidelity analog")
    regime: str = Field(description="UMCP-style regime")
    chi2_improvement: float = Field(description="χ² improvement ratio")


@app.get("/weyl/background", response_model=BackgroundResult, tags=["WEYL"])
async def compute_weyl_background(
    z: float = Query(..., ge=0.0, le=10.0, description="Redshift"),
    api_key: str = Security(require_public),
) -> BackgroundResult:
    """
    Compute background cosmology quantities at redshift z.

    Returns H(z), χ(z), D₁(z), σ8(z), Ω_m(z) for Planck 2018 cosmology.
    """
    try:
        from closures.weyl import (
            D1_of_z,
            H_of_z,
            Omega_m_of_z,
            chi_of_z,
            sigma8_of_z,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="WEYL closures not available",
        ) from e

    return BackgroundResult(
        z=z,
        H_z=float(H_of_z(z)),
        chi=float(chi_of_z(z)),
        D1=float(D1_of_z(z)),
        sigma8_z=float(sigma8_of_z(z)),
        Omega_m_z=float(Omega_m_of_z(z)),
    )


@app.get("/weyl/sigma", response_model=SigmaResult, tags=["WEYL"])
async def compute_weyl_sigma(
    z: float = Query(..., ge=0.0, le=3.0, description="Redshift"),
    Sigma_0: float = Query(default=0.24, ge=-1.0, le=1.0, description="Σ₀ amplitude"),
    g_model: str = Query(default="constant", description="g(z) model: constant, exponential, standard"),
    api_key: str = Security(require_public),
) -> SigmaResult:
    """
    Compute Σ(z) modified gravity parameter.

    Σ(z) = 1 + Σ₀ · g(z), where g(z) depends on the chosen model.
    DES Y3 finds Σ₀ ≈ 0.24 ± 0.14 (constant model).
    """
    try:
        from closures.weyl import GzModel, compute_Sigma
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="WEYL closures not available",
        ) from e

    try:
        model = GzModel(g_model)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid g_model: {g_model}. Use: constant, exponential, standard",
        ) from e

    result = compute_Sigma(z, Sigma_0, model)

    return SigmaResult(
        z=result.z,
        Sigma=result.Sigma,
        Sigma_0=result.Sigma_0,
        regime=result.regime,
        deviation_from_GR=result.deviation_from_GR,
    )


@app.get("/weyl/des-y3", response_model=DESY3Data, tags=["WEYL"])
async def get_des_y3_data(
    api_key: str = Security(require_public),
) -> DESY3Data:
    """
    Get DES Y3 reference data from Nature Communications 15:9295 (2024).

    Returns ĥJ measurements and Σ₀ fits for the 4 lens redshift bins.
    """
    try:
        from closures.weyl import DES_Y3_DATA
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="WEYL closures not available",
        ) from e

    return DESY3Data(
        z_bins=DES_Y3_DATA["z_bins"],
        hJ_mean=DES_Y3_DATA["hJ_cmb"]["mean"],
        hJ_sigma=DES_Y3_DATA["hJ_cmb"]["sigma"],
        Sigma_0_standard={
            "mean": DES_Y3_DATA["Sigma_0_fits"]["standard"]["mean"],
            "sigma": DES_Y3_DATA["Sigma_0_fits"]["standard"]["sigma"],
        },
        Sigma_0_constant={
            "mean": DES_Y3_DATA["Sigma_0_fits"]["constant"]["mean"],
            "sigma": DES_Y3_DATA["Sigma_0_fits"]["constant"]["sigma"],
        },
    )


@app.get("/weyl/umcp-mapping", response_model=UMCPMapping, tags=["WEYL"])
async def compute_weyl_umcp_mapping(
    Sigma_0: float = Query(default=0.24, ge=-1.0, le=1.0, description="Σ₀ amplitude"),
    chi2_Sigma: float = Query(default=1.1, ge=0.0, description="χ² of Σ model fit"),
    chi2_LCDM: float = Query(default=2.1, ge=0.0, description="χ² of ΛCDM fit"),
    api_key: str = Security(require_public),
) -> UMCPMapping:
    """
    Map WEYL Σ measurements to UMCP invariants.

    Returns ω (drift), F (fidelity), regime, and χ² improvement.
    """
    try:
        from closures.weyl import Sigma_to_UMCP_invariants
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="WEYL closures not available",
        ) from e

    result = Sigma_to_UMCP_invariants(Sigma_0, chi2_Sigma, chi2_LCDM)

    return UMCPMapping(
        omega_analog=result["omega_analog"],
        F_analog=result["F_analog"],
        regime=result["regime"],
        chi2_improvement=result["chi2_improvement"],
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the UMCP API server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        print("=" * 60)
        print("ERROR: uvicorn not installed")
        print("=" * 60)
        print("The API extension requires additional dependencies.")
        print("Install with: pip install umcp[api]")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("UMCP REST API Server")
    print("=" * 60)
    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"Version:  {__version__}")
    print(f"API:      v{API_VERSION}")
    print(f"Bind:     {host}:{port}")
    print(f"Docs:     http://{display_host}:{port}/docs")
    print(f"ReDoc:    http://{display_host}:{port}/redoc")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Use string import when reload is enabled (uvicorn requirement);
    # pass the app object directly otherwise to avoid double-importing
    # the module and duplicating middleware / router registration.
    app_ref: str | Any = "umcp.api_umcp:app" if reload else app
    uvicorn.run(  # type: ignore[no-untyped-call]
        app_ref,
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    """CLI entry point for umcp-api command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="UMCP REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  umcp-api                          # Start on default port 8000
  umcp-api --port 9000              # Start on custom port
  umcp-api --reload                 # Enable auto-reload for development
  umcp-api --host 127.0.0.1         # Bind to localhost only
""",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
