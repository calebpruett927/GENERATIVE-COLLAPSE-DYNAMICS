"""
UMCP REST API - FastAPI Communication Extension

Provides HTTP endpoints for remote validation, ledger access, and system health.
This is an optional extension that requires: pip install umcp[api]

Endpoints:
  GET  /health           - System health check
  GET  /version          - API and validator version info
  POST /validate         - Validate a casepack or repository
  GET  /casepacks        - List available casepacks
  GET  /casepacks/{id}   - Get casepack details
  POST /casepacks/{id}/run - Run a casepack
  GET  /ledger           - Query the return log ledger
  GET  /contracts        - List available contracts
  GET  /closures         - List available closures

Usage:
  uvicorn umcp.api_umcp:app --reload --host 0.0.0.0 --port 8000

Cross-references:
  - EXTENSION_INTEGRATION.md (extension architecture)
  - src/umcp/cli.py (CLI commands this mirrors)
  - src/umcp/validator.py (validation engine)
  - src/umcp/preflight.py (preflight validation)
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Import UMCP core modules
try:
    from . import __version__
except ImportError:
    # Fallback for direct execution
    __version__ = "1.5.0"

# ============================================================================
# Configuration
# ============================================================================

API_VERSION = "1.0.0"
API_TITLE = "UMCP REST API"
API_DESCRIPTION = """
Universal Measurement Contract Protocol REST API.

Provides HTTP endpoints for validating computational workflows,
querying the ledger, and managing casepacks.

**Authentication**: API key required via `X-API-Key` header.
"""

# API key from environment (production should use secrets management)
API_KEY = os.environ.get("UMCP_API_KEY", "umcp-dev-key")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_repo_root() -> Path:
    """Find the repository root (contains pyproject.toml)."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def verify_api_key(api_key: str | None = Security(api_key_header)) -> bool:
    """Verify the API key."""
    if api_key is None:
        return False
    return api_key == API_KEY


def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key and return it, or raise 401."""
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
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Utility Functions
# ============================================================================


def get_current_time() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """
    Classify the computational regime based on kernel invariants.

    Regimes (from KERNEL_SPECIFICATION.md):
      - STABLE: ω ∈ [0.3, 0.7], |s| ≤ 0.005
      - WATCH: ω ∈ [0.1, 0.3) ∪ (0.7, 0.9], |s| ≤ 0.01
      - COLLAPSE: ω < 0.1 or ω > 0.9
      - CRITICAL: |s| > 0.01

    Args:
        omega: Overlap fraction ω
        F: Freshness F = 1 - ω
        S: Seam residual (budget deviation)
        C: Curvature κ

    Returns:
        Regime classification string
    """
    # Seam-based critical overlay
    if abs(S) > 0.01:
        return "CRITICAL"

    # Omega-based primary classification
    if omega < 0.1 or omega > 0.9:
        return "COLLAPSE"
    elif 0.3 <= omega <= 0.7:
        return "STABLE"
    else:
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


@app.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    """API root - returns basic info."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
    api_key: str = Security(validate_api_key),
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
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore[import-untyped]

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # type: ignore[no-untyped-call]
    except ImportError:
        print("uvicorn not installed. Run: pip install umcp[api]")
        sys.exit(1)
