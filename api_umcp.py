
"""
UMCP Public Audit API

FastAPI endpoint exposing UMCP validation receipts and regime statistics.

Usage:
    uvicorn api_umcp:app --reload

Endpoints:
    GET /health - Health check
    GET /latest-receipt - Latest validation receipt
    GET /ledger - Historical validation ledger
    GET /stats - Aggregate statistics
    GET /regime - Current regime classification

Authentication:
    All endpoints are public by default. No API token required for local/dev use.
    For production, add authentication (see QUICKSTART_EXTENSIONS.md for JWT example).
"""
#!/usr/bin/env python3

__version__ = "1.0.0"

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

# --- API Key Authentication (Production) ---
# Set your API key here (for demo; use env vars in production)
API_KEY = "your-api-key-here"

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError("Please install FastAPI: pip install fastapi uvicorn") from e


app = FastAPI(
    title="UMCP Audit API",
    description="Public API for UMCP validation receipts and regime statistics",
    version="1.0.0",
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class RegimeInfo(BaseModel):
    regime: str
    omega: float
    F: float
    S: float
    C: float
    timestamp: str | None = None


class StatsResponse(BaseModel):
    total_validations: int
    conformant_count: int
    nonconformant_count: int
    current_regime: str
    regime_distribution: dict[str, int]
    latest_timestamp: str | None = None


def get_repo_root() -> Path:
    """Find repository root by looking for pyproject.toml"""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Classify regime based on UMCP thresholds"""
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    elif omega >= 0.30:
        return "Collapse"
    else:
        return "Watch"


@app.get("/", response_model=dict[str, str])
async def root():
    """API root with available endpoints"""
    return {
        "message": "UMCP Public Audit API",
        "endpoints": {
            "/health": "Health check",
            "/latest-receipt": "Latest validation receipt",
            "/ledger": "Historical validation ledger",
            "/stats": "Aggregate statistics",
            "/regime": "Current regime classification",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(status="ok", timestamp=datetime.now(UTC).isoformat(), version="1.0.0")


@app.get("/latest-receipt")
async def latest_receipt():
    """Get the most recent validation receipt"""
    repo_root = get_repo_root()
    receipt_path = repo_root / "receipt.json"

    if not receipt_path.exists():
        raise HTTPException(status_code=404, detail="No receipt found. Run 'umcp validate' first.")

    try:
        with open(receipt_path) as f:
            receipt = json.load(f)
        return JSONResponse(content=receipt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading receipt: {e!s}") from e


@app.get("/ledger")
async def get_ledger():
    """Get historical validation ledger"""
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No ledger found. Run 'umcp validate' to start collecting data.",
        )

    try:
        with open(ledger_path) as f:
            reader = csv.DictReader(f)
            records = list(reader)

        return JSONResponse(content={"records": records, "count": len(records)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ledger: {e!s}") from e


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get aggregate validation statistics"""
    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No ledger found. Run 'umcp validate' to start collecting data.",
        )

    try:
        with open(ledger_path) as f:
            reader = csv.DictReader(f)
            records = list(reader)

        if not records:
            raise HTTPException(status_code=404, detail="Ledger is empty")

        # Count statuses
        total = len(records)
        conformant = sum(1 for r in records if r["run_status"] == "CONFORMANT")
        nonconformant = total - conformant

        # Classify regimes
        regime_counts = {"Stable": 0, "Watch": 0, "Collapse": 0, "Unknown": 0}
        for record in records:
            try:
                omega = float(record["omega"]) if record["omega"] else 0
                F = 1.0 - omega
                S = float(record["stiffness"]) if record["stiffness"] else 0
                C = float(record["curvature"]) if record["curvature"] else 0
                regime = classify_regime(omega, F, S, C)
                regime_counts[regime] += 1
            except (ValueError, KeyError):
                regime_counts["Unknown"] += 1

        # Get current regime from latest record
        latest = records[-1]
        try:
            omega = float(latest["omega"]) if latest["omega"] else 0
            F = 1.0 - omega
            S = float(latest["stiffness"]) if latest["stiffness"] else 0
            C = float(latest["curvature"]) if latest["curvature"] else 0
            current_regime = classify_regime(omega, F, S, C)
        except (ValueError, KeyError):
            current_regime = "Unknown"

        return StatsResponse(
            total_validations=total,
            conformant_count=conformant,
            nonconformant_count=nonconformant,
            current_regime=current_regime,
            regime_distribution=regime_counts,
            latest_timestamp=latest.get("timestamp"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing stats: {e!s}") from e


@app.get("/regime", response_model=RegimeInfo)
async def get_current_regime(api_key: str = Depends(verify_api_key)):
    """Get current regime classification with invariants (API key required in production)"""
    repo_root = get_repo_root()
    inv_path = repo_root / "outputs" / "invariants.csv"

    if not inv_path.exists():
        raise HTTPException(status_code=404, detail="No invariants found. Run 'umcp validate' first.")

    try:
        with open(inv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise HTTPException(status_code=404, detail="Invariants file is empty")

        inv = rows[0]
        omega = float(inv.get("omega", 0))
        F = float(inv.get("F", 0))
        S = float(inv.get("S", 0))
        C = float(inv.get("C", 0))

        regime = classify_regime(omega, F, S, C)

        return RegimeInfo(
            regime=regime,
            omega=omega,
            F=F,
            S=S,
            C=C,
            timestamp=datetime.now(UTC).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading regime: {e!s}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# UMCP Extension Entry Points
def run_server():
    """Entry point for umcp-api command"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


class UMCPAuditAPI:
    """UMCP Extension: Public Audit API

    Provides REST API access to UMCP validation receipts and regime statistics.
    Automatically registered with UMCP extension system.

    Attributes:
        name: UMCP extension name
        version: Extension version
        description: Extension description
        requires: Required dependencies
    """

    name = "audit-api"
    version = "1.0.0"
    description = "Public REST API for UMCP validation receipts and regime statistics"
    requires: ClassVar[list[str]] = ["fastapi>=0.109.0", "uvicorn[standard]>=0.27.0"]

    @staticmethod
    def install():
        """Install extension dependencies"""
        import subprocess

        subprocess.check_call(["pip", "install", *UMCPAuditAPI.requires])

    @staticmethod
    def run():
        """Run the extension"""
        run_server()

    @staticmethod
    def info():
        """Return extension metadata"""
        return {
            "name": UMCPAuditAPI.name,
            "version": UMCPAuditAPI.version,
            "description": UMCPAuditAPI.description,
            "requires": UMCPAuditAPI.requires,
            "endpoints": [
                {"path": "/health", "method": "GET", "description": "Health check"},
                {
                    "path": "/latest-receipt",
                    "method": "GET",
                    "description": "Latest validation receipt",
                },
                {
                    "path": "/ledger",
                    "method": "GET",
                    "description": "Historical validation ledger",
                },
                {
                    "path": "/stats",
                    "method": "GET",
                    "description": "Aggregate statistics",
                },
                {
                    "path": "/regime",
                    "method": "GET",
                    "description": "Current regime classification",
                },
            ],
        }
