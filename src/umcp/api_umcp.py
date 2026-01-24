"""STUB: Placeholder for future REST API communication extension.

This file contains helper functions that would be used by a full FastAPI
implementation. The actual REST API is planned but not yet implemented.

For actual UMCP validation, use the core CLI:
  umcp validate <path>

This stub exists to:
1. Reserve the module name for future use
2. Provide example helper functions for API development
3. Support tests that check for optional dependencies

STATUS: Not functional - helpers only
FUTURE: Would require full FastAPI app with endpoints
"""

from datetime import datetime
from pathlib import Path

from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")


def validate_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Validate API key."""
    result: bool = api_key == "expected_key"
    return result


def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key."""
    result: bool = api_key == "expected_key"
    return result


def get_repo_root() -> Path:
    return Path(__file__).parent.resolve()


def classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega > 0 and F > 0 and S > 0 and C > 0:
        return "regime-positive"
    elif omega < 0 or F < 0 or S < 0 or C < 0:
        return "regime-negative"
    return "regime-unknown"


def get_current_time() -> str:
    return datetime.now().isoformat()
