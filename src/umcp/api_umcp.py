from datetime import datetime
from pathlib import Path

from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")


def validate_api_key(api_key: str = Security(api_key_header)):
    return api_key == "expected_key"


def verify_api_key(api_key: str = Security(api_key_header)):
    return api_key == "expected_key"


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
