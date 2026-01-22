def validate_api_key(api_key: str = Security(api_key_header)):
def verify_api_key(api_key: str = Security(api_key_header)):
def get_repo_root() -> Path:
def classify_regime(omega: float, F: float, S: float, C: float) -> str:
from datetime import datetime
