"""
Shared utilities, constants, and data loaders for UMCP Dashboard.

Contains:
  - Constants: REGIME_COLORS, KERNEL_SYMBOLS, STATUS_COLORS, THEMES
  - Data loaders: load_ledger, load_casepacks, load_contracts, load_closures
  - Helpers: classify_regime, get_regime_color, format_bytes, etc.
  - Path setup: _setup_closures_path, _ensure_closures_path, get_repo_root
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from umcp.dashboard._deps import _cache_data, np, pd

# Import UMCP core modules
try:
    from umcp import __version__
except ImportError:
    __version__ = "2.0.0"


def _setup_closures_path() -> None:
    """Add repo root to sys.path so closures package is importable.

    Works across editable installs, Docker, and Streamlit subprocess contexts.
    """
    import importlib
    from pathlib import Path

    # Check if closures is already importable via editable install
    spec = importlib.util.find_spec("closures")
    if spec is not None and spec.origin is not None:
        return  # Already resolvable — nothing to do

    # Find repo root (contains pyproject.toml)
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            repo_root = str(current)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return
        current = current.parent

    # Fallback: check common Docker paths
    for path in ["/app", "/workspaces/UMCP-Metadata-Runnable-Code"]:
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


_setup_closures_path()


def _ensure_closures_path() -> None:
    """Ensure closures/ package is importable (idempotent, safe to call multiple times).

    This is a safety net for Streamlit subprocesses where _setup_closures_path()
    may not have resolved the repo root correctly. Uses get_repo_root() which
    interrogates ``pyproject.toml`` location at runtime.
    """

    repo_root = get_repo_root()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


# ============================================================================
# Constants and Configuration
# ============================================================================

REGIME_COLORS = {
    "STABLE": "#28a745",  # Green
    "WATCH": "#ffc107",  # Yellow/Amber
    "COLLAPSE": "#dc3545",  # Red
    "CRITICAL": "#6f42c1",  # Purple
}

# Kernel invariants from KERNEL_SPECIFICATION.md
# Tier-1 outputs: F, ω, S, C, κ, IC computed from frozen trace Ψ(t)
KERNEL_SYMBOLS = {
    # Core Tier-1 Invariants
    "omega": "ω (Drift = 1-F)",
    "F": "F (Fidelity)",
    "S": "S (Bernoulli Field Entropy)",
    "C": "C (Curvature Proxy)",
    "tau_R": "τ_R (Return Time)",
    "IC": "IC (Integrity Composite = exp(κ))",
    "kappa": "κ (Log-Integrity = Σwᵢ ln cᵢ)",
    # Derived/Seam Values
    "stiffness": "Stiffness",
    "delta_kappa": "Δκ (Seam Curvature Change)",
    "curvature": "C (Curvature = std/0.5)",
    "freshness": "Freshness (1-ω)",
    "seam_residual": "s (Seam Residual)",
    # Meta
    "timestamp": "Timestamp",
    "run_status": "Status",
    "Phi_gen": "Φ_gen (Generative Flux)",
}

STATUS_COLORS = {
    "CONFORMANT": "#28a745",
    "NONCONFORMANT": "#dc3545",
    "NON_EVALUABLE": "#6c757d",
}

# Theme configurations
THEMES = {
    "Default": {
        "primary": "#007bff",
        "secondary": "#6c757d",
        "success": "#28a745",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        "bg_primary": "#ffffff",
        "bg_secondary": "#f8f9fa",
    },
    "Dark": {
        "primary": "#0d6efd",
        "secondary": "#adb5bd",
        "success": "#198754",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#0dcaf0",
        "bg_primary": "#212529",
        "bg_secondary": "#343a40",
    },
    "Ocean": {
        "primary": "#0077b6",
        "secondary": "#90e0ef",
        "success": "#2a9d8f",
        "danger": "#e63946",
        "warning": "#f4a261",
        "info": "#48cae4",
        "bg_primary": "#caf0f8",
        "bg_secondary": "#ade8f4",
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_repo_root() -> Path:
    """Find the repository root (contains pyproject.toml)."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


@_cache_data(ttl=60)
def load_ledger() -> Any:
    """Load the return log ledger as a DataFrame."""
    if pd is None:
        raise ImportError("pandas not installed. Run: pip install umcp[viz]")

    repo_root = get_repo_root()
    ledger_path = repo_root / "ledger" / "return_log.csv"

    if not ledger_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(ledger_path)

    # Ensure tau_R stays as string column — INF_REC is a typed sentinel
    # that cannot be coerced to int64 by PyArrow serialization.
    # Uses canonical tau_R_display for consistent formatting.
    if "tau_R" in df.columns:
        try:
            from .measurement_engine import tau_R_display
        except ImportError:
            from umcp.measurement_engine import tau_R_display  # type: ignore[no-redef]

        df["tau_R"] = df["tau_R"].apply(tau_R_display)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    return df


@_cache_data(ttl=60)
def load_casepacks() -> list[dict[str, Any]]:
    """Load casepack information with extended metadata."""
    repo_root = get_repo_root()
    casepacks_dir = repo_root / "casepacks"

    if not casepacks_dir.exists():
        return []

    casepacks: list[dict[str, Any]] = []
    for casepack_dir in sorted(casepacks_dir.iterdir()):
        if not casepack_dir.is_dir():
            continue

        manifest_path = casepack_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = casepack_dir / "manifest.yaml"

        casepack_info: dict[str, Any] = {
            "id": casepack_dir.name,
            "path": str(casepack_dir),
            "version": "unknown",
            "description": None,
            "contract": None,
            "closures_count": 0,
            "test_vectors": 0,
            "files_count": 0,
        }

        # Count files
        casepack_info["files_count"] = len(list(casepack_dir.rglob("*")))

        # Count closures
        closures_dir = casepack_dir / "closures"
        if closures_dir.exists():
            casepack_info["closures_count"] = len(list(closures_dir.glob("*.py")))

        # Count test vectors
        test_file = casepack_dir / "test_vectors.csv"
        if not test_file.exists():
            test_file = casepack_dir / "raw_measurements.csv"
        if test_file.exists():
            with open(test_file) as f:
                casepack_info["test_vectors"] = max(0, sum(1 for _ in f) - 1)

        # Load manifest
        if manifest_path.exists():
            try:
                if manifest_path.suffix == ".json":
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    import yaml

                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)

                if manifest and "casepack" in manifest:
                    cp = manifest["casepack"]
                    casepack_info["id"] = cp.get("id", casepack_dir.name)
                    casepack_info["version"] = cp.get("version", "unknown")
                    casepack_info["description"] = cp.get("description")
                    casepack_info["title"] = cp.get("title")
                if manifest and "refs" in manifest:
                    refs = manifest["refs"]
                    if "contract" in refs:
                        casepack_info["contract"] = refs["contract"].get("id")
            except Exception:
                pass

        casepacks.append(casepack_info)

    return casepacks


@_cache_data(ttl=60)
def load_contracts() -> list[dict[str, Any]]:
    """Load contract information with extended metadata."""
    repo_root = get_repo_root()
    contracts_dir = repo_root / "contracts"

    if not contracts_dir.exists():
        return []

    contracts: list[dict[str, Any]] = []
    for contract_path in sorted(contracts_dir.glob("*.yaml")):
        filename = contract_path.stem
        parts = filename.split(".")
        domain = parts[0] if parts else "unknown"
        version = parts[-1] if len(parts) > 1 and parts[-1].startswith("v") else "v1"

        # Get file size
        size_bytes = contract_path.stat().st_size

        contracts.append(
            {
                "id": filename,
                "domain": domain,
                "version": version,
                "path": str(contract_path),
                "size_bytes": size_bytes,
            }
        )

    return contracts


@_cache_data(ttl=60)
def load_closures() -> list[dict[str, Any]]:
    """Load closure information."""
    repo_root = get_repo_root()
    closures_dir = repo_root / "closures"

    if not closures_dir.exists():
        return []

    closures: list[dict[str, Any]] = []

    def _infer_domain(path: Path) -> str:
        """Infer domain from file name or parent directory name."""
        # Check parent directory name first (e.g., closures/gcd/, closures/rcft/)
        rel = path.relative_to(closures_dir)
        parts_lower = [p.lower() for p in rel.parts]
        combined = " ".join(parts_lower)
        if "gcd" in combined or "curvature" in combined or "gamma" in combined:
            return "GCD"
        if "kin" in combined:
            return "KIN"
        if "rcft" in combined:
            return "RCFT"
        if "weyl" in combined:
            return "WEYL"
        if "security" in combined:
            return "SECURITY"
        if "astro" in combined:
            return "ASTRO"
        if "nuclear" in combined or "nuc" in combined:
            return "NUC"
        if "quantum" in combined or "qm" in combined:
            return "QM"
        if "finance" in combined or "fin" in combined:
            return "FIN"
        return "unknown"

    # Python closures (recursive)
    for closure_path in sorted(closures_dir.rglob("*.py")):
        if closure_path.name.startswith("_"):
            continue

        name = closure_path.stem
        size_bytes = closure_path.stat().st_size
        domain = _infer_domain(closure_path)

        # Count lines
        with open(closure_path) as f:
            lines = len(f.readlines())

        closures.append(
            {
                "name": name,
                "domain": domain,
                "path": str(closure_path),
                "type": "python",
                "size_bytes": size_bytes,
                "lines": lines,
            }
        )

    # YAML closures (recursive)
    for closure_path in sorted(closures_dir.rglob("*.yaml")):
        if closure_path.name == "registry.yaml":
            continue

        name = closure_path.stem
        size_bytes = closure_path.stat().st_size
        domain = _infer_domain(closure_path)

        closures.append(
            {
                "name": name,
                "domain": domain,
                "path": str(closure_path),
                "type": "yaml",
                "size_bytes": size_bytes,
                "lines": 0,
            }
        )

    return closures


def classify_regime(omega: float, seam_residual: float = 0.0) -> str:
    """
    Classify the computational regime based on kernel invariants.

    Regimes (from KERNEL_SPECIFICATION.md):
      - STABLE: ω ∈ [0.3, 0.7], |s| ≤ 0.005
      - WATCH: ω ∈ [0.1, 0.3) ∪ (0.7, 0.9], |s| ≤ 0.01
      - COLLAPSE: ω < 0.1 or ω > 0.9
      - CRITICAL: |s| > 0.01
    """
    if abs(seam_residual) > 0.01:
        return "CRITICAL"
    if omega < 0.1 or omega > 0.9:
        return "COLLAPSE"
    if 0.3 <= omega <= 0.7:
        return "STABLE"
    return "WATCH"


def get_regime_color(regime: str) -> str:
    """Get color for regime visualization."""
    return REGIME_COLORS.get(regime, "#6c757d")


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size:.1f} TB"


def get_trend_indicator(current: float, previous: float, invert: bool = False) -> str:
    """Get trend arrow indicator."""
    threshold = 1.01
    if current > previous * threshold:
        return "📉" if invert else "📈"
    elif current < previous / threshold:
        return "📈" if invert else "📉"
    return "➡️"


def detect_anomalies(series: Any, threshold: float = 2.5) -> Any:
    """Detect anomalies using z-score method."""
    if pd is None or np is None:
        return []
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return abs(z_scores) > threshold


# ============================================================================
# Dashboard Pages
# ============================================================================
