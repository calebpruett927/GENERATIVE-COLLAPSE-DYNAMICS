"""UMCP Extensions — Default + On-Demand Plugin System

Extensions are split into two tiers:

  **Default** (loaded eagerly at startup, no optional deps):
    1. ledger          — Append-only validation logging
    2. formatter       — Contract auto-formatter
    3. thermodynamics  — τ_R* thermodynamic diagnostic

  **On-demand** (lazy-loaded on first use, may need optional deps):
    4. api             — REST API server (FastAPI / uvicorn)
    5. visualization   — Streamlit dashboard

Usage::

    from umcp.umcp_extensions import manager

    # Default extensions are already loaded
    manager.status()                   # Show what's loaded

    # On-demand: auto-loads on first access
    api_mod = manager.get("api")       # Lazy-loads if deps present

    # Or via CLI:
    #   umcp-ext list                  # List all (shows Default / On-demand)
    #   umcp-ext info api              # Details for one extension
    #   umcp-ext run visualization     # Launch dashboard

Cross-references:
  - src/umcp/api_umcp.py       (REST API extension)
  - src/umcp/dashboard/        (Streamlit dashboard extension)
  - pyproject.toml             (optional dependencies: api, viz extras)
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Extension Protocol
# ============================================================================


@runtime_checkable
class Extension(Protocol):
    """Protocol for UMCP extensions."""

    name: str
    description: str

    def run(self) -> None:
        """Run the extension."""
        ...

    def is_available(self) -> bool:
        """Check if extension dependencies are available."""
        ...


# ============================================================================
# Extension Registry
# ============================================================================


@dataclass
class ExtensionInfo:
    """Information about a registered extension."""

    name: str
    description: str
    type: str  # 'api', 'dashboard', 'logging', 'tool', 'validator'
    module: str
    default: bool = False  # True → loaded at startup; False → lazy / on-demand
    class_name: str | None = None
    requires: list[str] = field(default_factory=list)
    command: str | None = None
    port: int | None = None
    endpoints: list[dict[str, str]] = field(default_factory=list)
    features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "module": self.module,
            "default": self.default,
            "class": self.class_name,
            "requires": self.requires,
            "command": self.command,
            "port": self.port,
            "endpoints": self.endpoints,
            "features": self.features,
        }


# ---------------------------------------------------------------------------
#  Built-in extension registry
# ---------------------------------------------------------------------------
EXTENSIONS: dict[str, ExtensionInfo] = {
    # ── Default extensions (eager, no optional deps) ──────────────────────
    "ledger": ExtensionInfo(
        name="ledger",
        description="Continuous validation logging to ledger/return_log.csv",
        type="logging",
        module="umcp.validator",
        default=True,
        requires=[],
        features=[
            "Append-only ledger",
            "Validation receipts",
            "Audit trail",
        ],
    ),
    "formatter": ExtensionInfo(
        name="formatter",
        description="Contract auto-formatter and validator",
        type="tool",
        module="umcp.validator",
        default=True,
        requires=["pyyaml"],
        features=[
            "YAML formatting",
            "Schema validation",
            "Contract linting",
        ],
    ),
    "thermodynamics": ExtensionInfo(
        name="thermodynamics",
        description="τ_R* thermodynamic diagnostic — Tier-2 budget analysis with Tier-0 checks",
        type="validator",
        module="umcp.tau_r_star",
        default=True,
        class_name="ThermodynamicDiagnostic",
        requires=[],
        features=[
            "τ_R* critical return delay computation",
            "R_critical / R_min estimation",
            "Thermodynamic phase classification (surplus/deficit/trapped/pole)",
            "Budget dominance analysis (drift/curvature/memory)",
            "Trapping threshold computation",
            "Tier-1 identity verification (F=1−ω, IC≈exp(κ), IC≤F)",
            "Batch invariant diagnostics",
            "Prediction verification (§6 testable predictions)",
        ],
    ),
    # ── On-demand extensions (lazy, optional deps) ────────────────────────
    "api": ExtensionInfo(
        name="api",
        description="REST API server for remote validation and ledger access",
        type="api",
        module="umcp.api_umcp",
        default=False,
        class_name="app",
        requires=["fastapi", "uvicorn"],
        command="uvicorn umcp.api_umcp:app --reload --host 0.0.0.0 --port 8000",
        port=8000,
        endpoints=[
            {"method": "GET", "path": "/health", "description": "System health check"},
            {"method": "GET", "path": "/version", "description": "Version info"},
            {"method": "POST", "path": "/validate", "description": "Run validation"},
            {"method": "GET", "path": "/casepacks", "description": "List casepacks"},
            {"method": "GET", "path": "/ledger", "description": "Query ledger"},
            {"method": "GET", "path": "/contracts", "description": "List contracts"},
            {"method": "GET", "path": "/closures", "description": "List closures"},
            {"method": "POST", "path": "/regime", "description": "Classify regime"},
        ],
        features=[
            "RESTful API",
            "API key authentication",
            "OpenAPI documentation",
            "CORS support",
            "Health monitoring",
        ],
    ),
    "visualization": ExtensionInfo(
        name="visualization",
        description="Interactive Streamlit dashboard for exploring UMCP data",
        type="dashboard",
        module="umcp.dashboard",
        default=False,
        class_name="main",
        requires=["streamlit", "pandas", "plotly"],
        command="streamlit run src/umcp/dashboard.py",
        port=8501,
        features=[
            "Real-time system health",
            "Ledger exploration",
            "Regime visualization",
            "Kernel metrics analysis",
            "Casepack browser",
            "Contract explorer",
        ],
    ),
}


# Mapping from package names to import names (for packages where they differ)
_IMPORT_NAMES: dict[str, str] = {
    "pyyaml": "yaml",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "pandas": "pandas",
    "plotly": "plotly",
}


# ============================================================================
# Extension Manager — singleton orchestrator
# ============================================================================


class ExtensionManager:
    """Manages default and on-demand extensions.

    * Default extensions are loaded eagerly when ``startup()`` is called.
    * On-demand extensions are lazy-loaded via ``get()`` on first access.
    * Once loaded, modules are cached — subsequent calls return the same object.

    Usage::

        from umcp.umcp_extensions import manager

        manager.startup()              # Called once (e.g. from __init__)
        mod = manager.get("api")       # Lazy-loads on first call
        manager.status()               # Print what's loaded
    """

    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}  # name → module (or None if load failed)
        self._started: bool = False

    # ── Startup (eager defaults) ──────────────────────────────────────────

    def startup(self) -> list[str]:
        """Load all default extensions eagerly.

        Returns:
            Names of extensions successfully loaded.
        """
        loaded: list[str] = []
        for name, info in EXTENSIONS.items():
            if info.default:
                mod = self._do_load(name)
                if mod is not None:
                    loaded.append(name)
        self._started = True
        return loaded

    # ── On-demand access ──────────────────────────────────────────────────

    def get(self, name: str) -> Any | None:
        """Get an extension module, lazy-loading on first access.

        Args:
            name: Registered extension name.

        Returns:
            The extension module, or ``None`` if unavailable.
        """
        if name in self._loaded:
            return self._loaded[name]  # cached (may be None if previously failed)
        return self._do_load(name)

    # ── Query helpers ─────────────────────────────────────────────────────

    def is_loaded(self, name: str) -> bool:
        """True if *name* is already loaded (not just registered)."""
        return name in self._loaded and self._loaded[name] is not None

    @property
    def loaded_names(self) -> list[str]:
        """Names of currently-loaded extensions."""
        return [n for n, m in self._loaded.items() if m is not None]

    @property
    def available_names(self) -> list[str]:
        """Names of all registered extensions (loaded or not)."""
        return list(EXTENSIONS.keys())

    @property
    def default_names(self) -> list[str]:
        """Names of extensions marked as default."""
        return [n for n, info in EXTENSIONS.items() if info.default]

    @property
    def on_demand_names(self) -> list[str]:
        """Names of extensions NOT marked as default (lazy / on-demand)."""
        return [n for n, info in EXTENSIONS.items() if not info.default]

    def status(self) -> dict[str, Any]:
        """Return a status summary dict (also printable).

        Returns:
            Dictionary with 'default', 'on_demand', 'loaded' keys.
        """
        return {
            "started": self._started,
            "default": {n: {"loaded": self.is_loaded(n), "deps_ok": check_extension(n)} for n in self.default_names},
            "on_demand": {
                n: {"loaded": self.is_loaded(n), "deps_ok": check_extension(n)} for n in self.on_demand_names
            },
            "loaded": self.loaded_names,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _do_load(self, name: str) -> Any | None:
        """Attempt to import and cache an extension module."""
        if name not in EXTENSIONS:
            return None
        if not check_extension(name):
            self._loaded[name] = None
            return None
        info = EXTENSIONS[name]
        try:
            mod = importlib.import_module(info.module)
            self._loaded[name] = mod
            return mod
        except ImportError:
            self._loaded[name] = None
            return None

    def reset(self) -> None:
        """Clear all cached state (mainly for testing)."""
        self._loaded.clear()
        self._started = False


# Module-level singleton
manager = ExtensionManager()


# ============================================================================
# Backward-compatible module-level functions
# ============================================================================


def list_extensions(ext_type: str | None = None) -> list[dict[str, Any]]:
    """List all available extensions.

    Args:
        ext_type: Optional filter by type ('api', 'dashboard', 'logging', 'tool')

    Returns:
        List of extension info dictionaries
    """
    result = []
    for _name, info in EXTENSIONS.items():
        if ext_type is None or info.type == ext_type:
            result.append(info.to_dict())
    return result


def get_extension_info(name: str) -> dict[str, Any]:
    """Get detailed information about an extension.

    Args:
        name: Extension name to query

    Returns:
        Dictionary with extension metadata
    """
    if name in EXTENSIONS:
        return EXTENSIONS[name].to_dict()

    return {
        "name": name,
        "status": "not_found",
        "info": f"Extension '{name}' not found. Available: {', '.join(EXTENSIONS.keys())}",
    }


def check_extension(name: str) -> bool:
    """Check if an extension's dependencies are installed.

    Args:
        name: Extension name

    Returns:
        True if all dependencies are available
    """
    if name not in EXTENSIONS:
        return False

    info = EXTENSIONS[name]
    for req in info.requires:
        import_name = _IMPORT_NAMES.get(req, req)
        try:
            importlib.import_module(import_name)
        except ImportError:
            return False

    return True


def install_extension(name: str) -> bool:
    """Install extension dependencies.

    Args:
        name: Extension name

    Returns:
        True if installation succeeded
    """
    if name not in EXTENSIONS:
        return False

    info = EXTENSIONS[name]
    if not info.requires:
        return True

    extras_map = {
        "api": "api",
        "visualization": "viz",
        "ledger": "",
        "formatter": "",
    }

    extra = extras_map.get(name, "")
    if not extra:
        return True

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"umcp[{extra}]"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def load_extension(name: str) -> Any | None:
    """Load an extension module (delegates to manager for caching).

    Args:
        name: Extension name

    Returns:
        Extension module or None if failed
    """
    return manager.get(name)


def run_extension(name: str) -> int:
    """Run an extension.

    Args:
        name: Extension name

    Returns:
        Exit code (0 for success)
    """
    if name not in EXTENSIONS:
        print(f"Extension '{name}' not found")
        return 1

    if not check_extension(name):
        print(f"Extension '{name}' dependencies not installed")
        print(f"Run: pip install umcp[{name}]")
        return 1

    info = EXTENSIONS[name]
    if not info.command:
        print(f"Extension '{name}' has no run command")
        return 1

    try:
        result = subprocess.run(
            info.command.split(),
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nExtension stopped")
        return 0
    except Exception as e:
        print(f"Error running extension: {e}")
        return 1


# ============================================================================
# CLI Entry Point
# ============================================================================


def _print_status_table() -> None:
    """Print a human-readable extension status table."""
    st = manager.status()
    print("\n" + "=" * 70)
    print(" " * 20 + "UMCP EXTENSION STATUS")
    print("=" * 70)

    def _row(name: str, info: dict[str, Any], tier: str) -> None:
        loaded = "LOADED" if info["loaded"] else "—"
        deps = "deps OK" if info["deps_ok"] else "deps MISSING"
        print(f"  {name:<20s} {tier:<12s} {loaded:<10s} {deps}")

    print(f"\n  {'NAME':<20s} {'TIER':<12s} {'STATE':<10s} DEPS")
    print("  " + "-" * 56)
    for n, d in st["default"].items():
        _row(n, d, "default")
    for n, d in st["on_demand"].items():
        _row(n, d, "on-demand")

    print(f"\n  Currently loaded: {', '.join(st['loaded']) or '(none)'}")
    print("=" * 70 + "\n")


def main() -> int:
    """CLI entry point for extension manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="UMCP Extension Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  umcp-ext list                    # List all extensions
  umcp-ext list --type dashboard   # List dashboard extensions only
  umcp-ext status                  # Show loaded / on-demand status
  umcp-ext info api                # Show API extension details
  umcp-ext install api             # Install API dependencies
  umcp-ext run visualization       # Run visualization dashboard
  umcp-ext check api               # Check if API is installed
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    list_parser = subparsers.add_parser("list", help="List all extensions")
    list_parser.add_argument("--type", "-t", help="Filter by extension type")

    # status
    subparsers.add_parser("status", help="Show loaded / on-demand status")

    # info
    info_parser = subparsers.add_parser("info", help="Show extension details")
    info_parser.add_argument("name", help="Extension name")

    # install
    install_parser = subparsers.add_parser("install", help="Install extension")
    install_parser.add_argument("name", help="Extension name")

    # check
    check_parser = subparsers.add_parser("check", help="Check if extension is installed")
    check_parser.add_argument("name", help="Extension name")

    # run
    run_parser = subparsers.add_parser("run", help="Run an extension")
    run_parser.add_argument("name", help="Extension name")

    # Suppress unused-variable warnings for sub-parsers used only for argparse registration
    _ = (list_parser, info_parser, install_parser, check_parser, run_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "status":
        _print_status_table()
        return 0

    if args.command == "list":
        extensions = list_extensions(args.type)
        print("\n" + "=" * 70)
        print(" " * 25 + "UMCP EXTENSIONS")
        print("=" * 70)

        if not extensions:
            print("\nNo extensions found.\n")
            return 0

        for ext in extensions:
            installed = check_extension(ext["name"])
            status = "✅ INSTALLED" if installed else "❌ NOT INSTALLED"
            tier = "default" if ext.get("default") else "on-demand"

            print(f"\n📦 {ext['name']}  ({tier})")
            print(f"   Status:      {status}")
            print(f"   Type:        {ext['type']}")
            print(f"   Description: {ext['description']}")
            if ext.get("command"):
                print(f"   Command:     {ext['command']}")
            if ext.get("requires"):
                print(f"   Requires:    {', '.join(ext['requires'])}")

        print("\n" + "=" * 70)
        print(f"Total: {len(extensions)} extensions\n")
        return 0

    elif args.command == "info":
        info = get_extension_info(args.name)
        if info.get("status") == "not_found":
            print(f"❌ {info['info']}")
            return 1

        installed = check_extension(args.name)
        tier = "default" if info.get("default") else "on-demand"
        print("\n" + "=" * 70)
        print(f"Extension: {info['name']}  ({tier})")
        print("=" * 70)
        print(f"\nStatus:      {'✅ INSTALLED' if installed else '❌ NOT INSTALLED'}")
        print(f"Type:        {info['type']}")
        print(f"Description: {info['description']}")
        print(f"Module:      {info['module']}")

        if info.get("command"):
            print(f"Command:     {info['command']}")
        if info.get("port"):
            print(f"Port:        {info['port']}")
        if info.get("requires"):
            print("\nDependencies:")
            for dep in info["requires"]:
                print(f"  • {dep}")
        if info.get("endpoints"):
            print("\nAPI Endpoints:")
            for ep in info["endpoints"]:
                print(f"  • {ep['method']} {ep['path']} - {ep['description']}")
        if info.get("features"):
            print("\nFeatures:")
            for feature in info["features"]:
                print(f"  • {feature}")

        print("\n" + "=" * 70 + "\n")
        return 0

    elif args.command == "install":
        print(f"Installing dependencies for '{args.name}'...")
        if install_extension(args.name):
            print(f"✅ Successfully installed '{args.name}'")
            return 0
        else:
            print(f"❌ Failed to install '{args.name}'")
            return 1

    elif args.command == "check":
        installed = check_extension(args.name)
        if installed:
            print(f"✅ Extension '{args.name}' is installed")
            return 0
        else:
            print(f"❌ Extension '{args.name}' is not installed")
            print(f"\nTo install: umcp-ext install {args.name}")
            return 1

    elif args.command == "run":
        return run_extension(args.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
