"""UMCP Extensions - Production Plugin System

This module provides the complete extension/plugin system for UMCP.
Extensions enable optional communication and visualization features:

Built-in Extensions:
  1. api          - REST API server (FastAPI)
  2. visualization- Streamlit dashboard
  3. ledger       - Continuous ledger logging
  4. formatter    - Contract auto-formatter

Cross-references:
  - EXTENSION_INTEGRATION.md (architecture documentation)
  - src/umcp/api_umcp.py (REST API extension)
  - src/umcp/dashboard.py (Streamlit dashboard extension)
  - pyproject.toml (optional dependencies)
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
    """Information about an extension."""

    name: str
    description: str
    type: str  # 'api', 'dashboard', 'logging', 'tool', 'closure', 'validator'
    module: str
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
            "class": self.class_name,
            "requires": self.requires,
            "command": self.command,
            "port": self.port,
            "endpoints": self.endpoints,
            "features": self.features,
        }


# Built-in extension registry
EXTENSIONS: dict[str, ExtensionInfo] = {
    "api": ExtensionInfo(
        name="api",
        description="REST API server for remote validation and ledger access",
        type="api",
        module="umcp.api_umcp",
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
    "ledger": ExtensionInfo(
        name="ledger",
        description="Continuous validation logging to ledger/return_log.csv",
        type="logging",
        module="umcp.validator",
        class_name=None,
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
        class_name=None,
        requires=["pyyaml"],
        features=[
            "YAML formatting",
            "Schema validation",
            "Contract linting",
        ],
    ),
}


# ============================================================================
# Extension Discovery Functions
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


# Mapping from package names to import names (for packages where they differ)
_IMPORT_NAMES: dict[str, str] = {
    "pyyaml": "yaml",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "pandas": "pandas",
    "plotly": "plotly",
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
        # Map package name to import name if different
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

    # Map extension name to pip extras
    extras_map = {
        "api": "api",
        "visualization": "viz",
        "ledger": "",  # No extra dependencies
        "formatter": "",  # Included in core
    }

    extra = extras_map.get(name, "")
    if not extra:
        return True  # No installation needed

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
    """Load an extension module.

    Args:
        name: Extension name

    Returns:
        Extension module or None if failed
    """
    if name not in EXTENSIONS:
        return None

    if not check_extension(name):
        return None

    info = EXTENSIONS[name]
    try:
        module = importlib.import_module(info.module)
        return module
    except ImportError:
        return None


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
# Extension Manager CLI Entry Point
# ============================================================================


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
  umcp-ext info api                # Show API extension details
  umcp-ext install api             # Install API dependencies
  umcp-ext run visualization       # Run visualization dashboard
  umcp-ext check api               # Check if API is installed
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List all extensions")
    list_parser.add_argument("--type", "-t", help="Filter by extension type")

    # info command
    info_parser = subparsers.add_parser("info", help="Show extension details")
    info_parser.add_argument("name", help="Extension name")

    # install command
    install_parser = subparsers.add_parser("install", help="Install extension")
    install_parser.add_argument("name", help="Extension name")

    # check command
    check_parser = subparsers.add_parser("check", help="Check if extension is installed")
    check_parser.add_argument("name", help="Extension name")

    # run command
    run_parser = subparsers.add_parser("run", help="Run an extension")
    run_parser.add_argument("name", help="Extension name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
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

            print(f"\n📦 {ext['name']}")
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
        print("\n" + "=" * 70)
        print(f"Extension: {info['name']}")
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
