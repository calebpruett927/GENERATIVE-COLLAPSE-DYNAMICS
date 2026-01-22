"""
UMCP Extension Registry

Automatically discovers and registers UMCP extensions, including:
- Visualization dashboard
- Public audit API
- Continuous ledger
- Custom closures (GCD, RCFT)
- Contract validators
- Auto-formatters

Usage:
    from umcp_extensions import list_extensions, load_extension

    # List all available extensions
    extensions = list_extensions()

    # Load and run an extension
    viz = load_extension('visualization')
    viz.run()
"""

import importlib
import sys
from typing import Any, ClassVar


class ExtensionRegistry:
    """Registry for UMCP extensions"""

    # Built-in extensions shipped with UMCP
    BUILTIN_EXTENSIONS: ClassVar[dict] = {
        "visualization": {
            "module": "visualize_umcp",
            "class": "UMCPVisualization",
            "name": "Interactive Visualization Dashboard",
            "description": "Streamlit dashboard for UMCP validation monitoring",
            "requires": ["streamlit>=1.30.0", "pandas>=2.0.0", "plotly>=5.18.0"],
            "command": "umcp-visualize",
            "port": 8501,
            "type": "dashboard",
        },
        "api": {
            "module": "api_umcp",
            "class": "UMCPAuditAPI",
            "name": "Public Audit API",
            "description": "REST API for UMCP validation receipts and statistics",
            "requires": ["fastapi>=0.109.0", "uvicorn[standard]>=0.27.0"],
            "command": "umcp-api",
            "port": 8000,
            "type": "api",
        },
        "ledger": {
            "module": "umcp.cli",
        name: Extension identifier
    return _registry.check_installed(name)
        description: Description
    """
    for ext in extensions:
            status = "✅" if installed else "❌"

    """
    import argparse
        print("Available UMCP Extensions:")
        print()
        for ext_id, ext_info in extensions.items():
            status = "✅" if check_extension(ext_id) else "⚠️"
            print(f"{status} {ext_id}: {ext_info.get('name', 'Unknown')}")
            print(f"   {ext_info.get('description', '')}")
            print(f"   Command: {ext_info.get('command', 'N/A')}")
            print()
        return 0

    elif args.command == "info":
        ext = get_extension_info(args.extension)
