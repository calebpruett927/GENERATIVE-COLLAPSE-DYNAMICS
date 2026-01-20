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
from typing import Any, Dict, List, Optional


class ExtensionRegistry:
    """Registry for UMCP extensions"""
    
    # Built-in extensions shipped with UMCP
    BUILTIN_EXTENSIONS = {
        "visualization": {
            "module": "visualize_umcp",
            "class": "UMCPVisualization",
            "name": "Interactive Visualization Dashboard",
            "description": "Streamlit dashboard for UMCP validation monitoring",
            "requires": ["streamlit>=1.30.0", "pandas>=2.0.0", "plotly>=5.18.0"],
            "command": "umcp-visualize",
            "port": 8501,
            "type": "dashboard"
        },
        "api": {
            "module": "api_umcp",
            "class": "UMCPAuditAPI",
            "name": "Public Audit API",
            "description": "REST API for UMCP validation receipts and statistics",
            "requires": ["fastapi>=0.109.0", "uvicorn[standard]>=0.27.0"],
            "command": "umcp-api",
            "port": 8000,
            "type": "api"
        },
        "ledger": {
            "module": "umcp.cli",
            "class": "LedgerExtension",
            "name": "Continuous Ledger",
            "description": "Automatic logging of validation receipts to CSV ledger",
            "requires": [],  # Built into core
            "command": None,  # Automatically active
            "type": "logging"
        },
        "autoformat": {
            "module": "umcp_autoformat",
            "class": "UMCPContractFormatter",
            "name": "Contract Auto-Formatter",
            "description": "Automatic formatting and validation of UMCP contracts",
            "requires": [],  # Uses stdlib only
            "command": "python umcp_autoformat.py",
            "type": "tool"
        }
    }
    
    # Extension categories
    CATEGORIES = {
        "dashboard": "Interactive visualization and monitoring",
        "api": "REST API and web services",
        "logging": "Data collection and audit trails",
        "tool": "Command-line utilities",
        "closure": "Custom computational closures",
        "validator": "Custom validation logic"
    }
    
    def __init__(self):
        self._extensions: Dict[str, Dict[str, Any]] = {}
        self._load_builtin()
    
    def _load_builtin(self):
        """Load built-in extensions"""
        self._extensions.update(self.BUILTIN_EXTENSIONS)
    
    def register(
        self,
        name: str,
        module: str,
        cls: str,
        description: str,
        extension_type: str = "custom",
        requires: Optional[List[str]] = None,
        command: Optional[str] = None,
        **metadata
    ):
        """
        Register a custom extension
        
        Args:
            name: Extension identifier
            module: Python module path
            cls: Class name within module
            description: Human-readable description
            extension_type: Extension category
            requires: List of pip dependencies
            command: CLI command to run extension
            **metadata: Additional metadata
        """
        self._extensions[name] = {
            "module": module,
            "class": cls,
            "name": metadata.get("display_name", name),
            "description": description,
            "requires": requires or [],
            "command": command,
            "type": extension_type,
            **metadata
        }
    
    def list(self, extension_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered extensions
        
        Args:
            extension_type: Filter by type (None = all)
        
        Returns:
            List of extension metadata dicts
        """
        extensions = list(self._extensions.values())
        
        if extension_type:
            extensions = [e for e in extensions if e.get("type") == extension_type]
        
        return extensions
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get extension metadata by name"""
        return self._extensions.get(name)
    
    def load(self, name: str) -> Optional[Any]:
        """
        Load and return extension class
        
        Args:
            name: Extension identifier
        
        Returns:
            Extension class or None if not found
        """
        ext = self.get(name)
        if not ext:
            return None
        
        try:
            module = importlib.import_module(ext["module"])
            cls = getattr(module, ext["class"])
            return cls
        except (ImportError, AttributeError) as e:
            print(f"Failed to load extension '{name}': {e}")
            return None
    
    def install_dependencies(self, name: str) -> bool:
        """
        Install extension dependencies
        
        Args:
            name: Extension identifier
        
        Returns:
            True if successful, False otherwise
        """
        ext = self.get(name)
        if not ext:
            return False
        
        requires = ext.get("requires", [])
        if not requires:
            return True
        
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + requires)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_installed(self, name: str) -> bool:
        """
        Check if extension dependencies are installed
        
        Args:
            name: Extension identifier
        
        Returns:
            True if all dependencies installed, False otherwise
        """
        ext = self.get(name)
        if not ext:
            return False
        
        requires = ext.get("requires", [])
        if not requires:
            return True
        
        for dep in requires:
            # Extract package name (before >=, ==, etc.)
            pkg_name = dep.split(">=")[0].split("==")[0].split("[")[0]
            try:
                importlib.import_module(pkg_name)
            except ImportError:
                return False
        
        return True


# Global registry instance
_registry = ExtensionRegistry()


def list_extensions(extension_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all registered UMCP extensions
    
    Args:
        extension_type: Filter by type (dashboard, api, logging, tool, etc.)
    
    Returns:
        List of extension metadata dictionaries
    
    Example:
        >>> extensions = list_extensions()
        >>> for ext in extensions:
        ...     print(f"{ext['name']}: {ext['description']}")
    """
    return _registry.list(extension_type)


def load_extension(name: str) -> Optional[Any]:
    """
    Load an extension by name
    
    Args:
        name: Extension identifier
    
    Returns:
        Extension class or None if not found
    
    Example:
        >>> viz = load_extension('visualization')
        >>> if viz:
        ...     viz.run()
    """
    return _registry.load(name)


def get_extension_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for an extension
    
    Args:
        name: Extension identifier
    
    Returns:
        Extension metadata dict or None
    """
    return _registry.get(name)


def install_extension(name: str) -> bool:
    """
    Install extension dependencies
    
    Args:
        name: Extension identifier
    
    Returns:
        True if successful, False otherwise
    """
    return _registry.install_dependencies(name)


def check_extension(name: str) -> bool:
    """
    Check if extension is installed
    
    Args:
        name: Extension identifier
    
    Returns:
        True if installed, False otherwise
    """
    return _registry.check_installed(name)


def register_extension(
    name: str,
    module: str,
    cls: str,
    description: str,
    extension_type: str = "custom",
    **kwargs
):
    """
    Register a custom extension
    
    Args:
        name: Extension identifier
        module: Python module path
        cls: Class name
        description: Description
        extension_type: Extension category
        **kwargs: Additional metadata
    
    Example:
        >>> register_extension(
        ...     name="my_closure",
        ...     module="my_package.closures",
        ...     cls="MyCustomClosure",
        ...     description="Custom closure for special analysis",
        ...     extension_type="closure",
        ...     requires=["numpy>=1.24.0"]
        ... )
    """
    _registry.register(name, module, cls, description, extension_type, **kwargs)


def print_extensions_table():
    """Print a formatted table of all extensions"""
    extensions = list_extensions()
    
    print("\n" + "="*80)
    print(" " * 25 + "UMCP EXTENSIONS")
    print("="*80)
    
    categories = {}
    for ext in extensions:
        cat = ext.get("type", "custom")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ext)
    
    for cat, exts in sorted(categories.items()):
        cat_desc = _registry.CATEGORIES.get(cat, "Custom extensions")
        print(f"\n📦 {cat.upper()}: {cat_desc}")
        print("-" * 80)
        
        for ext in exts:
            installed = check_extension(ext.get("name", ""))
            status = "✅" if installed else "❌"
            name = ext.get("name", "Unknown")
            desc = ext.get("description", "")
            cmd = ext.get("command", "")
            
            print(f"  {status} {name}")
            print(f"     {desc}")
            if cmd:
                print(f"     Command: {cmd}")
            print()
    
    print("="*80)
    print(f"Total: {len(extensions)} extensions\n")


if __name__ == "__main__":
    # Print extension table when run directly
    print_extensions_table()
    
    # Show usage examples
    print("\nUsage Examples:")
    print("-" * 80)
    print("""
# List all extensions
from umcp_extensions import list_extensions
extensions = list_extensions()

# Load and run visualization
from umcp_extensions import load_extension
viz = load_extension('visualization')
viz.run()

# Check if API is installed
from umcp_extensions import check_extension
if check_extension('api'):
    api = load_extension('api')
    api.run()

# Install extension dependencies
from umcp_extensions import install_extension
install_extension('visualization')
""")
