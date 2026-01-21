"""
UMCP Closure Loading and Execution

This module provides utilities for loading and executing closures referenced
in the closures registry (closures/registry.yaml).

Interconnections:
- Reads: closures/registry.yaml (UMCP.CLOSURES.DEFAULT.v1)
- Loads: closures/*.py, closures/gcd/*.py, closures/rcft/*.py
- Used by: examples/interconnected_demo.py, tests/test_70_contract_closures.py
- Documentation: docs/interconnected_architecture.md, GLOSSARY.md#closure

API:
- ClosureLoader: Main class for loading and executing closures
- get_closure_loader(): Factory function for obtaining a ClosureLoader instance
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


class ClosureLoader:
    """
    Loads and executes closures from the closures/ directory.

    Example:
        >>> loader = ClosureLoader()
        >>> result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
        >>> print(result)
        {'F': 50.0}
    """

    def __init__(self, root_dir: Path | None = None):
        """
        Initialize closure loader.

        Args:
            root_dir: Repository root directory (default: auto-detect)
        """
        if root_dir is None:
            # Auto-detect: find pyproject.toml
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    root_dir = current
                    break
                current = current.parent
            else:
                root_dir = Path.cwd()

        self.root = root_dir
        self.closures_dir = self.root / "closures"
        self.registry_path = self.closures_dir / "registry.yaml"
        self._registry: dict[str, Any] | None = None
        self._loaded_modules: dict[str, Any] = {}

    @property
    def registry(self) -> dict[str, Any]:
        """Load and cache the closures registry."""
        if self._registry is None:
            if not self.registry_path.exists():
                raise FileNotFoundError(f"Closures registry not found: {self.registry_path}")
            with open(self.registry_path) as f:
                self._registry = yaml.safe_load(f)
        return self._registry

    def list_closures(self) -> dict[str, str]:
        """
        List all registered closures.

        Returns:
            Dict mapping closure names to their file paths
        """
        registry = self.registry.get("registry", {})
        closures_obj = registry.get("closures", {})

        result = {}
        for name, spec in closures_obj.items():
            if isinstance(spec, dict) and "path" in spec:
                result[name] = spec["path"]

        return result

    def load_closure_module(self, name: str) -> Any:
        """
        Load a Python closure module from closures/ directory.

        Args:
            name: Name of the closure (e.g., "F_from_omega")

        Returns:
            The loaded Python module
        """
        if name in self._loaded_modules:
            return self._loaded_modules[name]

        # Try direct .py file first
        py_file = self.closures_dir / f"{name}.py"
        if not py_file.exists():
            raise FileNotFoundError(f"Closure module not found: {py_file}")

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(f"umcp.closures.{name}", py_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load closure module: {py_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"umcp.closures.{name}"] = module
        spec.loader.exec_module(module)

        self._loaded_modules[name] = module
        return module

    def get_closure_function(self, name: str) -> Callable[..., Any]:
        """
        Get the compute function from a closure module.

        Args:
            name: Name of the closure

        Returns:
            The compute function
        """
        module = self.load_closure_module(name)
        if not hasattr(module, "compute"):
            raise AttributeError(f"Closure module {name} does not have a 'compute' function")
        return module.compute  # type: ignore[no-any-return]

    def execute_closure(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a closure with the given parameters.

        Args:
            name: Name of the closure
            **kwargs: Parameters to pass to the closure's compute function

        Returns:
            Result dict from the closure

        Example:
            >>> loader = ClosureLoader()
            >>> result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
        """
        compute_fn = self.get_closure_function(name)
        return compute_fn(**kwargs)  # type: ignore[no-any-return]

    def validate_closure_exists(self, name: str) -> bool:
        """
        Check if a closure exists and can be loaded.

        Args:
            name: Name of the closure

        Returns:
            True if closure exists and can be loaded
        """
        try:
            self.load_closure_module(name)
            return True
        except (FileNotFoundError, ImportError):
            return False


def get_closure_loader(root_dir: Path | None = None) -> ClosureLoader:
    """
    Factory function to create a ClosureLoader instance.

    Args:
        root_dir: Repository root directory (default: auto-detect)

    Returns:
        ClosureLoader instance
    """
    return ClosureLoader(root_dir=root_dir)
