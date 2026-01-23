"""
UMCP Closure Loading and Execution (repo-wide robust version)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


class ClosureLoader:
    def __init__(self, root_dir: Path | None = None):
        if root_dir is None:
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
        if self._registry is None:
            if not self.registry_path.exists():
                raise FileNotFoundError(f"Closures registry not found: {self.registry_path}")
            try:
                yaml = importlib.import_module("yaml")
                if yaml is None or getattr(yaml, "safe_load", None) is None:
                    raise ImportError("yaml missing safe_load")
                with open(self.registry_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    self._registry = cast(dict[str, Any], loaded or {})
            except Exception:
                self._registry = {}
                with open(self.registry_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if ":" in line:
                            k, v = line.split(":", 1)
                            self._registry[k.strip()] = v.strip()
        assert self._registry is not None
        return self._registry

    def list_closures(self) -> dict[str, str]:
        registry_obj = self.registry.get("registry", {}) or {}
        closures_obj = cast(dict[str, Any], registry_obj.get("closures", {}) or {})
        result: dict[str, str] = {}
        for name, spec in closures_obj.items():
            if isinstance(spec, dict) and "path" in spec:
                result[name] = spec["path"]
        return result

    def load_closure_module(self, name: str) -> Any:
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        py_file = self.closures_dir / f"{name}.py"
        if not py_file.exists():
            raise FileNotFoundError(f"Closure module not found: {py_file}")
        spec = importlib.util.spec_from_file_location(f"umcp.closures.{name}", py_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load closure module: {py_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"umcp.closures.{name}"] = module
        spec.loader.exec_module(module)
        self._loaded_modules[name] = module
        return module

    def get_closure_function(self, name: str) -> Callable[..., Any]:
        module = self.load_closure_module(name)
        if not hasattr(module, "compute"):
            raise AttributeError(f"Closure module {name} does not have a 'compute' function")
        return module.compute  # type: ignore[no-any-return]

    def execute_closure(self, name: str, **kwargs: Any) -> dict[str, Any]:
        compute_fn = self.get_closure_function(name)
        return compute_fn(**kwargs)  # type: ignore[no-any-return]

    def validate_closure_exists(self, name: str) -> bool:
        try:
            self.load_closure_module(name)
            return True
        except (FileNotFoundError, ImportError):
            return False


def get_closure_loader(root_dir: Path | None = None) -> ClosureLoader:
    return ClosureLoader(root_dir=root_dir)
