"""
UMCP File Reference Module

Provides easy access to root-level UMCP files and standardized paths.
Use this module to load manifest, contract, observables, and other root-level configuration files.

Interconnections:
- Reads: All root-level YAML/CSV files (manifest, contract, observables, etc.)
- Used by: examples/interconnected_demo.py, tests/test_96_file_references.py
- API: UMCPFiles class provides load_*() methods for all UMCP files
- Documentation: docs/file_reference.md, docs/interconnected_architecture.md

Design:
- Auto-detects repository root via pyproject.toml
- Provides standardized paths for all UMCP artifacts
- Supports YAML, JSON, CSV, and text file loading
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


class UMCPFiles:
    """
    Helper class to reference and load UMCP root-level files.

    Usage:
        >>> from umcp.file_refs import UMCPFiles
        >>> umcp = UMCPFiles()
        >>> manifest = umcp.load_manifest()
        >>> contract = umcp.load_contract()
        >>> observables = umcp.load_observables()
    """

    def __init__(self, root_path: Path | None = None):
        """
        Initialize with optional root path.

        Args:
            root_path: Path to UMCP repository root. If None, attempts to find it.
        """
        if root_path is None:
            # Try to find repository root
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "pyproject.toml").exists():
                    root_path = current
                    break
                current = current.parent
            if root_path is None:
                root_path = Path.cwd()

        self.root = Path(root_path)

        # Root-level configuration files
        self.manifest_yaml = self.root / "manifest.yaml"
        self.contract_yaml = self.root / "contract.yaml"
        self.observables_yaml = self.root / "observables.yaml"
        self.embedding_yaml = self.root / "embedding.yaml"
        self.return_yaml = self.root / "return.yaml"
        self.closures_yaml = self.root / "closures.yaml"
        self.weights_csv = self.root / "weights.csv"

        # Derived data
        self.derived_dir = self.root / "derived"
        self.trace_csv = self.derived_dir / "trace.csv"
        self.trace_meta_yaml = self.derived_dir / "trace_meta.yaml"

        # Outputs
        self.outputs_dir = self.root / "outputs"
        self.invariants_csv = self.outputs_dir / "invariants.csv"
        self.regimes_csv = self.outputs_dir / "regimes.csv"
        self.welds_csv = self.outputs_dir / "welds.csv"
        self.report_txt = self.outputs_dir / "report.txt"

        # Integrity
        self.integrity_dir = self.root / "integrity"
        self.sha256_txt = self.integrity_dir / "sha256.txt"
        self.env_txt = self.integrity_dir / "env.txt"
        self.code_version_txt = self.integrity_dir / "code_version.txt"

    def load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML file."""
        with open(path) as f:
            return yaml.safe_load(f)

    def load_json(self, path: Path) -> dict[str, Any]:
        """Load a JSON file."""
        with open(path) as f:
            return json.load(f)

    def load_csv(self, path: Path) -> list[dict[str, Any]]:
        """Load a CSV file as list of dictionaries."""
        with open(path) as f:
            return list(csv.DictReader(f))

    def load_text(self, path: Path) -> str:
        """Load a text file."""
        with open(path) as f:
            return f.read()

    # Convenience methods for root-level files

    def load_manifest(self) -> dict[str, Any]:
        """Load manifest.yaml from root."""
        return self.load_yaml(self.manifest_yaml)

    def load_contract(self) -> dict[str, Any]:
        """Load contract.yaml from root."""
        return self.load_yaml(self.contract_yaml)

    def load_observables(self) -> dict[str, Any]:
        """Load observables.yaml from root."""
        return self.load_yaml(self.observables_yaml)

    def load_embedding(self) -> dict[str, Any]:
        """Load embedding.yaml from root."""
        return self.load_yaml(self.embedding_yaml)

    def load_return(self) -> dict[str, Any]:
        """Load return.yaml from root."""
        return self.load_yaml(self.return_yaml)

    def load_closures(self) -> dict[str, Any]:
        """Load closures.yaml from root."""
        return self.load_yaml(self.closures_yaml)

    def load_weights(self) -> list[dict[str, Any]]:
        """Load weights.csv from root."""
        return self.load_csv(self.weights_csv)

    # Derived data methods

    def load_trace(self) -> list[dict[str, Any]]:
        """Load derived/trace.csv."""
        return self.load_csv(self.trace_csv)

    def load_trace_meta(self) -> dict[str, Any]:
        """Load derived/trace_meta.yaml."""
        return self.load_yaml(self.trace_meta_yaml)

    # Outputs methods

    def load_invariants(self) -> list[dict[str, Any]]:
        """Load outputs/invariants.csv."""
        return self.load_csv(self.invariants_csv)

    def load_regimes(self) -> list[dict[str, Any]]:
        """Load outputs/regimes.csv."""
        return self.load_csv(self.regimes_csv)

    def load_welds(self) -> list[dict[str, Any]]:
        """Load outputs/welds.csv."""
        return self.load_csv(self.welds_csv)

    def load_report(self) -> str:
        """Load outputs/report.txt."""
        return self.load_text(self.report_txt)

    # Integrity methods

    def load_sha256(self) -> str:
        """Load integrity/sha256.txt."""
        return self.load_text(self.sha256_txt)

    def load_env(self) -> str:
        """Load integrity/env.txt."""
        return self.load_text(self.env_txt)

    def load_code_version(self) -> str:
        """Load integrity/code_version.txt."""
        return self.load_text(self.code_version_txt)

    # Validation methods

    def verify_all_exist(self) -> dict[str, bool]:
        """
        Check which files exist.

        Returns:
            Dictionary mapping file descriptions to existence status.
        """
        return {
            "manifest.yaml": self.manifest_yaml.exists(),
            "contract.yaml": self.contract_yaml.exists(),
            "observables.yaml": self.observables_yaml.exists(),
            "embedding.yaml": self.embedding_yaml.exists(),
            "return.yaml": self.return_yaml.exists(),
            "closures.yaml": self.closures_yaml.exists(),
            "weights.csv": self.weights_csv.exists(),
            "derived/trace.csv": self.trace_csv.exists(),
            "derived/trace_meta.yaml": self.trace_meta_yaml.exists(),
            "outputs/invariants.csv": self.invariants_csv.exists(),
            "outputs/regimes.csv": self.regimes_csv.exists(),
            "outputs/welds.csv": self.welds_csv.exists(),
            "outputs/report.txt": self.report_txt.exists(),
            "integrity/sha256.txt": self.sha256_txt.exists(),
            "integrity/env.txt": self.env_txt.exists(),
            "integrity/code_version.txt": self.code_version_txt.exists(),
        }

    def get_missing_files(self) -> list[str]:
        """Get list of missing required files."""
        return [name for name, exists in self.verify_all_exist().items() if not exists]


# Convenience function for quick access
def get_umcp_files(root_path: Path | None = None) -> UMCPFiles:
    """
    Get UMCPFiles instance.

    Args:
        root_path: Optional path to repository root.

    Returns:
        UMCPFiles instance.

    Example:
        >>> from umcp.file_refs import get_umcp_files
        >>> umcp = get_umcp_files()
        >>> manifest = umcp.load_manifest()
    """
    return UMCPFiles(root_path)
