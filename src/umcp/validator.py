"""
UMCP Root File Validator

Validates the 16 root-level UMCP files for structural integrity and mathematical consistency.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml


class RootFileValidator:
    """
    Validates root-level UMCP configuration and data files.

    Checks:
    - File existence
    - Schema conformance (basic structure)
    - Mathematical identities (F = 1-ω, IC ≈ exp(κ))
    - Regime classification accuracy
    - Integrity checksums
    """

    def __init__(self, root_dir: Path | None = None):
        """
        Initialize validator.

        Args:
            root_dir: Repository root directory (default: auto-detect)
        """
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
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []

    def validate_all(self) -> dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            Dict with validation results
        """
        self.errors = []
        self.warnings = []
        self.passed = []

        # Check file existence
        self._check_file_existence()

        # Validate structure
        self._validate_manifest()
        self._validate_contract()
        self._validate_observables()
        self._validate_weights()

        # Validate mathematical consistency
        self._validate_trace_bounds()
        self._validate_invariant_identities()
        self._validate_regime_classification()

        # Validate integrity
        self._validate_checksums()

        return {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
            "total_checks": len(self.errors) + len(self.warnings) + len(self.passed),
        }

    def _check_file_existence(self) -> None:
        """Check that all 16 root files exist."""
        required_files = [
            "manifest.yaml",
            "contract.yaml",
            "observables.yaml",
            "embedding.yaml",
            "return.yaml",
            "closures.yaml",
            "weights.csv",
            "derived/trace.csv",
            "derived/trace_meta.yaml",
            "outputs/invariants.csv",
            "outputs/regimes.csv",
            "outputs/welds.csv",
            "outputs/report.txt",
            "integrity/sha256.txt",
            "integrity/env.txt",
            "integrity/code_version.txt",
        ]

        for file_path in required_files:
            full_path = self.root / file_path
            if full_path.exists():
                self.passed.append(f"✓ File exists: {file_path}")
            else:
                self.errors.append(f"✗ Missing file: {file_path}")

    def _validate_manifest(self) -> None:
        """Validate manifest.yaml structure."""
        try:
            manifest_path = self.root / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

            if "schema" not in manifest:
                self.errors.append("✗ manifest.yaml missing 'schema' field")
            elif "casepack" not in manifest:
                self.errors.append("✗ manifest.yaml missing 'casepack' field")
            else:
                self.passed.append("✓ manifest.yaml structure valid")
        except Exception as e:
            self.errors.append(f"✗ Error loading manifest.yaml: {e}")

    def _validate_contract(self) -> None:
        """Validate contract.yaml structure."""
        try:
            contract_path = self.root / "contract.yaml"
            with open(contract_path) as f:
                contract = yaml.safe_load(f)

            if "schema" not in contract:
                self.errors.append("✗ contract.yaml missing 'schema' field")
            elif "contract" not in contract:
                self.errors.append("✗ contract.yaml missing 'contract' field")
            else:
                self.passed.append("✓ contract.yaml structure valid")
        except Exception as e:
            self.errors.append(f"✗ Error loading contract.yaml: {e}")

    def _validate_observables(self) -> None:
        """Validate observables.yaml structure."""
        try:
            obs_path = self.root / "observables.yaml"
            with open(obs_path) as f:
                observables = yaml.safe_load(f)

            if "observables" not in observables:
                self.errors.append("✗ observables.yaml missing 'observables' field")
            else:
                self.passed.append("✓ observables.yaml structure valid")
        except Exception as e:
            self.errors.append(f"✗ Error loading observables.yaml: {e}")

    def _validate_weights(self) -> None:
        """Validate weights.csv sums to 1.0."""
        try:
            weights_path = self.root / "weights.csv"
            with open(weights_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self.errors.append("✗ weights.csv is empty")
                return

            row = rows[0]
            total = sum(float(row[k]) for k in row if k.startswith("w_"))

            if abs(total - 1.0) < 1e-9:
                self.passed.append(f"✓ Weights sum to 1.0 (sum={total:.10f})")
            else:
                self.errors.append(f"✗ Weights do not sum to 1.0 (sum={total:.10f})")
        except Exception as e:
            self.errors.append(f"✗ Error validating weights.csv: {e}")

    def _validate_trace_bounds(self) -> None:
        """Validate trace coordinates are in [0,1]."""
        try:
            trace_path = self.root / "derived" / "trace.csv"
            with open(trace_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self.errors.append("✗ trace.csv is empty")
                return

            row = rows[0]
            coords = [float(row[k]) for k in row if k.startswith("c_")]

            if all(0.0 <= c <= 1.0 for c in coords):
                self.passed.append(f"✓ All coordinates in [0,1]: {coords}")
            else:
                self.errors.append(f"✗ Coordinates out of bounds: {coords}")
        except Exception as e:
            self.errors.append(f"✗ Error validating trace.csv: {e}")

    def _validate_invariant_identities(self) -> None:
        """Validate F = 1-ω and IC ≈ exp(κ)."""
        try:
            inv_path = self.root / "outputs" / "invariants.csv"
            with open(inv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self.errors.append("✗ invariants.csv is empty")
                return

            row = rows[0]
            omega = float(row["omega"])
            F = float(row["F"])
            kappa = float(row["kappa"])
            IC = float(row["IC"])

            # Check F = 1 - ω
            expected_F = 1.0 - omega
            if abs(F - expected_F) < 1e-9:
                self.passed.append(f"✓ F = 1-ω identity satisfied (|{F} - {expected_F}| < 1e-9)")
            else:
                self.errors.append(f"✗ F ≠ 1-ω: F={F}, 1-ω={expected_F}, diff={abs(F - expected_F)}")

            # Check IC ≈ exp(κ)
            expected_IC = math.exp(kappa)
            if abs(IC - expected_IC) < 1e-6:
                self.passed.append(f"✓ IC ≈ exp(κ) identity satisfied (|{IC} - {expected_IC:.6f}| < 1e-6)")
            else:
                self.warnings.append(
                    f"⚠ IC ≈ exp(κ) slightly off: IC={IC}, exp(κ)={expected_IC:.6f}, diff={abs(IC - expected_IC)}"
                )
        except Exception as e:
            self.errors.append(f"✗ Error validating invariant identities: {e}")

    def _validate_regime_classification(self) -> None:
        """Validate regime label matches thresholds."""
        try:
            inv_path = self.root / "outputs" / "invariants.csv"
            with open(inv_path) as f:
                reader = csv.DictReader(f)
                inv_rows = list(reader)

            reg_path = self.root / "outputs" / "regimes.csv"
            with open(reg_path) as f:
                reader = csv.DictReader(f)
                reg_rows = list(reader)

            if not inv_rows or not reg_rows:
                self.errors.append("✗ invariants.csv or regimes.csv is empty")
                return

            inv = inv_rows[0]
            reg = reg_rows[0]

            omega = float(inv["omega"])
            F = float(inv["F"])
            S = float(inv["S"])
            C = float(inv["C"])

            regime_label_inv = inv.get("regime_label", "")
            regime_label_reg = reg.get("regime_label", "")

            # Determine expected regime
            if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
                expected = "Stable"
            elif omega >= 0.30:
                expected = "Collapse"
            else:
                expected = "Watch"

            if regime_label_inv == expected and regime_label_reg == expected:
                self.passed.append(
                    f"✓ Regime classification correct: {expected} (ω={omega:.6f}, F={F:.6f}, S={S:.6f}, C={C:.6f})"
                )
            else:
                self.errors.append(
                    f"✗ Regime mismatch: expected={expected}, invariants={regime_label_inv}, regimes={regime_label_reg}"
                )
        except Exception as e:
            self.errors.append(f"✗ Error validating regime classification: {e}")

    def _validate_checksums(self) -> None:
        """Validate SHA256 checksums in integrity/sha256.txt.

        Implements no_return_no_credit principle: only artifacts that return
        through the collapse-reconstruction cycle receive credit in validation.
        Ephemeral build artifacts are excluded from checksum validation.
        """
        try:
            checksum_path = self.root / "integrity" / "sha256.txt"
            with open(checksum_path) as f:
                lines = f.readlines()

            # Artifacts that do not return through collapse receive no credit
            # (AXIOM-0: What Returns Through Collapse Is Real)
            NON_RETURNING_PATTERNS = [
                ".mypy_cache/",  # Ephemeral type checker cache
                ".pytest_cache/",  # Ephemeral test cache
                "__pycache__/",  # Ephemeral bytecode cache
                ".venv/",  # Environment-specific virtual environment
                "validator.result.baseline.json",  # Generated output artifact
                "validator.result.strict.json",  # Generated output artifact
            ]

            mismatches = []
            validated_count = 0

            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue

                expected_hash, file_path = parts

                # Apply typed censoring: skip non-returning artifacts
                if any(pattern in file_path for pattern in NON_RETURNING_PATTERNS):
                    continue

                full_path = self.root / file_path

                if not full_path.exists():
                    mismatches.append(f"{file_path} (file missing)")
                    continue

                # Compute actual hash
                sha256 = hashlib.sha256()
                with open(full_path, "rb") as f:
                    sha256.update(f.read())
                actual_hash = sha256.hexdigest()

                if actual_hash != expected_hash:
                    mismatches.append(f"{file_path} (hash mismatch)")
                else:
                    validated_count += 1

            if mismatches:
                self.errors.append(f"✗ Checksum mismatches: {', '.join(mismatches)}")
            else:
                self.passed.append(f"✓ All checksums valid ({validated_count} files verified)")
        except Exception as e:
            self.errors.append(f"✗ Error validating checksums: {e}")


def get_root_validator(root_dir: Path | None = None) -> RootFileValidator:
    """
    Factory function to create a RootFileValidator instance.

    Args:
        root_dir: Repository root directory (default: auto-detect)

    Returns:
        RootFileValidator instance
    """
    return RootFileValidator(root_dir=root_dir)
