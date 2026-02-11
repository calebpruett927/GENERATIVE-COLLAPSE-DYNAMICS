"""
UMCP Root File Validator

Validates the 16 root-level UMCP files for structural integrity and mathematical consistency.

Interconnections:
- Validates: All root files (manifest, contract, observables, weights, trace, invariants, etc.)
- Checks: Tier-1 identities (F = 1-ω, IC ≈ exp(κ)), regime classification, checksums
- Implements: AXIOM-0 (no_return_no_credit), typed censoring rules
- Used by: umcp CLI (umcp validate), tests/test_97_root_integration.py
- Documentation: PROTOCOL_REFERENCE.md, docs/interconnected_architecture.md
- Optimizations: Uses compute_utils (OPT-17,20), kernel_optimized (OPT-1,2,12)

Validation layers:
1. File existence (16 required files)
2. Schema conformance (YAML/CSV structure)
3. Mathematical identities (Tier-1 kernel)
4. Regime thresholds (GCD classification)
5. Integrity checksums (SHA256)
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# Import optimization utilities
from .compute_utils import clip_coordinates
from .kernel_optimized import validate_kernel_bounds


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
        self._validate_thermodynamic_diagnostic()

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

    def _load_yaml(self, path: Path) -> Any:
        """Load a YAML file. Fallback to minimal parser if PyYAML unavailable and file is simple."""
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        if yaml is not None:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Minimal YAML parser for simple key: value pairs (no nested structures)
        result = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
        return result

    def _validate_manifest(self) -> None:
        """Validate manifest.yaml structure."""
        try:
            manifest_path = self.root / "manifest.yaml"
            manifest = self._load_yaml(manifest_path)

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
            contract = self._load_yaml(contract_path)

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
            observables = self._load_yaml(obs_path)

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
            # Support both 'w_*' column-prefix format and single 'weight' column
            weight_vals = [float(row[k]) for k in row if k.startswith("w_")]
            if not weight_vals and "weight" in row:
                weight_vals = [float(r["weight"]) for r in rows]
            total = sum(weight_vals)

            if abs(total - 1.0) < 1e-6:
                self.passed.append(f"✓ Weights sum to 1.0 (sum={total:.10f})")
            else:
                self.errors.append(f"✗ Weights do not sum to 1.0 (sum={total:.10f})")
        except Exception as e:
            self.errors.append(f"✗ Error validating weights.csv: {e}")

    def _validate_trace_bounds(self) -> None:
        """Validate trace coordinates are in [0,1] using optimized utilities (OPT-20)."""
        try:
            trace_path = self.root / "derived" / "trace.csv"
            with open(trace_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self.errors.append("✗ trace.csv is empty")
                return

            row = rows[0]
            coords = np.array([float(row[k]) for k in sorted(row.keys()) if k.startswith("c_")])

            # Use optimized clipping utility for diagnostics (OPT-20: vectorized)
            # Use 1e-9 tolerance: near-limit values (e.g. 0.99999999 in Stable regime)
            # are mathematically valid and should not be flagged as out-of-range.
            clip_result = clip_coordinates(coords, epsilon=1e-9)

            if clip_result.clip_count == 0:
                self.passed.append(f"✓ All {len(coords)} coordinates in [ε, 1-ε]")
            else:
                # Report which coordinates are out of range
                oor_info = f"{clip_result.clip_count} out-of-range at indices {clip_result.oor_indices}"
                self.errors.append(f"✗ Coordinates require clipping: {oor_info}")
        except Exception as e:
            self.errors.append(f"✗ Error validating trace.csv: {e}")

    def _validate_invariant_identities(self) -> None:
        """Validate F = 1-ω and IC ≈ exp(κ) using optimized bounds (OPT-2, OPT-12)."""
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
            C = float(row.get("C", 0.0))
            float(row.get("S", 0.0))

            # Use optimized kernel validation (Lemma 1 bounds)
            # Use 1e-9 epsilon: near-limit invariants (IC ≈ 1 in Stable regime)
            # are valid and should not trigger false violations.
            bounds_valid = validate_kernel_bounds(F, omega, C, IC, kappa, epsilon=1e-9)
            if bounds_valid:
                self.passed.append("✓ All kernel outputs satisfy Lemma 1 range bounds")
            else:
                self.errors.append("✗ Lemma 1 violation: kernel outputs outside valid ranges")

            # Check F = 1 - ω (Tier-1 identity)
            expected_F = 1.0 - omega
            if abs(F - expected_F) < 1e-9:
                self.passed.append(f"✓ F = 1-ω identity satisfied (|{F} - {expected_F}| < 1e-9)")
            else:
                self.errors.append(f"✗ F ≠ 1-ω: F={F}, 1-ω={expected_F}, diff={abs(F - expected_F)}")

            # Check IC ≈ exp(κ) (Lemma 2: IC is geometric mean)
            expected_IC = math.exp(kappa)
            if abs(IC - expected_IC) < 1e-6:
                self.passed.append(f"✓ IC ≈ exp(κ) identity satisfied (|{IC} - {expected_IC:.6f}| < 1e-6)")
            else:
                self.warnings.append(
                    f"⚠ IC ≈ exp(κ) slightly off: IC={IC}, exp(κ)={expected_IC:.6f}, diff={abs(IC - expected_IC)}"
                )

            # Check AM-GM inequality (Lemma 4: IC ≤ F)
            if IC <= F + 1e-9:
                self.passed.append(f"✓ Lemma 4 AM-GM satisfied: IC={IC:.6f} ≤ F={F:.6f}")
            else:
                self.errors.append(f"✗ Lemma 4 violated: IC={IC:.6f} > F={F:.6f}")

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

    def _validate_thermodynamic_diagnostic(self) -> None:
        """Run Tier-2 τ_R* thermodynamic diagnostic on invariant data.

        This is a Tier-2 diagnostic with Tier-0 checks. It reads Tier-1
        kernel invariants (read-only, no back-edges) and produces:
        - Budget decomposition (Γ, αC, Δκ)
        - Phase classification (surplus/deficit/trapped/pole)
        - Tier-1 identity verification (cross-check of existing checks)

        Results are reported as warnings/info since τ_R* is a diagnostic
        extension, not a core validation gate. Promotion to a Tier-0 gate
        requires a formal seam weld and contract version bump.

        Reference: KERNEL_SPECIFICATION.md §3, §5
        """
        try:
            inv_path = self.root / "outputs" / "invariants.csv"
            with open(inv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return  # No data to diagnose (file existence checked elsewhere)

            from .tau_r_star import diagnose

            row = rows[0]
            omega = float(row["omega"])
            F = float(row["F"])
            S = float(row.get("S", 0.0))
            C = float(row.get("C", 0.0))
            kappa = float(row["kappa"])
            IC = float(row["IC"])

            # Default R = 0.01 for diagnostic purposes
            diag = diagnose(omega=omega, F=F, S=S, C=C, kappa=kappa, IC=IC, R=0.01)

            # Report Tier-0 identity cross-check
            if diag.tier0_checks_pass:
                self.passed.append("✓ τ_R* Tier-0 identity cross-check passed")
            else:
                for w in diag.warnings:
                    if "≠" in w or "violated" in w:
                        self.warnings.append(f"⚠ τ_R* identity cross-check: {w}")

            # Report Tier-2 diagnostic results (informational)
            self.passed.append(
                f"✓ τ_R* diagnostic: phase={diag.phase.value}, "
                f"dominance={diag.dominance.value}, "
                f"trapped={diag.is_trapped}"
            )

        except Exception as e:
            self.warnings.append(f"⚠ τ_R* diagnostic skipped: {e}")

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
