"""
UMCP Preflight Validator - Failure Node Atlas Enforcement

Implements mechanical detection of failure nodes (meaning-drift choke points)
as defined in specs/failure_node_atlas_v1.yaml.

Cross-references:
  - specs/failure_node_atlas_v1.yaml (canonical node definitions)
  - schemas/failure_node_atlas.schema.json (schema validation)
  - docs/FAILURE_NODE_ATLAS.md (human-readable documentation)
  - KERNEL_SPECIFICATION.md, TIER_SYSTEM.md

Validation phases:
  Phase A: Structural conformance (fast, deterministic)
  Phase B: Drift conformance (comparability protection)
  Phase C: Statistical sentinels (suspicion detectors)

Exit codes:
  0 = PASS (no hits above INFO)
  1 = WARN (interpretation allowed only with explicit warning banner)
  2 = ERROR (nonconformant; do not interpret outputs)
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

Severity = Literal["ERROR", "WARN", "INFO"]
Phase = Literal["A", "B", "C"]

# Exit code mapping
EXIT_CODES = {"ERROR": 2, "WARN": 1, "PASS": 0}


@dataclass
class FailureHit:
    """A detected failure node hit."""

    node_id: str
    severity: Severity
    evidence: dict[str, Any]
    action: str
    phase: Phase

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "severity": self.severity,
            "evidence": self.evidence,
            "action": self.action,
            "phase": self.phase,
        }


@dataclass
class PreflightReport:
    """Complete preflight validation report."""

    run_id: str
    status: Literal["ERROR", "WARN", "PASS"]
    hits: list[FailureHit] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    validator_version: str = "1.0.0"
    atlas_version: str = "1.0.0"

    @property
    def error_count(self) -> int:
        return sum(1 for h in self.hits if h.severity == "ERROR")

    @property
    def warn_count(self) -> int:
        return sum(1 for h in self.hits if h.severity == "WARN")

    @property
    def info_count(self) -> int:
        return sum(1 for h in self.hits if h.severity == "INFO")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_utc": self.created_utc,
            "validator": {
                "version": self.validator_version,
                "atlas_version": self.atlas_version,
            },
            "hits": [h.to_dict() for h in self.hits],
            "summary": {
                "error_count": self.error_count,
                "warn_count": self.warn_count,
                "info_count": self.info_count,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def exit_code(self) -> int:
        """Return appropriate exit code based on hits."""
        if self.error_count > 0:
            return EXIT_CODES["ERROR"]
        elif self.warn_count > 0:
            return EXIT_CODES["WARN"]
        return EXIT_CODES["PASS"]


# -----------------------------------------------------------------------------
# Preflight Validator
# -----------------------------------------------------------------------------


class PreflightValidator:
    """
    Validates UMCP artifacts against the Failure Node Atlas.

    Performs three-phase validation:
    - Phase A: Structural conformance (fast, deterministic)
    - Phase B: Drift conformance (comparability protection)
    - Phase C: Statistical sentinels (suspicion detectors)
    """

    def __init__(
        self,
        root_dir: Path | None = None,
        freeze_dir: Path | None = None,
        run_id: str | None = None,
    ):
        """
        Initialize preflight validator.

        Args:
            root_dir: Repository root directory (default: auto-detect)
            freeze_dir: Directory containing frozen baseline artifacts
            run_id: Run identifier for the report
        """
        if root_dir is None:
            root_dir = self._find_repo_root()
        self.root = root_dir
        self.freeze_dir = freeze_dir or root_dir / "freeze"
        self.run_id = run_id or self._detect_run_id()
        self.hits: list[FailureHit] = []

        # Load atlas for reference
        self._atlas = self._load_atlas()

    def _find_repo_root(self) -> Path:
        """Auto-detect repository root."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _detect_run_id(self) -> str:
        """Detect run ID from manifest or generate placeholder."""
        manifest_path = self.root / "manifest.yaml"
        if manifest_path.exists():
            try:
                manifest = self._load_yaml(manifest_path)
                casepack = manifest.get("casepack", {})
                return f"{casepack.get('id', 'unknown')}-{casepack.get('version', '0.0.0')}"
            except Exception:
                pass
        return f"RUN-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    def _load_atlas(self) -> dict[str, Any]:
        """Load the Failure Node Atlas."""
        atlas_path = self.root / "specs" / "failure_node_atlas_v1.yaml"
        if atlas_path.exists():
            result = self._load_yaml(atlas_path)
            return result if isinstance(result, dict) else {}
        return {}

    def _load_yaml(self, path: Path) -> Any:
        """Load a YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required for preflight validation")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_csv(self, path: Path) -> list[dict[str, str]]:
        """Load a CSV file as list of dicts."""
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _add_hit(
        self,
        node_id: str,
        severity: Severity,
        evidence: dict[str, Any],
        action: str,
        phase: Phase,
    ) -> None:
        """Record a failure hit."""
        self.hits.append(FailureHit(node_id, severity, evidence, action, phase))

    # -------------------------------------------------------------------------
    # Phase A: Structural Conformance
    # -------------------------------------------------------------------------

    def phase_a_structural(self) -> None:
        """
        Phase A: Structural conformance checks.

        Fast, deterministic checks for:
        - Required files exist
        - Schemas validate
        - Manifest present
        - Hashes match
        - Weights sum to 1
        - Timestamps monotone
        - Required flags present

        Nodes: FN-003, FN-005, FN-007, FN-010, FN-011, FN-014
        """
        self._check_fn003_oor_flags()
        self._check_fn005_timestamps()
        self._check_fn007_weights()
        self._check_fn010_closures()
        self._check_fn011_diagnostic_laundering()
        self._check_fn014_manifest_integrity()

    def _check_fn003_oor_flags(self) -> None:
        """FN-003: Silent clipping / missing OOR flags."""
        observables_path = self.root / "observables.yaml"
        trace_path = self.root / "derived" / "trace.csv"

        if not observables_path.exists():
            return  # Will be caught by other checks

        try:
            obs = self._load_yaml(observables_path)
            face_policy = obs.get("face_policy", {})

            # Check if OOR policy requires flags
            if face_policy.get("clip_policy") in ["clip", "clamp"]:
                # Check if trace has OOR columns
                if trace_path.exists():
                    with open(trace_path) as f:
                        headers = f.readline().strip().split(",")
                    oor_cols = [h for h in headers if "oor" in h.lower() or "flag" in h.lower()]

                    if not oor_cols:
                        self._add_hit(
                            "FN-003",
                            "ERROR",
                            {
                                "clip_policy": face_policy.get("clip_policy"),
                                "expected_columns": "OOR flag columns",
                                "found_columns": headers,
                            },
                            "Add OOR flag columns to trace.csv or change face policy",
                            "A",
                        )
        except Exception as e:
            self._add_hit(
                "FN-003",
                "WARN",
                {"error": str(e)},
                "Could not verify OOR policy; manual review required",
                "A",
            )

    def _check_fn005_timestamps(self) -> None:
        """FN-005: Timebase drift (timestamp semantics change)."""
        trace_path = self.root / "derived" / "trace.csv"

        if not trace_path.exists():
            return

        try:
            rows = self._load_csv(trace_path)
            if not rows:
                return

            # Find timestamp column
            t_col = None
            for col in ["t", "timestamp", "time", "datetime"]:
                if col in rows[0]:
                    t_col = col
                    break

            if t_col is None:
                return

            timestamps = [row[t_col] for row in rows]

            # Check monotonicity
            prev = None
            non_monotone_indices = []
            for i, ts in enumerate(timestamps):
                if prev is not None and ts < prev:
                    non_monotone_indices.append(i)
                prev = ts

            if non_monotone_indices:
                self._add_hit(
                    "FN-005",
                    "ERROR",
                    {
                        "issue": "non-monotone timestamps",
                        "indices": non_monotone_indices[:10],  # First 10
                        "total_violations": len(non_monotone_indices),
                    },
                    "Fix timestamp ordering or declare non-monotone handling policy",
                    "A",
                )

            # Check for duplicates
            seen = set()
            duplicates = []
            for i, ts in enumerate(timestamps):
                if ts in seen:
                    duplicates.append(i)
                seen.add(ts)

            if duplicates:
                self._add_hit(
                    "FN-005",
                    "ERROR",
                    {
                        "issue": "duplicate timestamps",
                        "indices": duplicates[:10],
                        "total_duplicates": len(duplicates),
                    },
                    "Remove duplicate timestamps or declare aggregation policy",
                    "A",
                )

        except Exception as e:
            self._add_hit(
                "FN-005",
                "WARN",
                {"error": str(e)},
                "Could not verify timestamp integrity; manual review required",
                "A",
            )

    def _check_fn007_weights(self) -> None:
        """FN-007: Weight drift (w_i changes)."""
        weights_path = self.root / "weights.csv"

        if not weights_path.exists():
            self._add_hit(
                "FN-007",
                "ERROR",
                {"file": "weights.csv", "status": "missing"},
                "Create weights.csv with weights summing to 1",
                "A",
            )
            return

        try:
            rows = self._load_csv(weights_path)

            # Find weight column
            weight_col = None
            for col in ["w", "weight", "w_i", "weights"]:
                if any(col in row for row in rows):
                    weight_col = col
                    break

            if weight_col is None and rows:
                # Try first numeric column
                for key in rows[0]:
                    try:
                        float(rows[0][key])
                        weight_col = key
                        break
                    except (ValueError, TypeError):
                        pass

            if weight_col:
                weights = [float(row[weight_col]) for row in rows if row.get(weight_col)]
                weight_sum = sum(weights)
                tolerance = 1e-9

                if abs(weight_sum - 1.0) > tolerance:
                    self._add_hit(
                        "FN-007",
                        "ERROR",
                        {
                            "weight_sum": weight_sum,
                            "expected": 1.0,
                            "tolerance": tolerance,
                            "difference": abs(weight_sum - 1.0),
                        },
                        "Normalize weights to sum to 1.0",
                        "A",
                    )

        except Exception as e:
            self._add_hit(
                "FN-007",
                "WARN",
                {"error": str(e)},
                "Could not verify weight sum; manual review required",
                "A",
            )

    def _check_fn010_closures(self) -> None:
        """FN-010: Closure registry incompleteness or mismatch."""
        registry_path = self.root / "closures" / "registry.yaml"

        if not registry_path.exists():
            self._add_hit(
                "FN-010",
                "ERROR",
                {"file": "closures/registry.yaml", "status": "missing"},
                "Create closure registry with all required closure definitions",
                "A",
            )
            return

        try:
            registry = self._load_yaml(registry_path)
            reg_data = registry.get("registry", {})
            closures = reg_data.get("closures", {})

            # Check required closures exist
            required = ["gamma", "return_domain", "norms"]
            missing = [c for c in required if c not in closures]

            if missing:
                self._add_hit(
                    "FN-010",
                    "ERROR",
                    {"missing_closures": missing, "required": required},
                    f"Add missing closures to registry: {missing}",
                    "A",
                )

            # Verify closure files exist
            for name, closure_def in closures.items():
                if isinstance(closure_def, dict):
                    closure_path = closure_def.get("path")
                    if closure_path:
                        full_path = self.root / closure_path
                        if not full_path.exists():
                            self._add_hit(
                                "FN-010",
                                "ERROR",
                                {
                                    "closure": name,
                                    "path": closure_path,
                                    "status": "file missing",
                                },
                                f"Create closure file: {closure_path}",
                                "A",
                            )

        except Exception as e:
            self._add_hit(
                "FN-010",
                "WARN",
                {"error": str(e)},
                "Could not verify closure registry; manual review required",
                "A",
            )

    def _check_fn011_diagnostic_laundering(self) -> None:
        """FN-011: Diagnostic laundering into gates."""
        report_path = self.root / "outputs" / "report.txt"
        regimes_path = self.root / "outputs" / "regimes.csv"

        # Reserved Tier-2 prefixes that should not appear in gates
        tier2_prefixes = ["O.", "O_", "J.", "J_"]

        # Check report for Tier-2 symbols in gate context
        if report_path.exists():
            try:
                content = report_path.read_text()

                # Look for gate/decision context with Tier-2 symbols
                gate_patterns = [
                    r"gate[:\s]+[^\n]*(?:O\.|O_|J\.|J_)",
                    r"decision[:\s]+[^\n]*(?:O\.|O_|J\.|J_)",
                    r"threshold[:\s]+[^\n]*(?:O\.|O_|J\.|J_)",
                ]

                for pattern in gate_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self._add_hit(
                            "FN-011",
                            "ERROR",
                            {
                                "issue": "Tier-2 domain expansion symbols used in gate context",
                                "matches": matches[:5],
                                "reserved_prefixes": tier2_prefixes,
                            },
                            "Remove domain expansion symbols from decision gates; use kernel symbols only",
                            "A",
                        )
                        break

            except Exception as e:
                self._add_hit(
                    "FN-011",
                    "WARN",
                    {"error": str(e)},
                    "Could not verify gate symbols; manual review required",
                    "A",
                )

        # Check regime derivation uses only kernel symbols
        if regimes_path.exists():
            try:
                with open(regimes_path) as f:
                    headers = f.readline().strip().split(",")

                # Allowed kernel symbols for regime derivation
                # (reserved for future validation of regime derivation sources)
                _kernel_symbols = {
                    "omega",
                    "ω",
                    "S",
                    "C",
                    "F",
                    "IC",
                    "kappa",
                    "κ",
                    "tau_R",
                    "τ_R",
                    "regime",
                    "t",
                    "timestamp",
                }

                suspicious: list[str] = []
                for h in headers:
                    h_clean = h.strip().lower()
                    if any(h_clean.startswith(p.lower()) for p in tier2_prefixes):
                        suspicious.append(h)

                if suspicious:
                    self._add_hit(
                        "FN-011",
                        "ERROR",
                        {
                            "issue": "Tier-2 symbols in regime file",
                            "suspicious_columns": suspicious,
                        },
                        "Remove overlay columns from regimes.csv",
                        "A",
                    )

            except Exception:
                pass  # File read issues handled elsewhere

    def _check_fn014_manifest_integrity(self) -> None:
        """FN-014: Manifest/hash integrity failure."""
        manifest_path = self.root / "manifest.yaml"
        integrity_path = self.root / "integrity" / "sha256.txt"
        env_path = self.root / "integrity" / "env.txt"

        # Check manifest exists
        if not manifest_path.exists():
            self._add_hit(
                "FN-014",
                "ERROR",
                {"file": "manifest.yaml", "status": "missing"},
                "Create manifest.yaml",
                "A",
            )
            return

        # Check integrity directory
        if not integrity_path.exists():
            self._add_hit(
                "FN-014",
                "ERROR",
                {"file": "integrity/sha256.txt", "status": "missing"},
                "Run integrity generation to create sha256.txt",
                "A",
            )

        if not env_path.exists():
            self._add_hit(
                "FN-014",
                "WARN",
                {"file": "integrity/env.txt", "status": "missing"},
                "Add environment pin file for reproducibility",
                "A",
            )

        # Verify stored hashes match computed
        if integrity_path.exists():
            try:
                stored_hashes = {}
                with open(integrity_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                stored_hashes[parts[1]] = parts[0]

                # Verify a sample of critical files
                critical_files = [
                    "manifest.yaml",
                    "contract.yaml",
                    "observables.yaml",
                    "weights.csv",
                ]

                for fname in critical_files:
                    fpath = self.root / fname
                    if fpath.exists() and fname in stored_hashes:
                        computed = self._compute_sha256(fpath)
                        stored = stored_hashes[fname]
                        if computed != stored:
                            self._add_hit(
                                "FN-014",
                                "ERROR",
                                {
                                    "file": fname,
                                    "computed_hash": computed,
                                    "stored_hash": stored,
                                },
                                f"File {fname} has been modified; regenerate integrity or revert",
                                "A",
                            )

            except Exception as e:
                self._add_hit(
                    "FN-014",
                    "WARN",
                    {"error": str(e)},
                    "Could not verify hash integrity; manual review required",
                    "A",
                )

    # -------------------------------------------------------------------------
    # Phase B: Drift Conformance
    # -------------------------------------------------------------------------

    def phase_b_drift(self) -> None:
        """
        Phase B: Drift conformance checks.

        Compares current artifacts to frozen baseline for:
        - Adapter/embedding drift
        - Bounds drift
        - Norm/metric drift
        - Return settings drift

        Nodes: FN-001, FN-002, FN-008, FN-009
        Requires: freeze directory with baseline artifacts
        """
        if not self.freeze_dir.exists():
            # No freeze directory - skip drift checks but note it
            return

        self._check_fn001_adapter_drift()
        self._check_fn002_bounds_drift()
        self._check_fn008_norm_drift()
        self._check_fn009_return_drift()

    def _check_fn001_adapter_drift(self) -> None:
        """FN-001: Adapter drift (Ψ meaning drift)."""
        embedding_path = self.root / "embedding.yaml"
        freeze_hash_path = self.freeze_dir / "embedding.yaml.sha256"

        if not embedding_path.exists():
            return

        if freeze_hash_path.exists():
            try:
                frozen_hash = freeze_hash_path.read_text().strip()
                current_hash = self._compute_sha256(embedding_path)

                if frozen_hash != current_hash:
                    self._add_hit(
                        "FN-001",
                        "ERROR",
                        {
                            "expected_adapter_hash": frozen_hash,
                            "found_adapter_hash": current_hash,
                        },
                        "Mint new baseline Run_ID or declare seam; do not interpret outputs as comparable",
                        "B",
                    )
            except Exception as e:
                self._add_hit(
                    "FN-001",
                    "WARN",
                    {"error": str(e)},
                    "Could not verify adapter hash; manual review required",
                    "B",
                )

    def _check_fn002_bounds_drift(self) -> None:
        """FN-002: Bounds drift (comparability intervals K change)."""
        observables_path = self.root / "observables.yaml"
        freeze_bounds_path = self.freeze_dir / "bounds.json"

        if not observables_path.exists() or not freeze_bounds_path.exists():
            return

        try:
            obs = self._load_yaml(observables_path)
            # Get bounds from observables.yaml (may be nested under 'observables')
            obs_data = obs.get("observables", obs)
            current_bounds = obs_data.get("bounds", obs_data.get("coordinate_bounds", {}))

            with open(freeze_bounds_path) as f:
                frozen_bounds_doc = json.load(f)

            # Extract bounds from freeze doc (may be nested under 'bounds' key)
            frozen_bounds = frozen_bounds_doc.get("bounds", frozen_bounds_doc)

            if current_bounds != frozen_bounds:
                self._add_hit(
                    "FN-002",
                    "ERROR",
                    {
                        "frozen_bounds": frozen_bounds,
                        "current_bounds": current_bounds,
                    },
                    "Create new baseline with updated bounds or revert to frozen bounds",
                    "B",
                )

        except Exception as e:
            self._add_hit(
                "FN-002",
                "WARN",
                {"error": str(e)},
                "Could not verify bounds; manual review required",
                "B",
            )

    def _check_fn008_norm_drift(self) -> None:
        """FN-008: Norm / distance function drift."""
        registry_path = self.root / "closures" / "registry.yaml"
        freeze_norm_path = self.freeze_dir / "norms_closure.sha256"

        if not registry_path.exists() or not freeze_norm_path.exists():
            return

        try:
            registry = self._load_yaml(registry_path)
            closures = registry.get("registry", {}).get("closures", {})
            norm_def = closures.get("norms", {})

            if isinstance(norm_def, dict):
                norm_path = norm_def.get("path")
                if norm_path:
                    norm_file = self.root / norm_path
                    if norm_file.exists():
                        frozen_hash = freeze_norm_path.read_text().strip()
                        current_hash = self._compute_sha256(norm_file)

                        if frozen_hash != current_hash:
                            self._add_hit(
                                "FN-008",
                                "ERROR",
                                {
                                    "frozen_norm_hash": frozen_hash,
                                    "current_norm_hash": current_hash,
                                },
                                "Norm closure changed; mint new baseline or revert",
                                "B",
                            )

        except Exception as e:
            self._add_hit(
                "FN-008",
                "WARN",
                {"error": str(e)},
                "Could not verify norm closure; manual review required",
                "B",
            )

    def _check_fn009_return_drift(self) -> None:
        """FN-009: Return settings drift (η, H_rec, τ_R definition)."""
        return_path = self.root / "return.yaml"
        freeze_hash_path = self.freeze_dir / "return.yaml.sha256"

        if not return_path.exists():
            return

        if freeze_hash_path.exists():
            try:
                frozen_hash = freeze_hash_path.read_text().strip()
                current_hash = self._compute_sha256(return_path)

                if frozen_hash != current_hash:
                    self._add_hit(
                        "FN-009",
                        "ERROR",
                        {
                            "frozen_return_hash": frozen_hash,
                            "current_return_hash": current_hash,
                        },
                        "Return settings changed; mint new baseline or revert",
                        "B",
                    )
            except Exception as e:
                self._add_hit(
                    "FN-009",
                    "WARN",
                    {"error": str(e)},
                    "Could not verify return settings; manual review required",
                    "B",
                )

    # -------------------------------------------------------------------------
    # Phase C: Statistical Sentinels
    # -------------------------------------------------------------------------

    def phase_c_statistical(self) -> None:
        """
        Phase C: Statistical sentinel checks.

        Suspicion detectors that force human review:
        - Smoothing detection via autocorrelation
        - Distribution shift detection
        - Leakage detection
        - Rounding drift detection

        Nodes: FN-004, FN-006, FN-012, FN-013
        """
        self._check_fn004_smoothing()
        self._check_fn006_unit_drift()
        self._check_fn012_leakage()
        self._check_fn013_rounding()

    def _check_fn004_smoothing(self) -> None:
        """FN-004: Undeclared smoothing/filtering/resampling."""
        trace_path = self.root / "derived" / "trace.csv"
        observables_path = self.root / "observables.yaml"

        if not trace_path.exists():
            return

        try:
            # Check if preprocessing is declared
            declared_preprocessing = []
            if observables_path.exists():
                obs = self._load_yaml(observables_path)
                declared_preprocessing = obs.get("preprocessing_steps", [])

            # Load trace data for statistical tests
            rows = self._load_csv(trace_path)
            if len(rows) < 10:
                return

            # Find numeric columns for autocorrelation test
            numeric_cols = []
            for key in rows[0]:
                try:
                    float(rows[0][key])
                    numeric_cols.append(key)
                except (ValueError, TypeError):
                    pass

            # Simple lag-1 autocorrelation check
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                try:
                    values = [float(row[col]) for row in rows if row.get(col)]
                    if len(values) < 10:
                        continue

                    values_np = np.array(values)
                    if np.std(values_np) < 1e-10:
                        continue

                    # Compute lag-1 autocorrelation
                    n = len(values_np)
                    mean = np.mean(values_np)
                    var = np.var(values_np)
                    if var < 1e-10:
                        continue

                    autocorr = np.sum((values_np[:-1] - mean) * (values_np[1:] - mean)) / ((n - 1) * var)

                    # Very high autocorrelation suggests smoothing
                    if autocorr > 0.98 and not declared_preprocessing:
                        self._add_hit(
                            "FN-004",
                            "WARN",
                            {
                                "column": col,
                                "lag1_autocorrelation": float(autocorr),
                                "threshold": 0.98,
                                "declared_preprocessing": declared_preprocessing,
                            },
                            f"High autocorrelation in {col} suggests undeclared smoothing; review preprocessing",
                            "C",
                        )
                        break  # One warning is enough

                except (ValueError, TypeError):
                    continue

        except Exception:
            pass  # Statistical checks are advisory; don't fail on errors

    def _check_fn006_unit_drift(self) -> None:
        """FN-006: Unit drift before embedding."""
        observables_path = self.root / "observables.yaml"

        if not observables_path.exists():
            return

        try:
            obs = self._load_yaml(observables_path)
            observables = obs.get("observables", [])

            for observable in observables:
                if isinstance(observable, dict):
                    name = observable.get("name", observable.get("id", "unknown"))
                    units = observable.get("units")

                    if units is None:
                        self._add_hit(
                            "FN-006",
                            "WARN",
                            {"observable": name, "units": None},
                            f"Observable {name} missing units declaration",
                            "C",
                        )

        except Exception:
            pass

    def _check_fn012_leakage(self) -> None:
        """FN-012: Post hoc window selection / leakage."""
        # Check for receipt files with timestamps
        outputs_path = self.root / "outputs"
        if not outputs_path.exists():
            return

        try:
            receipt_files = list(outputs_path.glob("*.receipt.json")) + list(outputs_path.glob("*_receipt.json"))

            for receipt_path in receipt_files:
                try:
                    with open(receipt_path) as f:
                        receipt = json.load(f)

                    # Check for issuance timestamp
                    if "issuance_timestamp" not in receipt and "created_utc" not in receipt:
                        self._add_hit(
                            "FN-012",
                            "WARN",
                            {"receipt": receipt_path.name, "issue": "missing issuance timestamp"},
                            "Add issuance timestamp to receipt for audit trail",
                            "C",
                        )

                except (json.JSONDecodeError, KeyError):
                    pass

        except Exception:
            pass

    def _check_fn013_rounding(self) -> None:
        """FN-013: Rounding / formatting drift."""
        contract_path = self.root / "contract.yaml"

        if not contract_path.exists():
            return

        try:
            contract = self._load_yaml(contract_path)
            contract_data = contract.get("contract", {})

            # Check if rounding policy is declared
            if "rounding_policy" not in contract_data and "presentation" not in contract_data:
                self._add_hit(
                    "FN-013",
                    "WARN",
                    {"issue": "rounding policy not declared"},
                    "Add rounding_policy to contract for reproducible presentation",
                    "C",
                )

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Main validation
    # -------------------------------------------------------------------------

    def validate(self) -> PreflightReport:
        """
        Run all preflight validation phases.

        Returns:
            PreflightReport with all hits and status
        """
        self.hits = []

        # Run all phases
        self.phase_a_structural()
        self.phase_b_drift()
        self.phase_c_statistical()

        # Determine overall status
        status: Literal["ERROR", "WARN", "PASS"]
        if any(h.severity == "ERROR" for h in self.hits):
            status = "ERROR"
        elif any(h.severity == "WARN" for h in self.hits):
            status = "WARN"
        else:
            status = "PASS"

        atlas = self._atlas.get("atlas", {})
        return PreflightReport(
            run_id=self.run_id,
            status=status,
            hits=self.hits,
            atlas_version=atlas.get("version", "1.0.0"),
        )

    def validate_and_save(self, output_path: Path | None = None) -> PreflightReport:
        """
        Run validation and save report.

        Args:
            output_path: Path to save JSON report (default: casepacks/<id>/preflight/)

        Returns:
            PreflightReport
        """
        report = self.validate()

        if output_path is None:
            # Default to preflight directory in current casepack
            preflight_dir = self.root / "preflight"
            preflight_dir.mkdir(exist_ok=True)
            output_path = preflight_dir / "preflight_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json())

        return report


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def run_preflight(
    root_dir: Path | None = None,
    freeze_dir: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> int:
    """
    Run preflight validation and return exit code.

    Args:
        root_dir: Repository root
        freeze_dir: Freeze directory for drift checks
        output_path: Output path for JSON report
        verbose: Print verbose output

    Returns:
        Exit code (0=PASS, 1=WARN, 2=ERROR)
    """
    validator = PreflightValidator(root_dir=root_dir, freeze_dir=freeze_dir)
    report = validator.validate_and_save(output_path)

    if verbose:
        print(f"Preflight Report: {report.run_id}")
        print(f"Status: {report.status}")
        print(f"Errors: {report.error_count}, Warnings: {report.warn_count}, Info: {report.info_count}")

        if report.hits:
            print("\nHits:")
            for hit in report.hits:
                print(f"  [{hit.severity}] {hit.node_id}: {hit.action}")

    return report.exit_code


if __name__ == "__main__":
    import sys

    exit_code = run_preflight(verbose=True)
    sys.exit(exit_code)
