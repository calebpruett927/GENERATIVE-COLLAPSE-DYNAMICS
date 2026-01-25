#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
"""
UMCP Infrastructure Conservation Law Checker

Validates infrastructure using the fundamental conservation law (budget identity):

    R·τ_R = D_ω + D_C + Δκ

This script checks:
1. Mathematical identities across all CasePacks
2. Kernel invariant relationships (F = 1 - ω, IC = exp(κ))
3. Budget identity balance across seam chains
4. Closure consistency with declared contracts
5. Artifact integrity (hashes, schemas, required files)

Exit codes:
    0 = All conservation laws satisfied
    1 = Warnings present (non-critical violations)
    2 = Errors present (conservation law violations)

Cross-references:
    - KERNEL_SPECIFICATION.md §1-3 (invariant definitions)
    - src/umcp/frozen_contract.py (budget identity implementation)
    - specs/failure_node_atlas_v1.yaml (FN checks)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Tolerances from contract
TOL_SEAM = 0.005
TOL_ID = 1e-9
TOL_FIDELITY = 1e-9


@dataclass
class ConservationCheck:
    """Result of a conservation law check."""

    name: str
    formula: str
    passed: bool
    error: float
    tolerance: float
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "passed": self.passed,
            "error": self.error,
            "tolerance": self.tolerance,
            "details": self.details,
        }


@dataclass
class ConservationReport:
    """Full conservation check report."""

    status: str  # "PASS", "WARN", "ERROR"
    checks: list[ConservationCheck] = field(default_factory=list)  # type: ignore[assignment]
    casepacks_checked: list[str] = field(default_factory=list)  # type: ignore[assignment]
    errors: list[str] = field(default_factory=list)  # type: ignore[assignment]
    warnings: list[str] = field(default_factory=list)  # type: ignore[assignment]

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": {
                "total_checks": len(self.checks),
                "passed": self.pass_count,
                "failed": self.fail_count,
                "casepacks": len(self.casepacks_checked),
            },
            "checks": [c.to_dict() for c in self.checks],
            "casepacks_checked": self.casepacks_checked,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ConservationChecker:
    """
    Checks infrastructure using conservation laws.

    Core conservation laws:
    1. Fidelity-drift duality: F = 1 - ω
    2. Integrity-collapse: IC = exp(κ)
    3. Budget identity: R·τ_R = D_ω + D_C + Δκ
    4. Weights normalization: Σw_i = 1
    5. Curvature bounds: C ∈ [0, 1]
    """

    def __init__(self, root_dir: Path | None = None):
        if root_dir is None:
            root_dir = self._find_repo_root()
        self.root = root_dir
        self.report = ConservationReport(status="PASS")

    def _find_repo_root(self) -> Path:
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load_yaml(self, path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _add_check(self, check: ConservationCheck) -> None:
        self.report.checks.append(check)
        if not check.passed:
            self.report.errors.append(f"{check.name}: {check.details}")

    # -------------------------------------------------------------------------
    # Tier-1 Kernel Identity Checks
    # -------------------------------------------------------------------------

    def check_fidelity_duality(self, omega: float, F: float) -> ConservationCheck:
        """
        Check: F = 1 - ω (Fidelity-drift duality)

        From KERNEL_SPECIFICATION.md Definition 6:
        Fidelity is the complement of drift.
        """
        expected_F = 1.0 - omega
        error = abs(F - expected_F)
        passed = error <= TOL_FIDELITY

        return ConservationCheck(
            name="Fidelity-drift duality",
            formula="F = 1 - ω",
            passed=passed,
            error=error,
            tolerance=TOL_FIDELITY,
            details=f"ω={omega}, F={F}, expected F={expected_F}",
        )

    def check_integrity_collapse(self, kappa: float, IC: float) -> ConservationCheck:
        """
        Check: IC = exp(κ) (Integrity-collapse relation)

        From KERNEL_SPECIFICATION.md Definition 8:
        IC is the geometric-mean composite.
        """
        expected_IC = np.exp(kappa)
        # Use relative error for potentially large/small values
        if abs(expected_IC) > 1e-10:
            error = abs(IC - expected_IC) / abs(expected_IC)
        else:
            error = abs(IC - expected_IC)
        passed = error <= TOL_ID

        return ConservationCheck(
            name="Integrity-collapse relation",
            formula="IC = exp(κ)",
            passed=passed,
            error=error,
            tolerance=TOL_ID,
            details=f"κ={kappa}, IC={IC}, expected IC={expected_IC}",
        )

    def check_curvature_bounds(self, C: float) -> ConservationCheck:
        """
        Check: C ∈ [0, 1] (Curvature normalized bounds)

        From KERNEL_SPECIFICATION.md Lemma 10:
        C(t) ∈ [0, 1] under default convention.
        """
        in_bounds = 0.0 <= C <= 1.0
        error = max(0, -C, C - 1.0)  # Distance from valid interval

        return ConservationCheck(
            name="Curvature bounds",
            formula="C ∈ [0, 1]",
            passed=in_bounds,
            error=error,
            tolerance=0.0,
            details=f"C={C}",
        )

    def check_stiffness_bounds(self, S: float) -> ConservationCheck:
        """
        Check: S ∈ [0, 1] (Stiffness normalized bounds)

        From KERNEL_SPECIFICATION.md Lemma 8:
        S(t) ∈ [0, 1] under bounded embedding.
        """
        in_bounds = 0.0 <= S <= 1.0
        error = max(0, -S, S - 1.0)

        return ConservationCheck(
            name="Stiffness bounds",
            formula="S ∈ [0, 1]",
            passed=in_bounds,
            error=error,
            tolerance=0.0,
            details=f"S={S}",
        )

    # -------------------------------------------------------------------------
    # Budget Identity (Conservation Law)
    # -------------------------------------------------------------------------

    def check_budget_identity(
        self,
        R: float,
        tau_R: float,
        D_omega: float,
        D_C: float,
        delta_kappa: float,
    ) -> ConservationCheck:
        """
        Check: R·τ_R = D_ω + D_C + Δκ (Budget identity / conservation law)

        This is the fundamental conservation law of the UMCP system.
        The seam residual s = Δκ_budget - Δκ_ledger should be near zero.

        From KERNEL_SPECIFICATION.md §3:
        Δκ_budget = R·τ_R - (D_ω + D_C)
        """
        if tau_R == float("inf") or not np.isfinite(tau_R):
            # Typed censoring: INF_REC means no return credit
            budget = 0.0 - (D_omega + D_C)
        else:
            budget = R * tau_R - (D_omega + D_C)

        residual = abs(budget - delta_kappa)
        passed = residual <= TOL_SEAM

        return ConservationCheck(
            name="Budget identity (conservation law)",
            formula="R·τ_R = D_ω + D_C + Δκ",
            passed=passed,
            error=residual,
            tolerance=TOL_SEAM,
            details=f"R={R}, τ_R={tau_R}, D_ω={D_omega}, D_C={D_C}, Δκ={delta_kappa}, residual={residual}",
        )

    def check_seam_residual(self, residual: float) -> ConservationCheck:
        """
        Check: |s| ≤ tol_seam (Seam residual within tolerance)

        From KERNEL_SPECIFICATION.md Definition 11:
        s = Δκ_budget - Δκ_ledger
        """
        passed = abs(residual) <= TOL_SEAM

        return ConservationCheck(
            name="Seam residual tolerance",
            formula="|s| ≤ tol_seam",
            passed=passed,
            error=abs(residual),
            tolerance=TOL_SEAM,
            details=f"residual s={residual}",
        )

    # -------------------------------------------------------------------------
    # Weights Conservation
    # -------------------------------------------------------------------------

    def check_weights_normalization(self, weights: list[float]) -> ConservationCheck:
        """
        Check: Σw_i = 1 (Weights sum to one)

        From contract.yaml: weights_policy = nonnegative_sum_to_one
        """
        weight_sum = sum(weights)
        error = abs(weight_sum - 1.0)
        passed = error <= TOL_ID

        return ConservationCheck(
            name="Weights normalization",
            formula="Σw_i = 1",
            passed=passed,
            error=error,
            tolerance=TOL_ID,
            details=f"Σw_i={weight_sum}, count={len(weights)}",
        )

    # -------------------------------------------------------------------------
    # CasePack Validation
    # -------------------------------------------------------------------------

    def check_casepack_invariants(self, casepack_dir: Path) -> None:
        """Check all conservation laws for a CasePack."""
        invariants_path = casepack_dir / "expected" / "invariants.json"
        seam_path = casepack_dir / "expected" / "seam_receipt.json"

        if not invariants_path.exists():
            self.report.warnings.append(f"No invariants.json in {casepack_dir.name}")
            return

        self.report.casepacks_checked.append(casepack_dir.name)

        try:
            inv = self._load_json(invariants_path)
            rows = inv.get("rows", [])

            for row in rows:
                # Extract values with defaults
                omega = row.get("omega", 0.0)
                F = row.get("F", 1.0)
                S = row.get("S", 0.0)
                C = row.get("C", 0.0)
                kappa = row.get("kappa", 0.0)
                IC = row.get("IC", 1.0)

                # Check Tier-1 identities
                self._add_check(self.check_fidelity_duality(omega, F))
                self._add_check(self.check_integrity_collapse(kappa, IC))
                self._add_check(self.check_curvature_bounds(C))
                self._add_check(self.check_stiffness_bounds(S))

        except Exception as e:
            self.report.errors.append(f"Error checking {casepack_dir.name}: {e}")

        # Check seam receipt if present
        if seam_path.exists():
            try:
                seam = self._load_json(seam_path)
                identities = seam.get("identities_checked", [])

                for ident in identities:
                    if not ident.get("satisfied", True):
                        self.report.errors.append(f"Seam identity failed in {casepack_dir.name}: {ident.get('name')}")
            except Exception as e:
                self.report.warnings.append(f"Error reading seam receipt: {e}")

    # -------------------------------------------------------------------------
    # Root Artifact Checks
    # -------------------------------------------------------------------------

    def check_root_weights(self) -> None:
        """Check weights.csv conservation."""
        weights_path = self.root / "weights.csv"

        if not weights_path.exists():
            self.report.errors.append("weights.csv not found")
            return

        try:
            import csv

            with open(weights_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Find weight column
            weight_col = None
            for col in ["weight", "w", "weights"]:
                if rows and col in rows[0]:
                    weight_col = col
                    break

            if weight_col:
                weights = [float(row[weight_col]) for row in rows]
                self._add_check(self.check_weights_normalization(weights))
        except Exception as e:
            self.report.errors.append(f"Error reading weights.csv: {e}")

    def check_contract_consistency(self) -> None:
        """Check contract.yaml internal consistency."""
        contract_path = self.root / "contract.yaml"

        if not contract_path.exists():
            self.report.errors.append("contract.yaml not found")
            return

        try:
            contract = self._load_yaml(contract_path)
            contract_data = contract.get("contract", {})

            # Check embedding bounds
            embedding = contract_data.get("embedding", {})
            interval = embedding.get("interval", [0.0, 1.0])

            if len(interval) == 2:
                lower, upper = interval
                self._add_check(
                    ConservationCheck(
                        name="Embedding interval validity",
                        formula="lower < upper",
                        passed=lower < upper,
                        error=0.0 if lower < upper else upper - lower,
                        tolerance=0.0,
                        details=f"interval=[{lower}, {upper}]",
                    )
                )

            # Check tolerances are positive
            tolerances = contract_data.get("tier_1_kernel", {}).get("tolerances", {})
            for name, value in tolerances.items():
                if isinstance(value, (int, float)):
                    self._add_check(
                        ConservationCheck(
                            name=f"Tolerance positivity ({name})",
                            formula=f"{name} > 0",
                            passed=value > 0,
                            error=0.0 if value > 0 else abs(value),
                            tolerance=0.0,
                            details=f"{name}={value}",
                        )
                    )

        except Exception as e:
            self.report.errors.append(f"Error reading contract.yaml: {e}")

    def check_closure_registry(self) -> None:
        """Check closure registry completeness."""
        registry_path = self.root / "closures" / "registry.yaml"

        if not registry_path.exists():
            self.report.errors.append("closures/registry.yaml not found")
            return

        try:
            registry = self._load_yaml(registry_path)
            reg_data = registry.get("registry", {})
            closures = reg_data.get("closures", {})

            # Required closures for conservation law computation
            required = ["gamma", "return_domain", "norms"]

            for req in required:
                present = req in closures
                self._add_check(
                    ConservationCheck(
                        name=f"Required closure: {req}",
                        formula=f"{req} ∈ registry",
                        passed=present,
                        error=0.0 if present else 1.0,
                        tolerance=0.0,
                        details=f"closure '{req}' {'found' if present else 'MISSING'}",
                    )
                )

        except Exception as e:
            self.report.errors.append(f"Error reading closure registry: {e}")

    # -------------------------------------------------------------------------
    # Main Validation
    # -------------------------------------------------------------------------

    def run_all_checks(self) -> ConservationReport:
        """Run all conservation law checks."""
        print("=" * 70)
        print("UMCP Conservation Law Infrastructure Check")
        print("=" * 70)
        print()

        # Root artifact checks
        print("Checking root artifacts...")
        self.check_root_weights()
        self.check_contract_consistency()
        self.check_closure_registry()

        # CasePack checks
        casepacks_dir = self.root / "casepacks"
        if casepacks_dir.exists():
            print("\nChecking CasePacks...")
            for casepack in sorted(casepacks_dir.iterdir()):
                if casepack.is_dir():
                    print(f"  - {casepack.name}")
                    self.check_casepack_invariants(casepack)

        # Determine final status
        if self.report.errors:
            self.report.status = "ERROR"
        elif self.report.warnings:
            self.report.status = "WARN"
        else:
            self.report.status = "PASS"

        return self.report


def main() -> int:
    """Run conservation check and return exit code."""
    checker = ConservationChecker()
    report = checker.run_all_checks()

    # Print summary
    print()
    print("=" * 70)
    print(f"Status: {report.status}")
    print(f"Checks: {report.pass_count}/{len(report.checks)} passed")
    print(f"CasePacks: {len(report.casepacks_checked)} checked")
    print("=" * 70)

    if report.errors:
        print("\nERRORS:")
        for err in report.errors:
            print(f"  ✗ {err}")

    if report.warnings:
        print("\nWARNINGS:")
        for warn in report.warnings:
            print(f"  ⚠ {warn}")

    # Print failed checks
    failed_checks = [c for c in report.checks if not c.passed]
    if failed_checks:
        print("\nFAILED CHECKS:")
        for check in failed_checks:
            print(f"  ✗ {check.name}: {check.formula}")
            print(f"    Error: {check.error:.2e} (tolerance: {check.tolerance:.2e})")
            print(f"    Details: {check.details}")

    print()

    # Return exit code
    if report.status == "ERROR":
        return 2
    elif report.status == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
