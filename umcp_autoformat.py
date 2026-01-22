This file has been moved to src/umcp/ and is no longer available here.
#!/usr/bin/env python3
"""
UMCP Contract Auto-Formatter

Automatically formats and validates UMCP contracts, ensuring:
1. Proper tier hierarchy (UMCP -> GCD -> RCFT)
2. Core axiom encoding (no_return_no_credit: true)
3. Consistent structure and indentation
4. All required fields present
5. Extension compatibility

Usage:
    python umcp_autoformat.py <contract_file>
    python umcp_autoformat.py --all          # Format all contracts
    python umcp_autoformat.py --validate     # Validate without formatting
"""

import argparse
import sys
from pathlib import Path
from typing import Any, ClassVar

import yaml


class UMCPContractFormatter:
    """Auto-formatter for UMCP contract files"""

    REQUIRED_TIERS: ClassVar[dict] = {
        "UMA": {
            "tier_level": 0,
            "required_fields": [
                "id",
                "version",
                "timezone",
                "embedding",
                "tier_1_kernel",
                "typed_censoring",
            ],
            "axiom_check": "no_return_no_credit",
        },
        "GCD": {
            "tier_level": 1,
            "required_fields": [
                "id",
                "version",
                "parent_contract",
                "tier_level",
                "axioms",
                "regime_classification",
                "mathematical_identities",
            ],
            "axiom_check": "AX-0",
            "parent": "UMA",
        },
        "RCFT": {
            "tier_level": 2,
            "required_fields": [
                "id",
                "version",
                "parent_contract",
                "tier_level",
                "axioms",
                "regime_classification",
                "mathematical_identities",
            ],
            "axiom_check": "P-RCFT-0",
            "parent": "GCD",
        },
    }

    AXIOM_TEMPLATES: ClassVar[dict] = {
        "no_return_no_credit": {
            "statement": "What returns through collapse is real",
            "description": "Only measurements that return through collapse-reconstruction cycles receive credit. Non-returning quantities are flagged but not credited.",
            "enforcement": True,
        },
        "AX-0": {
            "id": "AX-0",
            "statement": "Collapse is generative",
            "description": "Every collapse event releases generative potential that can be harvested by downstream processes. Φ_gen ≥ 0 always.",
        },
        "P-RCFT-0": {
            "id": "P-RCFT-0",
            "statement": "Tier-2 augments, never overrides",
            "description": "RCFT adds new dimensions of analysis without modifying GCD's Tier-1 foundation. All Tier-1 invariants remain frozen and unchanged.",
        },
    }

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fixes: list[str] = []

    def format_contract(self, contract_path: Path, output_path: Path | None = None) -> bool:
        """
        Format a UMCP contract file

        Args:
            contract_path: Path to contract file
            output_path: Path to write formatted contract (None = overwrite)

        Returns:
            True if successful, False if errors
        """
        self.errors = []
        self.warnings = []
        self.fixes = []

        # Load contract
        try:
            with open(contract_path) as f:
                contract = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load contract: {e}")
            return False

        # Validate structure
        if not self._validate_structure(contract):
            return False

        # Detect tier
        tier = self._detect_tier(contract)
        if tier is None:
            self.errors.append("Could not determine contract tier")
            return False

        # Format contract
        formatted = self._format_tier(contract, tier)

        # Ensure axiom encoding
        formatted = self._ensure_axiom_encoding(formatted, tier)

        # Validate formatted contract
        if not self._validate_formatted(formatted, tier):
            return False

        # Write output
        output = output_path or contract_path
        try:
            with open(output, "w") as f:
                yaml.dump(formatted, f, default_flow_style=False, sort_keys=False, indent=2)
            self.fixes.append(f"Formatted contract written to {output}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to write formatted contract: {e}")
            return False

    def validate_contract(self, contract_path: Path) -> bool:
        """Validate contract without formatting"""
        try:
            with open(contract_path) as f:
                contract = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load contract: {e}")
            return False

        if not self._validate_structure(contract):
            return False

        tier = self._detect_tier(contract)
        if tier is None:
            self.errors.append("Could not determine contract tier")
            return False

        return self._validate_tier_requirements(contract, tier)

    def _validate_structure(self, contract: dict[str, Any]) -> bool:
        """Validate basic contract structure"""
        if "contract" not in contract:
            self.errors.append("Missing top-level 'contract' key")
            return False

        if "schema" not in contract:
            self.warnings.append("Missing 'schema' reference")

        return True

    def _detect_tier(self, contract: dict[str, Any]) -> str | None:
        """Detect contract tier from ID"""
        contract_id = contract.get("contract", {}).get("id", "")

        for tier in ["RCFT", "GCD", "UMA"]:
            if tier in contract_id:
                return tier

        return None

    def _format_tier(self, contract: dict[str, Any], tier: str) -> dict[str, Any]:
        """Format contract according to tier requirements"""
        formatted = contract.copy()
        tier_config = self.REQUIRED_TIERS[tier]

        # Ensure schema reference
        if "schema" not in formatted:
            formatted["schema"] = "schemas/contract.schema.json"
            self.fixes.append("Added schema reference")

        # Ensure tier level
        if "tier_level" not in formatted["contract"] and tier != "UMA":
            formatted["contract"]["tier_level"] = tier_config["tier_level"]
            self.fixes.append(f"Added tier_level: {tier_config['tier_level']}")

        # Ensure parent contract reference
        if "parent" in tier_config and "parent_contract" not in formatted["contract"]:
            parent_id = f"{tier_config['parent']}.INTSTACK.v1"
            formatted["contract"]["parent_contract"] = parent_id
            self.fixes.append(f"Added parent_contract: {parent_id}")

        return formatted

    def _ensure_axiom_encoding(self, contract: dict[str, Any], tier: str) -> dict[str, Any]:
        """Ensure core axiom is properly encoded"""
        tier_config = self.REQUIRED_TIERS[tier]
        axiom_key = tier_config["axiom_check"]

        if tier == "UMA":
            # Ensure no_return_no_credit in typed_censoring
            if "typed_censoring" not in contract["contract"]:
                contract["contract"]["typed_censoring"] = {}
                self.fixes.append("Added typed_censoring section")

            if "no_return_no_credit" not in contract["contract"]["typed_censoring"]:
                contract["contract"]["typed_censoring"]["no_return_no_credit"] = True
                self.fixes.append("Added no_return_no_credit: true (CORE AXIOM)")

        else:
            # Ensure axioms section exists
            if "axioms" not in contract["contract"]:
                contract["contract"]["axioms"] = []
                self.fixes.append("Added axioms section")

            # Check if axiom exists
            axiom_exists = any(
                a.get("id") == axiom_key or a.get("statement") == self.AXIOM_TEMPLATES[axiom_key]["statement"]
                for a in contract["contract"]["axioms"]
            )

            if not axiom_exists:
                # Add axiom
                contract["contract"]["axioms"].insert(0, self.AXIOM_TEMPLATES[axiom_key])
                self.fixes.append(f"Added {axiom_key} axiom")

        return contract

    def _validate_tier_requirements(self, contract: dict[str, Any], tier: str) -> bool:
        """Validate tier-specific requirements"""
        tier_config = self.REQUIRED_TIERS[tier]
        contract_data = contract.get("contract", {})

        # Check required fields
        for field in tier_config["required_fields"]:
            if field not in contract_data:
                self.errors.append(f"Missing required field: {field}")

        # Check axiom encoding
        axiom_key = tier_config["axiom_check"]

        if tier == "UMA":
            typed_censoring = contract_data.get("typed_censoring", {})
            if not typed_censoring.get("no_return_no_credit"):
                self.errors.append("Core axiom not encoded: no_return_no_credit must be true")
        else:
            axioms = contract_data.get("axioms", [])
            axiom_found = any(a.get("id") == axiom_key for a in axioms)
            if not axiom_found:
                self.errors.append(f"Required axiom not found: {axiom_key}")

        return len(self.errors) == 0

    def _validate_formatted(self, contract: dict[str, Any], tier: str) -> bool:
        """Validate formatted contract"""
        return self._validate_tier_requirements(contract, tier)

    def print_report(self):
        """Print formatting report"""
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.fixes:
            print("✅ FIXES APPLIED:")
            for fix in self.fixes:
                print(f"  • {fix}")

        if not self.errors and not self.warnings and not self.fixes:
            print("✅ Contract is valid and properly formatted")


def format_all_contracts(repo_root: Path, formatter: UMCPContractFormatter) -> int:
    """Format all contracts in repository"""
    contract_paths = list((repo_root / "contracts").glob("*.yaml"))

    total = len(contract_paths)
    success = 0
    failed = 0

    print(f"Found {total} contract files\n")

    for contract_path in contract_paths:
        print(f"Processing: {contract_path.name}")
        if formatter.format_contract(contract_path):
            formatter.print_report()
            success += 1
        else:
            formatter.print_report()
            failed += 1
        print()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {success} successful, {failed} failed out of {total} total")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Auto-format and validate UMCP contract files")
    parser.add_argument("contract", nargs="?", help="Contract file to format")
    parser.add_argument("--all", action="store_true", help="Format all contracts in repository")
    parser.add_argument("--validate", action="store_true", help="Validate without formatting")
    parser.add_argument("--strict", action="store_true", help="Strict mode (warnings become errors)")
    parser.add_argument("--output", "-o", help="Output path (default: overwrite input)")

    args = parser.parse_args()

    formatter = UMCPContractFormatter(strict=args.strict)

    # Find repository root
    repo_root = Path.cwd()
    while repo_root != repo_root.parent:
        if (repo_root / "pyproject.toml").exists():
            break
        repo_root = repo_root.parent
    else:
        repo_root = Path.cwd()

    if args.all:
        return format_all_contracts(repo_root, formatter)

    if not args.contract:
        parser.print_help()
        return 1

    contract_path = Path(args.contract)
    output_path = Path(args.output) if args.output else None

    if not contract_path.exists():
        print(f"❌ Contract file not found: {contract_path}")
        return 1

    if args.validate:
        success = formatter.validate_contract(contract_path)
    else:
        success = formatter.format_contract(contract_path, output_path)

    formatter.print_report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
