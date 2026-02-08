"""Schema-per-instance acceptance tests.

Validates that real data files in the repo conform to their declared schemas.
Schemas are the protocol's type system — if a schema accepts invalid data,
every downstream check is meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Optional jsonschema
try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"

pytestmark = pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")


def _load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_schema(name: str) -> dict[str, Any]:
    return _load_json(SCHEMAS_DIR / name)


def _validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    assert jsonschema is not None
    validator_cls = jsonschema.Draft202012Validator
    validator = validator_cls(schema)
    return [e.message for e in validator.iter_errors(instance)]


# ============================================================================
# Real data against its schema
# ============================================================================


class TestSchemaAcceptance:
    """Validate real repo data files against their schemas."""

    def test_contract_schema_accepts_uma_contract(self) -> None:
        """contracts/UMA.INTSTACK.v1.yaml validates against contract.schema.json."""
        contract = _load_yaml(REPO_ROOT / "contracts" / "UMA.INTSTACK.v1.yaml")
        schema = _load_schema("contract.schema.json")
        errors = _validate(contract, schema)
        assert errors == [], f"Contract validation errors: {errors}"

    def test_manifest_schema_accepts_hello_manifest(self) -> None:
        """casepacks/hello_world/manifest.json validates against manifest.schema.json."""
        manifest = _load_json(REPO_ROOT / "casepacks" / "hello_world" / "manifest.json")
        schema = _load_schema("manifest.schema.json")
        errors = _validate(manifest, schema)
        assert errors == [], f"Manifest validation errors: {errors}"

    def test_invariants_schema_accepts_hello_invariants(self) -> None:
        """casepacks/hello_world/expected/invariants.json validates."""
        inv = _load_json(REPO_ROOT / "casepacks" / "hello_world" / "expected" / "invariants.json")
        schema = _load_schema("invariants.schema.json")
        errors = _validate(inv, schema)
        assert errors == [], f"Invariants validation errors: {errors}"

    def test_receipt_schema_accepts_hello_receipt(self) -> None:
        """casepacks/hello_world/expected/ss1m_receipt.json validates."""
        receipt = _load_json(REPO_ROOT / "casepacks" / "hello_world" / "expected" / "ss1m_receipt.json")
        schema = _load_schema("receipt.ss1m.schema.json")
        errors = _validate(receipt, schema)
        assert errors == [], f"Receipt validation errors: {errors}"

    def test_closures_registry_schema(self) -> None:
        """closures/registry.yaml validates against closures.schema.json."""
        registry = _load_yaml(REPO_ROOT / "closures" / "registry.yaml")
        schema = _load_schema("closures.schema.json")
        errors = _validate(registry, schema)
        assert errors == [], f"Registry validation errors: {errors}"

    def test_canon_anchors_schema(self) -> None:
        """canon/anchors.yaml validates against canon.anchors.schema.json."""
        anchors = _load_yaml(REPO_ROOT / "canon" / "anchors.yaml")
        schema = _load_schema("canon.anchors.schema.json")
        errors = _validate(anchors, schema)
        assert errors == [], f"Anchors validation errors: {errors}"

    def test_validator_rules_schema(self) -> None:
        """validator_rules.yaml validates against validator.rules.schema.json."""
        rules = _load_yaml(REPO_ROOT / "validator_rules.yaml")
        schema = _load_schema("validator.rules.schema.json")
        errors = _validate(rules, schema)
        assert errors == [], f"Rules validation errors: {errors}"


# ============================================================================
# Schema rejection tests (invalid instances should fail)
# ============================================================================


class TestSchemaRejection:
    """Schemas must reject invalid data."""

    def test_contract_rejects_empty(self) -> None:
        schema = _load_schema("contract.schema.json")
        errors = _validate({}, schema)
        assert len(errors) > 0

    def test_manifest_rejects_empty(self) -> None:
        schema = _load_schema("manifest.schema.json")
        errors = _validate({}, schema)
        assert len(errors) > 0

    def test_invariants_rejects_no_rows(self) -> None:
        schema = _load_schema("invariants.schema.json")
        errors = _validate({"rows": "not_a_list"}, schema)
        assert len(errors) > 0

    def test_receipt_rejects_no_receipt(self) -> None:
        schema = _load_schema("receipt.ss1m.schema.json")
        errors = _validate({}, schema)
        assert len(errors) > 0


# ============================================================================
# Schema syntactic validity
# ============================================================================


class TestAllSchemasLoadable:
    """All 12 schema files are valid JSON and parseable."""

    @pytest.mark.parametrize(
        "schema_name",
        [
            "canon.anchors.schema.json",
            "closures.schema.json",
            "closures_registry.schema.json",
            "contract.schema.json",
            "failure_node_atlas.schema.json",
            "glossary.schema.json",
            "invariants.schema.json",
            "manifest.schema.json",
            "receipt.ss1m.schema.json",
            "trace.psi.schema.json",
            "validator.result.schema.json",
            "validator.rules.schema.json",
        ],
    )
    def test_schema_is_valid_json(self, schema_name: str) -> None:
        schema_path = SCHEMAS_DIR / schema_name
        assert schema_path.exists(), f"Missing schema: {schema_name}"
        schema = _load_json(schema_path)
        assert isinstance(schema, dict)
        # Must have "$schema" or "type" at minimum
        assert "$schema" in schema or "type" in schema or "properties" in schema
