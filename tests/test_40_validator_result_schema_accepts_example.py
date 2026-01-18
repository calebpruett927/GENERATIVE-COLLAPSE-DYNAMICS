from __future__ import annotations

from typing import Any, Dict

from .conftest import RepoPaths, load_schema, validate_instance


def test_validator_result_schema_accepts_minimal_example(repo_paths: RepoPaths) -> None:
    """
    Confirms schemas/validator.result.schema.json accepts a minimal-but-valid machine output.
    This does NOT test the validator implementation — it tests the schema contract.
    """
    schema = load_schema(repo_paths, "validator.result.schema.json")

    # Keep this example intentionally small, but structurally realistic.
    example: Dict[str, Any] = {
        "schema": "schemas/validator.result.schema.json",
        "created_utc": "2026-01-14T20:00:00Z",
        "validator": {
            "name": "umcp-validator",
            "version": "0.1.0",
            "implementation": {
                "language": "python",
                "git_commit": "deadbeef",
                "build": "dev",
            },
        },
        "run_status": "CONFORMANT",
        "summary": {
            "counts": {
                "errors": 0,
                "warnings": 0,
                "info": 1,
                "targets_total": 1,
                "targets_failed": 0,
            },
            "policy": {
                "strict": False,
                "fail_on_warning": False,
            },
        },
        "targets": [
            {
                "target_type": "repo",
                "target_path": ".",
                "run_status": "CONFORMANT",
                "counts": {"errors": 0, "warnings": 0, "info": 1},
                "issues": [],
                "artifacts": [{"kind": "canon_anchors", "path": "canon/anchors.yaml"}],
            }
        ],
        "issues": [],
        "notes": "schema acceptance test",
    }

    errors = validate_instance(example, schema)
    assert not errors, "validator.result schema rejected minimal example:\n" + "\n".join(errors)
