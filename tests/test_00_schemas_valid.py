from __future__ import annotations

from typing import Any

import pytest

from .conftest import RepoPaths, load_json, require_dir, require_file


def _get_draft202012_validator():
    """
    Resolve the Draft 2020-12 validator from jsonschema in a way that
    won’t explode if the dependency is missing.
    """
    try:
        from jsonschema.validators import Draft202012Validator  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"jsonschema (Draft202012Validator) not available: {exc}")
    return Draft202012Validator


def test_schemas_directory_exists(repo_paths: RepoPaths) -> None:
    require_dir(repo_paths.schemas_dir)


def test_all_schema_files_are_valid_json_and_valid_draft202012_schema(
    repo_paths: RepoPaths,
) -> None:
    """
    Ensures every *.json file under schemas/ is:
      - valid JSON
      - a valid Draft 2020-12 schema (Draft202012Validator.check_schema passes)
    """
    Draft202012Validator = _get_draft202012_validator()

    # Allow nested schema folders, not just flat schemas/*.json
    schema_files = sorted(repo_paths.schemas_dir.rglob("*.json"))
    assert schema_files, f"No schemas found under {repo_paths.schemas_dir.as_posix()}"

    for sf in schema_files:
        require_file(sf)
        schema: Any = load_json(sf)
        Draft202012Validator.check_schema(schema)


def test_schema_ids_are_present_and_local(repo_paths: RepoPaths) -> None:
    """
    Enforces the project convention:
      - every schema has a string $id
      - $id is a local repo-relative path starting with 'schemas/'
    """
    schema_files = sorted(repo_paths.schemas_dir.rglob("*.json"))
    assert schema_files, f"No schemas found under {repo_paths.schemas_dir.as_posix()}"

    for sf in schema_files:
        schema: Any = load_json(sf)
        assert "$id" in schema, f"Schema missing $id: {sf.as_posix()}"
        assert isinstance(
            schema["$id"], str
        ), f"Schema $id must be a string: {sf.as_posix()}"
        assert schema["$id"].startswith(
            "schemas/"
        ), f"Schema $id should be a local path starting with 'schemas/': {sf.as_posix()}"
