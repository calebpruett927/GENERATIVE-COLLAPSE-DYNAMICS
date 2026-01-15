from __future__ import annotations

from pathlib import Path

from .conftest import RepoPaths, load_json, require_dir, require_file, repo_paths, validate_instance, ensure_jsonschema_available
from .conftest import Draft202012Validator  # type: ignore


def test_schemas_directory_exists(repo_paths: RepoPaths) -> None:
    require_dir(repo_paths.schemas_dir)


def test_all_schema_files_are_valid_json_and_valid_jsonschema(repo_paths: RepoPaths) -> None:
    """
    Ensures every *.json file under schemas/ is:
      - valid JSON
      - a valid Draft 2020-12 schema (Draft202012Validator.check_schema passes)
    """
    ensure_jsonschema_available()
    schema_files = sorted(repo_paths.schemas_dir.glob("*.json"))
    assert schema_files, f"No schemas found in {repo_paths.schemas_dir.as_posix()}"

    for sf in schema_files:
        require_file(sf)
        schema = load_json(sf)
        Draft202012Validator.check_schema(schema)  # type: ignore


def test_schema_ids_are_present(repo_paths: RepoPaths) -> None:
    schema_files = sorted(repo_paths.schemas_dir.glob("*.json"))
    for sf in schema_files:
        schema = load_json(sf)
        assert "$id" in schema, f"Schema missing $id: {sf.as_posix()}"
        assert isinstance(schema["$id"], str) and schema["$id"].startswith("schemas/"), f"Schema $id should be a local path: {sf.as_posix()}"
