from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from .conftest import (
    RepoPaths,
    load_schema,
    load_yaml,
    require_file,
    validate_instance,
)


def _schema_basename(schema_ref: str) -> str:
    """
    Normalize schema references like:
      - "schemas/contract.schema.json" -> "contract.schema.json"
      - "contract.schema.json" -> "contract.schema.json"
    """
    s = str(schema_ref).strip()
    if not s:
        return s
    # Accept both "schemas/foo.json" and "./schemas/foo.json"
    s = s.replace("\\", "/")
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _declared_schema_name(doc: Any) -> Optional[str]:
    """
    Prefer an explicit per-document schema reference when present.
    Supports either 'schema' (your repo convention) or '$schema' (JSON Schema convention).
    """
    if not isinstance(doc, dict):
        return None
    for key in ("schema", "$schema"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return _schema_basename(val)
    return None


def _load_schema_for_doc(repo_paths: RepoPaths, doc: Any, fallback_schema_name: str) -> Any:
    """
    Load the schema referenced by the document when declared; otherwise fallback.
    """
    name = _declared_schema_name(doc) or fallback_schema_name
    return load_schema(repo_paths, name)


def _iter_yaml_files(dir_path: Path) -> List[Path]:
    """
    Return sorted YAML/YML files in a directory.
    """
    files = sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))
    return files


def test_canon_anchors_conform_to_schema(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.canon_anchors)
    anchors = load_yaml(repo_paths.canon_anchors)

    # Canon anchors are typically pinned to a single schema file name.
    schema = _load_schema_for_doc(repo_paths, anchors, "canon.anchors.schema.json")

    errors = validate_instance(anchors, schema)
    assert not errors, (
        "canon/anchors.yaml failed schema validation:\n"
        + "\n".join(errors)
    )


def test_all_contracts_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validate every YAML contract under contracts/ against its declared schema
    (or contract.schema.json if none is declared).
    """
    contracts_dir = getattr(repo_paths, "contracts_dir", (repo_paths.root / "contracts"))
    assert contracts_dir.exists(), f"Missing contracts directory: {contracts_dir.as_posix()}"

    contract_files = _iter_yaml_files(contracts_dir)
    assert contract_files, f"No contract YAML files found under: {contracts_dir.as_posix()}"

    failures: List[str] = []
    for cf in contract_files:
        require_file(cf)
        doc = load_yaml(cf)
        schema = _load_schema_for_doc(repo_paths, doc, "contract.schema.json")

        errors = validate_instance(doc, schema)
        if errors:
            failures.append(cf.as_posix() + ":\n" + "\n".join(errors))

    assert not failures, "One or more contract files failed schema validation:\n\n" + "\n\n".join(failures)


def test_closure_registry_and_referenced_files_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates:
      - closures/registry.yaml conforms to its declared schema (or closures.schema.json fallback)
      - referenced closure files exist
      - each referenced closure file conforms to *its own* declared schema when present,
        otherwise falls back to closures.schema.json
    """
    require_file(repo_paths.closures_registry)
    registry = load_yaml(repo_paths.closures_registry)

    # Registry may have its own schema; if it doesn't, fall back.
    registry_schema = _load_schema_for_doc(repo_paths, registry, "closures.schema.json")

    reg_errors = validate_instance(registry, registry_schema)
    assert not reg_errors, (
        "closures/registry.yaml failed schema validation:\n"
        + "\n".join(reg_errors)
    )

    # Resolve referenced closure file paths from registry.registry.closures.*.path
    reg_obj: Any = registry.get("registry", {}) if isinstance(registry, dict) else {}
    closures: Any = reg_obj.get("closures", {}) if isinstance(reg_obj, dict) else {}

    assert isinstance(closures, dict) and closures, (
        "closures/registry.yaml must include a non-empty mapping at registry.closures."
    )

    ref_paths: List[str] = []
    for _, spec in closures.items():
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            ref_paths.append(spec["path"])

    assert ref_paths, (
        "closures/registry.yaml must reference at least one closure file via registry.closures.<name>.path."
    )

    failures: List[str] = []
    root = repo_paths.root.resolve()

    for rel_path in ref_paths:
        closure_path = (repo_paths.root / rel_path).resolve()

        # Safety: ensure the resolved path stays within repo root
        try:
            closure_path.relative_to(root)
        except ValueError:
            failures.append(f"{rel_path}: resolves outside repo root -> {closure_path.as_posix()}")
            continue

        try:
            require_file(closure_path)
        except AssertionError as e:
            failures.append(f"{rel_path}: missing file ({e})")
            continue

        closure_doc = load_yaml(closure_path)

        # Each closure file may declare its own schema; respect it.
        # If absent, fall back to closures.schema.json.
        closure_schema = _load_schema_for_doc(repo_paths, closure_doc, "closures.schema.json")

        file_errors = validate_instance(closure_doc, closure_schema)
        if file_errors:
            failures.append(
                f"{closure_path.as_posix()}:\n" + "\n".join(file_errors)
            )

    assert not failures, "One or more closure files failed schema validation:\n\n" + "\n\n".join(failures)
