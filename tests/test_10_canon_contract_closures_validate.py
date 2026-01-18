from __future__ import annotations

from typing import Any, List

from .conftest import (
    RepoPaths,
    load_schema,
    load_yaml,
    require_file,
    validate_instance,
)


def test_canon_anchors_conform_to_schema(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.canon_anchors)
    anchors = load_yaml(repo_paths.canon_anchors)
    schema = load_schema(repo_paths, "canon.anchors.schema.json")

    errors = validate_instance(anchors, schema)
    assert not errors, "canon/anchors.yaml failed schema validation:\n" + "\n".join(errors)


def test_all_contracts_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates every *.yaml in contracts/ against schemas/contract.schema.json.
    Reports per-file errors so CI failures are actionable.
    """
    schema = load_schema(repo_paths, "contract.schema.json")

    contract_files = sorted(repo_paths.contracts_dir.glob("*.yaml"))
    assert contract_files, f"No contract files found in {repo_paths.contracts_dir.as_posix()}"

    failures: List[str] = []
    for f in contract_files:
        require_file(f)
        doc = load_yaml(f)
        errors = validate_instance(doc, schema)
        if errors:
            failures.append(f"{f.as_posix()}:\n  " + "\n  ".join(errors))

    assert not failures, "One or more contract files failed schema validation:\n" + "\n\n".join(failures)


def test_closure_registry_and_referenced_files_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates:
      - closures/registry.yaml conforms to schemas/closures.schema.json
      - every referenced closure file exists and conforms to schemas/closures.schema.json
    Reports per-file errors so CI failures are actionable.
    """
    require_file(repo_paths.closures_registry)
    registry = load_yaml(repo_paths.closures_registry)
    schema = load_schema(repo_paths, "closures.schema.json")

    reg_errors = validate_instance(registry, schema)
    assert not reg_errors, "closures/registry.yaml failed schema validation:\n" + "\n".join(reg_errors)

    reg_obj: Any = registry.get("registry", {})
    closures: Any = reg_obj.get("closures", {})

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
    for rel_path in ref_paths:
        closure_path = (repo_paths.root / rel_path).resolve()
        require_file(closure_path)

        closure_doc = load_yaml(closure_path)
        errors = validate_instance(closure_doc, schema)
        if errors:
            failures.append(f"{closure_path.as_posix()}:\n  " + "\n  ".join(errors))

    assert not failures, "One or more closure files failed schema validation:\n" + "\n\n".join(failures)
