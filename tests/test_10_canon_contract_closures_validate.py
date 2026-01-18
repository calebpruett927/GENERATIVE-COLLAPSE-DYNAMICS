from __future__ import annotations

from typing import Any, Iterable, List

from .conftest import (
    RepoPaths,
    load_schema,
    load_yaml,
    require_file,
    validate_instance,
)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def test_canon_anchors_conform_to_schema(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.canon_anchors)
    anchors = load_yaml(repo_paths.canon_anchors)
    schema = load_schema(repo_paths, "canon.anchors.schema.json")

    errors = validate_instance(anchors, schema)
    assert not errors, "canon/anchors.yaml failed schema validation:\n" + "\n".join(errors)


def test_contract_conforms_to_schema(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.contract)
    contract = load_yaml(repo_paths.contract)
    schema = load_schema(repo_paths, "contract.schema.json")

    errors = validate_instance(contract, schema)
    assert not errors, "contracts/* contract file failed schema validation:\n" + "\n".join(errors)


def test_closure_registry_and_referenced_files_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates:
      - closures/registry.yaml conforms to closures.schema.json
      - any referenced closure files exist and also conform to closures.schema.json
    """
    require_file(repo_paths.closures_registry)
    registry = load_yaml(repo_paths.closures_registry)
    schema = load_schema(repo_paths, "closures.schema.json")

    errors = validate_instance(registry, schema)
    assert not errors, "closures/registry.yaml failed schema validation:\n" + "\n".join(errors)

    # Resolve referenced closure file paths from registry.registry.closures.*.path
    reg_obj: Any = registry.get("registry", {})
    closures: Any = reg_obj.get("closures", {})

    assert isinstance(closures, dict) and closures, (
        "closures/registry.yaml must include a non-empty mapping at registry.closures."
    )

    ref_paths: List[str] = []
    for name, spec in closures.items():
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            ref_paths.append(spec["path"])

    assert ref_paths, (
        "closures/registry.yaml must reference at least one closure file via registry.closures.<name>.path."
    )

    for rel_path in ref_paths:
        closure_path = (repo_paths.root / rel_path).resolve()
        require_file(closure_path)

        closure_doc = load_yaml(closure_path)
        file_errors = validate_instance(closure_doc, schema)
        assert not file_errors, (
            f"Closure file failed schema validation: {closure_path.as_posix()}\n"
            + "\n".join(file_errors)
        )
