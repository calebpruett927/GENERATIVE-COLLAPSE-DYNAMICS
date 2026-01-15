from __future__ import annotations

from .conftest import (
    RepoPaths,
    load_schema,
    load_yaml,
    repo_paths,
    require_file,
    validate_instance,
)


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
    assert not errors, "contracts/UMA.INTSTACK.v1.yaml failed schema validation:\n" + "\n".join(errors)


def test_closure_registry_and_files_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates:
      - closures/registry.yaml is valid under closures.schema.json
      - referenced closure files exist and validate under same schema
    """
    require_file(repo_paths.closures_registry)
    registry = load_yaml(repo_paths.closures_registry)
    schema = load_schema(repo_paths, "closures.schema.json")

    errors = validate_instance(registry, schema)
    assert not errors, "closures/registry.yaml failed schema validation:\n" + "\n".join(errors)

    # Resolve referenced closure file paths
    reg_obj = registry.get("registry", {})
    closures = reg_obj.get("closures", {})
    assert isinstance(closures, dict) and closures, "closures/registry.yaml must include registry.closures mapping."

    ref_paths = []
    for key in ["gamma", "return_domain", "norms", "curvature_neighborhood"]:
        if key in closures and isinstance(closures[key], dict) and "path" in closures[key]:
            ref_paths.append(closures[key]["path"])

    assert ref_paths, "closures/registry.yaml should reference at least gamma, return_domain, norms closure files."

    for p in ref_paths:
        closure_path = (repo_paths.root / p).resolve()
        require_file(closure_path)
        closure_doc = load_yaml(closure_path)
        errors = validate_instance(closure_doc, schema)
        assert not errors, f"Closure file failed schema validation: {closure_path.as_posix()}\n" + "\n".join(errors)
