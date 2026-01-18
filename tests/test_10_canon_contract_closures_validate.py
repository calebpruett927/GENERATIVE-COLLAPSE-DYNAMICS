from __future__ import annotations

from typing import Any, List, Tuple

from .conftest import (
    RepoPaths,
    load_schema,
    load_yaml,
    require_file,
    validate_instance,
)


def _fmt_errors(prefix: str, errors: List[str], limit: int = 80) -> str:
    if not errors:
        return ""
    shown = errors[:limit]
    more = len(errors) - len(shown)
    tail = f"\n... ({more} more)" if more > 0 else ""
    return prefix + "\n" + "\n".join(shown) + tail


def test_canon_anchors_conform_to_schema(repo_paths: RepoPaths) -> None:
    require_file(repo_paths.canon_anchors)
    anchors = load_yaml(repo_paths.canon_anchors)
    schema = load_schema(repo_paths, "canon.anchors.schema.json")

    errors = validate_instance(anchors, schema)
    assert not errors, _fmt_errors(
        "canon/anchors.yaml failed schema validation:",
        errors,
    )


def test_all_contracts_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates every contracts/*.yaml against schemas/contract.schema.json
    and reports failures per file.
    """
    schema = load_schema(repo_paths, "contract.schema.json")
    contract_files = sorted(repo_paths.contracts_dir.glob("*.yaml"))

    assert contract_files, f"No contract files found in {repo_paths.contracts_dir.as_posix()}"

    failures: List[Tuple[str, List[str]]] = []
    for cf in contract_files:
        require_file(cf)
        doc = load_yaml(cf)
        errs = validate_instance(doc, schema)
        if errs:
            failures.append((cf.as_posix(), errs))

    if failures:
        msg_lines = ["One or more contract files failed schema validation:"]
        for path, errs in failures:
            msg_lines.append(f"\n--- {path} ---")
            msg_lines.extend(errs)
        raise AssertionError("\n".join(msg_lines))


def test_closure_registry_and_referenced_files_conform_to_schema(repo_paths: RepoPaths) -> None:
    """
    Validates:
      - closures/registry.yaml against schemas/closures.schema.json
      - every referenced closure file (registry.registry.closures.*.path) against same schema
    Reports failures per file.
    """
    schema = load_schema(repo_paths, "closures.schema.json")

    # Registry validation
    require_file(repo_paths.closures_registry)
    registry = load_yaml(repo_paths.closures_registry)

    reg_errors = validate_instance(registry, schema)
    assert not reg_errors, _fmt_errors(
        "closures/registry.yaml failed schema validation:",
        reg_errors,
    )

    reg_obj: Any = registry.get("registry", {})
    closures_obj: Any = reg_obj.get("closures", {})
    assert isinstance(closures_obj, dict) and closures_obj, (
        "closures/registry.yaml must include a non-empty mapping at registry.closures."
    )

    ref_paths: List[str] = []
    for _, spec in closures_obj.items():
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            ref_paths.append(spec["path"])

    assert ref_paths, (
        "closures/registry.yaml must reference at least one closure file via registry.closures.<name>.path."
    )

    failures: List[Tuple[str, List[str]]] = []
    for rel_path in ref_paths:
        closure_path = (repo_paths.root / rel_path).resolve()
        require_file(closure_path)
        closure_doc = load_yaml(closure_path)
        errs = validate_instance(closure_doc, schema)
        if errs:
            failures.append((closure_path.as_posix(), errs))

    if failures:
        msg_lines = ["One or more closure files failed schema validation:"]
        for path, errs in failures:
            msg_lines.append(f"\n--- {path} ---")
            msg_lines.extend(errs)
        raise AssertionError("\n".join(msg_lines))
