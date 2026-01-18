from __future__ import annotations

from .conftest import (
    RepoPaths,
    build_psi_doc,
    infer_psi_format,
    load_json,
    load_schema,
    parse_csv_as_rows,
    require_dir,
    require_file,
    validate_instance,
)


def test_hello_world_structure_exists(repo_paths: RepoPaths) -> None:
    """
    Verifies the hello_world CasePack exists and contains the minimum required files.
    """
    require_dir(repo_paths.hello_world_dir)
    require_file(repo_paths.hello_manifest)

    require_dir(repo_paths.hello_expected_dir)
    require_file(repo_paths.hello_psi_csv)
    require_file(repo_paths.hello_invariants_json)
    require_file(repo_paths.hello_ss1m_receipt_json)


def test_hello_world_manifest_conforms(repo_paths: RepoPaths) -> None:
    """
    Validates casepacks/hello_world/manifest.json against manifest.schema.json.
    """
    manifest = load_json(repo_paths.hello_manifest)
    schema = load_schema(repo_paths, "manifest.schema.json")

    errors = validate_instance(manifest, schema)
    assert not errors, (
        "casepacks/hello_world/manifest.json failed schema validation:\n" + "\n".join(errors)
    )


def test_hello_world_psi_conforms(repo_paths: RepoPaths) -> None:
    """
    Parses expected/psi.csv and validates the resulting document against trace.psi.schema.json.
    """
    rows = parse_csv_as_rows(repo_paths.hello_psi_csv)
    fmt = infer_psi_format(rows)
    psi_doc = build_psi_doc(rows, fmt)

    schema = load_schema(repo_paths, "trace.psi.schema.json")
    errors = validate_instance(psi_doc, schema)
    assert not errors, (
        "casepacks/hello_world/expected/psi.csv (parsed) failed trace.psi schema validation:\n"
        + "\n".join(errors)
    )


def test_hello_world_invariants_conform(repo_paths: RepoPaths) -> None:
    """
    Validates expected/invariants.json against invariants.schema.json.
    """
    inv = load_json(repo_paths.hello_invariants_json)
    schema = load_schema(repo_paths, "invariants.schema.json")

    errors = validate_instance(inv, schema)
    assert not errors, (
        "casepacks/hello_world/expected/invariants.json failed invariants schema validation:\n"
        + "\n".join(errors)
    )


def test_hello_world_ss1m_receipt_conforms(repo_paths: RepoPaths) -> None:
    """
    Validates expected/ss1m_receipt.json against receipt.ss1m.schema.json.
    """
    ss1m = load_json(repo_paths.hello_ss1m_receipt_json)
    schema = load_schema(repo_paths, "receipt.ss1m.schema.json")

    errors = validate_instance(ss1m, schema)
    assert not errors, (
        "casepacks/hello_world/expected/ss1m_receipt.json failed SS1m schema validation:\n"
        + "\n".join(errors)
    )
