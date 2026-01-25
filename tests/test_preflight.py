"""
Tests for UMCP Preflight Validator - Failure Node Atlas Enforcement

Tests the preflight validation system that detects meaning-drift failure nodes.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from pathlib import Path

import yaml


class TestPreflightValidator:
    """Test the PreflightValidator class."""

    def test_import(self) -> None:
        """Test that preflight module can be imported."""
        from umcp.preflight import PreflightValidator, PreflightReport, FailureHit

        assert PreflightValidator is not None
        assert PreflightReport is not None
        assert FailureHit is not None

    def test_preflight_report_structure(self) -> None:
        """Test PreflightReport dataclass structure."""
        from umcp.preflight import PreflightReport, FailureHit

        hit = FailureHit(
            node_id="FN-001",
            severity="ERROR",
            evidence={"test": "value"},
            action="Test action",
            phase="A",
        )

        report = PreflightReport(
            run_id="TEST-001",
            status="ERROR",
            hits=[hit],
        )

        assert report.run_id == "TEST-001"
        assert report.status == "ERROR"
        assert report.error_count == 1
        assert report.warn_count == 0
        assert report.exit_code == 2

    def test_preflight_exit_codes(self) -> None:
        """Test exit code mapping."""
        from umcp.preflight import PreflightReport, FailureHit

        # ERROR status -> exit code 2
        report_error = PreflightReport(
            run_id="TEST",
            status="ERROR",
            hits=[
                FailureHit("FN-001", "ERROR", {}, "action", "A"),
            ],
        )
        assert report_error.exit_code == 2

        # WARN status -> exit code 1
        report_warn = PreflightReport(
            run_id="TEST",
            status="WARN",
            hits=[
                FailureHit("FN-001", "WARN", {}, "action", "C"),
            ],
        )
        assert report_warn.exit_code == 1

        # PASS status -> exit code 0
        report_pass = PreflightReport(
            run_id="TEST",
            status="PASS",
            hits=[],
        )
        assert report_pass.exit_code == 0

    def test_preflight_report_json(self) -> None:
        """Test JSON serialization of preflight report."""
        from umcp.preflight import PreflightReport, FailureHit

        hit = FailureHit(
            node_id="FN-007",
            severity="ERROR",
            evidence={"weight_sum": 0.5, "expected": 1.0},
            action="Normalize weights",
            phase="A",
        )

        report = PreflightReport(
            run_id="TEST-001",
            status="ERROR",
            hits=[hit],
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "TEST-001"
        assert parsed["status"] == "ERROR"
        assert len(parsed["hits"]) == 1
        assert parsed["hits"][0]["node_id"] == "FN-007"
        assert parsed["summary"]["error_count"] == 1

    def test_validator_on_repo(self) -> None:
        """Test running validator on actual repo."""
        from umcp.preflight import PreflightValidator

        # Find repo root
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent
        
        validator = PreflightValidator(root_dir=current)
        report = validator.validate()

        # Report should have valid structure
        assert report.run_id is not None
        assert report.status in ("ERROR", "WARN", "PASS")
        assert report.exit_code in (0, 1, 2)


class TestFailureNodeAtlas:
    """Test the Failure Node Atlas specification."""

    def test_atlas_exists(self) -> None:
        """Test that atlas file exists."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        atlas_path = current / "specs" / "failure_node_atlas_v1.yaml"
        assert atlas_path.exists(), "Failure Node Atlas not found"

    def test_atlas_schema_exists(self) -> None:
        """Test that atlas schema exists."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        schema_path = current / "schemas" / "failure_node_atlas.schema.json"
        assert schema_path.exists(), "Failure Node Atlas schema not found"

    def test_atlas_validates_against_schema(self) -> None:
        """Test that atlas file validates against its schema."""
        from jsonschema import Draft202012Validator

        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        schema_path = current / "schemas" / "failure_node_atlas.schema.json"
        atlas_path = current / "specs" / "failure_node_atlas_v1.yaml"

        with open(schema_path) as f:
            schema = json.load(f)

        with open(atlas_path) as f:
            atlas = yaml.safe_load(f)

        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(atlas))  # type: ignore[arg-type]
        
        assert len(errors) == 0, f"Atlas validation errors: {[e.message for e in errors]}"

    def test_atlas_has_required_nodes(self) -> None:
        """Test that atlas contains all 14 failure nodes."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        atlas_path = current / "specs" / "failure_node_atlas_v1.yaml"

        with open(atlas_path) as f:
            atlas = yaml.safe_load(f)

        nodes = atlas["atlas"]["nodes"]
        node_ids = [n["id"] for n in nodes]

        expected_ids = [f"FN-{i:03d}" for i in range(1, 15)]
        
        for expected_id in expected_ids:
            assert expected_id in node_ids, f"Missing node: {expected_id}"

    def test_atlas_phases(self) -> None:
        """Test that all nodes are assigned to valid phases."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        atlas_path = current / "specs" / "failure_node_atlas_v1.yaml"

        with open(atlas_path) as f:
            atlas = yaml.safe_load(f)

        nodes = atlas["atlas"]["nodes"]
        
        for node in nodes:
            assert node["phase"] in ("A", "B", "C"), f"Invalid phase for {node['id']}"

    def test_atlas_severities(self) -> None:
        """Test that all nodes have valid severities."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        atlas_path = current / "specs" / "failure_node_atlas_v1.yaml"

        with open(atlas_path) as f:
            atlas = yaml.safe_load(f)

        nodes = atlas["atlas"]["nodes"]
        
        for node in nodes:
            assert node["severity"] in ("ERROR", "WARN", "INFO"), f"Invalid severity for {node['id']}"


class TestPhaseAChecks:
    """Test Phase A structural conformance checks."""

    def test_fn007_weights_sum(self) -> None:
        """Test FN-007: Weight sum check."""
        from umcp.preflight import PreflightValidator

        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        validator = PreflightValidator(root_dir=current)
        validator.hits = []
        validator._check_fn007_weights()

        # Check if FN-007 was detected (depends on actual weights.csv)
        fn007_hits = [h for h in validator.hits if h.node_id == "FN-007"]
        # Just verify the check runs without error
        assert isinstance(fn007_hits, list)

    def test_fn010_closures(self) -> None:
        """Test FN-010: Closure registry check."""
        from umcp.preflight import PreflightValidator

        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        validator = PreflightValidator(root_dir=current)
        validator.hits = []
        validator._check_fn010_closures()

        # Just verify the check runs without error
        fn010_hits = [h for h in validator.hits if h.node_id == "FN-010"]
        assert isinstance(fn010_hits, list)

    def test_fn014_manifest_integrity(self) -> None:
        """Test FN-014: Manifest integrity check."""
        from umcp.preflight import PreflightValidator

        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                break
            current = current.parent

        validator = PreflightValidator(root_dir=current)
        validator.hits = []
        validator._check_fn014_manifest_integrity()

        # Just verify the check runs without error
        fn014_hits = [h for h in validator.hits if h.node_id == "FN-014"]
        assert isinstance(fn014_hits, list)


class TestPreflightCLI:
    """Test CLI integration for preflight command."""

    def test_cli_preflight_command_exists(self) -> None:
        """Test that preflight command is registered in CLI."""
        from umcp.cli import build_parser

        parser = build_parser()
        # Parse with preflight command
        args = parser.parse_args(["preflight"])
        assert args.cmd == "preflight"

    def test_cli_preflight_options(self) -> None:
        """Test preflight command options."""
        from umcp.cli import build_parser

        parser = build_parser()
        
        # Test with all options
        args = parser.parse_args([
            "preflight",
            ".",
            "--freeze-dir", "/tmp/freeze",
            "--output", "/tmp/report.json",
            "--verbose",
            "--json",
        ])
        
        assert args.path == "."
        assert args.freeze_dir == "/tmp/freeze"
        assert args.output == "/tmp/report.json"
        assert args.verbose is True
        assert args.json is True
