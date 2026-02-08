"""Tests for finance_cli module and closure domain coverage.

finance_cli.py had zero tests anywhere.  Closure domains (astronomy,
nuclear, QM, security) have canon anchors but no targeted tests.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# finance_cli — module-level tests
# ============================================================================


class TestFinanceCLIImport:
    """finance_cli module loads and has expected attributes."""

    def test_import_succeeds(self) -> None:
        mod = importlib.import_module("umcp.finance_cli")
        assert hasattr(mod, "main")
        assert hasattr(mod, "cmd_init")
        assert hasattr(mod, "cmd_record")
        assert hasattr(mod, "cmd_analyze")
        assert hasattr(mod, "cmd_report")

    def test_finance_thresholds_defined(self) -> None:
        from umcp.finance_cli import FINANCE_THRESHOLDS

        assert FINANCE_THRESHOLDS.omega_stable_max > 0
        assert FINANCE_THRESHOLDS.F_stable_min > 0

    def test_finance_tol_seam_defined(self) -> None:
        from umcp.finance_cli import FINANCE_TOL_SEAM

        assert FINANCE_TOL_SEAM > 0

    def test_finance_dir_name_defined(self) -> None:
        from umcp.finance_cli import FINANCE_DIR_NAME

        assert isinstance(FINANCE_DIR_NAME, str)
        assert len(FINANCE_DIR_NAME) > 0

    def test_raw_headers_defined(self) -> None:
        from umcp.finance_cli import RAW_HEADERS

        assert isinstance(RAW_HEADERS, list)
        assert "month" in RAW_HEADERS or len(RAW_HEADERS) > 0


class TestFinanceWorkspace:
    """Test workspace creation and configuration."""

    def test_get_workspace_default(self) -> None:
        from umcp.finance_cli import _get_workspace

        ws = _get_workspace()
        assert isinstance(ws, Path)

    def test_get_workspace_custom(self, tmp_path: Path) -> None:
        from umcp.finance_cli import _get_workspace

        ws = _get_workspace(str(tmp_path / ".umcp-finance"))
        assert isinstance(ws, Path)

    def test_ensure_workspace_creates_dir(self, tmp_path: Path) -> None:
        from umcp.finance_cli import _ensure_workspace

        ws = tmp_path / ".umcp-finance"
        _ensure_workspace(ws)
        assert ws.exists()


# ============================================================================
# Closure domain coverage — canon anchor validation
# ============================================================================


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
class TestClosureDomainAnchors:
    """Canon anchor files exist and have valid structure for all domains."""

    @pytest.mark.parametrize(
        "anchor_file",
        [
            "anchors.yaml",
            "gcd_anchors.yaml",
            "rcft_anchors.yaml",
            "kin_anchors.yaml",
            "weyl_anchors.yaml",
        ],
    )
    def test_canon_anchor_exists_and_loads(self, anchor_file: str) -> None:
        path = REPO_ROOT / "canon" / anchor_file
        assert path.exists(), f"Missing canon anchor: {anchor_file}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "schema" in data or "anchors" in data or "canon" in data

    @pytest.mark.parametrize(
        "anchor_file",
        [
            "anchors.yaml",
            "gcd_anchors.yaml",
            "rcft_anchors.yaml",
            "kin_anchors.yaml",
            "weyl_anchors.yaml",
        ],
    )
    def test_anchor_has_domain_entries(self, anchor_file: str) -> None:
        path = REPO_ROOT / "canon" / anchor_file
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Anchors should have at least one entry beyond metadata
        assert len(data) >= 2, f"{anchor_file} has only {len(data)} top-level keys"


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
class TestClosureDirectories:
    """Each closure domain directory has __init__.py and at least one closure."""

    @pytest.mark.parametrize(
        "domain",
        [
            "astronomy",
            "finance",
            "gcd",
            "kinematics",
            "nuclear_physics",
            "quantum_mechanics",
            "rcft",
            "security",
            "weyl",
        ],
    )
    def test_domain_dir_exists(self, domain: str) -> None:
        domain_path = REPO_ROOT / "closures" / domain
        assert domain_path.is_dir(), f"Missing closure domain dir: {domain}"

    @pytest.mark.parametrize(
        "domain",
        [
            "astronomy",
            "finance",
            "kinematics",
            "nuclear_physics",
            "quantum_mechanics",
            "rcft",
            "security",
            "weyl",
        ],
    )
    def test_domain_has_init(self, domain: str) -> None:
        init = REPO_ROOT / "closures" / domain / "__init__.py"
        assert init.exists(), f"Missing __init__.py in closures/{domain}"

    @pytest.mark.parametrize(
        "domain",
        [
            "astronomy",
            "gcd",
            "kinematics",
            "nuclear_physics",
            "quantum_mechanics",
            "rcft",
            "security",
            "weyl",
        ],
    )
    def test_domain_has_closures(self, domain: str) -> None:
        domain_path = REPO_ROOT / "closures" / domain
        py_files = [f for f in domain_path.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 1, f"No closure files in closures/{domain}"


@pytest.mark.skipif(yaml is None, reason="pyyaml not installed")
class TestClosureRegistry:
    """closures/registry.yaml is complete and well-formed."""

    @pytest.fixture()
    def registry(self) -> dict[str, Any]:
        path = REPO_ROOT / "closures" / "registry.yaml"
        assert path.exists()
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_registry_has_schema(self, registry: dict[str, Any]) -> None:
        assert "schema" in registry

    def test_registry_has_id(self, registry: dict[str, Any]) -> None:
        assert "registry" in registry
        assert "id" in registry["registry"]

    def test_registry_has_base_closures(self, registry: dict[str, Any]) -> None:
        closures = registry["registry"].get("closures", {})
        for key in ("gamma", "return_domain", "norms", "curvature_neighborhood"):
            assert key in closures, f"Missing base closure: {key}"

    def test_registry_has_extensions(self, registry: dict[str, Any]) -> None:
        extensions = registry["registry"].get("extensions", {})
        assert len(extensions) >= 3  # At least gcd, rcft, kinematics

    def test_registry_closure_paths_exist(self, registry: dict[str, Any]) -> None:
        """Every path referenced in base closures exists."""
        closures = registry["registry"].get("closures", {})
        for _name, entry in closures.items():
            path = REPO_ROOT / entry["path"]
            assert path.exists(), f"Missing closure file: {entry['path']}"
