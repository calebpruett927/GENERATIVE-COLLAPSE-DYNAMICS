"""
test_215_casepack_roundtrip.py — End-to-end casepack validation for ALL 14 casepacks.

Previously only 6 of 14 casepacks had any test coverage. This module
validates every casepack through the full UMCP pipeline: directory
structure, manifest schema, raw data presence, expected output loading,
Tier-1 identity checking on expected invariants, and ``umcp validate``
round-trip.

Derivation chain:  Axiom-0 → Tier-0 protocol → casepack validation
    Every casepack is a collapse-return artefact. This test verifies
    that the return domain D_θ is reproducible: the expected outputs
    satisfy the structural identities F + ω = 1, IC ≤ F, IC ≈ exp(κ).

Sections
--------
§1  Structure:     Directory layout and manifest schema
§2  Raw data:      raw_measurements.csv loadable with correct columns
§3  Invariants:    Tier-1 identity checks on expected/invariants.json
§4  Regime:        Regime labels consistent with frozen thresholds
§5  CLI validate:  ``umcp validate casepacks/<name>`` returns CONFORMANT
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CASEPACKS_DIR = REPO_ROOT / "casepacks"

# ── Tolerances (Tier-1 checking) ──────────────────────────────────
TOL_DUALITY = 1e-6  # F + ω = 1
TOL_LOG_IC = 1e-3  # IC ≈ exp(κ)  (security_validation has precision to ~3 decimal)
TOL_BOUND = 0.002  # IC ≤ F  (ε-clamp artefacts: when F=0, IC=exp(κ_clamped)≈0.001)

# ── All 14 casepack directories ──────────────────────────────────
ALL_CASEPACKS: list[str] = sorted(d.name for d in CASEPACKS_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))

# Casepacks whose invariants.json uses the standard "rows" format
# (some casepacks have alternate structures or different file names)
_INVARIANT_FILES: dict[str, str] = {
    "confinement_T3": "expected/kernel_invariants.json",
}

# Casepacks with non-standard raw data paths
_RAW_DATA_FILES: dict[str, str] = {
    "weyl_des_y3": "data/hJ_measurements.csv",
    "retro_coherent_phys04": "data/raw_measurements.csv",
    "confinement_T3": "raw_particles.csv",
    "UMCP-REF-E2E-0001": "data/raw.csv",
}


def _get_manifest(name: str) -> dict:
    """Load and return parsed manifest.json for a casepack."""
    manifest_path = CASEPACKS_DIR / name / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def _get_invariant_rows(name: str) -> list[dict]:
    """Load expected invariant rows from the casepack's expected/ directory."""
    rel = _INVARIANT_FILES.get(name, "expected/invariants.json")
    inv_path = CASEPACKS_DIR / name / rel
    if not inv_path.exists():
        pytest.skip(f"No invariants file at {inv_path}")
    with open(inv_path) as f:
        data = json.load(f)
    return data.get("rows", [])


def _get_raw_data_path(name: str) -> Path:
    """Return the path to the raw measurements file."""
    rel = _RAW_DATA_FILES.get(name, "raw_measurements.csv")
    return CASEPACKS_DIR / name / rel


# ═══════════════════════════════════════════════════════════════════
# §1  STRUCTURE: Directory layout and manifest
# ═══════════════════════════════════════════════════════════════════


class TestCasepackStructure:
    """Every casepack must have a manifest.json and expected/ directory."""

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_directory_exists(self, name: str) -> None:
        assert (CASEPACKS_DIR / name).is_dir(), f"Casepack directory missing: {name}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_manifest_exists(self, name: str) -> None:
        assert (CASEPACKS_DIR / name / "manifest.json").exists(), f"manifest.json missing in {name}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_manifest_valid_json(self, name: str) -> None:
        manifest = _get_manifest(name)
        assert "casepack" in manifest, f"No 'casepack' key in manifest of {name}"
        assert "refs" in manifest, f"No 'refs' key in manifest of {name}"
        assert "artifacts" in manifest, f"No 'artifacts' key in manifest of {name}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_manifest_has_contract_ref(self, name: str) -> None:
        manifest = _get_manifest(name)
        contract_ref = manifest["refs"].get("contract", {})
        assert "id" in contract_ref, f"No contract id in manifest of {name}"
        assert "path" in contract_ref, f"No contract path in manifest of {name}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_expected_directory_exists(self, name: str) -> None:
        expected_dir = CASEPACKS_DIR / name / "expected"
        assert expected_dir.is_dir(), f"expected/ directory missing in {name}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_expected_has_output_files(self, name: str) -> None:
        expected_dir = CASEPACKS_DIR / name / "expected"
        files = list(expected_dir.iterdir())
        assert len(files) > 0, f"expected/ is empty in {name}"


# ═══════════════════════════════════════════════════════════════════
# §2  RAW DATA: CSV presence and basic structure
# ═══════════════════════════════════════════════════════════════════


class TestCasepackRawData:
    """Raw measurement data must exist and be loadable."""

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_raw_data_exists(self, name: str) -> None:
        raw_path = _get_raw_data_path(name)
        assert raw_path.exists(), f"Raw data missing: {raw_path}"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_raw_data_not_empty(self, name: str) -> None:
        raw_path = _get_raw_data_path(name)
        content = raw_path.read_text()
        lines = [ln for ln in content.strip().splitlines() if ln.strip()]
        assert len(lines) >= 2, f"Raw data has <2 lines in {name} (need header + data)"

    @pytest.mark.parametrize("name", ALL_CASEPACKS)
    def test_raw_data_has_header(self, name: str) -> None:
        raw_path = _get_raw_data_path(name)
        with open(raw_path) as f:
            header = f.readline().strip()
        assert "," in header, f"Header doesn't look like CSV in {name}: {header[:80]}"


# ═══════════════════════════════════════════════════════════════════
# §3  INVARIANTS: Tier-1 identity checks on expected outputs
# ═══════════════════════════════════════════════════════════════════

# Casepacks that have standard invariants.json with rows
_CASEPACKS_WITH_INVARIANTS = [
    name
    for name in ALL_CASEPACKS
    if (CASEPACKS_DIR / name / _INVARIANT_FILES.get(name, "expected/invariants.json")).exists()
]


class TestCasepackInvariants:
    """Tier-1 structural identities must hold in all expected invariant rows.

    For every row: F + ω = 1, IC ≤ F, and IC ≈ exp(κ).
    These are the non-negotiable identities of collapse.
    """

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_duality_identity(self, name: str) -> None:
        """F + ω = 1 for every expected row."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in invariants for {name}")
        for i, row in enumerate(rows):
            f_val = row.get("F")
            omega = row.get("omega")
            if f_val is None or omega is None:
                continue
            residual = abs(f_val + omega - 1.0)
            assert residual < TOL_DUALITY, (
                f"Duality violation in {name} row {i}: F={f_val} + ω={omega} = {f_val + omega}, residual={residual}"
            )

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_integrity_bound(self, name: str) -> None:
        """IC ≤ F for every expected row."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in invariants for {name}")
        for i, row in enumerate(rows):
            f_val = row.get("F")
            ic_val = row.get("IC")
            if f_val is None or ic_val is None:
                continue
            assert ic_val <= f_val + TOL_BOUND, f"Integrity bound violation in {name} row {i}: IC={ic_val} > F={f_val}"

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_log_integrity_relation(self, name: str) -> None:
        """IC ≈ exp(κ) for every expected row."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in invariants for {name}")
        for i, row in enumerate(rows):
            kappa = row.get("kappa")
            ic_val = row.get("IC")
            if kappa is None or ic_val is None:
                continue
            expected_ic = math.exp(kappa)
            residual = abs(ic_val - expected_ic)
            assert residual < TOL_LOG_IC, (
                f"Log-integrity violation in {name} row {i}: IC={ic_val}, exp(κ)={expected_ic}, residual={residual}"
            )

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_invariant_ranges(self, name: str) -> None:
        """F, ω, IC all in [0,1]; S ≥ 0; κ ≤ 0."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in invariants for {name}")
        for i, row in enumerate(rows):
            f_val = row.get("F")
            omega = row.get("omega")
            ic_val = row.get("IC")
            s_val = row.get("S")
            kappa = row.get("kappa")
            if f_val is not None:
                assert 0.0 <= f_val <= 1.0 + TOL_BOUND, f"F out of range in {name} row {i}: F={f_val}"
            if omega is not None:
                assert 0.0 <= omega <= 1.0 + TOL_BOUND, f"ω out of range in {name} row {i}: ω={omega}"
            if ic_val is not None:
                assert 0.0 <= ic_val <= 1.0 + TOL_BOUND, f"IC out of range in {name} row {i}: IC={ic_val}"
            if s_val is not None:
                assert s_val >= -TOL_BOUND, f"S negative in {name} row {i}: S={s_val}"
            if kappa is not None:
                assert kappa <= TOL_BOUND, f"κ positive in {name} row {i}: κ={kappa}"

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_contract_id_matches_manifest(self, name: str) -> None:
        """The invariants file's contract_id must match the manifest's."""
        manifest = _get_manifest(name)
        manifest_contract = manifest["refs"]["contract"]["id"]

        rel = _INVARIANT_FILES.get(name, "expected/invariants.json")
        inv_path = CASEPACKS_DIR / name / rel
        with open(inv_path) as f:
            inv_data = json.load(f)

        file_contract = inv_data.get("contract_id")
        if file_contract is None:
            pytest.skip(f"No contract_id in invariants file of {name}")

        assert file_contract == manifest_contract, (
            f"Contract mismatch in {name}: manifest={manifest_contract}, invariants={file_contract}"
        )


# ═══════════════════════════════════════════════════════════════════
# §4  REGIME: Labels consistent with frozen thresholds
# ═══════════════════════════════════════════════════════════════════


class TestCasepackRegime:
    """Regime labels in expected outputs must be consistent with
    the frozen gate thresholds (Stable/Watch/Collapse).

    Stable:   ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
    Watch:    0.038 ≤ ω < 0.30 (or Stable gates not all satisfied)
    Collapse: ω ≥ 0.30
    """

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_regime_labels_valid(self, name: str) -> None:
        """Every regime label must be one of the three canonical labels."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in {name}")
        valid_labels = {"Stable", "Watch", "Collapse"}
        for i, row in enumerate(rows):
            regime = row.get("regime")
            if regime is None:
                continue
            label = regime if isinstance(regime, str) else regime.get("label")
            assert label in valid_labels, f"Invalid regime '{label}' in {name} row {i}"

    @pytest.mark.parametrize("name", _CASEPACKS_WITH_INVARIANTS)
    def test_collapse_regime_consistency(self, name: str) -> None:
        """If regime is Collapse, ω should be ≥ 0.30."""
        rows = _get_invariant_rows(name)
        if not rows:
            pytest.skip(f"No rows in {name}")
        for i, row in enumerate(rows):
            regime = row.get("regime")
            omega = row.get("omega")
            if regime is None or omega is None:
                continue
            label = regime if isinstance(regime, str) else regime.get("label")
            if label == "Collapse":
                assert omega >= 0.29, (  # small tolerance
                    f"Collapse regime but ω={omega} < 0.30 in {name} row {i}"
                )


# ═══════════════════════════════════════════════════════════════════
# §5  CLI VALIDATE: ``umcp validate`` round-trip
# ═══════════════════════════════════════════════════════════════════

# Some casepacks may not pass strict due to missing optional fields;
# baseline validation should always succeed for well-formed casepacks.
_VALIDATE_CASEPACKS = ALL_CASEPACKS


class TestCasepackCLIValidate:
    """Run ``umcp validate casepacks/<name>`` and verify CONFORMANT.

    This is the definitive round-trip: raw data + contract + closures
    → validation pipeline → CONFORMANT verdict.
    """

    @pytest.mark.parametrize("name", _VALIDATE_CASEPACKS)
    def test_umcp_validate_conformant(self, name: str) -> None:
        """Baseline validation must return CONFORMANT for every casepack."""
        casepack_path = CASEPACKS_DIR / name
        result = subprocess.run(
            ["umcp", "validate", str(casepack_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # The validation output contains a summary line and JSON.
        # A return code of 0 means CONFORMANT.
        assert result.returncode == 0, (
            f"umcp validate failed for {name} (exit {result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Additionally check the output contains CONFORMANT
        combined = result.stdout + result.stderr
        assert "CONFORMANT" in combined, f"No CONFORMANT verdict in output for {name}:\n{combined[:500]}"


# ═══════════════════════════════════════════════════════════════════
# §6  CROSS-CASEPACK: Structural consistency across all casepacks
# ═══════════════════════════════════════════════════════════════════


class TestCrossPackConsistency:
    """Structural consistency properties that should hold across
    the entire casepack collection.
    """

    def test_all_casepacks_discovered(self) -> None:
        """Sanity check: we should discover at least 13 casepacks."""
        assert len(ALL_CASEPACKS) >= 13, f"Only {len(ALL_CASEPACKS)} casepacks found, expected ≥ 13"

    def test_contract_ids_reference_existing_files(self) -> None:
        """Every contract referenced by a manifest must exist on disk."""
        for name in ALL_CASEPACKS:
            manifest = _get_manifest(name)
            contract_path = manifest["refs"]["contract"]["path"]
            full_path = REPO_ROOT / contract_path
            assert full_path.exists(), f"Contract file missing for {name}: {contract_path}"

    def test_no_duplicate_casepack_ids(self) -> None:
        """No two casepacks should have the same id."""
        ids = []
        for name in ALL_CASEPACKS:
            manifest = _get_manifest(name)
            cp_id = manifest["casepack"]["id"]
            ids.append(cp_id)
        assert len(ids) == len(set(ids)), f"Duplicate casepack IDs found: {[x for x in ids if ids.count(x) > 1]}"

    def test_multiple_contracts_represented(self) -> None:
        """The casepack collection should exercise multiple contracts."""
        contracts = set()
        for name in ALL_CASEPACKS:
            manifest = _get_manifest(name)
            contracts.add(manifest["refs"]["contract"]["id"])
        assert len(contracts) >= 5, f"Only {len(contracts)} distinct contracts across casepacks: {contracts}"

    def test_invariant_files_have_rows(self) -> None:
        """Every invariants file with standard 'rows' format should have ≥1 row.

        confinement_T3 uses a non-standard structure (contract/fundamental/composite)
        and legitimately has zero rows — it is excluded.
        """
        _NON_ROW_FORMAT = {"confinement_T3"}  # alternate schema without 'rows'
        empty = []
        for name in _CASEPACKS_WITH_INVARIANTS:
            if name in _NON_ROW_FORMAT:
                continue
            rows = _get_invariant_rows(name)
            if len(rows) == 0:
                empty.append(name)
        assert not empty, f"Casepacks with empty invariant rows: {empty}"
