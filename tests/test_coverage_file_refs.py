"""Targeted tests to improve file_refs.py coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from umcp.file_refs import UMCPFiles

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestUMCPFilesLoading:
    """Test UMCPFiles loading methods."""

    def test_load_manifest_missing(self, tmp_path):
        """Test loading manifest when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_manifest()

    def test_load_contract_missing(self, tmp_path):
        """Test loading contract when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_contract()

    def test_load_observables_missing(self, tmp_path):
        """Test loading observables when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_observables()

    def test_load_embedding_missing(self, tmp_path):
        """Test loading embedding when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_embedding()

    def test_load_weights_missing(self, tmp_path):
        """Test loading weights when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_weights()

    def test_load_trace_missing(self, tmp_path):
        """Test loading trace when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_trace()

    def test_load_invariants_missing(self, tmp_path):
        """Test loading invariants when file doesn't exist."""
        files = UMCPFiles(root_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            files.load_invariants()

    def test_load_manifest_real_file(self):
        """Test loading actual manifest from repo."""
        files = UMCPFiles()

        if files.manifest_yaml.exists():
            manifest = files.load_manifest()
            assert manifest is not None
            assert isinstance(manifest, dict)
        else:
            pytest.skip("manifest.yaml not found")

    def test_load_contract_real_file(self):
        """Test loading actual contract from repo."""
        files = UMCPFiles()

        if files.contract_yaml.exists():
            contract = files.load_contract()
            assert contract is not None
            assert isinstance(contract, dict)
        else:
            pytest.skip("contract.yaml not found")

    def test_load_observables_real_file(self):
        """Test loading actual observables from repo."""
        files = UMCPFiles()

        if files.observables_yaml.exists():
            observables = files.load_observables()
            assert observables is not None
            assert isinstance(observables, dict)
        else:
            pytest.skip("observables.yaml not found")

    def test_load_embedding_real_file(self):
        """Test loading actual embedding from repo."""
        files = UMCPFiles()

        if files.embedding_yaml.exists():
            embedding = files.load_embedding()
            assert embedding is not None
        else:
            pytest.skip("embedding.yaml not found")

    def test_load_weights_real_file(self):
        """Test loading actual weights from repo."""
        files = UMCPFiles()

        if files.weights_csv.exists():
            weights = files.load_weights()
            assert weights is not None
            assert isinstance(weights, list)
        else:
            pytest.skip("weights.csv not found")

    def test_load_trace_real_file(self):
        """Test loading actual trace from repo."""
        files = UMCPFiles()

        trace_path = files.root / "derived" / "trace.csv"
        if trace_path.exists():
            trace = files.load_trace()
            assert trace is not None
            assert isinstance(trace, list)
        else:
            pytest.skip("trace.csv not found")

    def test_load_invariants_real_file(self):
        """Test loading actual invariants from repo."""
        files = UMCPFiles()

        inv_path = files.root / "outputs" / "invariants.csv"
        if inv_path.exists():
            invariants = files.load_invariants()
            assert invariants is not None
            assert isinstance(invariants, list)
        else:
            pytest.skip("invariants.csv not found")

    def test_all_path_attributes(self):
        """Test that all expected path attributes exist."""
        files = UMCPFiles()

        assert hasattr(files, "root")
        assert hasattr(files, "manifest_yaml")
        assert hasattr(files, "contract_yaml")
        assert hasattr(files, "observables_yaml")
        assert hasattr(files, "embedding_yaml")
        assert hasattr(files, "weights_csv")

        # Verify they're Path objects
        assert isinstance(files.root, Path)
        assert isinstance(files.manifest_yaml, Path)

    def test_root_detection_from_none(self):
        """Test root path auto-detection when None is passed."""
        files = UMCPFiles(root_path=None)

        # Should find something
        assert files.root is not None
        assert isinstance(files.root, Path)
