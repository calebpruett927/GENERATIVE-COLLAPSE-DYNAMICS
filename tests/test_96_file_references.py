"""
Test root-level UMCP file references.
"""
from __future__ import annotations

from pathlib import Path

from umcp import UMCPFiles, get_umcp_files


class TestUMCPFileReferences:
    """Test the UMCPFiles helper class."""
    
    def test_get_umcp_files_returns_instance(self):
        """get_umcp_files should return UMCPFiles instance."""
        umcp = get_umcp_files()
        assert isinstance(umcp, UMCPFiles)
    
    def test_umcp_files_with_explicit_root(self):
        """UMCPFiles should accept explicit root path."""
        root = Path(__file__).parent.parent
        umcp = UMCPFiles(root)
        assert umcp.root == root
    
    def test_all_root_files_exist(self):
        """All root-level UMCP files should exist."""
        umcp = get_umcp_files()
        
        # Required root files
        assert umcp.manifest_yaml.exists(), "manifest.yaml should exist"
        assert umcp.contract_yaml.exists(), "contract.yaml should exist"
        assert umcp.observables_yaml.exists(), "observables.yaml should exist"
        assert umcp.embedding_yaml.exists(), "embedding.yaml should exist"
        assert umcp.return_yaml.exists(), "return.yaml should exist"
        assert umcp.closures_yaml.exists(), "closures.yaml should exist"
        assert umcp.weights_csv.exists(), "weights.csv should exist"
    
    def test_derived_files_exist(self):
        """Derived files should exist."""
        umcp = get_umcp_files()
        
        assert umcp.trace_csv.exists(), "derived/trace.csv should exist"
        assert umcp.trace_meta_yaml.exists(), "derived/trace_meta.yaml should exist"
    
    def test_output_files_exist(self):
        """Output files should exist."""
        umcp = get_umcp_files()
        
        assert umcp.invariants_csv.exists(), "outputs/invariants.csv should exist"
        assert umcp.regimes_csv.exists(), "outputs/regimes.csv should exist"
        assert umcp.welds_csv.exists(), "outputs/welds.csv should exist"
        assert umcp.report_txt.exists(), "outputs/report.txt should exist"
    
    def test_integrity_files_exist(self):
        """Integrity files should exist."""
        umcp = get_umcp_files()
        
        assert umcp.sha256_txt.exists(), "integrity/sha256.txt should exist"
        assert umcp.env_txt.exists(), "integrity/env.txt should exist"
        assert umcp.code_version_txt.exists(), "integrity/code_version.txt should exist"
    
    def test_load_manifest(self):
        """Load manifest.yaml and validate structure."""
        umcp = get_umcp_files()
        manifest = umcp.load_manifest()
        
        assert 'casepack' in manifest
        assert 'refs' in manifest
        assert manifest['casepack']['id'] == 'umcp_complete'
        assert manifest['casepack']['version'] == '1.0.0'
    
    def test_load_contract(self):
        """Load contract.yaml and validate structure."""
        umcp = get_umcp_files()
        contract = umcp.load_contract()
        
        assert 'contract' in contract
        assert contract['contract']['id'] == 'UMA.INTSTACK.v1'
        assert 'embedding' in contract['contract']
        assert 'tier_1_kernel' in contract['contract']
    
    def test_load_observables(self):
        """Load observables.yaml and validate structure."""
        umcp = get_umcp_files()
        observables = umcp.load_observables()
        
        assert 'observables' in observables
        assert 'primary' in observables['observables']
        assert 'derived' in observables['observables']
        assert len(observables['observables']['primary']) == 3
        assert len(observables['observables']['derived']) == 3
    
    def test_load_weights(self):
        """Load weights.csv and validate values."""
        umcp = get_umcp_files()
        weights = umcp.load_weights()
        
        assert len(weights) == 1
        w_row = weights[0]
        w_values = [float(v) for v in w_row.values()]
        
        # Weights should sum to 1
        assert abs(sum(w_values) - 1.0) < 1e-9
        
        # All weights should be non-negative
        assert all(w >= 0 for w in w_values)
    
    def test_load_trace(self):
        """Load trace.csv and validate structure."""
        umcp = get_umcp_files()
        trace = umcp.load_trace()
        
        assert len(trace) > 0
        row = trace[0]
        assert 't' in row
        assert 'c_1' in row
        assert 'c_2' in row
        assert 'c_3' in row
        
        # All coordinates should be in [0, 1]
        coords = [float(row[f'c_{i}']) for i in range(1, 4)]
        assert all(0 <= c <= 1 for c in coords)
    
    def test_load_invariants(self):
        """Load invariants.csv and validate Tier-1 identities."""
        umcp = get_umcp_files()
        invariants = umcp.load_invariants()
        
        assert len(invariants) > 0
        inv = invariants[0]
        
        # Check required columns
        assert 'omega' in inv
        assert 'F' in inv
        assert 'S' in inv
        assert 'C' in inv
        assert 'kappa' in inv
        assert 'IC' in inv
        assert 'regime_label' in inv
        
        # Validate Tier-1 identity: F ≈ 1 - ω
        omega = float(inv['omega'])
        F = float(inv['F'])
        assert abs(F - (1 - omega)) < 1e-6, "F should equal 1 - omega"
    
    def test_load_regimes(self):
        """Load regimes.csv and validate structure."""
        umcp = get_umcp_files()
        regimes = umcp.load_regimes()
        
        assert len(regimes) > 0
        regime = regimes[0]
        assert 'regime_label' in regime
        assert regime['regime_label'] in ['Stable', 'Watch', 'Collapse']
    
    def test_load_integrity_files(self):
        """Load integrity files and validate content."""
        umcp = get_umcp_files()
        
        # SHA256 checksums
        sha256 = umcp.load_sha256()
        assert len(sha256) > 0
        # Check for any of the expected source files in checksums
        # The sha256.txt contains source files (.py, .yaml, .md, etc.) not data files
        assert any(x in sha256 for x in ['pyproject.toml', '.yaml', '.py', '.md'])
        
        # Environment info
        env = umcp.load_env()
        assert 'Python' in env
        
        # Code version
        version = umcp.load_code_version()
        assert len(version) > 0
    
    def test_verify_all_exist(self):
        """verify_all_exist should return status for all files."""
        umcp = get_umcp_files()
        status = umcp.verify_all_exist()
        
        assert isinstance(status, dict)
        assert len(status) > 0
        
        # All should exist
        assert all(status.values()), f"Some files missing: {[k for k, v in status.items() if not v]}"
    
    def test_get_missing_files_returns_empty(self):
        """get_missing_files should return empty list when all exist."""
        umcp = get_umcp_files()
        missing = umcp.get_missing_files()
        
        assert isinstance(missing, list)
        assert len(missing) == 0, f"Found missing files: {missing}"
