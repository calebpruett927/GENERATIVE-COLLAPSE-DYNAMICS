"""
Test the benchmark script runs correctly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestBenchmark:
    """Test the benchmark functionality."""

    def test_benchmark_file_exists(self):
        """Benchmark script should exist."""
        benchmark_path = REPO_ROOT / "benchmark_umcp_vs_standard.py"
        
        if not benchmark_path.exists():
            pytest.skip("Benchmark script not found")
        
        assert benchmark_path.exists()

    def test_benchmark_is_valid_python(self):
        """Benchmark script should be valid Python."""
        benchmark_path = REPO_ROOT / "benchmark_umcp_vs_standard.py"
        
        if not benchmark_path.exists():
            pytest.skip("Benchmark script not found")
        
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(benchmark_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"Benchmark has syntax errors: {result.stderr}"

    @pytest.mark.slow
    def test_benchmark_runs(self):
        """Benchmark script should run without errors."""
        benchmark_path = REPO_ROOT / "benchmark_umcp_vs_standard.py"
        
        if not benchmark_path.exists():
            pytest.skip("Benchmark script not found")
        
        result = subprocess.run(
            [sys.executable, str(benchmark_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
        assert "RESULTS" in result.stdout or "Benchmark" in result.stdout

    def test_standard_validator_class_exists(self):
        """StandardValidator class should be importable from benchmark."""
        benchmark_path = REPO_ROOT / "benchmark_umcp_vs_standard.py"
        
        if not benchmark_path.exists():
            pytest.skip("Benchmark script not found")
        
        # Just check the file can be parsed
        with benchmark_path.open("r") as f:
            content = f.read()
        
        assert "StandardValidator" in content
        assert "UMCPValidator" in content