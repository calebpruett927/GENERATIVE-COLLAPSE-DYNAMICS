from __future__ import annotations

import subprocess
import sys


def test_main_entry_callable():
    """Test that __main__.main is callable."""
    from umcp.__main__ import main

    assert callable(main)


def test_main_module_executable():
    """Test that umcp can be executed as a module."""
    result = subprocess.run([sys.executable, "-m", "umcp", "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_main_module_version():
    """Test that umcp --version works."""
    result = subprocess.run([sys.executable, "-m", "umcp", "--version"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert len(result.stdout.strip()) > 0


def test_main_module_no_args():
    """Test that umcp with no args shows help."""
    result = subprocess.run([sys.executable, "-m", "umcp"], capture_output=True, text=True, timeout=10)
    # Should show help or error, but not crash
    assert result.returncode in (0, 1, 2)
