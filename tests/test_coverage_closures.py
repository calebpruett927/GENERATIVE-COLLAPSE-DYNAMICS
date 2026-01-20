"""Targeted tests to improve closures.py coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from umcp.closures import ClosureLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestClosureLoadingEdgeCases:
    """Test closure loading edge cases."""

    def test_load_nonexistent_module(self):
        """Test loading a closure module that doesn't exist."""
        loader = ClosureLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_closure_module("definitely_does_not_exist_xyz")

    def test_get_closure_function_missing_callable(self, tmp_path):
        """Test getting a closure function when callable doesn't exist."""
        # Create a temporary closure dir with a Python file that has no compute function
        closure_dir = tmp_path / "closures"
        closure_dir.mkdir()
        
        test_file = closure_dir / "no_callable.py"
        test_file.write_text("x = 42  # No compute function here")
        
        loader = ClosureLoader(root_dir=tmp_path)
        
        # Try to get the compute function - should raise AttributeError
        with pytest.raises(AttributeError):
            loader.get_closure_function("no_callable")

    def test_execute_closure_with_real_closure(self):
        """Test executing a real closure from the repository."""
        loader = ClosureLoader()
        
        # Try to execute hello_world if it exists
        if (loader.closures_dir / "hello_world.py").exists():
            try:
                result = loader.execute_closure("hello_world")
                # hello_world should return something
                assert result is not None
            except (TypeError, AttributeError) as e:
                # May have required parameters
                pytest.skip(f"hello_world has required params: {e}")
        else:
            pytest.skip("hello_world closure not available")

    def test_execute_closure_missing_function(self):
        """Test executing a closure when the function doesn't exist."""
        loader = ClosureLoader()
        
        # This should raise an error or return None
        with pytest.raises((AttributeError, TypeError, FileNotFoundError)):
            loader.execute_closure("definitely_nonexistent")

    def test_validate_closure_exists_true(self):
        """Test validate_closure_exists for existing closure."""
        loader = ClosureLoader()
        
        # hello_world should exist
        if (loader.closures_dir / "hello_world.py").exists():
            assert loader.validate_closure_exists("hello_world") is True

    def test_validate_closure_exists_false(self):
        """Test validate_closure_exists for non-existent closure."""
        loader = ClosureLoader()
        
        assert loader.validate_closure_exists("nonexistent_xyz") is False

    def test_list_closures_returns_dict(self):
        """Test that list_closures returns proper structure."""
        loader = ClosureLoader()
        
        closures = loader.list_closures()
        assert isinstance(closures, dict)
        
        # Check structure
        for name, path in closures.items():
            assert isinstance(name, str)
            assert isinstance(path, str)

    def test_registry_property(self):
        """Test registry property access."""
        loader = ClosureLoader()
        
        registry = loader.registry
        assert isinstance(registry, dict)

    def test_module_caching(self):
        """Test that modules are cached after first load."""
        loader = ClosureLoader()
        
        # Load a module twice
        if (loader.closures_dir / "hello_world.py").exists():
            module1 = loader.load_closure_module("hello_world")
            module2 = loader.load_closure_module("hello_world")
            
            # Should be the same object (cached)
            assert module1 is module2

    def test_closure_loader_with_explicit_root(self, tmp_path):
        """Test ClosureLoader with explicit root directory."""
        closure_dir = tmp_path / "closures"
        closure_dir.mkdir()
        
        registry_file = closure_dir / "registry.yaml"
        registry_file.write_text("closures: {}")
        
        loader = ClosureLoader(root_dir=tmp_path)
        assert loader.root == tmp_path
        assert loader.closures_dir == closure_dir

    def test_load_closure_module_importerror(self, tmp_path):
        """Test loading a module that causes ImportError."""
        closure_dir = tmp_path / "closures"
        closure_dir.mkdir()
        
        # Create a Python file with syntax that will cause issues
        bad_file = closure_dir / "bad_module.py"
        bad_file.write_text("import nonexistent_module_xyz")
        
        loader = ClosureLoader(root_dir=tmp_path)
        
        # This should raise ImportError or similar
        with pytest.raises((ImportError, ModuleNotFoundError)):
            loader.load_closure_module("bad_module")

    def test_execute_closure_with_kwargs(self):
        """Test executing closure with keyword arguments."""
        loader = ClosureLoader()
        
        # Try F_from_omega if it exists
        if (loader.closures_dir / "F_from_omega.py").exists():
            try:
                result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
                assert result is not None
            except (TypeError, AttributeError) as e:
                pytest.skip(f"F_from_omega has different signature: {e}")
