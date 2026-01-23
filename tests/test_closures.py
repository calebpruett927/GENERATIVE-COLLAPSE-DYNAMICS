import sys

import pytest

from src.umcp.closures import ClosureLoader, get_closure_loader


def test_closure_loader_init():
    loader = ClosureLoader()
    assert isinstance(loader, ClosureLoader)
    assert hasattr(loader, "root")
    assert hasattr(loader, "closures_dir")


def test_registry_property_minimal_yaml(tmp_path, monkeypatch):
    # Simulate missing PyYAML
    monkeypatch.setitem(sys.modules, "yaml", None)
    closures_dir = tmp_path / "closures"
    closures_dir.mkdir()
    registry_path = closures_dir / "registry.yaml"
    registry_path.write_text("key1: value1\n# comment\nkey2: value2\n")
    loader = ClosureLoader(root_dir=tmp_path)
    loader.closures_dir = closures_dir
    loader.registry_path = registry_path
    reg = loader.registry
    assert reg["key1"] == "value1"
    assert reg["key2"] == "value2"


def test_list_closures_empty_registry(monkeypatch, tmp_path):
    closures_dir = tmp_path / "closures"
    closures_dir.mkdir()
    registry_path = closures_dir / "registry.yaml"
    registry_path.write_text("registry: { closures: {} }\n")
    loader = ClosureLoader(root_dir=tmp_path)
    loader.closures_dir = closures_dir
    loader.registry_path = registry_path
    monkeypatch.setattr(loader, "_registry", {"registry": {"closures": {}}})
    result = loader.list_closures()
    assert result == {}


def test_load_closure_module_not_found(tmp_path):
    loader = ClosureLoader(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_closure_module("nonexistent")


def test_get_closure_function_no_compute(tmp_path):
    closures_dir = tmp_path / "closures"
    closures_dir.mkdir()
    py_file = closures_dir / "testmod.py"
    py_file.write_text("def not_compute(): pass\n")
    loader = ClosureLoader(root_dir=tmp_path)
    loader.closures_dir = closures_dir
    with pytest.raises(AttributeError):
        loader.get_closure_function("testmod")


def test_validate_closure_exists_false():
    loader = ClosureLoader()
    assert loader.validate_closure_exists("nonexistent") is False


def test_get_closure_loader():
    loader = get_closure_loader()
    assert isinstance(loader, ClosureLoader)
