import json
import sys

from src.umcp.file_refs import UMCPFiles, get_umcp_files


def test_umcpfiles_init():
    files = UMCPFiles()
    assert isinstance(files, UMCPFiles)
    assert hasattr(files, "root")


def test_verify_all_exist(tmp_path):
    files = UMCPFiles(root_path=tmp_path)
    result = files.verify_all_exist()
    assert isinstance(result, dict)
    assert "manifest.yaml" in result


def test_get_missing_files(tmp_path):
    files = UMCPFiles(root_path=tmp_path)
    missing = files.get_missing_files()
    assert isinstance(missing, list)


def test_load_yaml_minimal(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("key: value\n# comment\nother: 123\n")
    files = UMCPFiles(root_path=tmp_path)
    result = files.load_yaml(yaml_file)
    assert result["key"] == "value"
    assert result["other"] == 123 or result["other"] == "123"


def test_load_json(tmp_path):
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps({"a": 1}))
    files = UMCPFiles(root_path=tmp_path)
    result = files.load_json(json_file)
    assert result["a"] == 1


def test_load_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b\n1,2\n")
    files = UMCPFiles(root_path=tmp_path)
    result = files.load_csv(csv_file)
    assert result[0]["a"] == "1"
    assert result[0]["b"] == "2"


def test_load_text(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello world")
    files = UMCPFiles(root_path=tmp_path)
    result = files.load_text(txt_file)
    assert result == "hello world"


def test_get_umcp_files():
    files = get_umcp_files()
    assert isinstance(files, UMCPFiles)
