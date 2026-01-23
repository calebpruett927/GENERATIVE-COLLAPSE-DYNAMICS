from src.umcp import api_umcp

def test_validate_api_key():
    # Should return True for expected key
    assert api_umcp.validate_api_key("expected_key") is True
    # Should return False for wrong key
    assert api_umcp.validate_api_key("wrong_key") is False

def test_verify_api_key():
    # Should return True for expected key
    assert api_umcp.verify_api_key("expected_key") is True
    # Should return False for wrong key
    assert api_umcp.verify_api_key("wrong_key") is False

def test_get_repo_root():
    # Should return a Path object
    root = api_umcp.get_repo_root()
    assert hasattr(root, 'exists')
    assert root.is_dir() or root.is_file()

def test_classify_regime_positive():
    # Should return "regime-positive" for positive values
    assert api_umcp.classify_regime(1, 1, 1, 1) == "regime-positive"

def test_classify_regime_negative():
    # Should return "regime-negative" for negative values
    assert api_umcp.classify_regime(-1, 1, 1, 1) == "regime-negative"
    assert api_umcp.classify_regime(1, -1, 1, 1) == "regime-negative"
    assert api_umcp.classify_regime(1, 1, -1, 1) == "regime-negative"
    assert api_umcp.classify_regime(1, 1, 1, -1) == "regime-negative"

def test_classify_regime_unknown():
    # Should return "regime-unknown" for zero values
    assert api_umcp.classify_regime(0, 0, 0, 0) == "regime-unknown"

def test_get_current_time():
    # Should return a string with time
    result = api_umcp.get_current_time()
    assert isinstance(result, str)
    assert "T" in result or ":" in result
