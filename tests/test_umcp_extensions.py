from src.umcp import umcp_extensions

def test_list_extensions():
    # Should return a list
    result = umcp_extensions.list_extensions()
    assert isinstance(result, list)

def test_get_extension_info():
    # Should return a dict with 'name' and 'info'
    info = umcp_extensions.get_extension_info("test")
    assert isinstance(info, dict)
    assert "name" in info
    assert "info" in info
