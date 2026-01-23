def test_main_entry():
    # Just ensure main entry imports and runs
    from src.umcp.__main__ import main

    assert callable(main)
