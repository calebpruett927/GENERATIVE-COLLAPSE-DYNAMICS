import pytest
from src.umcp import minimal_cli


def test_main_runs():
    # Just ensure main() runs without error
    try:
        minimal_cli.main()
    except Exception as e:
        pytest.fail(f"minimal_cli.main() raised {e}")
