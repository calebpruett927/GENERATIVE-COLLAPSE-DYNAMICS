"""Minimal CLI entry point for UMCP.

Provides a lightweight command-line interface that delegates
to the main CLI for full functionality. This module exists to
support minimal installations and quick health checks.
"""

from __future__ import annotations

import logging
import sys


def main(args: list[str] | None = None) -> int:
    """Minimal CLI entry point.

    When called with no arguments (args=None), defaults to showing version.
    This makes the function safe to call in tests without side effects.

    Args:
        args: Command line arguments. If None, defaults to empty list (shows version).
              Use sys.argv[1:] explicitly if you want CLI behavior.

    Returns:
        0 on success, non-zero on failure.
    """
    logger = logging.getLogger("umcp.minimal_cli")
    logger.debug("Minimal CLI main() called")

    # Default to empty args (show version) when called programmatically
    if args is None:
        args = []

    logger.debug(f"args: {args}")

    # For minimal mode with no args or version flag, print version and exit
    if len(args) == 0 or args[0] in ("--version", "-V"):
        from umcp import __version__

        print(f"umcp {__version__}")
        return 0

    # For other commands, delegate to full CLI
    try:
        from umcp.cli import main as cli_main

        # Pass args to cli_main if it accepts them, otherwise set sys.argv
        old_argv = sys.argv
        try:
            sys.argv = ["umcp", *args]
            return cli_main()
        finally:
            sys.argv = old_argv
    except ImportError:
        print("Full CLI not available. Install with: pip install umcp[dev]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
