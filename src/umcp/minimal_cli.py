from __future__ import annotations

import sys


def main():
    logger = logging.getLogger("umcp.minimal_cli")
    logger.debug("Minimal CLI main() called")
    logger.debug(f"sys.argv: {sys.argv}")
    # ... actual minimal CLI logic ...
    return 0


if __name__ == "__main__":
    sys.exit(main())
        # Only use sys.exit in CLI entry point
