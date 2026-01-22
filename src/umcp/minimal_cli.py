from __future__ import annotations
import sys

def main():
    print("[DEBUG] Minimal CLI main() called")
    print(f"[DEBUG] sys.argv: {sys.argv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
