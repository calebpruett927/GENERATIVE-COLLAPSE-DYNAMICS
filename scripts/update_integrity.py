#!/usr/bin/env python3
import hashlib
from pathlib import Path

# Path to the file and integrity file
target = Path("api_umcp.py")
integrity_file = Path("integrity/sha256.txt")


def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    checksum = sha256sum(target)
    integrity_file.write_text(f"{checksum}  {target.name}\n")
    print(f"Updated {integrity_file} with SHA256 for {target}")


if __name__ == "__main__":
    main()
