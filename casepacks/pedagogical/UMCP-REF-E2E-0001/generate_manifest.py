#!/usr/bin/env python3
"""Generate manifest with SHA256 hashes for all CasePack files"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CASEPACK = Path(__file__).resolve().parent
MANIFEST_DIR = CASEPACK / "manifest"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CASEPACK,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_python_version() -> str:
    """Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all files
    files = []
    exclude_dirs = {"__pycache__", ".git", "manifest"}

    for path in CASEPACK.rglob("*"):
        if path.is_file():
            # Skip manifest files themselves and cache
            if any(exc in path.parts for exc in exclude_dirs):
                continue
            if path.name in ["generate_manifest.py", "manifest.json", "sha256sums.txt"]:
                continue

            rel_path = path.relative_to(CASEPACK)
            sha256 = compute_sha256(path)
            size = path.stat().st_size

            files.append({"path": str(rel_path), "sha256": sha256, "size_bytes": size})

    # Sort by path
    files.sort(key=lambda x: x["path"])

    # Get environment info
    git_commit = get_git_commit()
    python_version = get_python_version()

    # Create manifest JSON
    manifest = {
        "schema": "schemas/manifest.schema.json",
        "manifest": {
            "id": "UMCP-REF-E2E-0001.manifest",
            "version": "1.0.0",
            "created_utc": datetime.now(UTC).isoformat(),
            "casepack_id": "UMCP-REF-E2E-0001",
            "environment": {
                "git_commit": git_commit,
                "python_version": python_version,
                "umcp_tool": "api_umcp.py",
                "os": "Linux",
            },
            "files": files,
            "file_count": len(files),
        },
    }

    # Compute manifest root hash (hash of all file hashes concatenated)
    manifest_root_sha256 = hashlib.sha256()
    for f in files:
        manifest_root_sha256.update(f["sha256"].encode())
    manifest["manifest"]["root_sha256"] = manifest_root_sha256.hexdigest()

    # Write manifest.json
    manifest_json = MANIFEST_DIR / "manifest.json"
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest: {manifest_json}")
    print(f"Files tracked: {len(files)}")
    print(f"Manifest root SHA256: {manifest['manifest']['root_sha256']}")

    # Write sha256sums.txt in standard format
    sha256sums_txt = MANIFEST_DIR / "sha256sums.txt"
    with open(sha256sums_txt, "w") as f:
        for file_info in files:
            f.write(f"{file_info['sha256']}  {file_info['path']}\n")

    print(f"Wrote checksums: {sha256sums_txt}")

    # Update receipt with manifest root hash
    receipt_path = CASEPACK / "receipts" / "ss1m.json"
    if receipt_path.exists():
        with open(receipt_path) as f:
            receipt = json.load(f)

        receipt["receipt"]["manifest"]["root_sha256"] = manifest["manifest"]["root_sha256"]

        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)

        print(f"Updated receipt with manifest root hash: {receipt_path}")


if __name__ == "__main__":
    main()
