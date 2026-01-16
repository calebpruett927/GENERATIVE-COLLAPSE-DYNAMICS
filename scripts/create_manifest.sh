#!/usr/bin/env bash
# Create manifest.json for a CasePack
# Usage: ./create_manifest.sh [casepack_dir]
# Example: ./create_manifest.sh casepacks/hello_world

CASEPACK_DIR="${1:-casepacks/hello_world}"

cat > "${CASEPACK_DIR}/manifest.json" <<'EOF'
{
  "schema": "schemas/manifest.schema.json",
  "casepack": {
    "id": "hello_world",
    "version": "1.0.0",
    "title": "Hello World CasePack (Minimal Wide Ψ)",
    "description": "Minimal runnable/publication CasePack that validates schema + semantic rules: wide ψ CSV with c_k columns; invariants satisfy F≈1−ω and IC≈exp(κ); regime label matches canon thresholds.",
    "created_utc": "2026-01-14T00:00:00Z",
    "timezone": "America/Chicago",
    "authors": ["Clement Paulus"]
  },
  "refs": {
    "canon_anchors": { "path": "canon/anchors.yaml" },
    "contract": { "id": "UMA.INTSTACK.v1", "path": "contracts/UMA.INTSTACK.v1.yaml" },
    "closures_registry": { "id": "UMCP.CLOSURES.DEFAULT.v1", "path": "closures/registry.yaml" }
  },
  "artifacts": {
    "raw_measurements": {
      "path": "raw_measurements.csv",
      "format": "csv"
    },
    "expected": {
      "psi_csv": { "path": "expected/psi.csv" },
      "invariants_json": { "path": "expected/invariants.json" },
      "ss1m_receipt_json": { "path": "expected/ss1m_receipt.json" }
    }
  },
  "run_intent": {
    "notes": "This CasePack uses a wide ψ CSV with three coordinates c_1..c_3. Values are chosen so Stable regime gates are satisfied under canon thresholds."
  }
}
EOF