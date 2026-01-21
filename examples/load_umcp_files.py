#!/usr/bin/env python3
"""
Example: Load and validate root-level UMCP files

This script demonstrates how to programmatically access and validate
the root-level UMCP configuration files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umcp import get_umcp_files


def main():
    """Load and display information from UMCP files."""

    print("=" * 70)
    print("UMCP Root-Level Files Loader Example")
    print("=" * 70)
    print()

    # Initialize file reference helper
    umcp = get_umcp_files()

    # Check file existence
    print("📁 File Existence Check:")
    print("-" * 70)
    all_exist = umcp.verify_all_exist()
    for filename, exists in all_exist.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")

    missing = umcp.get_missing_files()
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return 1

    print("\n✅ All files present!")
    print()

    # Load configuration files
    print("📋 Configuration Summary:")
    print("-" * 70)

    manifest = umcp.load_manifest()
    print(f"  CasePack ID: {manifest.get('casepack', {}).get('id', 'N/A')}")
    print(f"  Version: {manifest.get('casepack', {}).get('version', 'N/A')}")
    print(f"  Title: {manifest.get('casepack', {}).get('title', 'N/A')}")

    contract = umcp.load_contract()
    contract_info = contract.get("contract", {})
    print(f"  Contract ID: {contract_info.get('id', 'N/A')}")
    print(f"  Contract Version: {contract_info.get('version', 'N/A')}")

    embedding = umcp.load_embedding()
    emb_params = embedding.get("embedding", {}).get("parameters", {})
    print(f"  Embedding Interval: {emb_params.get('interval', 'N/A')}")
    print(f"  Epsilon: {emb_params.get('epsilon', 'N/A')}")
    print()

    # Load and validate observables
    print("🔬 Observables:")
    print("-" * 70)
    observables = umcp.load_observables()
    primary = observables.get("observables", {}).get("primary", [])
    derived = observables.get("observables", {}).get("derived", [])
    print(f"  Primary: {len(primary)} variables")
    print(f"  Derived: {len(derived)} variables")
    for obs in primary[:3]:  # Show first 3
        print(f"    - {obs.get('id', 'N/A')}: {obs.get('description', 'N/A')}")
    print()

    # Load weights
    print("⚖️  Weights:")
    print("-" * 70)
    weights = umcp.load_weights()
    if weights:
        w_row = weights[0]
        w_values = [float(v) for v in w_row.values()]
        print(f"  Values: {w_values}")
        print(f"  Sum: {sum(w_values):.6f}")
        print(f"  All non-negative: {all(w >= 0 for w in w_values)}")
    print()

    # Load trace
    print("📊 Trace Data:")
    print("-" * 70)
    trace = umcp.load_trace()
    trace_meta = umcp.load_trace_meta()
    print(f"  Format: {trace_meta.get('trace_meta', {}).get('format', 'N/A')}")
    print(f"  Rows: {len(trace)}")
    props = trace_meta.get("trace_meta", {}).get("properties", {})
    print(f"  Coordinates: {props.get('coordinate_count', 'N/A')}")
    if trace:
        t_row = trace[0]
        coords = [k for k in t_row if k.startswith("c_")]
        print(f"  Coordinate columns: {coords}")
        coord_values = [float(t_row[c]) for c in coords]
        print(f"  Values: {coord_values}")
        all_in_range = all(0 <= v <= 1 for v in coord_values)
        print(f"  All in [0,1]: {all_in_range}")
    print()

    # Load invariants
    print("📈 Tier-1 Invariants:")
    print("-" * 70)
    invariants = umcp.load_invariants()
    print(f"  Rows: {len(invariants)}")
    if invariants:
        inv = invariants[0]
        omega = float(inv["omega"])
        F = float(inv["F"])
        S = float(inv["S"])
        C = float(inv["C"])
        kappa = float(inv["kappa"])
        IC = float(inv["IC"])

        print(f"  ω (omega): {omega:.6f}")
        print(f"  F: {F:.6f}")
        print(f"  S: {S:.6f}")
        print(f"  C: {C:.6f}")
        print(f"  κ (kappa): {kappa:.6f}")
        print(f"  IC: {IC:.6f}")
        print(f"  Regime: {inv['regime_label']}")

        # Validate Tier-1 identities
        f_check = abs(F - (1 - omega))
        ic_check = abs(IC - 2.71828**kappa)  # exp(kappa)
        print("\n  Tier-1 Validation:")
        print(f"    F ≈ 1-ω: |{F:.6f} - {1 - omega:.6f}| = {f_check:.2e} {'✓' if f_check < 1e-6 else '✗'}")
        print(f"    IC ≈ exp(κ): {ic_check:.2e} {'✓' if ic_check < 1e-6 else '✗'}")
    print()

    # Load regimes
    print("🎯 Regimes:")
    print("-" * 70)
    regimes = umcp.load_regimes()
    print(f"  Classifications: {len(regimes)}")
    for regime in regimes:
        print(f"    t={regime['t']}: {regime['regime_label']} (critical: {regime['critical_overlay']})")
    print()

    # Load integrity information
    print("🔒 Integrity:")
    print("-" * 70)
    checksums = umcp.load_sha256()
    checksum_lines = [line for line in checksums.strip().split("\n") if line]
    print(f"  SHA256 checksums: {len(checksum_lines)} files")

    env_info = umcp.load_env()
    python_version = env_info.split("\n")[0] if env_info else "Unknown"
    print(f"  Environment: {python_version}")

    version_info = umcp.load_code_version()
    commit_hash = version_info.split("\n")[0] if version_info else "Unknown"
    print(f"  Code version: {commit_hash[:12]}...")
    print()

    # Load report
    print("📝 Validation Report:")
    print("-" * 70)
    report = umcp.load_report()
    report_lines = report.split("\n")
    # Show summary lines
    for line in report_lines[:15]:
        if line.strip():
            print(f"  {line}")
    print(f"  ... ({len(report_lines)} total lines)")
    print()

    print("=" * 70)
    print("✅ All UMCP files loaded successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
