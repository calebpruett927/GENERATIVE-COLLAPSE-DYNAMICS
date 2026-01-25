#!/usr/bin/env python3
"""
Collapse Integrity Audit Pipeline for Quantum Physics Experiments.

This script implements the Collapse Integrity Stack audit grammar for
quantum optics experiments, specifically:
- Seam classification (Type I, II, III)
- Budget reconciliation: Δκ = R·τ_R − (D_ω + D_C)
- AX-0 compliance: Only that which returns through collapse is real

Usage:
    python scripts/collapse_audit.py --input physics_data.csv
    python scripts/collapse_audit.py --single --tau-R -0.82 --D-C -0.82 --omega 0 --R 1.0
    python scripts/collapse_audit.py --example

Reference: PHYS-04 (Negative Excitation under Postselection)
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Audit parameters
TOL_SEAM = 0.005  # Maximum allowable residual for audit pass
CRITICAL_INTEGRITY = 0.30  # I < 0.30 flags integrity collapse

@dataclass
class AuditRow:
    """Single audit row in Collapse Integrity grammar."""
    event_id: str
    face: str
    tau_R: float        # Return delay (signed)
    D_C: float          # Curvature change (signed)
    D_omega: float      # Drift cost
    omega: float        # Drift load
    R: float            # Return credit [0, 1]
    delta_kappa: float  # Budget: Δκ = R·τ_R − (D_ω + D_C)
    s: float            # Residual seam discrepancy
    kappa: float        # Log-integrity
    I: float            # Integrity dial: I = exp(κ)
    seam_type: str      # Type I, II, or III
    status: str         # Pass or Fail
    ax0_pass: bool      # AX-0 compliance

def compute_audit(
    tau_R: float,
    D_C: float,
    omega: float = 0.0,
    R: float = 1.0,
    event_id: str = "UNNAMED",
    face: str = "default",
    lambda_drift: float = 0.2
) -> AuditRow:
    """
    Compute Collapse Integrity audit for a single event.
    
    Budget equation:
        Δκ = R·τ_R − (D_ω + D_C)
    
    Where:
        D_ω = λ · ω  (drift cost)
    
    Seam classification:
        Type I:   Δκ = 0, s = 0  (Return Without Loss)
        Type II:  Δκ < 0, |s| ≤ tol  (Return With Loss)
        Type III: |s| > tol  (Irreconcilable Collapse)
    """
    # Compute drift cost
    D_omega = lambda_drift * omega
    
    # Budget equation: Δκ = R·τ_R − (D_ω + D_C)
    delta_kappa = R * tau_R - (D_omega + D_C)
    
    # For this physics case, κ is the accumulated log-integrity
    # In a Type I seam with full return, κ = 0
    # The residual s measures budget discrepancy
    s = delta_kappa  # Residual is the budget imbalance
    
    # Compute kappa based on whether budget balances
    if abs(s) <= TOL_SEAM:
        kappa = 0.0  # Budget balanced
    else:
        kappa = delta_kappa  # Integrity shift
    
    # Integrity dial
    I = math.exp(kappa)
    
    # Seam classification
    if abs(delta_kappa) <= TOL_SEAM and abs(s) <= TOL_SEAM:
        seam_type = "Type I"
        status = "Pass"
    elif abs(s) <= TOL_SEAM:
        seam_type = "Type II"
        status = "Pass"
    else:
        seam_type = "Type III"
        status = "Fail"
    
    # AX-0: Only that which returns through collapse is real
    ax0_pass = (status == "Pass") and (abs(s) <= TOL_SEAM)
    
    # Regime classification based on omega (for future use)
    # regime = "Stable" if omega < 0.038 else ("Watch" if omega < 0.3 else "Collapse")
    _ = omega  # Regime calculation available for future extension
    
    # Critical overlay check
    if I < CRITICAL_INTEGRITY:
        status = f"{status} (Critical)"
    
    return AuditRow(
        event_id=event_id,
        face=face,
        tau_R=tau_R,
        D_C=D_C,
        D_omega=D_omega,
        omega=omega,
        R=R,
        delta_kappa=round(delta_kappa, 6),
        s=round(s, 6),
        kappa=round(kappa, 6),
        I=round(I, 6),
        seam_type=seam_type,
        status=status,
        ax0_pass=ax0_pass
    )

def classify_return(tau_R: float, D_C: float) -> str:
    """Classify return type based on τ_R and D_C."""
    if tau_R > 0:
        return "Class I: Forward-Causal Return"
    elif tau_R < 0 and D_C < 0:
        return "Class IIa: Curvature-Reversed Retro-Coherent Return"
    elif tau_R < 0 and D_C > 0:
        return "Class IIb: Anomalous Backward Return"
    elif tau_R == float('inf') or tau_R == float('-inf'):
        return "Class III: No Return (Unweldable)"
    else:
        return "Class I: Neutral"

def print_seamstamp(audit: AuditRow) -> None:
    """Print SeamStamp format as in the paper."""
    print(f"\nSS1m | {audit.event_id} | Face: {audit.face} | "
          f"τ_R = {audit.tau_R:.2f} | D_C = {audit.D_C:.2f} | ω = {audit.omega} | "
          f"R = {audit.R}")
    print(f"| Δκ = {audit.delta_kappa} | s = {audit.s:.3f} | κ = {audit.kappa} | "
          f"I = {audit.I:.3f} | {audit.seam_type} Weld | "
          f"AX-0 {'Pass' if audit.ax0_pass else 'Fail'}")

def run_batch_audit(input_file: Path) -> list[AuditRow]:
    """Run audit on batch of events from CSV."""
    results: list[AuditRow] = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audit = compute_audit(
                tau_R=float(row.get('tau_R', row.get('τ_R', 0))),
                D_C=float(row.get('D_C', row.get('DC', 0))),
                omega=float(row.get('omega', row.get('ω', 0))),
                R=float(row.get('R', 1.0)),
                event_id=row.get('event_id', row.get('id', 'UNNAMED')),
                face=row.get('face', 'default')
            )
            results.append(audit)
    
    return results

def generate_example_data() -> None:
    """Generate example physics data CSV."""
    example_data: list[dict[str, Any]] = [
        {"event_id": "PHYS-04", "face": "postselected-transmit", "tau_R": -0.82, "D_C": -0.82, "omega": 0, "R": 1.0},
        {"event_id": "PHYS-05", "face": "absorptive", "tau_R": 1.5, "D_C": 0.3, "omega": 0.02, "R": 0.95},
        {"event_id": "PHYS-06", "face": "partial-transmit", "tau_R": 0.5, "D_C": 0.1, "omega": 0.05, "R": 0.8},
        {"event_id": "PHYS-07", "face": "delayed-choice", "tau_R": -0.3, "D_C": -0.35, "omega": 0.01, "R": 1.0},
        {"event_id": "PHYS-08", "face": "weak-measurement", "tau_R": -0.1, "D_C": -0.1, "omega": 0, "R": 1.0},
    ]
    
    output_file = Path("physics_audit_example.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "face", "tau_R", "D_C", "omega", "R"])
        writer.writeheader()
        writer.writerows(example_data)
    
    print(f"Created example file: {output_file}")
    print("\nContents:")
    print("event_id,face,tau_R,D_C,omega,R")
    for row in example_data:
        print(f"{row['event_id']},{row['face']},{row['tau_R']},{row['D_C']},{row['omega']},{row['R']}")

def main():
    parser = argparse.ArgumentParser(
        description="Collapse Integrity Audit for Quantum Physics Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit PHYS-04 (negative excitation time)
  python scripts/collapse_audit.py --single --tau-R -0.82 --D-C -0.82 --omega 0 --R 1.0 --event-id PHYS-04 --face postselected-transmit

  # Batch audit from CSV
  python scripts/collapse_audit.py --input physics_data.csv

  # Generate example data
  python scripts/collapse_audit.py --example

Budget Equation:
  Δκ = R·τ_R − (D_ω + D_C)

Seam Types:
  Type I:   Δκ = 0, s = 0  (Return Without Loss)
  Type II:  Δκ < 0, |s| ≤ tol  (Return With Loss)  
  Type III: |s| > tol  (Irreconcilable Collapse)
        """
    )
    
    parser.add_argument('--input', '-i', type=Path, help='Input CSV file with physics data')
    parser.add_argument('--single', action='store_true', help='Single event audit mode')
    parser.add_argument('--tau-R', type=float, help='Return delay τ_R (signed)')
    parser.add_argument('--D-C', type=float, help='Curvature change D_C (signed)')
    parser.add_argument('--omega', type=float, default=0.0, help='Drift load ω [0, 1]')
    parser.add_argument('--R', type=float, default=1.0, help='Return credit R [0, 1]')
    parser.add_argument('--event-id', type=str, default='UNNAMED', help='Event identifier')
    parser.add_argument('--face', type=str, default='default', help='Observational face')
    parser.add_argument('--example', action='store_true', help='Generate example CSV')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if args.example:
        generate_example_data()
        return 0
    
    if args.single:
        if args.tau_R is None or args.D_C is None:
            print("Error: --single mode requires --tau-R and --D-C")
            return 1
        
        audit = compute_audit(
            tau_R=args.tau_R,
            D_C=args.D_C,
            omega=args.omega,
            R=args.R,
            event_id=args.event_id,
            face=args.face
        )
        
        return_class = classify_return(args.tau_R, args.D_C)
        
        if args.json:
            print(json.dumps(asdict(audit), indent=2))
        else:
            print("=" * 70)
            print("COLLAPSE INTEGRITY AUDIT")
            print("=" * 70)
            print(f"\nEvent: {audit.event_id}")
            print(f"Face: {audit.face}")
            print(f"Return Class: {return_class}")
            print()
            print("Runtime Invariants:")
            print(f"  τ_R = {audit.tau_R:.4f}  (return delay)")
            print(f"  D_C = {audit.D_C:.4f}  (curvature change)")
            print(f"  D_ω = {audit.D_omega:.4f}  (drift cost = λ·ω)")
            print(f"  ω   = {audit.omega:.4f}  (drift load)")
            print(f"  R   = {audit.R:.4f}  (return credit)")
            print()
            print("Budget Reconciliation:")
            print(f"  Δκ = R·τ_R − (D_ω + D_C)")
            print(f"  Δκ = {audit.R}×{audit.tau_R:.2f} − ({audit.D_omega:.2f} + {audit.D_C:.2f})")
            print(f"  Δκ = {audit.delta_kappa}")
            print()
            print("Audit Results:")
            print(f"  Residual s = {audit.s:.6f}")
            print(f"  κ (log-integrity) = {audit.kappa}")
            print(f"  I (integrity dial) = {audit.I:.6f}")
            print(f"  Seam Type: {audit.seam_type}")
            print(f"  Status: {audit.status}")
            print(f"  AX-0: {'✓ PASS' if audit.ax0_pass else '✗ FAIL'}")
            
            print_seamstamp(audit)
        
        return 0 if audit.status == "Pass" else 1
    
    if args.input:
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            return 1
        
        results = run_batch_audit(args.input)
        
        print("=" * 70)
        print("COLLAPSE INTEGRITY BATCH AUDIT")
        print("=" * 70)
        print(f"\nProcessed {len(results)} events from {args.input}")
        print()
        
        # Summary
        type_i = sum(1 for r in results if r.seam_type == "Type I")
        type_ii = sum(1 for r in results if r.seam_type == "Type II")
        type_iii = sum(1 for r in results if r.seam_type == "Type III")
        passed = sum(1 for r in results if r.status == "Pass")
        ax0_passed = sum(1 for r in results if r.ax0_pass)
        
        print("Summary:")
        print(f"  Type I (Return Without Loss): {type_i}")
        print(f"  Type II (Return With Loss):   {type_ii}")
        print(f"  Type III (Irreconcilable):    {type_iii}")
        print(f"  Audit Pass Rate: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
        print(f"  AX-0 Compliance: {ax0_passed}/{len(results)} ({100*ax0_passed/len(results):.1f}%)")
        print()
        
        # Individual results
        print("Individual Results:")
        print("-" * 70)
        for audit in results:
            return_class = classify_return(audit.tau_R, audit.D_C)
            status_symbol = "✓" if audit.status == "Pass" else "✗"
            print(f"{status_symbol} {audit.event_id:12} | {audit.seam_type:8} | "
                  f"τ_R={audit.tau_R:+.2f} | D_C={audit.D_C:+.2f} | "
                  f"Δκ={audit.delta_kappa:+.4f} | I={audit.I:.3f}")
        
        for audit in results:
            print_seamstamp(audit)
        
        return 0 if all(r.status == "Pass" for r in results) else 1
    
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())
