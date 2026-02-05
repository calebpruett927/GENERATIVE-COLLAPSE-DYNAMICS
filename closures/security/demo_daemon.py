#!/usr/bin/env python3
"""
UMCP Security Daemon - Quick Start Demo

Demonstrates the background antivirus running in UMCP mode:
"What Survives Validation Is Trusted"
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from closures.security.security_daemon import SecurityDaemon
from closures.security.response_engine import ResponseEngine

def demo():
    """Run a quick demo of the security daemon."""
    print("=" * 70)
    print("UMCP Security Daemon - Background Antivirus Demo")
    print("=" * 70)
    print()
    print("Core Principle:")
    print("  'What Returns Through Collapse Is Real'")
    print("  → 'What Survives Validation Is Trusted'")
    print()
    print("How it works:")
    print("  1. Daemon monitors files, network, processes continuously")
    print("  2. Collects security signals for each entity")
    print("  3. Validates through UMCP (computes T, θ, H, D, σ, TIC, τ_A)")
    print("  4. Takes action based on validation, not signatures")
    print()
    print("Response Logic:")
    print("  • τ_A = ∞ (no return) → BLOCK")
    print("  • T < 0.4 (low trust) → QUARANTINE")
    print("  • 0.4 ≤ T < 0.8 → ALLOW but MONITOR")
    print("  • T ≥ 0.8 and τ_A finite → TRUSTED (full access)")
    print()
    print("-" * 70)
    print()
    
    # Initialize daemon
    daemon = SecurityDaemon(
        log_dir="/tmp/umcp_security_demo/logs",
        state_dir="/tmp/umcp_security_demo/state",
        check_interval=0.5
    )
    
    # Initialize response engine
    response_engine = ResponseEngine(
        quarantine_dir="/tmp/umcp_security_demo/quarantine",
        dry_run=True  # Don't actually quarantine in demo
    )
    
    print("✓ Daemon initialized")
    print("✓ Response engine initialized")
    print()
    print("Simulating background monitoring...")
    print("(In production, this would run 24/7)")
    print()
    
    # Simulate some file validations
    test_files = [
        ("/usr/bin/python3", "trusted_system_binary"),
        ("/tmp/test_script.py", "user_script"),
        ("/tmp/suspicious.exe", "suspicious_file"),
        ("/tmp/malware.bin", "malicious_file"),
    ]
    
    for file_path, description in test_files:
        print(f"Validating: {file_path} ({description})")
        
        # Simulate signal collection and validation
        if "python3" in file_path:
            # System binary - high trust
            signals = [0.95, 0.92, 0.90, 0.93]
            T, tau_A = 0.92, 1
            status = "TRUSTED"
        elif "script" in file_path:
            # User script - medium trust
            signals = [0.70, 0.65, 0.72, 0.68]
            T, tau_A = 0.68, 8
            status = "SUSPICIOUS"
        elif "suspicious" in file_path:
            # Suspicious - low trust but may recover
            signals = [0.45, 0.38, 0.42, 0.40]
            T, tau_A = 0.41, 25
            status = "SUSPICIOUS"
        else:
            # Malicious - very low trust, no return
            signals = [0.15, 0.10, 0.12, 0.08]
            T, tau_A = 0.11, "INF_ANOMALY"
            status = "BLOCKED"
        
        # Make response decision
        decision = response_engine.decide_action(
            entity_id=f"file:{file_path}",
            entity_type="file",
            validation_status=status,
            T=T,
            tau_A=tau_A,
            threat_type="computed_from_invariants"
        )
        
        print(f"  Signals: {signals}")
        print(f"  T (Trust): {T:.3f}")
        print(f"  τ_A (Return Time): {tau_A}")
        print(f"  → Action: {decision.action.value}")
        print(f"  → Reason: {decision.reason}")
        print()
    
    print("-" * 70)
    print()
    print("Response Summary:")
    summary = response_engine.get_summary()
    print(f"  Total Decisions: {summary['total_decisions']}")
    print(f"  Actions Taken:")
    for action, count in summary['action_counts'].items():
        print(f"    - {action}: {count}")
    print()
    print(f"  Escalation Levels:")
    for level, count in summary['escalation_counts'].items():
        print(f"    - {level}: {count}")
    print()
    
    print("=" * 70)
    print("Demo Complete")
    print()
    print("To run the daemon for real:")
    print("  python closures/security/security_daemon.py start")
    print()
    print("Key Differences from Traditional Antivirus:")
    print()
    print("Traditional AV:")
    print("  • Signature-based detection")
    print("  • Binary yes/no decision")
    print("  • No trust accounting")
    print()
    print("UMCP Security:")
    print("  • Validation-based trust (T, τ_A)")
    print("  • Graduated response (ALLOW/MONITOR/QUARANTINE/BLOCK)")
    print("  • Trust ledger with seam accounting")
    print("  • 'No return = no trust credit' principle")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demo()
