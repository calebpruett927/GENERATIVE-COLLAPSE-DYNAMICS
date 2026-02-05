#!/usr/bin/env python3
"""
UMCP Security CasePack - Validation Test

Runs the security_validation casepack through UMCP validation.
Demonstrates the core axiom: "What Survives Validation Is Trusted"
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from closures.security.trust_fidelity import compute_trust_fidelity, classify_trust_status
from closures.security.security_entropy import compute_security_entropy, compute_signal_dispersion
from closures.security.trust_integrity import compute_trust_integrity
from closures.security.anomaly_return import compute_anomaly_return_series, detect_anomaly_events
from closures.security.threat_classifier import classify_threat_series, generate_threat_report
from closures.security.reputation_analyzer import analyze_url_reputation


def load_security_signals(path: Union[str, Path]) -> np.ndarray:
    """Load security signals from CSV."""
    df = pd.read_csv(path)
    signal_cols = ['integrity_score', 'reputation_score', 'behavior_score', 'identity_score']
    return df[signal_cols].values


def run_casepack_validation():
    """Run security validation casepack."""
    print("=" * 70)
    print("UMCP Security CasePack Validation")
    print("Axiom: 'What Survives Validation Is Trusted'")
    print("=" * 70)
    
    # Load casepack data
    casepack_dir = Path(__file__).parent
    signals_path = casepack_dir / "raw_security_signals.csv"
    urls_path = casepack_dir / "urls_to_check.csv"
    
    # Frozen weights from Tier-0
    weights = np.array([0.4, 0.2, 0.25, 0.15])
    
    # =========================================================================
    # PART 1: Signal Series Validation
    # =========================================================================
    print("\n" + "-" * 70)
    print("PART 1: Security Signal Series Validation")
    print("-" * 70)
    
    signal_series = load_security_signals(signals_path)
    T, n = signal_series.shape
    
    print(f"\nLoaded {T} samples with {n} signals each")
    
    # Compute Tier-1 invariants for each timestep
    invariants_list = []
    for t in range(T):
        signals = signal_series[t]
        
        trust = compute_trust_fidelity(signals, weights)
        entropy = compute_security_entropy(signals, weights)
        dispersion = compute_signal_dispersion(signals)
        integrity = compute_trust_integrity(signals, weights)
        
        invariants_list.append({
            "t": t + 1,
            "T": trust["T"],
            "theta": trust["theta"],
            "H": entropy["H"],
            "D": dispersion["D"],
            "sigma": integrity["sigma"],
            "TIC": integrity["TIC"],
            "tau_A": None  # Computed separately
        })
    
    # Compute τ_A for entire series
    tau_A_results = compute_anomaly_return_series(
        signal_series,
        eta=0.01,
        horizon=64,
        weights=weights
    )
    
    for i, tau_result in enumerate(tau_A_results):
        invariants_list[i]["tau_A"] = tau_result["tau_A"]
    
    # Display invariants
    print("\nTier-1 Invariants:")
    print(f"{'t':>3} {'T':>6} {'θ':>6} {'H':>6} {'D':>6} {'TIC':>6} {'τ_A':>10}")
    print("-" * 50)
    
    for inv in invariants_list:
        tau_str = str(inv["tau_A"]) if inv["tau_A"] is not None else "-"
        print(f"{inv['t']:>3} {inv['T']:>6.3f} {inv['theta']:>6.3f} "
              f"{inv['H']:>6.3f} {inv['D']:>6.3f} {inv['TIC']:>6.3f} {tau_str:>10}")
    
    # Detect anomaly events
    anomaly_events = detect_anomaly_events(tau_A_results)
    
    print(f"\nAnomaly Events Detected: {len(anomaly_events)}")
    for event in anomaly_events:
        print(f"  - {event['type']}: t={event['start']} to t={event['end']} "
              f"(duration={event.get('duration', '?')})")
    
    # Tier-2: Threat classification
    classifications = classify_threat_series(invariants_list)
    threat_report = generate_threat_report(classifications)
    
    print(f"\nTier-2 Threat Report:")
    print(f"  Overall Status: {threat_report['overall_status']}")
    print(f"  Threat Type Counts: {threat_report['threat_type_counts']}")
    
    # =========================================================================
    # PART 2: URL Validation
    # =========================================================================
    print("\n" + "-" * 70)
    print("PART 2: URL Reputation Validation")
    print("-" * 70)
    
    urls_df = pd.read_csv(urls_path)
    
    print(f"\n{'URL':<50} {'Status':>12} {'Score':>6}")
    print("-" * 70)
    
    for _, row in urls_df.iterrows():
        url = row['url']
        rep = analyze_url_reputation(url)
        
        # Classify based on reputation
        status = classify_trust_status(rep.score, 1 if rep.score > 0.5 else "INF_ANOMALY")
        
        display_url = url[:47] + "..." if len(url) > 50 else url
        print(f"{display_url:<50} {status:>12} {rep.score:>6.2f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    final_inv = invariants_list[-1]
    
    print(f"\nFinal State (t={final_inv['t']}):")
    print(f"  Trust Fidelity (T): {final_inv['T']:.4f}")
    print(f"  Trust IC (TIC): {final_inv['TIC']:.4f}")
    print(f"  Anomaly Return (τ_A): {final_inv['tau_A']}")
    
    # Determine final validation status
    final_status = classify_trust_status(final_inv['T'], final_inv['tau_A'])
    
    print(f"\n  FINAL STATUS: {final_status}")
    
    # Axiom verification
    print("\n" + "-" * 70)
    print("AXIOM VERIFICATION")
    print("-" * 70)
    print("  Axiom: 'What Returns Through Collapse Is Real'")
    print("  Security Translation: 'What Survives Validation Is Trusted'")
    print()
    
    if final_inv['tau_A'] is not None and final_inv['tau_A'] != "INF_ANOMALY":
        print(f"  ✓ Return detected (τ_A = {final_inv['tau_A']})")
        print(f"  ✓ Trust earned through validated return")
    else:
        print(f"  ✗ No return to baseline (τ_A = {final_inv['tau_A']})")
        print(f"  ✗ Trust NOT earned - entity remains non-canonical")
    
    print("\n" + "=" * 70)
    print("Validation complete.")
    print("=" * 70)
    
    return invariants_list, threat_report


if __name__ == "__main__":
    run_casepack_validation()
