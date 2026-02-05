#!/usr/bin/env python3
"""
UMCP Device Network Visualization

Shows the network topology with device trust levels
and VLAN assignments based on UMCP validation.
"""

from datetime import datetime, timedelta
from device_daemon import (
    DeviceDaemon, DeviceCategory, DeviceStatus, NetworkAction,
    NetworkDevice, INF_ANOMALY
)


def print_network_map():
    """Print ASCII network topology with trust levels"""
    
    print()
    print("=" * 80)
    print("                    UMCP DEVICE NETWORK MAP")
    print("=" * 80)
    print()
    
    # Create daemon with sample devices
    daemon = DeviceDaemon()
    
    devices_data = [
        ("AA:BB:CC:DD:EE:01", "192.168.1.10", "alice-laptop", DeviceCategory.WORKSTATION, 0.92),
        ("AA:BB:CC:DD:EE:02", "192.168.1.11", "bob-desktop", DeviceCategory.WORKSTATION, 0.88),
        ("AA:BB:CC:DD:EE:03", "192.168.1.50", "web-server", DeviceCategory.SERVER, 0.95),
        ("AA:BB:CC:DD:EE:04", "192.168.1.51", "db-server", DeviceCategory.SERVER, 0.93),
        ("AA:BB:CC:DD:EE:05", "192.168.1.100", "carol-iphone", DeviceCategory.MOBILE, 0.75),
        ("AA:BB:CC:DD:EE:06", "192.168.1.101", "dave-android", DeviceCategory.MOBILE, 0.70),
        ("AA:BB:CC:DD:EE:07", "192.168.1.200", "lobby-camera", DeviceCategory.IOT, 0.45),
        ("AA:BB:CC:DD:EE:08", "192.168.1.201", "smart-thermostat", DeviceCategory.IOT, 0.55),
        ("AA:BB:CC:DD:EE:09", "192.168.1.202", "door-lock", DeviceCategory.IOT, 0.60),
        ("AA:BB:CC:DD:EE:10", "192.168.1.1", "core-switch", DeviceCategory.NETWORK, 0.98),
        ("AA:BB:CC:DD:EE:11", "192.168.1.2", "wifi-ap-01", DeviceCategory.NETWORK, 0.90),
        ("AA:BB:CC:DD:EE:12", "192.168.1.250", "unknown-device", DeviceCategory.UNKNOWN, 0.30),
    ]
    
    # Add devices with simulated trust
    for mac, ip, hostname, category, trust in devices_data:
        device = daemon.add_device(mac, ip, hostname, category)
        device.current_T = trust
        if trust >= 0.8:
            device.current_tau_A = 1.0
            device.status = DeviceStatus.TRUSTED
        elif trust >= 0.6:
            device.current_tau_A = 30.0
            device.status = DeviceStatus.LIMITED
        elif trust >= 0.4:
            device.current_tau_A = INF_ANOMALY
            device.status = DeviceStatus.QUARANTINE
        else:
            device.current_tau_A = INF_ANOMALY
            device.status = DeviceStatus.BLOCKED
    
    # Print network diagram
    print("""
                           ┌─────────────────────────────────────┐
                           │          INTERNET / WAN             │
                           └───────────────┬─────────────────────┘
                                           │
                           ┌───────────────┴─────────────────────┐
                           │      core-switch (T=0.98) ✓         │
                           │      VLAN Router & Firewall         │
                           └───┬───────┬───────┬───────┬─────────┘
                               │       │       │       │
       ┌───────────────────────┤       │       │       │
       │                       │       │       │       │
       │      VLAN 100         │       │       │       │
       │     (Trusted)         │       │       │       │
       │                       │       │       │       │
    """)
    
    # VLAN 100 - Trusted
    print("┌──────────────────────────────────────┐")
    print("│           VLAN 100 (Trusted)         │")
    print("├──────────────────────────────────────┤")
    
    trusted = [(d.hostname, d.current_T) for d in daemon.devices.values() 
               if d.status == DeviceStatus.TRUSTED]
    for name, t in trusted:
        bar = "█" * int(t * 10)
        print(f"│  ✓ {name:<18} T={t:.2f} [{bar:<10}] │")
    print("│  Full access to all resources        │")
    print("└──────────────────────────────────────┘")
    
    print()
    
    # VLAN 200 - Limited
    print("┌──────────────────────────────────────┐")
    print("│           VLAN 200 (Limited)         │")
    print("├──────────────────────────────────────┤")
    
    limited = [(d.hostname, d.current_T) for d in daemon.devices.values() 
               if d.status == DeviceStatus.LIMITED]
    for name, t in limited:
        bar = "█" * int(t * 10)
        print(f"│  ◐ {name:<18} T={t:.2f} [{bar:<10}] │")
    print("│  Internet only, no internal servers  │")
    print("└──────────────────────────────────────┘")
    
    print()
    
    # VLAN 666 - Quarantine
    print("┌──────────────────────────────────────┐")
    print("│         VLAN 666 (Quarantine)        │")
    print("├──────────────────────────────────────┤")
    
    quarantine = [(d.hostname, d.current_T) for d in daemon.devices.values() 
                  if d.status == DeviceStatus.QUARANTINE]
    if quarantine:
        for name, t in quarantine:
            bar = "█" * int(t * 10)
            print(f"│  ◎ {name:<18} T={t:.2f} [{bar:<10}] │")
    else:
        print("│  (No devices currently quarantined)  │")
    print("│  Remediation server access only      │")
    print("└──────────────────────────────────────┘")
    
    print()
    
    # BLOCKED
    print("┌──────────────────────────────────────┐")
    print("│              BLOCKED                 │")
    print("├──────────────────────────────────────┤")
    
    blocked = [(d.hostname, d.current_T) for d in daemon.devices.values() 
               if d.status == DeviceStatus.BLOCKED]
    for name, t in blocked:
        bar = "█" * int(t * 10)
        print(f"│  ✗ {name:<18} T={t:.2f} [{bar:<10}] │")
    print("│  Port shutdown, no network access    │")
    print("└──────────────────────────────────────┘")
    
    print()
    
    # Legend
    print("=" * 80)
    print("LEGEND")
    print("=" * 80)
    print()
    print("  Status Symbols:")
    print("    ✓  TRUSTED     - High trust (T ≥ 0.80), stable return (τ_A finite)")
    print("    ◐  LIMITED     - Medium trust (0.60 ≤ T < 0.80), restricted access")
    print("    ◎  QUARANTINE  - Low trust (0.40 ≤ T < 0.60), isolated for remediation")
    print("    ✗  BLOCKED     - Very low trust (T < 0.40) or no return (τ_A = ∞)")
    print()
    print("  Trust Bar: [██████████] = T=1.0 (perfect), [███-------] = T=0.3 (low)")
    print()
    print("  VLAN Access Rules:")
    print("    100 (Trusted)    → All internal + external resources")
    print("    200 (Limited)    → External only (guest wifi equivalent)")
    print("    666 (Quarantine) → Remediation server only (10.0.0.1)")
    print("    BLOCKED          → Switch port shutdown, no L2/L3 access")
    print()


def print_trust_dashboard():
    """Print real-time trust dashboard"""
    
    print()
    print("=" * 80)
    print("               UMCP DEVICE TRUST DASHBOARD")
    print("=" * 80)
    print()
    
    # Simulate devices with different trust patterns
    devices = [
        ("alice-laptop", "workstation", 0.92, 1.0, "TRUSTED", "▲ Stable high trust"),
        ("bob-desktop", "workstation", 0.85, 1.0, "TRUSTED", "▲ Recently patched"),
        ("web-server", "server", 0.95, 1.0, "TRUSTED", "▲ Very stable"),
        ("db-server", "server", 0.78, 15.0, "LIMITED", "► Slight drift detected"),
        ("carol-iphone", "mobile", 0.72, 30.0, "LIMITED", "► Normal mobile pattern"),
        ("dave-android", "mobile", 0.68, 45.0, "LIMITED", "► Outdated OS version"),
        ("lobby-camera", "iot", 0.45, float('inf'), "QUARANTINE", "▼ Vuln CVE-2024-1234"),
        ("smart-thermostat", "iot", 0.55, 120.0, "QUARANTINE", "► Recovering from anomaly"),
        ("door-lock", "iot", 0.62, 60.0, "LIMITED", "► Firmware updated"),
        ("core-switch", "network", 0.98, 1.0, "TRUSTED", "▲ Infrastructure stable"),
        ("wifi-ap-01", "network", 0.88, 5.0, "TRUSTED", "▲ Normal operation"),
        ("unknown-device", "unknown", 0.25, float('inf'), "BLOCKED", "✗ No identity verified"),
    ]
    
    # Header
    print(f"  {'Device':<18} {'Type':<12} {'T':>6} {'τ_A':>8} {'Status':<12} {'Trend'}")
    print("  " + "-" * 75)
    
    for name, dtype, trust, tau, status, trend in devices:
        tau_str = f"{tau:.0f}s" if tau != float('inf') else "∞"
        
        # Trust bar
        trust_bar = "█" * int(trust * 10) + "░" * (10 - int(trust * 10))
        
        # Status icon
        icons = {"TRUSTED": "✓", "LIMITED": "◐", "QUARANTINE": "◎", "BLOCKED": "✗"}
        icon = icons.get(status, "?")
        
        print(f"  {icon} {name:<16} {dtype:<12} {trust:.2f}  {tau_str:>7} {status:<12} {trend}")
    
    print()
    
    # Summary stats
    trusted_count = sum(1 for _, _, _, _, s, _ in devices if s == "TRUSTED")
    limited_count = sum(1 for _, _, _, _, s, _ in devices if s == "LIMITED")
    quarantine_count = sum(1 for _, _, _, _, s, _ in devices if s == "QUARANTINE")
    blocked_count = sum(1 for _, _, _, _, s, _ in devices if s == "BLOCKED")
    avg_trust = sum(t for _, _, t, _, _, _ in devices) / len(devices)
    
    print("  " + "=" * 75)
    print()
    print(f"  Summary: {len(devices)} devices | Avg Trust: {avg_trust:.2f}")
    print(f"  ✓ Trusted: {trusted_count}  |  ◐ Limited: {limited_count}  |  ◎ Quarantine: {quarantine_count}  |  ✗ Blocked: {blocked_count}")
    print()


def print_device_lifecycle():
    """Show device trust lifecycle"""
    
    print()
    print("=" * 80)
    print("               DEVICE TRUST LIFECYCLE")
    print("=" * 80)
    print()
    print("  How a new device earns trust through UMCP validation:")
    print()
    
    lifecycle = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  t=0: DEVICE CONNECTS                                                   │
    │                                                                         │
    │    [New Device] ──→ MAC detected on switch port                        │
    │                      │                                                  │
    │                      ▼                                                  │
    │              ┌──────────────────┐                                       │
    │              │  STATUS: PENDING │  T=?, τ_A=∞                          │
    │              │  VLAN: Quarantine│  Isolated until validated            │
    │              └────────┬─────────┘                                       │
    │                       │                                                 │
    │  t=1: FIRST VALIDATION                                                  │
    │                       ▼                                                  │
    │              ┌──────────────────┐                                       │
    │              │  Signal Collection│  Identity, software, config         │
    │              └────────┬─────────┘                                       │
    │                       │                                                 │
    │                       ▼                                                  │
    │              ┌──────────────────┐                                       │
    │              │  UMCP Validation │  Compute T, θ, H, D, σ, τ_A          │
    │              └────────┬─────────┘                                       │
    │                       │                                                 │
    │         ┌─────────────┼─────────────┐                                   │
    │         ▼             ▼             ▼                                   │
    │    ┌─────────┐  ┌──────────┐  ┌──────────┐                              │
    │    │ T ≥ 0.8 │  │ T=0.4-0.8│  │ T < 0.4  │                              │
    │    │ τ_A < ∞ │  │ τ_A any  │  │ τ_A = ∞  │                              │
    │    └────┬────┘  └────┬─────┘  └────┬─────┘                              │
    │         ▼            ▼             ▼                                    │
    │    TRUSTED       LIMITED       BLOCKED                                  │
    │    VLAN 100      VLAN 200      No Access                                │
    │                                                                         │
    │  t=2+: CONTINUOUS VALIDATION                                            │
    │                                                                         │
    │    Device stays trusted if:                                             │
    │    • T remains high (signals stable)                                    │
    │    • τ_A remains finite (returns to baseline)                           │
    │    • σ (seam integrity) accumulates correctly                           │
    │                                                                         │
    │  RECOVERY PATH (Self-Healing):                                          │
    │                                                                         │
    │    BLOCKED ──→ (remediation) ──→ QUARANTINE ──→ LIMITED ──→ TRUSTED    │
    │     T < 0.4      update FW         T ≥ 0.4       T ≥ 0.6     T ≥ 0.8   │
    │                  fix vulns         τ_A finite    stable       stable    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    print(lifecycle)
    
    print("  KEY PRINCIPLE: Trust is EARNED through validated return, not asserted.")
    print()
    print("  Traditional NAC: One-time 802.1X auth → access forever")
    print("  UMCP Device:    Continuous validation → access earned each interval")
    print()


def print_comparison():
    """Compare UMCP device security with traditional approaches"""
    
    print()
    print("=" * 80)
    print("         UMCP DEVICE SECURITY vs TRADITIONAL NETWORK SECURITY")
    print("=" * 80)
    print()
    
    comparison = """
    ┌────────────────────────┬─────────────────────────┬─────────────────────────┐
    │        Aspect          │    Traditional NAC      │    UMCP Device Daemon   │
    ├────────────────────────┼─────────────────────────┼─────────────────────────┤
    │ Authentication         │ One-time (802.1X)       │ Continuous validation   │
    │                        │                         │                         │
    │ Trust Model            │ Static (allow/deny)     │ Dynamic (earned trust)  │
    │                        │                         │                         │
    │ Access Levels          │ Binary (on/off)         │ Graduated (4 levels)    │
    │                        │                         │                         │
    │ Device Assessment      │ Initial only            │ Continuous monitoring   │
    │                        │                         │                         │
    │ Anomaly Response       │ Manual intervention     │ Automatic isolation     │
    │                        │                         │                         │
    │ Recovery               │ Manual whitelist        │ Self-healing (if valid) │
    │                        │                         │                         │
    │ IoT Handling           │ MAC whitelist           │ Trust-based quarantine  │
    │                        │                         │                         │
    │ BYOD                   │ Guest VLAN forever      │ Trust earned over time  │
    │                        │                         │                         │
    │ Rogue Devices          │ Detected if not in DB   │ Blocked (T < threshold) │
    │                        │                         │                         │
    │ Compromised Device     │ Stays trusted until     │ Detected via T drop,    │
    │                        │ manually flagged        │ auto-quarantined        │
    │                        │                         │                         │
    │ Audit Trail            │ Auth logs only          │ Full trust ledger with  │
    │                        │                         │ σ (seam integrity)      │
    │                        │                         │                         │
    │ Mathematical Basis     │ None (rule-based)       │ UMCP axioms (T, τ_A, σ) │
    └────────────────────────┴─────────────────────────┴─────────────────────────┘
    """
    print(comparison)
    
    print()
    print("  UMCP Core Principle Applied to Devices:")
    print()
    print('    "What Returns Through Collapse Is Real"')
    print('    → "A device is trusted if it survives validation and returns to baseline"')
    print()
    print("  Signals Collected Per Device:")
    print("    • Identity: MAC, hostname, certificates, device ID")
    print("    • Software: OS type/version, firmware, patch level")
    print("    • Configuration: Firewall, encryption, antivirus, auto-updates")
    print("    • Behavior: Open ports, connections, bandwidth, auth failures")
    print("    • Risk: Vulnerabilities, security events, policy violations")
    print()
    print("  Invariants Computed:")
    print("    • T (Trust Fidelity): Weighted sum of signal scores")
    print("    • τ_A (Anomaly Return): Time to return to baseline")
    print("    • σ (Seam Integrity): Accumulated trust changes")
    print("    • H (Entropy): Uncertainty in device classification")
    print()


if __name__ == "__main__":
    print_network_map()
    print_trust_dashboard()
    print_device_lifecycle()
    print_comparison()
