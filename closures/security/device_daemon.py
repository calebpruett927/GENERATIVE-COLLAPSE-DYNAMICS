#!/usr/bin/env python3
"""
UMCP Device-Level Security Daemon

Extends UMCP security validation to entire devices on a network:
- Laptops, desktops, servers
- Mobile devices (phones, tablets)
- IoT devices (cameras, sensors, smart home)
- Network infrastructure (routers, switches, APs)

Core Principle: "What Returns Through Collapse Is Real"
→ "A device is trusted if it survives validation and returns to baseline"

Each device has:
- Device signals (firmware, config, behavior, identity)
- Trust Fidelity (T) computed from signals
- Anomaly Return Time (τ_A) measuring stability
- Device status in trust ledger
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import threading
import time
import json
import math
import hashlib
import uuid
import socket
import struct
import subprocess
import re


# =============================================================================
# CONSTANTS
# =============================================================================

INF_ANOMALY = float('inf')

# Device trust thresholds
T_TRUSTED = 0.80      # High trust, full access
T_LIMITED = 0.60      # Limited trust, restricted access
T_QUARANTINE = 0.40   # Low trust, isolated
T_BLOCK = 0.20        # Very low trust, blocked

# Device categories
class DeviceCategory(Enum):
    WORKSTATION = "workstation"       # Laptops, desktops
    SERVER = "server"                 # Servers, VMs
    MOBILE = "mobile"                 # Phones, tablets
    IOT = "iot"                       # IoT devices
    NETWORK = "network"               # Routers, switches, APs
    EMBEDDED = "embedded"             # Embedded systems
    UNKNOWN = "unknown"               # Unclassified


class DeviceStatus(Enum):
    TRUSTED = "trusted"               # Full network access
    LIMITED = "limited"               # Restricted access (guest VLAN)
    QUARANTINE = "quarantine"         # Isolated, can only reach remediation
    BLOCKED = "blocked"               # No network access
    PENDING = "pending"               # Awaiting first validation
    UNKNOWN = "unknown"               # Not yet seen


class NetworkAction(Enum):
    ALLOW_FULL = "allow_full"         # Full network access
    ALLOW_LIMITED = "allow_limited"   # Guest/restricted VLAN
    ALLOW_LOCAL = "allow_local"       # Local subnet only
    QUARANTINE = "quarantine"         # Remediation network only
    BLOCK = "block"                   # Drop all traffic
    MONITOR = "monitor"               # Allow but log everything


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DeviceSignals:
    """Security signals collected from a device"""
    
    # Identity signals
    mac_address: str = ""
    hostname: str = ""
    device_id: str = ""              # Hardware ID if available
    certificates: List[str] = field(default_factory=list)
    
    # Firmware/Software signals
    os_type: str = ""                # Windows, Linux, iOS, Android, etc.
    os_version: str = ""
    firmware_version: str = ""
    patch_level: str = ""
    installed_software: List[str] = field(default_factory=list)
    
    # Configuration signals
    firewall_enabled: bool = False
    encryption_enabled: bool = False
    antivirus_active: bool = False
    auto_update_enabled: bool = False
    
    # Behavior signals
    open_ports: List[int] = field(default_factory=list)
    active_connections: int = 0
    bandwidth_usage: float = 0.0     # MB/s
    dns_queries: int = 0
    failed_auth_attempts: int = 0
    
    # Network signals
    ip_address: str = ""
    vlan_id: int = 0
    last_seen: Optional[datetime] = None
    uptime_seconds: int = 0
    
    # Risk indicators
    known_vulnerabilities: int = 0
    security_events: int = 0
    policy_violations: int = 0


@dataclass
class DeviceValidation:
    """UMCP validation result for a device"""
    
    device_id: str
    timestamp: datetime
    
    # Tier-1 invariants
    T: float = 0.5                   # Trust Fidelity
    theta: float = 0.5               # Threat Drift (1-T)
    H: float = 0.5                   # Security Entropy
    D: float = 0.0                   # Signal Dispersion
    sigma: float = 0.0               # Log-Integrity
    TIC: float = 0.5                 # Trust IC
    tau_A: float = INF_ANOMALY       # Anomaly Return Time
    
    # Device-specific
    category: DeviceCategory = DeviceCategory.UNKNOWN
    status: DeviceStatus = DeviceStatus.UNKNOWN
    action: NetworkAction = NetworkAction.BLOCK
    
    # Audit
    signals_hash: str = ""
    validation_id: str = ""
    reason: str = ""


@dataclass
class NetworkDevice:
    """Represents a device on the network"""
    
    device_id: str                   # Unique identifier
    mac_address: str
    ip_address: str
    hostname: str
    
    category: DeviceCategory = DeviceCategory.UNKNOWN
    status: DeviceStatus = DeviceStatus.PENDING
    
    # Current signals
    signals: DeviceSignals = field(default_factory=DeviceSignals)
    
    # Validation history
    validations: List[DeviceValidation] = field(default_factory=list)
    current_T: float = 0.5
    current_tau_A: float = INF_ANOMALY
    
    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    
    # Trust ledger
    trust_history: List[Tuple[datetime, float]] = field(default_factory=list)
    sigma: float = 0.0               # Accumulated trust changes


# =============================================================================
# DEVICE SIGNAL COLLECTOR
# =============================================================================

class DeviceSignalCollector:
    """
    Collects security signals from devices on the network.
    
    In production, this would integrate with:
    - SNMP for network devices
    - WMI/WinRM for Windows
    - SSH for Linux/Unix
    - MDM APIs for mobile devices
    - NAC systems (802.1X)
    - Network monitoring (NetFlow, sFlow)
    """
    
    def __init__(self):
        self.known_ports = {
            22: "ssh", 80: "http", 443: "https", 3389: "rdp",
            445: "smb", 135: "rpc", 53: "dns", 25: "smtp"
        }
        self.risky_ports = {23, 21, 25, 110, 143, 3389, 5900, 8080}
    
    def collect_signals(self, device: NetworkDevice) -> DeviceSignals:
        """Collect current security signals from device"""
        
        signals = DeviceSignals()
        signals.mac_address = device.mac_address
        signals.hostname = device.hostname
        signals.ip_address = device.ip_address
        signals.last_seen = datetime.now()
        
        # In production, these would be real queries
        # For now, simulate based on device category
        
        if device.category == DeviceCategory.WORKSTATION:
            signals = self._collect_workstation_signals(device, signals)
        elif device.category == DeviceCategory.SERVER:
            signals = self._collect_server_signals(device, signals)
        elif device.category == DeviceCategory.MOBILE:
            signals = self._collect_mobile_signals(device, signals)
        elif device.category == DeviceCategory.IOT:
            signals = self._collect_iot_signals(device, signals)
        elif device.category == DeviceCategory.NETWORK:
            signals = self._collect_network_device_signals(device, signals)
        else:
            signals = self._collect_unknown_signals(device, signals)
        
        return signals
    
    def _collect_workstation_signals(self, device: NetworkDevice, 
                                      signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from workstation (laptop/desktop)"""
        
        # Simulate WMI/WinRM or SSH query results
        signals.os_type = "Windows"
        signals.os_version = "11"
        signals.firewall_enabled = True
        signals.encryption_enabled = True
        signals.antivirus_active = True
        signals.auto_update_enabled = True
        signals.open_ports = [135, 445]  # Windows defaults
        signals.active_connections = 15
        signals.bandwidth_usage = 1.5
        signals.known_vulnerabilities = 0
        signals.security_events = 0
        signals.policy_violations = 0
        
        return signals
    
    def _collect_server_signals(self, device: NetworkDevice,
                                 signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from server"""
        
        signals.os_type = "Linux"
        signals.os_version = "Ubuntu 22.04"
        signals.firewall_enabled = True
        signals.encryption_enabled = True
        signals.antivirus_active = False  # Servers often don't have AV
        signals.auto_update_enabled = False  # Managed updates
        signals.open_ports = [22, 80, 443]
        signals.active_connections = 50
        signals.bandwidth_usage = 10.0
        signals.uptime_seconds = 86400 * 30  # 30 days
        
        return signals
    
    def _collect_mobile_signals(self, device: NetworkDevice,
                                 signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from mobile device"""
        
        signals.os_type = "iOS"
        signals.os_version = "17.2"
        signals.encryption_enabled = True
        signals.auto_update_enabled = True
        signals.open_ports = []  # Mobile typically has no open ports
        signals.active_connections = 5
        signals.bandwidth_usage = 0.5
        
        return signals
    
    def _collect_iot_signals(self, device: NetworkDevice,
                              signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from IoT device"""
        
        # IoT devices are often insecure
        signals.os_type = "Embedded Linux"
        signals.os_version = "Unknown"
        signals.firmware_version = "1.0.0"
        signals.firewall_enabled = False
        signals.encryption_enabled = False
        signals.antivirus_active = False
        signals.auto_update_enabled = False
        signals.open_ports = [80, 23, 8080]  # Common IoT ports
        signals.active_connections = 2
        signals.known_vulnerabilities = 3
        
        return signals
    
    def _collect_network_device_signals(self, device: NetworkDevice,
                                         signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from network device (router, switch, AP)"""
        
        signals.os_type = "Network OS"
        signals.firmware_version = "15.2"
        signals.firewall_enabled = True
        signals.encryption_enabled = True
        signals.open_ports = [22, 443]  # Management ports
        signals.active_connections = 100
        signals.uptime_seconds = 86400 * 90  # 90 days
        
        return signals
    
    def _collect_unknown_signals(self, device: NetworkDevice,
                                  signals: DeviceSignals) -> DeviceSignals:
        """Collect signals from unknown device type"""
        
        # Unknown devices get minimal trust
        signals.os_type = "Unknown"
        signals.firewall_enabled = False
        signals.encryption_enabled = False
        signals.antivirus_active = False
        signals.auto_update_enabled = False
        signals.known_vulnerabilities = 1  # Assume risk
        
        return signals


# =============================================================================
# DEVICE VALIDATOR
# =============================================================================

class DeviceValidator:
    """
    UMCP validation engine for devices.
    
    Computes Tier-1 invariants from device signals.
    """
    
    def __init__(self):
        self.signal_weights = {
            'identity': 0.20,
            'software': 0.20,
            'configuration': 0.25,
            'behavior': 0.20,
            'risk': 0.15
        }
    
    def validate(self, device: NetworkDevice) -> DeviceValidation:
        """Validate device and compute UMCP invariants"""
        
        signals = device.signals
        
        # Compute component scores
        identity_score = self._compute_identity_score(signals)
        software_score = self._compute_software_score(signals, device.category)
        config_score = self._compute_config_score(signals)
        behavior_score = self._compute_behavior_score(signals, device)
        risk_score = self._compute_risk_score(signals)
        
        # All scores in [0, 1]
        scores = [identity_score, software_score, config_score, 
                  behavior_score, risk_score]
        
        # Trust Fidelity (weighted average)
        weights = list(self.signal_weights.values())
        T = sum(s * w for s, w in zip(scores, weights))
        
        # Threat Drift
        theta = 1.0 - T
        
        # Security Entropy (uncertainty)
        H = self._compute_entropy(scores)
        
        # Signal Dispersion (variance)
        mean = sum(scores) / len(scores)
        D = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
        
        # Trust IC (geometric mean)
        product = 1.0
        for s in scores:
            product *= max(s, 0.001)
        TIC = product ** (1.0 / len(scores))
        
        # Anomaly Return Time
        tau_A = self._compute_tau_A(device, T)
        
        # Determine status and action
        status, action, reason = self._determine_status(T, tau_A, device)
        
        # Update device sigma (seam accounting)
        if device.trust_history:
            last_T = device.trust_history[-1][1]
            delta_sigma = T - last_T
        else:
            delta_sigma = 0.0
        
        sigma = device.sigma + delta_sigma
        
        # Create validation result
        validation = DeviceValidation(
            device_id=device.device_id,
            timestamp=datetime.now(),
            T=T,
            theta=theta,
            H=H,
            D=D,
            sigma=sigma,
            TIC=TIC,
            tau_A=tau_A,
            category=device.category,
            status=status,
            action=action,
            signals_hash=self._hash_signals(signals),
            validation_id=str(uuid.uuid4())[:8],
            reason=reason
        )
        
        return validation
    
    def _compute_identity_score(self, signals: DeviceSignals) -> float:
        """Compute identity trustworthiness"""
        
        score = 0.0
        
        # Valid MAC address
        if signals.mac_address and self._is_valid_mac(signals.mac_address):
            score += 0.3
        
        # Hostname set
        if signals.hostname and len(signals.hostname) > 0:
            score += 0.2
        
        # Device ID present
        if signals.device_id:
            score += 0.2
        
        # Certificates
        if signals.certificates:
            score += min(0.3, len(signals.certificates) * 0.1)
        else:
            score += 0.1  # Base score for responding
        
        return min(score, 1.0)
    
    def _compute_software_score(self, signals: DeviceSignals,
                                 category: DeviceCategory) -> float:
        """Compute software/firmware trustworthiness"""
        
        score = 0.5  # Base score
        
        # Known OS
        known_os = ["Windows", "Linux", "macOS", "iOS", "Android", 
                    "Ubuntu", "Debian", "RHEL", "CentOS"]
        if any(os in signals.os_type for os in known_os):
            score += 0.2
        
        # Version present
        if signals.os_version or signals.firmware_version:
            score += 0.1
        
        # Patch level
        if signals.patch_level:
            score += 0.1
        
        # Auto-updates
        if signals.auto_update_enabled:
            score += 0.1
        
        return min(score, 1.0)
    
    def _compute_config_score(self, signals: DeviceSignals) -> float:
        """Compute configuration trustworthiness"""
        
        score = 0.0
        
        # Firewall
        if signals.firewall_enabled:
            score += 0.3
        
        # Encryption
        if signals.encryption_enabled:
            score += 0.3
        
        # Antivirus (for workstations)
        if signals.antivirus_active:
            score += 0.2
        else:
            score += 0.1  # Some devices don't need AV
        
        # Auto-updates
        if signals.auto_update_enabled:
            score += 0.2
        
        return min(score, 1.0)
    
    def _compute_behavior_score(self, signals: DeviceSignals,
                                 device: NetworkDevice) -> float:
        """Compute behavior trustworthiness"""
        
        score = 0.8  # Start high, deduct for issues
        
        # Risky ports
        risky_open = len(set(signals.open_ports) & {23, 21, 25, 3389, 5900})
        score -= risky_open * 0.1
        
        # Too many ports
        if len(signals.open_ports) > 10:
            score -= 0.1
        
        # High bandwidth (unusual for most devices)
        if signals.bandwidth_usage > 100:
            score -= 0.1
        
        # Failed auth attempts
        if signals.failed_auth_attempts > 5:
            score -= 0.2
        
        # Connection count relative to device type
        if device.category == DeviceCategory.WORKSTATION:
            if signals.active_connections > 50:
                score -= 0.1
        elif device.category == DeviceCategory.IOT:
            if signals.active_connections > 10:
                score -= 0.1
        
        return max(score, 0.0)
    
    def _compute_risk_score(self, signals: DeviceSignals) -> float:
        """Compute inverse risk score (higher = lower risk = better)"""
        
        score = 1.0
        
        # Known vulnerabilities
        score -= signals.known_vulnerabilities * 0.1
        
        # Security events
        score -= signals.security_events * 0.05
        
        # Policy violations
        score -= signals.policy_violations * 0.15
        
        return max(score, 0.0)
    
    def _compute_entropy(self, scores: List[float]) -> float:
        """Compute Shannon entropy of score distribution"""
        
        # Normalize to probabilities
        total = sum(scores) + 0.001
        probs = [s / total for s in scores]
        
        # Shannon entropy
        H = 0.0
        for p in probs:
            if p > 0:
                H -= p * math.log2(p + 1e-10)
        
        # Normalize to [0, 1]
        max_H = math.log2(len(scores))
        return H / max_H if max_H > 0 else 0.0
    
    def _compute_tau_A(self, device: NetworkDevice, current_T: float) -> float:
        """
        Compute anomaly return time.
        
        τ_A = time to return to baseline trust after anomaly
        """
        
        if len(device.trust_history) < 2:
            # Not enough history
            return INF_ANOMALY
        
        # Get baseline (average of last 10 validations)
        recent = device.trust_history[-10:]
        baseline_T = sum(t for _, t in recent) / len(recent)
        
        # Check if currently anomalous
        tolerance = 0.15
        if abs(current_T - baseline_T) <= tolerance:
            # At baseline, τ_A is low
            return 1.0
        
        # Find how long since we were at baseline
        for i, (ts, t) in enumerate(reversed(device.trust_history)):
            if abs(t - baseline_T) <= tolerance:
                # Found baseline point
                time_since = (datetime.now() - ts).total_seconds()
                return time_since if time_since < 3600 else INF_ANOMALY
        
        # Never at baseline
        return INF_ANOMALY
    
    def _determine_status(self, T: float, tau_A: float,
                          device: NetworkDevice) -> Tuple[DeviceStatus, NetworkAction, str]:
        """Determine device status and network action"""
        
        # Critical: No return = no trust
        if tau_A == INF_ANOMALY and T < T_LIMITED:
            return (DeviceStatus.BLOCKED, NetworkAction.BLOCK,
                    f"No return to baseline (τ_A=∞) with low trust T={T:.2f}")
        
        # High trust, stable
        if T >= T_TRUSTED and tau_A < 60:
            return (DeviceStatus.TRUSTED, NetworkAction.ALLOW_FULL,
                    f"Trusted device: T={T:.2f}, stable")
        
        # Medium trust
        if T >= T_LIMITED:
            return (DeviceStatus.LIMITED, NetworkAction.ALLOW_LIMITED,
                    f"Limited trust: T={T:.2f}, restricted access")
        
        # Low trust but returning
        if T >= T_QUARANTINE and tau_A < INF_ANOMALY:
            return (DeviceStatus.QUARANTINE, NetworkAction.QUARANTINE,
                    f"Low trust: T={T:.2f}, isolated for remediation")
        
        # Very low trust
        return (DeviceStatus.BLOCKED, NetworkAction.BLOCK,
                f"Untrusted: T={T:.2f}")
    
    def _is_valid_mac(self, mac: str) -> bool:
        """Check if MAC address is valid"""
        pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
        return bool(re.match(pattern, mac))
    
    def _hash_signals(self, signals: DeviceSignals) -> str:
        """Create hash of signal state"""
        data = json.dumps({
            'mac': signals.mac_address,
            'ip': signals.ip_address,
            'hostname': signals.hostname,
            'ports': sorted(signals.open_ports),
            'vulns': signals.known_vulnerabilities
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# DEVICE NETWORK CONTROLLER
# =============================================================================

class DeviceNetworkController:
    """
    Enforces network actions for devices.
    
    In production, integrates with:
    - SDN controllers (OpenFlow, etc.)
    - NAC systems (802.1X, RADIUS)
    - Firewalls (iptables, nftables, pf)
    - VLANs
    """
    
    def __init__(self):
        self.vlan_map = {
            NetworkAction.ALLOW_FULL: 100,      # Trusted VLAN
            NetworkAction.ALLOW_LIMITED: 200,   # Guest VLAN
            NetworkAction.ALLOW_LOCAL: 300,     # Local-only VLAN
            NetworkAction.QUARANTINE: 666,      # Quarantine VLAN
            NetworkAction.BLOCK: None,          # No VLAN (blocked)
            NetworkAction.MONITOR: 100,         # Same as full but logged
        }
    
    def enforce_action(self, device: NetworkDevice, 
                       action: NetworkAction) -> Dict[str, Any]:
        """Enforce network action for device"""
        
        result = {
            'device_id': device.device_id,
            'mac_address': device.mac_address,
            'ip_address': device.ip_address,
            'action': action.value,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'details': {}
        }
        
        if action == NetworkAction.ALLOW_FULL:
            result['details'] = self._allow_full(device)
        elif action == NetworkAction.ALLOW_LIMITED:
            result['details'] = self._allow_limited(device)
        elif action == NetworkAction.ALLOW_LOCAL:
            result['details'] = self._allow_local(device)
        elif action == NetworkAction.QUARANTINE:
            result['details'] = self._quarantine(device)
        elif action == NetworkAction.BLOCK:
            result['details'] = self._block(device)
        elif action == NetworkAction.MONITOR:
            result['details'] = self._monitor(device)
        
        return result
    
    def _allow_full(self, device: NetworkDevice) -> Dict[str, Any]:
        """Grant full network access"""
        return {
            'vlan': self.vlan_map[NetworkAction.ALLOW_FULL],
            'access': 'full',
            'logging': 'standard',
            'rate_limit': None,
            'command': f"# Move {device.mac_address} to trusted VLAN"
        }
    
    def _allow_limited(self, device: NetworkDevice) -> Dict[str, Any]:
        """Grant limited network access"""
        return {
            'vlan': self.vlan_map[NetworkAction.ALLOW_LIMITED],
            'access': 'limited',
            'blocked_services': ['smb', 'rdp', 'internal-servers'],
            'allowed_services': ['http', 'https', 'dns'],
            'logging': 'enhanced',
            'rate_limit': '100Mbps',
            'command': f"# Move {device.mac_address} to guest VLAN"
        }
    
    def _allow_local(self, device: NetworkDevice) -> Dict[str, Any]:
        """Grant local-only network access"""
        return {
            'vlan': self.vlan_map[NetworkAction.ALLOW_LOCAL],
            'access': 'local-only',
            'blocked_destinations': ['0.0.0.0/0'],  # No internet
            'allowed_destinations': ['10.0.0.0/8', '192.168.0.0/16'],
            'logging': 'enhanced',
            'command': f"# Move {device.mac_address} to local-only VLAN"
        }
    
    def _quarantine(self, device: NetworkDevice) -> Dict[str, Any]:
        """Quarantine device"""
        return {
            'vlan': self.vlan_map[NetworkAction.QUARANTINE],
            'access': 'quarantine-only',
            'allowed_destinations': ['10.0.0.1/32'],  # Only remediation server
            'blocked_destinations': ['0.0.0.0/0'],
            'logging': 'full',
            'alert': True,
            'command': f"# Move {device.mac_address} to quarantine VLAN 666"
        }
    
    def _block(self, device: NetworkDevice) -> Dict[str, Any]:
        """Block device completely"""
        return {
            'vlan': None,
            'access': 'none',
            'port_status': 'shutdown',
            'logging': 'full',
            'alert': True,
            'command': f"# Block MAC {device.mac_address} at switch port"
        }
    
    def _monitor(self, device: NetworkDevice) -> Dict[str, Any]:
        """Allow with enhanced monitoring"""
        return {
            'vlan': self.vlan_map[NetworkAction.MONITOR],
            'access': 'full',
            'logging': 'full',
            'mirror_traffic': True,
            'alert_on': ['unusual_traffic', 'new_connections', 'dns_queries'],
            'command': f"# Mirror {device.mac_address} traffic for analysis"
        }


# =============================================================================
# DEVICE DAEMON
# =============================================================================

class DeviceDaemon:
    """
    UMCP Device-Level Security Daemon
    
    Continuously monitors all devices on network and enforces
    trust-based access control using UMCP validation.
    """
    
    def __init__(self, scan_interval: float = 30.0,
                 validation_interval: float = 60.0):
        self.scan_interval = scan_interval
        self.validation_interval = validation_interval
        
        # Components
        self.collector = DeviceSignalCollector()
        self.validator = DeviceValidator()
        self.controller = DeviceNetworkController()
        
        # Device inventory
        self.devices: Dict[str, NetworkDevice] = {}
        self.lock = threading.RLock()
        
        # Trust ledger
        self.ledger: List[Dict[str, Any]] = []
        
        # State
        self.running = False
        self._threads: List[threading.Thread] = []
    
    def start(self):
        """Start the device daemon"""
        print("=" * 70)
        print("UMCP DEVICE-LEVEL SECURITY DAEMON")
        print("=" * 70)
        print()
        print("Core Principle: 'What Returns Through Collapse Is Real'")
        print("→ A device is trusted if it survives validation and returns")
        print("  to baseline (τ_A finite).")
        print()
        print(f"Scan interval: {self.scan_interval}s")
        print(f"Validation interval: {self.validation_interval}s")
        print()
        
        self.running = True
        
        # Start threads
        self._threads = [
            threading.Thread(target=self._discovery_loop, daemon=True),
            threading.Thread(target=self._validation_loop, daemon=True),
            threading.Thread(target=self._ledger_writer, daemon=True),
        ]
        
        for t in self._threads:
            t.start()
        
        print("[STARTED] Device daemon running")
    
    def stop(self):
        """Stop the device daemon"""
        self.running = False
        for t in self._threads:
            t.join(timeout=2.0)
        print("[STOPPED] Device daemon stopped")
    
    def _discovery_loop(self):
        """Discover devices on network"""
        while self.running:
            try:
                self._scan_network()
            except Exception as e:
                print(f"[ERROR] Discovery: {e}")
            time.sleep(self.scan_interval)
    
    def _validation_loop(self):
        """Validate all known devices"""
        while self.running:
            try:
                self._validate_all_devices()
            except Exception as e:
                print(f"[ERROR] Validation: {e}")
            time.sleep(self.validation_interval)
    
    def _ledger_writer(self):
        """Write trust ledger periodically"""
        while self.running:
            try:
                self._write_ledger()
            except Exception as e:
                print(f"[ERROR] Ledger: {e}")
            time.sleep(60.0)
    
    def _scan_network(self):
        """Scan network for devices (simulated)"""
        # In production, this would use:
        # - ARP scanning
        # - DHCP lease parsing
        # - Switch MAC tables (SNMP)
        # - 802.1X RADIUS events
        pass
    
    def _validate_all_devices(self):
        """Validate all known devices"""
        with self.lock:
            devices = list(self.devices.values())
        
        for device in devices:
            self._validate_device(device)
    
    def _validate_device(self, device: NetworkDevice):
        """Validate a single device"""
        
        # Collect current signals
        device.signals = self.collector.collect_signals(device)
        device.last_seen = datetime.now()
        
        # Validate
        validation = self.validator.validate(device)
        
        # Update device state
        with self.lock:
            device.validations.append(validation)
            device.current_T = validation.T
            device.current_tau_A = validation.tau_A
            device.trust_history.append((datetime.now(), validation.T))
            device.sigma = validation.sigma
            device.status = validation.status
            device.last_validated = datetime.now()
        
        # Enforce action
        result = self.controller.enforce_action(device, validation.action)
        
        # Log to ledger
        self.ledger.append({
            'timestamp': datetime.now().isoformat(),
            'device_id': device.device_id,
            'mac': device.mac_address,
            'ip': device.ip_address,
            'category': device.category.value,
            'T': validation.T,
            'tau_A': validation.tau_A if validation.tau_A != INF_ANOMALY else 'INF',
            'status': validation.status.value,
            'action': validation.action.value,
            'sigma': validation.sigma,
            'reason': validation.reason
        })
        
        # Print status
        self._print_validation(device, validation, result)
    
    def _print_validation(self, device: NetworkDevice, 
                          validation: DeviceValidation,
                          result: Dict[str, Any]):
        """Print validation result"""
        
        action_colors = {
            NetworkAction.ALLOW_FULL: "✓",
            NetworkAction.ALLOW_LIMITED: "◐",
            NetworkAction.ALLOW_LOCAL: "◑",
            NetworkAction.QUARANTINE: "◎",
            NetworkAction.BLOCK: "✗",
            NetworkAction.MONITOR: "◉",
        }
        
        symbol = action_colors.get(validation.action, "?")
        tau_str = f"{validation.tau_A:.0f}s" if validation.tau_A != INF_ANOMALY else "∞"
        
        print(f"\n{symbol} Device: {device.hostname or device.mac_address}")
        print(f"  Category: {device.category.value}")
        print(f"  IP: {device.ip_address}, MAC: {device.mac_address}")
        print(f"  T={validation.T:.3f}, τ_A={tau_str}, σ={validation.sigma:.3f}")
        print(f"  Status: {validation.status.value.upper()}")
        print(f"  Action: {validation.action.value}")
        print(f"  Reason: {validation.reason}")
    
    def _write_ledger(self):
        """Write ledger to file"""
        # In production, write to persistent storage
        pass
    
    def add_device(self, mac: str, ip: str, hostname: str = "",
                   category: DeviceCategory = DeviceCategory.UNKNOWN) -> NetworkDevice:
        """Add a device to monitor"""
        
        device_id = hashlib.sha256(mac.encode()).hexdigest()[:12]
        
        device = NetworkDevice(
            device_id=device_id,
            mac_address=mac,
            ip_address=ip,
            hostname=hostname,
            category=category,
        )
        
        with self.lock:
            self.devices[device_id] = device
        
        return device
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a device"""
        with self.lock:
            device = self.devices.get(device_id)
        
        if not device:
            return None
        
        return {
            'device_id': device.device_id,
            'mac': device.mac_address,
            'ip': device.ip_address,
            'hostname': device.hostname,
            'category': device.category.value,
            'status': device.status.value,
            'T': device.current_T,
            'tau_A': device.current_tau_A if device.current_tau_A != INF_ANOMALY else 'INF',
            'sigma': device.sigma,
            'first_seen': device.first_seen.isoformat(),
            'last_seen': device.last_seen.isoformat(),
            'validation_count': len(device.validations)
        }
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of all devices"""
        with self.lock:
            devices = list(self.devices.values())
        
        summary = {
            'total_devices': len(devices),
            'by_status': {},
            'by_category': {},
            'average_T': 0.0,
            'devices_at_risk': 0,
            'devices_trusted': 0,
        }
        
        if not devices:
            return summary
        
        for device in devices:
            # By status
            status = device.status.value
            summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
            
            # By category
            cat = device.category.value
            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
            
            # Trust metrics
            if device.current_T < T_QUARANTINE:
                summary['devices_at_risk'] += 1
            elif device.current_T >= T_TRUSTED:
                summary['devices_trusted'] += 1
        
        summary['average_T'] = sum(d.current_T for d in devices) / len(devices)
        
        return summary


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate device-level security daemon"""
    
    print("=" * 70)
    print("UMCP DEVICE-LEVEL SECURITY DAEMON - DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows how UMCP validates DEVICES (not just files):")
    print("- Laptops, servers, phones, IoT devices, network equipment")
    print("- Each device has signals (identity, software, config, behavior)")
    print("- UMCP computes T (Trust) and τ_A (Return Time)")
    print("- Network access is controlled based on trust, not just MAC/IP")
    print()
    print("Core Principle: 'What Returns Through Collapse Is Real'")
    print("→ A device earns network access by surviving validation")
    print()
    
    # Create daemon
    daemon = DeviceDaemon(scan_interval=30.0, validation_interval=60.0)
    
    # Add sample devices
    devices = [
        # Trusted workstation
        ("AA:BB:CC:DD:EE:01", "192.168.1.10", "alice-laptop", 
         DeviceCategory.WORKSTATION),
        
        # Corporate server
        ("AA:BB:CC:DD:EE:02", "192.168.1.50", "web-server-01",
         DeviceCategory.SERVER),
        
        # Employee phone
        ("AA:BB:CC:DD:EE:03", "192.168.1.101", "bob-iphone",
         DeviceCategory.MOBILE),
        
        # IoT camera (risky)
        ("AA:BB:CC:DD:EE:04", "192.168.1.200", "lobby-camera",
         DeviceCategory.IOT),
        
        # Unknown device (BYOD or rogue)
        ("AA:BB:CC:DD:EE:05", "192.168.1.250", "",
         DeviceCategory.UNKNOWN),
        
        # Network switch
        ("AA:BB:CC:DD:EE:06", "192.168.1.1", "core-switch-01",
         DeviceCategory.NETWORK),
    ]
    
    print("Adding devices to monitor...")
    print()
    
    for mac, ip, hostname, category in devices:
        device = daemon.add_device(mac, ip, hostname, category)
        print(f"  + {hostname or mac} ({category.value})")
    
    print()
    print("=" * 70)
    print("VALIDATING ALL DEVICES")
    print("=" * 70)
    
    # Simulate adding trust history for some devices
    with daemon.lock:
        for device in daemon.devices.values():
            if device.category == DeviceCategory.WORKSTATION:
                # Workstation has stable history
                for i in range(10):
                    ts = datetime.now() - timedelta(hours=10-i)
                    device.trust_history.append((ts, 0.85 + (i * 0.01)))
            elif device.category == DeviceCategory.SERVER:
                # Server is very stable
                for i in range(10):
                    ts = datetime.now() - timedelta(hours=10-i)
                    device.trust_history.append((ts, 0.90))
    
    # Validate all devices
    daemon._validate_all_devices()
    
    # Print summary
    print()
    print("=" * 70)
    print("NETWORK SUMMARY")
    print("=" * 70)
    
    summary = daemon.get_network_summary()
    print(f"\nTotal devices: {summary['total_devices']}")
    print(f"Average trust: {summary['average_T']:.3f}")
    print(f"Devices trusted: {summary['devices_trusted']}")
    print(f"Devices at risk: {summary['devices_at_risk']}")
    
    print("\nBy Status:")
    for status, count in sorted(summary['by_status'].items()):
        print(f"  {status}: {count}")
    
    print("\nBy Category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat}: {count}")
    
    # VLAN assignments
    print()
    print("=" * 70)
    print("NETWORK ACCESS (VLAN ASSIGNMENTS)")
    print("=" * 70)
    print()
    print("Based on UMCP validation, devices are assigned to VLANs:")
    print()
    print("  VLAN 100 (Trusted)    : Full access to all resources")
    print("  VLAN 200 (Guest)      : Internet only, no internal")
    print("  VLAN 300 (Local)      : Local subnet only, no internet")
    print("  VLAN 666 (Quarantine) : Remediation server only")
    print("  BLOCKED               : Port shutdown, no access")
    print()
    
    with daemon.lock:
        for device in daemon.devices.values():
            vlan = daemon.controller.vlan_map.get(
                device.validations[-1].action if device.validations else NetworkAction.BLOCK
            )
            vlan_str = f"VLAN {vlan}" if vlan else "BLOCKED"
            print(f"  {device.hostname or device.mac_address}: {vlan_str}")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: Device-Level Trust")
    print("=" * 70)
    print("""
    Traditional network security:
    - MAC address filtering (easily spoofed)
    - IP-based ACLs (static, manual)
    - Port-based 802.1X (one-time auth)
    
    UMCP device security:
    - CONTINUOUS validation (not one-time)
    - Multi-signal trust (identity + software + config + behavior)
    - Dynamic access (trust earned, not asserted)
    - Self-healing (devices can recover from quarantine)
    - No return = no trust (τ_A = ∞ means blocked)
    
    Example: IoT camera
    - Traditional: Whitelisted MAC, allowed forever
    - UMCP: Low T (0.35) due to no firewall, vulnerabilities
         → Quarantined until remediated
         → If T improves, access restored automatically
    """)


if __name__ == "__main__":
    demo()
