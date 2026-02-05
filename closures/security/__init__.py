"""
UMCP Security Domain

Applies UMCP architecture to security validation:
- Anti-virus/malware detection
- Anti-scam/phishing detection
- Identity verification
- Privacy protection

Core Axiom Translation:
    "What Returns Through Collapse Is Real" → "What Survives Validation Is Trusted"

Architecture:
    Tier-0: Frozen security policy (contracts, closures)
    Tier-1: Deterministic security invariants (T, θ, H, D, σ, TIC, τ_A)
    Tier-2: Diagnostic overlays (threat classification, reputation, etc.)
"""

# Tier-1 Kernel
from closures.security.trust_fidelity import (
    compute_trust_fidelity,
    compute_trust_fidelity_series,
    classify_trust_status
)
from closures.security.security_entropy import (
    compute_security_entropy,
    compute_signal_dispersion
)
from closures.security.trust_integrity import (
    compute_trust_integrity,
    compute_trust_seam,
    compute_seam_residual
)
from closures.security.anomaly_return import (
    compute_anomaly_return,
    compute_anomaly_return_series,
    detect_anomaly_events,
    TauAType
)

# Tier-2 Overlays
from closures.security.threat_classifier import (
    classify_threat,
    classify_threat_series,
    generate_threat_report,
    ThreatType,
    ThreatClassification
)
from closures.security.reputation_analyzer import (
    analyze_url_reputation,
    analyze_file_hash,
    analyze_ip_reputation,
    ReputationType,
    ReputationResult
)
from closures.security.behavior_profiler import (
    compute_baseline_profile,
    compute_deviation,
    analyze_trend,
    profile_behavior_series,
    detect_behavior_anomalies,
    BehaviorTrend,
    AnomalyLevel
)
from closures.security.privacy_auditor import (
    detect_pii,
    audit_data_privacy,
    generate_privacy_report,
    PIISeverity,
    ViolationType
)

# Main Validator
from closures.security.security_validator import (
    SecurityValidator,
    SecurityInvariants,
    ValidationResult
)

# Daemon Components (Background Service)
from closures.security.security_daemon import (
    SecurityDaemon,
    MonitoredEntity,
    SecurityEvent,
)
from closures.security.response_engine import (
    ResponseEngine,
    ResponseAction,
    ResponseDecision,
)
from closures.security.device_daemon import (
    DeviceDaemon,
    NetworkDevice,
    DeviceCategory,
    DeviceStatus,
    NetworkAction,
)

__all__ = [
    # Tier-1 Kernel
    "compute_trust_fidelity",
    "compute_trust_fidelity_series",
    "classify_trust_status",
    "compute_security_entropy",
    "compute_signal_dispersion",
    "compute_trust_integrity",
    "compute_trust_seam",
    "compute_seam_residual",
    "compute_anomaly_return",
    "compute_anomaly_return_series",
    "detect_anomaly_events",
    "TauAType",
    
    # Tier-2 Overlays
    "classify_threat",
    "classify_threat_series",
    "generate_threat_report",
    "ThreatType",
    "ThreatClassification",
    "analyze_url_reputation",
    "analyze_file_hash",
    "analyze_ip_reputation",
    "ReputationType",
    "ReputationResult",
    "compute_baseline_profile",
    "compute_deviation",
    "analyze_trend",
    "profile_behavior_series",
    "detect_behavior_anomalies",
    "BehaviorTrend",
    "AnomalyLevel",
    "detect_pii",
    "audit_data_privacy",
    "generate_privacy_report",
    "PIISeverity",
    "ViolationType",
    
    # Main Validator
    "SecurityValidator",
    "SecurityInvariants",
    "ValidationResult",
    
    # Daemon Components (Background Service)
    "SecurityDaemon",
    "ResponseEngine",
    "DeviceDaemon",
]
