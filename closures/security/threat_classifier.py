"""
UMCP Security Domain - Tier-2 Overlay: Threat Classifier

Classifies threat types based on Tier-1 invariants.
This is a DIAGNOSTIC overlay - it reads Tier-1 outputs but cannot modify them.

Tier-2 rules:
    - May read Tier-1 outputs (T, θ, H, D, σ, TIC, τ_A)
    - CANNOT reach upstream to alter interface, trace, or kernel
    - Results remain Tier-2 unless promoted through return-based canonization

Threat Types:
    - BENIGN: No threat detected, all signals normal
    - TRANSIENT_ANOMALY: Temporary deviation, returns to baseline
    - PERSISTENT_THREAT: Sustained low trust, no return
    - ATTACK_IN_PROGRESS: Active degradation of trust
    - RECOVERY: System recovering from threat
    - UNKNOWN: Cannot classify with confidence
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ThreatType(Enum):
    """Threat classification types."""

    BENIGN = "BENIGN"
    TRANSIENT_ANOMALY = "TRANSIENT_ANOMALY"
    PERSISTENT_THREAT = "PERSISTENT_THREAT"
    ATTACK_IN_PROGRESS = "ATTACK_IN_PROGRESS"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ThreatClassification:
    """Threat classification result."""

    threat_type: ThreatType
    confidence: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: list[str]
    invariants_used: dict[str, Any]


def classify_threat(
    T: float | None,
    theta: float | None,
    H: float | None,
    D: float | None,
    sigma: float | None,
    TIC: float | None,
    tau_A: Any,
    T_history: list[float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> ThreatClassification:
    """
    Classify threat type based on Tier-1 invariants.

    Args:
        T: Trust Fidelity (current)
        theta: Threat Drift (current)
        H: Security Entropy (current)
        D: Signal Dispersion (current)
        sigma: Log-Integrity (current)
        TIC: Trust Integrity Composite (current)
        tau_A: Anomaly Return Time (int, "INF_ANOMALY", or "UNIDENTIFIABLE")
        T_history: Optional history of T values for trend detection
        thresholds: Classification thresholds

    Returns:
        ThreatClassification with type, confidence, severity, recommendations
    """
    if thresholds is None:
        thresholds = {
            "T_trusted": 0.8,
            "T_suspicious": 0.4,
            "T_critical": 0.2,
            "H_high": 0.5,
            "D_high": 0.3,
            "tau_A_slow": 32,
        }

    invariants = {"T": T, "theta": theta, "H": H, "D": D, "sigma": sigma, "TIC": TIC, "tau_A": tau_A}

    # Check for non-evaluable (any required invariant is None)
    if T is None or theta is None or tau_A == "UNIDENTIFIABLE":
        return ThreatClassification(
            threat_type=ThreatType.UNKNOWN,
            confidence=0.0,
            severity="UNKNOWN",
            recommendations=["Insufficient data for classification", "Gather more signals"],
            invariants_used=invariants,
        )

    # Detect trend if history available
    trend = "stable"
    if T_history and len(T_history) >= 3:
        recent = T_history[-3:]
        if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
            trend = "declining"
        elif all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
            trend = "improving"

    # At this point T is guaranteed to be float (checked above)
    T_val = float(T) if T is not None else 0.0
    H_val = float(H) if H is not None else 0.0

    # Classification logic

    # BENIGN: High trust, low entropy, finite return
    if T_val >= thresholds["T_trusted"] and H_val < thresholds["H_high"] and tau_A != "INF_ANOMALY":
        return ThreatClassification(
            threat_type=ThreatType.BENIGN,
            confidence=min(0.95, T_val),
            severity="LOW",
            recommendations=["Continue monitoring", "No action required"],
            invariants_used=invariants,
        )

    # ATTACK_IN_PROGRESS: Declining trust trend
    if trend == "declining" and T_val < thresholds["T_trusted"]:
        severity = "CRITICAL" if T_val < thresholds["T_critical"] else "HIGH"
        return ThreatClassification(
            threat_type=ThreatType.ATTACK_IN_PROGRESS,
            confidence=0.85,
            severity=severity,
            recommendations=[
                "Active threat detected - immediate response required",
                "Isolate affected systems",
                "Enable enhanced logging",
                "Alert security team",
            ],
            invariants_used=invariants,
        )

    # PERSISTENT_THREAT: Low trust, no return
    if T_val < thresholds["T_suspicious"] and tau_A == "INF_ANOMALY":
        severity = "CRITICAL" if T_val < thresholds["T_critical"] else "HIGH"
        return ThreatClassification(
            threat_type=ThreatType.PERSISTENT_THREAT,
            confidence=0.90,
            severity=severity,
            recommendations=[
                "Persistent threat - system compromised",
                "Initiate incident response",
                "Consider system isolation or rebuild",
                "Preserve forensic evidence",
            ],
            invariants_used=invariants,
        )

    # RECOVERY: Improving trust trend after anomaly
    if trend == "improving" and T_val < thresholds["T_trusted"]:
        return ThreatClassification(
            threat_type=ThreatType.RECOVERY,
            confidence=0.75,
            severity="MEDIUM",
            recommendations=[
                "System recovering from threat",
                "Monitor for complete recovery",
                "Verify all indicators return to baseline",
                "Conduct post-incident review",
            ],
            invariants_used=invariants,
        )

    # TRANSIENT_ANOMALY: Low trust but finite return
    if T_val < thresholds["T_suspicious"] and isinstance(tau_A, int):
        return ThreatClassification(
            threat_type=ThreatType.TRANSIENT_ANOMALY,
            confidence=0.70,
            severity="MEDIUM" if tau_A <= thresholds["tau_A_slow"] else "HIGH",
            recommendations=[
                "Transient anomaly detected",
                f"Expected recovery in {tau_A} samples",
                "Monitor for escalation",
                "Review recent changes",
            ],
            invariants_used=invariants,
        )

    # SUSPICIOUS: Moderate concern
    if T_val < thresholds["T_trusted"]:
        return ThreatClassification(
            threat_type=ThreatType.TRANSIENT_ANOMALY,
            confidence=0.60,
            severity="MEDIUM",
            recommendations=["Elevated risk detected", "Increase monitoring frequency", "Review access logs"],
            invariants_used=invariants,
        )

    # Default: BENIGN with lower confidence
    return ThreatClassification(
        threat_type=ThreatType.BENIGN,
        confidence=0.70,
        severity="LOW",
        recommendations=["Continue standard monitoring"],
        invariants_used=invariants,
    )


def classify_threat_series(
    invariants_series: list[dict[str, Any]], thresholds: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """
    Classify threats over time series.

    Args:
        invariants_series: List of invariant dicts with T, theta, H, D, sigma, TIC, tau_A
        thresholds: Classification thresholds

    Returns:
        List of classification results per timestep
    """
    results = []
    T_history = []

    for i, inv in enumerate(invariants_series):
        T = inv.get("T")
        if T is not None:
            T_history.append(T)

        classification = classify_threat(
            T=inv.get("T"),
            theta=inv.get("theta"),
            H=inv.get("H"),
            D=inv.get("D"),
            sigma=inv.get("sigma"),
            TIC=inv.get("TIC"),
            tau_A=inv.get("tau_A"),
            T_history=T_history.copy(),
            thresholds=thresholds,
        )

        results.append(
            {
                "t": inv.get("t", i + 1),
                "threat_type": classification.threat_type.value,
                "confidence": classification.confidence,
                "severity": classification.severity,
                "recommendations": classification.recommendations,
            }
        )

    return results


def generate_threat_report(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate summary threat report from classification series.

    Args:
        classifications: List of classification results

    Returns:
        Summary report with threat counts, timeline, recommendations
    """
    # Count threat types
    type_counts = {}
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "UNKNOWN": 0}

    for c in classifications:
        threat_type = c.get("threat_type", "UNKNOWN")
        severity = c.get("severity", "UNKNOWN")

        type_counts[threat_type] = type_counts.get(threat_type, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Determine overall status
    if severity_counts["CRITICAL"] > 0:
        overall_status = "CRITICAL"
    elif severity_counts["HIGH"] > 0:
        overall_status = "HIGH_RISK"
    elif severity_counts["MEDIUM"] > 0:
        overall_status = "ELEVATED"
    else:
        overall_status = "NORMAL"

    # Find threat events
    events = []
    current_event = None

    for c in classifications:
        threat_type = c.get("threat_type")

        if threat_type not in ["BENIGN", "UNKNOWN"]:
            if current_event is None or current_event["type"] != threat_type:
                if current_event:
                    events.append(current_event)
                current_event = {"type": threat_type, "start": c["t"], "end": c["t"], "severity": c["severity"]}
            else:
                current_event["end"] = c["t"]
        elif current_event:
            events.append(current_event)
            current_event = None

    if current_event:
        events.append(current_event)

    return {
        "overall_status": overall_status,
        "total_samples": len(classifications),
        "threat_type_counts": type_counts,
        "severity_counts": severity_counts,
        "threat_events": events,
        "recommendations": _aggregate_recommendations(classifications),
    }


def _aggregate_recommendations(classifications: list[dict[str, Any]]) -> list[str]:
    """Aggregate and deduplicate recommendations."""
    all_recs = []
    for c in classifications:
        for rec in c.get("recommendations", []):
            if rec not in all_recs:
                all_recs.append(rec)
    return all_recs[:10]  # Top 10 recommendations


if __name__ == "__main__":
    # Example: classify threat from invariants
    result = classify_threat(T=0.35, theta=0.65, H=0.7, D=0.15, sigma=-1.05, TIC=0.35, tau_A="INF_ANOMALY")

    print(f"Threat Type: {result.threat_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Severity: {result.severity}")
    print("Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
