"""
UMCP Security Domain - Security Validator

Main entry point for UMCP-based security validation.
Applies the core axiom: "What Returns Through Collapse Is Real"
Translated for security: "What Survives Validation Is Trusted"

Architecture:
    Tier-0: Frozen security policy (contracts, closures)
    Tier-1: Deterministic security invariants (T, θ, H, D, σ, TIC, τ_A)
    Tier-2: Diagnostic overlays (threat classification, reputation, etc.)

Usage:
    from closures.security.security_validator import SecurityValidator

    validator = SecurityValidator()
    result = validator.validate_entity(signals, entity_type="file")
    print(result.status)  # TRUSTED, SUSPICIOUS, BLOCKED, NON_EVALUABLE
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from closures.security.anomaly_return import (
    compute_anomaly_return,
    compute_anomaly_return_series,
    detect_anomaly_events,
)
from closures.security.behavior_profiler import detect_behavior_anomalies, profile_behavior_series
from closures.security.privacy_auditor import audit_data_privacy, generate_privacy_report
from closures.security.reputation_analyzer import (
    ReputationType,
    analyze_file_hash,
    analyze_url_reputation,
)
from closures.security.security_entropy import compute_security_entropy, compute_signal_dispersion

# Tier-2 domain expansion imports
from closures.security.threat_classifier import (
    classify_threat,
    classify_threat_series,
    generate_threat_report,
)

# Tier-1 kernel imports
from closures.security.trust_fidelity import (
    classify_trust_status,
    compute_trust_fidelity,
)
from closures.security.trust_integrity import compute_trust_integrity


@dataclass
class SecurityInvariants:
    """Tier-1 security invariants."""

    T: float  # Trust Fidelity
    theta: float  # Threat Drift
    H: float  # Security Entropy
    D: float  # Signal Dispersion
    sigma: float  # Log-Integrity
    TIC: float  # Trust Integrity Composite
    tau_A: Any  # Anomaly Return Time (int, "INF_ANOMALY", or "UNIDENTIFIABLE")


@dataclass
class ValidationResult:
    """Complete validation result with receipt."""

    status: str  # TRUSTED, SUSPICIOUS, BLOCKED, NON_EVALUABLE
    invariants: SecurityInvariants
    confidence: float
    threat_type: str
    recommendations: list[str]
    receipt: dict[str, Any]


class SecurityValidator:
    """
    UMCP Security Validator.

    Applies UMCP architecture to security validation:
    - Tier-0: Frozen security policy
    - Tier-1: Deterministic invariant computation
    - Tier-2: Diagnostic overlays
    """

    def __init__(
        self,
        contract_id: str = "SECURITY.INTSTACK.v1",
        weights: np.ndarray | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize security validator.

        Args:
            contract_id: Security contract ID
            weights: Signal weights (default: equal weights)
            thresholds: Classification thresholds
        """
        self.contract_id = contract_id

        # Default weights from threat_patterns.v1.yaml
        self.weights = weights if weights is not None else np.array([0.4, 0.2, 0.25, 0.15])

        # Default thresholds from contract
        self.thresholds = thresholds or {
            "T_trusted": 0.8,
            "T_suspicious": 0.4,
            "eta_baseline": 0.01,
            "H_rec": 64,
            "tol_seam": 0.005,
        }

        self.epsilon = 1e-8

    def compute_invariants(self, signals: np.ndarray, signal_history: np.ndarray | None = None) -> SecurityInvariants:
        """
        Compute Tier-1 security invariants.

        Args:
            signals: Current security signals, shape (n,)
            signal_history: Optional signal history for τ_A computation

        Returns:
            SecurityInvariants object
        """
        # Adjust weights if needed
        weights = self.weights
        if len(signals) != len(weights):
            weights = np.ones(len(signals)) / len(signals)

        # T and θ
        trust = compute_trust_fidelity(signals, weights, self.epsilon)
        T = trust["T"]
        theta = trust["theta"]

        # H (Security Entropy)
        entropy = compute_security_entropy(signals, weights, self.epsilon)
        H = entropy["H"]

        # D (Signal Dispersion)
        dispersion = compute_signal_dispersion(signals)
        D = dispersion["D"]

        # σ and TIC
        integrity = compute_trust_integrity(signals, weights, self.epsilon)
        sigma = integrity["sigma"]
        TIC = integrity["TIC"]

        # τ_A (Anomaly Return Time)
        if signal_history is not None and len(signal_history) > 1:
            t = len(signal_history) - 1
            tau_result = compute_anomaly_return(
                t,
                signal_history,
                eta=self.thresholds["eta_baseline"],
                horizon=int(self.thresholds["H_rec"]),
                weights=weights,
            )
            tau_A = tau_result["tau_A"]
        else:
            tau_A = None

        return SecurityInvariants(T=T, theta=theta, H=H, D=D, sigma=sigma, TIC=TIC, tau_A=tau_A)

    def validate_signals(
        self, signals: np.ndarray, signal_history: np.ndarray | None = None, entity_id: str | None = None
    ) -> ValidationResult:
        """
        Validate security signals and produce result with receipt.

        Args:
            signals: Current security signals, shape (n,)
            signal_history: Optional signal history for return detection
            entity_id: Optional entity identifier

        Returns:
            ValidationResult with status, invariants, and receipt
        """
        # Compute Tier-1 invariants
        invariants = self.compute_invariants(signals, signal_history)

        # Tier-2: Classify threat
        classification = classify_threat(
            T=invariants.T,
            theta=invariants.theta,
            H=invariants.H,
            D=invariants.D,
            sigma=invariants.sigma,
            TIC=invariants.TIC,
            tau_A=invariants.tau_A,
        )

        # Determine final status
        status = classify_trust_status(
            invariants.T,
            invariants.tau_A,
            {
                "trusted": self.thresholds["T_trusted"],
                "suspicious": self.thresholds["T_suspicious"],
                "max_tau_A": self.thresholds["H_rec"] // 2,
            },
        )

        # Generate receipt
        receipt = self._generate_receipt(
            entity_id=entity_id, invariants=invariants, status=status, threat_type=classification.threat_type.value
        )

        return ValidationResult(
            status=status,
            invariants=invariants,
            confidence=classification.confidence,
            threat_type=classification.threat_type.value,
            recommendations=classification.recommendations,
            receipt=receipt,
        )

    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate URL reputation.

        Args:
            url: URL to validate

        Returns:
            ValidationResult
        """
        # Tier-2: Analyze reputation
        rep = analyze_url_reputation(url)

        # Convert to security signals
        # [reputation_score, structure_score, trust_score, history_score]
        signals = np.array(
            [
                rep.score,
                1.0 - len(rep.indicators) * 0.1,  # Structure score
                rep.score,  # Trust score (same as reputation for URLs)
                0.5 if rep.reputation_type == ReputationType.UNKNOWN else rep.score,
            ]
        )
        signals = np.clip(signals, 0, 1)

        result = self.validate_signals(signals, entity_id=url)

        # Add URL-specific details to receipt
        result.receipt["entity_type"] = "url"
        result.receipt["url"] = url
        result.receipt["reputation_indicators"] = rep.indicators[:5]

        return result

    def validate_file_hash(
        self,
        file_hash: str,
        hash_type: str = "sha256",
        blocklist: list[str] | None = None,
        allowlist: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate file hash.

        Args:
            file_hash: File hash to validate
            hash_type: Hash type (sha256, md5, sha1)
            blocklist: Known malicious hashes
            allowlist: Known trusted hashes

        Returns:
            ValidationResult
        """
        # Tier-2: Analyze hash reputation
        rep = analyze_file_hash(file_hash, hash_type, blocklist, allowlist)

        # Convert to security signals
        signals = np.array(
            [
                rep.score,  # Integrity score
                rep.score,  # Reputation score
                1.0 if rep.reputation_type != ReputationType.MALICIOUS else 0.0,
                0.5 if rep.reputation_type == ReputationType.UNKNOWN else rep.score,
            ]
        )
        signals = np.clip(signals, 0, 1)

        result = self.validate_signals(signals, entity_id=file_hash[:16])

        # Add file-specific details
        result.receipt["entity_type"] = "file_hash"
        result.receipt["hash"] = file_hash
        result.receipt["hash_type"] = hash_type

        return result

    def validate_data_privacy(
        self, data: dict[str, Any], consent_records: dict[str, bool] | None = None, is_encrypted: bool = True
    ) -> dict[str, Any]:
        """
        Validate data privacy.

        Args:
            data: Data to audit
            consent_records: Consent status for PII types
            is_encrypted: Whether storage is encrypted

        Returns:
            Privacy audit report
        """
        audit = audit_data_privacy(data=data, consent_records=consent_records, is_encrypted=is_encrypted)

        report = generate_privacy_report(audit)
        report["contract_id"] = self.contract_id
        report["validated_at_utc"] = datetime.utcnow().isoformat()

        return report

    def validate_signal_series(self, signal_series: np.ndarray, entity_id: str | None = None) -> dict[str, Any]:
        """
        Validate signal series over time.

        Args:
            signal_series: Shape (T, n) - signal history
            entity_id: Optional entity identifier

        Returns:
            Complete validation report with invariant series
        """
        weights = self.weights
        if signal_series.shape[1] != len(weights):
            weights = np.ones(signal_series.shape[1]) / signal_series.shape[1]

        # Compute invariant series
        invariants_list = []
        for t in range(signal_series.shape[0]):
            history = signal_series[: t + 1] if t > 0 else None
            inv = self.compute_invariants(signal_series[t], history)
            invariants_list.append(
                {
                    "t": t + 1,
                    "T": inv.T,
                    "theta": inv.theta,
                    "H": inv.H,
                    "D": inv.D,
                    "sigma": inv.sigma,
                    "TIC": inv.TIC,
                    "tau_A": inv.tau_A,
                }
            )

        # Tier-2: Classify threat series
        classifications = classify_threat_series(invariants_list)

        # Generate threat report
        threat_report = generate_threat_report(classifications)

        # Detect anomaly events
        tau_A_series = compute_anomaly_return_series(
            signal_series, eta=self.thresholds["eta_baseline"], horizon=int(self.thresholds["H_rec"]), weights=weights
        )
        anomaly_events = detect_anomaly_events(tau_A_series)

        # Behavior profiling
        behavior_profile = profile_behavior_series(signal_series, weights=weights)
        behavior_anomalies = detect_behavior_anomalies(behavior_profile)

        return {
            "contract_id": self.contract_id,
            "entity_id": entity_id,
            "validated_at_utc": datetime.utcnow().isoformat(),
            "summary": {
                "total_samples": len(invariants_list),
                "overall_status": threat_report["overall_status"],
                "anomaly_events": len(anomaly_events),
                "final_T": invariants_list[-1]["T"],
                "final_TIC": invariants_list[-1]["TIC"],
            },
            "invariants": invariants_list,
            "classifications": classifications,
            "threat_report": threat_report,
            "anomaly_events": anomaly_events,
            "behavior_anomalies": behavior_anomalies,
        }

    def _generate_receipt(
        self, entity_id: str | None, invariants: SecurityInvariants, status: str, threat_type: str
    ) -> dict[str, Any]:
        """Generate validation receipt."""
        return {
            "receipt_type": "security_validation",
            "contract_id": self.contract_id,
            "entity_id": entity_id,
            "validated_at_utc": datetime.utcnow().isoformat(),
            "status": status,
            "invariants": asdict(invariants),
            "threat_type": threat_type,
            "axiom_verification": {
                "axiom": "What Returns Through Collapse Is Real",
                "security_translation": "What Survives Validation Is Trusted",
                "return_status": "finite" if isinstance(invariants.tau_A, int) else str(invariants.tau_A),
                "trust_earned": status == "TRUSTED",
            },
        }


def main():
    """Example usage of SecurityValidator."""
    print("=" * 60)
    print("UMCP Security Validator")
    print("'What Survives Validation Is Trusted'")
    print("=" * 60)

    validator = SecurityValidator()

    # Example 1: Validate signals
    print("\n1. Signal Validation:")
    signals = np.array([0.95, 0.88, 0.92, 0.90])
    result = validator.validate_signals(signals, entity_id="test_entity")
    print(f"   Status: {result.status}")
    print(f"   T={result.invariants.T:.3f}, TIC={result.invariants.TIC:.3f}")
    print(f"   Threat: {result.threat_type}")

    # Example 2: Validate URL
    print("\n2. URL Validation:")
    urls = ["https://github.com/project", "http://secure-l0gin-verify.xyz/account"]
    for url in urls:
        result = validator.validate_url(url)
        print(f"   {url[:40]}...")
        print(f"      Status: {result.status}, T={result.invariants.T:.2f}")

    # Example 3: Privacy audit
    print("\n3. Privacy Audit:")
    test_data = {"email": "user@example.com", "notes": "SSN: 123-45-6789"}
    privacy = validator.validate_data_privacy(test_data, consent_records={"ssn": False}, is_encrypted=False)
    print(f"   Status: {privacy['status']}")
    print(f"   Privacy Score: {privacy['privacy_score']:.2f}")
    print(f"   PII Found: {privacy['pii_summary']['total_found']}")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == "__main__":
    main()
