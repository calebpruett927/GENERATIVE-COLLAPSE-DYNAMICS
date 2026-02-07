"""
UMCP Security Domain - Tier-2 Overlay: Privacy Auditor

Audits data for privacy violations (PII detection, consent, retention).
This is a DIAGNOSTIC overlay - it reads data and compares against
frozen privacy rules from Tier-0.

Tier-2 rules:
    - Reads frozen privacy rules from Tier-0 (privacy_rules.v1.yaml)
    - Scans data for PII patterns
    - Produces DIAGNOSTIC report - does not alter data
    - Privacy score feeds into Tier-0 signals for next validation

Privacy Checks:
    - PII Detection (SSN, credit cards, emails, phones, IPs)
    - Consent Verification
    - Retention Policy Compliance
    - Encryption Status
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PIISeverity(Enum):
    """PII severity levels."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ViolationType(Enum):
    """Privacy violation types."""

    PII_EXPOSURE = "PII_EXPOSURE"
    CONSENT_MISSING = "CONSENT_MISSING"
    RETENTION_EXCEEDED = "RETENTION_EXCEEDED"
    ENCRYPTION_MISSING = "ENCRYPTION_MISSING"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"


@dataclass
class PIIMatch:
    """Detected PII match."""

    pii_type: str
    value_masked: str  # Masked for security
    location: str  # Field or position
    severity: PIISeverity


@dataclass
class PrivacyViolation:
    """Privacy violation record."""

    violation_type: ViolationType
    severity: PIISeverity
    details: str
    remediation: str


@dataclass
class PrivacyAuditResult:
    """Complete privacy audit result."""

    pii_found: list[PIIMatch]
    violations: list[PrivacyViolation]
    privacy_score: float  # 0.0 (worst) to 1.0 (best)
    recommendations: list[str]


# PII Patterns (from privacy_rules.v1.yaml)
PII_PATTERNS = {
    "ssn": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "severity": PIISeverity.CRITICAL, "mask": "XXX-XX-{last4}"},
    "credit_card": {
        "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        "severity": PIISeverity.CRITICAL,
        "mask": "XXXX-XXXX-XXXX-{last4}",
        "validate": "luhn",
    },
    "email": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "severity": PIISeverity.MODERATE,
        "mask": "{first}***@{domain}",
    },
    "phone": {
        "pattern": r"\b(?:\+?1[-.]?)?(?:\(?\d{3}\)?[-.]?)?\d{3}[-.]?\d{4}\b",
        "severity": PIISeverity.MODERATE,
        "mask": "XXX-XXX-{last4}",
    },
    "ip_address": {
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "severity": PIISeverity.LOW,
        "mask": "{first}.{second}.XXX.XXX",
    },
}


def luhn_validate(card_number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.

    Args:
        card_number: Card number string (digits only)

    Returns:
        True if valid Luhn checksum
    """
    digits = [int(d) for d in re.sub(r"\D", "", card_number)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    # Luhn algorithm
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def mask_pii(value: str, pii_type: str) -> str:
    """
    Mask PII value for safe logging/display.

    Args:
        value: Original PII value
        pii_type: Type of PII

    Returns:
        Masked value
    """
    if pii_type == "ssn":
        return f"XXX-XX-{value[-4:]}"
    elif pii_type == "credit_card":
        digits = re.sub(r"\D", "", value)
        return f"XXXX-XXXX-XXXX-{digits[-4:]}"
    elif pii_type == "email":
        parts = value.split("@")
        if len(parts) == 2:
            return f"{parts[0][0]}***@{parts[1]}"
        return "***@***"
    elif pii_type == "phone":
        digits = re.sub(r"\D", "", value)
        return f"XXX-XXX-{digits[-4:]}"
    elif pii_type == "ip_address":
        parts = value.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.XXX.XXX"
        return "XXX.XXX.XXX.XXX"
    else:
        return "***REDACTED***"


def detect_pii(text: str, field_name: str = "unknown", pii_types: list[str] | None = None) -> list[PIIMatch]:
    """
    Detect PII in text.

    Args:
        text: Text to scan
        field_name: Name of field being scanned
        pii_types: Types to check (None = all)

    Returns:
        List of PIIMatch objects
    """
    matches = []

    types_to_check = pii_types or list(PII_PATTERNS.keys())

    for pii_type in types_to_check:
        if pii_type not in PII_PATTERNS:
            continue

        config = PII_PATTERNS[pii_type]
        pattern: str = str(config["pattern"])

        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group()

            # Additional validation if required
            if config.get("validate") == "luhn" and not luhn_validate(value):
                continue

            sev: PIISeverity = config["severity"]  # type: ignore[assignment]
            matches.append(
                PIIMatch(
                    pii_type=pii_type,
                    value_masked=mask_pii(value, pii_type),
                    location=f"{field_name}:{match.start()}-{match.end()}",
                    severity=sev,
                )
            )

    return matches


def audit_data_privacy(
    data: dict[str, Any],
    privacy_rules: dict[str, Any] | None = None,
    consent_records: dict[str, bool] | None = None,
    data_age_days: int | None = None,
    is_encrypted: bool = True,
) -> PrivacyAuditResult:
    """
    Perform complete privacy audit on data.

    Args:
        data: Data to audit (dict with field names → values)
        privacy_rules: Privacy rules config (None = use defaults)
        consent_records: Consent status for PII types
        data_age_days: Age of data in days (for retention check)
        is_encrypted: Whether data storage is encrypted

    Returns:
        PrivacyAuditResult with findings
    """
    if privacy_rules is None:
        privacy_rules = {
            "require_consent": ["ssn", "credit_card"],
            "retention_limits": {"pii_max_days": 365, "logs_max_days": 90},
        }

    all_pii = []
    violations = []
    recommendations = []

    # Scan each field for PII
    for field_name, value in data.items():
        if isinstance(value, str):
            pii_matches = detect_pii(value, field_name)
            all_pii.extend(pii_matches)
        elif isinstance(value, list | tuple):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    pii_matches = detect_pii(item, f"{field_name}[{i}]")
                    all_pii.extend(pii_matches)

    # Check for consent violations
    if consent_records is not None:
        require_consent = privacy_rules.get("require_consent", [])

        for pii in all_pii:
            if pii.pii_type in require_consent and not consent_records.get(pii.pii_type, False):
                violations.append(
                    PrivacyViolation(
                        violation_type=ViolationType.CONSENT_MISSING,
                        severity=PIISeverity.HIGH,
                        details=f"No consent for {pii.pii_type} at {pii.location}",
                        remediation=f"Obtain explicit consent for {pii.pii_type} collection",
                    )
                )

    # Check retention policy
    retention_limits = privacy_rules.get("retention_limits", {})
    if data_age_days is not None:
        max_days = retention_limits.get("pii_max_days", 365)

        if data_age_days > max_days and len(all_pii) > 0:
            violations.append(
                PrivacyViolation(
                    violation_type=ViolationType.RETENTION_EXCEEDED,
                    severity=PIISeverity.HIGH,
                    details=f"Data age ({data_age_days} days) exceeds limit ({max_days} days)",
                    remediation="Delete or anonymize PII data beyond retention period",
                )
            )

    # Check encryption
    if not is_encrypted and len(all_pii) > 0:
        # More severe for critical PII
        has_critical = any(p.severity == PIISeverity.CRITICAL for p in all_pii)

        violations.append(
            PrivacyViolation(
                violation_type=ViolationType.ENCRYPTION_MISSING,
                severity=PIISeverity.CRITICAL if has_critical else PIISeverity.HIGH,
                details="PII stored without encryption",
                remediation="Enable encryption at rest for all PII data",
            )
        )

    # Calculate privacy score
    base_score = 1.0

    # Deductions for PII found (not violations, just exposure risk)
    for pii in all_pii:
        if pii.severity == PIISeverity.CRITICAL:
            base_score -= 0.1
        elif pii.severity == PIISeverity.HIGH:
            base_score -= 0.05
        elif pii.severity == PIISeverity.MODERATE:
            base_score -= 0.02
        else:
            base_score -= 0.01

    # Deductions for violations
    for violation in violations:
        if violation.severity == PIISeverity.CRITICAL:
            base_score -= 0.2
        elif violation.severity == PIISeverity.HIGH:
            base_score -= 0.15
        else:
            base_score -= 0.1

    privacy_score = max(0.0, min(1.0, base_score))

    # Generate recommendations
    if any(p.severity == PIISeverity.CRITICAL for p in all_pii):
        recommendations.append("Audit access controls for critical PII (SSN, credit cards)")

    if not is_encrypted:
        recommendations.append("Implement encryption at rest for all data stores")

    if len([v for v in violations if v.violation_type == ViolationType.CONSENT_MISSING]) > 0:
        recommendations.append("Review consent collection process")

    if len(all_pii) > 10:
        recommendations.append("Consider data minimization - reduce PII collection")

    return PrivacyAuditResult(
        pii_found=all_pii, violations=violations, privacy_score=privacy_score, recommendations=recommendations
    )


def generate_privacy_report(audit_result: PrivacyAuditResult) -> dict[str, Any]:
    """
    Generate privacy audit report.

    Args:
        audit_result: Privacy audit result

    Returns:
        Report dictionary
    """
    pii_by_type: dict[str, int] = {}
    for pii in audit_result.pii_found:
        pii_by_type[pii.pii_type] = pii_by_type.get(pii.pii_type, 0) + 1

    violations_by_type: dict[str, int] = {}
    for v in audit_result.violations:
        key = v.violation_type.value
        violations_by_type[key] = violations_by_type.get(key, 0) + 1

    # Determine overall status
    if audit_result.privacy_score >= 0.9:
        status = "COMPLIANT"
    elif audit_result.privacy_score >= 0.7:
        status = "NEEDS_ATTENTION"
    elif audit_result.privacy_score >= 0.5:
        status = "AT_RISK"
    else:
        status = "NON_COMPLIANT"

    return {
        "status": status,
        "privacy_score": audit_result.privacy_score,
        "pii_summary": {"total_found": len(audit_result.pii_found), "by_type": pii_by_type},
        "violations_summary": {"total": len(audit_result.violations), "by_type": violations_by_type},
        "recommendations": audit_result.recommendations,
        "details": {
            "pii_locations": [
                {"type": p.pii_type, "location": p.location, "severity": p.severity.value}
                for p in audit_result.pii_found
            ],
            "violations": [
                {"type": v.violation_type.value, "severity": v.severity.value, "details": v.details}
                for v in audit_result.violations
            ],
        },
    }


if __name__ == "__main__":
    # Example: privacy audit
    test_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "555-123-4567",
        "ssn": "123-45-6789",
        "notes": "Customer called about card ending in 1234. IP: 192.168.1.100",
    }

    result = audit_data_privacy(
        data=test_data, consent_records={"email": True, "ssn": False}, data_age_days=400, is_encrypted=False
    )

    print("Privacy Audit Result:")
    print(f"  Privacy Score: {result.privacy_score:.2f}")
    print(f"  PII Found: {len(result.pii_found)}")
    for pii in result.pii_found:
        print(f"    - {pii.pii_type}: {pii.value_masked} ({pii.severity.value})")

    print(f"  Violations: {len(result.violations)}")
    for v in result.violations:
        print(f"    - {v.violation_type.value}: {v.details}")

    print("  Recommendations:")
    for rec in result.recommendations:
        print(f"    - {rec}")
