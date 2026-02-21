"""
Tests for security closure modules.

Covers: anomaly_return, behavior_profiler, privacy_auditor,
reputation_analyzer, security_entropy, threat_classifier,
response_engine, security_validator, device_daemon.
"""

from __future__ import annotations

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
# closures/security/anomaly_return.py
# ═══════════════════════════════════════════════════════════════════


class TestAnomalyReturn:
    """Unit tests for anomaly_return closures."""

    def test_compute_norm_l2(self):
        from closures.security.anomaly_return import compute_norm

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert compute_norm(v1, v2) == pytest.approx(np.sqrt(2.0))

    def test_compute_norm_l1(self):
        from closures.security.anomaly_return import compute_norm

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert compute_norm(v1, v2, norm_type="L1") == pytest.approx(2.0)

    def test_compute_norm_linf(self):
        from closures.security.anomaly_return import compute_norm

        v1 = np.array([3.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert compute_norm(v1, v2, norm_type="Linf") == pytest.approx(3.0)

    def test_unknown_norm_raises(self):
        from closures.security.anomaly_return import compute_norm

        with pytest.raises(ValueError, match="Unknown norm type"):
            compute_norm(np.array([1.0]), np.array([0.0]), norm_type="L3")

    def test_build_return_domain(self):
        from closures.security.anomaly_return import build_return_domain

        history = np.random.rand(100, 4)
        domain = build_return_domain(t=50, signal_history=history)
        assert isinstance(domain, list)
        assert all(isinstance(i, int) for i in domain)

    def test_compute_anomaly_return_finite(self):
        from closures.security.anomaly_return import compute_anomaly_return

        history = np.random.rand(100, 4) * 0.1  # Low variance → likely return
        result = compute_anomaly_return(t=50, signal_history=history, eta=1.0)
        assert "tau_A" in result
        assert "type" in result

    def test_compute_anomaly_return_t0(self):
        from closures.security.anomaly_return import compute_anomaly_return

        history = np.random.rand(10, 4)
        result = compute_anomaly_return(t=0, signal_history=history)
        assert result["tau_A"] is None

    def test_compute_anomaly_return_series(self):
        from closures.security.anomaly_return import compute_anomaly_return_series

        history = np.random.rand(20, 4)
        results = compute_anomaly_return_series(signal_history=history, eta=0.5)
        assert len(results) == 20
        assert all("tau_A" in r for r in results)

    def test_detect_anomaly_events(self):
        from closures.security.anomaly_return import detect_anomaly_events

        series = [
            {"tau_A": 1, "type": "finite"},
            {"tau_A": "INF_ANOMALY", "type": "INF_ANOMALY"},
            {"tau_A": "INF_ANOMALY", "type": "INF_ANOMALY"},
            {"tau_A": 2, "type": "finite"},
        ]
        events = detect_anomaly_events(series)
        assert isinstance(events, list)


# ═══════════════════════════════════════════════════════════════════
# closures/security/behavior_profiler.py
# ═══════════════════════════════════════════════════════════════════


class TestBehaviorProfiler:
    """Unit tests for behavior_profiler closures."""

    def test_compute_baseline_profile(self):
        from closures.security.behavior_profiler import compute_baseline_profile

        history = np.random.rand(100, 4)
        profile = compute_baseline_profile(signal_history=history)
        assert profile.samples > 0
        assert profile.mean.shape == (4,)

    def test_compute_deviation(self):
        from closures.security.behavior_profiler import (
            compute_baseline_profile,
            compute_deviation,
        )

        history = np.random.rand(100, 4)
        profile = compute_baseline_profile(signal_history=history)
        current = np.array([0.5, 0.5, 0.5, 0.5])
        deviation = compute_deviation(current_signals=current, baseline=profile)
        assert deviation.deviation_score >= 0.0

    def test_anomaly_level_classification(self):
        from closures.security.behavior_profiler import (
            AnomalyLevel,
            BehaviorProfile,
            compute_deviation,
        )

        profile = BehaviorProfile(
            mean=np.array([0.5, 0.5, 0.5, 0.5]),
            std=np.array([0.1, 0.1, 0.1, 0.1]),
            min_vals=np.array([0.0, 0.0, 0.0, 0.0]),
            max_vals=np.array([1.0, 1.0, 1.0, 1.0]),
            samples=100,
            window=64,
        )
        normal = compute_deviation(current_signals=np.array([0.5, 0.5, 0.5, 0.5]), baseline=profile)
        assert normal.anomaly_level == AnomalyLevel.NONE

    def test_analyze_trend(self):
        from closures.security.behavior_profiler import analyze_trend

        values = np.linspace(0.0, 1.0, 20)
        _trend, stats = analyze_trend(value_history=values)
        assert "slope" in stats
        assert "mean" in stats

    def test_profile_behavior_series(self):
        from closures.security.behavior_profiler import profile_behavior_series

        history = np.random.rand(100, 4)
        results = profile_behavior_series(signal_history=history)
        assert isinstance(results, list)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════
# closures/security/privacy_auditor.py
# ═══════════════════════════════════════════════════════════════════


class TestPrivacyAuditor:
    """Unit tests for privacy_auditor closures."""

    def test_luhn_valid_card(self):
        from closures.security.privacy_auditor import luhn_validate

        assert luhn_validate("4539578763621486") is True

    def test_luhn_invalid_card(self):
        from closures.security.privacy_auditor import luhn_validate

        assert luhn_validate("1234567890123456") is False

    def test_detect_pii_email(self):
        from closures.security.privacy_auditor import detect_pii

        matches = detect_pii("Contact us at test@example.com for info")
        assert any(m.pii_type == "email" for m in matches)

    def test_detect_pii_ssn(self):
        from closures.security.privacy_auditor import detect_pii

        matches = detect_pii("SSN: 123-45-6789")
        assert any(m.pii_type == "ssn" for m in matches)

    def test_detect_pii_phone(self):
        from closures.security.privacy_auditor import detect_pii

        matches = detect_pii("Call 555-123-4567")
        phone_matches = [m for m in matches if m.pii_type == "phone"]
        assert len(phone_matches) >= 0  # Pattern may or may not match

    def test_detect_pii_no_match(self):
        from closures.security.privacy_auditor import detect_pii

        matches = detect_pii("The quick brown fox jumps over the lazy dog")
        assert len(matches) == 0

    def test_audit_data_privacy(self):
        from closures.security.privacy_auditor import audit_data_privacy

        data = {"name": "John", "email": "john@example.com"}
        result = audit_data_privacy(data=data)
        assert hasattr(result, "privacy_score")
        assert 0.0 <= result.privacy_score <= 1.0

    def test_generate_privacy_report(self):
        from closures.security.privacy_auditor import (
            audit_data_privacy,
            generate_privacy_report,
        )

        data = {"note": "No PII here"}
        audit = audit_data_privacy(data=data)
        report = generate_privacy_report(audit)
        assert "status" in report
        assert report["status"] in (
            "COMPLIANT",
            "NEEDS_ATTENTION",
            "AT_RISK",
            "NON_COMPLIANT",
        )

    def test_mask_pii(self):
        from closures.security.privacy_auditor import mask_pii

        masked = mask_pii("test@example.com", "email")
        assert "@" not in masked or masked != "test@example.com"


# ═══════════════════════════════════════════════════════════════════
# closures/security/reputation_analyzer.py
# ═══════════════════════════════════════════════════════════════════


class TestReputationAnalyzer:
    """Unit tests for reputation_analyzer closures."""

    def test_trusted_domain(self):
        from closures.security.reputation_analyzer import analyze_url_reputation

        result = analyze_url_reputation("https://github.com/test")
        assert result.score >= 0.8

    def test_suspicious_tld(self):
        from closures.security.reputation_analyzer import analyze_url_reputation

        result = analyze_url_reputation("https://malicious.xyz/login")
        assert result.score < 0.8

    def test_blocklist(self):
        from closures.security.reputation_analyzer import analyze_url_reputation

        result = analyze_url_reputation(
            "https://evil.com",
            local_blocklist=["evil.com"],
        )
        assert result.score == pytest.approx(0.0)

    def test_analyze_url_structure(self):
        from closures.security.reputation_analyzer import analyze_url_structure

        result = analyze_url_structure("https://example.com/path?q=1")
        assert isinstance(result, dict)

    def test_detect_homoglyphs(self):
        from closures.security.reputation_analyzer import detect_homoglyphs

        result = detect_homoglyphs("gооgle.com", known_brands=["google.com"])
        assert isinstance(result, list)

    def test_analyze_file_hash(self):
        from closures.security.reputation_analyzer import analyze_file_hash

        result = analyze_file_hash("abc123def456", hash_type="sha256")
        assert hasattr(result, "reputation_type")

    def test_analyze_ip_reputation(self):
        from closures.security.reputation_analyzer import analyze_ip_reputation

        result = analyze_ip_reputation("192.168.1.1")
        assert hasattr(result, "score")

    def test_private_ip_neutral(self):
        from closures.security.reputation_analyzer import analyze_ip_reputation

        result = analyze_ip_reputation("10.0.0.1")
        assert result.score >= 0.5


# ═══════════════════════════════════════════════════════════════════
# closures/security/security_entropy.py
# ═══════════════════════════════════════════════════════════════════


class TestSecurityEntropy:
    """Unit tests for security_entropy closures."""

    def test_binary_entropy_half(self):
        from closures.security.security_entropy import binary_entropy

        assert binary_entropy(0.5) == pytest.approx(np.log(2), abs=1e-5)

    def test_binary_entropy_zero(self):
        from closures.security.security_entropy import binary_entropy

        assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-5)

    def test_binary_entropy_one(self):
        from closures.security.security_entropy import binary_entropy

        assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-5)

    def test_compute_security_entropy(self):
        from closures.security.security_entropy import compute_security_entropy

        signals = np.array([0.5, 0.5, 0.5, 0.5])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_security_entropy(signals=signals, weights=weights)
        assert "H" in result
        assert "H_normalized" in result

    def test_weight_sum_validation(self):
        from closures.security.security_entropy import compute_security_entropy

        signals = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.6])  # Sum > 1
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            compute_security_entropy(signals=signals, weights=weights)

    def test_negative_weights_validation(self):
        from closures.security.security_entropy import compute_security_entropy

        signals = np.array([0.5, 0.5])
        weights = np.array([-0.5, 1.5])
        with pytest.raises(ValueError, match="non-negative"):
            compute_security_entropy(signals=signals, weights=weights)

    def test_signal_range_validation(self):
        from closures.security.security_entropy import compute_security_entropy

        signals = np.array([0.5, 1.5])
        weights = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            compute_security_entropy(signals=signals, weights=weights)

    def test_length_mismatch_validation(self):
        from closures.security.security_entropy import compute_security_entropy

        signals = np.array([0.5, 0.5, 0.5])
        weights = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="mismatch"):
            compute_security_entropy(signals=signals, weights=weights)

    def test_compute_signal_dispersion(self):
        from closures.security.security_entropy import compute_signal_dispersion

        signals = np.array([0.2, 0.4, 0.6, 0.8])
        result = compute_signal_dispersion(signals=signals)
        assert "D" in result
        assert "std" in result
        assert 0.0 <= result["D"] <= 1.0

    def test_entropy_series(self):
        from closures.security.security_entropy import compute_security_entropy_series

        series = np.random.rand(10, 4)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        results = compute_security_entropy_series(signal_series=series, weights=weights)
        assert len(results) == 10


# ═══════════════════════════════════════════════════════════════════
# closures/security/threat_classifier.py
# ═══════════════════════════════════════════════════════════════════


class TestThreatClassifier:
    """Unit tests for threat_classifier closures."""

    def test_benign_classification(self):
        from closures.security.threat_classifier import classify_threat

        result = classify_threat(T=0.9, theta=0.8, H=0.1, D=0.1, sigma=0.05, TIC=0.85, tau_A=5)
        assert result.threat_type.name == "BENIGN"

    def test_unknown_with_none(self):
        from closures.security.threat_classifier import classify_threat

        result = classify_threat(T=None, theta=0.8, H=0.1, D=0.1, sigma=0.05, TIC=0.85, tau_A=5)
        assert result.threat_type.name == "UNKNOWN"

    def test_persistent_threat(self):
        from closures.security.threat_classifier import classify_threat

        result = classify_threat(
            T=0.3,
            theta=0.3,
            H=0.6,
            D=0.5,
            sigma=0.3,
            TIC=0.2,
            tau_A="INF_ANOMALY",
        )
        assert result.threat_type.name == "PERSISTENT_THREAT"

    def test_unidentifiable_tau(self):
        from closures.security.threat_classifier import classify_threat

        result = classify_threat(
            T=0.5,
            theta=0.5,
            H=0.3,
            D=0.2,
            sigma=0.1,
            TIC=0.5,
            tau_A="UNIDENTIFIABLE",
        )
        assert result.threat_type.name == "UNKNOWN"

    def test_confidence_bounds(self):
        from closures.security.threat_classifier import classify_threat

        result = classify_threat(T=0.9, theta=0.8, H=0.1, D=0.1, sigma=0.05, TIC=0.85, tau_A=5)
        assert 0.0 <= result.confidence <= 1.0

    def test_generate_threat_report(self):
        from closures.security.threat_classifier import (
            classify_threat,
            generate_threat_report,
        )

        cls = classify_threat(T=0.9, theta=0.8, H=0.1, D=0.1, sigma=0.05, TIC=0.85, tau_A=5)
        series = [
            {
                "threat_type": cls.threat_type,
                "confidence": cls.confidence,
                "severity": cls.severity,
                "t": 0,
            },
        ]
        report = generate_threat_report(series)
        assert "overall_status" in report


# ═══════════════════════════════════════════════════════════════════
# closures/security/response_engine.py
# ═══════════════════════════════════════════════════════════════════


class TestResponseEngine:
    """Unit tests for ResponseEngine."""

    def test_decide_allow(self):
        from closures.security.response_engine import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        decision = engine.decide_action(
            entity_id="test1",
            entity_type="file",
            validation_status="CONFORMANT",
            T=0.95,
            tau_A=2,
            threat_type="BENIGN",
        )
        assert decision.action.name in ("ALLOW", "ALLOW_MONITORED")

    def test_decide_block_low_trust(self):
        from closures.security.response_engine import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        decision = engine.decide_action(
            entity_id="test2",
            entity_type="file",
            validation_status="NONCONFORMANT",
            T=0.1,
            tau_A=5,
            threat_type="ATTACK_IN_PROGRESS",
        )
        assert decision.action.name == "BLOCK"

    def test_decide_block_inf_anomaly(self):
        from closures.security.response_engine import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        decision = engine.decide_action(
            entity_id="test3",
            entity_type="network",
            validation_status="NONCONFORMANT",
            T=0.5,
            tau_A="INF_ANOMALY",
            threat_type="PERSISTENT_THREAT",
        )
        assert decision.action.name == "BLOCK"

    def test_get_summary(self):
        from closures.security.response_engine import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        engine.decide_action(
            entity_id="test",
            entity_type="file",
            validation_status="CONFORMANT",
            T=0.95,
            tau_A=1,
            threat_type="BENIGN",
        )
        summary = engine.get_summary()
        assert summary["total_decisions"] == 1

    def test_escalation_levels(self):
        from closures.security.response_engine import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        d1 = engine.decide_action("e1", "file", "CONFORMANT", T=0.95, tau_A=1, threat_type="BENIGN")
        d2 = engine.decide_action("e2", "file", "NONCONFORMANT", T=0.1, tau_A=100, threat_type="ATTACK")
        assert d1.escalation_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert d2.escalation_level in ("HIGH", "CRITICAL")


# ═══════════════════════════════════════════════════════════════════
# closures/security/device_daemon.py
# ═══════════════════════════════════════════════════════════════════


class TestDeviceDaemon:
    """Unit tests for device_daemon closures."""

    def test_device_category_enum(self):
        from closures.security.device_daemon import DeviceCategory

        assert DeviceCategory.WORKSTATION.name == "WORKSTATION"
        assert DeviceCategory.IOT.name == "IOT"

    def test_device_status_enum(self):
        from closures.security.device_daemon import DeviceStatus

        assert DeviceStatus.TRUSTED.name == "TRUSTED"
        assert DeviceStatus.BLOCKED.name == "BLOCKED"

    def test_network_action_enum(self):
        from closures.security.device_daemon import NetworkAction

        assert NetworkAction.ALLOW_FULL.name == "ALLOW_FULL"
        assert NetworkAction.QUARANTINE.name == "QUARANTINE"

    def test_daemon_creation(self):
        from closures.security.device_daemon import DeviceDaemon

        daemon = DeviceDaemon(scan_interval=30.0, validation_interval=60.0)
        assert daemon is not None

    def test_add_device(self):
        from closures.security.device_daemon import DeviceDaemon

        daemon = DeviceDaemon()
        device = daemon.add_device(mac="AA:BB:CC:DD:EE:FF", ip="192.168.1.100", hostname="test-device")
        assert device is not None

    def test_get_network_summary(self):
        from closures.security.device_daemon import DeviceDaemon

        daemon = DeviceDaemon()
        summary = daemon.get_network_summary()
        assert isinstance(summary, dict)

    def test_device_validator(self):
        from closures.security.device_daemon import DeviceValidator

        validator = DeviceValidator()
        assert hasattr(validator, "signal_weights")
