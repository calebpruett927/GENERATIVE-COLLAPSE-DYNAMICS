"""
UMCP Security Response Engine

Automated response system that takes actions based on UMCP validation results.
This demonstrates how UMCP would respond to threats - not through signatures,
but through validation and return-based trust.

Core Principle:
    "What Survives Validation Is Trusted"

    Response is based on:
    1. Trust Fidelity (T): How much do we trust this entity?
    2. Anomaly Return (τ_A): Does it return to baseline?
    3. Trust Seam: Is trust continuity maintained?

    NOT based on:
    - Single signature match
    - Heuristic rules
    - Reputation alone

Response Levels:
    ALLOW: T ≥ 0.8, τ_A finite → Full access
    MONITOR: 0.4 ≤ T < 0.8 → Allow with enhanced logging
    QUARANTINE: T < 0.4, τ_A finite → Isolate, await recovery
    BLOCK: τ_A = INF_ANOMALY → No trust credit, hard block
"""

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ResponseAction(Enum):
    """Response actions the engine can take."""

    ALLOW = "ALLOW"
    ALLOW_MONITORED = "ALLOW_MONITORED"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"
    TERMINATE = "TERMINATE"
    ALERT = "ALERT"


@dataclass
class ResponseDecision:
    """Decision made by response engine."""

    action: ResponseAction
    entity_id: str
    entity_type: str
    validation_status: str
    T: float
    tau_A: Any
    reason: str
    timestamp: datetime
    escalation_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class ResponseEngine:
    """
    Automated response engine for UMCP security.

    Takes actions based on UMCP validation, not signatures.
    """

    def __init__(self, quarantine_dir: str = "/tmp/umcp_quarantine", dry_run: bool = False):
        """
        Initialize response engine.

        Args:
            quarantine_dir: Directory for quarantined files
            dry_run: If True, log actions but don't execute
        """
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.dry_run = dry_run

        self.logger = logging.getLogger("ResponseEngine")
        self.logger.setLevel(logging.INFO)

        # Response history
        self.decisions: list[ResponseDecision] = []

        # Escalation thresholds
        self.escalation = {
            "CRITICAL": {"T_max": 0.2, "tau_A": "INF_ANOMALY"},
            "HIGH": {"T_max": 0.4, "tau_A_min": 32},
            "MEDIUM": {"T_max": 0.6},
            "LOW": {},
        }

    def decide_action(
        self, entity_id: str, entity_type: str, validation_status: str, T: float, tau_A: Any, threat_type: str
    ) -> ResponseDecision:
        """
        Decide response action based on UMCP validation.

        UMCP Logic:
        1. τ_A = INF_ANOMALY → BLOCK (no return = no trust)
        2. T < 0.2 → BLOCK (critically low trust)
        3. T < 0.4 → QUARANTINE (low trust but may recover)
        4. 0.4 ≤ T < 0.8 → MONITOR (medium trust)
        5. T ≥ 0.8 and τ_A finite → ALLOW (trusted)

        Args:
            entity_id: Entity identifier
            entity_type: file, network, process, url
            validation_status: TRUSTED, SUSPICIOUS, BLOCKED, NON_EVALUABLE
            T: Trust Fidelity (0-1)
            tau_A: Anomaly Return Time (int or "INF_ANOMALY")
            threat_type: Threat classification

        Returns:
            ResponseDecision with action and details
        """
        # Determine escalation level
        escalation_level = self._determine_escalation(T, tau_A)

        # Decide action based on UMCP invariants
        if tau_A == "INF_ANOMALY":
            # No return through collapse = not real/trusted
            action = ResponseAction.BLOCK
            reason = "No return to baseline (τ_A = ∞) - trust not earned"

        elif T < 0.2:
            # Critically low trust
            action = ResponseAction.BLOCK
            reason = f"Critically low trust (T={T:.3f} < 0.2)"

        elif T < 0.4:
            # Low trust, but might recover
            action = ResponseAction.QUARANTINE
            reason = f"Low trust (T={T:.3f}) - quarantine pending recovery"

        elif T < 0.8:
            # Medium trust
            if tau_A and isinstance(tau_A, int) and tau_A > 32:
                # Slow return
                action = ResponseAction.ALLOW_MONITORED
                reason = f"Medium trust (T={T:.3f}) with slow return (τ_A={tau_A})"
            else:
                action = ResponseAction.ALLOW_MONITORED
                reason = f"Medium trust (T={T:.3f}) - allow with monitoring"

        else:
            # High trust (T ≥ 0.8)
            if tau_A is None or (isinstance(tau_A, int) and tau_A < 16):
                # Fast return or no anomaly
                action = ResponseAction.ALLOW
                reason = f"High trust (T={T:.3f}) with finite return - TRUSTED"
            else:
                action = ResponseAction.ALLOW_MONITORED
                reason = f"High trust (T={T:.3f}) but monitoring return"

        decision = ResponseDecision(
            action=action,
            entity_id=entity_id,
            entity_type=entity_type,
            validation_status=validation_status,
            T=T,
            tau_A=tau_A,
            reason=reason,
            timestamp=datetime.utcnow(),
            escalation_level=escalation_level,
        )

        self.decisions.append(decision)

        return decision

    def execute_decision(self, decision: ResponseDecision) -> bool:
        """
        Execute the response decision.

        Args:
            decision: ResponseDecision to execute

        Returns:
            True if successful
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute: {decision.action.value} on {decision.entity_id}")
            return True

        try:
            if decision.action == ResponseAction.ALLOW:
                return self._allow(decision)

            elif decision.action == ResponseAction.ALLOW_MONITORED:
                return self._allow_monitored(decision)

            elif decision.action == ResponseAction.QUARANTINE:
                return self._quarantine(decision)

            elif decision.action == ResponseAction.BLOCK:
                return self._block(decision)

            elif decision.action == ResponseAction.TERMINATE:
                return self._terminate(decision)

            elif decision.action == ResponseAction.ALERT:
                return self._alert(decision)

            else:
                self.logger.error(f"Unknown action: {decision.action}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing {decision.action}: {e}")
            return False

    def _allow(self, decision: ResponseDecision) -> bool:
        """Allow entity - no restrictions."""
        self.logger.info(f"ALLOW: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A})")
        return True

    def _allow_monitored(self, decision: ResponseDecision) -> bool:
        """Allow but with enhanced monitoring."""
        self.logger.warning(
            f"ALLOW_MONITORED: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A}) - {decision.reason}"
        )

        # In production: enable enhanced logging, rate limiting, etc.
        # For now, just log
        return True

    def _quarantine(self, decision: ResponseDecision) -> bool:
        """Quarantine entity."""
        self.logger.warning(f"QUARANTINE: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A})")

        if decision.entity_type == "file":
            # Move file to quarantine
            file_path = decision.entity_id.replace("file:", "")
            if os.path.exists(file_path):
                quarantine_path = self.quarantine_dir / f"{Path(file_path).name}.quarantine"
                shutil.move(file_path, quarantine_path)
                self.logger.info(f"File moved to quarantine: {quarantine_path}")
                return True

        elif decision.entity_type == "network":
            # Block network connection
            # In production: use iptables/nftables
            self.logger.info(f"Would block network connection: {decision.entity_id}")
            return True

        elif decision.entity_type == "process":
            # Suspend process
            # In production: use cgroups to pause process
            self.logger.info(f"Would suspend process: {decision.entity_id}")
            return True

        return True

    def _block(self, decision: ResponseDecision) -> bool:
        """Hard block entity."""
        self.logger.error(f"BLOCK: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A}) - {decision.reason}")

        if decision.entity_type == "file":
            # Delete or secure-wipe file
            file_path = decision.entity_id.replace("file:", "")
            if os.path.exists(file_path):
                # Move to quarantine first (don't delete immediately)
                return self._quarantine(decision)

        elif decision.entity_type == "network":
            # Block network with firewall
            # In production: iptables -A INPUT -s <ip> -j DROP
            self.logger.error(f"Would block network: {decision.entity_id}")
            return True

        elif decision.entity_type == "process":
            # Kill process
            return self._terminate(decision)

        return True

    def _terminate(self, decision: ResponseDecision) -> bool:
        """Terminate process."""
        self.logger.error(f"TERMINATE: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A})")

        if decision.entity_type == "process":
            # Extract PID
            try:
                pid = int(decision.entity_id.split(":")[-1])
                # In production: os.kill(pid, signal.SIGKILL)
                self.logger.error(f"Would kill process PID {pid}")
                return True
            except (ValueError, IndexError):
                self.logger.error(f"Could not extract PID from {decision.entity_id}")
                return False

        return True

    def _alert(self, decision: ResponseDecision) -> bool:
        """Send alert to security team."""
        self.logger.critical(
            f"ALERT: {decision.entity_id} (T={decision.T:.3f}, τ_A={decision.tau_A}) - {decision.reason}"
        )

        # In production: send email, webhook, SIEM alert, etc.
        return True

    def _determine_escalation(self, T: float, tau_A: Any) -> str:
        """Determine escalation level based on UMCP invariants."""
        if tau_A == "INF_ANOMALY" or T < 0.2:
            return "CRITICAL"
        elif T < 0.4 or (isinstance(tau_A, int) and tau_A > 32):
            return "HIGH"
        elif T < 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    def get_summary(self) -> dict[str, Any]:
        """Get summary of response actions."""
        action_counts = {}
        for decision in self.decisions:
            action = decision.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        escalation_counts = {}
        for decision in self.decisions:
            level = decision.escalation_level
            escalation_counts[level] = escalation_counts.get(level, 0) + 1

        return {
            "total_decisions": len(self.decisions),
            "action_counts": action_counts,
            "escalation_counts": escalation_counts,
            "recent_critical": [
                {"entity": d.entity_id, "action": d.action.value, "T": d.T, "tau_A": d.tau_A, "reason": d.reason}
                for d in self.decisions[-10:]
                if d.escalation_level == "CRITICAL"
            ],
        }


if __name__ == "__main__":
    # Example usage
    engine = ResponseEngine(dry_run=True)

    print("=" * 60)
    print("UMCP Response Engine Examples")
    print("'What Survives Validation Is Trusted'")
    print("=" * 60)

    # Example 1: Trusted entity
    print("\n1. Trusted Entity:")
    decision = engine.decide_action(
        entity_id="file:/usr/bin/python3",
        entity_type="file",
        validation_status="TRUSTED",
        T=0.92,
        tau_A=2,
        threat_type="BENIGN",
    )
    print(f"   Action: {decision.action.value}")
    print(f"   Reason: {decision.reason}")
    engine.execute_decision(decision)

    # Example 2: Suspicious entity (medium trust)
    print("\n2. Suspicious Entity:")
    decision = engine.decide_action(
        entity_id="file:/tmp/unknown.exe",
        entity_type="file",
        validation_status="SUSPICIOUS",
        T=0.55,
        tau_A=12,
        threat_type="TRANSIENT_ANOMALY",
    )
    print(f"   Action: {decision.action.value}")
    print(f"   Reason: {decision.reason}")
    engine.execute_decision(decision)

    # Example 3: Blocked entity (no return)
    print("\n3. Blocked Entity (No Return):")
    decision = engine.decide_action(
        entity_id="file:/tmp/malware.bin",
        entity_type="file",
        validation_status="BLOCKED",
        T=0.15,
        tau_A="INF_ANOMALY",
        threat_type="PERSISTENT_THREAT",
    )
    print(f"   Action: {decision.action.value}")
    print(f"   Reason: {decision.reason}")
    engine.execute_decision(decision)

    print("\n" + "=" * 60)
    print("Summary:")
    summary = engine.get_summary()
    print(f"  Total Decisions: {summary['total_decisions']}")
    print(f"  Actions: {summary['action_counts']}")
    print(f"  Escalations: {summary['escalation_counts']}")
