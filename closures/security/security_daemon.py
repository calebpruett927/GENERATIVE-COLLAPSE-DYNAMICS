#!/usr/bin/env python3
"""
UMCP Security Daemon - Background Antivirus Service

Runs continuously in the background, monitoring system activity and
validating security in real-time using UMCP principles.

Core Principle:
    "What Survives Validation Is Trusted"

    Instead of blocking/allowing based on signatures alone,
    the daemon validates entities through collapse-return cycles.
    Only entities that return to baseline (τ_A finite) are trusted.

Architecture:
    1. Signal Collectors: Monitor file, network, process activity
    2. Validation Engine: Continuous UMCP validation
    3. Action Handler: Respond based on validation status
    4. Ledger Logger: Maintain trust ledger for all entities

Usage:
    # Run as daemon
    sudo python security_daemon.py start

    # Stop daemon
    sudo python security_daemon.py stop

    # Check status
    python security_daemon.py status
"""

import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# UMCP Security imports
from closures.security.security_validator import SecurityValidator, ValidationResult
from closures.security.trust_integrity import compute_trust_seam


@dataclass
class MonitoredEntity:
    """Entity being monitored by the daemon."""

    entity_id: str
    entity_type: str  # file, url, process, network
    first_seen: datetime
    last_seen: datetime
    signal_history: list[np.ndarray]
    validation_history: list[ValidationResult]
    current_status: str  # TRUSTED, SUSPICIOUS, BLOCKED, NON_EVALUABLE
    tau_A_history: list[Any]
    trust_seam_ledger: list[float]  # Running ledger of σ changes


@dataclass
class SecurityEvent:
    """Security event detected by the daemon."""

    timestamp: datetime
    event_type: str  # file_access, network_connection, process_start, etc.
    entity_id: str
    entity_type: str
    signals: np.ndarray
    action_taken: str
    validation_status: str
    reason: str


class SignalCollector:
    """Collects security signals from system monitoring."""

    def __init__(self):
        self.file_access_count = {}
        self.network_connections = {}
        self.process_info = {}

    def collect_file_signals(self, file_path: str) -> np.ndarray:
        """
        Collect security signals for a file.

        Returns: [integrity_score, reputation_score, behavior_score, identity_score]
        """
        signals = []

        # 1. Integrity score (file hash check)
        try:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    hashlib.sha256(f.read()).hexdigest()
                    # Check against known good/bad hashes
                    # For demo: score based on file size (real would use hash DB)
                    file_size = os.path.getsize(file_path)
                    integrity_score = 0.9 if file_size < 10_000_000 else 0.7
            else:
                integrity_score = 0.0
        except (OSError, PermissionError):
            integrity_score = 0.0

        signals.append(integrity_score)

        # 2. Reputation score (file origin, signer, etc.)
        # For demo: check file extension
        ext = Path(file_path).suffix.lower()
        suspicious_exts = [".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js"]
        reputation_score = 0.5 if ext in suspicious_exts else 0.8
        signals.append(reputation_score)

        # 3. Behavior score (access patterns)
        access_count = self.file_access_count.get(file_path, 0)
        self.file_access_count[file_path] = access_count + 1
        # Rapid access is suspicious
        behavior_score = 0.9 if access_count < 10 else max(0.3, 0.9 - access_count * 0.05)
        signals.append(behavior_score)

        # 4. Identity score (caller identity, permissions)
        # For demo: check file permissions
        try:
            stat = os.stat(file_path)
            # Executable with write = suspicious
            is_executable = bool(stat.st_mode & 0o111)
            is_writable = bool(stat.st_mode & 0o222)
            identity_score = 0.6 if (is_executable and is_writable) else 0.9
        except (OSError, PermissionError):
            identity_score = 0.5

        signals.append(identity_score)

        return np.clip(np.array(signals), 0, 1)

    def collect_network_signals(self, connection: dict[str, Any]) -> np.ndarray:
        """
        Collect security signals for network connection.

        Returns: [integrity_score, reputation_score, behavior_score, identity_score]
        """
        signals = []

        # 1. Integrity score (TLS/cert validation)
        is_https = connection.get("protocol") == "https"
        signals.append(0.9 if is_https else 0.4)

        # 2. Reputation score (domain/IP reputation)
        domain = connection.get("domain", "")
        trusted = any(d in domain for d in ["google.com", "github.com", "microsoft.com"])
        suspicious = any(ext in domain for ext in [".xyz", ".click", ".tk"])

        if trusted:
            reputation = 0.95
        elif suspicious:
            reputation = 0.2
        else:
            reputation = 0.5

        signals.append(reputation)

        # 3. Behavior score (data transfer patterns)
        bytes_sent = connection.get("bytes_sent", 0)
        # Large uploads = potential exfiltration
        behavior = 0.9 if bytes_sent < 1_000_000 else max(0.3, 0.9 - bytes_sent / 10_000_000)
        signals.append(behavior)

        # 4. Identity score (process initiating connection)
        # For demo: trusted processes get high score
        process = connection.get("process", "unknown")
        trusted_processes = ["chrome", "firefox", "code", "python"]
        identity = 0.9 if any(p in process.lower() for p in trusted_processes) else 0.6
        signals.append(identity)

        return np.clip(np.array(signals), 0, 1)

    def collect_process_signals(self, process_info: dict[str, Any]) -> np.ndarray:
        """
        Collect security signals for a process.

        Returns: [integrity_score, reputation_score, behavior_score, identity_score]
        """
        signals = []

        # 1. Integrity score (binary hash, signature)
        has_signature = process_info.get("signed", False)
        signals.append(0.9 if has_signature else 0.5)

        # 2. Reputation score (known process, publisher)
        name = process_info.get("name", "")
        known_good = any(n in name.lower() for n in ["systemd", "kernel", "chrome", "code"])
        signals.append(0.95 if known_good else 0.5)

        # 3. Behavior score (CPU/memory usage, syscalls)
        cpu_percent = process_info.get("cpu_percent", 0)
        behavior = 0.9 if cpu_percent < 50 else max(0.3, 0.9 - cpu_percent / 100)
        signals.append(behavior)

        # 4. Identity score (user, privileges)
        is_root = process_info.get("user") == "root"
        needs_root = process_info.get("needs_root", False)
        identity = 0.7 if (is_root and not needs_root) else 0.9
        signals.append(identity)

        return np.clip(np.array(signals), 0, 1)


class SecurityDaemon:
    """
    UMCP Security Daemon - Background antivirus service.

    Continuously validates system activity using UMCP principles.
    """

    def __init__(
        self,
        log_dir: str = "/var/log/umcp_security",
        state_dir: str = "/var/lib/umcp_security",
        check_interval: float = 1.0,
    ):
        """
        Initialize security daemon.

        Args:
            log_dir: Directory for logs
            state_dir: Directory for state persistence
            check_interval: Seconds between validation cycles
        """
        self.log_dir = Path(log_dir)
        self.state_dir = Path(state_dir)
        self.check_interval = check_interval

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # UMCP validator
        self.validator = SecurityValidator()

        # Signal collector
        self.collector = SignalCollector()

        # Monitored entities
        self.entities: dict[str, MonitoredEntity] = {}

        # Event log
        self.events: deque = deque(maxlen=10000)

        # Trust ledger (entity_id → cumulative σ)
        self.trust_ledger: dict[str, float] = {}

        # Daemon state
        self.running = False
        self.threads: list[threading.Thread] = []

        # Statistics
        self.stats = {"validated": 0, "trusted": 0, "suspicious": 0, "blocked": 0, "non_evaluable": 0}

        # Daemon uptime tracking
        self.start_time: datetime = datetime.utcnow()

        self.logger.info("UMCP Security Daemon initialized")
        self.logger.info("Axiom: 'What Survives Validation Is Trusted'")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("UMCPSecurityDaemon")
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_dir / "security_daemon.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def start(self):
        """Start the security daemon."""
        if self.running:
            self.logger.warning("Daemon already running")
            return

        self.running = True
        self.logger.info("Starting UMCP Security Daemon")

        # Load state
        self._load_state()

        # Start monitoring threads
        self.threads = [
            threading.Thread(target=self._file_monitor, daemon=True),
            threading.Thread(target=self._network_monitor, daemon=True),
            threading.Thread(target=self._validation_loop, daemon=True),
            threading.Thread(target=self._ledger_writer, daemon=True),
        ]

        for thread in self.threads:
            thread.start()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Daemon started successfully")

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the daemon gracefully."""
        self.logger.info("Stopping UMCP Security Daemon")
        self.running = False

        # Save state
        self._save_state()

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)

        self.logger.info("Daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _file_monitor(self):
        """Monitor file system activity."""
        self.logger.info("File monitor started")

        # In production, this would use inotify/FSEvents
        # For demo, simulate file access
        watch_dirs = ["/tmp", "/home"]

        while self.running:
            try:
                # Simulate: check recent files
                for watch_dir in watch_dirs:
                    if not os.path.exists(watch_dir):
                        continue

                    # Check a few files (in production, triggered by inotify)
                    for root, _dirs, files in os.walk(watch_dir):
                        for file in files[:5]:  # Limit for demo
                            file_path = os.path.join(root, file)
                            self._validate_file(file_path)
                        break  # Don't recurse deeply in demo

                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"File monitor error: {e}")
                time.sleep(5)

    def _network_monitor(self):
        """Monitor network activity."""
        self.logger.info("Network monitor started")

        # In production, this would use netfilter/BPF
        # For demo, simulate connections

        while self.running:
            try:
                # Simulate network connections
                # In production: hook into netfilter, parse /proc/net/tcp, etc.
                time.sleep(self.check_interval * 2)

            except Exception as e:
                self.logger.error(f"Network monitor error: {e}")
                time.sleep(5)

    def _validation_loop(self):
        """Main validation loop."""
        self.logger.info("Validation loop started")

        while self.running:
            try:
                # Re-validate entities with history
                for _entity_id, entity in list(self.entities.items()):
                    if len(entity.signal_history) > 1:
                        self._revalidate_entity(entity)

                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Validation loop error: {e}")
                time.sleep(5)

    def _validate_file(self, file_path: str):
        """Validate a file through UMCP."""
        entity_id = f"file:{file_path}"

        # Collect signals
        try:
            signals = self.collector.collect_file_signals(file_path)
        except Exception as e:
            self.logger.error(f"Error collecting signals for {file_path}: {e}")
            return

        # Get or create entity
        entity = self.entities.get(entity_id)
        if entity is None:
            entity = MonitoredEntity(
                entity_id=entity_id,
                entity_type="file",
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                signal_history=[],
                validation_history=[],
                current_status="NON_EVALUABLE",
                tau_A_history=[],
                trust_seam_ledger=[0.0],
            )
            self.entities[entity_id] = entity

        entity.last_seen = datetime.utcnow()
        entity.signal_history.append(signals)

        # Keep history bounded
        if len(entity.signal_history) > 100:
            entity.signal_history = entity.signal_history[-100:]

        # Validate with history
        signal_history = np.array(entity.signal_history)
        result = self.validator.validate_signals(
            signals, signal_history=signal_history if len(signal_history) > 1 else None, entity_id=entity_id
        )

        entity.validation_history.append(result)
        entity.current_status = result.status
        entity.tau_A_history.append(result.invariants.tau_A)

        # Update trust ledger (seam accounting)
        if len(entity.validation_history) > 1:
            prev = entity.validation_history[-2]
            seam = compute_trust_seam(
                prev.invariants.sigma, result.invariants.sigma, prev.invariants.TIC, result.invariants.TIC
            )
            entity.trust_seam_ledger.append(entity.trust_seam_ledger[-1] + seam["delta_sigma_ledger"])

        # Take action based on validation
        action = self._take_action(entity, result)

        # Log event
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="file_access",
            entity_id=entity_id,
            entity_type="file",
            signals=signals,
            action_taken=action,
            validation_status=result.status,
            reason=result.threat_type,
        )
        self.events.append(event)

        # Update stats
        self.stats["validated"] += 1
        self.stats[result.status.lower()] = self.stats.get(result.status.lower(), 0) + 1

        # Log if not trusted
        if result.status != "TRUSTED":
            self.logger.warning(
                f"File: {file_path} - Status: {result.status} "
                f"(T={result.invariants.T:.3f}, τ_A={result.invariants.tau_A}) "
                f"- Action: {action}"
            )

    def _revalidate_entity(self, entity: MonitoredEntity):
        """Re-validate entity with accumulated history."""
        if len(entity.signal_history) < 2:
            return

        # Most recent signals
        signals = entity.signal_history[-1]
        signal_history = np.array(entity.signal_history)

        # Validate
        result = self.validator.validate_signals(signals, signal_history=signal_history, entity_id=entity.entity_id)

        # Check if status changed
        if result.status != entity.current_status:
            self.logger.info(f"Status change for {entity.entity_id}: {entity.current_status} → {result.status}")

            entity.current_status = result.status

            # Re-take action if needed
            action = self._take_action(entity, result)

            # Log event
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                event_type="revalidation",
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                signals=signals,
                action_taken=action,
                validation_status=result.status,
                reason=f"Status changed to {result.status}",
            )
            self.events.append(event)

    def _take_action(self, entity: MonitoredEntity, result: ValidationResult) -> str:
        """
        Take action based on validation result.

        UMCP Principle: "What Survives Validation Is Trusted"
        - TRUSTED: Allow, no restrictions
        - SUSPICIOUS: Allow but log/monitor
        - BLOCKED: Quarantine/block
        - NON_EVALUABLE: Allow but track

        Returns: Action taken
        """
        if result.status == "TRUSTED":
            # Entity passed validation and returned to baseline
            return "ALLOW"

        elif result.status == "SUSPICIOUS":
            # Low trust or slow return
            self.logger.warning(
                f"SUSPICIOUS: {entity.entity_id} (T={result.invariants.T:.3f}, τ_A={result.invariants.tau_A})"
            )
            return "ALLOW_MONITORED"

        elif result.status == "BLOCKED":
            # Failed validation (low trust, no return)
            self.logger.error(
                f"BLOCKED: {entity.entity_id} (T={result.invariants.T:.3f}, τ_A={result.invariants.tau_A})"
            )

            # In production: quarantine file, block network, kill process
            if entity.entity_type == "file":
                return "QUARANTINE"
            elif entity.entity_type == "network":
                return "BLOCK_CONNECTION"
            elif entity.entity_type == "process":
                return "TERMINATE"
            else:
                return "BLOCK"

        else:  # NON_EVALUABLE
            # Insufficient data
            return "ALLOW_TRACK"

    def _ledger_writer(self):
        """Write trust ledger to disk periodically."""
        self.logger.info("Ledger writer started")

        while self.running:
            try:
                # Write ledger every minute
                time.sleep(60)

                ledger_file = self.state_dir / "trust_ledger.jsonl"
                with open(ledger_file, "a") as f:
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "stats": self.stats.copy(),
                        "active_entities": len(self.entities),
                        "trusted_entities": sum(1 for e in self.entities.values() if e.current_status == "TRUSTED"),
                        "blocked_entities": sum(1 for e in self.entities.values() if e.current_status == "BLOCKED"),
                    }
                    f.write(json.dumps(entry) + "\n")

            except Exception as e:
                self.logger.error(f"Ledger writer error: {e}")

    def _save_state(self):
        """Save daemon state to disk."""
        try:
            state_file = self.state_dir / "daemon_state.json"

            # Convert entities to serializable format
            entities_data = {}
            for entity_id, entity in self.entities.items():
                entities_data[entity_id] = {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "current_status": entity.current_status,
                    "first_seen": entity.first_seen.isoformat(),
                    "last_seen": entity.last_seen.isoformat(),
                    "signal_count": len(entity.signal_history),
                    "trust_ledger": entity.trust_seam_ledger[-1] if entity.trust_seam_ledger else 0.0,
                }

            state = {"saved_at": datetime.utcnow().isoformat(), "stats": self.stats, "entities": entities_data}

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"State saved to {state_file}")

        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load daemon state from disk."""
        try:
            state_file = self.state_dir / "daemon_state.json"

            if not state_file.exists():
                self.logger.info("No previous state found")
                return

            with open(state_file) as f:
                state = json.load(f)

            self.stats = state.get("stats", self.stats)

            self.logger.info(f"State loaded from {state_file}")
            self.logger.info(f"Previous stats: {self.stats}")

        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self.running,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            if hasattr(self, "start_time")
            else 0,
            "stats": self.stats,
            "active_entities": len(self.entities),
            "trusted_entities": sum(1 for e in self.entities.values() if e.current_status == "TRUSTED"),
            "blocked_entities": sum(1 for e in self.entities.values() if e.current_status == "BLOCKED"),
            "recent_events": len(self.events),
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="UMCP Security Daemon")
    parser.add_argument("action", choices=["start", "stop", "status"], help="Action to perform")
    parser.add_argument("--log-dir", default="/tmp/umcp_security/logs", help="Log directory")
    parser.add_argument("--state-dir", default="/tmp/umcp_security/state", help="State directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Check interval in seconds")

    args = parser.parse_args()

    daemon = SecurityDaemon(log_dir=args.log_dir, state_dir=args.state_dir, check_interval=args.interval)

    if args.action == "start":
        print("Starting UMCP Security Daemon...")
        print("Axiom: 'What Survives Validation Is Trusted'")
        print()
        daemon.start_time = datetime.utcnow()  # Reset start time on daemon start
        daemon.start()

    elif args.action == "stop":
        print("Stopping UMCP Security Daemon...")
        daemon.stop()

    elif args.action == "status":
        status = daemon.get_status()
        print("UMCP Security Daemon Status:")
        print(f"  Running: {status['running']}")
        print(f"  Active Entities: {status['active_entities']}")
        print(f"  Trusted: {status['trusted_entities']}")
        print(f"  Blocked: {status['blocked_entities']}")
        print(f"  Total Validated: {status['stats']['validated']}")


if __name__ == "__main__":
    main()
