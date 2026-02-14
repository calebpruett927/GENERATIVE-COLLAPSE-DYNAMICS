# UMCP Security Domain

**Applies UMCP architecture to security validation.**

## Core Axiom Translation

> **"What Returns Through Collapse Is Real"** → **"What Survives Validation Is Trusted"**

An entity (file, URL, identity) is only "trusted" if it:
1. Passes through frozen Tier-0 security policy
2. Generates valid Tier-1 invariants
3. Returns to baseline (τ_A finite) after validation
4. Produces cryptographic receipt for audit

## Architecture

```
Tier-0 (Frozen Policy)
├── contracts/SECURITY.INTSTACK.v1.yaml    # Security contract
├── closures/security/threat_patterns.v1.yaml
├── closures/security/trust_baseline.v1.yaml
├── closures/security/privacy_rules.v1.yaml
└── closures/security/identity_verification.v1.yaml
    ↓
Tier-1 (Kernel Invariants) — Pure functions of frozen Tier-0
├── T = Trust Fidelity (analogous to F)
├── θ = Threat Drift (analogous to ω)
├── H = Security Entropy (analogous to S)
├── D = Signal Dispersion (analogous to C)
├── σ = Log-Integrity (analogous to κ)
├── TIC = Trust Integrity Composite (analogous to IC)
└── τ_A = Anomaly Return Time (analogous to τ_R)
    ↓
Tier-2 (Diagnostic Overlays) — Read-only, no back-edges
├── Threat Classifier
├── Reputation Analyzer
├── Behavior Profiler
└── Privacy Auditor
```

## Security Invariants (Tier-1)

| Symbol | Name | Formula | Range | Interpretation |
|--------|------|---------|-------|----------------|
| **T** | Trust Fidelity | T = Σ wᵢ · sᵢ | [0,1] | Weighted trust from signals |
| **θ** | Threat Drift | θ = 1 - T | [0,1] | Deviation from perfect trust |
| **H** | Security Entropy | H = -Σ wᵢ [sᵢ ln(sᵢ) + (1-sᵢ)ln(1-sᵢ)] | ≥0 | Classification uncertainty |
| **D** | Signal Dispersion | D = std(sᵢ)/0.5 | [0,1] | Signal variance |
| **σ** | Log-Integrity | σ = Σ wᵢ ln(sᵢ,ε) | ≤0 | Trust ledger |
| **TIC** | Trust Integrity Composite | TIC = exp(σ) | (0,1] | Geometric trust mean |
| **τ_A** | Anomaly Return Time | min{t-u : ‖S(t)-S(u)‖≤η} | ℕ∪{∞} | Time to baseline return |

## Validation Status

| Status | Condition | Meaning |
|--------|-----------|---------|
| **TRUSTED** | T ≥ 0.8 AND τ_A ≠ INF_ANOMALY | High trust, returns to baseline |
| **SUSPICIOUS** | 0.4 ≤ T < 0.8 OR τ_A > H_rec/2 | Moderate concern |
| **BLOCKED** | T < 0.4 OR τ_A = INF_ANOMALY | Low trust, no return |
| **NON_EVALUABLE** | Insufficient data | Cannot classify |

## Usage

### Python API

```python
from closures.security import SecurityValidator

validator = SecurityValidator()

# Validate security signals
signals = np.array([0.95, 0.88, 0.92, 0.90])
result = validator.validate_signals(signals)
print(result.status)  # "TRUSTED"

# Validate URL
result = validator.validate_url("https://github.com/project")
print(result.status)  # "TRUSTED"

result = validator.validate_url("http://secure-l0gin-verify.xyz")
print(result.status)  # "BLOCKED" (phishing indicators)

# Privacy audit
data = {"email": "user@example.com", "ssn": "123-45-6789"}
report = validator.validate_data_privacy(data)
print(report["privacy_score"])  # 0.7 (PII detected)
```

### Signal-Based Validation

Security signals are mapped to [0,1] bounded values:
- `integrity_score`: Hash match ratio (1 = verified)
- `reputation_score`: External reputation (1 = trusted)
- `behavior_score`: Deviation from baseline (1 = normal)
- `identity_score`: Authentication strength (1 = verified)

## Components

### Tier-1 Kernel (Pure Functions)

- `trust_fidelity.py` - Compute T and θ
- `security_entropy.py` - Compute H and D
- `trust_integrity.py` - Compute σ and TIC, seam accounting
- `anomaly_return.py` - Compute τ_A, detect return events

### Tier-2 Overlays (Diagnostics)

- `threat_classifier.py` - Classify threat type from invariants
- `reputation_analyzer.py` - URL/hash/IP reputation analysis
- `behavior_profiler.py` - Baseline profiling, deviation detection
- `privacy_auditor.py` - PII detection, privacy compliance

### Main Validator

- `security_validator.py` - Unified validation interface

## Closures (Tier-0)

- `threat_patterns.v1.yaml` - Frozen threat signatures
- `trust_baseline.v1.yaml` - Baseline definition for return
- `privacy_rules.v1.yaml` - PII patterns, consent rules
- `identity_verification.v1.yaml` - Auth signal definitions
- `registry.yaml` - Closure registry

## Contract

See `contracts/SECURITY.INTSTACK.v1.yaml` for the complete security contract including:
- Reserved invariant symbols
- Frozen parameters (thresholds, tolerances)
- Typed censoring (INF_ANOMALY, UNIDENTIFIABLE)
- Validation status enum

## CasePack

See `casepacks/security_validation/` for a complete example with:
- Sample security signals
- Expected invariant outputs
- Validation receipts
- Threat classifications

## Return-Based Canonization

Trust is NOT claimed by assertion. Trust is EARNED through validated return:

1. Entity enters validation (collapse event)
2. Tier-1 invariants computed from frozen policy
3. If τ_A finite (returns to baseline) AND TIC > threshold
4. THEN entity is TRUSTED with cryptographic receipt
5. ELSE entity remains SUSPICIOUS or BLOCKED

**"No return, no credit"** - Entities that never return to baseline (τ_A = INF_ANOMALY) do not earn trust, regardless of any other claims.

---

## Background Daemon (Real-Time Protection)

The UMCP security system can run as a **background daemon** providing continuous real-time protection:

### Quick Demo

```bash
python closures/security/demo_daemon.py
```

### Run as Service

```bash
# Start daemon
python closures/security/security_daemon.py start

# Stop daemon
python closures/security/security_daemon.py stop

# Check status
python closures/security/security_daemon.py status
```

### How It Works

1. **Continuous Monitoring**: Monitors files, network, processes in real-time
2. **Signal Collection**: Collects security signals (integrity, reputation, behavior, identity)
3. **UMCP Validation**: Computes invariants (T, θ, H, D, σ, TIC, τ_A) continuously
4. **Automated Response**: Takes action based on validation, not signatures

### Response Logic

| Validation Result | Action | Reason |
|-------------------|--------|--------|
| T ≥ 0.8, τ_A finite | **ALLOW** | High trust, returns to baseline |
| 0.4 ≤ T < 0.8 | **MONITOR** | Medium trust, enhanced logging |
| T < 0.4, τ_A finite | **QUARANTINE** | Low trust, isolate pending recovery |
| τ_A = INF_ANOMALY | **BLOCK** | No return = no trust credit |

### Key Differences from Traditional AV

**Traditional Antivirus:**
- Signature-based detection
- Binary allow/block decision
- No trust accounting
- Reactive (detects known threats)

**UMCP Security Daemon:**
- Validation-based trust (T, τ_A)
- Graduated response (4 levels)
- Trust ledger with seam accounting
- Proactive (validates through collapse-return)
- **"No return = no trust credit"** principle

### Architecture

```
┌─────────────────────────────────────────┐
│  UMCP Security Daemon (Background)      │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────┐  ┌────────────────┐    │
│  │  Signal   │→ │  Validation    │    │
│  │Collectors │  │  Engine        │    │
│  └───────────┘  │  (T, θ, τ_A)   │    │
│       ↓         └────────────────┘    │
│  Files, Net,            ↓              │
│  Processes         ┌────────────┐     │
│                    │  Response  │     │
│                    │  Engine    │     │
│                    └────────────┘     │
│                         ↓              │
│              ALLOW / MONITOR /         │
│              QUARANTINE / BLOCK        │
└─────────────────────────────────────────┘
```

### Components

- **`security_daemon.py`**: Main daemon process (783 lines)
- **`response_engine.py`**: Automated response system (382 lines)
- **`demo_daemon.py`**: Interactive demo (183 lines)

### Production Deployment

For production use, the daemon would:
1. Run as systemd service
2. Hook into kernel monitoring (inotify, netfilter, BPF)
3. Integrate with external threat feeds
4. Send alerts to SIEM/SOC
5. Maintain persistent trust ledger
6. Support policy updates via seam accounting

---

## Device-Level Security

The UMCP Security Daemon extends to **entire devices** on the network, not just files/processes.

### Supported Device Types

| Category | Examples | Signals Collected |
|----------|----------|-------------------|
| **Workstation** | Laptops, desktops | OS, firewall, AV, patches, connections |
| **Server** | VMs, bare metal | Services, ports, uptime, bandwidth |
| **Mobile** | Phones, tablets | OS version, encryption, MDM status |
| **IoT** | Cameras, sensors, smart home | Firmware, vulnerabilities, behavior |
| **Network** | Routers, switches, APs | Firmware, config, ports, uptime |
| **Unknown** | BYOD, rogue devices | Limited signals, low default trust |

### Device Trust Levels

| Trust (T) | τ_A | Status | Network Action | VLAN |
|-----------|-----|--------|----------------|------|
| T ≥ 0.80 | finite | TRUSTED | Full access | 100 |
| 0.60 ≤ T < 0.80 | any | LIMITED | Guest/external only | 200 |
| 0.40 ≤ T < 0.60 | any | QUARANTINE | Remediation only | 666 |
| T < 0.40 | ∞ | BLOCKED | Port shutdown | None |

### Usage

```bash
# Device daemon demo
python closures/security/device_daemon.py

# Network visualization
python closures/security/device_network_viz.py
```

### How Device Trust Works

```
Device Connects → Signals Collected → UMCP Validation → VLAN Assignment
                       │                    │
                       ▼                    ▼
              Identity, Software,    T (Trust Fidelity)
              Config, Behavior,      τ_A (Anomaly Return)
              Risk indicators        σ (Seam Integrity)
```

### Key Differences from Traditional NAC

| Traditional NAC | UMCP Device Security |
|-----------------|---------------------|
| One-time 802.1X auth | Continuous validation |
| Static allow/deny | Dynamic earned trust |
| Binary (on/off) | Graduated (4 levels) |
| Manual intervention | Self-healing |
| MAC whitelist for IoT | Trust-based quarantine |

### Device Components

- **`device_daemon.py`**: Device-level security daemon (900+ lines)
- **`device_network_viz.py`**: Network visualization with trust levels
- **Device signal collectors**: Identity, software, config, behavior, risk
- **Network controller**: VLAN assignment, port control, traffic mirroring
