# UMCP Production Deployment Guide

This guide provides best practices for deploying UMCP validator in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Health Checks](#health-checks)
- [Performance Tuning](#performance-tuning)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.11+ (3.12 recommended for production)
- **Memory**: Minimum 512MB RAM, 1GB+ recommended
- **Disk**: 100MB for installation, additional space for casepacks and logs
- **CPU**: Multi-core recommended for parallel validation

### Optional Dependencies

For production monitoring and system metrics:

```bash
pip install umcp[production]
```

This installs:
- `psutil` - System resource monitoring (CPU, memory, disk)

---

## Installation

### Production Installation

```bash
# Clone repository
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with production dependencies
pip install -e ".[production]"

# Verify installation
umcp --version
umcp health
```

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e ".[production]"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD umcp health --json || exit 1

# Run validation
ENTRYPOINT ["umcp"]
CMD ["validate", "--strict"]
```

Build and run:

```bash
docker build -t umcp-validator .
docker run --rm -v $(pwd):/workspace umcp-validator validate /workspace --strict
```

---

## Configuration

### Environment Variables

```bash
# Enable JSON-formatted logs for log aggregation systems (ELK, Splunk, CloudWatch)
export UMCP_JSON_LOGS=1

# Set log level (DEBUG, INFO, WARNING, ERROR)
export UMCP_LOG_LEVEL=INFO

# Performance monitoring
export UMCP_ENABLE_METRICS=1
```

### Validation Policies

Two operating modes:

**1. Non-Strict Mode (Default)**
- Validates structural correctness (schemas)
- Warnings are informational only
- Suitable for development and CI

```bash
umcp validate .
```

**2. Strict Mode (Production)**
- All warnings treated as errors
- Publication-grade enforcement
- Suitable for release gates and audits

```bash
umcp validate --strict
```

---

## Monitoring & Observability

### Structured Logging

UMCP provides structured JSON logs for production monitoring:

```bash
# Enable JSON logs
export UMCP_JSON_LOGS=1
umcp validate --strict --verbose > validation.log 2>&1
```

**Log Format:**

```json
{
  "timestamp": "2026-01-18T08:00:00.000000+00:00",
  "level": "INFO",
  "logger": "umcp",
  "message": "Completed: validate_repo",
  "context": {
    "operation": "validate_repo",
    "duration_ms": 2450.32,
    "memory_mb": 125.45,
    "cpu_percent": 15.2,
    "path": "/workspace"
  }
}
```

### Integration with Monitoring Systems

#### Prometheus

```python
# Example: Export metrics endpoint
from umcp.logging_utils import HealthCheck

def metrics_endpoint(repo_root):
    health = HealthCheck.check(repo_root)
    # Convert to Prometheus format
    metrics = [
        f"umcp_health_status{{status=\"{health['status']}\"}} 1",
        f"umcp_schemas_count {health['metrics']['schemas_count']}",
    ]
    if 'system' in health['metrics']:
        metrics.extend([
            f"umcp_cpu_percent {health['metrics']['system']['cpu_percent']}",
            f"umcp_memory_percent {health['metrics']['system']['memory_percent']}",
        ])
    return "\n".join(metrics)
```

#### CloudWatch

```bash
# Send validation results to CloudWatch
umcp validate --strict --out result.json
aws cloudwatch put-metric-data \
  --namespace UMCP \
  --metric-name ValidationErrors \
  --value $(jq '.summary.counts.errors' result.json)
```

#### ELK Stack

```bash
# Enable JSON logs and ship to Logstash
export UMCP_JSON_LOGS=1
umcp validate --strict --verbose 2>&1 | \
  filebeat -e -c filebeat.yml
```

---

## Health Checks

### System Health Check

Check validator system readiness:

```bash
# Human-readable output
umcp health

# JSON output for monitoring systems
umcp health --json
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T08:00:00Z",
  "checks": {
    "dir_schemas": {"status": "pass", "exists": true},
    "dir_contracts": {"status": "pass", "exists": true},
    "dir_closures": {"status": "pass", "exists": true},
    "schemas": {"status": "pass", "count": 10}
  },
  "metrics": {
    "schemas_count": 10,
    "system": {
      "cpu_percent": 5.0,
      "memory_percent": 41.5,
      "disk_percent": 33.7
    }
  }
}
```

### Kubernetes Liveness/Readiness Probes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: umcp-validator
spec:
  containers:
  - name: validator
    image: umcp-validator:latest
    livenessProbe:
      exec:
        command:
        - umcp
        - health
        - --json
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      exec:
        command:
        - umcp
        - health
        - --json
      initialDelaySeconds: 5
      periodSeconds: 10
```

---

## Performance Tuning

### Parallel Execution

For large repositories with multiple casepacks:

```bash
# Use pytest-xdist for parallel test execution
pytest -n auto tests/

# Process casepacks in parallel (custom script)
find casepacks -type d -maxdepth 1 | parallel -j 4 umcp validate {}
```

### Performance Metrics

Enable verbose mode to see performance breakdown:

```bash
umcp validate --strict --verbose
```

**Expected Performance:**
- Schema validation: ~1-5ms per schema
- Semantic rules: ~10-50ms per casepack
- Full repo validation: 2-5 seconds (typical)

### Optimization Tips

1. **Cache Schemas**: Load schemas once, reuse for multiple validations
2. **Batch Processing**: Group casepacks for validation
3. **Resource Limits**: Set memory/CPU limits in containerized environments
4. **Disable Metrics**: For maximum speed, skip performance monitoring

---

## Security

### Best Practices

1. **Input Validation**: All input paths are validated and sanitized
2. **No Code Execution**: YAML/JSON parsing only, no arbitrary code execution
3. **Sandboxing**: Run in containers with restricted permissions
4. **Audit Trails**: All validations produce cryptographically signed receipts

### Secure Deployment

```bash
# Run as non-root user
docker run --user 1000:1000 umcp-validator

# Read-only filesystem
docker run --read-only --tmpfs /tmp umcp-validator

# Network isolation
docker run --network=none umcp-validator
```

### Compliance

UMCP validator produces audit-ready receipts:
- SHA256 checksums of all validation results
- Git commit provenance
- Timestamp (UTC) with timezone awareness
- Python version and build information

---

## Troubleshooting

### Common Issues

#### Issue: "Could not find repo root"

**Solution**: Ensure `pyproject.toml` exists in repo root:

```bash
cd /path/to/repo
umcp validate .
```

#### Issue: Health check fails

**Solution**: Verify required directories exist:

```bash
ls -la schemas/ contracts/ closures/
umcp health --json | jq '.checks'
```

#### Issue: Slow validation

**Solution**: Enable verbose mode to identify bottlenecks:

```bash
umcp validate --strict --verbose 2>&1 | grep duration_ms
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use verbose flag:

```bash
umcp validate --verbose
```

### Support

- **GitHub Issues**: https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/issues
- **Documentation**: See `docs/` directory
- **Health Check**: Run `umcp health` for system diagnostics

---

## Production Checklist

- [ ] Python 3.12 installed
- [ ] Production dependencies installed (`psutil`)
- [ ] Environment variables configured
- [ ] Health check passing
- [ ] Validation runs in strict mode
- [ ] Logging configured (JSON format for production)
- [ ] Monitoring integrated (Prometheus/CloudWatch/ELK)
- [ ] Docker image built and tested
- [ ] Health check endpoints configured (K8s probes)
- [ ] Resource limits set (CPU/memory)
- [ ] Audit trail receipts collected
- [ ] Backup strategy for validation results

---

## Production Monitoring Dashboard (Example)

### Key Metrics to Track

1. **Validation Success Rate**: `(CONFORMANT / total) * 100`
2. **Average Validation Time**: Track `duration_ms` over time
3. **Error Rate**: Monitor `summary.counts.errors`
4. **System Resources**: CPU, memory, disk usage
5. **Schema Count**: Ensure all schemas loadable
6. **Health Status**: Track health check results

### Alerting Thresholds

- **Critical**: Health status = "unhealthy" OR validation errors > 0
- **Warning**: Health status = "degraded" OR duration_ms > 5000
- **Info**: New schemas detected OR casepacks added

---

## Version History

- **v0.1.0**: Production-ready release with monitoring
- Beta: Added structured logging and health checks
- Alpha: Initial validator implementation
