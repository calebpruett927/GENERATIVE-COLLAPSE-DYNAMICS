"""
Tests for production-grade logging and monitoring features.
"""

from __future__ import annotations

import json

import pytest

from umcp.logging_utils import HealthCheck, PerformanceMetrics, StructuredLogger, get_logger


def test_performance_metrics():
    """Test performance metrics collection."""
    metrics = PerformanceMetrics(operation="test_operation")
    metrics.finish()

    assert metrics.operation == "test_operation"
    assert metrics.duration_ms is not None
    assert metrics.duration_ms >= 0

    data = metrics.to_dict()
    assert data["operation"] == "test_operation"
    assert "duration_ms" in data


def test_structured_logger():
    """Test structured logger basic functionality."""
    logger = StructuredLogger(name="test", json_output=False, include_metrics=True)

    # These should not raise exceptions
    logger.debug("Debug message", key="value")
    logger.info("Info message", key="value")
    logger.warning("Warning message", key="value")
    logger.error("Error message", key="value")


def test_structured_logger_operation_context():
    """Test logger operation context manager."""
    logger = StructuredLogger(name="test", json_output=False, include_metrics=True)

    with logger.operation("test_op", file="test.json") as metrics:
        # Simulate work
        pass

    assert metrics.duration_ms is not None
    assert metrics.duration_ms >= 0


def test_structured_logger_operation_with_exception():
    """Test logger operation context with exception."""
    logger = StructuredLogger(name="test", json_output=False, include_metrics=True)

    with pytest.raises(ValueError), logger.operation("failing_op"):
        raise ValueError("Test error")


def test_health_check(repo_paths):
    """Test health check functionality."""
    health = HealthCheck.check(repo_paths.root)

    assert "status" in health
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert "timestamp" in health
    assert "checks" in health
    assert "metrics" in health

    # Check required directories
    assert "dir_schemas" in health["checks"]
    assert "dir_contracts" in health["checks"]
    assert "dir_closures" in health["checks"]

    # Schemas should be loadable
    assert "schemas" in health["checks"]
    assert health["checks"]["schemas"]["status"] == "pass"
    assert health["checks"]["schemas"]["count"] > 0


def test_health_check_json_serializable(repo_paths):
    """Test health check output is JSON serializable."""
    health = HealthCheck.check(repo_paths.root)

    # Should not raise
    json_str = json.dumps(health)
    loaded = json.loads(json_str)

    assert loaded["status"] == health["status"]


def test_get_logger_singleton():
    """Test get_logger returns consistent instance."""
    logger1 = get_logger()
    logger2 = get_logger()

    # Should be the same instance (singleton pattern)
    assert logger1 is logger2


def test_logger_json_output():
    """Test JSON output mode for log aggregation."""
    logger = StructuredLogger(name="test", json_output=True, include_metrics=False)

    # Should not raise with JSON formatting
    logger.info("Test message", key="value", count=42)


def test_health_check_metrics(repo_paths):
    """Test health check includes system metrics when psutil available."""
    health = HealthCheck.check(repo_paths.root)

    assert "metrics" in health
    assert "schemas_count" in health["metrics"]
    assert health["metrics"]["schemas_count"] > 0

    # System metrics may or may not be present depending on psutil availability
    # Just check they're dict or not present
    if "system" in health["metrics"]:
        assert isinstance(health["metrics"]["system"], dict)
