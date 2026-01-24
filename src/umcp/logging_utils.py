"""
Production-grade structured logging for UMCP validator.

Provides JSON-structured logging with performance metrics, error context,
and observability hooks for production deployment.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# psutil availability check
_has_psutil = False
try:
    import psutil

    _has_psutil = True
except ImportError:
    psutil = None  # type: ignore[assignment]


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and observability."""

    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    duration_ms: float | None = None
    memory_used_mb: float | None = None
    cpu_percent: float | None = None

    def finish(self) -> None:
        """Mark operation complete and calculate metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.memory_used_mb = None
        self.cpu_percent = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        data = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
        }
        if self.memory_used_mb:
            data["memory_mb"] = round(self.memory_used_mb, 2)
        if self.cpu_percent:
            data["cpu_percent"] = round(self.cpu_percent, 2)
        return data


class StructuredLogger:
    """
    Structured JSON logger for production observability.

    Outputs JSON-formatted logs suitable for log aggregation systems
    (ELK, Splunk, CloudWatch, etc.) while maintaining human readability.
    """

    def __init__(
        self,
        name: str = "umcp",
        level: int = logging.INFO,
        json_output: bool = False,
        include_metrics: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_output = json_output
        self.include_metrics = include_metrics

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        if json_output:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        self.logger.addHandler(handler)

    def _log(self, level: int, message: str, **context: Any) -> None:
        """Internal log method with structured context."""
        extra = {"timestamp": datetime.now(UTC).isoformat(), "context": context}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **context: Any) -> None:
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._log(logging.ERROR, message, **context)

    def critical(self, message: str, **context: Any) -> None:
        self._log(logging.CRITICAL, message, **context)

    @contextmanager
    def operation(self, name: str, **context: Any) -> Iterator[PerformanceMetrics]:
        """
        Context manager for timing and monitoring operations.

        Example:
            with logger.operation("validate_schema", file="manifest.json") as metrics:
                # do work
                pass
            # Automatically logs performance metrics
        """
        metrics = PerformanceMetrics(operation=name)
        self.debug(f"Starting: {name}", **context)

        try:
            yield metrics
        except Exception as e:
            metrics.finish()
            self.error(
                f"Failed: {name}",
                error=str(e),
                error_type=type(e).__name__,
                **context,
                **metrics.to_dict(),
            )
            raise
        else:
            metrics.finish()
            if self.include_metrics:
                self.info(f"Completed: {name}", **context, **metrics.to_dict())


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if available
        context = getattr(record, "context", None)
        if context is not None:
            log_data["context"] = context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class HealthCheck:
    """System health check for production monitoring."""

    @staticmethod
    def check(repo_root: Path) -> dict[str, Any]:
        """
        Perform health check on UMCP validator system.

        Returns comprehensive health status suitable for monitoring endpoints.
        """
        checks: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        health: dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": checks,
            "metrics": metrics,
        }

        # Check: Required directories exist
        try:
            required_dirs = ["schemas", "contracts", "closures"]
            for dir_name in required_dirs:
                dir_path = repo_root / dir_name
                checks[f"dir_{dir_name}"] = {
                    "status": "pass" if dir_path.exists() else "fail",
                    "exists": dir_path.exists(),
                }
        except Exception as e:
            checks["directories"] = {"status": "fail", "error": str(e)}
            health["status"] = "unhealthy"

        # Check: Schemas loadable
        try:
            schemas_dir = repo_root / "schemas"
            if schemas_dir.exists():
                schema_count = len(list(schemas_dir.glob("*.json")))
                checks["schemas"] = {
                    "status": "pass" if schema_count > 0 else "fail",
                    "count": schema_count,
                }
                metrics["schemas_count"] = schema_count
        except Exception as e:
            checks["schemas"] = {"status": "fail", "error": str(e)}
            health["status"] = "degraded"

        # System metrics
        if _has_psutil and psutil is not None:
            with suppress(Exception):
                metrics["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                }

        # Overall status
        failed_checks = sum(1 for check in checks.values() if isinstance(check, dict) and check.get("status") == "fail")

        if failed_checks > 0:
            health["status"] = "unhealthy"
        elif any(check.get("status") == "degraded" for check in checks.values() if isinstance(check, dict)):
            health["status"] = "degraded"

        return health


# Global logger instance
_default_logger: StructuredLogger | None = None


def get_logger(name: str = "umcp", json_output: bool = False, include_metrics: bool = True) -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name
        json_output: Use JSON formatting (for production log aggregation)
        include_metrics: Include performance metrics in logs

    Returns:
        StructuredLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger(name=name, json_output=json_output, include_metrics=include_metrics)
    return _default_logger
