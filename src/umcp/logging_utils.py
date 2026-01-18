"""
Production-grade structured logging for UMCP validator.

Provides JSON-structured logging with performance metrics, error context,
and observability hooks for production deployment.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and observability."""
    
    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_used_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    def finish(self) -> None:
        """Mark operation complete and calculate metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self.memory_used_mb = process.memory_info().rss / 1024 / 1024
                self.cpu_percent = process.cpu_percent(interval=0.1)
            except Exception:
                pass
    
    def to_dict(self) -> Dict[str, Any]:
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
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
        
        self.logger.addHandler(handler)
    
    def _log(self, level: int, message: str, **context: Any) -> None:
        """Internal log method with structured context."""
        extra = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
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
                **metrics.to_dict()
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
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add context if available
        if hasattr(record, "context"):
            log_data["context"] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class HealthCheck:
    """System health check for production monitoring."""
    
    @staticmethod
    def check(repo_root: Path) -> Dict[str, Any]:
        """
        Perform health check on UMCP validator system.
        
        Returns comprehensive health status suitable for monitoring endpoints.
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "metrics": {}
        }
        
        # Check: Required directories exist
        try:
            required_dirs = ["schemas", "contracts", "closures"]
            for dir_name in required_dirs:
                dir_path = repo_root / dir_name
                health["checks"][f"dir_{dir_name}"] = {
                    "status": "pass" if dir_path.exists() else "fail",
                    "exists": dir_path.exists()
                }
        except Exception as e:
            health["checks"]["directories"] = {"status": "fail", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check: Schemas loadable
        try:
            schemas_dir = repo_root / "schemas"
            if schemas_dir.exists():
                schema_count = len(list(schemas_dir.glob("*.json")))
                health["checks"]["schemas"] = {
                    "status": "pass" if schema_count > 0 else "fail",
                    "count": schema_count
                }
                health["metrics"]["schemas_count"] = schema_count
        except Exception as e:
            health["checks"]["schemas"] = {"status": "fail", "error": str(e)}
            health["status"] = "degraded"
        
        # System metrics
        if HAS_PSUTIL:
            try:
                health["metrics"]["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent
                }
            except Exception:
                pass
        
        # Overall status
        failed_checks = sum(
            1 for check in health["checks"].values()
            if isinstance(check, dict) and check.get("status") == "fail"
        )
        
        if failed_checks > 0:
            health["status"] = "unhealthy"
        elif any(
            check.get("status") == "degraded"
            for check in health["checks"].values()
            if isinstance(check, dict)
        ):
            health["status"] = "degraded"
        
        return health


# Global logger instance
_default_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "umcp",
    json_output: bool = False,
    include_metrics: bool = True
) -> StructuredLogger:
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
        _default_logger = StructuredLogger(
            name=name,
            json_output=json_output,
            include_metrics=include_metrics
        )
    return _default_logger
