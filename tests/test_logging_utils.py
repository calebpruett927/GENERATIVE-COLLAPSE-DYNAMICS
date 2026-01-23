import pytest
from src.umcp.logging_utils import StructuredLogger, get_logger, HealthCheck

def test_structured_logger_basic():
    logger = StructuredLogger(json_output=False)
    logger.info("Test info log")
    logger.debug("Test debug log")
    logger.warning("Test warning log")
    logger.error("Test error log")
    logger.critical("Test critical log")

def test_logger_operation_context_exception():
    logger = StructuredLogger(json_output=False)
    with pytest.raises(ValueError):
        with logger.operation("fail_operation"):
            raise ValueError("fail")

def test_get_logger():
    logger = get_logger()
    assert isinstance(logger, StructuredLogger)

def test_health_check(tmp_path):
    result = HealthCheck.check(tmp_path)
    assert isinstance(result, dict)
    assert "status" in result

def test_logger_json_output():
    logger = StructuredLogger(json_output=True)
    logger.info("Test JSON log")
