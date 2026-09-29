"""Structured JSON logging configuration for enterprise observability.

Adheres strictly to Hexagonal Architecture: uses standard library `logging` and `json`.
Formats records with timestamp, level, session_id, step_index, action_type, and execution_time_ms.
"""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON objects with contextual metadata."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Contextual audit fields
        for attr in ("session_id", "step_index", "step", "action_type", "execution_time_ms"):
            val = getattr(record, attr, None)
            if val is not None:
                log_payload[attr] = val

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-calls
    has_json_handler = any(
        isinstance(h.formatter, StructuredJSONFormatter)
        for h in root_logger.handlers
        if h.formatter
    )
    if not has_json_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Retrieve logger instance with structured capability."""
    return logging.getLogger(name)
