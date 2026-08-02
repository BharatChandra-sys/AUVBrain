"""Structured JSON logging configuration.

In production every log line is emitted as a JSON object so that log
aggregators (Loki, CloudWatch, Datadog, etc.) can parse fields without
fragile regex extraction.

In development (``LOG_FORMAT=text``) a human-readable format is used instead.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Correlation ID propagated through async tasks via contextvars
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        exc_text: str | None = None
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)

        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": correlation_id.get("-"),
        }

        if exc_text:
            payload["exc"] = exc_text
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Attach any extra fields passed via ``logger.info(..., extra={...})``
        for key in ("tick", "mode", "source", "mission_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FMT = "%(asctime)s %(levelname)-8s [%(correlation_id)s] %(name)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        # Inject correlation_id into LogRecord for the format string
        record.correlation_id = correlation_id.get("-")  # type: ignore[attr-defined]
        return super().format(record)


def configure_logging(level: str, *, fmt: str = "json") -> None:
    """Configure root logger.

    Args:
        level:  Log level string, e.g. ``"INFO"``.
        fmt:    ``"json"`` (default, for production) or ``"text"`` (local dev).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if fmt.lower() == "text":
        formatter: logging.Formatter = _TextFormatter(
            fmt=_TextFormatter.FMT, datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        formatter = _JSONFormatter()

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Replace any existing handlers (avoid duplicate output on re-init)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers at WARNING
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
