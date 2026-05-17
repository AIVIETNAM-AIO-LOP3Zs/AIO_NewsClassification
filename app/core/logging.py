"""
Structured logging configuration.
Supports both human-readable (text) and machine-parseable (JSON) formats.
"""
import logging
import sys
from typing import Any

try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False


def _setup_stdlib_logging(log_level: str) -> None:
    """Fallback: standard library logging with a clean formatter."""
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )


def _setup_structlog(log_level: str, log_format: str) -> None:
    """Configure structlog with optional JSON output."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format.lower() == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so uvicorn/httpx logs are captured.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Entry point called once in the app lifespan."""
    if _STRUCTLOG_AVAILABLE:
        _setup_structlog(log_level, log_format)
    else:
        _setup_stdlib_logging(log_level)


def get_logger(name: str) -> Any:
    """Return an appropriate logger (structlog or stdlib)."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)
