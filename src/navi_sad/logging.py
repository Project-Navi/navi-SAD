"""Structured logging configuration for navi-SAD.

Call configure_logging() once at application startup, before any
module-level loggers are used. Bridges structlog with stdlib logging
so both structlog.get_logger() and logging.getLogger() emit
structured output through the same processor chain.

Usage in scripts:
    from navi_sad.logging import configure_logging
    configure_logging()  # colored console
    configure_logging(json=True)  # JSON lines

Usage in library modules:
    import structlog
    log = structlog.get_logger()
    log.info("event_name", key=value, ...)
"""

from __future__ import annotations

import logging
import logging.config

import structlog

_configured = False


def configure_logging(
    *,
    json: bool = False,
    level: str = "INFO",
) -> None:
    """Configure structured logging for the application.

    Args:
        json: If True, emit JSON lines. If False, colored console output.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).

    Must be called exactly once, before any log calls. Subsequent
    calls are no-ops (one-shot guard).
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for both structlog and stdlib paths.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure stdlib logging to route through structlog processors.
    # This ensures logging.getLogger() calls (e.g., from core/) emit
    # structured output alongside structlog.get_logger() calls.
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        renderer,
                    ],
                    "foreign_pre_chain": shared_processors,
                },
            },
            "handlers": {
                "default": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "structlog",
                },
            },
            "root": {
                "handlers": ["default"],
                "level": log_level,
            },
        }
    )

    # Configure structlog to use stdlib as the backend.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
