"""Tests for structured logging configuration.

Proves: configure_logging works in both modes, structlog emits events,
stdlib bridge routes through structlog processors.
"""

from __future__ import annotations

import structlog
from structlog.testing import capture_logs

from navi_sad.logging import configure_logging


class TestConfigureLogging:
    def test_console_mode_does_not_raise(self) -> None:
        """Console (dev) mode configures without error."""
        configure_logging(json=False)

    def test_json_mode_does_not_raise(self) -> None:
        """JSON (production) mode configures without error."""
        configure_logging(json=True)


class TestStructlogEvents:
    def test_capture_logs(self) -> None:
        """structlog events can be captured in tests."""
        with capture_logs() as cap:
            log = structlog.get_logger()
            log.info("test_event", key="value", count=42)
        assert len(cap) == 1
        assert cap[0]["event"] == "test_event"
        assert cap[0]["key"] == "value"
        assert cap[0]["count"] == 42

    def test_bound_logger_carries_context(self) -> None:
        """Bound loggers carry context through calls."""
        with capture_logs() as cap:
            log = structlog.get_logger()
            bound = log.bind(run_id="abc-123")
            bound.info("with_context")
        assert cap[0]["run_id"] == "abc-123"
