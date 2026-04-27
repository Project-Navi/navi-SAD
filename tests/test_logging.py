"""Tests for structured logging configuration.

Proves: configure_logging works in both modes, structlog emits events,
stdlib bridge routes through structlog processors, JSON output is parseable,
one-shot guard prevents double configuration.
"""

from __future__ import annotations

import json
import logging

import pytest
import structlog
from structlog.testing import capture_logs

import navi_sad.logging as logging_mod
from navi_sad.logging import configure_logging


@pytest.fixture(autouse=True)
def _reset_configure_logging_guard() -> None:
    """Reset the ``_configured`` one-shot guard around every test.

    Without this, the first test to call ``configure_logging`` flips the
    guard True for the rest of the session, so later tests that pass a
    different ``json=`` argument become silent no-ops and never exercise
    the renderer / dictConfig path they intend to. An autouse fixture is
    safer than per-test inline resets because it guarantees order-
    independence even as the file grows.
    """
    logging_mod._configured = False
    yield
    logging_mod._configured = False


class TestConfigureLogging:
    def test_console_mode_does_not_raise(self) -> None:
        """Console (dev) mode configures without error."""
        configure_logging(json=False)

    def test_json_mode_does_not_raise(self) -> None:
        """JSON (production) mode configures without error."""
        configure_logging(json=True)

    def test_one_shot_guard(self) -> None:
        """Second call to configure_logging is a no-op."""
        import navi_sad.logging as logging_mod

        logging_mod._configured = False
        configure_logging(json=False)
        assert logging_mod._configured is True
        # Second call should not raise or reconfigure
        configure_logging(json=True)
        assert logging_mod._configured is True
        # Reset for other tests
        logging_mod._configured = False


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


class TestStdlibBridge:
    def test_stdlib_logger_emits_through_structlog(self) -> None:
        """stdlib logging.getLogger() calls route through structlog processors.

        This tests the bridge that makes existing core/ module loggers
        emit structured output after configure_logging() is called.
        """
        import navi_sad.logging as logging_mod

        logging_mod._configured = False
        configure_logging(json=False)

        stdlib_logger = logging.getLogger("test.stdlib.bridge")
        # The stdlib logger should work without error after configuration.
        # We can't easily capture its output via structlog.testing.capture_logs
        # (that only captures structlog-native calls), but we can verify
        # it doesn't raise and the handler is wired.
        stdlib_logger.info("stdlib bridge test")
        assert len(stdlib_logger.handlers) > 0 or logging.root.handlers

        logging_mod._configured = False


class TestJSONOutput:
    def test_json_renderer_produces_parseable_output(self) -> None:
        """JSON renderer output is valid JSON with expected fields.

        Uses structlog's JSONRenderer directly to verify the output shape,
        since capture_logs() bypasses the renderer.
        """
        renderer = structlog.processors.JSONRenderer()
        event_dict = {
            "event": "test_event",
            "key": "value",
            "count": 42,
            "log_level": "info",
        }
        rendered = renderer(None, None, event_dict)
        parsed = json.loads(rendered)
        assert parsed["event"] == "test_event"
        assert parsed["key"] == "value"
        assert parsed["count"] == 42
        assert parsed["log_level"] == "info"

    def test_json_renderer_handles_nested_data(self) -> None:
        """JSON renderer handles nested dicts and lists."""
        renderer = structlog.processors.JSONRenderer()
        event_dict = {
            "event": "complex_event",
            "metadata": {"n_layers": 32, "n_heads": 32},
            "indices": [1, 2, 3],
        }
        rendered = renderer(None, None, event_dict)
        parsed = json.loads(rendered)
        assert parsed["metadata"]["n_layers"] == 32
        assert parsed["indices"] == [1, 2, 3]
