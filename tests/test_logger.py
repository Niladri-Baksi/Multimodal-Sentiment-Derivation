"""Unit tests for shared logging configuration."""

from __future__ import annotations

import importlib
import logging
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.utils import logger as logger_module  # noqa: E402
from sentiment_derivation.utils.logger import configure_logging  # noqa: E402


@pytest.fixture
def isolated_logger_name() -> Iterator[str]:
    """Create a unique logger and remove its handlers after the test."""
    name = f"test.sentiment_derivation.{uuid.uuid4().hex}"
    yield name
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_configure_logging_returns_named_logger(isolated_logger_name: str) -> None:
    logger = configure_logging(isolated_logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == isolated_logger_name
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)


@pytest.mark.parametrize(
    ("level_name", "level_value"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_configure_logging_accepts_standard_levels(
    isolated_logger_name: str, level_name: str, level_value: int
) -> None:
    logger = configure_logging(isolated_logger_name, level_name.lower())

    assert logger.level == level_value
    assert logger.handlers[0].level == level_value


def test_configure_logging_rejects_invalid_level(isolated_logger_name: str) -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        configure_logging(isolated_logger_name, "TRACE")


def test_configure_logging_uses_consistent_formatter(isolated_logger_name: str) -> None:
    logger = configure_logging(isolated_logger_name)
    formatter = logger.handlers[0].formatter

    assert formatter is not None
    assert formatter._fmt == "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"


def test_repeated_configuration_does_not_add_handlers(isolated_logger_name: str) -> None:
    logger = configure_logging(isolated_logger_name, "INFO")
    initial_handlers = tuple(logger.handlers)

    same_logger = configure_logging(isolated_logger_name, "DEBUG")

    assert same_logger is logger
    assert tuple(logger.handlers) == initial_handlers
    assert logger.level == logging.DEBUG


def test_import_does_not_configure_root_logger() -> None:
    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_level = root_logger.level

    importlib.reload(logger_module)

    assert tuple(root_logger.handlers) == original_handlers
    assert root_logger.level == original_level


def test_configure_logging_does_not_modify_root_logger() -> None:
    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_level = root_logger.level
    name = f"test.sentiment_derivation.root_isolation.{uuid.uuid4().hex}"

    try:
        configure_logging(name, "DEBUG")

        assert tuple(root_logger.handlers) == original_handlers
        assert root_logger.level == original_level
    finally:
        named_logger = logging.getLogger(name)
        for handler in named_logger.handlers[:]:
            named_logger.removeHandler(handler)
            handler.close()
