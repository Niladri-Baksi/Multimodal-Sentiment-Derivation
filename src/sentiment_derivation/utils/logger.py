"""Shared application logging configuration."""

from __future__ import annotations

import logging
from typing import Final

_HANDLER_MARKER: Final[str] = "_sentiment_derivation_console_handler"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_LEVELS: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_level(level: str | int) -> int:
    """Return a supported logging level or raise a clear ``ValueError``."""
    if isinstance(level, str):
        resolved = _LEVELS.get(level.upper())
        if resolved is None:
            supported = ", ".join(_LEVELS)
            raise ValueError(f"Unsupported log level {level!r}; choose one of: {supported}.")
        return resolved
    if isinstance(level, int) and not isinstance(level, bool) and level in _LEVELS.values():
        return level
    supported = ", ".join(_LEVELS)
    raise ValueError(f"Unsupported log level {level!r}; choose one of: {supported}.")


def configure_logging(
    name: str, level: str | int = "INFO"
) -> logging.Logger:
    """Configure and return a named logger with one console handler.

    Repeated calls for the same logger reuse this utility's handler and update
    its level. The root logger is not configured. Propagation is disabled on
    the returned named logger so its console output is not duplicated by root
    handlers.

    Args:
        name: Name for the application logger.
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL, or the matching
            standard-library numeric logging constant.

    Returns:
        A configured :class:`logging.Logger` instance.

    Raises:
        ValueError: If ``level`` is not one of the supported levels.
    """
    resolved_level = _resolve_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    handler = next(
        (candidate for candidate in logger.handlers
         if getattr(candidate, _HANDLER_MARKER, False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return logger
