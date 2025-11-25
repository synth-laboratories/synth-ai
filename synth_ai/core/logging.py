"""Logging utilities for Synth AI SDK.

This module provides structured logging setup for SDK modules.
Uses standard logging library, not print statements.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

# SDK logger name prefix
SDK_LOGGER_NAME = "synth_ai"


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the synth_ai namespace.

    Args:
        name: Module name (will be prefixed with synth_ai.)

    Returns:
        Configured logger instance
    """
    if name.startswith(SDK_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{SDK_LOGGER_NAME}.{name}")


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None,
    *,
    format_style: Literal["simple", "detailed"] = "simple",
) -> None:
    """Configure SDK-wide logging.

    Args:
        level: Log level (or read from SYNTH_LOG_LEVEL env var)
        format_style: 'simple' for minimal output, 'detailed' for timestamps
    """
    if level is None:
        level_str = os.environ.get("SYNTH_LOG_LEVEL", "INFO").upper()
        if level_str not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            level_str = "INFO"
        level = level_str  # type: ignore[assignment]

    # After assignment, level is guaranteed to be str
    assert isinstance(level, str), "level must be str at this point"
    log_level = getattr(logging, level)

    if format_style == "detailed":
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    else:
        fmt = "%(levelname)s: %(message)s"

    # Configure the SDK root logger
    logger = logging.getLogger(SDK_LOGGER_NAME)
    logger.setLevel(log_level)

    # Only add handler if none exist (avoid duplicates)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)


def suppress_noisy_loggers() -> None:
    """Suppress verbose third-party loggers."""
    for name in ("httpx", "httpcore", "aiohttp", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


__all__ = [
    "get_logger",
    "configure_logging",
    "suppress_noisy_loggers",
    "SDK_LOGGER_NAME",
]

