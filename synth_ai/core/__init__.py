"""Synth AI Core Layer.

This module provides internal plumbing that SDK and CLI can share:
- Environment resolution
- HTTP client utilities
- Error types
- Logging
- Config base classes

Dependency rules:
- core/ can import data/
- core/ should NOT import sdk/ or cli/
- core/ should NOT import click (leave that to cli/)
"""

# Config base classes
from synth_ai.core.config import BaseJobConfig, ConfigValidator

# Error types
from synth_ai.core.errors import (
    AuthenticationError,
    ConfigError,
    HTTPError,
    JobError,
    ModelNotSupportedError,
    StorageError,
    SynthError,
    TimeoutError,
    ValidationError,
)

# Logging
from synth_ai.core.logging import (
    SDK_LOGGER_NAME,
    configure_logging,
    get_logger,
    suppress_noisy_loggers,
)

# Environment utilities
from synth_ai.core.utils.env import get_api_key, mask_value
from synth_ai.core.utils.urls import BACKEND_URL_BASE

__all__ = [
    # Errors
    "SynthError",
    "ConfigError",
    "AuthenticationError",
    "ValidationError",
    "HTTPError",
    "JobError",
    "TimeoutError",
    "StorageError",
    "ModelNotSupportedError",
    # Environment
    "get_api_key",
    "mask_value",
    "BACKEND_URL_BASE",
    # Logging
    "get_logger",
    "configure_logging",
    "suppress_noisy_loggers",
    "SDK_LOGGER_NAME",
    # Config
    "BaseJobConfig",
    "ConfigValidator",
]
