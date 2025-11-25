"""Synth AI Core Layer.

This module provides internal plumbing that SDK and CLI can share:
- Environment resolution
- HTTP client utilities
- Error types
- Logging
- Pricing
- Config base classes

Dependency rules:
- core/ can import data/
- core/ should NOT import sdk/ or cli/
- core/ should NOT import click (leave that to cli/)
"""

from __future__ import annotations

# Config base classes
from synth_ai.core.config import BaseJobConfig, ConfigValidator

# Environment utilities
from synth_ai.core.env import (
    PROD_BASE_URL,
    get_api_key,
    get_backend_url,
    load_env_file,
    mask_value,
    resolve_env_file,
)

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

# HTTP utilities
from synth_ai.core.http import AsyncHttpClient, sleep

# Logging
from synth_ai.core.logging import (
    SDK_LOGGER_NAME,
    configure_logging,
    get_logger,
    suppress_noisy_loggers,
)

# Pricing
from synth_ai.core.pricing import (
    MODEL_PRICES,
    TokenRates,
    estimate_cost,
    get_token_rates,
)

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
    "get_backend_url",
    "resolve_env_file",
    "load_env_file",
    "mask_value",
    "PROD_BASE_URL",
    # HTTP
    "AsyncHttpClient",
    "sleep",
    # Logging
    "get_logger",
    "configure_logging",
    "suppress_noisy_loggers",
    "SDK_LOGGER_NAME",
    # Pricing
    "TokenRates",
    "MODEL_PRICES",
    "get_token_rates",
    "estimate_cost",
    # Config
    "BaseJobConfig",
    "ConfigValidator",
]


