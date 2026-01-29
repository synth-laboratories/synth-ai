"""Synth AI Core Layer.

This module provides internal plumbing that SDK and CLI can share:
- Environment resolution
- HTTP client utilities
- Error types

Dependency rules:
- core/ can import data/
- core/ should NOT import sdk/ or cli/
- core/ should NOT import click (leave that to cli/)
"""

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
]
