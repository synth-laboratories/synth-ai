"""Shared runtime helpers for the narrowed Synth SDK.

The live core surface is intentionally small and only exposes the pieces used by
containers, tunnels, and pools.
"""

# Error types
from synth_ai.core.errors import (
    AuthenticationError,
    ConfigError,
    HTTPError,
    JobError,
    ModelNotSupportedError,
    PaymentRequiredError,
    StorageError,
    SynthError,
    TimeoutError,
    ValidationError,
)
from synth_ai.core.utils.env import get_api_key, mask_value
from synth_ai.core.utils.urls import BACKEND_URL_BASE

__all__ = [
    # Errors
    "SynthError",
    "ConfigError",
    "AuthenticationError",
    "ValidationError",
    "HTTPError",
    "PaymentRequiredError",
    "JobError",
    "TimeoutError",
    "StorageError",
    "ModelNotSupportedError",
    # Environment
    "get_api_key",
    "mask_value",
    "BACKEND_URL_BASE",
]
