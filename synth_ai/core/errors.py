"""Consolidated error types for Synth AI SDK.

This module provides base exception classes used throughout the SDK.
CLI-specific errors remain in cli/ modules; these are for SDK/core use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SynthError(Exception):
    """Base exception for all Synth AI SDK errors."""

    pass


class ConfigError(SynthError):
    """Raised when configuration is invalid or missing."""

    pass


class AuthenticationError(SynthError):
    """Raised when API authentication fails."""

    pass


class ValidationError(SynthError):
    """Raised when data validation fails."""

    pass


@dataclass
class HTTPError(SynthError):
    """Raised when an HTTP request fails."""

    status: int
    url: str
    message: str
    body_snippet: str | None = None
    detail: Any | None = None

    def __str__(self) -> str:
        base = f"HTTP {self.status} for {self.url}: {self.message}"
        if self.body_snippet:
            base += f" | body[0:200]={self.body_snippet[:200]}"
        return base


class JobError(SynthError):
    """Raised when a job operation fails."""

    pass


class TimeoutError(SynthError):
    """Raised when an operation times out."""

    pass


class StorageError(SynthError):
    """Raised when storage operations fail."""

    pass


class ModelNotSupportedError(SynthError):
    """Raised when a model is not supported."""

    def __init__(self, model: str, provider: str | None = None) -> None:
        self.model = model
        self.provider = provider
        msg = f"Model '{model}' is not supported"
        if provider:
            msg += f" by provider '{provider}'"
        super().__init__(msg)


@dataclass
class UsageLimitError(SynthError):
    """Raised when an org rate limit is exceeded.

    Attributes:
        limit_type: The type of limit exceeded (e.g., "inference_tokens_per_day")
        api: The API that hit the limit (e.g., "inference", "judges", "prompt_opt")
        current: Current usage value
        limit: The limit value
        tier: The org's tier (e.g., "free", "starter", "growth")
        retry_after_seconds: Seconds until the limit resets (if available)
        upgrade_url: URL to upgrade tier
    """

    limit_type: str
    api: str
    current: int | float
    limit: int | float
    tier: str = "free"
    retry_after_seconds: int | None = None
    upgrade_url: str = "https://usesynth.ai/pricing"

    def __str__(self) -> str:
        return (
            f"Rate limit exceeded: {self.limit_type} "
            f"({self.current}/{self.limit}) for tier '{self.tier}'. "
            f"Upgrade at {self.upgrade_url}"
        )


__all__ = [
    "SynthError",
    "ConfigError",
    "AuthenticationError",
    "ValidationError",
    "HTTPError",
    "JobError",
    "TimeoutError",
    "StorageError",
    "ModelNotSupportedError",
    "UsageLimitError",
]

