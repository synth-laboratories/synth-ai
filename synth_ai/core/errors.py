"""Consolidated error types for Synth AI SDK.

This module provides base exception classes used throughout the SDK.
CLI-specific errors remain in cli/ modules; these are for SDK/core use.
"""

from __future__ import annotations

import base64
import json
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


def _extract_x402_payload(detail: Any, body_snippet: str | None) -> tuple[Any | None, str | None]:
    if isinstance(detail, dict):
        x402_value = detail.get("x402")
        if isinstance(x402_value, dict):
            return x402_value.get("challenge"), x402_value.get("payment_required_header")
    if not body_snippet:
        return None, None
    try:
        payload = json.loads(body_snippet)
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    detail_payload = payload.get("detail")
    if not isinstance(detail_payload, dict):
        return None, None
    x402_value = detail_payload.get("x402")
    if not isinstance(x402_value, dict):
        return None, None
    return x402_value.get("challenge"), x402_value.get("payment_required_header")


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
        api: The API that hit the limit (e.g., "inference", "verifiers", "prompt_opt")
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


@dataclass
class PlanGatingError(SynthError):
    """Raised when a feature requires a higher subscription plan.

    Attributes:
        feature: The feature that was denied (e.g., "environment_pools")
        current_plan: The user's current plan tier
        required_plans: Plans that grant access to this feature
        upgrade_url: URL to upgrade plan
    """

    feature: str
    current_plan: str = "free"
    required_plans: tuple[str, ...] = ("pro", "team")
    upgrade_url: str = "https://usesynth.ai/pricing"

    def __str__(self) -> str:
        plans = ", ".join(self.required_plans)
        return (
            f"Environment Pools requires a {plans} plan. "
            f"Your current plan is '{self.current_plan}'. "
            f"Upgrade at {self.upgrade_url}"
        )


@dataclass
class PaymentRequiredError(HTTPError):
    """Raised when an endpoint requires x402 payment before retry."""

    challenge: Any | None = None
    payment_required_header: str | None = None
    payment_response_header: str = "PAYMENT-RESPONSE"

    @classmethod
    def from_http_error(cls, error: HTTPError) -> PaymentRequiredError:
        challenge, header_value = _extract_x402_payload(error.detail, error.body_snippet)
        return cls(
            status=error.status,
            url=error.url,
            message=error.message,
            body_snippet=error.body_snippet,
            detail=error.detail,
            challenge=challenge,
            payment_required_header=header_value,
        )

    def build_payment_response_header(self, *, payment_reference: str) -> str:
        """Build PAYMENT-RESPONSE header value for the x402 mock implementation."""
        payload = {
            "challenge": self.challenge,
            "payment_reference": payment_reference,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


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
    "PlanGatingError",
    "PaymentRequiredError",
]
