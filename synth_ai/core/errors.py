"""Consolidated error types for Synth AI SDK.

This module provides base exception classes used throughout the SDK.
CLI-specific errors remain in cli/ modules; these are for SDK/core use.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

HTTP_ERROR_BODY_CAPTURE_CHARS_MAX = 8192
_DISPLAY_MESSAGE_CHARS_MAX = 4096
_DISPLAY_BODY_CHARS_MAX = 4096
_DISPLAY_URL_CHARS_MAX = 1024
_SENSITIVE_NAME = (
    r"(?:[A-Za-z0-9]+[_-]+)*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password|authorization|secret|payment[_-]?signature|"
    r"payment[_-]?required[_-]?header)"
)
_SENSITIVE_FIELD = re.compile(
    r"""(?i)(?P<prefix>["']""" + _SENSITIVE_NAME + r"""["']\s*:\s*["'])"""
    r"""[^"']*(?P<suffix>["'])"""
)
_SENSITIVE_PARAMETER = re.compile(
    r"(?i)(?P<prefix>(?:[?&]|\b)" + _SENSITIVE_NAME + r"\s*[:=]\s*)"
    r"[^&#\s,;]+"
)
_AUTHORIZATION_HEADER = re.compile(r"(?i)\bauthorization\s*:\s*(?:Bearer|Basic)\s+[^\s\"',;]+")
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[^\s\"',;]+")
_SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_URL_USERINFO = re.compile(r"(?P<scheme>https?://)[^/@\s]+@")


def _safe_error_display(value: str, *, limit: int) -> str:
    """Bound and redact exception text before it enters logs or a terminal."""
    clean = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    clean = _SENSITIVE_FIELD.sub(r"\g<prefix>[REDACTED]\g<suffix>", clean)
    clean = _AUTHORIZATION_HEADER.sub("Authorization: [REDACTED]", clean)
    clean = _SENSITIVE_PARAMETER.sub(r"\g<prefix>[REDACTED]", clean)
    clean = _BEARER_TOKEN.sub("Bearer [REDACTED]", clean)
    clean = _SK_TOKEN.sub("[REDACTED]", clean)
    clean = _URL_USERINFO.sub(r"\g<scheme>[REDACTED]@", clean)
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]}…[truncated {len(clean) - limit} chars]"


class SynthErrorCode(str):
    """Stable server error code, preserving unknown future values."""


class SynthErrorCategory(StrEnum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    CONFLICT = "conflict"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RATE_LIMITED = "rate_limited"
    TRANSIENT_SERVICE = "transient_service"
    CONTRACT_MISMATCH = "contract_mismatch"
    OPERATION = "operation"


@dataclass(frozen=True, slots=True)
class RetryDirective:
    retryable: bool
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ResourceRef:
    kind: str
    resource_id: str


@dataclass(frozen=True, slots=True)
class SynthFailure:
    """Language-neutral failure record carried by every operation exception."""

    code: SynthErrorCode
    category: SynthErrorCategory
    operation: str | None
    request_id: str | None
    correlation_id: str | None
    retry: RetryDirective
    status: int | None
    detail: str
    resource: ResourceRef | None = None


class SynthError(Exception):
    """Base exception for all Synth AI SDK errors."""

    def __init__(
        self,
        message: str = "Synth operation failed",
        *,
        failure: SynthFailure | None = None,
    ) -> None:
        self.message = message
        self.failure = failure
        super().__init__(message)

    @property
    def error_code(self) -> SynthErrorCode | None:
        return self.failure.code if self.failure is not None else None

    @property
    def operation(self) -> str | None:
        return self.failure.operation if self.failure is not None else None

    @property
    def request_id(self) -> str | None:
        return self.failure.request_id if self.failure is not None else None

    @property
    def correlation_id(self) -> str | None:
        return self.failure.correlation_id if self.failure is not None else None

    @property
    def retryable(self) -> bool:
        return self.failure.retry.retryable if self.failure is not None else False

    @property
    def retry_after_seconds(self) -> float | None:
        return self.failure.retry.retry_after_seconds if self.failure is not None else None


class ConfigError(SynthError):
    """Raised when local Synth configuration exists but cannot be used."""


class AuthenticationError(SynthError):
    """Raised when API authentication fails."""


class HTTPError(SynthError):
    """Raised when an HTTP request fails."""

    def __init__(
        self,
        status: int,
        url: str,
        message: str,
        body_snippet: str | None = None,
        detail: Any | None = None,
        *,
        failure: SynthFailure | None = None,
    ) -> None:
        self.status = status
        self.url = url
        self.body_snippet = body_snippet
        self.detail = detail
        # Exception repr and args are display surfaces too. Retain the raw
        # message on the structured fields while keeping those surfaces safe.
        super().__init__(
            _safe_error_display(message, limit=_DISPLAY_MESSAGE_CHARS_MAX),
            failure=failure,
        )
        self.message = message

    def __str__(self) -> str:
        # Query strings and fragments are caller-controlled and may carry
        # credentials under names the redactor cannot anticipate.
        display_url = self.url.partition("?")[0].partition("#")[0]
        url = _safe_error_display(display_url, limit=_DISPLAY_URL_CHARS_MAX)
        message = _safe_error_display(self.message, limit=_DISPLAY_MESSAGE_CHARS_MAX)
        base = f"HTTP {self.status} for {url}: {message}"
        if self.body_snippet:
            body = _safe_error_display(self.body_snippet, limit=_DISPLAY_BODY_CHARS_MAX)
            base += f" | body_excerpt={body}"
        return base


class AuthorizationError(HTTPError):
    """The credential is valid but lacks authority for the operation."""


class ConflictError(HTTPError):
    """The requested mutation conflicts with authoritative resource state."""


class ResourceExhaustedError(HTTPError):
    """Budget, credit, or admitted capacity is exhausted."""


class RateLimitedError(HTTPError):
    """A rate limit denied the request and may include retry timing."""


class TransientServiceError(HTTPError):
    """A classified transient service failure safe to reconsider by policy."""


class ContractMismatchError(HTTPError):
    """The server response violates the versioned SDK contract."""


class ResearchOperationError(HTTPError):
    """A stable Research operation failed without a more specific category."""


def _extract_x402_payload(detail: Any, body_snippet: str | None) -> tuple[Any | None, str | None]:
    if isinstance(detail, dict):
        x402_value = detail.get("x402")
        if isinstance(x402_value, dict):
            payment_required = x402_value.get("payment_required")
            if payment_required is None:
                payment_required = x402_value.get("challenge")
            return payment_required, x402_value.get("payment_required_header")
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
    payment_required = x402_value.get("payment_required")
    if payment_required is None:
        payment_required = x402_value.get("challenge")
    return payment_required, x402_value.get("payment_required_header")


class TimeoutError(SynthError):
    """Raised when an operation times out."""

    pass


class PaymentRequiredError(HTTPError):
    """Raised when an endpoint requires x402 payment before retry."""

    payment_signature_header = "PAYMENT-SIGNATURE"
    strict_payment_signature_header = "X-PAYMENT"
    payment_response_header = "PAYMENT-RESPONSE"

    def __init__(
        self,
        status: int,
        url: str,
        message: str,
        body_snippet: str | None = None,
        detail: Any | None = None,
        *,
        failure: SynthFailure | None = None,
        challenge: Any | None = None,
        payment_required_header: str | None = None,
    ) -> None:
        self.challenge = challenge
        self.payment_required_header = payment_required_header
        super().__init__(
            status,
            url,
            message,
            body_snippet,
            detail,
            failure=failure,
        )

    @classmethod
    def from_http_error(cls, error: HTTPError) -> PaymentRequiredError:
        challenge, header_value = _extract_x402_payload(error.detail, error.body_snippet)
        return cls(
            status=error.status,
            url=error.url,
            message=error.message,
            body_snippet=error.body_snippet,
            detail=error.detail,
            failure=error.failure,
            challenge=challenge,
            payment_required_header=header_value,
        )

    def build_payment_response_header(self, *, payment_reference: str) -> str:
        """Build PAYMENT-RESPONSE header value for the canonical x402 mock implementation.

        Note: Real x402 clients should send a `PAYMENT-SIGNATURE` header, not `PAYMENT-RESPONSE`.
        """
        payload = {
            "challenge": self.challenge,
            "payment_reference": payment_reference,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


__all__ = [
    "SynthError",
    "SynthErrorCategory",
    "SynthErrorCode",
    "SynthFailure",
    "RetryDirective",
    "ResourceRef",
    "AuthenticationError",
    "ConfigError",
    "HTTPError",
    "AuthorizationError",
    "ConflictError",
    "ResourceExhaustedError",
    "RateLimitedError",
    "TransientServiceError",
    "ContractMismatchError",
    "ResearchOperationError",
    "TimeoutError",
    "PaymentRequiredError",
]
