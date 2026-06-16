"""Package error types."""

from __future__ import annotations

from typing import Any


class SmrApiError(RuntimeError):
    """Raised when the Managed Research API returns an error response.

    When the backend returns a structured body (``{failure_class, remediation,
    cause, ...}``) the SDK preserves it on the exception so drivers can show
    the *class* of failure (e.g. ``db_schema_missing`` + the missing
    relation), not just a bare ``500``.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        failure_class: str | None = None,
        remediation: str | None = None,
        cause: list[dict[str, Any]] | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.failure_class = failure_class
        self.remediation = remediation
        self.cause_chain = list(cause) if cause else []
        self.body: dict[str, Any] = dict(body) if body else {}

    def __str__(self) -> str:  # noqa: D401 - simple stringer
        base = super().__str__()
        if not (self.failure_class or self.remediation or self.cause_chain):
            return base
        parts: list[str] = [base]
        if self.failure_class:
            parts.append(f"failure_class={self.failure_class}")
        if self.cause_chain:
            top = self.cause_chain[0]
            parts.append(f"cause[0]={top.get('type')}({top.get('module')}): {top.get('message')!r}")
            if len(self.cause_chain) > 1:
                parts.append(f"cause_chain_depth={len(self.cause_chain)}")
        for key in (
            "missing_object_name",
            "missing_object_kind",
            "constraint_name",
            "constraint_kind",
            "table",
            "column",
        ):
            value = self.body.get(key)
            if value:
                parts.append(f"{key}={value}")
        if self.remediation:
            parts.append(f"remediation={self.remediation}")
        return " | ".join(parts)


class UnsupportedProvider(ValueError):  # noqa: N818 - public compatibility name
    """Raised when an SDK helper only supports one provider for v1."""


class FeatureGated(SmrApiError):  # noqa: N818 - public compatibility name
    """Raised when a requested capability is behind a backend feature gate."""

    def __init__(
        self,
        feature: str,
        message: str | None = None,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message or f"Feature is not enabled: {feature}",
            status_code=status_code,
            response_text=response_text,
            body=detail,
        )
        self.feature = feature
        self.detail = dict(detail) if detail else {}


class SmrLimitExceededError(SmrApiError):
    """Raised when the backend rejects a request with ``smr_limit_exceeded`` (HTTP 429)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrFundingLaneInvariantError(SmrApiError):
    """Raised when the backend rejects with ``smr_free_tier_routing_violation`` (HTTP 409)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrInsufficientCreditsError(SmrApiError):
    """Raised when run start is blocked for credit headroom (HTTP 402, ``smr_insufficient_credits``)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrProjectMonthlyBudgetExhaustedError(SmrApiError):
    """Raised when the project monthly budget is exhausted (HTTP 402, ``smr_project_monthly_budget_exhausted``)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrManagedInferenceUnavailableError(SmrApiError):
    """Raised when managed Nemotron (or similar) is unreachable at run materialization (HTTP 503)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrCheckpointQuotaExceededError(SmrApiError):
    """Raised when checkpoint storage quota prevents restore or branch recovery."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}


class SmrConcurrentRunLimitExceededError(SmrApiError):
    """Raised when the org has reached the concurrent run limit for their plan (HTTP 429)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_text=response_text)
        self.detail = dict(detail) if detail else {}
        self.concurrent_limit: int | None = self.detail.get("concurrent_limit")
        self.current_concurrent: int | None = self.detail.get("current_concurrent")


class SmrStructuredDenialError(SmrApiError):
    """Raised for other JSON error bodies that include a string ``error_code`` (forward-compatible)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            body=detail,
        )
        self.detail = dict(detail) if detail else {}


ManagedResearchError = SmrApiError


__all__ = [
    "ManagedResearchError",
    "SmrApiError",
    "SmrCheckpointQuotaExceededError",
    "SmrConcurrentRunLimitExceededError",
    "SmrFundingLaneInvariantError",
    "SmrInsufficientCreditsError",
    "SmrLimitExceededError",
    "SmrManagedInferenceUnavailableError",
    "SmrProjectMonthlyBudgetExhaustedError",
    "SmrStructuredDenialError",
]
