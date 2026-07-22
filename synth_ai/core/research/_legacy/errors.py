"""Package error types."""

from __future__ import annotations

from typing import Any

from synth_ai.core.errors import SynthError


class SmrApiError(SynthError, RuntimeError):
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
        self.request_context: str | None = None
        self.failure_class = failure_class
        self.remediation = remediation
        self.cause_chain = list(cause) if cause else []
        self.body: dict[str, Any] = dict(body) if body else {}

    def __str__(self) -> str:  # noqa: D401 - simple stringer
        base = super().__str__()
        request_context = self.request_context
        if request_context and request_context not in base:
            base = f"{request_context}: {base}"
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


class SmrHostedModelOverridesError(SmrApiError):
    """Raised client-side when a hosted launch carries actor model overrides.

    On hosted launches (no ``local_execution``) the platform resolves actor
    harness/model/profile, so ``agent_model`` and related override kwargs are not
    permitted. This mirrors the backend 422 ``model_overrides_not_supported_on_hosted``
    gate and fails fast before any HTTP request. Local launches keep full control.
    """

    error_code = "model_overrides_not_supported_on_hosted"

    def __init__(
        self,
        message: str,
        *,
        rejected_fields: list[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class=self.error_code)
        self.rejected_fields = list(rejected_fields) if rejected_fields else []
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


class CloudDeploymentClaimError(SmrApiError):
    """Base for typed CloudDeployment claim/fencing denials (named 409/410 reasons).

    ``reason`` carries the backend's named reason string verbatim (e.g.
    ``claim_conflict:worker-a``, ``claim_expired``, ``fencing_token_stale``).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            failure_class=reason,
            body=detail,
        )
        self.reason = reason
        self.detail = dict(detail) if detail else {}


class ClaimConflictError(CloudDeploymentClaimError):
    """Acquire refused: another holder owns the claim (HTTP 409, ``claim_conflict:<holder>``)."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        holder: str | None = None,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason=reason,
            status_code=status_code,
            response_text=response_text,
            detail=detail,
        )
        self.holder = holder


class ClaimExpiredError(CloudDeploymentClaimError):
    """Heartbeat refused: the claim's TTL lapsed (HTTP 410, ``claim_expired``)."""


class ClaimSupersededError(CloudDeploymentClaimError):
    """Heartbeat refused: a newer claim replaced this one (HTTP 409, ``claim_superseded``)."""


class FencingTokenRequiredError(CloudDeploymentClaimError):
    """Mutating op refused: an active claim requires ``X-Fencing-Token`` (HTTP 409, ``fencing_token_required``)."""


class FencingTokenStaleError(CloudDeploymentClaimError):
    """Mutating op refused: the presented fencing token was superseded (HTTP 409, ``fencing_token_stale``)."""


_CLAIM_REASON_ERRORS: dict[str, type[CloudDeploymentClaimError]] = {
    "claim_expired": ClaimExpiredError,
    "claim_superseded": ClaimSupersededError,
    "fencing_token_required": FencingTokenRequiredError,
    "fencing_token_stale": FencingTokenStaleError,
}

_CLAIM_CONFLICT_REASON_PREFIX = "claim_conflict"


def _claim_reason_candidates(exc: SmrApiError) -> list[str]:
    candidates: list[Any] = [exc.failure_class]
    for source in (exc.body, getattr(exc, "detail", None)):
        if isinstance(source, dict):
            candidates.extend(source.get(key) for key in ("error_code", "error", "reason"))
            nested = source.get("detail")
            if isinstance(nested, dict):
                candidates.extend(nested.get(key) for key in ("error_code", "error", "reason"))
    normalized: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized.append(candidate.strip())
    return normalized


def raise_cloud_deployment_claim_error(exc: SmrApiError) -> None:
    """Re-raise ``exc`` as a typed claim/fencing error when its named reason is recognized.

    Only 409/410 responses are inspected; unrecognized reasons return so the
    caller can re-raise the original ``SmrApiError`` unchanged. Reasons are read
    from the structured error body (``failure_class`` / ``error_code`` /
    ``error`` / ``reason``), never phrase-matched from prose.
    """

    if exc.status_code not in (409, 410):
        return
    message = str(exc)
    detail = dict(getattr(exc, "detail", None) or exc.body or {})
    for reason in _claim_reason_candidates(exc):
        if reason == _CLAIM_CONFLICT_REASON_PREFIX or reason.startswith(
            f"{_CLAIM_CONFLICT_REASON_PREFIX}:"
        ):
            holder = reason.partition(":")[2].strip() or None
            raise ClaimConflictError(
                message,
                reason=reason,
                holder=holder,
                status_code=exc.status_code,
                response_text=exc.response_text,
                detail=detail,
            ) from exc
        error_cls = _CLAIM_REASON_ERRORS.get(reason)
        if error_cls is not None:
            raise error_cls(
                message,
                reason=reason,
                status_code=exc.status_code,
                response_text=exc.response_text,
                detail=detail,
            ) from exc


ManagedResearchError = SmrApiError


__all__ = [
    "ClaimConflictError",
    "ClaimExpiredError",
    "ClaimSupersededError",
    "CloudDeploymentClaimError",
    "FencingTokenRequiredError",
    "FencingTokenStaleError",
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
    "raise_cloud_deployment_claim_error",
]
