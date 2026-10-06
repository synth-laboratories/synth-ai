"""Research API errors (public ``Research*`` names).

Catch these typed exceptions from ``SynthClient().research`` call sites.

The legacy ``Smr*`` names are deprecated aliases of the ``Research*`` classes.
They remain importable for compatibility but emit a ``DeprecationWarning`` on
first access and will be removed in a future release.

| Exception | Typical cause |
| --- | --- |
| ``ResearchApiError`` | Base API error with structured ``message`` |
| ``ResearchStructuredDenialError`` | Policy or preflight denial |
| ``ResearchLimitExceededError`` | Org or project limit exceeded |
| ``ResearchConcurrentRunLimitExceededError`` | Too many concurrent runs |
| ``ResearchInsufficientCreditsError`` | Insufficient account credits |
| ``ResearchProjectMonthlyBudgetExhaustedError`` | Project monthly budget exhausted |
| ``ResearchInferenceProviderUnavailableError`` | Upstream provider temporarily unavailable |
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from synth_ai.core.errors import (
    AuthorizationError,
    ConflictError,
    ContractMismatchError,
    RateLimitedError,
    ResearchOperationError,
    ResourceExhaustedError,
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
    TransientServiceError,
)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ResearchApiError(SynthError, RuntimeError):
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
        failure: SynthFailure | None = None,
        operation_id: str | None = None,
    ) -> None:
        super().__init__(message, failure=failure)
        self.operation_id = operation_id
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


class FeatureGated(ResearchApiError):  # noqa: N818 - public compatibility name
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


class ResearchLimitExceededError(ResearchApiError):
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


class ResearchHostedModelOverridesError(ResearchApiError):
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


class ResearchFundingLaneInvariantError(ResearchApiError):
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


class ResearchInsufficientCreditsError(ResearchApiError):
    """Run start lacks credit headroom (402, ``smr_insufficient_credits``)."""

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


class ResearchProjectMonthlyBudgetExhaustedError(ResearchApiError):
    """Project budget exhausted (402, ``smr_project_monthly_budget_exhausted``)."""

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


class ResearchInferenceProviderUnavailableError(ResearchApiError):
    """Raised for a retryable, durable upstream inference-provider failure."""

    error_code = "inference_provider_unavailable"

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
            failure_class=self.error_code,
            body=detail,
        )
        self.detail = dict(detail) if detail else {}
        route = self.detail.get("route")
        self.route = dict(route) if isinstance(route, Mapping) else {}
        self.provider = (
            str(self.detail.get("provider") or self.route.get("provider") or "").strip() or None
        )
        self.model = str(self.detail.get("model") or self.route.get("model") or "").strip() or None
        self._provider_retryable = bool(self.detail.get("retryable", True))
        self.upstream_status = _optional_int(self.detail.get("upstream_status"))
        self._provider_retry_after_seconds = _optional_int(self.detail.get("retry_after_seconds"))

    @property
    def retryable(self) -> bool:
        """Whether the durable provider fact says a fresh attempt may succeed."""

        return self._provider_retryable

    @property
    def retry_after_seconds(self) -> int | None:
        """Provider retry delay when the durable fact supplies one."""

        return self._provider_retry_after_seconds


class ResearchManagedInferenceUnavailableError(ResearchInferenceProviderUnavailableError):
    """Compatibility name for managed-inference provider unavailability."""


class ResearchCheckpointQuotaExceededError(ResearchApiError):
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


class ResearchConcurrentRunLimitExceededError(ResearchApiError):
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


class ResearchStructuredDenialError(ResearchApiError):
    """Forward-compatible JSON refusal carrying a string ``error_code``."""

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


class ResearchScientificRefusalError(ResearchApiError):
    """A definitive scientific rejection retains code, identity and retry policy.

    See forge_scientific_delivery.md. A scorer dependency refusal cannot become
    a status-based retry or an uncertain effect.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_text: str,
        detail: dict[str, Any],
        operation_id: str | None,
    ):
        code = detail["error_code"]
        operation_id = detail.get("operation_id") or operation_id
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            body=detail,
            operation_id=operation_id,
            failure=SynthFailure(
                code=SynthErrorCode(code),
                category=SynthErrorCategory.OPERATION,
                operation=operation_id,
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=status_code,
                detail=message,
            ),
        )
        self.detail = dict(detail)
        self.authority_code = detail.get("authority_code")
        self.receipt_lookup_required = code == "admission_expired"


class ResearchOutcomeUncertainError(ResearchApiError):
    """An original scientific operation needs receipt reconciliation.

    See forge_scientific_delivery.md. Never turn an uncertain effect into a
    deterministic denial or automatically submit a replacement operation.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        response_text: str,
        detail: dict[str, Any],
        operation_id: str | None,
    ):
        operation_id = detail.get("operation_id") or operation_id
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            body=detail,
            operation_id=operation_id,
            failure=SynthFailure(
                code=SynthErrorCode("outcome_uncertain"),
                category=SynthErrorCategory.OPERATION,
                operation=operation_id,
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=status_code,
                detail=message,
            ),
        )
        self.detail = dict(detail)
        self.receipt_lookup_required = True


LAUNCH_REFUSAL_CODES = frozenset(
    {
        "project_archived",
        "provenance_unbound",
        "run_provenance_mode_invalid",
        "run_deployment_pins_missing",
        "run_deployment_pin_invalid",
        "run_deployment_pin_placeholder",
        "run_trace_store_not_provisioned",
        "resource_inventory_unspecified",
        "resource_delivery_limit_exceeded",
        "native_budget_unbound",
        "native_role_budget_exceeds_run",
        "orchestra_plan_budget_unbound",
        "orchestra_plan_token_ceiling_unbound",
        "scientific_writer_transferred",
        "transfer_fence_active",
    }
)


class ResearchLaunchRefusalError(ResearchStructuredDenialError):
    """First-class launch refusal retaining authority, limits and retry posture.

    See backend launch_resource_inventory.md and packages/smr/run_provenance.py.
    A fence may require waiting; missing inputs and permanent writer changes do not retry.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, response_text=response_text, detail=detail
        )
        code = self.detail.get("error_code")
        if code not in LAUNCH_REFUSAL_CODES:
            raise ValueError("Unknown launch refusal code")
        retryable = self.detail.get("retryable", code == "transfer_fence_active")
        if not isinstance(retryable, bool):
            raise ValueError("launch refusal retryable must be a boolean")
        self.failure = SynthFailure(
            code=SynthErrorCode(code),
            category=(
                SynthErrorCategory.TRANSIENT_SERVICE
                if retryable
                else SynthErrorCategory.RESOURCE_EXHAUSTED
                if code in {"native_role_budget_exceeds_run", "resource_delivery_limit_exceeded"}
                else SynthErrorCategory.CONFLICT
                if code in {"scientific_writer_transferred", "project_archived"}
                else SynthErrorCategory.VALIDATION
            ),
            operation=operation_id,
            request_id=None,
            correlation_id=None,
            retry=RetryDirective(retryable=retryable),
            status=status_code,
            detail=message,
        )
        self.retry_after = self.detail.get("retry_after")
        self.operation_id = self.detail.get("operation_id", operation_id)


class ResearchNotFoundError(ResearchStructuredDenialError):
    """Raised when the backend reports a typed ``*_not_found`` condition (HTTP 404).

    The backend scopes every Research Intern lookup to the organization bound
    to the caller's API key, so a 404 only proves the resource is absent *for
    that organization* -- it never proves global absence. Release evidence must
    be able to distinguish a genuine miss from a wrong-organization lookup, so
    this error preserves the backend's typed condition instead of collapsing it
    into an opaque denial:

    - ``backend_error_code``: the exact backend condition, for example
      ``intern_async_runtime_not_found``, ``intern_sync_session_not_found``,
      or a retention condition such as
      ``intern_async_runtime_retention_expired`` /
      ``intern_acceptance_fixture_retention_expired`` (the resource existed;
      its read-only retention window ended). (Named to avoid shadowing the
      read-only ``SynthError.error_code`` transport-failure property.)
    - ``resource``: the resource segment of the condition (``async_runtime``,
      ``sync_session``, ``acceptance_fixture``, ...), or ``None`` when the
      code has neither the ``intern_*_not_found`` nor the
      ``intern_*_retention_expired`` shape.
    - ``scope_identifier``: the lookup key the backend echoed back (the
      organization id for org-singleton lookups such as the Async Intern, the
      resource id otherwise), when the backend provided one.

    Evidence that records ``backend_error_code`` + ``scope_identifier`` next to
    the caller's organization binding can attribute the miss: a not-found under
    the expected organization is a true absence for that organization, while a
    mismatch between the expected resource's organization and the caller's
    binding identifies a wrong-organization lookup.
    """

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
            detail=detail,
        )
        code = self.detail.get("error_code")
        self.backend_error_code: str = code.strip() if isinstance(code, str) else ""
        resource: str | None = None
        if self.backend_error_code.startswith("intern_"):
            for suffix in ("_not_found", "_retention_expired"):
                if self.backend_error_code.endswith(suffix):
                    resource = self.backend_error_code.removeprefix("intern_").removesuffix(suffix)
                    break
        self.resource: str | None = resource
        scope: str | None = None
        for key in ("runtime_id", "resource_id", "fixture_id", "async_runtime_id", "org_id"):
            value = self.detail.get(key)
            if isinstance(value, str) and value.strip():
                scope = value.strip()
                break
        self.scope_identifier: str | None = scope


class ResearchLimitExtensionError(ResearchApiError):
    """Base class for durable run-limit extension refusals."""

    error_code = "limit_extension_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        exact_detail = dict(detail) if detail else {}
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
            failure_class=self.error_code,
            body=exact_detail,
        )
        self.detail = exact_detail
        self._limit_extension_retryable = bool(exact_detail.get("retryable", False))
        self.refusal_receipt_id = exact_detail.get("refusal_receipt_id")

    @property
    def retryable(self) -> bool:
        """Whether the durable refusal says a fresh attempt may succeed."""

        return self._limit_extension_retryable


class ResearchLimitRevisionConflictError(ResearchLimitExtensionError):
    """Raised when ``expected_revision`` no longer names the current cap."""

    error_code = "limit_revision_conflict"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.expected_revision = _optional_int(self.detail.get("expected_revision"))
        self.current_revision = _optional_int(self.detail.get("current_revision"))


class ResearchLimitExtensionIdempotencyConflictError(ResearchLimitExtensionError):
    """Raised when an idempotency key is replayed with different inputs."""

    error_code = "limit_extension_idempotency_conflict"


class ResearchLimitExtensionGuardedResumeBlockedError(ResearchLimitExtensionError):
    """Raised when a requested guarded blocker release or resume is unavailable."""

    error_code = "limit_extension_guarded_action_not_enabled"


class ResearchUnsafeLimitExtensionError(ResearchLimitExtensionError):
    """Raised when extending the selected cap would not be safe."""

    error_code = "limit_extension_not_safe"


class CloudDeploymentClaimError(ResearchApiError):
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
    """Active claim requires ``X-Fencing-Token`` (409, ``fencing_token_required``)."""


class FencingTokenStaleError(CloudDeploymentClaimError):
    """Presented token was superseded (409, ``fencing_token_stale``)."""


_CLAIM_REASON_ERRORS: dict[str, type[CloudDeploymentClaimError]] = {
    "claim_expired": ClaimExpiredError,
    "claim_superseded": ClaimSupersededError,
    "fencing_token_required": FencingTokenRequiredError,
    "fencing_token_stale": FencingTokenStaleError,
}

_CLAIM_CONFLICT_REASON_PREFIX = "claim_conflict"


def _claim_reason_candidates(exc: ResearchApiError) -> list[str]:
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


def raise_cloud_deployment_claim_error(exc: ResearchApiError) -> None:
    """Re-raise ``exc`` as a typed claim/fencing error when its named reason is recognized.

    Only 409/410 responses are inspected; unrecognized reasons return so the
    caller can re-raise the original ``ResearchApiError`` unchanged. Reasons are read
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


# Deprecated ``Smr*`` -> ``Research*`` aliases, resolved lazily via module
# ``__getattr__`` so any access emits a DeprecationWarning (once per name).
_DEPRECATED_SMR_ALIASES: dict[str, type[ResearchApiError]] = {
    "SmrApiError": ResearchApiError,
    "SmrCheckpointQuotaExceededError": ResearchCheckpointQuotaExceededError,
    "SmrConcurrentRunLimitExceededError": ResearchConcurrentRunLimitExceededError,
    "SmrFundingLaneInvariantError": ResearchFundingLaneInvariantError,
    "SmrHostedModelOverridesError": ResearchHostedModelOverridesError,
    "SmrInsufficientCreditsError": ResearchInsufficientCreditsError,
    "SmrLimitExceededError": ResearchLimitExceededError,
    "SmrLimitExtensionError": ResearchLimitExtensionError,
    "SmrLimitExtensionGuardedResumeBlockedError": (ResearchLimitExtensionGuardedResumeBlockedError),
    "SmrLimitExtensionIdempotencyConflictError": (ResearchLimitExtensionIdempotencyConflictError),
    "SmrLimitRevisionConflictError": ResearchLimitRevisionConflictError,
    "SmrUnsafeLimitExtensionError": ResearchUnsafeLimitExtensionError,
    "SmrInferenceProviderUnavailableError": ResearchInferenceProviderUnavailableError,
    "SmrManagedInferenceUnavailableError": ResearchManagedInferenceUnavailableError,
    "SmrProjectMonthlyBudgetExhaustedError": ResearchProjectMonthlyBudgetExhaustedError,
    "SmrStructuredDenialError": ResearchStructuredDenialError,
}


def __getattr__(name: str) -> type[ResearchApiError]:
    try:
        replacement = _DEPRECATED_SMR_ALIASES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    warnings.warn(
        f"{name} is deprecated; use {replacement.__name__} instead; "
        "Smr aliases will be removed in a future release",
        DeprecationWarning,
        stacklevel=2,
    )
    # Cache so the warning fires once per name and later lookups skip __getattr__.
    globals()[name] = replacement
    return replacement


__all__ = [
    "AuthorizationError",
    "ClaimConflictError",
    "ClaimExpiredError",
    "ClaimSupersededError",
    "CloudDeploymentClaimError",
    "ConflictError",
    "ContractMismatchError",
    "FeatureGated",
    "FencingTokenRequiredError",
    "FencingTokenStaleError",
    "RateLimitedError",
    "ResearchApiError",
    "ResearchOutcomeUncertainError",
    "ResearchScientificRefusalError",
    "ResearchCheckpointQuotaExceededError",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchFundingLaneInvariantError",
    "ResearchHostedModelOverridesError",
    "ResearchInsufficientCreditsError",
    "ResearchInferenceProviderUnavailableError",
    "ResearchLimitExceededError",
    "ResearchLimitExtensionError",
    "ResearchLimitExtensionGuardedResumeBlockedError",
    "ResearchLimitExtensionIdempotencyConflictError",
    "ResearchLimitRevisionConflictError",
    "ResearchManagedInferenceUnavailableError",
    "ResearchNotFoundError",
    "ResearchOperationError",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchStructuredDenialError",
    "ResearchLaunchRefusalError",
    "ResearchUnsafeLimitExtensionError",
    "ResourceExhaustedError",
    "RetryDirective",
    # Deprecated aliases served by module __getattr__ (invisible to static
    # analysis, hence the noqa markers).
    "SmrApiError",  # noqa: F822
    "SmrCheckpointQuotaExceededError",  # noqa: F822
    "SmrConcurrentRunLimitExceededError",  # noqa: F822
    "SmrFundingLaneInvariantError",  # noqa: F822
    "SmrHostedModelOverridesError",  # noqa: F822
    "SmrInsufficientCreditsError",  # noqa: F822
    "SmrInferenceProviderUnavailableError",  # noqa: F822
    "SmrLimitExceededError",  # noqa: F822
    "SmrLimitExtensionError",  # noqa: F822
    "SmrLimitExtensionGuardedResumeBlockedError",  # noqa: F822
    "SmrLimitExtensionIdempotencyConflictError",  # noqa: F822
    "SmrLimitRevisionConflictError",  # noqa: F822
    "SmrManagedInferenceUnavailableError",  # noqa: F822
    "SmrProjectMonthlyBudgetExhaustedError",  # noqa: F822
    "SmrStructuredDenialError",  # noqa: F822
    "SmrUnsafeLimitExtensionError",  # noqa: F822
    "SynthError",
    "SynthErrorCategory",
    "SynthErrorCode",
    "SynthFailure",
    "TransientServiceError",
    "UnsupportedProvider",
    "raise_cloud_deployment_claim_error",
]
