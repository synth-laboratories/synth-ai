"""HTTP transport helpers for the Managed Research SDK.

Maps backend error bodies to typed SDK exceptions; non-JSON bodies fall back to
generic messages rather than failing the transport with a parse error.

# See: Synth Style — translate at the edge; preserve causes with ``from exc``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from synth_ai.core.contracts.json_value import JsonValue
from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.errors import (
    SmrApiError,
    SmrCheckpointQuotaExceededError,
    SmrConcurrentRunLimitExceededError,
    SmrFundingLaneInvariantError,
    SmrInferenceProviderUnavailableError,
    SmrInsufficientCreditsError,
    SmrLimitExceededError,
    SmrManagedInferenceUnavailableError,
    SmrProjectMonthlyBudgetExhaustedError,
    SmrStructuredDenialError,
)

# Backend billing-admission blocker codes for the wallet/allowance family
# (services/smr/billing/preflight.py + admission.py). These arrive as HTTP 402
# ``{"detail": {"error_code": ..., "billing_preflight": ...}}`` bodies and mean
# the org cannot fund the request — the SDK's insufficient-credits condition.
# ``factory_*`` budget codes are deliberately excluded: factory budget
# exhaustion is a policy denial on the Factory effort, not an org credit state,
# and stays a structured denial.
_INSUFFICIENT_CREDITS_CODES = frozenset(
    {
        "smr_insufficient_credits",
        "smr_allowance_and_wallet_insufficient",
        "smr_allowance_and_wallet_exhausted",
        "smr_wallet_exhausted",
        "smr_allowance_manual_reset_required",
        "smr_billing_blocked",
    }
)
# Backend emits per-model-class / per-window variants (e.g.
# ``smr_allowance_unprovisioned_premium``); its own blocker classifier
# (admission.py::is_billing_admission_blocker) matches these prefixes.
_INSUFFICIENT_CREDITS_CODE_PREFIXES = (
    "smr_allowance_unprovisioned_",
    "smr_window_exhausted_",
)


def _is_insufficient_credits_code(code: str) -> bool:
    if code in _INSUFFICIENT_CREDITS_CODES:
        return True
    return code.startswith(_INSUFFICIENT_CREDITS_CODE_PREFIXES)


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        # New structured shape: top-level {failure_class, message, remediation, cause}
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, dict):
            msg = detail.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            err = detail.get("error")
            if isinstance(err, str) and err.strip():
                return err.strip()
    return (
        f"{response.request.method} {response.request.url.path} failed with {response.status_code}"
    )


def _structured_body_fields(response: httpx.Response) -> dict[str, Any]:
    """Extract failure_class / remediation / cause from a backend error body.

    Looks at both the new top-level shape (``{failure_class, remediation, cause, ...}``)
    and the legacy ``{detail: {error, error_code, message, ...}}`` shape.
    """
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "failure_class",
        "remediation",
        "cause",
        "missing_object_name",
        "missing_object_kind",
        "constraint_name",
        "constraint_kind",
        "table",
        "column",
        "error",
        "error_code",
    ):
        value = payload.get(key)
        if value is not None:
            out[key] = value
    detail = payload.get("detail")
    if isinstance(detail, dict):
        for key in ("error_code", "error", "message", "remediation"):
            value = detail.get(key)
            if value is not None and key not in out:
                out[key] = value
    if "failure_class" not in out and isinstance(out.get("error"), str):
        out["failure_class"] = out["error"]
    return {"raw_body": payload, **out}


def _raise_for_error_response(
    response: httpx.Response,
    operation_id: str | None = None,
) -> None:
    """Map FastAPI ``{"detail": {"error_code": ...}}`` bodies to typed SDK errors."""
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            code = detail.get("error_code")
            if isinstance(code, str) and code.strip():
                message = _error_message(response)
                status_code = response.status_code
                response_text = response.text
                stripped = code.strip()
                if stripped == "smr_limit_exceeded":
                    raise SmrLimitExceededError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped in {
                    "smr_concurrent_run_limit_exceeded",
                    "smr_launch_promo_concurrent_limit",
                }:
                    raise SmrConcurrentRunLimitExceededError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped == "smr_free_tier_routing_violation":
                    raise SmrFundingLaneInvariantError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if _is_insufficient_credits_code(stripped):
                    raise SmrInsufficientCreditsError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped == "smr_project_monthly_budget_exhausted":
                    raise SmrProjectMonthlyBudgetExhaustedError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped == "inference_provider_unavailable":
                    raise SmrInferenceProviderUnavailableError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped in {
                    "smr_managed_inference_unavailable",
                    "smr_managed_inference_upstream_unavailable",
                }:
                    raise SmrManagedInferenceUnavailableError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                if stripped == "checkpoint_storage_quota_exceeded":
                    raise SmrCheckpointQuotaExceededError(
                        message,
                        status_code=status_code,
                        response_text=response_text,
                        detail=detail,
                    )
                raise SmrStructuredDenialError(
                    message,
                    status_code=status_code,
                    response_text=response_text,
                    detail=detail,
                )
            structured = _structured_body_fields(response)
            raise SmrApiError(
                _error_message(response),
                status_code=response.status_code,
                response_text=response.text,
                failure_class=structured.get("failure_class"),
                remediation=structured.get("remediation"),
                cause=structured.get("cause")
                if isinstance(structured.get("cause"), list)
                else None,
                body=structured.get("raw_body")
                if isinstance(structured.get("raw_body"), dict)
                else None,
            )
    structured = _structured_body_fields(response)
    raise SmrApiError(
        _error_message(response),
        status_code=response.status_code,
        response_text=response.text,
        failure_class=structured.get("failure_class"),
        remediation=structured.get("remediation"),
        cause=structured.get("cause") if isinstance(structured.get("cause"), list) else None,
        body=structured.get("raw_body") if isinstance(structured.get("raw_body"), dict) else None,
    )


def _raise_for_transport_exception(
    method: str,
    path: str,
    error: httpx.HTTPError,
    operation_id: str | None = None,
) -> None:
    if isinstance(error, httpx.TimeoutException):
        raise SmrApiError(f"{method} {path} timed out") from error
    raise SmrApiError(f"{method} {path} failed: network error ({type(error).__name__})") from error


def _raise_for_decode_error(
    method: str,
    path: str,
    response: httpx.Response,
    error: Exception,
    operation_id: str | None = None,
) -> None:
    raise SmrApiError(
        f"{method} {path} returned a non-JSON response",
        status_code=response.status_code,
        response_text=response.text,
    ) from error


class SmrHttpTransport(HttpTransport):
    """The Research transport: core's HttpTransport wired to Research error mapping.

    Every ``ResearchSession`` request goes through this.
    """

    def __init__(self, *, base_url: str, headers: dict[str, str], timeout: float) -> None:
        super().__init__(
            base_url=base_url,
            headers=headers,
            timeout_seconds=timeout,
            error_handler=_raise_for_error_response,
            exception_handler=_raise_for_transport_exception,
            decode_error_handler=_raise_for_decode_error,
        )
        self.timeout = timeout

    def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> Iterator[SseEvent]:
        try:
            yield from super().stream_sse(
                path,
                params=params,
                last_event_id=last_event_id,
                timeout_seconds=timeout_seconds,
                operation_id=operation_id,
            )
        except SmrApiError as error:
            if isinstance(error.__cause__, httpx.TimeoutException):
                raise SmrApiError(f"GET {path} SSE stream timed out") from error.__cause__
            if isinstance(error.__cause__, httpx.TransportError):
                raise SmrApiError(
                    f"GET {path} SSE stream failed: network error "
                    f"({type(error.__cause__).__name__})"
                ) from error.__cause__
            raise


__all__ = ["SmrHttpTransport"]
