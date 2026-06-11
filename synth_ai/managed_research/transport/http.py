"""HTTP transport helpers for the Managed Research SDK.

Maps backend error bodies to typed SDK exceptions; non-JSON bodies fall back to
generic messages rather than failing the transport with a parse error.

# See: Synth Style — translate at the edge; preserve causes with ``from exc``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from synth_ai.managed_research.errors import (
    SmrApiError,
    SmrCheckpointQuotaExceededError,
    SmrConcurrentRunLimitExceededError,
    SmrFundingLaneInvariantError,
    SmrInsufficientCreditsError,
    SmrLimitExceededError,
    SmrManagedInferenceUnavailableError,
    SmrProjectMonthlyBudgetExhaustedError,
    SmrStructuredDenialError,
)
from synth_ai.managed_research.transport.streaming import SseEvent, iter_sse_events


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


def _raise_for_error_response(response: httpx.Response) -> None:
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
                if stripped == "smr_concurrent_run_limit_exceeded":
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
                if stripped == "smr_insufficient_credits":
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
                if stripped == "smr_managed_inference_unavailable":
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
                cause=structured.get("cause") if isinstance(structured.get("cause"), list) else None,
                body=structured.get("raw_body") if isinstance(structured.get("raw_body"), dict) else None,
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


@dataclass
class SmrHttpTransport:
    """Simple JSON HTTP transport used by the rewritten public client."""

    base_url: str
    headers: dict[str, str]
    timeout: float
    client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=self.headers,
            timeout=self.timeout,
        )

    def close(self) -> None:
        self.client.close()

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        try:
            response = self.client.request(
                method,
                path,
                params=params,
                json=json_body,
            )
        except httpx.TimeoutException as exc:
            raise SmrApiError(f"{method} {path} timed out") from exc
        except httpx.TransportError as exc:
            raise SmrApiError(
                f"{method} {path} failed: network error ({type(exc).__name__})"
            ) from exc
        if allow_not_found and response.status_code == 404:
            return None
        if response.is_error:
            _raise_for_error_response(response)
        if not response.content:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SmrApiError(
                f"{method} {path} returned a non-JSON response",
                status_code=response.status_code,
                response_text=response.text,
            ) from exc

    def stream_sse(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        last_event_id: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[SseEvent]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            with self.client.stream(
                "GET",
                path,
                params=params,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.is_error:
                    response.read()
                    _raise_for_error_response(response)
                yield from iter_sse_events(response.iter_lines())
        except httpx.TimeoutException as exc:
            raise SmrApiError(f"GET {path} SSE stream timed out") from exc
        except httpx.TransportError as exc:
            raise SmrApiError(
                f"GET {path} SSE stream failed: network error ({type(exc).__name__})"
            ) from exc


__all__ = ["SmrHttpTransport"]
