"""Dependency-clean synchronous HTTP transport.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import NoReturn, cast

import httpx

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.errors import (
    AuthorizationError,
    ConflictError,
    ContractMismatchError,
    HTTPError,
    PaymentRequiredError,
    RateLimitedError,
    ResearchOperationError,
    ResourceExhaustedError,
    RetryDirective,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
    TimeoutError,
    TransientServiceError,
)
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.streaming import SseEvent, iter_sse_events


ErrorHandler = Callable[[httpx.Response, str | None], NoReturn]
ExceptionHandler = Callable[[str, str, httpx.HTTPError, str | None], NoReturn]
DecodeErrorHandler = Callable[[str, str, httpx.Response, Exception, str | None], NoReturn]


def _decode_json_value(value: object, *, context: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_decode_json_value(item, context=context) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{context} JSON object contains a non-string key")
        return {
            str(key): _decode_json_value(item, context=context)
            for key, item in value.items()
        }
    raise ValueError(f"{context} is not a JSON value")


def raise_http_error(response: httpx.Response, operation_id: str | None = None) -> NoReturn:
    """Translate one failed response into the stable SDK failure hierarchy."""

    detail: JsonValue | None = None
    message = f"{response.request.method} {response.request.url.path} failed"
    try:
        decoded = _decode_json_value(response.json(), context="error response")
    except (json.JSONDecodeError, ValueError):
        decoded = None
    if isinstance(decoded, dict):
        detail = decoded
        raw_message = decoded.get("message")
        if isinstance(raw_message, str) and raw_message.strip():
            message = raw_message.strip()
        nested = decoded.get("detail")
        if isinstance(nested, str) and nested.strip():
            message = nested.strip()
        elif isinstance(nested, dict):
            nested_message = nested.get("message") or nested.get("error")
            if isinstance(nested_message, str) and nested_message.strip():
                message = nested_message.strip()
    failure = _failure_from_response(
        response,
        decoded=detail,
        message=message,
        operation_id=operation_id,
    )
    error_type: type[HTTPError]
    if response.status_code == 402:
        error_type = ResourceExhaustedError
    elif response.status_code == 403:
        error_type = AuthorizationError
    elif response.status_code == 409:
        error_type = ConflictError
    elif response.status_code == 429:
        error_type = RateLimitedError
    elif response.status_code in {500, 502, 503, 504}:
        error_type = TransientServiceError
    else:
        error_type = ResearchOperationError
    error = error_type(
        status=response.status_code,
        url=str(response.request.url),
        message=message,
        body_snippet=response.text[:200] or None,
        detail=detail,
        failure=failure,
    )
    if response.status_code == 402:
        raise PaymentRequiredError.from_http_error(error)
    raise error


def _failure_from_response(
    response: httpx.Response,
    *,
    decoded: JsonValue | None,
    message: str,
    operation_id: str | None,
) -> SynthFailure:
    detail = decoded.get("detail") if isinstance(decoded, dict) else None
    sources = tuple(item for item in (detail, decoded) if isinstance(item, dict))
    error_code = next(
        (
            value
            for source in sources
            for key in ("error_code", "code", "error")
            if isinstance((value := source.get(key)), str) and value.strip()
        ),
        f"http_{response.status_code}",
    )
    request_id = _response_identity(response, sources, "request_id", "x-request-id")
    correlation_id = _response_identity(
        response,
        sources,
        "correlation_id",
        "x-correlation-id",
    )
    retry_after_seconds = _retry_after_seconds(response)
    category = _error_category(response.status_code)
    return SynthFailure(
        code=SynthErrorCode(error_code),
        category=category,
        operation=operation_id,
        request_id=request_id,
        correlation_id=correlation_id,
        retry=RetryDirective(
            retryable=category
            in {SynthErrorCategory.RATE_LIMITED, SynthErrorCategory.TRANSIENT_SERVICE},
            retry_after_seconds=retry_after_seconds,
        ),
        status=response.status_code,
        detail=message,
    )


def _response_identity(
    response: httpx.Response,
    sources: tuple[JsonObject, ...],
    field_name: str,
    header_name: str,
) -> str | None:
    header_value = response.headers.get(header_name)
    if header_value and header_value.strip():
        return header_value.strip()
    for source in sources:
        value = source.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_category(status: int) -> SynthErrorCategory:
    if status == 401:
        return SynthErrorCategory.AUTHENTICATION
    if status == 403:
        return SynthErrorCategory.AUTHORIZATION
    if status in {400, 404, 405, 422}:
        return SynthErrorCategory.VALIDATION
    if status == 409:
        return SynthErrorCategory.CONFLICT
    if status == 402:
        return SynthErrorCategory.RESOURCE_EXHAUSTED
    if status == 429:
        return SynthErrorCategory.RATE_LIMITED
    if status in {500, 502, 503, 504}:
        return SynthErrorCategory.TRANSIENT_SERVICE
    return SynthErrorCategory.OPERATION


def raise_transport_exception(
    method: str,
    path: str,
    error: httpx.HTTPError,
    operation_id: str | None = None,
) -> NoReturn:
    """Translate timeout and network exceptions without changing authority."""

    if isinstance(error, httpx.TimeoutException):
        raise TimeoutError(
            f"{method} {path} timed out",
            failure=SynthFailure(
                code=SynthErrorCode("transport_timeout"),
                category=SynthErrorCategory.TRANSIENT_SERVICE,
                operation=operation_id,
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=True),
                status=None,
                detail=f"{method} {path} timed out",
            ),
        ) from error
    raise TransientServiceError(
        0,
        path,
        f"network error ({type(error).__name__})",
        failure=SynthFailure(
            code=SynthErrorCode("transport_error"),
            category=SynthErrorCategory.TRANSIENT_SERVICE,
            operation=operation_id,
            request_id=None,
            correlation_id=None,
            retry=RetryDirective(retryable=True),
            status=None,
            detail=f"{method} {path} transport failure",
        ),
    ) from error


def raise_json_decode_error(
    method: str,
    path: str,
    response: httpx.Response,
    error: Exception,
    operation_id: str | None = None,
) -> NoReturn:
    """Reject non-JSON and structurally invalid JSON responses at the boundary."""

    raise ContractMismatchError(
        response.status_code,
        str(response.request.url),
        f"{method} {path} response was not valid JSON",
        response.text[:200],
        failure=SynthFailure(
            code=SynthErrorCode("response_not_json"),
            category=SynthErrorCategory.CONTRACT_MISMATCH,
            operation=operation_id,
            request_id=response.headers.get("x-request-id"),
            correlation_id=response.headers.get("x-correlation-id"),
            retry=RetryDirective(retryable=False),
            status=response.status_code,
            detail=f"{method} {path} response was not valid JSON",
        ),
    ) from error


@dataclass(slots=True)
class HttpTransport:
    """One strict sync transport for JSON, bytes, and SSE operations."""

    base_url: str
    headers: Mapping[str, str]
    timeout_seconds: float = 30.0
    error_handler: ErrorHandler = raise_http_error
    exception_handler: ExceptionHandler = raise_transport_exception
    decode_error_handler: DecodeErrorHandler = raise_json_decode_error
    client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=dict(self.headers),
            timeout=self.timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        json_body: JsonObject | None = None,
        headers: Mapping[str, str] | None = None,
        allow_not_found: bool = False,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> JsonValue:
        try:
            response = self.client.request(
                method,
                path,
                params=cast(Mapping[str, object] | None, params),
                json=json_body,
                headers=headers,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            self.exception_handler(method, path, exc, operation_id)
        except httpx.TransportError as exc:
            self.exception_handler(method, path, exc, operation_id)
        if allow_not_found and response.status_code == 404:
            return None
        if response.is_error:
            self.error_handler(response, operation_id)
        if not response.content:
            return {}
        try:
            return _decode_json_value(response.json(), context=f"{method} {path} response")
        except (json.JSONDecodeError, ValueError) as exc:
            self.decode_error_handler(method, path, response, exc, operation_id)

    def execute(self, request: HttpRequest) -> JsonValue:
        return self.request_json(
            request.operation.method.value,
            request.path,
            params=request.query,
            json_body=request.body,
            headers=request.headers,
            timeout_seconds=request.timeout_seconds,
            operation_id=str(request.operation.operation_id),
        )

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> bytes:
        try:
            response = self.client.request(
                method,
                path,
                params=cast(Mapping[str, object] | None, params),
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            self.exception_handler(method, path, exc, operation_id)
        except httpx.TransportError as exc:
            self.exception_handler(method, path, exc, operation_id)
        if response.is_error:
            self.error_handler(response, operation_id)
        return bytes(response.content)

    def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> Iterator[SseEvent]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        try:
            with self.client.stream(
                "GET",
                path,
                params=cast(Mapping[str, object] | None, params),
                headers=headers,
                timeout=timeout_seconds,
            ) as response:
                if response.is_error:
                    response.read()
                    self.error_handler(response, operation_id)
                yield from iter_sse_events(response.iter_lines())
        except httpx.TimeoutException as exc:
            self.exception_handler("GET", path, exc, operation_id)
        except httpx.TransportError as exc:
            self.exception_handler("GET", path, exc, operation_id)


__all__ = [
    "ErrorHandler",
    "ExceptionHandler",
    "DecodeErrorHandler",
    "HttpTransport",
    "raise_http_error",
    "raise_json_decode_error",
    "raise_transport_exception",
]
