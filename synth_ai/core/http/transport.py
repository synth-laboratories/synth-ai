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
from synth_ai.core.errors import HTTPError, TimeoutError
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.streaming import SseEvent, iter_sse_events


ErrorHandler = Callable[[httpx.Response], NoReturn]
ExceptionHandler = Callable[[str, str, httpx.HTTPError], NoReturn]
DecodeErrorHandler = Callable[[str, str, httpx.Response, Exception], NoReturn]


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


def raise_http_error(response: httpx.Response) -> NoReturn:
    """Translate one failed response into the general SDK HTTP error."""

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
    raise HTTPError(
        status=response.status_code,
        url=str(response.request.url),
        message=message,
        body_snippet=response.text[:200] or None,
        detail=detail,
    )


def raise_transport_exception(method: str, path: str, error: httpx.HTTPError) -> NoReturn:
    """Translate timeout and network exceptions without changing authority."""

    if isinstance(error, httpx.TimeoutException):
        raise TimeoutError(f"{method} {path} timed out") from error
    raise HTTPError(0, path, f"network error ({type(error).__name__})") from error


def raise_json_decode_error(
    method: str,
    path: str,
    response: httpx.Response,
    error: Exception,
) -> NoReturn:
    """Reject non-JSON and structurally invalid JSON responses at the boundary."""

    raise HTTPError(
        response.status_code,
        str(response.request.url),
        f"{method} {path} response was not valid JSON",
        response.text[:200],
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
            self.exception_handler(method, path, exc)
        except httpx.TransportError as exc:
            self.exception_handler(method, path, exc)
        if allow_not_found and response.status_code == 404:
            return None
        if response.is_error:
            self.error_handler(response)
        if not response.content:
            return {}
        try:
            return _decode_json_value(response.json(), context=f"{method} {path} response")
        except (json.JSONDecodeError, ValueError) as exc:
            self.decode_error_handler(method, path, response, exc)

    def execute(self, request: HttpRequest) -> JsonValue:
        return self.request_json(
            request.operation.method.value,
            request.path,
            params=request.query,
            json_body=request.body,
            headers=request.headers,
            timeout_seconds=request.timeout_seconds,
        )

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
    ) -> bytes:
        try:
            response = self.client.request(
                method,
                path,
                params=cast(Mapping[str, object] | None, params),
            )
        except httpx.TimeoutException as exc:
            self.exception_handler(method, path, exc)
        except httpx.TransportError as exc:
            self.exception_handler(method, path, exc)
        if response.is_error:
            self.error_handler(response)
        return bytes(response.content)

    def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
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
                    self.error_handler(response)
                yield from iter_sse_events(response.iter_lines())
        except httpx.TimeoutException as exc:
            self.exception_handler("GET", path, exc)
        except httpx.TransportError as exc:
            self.exception_handler("GET", path, exc)


__all__ = [
    "ErrorHandler",
    "ExceptionHandler",
    "DecodeErrorHandler",
    "HttpTransport",
    "raise_http_error",
    "raise_json_decode_error",
    "raise_transport_exception",
]
