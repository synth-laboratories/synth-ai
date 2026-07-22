"""Native asynchronous HTTP transport with parity to the sync substrate.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import cast

import httpx

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.retry import (
    RetryPolicy,
    retry_after_from_error,
    should_retry_failure,
)
from synth_ai.core.http.streaming import SseEvent, iter_sse_events_async
from synth_ai.core.http.transport import (
    ErrorHandler,
    ExceptionHandler,
    DecodeErrorHandler,
    _decode_json_value,
    raise_http_error,
    raise_json_decode_error,
    raise_transport_exception,
)


@dataclass(slots=True)
class AsyncHttpTransport:
    """One native async transport for JSON, bytes, and SSE operations."""

    base_url: str
    headers: Mapping[str, str]
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    error_handler: ErrorHandler = raise_http_error
    exception_handler: ExceptionHandler = raise_transport_exception
    decode_error_handler: DecodeErrorHandler = raise_json_decode_error
    client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=dict(self.headers),
            timeout=self.timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=0),
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def request_json(
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
            response = await self.client.request(
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

    async def execute(self, request: HttpRequest) -> JsonValue:
        attempt_index = 0
        while True:
            try:
                return await self.request_json(
                    request.operation.method.value,
                    request.path,
                    params=request.query,
                    json_body=request.body,
                    headers=request.headers,
                    timeout_seconds=request.timeout_seconds,
                    operation_id=str(request.operation.operation_id),
                )
            except Exception as error:
                if attempt_index + 1 >= self.retry_policy.attempts_max:
                    raise
                if not should_retry_failure(self.retry_policy, request, error):
                    raise
                delay = self.retry_policy.delay_seconds(
                    attempt_index,
                    retry_after_seconds=retry_after_from_error(error),
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt_index += 1

    async def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> bytes:
        try:
            response = await self.client.request(
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

    async def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> AsyncIterator[SseEvent]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        try:
            async with self.client.stream(
                "GET",
                path,
                params=cast(Mapping[str, object] | None, params),
                headers=headers,
                timeout=timeout_seconds,
            ) as response:
                if response.is_error:
                    await response.aread()
                    self.error_handler(response, operation_id)
                async for event in iter_sse_events_async(response.aiter_lines()):
                    yield event
        except httpx.TimeoutException as exc:
            self.exception_handler("GET", path, exc, operation_id)
        except httpx.TransportError as exc:
            self.exception_handler("GET", path, exc, operation_id)


__all__ = ["AsyncHttpTransport"]
