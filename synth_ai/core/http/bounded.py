"""Single-attempt bounded JSON reads on caller-owned authenticated transports.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
Async reads have a cancellation deadline. Sync reads reject any observation
after the monotonic deadline; an in-flight blocking read can finish after it,
bounded by the HTTP client's per-I/O timeout. No threads or global signals are
introduced to pretend arbitrary synchronous transports are cancellable.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from synth_ai.core.contracts.json_value import JsonValue
from synth_ai.core.errors import ContractMismatchError
from synth_ai.core.http.transport import _decode_json_value

if TYPE_CHECKING:
    from synth_ai.core.http.async_transport import AsyncHttpTransport
    from synth_ai.core.http.transport import HttpTransport


@dataclass(frozen=True)
class JsonReadBudget:
    response_bytes_max: int
    deadline_seconds: float

    def __post_init__(self) -> None:
        if (
            type(self.response_bytes_max) is not int
            or not 1 <= self.response_bytes_max <= 1_048_576
        ):
            raise ValueError("JSON response bound must be an integer from 1 to 1048576 bytes")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("JSON deadline must be finite and positive")

    def require_time(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout("Bounded JSON absolute deadline exceeded")

    def io_timeout(self, transport_seconds: float) -> float:
        if not math.isfinite(transport_seconds) or transport_seconds <= 0:
            raise ValueError("Transport timeout must be finite and positive")
        return min(transport_seconds, self.deadline_seconds)

    def require_headers(self, response: httpx.Response) -> None:
        if response.is_redirect:
            self.reject(response, "redirects are forbidden for bounded JSON reads")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            self.reject(response, "compressed bounded JSON responses are forbidden")
        length = response.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                self.reject(response, "invalid JSON content length")
            if int(length) > self.response_bytes_max:
                self.reject(response, "JSON response exceeds byte budget")

    def append(self, response: httpx.Response, body: bytearray, chunk: bytes) -> None:
        if len(body) + len(chunk) > self.response_bytes_max:
            self.reject(response, "JSON response exceeds byte budget")
        body.extend(chunk)

    @staticmethod
    def reject(response: httpx.Response, message: str) -> None:
        raise ContractMismatchError(response.status_code, str(response.request.url), message)


def _decode(
    transport: HttpTransport | AsyncHttpTransport,
    response: httpx.Response,
    operation_id: str | None,
) -> JsonValue:
    if response.is_error:
        transport.error_handler(response, operation_id)
    try:
        return _decode_json_value(response.json(), context="bounded GET response")
    except (json.JSONDecodeError, ValueError) as error:
        transport.decode_error_handler(
            "GET",
            response.request.url.path,
            response,
            error,
            operation_id,
        )


def _buffered(response: httpx.Response, body: bytearray) -> httpx.Response:
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        content=bytes(body),
        request=response.request,
    )


def bounded_json_get(
    transport: HttpTransport,
    path: str,
    budget: JsonReadBudget,
    operation_id: str | None,
) -> JsonValue:
    deadline = time.monotonic() + budget.deadline_seconds
    try:
        with transport.client.stream(
            "GET",
            path,
            headers={"Accept-Encoding": "identity"},
            follow_redirects=False,
            timeout=budget.io_timeout(transport.timeout_seconds),
        ) as response:
            budget.require_time(deadline)
            budget.require_headers(response)
            body = bytearray()
            for chunk in response.iter_bytes():
                budget.require_time(deadline)
                budget.append(response, body, chunk)
            budget.require_time(deadline)
            return _decode(transport, _buffered(response, body), operation_id)
    except httpx.TransportError as error:
        transport.exception_handler("GET", path, error, operation_id)


async def bounded_json_get_async(
    transport: AsyncHttpTransport,
    path: str,
    budget: JsonReadBudget,
    operation_id: str | None,
) -> JsonValue:
    deadline = time.monotonic() + budget.deadline_seconds
    try:
        async with asyncio.timeout(budget.deadline_seconds):
            async with transport.client.stream(
                "GET",
                path,
                headers={"Accept-Encoding": "identity"},
                follow_redirects=False,
                timeout=budget.io_timeout(transport.timeout_seconds),
            ) as response:
                budget.require_time(deadline)
                budget.require_headers(response)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    budget.require_time(deadline)
                    budget.append(response, body, chunk)
                budget.require_time(deadline)
                return _decode(transport, _buffered(response, body), operation_id)
    except TimeoutError as error:
        transport.exception_handler(
            "GET",
            path,
            httpx.ReadTimeout("Bounded JSON absolute deadline exceeded"),
            operation_id,
        )
        raise AssertionError("exception handler must raise") from error
    except httpx.TransportError as error:
        transport.exception_handler("GET", path, error, operation_id)
