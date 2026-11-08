"""One transport for generated operations, with bounded same-identity retries."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from ._generated.models import MODELS, RunEvent
from ._generated.operations import OPERATIONS
from ._version import __version__
from .errors import SynthError, TransportFailureError, local_problem, raise_for_response

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class RetryPolicy:
    """Bound one operation; ambiguous mutations are never automatically replayed."""

    attempts: int = 3
    deadline_seconds: float = 90.0
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 8.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or not 1 <= self.attempts <= 6:
            raise ValueError("retry attempts must be between 1 and 6")
        values = (
            self.deadline_seconds,
            self.backoff_seconds,
            self.backoff_max_seconds,
            self.jitter_seconds,
        )
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or self.deadline_seconds == 0
        ):
            raise ValueError(
                "retry durations must be finite and nonnegative; deadline must be positive"
            )

    def delay(self, value: str | None, attempt: int) -> float | None:
        server_delay = retry_delay(value, attempt)
        if server_delay is None:
            return None
        backoff = min(self.backoff_max_seconds, self.backoff_seconds * float(2**attempt))
        return max(server_delay if value is not None else 0, backoff) + random.uniform(
            0, self.jitter_seconds
        )


def ambiguous_response(response: httpx.Response) -> bool:
    """Honor the owner's explicit uncertainty even on a retryable HTTP status."""
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        return False
    return isinstance(body, dict) and (
        body.get("outcome") == "unknown" or body.get("outcome_unknown") is True
    )


class Transport:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        url = httpx.URL(self.base_url)
        if (
            url.scheme not in {"https", "http"}
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base_url must be an HTTP origin")
        if url.scheme == "http" and url.host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("credentials require HTTPS outside loopback")
        if retries < 0 or retries > 5:
            raise ValueError("retries must be between 0 and 5")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy(
            attempts=retries + 1, deadline_seconds=timeout * (retries + 1)
        )
        self.token = token
        self.client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self.owned = client is None
        self.retries = retries

    async def close(self) -> None:
        if self.owned:
            await self.client.aclose()

    def headers(self, auth: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "synth-ai/" + __version__,
            "X-Request-Id": str(uuid4()),
        }
        if auth:
            if not self.token:
                raise SynthError("Set SYNTH_API_KEY or run synth login")
            headers["Authorization"] = "Bearer " + self.token
        return headers

    async def invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, Any] | None = None,
        body: BaseModel | Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        op = OPERATIONS[operation_id]
        route = op["path"]
        if set(path or {}) != set(op["path_params"]):
            raise ValueError("path parameters do not match operation")
        for name, value in (path or {}).items():
            route = route.replace("{" + name + "}", quote(value, safe=""))
        query = {k: v for k, v in (query or {}).items() if v is not None}
        if set(query) - set(op["query_params"]):
            raise ValueError("query parameters do not match operation")
        payload = (
            body.model_dump(mode="json", exclude_unset=True)
            if isinstance(body, BaseModel)
            else body
        )
        if op["request_model"] and payload is not None:
            payload = (
                MODELS[op["request_model"]]
                .model_validate(payload)
                .model_dump(mode="json", exclude_unset=True)
            )
        headers = self.headers(op["auth"])
        if op["idempotency"] == "durable_key":
            headers["Idempotency-Key"] = idempotency_key or str(uuid4())
        deadline = time.monotonic() + self.retry_policy.deadline_seconds
        for attempt in range(self.retry_policy.attempts):
            try:
                remaining = deadline - time.monotonic()
                async with asyncio.timeout(max(0, remaining)):
                    response = await self.client.request(
                        op["method"],
                        self.base_url + route,
                        params=query,
                        json=payload,
                        headers=headers,
                        timeout=min(self.timeout, max(0, remaining)),
                    )
            except (httpx.TransportError, TimeoutError) as error:
                # A timeout after admission is unknown. Keep the same key for caller-directed retry.
                raise TransportFailureError(
                    local_problem(
                        "transport_failed",
                        operation_id,
                        unknown=op["method"] not in {"GET", "HEAD"},
                    ),
                    request_id=headers["X-Request-Id"],
                    idempotency_key=headers.get("Idempotency-Key"),
                ) from error
            if (
                response.status_code not in {429, 503}
                or attempt + 1 == self.retry_policy.attempts
                or op["retry"] == "never"
                or ambiguous_response(response)
            ):
                break
            delay = self.retry_policy.delay(response.headers.get("Retry-After"), attempt)
            if delay is None or delay >= deadline - time.monotonic():
                break
            await asyncio.sleep(delay)
        raise_for_response(response)
        if op["binary"]:
            return response.content
        if op["response_model"]:
            try:
                return MODELS[op["response_model"]].model_validate_json(response.content)
            except ValidationError as error:
                raise SynthError("Response does not conform to " + operation_id) from error
        return None

    async def put_upload(
        self, grant: dict[str, Any], content: bytes, *, idempotency_key: str
    ) -> None:
        target = httpx.URL(grant["url"])
        if (
            target.username
            or target.password
            or target.fragment
            or target.scheme not in {"https", "http"}
            or (target.scheme == "http" and target.host not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise SynthError("Upload grant has an invalid URL")
        headers = grant["headers"]
        if any(
            name.lower()
            not in {"content-type", "x-amz-content-sha256", "if-none-match", "content-length"}
            for name in headers
        ):
            raise SynthError("Upload grant has unsupported headers")
        # Construct the request directly: never merge API/client headers, cookies
        # or default authentication into a storage-origin request.
        request = httpx.Request("PUT", target, content=content, headers=headers)
        try:
            response = await self.client.send(request, auth=None, follow_redirects=False)
        except httpx.TransportError as error:
            raise TransportFailureError(
                local_problem("upload_transfer_failed", "uploadFile", unknown=True),
                idempotency_key=idempotency_key,
            ) from error
        if response.status_code not in {200, 201, 204, 412}:
            raise SynthError("Upload transfer refused: HTTP " + str(response.status_code))

    async def request(self, operation_id: str, model: type[T], **kwargs: Any) -> T:
        result = await self.invoke(operation_id, **kwargs)
        if not isinstance(result, model):
            raise SynthError("Unexpected response model for " + operation_id)
        return result

    async def stream_events(
        self, run_id: str, *, last_event_id: str | None = None
    ) -> AsyncIterator[RunEvent]:
        route = OPERATIONS["listRunEvents"]["path"].replace("{run_id}", quote(run_id, safe=""))
        headers = self.headers()
        headers["Accept"] = "text/event-stream"
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        try:
            async with self.client.stream(
                "GET", self.base_url + route, headers=headers, timeout=None
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise_for_response(response)
                if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                    raise SynthError("Expected event stream")
                data: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        if data:
                            yield RunEvent.model_validate(json.loads("\n".join(data)))
                            data = []
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
        except httpx.TransportError as error:
            raise TransportFailureError(
                local_problem("event_stream_interrupted", "listRunEvents")
            ) from error


def retry_delay(value: str | None, attempt: int) -> float | None:
    if value is None:
        return float(2**attempt)
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) and seconds <= 60 else None
