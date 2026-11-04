"""One transport for generated operations, with bounded same-identity retries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from ._generated.models import MODELS, RunEvent
from ._generated.operations import OPERATIONS
from .errors import SynthError, TransportFailureError, local_problem, raise_for_response

T = TypeVar("T", bound=BaseModel)


class Transport:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
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
            "User-Agent": "synth-ai/0.26.0",
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
        if op["idempotent"]:
            headers["Idempotency-Key"] = idempotency_key or str(uuid4())
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.request(
                    op["method"], self.base_url + route, params=query, json=payload, headers=headers
                )
            except httpx.TransportError as error:
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
            if response.status_code not in {429, 503} or attempt == self.retries:
                break
            if op["method"] not in {"GET", "HEAD"} and not op["idempotent"]:
                break
            delay = retry_delay(response.headers.get("Retry-After"), attempt)
            if delay is None:
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
    return max(0, seconds) if seconds <= 60 else None
