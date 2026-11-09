"""One transport for generated operations, with bounded same-identity retries."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from ._generated.models import Problem, RunEvent, RunSnapshot, UploadGrant
from ._generated.operations import OPERATIONS
from ._http_errors import raise_stream_error
from ._limits import CHECKPOINT_MAX_EVENTS, CHECKPOINT_MAX_ID_LENGTH
from ._operations import OperationExecutor
from ._retry import RetryPolicy as RetryPolicy
from ._retry import retry_delay as retry_delay
from ._stream import StreamCheckpoint, bounded_lines, frames
from ._transfers import ArtifactTransfer
from ._version import __version__
from .errors import (
    ResponseContractError,
    SwarmsAPIError,
    SynthError,
    TransportFailureError,
    local_problem,
    raise_problem,
)

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
            if self.token is None or self.token == "":
                raise SynthError("Set SYNTH_API_KEY or run synth login")
            headers["Authorization"] = "Bearer " + self.token
        return headers

    async def invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> BaseModel | bytes | None:
        return await OperationExecutor(
            self.client, self.base_url, self.timeout, self.retry_policy, self.headers
        ).invoke(operation_id, path=path, query=query, body=body, idempotency_key=idempotency_key)

    def _transfers(self) -> ArtifactTransfer:
        return ArtifactTransfer(
            self.client, self.base_url, self.timeout, self.retry_policy, self.headers
        )

    async def put_upload(self, grant: UploadGrant, content: bytes, *, idempotency_key: str) -> None:
        await self._transfers().put_upload(grant, content, idempotency_key=idempotency_key)

    async def download_output(
        self,
        run_id: str,
        output_id: str,
        destination: Path,
        *,
        revision_id: str,
        digest_sha256: str,
        max_bytes: int,
        expected_bytes: int | None = None,
    ) -> None:
        await self._transfers().download_output(
            run_id,
            output_id,
            destination,
            revision_id=revision_id,
            digest_sha256=digest_sha256,
            max_bytes=max_bytes,
            expected_bytes=expected_bytes,
        )

    async def request(
        self,
        operation_id: str,
        model: type[T],
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> T:
        result = await self.invoke(
            operation_id, path=path, query=query, body=body, idempotency_key=idempotency_key
        )
        if not isinstance(result, model):
            raise SynthError("Unexpected response model for " + operation_id)
        return result

    async def stream_events(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        checkpoint: StreamCheckpoint | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[RunEvent]:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("stream timeout must be finite and positive")
        state = checkpoint if checkpoint is not None else StreamCheckpoint()
        state.bind(run_id)
        if last_event_id is not None:
            if state.last_event_id is not None and state.last_event_id != last_event_id:
                raise ValueError("conflicting stream resume identities")
            state.last_event_id = last_event_id
        state.__post_init__()
        route = OPERATIONS["listRunEvents"]["path"].replace("{run_id}", quote(run_id, safe=""))
        try:
            async with asyncio.timeout(timeout):
                for attempt in range(self.retry_policy.attempts):
                    headers = self.headers()
                    headers["Accept"] = "text/event-stream"
                    headers["Accept-Encoding"] = "identity"
                    if state.last_event_id is not None:
                        headers["Last-Event-ID"] = state.last_event_id
                    disconnected = False
                    try:
                        async with self.client.stream(
                            "GET", self.base_url + route, headers=headers, timeout=self.timeout
                        ) as response:
                            if not response.is_success:
                                await raise_stream_error(response)
                            if (
                                response.headers.get("content-type", "").split(";")[0].strip()
                                != "text/event-stream"
                            ):
                                raise ValueError("expected event stream")
                            if response.headers.get("content-encoding", "identity") != "identity":
                                raise ValueError("compressed event stream refused")
                            async for frame in frames(bounded_lines(response.aiter_bytes())):
                                if frame.event == "error":
                                    raise_problem(
                                        Problem.model_validate_json(frame.data),
                                        request_id=response.headers.get("x-request-id"),
                                    )
                                if frame.event not in {"message", "event"}:
                                    raise ValueError("unknown event frame type")
                                item = (
                                    RunEvent.model_validate_json(frame.data) if frame.data else None
                                )
                                if item is not None and item.run_id != run_id:
                                    raise ValueError("event belongs to another run")
                                duplicate = (
                                    item is not None and item.event_id in state.pending_event_ids
                                )
                                if item is not None:
                                    state.pending_event_ids.add(item.event_id)
                                if len(state.pending_event_ids) > CHECKPOINT_MAX_EVENTS:
                                    raise ValueError(
                                        "expanded event sequence exceeds checkpoint bound"
                                    )
                                if frame.id is not None:
                                    if len(frame.id) > CHECKPOINT_MAX_ID_LENGTH:
                                        raise ValueError("event cursor exceeds checkpoint bound")
                                    state.last_event_id = frame.id or None
                                    state.pending_event_ids.clear()
                                if item is not None and not duplicate:
                                    yield item
                    except httpx.TransportError:
                        disconnected = True
                    if not disconnected:
                        snapshot = await self.request(
                            "getRun", RunSnapshot, path={"run_id": run_id}
                        )
                        if snapshot.execution_closed:
                            return
                    if attempt + 1 < self.retry_policy.attempts:
                        delay = self.retry_policy.delay(None, attempt)
                        await asyncio.sleep(delay or 0)
                error = TransportFailureError(
                    local_problem("event_stream_interrupted", "listRunEvents")
                )
                error.checkpoint = state
                raise error
        except SwarmsAPIError as api_error:
            api_error.checkpoint = state
            raise
        except (ValueError, ValidationError) as invalid_frame:
            contract_error = ResponseContractError(
                local_problem("event_stream_invalid", "listRunEvents")
            )
            contract_error.checkpoint = state
            raise contract_error from invalid_frame
        except TimeoutError as cause:
            error = TransportFailureError(local_problem("event_stream_timeout", "listRunEvents"))
            error.checkpoint = state
            raise error from cause
