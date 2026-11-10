"""One transport for generated operations, with bounded same-identity retries."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from ._event_stream import RunMilestone, decode_events, validate_stream_response
from ._file_source import FileIO
from ._generated.models import RunEvent, RunSnapshot, UploadGrant
from ._generated.operations import OPERATIONS
from ._http_errors import raise_stream_error
from ._operations import OperationExecutor
from ._retry import RetryPolicy as RetryPolicy
from ._retry import retry_delay as retry_delay
from ._stream import StreamCheckpoint
from ._transfers import ArtifactTransfer
from ._version import __version__
from .diagnostics import DiagnosticHook
from .errors import (
    ClientClosedError,
    ResponseContractError,
    SwarmsAPIError,
    SynthError,
    TransportFailureError,
    local_problem,
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
        diagnostic_hook: DiagnosticHook | None = None,
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
        self.diagnostic_hook = diagnostic_hook
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy(
            attempts=retries + 1, deadline_seconds=timeout * (retries + 1)
        )
        self.token = token
        self.client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self.owned = client is None
        self.file_io = FileIO()
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.owned:
            await self.client.aclose()

    def headers(self, auth: bool = True) -> dict[str, str]:
        if self._closed:
            raise ClientClosedError("client is closed")
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
            self.client,
            self.base_url,
            self.timeout,
            self.retry_policy,
            self.headers,
            self.diagnostic_hook,
        ).invoke(operation_id, path=path, query=query, body=body, idempotency_key=idempotency_key)

    def _transfers(self) -> ArtifactTransfer:
        return ArtifactTransfer(
            self.client,
            self.base_url,
            self.timeout,
            self.retry_policy,
            self.headers,
            self.file_io,
        )

    async def put_upload(
        self, grant: UploadGrant, content: bytes | AsyncIterable[bytes], *, idempotency_key: str
    ) -> None:
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

    @asynccontextmanager
    async def subscribe_events(
        self, run_id: str, state: StreamCheckpoint
    ) -> AsyncGenerator[AsyncIterator[RunEvent | RunMilestone], None]:
        """Open the durable feed before an observer performs its authoritative read."""
        state.bind(run_id)
        state.__post_init__()
        headers = self.headers()
        headers["Accept"] = "text/event-stream"
        headers["Accept-Encoding"] = "identity"
        if state.last_event_id is not None:
            headers["Last-Event-ID"] = state.last_event_id
        route = OPERATIONS["listRunEvents"]["path"].replace("{run_id}", quote(run_id, safe=""))
        try:
            async with self.client.stream(
                "GET",
                self.base_url + route,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=False,
            ) as response:
                if not response.is_success:
                    await raise_stream_error(response)
                validate_stream_response(response)
                async with aclosing(decode_events(response, run_id, state)) as events:
                    yield events
        except SwarmsAPIError as error:
            error.checkpoint = state
            raise
        except (ValueError, ValidationError) as cause:
            error = ResponseContractError(local_problem("event_stream_invalid", "listRunEvents"))
            error.checkpoint = state
            raise error from cause

    async def stream_events(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        checkpoint: StreamCheckpoint | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("stream timeout must be finite and positive")
        state = checkpoint if checkpoint is not None else StreamCheckpoint()
        state.bind(run_id)
        if last_event_id is not None:
            if state.last_event_id is not None and state.last_event_id != last_event_id:
                raise ValueError("conflicting stream resume identities")
            state.last_event_id = last_event_id
        state.__post_init__()
        try:
            async with asyncio.timeout(timeout):
                for attempt in range(self.retry_policy.attempts):
                    disconnected = False
                    try:
                        async with self.subscribe_events(run_id, state) as events:
                            async for item in events:
                                # Milestones advance the checkpoint; this public
                                # iterator yields run events only.
                                if isinstance(item, RunEvent):
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
