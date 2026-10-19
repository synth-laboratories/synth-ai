"""Native async bounded exchange. See backend SDK index_contract.md."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from synth_ai.contracts.operations import CAPABILITIES, Operation
from synth_ai.core.capabilities import require_operation, validate_release
from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    MutationCancelledError,
    OutcomeUncertainError,
    RequestTimeoutError,
    SynthError,
    TransportError,
)
from synth_ai.options import ClientOptions, ReadOptions
from synth_ai.types.base import ClientTiming
from synth_ai.types.capabilities import Capabilities, NegotiatedCapability

from .backoff import delay_seconds, retry_after_seconds, retry_failure
from .cleanup import handle_cleanup_failure
from .credentials import CredentialProvider
from .lifetime import AsyncCloseable, await_cleanup
from .request import build_request, encode_body, mutation_identity
from .response import (
    annotate_response_error,
    classify_exchange,
    decode_response,
    validate_negotiated_response,
)

T = TypeVar("T", bound=BaseModel)


class AsyncHttpTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        options: ClientOptions,
        *,
        credential_provider: CredentialProvider | None = None,
        _sleep: Callable[[float], Awaitable[None]] | None = None,
        _random: random.Random | None = None,
    ) -> None:
        if options.local_development_retention and urlsplit(base_url).hostname not in {
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Local development retention requires a loopback API address")
        self.local_development_retention = options.local_development_retention
        self._base_url = base_url
        self._api_key = api_key
        self._credential_provider = credential_provider
        self._credential_tasks: set[asyncio.Task[str]] = set()
        self._options = options
        self._sleep = _sleep or asyncio.sleep
        self._random = _random or random.Random()
        self.transfer_hosts = options.upload_hosts or (urlsplit(base_url).hostname or "",)
        self._owned = options.async_http_client is None
        self.client = options.async_http_client or httpx.AsyncClient(
            trust_env=False, limits=httpx.Limits(max_connections=options.maximum_connections)
        )
        self._closed = False
        self._streams: set[AsyncCloseable] = set()
        self._responses: set[httpx.Response] = set()
        self._closing_task: asyncio.Task[None] | None = None
        self._closing_deadline: float | None = None
        self._response_cleanup_tasks: dict[httpx.Response, asyncio.Task[None]] = {}
        self._send_tasks: set[asyncio.Task[httpx.Response]] = set()

    async def _send_response(self, request: httpx.Request) -> httpx.Response:
        # Reserve before network dispatch: borrowed pools cannot bound SDK custody.
        if self._closed:
            raise ContractError("SDK client is closed")
        if len(self._responses) + len(self._send_tasks) >= self._options.maximum_connections:
            raise ContractError("SDK active response limit reached")
        task = asyncio.create_task(self._send_owned(request))
        self._send_tasks.add(task)
        try:
            return await task
        except asyncio.CancelledError as primary:
            # A completed child may have transferred custody before cancellation
            # prevents the caller from receiving its response.
            if task.done() and not task.cancelled() and task.exception() is None:
                try:
                    await self._release_response(task.result())
                except (Exception, asyncio.CancelledError) as cause:
                    handle_cleanup_failure(cause, primary, mutation=False, identity=None)
            raise
        finally:
            self._send_tasks.discard(task)

    async def _send_owned(self, request: httpx.Request) -> httpx.Response:
        task = asyncio.current_task()
        try:
            response = await self.client.send(
                request, stream=True, auth=None, follow_redirects=False
            )
        finally:
            if task is not None:
                self._send_tasks.discard(task)
        # Transfer the reservation without an intervening suspension. Failed
        # cleanup retains this response and consumes capacity until client close.
        self._responses.add(response)
        if self._closed:
            await self._release_response(response)
            raise ContractError("SDK client closed during dispatch")
        return response

    async def _release_response(self, response: httpx.Response) -> None:
        task = self._response_cleanup_tasks.get(response)
        if task is None:
            task = asyncio.create_task(self._close_response(response))
            self._response_cleanup_tasks[response] = task
        await await_cleanup(task)

    async def _close_response(self, response: httpx.Response) -> None:
        timeout = self._options.timeout_seconds
        if self._closing_deadline is not None:
            timeout = min(timeout, max(0, self._closing_deadline - time.monotonic()))
        async with asyncio.timeout(timeout):
            await response.aclose()
        self._responses.discard(response)
        self._response_cleanup_tasks.pop(response, None)

    @property
    def candidate_manifest_sha256(self) -> str | None:
        return self._options.candidate_manifest_sha256

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def request_timeout_seconds(self) -> float:
        return self._options.timeout_seconds

    @property
    def maximum_response_bytes(self) -> int:
        return self._options.maximum_response_bytes

    def _register_stream(self, stream: AsyncCloseable) -> None:
        if self._closed:
            raise RuntimeError("SDK client is closed")
        if len(self._streams) >= self._options.maximum_connections:
            raise RuntimeError("SDK active stream limit reached")
        self._streams.add(stream)

    def _unregister_stream(self, stream: AsyncCloseable) -> None:
        self._streams.discard(stream)

    async def aclose(self) -> None:
        if self._closing_task is None:
            self._closed = True
            self._closing_deadline = time.monotonic() + self._options.timeout_seconds
            self._closing_task = asyncio.create_task(self._close_owned())
        await await_cleanup(self._closing_task)

    async def _close_owned(self) -> None:
        """Finish one retained cleanup task before acknowledging local cancellation."""
        async with asyncio.timeout(self._options.timeout_seconds):
            await self._close_resources()

    async def _close_resources(self) -> None:
        errors: list[Exception] = []
        credentials = tuple(self._credential_tasks)
        for task in credentials:
            task.cancel()
        if credentials:
            await asyncio.gather(*credentials, return_exceptions=True)
        if self._credential_provider is not None:
            self._credential_provider.close()
        sends = tuple(self._send_tasks)
        for task in sends:
            task.cancel()
        if sends:
            await asyncio.gather(*sends, return_exceptions=True)
        for stream in tuple(self._streams):
            try:
                await stream.aclose()
            except Exception as error:
                errors.append(error)
        for response in tuple(self._responses):
            try:
                await self._release_response(response)
            except Exception as error:
                errors.append(error)
        if self._owned:
            try:
                await self.client.aclose()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("SDK response cleanup failed", errors)

    async def _negotiate(
        self, operation: Operation, deadline: float
    ) -> NegotiatedCapability | None:
        if operation.keyed and self._api_key is None and self._credential_provider is None:
            raise AuthenticationError("This operation requires an API key")
        if operation.contract == "tunnels.v2":
            from synth_ai.contracts.tunnels import require_control_operation

            require_control_operation(operation, self._options)
            return None
        if operation.id == CAPABILITIES.id:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Capability negotiation exceeded the request deadline")
        receipt = await self.request(
            CAPABILITIES,
            Capabilities,
            options=ReadOptions(timeout_seconds=remaining, read_retry_count=0),
        )
        return require_operation(
            receipt,
            operation,
            surface="http",
            expected_manifest_sha256=self._options.release_manifest_sha256,
            candidate_manifest_sha256=self._options.candidate_manifest_sha256,
        )

    async def _open_stream(
        self,
        operation: Operation,
        *,
        path: str,
        query: Mapping[str, str | int],
        remaining_seconds: float,
        digest: str,
    ) -> httpx.Response:
        """Open one SDK-owned read response, without retries or redirects."""
        if self._closed:
            raise RuntimeError("SDK client is closed")
        if operation.mutation or operation.method != "GET":
            raise ValueError("byte streams require an observational GET operation")
        deadline = time.monotonic() + remaining_seconds
        key = await self._resolve_credential(operation, deadline)
        negotiated = await self._negotiate(operation, deadline)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise RequestTimeoutError("Capability negotiation exceeded the stream deadline")
        request = build_request(
            self.client,
            self._base_url,
            key,
            operation,
            path,
            b"",
            query,
            ReadOptions(timeout_seconds=remaining_seconds, read_retry_count=0),
            None,
            remaining_seconds,
            self._options.connect_timeout_seconds,
            negotiated=negotiated,
        )
        request.headers["Accept"] = "application/octet-stream"
        request.headers["If-Match"] = '"' + digest + '"'
        response = await self._send_response(request)
        try:
            validate_negotiated_response(response, negotiated, None, False)
        except BaseException as primary:
            try:
                await self._release_response(response)
            except (Exception, asyncio.CancelledError) as cause:
                handle_cleanup_failure(cause, primary, mutation=False, identity=None)
            raise
        return response

    def mutation_events(self, operation: Operation, *, path: str, body: BaseModel, options=None):
        from .stream import AsyncMutationEventStream

        return AsyncMutationEventStream(self, operation, path=path, body=body, options=options)

    async def _resolve_credential(self, operation: Operation, deadline: float) -> str | None:
        if not operation.keyed or self._credential_provider is None:
            return self._api_key
        task = asyncio.create_task(self._credential_provider.aresolve(deadline))
        self._credential_tasks.add(task)
        try:
            return await task
        finally:
            self._credential_tasks.discard(task)

    async def request(
        self,
        operation: Operation,
        model: type[T],
        *,
        body: BaseModel | None = None,
        path: str | None = None,
        query: Mapping[str, str | int] | None = None,
        options: ReadOptions | None = None,
        validate: Callable[[T], T] | None = None,
    ) -> T:
        if self._closed:
            raise RuntimeError("SDK client is closed")
        if options is not None and options.response_metadata is not None:
            options.response_metadata.rate_limits = ()
        payload = encode_body(body)
        identity = mutation_identity(operation, path, payload, options)
        started = time.monotonic()
        timeout = (
            options.timeout_seconds
            if options is not None and options.timeout_seconds is not None
            else self._options.request_timeout(operation.id)
        )
        deadline = started + timeout
        key = await self._resolve_credential(operation, deadline)
        negotiated = await self._negotiate(operation, deadline)
        retry_count = (
            options.read_retry_count
            if options is not None and options.read_retry_count is not None
            else self._options.read_retry_count
        )
        attempts = 1 if operation.mutation else retry_count + 1
        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Request deadline expired", identity=identity)
            request = build_request(
                self.client,
                self._base_url,
                key,
                operation,
                path,
                payload,
                query,
                options,
                identity,
                remaining,
                self._options.connect_timeout_seconds,
                negotiated=negotiated,
            )
            response: httpx.Response | None = None
            try:
                async with asyncio.timeout(remaining):
                    response = await self._send_response(request)
                    if options is not None and options.response_metadata is not None:
                        options.response_metadata.capture(response.headers)
                    exchange = classify_exchange(
                        response,
                        None,
                        contract=operation.contract,
                        mutation=operation.mutation,
                        identity=identity,
                        attempt=attempt,
                        attempts=attempts,
                        operation=operation.id,
                    )
                    content = bytearray()
                    first_byte: float | None = None
                    if exchange.kind == "decode":
                        if (
                            response.headers.get("Content-Encoding", "identity").lower()
                            != "identity"
                        ):
                            raise ContractError(
                                "Compressed responses are outside the bounded wire contract"
                            )
                        async for chunk in response.aiter_bytes():
                            if self._closed:
                                raise ContractError("SDK client closed during response")
                            if first_byte is None:
                                first_byte = time.monotonic()
                            if len(content) + len(chunk) > self._options.maximum_response_bytes:
                                raise ContractError("Response exceeded the configured byte bound")
                            content.extend(chunk)
                        exchange = classify_exchange(
                            response,
                            bytes(content),
                            contract=operation.contract,
                            mutation=operation.mutation,
                            identity=identity,
                            attempt=attempt,
                            attempts=attempts,
                            operation=operation.id,
                        )
                finished = time.monotonic()
                if exchange.error is not None:
                    raise annotate_response_error(response, exchange.error)
                if exchange.kind == "retry":
                    hint = retry_after_seconds(response.headers.get("Retry-After"))
                    delay = delay_seconds(
                        attempt,
                        self._options.retry_base_delay_seconds,
                        self._options.retry_max_delay_seconds,
                        hint,
                        rng=self._random,
                    )
                    failure = retry_failure(
                        status=response.status_code,
                        delay=delay,
                        remaining=deadline - time.monotonic(),
                        retry_after=hint,
                    )
                    if failure is not None:
                        annotate_response_error(response, failure)
                    await self._release_response(response)
                    response = None
                    if failure is not None:
                        raise failure
                    await self._sleep(delay)
                    continue
                timing = ClientTiming(
                    total_ms=(finished - started) * 1000,
                    first_byte_ms=((first_byte or finished) - started) * 1000,
                    attempts=attempt,
                )
                decoded = decode_response(
                    response,
                    bytes(content),
                    model,
                    timing,
                    identity,
                    operation.mutation,
                    operation.contract,
                    negotiated=negotiated,
                )
                if operation.id == CAPABILITIES.id and isinstance(decoded, Capabilities):
                    validate_release(
                        decoded,
                        self._options.release_manifest_sha256,
                        self._options.candidate_manifest_sha256,
                    )
                return validate(decoded) if validate is not None else decoded
            except asyncio.CancelledError as cause:
                if operation.mutation:
                    raise MutationCancelledError(identity) from cause
                raise
            except ContractError as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation response is invalid; reconcile its identity",
                        identity=identity,
                        reason="response_invalid",
                        status=response.status_code if response is not None else None,
                    ) from cause
                raise
            except (httpx.TransportError, TimeoutError, RequestTimeoutError) as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation delivery is uncertain; retain its identity",
                        identity=identity,
                        reason="response_lost",
                    ) from cause
                if isinstance(cause, httpx.TransportError) and attempt < attempts:
                    delay = delay_seconds(
                        attempt,
                        self._options.retry_base_delay_seconds,
                        self._options.retry_max_delay_seconds,
                        None,
                        rng=self._random,
                    )
                    failure = retry_failure(
                        status=None,
                        delay=delay,
                        remaining=deadline - time.monotonic(),
                        retry_after=None,
                    )
                    if response is not None:
                        await self._release_response(response)
                        response = None
                    if failure is not None:
                        raise failure from cause
                    await self._sleep(delay)
                    continue
                if isinstance(cause, (httpx.TimeoutException, TimeoutError)):
                    raise RequestTimeoutError("HTTP request timed out") from cause
                if isinstance(cause, httpx.TransportError):
                    raise TransportError("HTTP exchange failed") from cause
                raise
            except SynthError:
                raise
            except Exception as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation exchange failed unexpectedly; reconcile its identity",
                        identity=identity,
                    ) from cause
                raise TransportError("HTTP exchange failed unexpectedly") from cause
            finally:
                if response is not None:
                    primary = sys.exception()
                    try:
                        await self._release_response(response)
                    except (Exception, asyncio.CancelledError) as cause:
                        handle_cleanup_failure(
                            cause, primary, mutation=operation.mutation, identity=identity
                        )
        raise RuntimeError("Bounded request loop exhausted without an outcome")
