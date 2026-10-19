"""Bounded one-request mutation SSE. No reconnect, retry or intent regeneration.

See the public Index retention contract. Closing this response disconnects the
active owner; only metadata can be recovered after losing strict content.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from synth_ai.errors import ContractError, MutationCancelledError, OutcomeUncertainError, SynthError
from synth_ai.options import MutationOptions
from synth_ai.types.base import ClientTiming

from .cleanup import handle_cleanup_failure
from .identity import annotate_request_error, request_options
from .request import build_request, encode_body, mutation_identity
from .response import (
    annotate_response_error,
    classify_exchange,
    decode_response,
    validate_negotiated_response,
)


@dataclass(frozen=True)
class StreamFrame:
    event: str
    data: bytes


class FrameDecoder:
    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.buffer = bytearray()
        self.frames = 0

    def feed(self, chunk):
        offset = 0
        while offset < len(chunk):
            take = min(len(chunk) - offset, self.maximum_bytes + 2 - len(self.buffer))
            if take <= 0:
                raise ContractError("Event frame exceeded the configured byte bound")
            self.buffer.extend(chunk[offset : offset + take])
            offset += take
            yield from self._drain()

    def _drain(self):
        while (end := self.buffer.find(b"\n\n")) >= 0:
            if end > self.maximum_bytes:
                raise ContractError("Event frame exceeded the configured byte bound")
            raw = bytes(self.buffer[:end])
            del self.buffer[: end + 2]
            if raw.startswith(b":") and b"\n" not in raw:
                continue
            lines = raw.split(b"\n")
            if (
                len(lines) != 2
                or not lines[0].startswith(b"event: ")
                or not lines[1].startswith(b"data: ")
            ):
                raise ContractError("Malformed bounded event frame")
            try:
                kind = lines[0][7:].decode("ascii")
                lines[1][6:].decode("utf-8")
            except UnicodeError as error:
                raise ContractError("Malformed event encoding") from error
            self.frames += 1
            if kind not in {"progress", "terminal", "error"} or self.frames > 65:
                raise ContractError("Event stream discriminant or count is outside its bound")
            yield StreamFrame(kind, lines[1][6:])
        if len(self.buffer) > self.maximum_bytes and not (
            len(self.buffer) == self.maximum_bytes + 1 and self.buffer.endswith(b"\n")
        ):
            raise ContractError("Event frame exceeded the configured byte bound")


def _prepare(owner, operation, path, body, options):
    if not operation.keyed or not operation.mutation or operation.method != "POST":
        raise ValueError("Event delivery requires an authenticated POST mutation")
    options = request_options(operation, options or MutationOptions())
    payload = encode_body(body)
    identity = mutation_identity(operation, path, payload, options)
    timeout = options.timeout_seconds or owner._options.request_timeout(operation.id)
    return options, payload, identity, time.monotonic() + timeout


def _request(owner, operation, path, payload, options, identity, deadline, key, negotiated):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OutcomeUncertainError("Event request deadline expired", identity=identity)
    request = build_request(
        owner.client,
        owner.base_url,
        key,
        operation,
        path,
        payload,
        None,
        options,
        identity,
        remaining,
        owner._options.connect_timeout_seconds,
        negotiated=negotiated,
    )
    request.headers["Accept"] = "text/event-stream"
    return request


def _check_headers(response, negotiated, identity, contract):
    exchange = classify_exchange(
        response,
        None,
        contract=contract,
        mutation=True,
        identity=identity,
        attempt=1,
        attempts=1,
    )
    if exchange.error is not None:
        raise annotate_response_error(response, exchange.error)
    validate_negotiated_response(response, negotiated, identity, True)
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ContractError(
            "Compressed event responses are outside the bounded contract", identity=identity
        )
    if (
        response.is_success
        and response.headers.get("Content-Type", "").split(";")[0].strip() != "text/event-stream"
    ):
        raise OutcomeUncertainError(
            "Mutation did not return the negotiated event stream", identity=identity
        )


def _definite_error(response, content, identity, operation, negotiated):
    decode_response(
        response,
        content,
        BaseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        identity,
        True,
        operation.contract,
        negotiated=negotiated,
        operation=operation.id,
    )
    raise OutcomeUncertainError("Event mutation outcome is unknown", identity=identity)


class MutationEventStream:
    def __init__(self, owner, operation, *, path, body, options=None):
        self.owner, self.operation, self.path = owner, operation, path
        self.options, self.payload, self.identity, self.deadline = _prepare(
            owner, operation, path, body, options
        )
        self.response = None
        self.closed = False
        self.used = False
        owner._register_stream(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.response is not None:
                self.owner._release_response(self.response)
        finally:
            self.payload = b""
            self.owner._unregister_stream(self)

    def __iter__(self):
        if self.used or self.closed:
            raise ContractError("Event mutation stream is single-use", identity=self.identity)
        self.used = True
        primary = None
        try:
            key = (
                self.owner._credential_provider.resolve(self.deadline)
                if self.owner._credential_provider is not None
                else self.owner._api_key
            )
            negotiated = self.owner._negotiate(self.operation, self.deadline)
            request = _request(
                self.owner,
                self.operation,
                self.path,
                self.payload,
                self.options,
                self.identity,
                self.deadline,
                key,
                negotiated,
            )
            self.response = self.owner._adopt_response(
                self.owner.client.send(request, stream=True, auth=None, follow_redirects=False)
            )
            self.payload = b""
            if self.options.response_metadata is not None:
                self.options.response_metadata.capture(self.response.headers)
            _check_headers(self.response, negotiated, self.identity, self.operation.contract)
            decoder = FrameDecoder(self.owner.maximum_response_bytes)
            content = bytearray()
            terminal = False
            for chunk in self.response.iter_bytes():
                if self.closed or time.monotonic() >= self.deadline:
                    raise OutcomeUncertainError(
                        "Event stream closed or deadline expired", identity=self.identity
                    )
                if not self.response.is_success:
                    if len(content) + len(chunk) > self.owner.maximum_response_bytes:
                        raise OutcomeUncertainError(
                            "Error response exceeded the byte bound", identity=self.identity
                        )
                    content.extend(chunk)
                    continue
                for frame in decoder.feed(chunk):
                    if terminal:
                        raise ContractError(
                            "Event followed the terminal frame", identity=self.identity
                        )
                    terminal = frame.event in {"terminal", "error"}
                    yield frame
            if not self.response.is_success:
                _definite_error(
                    self.response, bytes(content), self.identity, self.operation, negotiated
                )
            if not terminal or decoder.buffer:
                raise OutcomeUncertainError(
                    "Strict event stream ended without a complete terminal", identity=self.identity
                )
        except (httpx.HTTPError, OSError) as error:
            primary = OutcomeUncertainError(
                "Strict event transport failed; recover metadata only", identity=self.identity
            )
            annotate_request_error(primary, self.operation, self.options)
            raise primary from error
        except BaseException as error:
            primary = error
            if isinstance(error, SynthError):
                annotate_request_error(error, self.operation, self.options)
            raise
        finally:
            try:
                self.close()
            except Exception as cause:
                handle_cleanup_failure(cause, primary, mutation=True, identity=self.identity)


class AsyncMutationEventStream:
    def __init__(self, owner, operation, *, path, body, options=None):
        self.owner, self.operation, self.path = owner, operation, path
        self.options, self.payload, self.identity, self.deadline = _prepare(
            owner, operation, path, body, options
        )
        self.response = None
        self.closed = False
        self.used = False
        self._read_task = None
        owner._register_stream(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self._read_task is not None:
                self._read_task.cancel()
                await asyncio.gather(self._read_task, return_exceptions=True)
            if self.response is not None:
                await self.owner._release_response(self.response)
        finally:
            self.payload = b""
            self.owner._unregister_stream(self)

    async def _chunks(self):
        response = self.response
        if response is None:
            raise ContractError("Event response is unavailable before stream entry")
        iterator = response.aiter_bytes()
        while True:

            async def read():
                return await anext(iterator)

            child = asyncio.create_task(read())
            self._read_task = child
            try:
                yield await child
            except StopAsyncIteration:
                return
            finally:
                if self._read_task is child:
                    self._read_task = None

    async def __aiter__(self):
        if self.used or self.closed:
            raise ContractError("Event mutation stream is single-use", identity=self.identity)
        self.used = True
        primary = None
        try:
            key = await self.owner._resolve_credential(self.operation, self.deadline)
            negotiated = await self.owner._negotiate(self.operation, self.deadline)
            request = _request(
                self.owner,
                self.operation,
                self.path,
                self.payload,
                self.options,
                self.identity,
                self.deadline,
                key,
                negotiated,
            )
            self.response = await self.owner._send_response(request)
            self.payload = b""
            if self.options.response_metadata is not None:
                self.options.response_metadata.capture(self.response.headers)
            _check_headers(self.response, negotiated, self.identity, self.operation.contract)
            decoder = FrameDecoder(self.owner.maximum_response_bytes)
            content = bytearray()
            terminal = False
            async with asyncio.timeout(max(0, self.deadline - time.monotonic())):
                async for chunk in self._chunks():
                    if self.closed or time.monotonic() >= self.deadline:
                        raise OutcomeUncertainError(
                            "Event stream closed or deadline expired", identity=self.identity
                        )
                    if not self.response.is_success:
                        if len(content) + len(chunk) > self.owner.maximum_response_bytes:
                            raise OutcomeUncertainError(
                                "Error response exceeded the byte bound", identity=self.identity
                            )
                        content.extend(chunk)
                        continue
                    for frame in decoder.feed(chunk):
                        if terminal:
                            raise ContractError(
                                "Event followed the terminal frame", identity=self.identity
                            )
                        terminal = frame.event in {"terminal", "error"}
                        yield frame
            if not self.response.is_success:
                _definite_error(
                    self.response, bytes(content), self.identity, self.operation, negotiated
                )
            if not terminal or decoder.buffer:
                raise OutcomeUncertainError(
                    "Strict event stream ended without a complete terminal", identity=self.identity
                )
        except asyncio.CancelledError as error:
            primary = MutationCancelledError(self.identity)
            raise primary from error
        except (httpx.HTTPError, OSError, TimeoutError) as error:
            primary = OutcomeUncertainError(
                "Strict event transport failed; recover metadata only", identity=self.identity
            )
            annotate_request_error(primary, self.operation, self.options)
            raise primary from error
        except BaseException as error:
            primary = error
            if isinstance(error, SynthError):
                annotate_request_error(error, self.operation, self.options)
            raise
        finally:
            try:
                await self.aclose()
            except (Exception, asyncio.CancelledError) as cause:
                handle_cleanup_failure(cause, primary, mutation=True, identity=self.identity)
