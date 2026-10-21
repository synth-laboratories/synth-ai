"""Resumable run snapshot stream. See backend swarms stream contract.

Each frame carries a complete ``RunSnapshot``; its event id is the run version
token. ``rotate`` and bounded transient failures reconnect with
``Last-Event-ID``; ``final`` ends observation. Closing never changes remote work.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator, Generator, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    ValidationError,
)

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.backoff import delay_seconds
from synth_ai.core.http.response import (
    TRANSIENT_STATUSES,
    classify_exchange,
    decode_response,
)
from synth_ai.errors import ContractError, RequestTimeoutError, SynthError, TransportError
from synth_ai.options import ObservationOptions
from synth_ai.types.base import ClientTiming

from .types.runs import RunSnapshot

if TYPE_CHECKING:
    from synth_ai.core.http.async_client import AsyncHttpTransport
    from synth_ai.core.http.client import HttpTransport

    from .run import AsyncSwarmRun, SwarmRun

EndReason = Literal["final", "rotate"]
EVENTS = frozenset({"snapshot", "end", "error"})
# The backend rotates each connection after 15 minutes; this window only bounds
# a connection whose rotation never arrives.
CONNECTION_WINDOW_SECONDS = 960.0
DEFAULT_RETRY_COUNT = 5
_EVENT_ID = re.compile(r"[!-~]{1,512}")
_END = TypeAdapter(dict[Literal["reason"], Literal["final", "rotate"]])


class _ErrorFrame(BaseModel):
    """The standard error body plus the HTTP status the refusal would have had."""

    model_config = ConfigDict(extra="forbid")
    status: StrictInt = Field(ge=400, le=599)
    error: dict[str, JsonValue]


def event_id(value: str) -> str:
    if not isinstance(value, str) or _EVENT_ID.fullmatch(value) is None:
        raise ValueError("Last-Event-ID must be 1-512 printable ASCII characters")
    return value


@dataclass(frozen=True, slots=True)
class SnapshotFrame:
    event: str
    data: bytes
    event_id: str | None


class SnapshotFrameDecoder:
    """Bounded ``id``/``event``/``data`` frames; single-line comments are keepalives."""

    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[SnapshotFrame]:
        offset = 0
        while offset < len(chunk):
            take = min(len(chunk) - offset, self.maximum_bytes + 2 - len(self.buffer))
            if take <= 0:
                raise ContractError("Run stream frame exceeded the configured byte bound")
            self.buffer.extend(chunk[offset : offset + take])
            offset += take
            yield from self._drain()

    def _drain(self) -> Iterator[SnapshotFrame]:
        while (end := self.buffer.find(b"\n\n")) >= 0:
            if end > self.maximum_bytes:
                raise ContractError("Run stream frame exceeded the configured byte bound")
            raw = bytes(self.buffer[:end])
            del self.buffer[: end + 2]
            frame = _parse(raw)
            if frame is not None:
                yield frame
        if len(self.buffer) > self.maximum_bytes and not (
            len(self.buffer) == self.maximum_bytes + 1 and self.buffer.endswith(b"\n")
        ):
            raise ContractError("Run stream frame exceeded the configured byte bound")


def _parse(raw: bytes) -> SnapshotFrame | None:
    fields: dict[bytes, bytes] = {}
    for line in raw.split(b"\n"):
        if line.startswith(b":"):
            continue
        name, separator, value = line.partition(b":")
        if not separator or name not in {b"id", b"event", b"data"} or name in fields:
            raise ContractError("Malformed run stream frame")
        fields[name] = value[1:] if value.startswith(b" ") else value
    if not fields:
        return None
    try:
        kind = fields[b"event"].decode("ascii")
        identifier = fields[b"id"].decode("ascii") if b"id" in fields else None
        data = fields[b"data"]
        data.decode("utf-8")
    except (KeyError, UnicodeError) as error:
        raise ContractError("Malformed run stream frame") from error
    if kind not in EVENTS:
        raise ContractError("Run stream event is outside its contract")
    if identifier is not None and _EVENT_ID.fullmatch(identifier) is None:
        raise ContractError("Run stream event id is outside its contract")
    if kind == "snapshot" and identifier is None:
        raise ContractError("Run stream snapshot lacks its version token")
    return SnapshotFrame(kind, data, identifier)


def _snapshot(frame: SnapshotFrame) -> RunSnapshot:
    try:
        snapshot = RunSnapshot.model_validate_json(frame.data)
    except ValidationError as error:
        raise ContractError("Run stream snapshot does not match the selected contract") from error
    if snapshot.metadata.contract != "swarms.v1":
        raise ContractError("Run stream snapshot metadata names a different contract")
    return snapshot


def _end_reason(frame: SnapshotFrame) -> EndReason:
    try:
        return _END.validate_json(frame.data)["reason"]
    except (ValidationError, KeyError) as error:
        raise ContractError("Run stream end frame is outside its contract") from error


def _error_frame(response: httpx.Response, frame: SnapshotFrame) -> SynthError:
    """Decode an error frame as the HTTP refusal its status and body describe."""
    try:
        value = _ErrorFrame.model_validate_json(frame.data)
    except ValidationError as error:
        raise ContractError("Run stream error frame is outside its contract") from error
    body = json.dumps({"error": value.error}, separators=(",", ":")).encode()
    synthetic = httpx.Response(
        value.status,
        headers={"X-Synth-Contract": "swarms.v1", "Content-Type": "application/json"},
        content=body,
        request=response.request,
    )
    try:
        decode_response(
            synthetic,
            body,
            RunSnapshot,
            ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
            None,
            False,
            "swarms.v1",
            operation=operations.STREAM.id,
        )
    except SynthError as error:
        return error
    return ContractError("Run stream error frame did not decode as a refusal")


def _check_stream(response: httpx.Response) -> None:
    exchange = classify_exchange(
        response,
        None,
        contract="swarms.v1",
        mutation=False,
        identity=None,
        attempt=1,
        attempts=1,
        operation=operations.STREAM.id,
    )
    if exchange.error is not None:
        raise exchange.error
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ContractError("Compressed run streams are outside the bounded contract")
    if response.headers.get("Content-Type", "").split(";")[0].strip() != "text/event-stream":
        raise ContractError("Run stream did not return the negotiated event stream")


def _refusal(response: httpx.Response, body: bytes) -> SynthError:
    try:
        decode_response(
            response,
            body,
            RunSnapshot,
            ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
            None,
            False,
            "swarms.v1",
            operation=operations.STREAM.id,
        )
    except SynthError as error:
        if response.status_code not in TRANSIENT_STATUSES:
            raise
        return error
    raise ContractError("Run stream returned an unexpected status", status=response.status_code)


class _Policy:
    def __init__(self, observation: ObservationOptions | None) -> None:
        self.deadline = (
            None if observation is None else time.monotonic() + observation.timeout_seconds
        )
        self.retry_count = (
            DEFAULT_RETRY_COUNT if observation is None else observation.retry_count_max
        )
        self.maximum_events = None if observation is None else observation.maximum_events
        self.failures = 0
        self.events = 0

    def window(self) -> float:
        if self.deadline is None:
            return CONNECTION_WINDOW_SECONDS
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _expired()
        return min(remaining, CONNECTION_WINDOW_SECONDS)

    def check(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _expired()

    def observed(self) -> None:
        self.failures = 0
        self.events += 1
        if self.maximum_events is not None and self.events > self.maximum_events:
            raise RequestTimeoutError(
                "Run watch reached its snapshot bound; remote work continues",
                code="run_observation_expired",
            )

    def backoff(self, failure: SynthError, transport: HttpTransport | AsyncHttpTransport) -> float:
        self.failures += 1
        if self.failures > self.retry_count:
            raise failure
        delay = delay_seconds(
            self.failures,
            transport._options.retry_base_delay_seconds,
            transport._options.retry_max_delay_seconds,
            failure.retry_after_seconds,
            rng=transport._random,
        )
        if self.deadline is not None and delay >= self.deadline - time.monotonic():
            raise _expired() from failure
        return delay


def _expired() -> RequestTimeoutError:
    return RequestTimeoutError(
        "Run watch deadline expired; remote work continues", code="run_observation_expired"
    )


def _interrupted(cause: httpx.TransportError) -> SynthError:
    if isinstance(cause, httpx.TimeoutException):
        return RequestTimeoutError("Run stream read timed out", retryable=True)
    return TransportError("Run stream interrupted", retryable=True)


def _open_failure(error: SynthError) -> bool:
    return isinstance(error, (TransportError, RequestTimeoutError))


class RunWatch:
    """Lazy single-consumer iterator of verified run snapshots.

    ``last_event_id`` is the newest snapshot version token, usable to resume;
    ``end_reason`` is ``"final"`` once the backend closes observation."""

    def __init__(
        self,
        run: SwarmRun,
        transport: HttpTransport,
        observation: ObservationOptions | None,
        last_event_id: str | None,
    ) -> None:
        self.last_event_id = None if last_event_id is None else event_id(last_event_id)
        self.end_reason: EndReason | None = None
        self._run = run
        self._transport = transport
        self._policy = _Policy(observation)
        self._iterator = self._snapshots()

    def __iter__(self) -> RunWatch:
        return self

    def __next__(self) -> RunSnapshot:
        return next(self._iterator)

    def __enter__(self) -> RunWatch:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop observing locally; the remote run continues."""
        self._iterator.close()

    def _snapshots(self) -> Generator[RunSnapshot, None, None]:
        transport, policy = self._transport, self._policy
        path = f"/swarms/runs/{self._run.run_id}/stream"
        while True:
            window = policy.window()
            headers = {} if self.last_event_id is None else {"Last-Event-ID": self.last_event_id}
            response: httpx.Response | None = None
            failure: SynthError
            try:
                try:
                    response = transport._open_stream(
                        operations.STREAM,
                        path=path,
                        query={},
                        remaining_seconds=window,
                        digest=None,
                        accept="text/event-stream",
                        headers=headers,
                    )
                except SynthError as error:
                    if not _open_failure(error):
                        raise
                    failure = error
                else:
                    if response.status_code != 200:
                        failure = _refusal(response, self._read_error(response))
                    else:
                        _check_stream(response)
                        decoder = SnapshotFrameDecoder(transport.maximum_response_bytes)
                        reason: EndReason | None = None
                        try:
                            for chunk in response.iter_bytes():
                                policy.check()
                                for frame in decoder.feed(chunk):
                                    if frame.event == "error":
                                        raise _error_frame(response, frame)
                                    if frame.event == "end":
                                        reason = _end_reason(frame)
                                        break
                                    snapshot = self._run._observe_snapshot(_snapshot(frame))
                                    self.last_event_id = frame.event_id
                                    policy.observed()
                                    yield snapshot
                                if reason is not None:
                                    break
                        except httpx.TransportError as cause:
                            policy.check()
                            failure = _interrupted(cause)
                        else:
                            if reason == "final":
                                self.end_reason = "final"
                                return
                            if reason == "rotate":
                                continue
                            failure = TransportError(
                                "Run stream ended without an end frame", retryable=True
                            )
            finally:
                if response is not None:
                    transport._release_response(response)
            transport._sleep(policy.backoff(failure, transport))

    def _read_error(self, response: httpx.Response) -> bytes:
        body = bytearray()
        for chunk in response.iter_bytes():
            if len(body) + len(chunk) > self._transport.maximum_response_bytes:
                raise ContractError("Run stream refusal exceeded the configured byte bound")
            body.extend(chunk)
        return bytes(body)


class AsyncRunWatch:
    """Native-async ``RunWatch``; no background task is started."""

    def __init__(
        self,
        run: AsyncSwarmRun,
        transport: AsyncHttpTransport,
        observation: ObservationOptions | None,
        last_event_id: str | None,
    ) -> None:
        self.last_event_id = None if last_event_id is None else event_id(last_event_id)
        self.end_reason: EndReason | None = None
        self._run = run
        self._transport = transport
        self._policy = _Policy(observation)
        self._iterator = self._snapshots()

    def __aiter__(self) -> AsyncRunWatch:
        return self

    async def __anext__(self) -> RunSnapshot:
        return await anext(self._iterator)

    async def __aenter__(self) -> AsyncRunWatch:
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop observing locally; the remote run continues."""
        await self._iterator.aclose()

    async def _snapshots(self) -> AsyncGenerator[RunSnapshot, None]:
        transport, policy = self._transport, self._policy
        path = f"/swarms/runs/{self._run.run_id}/stream"
        while True:
            window = policy.window()
            headers = {} if self.last_event_id is None else {"Last-Event-ID": self.last_event_id}
            response: httpx.Response | None = None
            failure: SynthError
            try:
                try:
                    response = await transport._open_stream(
                        operations.STREAM,
                        path=path,
                        query={},
                        remaining_seconds=window,
                        digest=None,
                        accept="text/event-stream",
                        headers=headers,
                    )
                except SynthError as error:
                    if not _open_failure(error):
                        raise
                    failure = error
                else:
                    if response.status_code != 200:
                        failure = _refusal(response, await self._read_error(response))
                    else:
                        _check_stream(response)
                        decoder = SnapshotFrameDecoder(transport.maximum_response_bytes)
                        reason: EndReason | None = None
                        try:
                            async for chunk in response.aiter_bytes():
                                policy.check()
                                for frame in decoder.feed(chunk):
                                    if frame.event == "error":
                                        raise _error_frame(response, frame)
                                    if frame.event == "end":
                                        reason = _end_reason(frame)
                                        break
                                    snapshot = self._run._observe_snapshot(_snapshot(frame))
                                    self.last_event_id = frame.event_id
                                    policy.observed()
                                    yield snapshot
                                if reason is not None:
                                    break
                        except httpx.TransportError as cause:
                            policy.check()
                            failure = _interrupted(cause)
                        else:
                            if reason == "final":
                                self.end_reason = "final"
                                return
                            if reason == "rotate":
                                continue
                            failure = TransportError(
                                "Run stream ended without an end frame", retryable=True
                            )
            finally:
                if response is not None:
                    await transport._release_response(response)
            await transport._sleep(policy.backoff(failure, transport))

    async def _read_error(self, response: httpx.Response) -> bytes:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self._transport.maximum_response_bytes:
                raise ContractError("Run stream refusal exceeded the configured byte bound")
            body.extend(chunk)
        return bytes(body)
