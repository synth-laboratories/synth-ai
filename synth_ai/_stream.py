"""SSE framing and resumable delivery state, independent of run completion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

from ._limits import CHECKPOINT_MAX_EVENTS, CHECKPOINT_MAX_ID_LENGTH, SSE_FRAME_MAX_BYTES


class StreamCheckpointData(BaseModel):
    """Versioned persistence protocol; mutable stream state is kept separately."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    run_id: str | None = None
    last_event_id: str | None = None
    pending_event_ids: list[str] = Field(default_factory=list, max_length=CHECKPOINT_MAX_EVENTS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("checkpoint schema version must be an integer")
        return value


@dataclass
class StreamCheckpoint:
    run_id: str | None = None
    last_event_id: str | None = None
    pending_event_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for value in (self.run_id, self.last_event_id, *self.pending_event_ids):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or len(value) > CHECKPOINT_MAX_ID_LENGTH
                or any(c in value for c in "\r\n\0")
            ):
                raise ValueError("invalid stream checkpoint identity")
        if len(self.pending_event_ids) > CHECKPOINT_MAX_EVENTS:
            raise ValueError("stream checkpoint exceeds the event bound")

    def bind(self, run_id: str) -> None:
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError("stream checkpoint belongs to another run")
        self.run_id = run_id

    def to_data(self) -> StreamCheckpointData:
        self.__post_init__()
        return StreamCheckpointData(
            run_id=self.run_id,
            last_event_id=self.last_event_id,
            pending_event_ids=sorted(self.pending_event_ids),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return TypeAdapter(dict[str, JsonValue]).validate_python(
            self.to_data().model_dump(mode="json"), strict=True
        )

    @classmethod
    def from_data(cls, value: StreamCheckpointData) -> StreamCheckpoint:
        return cls(
            run_id=value.run_id,
            last_event_id=value.last_event_id,
            pending_event_ids=set(value.pending_event_ids),
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> StreamCheckpoint:
        # An absent version reads the original v1 checkpoint format.
        return cls.from_data(StreamCheckpointData.model_validate(value))


@dataclass(frozen=True)
class Frame:
    event: str
    data: str
    id: str | None


async def bounded_lines(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Bound bytes before line assembly/UTF-8 decoding, including missing newlines.

    SSE permits LF, CRLF and CR. A delimiter may cross transport chunks; UTF-8
    code points may too. No unbounded text decoder or HTTP line buffer precedes
    this check. A trailing partial line cannot dispatch an event.
    """
    pending = bytearray()
    skip_lf, first = False, True
    async for chunk in chunks:
        offset = 0
        while offset < len(chunk):
            if skip_lf:
                skip_lf = False
                if chunk[offset] == 10:
                    offset += 1
                    continue
            lf, cr = chunk.find(b"\n", offset), chunk.find(b"\r", offset)
            end = cr if lf < 0 else lf if cr < 0 else min(lf, cr)
            stop = len(chunk) if end < 0 else end
            if len(pending) + stop - offset > SSE_FRAME_MAX_BYTES:
                raise ValueError("event line exceeds 1 MiB")
            pending.extend(memoryview(chunk)[offset:stop])
            if end < 0:
                break
            line = pending.decode("utf-8", errors="strict")
            pending.clear()
            if first:
                line = line.removeprefix("\ufeff")
                first = False
            yield line
            skip_lf = chunk[end] == 13
            offset = end + 1


async def frames(lines: AsyncIterator[str]) -> AsyncIterator[Frame]:
    data: list[str] = []
    event, event_id, size = "message", None, 0
    async for line in lines:
        size += len(line.encode("utf-8")) + 1
        if size > SSE_FRAME_MAX_BYTES:
            raise ValueError("event frame exceeds 1 MiB")
        if line == "":
            if data or event_id is not None:
                yield Frame(event, "\n".join(data), event_id)
            event, event_id, data, size = "message", None, cast(list[str], []), 0
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
        elif name == "id" and "\0" not in value:
            event_id = value
    # EOF does not dispatch an incomplete frame.
