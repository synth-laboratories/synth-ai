"""SSE framing and resumable delivery state, independent of run completion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class StreamCheckpoint:
    run_id: str | None = None
    last_event_id: str | None = None
    pending_event_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for value in (self.run_id, self.last_event_id, *self.pending_event_ids):
            if value is not None and (
                not isinstance(value, str) or not value or len(value) > 1024
                or any(c in value for c in "\r\n\0")
            ):
                raise ValueError("invalid stream checkpoint identity")
        if len(self.pending_event_ids) > 4096:
            raise ValueError("stream checkpoint exceeds the event bound")

    def bind(self, run_id: str) -> None:
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError("stream checkpoint belongs to another run")
        self.run_id = run_id

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "last_event_id": self.last_event_id,
            "pending_event_ids": sorted(self.pending_event_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> StreamCheckpoint:
        if set(value) != {"run_id", "last_event_id", "pending_event_ids"}:
            raise ValueError("invalid stream checkpoint")
        run_id, last_id, pending = (
            value["run_id"],
            value["last_event_id"],
            value["pending_event_ids"],
        )
        if (
            run_id is not None
            and not isinstance(run_id, str)
            or last_id is not None
            and not isinstance(last_id, str)
            or not isinstance(pending, list)
            or any(not isinstance(item, str) for item in pending)
        ):
            raise ValueError("invalid stream checkpoint")
        return cls(run_id=run_id, last_event_id=last_id, pending_event_ids=set(pending))


@dataclass(frozen=True)
class Frame:
    event: str
    data: str
    id: str | None


async def frames(lines: AsyncIterator[str]) -> AsyncIterator[Frame]:
    data: list[str] = []
    event, event_id, size = "message", None, 0
    async for line in lines:
        size += len(line.encode("utf-8"))
        if size > 1 << 20:
            raise ValueError("event frame exceeds 1 MiB")
        if line == "":
            if data or event_id is not None:
                yield Frame(event, "\n".join(data), event_id)
            event, event_id, data, size = "message", None, [], 0
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
