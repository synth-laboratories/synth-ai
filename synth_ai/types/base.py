"""Closed customer contracts. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

from typing import Annotated, Generic, Literal, NewType, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, model_validator
from typing_extensions import TypeVar

Identifier = Annotated[
    str, Field(strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Cursor = Annotated[str, Field(strict=True, min_length=1, max_length=2048)]
NonNegative = Annotated[StrictInt, Field(ge=0)]
PageCursor = NewType("PageCursor", str)
EventCursor = NewType("EventCursor", str)
T = TypeVar("T")
C = TypeVar("C", default=PageCursor)


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class TimingSpan(Model):
    name: Literal["request", "queue", "first_useful", "active", "wall", "settlement"]
    duration_ms: Annotated[float, Field(ge=0)] | None
    source: Literal["server", "worker", "unknown"]
    clock: Literal["monotonic", "utc", "unknown"]
    uncertainty_ms: Annotated[float, Field(ge=0)] | None

    @model_validator(mode="after")
    def validate_timing_source(self) -> Self:
        if self.duration_ms is not None and (self.source == "unknown" or self.clock == "unknown"):
            raise ValueError("observed timing requires a known source and clock")
        return self


class ClientTiming(Model):
    total_ms: Annotated[float, Field(ge=0)]
    first_byte_ms: Annotated[float, Field(ge=0)]
    attempts: Annotated[StrictInt, Field(ge=1)]
    source: Literal["sdk_monotonic"] = "sdk_monotonic"


class ResponseMetadata(Model):
    contract: Literal["index.v1", "swarms.v1"]
    request_id: Identifier
    operation_id: Identifier | None
    observed_at: AwareDatetime
    server_timing: tuple[TimingSpan, ...] = Field(default=(), max_length=32)
    client_timing: ClientTiming | None = None


class ResponseModel(Model):
    metadata: ResponseMetadata


class Page(ResponseModel, Generic[T, C]):
    items: tuple[T, ...] = Field(max_length=200)
    next_cursor: C | None
    complete: bool
