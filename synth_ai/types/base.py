"""Strict requests and tolerant observations. See the contract snapshots in contracts/."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated, Any, Generic, Literal, NewType, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)
from typing_extensions import TypeVar

# Set only during synchronous response validation, including shared nested values.
# A ContextVar keeps threads/tasks isolated and the token is always reset.
_RESPONSE_DECODING: ContextVar[bool] = ContextVar("synth_response_decoding", default=False)
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
ResponseIdentifier = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
_RESPONSE_IDENTIFIER = TypeAdapter(ResponseIdentifier)


def _identifier(value: object, handler: ValidatorFunctionWrapHandler) -> str:
    if _RESPONSE_DECODING.get():
        return _RESPONSE_IDENTIFIER.validate_python(value, strict=True)
    return handler(value)


Identifier = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN),
    WrapValidator(_identifier),
]
# OAuth clients may use HTTPS metadata URLs; resource identifiers may not.
OAuthClientIdentifier = Annotated[str, Field(strict=True, min_length=1, max_length=2048)]
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


RequestModel = Model


class ResponseBase(BaseModel):
    """Forward-compatible observations; known values remain strict and immutable."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="never",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        from .response_values import response_json

        token = _RESPONSE_DECODING.set(True)
        try:
            return super().model_validate_json(response_json(json_data, cls), **kwargs)
        finally:
            _RESPONSE_DECODING.reset(token)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:
        from .response_values import response_value

        token = _RESPONSE_DECODING.set(True)
        try:
            return super().model_validate(response_value(obj, cls), **kwargs)
        finally:
            _RESPONSE_DECODING.reset(token)


class TimingSpan(ResponseBase):
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


class ClientTiming(ResponseBase):
    total_ms: Annotated[float, Field(ge=0)]
    first_byte_ms: Annotated[float, Field(ge=0)]
    attempts: Annotated[StrictInt, Field(ge=1)]
    source: Literal["sdk_monotonic"] = "sdk_monotonic"


class ResponseMetadata(ResponseBase):
    contract: Literal["index.v1", "swarms.v1", "core.v1"]
    request_id: ResponseIdentifier
    operation_id: ResponseIdentifier | None
    observed_at: AwareDatetime
    server_timing: tuple[TimingSpan, ...] = Field(default=(), max_length=32)
    client_timing: ClientTiming | None = None


class ResponseModel(ResponseBase):
    metadata: ResponseMetadata


class Page(ResponseModel, Generic[T, C]):
    """One server-authorized page. Wire bytes and requested counts bound reads.

    The advertised server limit is 1000; response decoding accepts future larger
    pages rather than rejecting otherwise valid observations.
    """

    items: tuple[T, ...] = Field(json_schema_extra={"maxItems": 1000})
    next_cursor: C | None
    complete: bool
