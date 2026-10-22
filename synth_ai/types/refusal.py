"""Versioned refusal document; see backend refusal_custody.md and contracts/refusal-v1/."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

MAXIMUM_DEPTH = 16

Token = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9_.:/{}-]{1,256}$")]
Code = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,127}$")]
Stage = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_.]{0,127}$")]


class Refusal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal["synth.refusal.v1"]
    code: Code
    service: Token
    stage: Stage
    retryable: StrictBool
    in_doubt: StrictBool
    operation_id: Token | None
    request_id: Token | None
    status: Annotated[StrictInt, Field(ge=100, le=599)] | None
    upstream: Refusal | None
    detail: dict[Token, Token] = Field(max_length=16)

    @model_validator(mode="before")
    @classmethod
    def bounded_chain(cls, value):
        cursor, depth = value, 0
        while isinstance(cursor, dict) and cursor.get("upstream") is not None:
            depth += 1
            if depth > MAXIMUM_DEPTH:
                raise ValueError("refusal_upstream_depth_invalid")
            cursor = cursor["upstream"]
        return value
