"""Decode ``synth.menagerie.v1`` responses; never render raw bodies."""

from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synth_ai.errors import (
    AuthenticationError,
    ConflictError,
    ContractError,
    NotFoundError,
    OutcomeUncertainError,
    PermissionDeniedError,
    RateLimitError,
    RefusedError,
)
from synth_ai.types.identity import MutationIdentity

T = TypeVar("T", bound=BaseModel)


class _Detail(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    error_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    service: str = Field(pattern=r"^backend\.menagerie$")
    stage: str = Field(max_length=64)
    cause: str = Field(max_length=2048)


class _Refusal(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    detail: _Detail


def decode_menagerie_response(
    response: httpx.Response,
    body: bytes,
    model: type[T],
    identity: MutationIdentity | None,
    mutation: bool,
) -> T:
    status = response.status_code
    if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != (
        "application/json"
    ):
        raise ContractError("Menagerie responses require application/json", status=status)
    if status == 200:
        try:
            return model.model_validate_json(body)
        except ValidationError as error:
            raise ContractError("Menagerie response violated its typed contract") from error
    try:
        detail = _Refusal.model_validate_json(body).detail
    except ValidationError as error:
        if mutation:
            raise OutcomeUncertainError(
                "Menagerie write outcome is uncertain; repeat the same idempotent call",
                identity=identity,
                status=status,
            ) from error
        raise ContractError(
            "Menagerie refusal violated its typed contract", status=status
        ) from error
    if mutation and status >= 500:
        raise OutcomeUncertainError(
            "Menagerie write outcome is uncertain; repeat the same idempotent call",
            code=detail.error_code,
            identity=identity,
            status=status,
            stage=detail.stage,
            cause=detail.cause,
        )
    kind = {
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        409: ConflictError,
        429: RateLimitError,
    }.get(status, RefusedError)
    raise kind(
        f"Menagerie refused ({detail.error_code}) at {detail.stage}: {detail.cause}",
        code=detail.error_code,
        status=status,
        identity=identity,
        retryable=status == 503,
        stage=detail.stage,
        cause=detail.cause,
    )
