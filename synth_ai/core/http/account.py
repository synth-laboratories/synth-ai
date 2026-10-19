"""Strict account read decoding; never render provider bodies or credentials."""

from __future__ import annotations

import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RefusedError,
)

T = TypeVar("T", bound=BaseModel)


class _Refusal(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    service: str = Field(pattern=r"^backend\.account_billing$")
    status: int = Field(ge=400, le=599)


class _Error(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    error: _Refusal


def decode_account_read(response: httpx.Response, body: bytes, model: type[T]) -> T:
    if response.headers.get("X-Synth-Contract") != "account.v1":
        raise ContractError("Backend did not select account.v1", status=response.status_code)
    if (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ContractError(
            "Account response requires application/json", status=response.status_code
        )
    if response.status_code == 200:
        try:
            return model.model_validate_json(body)
        except ValidationError as error:
            raise ContractError("Account response violated its typed contract") from error
    try:
        refused = _Error.model_validate_json(body).error
        if refused.status != response.status_code:
            raise ValueError("refusal status differs")
    except (ValueError, ValidationError) as error:
        raise ContractError(
            "Account refusal violated its typed contract", status=response.status_code
        ) from error
    request_id = response.headers.get("X-Request-Id", "")
    request_id = request_id if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id) else None
    kind = {
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        429: RateLimitError,
    }.get(response.status_code, RefusedError)
    raise kind(
        "Account read refused",
        code=refused.code,
        status=response.status_code,
        request_id=request_id,
        retryable=response.status_code in {429, 502, 503, 504},
    )
