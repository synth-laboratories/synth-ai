"""Decode the existing catalog v2 wire protocol; see backend tunnels/v2_catalog.md."""

from __future__ import annotations

import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

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


def decode_control_response(
    response: httpx.Response,
    body: bytes,
    model: type[T],
    identity: MutationIdentity | None,
    mutation: bool,
) -> T:
    if (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ContractError("Tunnel control responses require application/json")
    if response.headers.get("X-Synth-Contract") not in {None, "tunnels.v2"}:
        raise ContractError("Tunnel control response selected another contract")
    if response.status_code == 200:
        try:
            return model.model_validate_json(body)
        except ValidationError as error:
            raise ContractError("Tunnel response violated its typed contract") from error
    status = response.status_code
    if mutation and (status >= 500 or status in {408, 429} or 300 <= status < 400):
        raise OutcomeUncertainError(
            "Tunnel command outcome is uncertain; reconcile its UUID",
            identity=identity,
            status=status,
        )
    code = None
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise ContractError(
            "Tunnel refusal must be a bounded JSON object", status=status
        ) from error
    detail = value.get("detail") if isinstance(value, dict) else None
    if isinstance(detail, dict):
        detail = detail.get("code")
    if isinstance(detail, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", detail):
        code = detail
    if mutation and code is None:
        raise OutcomeUncertainError(
            "Unverified Tunnel refusal; reconcile its UUID", identity=identity, status=status
        )
    kind = {
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        409: ConflictError,
        412: ConflictError,
        429: RateLimitError,
    }.get(status, RefusedError)
    raise kind(
        "Tunnel control operation refused",
        code=code,
        status=status,
        identity=identity,
        retryable=False,
    )
