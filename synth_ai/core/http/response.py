"""Bounded response decoding and safe errors. See SDK index_contract.md."""

from __future__ import annotations

import math
from typing import Literal, TypeVar

import httpx
from pydantic import Field, ValidationError

from synth_ai.errors import (
    AuthenticationError,
    BudgetExceededError,
    ConflictError,
    ContractError,
    NotFoundError,
    OutcomeUncertainError,
    PermissionDeniedError,
    RateLimitError,
    ResultNotReadyError,
    SynthError,
)
from synth_ai.types.base import ClientTiming, Identifier, Model, ResponseModel
from synth_ai.types.identity import MutationIdentity

T = TypeVar("T", bound=ResponseModel)


class ErrorDetail(Model):
    code: Identifier
    request_id: Identifier | None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    outcome: Literal["not_accepted", "unknown"] = "unknown"


class ErrorResponse(Model):
    error: ErrorDetail


def decode_response(
    response: httpx.Response,
    body: bytes,
    model: type[T],
    timing: ClientTiming,
    identity: MutationIdentity | None,
    mutation: bool,
    contract: str,
) -> T:
    """Validate negotiated version before accepting a body; never display raw bodies."""
    if response.headers.get("X-Synth-Contract") != contract:
        if mutation:
            raise OutcomeUncertainError(
                "Mutation response has an incompatible contract; reconcile its identity",
                identity=identity,
                status=response.status_code,
            )
        raise ContractError(
            "Backend did not negotiate the selected contract", status=response.status_code
        )
    if (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ContractError("Contract responses must be application/json")
    if response.status_code != 200:
        try:
            error = ErrorResponse.model_validate_json(body).error
        except ValidationError as cause:
            if mutation and response.status_code >= 500:
                raise OutcomeUncertainError(
                    "Mutation outcome is uncertain", identity=identity, status=response.status_code
                ) from cause
            raise ContractError(
                "Backend error response does not match the selected contract",
                status=response.status_code,
            ) from cause
        if mutation and error.outcome == "unknown":
            raise OutcomeUncertainError(
                "Mutation outcome is uncertain",
                code=error.code,
                request_id=error.request_id,
                status=response.status_code,
                identity=identity,
                retry_after_seconds=error.retry_after_seconds,
            )
        kind: type[SynthError] = SynthError
        if error.code == "result_not_ready":
            kind = ResultNotReadyError
        elif response.status_code == 401:
            kind = AuthenticationError
        elif response.status_code == 403:
            kind = PermissionDeniedError
        elif response.status_code == 404:
            kind = NotFoundError
        elif response.status_code in {409, 412}:
            kind = ConflictError
        elif response.status_code == 429:
            kind = RateLimitError
        elif response.status_code == 402:
            kind = BudgetExceededError
        raise kind(
            "Operation refused",
            code=error.code,
            request_id=error.request_id,
            status=response.status_code,
            identity=identity,
            retry_after_seconds=error.retry_after_seconds,
        )
    try:
        result = model.model_validate_json(body)
    except ValidationError as cause:
        if mutation:
            raise OutcomeUncertainError(
                "Mutation response is invalid; reconcile its identity",
                identity=identity,
                status=200,
            ) from cause
        raise ContractError(
            "Backend response does not match the typed contract", status=200
        ) from cause
    if result.metadata.contract != contract:
        raise ContractError("Response metadata contract does not match negotiated contract")
    return result.model_copy(
        update={"metadata": result.metadata.model_copy(update={"client_timing": timing})}
    )


def retry_delay(response: httpx.Response, fallback: float) -> float:
    value = response.headers.get("Retry-After")
    if value is None:
        return fallback
    try:
        seconds = float(value)
    except ValueError:
        return fallback
    return seconds if math.isfinite(seconds) and seconds >= 0 else fallback
