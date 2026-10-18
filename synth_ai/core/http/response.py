"""Bounded response decoding and safe errors. See SDK index_contract.md."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synth_ai.errors import (
    POLICY_ERROR_TYPES,
    AuthenticationError,
    ConflictError,
    ContractError,
    NotFoundError,
    OutcomeUncertainError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    RefusedError,
    RequestTimeoutError,
    ResultNotReadyError,
    RetentionModeNotAllowed,
    SearchContentPurged,
    SynthError,
    UnsupportedOperationError,
)
from synth_ai.types.base import ClientTiming, Identifier, Model, ResponseModel
from synth_ai.types.capabilities import NegotiatedCapability
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import parse_rate_limits

T = TypeVar("T", bound=BaseModel)


class ErrorDetail(Model):
    code: Identifier
    request_id: Identifier | None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    # The exported index.v1 schema (and so its fingerprint, which the backend
    # capability receipt must equal) pins this default. Whether the server sent
    # the field at all is read from ``model_fields_set``, never from a nullable type.
    outcome: Literal["not_accepted", "unknown"] = "unknown"


class ErrorResponse(Model):
    error: ErrorDetail


class _TolerantErrorDetail(BaseModel):
    """Read-only view of a refusal that carries extra typed detail.

    ``ErrorDetail`` stays closed (its schema is fingerprinted). A refusal such as
    ``retention_mode_not_allowed`` may add ``detail.allowed_modes``; this view is
    consulted only when the closed model rejects the body, and only for the codes
    in ``_TOLERANT_REFUSAL_CODES``."""

    model_config = ConfigDict(extra="ignore")
    code: Identifier
    request_id: Identifier | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    outcome: Literal["not_accepted", "unknown"] = "unknown"
    detail: dict[str, object] | None = None
    allowed_modes: tuple[str, ...] | None = None

    def modes(self) -> tuple[str, ...]:
        candidate: object = self.allowed_modes
        if candidate is None and self.detail is not None:
            candidate = self.detail.get("allowed_modes")
        if isinstance(candidate, (list, tuple)) and all(isinstance(m, str) for m in candidate):
            return tuple(candidate)
        return ()


class _TolerantErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    error: _TolerantErrorDetail


_TOLERANT_REFUSAL_CODES = frozenset({"retention_mode_not_allowed"})


RETENTION_ALLOWED_MODES_HEADER = "X-Synth-Retention-Allowed-Modes"


def _header_text(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _header_modes(headers: Mapping[str, str]) -> tuple[str, ...]:
    """Comma-separated allowed retention modes from the response header."""
    raw = _header_text(headers, RETENTION_ALLOWED_MODES_HEADER)
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


UNCERTAIN_CLIENT_STATUSES = frozenset({408, 429})


def is_definite_refusal(status: int, outcome: str | None) -> bool:
    """A typed refusal proves nothing was created, unless the server says otherwise.

    An explicit ``outcome`` always wins. Without one, a typed 4xx other than
    408/429 is definite; 5xx, 408 and 429 stay ambiguous for mutations."""
    if outcome is not None:
        return outcome == "not_accepted"
    return 400 <= status < 500 and status not in UNCERTAIN_CLIENT_STATUSES


@dataclass(frozen=True, slots=True)
class RefusalDetail:
    """Allowlisted refusal headers (backend SYN-4730 / SYN-4763); nothing else is read."""

    stage: str | None = None
    cause: str | None = None
    closure_pending: str | None = None
    closure_expected_by: datetime | None = None


def _token(value: str | None) -> str | None:
    """Short identifier tokens only, mirroring the backend's own bound."""
    if value and len(value) <= 64 and value.isascii() and value.isidentifier():
        return value
    return None


def parse_refusal_detail(headers: Mapping[str, str]) -> RefusalDetail:
    """Parse the refusal stage/cause and pending-closure headers; malformed values are dropped.

    Closure pending and expected-by are valid together (timezone-aware ISO-8601) or not at all."""
    pending = _token(headers.get("X-Synth-Closure-Pending"))
    expected_by: datetime | None = None
    raw = headers.get("X-Synth-Closure-Expected-By")
    if pending is not None and raw is not None and len(raw) <= 64:
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            expected_by = parsed
    if expected_by is None:
        pending = None
    return RefusalDetail(
        stage=_token(headers.get("X-Synth-Refusal-Stage")),
        cause=_token(headers.get("X-Synth-Refusal-Cause")),
        closure_pending=pending,
        closure_expected_by=expected_by,
    )


def decode_response(
    response: httpx.Response,
    body: bytes,
    model: type[T],
    timing: ClientTiming,
    identity: MutationIdentity | None,
    mutation: bool,
    contract: str,
    *,
    negotiated: NegotiatedCapability | None = None,
) -> T:
    """Preserve bounded response capacity on every typed decoder refusal."""
    try:
        return _decode_response(
            response,
            body,
            model,
            timing,
            identity,
            mutation,
            contract,
            negotiated=negotiated,
        )
    except SynthError as error:
        error.rate_limits = parse_rate_limits(response.headers)
        detail = parse_refusal_detail(response.headers)
        error.stage = detail.stage
        error.cause = detail.cause
        error.closure_pending = detail.closure_pending
        error.closure_expected_by = detail.closure_expected_by
        if error.retry_after_seconds is None:
            delay = retry_delay(response, float("nan"))
            if math.isfinite(delay):
                error.retry_after_seconds = delay
        raise


def _decode_response(
    response: httpx.Response,
    body: bytes,
    model: type[T],
    timing: ClientTiming,
    identity: MutationIdentity | None,
    mutation: bool,
    contract: str,
    *,
    negotiated: NegotiatedCapability | None = None,
) -> T:
    """Validate negotiated version before accepting a body; never display raw bodies."""
    if contract == "tunnels.v2":
        from .tunnel_control import decode_control_response

        return decode_control_response(response, body, model, identity, mutation)
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
        allowed_modes: tuple[str, ...] = ()
        try:
            error = ErrorResponse.model_validate_json(body).error
            sent_outcome = error.outcome if "outcome" in error.model_fields_set else None
        except ValidationError as cause:
            try:
                tolerant = _TolerantErrorResponse.model_validate_json(body).error
            except ValidationError:
                tolerant = None
            if tolerant is None or tolerant.code not in _TOLERANT_REFUSAL_CODES:
                if mutation and response.status_code >= 500:
                    raise OutcomeUncertainError(
                        "Mutation outcome is uncertain",
                        identity=identity,
                        status=response.status_code,
                    ) from cause
                raise ContractError(
                    "Backend error response does not match the selected contract",
                    status=response.status_code,
                ) from cause
            allowed_modes = tolerant.modes()
            sent_outcome = tolerant.outcome if "outcome" in tolerant.model_fields_set else None
            error = ErrorDetail(
                code=tolerant.code,
                request_id=tolerant.request_id,
                retry_after_seconds=tolerant.retry_after_seconds,
            )
        definite = is_definite_refusal(response.status_code, sent_outcome)
        if mutation and not definite:
            raise OutcomeUncertainError(
                "Mutation outcome is uncertain",
                code=error.code,
                request_id=error.request_id,
                status=response.status_code,
                identity=identity,
                retry_after_seconds=error.retry_after_seconds,
            )
        kind: type[SynthError] = RefusedError if definite else SynthError
        if (response.status_code, error.code) in POLICY_ERROR_TYPES:
            kind = POLICY_ERROR_TYPES[(response.status_code, error.code)]
        elif error.code in {
            "operation_unavailable",
            "capabilities_unavailable",
            "schema_mismatch",
            "capability_mismatch",
            "release_mismatch",
            "capability_negotiation_required",
        }:
            kind = UnsupportedOperationError
        elif error.code == "result_not_ready":
            kind = ResultNotReadyError
        elif response.status_code == 401:
            kind = AuthenticationError
        elif response.status_code == 403:
            kind = PermissionDeniedError
        elif response.status_code == 404:
            kind = NotFoundError
        elif response.status_code in {409, 412}:
            kind = ConflictError
        elif response.status_code == 408:
            kind = RequestTimeoutError
        elif response.status_code == 429:
            kind = RateLimitError
        elif response.status_code == 402:
            kind = PaymentRequiredError
        refusal = kind(
            "Operation refused",
            retryable=False if definite else None,
            code=error.code,
            request_id=error.request_id,
            status=response.status_code,
            identity=identity,
            retry_after_seconds=error.retry_after_seconds,
        )
        if isinstance(refusal, RetentionModeNotAllowed):
            refusal.allowed_modes = allowed_modes or _header_modes(response.headers)
        if isinstance(refusal, SearchContentPurged):
            refusal.search_id = _header_text(response.headers, "X-Synth-Search-Id")
            refusal.purges_at = _header_text(response.headers, "X-Synth-Purges-At")
            refusal.purged_at = _header_text(response.headers, "X-Synth-Purged-At")
        raise refusal
    validate_negotiated_response(response, negotiated, identity, mutation)
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
    if not isinstance(result, ResponseModel):
        raise ContractError("Customer API responses require metadata")
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


def validate_negotiated_response(
    response: httpx.Response,
    negotiated: NegotiatedCapability | None,
    identity: MutationIdentity | None,
    mutation: bool,
) -> None:
    """Require a successful response from the same schema/capability/release.

    See: backend/notes/specifications/sdk/index_contract.md."""
    if negotiated is None or response.status_code not in {200, 206}:
        return
    pairs = (
        ("X-Synth-Schema-SHA256", negotiated.schema_sha256),
        ("X-Synth-Capabilities-SHA256", negotiated.capability_sha256),
        ("X-Synth-Release-Manifest-SHA256", negotiated.release_manifest_sha256),
    )
    if any(response.headers.get(name) != value for name, value in pairs):
        if mutation:
            raise OutcomeUncertainError(
                "Mutation response changed deployment identity; reconcile its identity",
                identity=identity,
                status=response.status_code,
            )
        raise ContractError("Response deployment identity does not match negotiation")
