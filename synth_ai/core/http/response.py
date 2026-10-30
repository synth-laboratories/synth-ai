"""Bounded response decoding and safe errors. See SDK index_contract.md."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.core.version_skew import schema_error
from synth_ai.errors import (
    LIFECYCLE_WRITE_REFUSAL_CODES,
    POLICY_ERROR_TYPES,
    AuthenticationError,
    ConflictError,
    ContractError,
    LifecycleWriteRefusedError,
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
    TransportError,
    UnsupportedOperationError,
)
from synth_ai.types.base import (
    ClientTiming,
    Identifier,
    ResponseBase,
    ResponseIdentifier,
    ResponseModel,
)
from synth_ai.types.capabilities import NegotiatedCapability
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import parse_rate_limits
from synth_ai.types.refusal import Refusal

from .backoff import retry_after_seconds

T = TypeVar("T", bound=BaseModel)


class ErrorDetail(ResponseBase):
    code: ResponseIdentifier
    request_id: ResponseIdentifier | None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    # The exported index.v1 schema (and so its fingerprint, which the backend
    # capability receipt must equal) pins this default. Whether the server sent
    # the field at all is read from ``model_fields_set``, never from a nullable type.
    outcome: Literal["not_accepted", "unknown"] = "unknown"


class ErrorResponse(ResponseBase):
    error: ErrorDetail


# SYN-4551: rich error bodies are negotiated per request, never by environment.
# This SDK sends ``X-Synth-Accept-Errors: synth.errors.v2``; a backend that
# honours it marks the body with ``X-Synth-Error-Contract: synth.errors.v2``.
# Without that response header the body is decoded as v1 (older backends).
ACCEPT_ERRORS_HEADER = "X-Synth-Accept-Errors"
ERROR_CONTRACT_HEADER = "X-Synth-Error-Contract"
ERRORS_V2 = "synth.errors.v2"
_CAUSE_TOKEN = r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"


class ErrorDetailV2(ErrorDetail):
    """``ErrorDetail`` plus the typed cause fields; each is omitted when unknown.

    Not part of the fingerprinted ``index.v1`` export: v2 is a separately
    negotiated error contract, so old SDKs keep their pinned schema."""

    service: str | None = Field(default=None, max_length=32, pattern=_CAUSE_TOKEN)
    stage: str | None = Field(default=None, max_length=64, pattern=_CAUSE_TOKEN)
    cause_code: str | None = Field(default=None, max_length=64, pattern=_CAUSE_TOKEN)
    upstream_status: int | None = Field(default=None, ge=100, le=599)


class ErrorResponseV2(ResponseBase):
    error: ErrorDetailV2


class RefusalErrorDetail(ErrorDetailV2):
    model_config = ConfigDict(extra="forbid")
    refusal: Refusal


class RefusalErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: RefusalErrorDetail


def _decode_error(
    response: httpx.Response, body: bytes
) -> ErrorDetail | ErrorDetailV2 | RefusalErrorDetail:
    error_contract = response.headers.get(ERROR_CONTRACT_HEADER)
    if error_contract is not None and error_contract != ERRORS_V2:
        raise ContractError("Unsupported error contract", status=response.status_code)
    contract = response.headers.get("X-Synth-Refusal-Contract")
    if contract is None:
        if error_contract == ERRORS_V2:
            return ErrorResponseV2.model_validate_json(body).error
        return ErrorResponse.model_validate_json(body).error
    if contract != "synth.refusal.v1":
        raise ContractError("Unsupported refusal contract", status=response.status_code)
    if len(body) > 262_144:
        raise ContractError(
            "Versioned refusal document exceeds its bound", status=response.status_code
        )
    error = RefusalErrorResponse.model_validate_json(body).error
    if error.code != error.refusal.code:
        raise ContractError("Refusal code does not match error code", status=response.status_code)
    return error


def _cause_fields(error: ErrorDetail) -> dict[str, object]:
    """Typed v2 cause fields for the exception; empty for a v1 body."""
    if not isinstance(error, ErrorDetailV2):
        return {}
    return {
        "service": error.service,
        "stage": error.stage,
        "cause": error.cause_code,
        "upstream_status": error.upstream_status,
    }


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


def refusal_document_is_definite(status: int, refusal: Refusal | None) -> bool:
    """A versioned ``synth.refusal.v1`` that says not retryable and not in doubt is
    definite, even when the legacy ``outcome`` field says ``unknown`` (SYN-5273).

    The backend derives ``in_doubt`` from the route outcome (backend#3595), so the
    document's own ``in_doubt`` is trusted for every code; no code is exempt. Only
    a 4xx other than 408/429 qualifies: a 5xx document stays ambiguous for
    mutations because older backends synthesize ``in_doubt=false`` on outages."""
    return (
        refusal is not None
        and refusal.retryable is False
        and refusal.in_doubt is False
        and 400 <= status < 500
        and status not in UNCERTAIN_CLIENT_STATUSES
    )


@dataclass(frozen=True, slots=True)
class RefusalDetail:
    """Allowlisted refusal headers (backend SYN-4730 / SYN-4763); nothing else is read."""

    stage: str | None = None
    cause: str | None = None
    closure_pending: str | None = None
    closure_expected_by: datetime | None = None
    durable_search_id: str | None = None
    sync_cap_seconds: float | None = None


def _durable_search_id(value: str | None) -> str | None:
    """A canonical UUID only; anything else is dropped (SYN-4999)."""
    if value is None or len(value) > 64:
        return None
    try:
        return str(UUID(value.strip()))
    except ValueError:
        return None


def _cap_seconds(value: str | None) -> float | None:
    if value is None or len(value) > 16:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


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
        durable_search_id=_durable_search_id(headers.get("X-Synth-Durable-Search-Id")),
        sync_cap_seconds=_cap_seconds(headers.get("X-Synth-Deep-Sync-Cap-Seconds")),
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
    operation: str | None = None,
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
            operation=operation,
        )
    except SynthError as error:
        annotate_response_error(response, error)
        raise


def annotate_response_error(response: httpx.Response, error: SynthError) -> SynthError:
    """Attach only bounded, allowlisted capacity and refusal metadata."""
    error.rate_limits = parse_rate_limits(response.headers)
    detail = parse_refusal_detail(response.headers)
    error.stage = detail.stage or error.stage
    error.cause = detail.cause or error.cause
    error.closure_pending = detail.closure_pending
    error.closure_expected_by = detail.closure_expected_by
    error.durable_search_id = detail.durable_search_id or error.durable_search_id
    error.sync_cap_seconds = detail.sync_cap_seconds or error.sync_cap_seconds
    if error.retry_after_seconds is None:
        delay = retry_delay(response, float("nan"))
        if math.isfinite(delay):
            error.retry_after_seconds = delay
    return error


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
    operation: str | None = None,
) -> T:
    """Validate negotiated version before accepting a body; never display raw bodies."""
    if contract == "tunnels.v2":
        from .tunnel_control import decode_control_response

        return decode_control_response(response, body, model, identity, mutation)
    if contract == "menagerie.v1":
        from .menagerie import decode_menagerie_response

        return decode_menagerie_response(response, body, model, identity, mutation)
    if contract == "account.v1":
        from .account import decode_account_read

        if mutation:
            raise ContractError("Account companion supports reads only")
        return decode_account_read(response, body, model)
    exchange = classify_exchange(
        response,
        body,
        contract=contract,
        mutation=mutation,
        identity=identity,
        attempt=1,
        attempts=1,
    )
    if exchange.error is not None:
        raise exchange.error
    if (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ContractError("Contract responses must be application/json")
    if response.status_code != 200:
        allowed_modes: tuple[str, ...] = ()
        try:
            error = _decode_error(response, body)
            sent_outcome = error.outcome if "outcome" in error.model_fields_set else None
            if error.code in _TOLERANT_REFUSAL_CODES:
                allowed_modes = _TolerantErrorResponse.model_validate_json(body).error.modes()
        except (ValidationError, ContractError) as cause:
            if response.headers.get("X-Synth-Refusal-Contract") is not None:
                if mutation:
                    raise OutcomeUncertainError(
                        "Malformed versioned refusal; reconcile the mutation identity",
                        identity=identity,
                        status=response.status_code,
                        reason="response_invalid",
                    ) from cause
                raise ContractError(
                    "Malformed versioned refusal document", status=response.status_code
                ) from cause
            try:
                tolerant = _TolerantErrorResponse.model_validate_json(body).error
            except ValidationError:
                tolerant = None
            if tolerant is None or tolerant.code not in _TOLERANT_REFUSAL_CODES:
                from .quota import decode_quota_refusal, quota_code

                if quota_code(body) is not None:
                    # SYN-4844: a typed SMR quota refusal (403/409) is definite;
                    # a malformed one raises the typed ContractError.
                    raise decode_quota_refusal(
                        response.status_code,
                        body,
                        operation=operation,
                        request_id=response.headers.get("X-Request-Id"),
                    ) from cause
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
        definite = is_definite_refusal(
            response.status_code, sent_outcome
        ) or refusal_document_is_definite(response.status_code, getattr(error, "refusal", None))
        if mutation and not definite:
            uncertain = (
                LifecycleWriteRefusedError
                if error.code in LIFECYCLE_WRITE_REFUSAL_CODES
                else OutcomeUncertainError
            )
            raise uncertain(
                "Mutation outcome is uncertain",
                code=error.code,
                refusal=getattr(error, "refusal", None),
                request_id=error.request_id,
                status=response.status_code,
                identity=identity,
                retry_after_seconds=error.retry_after_seconds,
                **_cause_fields(error),
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
            f"{operation or contract} refused: {error.code}",
            refusal=getattr(error, "refusal", None),
            retryable=False if definite else None,
            code=error.code,
            request_id=error.request_id,
            status=response.status_code,
            identity=identity,
            retry_after_seconds=error.retry_after_seconds,
            **_cause_fields(error),
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
        skew = _response_schema_error(response, contract, negotiated)
        if mutation:
            raise OutcomeUncertainError(
                "Mutation response is invalid; reconcile its identity. " + str(skew),
                identity=identity,
                reason="response_invalid",
                status=200,
            ) from cause
        raise skew from cause
    if not isinstance(result, ResponseModel):
        raise ContractError("Customer API responses require metadata")
    if result.metadata.contract != contract:
        raise ContractError("Response metadata contract does not match negotiated contract")
    return result.model_copy(
        update={"metadata": result.metadata.model_copy(update={"client_timing": timing})}
    )


@dataclass(frozen=True, slots=True)
class Exchange:
    kind: Literal["decode", "retry", "fail"]
    delay: float = 0.0
    error: SynthError | None = None


TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def classify_exchange(
    response: httpx.Response,
    body: bytes | None,
    *,
    contract: str,
    mutation: bool,
    identity: MutationIdentity | None,
    attempt: int,
    attempts: int,
    operation: str | None = None,
) -> Exchange:
    """Classify infrastructure before contract skew; never retry a mutation.

    A headerless 500 is transient for reads; a negotiated, typed 500 keeps its
    existing final decoder behavior. Matching typed 408/429/502/503/504 reads
    retain bounded retry. A missing header on 2xx remains contract skew.
    ``body=None`` permits rejecting infrastructure without reading untrusted
    compressed content; a matching identity response needs its bounded body.
    Tunnel control keeps its separately versioned decoder.
    """
    if contract in {"tunnels.v2", "account.v1", "menagerie.v1"}:
        return Exchange("decode")
    negotiated = response.headers.get("X-Synth-Contract") == contract
    if response.status_code in TRANSIENT_STATUSES:
        typed = False
        explicit_retryable = None
        bounded_encoding = (
            response.headers.get("Content-Encoding", "identity").lower() == "identity"
        )
        if negotiated and bounded_encoding and body is None:
            return Exchange("decode")
        if (
            negotiated
            and bounded_encoding
            and response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            == "application/json"
        ):
            try:
                parsed = _decode_error(response, body or b"")
                if isinstance(parsed, RefusalErrorDetail):
                    explicit_retryable = parsed.refusal.retryable
                typed = True
            except (ValidationError, ContractError):
                if response.headers.get("X-Synth-Refusal-Contract") is not None:
                    return Exchange("decode")
                pass  # The infrastructure disposition below retains this failure.
        if not typed:
            if mutation:
                return Exchange(
                    "fail",
                    error=OutcomeUncertainError(
                        "Upstream infrastructure response; reconcile the mutation identity",
                        reason="infrastructure_status",
                        identity=identity,
                        status=response.status_code,
                        stage="infrastructure",
                    ),
                )
            if attempt < attempts:
                return Exchange("retry", retry_delay(response, 0.0))
            return Exchange(
                "fail",
                error=TransportError(
                    f"{operation or contract}: upstream returned HTTP {response.status_code} without a valid Synth contract (proxy, CDN or deploy in progress)",
                    status=response.status_code,
                    retryable=True,
                    stage="infrastructure",
                ),
            )
        if (
            not mutation
            and response.status_code != 500
            and attempt < attempts
            and explicit_retryable is not False
        ):
            return Exchange("retry", retry_delay(response, 0.0))
    if not negotiated:
        if mutation:
            return Exchange(
                "fail",
                error=OutcomeUncertainError(
                    "Mutation response has an incompatible contract; reconcile its identity",
                    reason="contract_header_missing",
                    identity=identity,
                    status=response.status_code,
                ),
            )
        return Exchange(
            "fail",
            error=ContractError(
                "Backend did not negotiate the selected contract; upgrade synth-ai if this persists",
                status=response.status_code,
            ),
        )
    return Exchange("decode")


def retry_delay(response: httpx.Response, fallback: float) -> float:
    seconds = retry_after_seconds(response.headers.get("Retry-After"))
    return fallback if seconds is None else seconds


def validate_negotiated_response(
    response: httpx.Response,
    negotiated: NegotiatedCapability | None,
    identity: MutationIdentity | None,
    mutation: bool,
) -> None:
    """Require a successful response from the same schema/capability/release.

    See: the public contract snapshots in contracts/"""
    if (
        negotiated is None
        or not negotiated.strict_contract
        or response.status_code not in {200, 206}
    ):
        return
    pairs = (
        ("X-Synth-Schema-SHA256", negotiated.schema_sha256),
        ("X-Synth-Capabilities-SHA256", negotiated.capability_sha256),
        ("X-Synth-Release-Manifest-SHA256", negotiated.release_manifest_sha256),
    )
    if any(response.headers.get(name) != value for name, value in pairs):
        skew = _response_schema_error(
            response, negotiated.contract or "selected contract", negotiated
        )
        if mutation:
            raise OutcomeUncertainError(
                "Mutation response changed deployment identity; reconcile its identity",
                identity=identity,
                reason="response_invalid",
                status=response.status_code,
            ) from skew
        raise skew


def _response_schema_error(
    response: httpx.Response, contract: str, negotiated: NegotiatedCapability | None
) -> ContractError:
    sdk_schema = negotiated.sdk_schema_sha256 if negotiated is not None else None
    if sdk_schema is None:
        sdk_schema = (
            schema_sha256(contract)
            if contract in {"index.v1", "swarms.v1", "core.v1"}
            else "unknown"
        )
    raw = response.headers.get("X-Synth-Schema-SHA256", "")
    backend_schema = (
        raw
        if re.fullmatch(r"[0-9a-f]{64}", raw)
        else (negotiated.schema_sha256 if negotiated is not None else None)
    )
    error = schema_error(
        contract=contract,
        sdk_schema=sdk_schema,
        backend_schema=backend_schema,
        backend_release=negotiated.backend_release if negotiated is not None else None,
        minimum_sdk_version=negotiated.minimum_sdk_version if negotiated is not None else None,
        stage="response_decode",
    )
    error.status = response.status_code
    return error
