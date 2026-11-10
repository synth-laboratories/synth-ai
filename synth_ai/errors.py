"""Public Problem Details retain the server's refusal and outcome semantics."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ._stream import StreamCheckpoint

import httpx
from pydantic import ValidationError

from ._generated.models import Problem, ProblemOutcome, RunSnapshot


class ErrorKind(StrEnum):
    API = "api"
    TRANSPORT = "transport"
    CONTRACT = "contract"
    UPLOAD = "upload"
    OBSERVATION = "observation"
    CONFIGURATION = "configuration"
    INTEGRITY = "integrity"
    LIFECYCLE = "lifecycle"


class SynthError(Exception):
    """Base SDK error with a stable category and machine-readable code."""

    kind: ClassVar[ErrorKind] = ErrorKind.CONFIGURATION
    code: str = "sdk_error"


class ClientClosedError(SynthError, RuntimeError):
    kind = ErrorKind.LIFECYCLE
    code = "client_closed"


class DataIntegrityError(SynthError, ValueError):
    kind = ErrorKind.INTEGRITY
    code = "content_integrity_failed"


class PaginationError(SynthError, ValueError):
    code = "pagination_invalid"

    def __init__(self, message: str, *, cursor: str | None) -> None:
        super().__init__(message)
        self.cursor = cursor


class SwarmsAPIError(SynthError):
    kind = ErrorKind.API
    checkpoint: StreamCheckpoint | None = None

    def __init__(
        self, problem: Problem, *, request_id: str | None = None, idempotency_key: str | None = None
    ) -> None:
        super().__init__(f"{problem.code}: {problem.detail}")
        self.problem = problem
        self.code = problem.code
        self.status = problem.status
        self.service = problem.service
        self.stage = problem.stage
        self.outcome = problem.outcome
        self.request_id = request_id
        self.idempotency_key = idempotency_key


class InvalidRequestError(SwarmsAPIError):
    pass


class UnauthenticatedError(SwarmsAPIError):
    pass


class PaymentRequiredError(SwarmsAPIError):
    pass


class ForbiddenError(SwarmsAPIError):
    pass


class NotFoundError(SwarmsAPIError):
    pass


class ConflictError(SwarmsAPIError):
    pass


class RateLimitedError(SwarmsAPIError):
    pass


class UpstreamUnavailableError(SwarmsAPIError):
    pass


class ServerError(SwarmsAPIError):
    pass


class UpgradeRequiredError(SwarmsAPIError):
    """The API no longer serves this synth-ai version (426 client_upgrade_required).

    SDK releases are full cutovers with no support window; upgrade with
    `pip install -U synth-ai`. Never retried.
    """

    UPGRADE_HINT: ClassVar[str] = "pip install -U synth-ai"

    def __init__(
        self, problem: Problem, *, request_id: str | None = None, idempotency_key: str | None = None
    ) -> None:
        super().__init__(problem, request_id=request_id, idempotency_key=idempotency_key)
        self.minimum_version: str | None = problem.minimum_version
        self.client_version: str | None = problem.client_version


class ProviderPolicyError(InvalidRequestError):
    pass


class ProviderPresetNotAllowedError(ForbiddenError):
    pass


class ResourceAdmissionError(InvalidRequestError):
    pass


class MalformedErrorResponseError(SwarmsAPIError):
    kind = ErrorKind.CONTRACT


class TransportFailureError(SwarmsAPIError):
    """A transport or retryable stream failure outlasted its bounded reconnects.

    Observation failures retain the delivery checkpoint, the number of feed
    connections attempted and the last authoritative snapshot, when one was read.
    """

    kind = ErrorKind.TRANSPORT
    snapshot: RunSnapshot | None = None
    connection_attempts: int | None = None


class UploadTransferError(SynthError):
    kind = ErrorKind.UPLOAD
    code = "upload_incomplete"

    def __init__(self, upload_id: str, idempotency_key: str) -> None:
        super().__init__("Upload is incomplete; resume the retained upload")
        self.upload_id = upload_id
        self.idempotency_key = idempotency_key


class ResponseContractError(SwarmsAPIError):
    """A response could not be decoded; mutation identity remains recoverable."""

    kind = ErrorKind.CONTRACT


class RunObservationError(SynthError):
    kind = ErrorKind.OBSERVATION

    def __init__(
        self, message: str, *, snapshot: RunSnapshot | None = None, run_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.snapshot = snapshot
        self.run_id = snapshot.run_id if snapshot is not None else run_id


class RunFailedError(RunObservationError):
    code = "run_failed"


class WaitTimeoutError(RunObservationError):
    code = "wait_timeout"


class ResultUnavailableError(RunObservationError):
    code = "result_unavailable"


class RunHistoryUnavailableError(RunObservationError):
    code = "run_history_unavailable"


_BY_STATUS: dict[int, type[SwarmsAPIError]] = {
    400: InvalidRequestError,
    401: UnauthenticatedError,
    402: PaymentRequiredError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    410: NotFoundError,
    422: InvalidRequestError,
    426: UpgradeRequiredError,
    429: RateLimitedError,
    502: UpstreamUnavailableError,
    503: UpstreamUnavailableError,
    504: UpstreamUnavailableError,
}


_BY_CODE: dict[str, type[SwarmsAPIError]] = {
    "client_upgrade_required": UpgradeRequiredError,
    "provider_preset_not_allowed": ProviderPresetNotAllowedError,
    **dict.fromkeys(
        (
            "provider_model_unavailable",
            "provider_preset_unavailable",
            "provider_policy_invalid",
            "provider_policy_mismatch",
            "provider_model_mismatch",
            "routing_constraints_invalid",
            "routing_constraints_unsatisfiable",
        ),
        ProviderPolicyError,
    ),
    **dict.fromkeys(
        (
            "resource_request_invalid",
            "resource_limits_required",
            "grant_not_widenable",
            "resource_routing_unverified",
        ),
        ResourceAdmissionError,
    ),
}


def local_problem(
    code: str,
    stage: str,
    *,
    status: int = 503,
    unknown: bool = False,
    upstream_status: int | None = None,
    detail: str | None = None,
) -> Problem:
    return Problem(
        type="about:blank",
        title=code,
        status=status,
        detail=detail if detail is not None else code,
        instance="urn:synth-ai:request",
        code=code,
        service="synth-ai",
        stage=stage,
        outcome=ProblemOutcome.UNKNOWN if unknown else ProblemOutcome.NOT_ACCEPTED,
        upstream_status=upstream_status,
    )


def stream_interrupted(
    attempts: int, checkpoint: StreamCheckpoint, snapshot: RunSnapshot | None
) -> TransportFailureError:
    """Exhausted feed reconnects; recovery state stays on the error."""
    reconnects = attempts - 1
    error = TransportFailureError(
        local_problem(
            "event_stream_interrupted",
            "listRunEvents",
            detail=(
                f"event stream ended without completion after {attempts} connection "
                f"attempt{'s' if attempts != 1 else ''} ({reconnects} "
                f"reconnect{'s' if reconnects != 1 else ''})"
            ),
        )
    )
    error.checkpoint = checkpoint
    error.snapshot = snapshot
    error.connection_attempts = attempts
    return error


def raise_for_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        problem = Problem.model_validate_json(response.content)
        if problem.status != response.status_code:
            raise ValueError("status mismatch")
    except (ValidationError, ValueError):
        raise MalformedErrorResponseError(
            local_problem(
                "malformed_error_response",
                "decode",
                status=response.status_code if 400 <= response.status_code <= 599 else 502,
                upstream_status=response.status_code,
                unknown=response.request.method not in {"GET", "HEAD"},
            ),
            request_id=response.headers.get("x-request-id")
            or response.request.headers.get("x-request-id"),
            idempotency_key=response.request.headers.get("idempotency-key"),
        ) from None
    raise_problem(
        problem,
        request_id=response.headers.get("x-request-id")
        or response.request.headers.get("x-request-id"),
        idempotency_key=response.request.headers.get("idempotency-key"),
    )


def raise_problem(
    problem: Problem, *, request_id: str | None = None, idempotency_key: str | None = None
) -> None:
    kind = _BY_CODE.get(problem.code) or _BY_STATUS.get(
        problem.status, ServerError if problem.status >= 500 else InvalidRequestError
    )
    raise kind(problem, request_id=request_id, idempotency_key=idempotency_key)
