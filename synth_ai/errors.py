"""Public Problem Details retain the server's refusal and outcome semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._stream import StreamCheckpoint

import httpx
from pydantic import ValidationError

from ._generated.models import Problem, RunSnapshot


class SynthError(Exception):
    """Base SDK error."""


class SwarmsAPIError(SynthError):
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


class ProviderPolicyError(InvalidRequestError):
    pass


class ProviderPresetNotAllowedError(ForbiddenError):
    pass


class ResourceAdmissionError(InvalidRequestError):
    pass


class MalformedErrorResponseError(SwarmsAPIError):
    pass


class TransportFailureError(SwarmsAPIError):
    pass


class UploadTransferError(SynthError):
    def __init__(self, upload_id: str, idempotency_key: str) -> None:
        super().__init__("Upload is incomplete; resume the retained upload")
        self.upload_id = upload_id
        self.idempotency_key = idempotency_key


class ResponseContractError(SwarmsAPIError):
    """A response could not be decoded; mutation identity remains recoverable."""


class RunObservationError(SynthError):
    def __init__(
        self, message: str, *, snapshot: RunSnapshot | None = None, run_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.snapshot = snapshot
        self.run_id = snapshot.run_id if snapshot is not None else run_id


class RunFailedError(RunObservationError):
    pass


class WaitTimeoutError(RunObservationError):
    pass


class ResultUnavailableError(RunObservationError):
    pass


class RunHistoryUnavailableError(RunObservationError):
    pass


_BY_STATUS: dict[int, type[SwarmsAPIError]] = {
    400: InvalidRequestError,
    401: UnauthenticatedError,
    402: PaymentRequiredError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    410: NotFoundError,
    422: InvalidRequestError,
    429: RateLimitedError,
    502: UpstreamUnavailableError,
    503: UpstreamUnavailableError,
    504: UpstreamUnavailableError,
}


_BY_CODE: dict[str, type[SwarmsAPIError]] = {
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


def local_problem(code: str, stage: str, *, status: int = 503, unknown: bool = False) -> Problem:
    return Problem(
        type="about:blank",
        title=code,
        status=status,
        detail=code,
        instance="urn:synth-ai:request",
        code=code,
        service="synth-ai",
        stage=stage,
        outcome="unknown" if unknown else "not_accepted",
    )


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
                status=response.status_code,
                unknown=response.request.method not in {"GET", "HEAD"},
            ),
            request_id=response.headers.get("x-request-id"),
            idempotency_key=response.request.headers.get("idempotency-key"),
        ) from None
    raise_problem(
        problem,
        request_id=response.headers.get("x-request-id"),
        idempotency_key=response.request.headers.get("idempotency-key"),
    )


def raise_problem(
    problem: Problem, *, request_id: str | None = None, idempotency_key: str | None = None
) -> None:
    kind = _BY_CODE.get(problem.code) or _BY_STATUS.get(
        problem.status, ServerError if problem.status >= 500 else InvalidRequestError
    )
    raise kind(problem, request_id=request_id, idempotency_key=idempotency_key)
