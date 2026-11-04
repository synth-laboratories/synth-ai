"""Public Problem Details retain the server's refusal and outcome semantics."""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from ._generated.models import Problem


class SynthError(Exception):
    """Base SDK error."""


class SwarmsAPIError(SynthError):
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


class RunFailedError(SynthError):
    pass


class WaitTimeoutError(SynthError):
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
        ),
        ProviderPolicyError,
    ),
    **dict.fromkeys(
        ("resource_request_invalid", "resource_limits_required", "grant_not_widenable"),
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
    kind = _BY_CODE.get(problem.code) or _BY_STATUS.get(
        response.status_code, ServerError if response.status_code >= 500 else InvalidRequestError
    )
    raise kind(
        problem,
        request_id=response.headers.get("x-request-id"),
        idempotency_key=response.request.headers.get("idempotency-key"),
    )
