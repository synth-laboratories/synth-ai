"""One typed exception hierarchy for the public Swarms v1 error body.

Every non-2xx response carries ``{"error": {code, message, service, stage,
upstream_status, cause, ...}}``. :func:`raise_for_response` parses that body and
raises the subclass chosen by HTTP status; the stable ``code`` is kept verbatim.
"""

from __future__ import annotations

import httpx

from synth_ai.errors import SynthError

from .models import ErrorBody, ErrorDetail


class SwarmsAPIError(SynthError):
    """A typed refusal from the Synth API."""

    def __init__(self, status: int, error: ErrorDetail) -> None:
        text = error.message or error.code
        super().__init__(
            f"{error.code}: {text} (service={error.service}, stage={error.stage}, status={status})",
            code=error.code,
            status=status,
            stage=error.stage,
            cause=error.cause,
            service=error.service,
            upstream_status=error.upstream_status,
        )
        self.http_status = status
        self.error = error

    @property
    def message(self) -> str | None:
        return self.error.message


class InvalidRequestError(SwarmsAPIError):
    """400 or 422: the request does not match the contract."""


class UnauthenticatedError(SwarmsAPIError):
    """401: no valid credential."""


class PaymentRequiredError(SwarmsAPIError):
    """402: budget, balance or plan refused the request."""


class ForbiddenError(SwarmsAPIError):
    """403: credential lacks the scope or permission."""


class NotFoundError(SwarmsAPIError):
    """404."""


class ConflictError(SwarmsAPIError):
    """409: idempotency or state conflict."""


class RateLimitedError(SwarmsAPIError):
    """429."""


class UpstreamUnavailableError(SwarmsAPIError):
    """502, 503 or 504: a dependency refused or timed out."""


class ServerError(SwarmsAPIError):
    """Any other 5xx."""


class MalformedErrorResponseError(SwarmsAPIError):
    """A non-2xx response whose body is not the contract error shape."""


class TransportFailureError(SwarmsAPIError):
    """The request did not complete at the HTTP layer."""


class RunFailedError(SwarmsAPIError):
    """``swarms.run`` waited for a run that ended in state ``failed``."""


class WaitTimeoutError(SwarmsAPIError):
    """``swarms.wait`` reached its timeout before the run closed."""


_BY_STATUS: dict[int, type[SwarmsAPIError]] = {
    400: InvalidRequestError,
    401: UnauthenticatedError,
    402: PaymentRequiredError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: InvalidRequestError,
    429: RateLimitedError,
    502: UpstreamUnavailableError,
    503: UpstreamUnavailableError,
    504: UpstreamUnavailableError,
}


def error_class(status: int) -> type[SwarmsAPIError]:
    if status in _BY_STATUS:
        return _BY_STATUS[status]
    if status >= 500:
        return ServerError
    return InvalidRequestError


def client_error(code: str, stage: str, *, cause: str | None = None, status: int = 0,
                 kind: type[SwarmsAPIError] = SwarmsAPIError) -> SwarmsAPIError:
    """Build an SDK-side error in the same shape as a server refusal."""
    return kind(status, ErrorDetail(code=code, service="synth-ai", stage=stage, cause=cause))


def raise_for_response(response: httpx.Response, *, stage: str) -> None:
    if response.is_success:
        return
    try:
        body = ErrorBody.model_validate_json(response.content)
    except ValueError as error:
        raise MalformedErrorResponseError(
            response.status_code,
            ErrorDetail(
                code="malformed_error_response",
                service="synth-ai",
                stage=stage,
                upstream_status=response.status_code,
                cause=type(error).__name__,
            ),
        ) from None
    raise error_class(response.status_code)(response.status_code, body.error)
