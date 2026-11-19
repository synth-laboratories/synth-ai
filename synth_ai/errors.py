"""Public Problem Details retain the server's refusal and outcome semantics."""

from __future__ import annotations

import asyncio
import math
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ._stream import StreamCheckpoint

import httpx
from pydantic import ValidationError

from ._generated.models import File, Problem, ProblemOutcome, RunSnapshot
from ._limits import UPLOAD_MEDIA_TYPE_MAX_BYTES
from ._retry import retry_delay


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
    retry_after_seconds: int | None = None
    """The refusal's `Retry-After` in whole seconds, when the server sent one."""

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


class LaunchValidationError(InvalidRequestError, ValueError):
    """A local launch refusal; also preserves ValueError compatibility."""


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


class ContinueFromNotFoundError(NotFoundError):
    """The parent run is absent or inaccessible in the caller's organization."""


class ContinueFromNotTerminalError(ConflictError):
    """The selected parent has not reached a terminal state."""


class ContinueFromInputsUnavailableError(ConflictError):
    """The parent's required retained input references are unavailable."""


class LimitsRevisionConflictError(ConflictError):
    @property
    def current_revision(self) -> int | None:
        value = (self.problem.model_extra or {}).get("current_revision")
        return int(value) if isinstance(value, (int, str)) and str(value).isdigit() else None


class LimitAboveCeilingError(InvalidRequestError):
    pass


class LimitStillReachedError(ConflictError):
    pass


class RunExpiredError(ConflictError):
    pass


class SpendEntitlementExceededError(PaymentRequiredError):
    pass


class RateLimitedError(SwarmsAPIError):
    pass


class UpstreamUnavailableError(SwarmsAPIError):
    pass


class MessageTooLargeError(InvalidRequestError):
    """The UTF-8 message body exceeds the public message limit."""


class RunMessageRateLimitedError(RateLimitedError):
    """The run's send allowance is exhausted; the Problem owns retryability."""


class RunNotAcceptingMessagesError(ConflictError):
    """The run cannot currently accept a customer message."""


class RunMessageQueueFullError(ConflictError):
    """The bounded customer queue has no capacity."""


class MessagingUnavailableError(UpstreamUnavailableError):
    """The run has no admitted customer messaging interface."""


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


class UploadFormatUnsupportedError(InvalidRequestError):
    """422 upload_format_unsupported: the media type is not an accepted upload format.

    Raised before transfer for a media type outside Core's bounded ASCII
    type/subtype grammar. Binary media types are accepted.
    """

    maximum_media_type_bytes: ClassVar[int] = UPLOAD_MEDIA_TYPE_MAX_BYTES


class UploadSizeExceededError(InvalidRequestError):
    """422 upload_size_exceeded: the file is larger than Core accepts for one upload.

    Raised before transfer for a file over `maximum_size_bytes` (16 MiB), and
    mapped from Core's own 422. The Problem carries `maximum_size_bytes`.
    """


class OutputContentTooLargeError(InvalidRequestError):
    """413 output_content_too_large_for_mcp: the output is over the MCP inline bound.

    The Problem carries `byte_size` (a lower bound when `byte_size_lower_bound`),
    `max_bytes` and `download_operation`; read it with `download_output` instead.
    """


def output_content_too_large(
    byte_size: int, max_bytes: int, *, lower_bound: bool
) -> OutputContentTooLargeError:
    base = local_problem(
        "output_content_too_large_for_mcp",
        "getRunOutputContent",
        status=413,
        detail=f"output is {'over ' if lower_bound else ''}{byte_size} bytes; MCP returns at most {max_bytes}. Download it over HTTP (getRunOutputContent).",
    )
    return OutputContentTooLargeError(
        Problem.model_validate(
            {
                **base.model_dump(),
                "byte_size": byte_size,
                "byte_size_lower_bound": lower_bound,
                "max_bytes": max_bytes,
                "download_operation": "getRunOutputContent",
            }
        )
    )


class LaunchBudgetBelowActorFloorError(InvalidRequestError):
    """422 launch_budget_below_actor_floor: maximum_charge cannot fund every actor at once.

    The budget must cover the planner, the reviewer and each of
    `limits.parallel_workers` workers; the Problem detail states the minimum in
    micro-USD. Raise the budget or lower parallel_workers.
    """


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


class DirectoryUploadError(SynthError):
    """A partial directory upload retains committed files and retry identity."""

    kind = ErrorKind.UPLOAD
    code = "directory_upload_incomplete"

    def __init__(
        self,
        name: str,
        completed_files: tuple[File, ...],
        idempotency_key: str,
        file_idempotency_key: str,
    ) -> None:
        super().__init__(f"Directory upload is incomplete at {name}")
        self.name = name
        self.completed_files = completed_files
        self.idempotency_key = idempotency_key
        self.file_idempotency_key = file_idempotency_key


class DirectoryUploadCancelledError(asyncio.CancelledError):
    """Cancellation retains directory custody while remaining a CancelledError."""

    code = "directory_upload_cancelled"

    def __init__(
        self,
        name: str,
        completed_files: tuple[File, ...],
        idempotency_key: str,
        file_idempotency_key: str,
    ) -> None:
        super().__init__(f"Directory upload cancelled at {name}")
        self.name = name
        self.completed_files = completed_files
        self.idempotency_key = idempotency_key
        self.file_idempotency_key = file_idempotency_key


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


class MessageReplyTimeoutError(RunObservationError):
    """A local observation deadline; the server message remains unchanged."""

    code = "message_reply_timeout"

    def __init__(self, *, run_id: str, message_id: str) -> None:
        super().__init__("Message reply was not observed before the deadline", run_id=run_id)
        self.message_id = message_id


class MessageReplyUnavailableError(RunObservationError):
    """The feed reached finality without a correlated reply."""

    code = "message_reply_unavailable"

    def __init__(self, *, run_id: str, message_id: str) -> None:
        super().__init__("Run closed without an observed message reply", run_id=run_id)
        self.message_id = message_id


class RunFailedError(RunObservationError):
    code = "run_failed"


class RunPausedError(RunObservationError):
    code = "run_paused"


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
    "continue_from_not_found": ContinueFromNotFoundError,
    "continue_from_not_terminal": ContinueFromNotTerminalError,
    "continue_from_inputs_unavailable": ContinueFromInputsUnavailableError,
    "message_too_large": MessageTooLargeError,
    "run_message_rate_limited": RunMessageRateLimitedError,
    "run_not_accepting_messages": RunNotAcceptingMessagesError,
    "run_message_queue_full": RunMessageQueueFullError,
    "messaging_unavailable": MessagingUnavailableError,
    "limits_revision_conflict": LimitsRevisionConflictError,
    "limit_above_ceiling": LimitAboveCeilingError,
    "limit_still_reached": LimitStillReachedError,
    "run_expired": RunExpiredError,
    "spend_entitlement_exceeded": SpendEntitlementExceededError,
    "client_upgrade_required": UpgradeRequiredError,
    "upload_format_unsupported": UploadFormatUnsupportedError,
    "upload_size_exceeded": UploadSizeExceededError,
    "launch_budget_below_actor_floor": LaunchBudgetBelowActorFloorError,
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
        retry_after_seconds=retry_after_seconds(response.headers.get("retry-after")),
    )


def retry_after_seconds(value: str | None) -> int | None:
    """Preserve delta-seconds or an HTTP date without rounding the server delay down."""
    if value is None:
        return None
    delay = retry_delay(value, 0)
    return math.ceil(delay) if delay is not None else None


def raise_problem(
    problem: Problem,
    *,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    retry_after_seconds: int | None = None,
) -> None:
    kind = _BY_CODE.get(problem.code) or _BY_STATUS.get(
        problem.status, ServerError if problem.status >= 500 else InvalidRequestError
    )
    error = kind(problem, request_id=request_id, idempotency_key=idempotency_key)
    error.retry_after_seconds = retry_after_seconds
    raise error


def upload_format_unsupported(media_type: str | None) -> UploadFormatUnsupportedError:
    """The typed local refusal for a media type Core does not accept for uploads."""
    return UploadFormatUnsupportedError(
        local_problem(
            "upload_format_unsupported",
            "upload",
            status=422,
            detail=f"Media type must be ASCII type/subtype without whitespace, at most {UPLOAD_MEDIA_TYPE_MAX_BYTES} bytes.",
        )
    )


def upload_size_exceeded(maximum_bytes: int) -> UploadSizeExceededError:
    """The typed local refusal for a file over Core's per-upload byte limit."""
    base = local_problem(
        "upload_size_exceeded",
        "upload",
        status=422,
        detail=f"File exceeds the {maximum_bytes}-byte upload limit.",
    )
    return UploadSizeExceededError(
        Problem.model_validate({**base.model_dump(), "maximum_size_bytes": str(maximum_bytes)})
    )


def check_upload_format(media_type: str | None) -> str:
    """Return `media_type` when Core accepts it as an upload, else raise the typed refusal."""
    if media_type is None:
        raise upload_format_unsupported(media_type)
    slash = media_type.find("/")
    if (
        not media_type
        or len(media_type) > UPLOAD_MEDIA_TYPE_MAX_BYTES
        or slash <= 0
        or slash == len(media_type) - 1
        or any(ord(character) <= 32 or ord(character) >= 127 for character in media_type)
    ):
        raise upload_format_unsupported(media_type)
    return media_type
