"""Stable Index failure codes carried by the SDK's typed HTTP exceptions.

Transport already raises typed ``SynthError`` subclasses (AuthorizationError,
PaymentRequiredError, RateLimitedError, ConflictError, TransientServiceError...).
These codes distinguish Index outcomes inside those classes; an unavailable
service is never reported as an empty result.
"""

from enum import StrEnum

from synth_ai.core.errors import SynthError


class IndexErrorCode(StrEnum):
    FORBIDDEN = "index_forbidden"
    PAYMENT_REQUIRED = "index_payment_required"
    RATE_LIMITED = "index_rate_limited"
    IDEMPOTENCY_CONFLICT = "index_idempotency_conflict"
    UNAVAILABLE = "index_unavailable"
    DEADLINE_EXCEEDED = "index_deadline_exceeded"
    REQUEST_TOO_LARGE = "index_request_too_large"
    CONTRIBUTION_FORBIDDEN = "contribution_forbidden"
    CONTRIBUTION_NOT_FOUND = "contribution_not_found"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"


def index_error_code(error: BaseException) -> IndexErrorCode | None:
    """Return the stable Index code of an SDK exception, or None if not an Index one."""
    if not isinstance(error, SynthError):
        return None
    code = getattr(getattr(error, "failure", None), "code", None)
    try:
        return IndexErrorCode(str(code))
    except ValueError:
        return None
