"""Stable Index failure codes carried by the SDK's typed HTTP exceptions.

Transport already raises typed ``SynthError`` subclasses (AuthorizationError,
PaymentRequiredError, RateLimitedError, ConflictError, TransientServiceError...).
These codes distinguish Index outcomes inside those classes; an unavailable
service is never reported as an empty result.
"""

from enum import StrEnum

from synth_ai.core.errors import SynthError


class IndexErrorCode(StrEnum):
    SEARCH_MODE_UNSUPPORTED = "index_search_mode_unsupported"
    SEARCH_NOT_FOUND = "index_search_not_found"
    SEARCH_RESULT_NOT_READY = "index_search_result_not_ready"
    SEARCH_NOT_CANCELLABLE = "index_search_not_cancellable"
    FORBIDDEN = "index_forbidden"
    PAYMENT_REQUIRED = "index_payment_required"
    # A private search refused because the promotional balance is spent and the
    # organization has no paid authority. It arrives inside the same 402
    # PaymentRequiredError as PAYMENT_REQUIRED and is never a charge: recheck
    # ``account.promo_credit()`` for the reset instant before retrying.
    PRIVATE_CREDIT_EXHAUSTED = "index_private_credit_exhausted"
    DEEP_ALLOWANCE_EXHAUSTED = "index_deep_allowance_exhausted"
    RATE_LIMITED = "index_rate_limited"
    CONCURRENCY_LIMITED = "index_concurrency_limited"
    IDEMPOTENCY_CONFLICT = "index_idempotency_conflict"
    UNAVAILABLE = "index_unavailable"
    ENTITLEMENT_UNAVAILABLE = "index_entitlement_unavailable"
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
