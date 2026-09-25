"""Stable Index failure codes carried by the SDK's typed HTTP exceptions.

Transport already raises typed ``SynthError`` subclasses (AuthorizationError,
PaymentRequiredError, RateLimitedError, ConflictError, TransientServiceError...).
These codes distinguish Index outcomes inside those classes; an unavailable
service is never reported as an empty result.

The backend owns the error vocabulary. Historical codes remain importable for
client compatibility; unknown future codes remain available as raw
``error.failure.code`` values.
"""

from enum import StrEnum
from types import MappingProxyType

from synth_ai.core.errors import SynthError


class IndexErrorCode(StrEnum):
    # Search, contents and account boundary.
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
    # Retained for clients of older Index deployments.
    OVERLOADED = "index_overloaded"
    REQUEST_CANCELLED = "index_request_cancelled"
    SCOPE_INVALID = "index_scope_invalid"
    USER_REQUIRED = "user_required"

    # Request framing, rejected before any route runs.
    REQUEST_TOO_LARGE = "index_request_too_large"
    REQUEST_TOO_FRAGMENTED = "index_request_too_fragmented"
    REQUEST_READ_TIMEOUT = "index_request_read_timeout"
    INVALID_CONTENT_LENGTH = "index_invalid_content_length"
    CONTENT_ENCODING_UNSUPPORTED = "index_content_encoding_unsupported"

    # Contribution lifecycle.
    CONTRIBUTION_FORBIDDEN = "contribution_forbidden"
    CONTRIBUTION_NOT_FOUND = "contribution_not_found"
    CONTRIBUTION_MISMATCH = "contribution_mismatch"
    CONTRIBUTION_IDENTITY_MISMATCH = "contribution_identity_mismatch"
    REVISION_NOT_FOUND = "revision_not_found"
    COLLECTION_NOT_FOUND = "collection_not_found"
    ASSET_TOO_LARGE = "asset_too_large"
    ASSET_UNAVAILABLE = "asset_unavailable"
    # Draft/revision creation replayed a key with a different request. The
    # search route uses IDEMPOTENCY_CONFLICT for the same situation.
    DRAFT_IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    MUTATION_RECEIPT_CONFLICT = "mutation_receipt_conflict"
    INVALID_ORIGIN = "invalid_origin"
    INVALID_TRANSITION = "invalid_transition"
    INVALID_MANIFEST_DIGEST = "invalid_manifest_digest"
    GENERATION_CONFLICT = "generation_conflict"
    TRANSACTION_REQUIRED = "transaction_required"

    # Upload, finalize and submit.
    ARTIFACT_BINDING_MISMATCH = "artifact_binding_mismatch"
    ARTIFACT_NOT_COMMITTED = "artifact_not_committed"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    CONTRIBUTION_PUBLICATION_MISMATCH = "contribution_publication_mismatch"
    CONTRIBUTION_STORAGE_UNAVAILABLE = "contribution_storage_unavailable"
    CONTRIBUTION_FINALIZATION_CONFLICT = "contribution_finalization_conflict"
    CONTRIBUTION_FINALIZATION_TIMEOUT = "contribution_finalization_timeout"
    CONTRIBUTION_SUBMISSION_TIMEOUT = "contribution_submission_timeout"
    CONTRIBUTION_DELIVERY_INVALID = "contribution_delivery_invalid"
    CONTRIBUTION_DELIVERY_UNAVAILABLE = "contribution_delivery_unavailable"
    SUBMISSION_CONFLICT = "submission_conflict"
    SUBMISSION_CONTENT_TOO_LARGE = "submission_content_too_large"
    SUBMISSION_DELIVERY_FAILED = "submission_delivery_failed"

    # Review, qualification and publication.
    REVIEW_FORBIDDEN = "review_forbidden"
    REVIEW_CONFLICT = "review_conflict"
    SELF_REVIEW = "self_review"
    ASSESSMENT_MISMATCH = "assessment_mismatch"
    INVALID_ASSESSMENT = "invalid_assessment"
    INVALID_COMMENTS = "invalid_comments"
    INVALID_CURSOR = "invalid_cursor"
    UNKNOWN_CLAIM = "unknown_claim"
    MANIFEST_MISMATCH = "manifest_mismatch"
    UNSEALED_REVISION = "unsealed_revision"
    UNQUALIFIED_REVISION = "unqualified_revision"
    AUDIENCE_MISMATCH = "audience_mismatch"
    RIGHTS_NOT_ATTESTED = "rights_not_attested"
    ASSET_LICENSE_NOT_PUBLIC = "asset_license_not_public"
    RELEASE_ISOLATION_REQUIRED = "release_isolation_required"

    # Sharing and profiles.
    COLLECTION_NOT_SHAREABLE = "collection_not_shareable"
    GRANT_NOT_FOUND = "grant_not_found"
    GRANTEE_NOT_FOUND = "grantee_not_found"
    GRANTEE_IS_OWNER = "grantee_is_owner"
    ORG_GRANTS_UNSUPPORTED = "org_grants_unsupported"
    PIN_NOT_ELIGIBLE = "pin_not_eligible"

    # Rewards and contests.
    AWARD_NOT_FOUND = "award_not_found"
    AWARD_NOT_ELIGIBLE = "award_not_eligible"
    AWARD_LEDGER_MISSING = "award_ledger_missing"
    INVALID_AWARD_EXPIRY = "invalid_award_expiry"
    SELF_AWARD = "self_award"
    CONTEST_NOT_FOUND = "contest_not_found"
    CONTEST_CLOSED = "contest_closed"
    CONTEST_CONFLICT = "contest_conflict"
    INVALID_CONTEST_WINDOW = "invalid_contest_window"
    ENTRY_NOT_FOUND = "entry_not_found"
    SELF_SCORE = "self_score"

    # Private research intake: allocation and its non-mutating lookup.
    RESEARCH_IMPORT_FORBIDDEN = "research_import_forbidden"
    CROSS_ORG_SOURCE_FORBIDDEN = "cross_org_source_forbidden"
    RESEARCH_SOURCE_PATH_FORBIDDEN = "research_source_path_forbidden"
    RESEARCH_DIGEST_REQUIRED = "research_digest_required"
    RESEARCH_LOOKUP_FORBIDDEN = "research_lookup_forbidden"
    # Nothing was allocated under this key for this account on this backend.
    RESEARCH_RECEIPT_ABSENT = "research_receipt_absent"
    # The key was used for a different research request.
    RESEARCH_RECEIPT_CONFLICT = "research_receipt_conflict"
    # An allocation under this key is still committing; retry the lookup.
    RESEARCH_RECEIPT_PENDING = "research_receipt_pending"
    RESEARCH_RECEIPT_UNOBSERVABLE = "research_receipt_unobservable"
    RESEARCH_REVISION_UNOBSERVABLE = "research_revision_unobservable"
    RESEARCH_REGISTRATION_CONFLICT = "research_registration_conflict"
    RESEARCH_REGISTRATION_MISSING = "research_registration_missing"
    RESEARCH_REGISTRATION_INVALID = "research_registration_invalid"
    RESEARCH_ACTOR_ORG_MISMATCH = "research_actor_org_mismatch"
    RESEARCH_ACTOR_UNBOUND = "research_actor_unbound"
    RESEARCH_ACTOR_USER_MISMATCH = "research_actor_user_mismatch"
    # The bundle's manifest is already registered to an allocation. Resume that
    # allocation through the lookup with its original idempotency key.
    RESEARCH_MANIFEST_ALREADY_REGISTERED = "research_manifest_already_registered"
    RESEARCH_ORG_CONTEXT_INVALID = "research_org_context_invalid"
    RESEARCH_ORG_CONTEXT_UNBOUND = "research_org_context_unbound"
    RESEARCH_PACKAGE_POLICY = "research_package_policy"
    RESEARCH_SOURCE_ASSET_MISSING = "research_source_asset_missing"
    RESEARCH_SOURCE_MISMATCH = "research_source_mismatch"

    # Private research intake: the server gate run at submission.
    RESEARCH_PACKAGE_INVALID = "research_package_invalid"
    RESEARCH_ASSET_MISMATCH = "research_asset_mismatch"
    RESEARCH_BUNDLE_ASSET_MISMATCH = "research_bundle_asset_mismatch"
    RESEARCH_BUNDLE_CONTENT_MISMATCH = "research_bundle_content_mismatch"
    RESEARCH_BUNDLE_MANIFEST_INVALID = "research_bundle_manifest_invalid"
    RESEARCH_BUNDLE_MANIFEST_MISSING = "research_bundle_manifest_missing"
    RESEARCH_BUNDLE_MISMATCH = "research_bundle_mismatch"
    RESEARCH_CORPUS_PREFLIGHT_FAILED = "research_corpus_preflight_failed"
    RESEARCH_DECISION_LOG_INVALID = "research_decision_log_invalid"
    RESEARCH_NUMBER_UNGROUNDED = "research_number_ungrounded"
    RESEARCH_POLICY_INVALID = "research_policy_invalid"
    RESEARCH_POLICY_UNAVAILABLE = "research_policy_unavailable"
    RESEARCH_PROVENANCE_MISSING = "research_provenance_missing"
    RESEARCH_REPORT_MISSING = "research_report_missing"
    RESEARCH_RUN_LEDGER_INVALID = "research_run_ledger_invalid"
    RESEARCH_RUN_PROVENANCE_MISMATCH = "research_run_provenance_mismatch"
    RESEARCH_SCAN_LIMIT = "research_scan_limit"
    RESEARCH_SECRET_DETECTED = "research_secret_detected"
    RESEARCH_RAW_TRANSCRIPT = "research_raw_transcript"


# Codes an Index route can return only inside an operator acceptance run, which
# needs a signed acceptance context no customer credential carries. They are not
# customer outcomes, so the SDK does not name them; the raw string still reaches
# callers through ``error.failure.code``.
OPERATOR_ONLY_ERROR_CODES = MappingProxyType(
    {
        "acceptance_use_replay_conflict": (
            "An acceptance-run capability use was replayed with different intent."
        ),
        "research_run_use_absent": (
            "An acceptance-run lookup found no recorded capability use for the run."
        ),
    }
)


def index_error_code(error: BaseException) -> IndexErrorCode | None:
    """Return the stable Index code of an SDK exception, or None if not an Index one."""
    if not isinstance(error, SynthError):
        return None
    code = getattr(getattr(error, "failure", None), "code", None)
    try:
        return IndexErrorCode(str(code))
    except ValueError:
        return None
