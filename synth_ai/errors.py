"""Safe typed errors. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType

from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import RateLimitSnapshot


class SynthError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
        status: int | None = None,
        identity: MutationIdentity | None = None,
        retry_after_seconds: float | None = None,
        rate_limits: tuple[RateLimitSnapshot, ...] = (),
        retryable: bool | None = None,
        stage: str | None = None,
        cause: str | None = None,
        closure_pending: str | None = None,
        closure_expected_by: datetime | None = None,
    ) -> None:
        self.retryable = retryable
        self.stage = stage
        self.cause = cause
        self.closure_pending = closure_pending
        self.closure_expected_by = closure_expected_by
        self.code = code
        self.request_id = request_id
        self.status = status
        self.identity = identity
        self.retry_after_seconds = retry_after_seconds
        self.rate_limits = rate_limits
        super().__init__(message)

    def __str__(self) -> str:
        message = super().__str__()
        capacity = "; ".join(
            f"{item.dimension}: {item.remaining}/{item.limit} remaining"
            for item in self.rate_limits
            if item.remaining is not None and item.limit is not None
        )
        detail = self._detail()
        if detail:
            message = detail
        return f"{message} ({capacity})" if capacity else message

    def _detail(self) -> str:
        """``<Class>: <code> (status n) stage=.. cause=.. request_id=..``; absent parts omitted.

        Only allowlisted, already-validated fields are rendered: never headers or
        bodies beyond these."""
        if not any((self.code, self.status, self.stage, self.cause, self.request_id)):
            return ""
        message = Exception.__str__(self)
        head = self.code or message
        parts = [f"{type(self).__name__}: {head}"]
        if self.status is not None:
            parts.append(f"(status {self.status})")
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.cause:
            parts.append(f"cause={self.cause}")
        if self.closure_pending:
            expected = self.closure_expected_by
            parts.append(f"closure_pending={self.closure_pending}")
            if expected is not None:
                parts.append(f"closure_expected_by={expected.isoformat()}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        text = " ".join(parts)
        if self.code and message and message != "Operation refused" and message != self.code:
            text = f"{text} - {message}"
        return text


class AuthenticationError(SynthError):
    """No valid selected principal."""


class PermissionDeniedError(SynthError):
    """Current grants refuse the operation."""


class EntitlementDeniedError(PermissionDeniedError):
    """Organization policy refuses the requested sub-product."""


class ActivationRequiredError(EntitlementDeniedError):
    """The requested sub-product requires activation."""


class OwnCredentialRequiredError(EntitlementDeniedError):
    """The organization must supply its own provider credential."""


class ByokForbiddenError(EntitlementDeniedError):
    """The selected sub-product does not permit a customer provider credential."""


class PlanRequiredError(EntitlementDeniedError):
    """The requested sub-product requires an eligible plan."""


class PolicyUnavailableError(SynthError):
    """Authoritative account policy could not be read; no access is inferred."""


class NotFoundError(SynthError):
    """The authorized resource is unavailable."""


class ConflictError(SynthError):
    """Version or retained mutation identity conflicts."""


class RateLimitError(SynthError):
    """Retryable admission limit; no implicit paid retry."""


class BudgetExceededError(SynthError):
    """Customer charge ceiling or allowance refused."""


class PaymentRequiredError(BudgetExceededError):
    """HTTP 402: payment or credit is required; nothing was created or charged."""


class RefusedError(SynthError):
    """Definite typed refusal (4xx with a code): the operation was not accepted.

    ``retryable`` is False; repeating the identical request without changing the
    refused condition will fail the same way."""


class RequestTimeoutError(SynthError):
    """Local observation/request deadline expired."""


class TransportError(SynthError):
    """HTTP exchange failed."""


class ContractError(SynthError):
    """Version, shape, identity or bound mismatch."""


class UnsupportedOperationError(SynthError):
    """The selected backend has not enabled a compatible operation surface."""


class CapabilityUnavailableError(UnsupportedOperationError):
    """The requested sub-product has no qualified native capability."""


class SearchHistoryDeletedError(SynthError):
    """The exact customer archive was deleted; it cannot be delivered again."""


class SearchHistoryExpiredError(SynthError):
    """The original archive lifetime ended; a read never renews it."""


class SearchHistoryUnavailableError(SynthError):
    """The selected archive has no authorized deliverable content."""


class RetentionModeNotAllowed(PermissionDeniedError):  # noqa: N818
    """The organization's enforced retention policy refuses the requested mode.

    ``allowed_modes`` lists the modes the organization permits (empty when the
    backend did not send them). ``retryable`` is False: change the mode."""

    allowed_modes: tuple[str, ...] = ()


class SearchContentPurged(SynthError):  # noqa: N818
    """The search's ZDR content was purged; only metadata remains (HTTP 410)."""


class StrictZdrResultUnavailable(ConflictError):  # noqa: N818
    """A ``zdr_strict`` search never stores its result.

    Use ``run.inline_result`` (FAST create response) or ``run.events()`` (DEEP)."""


POLICY_ERROR_TYPES: Mapping[tuple[int, str], type[SynthError]] = MappingProxyType(
    {
        (403, "entitlement_denied"): EntitlementDeniedError,
        (403, "activation_required"): ActivationRequiredError,
        (403, "own_credential_required"): OwnCredentialRequiredError,
        (403, "byok_forbidden"): ByokForbiddenError,
        (403, "plan_required"): PlanRequiredError,
        (503, "policy_unavailable"): PolicyUnavailableError,
        (503, "capability_unavailable"): CapabilityUnavailableError,
        (403, "retention_mode_not_allowed"): RetentionModeNotAllowed,
        (410, "search_content_purged"): SearchContentPurged,
        (409, "result_not_retained_strict_zdr"): StrictZdrResultUnavailable,
        (410, "search_history_deleted"): SearchHistoryDeletedError,
        (410, "search_history_expired"): SearchHistoryExpiredError,
        (409, "search_history_unavailable"): SearchHistoryUnavailableError,
        (410, "search_history_unavailable"): SearchHistoryUnavailableError,
    }
)


class ResultNotReadyError(SynthError):
    """The canonical result is not yet available."""


class OutcomeUncertainError(SynthError):
    """Retain identity and reconcile before repeating work."""


class OperationFailedError(SynthError):
    """Explicit opt-in failure raising for an inspectable result."""


class MutationCancelledError(asyncio.CancelledError):
    def __init__(self, identity: MutationIdentity | None) -> None:
        self.identity = identity
        super().__init__("Local mutation observation cancelled; remote outcome is unknown")


class OutputIntegrityError(ContractError):
    """Output bytes or identity failed verification; never publish the file."""
