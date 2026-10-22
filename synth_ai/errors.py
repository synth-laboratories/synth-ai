"""Safe typed errors. See the public contract snapshots in contracts/"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import RateLimitSnapshot
from synth_ai.types.refusal import Refusal


class SynthError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        refusal: Refusal | None = None,
        request_id: str | None = None,
        operation: str | None = None,
        client_request_id: str | None = None,
        status: int | None = None,
        identity: MutationIdentity | None = None,
        retry_after_seconds: float | None = None,
        rate_limits: tuple[RateLimitSnapshot, ...] = (),
        retryable: bool | None = None,
        stage: str | None = None,
        cause: str | None = None,
        closure_pending: str | None = None,
        closure_expected_by: datetime | None = None,
        reason: Literal[
            "request_not_sent",
            "response_lost",
            "response_invalid",
            "infrastructure_status",
            "contract_header_missing",
        ]
        | None = None,
    ) -> None:
        self.refusal = refusal
        self.reason = reason
        self.retryable = retryable
        self.stage = stage
        self.cause = cause
        self.closure_pending = closure_pending
        self.closure_expected_by = closure_expected_by
        self.code = code
        self.request_id = request_id
        self.operation = operation
        self.client_request_id = client_request_id
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
        if self.refusal is not None:
            leaf = self.refusal
            request_ids = []
            while True:
                if leaf.request_id is not None:
                    request_ids.append(leaf.request_id)
                if leaf.upstream is None:
                    break
                leaf = leaf.upstream
            detail += f" [leaf={leaf.service}:{leaf.code} request_ids={','.join(dict.fromkeys(request_ids))}]"
        if detail:
            message = detail
        return f"{message} ({capacity})" if capacity else message

    def _detail(self) -> str:
        """Render safe identity/refusal facts; never include bodies or headers."""
        fields = (
            ("code", self.code),
            ("status", self.status),
            ("request_id", self.request_id),
            ("operation", self.operation),
            ("stage", self.stage),
            ("cause", self.cause),
            ("closure_pending", self.closure_pending),
            (
                "closure_expected_by",
                self.closure_expected_by.isoformat()
                if self.closure_expected_by is not None
                else None,
            ),
        )
        parts = [f"{name}={value}" for name, value in fields if value is not None]
        return f"{Exception.__str__(self)} [{' '.join(parts)}]" if parts else ""

    def __repr__(self) -> str:
        parts = [repr(Exception.__str__(self))]
        for name in ("code", "status", "request_id", "operation", "stage", "cause"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


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
    """The search's ZDR content was purged; only metadata remains (HTTP 410).

    ``search_id``, ``purges_at`` and ``purged_at`` come from the
    ``X-Synth-Search-Id`` / ``X-Synth-Purges-At`` / ``X-Synth-Purged-At`` response
    headers when the backend sends them (``None`` otherwise)."""

    search_id: str | None = None
    purges_at: str | None = None
    purged_at: str | None = None


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
