"""Safe typed errors. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
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
    ) -> None:
        self.retryable = retryable
        self.code = code
        self.request_id = request_id
        self.status = status
        self.identity = identity
        self.retry_after_seconds = retry_after_seconds
        self.rate_limits = rate_limits
        super().__init__(message)

    def __str__(self) -> str:
        capacity = "; ".join(
            f"{item.dimension}: {item.remaining}/{item.limit} remaining"
            for item in self.rate_limits
            if item.remaining is not None and item.limit is not None
        )
        message = super().__str__()
        return f"{message} ({capacity})" if capacity else message


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


POLICY_ERROR_TYPES: Mapping[tuple[int, str], type[SynthError]] = MappingProxyType(
    {
        (403, "entitlement_denied"): EntitlementDeniedError,
        (403, "activation_required"): ActivationRequiredError,
        (403, "own_credential_required"): OwnCredentialRequiredError,
        (403, "byok_forbidden"): ByokForbiddenError,
        (403, "plan_required"): PlanRequiredError,
        (503, "policy_unavailable"): PolicyUnavailableError,
        (503, "capability_unavailable"): CapabilityUnavailableError,
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
