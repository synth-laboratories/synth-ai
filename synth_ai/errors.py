"""Safe typed errors. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

import asyncio

from synth_ai.types.identity import MutationIdentity


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
    ) -> None:
        self.code = code
        self.request_id = request_id
        self.status = status
        self.identity = identity
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class AuthenticationError(SynthError):
    """No valid selected principal."""


class PermissionDeniedError(SynthError):
    """Current grants refuse the operation."""


class NotFoundError(SynthError):
    """The authorized resource is unavailable."""


class ConflictError(SynthError):
    """Version or retained mutation identity conflicts."""


class RateLimitError(SynthError):
    """Retryable admission limit; no implicit paid retry."""


class BudgetExceededError(SynthError):
    """Customer charge ceiling or allowance refused."""


class RequestTimeoutError(SynthError):
    """Local observation/request deadline expired."""


class TransportError(SynthError):
    """HTTP exchange failed."""


class ContractError(SynthError):
    """Version, shape, identity or bound mismatch."""


class UnsupportedOperationError(SynthError):
    """The selected backend has not enabled a compatible operation surface."""


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
