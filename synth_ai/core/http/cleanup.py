"""Keep response-cleanup faults from erasing a mutation's recovery identity."""

from __future__ import annotations

import asyncio
import logging

from synth_ai.errors import MutationCancelledError, OutcomeUncertainError, TransportError
from synth_ai.types.identity import MutationIdentity

logger = logging.getLogger(__name__)


def handle_cleanup_failure(
    cause: BaseException,
    primary: BaseException | None,
    *,
    mutation: bool,
    identity: MutationIdentity | None,
) -> None:
    if primary is not None:
        primary.add_note("HTTP response cleanup also failed")
        logger.error("HTTP response cleanup failed: error_type=%s", type(cause).__name__)
        return
    if isinstance(cause, asyncio.CancelledError):
        if mutation:
            raise MutationCancelledError(identity) from cause
        raise cause
    if mutation:
        raise OutcomeUncertainError(
            "Mutation response cleanup failed; reconcile before repeating", identity=identity
        ) from cause
    raise TransportError("HTTP response cleanup failed") from cause
