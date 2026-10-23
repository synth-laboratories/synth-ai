"""One operation lookup for recovery, transport and adapters."""

from __future__ import annotations

from synth_ai.types.identity import MutationIdentity

from . import core, forge, operations, swarms
from .operations import Operation

ALL_OPERATIONS = (
    *operations.ALL_OPERATIONS,
    *swarms.ALL_OPERATIONS,
    *core.ALL_OPERATIONS,
    *forge.ALL_OPERATIONS,
)
OPERATIONS_BY_ID = {operation.id: operation for operation in ALL_OPERATIONS}
if len(OPERATIONS_BY_ID) != len(ALL_OPERATIONS):
    raise RuntimeError("Operation identifiers must be unique")


def recovery_operation(identity: MutationIdentity) -> Operation:
    operation = OPERATIONS_BY_ID.get(identity.operation)
    if operation is None or not operation.mutation or not operation.keyed:
        raise ValueError("Identity must reference a supported keyed mutation")
    if not operation.matches_resource_path(identity.resource_path):
        raise ValueError("Identity resource path does not match its operation")
    return swarms.RESOLVE if operation.contract == "swarms.v1" else operations.RESOLVE


def mutation_scope(identity: MutationIdentity) -> str | None:
    recovery_operation(identity)
    return OPERATIONS_BY_ID[identity.operation].required_scope
