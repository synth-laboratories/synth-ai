"""Receipts for backend-owned native Swarm answer and steer operations.

# See: openapi/research-v1.json SmrOrchestraOperationReceiptResponse
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from synth_ai.core.contracts.json_value import JsonValue
from synth_ai.sdk.research.contracts._wire import object_value, optional_text, required_text
from synth_ai.sdk.research.contracts.common import SwarmId

ControlKind = Literal["action_answer", "steer"]
ControlStatus = Literal["accepted", "duplicate"]


@dataclass(frozen=True, slots=True)
class SwarmControlReceipt:
    """Durable acceptance acknowledgement; execution delivery remains asynchronous."""

    swarm_id: SwarmId
    operation_id: str
    kind: ControlKind
    status: ControlStatus
    action_id: str | None = None
    interaction_id: str | None = None
    control_seq: int | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmControlReceipt:
        payload = object_value(value, operation_id="swarm control receipt")
        required = {"run_id", "operation_id", "kind", "status"}
        allowed = required | {"action_id", "interaction_id", "control_seq"}
        if required - payload.keys() or payload.keys() - allowed:
            raise ValueError("swarm control receipt fields drifted")
        kind, status = required_text(payload, "kind"), required_text(payload, "status")
        if kind not in ("action_answer", "steer") or status not in ("accepted", "duplicate"):
            raise ValueError("swarm control receipt vocabulary drifted")
        sequence = payload.get("control_seq")
        if sequence is not None and (type(sequence) is not int or sequence < 0):
            raise ValueError("swarm control sequence must be a non-negative integer")
        return cls(
            SwarmId(required_text(payload, "run_id")),
            required_text(payload, "operation_id"),
            cast(ControlKind, kind),
            cast(ControlStatus, status),
            optional_text(payload, "action_id"),
            optional_text(payload, "interaction_id"),
            cast(int | None, sequence),
        )


__all__ = ["SwarmControlReceipt"]
