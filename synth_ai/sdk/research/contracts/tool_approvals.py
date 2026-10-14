"""Backend-authored Orchestra tool approvals; see openapi/research-v1.json."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import object_value, required_text
from synth_ai.sdk.research.contracts.common import SwarmId


class ToolApprovalDecision(StrEnum):
    APPROVE = "approve"
    REFUSE = "refuse"


def approval_path(swarm_id: SwarmId, approval_id: str | None = None) -> str:
    path = f"/smr/runs/{UUID(str(swarm_id))}/tool-approvals"
    return path if approval_id is None else path + f"/{UUID(approval_id)}"


def decision_body(
    decision: ToolApprovalDecision, expected_revision: int, reason: str | None
) -> JsonObject:
    if type(expected_revision) is not int or not 1 <= expected_revision <= 2**63 - 1:
        raise ValueError("tool approval revision must be a positive 64-bit integer")
    if reason is not None and (not isinstance(reason, str) or not 1 <= len(reason) <= 4096):
        raise ValueError("tool approval reason must contain 1..4096 characters")
    return {
        "decision": ToolApprovalDecision(decision).value,
        "expected_revision": expected_revision,
        "reason": reason,
    }


@dataclass(frozen=True, slots=True)
class PendingToolApproval:
    approval_id: str
    revision: int
    state: str
    attempt_id: str
    actor_id: str
    actor_class: str
    call_id: str
    tool: str
    capability: str
    argument_digest: str
    parked_at_ms: int

    @classmethod
    def from_wire(cls, value: JsonValue) -> PendingToolApproval:
        data = object_value(value, operation_id="pending tool approval")
        revision, parked = data.get("revision"), data.get("parked_at_ms")
        if (
            type(revision) is not int
            or not 1 <= revision <= 2**63 - 1
            or type(parked) is not int
            or parked < 0
        ):
            raise ValueError("invalid approval revision or timestamp")
        state = required_text(data, "state")
        if state not in {
            "awaiting",
            "approved",
            "refused",
            "permit_delivered",
            "permit_refused",
            "permit_in_doubt",
        }:
            raise ValueError("unknown tool approval state")
        return cls(
            required_text(data, "approval_id"),
            revision,
            state,
            required_text(data, "attempt_id"),
            required_text(data, "actor_id"),
            required_text(data, "actor_class"),
            required_text(data, "call_id"),
            required_text(data, "tool"),
            required_text(data, "capability"),
            required_text(data, "argument_digest"),
            parked,
        )


@dataclass(frozen=True, slots=True)
class ToolApprovals:
    swarm_id: SwarmId
    approvals: tuple[PendingToolApproval, ...]

    @classmethod
    def from_wire(cls, value: JsonValue, swarm_id: SwarmId) -> ToolApprovals:
        data = object_value(value, operation_id="list_run_tool_approvals")
        scope = object_value(data.get("scope"), operation_id="tool approvals scope")
        if data.get("schema_version") != "synth.orchestra-tool-approvals.v1" or scope.get(
            "stream_id"
        ) != str(UUID(str(swarm_id))):
            raise ValueError("tool approval scope or schema mismatch")
        values = data.get("approvals")
        if not isinstance(values, list):
            raise ValueError("tool approvals must be an array")
        rows = tuple(PendingToolApproval.from_wire(item) for item in values)
        if len({row.approval_id for row in rows}) != len(rows):
            raise ValueError("duplicate tool approval")
        return cls(swarm_id, rows)


@dataclass(frozen=True, slots=True)
class ToolApprovalVerdict:
    verdict: str
    detail: JsonObject

    @classmethod
    def from_wire(cls, value: JsonValue) -> ToolApprovalVerdict:
        data = object_value(value, operation_id="decide_run_tool_approval")
        verdict = required_text(data, "verdict")
        if verdict not in {"enacted", "refused"}:
            raise ValueError("unknown tool approval verdict")
        return cls(verdict, dict(data))
