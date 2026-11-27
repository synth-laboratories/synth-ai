"""MCP elicitation (`elicitation/create`) for interactive run approval prompts (SYN-6057).

When a connected MCP client advertises the `elicitation` capability, pending
`RunApproval` requests prompt the human directly in-client and submit the
decision through Core's idempotent approval endpoint. When elicitation is not
supported, callers gracefully degrade to explicit approval tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mcp import McpError
from mcp.types import ClientCapabilities, ElicitationCapability

from synth_ai._generated.models import (
    RunApprovalDecisionRequest,
    RunApprovalDecisionRequestDecision,
)
from synth_ai.errors import SwarmsAPIError

if TYPE_CHECKING:
    from mcp.server.session import ServerSession
    from mcp.types import RequestId

    from synth_ai.aclient import AsyncSynthClient

__all__ = [
    "APPROVAL_ELICITATION_SCHEMA",
    "ElicitationOutcome",
    "client_supports_elicitation",
    "elicit_approval_decision",
]

APPROVAL_ELICITATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "title": "Decision",
            "description": "Approve or reject this swarm action.",
            "enum": ["approve", "reject"],
        },
        "reason": {
            "type": "string",
            "title": "Reason",
            "description": "Optional rationale recorded on the approval receipt.",
            "maxLength": 1024,
        },
    },
    "required": ["decision"],
}


@dataclass(frozen=True, slots=True)
class ElicitationOutcome:
    """Result of attempting an interactive MCP approval prompt."""

    status: Literal["decided", "deferred", "unsupported", "failed"]
    approval_id: str
    decision: str | None = None
    detail: str | None = None


def client_supports_elicitation(session: ServerSession) -> bool:
    """Return True when the connected MCP client advertised elicitation support."""
    return session.check_client_capability(ClientCapabilities(elicitation=ElicitationCapability()))


async def elicit_approval_decision(
    client: AsyncSynthClient,
    session: ServerSession,
    run_id: str,
    *,
    approval_id: str,
    action_kind: str,
    summary: str,
    revision: int,
    related_request_id: RequestId | None = None,
) -> ElicitationOutcome:
    """Prompt the user via `elicitation/create` and submit the approval decision."""
    if not client_supports_elicitation(session):
        return ElicitationOutcome(
            status="unsupported",
            approval_id=approval_id,
            detail="Client does not advertise elicitation capability.",
        )
    prompt = f"Swarm run {run_id} requires approval ({approval_id}, {action_kind}): {summary}"
    try:
        response = await session.elicit_form(
            message=prompt,
            requestedSchema=APPROVAL_ELICITATION_SCHEMA,
            related_request_id=related_request_id,
        )
    except McpError as error:
        return ElicitationOutcome(
            status="unsupported",
            approval_id=approval_id,
            detail=str(error),
        )
    if response.action != "accept" or not isinstance(response.content, dict):
        return ElicitationOutcome(
            status="deferred",
            approval_id=approval_id,
            detail=f"User {response.action}ed approval prompt; decide later via swarms_run_approval_decide.",
        )
    raw_decision = response.content.get("decision")
    if raw_decision not in {"approve", "reject"}:
        return ElicitationOutcome(
            status="failed",
            approval_id=approval_id,
            detail="Invalid decision returned from elicitation form.",
        )
    raw_reason = response.content.get("reason")
    reason = raw_reason if isinstance(raw_reason, str) and raw_reason.strip() else None
    request = RunApprovalDecisionRequest(
        decision=RunApprovalDecisionRequestDecision(raw_decision),
        reason=reason,
        expected_revision=revision if revision >= 1 else None,
    )
    try:
        await client.swarms.decide_approval(
            run_id,
            approval_id,
            request,
            idempotency_key=f"mcp-elicit-{approval_id}-{revision}",
        )
    except SwarmsAPIError as error:
        return ElicitationOutcome(
            status="failed",
            approval_id=approval_id,
            decision=str(raw_decision),
            detail=error.problem.detail or error.problem.code,
        )
    return ElicitationOutcome(
        status="decided",
        approval_id=approval_id,
        decision=str(raw_decision),
    )
