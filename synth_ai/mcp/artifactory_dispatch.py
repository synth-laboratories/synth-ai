"""MCP dispatch for notebook discovery and Human Artifactory records (SYN-6067)."""

from collections.abc import Mapping

from pydantic import BaseModel, JsonValue

from synth_ai._generated.models import (
    CreateArtifactoryRecordRequest,
    ReviseArtifactoryRecordRequest,
)
from synth_ai.aclient import AsyncSynthClient
from synth_ai.artifactory import ArtifactoryOrigin, ArtifactoryPayload, payload_category
from synth_ai.errors import InvalidRequestError, ResponseContractError, local_problem

from ._arguments import optional, page_limit, required

__all__ = ["dispatch_artifactory"]


def _payload(
    body: CreateArtifactoryRecordRequest | ReviseArtifactoryRecordRequest,
) -> ArtifactoryPayload | None:
    present = [p for p in (body.preference, body.hypothesis, body.finding) if p is not None]
    if len(present) > 1:
        raise InvalidRequestError(
            local_problem(
                "tool_arguments_invalid",
                "mcp_input",
                status=422,
                detail="body carries more than one of preference, hypothesis, finding",
            )
        )
    return present[0] if present else None


def _create_body(arguments: Mapping[str, JsonValue]) -> CreateArtifactoryRecordRequest:
    raw = arguments["body"]
    if isinstance(raw, dict) and "origin" not in raw:
        # MCP OAuth grants are refused origin=human; the catalog defaults to agent.
        raw = {**raw, "origin": ArtifactoryOrigin.AGENT.value}
    return required({"body": raw}, "body", CreateArtifactoryRecordRequest)


async def dispatch_artifactory(
    client: AsyncSynthClient, operation: str, arguments: Mapping[str, JsonValue]
) -> BaseModel | None:
    match operation:
        case "listNotebooks":
            return await client.notebooks.list_notebooks(
                limit=page_limit(arguments), cursor=optional(arguments, "cursor", str)
            )
        case "createArtifactoryRecord":
            body = _create_body(arguments)
            payload = _payload(body)
            if payload is None:
                raise InvalidRequestError(
                    local_problem(
                        "tool_arguments_invalid",
                        "mcp_input",
                        status=422,
                        detail="body needs the preference, hypothesis or finding for its category",
                    )
                )
            if payload_category(payload).value != body.category.value:
                raise InvalidRequestError(
                    local_problem(
                        "tool_arguments_invalid",
                        "mcp_input",
                        status=422,
                        detail="body category does not match its payload",
                    )
                )
            return await client.artifactory.create_record(
                required(arguments, "notebook_id", str),
                body.title,
                payload,
                origin=body.origin.value,
                run_id=body.run_id,
                author_kind=body.author_kind.value if body.author_kind is not None else None,
                pinned=body.pinned,
                summary=body.summary,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listArtifactoryRecords":
            return await client.artifactory.list_records(
                required(arguments, "notebook_id", str),
                category=optional(arguments, "category", str),
                origin=optional(arguments, "origin", str),
                run_id=optional(arguments, "run_id", str),
                pinned_only=optional(arguments, "pinned", str) == "true",
            )
        case "resolveEffectivePreference":
            return await client.artifactory.resolve_preference(
                required(arguments, "notebook_id", str),
                run_id=optional(arguments, "run_id", str),
            )
        case "getArtifactoryRecord":
            return await client.artifactory.get_record(required(arguments, "record_id", str))
        case "reviseArtifactoryRecord":
            revision = required(arguments, "body", ReviseArtifactoryRecordRequest)
            return await client.artifactory.revise_record(
                required(arguments, "record_id", str),
                payload=_payload(revision),
                title=revision.title,
                summary=revision.summary,
                pinned=revision.pinned,
                author_kind=revision.author_kind.value
                if revision.author_kind is not None
                else None,
                expected_revision=revision.expected_revision,
                expected_supersedes_digest=revision.expected_supersedes_digest,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case _:
            raise ResponseContractError(
                local_problem("mcp_operation_unbound", "mcp_dispatch", detail=operation)
            )
