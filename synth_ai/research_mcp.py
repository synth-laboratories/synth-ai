"""Typed MCP tool definitions and dispatcher for Track B Artifacts, Experiments, and Research Logs.

Kept separate from the generated `/v1` OpenAPI MCP catalog (`synth_ai.mcp.tools`) until the
backend OpenAPI contract publishes these routes (SYN-5973, SYN-5974).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

from pydantic import ValidationError

from ._validation import require_non_empty, safe_validation_detail, validate_idempotency_key
from .errors import InvalidRequestError, local_problem
from .research import ExactReference, ReferenceSource

if TYPE_CHECKING:
    from pydantic import BaseModel, JsonValue

    from .aclient import AsyncSynthClient

__all__ = ["RESEARCH_MCP_TOOLS", "ResearchToolSchema", "call_research_tool"]


class ResearchToolSchema(TypedDict):
    """JSON-Schema definition for one Track B MCP tool."""

    name: str
    description: str
    inputSchema: dict[str, JsonValue]


_EXACT_REF_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source", "kind", "record_id", "revision", "digest_sha256"],
    "properties": {
        "source": {"type": "string", "enum": ["research", "artifact", "run"]},
        "kind": {"type": "string"},
        "record_id": {"type": "string"},
        "revision": {"type": "string"},
        "digest_sha256": {"type": "string"},
    },
}

RESEARCH_MCP_TOOLS: tuple[ResearchToolSchema, ...] = (
    {
        "name": "artifacts_create",
        "description": "Create a hosted interactive artifact (v1) for a project or run.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id", "title", "kind", "content", "summary"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
                "title": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["html", "react", "svg", "mermaid", "markdown", "chart"],
                },
                "origin": {"type": "string", "enum": ["agent", "human"]},
                "content": {"type": "string"},
                "summary": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "artifacts_update",
        "description": "Append a new immutable revision (v2, v3, ...) to an artifact.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["artifact_id", "content", "summary"],
            "properties": {
                "artifact_id": {"type": "string"},
                "content": {"type": "string"},
                "summary": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "artifacts_list",
        "description": "List hosted interactive artifacts for a project or run.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
            },
        },
    },
    {
        "name": "artifacts_get",
        "description": "Fetch an interactive artifact and its revision history.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["artifact_id"],
            "properties": {"artifact_id": {"type": "string"}},
        },
    },
    {
        "name": "artifacts_share",
        "description": "Create or update an org or link share grant for an artifact.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["artifact_id"],
            "properties": {
                "artifact_id": {"type": "string"},
                "scope": {"type": "string", "enum": ["org", "link"]},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "artifacts_revoke_share",
        "description": "Revoke the active share grant for an interactive artifact.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["artifact_id"],
            "properties": {"artifact_id": {"type": "string"}},
        },
    },
    {
        "name": "experiments_create",
        "description": "Create a versioned multi-arm experiment specification.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id", "run_id", "title", "specification", "arms"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
                "title": {"type": "string"},
                "specification": {"type": "string"},
                "hypothesis": {"type": "string"},
                "arms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "pinned_inputs": {"type": "array", "items": {"type": "string"}},
                "stopping_criteria": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "experiments_revise",
        "description": "Create a new immutable revision of an experiment plan.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["experiment_id"],
            "properties": {
                "experiment_id": {"type": "string"},
                "title": {"type": "string"},
                "specification": {"type": "string"},
                "hypothesis": {"type": "string"},
                "arms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "stopping_criteria": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "experiments_get",
        "description": "Read an experiment and its associated trials and scores.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["experiment_id"],
            "properties": {"experiment_id": {"type": "string"}},
        },
    },
    {
        "name": "experiments_list",
        "description": "List experiments for a project, optionally filtered by run.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
            },
        },
    },
    {
        "name": "research_log_append",
        "description": "Append an entry (hypothesis, observation, failure, preference, ...) to the project research log.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id", "category", "text"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "hypothesis",
                        "attempt",
                        "observation",
                        "failure",
                        "decision",
                        "correction",
                        "preference",
                        "rationale",
                    ],
                },
                "text": {"type": "string"},
                "author_kind": {
                    "type": "string",
                    "enum": ["human", "local_agent", "swarm_actor"],
                },
                "author_label": {"type": "string"},
                "sources": {"type": "array", "items": _EXACT_REF_SCHEMA},
                "artifact_ids": {"type": "array", "items": {"type": "string"}},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "research_log_correct",
        "description": "Append an explicit correction entry pointing to an earlier research-log entry.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id", "entry_id", "text"],
            "properties": {
                "notebook_id": {"type": "string"},
                "entry_id": {"type": "string"},
                "text": {"type": "string"},
                "sources": {"type": "array", "items": _EXACT_REF_SCHEMA},
                "artifact_ids": {"type": "array", "items": {"type": "string"}},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "research_log_search",
        "description": "Search project research-log entries by query, category, run, or author.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id"],
            "properties": {
                "notebook_id": {"type": "string"},
                "query": {"type": "string"},
                "category": {"type": "string"},
                "run_id": {"type": "string"},
                "author_kind": {"type": "string"},
            },
        },
    },
    {
        "name": "research_log_export",
        "description": "Export a project or run research log as markdown or JSON.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["notebook_id"],
            "properties": {
                "notebook_id": {"type": "string"},
                "run_id": {"type": "string"},
                "format": {"type": "string", "enum": ["markdown", "json"]},
            },
        },
    },
)

_BY_NAME: dict[str, ResearchToolSchema] = {tool["name"]: tool for tool in RESEARCH_MCP_TOOLS}


def _invalid_input(detail: str) -> InvalidRequestError:
    return InvalidRequestError(
        local_problem("tool_arguments_invalid", "mcp_input", status=422, detail=detail)
    )


def _str(arguments: dict[str, JsonValue], key: str) -> str:
    value = arguments[key]
    if not isinstance(value, str):
        raise _invalid_input(f"{key} must be a string")
    return value


def _id_str(arguments: dict[str, JsonValue], key: str) -> str:
    try:
        return require_non_empty(key, _str(arguments, key))
    except ValueError as error:
        raise _invalid_input(str(error)) from None


def _opt_str(arguments: dict[str, JsonValue], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_input(f"{key} must be a string")
    return value


def _opt_id_str(arguments: dict[str, JsonValue], key: str) -> str | None:
    value = _opt_str(arguments, key)
    if value is None:
        return None
    try:
        return require_non_empty(key, value)
    except ValueError as error:
        raise _invalid_input(str(error)) from None


def _opt_key(arguments: dict[str, JsonValue]) -> str | None:
    try:
        return validate_idempotency_key(_opt_str(arguments, "idempotency_key"))
    except ValueError as error:
        raise _invalid_input(str(error)) from None


def _str_list(arguments: dict[str, JsonValue], key: str) -> list[str]:
    value = arguments.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _invalid_input(f"{key} must be a list of strings")
    return [item for item in value if isinstance(item, str)]


def _refs(arguments: dict[str, JsonValue], key: str) -> list[ExactReference]:
    value = arguments.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _invalid_input(f"{key} must be a list")
    refs: list[ExactReference] = []
    for item in value:
        if not isinstance(item, dict):
            raise _invalid_input(f"{key} entries must be objects")
        try:
            refs.append(
                ExactReference(
                    source=ReferenceSource(_str(item, "source")),
                    kind=_str(item, "kind"),
                    record_id=_id_str(item, "record_id"),
                    revision=_id_str(item, "revision"),
                    digest_sha256=_id_str(item, "digest_sha256"),
                )
            )
        except ValidationError as error:
            raise _invalid_input(safe_validation_detail(error)) from None
        except ValueError as error:
            raise _invalid_input(str(error)) from None
    return refs


async def call_research_tool(
    client: AsyncSynthClient, name: str, arguments: dict[str, JsonValue]
) -> str:
    """Validate and dispatch a Track B MCP tool call through the public SDK."""
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ValueError(f"Unknown research tool: {name}")
    from jsonschema import Draft202012Validator

    error = next(iter(Draft202012Validator(tool["inputSchema"]).iter_errors(arguments)), None)
    if error is not None:
        raise _invalid_input(error.message)
    result: BaseModel
    match name:
        case "artifacts_create":
            result = await client.artifacts.create_artifact(
                _id_str(arguments, "notebook_id"),
                _str(arguments, "title"),
                _str(arguments, "kind"),
                _str(arguments, "content"),
                _str(arguments, "summary"),
                origin=_opt_str(arguments, "origin") or "agent",
                run_id=_opt_id_str(arguments, "run_id"),
                idempotency_key=_opt_key(arguments),
            )
        case "artifacts_update":
            result = await client.artifacts.update_artifact(
                _id_str(arguments, "artifact_id"),
                _str(arguments, "content"),
                _str(arguments, "summary"),
                idempotency_key=_opt_key(arguments),
            )
        case "artifacts_list":
            result = await client.artifacts.list_artifacts(
                _id_str(arguments, "notebook_id"),
                run_id=_opt_id_str(arguments, "run_id"),
            )
        case "artifacts_get":
            result = await client.artifacts.get_artifact(_id_str(arguments, "artifact_id"))
        case "artifacts_share":
            result = await client.artifacts.share_artifact(
                _id_str(arguments, "artifact_id"),
                scope=_opt_str(arguments, "scope") or "org",
                idempotency_key=_opt_key(arguments),
            )
        case "artifacts_revoke_share":
            result = await client.artifacts.revoke_artifact_share(_id_str(arguments, "artifact_id"))
        case "experiments_create":
            result = await client.experiments.create_experiment(
                _id_str(arguments, "notebook_id"),
                _id_str(arguments, "run_id"),
                _str(arguments, "title"),
                _str(arguments, "specification"),
                _str_list(arguments, "arms"),
                hypothesis=_opt_str(arguments, "hypothesis"),
                pinned_inputs=_str_list(arguments, "pinned_inputs"),
                stopping_criteria=_opt_str(arguments, "stopping_criteria"),
                idempotency_key=_opt_key(arguments),
            )
        case "experiments_revise":
            arms = _str_list(arguments, "arms") if "arms" in arguments else None
            result = await client.experiments.revise_experiment(
                _id_str(arguments, "experiment_id"),
                title=_opt_str(arguments, "title"),
                specification=_opt_str(arguments, "specification"),
                hypothesis=_opt_str(arguments, "hypothesis"),
                arms=arms,
                stopping_criteria=_opt_str(arguments, "stopping_criteria"),
                idempotency_key=_opt_key(arguments),
            )
        case "experiments_get":
            result = await client.experiments.get_experiment(_id_str(arguments, "experiment_id"))
        case "experiments_list":
            result = await client.experiments.list_experiments(
                _id_str(arguments, "notebook_id"),
                run_id=_opt_id_str(arguments, "run_id"),
            )
        case "research_log_append":
            result = await client.research_logs.append_entry(
                _id_str(arguments, "notebook_id"),
                _str(arguments, "category"),
                _str(arguments, "text"),
                run_id=_opt_id_str(arguments, "run_id"),
                author_kind=_opt_str(arguments, "author_kind") or "local_agent",
                author_label=_opt_str(arguments, "author_label"),
                sources=_refs(arguments, "sources"),
                artifact_ids=_str_list(arguments, "artifact_ids"),
                idempotency_key=_opt_key(arguments),
            )
        case "research_log_correct":
            result = await client.research_logs.correct_entry(
                _id_str(arguments, "notebook_id"),
                _id_str(arguments, "entry_id"),
                _str(arguments, "text"),
                sources=_refs(arguments, "sources"),
                artifact_ids=_str_list(arguments, "artifact_ids"),
                idempotency_key=_opt_key(arguments),
            )
        case "research_log_search":
            result = await client.research_logs.search_entries(
                _id_str(arguments, "notebook_id"),
                query=_opt_str(arguments, "query"),
                category=_opt_str(arguments, "category"),
                run_id=_opt_id_str(arguments, "run_id"),
                author_kind=_opt_str(arguments, "author_kind"),
            )
        case "research_log_export":
            result = await client.research_logs.export_log(
                _id_str(arguments, "notebook_id"),
                run_id=_opt_id_str(arguments, "run_id"),
                fmt=_opt_str(arguments, "format") or "markdown",
            )
        case _:
            raise ValueError(f"Unhandled research tool: {name}")
    return json.dumps(result.model_dump(mode="json", exclude_unset=True))
