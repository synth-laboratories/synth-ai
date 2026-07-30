"""Stable MCP adapters for first-class Research Visuals."""

from __future__ import annotations

import base64
from collections.abc import Callable

from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    JSONDict,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import (
    optional_bool,
    optional_int,
    optional_string,
    require_string,
)
from synth_ai.sdk.research.client import Client as ResearchClient
from synth_ai.sdk.research.contracts.common import (
    ArtifactId,
    EffortId,
    FactoryId,
    ProjectId,
)
from synth_ai.sdk.research.contracts.visuals import Visual, VisualVisibility

CoreClientFactory = Callable[[JSONDict], ResearchClient]
_MCP_BINARY_CHUNK_BYTES_DEFAULT = 65_536
_MCP_BINARY_CHUNK_BYTES_MAX = 131_072


def _optional_visibility(args: JSONDict) -> VisualVisibility | None:
    value = optional_string(args, "visibility")
    return VisualVisibility(value) if value is not None else None


def _content_payload(
    visual: Visual,
    content: bytes,
    *,
    content_type: str,
    offset: int,
    max_bytes: int,
    include_content_digest: bool,
) -> JSONDict:
    if offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if max_bytes < 1 or max_bytes > _MCP_BINARY_CHUNK_BYTES_MAX:
        raise ValueError(f"max_bytes must be between 1 and {_MCP_BINARY_CHUNK_BYTES_MAX}")
    chunk = content[offset : offset + max_bytes]
    next_offset = offset + len(chunk)
    payload: JSONDict = {
        "schema_version": "visual-content-chunk-v1",
        "hosted_artifact_id": str(visual.hosted_artifact_id),
        "root_artifact_id": str(visual.root_artifact_id),
        "project_id": str(visual.project_id),
        "artifact_version": visual.artifact_version,
        "content_type": content_type,
        "encoding": "base64",
        "content_base64": base64.b64encode(chunk).decode("ascii"),
        "size_bytes": len(content),
        "offset": offset,
        "bytes_returned": len(chunk),
        "eof": next_offset >= len(content),
        "next_offset": next_offset if next_offset < len(content) else None,
    }
    if include_content_digest:
        payload["content_digest"] = visual.content_digest
    return payload


def _chunk_request(args: JSONDict) -> tuple[int, int]:
    offset = optional_int(args, "offset")
    max_bytes = optional_int(args, "max_bytes")
    return (
        offset if offset is not None else 0,
        max_bytes if max_bytes is not None else _MCP_BINARY_CHUNK_BYTES_DEFAULT,
    )


def build_visual_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build the stable authenticated Visual read adapters."""

    def list_visuals(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        factory_id = optional_string(args, "factory_id")
        effort_id = optional_string(args, "effort_id")
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return client.visuals.list(
                project_id,
                visual_kind=optional_string(args, "visual_kind"),
                visibility=_optional_visibility(args),
                factory_id=FactoryId(factory_id) if factory_id is not None else None,
                effort_id=EffortId(effort_id) if effort_id is not None else None,
                include_deleted=optional_bool(args, "include_deleted"),
                cursor=optional_string(args, "cursor"),
                limit=limit if limit is not None else 100,
            ).to_wire()

    def get_visual(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        with client_from_args(args) as client:
            return client.visuals.retrieve(visual_id).to_wire()

    def get_visual_content(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        offset, max_bytes = _chunk_request(args)
        with client_from_args(args) as client:
            visual = client.visuals.retrieve(visual_id)
            content = client.visuals.retrieve_content(visual_id)
        return _content_payload(
            visual,
            content,
            content_type="text/html",
            offset=offset,
            max_bytes=max_bytes,
            include_content_digest=True,
        )

    def get_visual_preview(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        offset, max_bytes = _chunk_request(args)
        with client_from_args(args) as client:
            visual = client.visuals.retrieve(visual_id)
            content = client.visuals.retrieve_preview(visual_id)
        return _content_payload(
            visual,
            content,
            content_type="image/png",
            offset=offset,
            max_bytes=max_bytes,
            include_content_digest=False,
        )

    visual_id_schema = {
        "type": "string",
        "description": "Stable hosted artifact identifier for one Visual version.",
    }
    chunk_schema = {
        "offset": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
        },
        "max_bytes": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MCP_BINARY_CHUNK_BYTES_MAX,
            "default": _MCP_BINARY_CHUNK_BYTES_DEFAULT,
        },
    }
    return [
        ToolDefinition(
            name="research_list_visuals",
            description="List one cursor-paginated page of project Visuals.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "visual_kind": {"type": "string"},
                    "visibility": {
                        "type": "string",
                        "enum": [visibility.value for visibility in VisualVisibility],
                    },
                    "factory_id": {"type": "string"},
                    "effort_id": {"type": "string"},
                    "include_deleted": {"type": "boolean"},
                    "cursor": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 250},
                },
                required=["project_id"],
            ),
            handler=list_visuals,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_get_visual",
            description="Retrieve authenticated metadata for one Visual version.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema},
                required=["visual_id"],
            ),
            handler=get_visual,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_get_visual_content",
            description="Read one bounded base64 chunk of authenticated Visual HTML.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema, **chunk_schema},
                required=["visual_id"],
            ),
            handler=get_visual_content,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_get_visual_preview",
            description="Read one bounded base64 chunk of an authenticated Visual PNG preview.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema, **chunk_schema},
                required=["visual_id"],
            ),
            handler=get_visual_preview,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_visual_tools"]
