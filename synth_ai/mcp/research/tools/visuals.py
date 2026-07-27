"""Stable MCP adapters for first-class Research Visuals.

# See: Jstack/.jstack/daily_notes/2026-07-27/SPEC_visuals_resource.md
"""

from __future__ import annotations

import base64
from collections.abc import Callable

from synth_ai.core.research.client import Client as ResearchClient
from synth_ai.core.research.contracts.common import (
    ArtifactId,
    EffortId,
    FactoryId,
    ProjectId,
)
from synth_ai.core.research.contracts.visuals import Visual, VisualVisibility
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

CoreClientFactory = Callable[[JSONDict], ResearchClient]


def _optional_visibility(args: JSONDict) -> VisualVisibility | None:
    value = optional_string(args, "visibility")
    return VisualVisibility(value) if value is not None else None


def _content_payload(
    visual: Visual,
    content: bytes,
    *,
    content_type: str,
) -> JSONDict:
    return {
        "hosted_artifact_id": str(visual.hosted_artifact_id),
        "root_artifact_id": str(visual.root_artifact_id),
        "project_id": str(visual.project_id),
        "artifact_version": visual.artifact_version,
        "content_type": content_type,
        "encoding": "base64",
        "content_base64": base64.b64encode(content).decode("ascii"),
        "size_bytes": len(content),
        "content_digest": visual.content_digest,
    }


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
        with client_from_args(args) as client:
            visual = client.visuals.retrieve(visual_id)
            content = client.visuals.retrieve_content(visual_id)
        return _content_payload(visual, content, content_type="text/html")

    def get_visual_preview(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        with client_from_args(args) as client:
            visual = client.visuals.retrieve(visual_id)
            content = client.visuals.retrieve_preview(visual_id)
        return _content_payload(visual, content, content_type="image/png")

    visual_id_schema = {
        "type": "string",
        "description": "Stable hosted artifact identifier for one Visual version.",
    }
    return [
        ToolDefinition(
            name="smr_list_visuals",
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
            name="smr_get_visual",
            description="Retrieve authenticated metadata for one Visual version.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema},
                required=["visual_id"],
            ),
            handler=get_visual,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_visual_content",
            description="Retrieve authenticated Visual HTML as an explicit base64 payload.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema},
                required=["visual_id"],
            ),
            handler=get_visual_content,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_visual_preview",
            description="Retrieve an authenticated Visual PNG preview as an explicit base64 payload.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema},
                required=["visual_id"],
            ),
            handler=get_visual_preview,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_visual_tools"]
