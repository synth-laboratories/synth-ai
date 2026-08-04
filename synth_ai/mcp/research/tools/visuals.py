"""Stable MCP adapters for first-class Research Visuals.

Mutation surface (v1): create, update, promote, unpublish. ``delete`` and
``restore`` exist in the SDK but are deliberately omitted here — they are
destructive lifecycle operations and stay SDK-only until an explicit MCP
ruling admits them.

Score-series authority: score-series Visuals are server-authoritative. The
backend appends score points as results are scored; no MCP mutation path may
create or rewrite a Visual claiming a score-series kind. See
:func:`_reject_score_series_kind`.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from typing import Any

from synth_ai.core.errors import (
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
)
from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
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
from synth_ai.sdk.research.contracts.visuals import (
    Visual,
    VisualPatch,
    VisualPromotion,
    VisualVisibility,
)

CoreClientFactory = Callable[[JSONDict], ResearchClient]
_MCP_BINARY_CHUNK_BYTES_DEFAULT = 65_536
_MCP_BINARY_CHUNK_BYTES_MAX = 131_072

VISUAL_SCORE_SERIES_ERROR_CODE = "visual_score_series_server_authoritative"

_SCORE_SERIES_GUARD_NOTE = (
    "Score-series Visuals are server-authoritative (the backend appends points "
    "on scored results); any visual_kind claiming a score series is refused "
    f"with error code '{VISUAL_SCORE_SERIES_ERROR_CODE}'."
)


class VisualScoreSeriesServerAuthoritativeError(SynthError):
    """Refusal: MCP mutation must never write score-series Visual data.

    Score series are appended by the backend as results are scored. Allowing
    a client-supplied Visual to claim a score-series kind would let arbitrary
    HTML masquerade as authoritative score data, so the MCP layer refuses the
    payload before any client or network request is constructed.
    """

    def __init__(self, visual_kind: str) -> None:
        detail = (
            f"visual_kind {visual_kind!r} claims a score-series Visual; score "
            "series are server-authoritative (the backend appends points on "
            "scored results) and cannot be created or rewritten through "
            "MCP/SDK visual mutation."
        )
        super().__init__(
            detail,
            failure=SynthFailure(
                code=SynthErrorCode(VISUAL_SCORE_SERIES_ERROR_CODE),
                category=SynthErrorCategory.VALIDATION,
                operation="mutate_visual",
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=None,
                detail=detail,
            ),
        )
        self.visual_kind = visual_kind


def _reject_score_series_kind(visual_kind: str | None) -> None:
    """Refuse any visual_kind that claims score-series authority.

    Separator- and case-insensitive: ``score_series``, ``score-series``,
    ``Score Series``, ``scoreSeries``, and namespaced spellings such as
    ``factory.score_series`` all collapse to the same claim and are refused.
    """
    if visual_kind is None:
        return
    collapsed = "".join(char for char in visual_kind.lower() if char.isalnum())
    if "scoreseries" in collapsed:
        raise VisualScoreSeriesServerAuthoritativeError(visual_kind)


def _optional_visibility(args: JSONDict) -> VisualVisibility | None:
    value = optional_string(args, "visibility")
    return VisualVisibility(value) if value is not None else None


def _optional_metadata(args: JSONDict) -> dict[str, Any] | None:
    value = args.get("metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("'metadata' must be an object when provided")
    return value


def _optional_string_array(args: JSONDict, key: str) -> tuple[str, ...]:
    value = args.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"'{key}' must be an array of non-empty strings when provided")
    return tuple(item.strip() for item in value)


def _optional_base64_bytes(args: JSONDict, key: str) -> bytes | None:
    value = optional_string(args, key)
    if value is None:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"'{key}' must be valid base64 when provided") from exc


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
    """Build the stable authenticated Visual read and mutate adapters."""

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

    def create_visual(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        title = require_string(args, "title")
        html = require_string(args, "html")
        visual_kind = optional_string(args, "visual_kind") or "research_visual"
        _reject_score_series_kind(visual_kind)
        summary = optional_string(args, "summary")
        source_run_ids = _optional_string_array(args, "source_run_ids")
        metadata = _optional_metadata(args)
        root_artifact_id = optional_string(args, "root_artifact_id")
        preview_png = _optional_base64_bytes(args, "preview_png_base64")
        with client_from_args(args) as client:
            return client.visuals.create(
                project_id,
                title=title,
                summary=summary,
                html=html,
                visual_kind=visual_kind,
                source_run_ids=source_run_ids,
                metadata=metadata,
                root_artifact_id=(
                    ArtifactId(root_artifact_id) if root_artifact_id is not None else None
                ),
                preview_png=preview_png,
            ).to_wire()

    def update_visual(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        visual_kind = optional_string(args, "visual_kind")
        _reject_score_series_kind(visual_kind)
        patch = VisualPatch(
            title=optional_string(args, "title"),
            summary=optional_string(args, "summary"),
            visual_kind=visual_kind,
        )
        if patch.title is None and patch.summary is None and patch.visual_kind is None:
            raise ValueError("update must change at least one of 'title', 'summary', 'visual_kind'")
        with client_from_args(args) as client:
            return client.visuals.update(visual_id, patch).to_wire()

    def promote_visual(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        promotion = VisualPromotion(slug=require_string(args, "slug"))
        with client_from_args(args) as client:
            return client.visuals.promote(visual_id, promotion).to_wire()

    def unpublish_visual(args: JSONDict) -> JSONDict:
        visual_id = ArtifactId(require_string(args, "visual_id"))
        with client_from_args(args) as client:
            return client.visuals.unpublish(visual_id).to_wire()

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
        ToolDefinition(
            name="research_create_visual",
            description=(
                "Publish a project-owned Visual from self-contained HTML. "
                f"{_SCORE_SERIES_GUARD_NOTE}"
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Research project ID that owns the Visual.",
                    },
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Human-readable Visual title.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional short summary.",
                    },
                    "html": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Complete self-contained HTML document (UTF-8).",
                    },
                    "visual_kind": {
                        "type": "string",
                        "default": "research_visual",
                        "description": (
                            "Visual kind label. Score-series kinds are refused; "
                            "score series are server-authoritative."
                        ),
                    },
                    "source_run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Run IDs whose evidence backs this Visual.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional JSON metadata for the Visual.",
                    },
                    "root_artifact_id": {
                        "type": "string",
                        "description": (
                            "Existing root artifact ID to publish a new version under."
                        ),
                    },
                    "preview_png_base64": {
                        "type": "string",
                        "description": "Optional base64-encoded PNG preview image.",
                    },
                },
                required=["project_id", "title", "html"],
            ),
            handler=create_visual,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_update_visual",
            description=(
                "Update the mutable metadata (title, summary, visual_kind) of one "
                f"Visual. {_SCORE_SERIES_GUARD_NOTE}"
            ),
            input_schema=tool_schema(
                {
                    "visual_id": visual_id_schema,
                    "title": {"type": "string", "minLength": 1},
                    "summary": {"type": "string"},
                    "visual_kind": {
                        "type": "string",
                        "description": (
                            "New visual kind label. Score-series kinds are refused; "
                            "score series are server-authoritative."
                        ),
                    },
                },
                required=["visual_id"],
            ),
            handler=update_visual,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_promote_visual",
            description=(
                "Promote a Visual to a stable public slug through the backend's "
                "ADMIN/OWNER authority."
            ),
            input_schema=tool_schema(
                {
                    "visual_id": visual_id_schema,
                    "slug": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Requested stable public slug.",
                    },
                },
                required=["visual_id", "slug"],
            ),
            handler=promote_visual,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_unpublish_visual",
            description="Remove a Visual from its published public surface.",
            input_schema=tool_schema(
                {"visual_id": visual_id_schema},
                required=["visual_id"],
            ),
            handler=unpublish_visual,
            required_scopes=WRITE_SCOPES,
        ),
    ]


__all__ = [
    "VISUAL_SCORE_SERIES_ERROR_CODE",
    "VisualScoreSeriesServerAuthoritativeError",
    "build_visual_tools",
]
