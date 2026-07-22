"""Stable MCP adapters for customer image releases."""

from __future__ import annotations

from collections.abc import Callable

from synth_ai.core.research.client import ResearchClient
from synth_ai.core.research.contracts.image_releases import (
    ImageReleaseFinalizeRequest,
    ImageReleaseId,
    ImageReleaseUploadRequest,
    RuntimeImageReleaseId,
)
from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    JSONDict,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import require_string

CoreClientFactory = Callable[[JSONDict], ResearchClient]


def _declaration(args: JSONDict) -> object:
    value = args.get("declaration")
    if not isinstance(value, dict):
        raise ValueError("'declaration' is required and must be an object")
    return value


def build_image_release_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build the five stable image-release adapters."""

    def create_upload(args: JSONDict) -> JSONDict:
        payload: JSONDict = {"declaration": _declaration(args)}
        if "expires_in" in args:
            payload["expires_in"] = args["expires_in"]
        request = ImageReleaseUploadRequest.from_wire(payload)
        with client_from_args(args) as client:
            return client.image_releases.create_upload(request).to_wire()

    def finalize(args: JSONDict) -> JSONDict:
        request = ImageReleaseFinalizeRequest.from_wire(
            {
                "upload_id": require_string(args, "upload_id"),
                "declaration": _declaration(args),
            }
        )
        with client_from_args(args) as client:
            return client.image_releases.finalize(request).to_wire()

    def list_actor_images(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.image_releases.list().to_wire()

    def archive_actor_image(args: JSONDict) -> JSONDict:
        release_id = RuntimeImageReleaseId(require_string(args, "runtime_image_release_id"))
        with client_from_args(args) as client:
            return client.image_releases.archive(release_id).to_wire()

    def retrieve(args: JSONDict) -> JSONDict:
        release_id = ImageReleaseId(require_string(args, "release_id"))
        with client_from_args(args) as client:
            return client.image_releases.retrieve(release_id).to_wire()

    declaration_schema = {"type": "object", "additionalProperties": True}
    release_id_schema = {
        "type": "string",
        "pattern": "^imgrel_[0-9a-f]{64}$",
    }
    upload_id_schema = {"type": "string", "pattern": "^imgup_[0-9a-f]{32}$"}
    runtime_id_schema = {
        "type": "string",
        "pattern": "^[0-9a-fA-F-]{36}$",
    }
    return [
        ToolDefinition(
            name="smr_create_image_release_upload",
            description="Create a presigned upload URL for one image release declaration.",
            input_schema=tool_schema(
                {
                    "declaration": declaration_schema,
                    "expires_in": {"type": "integer", "minimum": 60, "maximum": 86400},
                },
                required=["declaration"],
            ),
            handler=create_upload,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_finalize_image_release",
            description="Finalize one uploaded image release and return its receipt.",
            input_schema=tool_schema(
                {"upload_id": upload_id_schema, "declaration": declaration_schema},
                required=["upload_id", "declaration"],
            ),
            handler=finalize,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_customer_actor_images",
            description="List customer actor runtime image materializations.",
            input_schema=tool_schema({}, required=[]),
            handler=list_actor_images,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_archive_customer_actor_image",
            description="Archive one customer actor runtime image materialization.",
            input_schema=tool_schema(
                {"runtime_image_release_id": runtime_id_schema},
                required=["runtime_image_release_id"],
            ),
            handler=archive_actor_image,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_retrieve_image_release",
            description="Retrieve one immutable image-release receipt.",
            input_schema=tool_schema({"release_id": release_id_schema}, required=["release_id"]),
            handler=retrieve,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_image_release_tools"]
