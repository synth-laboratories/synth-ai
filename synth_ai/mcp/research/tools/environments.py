"""Stable MCP adapters for the typed Environment catalog."""

from __future__ import annotations

from collections.abc import Callable

from synth_ai.core.research.client import ResearchClient
from synth_ai.core.research.contracts.common import EnvironmentDigest, EnvironmentName
from synth_ai.core.research.contracts.environment_manifest import (
    ENVIRONMENT_SCHEMA_VERSION,
    EnvironmentManifest,
    RuntimeImageKind,
)
from synth_ai.mcp.research.registry import (
    JSONDict,
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import optional_int, optional_string, require_string


CoreClientFactory = Callable[[JSONDict], ResearchClient]


def _manifest(args: JSONDict) -> EnvironmentManifest:
    value = args.get("manifest")
    if not isinstance(value, dict):
        raise ValueError("'manifest' is required and must be an object")
    return EnvironmentManifest.from_input(value)


def _manifest_digest(args: JSONDict) -> EnvironmentDigest | None:
    value = optional_string(args, "manifest_digest")
    return EnvironmentDigest(value) if value is not None else None


def _nullable_string() -> JSONDict:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _closed_object(properties: JSONDict, *, required: list[str]) -> JSONDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _manifest_schema() -> JSONDict:
    nullable = _nullable_string()
    image = _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": [kind.value for kind in RuntimeImageKind],
            },
            "ref": {"type": "string", "minLength": 1, "maxLength": 4000},
            "digest": nullable,
        },
        required=["ref"],
    )
    secret = _closed_object(
        {
            "label": {"type": "string", "minLength": 1, "maxLength": 255},
            "kind": nullable,
            "provider": nullable,
            "required": {"type": "boolean"},
        },
        required=["label"],
    )
    repository = _closed_object(
        {
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "url": nullable,
            "role": {"type": "string", "minLength": 1, "maxLength": 128},
            "branch": nullable,
            "required": {"type": "boolean"},
        },
        required=["name"],
    )
    mount = _closed_object(
        {
            "path": {"type": "string", "minLength": 1, "maxLength": 4000},
            "source": nullable,
            "required": {"type": "boolean"},
        },
        required=["path"],
    )
    isolation = _closed_object(
        {
            "egress_allowlist": {"type": "array", "items": {"type": "string"}},
            "network_mode": nullable,
        },
        required=[],
    )
    preflight = _closed_object(
        {
            "cmd": nullable,
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1800,
            },
        },
        required=[],
    )
    return _closed_object(
        {
            "schema_version": {
                "type": "string",
                "const": ENVIRONMENT_SCHEMA_VERSION,
            },
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "digest": nullable,
            "images": {"type": "array", "items": image, "minItems": 1},
            "secrets": {"type": "array", "items": secret},
            "repos": {"type": "array", "items": repository},
            "mounts": {"type": "array", "items": mount},
            "isolation": isolation,
            "preflight": preflight,
            "metadata": {"type": "object"},
        },
        required=["name", "images"],
    )


def build_environment_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build the four stable adapters in the bounded backend contract."""

    def list_environments(args: JSONDict) -> list[JSONDict]:
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return [
                environment.to_wire()
                for environment in client.environments.list(
                    limit=100 if limit is None else limit
                )
            ]

    def create_environment(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.environments.create(_manifest(args)).to_wire()

    def get_environment(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.environments.retrieve(
                EnvironmentName(require_string(args, "name")),
                manifest_digest=_manifest_digest(args),
            ).to_wire()

    def preflight_environment(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.environments.preflight(
                EnvironmentName(require_string(args, "name")),
                manifest_digest=_manifest_digest(args),
            ).to_wire()

    selector = {
        "name": {"type": "string", "minLength": 1, "maxLength": 255},
        "manifest_digest": {
            "type": "string",
            "pattern": "^sha256:[0-9a-f]{64}$",
        },
    }
    return [
        ToolDefinition(
            name="smr_list_environments",
            description="List immutable Research Environment manifest versions.",
            input_schema=tool_schema(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
                required=[],
            ),
            handler=list_environments,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_create_environment",
            description="Catalog one strict, content-addressed Environment manifest.",
            input_schema=tool_schema(
                {"manifest": _manifest_schema()},
                required=["manifest"],
            ),
            handler=create_environment,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_environment",
            description="Retrieve one immutable Environment manifest version.",
            input_schema=tool_schema(selector, required=["name"]),
            handler=get_environment,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_preflight_environment",
            description="Run backend-owned preflight for one Environment version.",
            input_schema=tool_schema(selector, required=["name"]),
            handler=preflight_environment,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_environment_tools"]
