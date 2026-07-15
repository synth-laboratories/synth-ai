"""CloudDeployment MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema

_CLOUD_DEPLOYMENT_ID = {
    "deployment_id": {
        "type": "string",
        "description": "CloudDeployment id.",
    }
}

_REASON = {
    "reason": {
        "type": "string",
        "description": "Optional human-readable lifecycle reason.",
    }
}

_FENCING_TOKEN = {
    "fencing_token": {
        "type": "integer",
        "minimum": 1,
        "description": "Active claim fencing token required for a claimed deployment.",
    }
}


def _cloud_deployment_action_schema() -> dict[str, Any]:
    return tool_schema(
        {
            **_CLOUD_DEPLOYMENT_ID,
            **_REASON,
            **_FENCING_TOKEN,
        },
        required=["deployment_id"],
    )


def build_cloud_deployment_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_cloud_deployments",
            description="List durable CloudDeployments visible to the authenticated org.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum row count.",
                    },
                },
                required=[],
            ),
            handler=server._tool_list_cloud_deployments,
        ),
        ToolDefinition(
            name="smr_create_cloud_deployment",
            description="Create a durable service CloudDeployment request on the exe.dev service lane.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed Research project id.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable CloudDeployment name, unique within the org.",
                    },
                    "topology_id": {
                        "type": "string",
                        "description": "Service topology id, for example synth-dev.",
                    },
                    "topology_version": {
                        "type": "string",
                        "description": "Optional topology version.",
                    },
                    "host_kind": {
                        "type": "string",
                        "description": "Service substrate kind. Defaults to exe_dev.",
                    },
                    "cloud_slot": {
                        "type": "string",
                        "enum": ["slot1-cloud", "slot2-cloud"],
                        "description": "Optional canonical cloud-slot identity.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional caller metadata for lifecycle receipts.",
                    },
                    "source": {
                        "type": "object",
                        "description": (
                            "Optional immutable project-git source binding. Remote URLs "
                            "and credentials are not accepted."
                        ),
                        "properties": {
                            "kind": {
                                "type": "string",
                                "const": "project_git",
                                "description": "Project-git source authority.",
                            },
                            "source_commit_sha": {
                                "type": "string",
                                "pattern": "^[0-9a-fA-F]{40}$",
                                "description": "Full source commit SHA.",
                            },
                            "evidence_commit_sha": {
                                "type": "string",
                                "pattern": "^[0-9a-fA-F]{40}$",
                                "description": "Full evidence commit SHA.",
                            },
                            "instance_id": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Reflexion instance identity.",
                            },
                        },
                        "required": [
                            "kind",
                            "source_commit_sha",
                            "evidence_commit_sha",
                            "instance_id",
                        ],
                        "additionalProperties": False,
                    },
                },
                required=["project_id", "name", "topology_id"],
            ),
            handler=server._tool_create_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_get_cloud_deployment",
            description="Fetch one durable CloudDeployment.",
            input_schema=tool_schema(_CLOUD_DEPLOYMENT_ID, required=["deployment_id"]),
            handler=server._tool_get_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_get_cloud_deployment_services",
            description="Discover topology-declared services, health checks, and routed endpoints.",
            input_schema=tool_schema(_CLOUD_DEPLOYMENT_ID, required=["deployment_id"]),
            handler=server._tool_get_cloud_deployment_services,
        ),
        ToolDefinition(
            name="smr_get_cloud_deployment_workspace",
            description="Inspect declared repositories and live Git/source materialization proof.",
            input_schema=tool_schema(_CLOUD_DEPLOYMENT_ID, required=["deployment_id"]),
            handler=server._tool_get_cloud_deployment_workspace,
        ),
        ToolDefinition(
            name="smr_materialize_cloud_deployment_workspace",
            description=(
                "Materialize an exact commit for a topology-declared repository; "
                "requires the active claim fencing token."
            ),
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    **_FENCING_TOKEN,
                    "repository": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Topology-declared repository selector.",
                    },
                    "branch": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Declared branch containing the exact commit.",
                    },
                    "source_commit_sha": {
                        "type": "string",
                        "pattern": "^[0-9a-fA-F]{40}$",
                        "description": "Exact full Git commit SHA to materialize.",
                    },
                },
                required=[
                    "deployment_id",
                    "repository",
                    "branch",
                    "source_commit_sha",
                    "fencing_token",
                ],
            ),
            handler=server._tool_materialize_cloud_deployment_workspace,
        ),
        ToolDefinition(
            name="smr_exec_cloud_deployment",
            description=(
                "Execute argv inside a CloudDeployment workspace with bounded output; "
                "requires the active claim fencing token."
            ),
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    **_FENCING_TOKEN,
                    "argv": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 128,
                        "items": {"type": "string", "maxLength": 4096},
                        "description": "Command argv; shell command strings are not accepted.",
                    },
                    "cwd": {
                        "type": "string",
                        "maxLength": 512,
                        "description": "Optional path relative to the declared workspace root.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 900,
                        "default": 300,
                    },
                    "max_output_bytes": {
                        "type": "integer",
                        "minimum": 1024,
                        "maximum": 262144,
                        "default": 65536,
                    },
                },
                required=["deployment_id", "argv", "fencing_token"],
            ),
            handler=server._tool_exec_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_get_cloud_deployment_logs",
            description="Read bounded logs for one topology-declared service.",
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    "service_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Service id returned by service discovery.",
                    },
                    "tail": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000,
                        "default": 200,
                    },
                },
                required=["deployment_id", "service_id"],
            ),
            handler=server._tool_get_cloud_deployment_logs,
        ),
        ToolDefinition(
            name="smr_observe_cloud_deployment",
            description="Observe one CloudDeployment against its substrate and update lifecycle state.",
            input_schema=tool_schema(
                {**_CLOUD_DEPLOYMENT_ID, **_FENCING_TOKEN},
                required=["deployment_id"],
            ),
            handler=server._tool_observe_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_deploy_cloud_deployment",
            description=(
                "Request deployment steps for a VM-ready CloudDeployment or retry "
                "a failed CloudDeployment after fixing the reported cause."
            ),
            input_schema=_cloud_deployment_action_schema(),
            handler=server._tool_deploy_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_retire_cloud_deployment",
            description="Retire a CloudDeployment, retaining the VM unless delete_vm is explicitly true.",
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    **_REASON,
                    **_FENCING_TOKEN,
                    "delete_vm": {
                        "type": "boolean",
                        "description": "Whether to delete the substrate VM. Defaults to false.",
                    },
                    "confirm_vm_name": {
                        "type": "string",
                        "description": "Required when delete_vm is true; must match the VM name.",
                    },
                },
                required=["deployment_id"],
            ),
            handler=server._tool_retire_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_acquire_cloud_deployment_claim",
            description="Acquire the TTL-bounded claim and mint a fencing token.",
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    "holder": {
                        "type": "string",
                        "description": "Stable operator or agent identity.",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Human-readable reason for the claim.",
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Claim lifetime before a heartbeat is required.",
                    },
                },
                required=["deployment_id", "holder", "purpose", "ttl_seconds"],
            ),
            handler=server._tool_acquire_cloud_deployment_claim,
        ),
        ToolDefinition(
            name="smr_heartbeat_cloud_deployment_claim",
            description="Renew the active cloud-deployment claim TTL.",
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    "claim_id": {
                        "type": "string",
                        "description": "Claim id returned by acquire.",
                    },
                },
                required=["deployment_id", "claim_id"],
            ),
            handler=server._tool_heartbeat_cloud_deployment_claim,
        ),
        ToolDefinition(
            name="smr_release_cloud_deployment_claim",
            description="Release a cloud-deployment claim idempotently.",
            input_schema=tool_schema(
                {
                    **_CLOUD_DEPLOYMENT_ID,
                    "claim_id": {
                        "type": "string",
                        "description": "Claim id returned by acquire.",
                    },
                },
                required=["deployment_id", "claim_id"],
            ),
            handler=server._tool_release_cloud_deployment_claim,
        ),
        ToolDefinition(
            name="smr_get_cloud_deployment_claims",
            description="Read active claim and last-issued fencing-token truth.",
            input_schema=tool_schema(_CLOUD_DEPLOYMENT_ID, required=["deployment_id"]),
            handler=server._tool_get_cloud_deployment_claims,
        ),
    ]


__all__ = ["build_cloud_deployment_tools"]
