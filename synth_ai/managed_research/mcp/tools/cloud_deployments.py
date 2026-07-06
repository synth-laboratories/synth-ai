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


def _cloud_deployment_action_schema() -> dict[str, Any]:
    return tool_schema(
        {
            **_CLOUD_DEPLOYMENT_ID,
            **_REASON,
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
                    "metadata": {
                        "type": "object",
                        "description": "Optional caller metadata for lifecycle receipts.",
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
            name="smr_observe_cloud_deployment",
            description="Observe one CloudDeployment against its substrate and update lifecycle state.",
            input_schema=tool_schema(_CLOUD_DEPLOYMENT_ID, required=["deployment_id"]),
            handler=server._tool_observe_cloud_deployment,
        ),
        ToolDefinition(
            name="smr_deploy_cloud_deployment",
            description="Request deployment steps for a CloudDeployment whose VM is ready.",
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
    ]


__all__ = ["build_cloud_deployment_tools"]
