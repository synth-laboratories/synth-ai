"""Results-stage WorkProduct tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_output_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_run_work_products",
            description=(
                "List durable WorkProducts for a run: Models, Containers / Evals, "
                "and Reports. Prefer this over raw artifact/output tools."
            ),
            input_schema=tool_schema(
                {"project_id": {"type": "string"}, "run_id": {"type": "string"}},
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_run_work_products,
        ),
        ToolDefinition(
            name="smr_get_run_work_product",
            description="Fetch a WorkProduct by id.",
            input_schema=tool_schema(
                {"work_product_id": {"type": "string"}},
                required=["work_product_id"],
            ),
            handler=server._tool_get_run_work_product,
        ),
        ToolDefinition(
            name="smr_get_run_work_product_content",
            description="Fetch readable content for a report or artifact-backed WorkProduct.",
            input_schema=tool_schema(
                {"work_product_id": {"type": "string"}},
                required=["work_product_id"],
            ),
            handler=server._tool_get_run_work_product_content,
        ),
        ToolDefinition(
            name="smr_export_run_work_product",
            description=(
                "Export a downloadable/importable WorkProduct to a backend-mediated "
                "destination. Destination secrets are redacted in responses."
            ),
            input_schema=tool_schema(
                {
                    "work_product_id": {"type": "string"},
                    "destination": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                required=["work_product_id", "destination"],
            ),
            handler=server._tool_export_run_work_product,
        ),
        ToolDefinition(
            name="smr_explain_work_product_blocker",
            description="Explain why a WorkProduct is blocked or unavailable.",
            input_schema=tool_schema(
                {"work_product_id": {"type": "string"}},
                required=["work_product_id"],
            ),
            handler=server._tool_explain_work_product_blocker,
        ),
        ToolDefinition(
            name="smr_upload_container_eval_package",
            description=(
                "Register a container/eval package WorkProduct for a run. The "
                "package should include dataset, evaluator, runtime contract, and "
                "validation manifest."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "synth_container",
                            "eval_package",
                            "benchmark_harness",
                        ],
                    },
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "artifact_id": {"type": "string"},
                    "storage_uri": {"type": "string"},
                    "archive_size_bytes": {"type": "integer"},
                    "manifest": {"type": "object"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "run_id", "kind", "name"],
            ),
            handler=server._tool_upload_container_eval_package,
        ),
        ToolDefinition(
            name="smr_validate_container_eval_package",
            description="Validate a registered container/eval package.",
            input_schema=tool_schema(
                {"package_id": {"type": "string"}},
                required=["package_id"],
            ),
            handler=server._tool_validate_container_eval_package,
        ),
    ]


__all__ = ["build_output_tools"]
