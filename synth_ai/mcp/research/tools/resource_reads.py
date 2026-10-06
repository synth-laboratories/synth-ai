"""Read-only native dataset/pool MCP adapters over the public typed SDK.

# See: testing-resources/specifications/sdk/forge_resource_reads.md
Discovery and dispatch share one registry; backend authority remains required.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from synth_ai.mcp.research.registry import READ_SCOPES, ToolDefinition
from synth_ai.sdk.research.client import Client
from synth_ai.sdk.research.contracts.resource_reads import (
    DataPoolReadReference,
    DatasetRevisionReadReference,
)

MCP_RESOURCE_BYTES_MAX = 4 * 1024 * 1024
RESOURCE_READ_TOOL_NAMES = frozenset(
    {
        "research_get_exact_dataset_revision",
        "research_read_dataset_revision_manifest",
        "research_read_dataset_revision_content",
        "research_list_project_data_pools",
        "research_get_project_data_pool_inventory",
        "research_read_project_data_pool_file",
    }
)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRevisionArguments(_Arguments):
    reference: DatasetRevisionReadReference


class DatasetRevisionContentArguments(DatasetRevisionArguments):
    logical_path: str = Field(min_length=1, max_length=1024)


class PoolListArguments(_Arguments):
    project_id: str = Field(min_length=1, max_length=255)


class PoolInventoryArguments(_Arguments):
    reference: DataPoolReadReference


class PoolFileArguments(PoolInventoryArguments):
    file_id: str = Field(min_length=1, max_length=255)
    content_digest: str = Field(pattern=r"^(?:sha256:)?[0-9a-f]{64}$")


def _byte_payload(content: bytes) -> dict[str, Any]:
    return {
        "encoding": "base64",
        "content": base64.b64encode(content).decode("ascii"),
        "content_digest": "sha256:" + hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def build_resource_read_tools(
    client_factory: Callable[[dict[str, Any]], AbstractContextManager[Client]],
) -> list[ToolDefinition]:
    """Expose all six exact resource reads with explicit existing read scopes.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    Byte replies are bounded to 4 MiB, verified by the SDK, then base64 encoded.
    """

    def revision(arguments):
        selected = DatasetRevisionArguments.model_validate(arguments)
        with client_factory({}) as client:
            return client.projects.dataset_revisions.get(selected.reference).model_dump(mode="json")

    def manifest(arguments):
        selected = DatasetRevisionArguments.model_validate(arguments)
        with client_factory({}) as client:
            content = client.projects.dataset_revisions.download_manifest(
                selected.reference, max_bytes=MCP_RESOURCE_BYTES_MAX
            )
        return {"reference": selected.reference.model_dump(mode="json"), **_byte_payload(content)}

    def content(arguments):
        selected = DatasetRevisionContentArguments.model_validate(arguments)
        with client_factory({}) as client:
            body = client.projects.dataset_revisions.download_content(
                selected.reference, selected.logical_path, max_bytes=MCP_RESOURCE_BYTES_MAX
            )
        return {
            "reference": selected.reference.model_dump(mode="json"),
            "logical_path": selected.logical_path,
            **_byte_payload(body),
        }

    def pools(arguments):
        selected = PoolListArguments.model_validate(arguments)
        with client_factory({}) as client:
            return [
                item.model_dump(mode="json")
                for item in client.projects.data_pools.list(selected.project_id)
            ]

    def inventory(arguments):
        selected = PoolInventoryArguments.model_validate(arguments)
        with client_factory({}) as client:
            return client.projects.data_pools.get(selected.reference).model_dump(mode="json")

    def file_content(arguments):
        selected = PoolFileArguments.model_validate(arguments)
        with client_factory({}) as client:
            body = client.projects.data_pools.download_file(
                selected.reference,
                selected.file_id,
                content_digest=selected.content_digest,
                max_bytes=MCP_RESOURCE_BYTES_MAX,
            )
        return {
            "reference": selected.reference.model_dump(mode="json"),
            "file_id": selected.file_id,
            **_byte_payload(body),
        }

    return [
        ToolDefinition(
            name=name,
            description=description,
            input_schema=model.model_json_schema(),
            handler=handler,
            required_scopes=READ_SCOPES,
        )
        for name, description, model, handler in (
            (
                "research_get_exact_dataset_revision",
                "Read one exact native sealed DatasetRevision.",
                DatasetRevisionArguments,
                revision,
            ),
            (
                "research_read_dataset_revision_manifest",
                "Read original canonical dataset manifest bytes up to 4 MiB.",
                DatasetRevisionArguments,
                manifest,
            ),
            (
                "research_read_dataset_revision_content",
                "Read one original declared dataset object up to 4 MiB.",
                DatasetRevisionContentArguments,
                content,
            ),
            (
                "research_list_project_data_pools",
                "Read existing native project data-pool descriptors.",
                PoolListArguments,
                pools,
            ),
            (
                "research_get_project_data_pool_inventory",
                "Read a complete scoped native data-pool inventory.",
                PoolInventoryArguments,
                inventory,
            ),
            (
                "research_read_project_data_pool_file",
                "Read an exact digest-pinned native pool file up to 4 MiB.",
                PoolFileArguments,
                file_content,
            ),
        )
    ]
