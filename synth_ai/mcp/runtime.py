"""MCP execution boundaries; backend remains the authorization authority."""

from __future__ import annotations

from typing import Protocol

from synth_ai import AsyncSynthClient
from synth_ai.contracts.registry import OPERATIONS_BY_ID
from synth_ai.core.capabilities import require_operation, supports_operation
from synth_ai.errors import PermissionDeniedError
from synth_ai.options import ReadOptions
from synth_ai.types.auth import AuthContext
from synth_ai.types.capabilities import Capabilities

from .catalog import TOOLS, ToolSpec
from .contracts import JsonObject


class Executor(Protocol):
    async def available_tools(self) -> frozenset[str]: ...
    async def invoke(self, tool: ToolSpec, arguments: JsonObject) -> JsonObject: ...


class SDKExecutor:
    def __init__(
        self,
        client: AsyncSynthClient,
        public_client: AsyncSynthClient,
        enabled_tools: frozenset[str],
    ) -> None:
        if (
            client.base_url != public_client.base_url
            or client.candidate_manifest_sha256 != public_client.candidate_manifest_sha256
        ):
            raise ValueError(
                "MCP public and keyed clients must share one backend origin and candidate policy"
            )
        known = {tool.name for tool in TOOLS}
        if not enabled_tools <= known:
            raise ValueError("MCP profile contains an unknown tool")
        self._client = client
        self._public_client = public_client
        self._enabled = enabled_tools

    async def _authorization(self, receipt: Capabilities) -> AuthContext | None:
        if not any(
            tool.keyed
            and tool.name in self._enabled
            and supports_operation(
                receipt,
                OPERATIONS_BY_ID[tool.operation],
                surface="http",
                candidate_manifest_sha256=self._client.candidate_manifest_sha256,
            )
            for tool in TOOLS
        ):
            return None
        return await self._client.auth_context(
            request=ReadOptions(timeout_seconds=10, read_retry_count=0)
        )

    def _permitted(self, tool: ToolSpec, context: AuthContext | None, scope: str | None) -> bool:
        return tool.name in self._enabled and (
            not tool.keyed or context is not None and (scope is None or scope in context.scopes)
        )

    async def available_tools(self) -> frozenset[str]:
        receipt = await self._public_client.capabilities(
            request=ReadOptions(timeout_seconds=10, read_retry_count=0)
        )
        context = await self._authorization(receipt)
        return frozenset(
            tool.name
            for tool in TOOLS
            if self._permitted(tool, context, tool.scope)
            and supports_operation(
                receipt,
                OPERATIONS_BY_ID[tool.operation],
                surface="http",
                candidate_manifest_sha256=self._client.candidate_manifest_sha256,
            )
        )

    async def invoke(self, tool: ToolSpec, arguments: JsonObject) -> JsonObject:
        if tool.name not in self._enabled:
            raise PermissionDeniedError(
                "Tool is disabled for this connection", code="tool_disabled"
            )
        receipt = await self._public_client.capabilities(
            request=ReadOptions(timeout_seconds=10, read_retry_count=0)
        )
        require_operation(
            receipt,
            OPERATIONS_BY_ID[tool.operation],
            surface="http",
            candidate_manifest_sha256=self._client.candidate_manifest_sha256,
        )
        context = await self._authorization(receipt)
        if not self._permitted(tool, context, tool.scope_for(arguments)):
            raise PermissionDeniedError(
                "Tool is outside the current grants", code="insufficient_scope"
            )
        return await tool.invoke(self._client if tool.keyed else self._public_client, arguments)
