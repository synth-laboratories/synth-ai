"""MCP execution boundaries; backend remains the authorization authority."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from synth_ai import AsyncSynthClient
from synth_ai.contracts.registry import OPERATIONS_BY_ID
from synth_ai.core.capabilities import require_operation, supports_operation
from synth_ai.errors import AuthenticationError, PermissionDeniedError, SynthError
from synth_ai.options import ReadOptions
from synth_ai.types.auth import AuthContext
from synth_ai.types.capabilities import Capabilities

from .catalog import TOOLS, ToolSpec
from .contracts import JsonObject
from .policy import POLICY_ACTION_TOOLS, visible_policy_tools


class Executor(Protocol):
    async def available_tools(self) -> frozenset[str]: ...
    async def invoke(self, tool: ToolSpec, arguments: JsonObject) -> JsonObject: ...


@runtime_checkable
class PolicyDiscoveryExecutor(Protocol):
    """Optional discovery projection for executors with account-policy support."""

    async def visible_tools(self) -> frozenset[str]: ...


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
            and tool.operation in OPERATIONS_BY_ID
            and supports_operation(
                receipt,
                OPERATIONS_BY_ID[tool.operation],
                surface="http",
                candidate_manifest_sha256=self._client.candidate_manifest_sha256,
                strict_contract=self._client._transport.strict_contract,
            )
            for tool in TOOLS
        ):
            return None
        return await self._client.auth_context(
            request=ReadOptions(timeout_seconds=10, read_retry_count=0)
        )

    def _permitted(self, tool: ToolSpec, context: AuthContext | None, scope: str | None) -> bool:
        return tool.name in self._enabled and (
            not tool.keyed
            or context is not None
            and (scope is None or scope in context.scopes)
            and all(required in context.scopes for required in tool.additional_scopes)
        )

    async def _current_tools(self) -> tuple[frozenset[str], AuthContext | None]:
        receipt = await self._public_client.capabilities(
            request=ReadOptions(timeout_seconds=10, read_retry_count=0)
        )
        context = await self._authorization(receipt)
        available = frozenset(
            tool.name
            for tool in TOOLS
            if self._permitted(tool, context, tool.scope)
            and tool.operation in OPERATIONS_BY_ID
            and supports_operation(
                receipt,
                OPERATIONS_BY_ID[tool.operation],
                surface="http",
                candidate_manifest_sha256=self._client.candidate_manifest_sha256,
                strict_contract=self._client._transport.strict_contract,
            )
        )
        return available, context

    async def available_tools(self) -> frozenset[str]:
        # Invocation still reaches native policy for exact request conditions
        # and accepted-mutation replay. Discovery is not a second admission gate.
        available, _ = await self._current_tools()
        return available

    async def visible_tools(self) -> frozenset[str]:
        available, context = await self._current_tools()
        if not available & POLICY_ACTION_TOOLS:
            return available
        if "account_entitlements" not in available or context is None:
            return visible_policy_tools(available, None)
        try:
            policy = await self._client.account.entitlements(
                request=ReadOptions(timeout_seconds=10, read_retry_count=0)
            )
        except AuthenticationError:
            raise
        except SynthError:
            # Only new-work discovery is unavailable. Status, cancellation and
            # recovery retain their independent authorization paths.
            return visible_policy_tools(available, None)
        if policy.tenant_id != context.tenant_id:
            raise AuthenticationError("Account policy principal changed")
        return visible_policy_tools(available, policy)

    async def invoke(self, tool: ToolSpec, arguments: JsonObject) -> JsonObject:
        if tool.name not in self._enabled or tool.operation not in OPERATIONS_BY_ID:
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
            strict_contract=self._client._transport.strict_contract,
        )
        context = await self._authorization(receipt)
        if not self._permitted(tool, context, tool.scope_for(arguments)):
            raise PermissionDeniedError(
                "Tool is outside the current grants", code="insufficient_scope"
            )
        return await tool.invoke(self._client if tool.keyed else self._public_client, arguments)
