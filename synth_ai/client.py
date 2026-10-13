"""Explicit authenticated/anonymous composition. See backend SDK index_contract.md."""

from __future__ import annotations

import os
from types import TracebackType

from synth_ai.contracts.operations import AUTH_CONTEXT
from synth_ai.contracts.registry import recovery_operation
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.http.request import validate_base_url
from synth_ai.errors import AuthenticationError, ContractError
from synth_ai.options import ClientOptions, ReadOptions
from synth_ai.sdk.index.client import AsyncIndexClient, IndexClient
from synth_ai.sdk.swarms.client import AsyncSwarmsClient, SwarmsClient
from synth_ai.types.auth import AuthContext
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import MutationResolution

DEFAULT_BASE_URL = "https://api.usesynth.ai"


def resolve_key(api_key: str | None) -> str:
    selected = api_key if api_key is not None else os.environ.get("SYNTH_API_KEY")
    if not selected or not selected.strip() or any(character in selected for character in "\r\n"):
        raise AuthenticationError("Provide an API key explicitly or through SYNTH_API_KEY")
    return selected.strip()


def resolve_url(base_url: str | None) -> str:
    return validate_base_url(
        base_url if base_url is not None else os.environ.get("SYNTH_BACKEND_URL", DEFAULT_BASE_URL)
    )


class SynthClient:
    """An owned SDK lifetime, with an optional borrowed HTTP client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: ClientOptions | None = None,
    ) -> None:
        selected = options or ClientOptions()
        if selected.async_http_client is not None:
            raise ValueError("SynthClient requires a synchronous injected client")
        self._transport = HttpTransport(resolve_url(base_url), resolve_key(api_key), selected)
        self.index = IndexClient(self._transport, True)
        self.swarms = SwarmsClient(self._transport)

    @classmethod
    def anonymous(
        cls, *, base_url: str | None = None, options: ClientOptions | None = None
    ) -> SynthClient:
        """Construct a public client without reading ambient credentials or doing network IO.

        See: backend/notes/specifications/sdk/index_contract.md."""
        selected = options or ClientOptions()
        if selected.async_http_client is not None:
            raise ValueError("SynthClient requires a synchronous injected client")
        client = cls.__new__(cls)
        client._transport = HttpTransport(resolve_url(base_url), None, selected)
        client.index = IndexClient(client._transport, False)
        client.swarms = SwarmsClient(client._transport)
        return client

    def auth_context(self, *, request: ReadOptions | None = None) -> AuthContext:
        """Read verified identity and current grants without secrets or tenant overrides.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return self._transport.request(AUTH_CONTEXT, AuthContext, options=request)

    def resolve_mutation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> MutationResolution:
        """Reconcile known scoped mutation identity; unavailable is not proof of nonacceptance.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            recovery_operation(identity),
            MutationResolution,
            query={
                "key": identity.key,
                "operation": identity.operation,
                "resource_path": identity.resource_path,
                "payload_sha256": identity.payload_sha256,
            },
            options=request,
        )
        if result.identity != identity:
            raise ContractError("Resolver returned a different mutation identity")
        return result

    def close(self) -> None:
        """End local ownership; borrowed transports and durable remote work are not cancelled.

        See: backend/notes/specifications/sdk/index_contract.md."""
        self._transport.close()

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncSynthClient:
    """Native async client; construction and anonymous factory do no IO."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: ClientOptions | None = None,
    ) -> None:
        selected = options or ClientOptions()
        if selected.http_client is not None:
            raise ValueError("AsyncSynthClient requires an asynchronous injected client")
        self._transport = AsyncHttpTransport(resolve_url(base_url), resolve_key(api_key), selected)
        self.index = AsyncIndexClient(self._transport, True)
        self.swarms = AsyncSwarmsClient(self._transport)

    @classmethod
    def anonymous(
        cls, *, base_url: str | None = None, options: ClientOptions | None = None
    ) -> AsyncSynthClient:
        """Construct a public client without reading ambient credentials or doing network IO.

        See: backend/notes/specifications/sdk/index_contract.md."""
        selected = options or ClientOptions()
        if selected.http_client is not None:
            raise ValueError("AsyncSynthClient requires an asynchronous injected client")
        client = cls.__new__(cls)
        client._transport = AsyncHttpTransport(resolve_url(base_url), None, selected)
        client.index = AsyncIndexClient(client._transport, False)
        client.swarms = AsyncSwarmsClient(client._transport)
        return client

    async def auth_context(self, *, request: ReadOptions | None = None) -> AuthContext:
        """Read verified identity and current grants without secrets or tenant overrides.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return await self._transport.request(AUTH_CONTEXT, AuthContext, options=request)

    async def resolve_mutation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> MutationResolution:
        """Reconcile known scoped mutation identity; unavailable is not proof of nonacceptance.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            recovery_operation(identity),
            MutationResolution,
            query={
                "key": identity.key,
                "operation": identity.operation,
                "resource_path": identity.resource_path,
                "payload_sha256": identity.payload_sha256,
            },
            options=request,
        )
        if result.identity != identity:
            raise ContractError("Resolver returned a different mutation identity")
        return result

    async def aclose(self) -> None:
        """End async local ownership without cancelling durable work or closing borrowed clients.

        See: backend/notes/specifications/sdk/index_contract.md."""
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncSynthClient:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
