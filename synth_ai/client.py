"""Explicit authenticated/anonymous composition. See backend SDK index_contract.md."""

from __future__ import annotations

import asyncio
from types import TracebackType

from synth_ai.auth.provider import StoredCredentialProvider
from synth_ai.contracts.operations import AUTH_CONTEXT, CAPABILITIES
from synth_ai.contracts.registry import recovery_operation
from synth_ai.core.environment import environment_value
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.cleanup import handle_cleanup_failure
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.http.request import validate_base_url
from synth_ai.errors import AuthenticationError, ContractError
from synth_ai.options import ClientOptions, ReadOptions
from synth_ai.sdk.account.client import AccountClient, AsyncAccountClient
from synth_ai.sdk.core.artifacts import Artifacts, AsyncArtifacts
from synth_ai.sdk.core.links import AsyncLinks, Links
from synth_ai.sdk.core.repos import AsyncRepositories, Repositories
from synth_ai.sdk.forge.client import AsyncForgeClient, ForgeClient
from synth_ai.sdk.index.client import AsyncIndexClient, IndexClient
from synth_ai.sdk.menagerie.client import AsyncMenagerieClient, MenagerieClient
from synth_ai.sdk.swarms.client import AsyncSwarmsClient, SwarmsClient
from synth_ai.sdk.tunnels.client import AsyncTunnelsClient, TunnelsClient
from synth_ai.types.auth import AuthContext
from synth_ai.types.capabilities import Capabilities
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import MutationResolution

DEFAULT_BASE_URL = "https://api.usesynth.ai"


def resolve_key(api_key: str | None) -> str:
    selected = (
        api_key
        if api_key is not None
        else environment_value("SYNTH_API_KEY", "SYNTH_USER_KEY", "SYNTH_BACKEND_API_KEY")
    )
    if not selected or not selected.strip() or any(character in selected for character in "\r\n"):
        raise AuthenticationError("Provide an API key explicitly or through SYNTH_API_KEY")
    return selected.strip()


def resolve_url(base_url: str | None) -> str:
    selected = (
        base_url
        if base_url is not None
        else environment_value("SYNTH_API_URL", "SYNTH_BACKEND_URL", "SYNTH_API_BASE_URL")
    )
    return validate_base_url(DEFAULT_BASE_URL if selected is None else selected)


def _credentials(
    api_key: str | None, origin: str
) -> tuple[str | None, StoredCredentialProvider | None]:
    selected = (
        api_key
        if api_key is not None
        else environment_value("SYNTH_API_KEY", "SYNTH_USER_KEY", "SYNTH_BACKEND_API_KEY")
    )
    if selected is not None:
        return resolve_key(selected), None
    return None, StoredCredentialProvider(origin)


class SynthClient:
    """An owned SDK lifetime with cancellable network IO."""

    _transport: HttpTransport
    index: IndexClient
    swarms: SwarmsClient
    account: AccountClient
    tunnels: TunnelsClient
    menagerie: MenagerieClient
    repos: Repositories
    links: Links
    artifacts: Artifacts
    forge: ForgeClient

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: ClientOptions | None = None,
    ) -> None:
        selected = options or ClientOptions()
        if selected.async_http_client is not None:
            raise ValueError(
                "async_http_client requires AsyncSynthClient; use http_client with SynthClient"
            )
        origin = resolve_url(base_url)
        key, provider = _credentials(api_key, origin)
        self._transport = HttpTransport(origin, key, selected, credential_provider=provider)
        self.index = IndexClient(self._transport, True)
        self.swarms = SwarmsClient(self._transport)
        self.account = AccountClient(self._transport)
        self.tunnels = TunnelsClient(self._transport)
        self.menagerie = MenagerieClient(self._transport)
        self.repos = Repositories(self._transport)
        self.links = Links(self._transport)
        self.artifacts = Artifacts(self._transport)
        self.forge = ForgeClient(self._transport)

    @classmethod
    def anonymous(
        cls, *, base_url: str | None = None, options: ClientOptions | None = None
    ) -> SynthClient:
        """Construct a public client without reading ambient credentials or doing network IO.

        See: the public contract snapshots in contracts/"""
        selected = options or ClientOptions()
        if selected.async_http_client is not None:
            raise ValueError(
                "async_http_client requires AsyncSynthClient; use http_client with SynthClient"
            )
        client = cls.__new__(cls)
        client._transport = HttpTransport(resolve_url(base_url), None, selected)
        client.index = IndexClient(client._transport, False)
        client.swarms = SwarmsClient(client._transport)
        client.account = AccountClient(client._transport)
        client.tunnels = TunnelsClient(client._transport)
        client.menagerie = MenagerieClient(client._transport)
        client.repos = Repositories(client._transport)
        client.links = Links(client._transport)
        client.artifacts = Artifacts(client._transport)
        client.forge = ForgeClient(client._transport)
        return client

    def capabilities(self, *, request: ReadOptions | None = None) -> Capabilities:
        """Read nonsecret deployment facts; these do not grant resource access.

        See: the public contract snapshots in contracts/"""
        return self._transport.request(CAPABILITIES, Capabilities, options=request)

    @property
    def candidate_manifest_sha256(self) -> str | None:
        """Explicit nonproduction candidate pin; absent for ordinary clients."""
        return self._transport.candidate_manifest_sha256

    @property
    def base_url(self) -> str:
        """The validated API origin selected for this client."""
        return self._transport.base_url

    def auth_context(self, *, request: ReadOptions | None = None) -> AuthContext:
        """Read verified identity and current grants without secrets or tenant overrides.

        See: the public contract snapshots in contracts/"""
        return self._transport.request(AUTH_CONTEXT, AuthContext, options=request)

    def resolve_mutation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> MutationResolution:
        """Reconcile known scoped mutation identity; unavailable is not proof of nonacceptance.

        See: the public contract snapshots in contracts/"""
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
        """Close owned network resources without cancelling durable remote work.

        See: the public contract snapshots in contracts/"""
        self._transport.close()

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except Exception as cause:
            handle_cleanup_failure(cause, exception, mutation=False, identity=None)


class AsyncSynthClient:
    """Native async client; construction and anonymous factory do no IO."""

    _transport: AsyncHttpTransport
    index: AsyncIndexClient
    swarms: AsyncSwarmsClient
    account: AsyncAccountClient
    tunnels: AsyncTunnelsClient
    menagerie: AsyncMenagerieClient
    repos: AsyncRepositories
    links: AsyncLinks
    artifacts: AsyncArtifacts
    forge: AsyncForgeClient

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: ClientOptions | None = None,
    ) -> None:
        selected = options or ClientOptions()
        origin = resolve_url(base_url)
        key, provider = _credentials(api_key, origin)
        self._transport = AsyncHttpTransport(origin, key, selected, credential_provider=provider)
        self.index = AsyncIndexClient(self._transport, True)
        self.swarms = AsyncSwarmsClient(self._transport)
        self.account = AsyncAccountClient(self._transport)
        self.tunnels = AsyncTunnelsClient(self._transport)
        self.menagerie = AsyncMenagerieClient(self._transport)
        self.repos = AsyncRepositories(self._transport)
        self.links = AsyncLinks(self._transport)
        self.artifacts = AsyncArtifacts(self._transport)
        self.forge = AsyncForgeClient(self._transport)

    @classmethod
    def anonymous(
        cls, *, base_url: str | None = None, options: ClientOptions | None = None
    ) -> AsyncSynthClient:
        """Construct a public client without reading ambient credentials or doing network IO.

        See: the public contract snapshots in contracts/"""
        selected = options or ClientOptions()
        client = cls.__new__(cls)
        client._transport = AsyncHttpTransport(resolve_url(base_url), None, selected)
        client.index = AsyncIndexClient(client._transport, False)
        client.swarms = AsyncSwarmsClient(client._transport)
        client.account = AsyncAccountClient(client._transport)
        client.tunnels = AsyncTunnelsClient(client._transport)
        client.menagerie = AsyncMenagerieClient(client._transport)
        client.repos = AsyncRepositories(client._transport)
        client.links = AsyncLinks(client._transport)
        client.artifacts = AsyncArtifacts(client._transport)
        client.forge = AsyncForgeClient(client._transport)
        return client

    async def capabilities(self, *, request: ReadOptions | None = None) -> Capabilities:
        """Read nonsecret deployment facts; these do not grant resource access.

        See: the public contract snapshots in contracts/"""
        return await self._transport.request(CAPABILITIES, Capabilities, options=request)

    @property
    def candidate_manifest_sha256(self) -> str | None:
        """Explicit nonproduction candidate pin; absent for ordinary clients."""
        return self._transport.candidate_manifest_sha256

    @property
    def base_url(self) -> str:
        """The validated API origin selected for this client."""
        return self._transport.base_url

    async def auth_context(self, *, request: ReadOptions | None = None) -> AuthContext:
        """Read verified identity and current grants without secrets or tenant overrides.

        See: the public contract snapshots in contracts/"""
        return await self._transport.request(AUTH_CONTEXT, AuthContext, options=request)

    async def resolve_mutation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> MutationResolution:
        """Reconcile known scoped mutation identity; unavailable is not proof of nonacceptance.

        See: the public contract snapshots in contracts/"""
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

        See: the public contract snapshots in contracts/"""
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncSynthClient:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.aclose()
        except (Exception, asyncio.CancelledError) as cause:
            handle_cleanup_failure(cause, exception, mutation=False, identity=None)
