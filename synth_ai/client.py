"""Public client composition: Swarms, account and billing (``PUBLIC_PRODUCTS``)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import TracebackType
from typing import Any

import httpx

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
from synth_ai.public.client import Account, Billing, PublicTransport, Swarms
from synth_ai.sdk.account.client import AccountClient, AsyncAccountClient
from synth_ai.sdk.core.artifacts import Artifacts, AsyncArtifacts
from synth_ai.sdk.core.links import AsyncLinks, Links
from synth_ai.sdk.core.repos import AsyncRepositories, Repositories
from synth_ai.sdk.forge.client import AsyncForgeClient, ForgeClient
from synth_ai.sdk.index.client import AsyncIndexClient, IndexClient
from synth_ai.sdk.menagerie.client import AsyncMenagerieClient, MenagerieClient
from synth_ai.sdk.mnemosyne import AsyncMnemosyneClient, MnemosyneClient
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
        else environment_value("SYNTH_API_BASE_URL", "SYNTH_API_URL", "SYNTH_BACKEND_URL")
    )
    return validate_base_url(DEFAULT_BASE_URL if selected is None else selected)


def resolve_project(project_id: str | None) -> str | None:
    """Default Swarms project: explicit argument, else the SYNTH_PROJECT_ID config value."""
    selected = project_id if project_id is not None else environment_value("SYNTH_PROJECT_ID")
    return selected.strip() if selected and selected.strip() else None


# Hidden products: importable and constructible from code, absent from the public client.
_ASYNC_HIDDEN_FACTORIES: dict[str, Callable[[AsyncHttpTransport], object]] = {
    "index": lambda transport: AsyncIndexClient(transport, True),
    "forge": AsyncForgeClient,
    "mnemosyne": AsyncMnemosyneClient,
    "menagerie": AsyncMenagerieClient,
    "tunnels": AsyncTunnelsClient,
    "swarms_legacy": AsyncSwarmsClient,
    "account_legacy": AsyncAccountClient,
}

_HIDDEN_FACTORIES: dict[str, Callable[[HttpTransport], object]] = {
    "index": lambda transport: IndexClient(transport, True),
    "forge": ForgeClient,
    "mnemosyne": MnemosyneClient,
    "menagerie": MenagerieClient,
    "tunnels": TunnelsClient,
    "swarms_legacy": SwarmsClient,
    "account_legacy": AccountClient,
}


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
    """An owned SDK lifetime for the public Swarms, account and billing surface."""

    _transport: HttpTransport
    _public: PublicTransport
    swarms: Swarms
    account: Account
    billing: Billing
    repos: Repositories
    links: Links
    artifacts: Artifacts

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: ClientOptions | None = None,
        session_token: str | None = None,
        project_id: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        selected = options or ClientOptions()
        if selected.async_http_client is not None:
            raise ValueError(
                "async_http_client requires AsyncSynthClient; use http_client with SynthClient"
            )
        origin = resolve_url(base_url)
        key, provider = _credentials(api_key, origin) if session_token is None else (api_key, None)
        self._transport = HttpTransport(origin, key, selected, credential_provider=provider)
        self._public = PublicTransport(
            origin,
            api_key=key,
            credential_provider=provider,
            session_token=session_token,
            http_client=http_client or selected.http_client,
        )
        self.swarms = Swarms(self._public, project_id=resolve_project(project_id))
        self.account = Account(self._public)
        self.billing = Billing(self._public)
        self.repos = Repositories(self._transport)
        self.links = Links(self._transport)
        self.artifacts = Artifacts(self._transport)
        self._hidden: dict[str, object] = {}

    def hidden(self, product: str) -> object:
        """Construct a client for a product outside ``PUBLIC_PRODUCTS``.

        Hidden products keep their modules; they are not part of the public surface."""
        if product not in _HIDDEN_FACTORIES:
            raise KeyError(product)
        if product not in self._hidden:
            self._hidden[product] = _HIDDEN_FACTORIES[product](self._transport)
        return self._hidden[product]

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
        self._public.close()
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
    """Native async client for repos, links and artifacts; construction does no IO.

    The Swarms v1 run, account and billing calls are on :class:`SynthClient`."""

    _transport: AsyncHttpTransport
    swarms: AsyncSwarmsClient
    account: AsyncAccountClient
    repos: AsyncRepositories
    links: AsyncLinks
    artifacts: AsyncArtifacts

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
        self.swarms = AsyncSwarmsClient(self._transport)
        self.account = AsyncAccountClient(self._transport)
        self.repos = AsyncRepositories(self._transport)
        self.links = AsyncLinks(self._transport)
        self.artifacts = AsyncArtifacts(self._transport)
        self._hidden: dict[str, Any] = {}

    def hidden(self, product: str) -> Any:
        """Construct an async client for a product outside ``PUBLIC_PRODUCTS``."""
        if product not in _ASYNC_HIDDEN_FACTORIES:
            raise KeyError(product)
        if product not in self._hidden:
            self._hidden[product] = _ASYNC_HIDDEN_FACTORIES[product](self._transport)
        return self._hidden[product]

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
