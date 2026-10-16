"""Hosted resource-server adapter with explicit backend-owned verification.

See SDK_0230_MCP.md. Implement AuthorizationBackend inside the backend using its
shared principal/services. No admin-key substitution or OAuth token passthrough.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware, get_access_token
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl, Field, StrictInt
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from synth_ai.contracts.registry import OPERATIONS_BY_ID
from synth_ai.core.capabilities import require_operation, supports_operation
from synth_ai.errors import (
    AuthenticationError,
    PermissionDeniedError,
    SynthError,
    UnsupportedOperationError,
)
from synth_ai.types.account import AccountEntitlements
from synth_ai.types.base import Identifier, Model, OAuthClientIdentifier
from synth_ai.types.capabilities import Capabilities

from .catalog import TOOLS, ToolSpec
from .contracts import JSON_OBJECT, JsonObject
from .policy import POLICY_ACTION_TOOLS, visible_policy_tools
from .server import create_server

# Clerk OAuth consent scopes and per-operation resource grants are different
# authorities. Advertise the issuer's consent scopes; tool gates stay granular.
AUTHORIZATION_SCOPES = ("user:org:read", "synth:read")


class AuthorizationScopeChallengeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_challenge(message):
            if message["type"] == "http.response.start" and message["status"] == 401:
                message = {
                    **message,
                    "headers": [
                        (name, value + b', scope="user:org:read synth:read"')
                        if name.lower() == b"www-authenticate"
                        and value.startswith(b"Bearer ")
                        and b'scope="' not in value
                        else (name, value)
                        for name, value in message.get("headers", [])
                    ],
                }
            await send(message)

        await self.app(scope, receive, send_challenge)


class AuthorizationUnavailableError(Exception):
    """Verifier outage; never reinterpret the credential or degrade to anonymous."""


class AuthorizationAvailabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._app(scope, receive, send)
        except AuthorizationUnavailableError:
            response = JSONResponse(
                {"error": "authorization_unavailable"},
                status_code=503,
                headers={"Retry-After": "5"},
            )
            await response(scope, receive, send)


class VerifiedPrincipal(Model):
    """Server-produced identity. Never accepted as a tool argument or client header."""

    subject: Identifier
    organization_id: Identifier
    client_id: OAuthClientIdentifier
    credential_kind: Literal["oauth", "api_key"]
    resource: str = Field(min_length=1, max_length=2048)
    issuer: str = Field(min_length=1, max_length=2048)
    expires_at: StrictInt = Field(gt=0)
    scopes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,127}$")], ...] = Field(
        max_length=64
    )
    allowed_tools: tuple[Identifier, ...] = Field(max_length=128)
    authorization_handle: str = Field(min_length=1, max_length=4096, repr=False)


class AuthorizationBackend(Protocol):
    async def verify(self, credential: str, *, resource: str) -> VerifiedPrincipal | None:
        """Recognize credential by authority, not caller hint. Validate issuer/audience,
        expiry/revocation, organization membership, scopes and tool grants. Return
        None only for invalid credentials; outages must raise (fail closed).
        """
        ...

    async def invoke(
        self, principal: VerifiedPrincipal, tool: str, arguments: JsonObject
    ) -> JsonObject:
        """Recheck live grants/resource rights/entitlements/budget and call shared services.
        Treat authorization_handle as server-only evidence, not unconditional authority.
        Never execute by forwarding a browser token to a different resource.
        """
        ...


class PrincipalToken(AccessToken):
    principal: VerifiedPrincipal = Field(repr=False)


class PrincipalVerifier:
    def __init__(
        self,
        backend: AuthorizationBackend,
        resource: str,
        issuer: str,
        *,
        legacy_issuers: tuple[str, ...] = (),
    ) -> None:
        self._backend = backend
        self._resource = resource
        self._issuers = frozenset((issuer, *legacy_issuers))

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            async with asyncio.timeout(10):
                principal = await self._backend.verify(token, resource=self._resource)
        except Exception as cause:
            raise AuthorizationUnavailableError() from cause
        if (
            principal is None
            or principal.resource != self._resource
            or principal.expires_at <= time.time()
        ):
            return None
        if principal.credential_kind == "oauth" and principal.issuer not in self._issuers:
            return None
        # A connection never stores the incoming bearer token in its execution context.
        return PrincipalToken(
            token="redacted",
            client_id=principal.client_id,
            subject=principal.subject,
            scopes=list(principal.scopes),
            expires_at=principal.expires_at,
            resource=principal.resource,
            principal=principal,
        )


class HostedExecutor:
    def __init__(
        self,
        backend: AuthorizationBackend,
        enabled_tools: frozenset[str],
        capability_provider: Callable[[], Capabilities] | None,
        candidate_manifest_sha256: str | None,
    ) -> None:
        self._backend = backend
        self._enabled = enabled_tools
        self._capability_provider = capability_provider
        self._candidate_manifest_sha256 = candidate_manifest_sha256

    def _capabilities(self) -> Capabilities:
        if self._capability_provider is None:
            raise UnsupportedOperationError(
                "Backend capability identity is unavailable",
                code="capabilities_unavailable",
                status=503,
            )
        return self._capability_provider()

    def principal(self) -> VerifiedPrincipal:
        token = get_access_token()
        if not isinstance(token, PrincipalToken) or token.principal.expires_at <= time.time():
            raise AuthenticationError("Verified authorization is unavailable or expired")
        return token.principal

    async def available_tools(self) -> frozenset[str]:
        principal = self.principal()
        receipt = self._capabilities()
        return frozenset(
            tool.name
            for tool in TOOLS
            if tool.name in self._enabled
            and supports_operation(
                receipt,
                OPERATIONS_BY_ID[tool.operation],
                surface="mcp",
                candidate_manifest_sha256=self._candidate_manifest_sha256,
            )
            and tool.name in principal.allowed_tools
            and (tool.scope is None or tool.scope in principal.scopes)
        )

    async def unavailable_cause(self, tool: ToolSpec) -> str:
        """Name the first gate that hides ``tool``; a closed, secret-free vocabulary."""
        principal = self.principal()
        if tool.name not in self._enabled:
            return "tool_not_enabled"
        try:
            require_operation(
                self._capabilities(),
                OPERATIONS_BY_ID[tool.operation],
                surface="mcp",
                candidate_manifest_sha256=self._candidate_manifest_sha256,
            )
        except SynthError as error:
            return f"capability:{error.code or type(error).__name__}"
        if tool.name not in principal.allowed_tools:
            return "principal_not_authorized"
        if tool.scope is not None and tool.scope not in principal.scopes:
            return f"scope_missing:{tool.scope}"
        return "unknown"

    async def visible_tools(self) -> frozenset[str]:
        available = await self.available_tools()
        if not available & POLICY_ACTION_TOOLS:
            return available
        if "account_entitlements" not in available:
            return visible_policy_tools(available, None)
        principal = self.principal()
        try:
            value = await self._backend.invoke(principal, "account_entitlements", {})
            policy = AccountEntitlements.model_validate_json(JSON_OBJECT.dump_json(value))
        except AuthenticationError:
            raise
        except (SynthError, ValueError):
            return visible_policy_tools(available, None)
        if policy.tenant_id != principal.organization_id:
            raise AuthenticationError("Account policy principal changed")
        return visible_policy_tools(available, policy)

    async def invoke(self, tool: ToolSpec, arguments: JsonObject) -> JsonObject:
        principal = self.principal()
        if tool.name not in self._enabled or tool.name not in principal.allowed_tools:
            raise PermissionDeniedError("Tool is outside the active authorization")
        require_operation(
            self._capabilities(),
            OPERATIONS_BY_ID[tool.operation],
            surface="mcp",
            candidate_manifest_sha256=self._candidate_manifest_sha256,
        )
        scope = tool.scope_for(arguments)
        if scope is not None and scope not in principal.scopes:
            raise PermissionDeniedError(
                "Original mutation scope is required", code="insufficient_scope"
            )
        return await self._backend.invoke(principal, tool.name, arguments)


def create_http_app(
    backend: AuthorizationBackend,
    *,
    resource_url: str,
    issuer_url: str,
    enabled_tools: frozenset[str],
    capability_provider: Callable[[], Capabilities] | None = None,
    candidate_manifest_sha256: str | None = None,
    allowed_origins: tuple[str, ...] = (),
    legacy_issuer_urls: tuple[str, ...] = (),
) -> Starlette:
    """Create one stateless protected MCP endpoint; backend supplies live authorization.

    Deployment must pass the qualified tool allowlist explicitly. No socket is
    opened here, and no OAuth authorization server or registration flow is fabricated.
    """
    for value in (resource_url, issuer_url, *legacy_issuer_urls):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Hosted resource and issuer must be explicit HTTPS URLs")
    known = {tool.name for tool in TOOLS}
    if not enabled_tools <= known:
        raise ValueError("Hosted allowlist contains an unknown tool")
    resource = AnyHttpUrl(resource_url)
    issuer = AnyHttpUrl(issuer_url)
    executor = HostedExecutor(
        backend, enabled_tools, capability_provider, candidate_manifest_sha256
    )
    server = create_server(executor)
    host = urlsplit(resource_url).netloc
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        max_request_body_size=128 * 1024,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[host],
            allowed_origins=list(allowed_origins),
        ),
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        async with manager.run():
            yield

    routes = create_protected_resource_routes(
        resource,
        [issuer],
        scopes_supported=list(AUTHORIZATION_SCOPES),
        resource_name="Synth Index and Swarms",
    )
    routes.append(
        Route(
            urlsplit(resource_url).path or "/",
            endpoint=RequireAuthMiddleware(
                manager.handle_request,
                required_scopes=[],
                resource_metadata_url=build_resource_metadata_url(resource),
            ),
        )
    )
    return Starlette(
        routes=routes,
        lifespan=lifespan,
        # Starlette's permissive factory protocol is wider than its public
        # ASGIApp alias; these constructors implement the actual ASGI contract.
        middleware=[
            Middleware(AuthorizationAvailabilityMiddleware),
            Middleware(AuthorizationScopeChallengeMiddleware),
            Middleware(
                AuthenticationMiddleware,
                backend=BearerAuthBackend(
                    PrincipalVerifier(
                        backend, resource_url, issuer_url, legacy_issuers=legacy_issuer_urls
                    ),
                    resource_server_url=resource,
                ),
            ),
            Middleware(AuthContextMiddleware),
        ],
    )
