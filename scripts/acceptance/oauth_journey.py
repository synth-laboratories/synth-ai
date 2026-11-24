"""Bound OAuth protocol reads and Console consent for J3 acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import parse_qs, urlparse

import httpx

from synth_ai import auth

if TYPE_CHECKING:
    from pydantic import JsonValue

    from synth_ai._auth_models import OAuthMetadata

PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource/mcp"
SAFE_ERROR_CODES = frozenset(
    {
        "access_denied",
        "invalid_client",
        "invalid_client_metadata",
        "invalid_grant",
        "invalid_redirect_uri",
        "invalid_request",
        "invalid_scope",
        "invalid_target",
        "oauth_grant_not_found",
        "server_error",
        "session_required",
        "temporarily_unavailable",
        "unsupported_grant_type",
        "unsupported_response_type",
    }
)


@dataclass(slots=True)
class OAuthJourneyError(Exception):
    stage: str
    code: str


def origin_of(value: str, stage: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not (
            parsed.scheme == "https"
            or (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"})
        )
    ):
        raise OAuthJourneyError(stage, "oauth_url_invalid")
    return f"{parsed.scheme}://{parsed.netloc}"


def response_object(
    response: httpx.Response, stage: str, expected: int, notes: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    notes[f"{stage}_status"] = response.status_code
    try:
        value = response.json()
    except ValueError:
        value = None
    if response.status_code != expected:
        code = None
        if isinstance(value, dict):
            code = value.get("error", value.get("code"))
        raise OAuthJourneyError(
            stage, code if isinstance(code, str) and code in SAFE_ERROR_CODES else "http_refused"
        )
    if not isinstance(value, dict):
        raise OAuthJourneyError(stage, "oauth_response_invalid")
    return cast("dict[str, JsonValue]", value)


def approve(
    oauth: str, session: str, authorization_url: str, resource: str, notes: dict[str, JsonValue]
) -> str:
    """Use the issuer's redirect to bind the Console origin without following it."""
    if origin_of(authorization_url, "authorize") != oauth:
        raise OAuthJourneyError("authorize", "oauth_origin_mismatch")
    parameters = parse_qs(urlparse(authorization_url).query)
    client_ids = parameters.get("client_id")
    if client_ids is None or len(client_ids) != 1 or parameters.get("resource") != [resource]:
        raise OAuthJourneyError("authorize", "oauth_request_identity_invalid")
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        redirect = client.get(authorization_url)
        notes["authorize_status"] = redirect.status_code
        if redirect.status_code != 303:
            response_object(redirect, "authorize", 303, notes)
        location = redirect.headers.get("location", "")
        console = origin_of(location, "authorize")
        link = urlparse(location)
        request = parse_qs(link.query).get("request")
        if link.path != "/mcp/authorize" or request is None or len(request) != 1:
            raise OAuthJourneyError("authorize", "oauth_consent_redirect_invalid")
        shown = response_object(
            client.get(oauth + "/mcp/oauth/request", params={"request": request[0]}),
            "consent_request",
            200,
            notes,
        )
        if shown.get("resource") != resource:
            raise OAuthJourneyError("consent_request", "oauth_resource_mismatch")
        if shown.get("client_id") != client_ids[0]:
            raise OAuthJourneyError("consent_request", "oauth_client_mismatch")
        decided = response_object(
            client.post(
                oauth + "/mcp/oauth/consent",
                json={"request": request[0], "approve": True},
                headers={"Origin": console, "X-Clerk-Session-Token": session},
            ),
            "consent",
            200,
            notes,
        )
        callback = decided.get("redirect_uri")
        if not isinstance(callback, str) or not callback:
            raise OAuthJourneyError("consent", "oauth_callback_missing")
        return callback


@dataclass(frozen=True, slots=True)
class HostedOAuth:
    origin: str
    resource: str
    resource_metadata: str
    metadata: OAuthMetadata

    @classmethod
    def discover(
        cls, client: httpx.Client, mcp_url: str, notes: dict[str, JsonValue]
    ) -> HostedOAuth:
        resource = mcp_url.rstrip("/")
        origin = origin_of(resource, "resource_metadata")
        if resource != origin + "/mcp":
            raise OAuthJourneyError("resource_metadata", "oauth_resource_url_invalid")
        path = origin + PROTECTED_RESOURCE_PATH
        protected = response_object(client.get(path), "resource_metadata", 200, notes)
        if protected.get("resource") != resource:
            raise OAuthJourneyError("resource_metadata", "oauth_resource_mismatch")
        if protected.get("authorization_servers") not in ([origin + "/"], [origin]):
            raise OAuthJourneyError("resource_metadata", "oauth_issuer_mismatch")
        metadata = auth.discover(client, origin)
        notes["issuer_discovered"] = True
        return cls(origin, resource, path, metadata)

    def endpoint(self, name: str) -> str:
        return auth.endpoint(self.metadata, name, self.origin)

    def cleanup_token(
        self,
        client: httpx.Client,
        endpoint: str,
        client_id: str,
        access: str,
        notes: dict[str, JsonValue],
    ) -> None:
        """Revoke only the token issued to this journey, preserving its original failure."""
        notes["fallback_revoke_verified"] = False
        try:
            response = client.post(
                endpoint,
                data={"token": access, "client_id": client_id, "resource": self.resource},
            )
            notes["fallback_revoke_status"] = response.status_code
            if response.status_code == 200:
                after = client.get(self.resource, headers={"Authorization": "Bearer " + access})
                notes["fallback_revoke_verified"] = self.revoked(after, notes)
        except httpx.HTTPError:
            notes["fallback_revoke_transport_error"] = True

    def revoked(self, response: httpx.Response, notes: dict[str, JsonValue]) -> bool:
        notes["revoked_transport_status"] = response.status_code
        challenge = response.headers.get("www-authenticate", "")
        scheme, _, parameters = challenge.partition(" ")
        resource_bound = f'resource_metadata="{self.resource_metadata}"' in {
            item.strip() for item in parameters.split(",")
        }
        if response.status_code != 401 or scheme.lower() != "bearer" or not resource_bound:
            return False
        try:
            document = response.json()
        except ValueError:
            return False
        error = document.get("error") if isinstance(document, dict) else None
        if (
            not isinstance(document, dict)
            or document.get("jsonrpc") != "2.0"
            or "id" not in document
            or document["id"] is not None
            or not isinstance(error, dict)
            or error.get("code") != -32600
        ):
            return False
        notes["revoked_transport_jsonrpc_code"] = -32600
        notes["revoked_transport_challenge_bound"] = True
        return True
