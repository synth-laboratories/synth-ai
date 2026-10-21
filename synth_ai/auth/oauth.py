"""Public-client PKCE and token custody. See backend oauth-families.md.

Stored legacy issuer credentials keep their issuer-bound transport; new CLI
connections use the Synth family issuer with Clerk-authenticated consent.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import NotRequired, TypedDict
from urllib.parse import urlencode, urlsplit

import httpx

from .storage import Credentials, CredentialStore


class OAuthMetadata(TypedDict):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str
    device_authorization_endpoint: NotRequired[str]
    grant_types_supported: NotRequired[tuple[str, ...]]
    authorization_response_iss_parameter_supported: NotRequired[bool]


@dataclass(slots=True)
class AuthRefusalError(Exception):
    code: str
    status: int = 401
    service: str = "clerk"
    route: str = "oauth"
    cause: str = "Authentication failed"


def trusted_endpoint(issuer: str, endpoint: str) -> str:
    source, target = urlsplit(issuer), urlsplit(endpoint)
    if (
        source.scheme != "https"
        or not source.hostname
        or source.username
        or source.password
        or source.path not in ("", "/")
        or source.query
        or source.fragment
        or source.port
        or target.scheme != "https"
        or target.netloc != source.netloc
        or target.username
        or target.password
        or target.fragment
        or target.query
    ):
        raise AuthRefusalError("oauth_endpoint_untrusted", 503)
    return endpoint


@dataclass(frozen=True, slots=True)
class OAuthConfig:
    issuer: str
    client_id: str
    resource: str
    scopes: tuple[str, ...] = (
        "offline_access",
        "user:org:read",
        "synth:read",
        "synth:action",
    )
    resource_exchange_url: str | None = None

    def __post_init__(self) -> None:
        trusted_endpoint(self.issuer, self.issuer)
        resource = urlsplit(self.resource)
        if (
            not self.client_id
            or resource.scheme != "https"
            or not resource.hostname
            or resource.username
            or resource.password
            or resource.query
            or resource.fragment
        ):
            raise AuthRefusalError("oauth_configuration_invalid", 503)
        if self.resource_exchange_url:
            exchange = urlsplit(self.resource_exchange_url)
            if (
                exchange.username
                or exchange.password
                or exchange.query
                or exchange.fragment
                or exchange.path != "/auth/oauth/resource-token"
                or (
                    exchange.scheme != "https"
                    and not (
                        exchange.scheme == "http"
                        and exchange.hostname in {"127.0.0.1", "localhost"}
                    )
                )
            ):
                raise AuthRefusalError("oauth_resource_exchange_untrusted", 503)


class OAuthClient:
    def __init__(
        self,
        config: OAuthConfig,
        http: httpx.Client,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config, self.http, self.clock, self.sleep = config, http, clock, sleep
        self._development_browser_token: str | None = None

    def _post_response(self, endpoint: str, data: dict[str, str]) -> httpx.Response:
        params = (
            {"__dev_session": self._development_browser_token}
            if self._development_browser_token
            else None
        )
        response = self.http.post(endpoint, data=data, params=params)
        # Clerk development FAPI requires its documented anonymous DevBrowser
        # bootstrap even for public OAuth endpoints. Production never uses it.
        if (
            response.status_code == 401
            and self._development_browser_token is None
            and (urlsplit(self.config.issuer).hostname or "").endswith(".clerk.accounts.dev")
        ):
            try:
                payload = response.json()
            except ValueError:
                return response
            errors = payload.get("errors", []) if isinstance(payload, dict) else []
            if any(
                isinstance(error, dict) and error.get("code") == "dev_browser_unauthenticated"
                for error in errors
            ):
                bootstrap = self.http.post(self.config.issuer + "/v1/dev_browser")
                if bootstrap.status_code == 429:
                    raise AuthRefusalError("oauth_rate_limited", 429)
                if not bootstrap.is_success:
                    raise AuthRefusalError("oauth_development_bootstrap_unavailable", 503)
                value = bootstrap.json()
                token = value.get("token") if isinstance(value, dict) else None
                if not isinstance(token, str) or not token:
                    raise AuthRefusalError("oauth_development_bootstrap_malformed", 503)
                self._development_browser_token = token
                return self.http.post(endpoint, data=data, params={"__dev_session": token})
        return response

    def metadata(self) -> OAuthMetadata:
        try:
            response = self.http.get(
                self.config.issuer.rstrip("/") + "/.well-known/oauth-authorization-server"
            )
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise AuthRefusalError("oauth_discovery_unavailable", 503) from error
        if not isinstance(value, dict) or value.get("issuer") != self.config.issuer:
            raise AuthRefusalError("oauth_issuer_mismatch", 503)
        for name in ("authorization_endpoint", "token_endpoint", "revocation_endpoint"):
            if not isinstance(value.get(name), str):
                raise AuthRefusalError("oauth_discovery_malformed", 503)
            trusted_endpoint(self.config.issuer, value[name])
        result: OAuthMetadata = {
            "issuer": self.config.issuer,
            "authorization_endpoint": value["authorization_endpoint"],
            "token_endpoint": value["token_endpoint"],
            "revocation_endpoint": value["revocation_endpoint"],
        }
        if isinstance(value.get("device_authorization_endpoint"), str):
            result["device_authorization_endpoint"] = trusted_endpoint(
                self.config.issuer, value["device_authorization_endpoint"]
            )
        if isinstance(value.get("grant_types_supported"), list) and all(
            isinstance(grant, str) for grant in value["grant_types_supported"]
        ):
            result["grant_types_supported"] = tuple(value["grant_types_supported"])
        if "authorization_response_iss_parameter_supported" in value:
            supported = value["authorization_response_iss_parameter_supported"]
            if type(supported) is not bool:
                raise AuthRefusalError("oauth_discovery_malformed", 503)
            result["authorization_response_iss_parameter_supported"] = supported
        return result

    def post(self, endpoint: str, data: dict[str, str]) -> dict[str, object]:
        trusted_endpoint(self.config.issuer, endpoint)
        try:
            response = self._post_response(endpoint, data)
            value = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise AuthRefusalError("oauth_authority_unavailable", 503) from error
        if not isinstance(value, dict):
            raise AuthRefusalError("oauth_response_malformed", 503)
        if response.status_code == 429:
            raise AuthRefusalError("oauth_rate_limited", 429)
        if response.status_code >= 500:
            raise AuthRefusalError("oauth_authority_unavailable", 503)
        if response.is_error or value.get("error"):
            # Only known protocol codes are exposed; provider text can contain secrets.
            code = value.get("error")
            allowed = {
                "authorization_pending",
                "slow_down",
                "access_denied",
                "expired_token",
                "invalid_grant",
                "invalid_client",
                "invalid_scope",
                "unsupported_grant_type",
            }
            raise AuthRefusalError(
                code if code in allowed else "oauth_request_refused", response.status_code
            )
        return value

    def credentials(
        self, value: dict[str, object], metadata: OAuthMetadata, previous: Credentials | None = None
    ) -> Credentials:
        token = value.get("access_token")
        token_type = value.get("token_type")
        expiry = value.get("expires_in")
        refresh = value.get("refresh_token", previous.refresh_token if previous else None)
        if (
            not isinstance(token, str)
            or not token
            or (not isinstance(token_type, str) or token_type.lower() != "bearer")
            or not isinstance(expiry, (int, float))
            or isinstance(expiry, bool)
            or not 0 < expiry <= 86400 * 30
            or (refresh is not None and not isinstance(refresh, str))
        ):
            raise AuthRefusalError("oauth_response_malformed", 503)
        scope = value.get("scope", " ".join(previous.scopes if previous else self.config.scopes))
        if not isinstance(scope, str) or not set(scope.split()) <= set(self.config.scopes):
            raise AuthRefusalError("oauth_scopes_malformed", 503)
        return Credentials(
            token,
            refresh,
            self.clock() + expiry,
            self.config.issuer,
            self.config.client_id,
            metadata["token_endpoint"],
            metadata["revocation_endpoint"],
            tuple(scope.split()),
            self.config.resource,
        )

    def pkce(self, metadata: OAuthMetadata, redirect_uri: str) -> tuple[str, str, str]:
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        url = (
            metadata["authorization_endpoint"]
            + "?"
            + urlencode(
                {
                    "response_type": "code",
                    "client_id": self.config.client_id,
                    "redirect_uri": redirect_uri,
                    "scope": " ".join(self.config.scopes),
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "resource": self.config.resource,
                }
            )
        )
        return url, state, verifier

    def exchange(
        self, metadata: OAuthMetadata, code: str, verifier: str, redirect_uri: str
    ) -> Credentials:
        return self.credentials(
            self.post(
                metadata["token_endpoint"],
                {
                    "grant_type": "authorization_code",
                    "client_id": self.config.client_id,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "resource": self.config.resource,
                },
            ),
            metadata,
        )

    def device(self, metadata: OAuthMetadata, notify) -> Credentials:
        if "urn:ietf:params:oauth:grant-type:token-exchange" not in metadata.get(
            "grant_types_supported", ()
        ):
            raise AuthRefusalError("device_resource_binding_unavailable", 503)
        endpoint = metadata.get("device_authorization_endpoint")
        if not isinstance(endpoint, str):
            raise AuthRefusalError("device_authorization_unavailable", 503)
        value = self.post(
            endpoint,
            {
                "client_id": self.config.client_id,
                "scope": " ".join(self.config.scopes),
                "resource": self.config.resource,
            },
        )
        device_code, user_code, verification_uri = (
            value.get(k) for k in ("device_code", "user_code", "verification_uri")
        )
        expires_in, interval = value.get("expires_in"), value.get("interval", 5)
        if (
            not isinstance(device_code, str)
            or not device_code
            or not isinstance(user_code, str)
            or not user_code
            or not isinstance(verification_uri, str)
            or not verification_uri
            or not isinstance(expires_in, (int, float))
            or isinstance(expires_in, bool)
            or not 0 < expires_in <= 3600
            or not isinstance(interval, (int, float))
            or isinstance(interval, bool)
            or not 0 < interval <= 60
        ):
            raise AuthRefusalError("device_response_malformed", 503)
        # The hosted verification UI may use a separate trusted Account Portal origin.
        verification = urlsplit(verification_uri)
        if (
            verification.scheme != "https"
            or not verification.hostname
            or verification.username
            or verification.password
        ):
            raise AuthRefusalError("device_verification_untrusted", 503)
        notify(verification_uri, user_code)
        deadline = self.clock() + expires_in
        while self.clock() + interval < deadline:
            self.sleep(interval)
            try:
                token = self.post(
                    metadata["token_endpoint"],
                    {
                        "client_id": self.config.client_id,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code,
                        "resource": self.config.resource,
                    },
                )
            except AuthRefusalError as error:
                if error.code == "slow_down":
                    interval += 5
                elif error.code != "authorization_pending":
                    raise
            else:
                return self.resource_exchange(self.credentials(token, metadata), metadata)
        raise AuthRefusalError("expired_token")

    def resource_exchange(self, source: Credentials, metadata: OAuthMetadata) -> Credentials:
        """Bind a device grant to the declared API using Clerk RFC 8693.

        The original opaque grant remains private for refresh and logout; it is
        never presented to Synth as a resource-bound credential.
        """
        scopes = tuple(
            scope
            for scope in source.scopes
            if scope not in {"openid", "email", "profile", "offline_access"}
        )
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": self.config.client_id,
            "subject_token": source.access_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "resource": self.config.resource,
            "scope": " ".join(scopes),
        }
        if self.config.resource_exchange_url:
            try:
                response = self.http.post(
                    self.config.resource_exchange_url,
                    headers={"Authorization": "Bearer " + source.access_token},
                )
                if response.status_code == 429:
                    raise AuthRefusalError("oauth_rate_limited", 429)
                if not response.is_success:
                    refusal = response.json()
                    detail = refusal.get("detail", {}) if isinstance(refusal, dict) else {}
                    cause = detail.get("cause") if isinstance(detail, dict) else None
                    known = {
                        f"Clerk exchange refused during {phase}": phase
                        for phase in (
                            "source_verification",
                            "organization_selection",
                            "resource_exchange",
                        )
                    }
                    code = (
                        "oauth_resource_exchange_" + known[cause] + "_refused"
                        if isinstance(cause, str) and cause in known
                        else "oauth_resource_exchange_refused"
                    )
                    raise AuthRefusalError(code, response.status_code)
                value = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise AuthRefusalError("oauth_resource_exchange_unavailable", 503) from error
            if not isinstance(value, dict):
                raise AuthRefusalError("oauth_exchange_response_malformed", 503)
        else:
            value = self.post(
                metadata["token_endpoint"],
                data,
            )
        if value.get("issued_token_type") != "urn:ietf:params:oauth:token-type:access_token":
            raise AuthRefusalError("oauth_exchange_response_malformed", 503)
        returned_scope = value.get("scope", " ".join(scopes))
        if not isinstance(returned_scope, str) or not set(returned_scope.split()) <= set(scopes):
            raise AuthRefusalError("oauth_exchange_scopes_expanded", 503)
        value = {**value, "scope": returned_scope}
        # A derived refresh token belongs to the confidential resource client.
        # Continue rotating the public client's source grant, then rebind it.
        value.pop("refresh_token", None)
        bound = self.credentials(value, metadata, source)
        return replace(
            bound,
            token_exchange_required=True,
            source_access_token=source.access_token,
            source_scopes=source.scopes,
            resource_exchange_url=self.config.resource_exchange_url,
        )

    def revoke(self, endpoint: str, token: str) -> None:
        trusted_endpoint(self.config.issuer, endpoint)
        try:
            response = self._post_response(
                endpoint, {"token": token, "client_id": self.config.client_id}
            )
        except (httpx.HTTPError, ValueError) as error:
            raise AuthRefusalError("oauth_authority_unavailable", 503) from error
        if response.status_code != 200:
            raise AuthRefusalError("oauth_revocation_refused", response.status_code)

    def refresh(self, credentials: Credentials) -> Credentials:
        if not credentials.refresh_token:
            raise AuthRefusalError("login_required")
        metadata = self.metadata()
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "refresh_token": credentials.refresh_token,
        }
        if not credentials.token_exchange_required:
            data["resource"] = self.config.resource
        refreshed = self.credentials(
            self.post(metadata["token_endpoint"], data),
            metadata,
            replace(credentials, scopes=credentials.source_scopes)
            if credentials.token_exchange_required
            else credentials,
        )
        return (
            self.resource_exchange(refreshed, metadata)
            if credentials.token_exchange_required
            else refreshed
        )


def stored_access_token(
    store: CredentialStore, *, api_origin: str, http: httpx.Client | None = None
) -> str | None:
    credential = store.load()
    if credential is None:
        return None
    if credential.api_origin != api_origin:
        raise AuthRefusalError("credential_origin_mismatch", 401, service="synth-ai")
    if credential.expires_at is None or credential.expires_at > time.time() + 30:
        return credential.access_token
    if not credential.issuer or not credential.client_id:
        raise AuthRefusalError("login_required")
    config = OAuthConfig(
        credential.issuer,
        credential.client_id,
        credential.resource or "https://api.usesynth.ai/mcp",
        credential.source_scopes or credential.scopes,
        credential.resource_exchange_url,
    )
    with (
        nullcontext(http)
        if http is not None
        else httpx.Client(timeout=15, follow_redirects=False, trust_env=True, auth=None)
    ) as connection:
        updated = OAuthClient(config, connection).refresh(credential)
    updated = replace(updated, api_origin=credential.api_origin)
    store.save(updated)
    return updated.access_token
