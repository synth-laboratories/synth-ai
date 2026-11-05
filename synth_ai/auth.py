"""Environment credentials and discovery-based OAuth PKCE; no embedded client identity."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .errors import SynthError

DEFAULT_BASE_URL = "https://api.usesynth.ai"
DEFAULT_OAUTH_BASE_URL = "https://mcp.usesynth.ai"


def token_path() -> Path:
    configured = os.getenv("SYNTH_AUTH_FILE")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".config" / "synth" / "credentials.json"
    )


def read_tokens() -> dict[str, Any]:
    path = token_path()
    if not path.exists():
        return {}
    if path.stat().st_mode & 0o077:
        raise SynthError("Credential file must have permissions 0600")
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, ValueError) as error:
        raise SynthError("Credential file cannot be read") from error


def save_tokens(records: dict[str, Any]) -> None:
    path = token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(records, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def oauth_base(base: str) -> str:
    """The OAuth origin for an API base. Production signs in on the MCP host; every
    other stack serves both from one origin. Tokens stay stored under the API base."""
    override = os.getenv("SYNTH_OAUTH_BASE_URL")
    if override:
        return override.rstrip("/")
    return DEFAULT_OAUTH_BASE_URL if base == DEFAULT_BASE_URL else base


def credentials(base_url: str | None = None, api_key: str | None = None) -> tuple[str, str | None]:
    base = (base_url or os.getenv("SYNTH_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.getenv("SYNTH_API_KEY")
    if key:
        return base, key
    record = read_tokens().get(base)
    if record is None:
        return base, None
    if not isinstance(record, dict):
        raise SynthError("Invalid stored credential")
    if float(record.get("expires_at", 0)) <= time.time() + 30:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            oauth = oauth_base(base)
            metadata = discover(client, oauth)
            response = client.post(
                endpoint(metadata, "token_endpoint", oauth),
                data={
                    "grant_type": "refresh_token",
                    "client_id": record["client_id"],
                    "refresh_token": record["refresh_token"],
                    "resource": oauth + "/mcp",
                },
            )
            if not response.is_success:
                raise SynthError("OAuth refresh refused; sign in again")
            token = response.json()
            record.update(token)
            record["expires_at"] = time.time() + float(token["expires_in"])
        records = read_tokens()
        records[base] = record
        save_tokens(records)
    return base, str(record["access_token"])


def endpoint(metadata: dict[str, Any], name: str, base: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str):
        raise SynthError("OAuth metadata missing " + name)
    url = urlparse(value)
    origin = urlparse(base)
    if (
        (url.scheme, url.netloc) != (origin.scheme, origin.netloc)
        or url.username
        or url.password
        or url.fragment
    ):
        raise SynthError("OAuth endpoint origin mismatch")
    return value


def discover(client: httpx.Client, base: str) -> dict[str, Any]:
    parsed = urlparse(base)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    ):
        raise SynthError("OAuth requires HTTPS outside loopback")
    response = client.get(base + "/.well-known/oauth-authorization-server")
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict) or str(value.get("issuer", "")).rstrip("/") != base:
        raise SynthError("OAuth issuer mismatch")
    if "S256" not in value.get("code_challenge_methods_supported", []):
        raise SynthError("OAuth server lacks PKCE S256")
    return value


def login(base_url: str | None = None, *, timeout: float = 180) -> None:
    base = (base_url or os.getenv("SYNTH_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    oauth = oauth_base(base)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    result: dict[str, str] = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            valid = (
                parsed.path == "/callback"
                and query.get("state") == [state]
                and query.get("iss") == [oauth + "/"]
            )
            if not valid:
                self.send_response(400)
                self.end_headers()
                return
            for key in ["code", "error"]:
                values = query.get(key, [])
                if len(values) == 1:
                    result[key] = values[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Sign-in complete. You can close this window.")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with (
        HTTPServer(("127.0.0.1", 0), Callback) as server,
        httpx.Client(timeout=30, follow_redirects=False) as client,
    ):
        redirect = f"http://127.0.0.1:{server.server_port}/callback"
        metadata = discover(client, oauth)
        response = client.post(
            endpoint(metadata, "registration_endpoint", oauth),
            json={
                "client_name": "Synth CLI",
                "redirect_uris": [redirect],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        response.raise_for_status()
        client_id = response.json()["client_id"]
        authorization = (
            endpoint(metadata, "authorization_endpoint", oauth)
            + "?"
            + urlencode(
                {
                    "client_id": client_id,
                    "redirect_uri": redirect,
                    "response_type": "code",
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": state,
                    "resource": oauth + "/mcp",
                    "scope": "offline_access user:org:read synth:read synth:action account:billing:read account:billing:write",
                }
            )
        )
        if not webbrowser.open(authorization):
            raise SynthError("Browser could not open OAuth authorization")
        deadline = time.monotonic() + timeout
        server.timeout = 1
        while not result and time.monotonic() < deadline:
            server.handle_request()
        if "code" not in result:
            raise SynthError("OAuth authorization denied or timed out")
        response = client.post(
            endpoint(metadata, "token_endpoint", oauth),
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": result["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect,
                "resource": oauth + "/mcp",
            },
        )
        response.raise_for_status()
        token = response.json()
        token["client_id"] = client_id
        token["expires_at"] = time.time() + float(token["expires_in"])
        records = read_tokens()
        records[base] = token
        save_tokens(records)


def logout(base_url: str | None = None) -> None:
    base = (base_url or os.getenv("SYNTH_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    records = read_tokens()
    record = records.get(base)
    if record:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            oauth = oauth_base(base)
            metadata = discover(client, oauth)
            response = client.post(
                endpoint(metadata, "revocation_endpoint", oauth),
                data={
                    "token": record.get("refresh_token", record["access_token"]),
                    "client_id": record["client_id"],
                },
            )
            response.raise_for_status()
        records.pop(base)
        save_tokens(records)


def status(base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or os.getenv("SYNTH_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    record = read_tokens().get(base, {})
    return {
        "base_url": base,
        "authenticated": bool(os.getenv("SYNTH_API_KEY") or record),
        "source": "environment" if os.getenv("SYNTH_API_KEY") else "oauth" if record else "none",
        "expires_at": record.get("expires_at"),
    }
