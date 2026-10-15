"""synth-ai login/logout/auth status. See backend auth-023.md, CLI login."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from synth_ai.auth.oauth import AuthRefusalError, OAuthClient, OAuthConfig, stored_access_token
from synth_ai.auth.storage import Credentials, default_store
from synth_ai.errors import SynthError


def private_login_file(text: str) -> Path:
    root = Path(os.environ.get("SYNTH_CONFIG_DIR", str(Path.home() / ".synth")))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / "login.txt"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)
    return path


def browser_login(
    client: OAuthClient,
    *,
    port: int = 8765,
    timeout_seconds: float = 300,
    open_browser=webbrowser.open,
) -> Credentials:
    metadata = client.metadata()
    outcome: dict[str, str] = {}
    state = ""

    class Callback(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(1.0)

        def log_message(self, *_):
            # HTTP request logs would expose the one-time authorization code.
            return

        def do_GET(self):
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            incoming = params.get("state", [])
            valid = (
                parsed.path == "/callback"
                and len(incoming) == 1
                and secrets.compare_digest(incoming[0], state)
            )
            if not valid:
                self.send_response(400)
                self.end_headers()
                return
            codes = params.get("code", [])
            if params.get("error") or len(codes) != 1 or not codes[0]:
                outcome["error"] = "oauth_callback_refused"
            else:
                outcome["code"] = codes[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.end_headers()
            self.wfile.write(b"Signed in" if "code" in outcome else b"Sign-in refused")

    try:
        listener = HTTPServer(("127.0.0.1", port), Callback)
    except OSError as error:
        raise AuthRefusalError(
            "oauth_callback_unavailable", 503, cause="Loopback listener unavailable"
        ) from error
    with listener as server:
        redirect = f"http://127.0.0.1:{server.server_port}/callback"
        url, state, verifier = client.pkce(metadata, redirect)
        path = private_login_file(url + "\n")
        print(path)
        open_browser(url)
        deadline = time.monotonic() + timeout_seconds
        while not outcome and time.monotonic() < deadline:
            server.timeout = min(1.0, max(0.01, deadline - time.monotonic()))
            server.handle_request()
        if "code" not in outcome:
            raise AuthRefusalError(outcome.get("error", "oauth_callback_timeout"))
        return client.exchange(metadata, outcome["code"], verifier, redirect)


def oauth_config() -> OAuthConfig:
    issuer = os.environ.get("SYNTH_OAUTH_ISSUER", "https://clerk.usesynth.ai")
    client_id = os.environ.get("SYNTH_OAUTH_CLIENT_ID", "")
    if not client_id:
        raise AuthRefusalError("oauth_client_not_configured", 503)
    return OAuthConfig(
        issuer,
        client_id,
        os.environ.get("SYNTH_OAUTH_RESOURCE", "https://api.usesynth.ai/mcp"),
        resource_exchange_url=os.environ.get(
            "SYNTH_OAUTH_RESOURCE_EXCHANGE_URL",
            os.environ.get("SYNTH_API_URL", "https://api.usesynth.ai").rstrip("/")
            + "/auth/oauth/resource-token",
        ),
    )


def run(args: argparse.Namespace, *, store=None) -> int:
    store = store or default_store()
    if args.command == "login":
        if args.with_token:
            value = sys.stdin.read(8193).strip()
            if len(value) > 8192 or not value.startswith("sk_synth_"):
                raise AuthRefusalError("synth_key_required")
            store.save(Credentials(value))
        else:
            with httpx.Client(timeout=15, follow_redirects=False) as http:
                client = OAuthClient(oauth_config(), http)
                if args.device_auth:
                    credential = client.device(
                        client.metadata(),
                        lambda url, code: print(private_login_file(url + "\n" + code + "\n")),
                    )
                else:
                    credential = browser_login(
                        client,
                        port=args.callback_port,
                        open_browser=(lambda _: False) if args.no_browser else webbrowser.open,
                    )
                store.save(credential)
        print("Signed in")
    elif args.command == "logout":
        credentials = store.load()
        if credentials and credentials.issuer and credentials.client_id:
            with httpx.Client(timeout=15, follow_redirects=False) as http:
                client = OAuthClient(
                    OAuthConfig(
                        credentials.issuer,
                        credentials.client_id,
                        credentials.resource or "https://api.usesynth.ai/mcp",
                        credentials.source_scopes or credentials.scopes,
                        credentials.resource_exchange_url,
                    ),
                    http,
                )
                endpoint = client.metadata()["revocation_endpoint"]
                for token in (
                    credentials.refresh_token,
                    credentials.source_access_token,
                    credentials.access_token,
                ):
                    if token:
                        client.revoke(endpoint, token)
        store.clear()
        print("Signed out")
    else:
        token = os.environ.get("SYNTH_API_KEY") or stored_access_token(store)
        if not token:
            print(json.dumps({"state": "signed_out"}) if args.json else "Signed out")
            return 1
        from synth_ai import SynthClient
        from synth_ai.options import ClientOptions

        try:
            options = ClientOptions(candidate_manifest_sha256=args.candidate_manifest_sha256)
        except ValueError as error:
            raise AuthRefusalError(
                "candidate_manifest_invalid", 400, service="synth-ai", route="auth.status"
            ) from error
        with SynthClient(api_key=token, options=options) as client:
            principal = client.auth_context()
        status = {
            "state": "signed_in",
            "source": "environment" if os.environ.get("SYNTH_API_KEY") else "stored",
            "credential_kind": principal.credential_kind,
            "subject_id": principal.subject_id,
            "tenant_id": principal.tenant_id,
            "scopes": list(principal.scopes),
        }
        print(
            json.dumps(status)
            if args.json
            else f"Signed in · {principal.credential_kind} · {principal.tenant_id}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="synth-ai")
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login")
    modes = login.add_mutually_exclusive_group()
    modes.add_argument("--device-auth", action="store_true")
    modes.add_argument("--with-token", action="store_true")
    login.add_argument("--no-browser", action="store_true")
    login.add_argument("--callback-port", type=int, default=8765)
    commands.add_parser("logout")
    auth = commands.add_parser("auth")
    status = auth.add_subparsers(dest="auth_command", required=True).add_parser("status")
    status.add_argument("--json", action="store_true")
    status.add_argument("--candidate-manifest-sha256")
    try:
        return run(parser.parse_args())
    except AuthRefusalError as error:
        print(
            json.dumps(
                {
                    "code": error.code,
                    "service": error.service,
                    "route": error.route,
                    "status": error.status,
                    "cause": error.cause,
                }
            ),
            file=sys.stderr,
        )
        return 1
    except SynthError as error:
        print(
            json.dumps(
                {
                    "code": "auth_status_refused",
                    "service": "backend",
                    "route": "auth.status",
                    "status": error.status or 503,
                    "cause": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "code": "credential_storage_unavailable",
                    "service": "synth-ai",
                    "route": "auth",
                    "status": 503,
                    "cause": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
