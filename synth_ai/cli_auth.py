"""Customer login, logout and verified status. See SDK_0230_MCP.md."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
import time
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from synth_ai.auth.oauth import AuthRefusalError, OAuthClient, OAuthConfig, stored_access_token
from synth_ai.auth.storage import (
    Credentials,
    CredentialStore,
    default_store,
    require_private_file_support,
)
from synth_ai.cli_account import add_account_commands, run_account
from synth_ai.cli_swarms import add_swarms_commands, run_swarms
from synth_ai.client import SynthClient, resolve_key, resolve_url
from synth_ai.core.environment import environment_value
from synth_ai.errors import AuthenticationError, SynthError
from synth_ai.options import ClientOptions
from synth_ai.types.account_cloud import AccountContext
from synth_ai.types.auth import AuthContext

# Commands whose refusals are reported against their own route, not auth.
OPERATION_ROUTES = frozenset({"account", "swarms"})


def private_login_file(text: str) -> Path:
    """Keep one-time authorization URLs out of terminal history and shared logs."""
    require_private_file_support()
    root = Path(os.environ.get("SYNTH_CONFIG_DIR", str(Path.home() / ".synth")))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077 or root.stat().st_uid != os.getuid():
        raise PermissionError("Credential directory must be private (0700)")
    path = root / "login.txt"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            raise PermissionError("Login file must be private (0600)")
        os.ftruncate(stream.fileno(), 0)
        stream.write(text)
    return path


def browser_login(
    client: OAuthClient,
    *,
    port: int = 8765,
    timeout_seconds: float = 300,
    open_browser=webbrowser.open,
) -> Credentials:
    """Use a loopback listener and accept only the exact S256 flow state."""
    metadata = client.metadata()
    outcome: dict[str, str] = {}
    state = ""

    class Callback(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(1.0)

        def log_message(self, format: str, *args: object) -> None:
            # Request paths contain one-time authorization codes.
            return

        def do_GET(self):
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            incoming = params.get("state", [])
            if (
                parsed.path != "/callback"
                or len(incoming) != 1
                or not secrets.compare_digest(incoming[0], state)
            ):
                self.send_response(400)
                self.end_headers()
                return
            codes = params.get("code", [])
            issuers = params.get("iss", [])
            issuer_required = metadata.get("authorization_response_iss_parameter_supported", False)
            if (
                params.get("error")
                or len(codes) != 1
                or not codes[0]
                or ((issuer_required or issuers) and issuers != [client.config.issuer])
            ):
                outcome["error"] = "oauth_callback_refused"
            else:
                outcome["code"] = codes[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.end_headers()
            self.wfile.write(b"Authorization received" if "code" in outcome else b"Sign-in refused")

    try:
        listener = HTTPServer(("127.0.0.1", port), Callback)
    except OSError as error:
        raise AuthRefusalError("oauth_callback_unavailable", 503) from error
    with listener as server:
        redirect = f"http://127.0.0.1:{server.server_port}/callback"
        url, state, verifier = client.pkce(metadata, redirect)
        path = private_login_file(url + "\n")
        try:
            print(path)
            open_browser(url)
            deadline = time.monotonic() + timeout_seconds
            while not outcome and time.monotonic() < deadline:
                server.timeout = min(1.0, max(0.01, deadline - time.monotonic()))
                server.handle_request()
            if "code" not in outcome:
                raise AuthRefusalError(outcome.get("error", "oauth_callback_timeout"))
            return client.exchange(metadata, outcome["code"], verifier, redirect)
        finally:
            path.unlink(missing_ok=True)


def oauth_config(api_origin: str) -> OAuthConfig:
    client_id = os.environ.get("SYNTH_OAUTH_CLIENT_ID", "synth-ai-cli")
    if not client_id:
        raise AuthRefusalError("oauth_client_not_configured", 503)
    configured = OAuthConfig(
        os.environ.get("SYNTH_OAUTH_ISSUER", api_origin.rstrip("/") + "/"),
        client_id,
        os.environ.get("SYNTH_OAUTH_RESOURCE", api_origin + "/mcp"),
    )
    return replace(
        configured,
        scopes=(*configured.scopes, "account:read", "account:usage:read", "account:billing:read"),
    )


def verify_principal(
    token: str, origin: str, args: argparse.Namespace
) -> AuthContext | AccountContext:
    """Validate the selected backend's real principal before accepting a login."""
    try:
        options = ClientOptions(candidate_manifest_sha256=args.candidate_manifest_sha256)
    except ValueError as error:
        raise AuthRefusalError("candidate_manifest_invalid", 400, service="synth-ai") from error
    with SynthClient(api_key=token, base_url=origin, options=options) as client:
        # An explicit legacy candidate pin keeps its negotiated auth primitive.
        # Ordinary account setup never waits on workload capability discovery.
        if args.candidate_manifest_sha256 is not None:
            return client.auth_context()
        return client.account.context()


def run(args: argparse.Namespace, *, store: CredentialStore | None = None) -> int:
    if args.command == "account":
        return run_account(args)
    if args.command == "swarms":
        return run_swarms(args)
    if args.command == "login" and args.device_auth:
        raise AuthRefusalError(
            "device_authorization_unavailable",
            503,
            service="synth-ai",
            cause="Device login is not available in this release; use browser login or --with-token",
        )
    if (
        args.command == "auth"
        and environment_value("SYNTH_API_KEY", "SYNTH_USER_KEY", "SYNTH_BACKEND_API_KEY")
        is not None
    ):
        return _run_locked(args, None)
    selected = store if store is not None else default_store()
    with selected.locked():
        return _run_locked(args, selected)


def _run_locked(args: argparse.Namespace, store: CredentialStore | None) -> int:
    origin = resolve_url(None)
    if args.command == "login":
        if store is None:
            raise ValueError("Credential store is required")
        if args.with_token:
            value = sys.stdin.read(8193).strip()
            if (
                len(value) > 8192
                or not value.startswith("sk_synth_")
                or any(c.isspace() for c in value)
            ):
                raise AuthRefusalError("synth_key_required")
            credential = Credentials(value)
        else:
            with httpx.Client(
                timeout=15, follow_redirects=False, trust_env=True, auth=None
            ) as http:
                client = OAuthClient(oauth_config(origin), http)
                credential = browser_login(
                    client,
                    port=args.callback_port,
                    open_browser=(lambda _: False) if args.no_browser else webbrowser.open,
                )
        principal = verify_principal(credential.access_token, origin, args)
        expected_kind = "api_key" if args.with_token else "oauth"
        if principal.credential_kind != expected_kind:
            raise AuthRefusalError("credential_kind_mismatch", service="synth-ai")
        store.save(replace(credential, api_origin=origin))
        print("Signed in")
    elif args.command == "logout":
        if store is None:
            raise ValueError("Credential store is required")
        credential = store.load()
        if credential and credential.issuer and credential.client_id:
            with httpx.Client(
                timeout=15, follow_redirects=False, trust_env=True, auth=None
            ) as http:
                client = OAuthClient(
                    OAuthConfig(
                        credential.issuer,
                        credential.client_id,
                        credential.resource or "https://api.usesynth.ai/mcp",
                        credential.source_scopes or credential.scopes,
                        credential.resource_exchange_url,
                    ),
                    http,
                )
                endpoint = client.metadata()["revocation_endpoint"]
                for token in dict.fromkeys(
                    (
                        credential.refresh_token,
                        credential.source_access_token,
                        credential.access_token,
                    )
                ):
                    if token:
                        client.revoke(endpoint, token)
        store.clear()
        print("Stored credentials cleared")
    else:
        value = environment_value("SYNTH_API_KEY", "SYNTH_USER_KEY", "SYNTH_BACKEND_API_KEY")
        if value is not None:
            token = resolve_key(value)
        elif store is not None:
            token = stored_access_token(store, api_origin=origin)
        else:
            raise ValueError("Credential store is required")
        if not token:
            print(json.dumps({"state": "signed_out"}) if args.json else "Signed out")
            return 1
        principal = verify_principal(token, origin, args)
        if isinstance(principal, AccountContext):
            subject_id, tenant_id, observed_at = (
                principal.user_id,
                principal.org_id,
                principal.observed_at,
            )
        else:
            subject_id, tenant_id, observed_at = (
                principal.subject_id,
                principal.tenant_id,
                principal.metadata.observed_at,
            )
        status = {
            "state": "signed_in",
            "source": "environment" if value is not None else "stored",
            "credential_kind": principal.credential_kind,
            "subject_id": subject_id,
            "tenant_id": tenant_id,
            "scopes": list(principal.scopes),
            "credential_id": principal.credential_id,
            "api_origin": origin,
            "oauth_client_id": principal.oauth_client_id,
            "observed_at": observed_at.isoformat(),
        }
        print(
            json.dumps(status)
            if args.json
            else f"Signed in · {principal.credential_kind} · {tenant_id}"
        )
    return 0


def parser() -> argparse.ArgumentParser:
    from synth_ai import __version__

    root = argparse.ArgumentParser(prog="synth-ai")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login")
    modes = login.add_mutually_exclusive_group()
    modes.add_argument("--device-auth", action="store_true")
    modes.add_argument("--with-token", action="store_true")
    login.add_argument("--no-browser", action="store_true")
    login.add_argument("--callback-port", type=int, default=8765)
    login.add_argument("--candidate-manifest-sha256")
    commands.add_parser("logout")
    status = (
        commands.add_parser("auth")
        .add_subparsers(dest="auth_command", required=True)
        .add_parser("status")
    )
    status.add_argument("--json", action="store_true")
    status.add_argument("--candidate-manifest-sha256")
    add_account_commands(commands)
    add_swarms_commands(commands)
    return root


def main() -> int:
    args = None
    try:
        args = parser().parse_args()
        return run(args)
    except AuthRefusalError as error:
        refusal = {
            "code": error.code,
            "service": error.service,
            "route": error.route,
            "status": error.status,
            "cause": error.cause,
        }
    except SynthError as error:
        refusal = {
            "code": error.code
            or (
                "authentication_required"
                if isinstance(error, AuthenticationError)
                else "operation_refused"
                if args is not None and args.command in OPERATION_ROUTES
                else "authentication_refused"
            ),
            "service": "backend",
            "route": args.command
            if args is not None and args.command in OPERATION_ROUTES
            else "auth",
            "status": error.status or (401 if isinstance(error, AuthenticationError) else 503),
            "cause": type(error).__name__,
        }
        if args is not None and args.command in OPERATION_ROUTES:
            if error.retry_after_seconds is not None:
                refusal["retry_after_seconds"] = error.retry_after_seconds
            if error.request_id:
                refusal["request_id"] = error.request_id
    except ValueError as error:
        if args is not None and args.command in OPERATION_ROUTES:
            print(
                json.dumps(
                    {
                        "code": f"{args.command}_request_invalid",
                        "service": "synth-ai",
                        "route": args.command,
                        "status": 400,
                        "cause": type(error).__name__,
                    }
                ),
                file=sys.stderr,
            )
            return 2
        refusal = {
            "code": "credential_storage_or_configuration_unavailable",
            "service": "synth-ai",
            "route": "auth",
            "status": 503,
            "cause": type(error).__name__,
        }
    except OSError as error:
        refusal = {
            "code": "credential_storage_or_configuration_unavailable",
            "service": "synth-ai",
            "route": "auth",
            "status": 503,
            "cause": type(error).__name__,
        }
    print(json.dumps(refusal), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
