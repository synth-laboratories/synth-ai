"""J3 OAuth conformance over loopback with native SDK, CLI, and hosted MCP."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from scripts.acceptance.journey_main import surface_factory
from scripts.acceptance.journeys import JourneyContext, Secret, Step, Verdict, run_step
from scripts.acceptance.surfaces import SurfaceName
from synth_ai.errors import local_problem

SESSION = "fixture-clerk-session"
ACCESS = "fixture-oauth-access"
REFRESH = "fixture-oauth-refresh"
DISCOVERY = "/.well-known/oauth-protected-resource/mcp"
GRANT = "00000000-0000-4000-8000-000000000001"
CLIENT = "fixture-client"
CONSOLE = "https://console.fixture.invalid"


@contextmanager
def serve(handler):
    class Handler(BaseHTTPRequestHandler):
        def answer(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status, body, headers = handler(self.command, urlsplit(self.path), self.headers, raw)
            encoded = json.dumps(body).encode() if status != 202 else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_DELETE = answer

        def log_message(self, *_):
            return

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join()


class Authority:
    def __init__(self, contract, defect=""):
        self.defect = defect
        self.account = copy.deepcopy(
            contract["paths"]["/v1/account"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        )
        self.account["principal"]["credential_kind"] = "oauth"
        self.events = []
        self.oauth = ""
        self.request = {}
        self.revoked = False

    def core(self, method, url, headers, raw):
        if url.path in {"/mcp/oauth/register", "/mcp/oauth/authorize"}:
            return self.handle(method, url, headers, raw)
        if url.path != "/v1/account":
            return 404, {}, {}
        assert (method, url.path, headers.get("Authorization")) == (
            "GET",
            "/v1/account",
            "Bearer " + ACCESS,
        )
        if self.revoked:
            self.events.append("account_after_revoke")
            code = "api_key_invalid" if self.defect == "pkce_refusal" else "oauth_token_invalid"
            return 401, local_problem(code, "authenticate", status=401).model_dump(mode="json"), {}
        self.events.append("account")
        return 200, self.account, {}

    def handle(self, method, url, headers, raw):
        path = url.path
        resource = self.oauth + "/mcp"
        if path == "/v1/account":
            return self.core(method, url, headers, raw)
        if path == DISCOVERY:
            self.events.append("resource")
            return (
                200,
                {
                    "resource": resource
                    if self.defect != "resource"
                    else "https://foreign.invalid/mcp",
                    "authorization_servers": [
                        self.oauth + "/" if self.defect != "issuer" else "https://foreign.invalid/"
                    ],
                },
                {},
            )
        if path == "/.well-known/oauth-authorization-server":
            self.events.append("issuer")
            return (
                200,
                {
                    "issuer": self.oauth + "/",
                    "code_challenge_methods_supported": ["S256"],
                    "registration_endpoint": self.oauth + "/mcp/oauth/register",
                    "authorization_endpoint": self.oauth + "/mcp/oauth/authorize",
                    "token_endpoint": self.oauth + "/mcp/oauth/token",
                    "revocation_endpoint": self.oauth + "/mcp/oauth/revoke",
                },
                {},
            )
        if path == "/mcp/oauth/register":
            self.events.append("register")
            assert method == "POST"
            assert json.loads(raw)["token_endpoint_auth_method"] == "none"
            return 201, {"client_id": CLIENT}, {}
        if path == "/mcp/oauth/authorize":
            self.events.append("authorize")
            self.request = {k: v[0] for k, v in parse_qs(url.query).items()}
            if self.request["resource"] != resource:
                return 400, {"error": "invalid_target"}, {}
            # Core mcpoauth.ConsentScopes: every authorize must carry the required
            # scope (publicpolicy.RequiredScopes = user:org:read) or it is refused.
            if "user:org:read" not in self.request.get("scope", "").split():
                return 400, {"error": "invalid_scope"}, {}
            return 303, {}, {"Location": CONSOLE + "/mcp/authorize?request=" + "r" * 36}
        if path == "/mcp/oauth/request":
            self.events.append("request")
            assert headers.get("Authorization") is None
            client_id = CLIENT if self.defect != "client" else "other-client"
            return 200, {"client_id": client_id, "resource": resource}, {}
        if path == "/mcp/oauth/consent":
            self.events.append("consent")
            if (
                headers.get("Authorization") is not None
                or headers.get("Origin") != CONSOLE
                or headers.get_all("X-Clerk-Session-Token") != [SESSION]
                or self.defect == "consent"
            ):
                return 403, {"error": "access_denied"}, {}
            assert json.loads(raw) == {"request": "r" * 36, "approve": True}
            callback = self.request["redirect_uri"]
            query = urlencode(
                {
                    "state": "mismatch" if self.defect == "state" else self.request["state"],
                    "iss": self.oauth + "/",
                    "code": "fixture-code",
                }
            )
            return 200, {"redirect_uri": callback + "?" + query}, {}
        if path == "/mcp/oauth/token":
            self.events.append("token")
            form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            assert form["resource"] == self.request["resource"] == resource
            assert form["redirect_uri"] == self.request["redirect_uri"]
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest())
                .decode()
                .rstrip("=")
            )
            assert challenge == self.request["code_challenge"]
            return 200, {"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 600}, {}
        if path == "/mcp/oauth/revoke":
            self.events.append("token_revoke")
            assert method == "POST"
            assert headers.get("Authorization") is None
            assert headers.get("X-Clerk-Session-Token") is None
            form = parse_qs(raw.decode())
            assert form["client_id"] == [CLIENT]
            if "resource" in form:
                assert form["resource"] == [resource]
                assert form["token"] == [ACCESS]
            else:
                assert form["token"] == [REFRESH]
            self.revoked = True
            return 200, {}, {}
        if path == "/mcp/oauth/grants":
            self.events.append("grants")
            assert headers.get_all("X-Clerk-Session-Token") == [SESSION]
            assert headers.get("Authorization") is None
            if self.defect == "grants":
                return 403, {"code": "session_required"}, {}
            return (
                200,
                {
                    "grants": [
                        {
                            "client_id": CLIENT,
                            "grant_id": GRANT,
                            "status": "active",
                            "owned_by_viewer": self.defect != "foreign_grant",
                        }
                    ],
                },
                {},
            )
        if path == "/mcp/oauth/grants/" + GRANT:
            self.events.append("revoke")
            assert method == "DELETE"
            assert headers.get_all("X-Clerk-Session-Token") == [SESSION]
            assert headers.get("Authorization") is None
            if self.defect == "revoke":
                return 403, {"code": "session_required"}, {}
            self.revoked = True
            return 200, {"grant_id": GRANT, "status": "revoked", "outcome": "revoked"}, {}
        assert path == "/mcp"
        if method == "GET":
            self.events.append("revoked_read")
            assert headers.get("Authorization") == "Bearer " + ACCESS
            if not self.revoked:
                return 200, {}, {}
            challenge = self.oauth + ("/foreign" if self.defect == "challenge" else DISCOVERY)
            return (
                401,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Authentication required."},
                },
                {
                    "WWW-Authenticate": f'Bearer resource_metadata="{challenge}", scope="account:read"'
                },
            )
        if method == "DELETE":
            return 200, {}, {}
        rpc = json.loads(raw)
        message = rpc["method"]
        if message == "initialize":
            result = {
                "protocolVersion": rpc["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif message == "tools/list":
            result = {"tools": [{"name": "account_get", "inputSchema": {"type": "object"}}]}
        elif message == "tools/call":
            self.events.append("tool")
            assert headers.get("Authorization") == "Bearer " + ACCESS
            assert rpc["params"]["name"] == "account_get"
            result = {"content": [{"type": "text", "text": json.dumps(self.account)}]}
        else:
            assert message == "notifications/initialized"
            return 202, {}, {}
        return 200, {"jsonrpc": "2.0", "id": rpc["id"], "result": result}, {}


@pytest.mark.parametrize("surface", [SurfaceName.SDK, SurfaceName.CLI])
@pytest.mark.parametrize("split_origin", [False, True], ids=["same-issuer", "split-issuer"])
@pytest.mark.parametrize("defect", ["", "pkce_refusal"], ids=["exact-refusal", "wrong-refusal"])
def test_pkce_uses_clerk_header_on_discovered_issuer(
    contract, tmp_path, monkeypatch, surface, split_origin, defect
):
    authority = Authority(contract, defect)
    with serve(authority.handle) as oauth, serve(authority.core) as core:
        authority.oauth = oauth
        monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", oauth)
        origin = core if split_origin else oauth
        make = surface_factory(surface, origin=origin, work_directory=tmp_path, mcp_url=None)
        with make("fixture-setup") as opened:
            result = run_step(
                JourneyContext(
                    opened, surface, origin, make, tmp_path, secrets={Secret.SESSION: SESSION}
                ),
                Step.CLI_OAUTH,
            )
    assert result.verdict == (Verdict.FAIL if defect else Verdict.PASS), result
    if defect:
        assert result.cause == "oauth_token_revocation_unproven"
    assert authority.events == [
        "issuer",
        "register",
        "authorize",
        "request",
        "consent",
        "token",
        "account",
        "issuer",
        "token_revoke",
        "account_after_revoke",
    ]
    assert not (tmp_path / "oauth-credentials.json").exists()
    assert all(
        value not in json.dumps(result.to_json())
        for value in (SESSION, ACCESS, REFRESH, "fixture-code")
    )


@pytest.mark.parametrize(
    ("defect", "verdict", "last_event"),
    [
        ("", Verdict.PASS, "revoked_read"),
        ("resource", Verdict.FAIL, "resource"),
        ("issuer", Verdict.FAIL, "resource"),
        ("client", Verdict.FAIL, "request"),
        ("consent", Verdict.FAIL, "consent"),
        ("state", Verdict.FAIL, "consent"),
        ("grants", Verdict.FAIL, "revoked_read"),
        ("foreign_grant", Verdict.FAIL, "revoked_read"),
        ("revoke", Verdict.FAIL, "revoked_read"),
        ("challenge", Verdict.FAIL, "revoked_read"),
    ],
)
def test_hosted_mcp_oauth_binds_resource_and_grant_revocation(
    contract,
    tmp_path,
    defect,
    verdict,
    last_event,
):
    authority = Authority(contract, defect)
    surface = SurfaceName.MCP_HOSTED
    with serve(authority.handle) as oauth, serve(authority.core) as core:
        authority.oauth = oauth
        mcp = oauth + "/mcp"
        make = surface_factory(surface, origin=core, work_directory=tmp_path, mcp_url=mcp)
        with make("fixture-setup") as opened:
            result = run_step(
                JourneyContext(
                    opened,
                    surface,
                    core,
                    make,
                    tmp_path,
                    secrets={Secret.SESSION: SESSION},
                    mcp_url=mcp,
                ),
                Step.MCP_CONSENT,
            )
    assert result.verdict == verdict, result
    assert authority.events[-1] == last_event
    if defect == "consent":
        assert result.cause == "oauth_failed:consent:access_denied"
        assert result.evidence[-1]["consent_status"] == 403
    if defect in {"grants", "foreign_grant", "revoke"}:
        stage = "revoke" if defect == "revoke" else "grants"
        assert result.cause.startswith(f"oauth_failed:{stage}:")
        assert authority.events[-2:] == ["token_revoke", "revoked_read"]
        assert result.evidence[-1]["fallback_revoke_status"] == 200
        assert result.evidence[-1]["fallback_revoke_verified"] is True
        assert ("revoke" in authority.events) == (defect == "revoke")
    if verdict == Verdict.PASS:
        assert authority.events == [
            "resource",
            "issuer",
            "register",
            "authorize",
            "request",
            "consent",
            "token",
            "grants",
            "tool",
            "revoke",
            "revoked_read",
        ]
    assert all(
        value not in json.dumps(result.to_json())
        for value in (SESSION, ACCESS, REFRESH, "fixture-code")
    )
