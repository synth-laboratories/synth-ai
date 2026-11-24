"""SYN-5737 open cells through the real SDK, CLI child and MCP stdio child against loopback Core.

Org B addresses org A's owned fixture and an absent ID on every open cell. The loopback
answers both with the same typed 404, and in ``leaky`` mode it still mutates A's state
for B's message send and grant revoke, which only the owner readback can catch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Any

import pytest
import yaml
from typing_extensions import override

from scripts.acceptance import isolation_remaining as remaining
from scripts.acceptance.journeys import Secret
from scripts.acceptance.surfaces import ROOT

OWNER_ORG, FOREIGN_ORG = (
    "00000000-0000-4000-8000-00000000000a",
    "00000000-0000-4000-8000-00000000000b",
)
RUN, OUTPUT, UPLOAD, KEY = (
    "run_" + "A" * 22,
    "out_" + "A" * 22,
    "upl_" + "A" * 22,
    "key_" + "A" * 22,
)
MESSAGE, REQUEST = "msg_" + "A" * 22, "00000000-0000-4000-8000-000000000054"
GRANT = "00000000-0000-4000-8000-0000000000c1"
CONTENT = b"owned output"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
TOKENS = {
    "a_key": ("a", "api_key"),
    "a_session": ("a", "session"),
    "b_key": ("b", "api_key"),
    "b_session": ("b", "session"),
}
OPEN_OPERATIONS = (
    "finalizeUpload",
    "getUpload",
    "getRunOutput",
    "getRunOutputContent",
    "listRunOutputs",
    "updateApiKey",
    "updateRunLabels",
    "waitRun",
    "withdrawPlanRequest",
    "sendRunMessage",
    "listRunMessages",
    "getRunMessage",
    "revokeOAuthGrant",
)


def contract():
    return yaml.safe_load((ROOT / "contracts/synth-v1.openapi.yaml").read_text())


def example(document, path, method="get"):
    response = document["paths"][path][method]["responses"]["200"]["content"]
    return copy.deepcopy(response["application/json"]["example"])


def routes(document: dict[str, Any]) -> list[tuple[str, re.Pattern[str], str]]:
    """(method, compiled path template, operationId) for every contract operation."""
    table: list[tuple[str, re.Pattern[str], str]] = []
    for path, methods in document["paths"].items():
        pattern = re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", path) + "$")
        for method, operation in methods.items():
            if isinstance(operation, dict) and isinstance(operation.get("operationId"), str):
                table.append((method.upper(), pattern, operation["operationId"]))
    return table


class LoopbackCore:
    def __init__(self, *, leaky: bool) -> None:
        self.leaky = leaky
        self.document = contract()
        self.table = routes(self.document)
        self.lock = Lock()
        self.requests: list[tuple[str, str, str, str]] = []  # (side, operationId, path, token)
        self.messages = [self.message(MESSAGE, 1)]
        self.grant_status = "active"

    def operation(self, method: str, path: str) -> str:
        matches = [op for verb, pattern, op in self.table if verb == method and pattern.match(path)]
        return matches[0] if len(matches) == 1 else "unrouted"

    @staticmethod
    def message(message_id: str, sequence: int):
        return {
            "run_id": RUN,
            "message_id": message_id,
            "thread_id": "thr_" + "A" * 22,
            "author": "customer",
            "kind": "question",
            "sequence": sequence,
            "state": {"kind": "queued"},
            "body": {"kind": "available", "text": "fixture"},
            "created_at": "2026-11-17T00:00:00Z",
            "updated_at": "2026-11-17T00:00:00Z",
        }

    @staticmethod
    def not_found(code: str):
        return 404, {
            "type": "https://docs.usesynth.ai/errors/" + code,
            "title": "not found",
            "status": 404,
            "detail": "not found",
            "instance": "urn:request:loopback",
            "code": code,
            "service": "synth-backend",
            "stage": "public",
            "outcome": "not_accepted",
            "retryable": False,
        }

    def account(self, side: str, kind: str):
        value = example(self.document, "/v1/account")
        org = OWNER_ORG if side == "a" else FOREIGN_ORG
        value["org"]["id"] = value["principal"]["org_id"] = org
        value["principal"]["user_id"] = "user_" + side
        value["principal"]["credential_kind"] = kind
        value["principal"]["scopes"] = ["account:read", "account:write"]
        return 200, value

    def owner(self, operation: str, path: str):
        document = self.document
        if operation == "getRun":
            value = example(document, "/v1/runs/{run_id}")
            value.update(run_id=RUN, execution_closed=True, state="completed")
            return 200, value
        if operation == "listApiKeys":
            item = {
                "key_id": KEY,
                "prefix": "sk_synth_user_",
                "last4": "abcd",
                "key_type": "user",
                "is_active": True,
                "purpose": "personal",
                "name": "disposable",
                "scopes": ["account:read"],
                "authorization_version": 4,
                "rotation_family_id": None,
            }
            return 200, {"has_more": False, "items": [item]}
        if operation == "listPlanRequests":
            value = example(document, "/v1/plan-requests")
            value["items"][0].update(state="pending", decided_at=None, decision_reason=None)
            return 200, value
        if operation == "getUpload":
            return 200, example(document, "/v1/uploads/{upload_id}")
        if operation == "getRunOutput":
            value = example(document, "/v1/runs/{run_id}/outputs/{output_id}")
            value["reference"] = {"revision_id": "rev_" + "A" * 22, "digest_sha256": DIGEST}
            return 200, value
        if operation == "getRunOutputContent":
            return 200, CONTENT
        if operation == "listRunMessages":
            return 200, {
                "items": list(self.messages),
                "has_more": False,
                "next_cursor": None,
                "availability": {"kind": "available"},
            }
        if operation == "listOAuthGrants":
            value = example(document, "/mcp/oauth/grants")
            value["grants"] = [
                {
                    "grant_id": GRANT,
                    "client_id": "client",
                    "client_name": "fixture",
                    "scopes": ["mcp"],
                    "created_at": "2026-11-17T00:00:00Z",
                    "last_used_at": None,
                    "expires_at": None,
                    "revoked_at": None,
                    "status": self.grant_status,
                    "owner_user_id": "user_a",
                    "owned_by_viewer": True,
                }
            ]
            return 200, value
        return self.not_found("resource_not_found")

    def foreign(self, operation: str, path: str):
        # B never sees A's state; a leaky Core still applies B's mutation to it.
        if self.leaky and operation == "sendRunMessage" and RUN in path:
            self.messages.append(self.message("msg_" + "B" * 22, len(self.messages) + 1))
        if self.leaky and operation == "revokeOAuthGrant" and GRANT in path:
            self.grant_status = "revoked"
        code = "oauth_grant_not_found" if operation == "revokeOAuthGrant" else "resource_not_found"
        return self.not_found(code)

    def answer(self, method: str, path: str, token: str):
        side, kind = TOKENS.get(token, ("?", "?"))
        operation = self.operation(method, path.split("?", 1)[0])
        with self.lock:
            self.requests.append((side, operation, path, token))
            if operation == "getAccount":
                return self.account(side, kind)
            return self.owner(operation, path) if side == "a" else self.foreign(operation, path)


@contextmanager
def loopback(core: LoopbackCore):
    class Handler(BaseHTTPRequestHandler):
        def handle_any(self):
            length = int(self.headers.get("content-length", "0"))
            if length:
                self.rfile.read(length)
            bearer = self.headers.get("Authorization", "").removeprefix("Bearer ")
            token = self.headers.get("X-Clerk-Session-Token") or bearer
            status, value = core.answer(self.command, self.path, token)
            raw = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            media = "application/problem+json" if status >= 400 else "application/json"
            self.send_header(
                "Content-Type", "application/octet-stream" if isinstance(value, bytes) else media
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = do_PATCH = do_DELETE = do_PUT = handle_any

        @override
        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture_ids():
    return {
        "owner_org_id": OWNER_ORG,
        "owner_user_id": "user_a",
        "foreign_org_id": FOREIGN_ORG,
        "foreign_user_id": "user_b",
        "run_id": RUN,
        "output_id": OUTPUT,
        "key_id": KEY,
        "upload_id": UPLOAD,
        "request_id": REQUEST,
        "fixture_key_name": "disposable",
        "message_id": MESSAGE,
        "grant_id": GRANT,
    }


def open_rows():
    return remaining.cell_rows(
        tuple(
            remaining.Cell(driver, operation, "uncovered:surface_unmapped")
            for driver in remaining.DRIVERS
            for operation in OPEN_OPERATIONS
        )
    )


def execute(monkeypatch, tmp_path, *, leaky):
    for name, token in {
        "SYNTH_API_KEY": "a_key",
        Secret.SESSION: "a_session",
        Secret.FOREIGN: "b_key",
        Secret.FOREIGN_SESSION: "b_session",
    }.items():
        monkeypatch.setenv(str(name), token)
    core = LoopbackCore(leaky=leaky)
    rows = open_rows()
    with loopback(core) as origin:
        try:
            remaining.run(
                rows, tmp_path, fixture_ids(), core.document, origin=origin, profile="local"
            )
        except remaining.FixtureBlocked as error:
            return core, rows, str(error)
    return core, rows, None


@pytest.mark.acceptance
def test_open_cells_reach_loopback_core_and_pass_on_every_driver(monkeypatch, tmp_path):
    core, rows, error = execute(monkeypatch, tmp_path, leaky=False)

    assert error is None
    assert len(rows) == 3 * len(OPEN_OPERATIONS) == 39
    assert {(row["driver"], row["operation"]): row["result"] for row in rows} == {
        (row["driver"], row["operation"]): "pass" for row in rows
    }
    foreign = Counter(operation for side, operation, _, _ in core.requests if side == "b")
    # Each driver sends the foreign ID and an absent ID: the binding reached Core twice.
    assert {operation: foreign[operation] for operation in OPEN_OPERATIONS} == dict.fromkeys(
        OPEN_OPERATIONS, 6
    )
    assert "unrouted" not in {operation for _, operation, _, _ in core.requests}
    # revokeOAuthGrant goes through each driver's own surface with B's API key (1.40.0).
    revoke = [row for row in rows if row["operation"] == "revokeOAuthGrant"]
    assert [row["evidence"]["actor_credential"] for row in revoke] == ["api_key"] * 3
    assert {row["evidence"]["route"] for row in revoke} == {"/mcp/oauth/grants/{grant_id}"}
    revoke_tokens = {token for side, op, _, token in core.requests if op == "revokeOAuthGrant"}
    assert revoke_tokens == {"b_key"}
    for row in rows:
        if row["operation"] in {"sendRunMessage", "revokeOAuthGrant"}:
            assert row["owner_state"] == "unchanged"


@pytest.mark.acceptance
def test_foreign_404_that_still_mutates_owner_state_fails_the_cell(monkeypatch, tmp_path):
    _core, rows, error = execute(monkeypatch, tmp_path, leaky=True)

    assert error == "owner_fixture_changed_stop_remaining_drivers"
    first = {row["operation"]: row for row in rows if row["driver"] == remaining.DRIVERS[0]}
    assert first["sendRunMessage"]["result"] == "fail:owner_state_changed"
    assert first["revokeOAuthGrant"]["result"] == "fail:owner_state_changed"
    later = [row for row in rows if row["driver"] != remaining.DRIVERS[0]]
    assert {row["result"] for row in later} == {"pending:owned_fixture"}
