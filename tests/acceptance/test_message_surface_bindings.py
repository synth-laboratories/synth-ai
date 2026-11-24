"""Actual SDK/CLI adapters against owned fake HTTP; no Core or installed proof."""

from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from typing_extensions import override

from scripts.acceptance.surfaces import Call, CLISurface, SDKSurface, accepted
from synth_ai import SynthClient
from synth_ai._generated.models import RunMessagePage

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
MESSAGE = "msg_AAAAAAAAAAAAAAAAAAAAAA"
PARENT = "run_AAAAAAAAAAAAAAAAAAAAAB"


def message():
    return {
        "run_id": RUN,
        "message_id": MESSAGE,
        "thread_id": "thr_AAAAAAAAAAAAAAAAAAAAAA",
        "author": "customer",
        "kind": "question",
        "sequence": 1,
        "state": {"kind": "queued"},
        "body": {"kind": "available", "text": "fixture"},
        "created_at": "2026-11-17T00:00:00Z",
        "updated_at": "2026-11-17T00:00:00Z",
    }


@contextmanager
def fake_http():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        @override
        def log_message(self, format: str, *args: Any) -> None:
            # Fixture HTTP access logs would otherwise expose paths; retain only
            # the structured in-memory assertions below.
            return

        def reply(self):
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            requests.append(
                (self.command, self.path, dict(self.headers), json.loads(raw) if raw else None)
            )
            if self.command == "POST" and self.path == "/v1/runs":
                value = {
                    "type": "about:blank",
                    "title": "continue_from_inputs_unavailable",
                    "code": "continue_from_inputs_unavailable",
                    "status": 409,
                    "detail": "continue_from_inputs_unavailable",
                    "instance": "urn:request:fixture",
                    "service": "synth-backend",
                    "stage": "continuation",
                    "outcome": "not_accepted",
                    "retryable": False,
                }
                status = 409
            elif self.command == "GET" and "/messages?" in self.path:
                value = {
                    "items": [message()],
                    "has_more": False,
                    "next_cursor": None,
                    "availability": {"kind": "available"},
                }
                status = 200
            else:
                value, status = message(), 200
            encoded = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def calls():
    return [
        Call(
            "sendRunMessage",
            path={"run_id": RUN},
            body={"kind": "question", "text": "fixture"},
            idempotency_key="same-message-key",
        ),
        Call(
            "listRunMessages", path={"run_id": RUN}, query={"cursor": "fixture-cursor", "limit": 17}
        ),
        Call("getRunMessage", path={"run_id": RUN, "message_id": MESSAGE}),
    ]


def verify_messages(surface, requests):
    outcomes = [surface.call(call) for call in calls()]
    assert all(outcome.accepted for outcome in outcomes), [
        outcome.evidence() for outcome in outcomes
    ]
    assert len(requests) == 3
    assert requests[0][0:2] == ("POST", f"/v1/runs/{RUN}/messages")
    assert requests[0][2]["Idempotency-Key"] == "same-message-key"
    assert requests[0][3] == {"kind": "question", "text": "fixture"}
    assert requests[1][0:2] == ("GET", f"/v1/runs/{RUN}/messages?cursor=fixture-cursor&limit=17")
    assert requests[2][0:2] == ("GET", f"/v1/runs/{RUN}/messages/{MESSAGE}")


def continuation():
    return Call(
        "createRun",
        body={
            "message": "fixture",
            "continue_from": PARENT,
            "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
        },
        idempotency_key="parent-key",
    )


def test_sdk_message_bindings_reach_real_sdk_http_methods():
    with fake_http() as (origin, requests), SynthClient("fixture", base_url=origin) as client:
        verify_messages(SDKSurface(client), requests)


def test_sdk_message_page_preserves_missing_posted_state_fields():
    reply = {
        **message(),
        "author": "manager",
        "kind": "reply",
        "state": {"kind": "posted"},
    }
    page = RunMessagePage.model_validate(
        {
            "items": [reply],
            "has_more": False,
            "next_cursor": None,
            "availability": {"kind": "available"},
        }
    )
    outcome = accepted("listRunMessages", page)
    assert isinstance(outcome.value, dict)
    items = outcome.value["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    assert first["state"] == {"kind": "posted"}


def test_cli_message_bindings_reach_real_child_http_methods(tmp_path):
    import os
    from pathlib import Path

    with fake_http() as (origin, requests):
        surface = CLISurface(
            python=Path.cwd() / ".venv/bin/python",
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_KEY": "fixture",
            },
            work_directory=tmp_path,
        )
        verify_messages(surface, requests)


def test_sdk_continuation_adapter_cannot_drop_parent_into_ordinary_launch():
    with fake_http() as (origin, requests), SynthClient("fixture", base_url=origin) as client:
        outcome = SDKSurface(client).call(continuation())
        assert outcome.code == "continue_from_inputs_unavailable"
        assert len(requests) == 1
        assert requests[0][3]["continue_from"] == PARENT


def test_cli_continuation_adapter_cannot_drop_parent_into_ordinary_launch(tmp_path):
    import os
    from pathlib import Path

    with fake_http() as (origin, requests):
        surface = CLISurface(
            python=Path.cwd() / ".venv/bin/python",
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_KEY": "fixture",
            },
            work_directory=tmp_path,
        )
        outcome = surface.call(continuation())
        assert outcome.code == "continue_from_inputs_unavailable"
        assert len(requests) == 1
        assert requests[0][3]["continue_from"] == PARENT
