import json
import os
import re
import subprocess
import sys

import pytest

from synth_ai._generated.operations import OPERATIONS
from synth_ai.cli import COMMANDS, parser

TASK = "craftax-classic-enginebench-policy-v1"


def test_cli_product_operation_inventory():
    declared = {oid for commands in COMMANDS.values() for oid in commands.values()}
    assert declared | {"createRun"} == set(OPERATIONS)
    for group in COMMANDS:
        result = subprocess.run(
            [sys.executable, "-m", "synth_ai.cli", group, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert all(verb in result.stdout for verb in COMMANDS[group])
    for verb in ("watch", "wait", "follow", "attach"):
        assert parser().parse_args(["swarms", verb, "run_x"]).verb == verb
    for command in (["login"], ["logout"], ["auth", "status"]):
        parser().parse_args(command)


@pytest.mark.parametrize("state,exit_code", [("completed", 0), ("failed", 1), ("cancelled", 0)])
def test_watch_streams_then_reports_terminal_state(contract, state, exit_code):
    snapshot = dict(
        contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    snapshot.update(state=state, execution_closed=True)
    event = {
        "event_id": "evt_AAAAAAAAAAAAAAAAAAAAAA",
        "run_id": snapshot["run_id"],
        "sequence": 1,
        "kind": "progress",
        "original_kind": "progress",
        "observed_at": None,
        "usage": None,
        "approval_id": None,
    }
    # Exercise the real CLI entry point against loopback HTTP, including exit codes.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            if self.headers.get("Accept") == "text/event-stream":
                payload = ("id: 1\ndata: " + json.dumps(event) + "\n\n").encode()
                self.send_header("Content-Type", "text/event-stream")
            else:
                payload = json.dumps(snapshot).encode()
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "synth_ai.cli",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "swarms",
                "watch",
                snapshot["run_id"],
                "--timeout",
                "5",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "SYNTH_API_KEY": "fixture"},
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert result.returncode == exit_code, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["sequence"] == 1
    assert rows[-1]["state"] == state


@pytest.mark.parametrize(
    "method,release,overrides",
    [
        ("status-quo", None, None),
        ("fork-flush", "research.fork_flush@1", {"branches": 2}),
        ("forgetting", "research.forgetting@1", {"peers": 2}),
    ],
)
def test_launch_method_reaches_http_contract(contract, method, release, overrides):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    snapshot = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(snapshot).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    args = [
        sys.executable,
        "-m",
        "synth_ai.cli",
        "--base-url",
        f"http://127.0.0.1:{server.server_port}",
        "swarms",
        "launch",
        "Improve policy",
        "--budget",
        "2.00",
        "--method",
        method,
    ]
    if release:
        args += ["--method-task", TASK]
    if overrides:
        args += ["--method-overrides", json.dumps(overrides)]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env={**os.environ, "SYNTH_API_KEY": "fixture"},
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert result.returncode == 0, result.stderr
    assert len(requests) == 1
    expected = {"release": release, "task": TASK, "overrides": overrides} if release else None
    assert requests[0].get("method") == expected
    assert "project_id" not in requests[0]
    if expected:
        schema = contract["components"]["schemas"]["LaunchRequest"]["properties"]["method"]
        assert set(schema["required"]) <= set(expected) and set(expected) <= set(
            schema["properties"]
        )
        assert re.fullmatch(schema["properties"]["task"]["pattern"], expected["task"])
        assert expected["release"] in schema["properties"]["release"]["enum"]
    assert requests[0]["limits"]["maximum_charge"]["amount"] == "2.00"


@pytest.mark.parametrize(
    "extra,refusal",
    [
        (["--method", "fork-flush"], "requires --method-task"),
        (["--method", "status-quo", "--method-task", TASK], "require fork-flush or forgetting"),
        (["--method-task", TASK], "require fork-flush or forgetting"),
    ],
)
def test_method_task_refuses_before_any_request(extra, refusal):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "synth_ai.cli",
            "--base-url",
            "http://127.0.0.1:9",
            "swarms",
            "launch",
            "Task",
            "--budget",
            "1.00",
            *extra,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SYNTH_API_KEY": "fixture"},
        timeout=10,
    )
    assert result.returncode != 0 and refusal in result.stderr
