"""Restart owned test clients against the existing Swarms fixture (SYN-5567)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
from test_journey import ANSWER, OUTPUT_ID, RUN_ID, FakeApi, frame
from typing_extensions import override

TIMEOUT_SECONDS = 15
RUN_PATH = f"/v1/runs/{RUN_ID}"
CLI = ("-m", "synth_ai.cli", "--env", "local")
MCP = ("-c", "from synth_ai.mcp.server import main; main()", "--env", "local")
SDK = """
import json
import sys
from pathlib import Path

from synth_ai import SynthClient
from synth_ai._stream import StreamCheckpoint

origin, operation, run_id, state_path = sys.argv[1:]
path = Path(state_path)
with SynthClient("fixture", base_url=origin) as client:
    if operation == "follow":
        checkpoint = (
            StreamCheckpoint.from_dict(json.loads(path.read_bytes()))
            if path.exists()
            else StreamCheckpoint(run_id=run_id)
        )
        for item in client.swarms.follow(run_id, checkpoint=checkpoint):
            print(item.model_dump_json(exclude_unset=True), flush=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(checkpoint.to_dict()))
            temporary.replace(path)
    else:
        values = {
            "launch": lambda: client.swarms.launch(
                "Fixture stay alive", budget="2.00", idempotency_key="restart-once"
            ),
            "wait": lambda: client.swarms.wait(run_id, timeout=30),
            "result": lambda: client.swarms.result(run_id),
            "outputs": lambda: client.swarms.outputs(run_id),
        }
        print(values[operation]().model_dump_json(exclude_unset=True), flush=True)
"""


@contextlib.contextmanager
def loopback_run(contract, barrier):
    fixture = FakeApi(contract)
    blocked = threading.Event()
    resumed = threading.Event()
    opens = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.answer()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == f"{RUN_PATH}/events":
                requests.append((self.command, path))
                opens.append(self.headers.get("Last-Event-ID"))
                first = frame(None, fixture.event(1), "1").encode()
                second = frame(None, fixture.event(2), "2").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(first)
                self.wfile.flush()
                if resumed.is_set():
                    # Model a replay of the last committed event from the fixture.
                    self.wfile.write(second + second)
                else:
                    if barrier == "follow":
                        blocked.set()
                    resumed.wait(TIMEOUT_SECONDS)
                return
            if path == f"{RUN_PATH}/wait":
                requests.append((self.command, path))
                if not resumed.is_set():
                    blocked.set()
                    resumed.wait(TIMEOUT_SECONDS)
                    return
                self.reply(
                    httpx.Response(200, json={"outcome": "terminal", "run": fixture.snapshot()})
                )
                return
            self.answer()
            if path == RUN_PATH and barrier == "wait":
                blocked.set()

        def answer(self):
            assert isinstance(self.server, ThreadingHTTPServer)
            length = int(self.headers.get("Content-Length", "0"))
            request = httpx.Request(
                self.command,
                f"http://127.0.0.1:{self.server.server_port}{self.path}",
                headers=dict(self.headers),
                content=self.rfile.read(length),
            )
            requests.append((self.command, request.url.path))
            self.reply(fixture(request))

        def reply(self, response):
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)
            self.wfile.flush()

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def settle():
        fixture.state = "completed"
        resumed.set()

    try:
        yield SimpleNamespace(
            origin=f"http://127.0.0.1:{server.server_port}",
            blocked=blocked,
            settle=settle,
            opens=opens,
            requests=requests,
            fixture=fixture,
        )
    finally:
        resumed.set()
        server.shutdown()
        server.server_close()
        thread.join()


def start(args, workspace, origin):
    # A fresh directory and a complete minimal environment keep the client on
    # the disposable route, with a fixture bearer and private local checkpoint.
    return subprocess.Popen(
        (sys.executable, "-B", *args),
        cwd=workspace,
        env={
            "PATH": os.defpath,
            "SYNTH_API_KEY": "fixture",
            "SYNTH_AUTH_FILE": str(workspace / "absent-auth.json"),
            "SYNTH_LOCAL_API_BASE_URL": origin,
            "SYNTH_STATE_DIR": str(workspace / "state"),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@contextlib.contextmanager
def running(args, workspace, origin):
    with start(args, workspace, origin) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=TIMEOUT_SECONDS)


def client_args(driver, operation, origin, checkpoint):
    if driver == "sdk":
        return ("-c", SDK, origin, operation, RUN_ID, str(checkpoint))
    if operation == "follow":
        return (*CLI, "logs", RUN_ID, "--follow", "--json")
    return (*CLI, operation, RUN_ID, "--json")


def collect(process):
    out, err = process.communicate(timeout=TIMEOUT_SECONDS)
    assert process.returncode == 0, err.decode()
    return [json.loads(line) for line in out.splitlines()]


def await_barrier(event, process):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if event.wait(0.02):
            assert process.poll() is None
            return
        if process.poll() is not None:
            _, diagnostics = process.communicate()
            pytest.fail("client exited before its restart barrier: " + diagnostics.decode())
    pytest.fail("fixture did not observe the client restart barrier")


def checkpoint_committed(path, process):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        assert process.poll() is None
        if path.is_file():
            saved = json.loads(path.read_bytes())
            if saved["last_event_id"] == "1":
                assert saved["run_id"] == RUN_ID
                return
        time.sleep(0.01)
    pytest.fail("first stream event was not checkpointed")


def killed_output(process):
    assert process.poll() is None
    process.kill()
    out, _ = process.communicate(timeout=TIMEOUT_SECONDS)
    assert process.returncode < 0
    return [json.loads(line) for line in out.splitlines()]


def launch_once(origin, workspace):
    args = client_args("sdk", "launch", origin, workspace / "unused-checkpoint.json")
    with running(args, workspace, origin) as process:
        [initial] = collect(process)
    assert initial["run_id"] == RUN_ID and initial["state"] == "running"


def assert_finished(run):
    assert run["run_id"] == RUN_ID and run["state"] == "completed"
    assert run["execution_closed"] and run["accounting_final"] and run["cleanup_complete"]


def assert_one_output(server, result, outputs):
    assert result["run_id"] == RUN_ID and result["answer"] == ANSWER.decode()
    assert result["answer_output"]["output_id"] == OUTPUT_ID
    assert [(item["run_id"], item["output_id"]) for item in outputs["items"]] == [
        (RUN_ID, OUTPUT_ID)
    ]
    assert outputs["items"][0]["reference"]["digest_sha256"] == hashlib.sha256(ANSWER).hexdigest()
    assert server.requests.count(("POST", "/v1/runs")) == 1
    assert all(
        (method, path) == ("POST", "/v1/runs")
        or (method == "GET" and (path == RUN_PATH or path.startswith(RUN_PATH + "/")))
        for method, path in server.requests
    )
    assert len(server.fixture.launches) == 1


@pytest.mark.parametrize("driver", ["sdk", "cli"])
@pytest.mark.parametrize("operation", ["wait", "follow"])
def test_restarts_sdk_and_cli_on_the_same_run(contract, tmp_path, driver, operation):
    checkpoint = (
        tmp_path / "sdk-checkpoint.json"
        if driver == "sdk"
        else tmp_path / "state" / "streams" / f"{RUN_ID}.json"
    )
    with loopback_run(contract, operation) as server:
        launch_once(server.origin, tmp_path)
        args = client_args(driver, operation, server.origin, checkpoint)
        with running(args, tmp_path, server.origin) as first:
            await_barrier(server.blocked, first)
            if operation == "follow":
                checkpoint_committed(checkpoint, first)
                if driver == "cli":
                    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
            initial = killed_output(first)

        server.settle()
        with running(args, tmp_path, server.origin) as second:
            remaining = collect(second)
        if operation == "wait":
            assert initial == []
            [run] = remaining
            assert_finished(run)
        else:
            assert [event["sequence"] for event in initial] == [1]
            assert [event["sequence"] for event in remaining] == [2]
            assert {event["run_id"] for event in initial + remaining} == {RUN_ID}
            assert len({event["event_id"] for event in initial + remaining}) == 2
            assert server.opens == [None, "1"]
            assert json.loads(checkpoint.read_bytes())["last_event_id"] == "2"

        def read(name):
            args = client_args(driver, name, server.origin, checkpoint)
            with running(args, tmp_path, server.origin) as reader:
                [result] = collect(reader)
            return result

        assert_one_output(server, read("result"), read("outputs"))


def send(process, method, params=None, *, identity=None):
    message = {"jsonrpc": "2.0", "method": method}
    if identity is not None:
        message["id"] = identity
    if params is not None:
        message["params"] = params
    assert process.stdin is not None
    process.stdin.write(json.dumps(message).encode() + b"\n")
    process.stdin.flush()


def response(process, identity):
    assert process.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(TIMEOUT_SECONDS), "MCP stdio response was not ready"
    raw = json.loads(process.stdout.readline())
    assert raw["id"] == identity and "error" not in raw, raw
    return raw["result"]


def initialize(process):
    send(
        process,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "restart-fixture", "version": "1"},
        },
        identity=1,
    )
    assert response(process, 1)["protocolVersion"]
    send(process, "notifications/initialized")


def tool_call(process, name, identity, **arguments):
    send(process, "tools/call", {"name": name, "arguments": arguments}, identity=identity)


def tool_result(process, identity):
    result = response(process, identity)
    assert result.get("isError") is False, result
    return json.loads(result["content"][0]["text"])


def test_restarts_mcp_stdio_mid_wait_on_the_same_run(contract, tmp_path):
    pytest.importorskip("mcp")
    with loopback_run(contract, "mcp_wait") as server:
        launch_once(server.origin, tmp_path)
        with running(MCP, tmp_path, server.origin) as first:
            initialize(first)
            tool_call(first, "swarms_run_wait", 2, run_id=RUN_ID, timeout_seconds=10)
            await_barrier(server.blocked, first)
            assert killed_output(first) == []

        server.settle()
        with running(MCP, tmp_path, server.origin) as second:
            initialize(second)
            tool_call(second, "swarms_run_wait", 2, run_id=RUN_ID, timeout_seconds=10)
            final = tool_result(second, 2)
            assert final["outcome"] == "terminal"
            assert_finished(final["run"])
            tool_call(second, "swarms_run_result", 3, run_id=RUN_ID)
            result = tool_result(second, 3)
            tool_call(second, "swarms_run_outputs", 4, run_id=RUN_ID)
            outputs = tool_result(second, 4)
        assert server.requests.count(("GET", f"{RUN_PATH}/wait")) == 2
        assert_one_output(server, result, outputs)
