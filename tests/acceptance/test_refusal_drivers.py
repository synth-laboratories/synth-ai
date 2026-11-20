"""SYN-5723 client-path qualification on controlled loopback HTTP.

Real SDK, CLI subprocess and MCP stdio use the same server refusal sequence.
These fixtures are not a cached run or live J-refusals acceptance receipt.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acceptance.j1 import AdapterError, CLIDriver, RecordedLaunch, SDKDriver
from scripts.acceptance.launch_journal import LaunchJournal
from scripts.acceptance.mcp_driver import MCPDriver
from scripts.acceptance.mcp_transport import SessionToolCaller, StdioCommand, stdio_session
from synth_ai import SynthClient
from synth_ai.errors import SwarmsAPIError, local_problem

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).parents[2]
RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
RECORDING = RecordedLaunch("fixture task", "0.10", 30, {}, ())
STANDING = {
    "position": 3,
    "ahead": 2,
    "estimated_start_at": None,
    "expires_at": "2026-11-17T06:00:00Z",
}


@contextmanager
def pressure_api(contract, *, status, retryable):
    snapshot = json.loads(
        json.dumps(
            contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        )
    )
    snapshot.update(
        run_id=RUN,
        state="queued",
        queue=STANDING,
        optimizer={
            "topology": "swarms_v2",
            "version": 2,
            "params": {},
            "reason": "environment_default",
            "definition_digest": "sha256:" + "a" * 64,
        },
        execution_closed=False,
        accounting_final=False,
        cleanup_complete=False,
    )
    problem = local_problem(
        "overloaded" if retryable else "quota_exhausted", "admission", status=status
    ).model_dump(mode="json")
    problem.update(retryable=retryable, outcome="not_accepted")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(
                (self.path, self.headers.get("Idempotency-Key"), body, time.monotonic())
            )
            refused = not retryable or len(requests) == 1
            raw = json.dumps(problem if refused else snapshot).encode()
            self.send_response(status if refused else 202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if refused:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def driver_for(surface, origin, tmp_path):
    environment = {**os.environ, "SYNTH_LOCAL_API_BASE_URL": origin, "SYNTH_API_KEY": "fixture"}
    with ExitStack() as stack:
        if surface == "sdk":
            client = stack.enter_context(SynthClient("fixture", base_url=origin))
            driver = SDKDriver(client)
        elif surface == "cli":
            driver = CLIDriver(
                python=Path(sys.executable), environment=environment, working_directory=ROOT
            )
        else:
            command = StdioCommand(
                Path(sys.executable).parent / "synth-ai-mcp", ("--env", "local"), environment, ROOT
            )
            caller = stack.enter_context(SessionToolCaller(stdio_session(command)))
            driver = MCPDriver(caller, surface="mcp_stdio")
        yield driver


def launch(driver, tmp_path):
    arguments = dict(
        recording=RECORDING,
        idempotency_key="fixture-queue-retry",
        journal=LaunchJournal(tmp_path / "launch"),
    )
    if isinstance(driver, CLIDriver):
        return driver.launch_recorded(**arguments, work_directory=tmp_path)
    return driver.launch_recorded(**arguments)


@pytest.mark.parametrize("surface", ["sdk", "cli", "mcp"])
@pytest.mark.parametrize("status", [429, 503])
def test_retry_preserves_launch_identity_and_queued_ack(contract, tmp_path, surface, status):
    with pressure_api(contract, status=status, retryable=True) as (origin, requests):
        with driver_for(surface, origin, tmp_path) as driver:
            queued = launch(driver, tmp_path)
    assert queued.run_id == RUN and queued.state == "queued"
    assert queued.queue is not None
    assert (queued.queue.position, queued.queue.ahead, queued.queue.estimated_start_at) == (
        3,
        2,
        None,
    )
    assert len(requests) == 2
    assert requests[0][:3] == requests[1][:3]
    assert requests[0][0:2] == ("/v1/runs", "fixture-queue-retry")
    assert requests[1][3] - requests[0][3] >= 1.0
    assert LaunchJournal(tmp_path / "launch").read_accepted() == queued


@pytest.mark.parametrize("surface", ["sdk", "cli", "mcp"])
def test_quota_refusal_is_not_retried_or_recorded_as_accepted(contract, tmp_path, surface):
    with pressure_api(contract, status=429, retryable=False) as (origin, requests):
        with driver_for(surface, origin, tmp_path) as driver:
            with pytest.raises((SwarmsAPIError, AdapterError)):
                launch(driver, tmp_path)
    assert len(requests) == 1
    assert (tmp_path / "launch/launch-intent.json").exists()
    assert not (tmp_path / "launch/launch-accepted.json").exists()
