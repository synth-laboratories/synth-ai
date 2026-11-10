"""One customer journey, driven through the typed client, the CLI and the stdio MCP tools.

Every surface runs the same steps against one in-memory fake of the API (contract
examples behind `httpx.MockTransport`, with run state) and must reach the same final
facts. `SURFACES` maps each step to its implementation per surface; a surface that
lacks a step fails the parity check by name.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from synth_ai import AsyncSynthClient, cli
from synth_ai._generated.operations import OPERATIONS
from synth_ai.mcp.server import call_tool
from synth_ai.mcp.tools import TOOLS

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"
OUTPUT_ID = "out_AAAAAAAAAAAAAAAAAAAAAA"
ANSWER = b"The fault is in the retry loop.\n"
TASK = b"# Task\nFind the fault.\n"
BUDGET = "2.00"

# Step -> operations that prove the step reached the API, per surface family.
STEPS: dict[str, dict[str, set[str]]] = {
    "discover_profiles": {"*": {"listProfiles"}},
    "upload_file": {"sdk": {"createUpload"}, "cli": {"createUpload"}, "mcp": {"uploadFile"}},
    "launch": {"*": {"createRun"}},
    "pause": {"*": {"pauseRun"}},
    "resume": {"*": {"resumeRun"}},
    "watch": {"sdk": {"listRunEvents"}, "cli": {"listRunEvents"}, "mcp": {"waitRun"}},
    "outputs": {"*": {"listRunOutputs"}},
    "answer": {
        "sdk": {"getRunOutputContent"},
        "cli": {"getRunOutputContent"},
        "mcp": {"getRunOutput"},
    },
    "usage": {"*": {"listUsageRecords"}},
}


def example(contract, oid):
    op = OPERATIONS[oid]
    responses = contract["paths"][op["path"]][op["method"].lower()]["responses"]
    status = next(code for code in responses if code.startswith("2"))
    return int(status), json.loads(
        json.dumps(responses[status]["content"]["application/json"]["example"])
    )


def route(request: httpx.Request) -> str:
    for oid, op in OPERATIONS.items():
        pattern = "^" + re.sub(r"\{[^}]+\}", "[^/]+", op["path"]) + "$"
        if request.method == op["method"].upper() and re.match(pattern, request.url.path):
            return oid
    raise AssertionError("unrouted " + request.method + " " + request.url.path)


def frame(event: str | None, body: dict, event_id: str | None = None) -> str:
    head = ("event: " + event + "\n" if event else "") + (
        "id: " + event_id + "\n" if event_id else ""
    )
    return head + "data: " + json.dumps(body) + "\n\n"


class FakeApi:
    """A stateful run over contract examples; drops the first event stream once."""

    def __init__(self, contract):
        self.contract = contract
        self.calls: list[str] = []
        self.launches: list[dict] = []
        self.state = "queued"
        self.stream_opens: list[str | None] = []
        self.waits = 0
        digest = hashlib.sha256(ANSWER).hexdigest()
        self.output = {
            "run_id": RUN_ID,
            "output_id": OUTPUT_ID,
            "state": "available",
            "filename": "answer.md",
            "reference": {
                "revision_id": "rev_answer",
                "digest_sha256": digest,
                "media_type": "text/markdown",
                "byte_size": str(len(ANSWER)),
            },
        }

    def snapshot(self) -> dict:
        _, body = example(self.contract, "getRun")
        done = self.state == "completed"
        body.update(
            run_id=RUN_ID,
            state=self.state,
            execution_closed=done,
            accounting_final=done,
            cleanup_complete=done,
        )
        return body

    def event(self, sequence: int) -> dict:
        return {
            "event_id": "evt_" + str(sequence).rjust(22, "A"),
            "run_id": RUN_ID,
            "sequence": sequence,
            "kind": "progress",
            "original_kind": "progress",
            "observed_at": None,
            "usage": None,
            "approval_id": None,
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(200)
        oid = route(request)
        self.calls.append(oid)
        if oid == "listRunEvents" and request.headers.get("Accept") == "text/event-stream":
            last = request.headers.get("Last-Event-ID")
            self.stream_opens.append(last)
            if last is None:
                # Deliver one event, then a retryable in-stream refusal (an owner handoff).
                refusal = {
                    "type": "about:blank",
                    "title": "Event owner unavailable",
                    "status": 503,
                    "detail": "event owner is moving",
                    "instance": "urn:journey",
                    "code": "event_owner_unavailable",
                    "service": "core",
                    "stage": "events",
                    "outcome": "not_accepted",
                    "retryable": True,
                    "last_event_id": "1",
                }
                body = frame(None, self.event(1), "1") + frame("error", refusal)
            else:
                self.state = "completed"
                body = frame(None, self.event(2), "2")
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=body.encode()
            )
        if oid == "createUpload":
            return httpx.Response(201, json=self.upload("committed"))
        if oid == "uploadFile":
            payload = json.loads(request.content)
            assert payload["content"].encode() == TASK
            return httpx.Response(201, json=self.manifest())
        if oid == "createRun":
            launch = json.loads(request.content)
            self.launches.append(launch)
            self.state = "running"
            return httpx.Response(202, json=self.snapshot())
        if oid in ("getRun", "cancelRun"):
            return httpx.Response(200, json=self.snapshot())
        if oid in ("pauseRun", "resumeRun"):
            assert self.state == ("running" if oid == "pauseRun" else "paused")
            self.state = "paused" if oid == "pauseRun" else "running"
            status, receipt = example(self.contract, oid)
            receipt.update(resource_id=RUN_ID)
            return httpx.Response(status, json=receipt)
        if oid == "waitRun":
            # The bounded wait times out once, then the run reaches its terminal state.
            self.waits += 1
            if self.waits > 1:
                self.state = "completed"
            outcome = "terminal" if self.state == "completed" else "timeout"
            return httpx.Response(200, json={"outcome": outcome, "run": self.snapshot()})
        if oid == "listRunOutputs":
            return httpx.Response(200, json={"items": [self.output], "has_more": False})
        if oid == "getRunOutput":
            return httpx.Response(200, json=self.output)
        if oid == "getRunOutputContent":
            return httpx.Response(
                200, content=ANSWER, headers={"Content-Type": "application/octet-stream"}
            )
        if oid == "listUsageRecords":
            assert request.url.params["run_id"] == RUN_ID
            record = {
                "record_id": "usg_journey",
                "run_id": RUN_ID,
                "session_kind": "run",
                "kind": "gateway",
                "provider": "example",
                "preset": "example",
                "program_pattern": "example",
                "actor_kind": "worker",
                "key_id": None,
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_tokens": 0,
                "charge": {"amount": "0.01", "currency": "USD"},
                "created_at": "2026-11-10T00:00:00Z",
            }
            return httpx.Response(200, json={"items": [record], "has_more": False})
        status, body = example(self.contract, oid)
        return httpx.Response(status, json=body)

    def manifest(self) -> dict:
        _, manifest = example(self.contract, "uploadFile")
        manifest.update(size_bytes=len(TASK), digest_sha256=hashlib.sha256(TASK).hexdigest())
        return manifest

    def upload(self, state: str) -> dict:
        return {"upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA", "state": state, "file": self.manifest()}


class Facts:
    """What a surface observed at the end of the journey."""

    def __init__(self) -> None:
        self.state: str | None = None
        self.verified_outputs: list[str] = []
        self.answer: str | None = None
        self.usage_records = 0


Step = Callable[[], Awaitable[None]]


def sdk_surface(api: FakeApi, task: Path, facts: Facts) -> tuple[dict[str, Step], Callable]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(api))
    client = AsyncSynthClient("fixture", base_url="https://api.example.test", http_client=http)
    swarms = client.swarms
    files = []

    async def discover_profiles():
        await swarms.profiles()

    async def upload_file():
        files.append(await swarms.upload_file(str(task)))

    async def launch():
        from synth_ai._generated.models import InputFile, RunInputs

        inputs = RunInputs(
            files=[
                InputFile(
                    file_id=f.file_id, revision_id=f.revision_id, digest_sha256=f.digest_sha256
                )
                for f in files
            ]
        )
        await swarms.launch("Find the fault", budget=BUDGET, inputs=inputs)

    async def pause():
        await swarms.pause(RUN_ID)

    async def resume():
        await swarms.resume(RUN_ID)

    async def watch():
        facts.state = (await swarms.wait(RUN_ID, timeout=30, poll_interval=0.01)).state

    async def outputs():
        page = await swarms.outputs(RUN_ID)
        facts.verified_outputs = [
            o.output_id for o in page.items if (o.reference or {}).get("digest_sha256")
        ]

    async def answer():
        facts.answer = (await swarms.result(RUN_ID)).content

    async def usage():
        facts.usage_records = len((await client.account.usage_records(run_id=RUN_ID)).items)

    steps = dict(
        discover_profiles=discover_profiles,
        upload_file=upload_file,
        launch=launch,
        pause=pause,
        resume=resume,
        watch=watch,
        outputs=outputs,
        answer=answer,
        usage=usage,
    )
    return steps, client.close


def cli_surface(api: FakeApi, task: Path, facts: Facts, capsys) -> dict[str, Step]:
    async def synth(*argv: str) -> dict | list:
        # The CLI owns its event loop (asyncio.run), so it runs on a worker thread here.
        code = await asyncio.to_thread(
            lambda: cli.main(
                [*argv, "--json"],
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
            )
        )
        captured = capsys.readouterr()
        assert code == cli.EXIT_OK, captured.err
        return json.loads(captured.out)

    file_ids: list[str] = []

    async def discover_profiles():
        await synth("profiles")

    async def upload_file():
        file_ids.extend(f["file_id"] for f in await synth("upload", str(task)))

    async def launch():
        await synth("run", "Find the fault", "--budget", BUDGET, "--file", str(task))

    async def pause():
        await synth("pause", RUN_ID)

    async def resume():
        await synth("resume", RUN_ID)

    async def watch():
        facts.state = (await synth("wait", RUN_ID, "--timeout", "30"))["state"]

    async def outputs():
        page = await synth("outputs", RUN_ID)
        facts.verified_outputs = [
            o["output_id"] for o in page["items"] if o["reference"].get("digest_sha256")
        ]

    async def answer():
        facts.answer = (await synth("result", RUN_ID))["content"]

    async def usage():
        facts.usage_records = len((await synth("usage", RUN_ID))["items"])

    return dict(
        discover_profiles=discover_profiles,
        upload_file=upload_file,
        launch=launch,
        pause=pause,
        resume=resume,
        watch=watch,
        outputs=outputs,
        answer=answer,
        usage=usage,
    )


def mcp_surface(api: FakeApi, task: Path, facts: Facts) -> tuple[dict[str, Step], Callable]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(api))
    client = AsyncSynthClient("fixture", base_url="https://api.example.test", http_client=http)
    listed = {tool["name"] for tool in TOOLS}
    keys = iter(range(1000))
    files: list[dict] = []

    async def tool(name: str, **arguments) -> dict:
        assert name in listed, "stdio MCP server does not list " + name
        return json.loads(await call_tool(client, name, arguments))

    def key() -> str:
        return "journey-" + str(next(keys))

    async def discover_profiles():
        await tool("swarms_profiles_list")

    async def upload_file():
        body = {"name": task.name, "media_type": "text/markdown", "content": task.read_text()}
        files.append(await tool("swarms_upload_file", body=body, idempotency_key=key()))

    async def launch():
        body = {
            "message": "Find the fault",
            "limits": {"maximum_charge": {"amount": BUDGET, "currency": "USD"}},
            "inputs": {
                "files": [
                    {k: f[k] for k in ("file_id", "revision_id", "digest_sha256")} for f in files
                ]
            },
        }
        await tool("swarms_run_launch", body=body, idempotency_key=key())

    async def pause():
        await tool("swarms_run_pause", run_id=RUN_ID, idempotency_key=key())

    async def resume():
        await tool("swarms_run_resume", run_id=RUN_ID, idempotency_key=key())

    async def watch():
        while (waited := await tool("swarms_run_wait", run_id=RUN_ID, timeout_seconds=1))[
            "outcome"
        ] != "terminal":
            pass
        facts.state = waited["run"]["state"]

    async def outputs():
        page = await tool("swarms_run_outputs", run_id=RUN_ID)
        facts.verified_outputs = [
            o["output_id"] for o in page["items"] if o["reference"].get("digest_sha256")
        ]

    async def answer():
        # MCP returns output metadata; the content is fetched by digest outside the tool.
        output = await tool("swarms_run_output_get", run_id=RUN_ID, output_id=OUTPUT_ID)
        facts.answer = output["reference"]["digest_sha256"]

    async def usage():
        facts.usage_records = len((await tool("usage_for_run", run_id=RUN_ID))["items"])

    steps = dict(
        discover_profiles=discover_profiles,
        upload_file=upload_file,
        launch=launch,
        pause=pause,
        resume=resume,
        watch=watch,
        outputs=outputs,
        answer=answer,
        usage=usage,
    )
    return steps, client.close


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_STATE_DIR", str(tmp_path / "state"))
    task = tmp_path / "task.md"
    task.write_bytes(TASK)
    return task


@pytest.mark.parametrize("surface", ["sdk", "cli", "mcp"])
async def test_the_same_journey_on_every_surface(contract, env, capsys, surface):
    api, facts = FakeApi(contract), Facts()
    close = None
    if surface == "sdk":
        steps, close = sdk_surface(api, env, facts)
    elif surface == "cli":
        steps = cli_surface(api, env, facts, capsys)
    else:
        steps, close = mcp_surface(api, env, facts)

    # Parity: every journey step exists on this surface.
    missing = [step for step in STEPS if step not in steps]
    assert not missing, surface + " is missing journey steps: " + ", ".join(missing)

    try:
        for step, expected_by_surface in STEPS.items():
            before = len(api.calls)
            await steps[step]()
            expected = expected_by_surface.get(surface, expected_by_surface.get("*", set()))
            reached = set(api.calls[before:])
            assert expected <= reached, (surface, step, sorted(reached))
    finally:
        if close is not None:
            await close()

    # The same final facts on every surface.
    assert facts.state == "completed"
    assert facts.verified_outputs == [OUTPUT_ID]
    assert facts.usage_records == 1
    launch = api.launches[-1]
    assert launch["message"] == "Find the fault"
    assert launch["limits"]["maximum_charge"]["amount"] == BUDGET
    assert [f["file_id"] for f in launch["inputs"]["files"]] == [api.manifest()["file_id"]]
    if surface == "mcp":
        assert api.waits == 2  # one bounded-wait timeout, then terminal
        assert facts.answer == hashlib.sha256(ANSWER).hexdigest()
    else:
        assert api.stream_opens == [None, "1"]  # one retryable drop, resumed after event 1
        assert facts.answer == ANSWER.decode()
