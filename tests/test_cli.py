"""The `synth` task-verb CLI against an in-memory HTTP boundary."""

import hashlib
import json
import re
import stat
import subprocess
import sys

import httpx
import pytest

from synth_ai import cli
from synth_ai._generated.operations import OPERATIONS

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"
EXPECTED_COMMANDS = [
    "login",
    "whoami",
    "logout",
    "profiles",
    "upload",
    "run",
    "status",
    "wait",
    "logs",
    "tasks",
    "ls",
    "pause",
    "resume",
    "cancel",
    "outputs",
    "download",
    "report",
    "result",
    "usage",
    "billing",
    "limits",
]


def example(contract, operation_id):
    op = OPERATIONS[operation_id]
    responses = contract["paths"][op["path"]][op["method"].lower()]["responses"]
    status = next(code for code in responses if code.startswith("2"))
    return int(status), dict(responses[status]["content"]["application/json"]["example"])


def route(request):
    for oid, op in OPERATIONS.items():
        pattern = "^" + re.sub(r"\{[^}]+\}", "[^/]+", op["path"]) + "$"
        if request.method == op["method"].upper() and re.match(pattern, request.url.path):
            return oid
    raise AssertionError("unrouted " + request.method + " " + request.url.path)


class Api:
    """Contract examples per operation, with per-test overrides."""

    def __init__(self, contract):
        self.contract = contract
        self.calls = []
        self.overrides = {}
        self.events = []

    def __call__(self, request):
        if request.method == "PUT":
            return httpx.Response(200)
        oid = route(request)
        self.calls.append((oid, request))
        if oid == "listRunEvents" and request.headers.get("Accept") == "text/event-stream":
            last = request.headers.get("Last-Event-ID")
            start = 0 if last is None else int(last)
            body = "".join(
                "id: " + str(e["sequence"]) + "\ndata: " + json.dumps(e) + "\n\n"
                for e in self.events[start:]
            )
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=body.encode()
            )
        override = self.overrides.get(oid)
        if callable(override):
            return override(request)
        if override is not None:
            return httpx.Response(override[0], json=override[1])
        status, body = example(self.contract, oid)
        return httpx.Response(status, json=body)


@pytest.fixture
def api(contract, monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_STATE_DIR", str(tmp_path / "state"))
    return Api(contract)


def invoke(api, capsys, *argv):
    http = httpx.AsyncClient(transport=httpx.MockTransport(api))
    code = cli.main(list(argv), http_client=http)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def snapshot_in(contract, state):
    _, body = example(contract, "getRun")
    body.update(state=state, execution_closed=True, accounting_final=True, cleanup_complete=True)
    return body


def test_help_lists_exactly_the_task_verbs():
    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    listed = re.findall(r"^    (\S+)\s", result.stdout, re.MULTILINE)
    assert listed == EXPECTED_COMMANDS
    for code in ("0", "1", "2", "3", "4"):
        assert re.search("^  " + code + "  ", result.stdout, re.MULTILINE)
    assert list(cli.COMMANDS) == EXPECTED_COMMANDS
    for removed in ("swarms", "api-keys", "account", "auth"):
        assert removed not in cli.COMMANDS


@pytest.mark.parametrize(
    "argv,oid,human",
    [
        (["status", RUN_ID], "getRun", "state"),
        (["tasks", RUN_ID], "listRunTasks", "TASK"),
        (["ls"], "listRuns", "RUN"),
        (["logs", RUN_ID], "listRunEvents", ""),
        (["pause", RUN_ID], "pauseRun", "command"),
        (["resume", RUN_ID], "resumeRun", "command"),
        (["cancel", RUN_ID], "cancelRun", "state"),
        (["outputs", RUN_ID], "listRunOutputs", "OUTPUT"),
        (["report", RUN_ID], "getRunReport", "settlement"),
        (["profiles"], "listProfiles", "PROFILE"),
        (["usage"], "getUsageSummary", "charge"),
        (["usage", RUN_ID], "listUsageRecords", "RECORD"),
        (["billing"], "getBalance", "available"),
        (["limits"], "listLimits", "plan"),
        (["whoami"], "getAccount", "org"),
    ],
)
def test_command_human_and_json_output(api, capsys, contract, argv, oid, human):
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_OK, err
    assert human in out and not out.lstrip().startswith("{")
    assert api.calls[-1][0] == oid
    code, out, err = invoke(api, capsys, *argv, "--json")
    assert code == cli.EXIT_OK, err
    document = json.loads(out)
    _, served = example(contract, oid)
    assert document.keys() <= served.keys() | {"items"}
    assert len(out.strip().splitlines()) == 1


def test_run_requires_a_goal(api, capsys):
    code, out, err = invoke(api, capsys, "run", "--budget", "1.00")
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("usage_error: ")
    assert api.calls == []


def test_run_uploads_files_labels_and_prints_the_run_id(api, capsys, contract, tmp_path):
    content = b"# Notes\n"
    path = tmp_path / "notes.md"
    path.write_bytes(content)
    _, manifest = example(contract, "uploadFile")
    manifest.update(size_bytes=len(content), digest_sha256=hashlib.sha256(content).hexdigest())
    pending = {
        "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
        "state": "pending",
        "file": manifest,
        "put": {
            "url": "https://storage.example.test/object",
            "headers": {"content-type": "text/markdown"},
            "expires_at": "2026-11-06T00:00:00Z",
        },
    }
    api.overrides["createUpload"] = (201, pending)
    api.overrides["finalizeUpload"] = (200, {**pending, "state": "committed", "put": None})
    code, out, err = invoke(
        api,
        capsys,
        *["run", "Find the fault", "--budget", "2.00", "--file", str(path)],
        *["--label", "team=core", "--wall", "600"],
    )
    assert code == cli.EXIT_OK, err
    _, created = example(contract, "createRun")
    assert out == created["run_id"] + "\n"
    launch = json.loads(next(r for oid, r in api.calls if oid == "createRun").content)
    assert launch["message"] == "Find the fault"
    assert launch["labels"] == {"team": "core"}
    assert launch["limits"]["wall_time_seconds"] == 600
    assert launch["limits"]["maximum_charge"]["amount"] == "2.00"
    assert launch["inputs"]["files"] == [
        {k: manifest[k] for k in ("file_id", "revision_id", "digest_sha256")}
    ]


def test_upload_is_one_step(api, capsys, contract, tmp_path):
    content = b"{}"
    path = tmp_path / "data.json"
    path.write_bytes(content)
    _, manifest = example(contract, "uploadFile")
    manifest.update(size_bytes=len(content), digest_sha256=hashlib.sha256(content).hexdigest())
    committed = {"upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA", "state": "committed", "file": manifest}
    api.overrides["createUpload"] = (201, committed)
    code, out, err = invoke(api, capsys, "upload", str(path), "--json")
    assert code == cli.EXIT_OK, err
    assert [item["file_id"] for item in json.loads(out)] == [manifest["file_id"]]
    assert [oid for oid, _ in api.calls] == ["createUpload"]
    code, out, _ = invoke(api, capsys, "upload", str(path))
    assert out.splitlines()[0].split() == ["FILE", "NAME", "BYTES"]


@pytest.mark.parametrize(
    "state,expected",
    [("completed", cli.EXIT_OK), ("failed", cli.EXIT_RUN_UNSUCCESSFUL), ("cancelled", 3)],
)
def test_wait_exit_codes(api, capsys, contract, state, expected):
    api.overrides["getRun"] = (200, snapshot_in(contract, state))
    code, out, err = invoke(api, capsys, "wait", RUN_ID, "--timeout", "5", "--json")
    assert code == expected, err
    assert json.loads(out)["state"] == state
    code, out, _ = invoke(
        api, capsys, "run", "Goal", "--budget", "1.00", "--wait", "--timeout", "5"
    )
    assert code == expected
    assert out.splitlines()[-1] == "state  " + state


def test_wait_timeout_exits_two(api, capsys, contract):
    _, running = example(contract, "getRun")
    running.update(state="running", execution_closed=False)
    api.overrides["getRun"] = (200, running)
    code, out, err = invoke(api, capsys, "wait", RUN_ID, "--timeout", "0.2")
    assert code == cli.EXIT_WAIT_TIMEOUT
    assert out == "" and err.startswith("wait_timeout: ")


def test_server_refusal_exits_four_with_problem(api, capsys, contract):
    problem = {
        "type": "about:blank",
        "title": "Not found",
        "status": 404,
        "detail": "run not found",
        "instance": "urn:request",
        "code": "run_not_found",
        "service": "core",
        "stage": "getRun",
        "outcome": "not_accepted",
    }
    api.overrides["getRun"] = (404, problem)
    code, out, err = invoke(api, capsys, "status", RUN_ID)
    assert code == cli.EXIT_SERVER_REFUSAL and out == ""
    assert err == "run_not_found: run not found\n"
    code, out, err = invoke(api, capsys, "status", RUN_ID, "--json")
    assert code == cli.EXIT_SERVER_REFUSAL and out == ""
    assert json.loads(err)["code"] == "run_not_found"


def test_logs_follow_resumes_from_checkpoint_without_duplicates(api, capsys, contract, tmp_path):
    api.overrides["getRun"] = (200, snapshot_in(contract, "completed"))

    def event(sequence):
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

    api.events = [event(1), event(2)]
    code, first, err = invoke(api, capsys, "logs", RUN_ID, "-f")
    assert code == cli.EXIT_OK, err
    assert [line.split()[0] for line in first.splitlines()] == ["1", "2"]
    stored = tmp_path / "state" / "streams" / (RUN_ID + ".json")
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert json.loads(stored.read_text())["last_event_id"] == "2"
    api.events.append(event(3))
    code, second, err = invoke(api, capsys, "logs", RUN_ID, "-f", "--json")
    assert code == cli.EXIT_OK, err
    assert [json.loads(line)["sequence"] for line in second.splitlines()] == [3]
    resumed = [r for oid, r in api.calls if oid == "listRunEvents"]
    assert resumed[-1].headers["Last-Event-ID"] == "2"


def test_download_writes_verified_outputs(api, capsys, contract, tmp_path):
    content = b"hello"
    item = {
        "run_id": RUN_ID,
        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
        "state": "available",
        "filename": "../hello.txt",
        "reference": {
            "revision_id": "rev",
            "digest_sha256": hashlib.sha256(content).hexdigest(),
        },
    }
    api.overrides["listRunOutputs"] = (200, {"items": [item], "has_more": False})
    api.overrides["getRunOutputContent"] = lambda _r: httpx.Response(
        200, content=content, headers={"Content-Type": "application/octet-stream"}
    )
    target = tmp_path / "out"
    code, out, err = invoke(api, capsys, "download", RUN_ID, "-o", str(target), "--json")
    assert code == cli.EXIT_OK, err
    assert (target / "hello.txt").read_bytes() == content
    assert json.loads(out)[0]["path"] == str(target / "hello.txt")
    code, _, err = invoke(api, capsys, "download", RUN_ID, "out_missing", "-o", str(target))
    assert code == cli.EXIT_CLIENT_ERROR and "not found" in err


def serve_answer(api, contract, text):
    content = text.encode()
    item = {
        "run_id": RUN_ID,
        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
        "state": "available",
        "filename": "answer.md",
        "reference": {
            "revision_id": "rev",
            "digest_sha256": hashlib.sha256(content).hexdigest(),
            "media_type": "text/markdown",
        },
    }
    api.overrides["getRun"] = (200, snapshot_in(contract, "completed"))
    api.overrides["listRunOutputs"] = (200, {"items": [item], "has_more": False})
    api.overrides["getRunOutputContent"] = lambda _r: httpx.Response(
        200, content=content, headers={"Content-Type": "application/octet-stream"}
    )


def test_result_prints_the_answer_or_the_run_result(api, capsys, contract):
    serve_answer(api, contract, "The answer is 42.")
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "The answer is 42.\n"
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    document = json.loads(out)
    assert document["run"]["state"] == "completed"
    assert document["content"] == "The answer is 42."
    assert document["answer"]["output_id"] == document["outputs"][0]["output_id"]
    assert len(out.strip().splitlines()) == 1


def test_result_without_a_text_answer_says_so(api, capsys, contract):
    api.overrides["getRun"] = (200, snapshot_in(contract, "completed"))
    api.overrides["listRunOutputs"] = (200, {"items": [], "has_more": False})
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert "answer" in out and "none" in out


def test_result_on_an_unfinished_run_exits_one_with_its_code(api, capsys, contract):
    _, running = example(contract, "getRun")
    running.update(state="running", execution_closed=False)
    api.overrides["getRun"] = (200, running)
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("result_unavailable: ")


def test_run_wait_prints_the_answer_last(api, capsys, contract):
    serve_answer(api, contract, "Done: shipped.")
    code, out, err = invoke(api, capsys, "run", "Goal", "--budget", "1.00", "--wait")
    assert code == cli.EXIT_OK, err
    assert out.splitlines()[-2:] == ["state  completed", "Done: shipped."]
    code, out, err = invoke(api, capsys, "run", "Goal", "--budget", "1.00", "--wait", "--json")
    assert code == cli.EXIT_OK, err
    assert json.loads(out)["state"] == "completed"


def test_no_browser_login_accepts_the_pasted_redirect():
    from synth_ai.auth import callback_result

    issuer = "https://mcp.example.test"
    pasted = "http://127.0.0.1:5555/callback?code=abc&state=s1&iss=" + issuer + "/\n"
    assert callback_result(pasted, state="s1", issuer=issuer) == {"code": "abc"}
    assert callback_result(pasted, state="other", issuer=issuer) is None
    forged = pasted.replace("127.0.0.1:5555", "evil.example.test")
    assert callback_result(forged, state="s1", issuer=issuer) is None
