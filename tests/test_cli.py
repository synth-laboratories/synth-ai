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
    "content",
    "labels",
    "files",
    "uploads",
    "keys",
    "organization",
    "inspect",
    "families",
    "runs",
    "login",
    "whoami",
    "logout",
    "profiles",
    "choices",
    "upload",
    "run",
    "status",
    "wait",
    "logs",
    "message",
    "messages",
    "tasks",
    "ls",
    "pause",
    "resume",
    "cancel",
    "outputs",
    "download",
    "report",
    "allocation",
    "result",
    "usage",
    "billing",
    "limits",
    "plan",
    "grants",
    "promo",
    "artifacts",
    "experiments",
    "research-log",
    "notebooks",
    "artifactory",
    "version",
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
        if request.method == "PUT" and request.url.host != "api.example.test":
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


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_package_version_and_exits_zero(flag, capsys):
    from synth_ai import __version__

    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", flag], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"synth-ai {__version__}"
    assert result.stderr == ""

    code = cli.main([flag])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == f"synth-ai {__version__}"
    assert captured.err == ""


@pytest.mark.parametrize(
    ("argv", "oid", "human"),
    [
        (["status", RUN_ID], "getRun", "state"),
        (["tasks", RUN_ID], "listRunTasks", "TASK"),
        (["ls"], "listRuns", "RUN"),
        (["logs", RUN_ID], "listRunEvents", ""),
        (["pause", RUN_ID], "pauseRun", "command"),
        (["resume", RUN_ID], "resumeRun", "command"),
        (["cancel", RUN_ID], "cancelRun", "state"),
        (["outputs", RUN_ID], "listRunOutputs", "OUTPUT"),
        (["message", RUN_ID, "fixture"], "sendRunMessage", "customer"),
        (["messages", RUN_ID], "listRunMessages", "messaging_unavailable"),
        (
            ["messages", RUN_ID, "--message", "msg_AAAAAAAAAAAAAAAAAAAAAA"],
            "getRunMessage",
            "customer",
        ),
        (["report", RUN_ID], "getRunReport", "settlement"),
        (["allocation", RUN_ID], "getRunAllocation", "menu revision"),
        (["choices"], "getLaunchChoices", "SANDBOX"),
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


@pytest.mark.parametrize(
    "argv",
    [
        ("index", "search", "q"),
        ("--json", "index"),
        ("index",),
        ("--env", "local", "index", "--json"),
    ],
)
def test_index_is_a_typed_local_refusal(api, capsys, argv):
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    if "--json" in argv:
        assert json.loads(err)["code"] == "index_not_available"
    else:
        assert err.startswith("index_not_available: ")
    assert api.calls == []


def test_index_as_an_option_value_stays_a_usage_error(api, capsys):
    code, _, err = invoke(api, capsys, "--base-url", "index")
    assert code == cli.EXIT_CLIENT_ERROR and err.startswith("usage_error: ")


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
        *["--label", "team=core", "--wall", "600", "--parallel-workers", "4"],
    )
    assert code == cli.EXIT_OK, err
    _, created = example(contract, "createRun")
    assert out == created["run_id"] + "\n"
    launch = json.loads(next(r for oid, r in api.calls if oid == "createRun").content)
    assert launch["message"] == "Find the fault"
    assert launch["labels"] == {"team": "core"}
    assert launch["limits"]["wall_time_seconds"] == 600
    assert launch["limits"]["parallel_workers"] == 4
    assert launch["limits"]["maximum_charge"]["amount"] == "2.00"
    assert launch["inputs"]["files"] == [
        {k: manifest[k] for k in ("file_id", "revision_id", "digest_sha256")}
    ]


@pytest.mark.parametrize(
    "flags",
    [
        ["--parallel-workers", "33"],
        ["--parallel-workers", "0"],
        ["--wall", "86401"],
        ["--wall", "0"],
        ["--label", "no-equals-sign"],
        ["--budget", "-1"],
        ["--idempotency-key", ""],
        ["--idempotency-key", "x" * 129],
    ],
)
def test_run_refuses_bad_flags_before_any_upload(api, capsys, tmp_path, flags):
    path = tmp_path / "notes.md"
    path.write_bytes(b"# Notes\n")
    argv = ["run", "Find the fault", "--budget", "2.00", "--file", str(path), *flags]
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_CLIENT_ERROR and out == "", err
    assert api.calls == []


RECORDED_CRAFTAX_RESOURCES = [
    {
        "plugin": "pool",
        "version": 1,
        "roles": ["planner", "worker"],
        "limits": {"rollouts": 90, "spend_micros": 11000},
        "selection": {},
        "evaluation_templates": [
            "sha256:5c6763bbbb0a481d071ccd340b02133e2a4907c3b1088d16c87bd1d9d40e6007"
        ],
    }
]


def test_run_sends_recorded_resources_exactly(api, capsys, tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(RECORDED_CRAFTAX_RESOURCES))
    argv = ["run", "Goal", "--budget", "10.00", "--resources", str(path)]
    code, _, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_OK, err
    launch = json.loads(next(r for oid, r in api.calls if oid == "createRun").content)
    assert launch["resources"] == RECORDED_CRAFTAX_RESOURCES


@pytest.mark.parametrize(
    "document",
    [
        "{}",
        "not json",
        json.dumps([{**RECORDED_CRAFTAX_RESOURCES[0], "plugin": "other"}]),
        json.dumps([{**RECORDED_CRAFTAX_RESOURCES[0], "local_cache": {"mode": "replay"}}]),
    ],
)
def test_run_refuses_bad_resources_before_any_call(api, capsys, tmp_path, document):
    path = tmp_path / "resources.json"
    path.write_text(document)
    upload = tmp_path / "notes.md"
    upload.write_bytes(b"# Notes\n")
    argv = ["run", "Goal", "--budget", "1.00", "--file", str(upload), "--resources", str(path)]
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_CLIENT_ERROR and out == "", err
    assert err.startswith("client_error: --resources ")
    assert api.calls == []


@pytest.mark.parametrize("as_json", [False, True])
def test_run_refuses_invalid_optimizer_in_one_line(api, capsys, tmp_path, as_json):
    """SYN-5666: the refusal detail is one line that names the field, not pydantic's dump."""
    upload = tmp_path / "notes.md"
    upload.write_bytes(b"# Notes\n")
    argv = ["run", "Goal", "--budget", "1.00", "--file", str(upload)]
    argv += ["--optimizer", "swarms_v2@0", *(["--json"] if as_json else [])]
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_CLIENT_ERROR and out == "" and api.calls == [], err
    assert err.count("\n") == 1, err
    if as_json:
        problem = json.loads(err)
        assert problem["code"] == "client_error", err
        detail = problem["detail"]
    else:
        assert err.startswith("client_error: "), err
        detail = err.removeprefix("client_error: ")
    assert "\n" not in detail.strip() and "--optimizer" in detail, err
    assert "version" in detail and "swarms_v2@0" in detail, err


def test_run_refuses_unreadable_resources_file(api, capsys, tmp_path):
    argv = ["run", "Goal", "--budget", "1.00", "--resources", str(tmp_path / "absent.json")]
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_CLIENT_ERROR and out == "", err
    assert api.calls == []


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
    ("state", "expected"),
    [("completed", cli.EXIT_OK), ("failed", cli.EXIT_RUN_UNSUCCESSFUL), ("cancelled", 3)],
)
def test_wait_exit_codes(api, capsys, contract, state, expected):
    serve_result(api, contract, None)
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
    assert code == cli.EXIT_WAIT_TIMEOUT, err
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


def serve_result(
    api, contract, answer, *, media_type="text/markdown", objective=None, result_file=None
):
    """Serve getRunResult from whatever getRun currently returns (so charge edits apply)."""
    api.overrides["getRun"] = (200, snapshot_in(contract, "completed"))
    source = {
        "run_id": RUN_ID,
        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
        "state": "available",
        "filename": "answer.md",
        "reference": {"revision_id": "rev", "digest_sha256": "0" * 64, "media_type": media_type},
    }

    def result(_request):
        run = api.overrides["getRun"][1]
        return httpx.Response(
            200,
            json={
                "run_id": run["run_id"],
                "status": run["state"],
                "accounting": None,
                "run": run,
                "state": run["state"],
                "answer": answer,
                "answer_output": None if answer is None else source,
                "objective_answer": objective,
                "charge": run.get("charge"),
                "attempt_tree": {
                    "state": "unavailable",
                    "run_id": run["run_id"],
                    "code": "run_records_projection_unavailable",
                    "stage": "run_result",
                    "cause": "Canonical research record projection is unavailable for this cohort.",
                },
                "result_file": result_file,
            },
        )

    api.overrides["getRunResult"] = result


def test_result_names_result_md_on_stderr(api, capsys, contract):
    result_md = {
        "run_id": RUN_ID,
        "output_id": "out_BBBBBBBBBBBBBBBBBBBBBB",
        "state": "available",
        "filename": "result.md",
        "reference": {
            "revision_id": "rev",
            "digest_sha256": "1" * 64,
            "byte_size": "12",
            "media_type": "text/markdown",
        },
    }
    serve_result(api, contract, "The answer is 42.", result_file=result_md)
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "The answer is 42.\n"
    assert "result.md  out_BBBBBBBBBBBBBBBBBBBBBB" in err
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    assert json.loads(out)["result_file"]["output_id"] == "out_BBBBBBBBBBBBBBBBBBBBBB"


def serve_answer(api, contract, text):
    serve_result(api, contract, text)


def test_result_prints_the_answer_or_the_run_result(api, capsys, contract):
    serve_answer(api, contract, "The answer is 42.")
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "The answer is 42.\n"
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    document = json.loads(out)
    assert document["run"]["state"] == "completed" and document["state"] == "completed"
    assert document["answer"] == "The answer is 42."
    assert document["answer_output"]["output_id"] == "out_AAAAAAAAAAAAAAAAAAAAAA"
    assert document["objective_answer"] is None
    assert document["attempt_tree"]["state"] == "unavailable"
    assert document["attempt_tree"]["code"] == "run_records_projection_unavailable"
    assert len(out.strip().splitlines()) == 1


def serve_objective_answer(api, contract, text):
    objective = {
        "schema_version": "synth.objective-answer.v1",
        "objective_id": "codename",
        "definition_revision": 1,
        "answer": {"text": text, "by": "planner", "revision": 1, "sources": []},
    }
    serve_result(api, contract, text, media_type="application/json", objective=objective)


def test_result_prints_an_objective_answer(api, capsys, contract):
    serve_objective_answer(api, contract, "nonce-6b98")
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "nonce-6b98\n"
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    document = json.loads(out)
    assert document["answer"] == "nonce-6b98"
    assert document["objective_answer"]["objective_id"] == "codename"


def test_billing_human_renderers_and_checkout_flags(api, capsys):
    code, out, err = invoke(api, capsys, "billing", "balance")
    assert code == cli.EXIT_OK, err
    assert "available" in out and "weekly allowance" in out and "payg_credits" in out
    assert not out.lstrip().startswith("{")

    code, out, err = invoke(api, capsys, "billing", "plans")
    assert code == cli.EXIT_OK, err
    assert "PLAN" in out and "PACK" in out and "custom" in out
    assert not out.lstrip().startswith("{")

    code, out, err = invoke(api, capsys, "billing", "checkout", "--pack", "credits.25", "--json")
    assert code == cli.EXIT_OK, err
    oid, req = api.calls[-1]
    assert oid == "createCheckout"
    assert json.loads(req.content) == {"pack_id": "credits.25"}

    code, out, err = invoke(api, capsys, "billing", "checkout", "--amount", "40.00", "--json")
    assert code == cli.EXIT_OK, err
    oid, req = api.calls[-1]
    assert oid == "createCheckout"
    assert json.loads(req.content) == {"credit": {"amount": "40.00", "currency": "USD"}}

    code, out, err = invoke(
        api, capsys, "billing", "change-plan", "--plan", "max.monthly", "--json"
    )
    assert code == cli.EXIT_OK, err
    oid, req = api.calls[-1]
    assert oid == "changeSubscriptionPlan"
    assert json.loads(req.content) == {"plan_id": "max.monthly"}

    code, out, err = invoke(api, capsys, "billing", "set-spend-alerts", "--clear", "--json")
    assert code == cli.EXIT_OK, err
    oid, req = api.calls[-1]
    assert oid == "setSpendAlerts"
    assert json.loads(req.content) == {"monthly_amount": None}


def test_usage_accepts_date_and_page_flags(api, capsys):
    code, out, err = invoke(
        api, capsys, "usage", "--since", "2026-11-01T00:00:00Z", "--until", "2026-11-27T00:00:00Z"
    )
    assert code == cli.EXIT_OK, err
    assert "weekly allowance" in out and "available" in out
    oid, req = api.calls[-1]
    assert oid == "getUsageSummary"
    assert req.url.params["since"] == "2026-11-01T00:00:00Z"
    assert req.url.params["until"] == "2026-11-27T00:00:00Z"


def test_run_help_hides_deprecated_tokens_flag():
    result = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "run", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--tokens" not in result.stdout
