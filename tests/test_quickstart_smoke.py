"""Safety and surface checks for the opt-in quickstart qualification."""

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from scripts import quickstart_smoke as smoke
from synth_ai.mcp.tools import TOOLS


def test_receipt_round_trip_and_spend_boundary(tmp_path):
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    output = tmp_path / "receipt.json"
    receipt.save(output)
    restored = smoke.Receipt.load(output)
    restored.validate_intent("https://api.example.test", "fixture-org", "a" * 64)
    assert restored.aggregate_maximum_usd == "12.00"
    assert [arm.surface for arm in restored.arms] == ["sdk", "cli", "mcp"]
    assert not restored.journey_passed
    assert not restored.provider_invoice_reconciled
    assert restored.cohort_sha256 == "a" * 64
    assert oct(output.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError, match="target or surface"):
        restored.validate_intent("https://other.example.test", "fixture-org", "a" * 64)
    for amount in ("-0.01", "4.01", "NaN", "invalid"):
        with pytest.raises(ValueError):
            smoke.check_charge(amount)
    smoke.check_charge("4.00")


def test_receipt_rejects_false_pass_or_modified_mutation_identity():
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    receipt.arms[0].status = "passed"
    receipt.arms[0].run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    with pytest.raises(ValueError, match="lacks finality"):
        receipt.validate_intent(receipt.base_url, receipt.fixture_org_id, receipt.cohort_sha256)
    receipt.arms[0].status = "not_started"
    receipt.arms[0].run_id = None
    receipt.arms[1].launch_key = "different-key"
    with pytest.raises(ValueError, match="mutation keys"):
        receipt.validate_intent(receipt.base_url, receipt.fixture_org_id, receipt.cohort_sha256)


async def test_mcp_launch_arguments_match_real_tool_schemas(tmp_path, monkeypatch):
    task = tmp_path / "task.md"
    task.write_text("# Task\nA retained nonce\n")
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    arm = receipt.arms[2]
    observed = []
    schemas = {tool["name"]: tool["inputSchema"] for tool in TOOLS}

    async def call(_session, name, arguments):
        jsonschema.Draft202012Validator(schemas[name]).validate(arguments)
        observed.append((name, arguments))
        if name == "swarms_upload_file":
            return {
                "file_id": "fil_AAAAAAAAAAAAAAAAAAAAAA",
                "revision_id": "rev_AAAAAAAAAAAAAAAAAAAAAA",
                "digest_sha256": "a" * 64,
            }
        return {"run_id": "run_AAAAAAAAAAAAAAAAAAAAAA"}

    monkeypatch.setattr(smoke, "mcp_call", call)
    run_id = await smoke.launch(None, None, arm, task, "Follow the uploaded task")
    assert run_id == "run_AAAAAAAAAAAAAAAAAAAAAA"
    assert [name for name, _ in observed] == ["swarms_upload_file", "swarms_run_launch"]
    assert observed[0][1]["idempotency_key"] == arm.upload_key
    assert observed[1][1]["idempotency_key"] == arm.launch_key
    assert observed[1][1]["body"]["limits"]["maximum_charge"]["amount"] == "4.00"


async def test_uncertain_launch_reuses_intent_before_any_other_arm(tmp_path, monkeypatch):
    import mcp
    import mcp.client.stdio

    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    output = tmp_path / "receipt.json"
    receipt.save(output)
    calls = []
    run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"

    class Context:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *_args):
            return False

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            pass

    class Swarms:
        async def get(self, _run_id):
            return SimpleNamespace(
                run_id=run_id,
                state="completed",
                execution_closed=True,
                accounting_final=True,
                cleanup_complete=True,
                finished=True,
                charge=None,
            )

    class Client:
        def __init__(self):
            self.swarms = Swarms()
            self.account = SimpleNamespace(get=self.account_get)

        async def account_get(self):
            return SimpleNamespace(org=SimpleNamespace(id="fixture-org"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def launch(_client, _session, arm, _task, _goal):
        calls.append((arm.surface, arm.launch_key))
        if len(calls) == 1:
            raise RuntimeError("response lost after admission")
        return run_id

    async def verify(_client, _session, _arm, _nonce, _path):
        return "0.10", True, "out_AAAAAAAAAAAAAAAAAAAAAA", "a" * 64, "fixture-output"

    monkeypatch.setattr(smoke, "AsyncSynthClient", Client)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", lambda *_a, **_kw: Context((None, None)))
    monkeypatch.setattr(smoke, "ClientSession", Session)
    monkeypatch.setattr(smoke, "launch", launch)
    monkeypatch.setattr(smoke, "verify_surface", verify)
    with pytest.raises(RuntimeError, match="response lost"):
        await smoke.execute(receipt, output)
    saved = smoke.Receipt.load(output)
    assert saved.arms[0].status == "launch_pending"
    assert saved.arms[0].run_id is None
    assert saved.arms[1].status == saved.arms[2].status == "not_started"
    assert calls == [("sdk", saved.arms[0].launch_key)]

    # Stop after the recovered first arm: duplicate IDs must not advance a later arm.
    async def second_launch(_client, _session, arm, _task, _goal):
        calls.append((arm.surface, arm.launch_key))
        if arm.surface == "sdk":
            return run_id
        raise RuntimeError("stop before next admission")

    monkeypatch.setattr(smoke, "launch", second_launch)
    with pytest.raises(RuntimeError, match="stop before next admission"):
        await smoke.execute(saved, output)
    assert calls[0] == calls[1]
    final = smoke.Receipt.load(output)
    assert final.arms[0].run_id == run_id and final.arms[0].status == "passed"
    assert final.arms[1].status == "launch_pending" and final.arms[1].run_id is None
    assert final.arms[2].status == "not_started"
    assert not final.journey_passed


async def test_verified_answer_requires_a_digest_checked_output(tmp_path):
    import hashlib

    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    arm = receipt.arms[0]
    arm.run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    result = {
        "run": {"run_id": arm.run_id},
        "answer": receipt.nonce,
        "charge": {"amount": {"amount": "0.10", "currency": "USD"}, "final": True},
    }

    class Swarms:
        async def result(self, _run_id):
            return SimpleNamespace(model_dump=lambda **_kwargs: result)

        async def outputs(self, _run_id):
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        output_id="out_AAAAAAAAAAAAAAAAAAAAAA",
                        reference={
                            "revision_id": "rev_test",
                            "digest_sha256": hashlib.sha256(
                                ("answer: " + receipt.nonce).encode()
                            ).hexdigest(),
                        },
                    )
                ]
            )

        async def download_output(self, _run_id, _output_id, filename, **_kwargs):
            Path(filename).write_bytes(("answer: " + receipt.nonce).encode())

    client = SimpleNamespace(swarms=Swarms())
    proof = await smoke.verify_surface(client, None, arm, receipt.nonce, tmp_path / "receipt.json")
    assert proof[:2] == ("0.10", True)
    receipt.nonce = "different"
    with pytest.raises(ValueError, match="answer did not contain"):
        await smoke.verify_surface(client, None, arm, receipt.nonce, tmp_path / "receipt.json")


def test_no_execution_without_explicit_flag(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["quickstart_smoke", "--output", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        smoke.main()
    assert not (tmp_path / "r.json").exists()


async def test_cancel_reuses_retained_key_after_lost_response(tmp_path):
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    arm = receipt.arms[0]
    arm.run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    arm.status = "running"
    output = tmp_path / "receipt.json"
    receipt.save(output)
    calls = []

    class Swarms:
        async def cancel(self, run_id, *, idempotency_key):
            calls.append((run_id, idempotency_key))
            if len(calls) == 1:
                raise RuntimeError("response lost")

        async def get(self, _run_id):
            done = len(calls) >= 2
            return SimpleNamespace(
                run_id=arm.run_id,
                execution_closed=done,
                accounting_final=done,
                cleanup_complete=done,
                charge=SimpleNamespace(amount=SimpleNamespace(amount="0.11"), final=True)
                if done
                else None,
                finished=done,
            )

    client = SimpleNamespace(swarms=Swarms())
    with pytest.raises(RuntimeError, match="response lost"):
        await smoke.cancel_and_reconcile(client, receipt, arm, output)
    saved = smoke.Receipt.load(output)
    assert saved.arms[0].cancel_requested
    saved.validate_intent(receipt.base_url, receipt.fixture_org_id, receipt.cohort_sha256)
    await smoke.cancel_and_reconcile(client, saved, saved.arms[0], output)
    assert calls == [(arm.run_id, arm.cancel_key), (arm.run_id, arm.cancel_key)]
    final = smoke.Receipt.load(output).arms[0]
    assert final.status == "failed" and final.cleanup_complete
    assert final.charge_usd == "0.11" and final.charge_final


async def test_three_surfaces_use_real_local_transports(tmp_path, monkeypatch, contract):
    import hashlib
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from synth_ai._generated.operations import OPERATIONS

    receipt = smoke.new_receipt("http://127.0.0.1", "fixture-org", "a" * 64)
    text = f"# Task\nReply with exactly this token: {receipt.nonce}\n"
    task_bytes = text.encode()
    answer = receipt.nonce.encode()
    digest = hashlib.sha256(answer).hexdigest()
    mutations = []
    runs = {}

    def example(operation):
        op = OPERATIONS[operation]
        responses = contract["paths"][op["path"]][op["method"].lower()]["responses"]
        code = next(value for value in responses if value.startswith("2"))
        return json.loads(json.dumps(responses[code]["content"]["application/json"]["example"]))

    def manifest():
        value = example("uploadFile")
        value.update(
            size_bytes=len(task_bytes), digest_sha256=hashlib.sha256(task_bytes).hexdigest()
        )
        return value

    def snapshot(run_id):
        value = example("getRun")
        value.update(
            run_id=run_id,
            state="completed",
            execution_closed=True,
            accounting_final=True,
            cleanup_complete=True,
            charge={"amount": {"amount": "0.10", "currency": "USD"}, "final": True},
        )
        return value

    def output(run_id):
        return {
            "run_id": run_id,
            "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
            "state": "available",
            "filename": "answer.txt",
            "reference": {"revision_id": "rev_answer", "digest_sha256": digest},
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self):
            path = self.path.split("?", 1)[0]
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            payload = json.loads(raw) if raw else None
            key = self.headers.get("Idempotency-Key")
            if self.command == "GET" and path == "/v1/account":
                account = example("getAccount")
                account["org"]["id"] = "fixture-org"
                return self.json_reply(200, account)
            if self.command == "POST" and path == "/v1/uploads":
                assert payload["digest_sha256"] == hashlib.sha256(task_bytes).hexdigest()
                mutations.append(("upload", key))
                return self.json_reply(
                    201,
                    {
                        "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                        "state": "committed",
                        "file": manifest(),
                    },
                )
            if self.command == "POST" and path == "/v1/files":
                assert payload["content"] == text
                mutations.append(("mcp_upload", key))
                return self.json_reply(201, manifest())
            if self.command == "POST" and path == "/v1/runs":
                mutations.append(("launch", key))
                assert payload["limits"]["maximum_charge"]["amount"] == "4.00"
                assert (
                    payload["inputs"]["files"][0]["digest_sha256"]
                    == hashlib.sha256(task_bytes).hexdigest()
                )
                run_id = runs.setdefault(key, "run_" + chr(ord("A") + len(runs)) * 22)
                return self.json_reply(202, snapshot(run_id))
            match = re.fullmatch(r"/v1/runs/(run_[A-Z]{22})(.*)", path)
            assert match, f"unexpected request: {self.command} {path}"
            run_id, suffix = match.groups()
            if suffix == "":
                return self.json_reply(200, snapshot(run_id))
            if suffix == "/result":
                return self.json_reply(
                    200,
                    {
                        "run": snapshot(run_id),
                        "state": "completed",
                        "answer": receipt.nonce,
                        "answer_output": output(run_id),
                        "objective_answer": None,
                        "charge": snapshot(run_id)["charge"],
                    },
                )
            if suffix == "/outputs":
                return self.json_reply(200, {"items": [output(run_id)], "has_more": False})
            if suffix.endswith("/content"):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)
                return
            raise AssertionError(f"unexpected suffix {suffix}")

        def json_reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = respond
        do_POST = respond

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = f"http://127.0.0.1:{server.server_port}"
    receipt.base_url = target
    path = tmp_path / "receipt.json"
    receipt.save(path)
    monkeypatch.setenv("SYNTH_API_BASE_URL", target)
    monkeypatch.setenv("SYNTH_API_KEY", "test-key")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "absent-auth.json"))
    try:
        await smoke.execute(receipt, path)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    restored = smoke.Receipt.load(path)
    restored.validate_intent(target, "fixture-org", "a" * 64)
    smoke.validate_evidence(restored, path)
    assert restored.journey_passed
    assert not restored.provider_invoice_reconciled
    assert len(runs) == 3
    assert [kind for kind, _ in mutations] == [
        "upload",
        "launch",
        "upload",
        "launch",
        "mcp_upload",
        "launch",
    ]
    assert [arm.charge_usd for arm in restored.arms] == ["0.10"] * 3


async def test_duplicate_run_identity_is_retained_without_cancelling_other_arm(
    tmp_path, monkeypatch
):
    import mcp.client.stdio

    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    prior = receipt.arms[0]
    prior.run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    prior.status = "passed"
    prior.execution_closed = prior.accounting_final = prior.cleanup_complete = True
    prior.charge_usd = "0.10"
    prior.charge_final = prior.answer_verified = True
    prior.output_id, prior.output_sha256, prior.output_file = (
        "out_AAAAAAAAAAAAAAAAAAAAAA",
        "a" * 64,
        "fixture-output",
    )
    output = tmp_path / "receipt.json"
    receipt.save(output)
    cancelled = []

    class Context:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *_args):
            return False

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            pass

    class Swarms:
        async def cancel(self, run_id, **_kwargs):
            cancelled.append(run_id)

    class Client:
        def __init__(self):
            self.swarms = Swarms()
            self.account = SimpleNamespace(get=self.account_get)

        async def account_get(self):
            return SimpleNamespace(org=SimpleNamespace(id="fixture-org"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def duplicate(*_args):
        return prior.run_id

    monkeypatch.setattr(smoke, "AsyncSynthClient", Client)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", lambda *_a, **_kw: Context((None, None)))
    monkeypatch.setattr(smoke, "ClientSession", Session)
    monkeypatch.setattr(smoke, "launch", duplicate)
    with pytest.raises(ValueError, match="duplicate run identities"):
        await smoke.execute(receipt, output)
    assert not cancelled
    saved = smoke.Receipt.load(output)
    assert saved.arms[1].run_id == prior.run_id
    with pytest.raises(ValueError, match="duplicate run identities"):
        saved.validate_intent(saved.base_url, saved.fixture_org_id, saved.cohort_sha256)


def test_target_requires_api_key_and_tls_except_loopback(monkeypatch):
    monkeypatch.setenv("SYNTH_API_KEY", "test-key")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "http://api.example.test")
    with pytest.raises(ValueError, match="require HTTPS"):
        smoke.target_url()
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    assert smoke.target_url() == "https://api.example.test"
    monkeypatch.setenv("SYNTH_API_BASE_URL", "http://127.0.0.1:8092")
    assert smoke.target_url() == "http://127.0.0.1:8092"
    monkeypatch.delenv("SYNTH_API_KEY")
    with pytest.raises(ValueError, match="SYNTH_API_KEY"):
        smoke.target_url()


async def test_interrupted_cli_subprocess_is_reaped(monkeypatch):
    import asyncio

    started = asyncio.Event()

    class Process:
        returncode = None
        terminated = False
        waited = False

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

        def terminate(self):
            self.terminated = True

        async def wait(self):
            self.waited = True
            self.returncode = -15
            return self.returncode

    process = Process()

    async def spawn(*_args, **_kwargs):
        return process

    monkeypatch.setattr(smoke.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(smoke.cli("run", "goal"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated and process.waited


async def test_wrong_fixture_organization_refuses_before_admission(tmp_path, monkeypatch):
    import mcp.client.stdio

    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    output = tmp_path / "receipt.json"
    receipt.save(output)

    class Context:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *_args):
            return False

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            pass

    class Client:
        def __init__(self):
            self.account = SimpleNamespace(get=self.account_get)

        async def account_get(self):
            return SimpleNamespace(org=SimpleNamespace(id="other-org"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def no_launch(*_args):
        pytest.fail("wrong organization must refuse before any run admission")

    monkeypatch.setattr(smoke, "AsyncSynthClient", Client)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", lambda *_a, **_kw: Context((None, None)))
    monkeypatch.setattr(smoke, "ClientSession", Session)
    monkeypatch.setattr(smoke, "launch", no_launch)
    with pytest.raises(ValueError, match="fixture organization"):
        await smoke.execute(receipt, output)
    assert all(arm.status == "not_started" for arm in smoke.Receipt.load(output).arms)


def test_resume_refuses_changed_sdk_or_runner_identity(monkeypatch):
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    monkeypatch.setattr(smoke, "sdk_tree_digest", lambda: "0" * 64)
    with pytest.raises(ValueError, match="SDK or runner identity"):
        receipt.validate_intent(receipt.base_url, receipt.fixture_org_id, receipt.cohort_sha256)


def test_child_environment_excludes_unrelated_secrets(monkeypatch):
    monkeypatch.setenv("SYNTH_API_KEY", "authorized-test-key")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-child")
    env = smoke.child_environment()
    assert env["SYNTH_API_KEY"] == "authorized-test-key"
    assert env["SYNTH_API_BASE_URL"] == "https://api.example.test"
    assert "UNRELATED_SECRET" not in env


def test_resume_refuses_tampered_output_evidence(tmp_path):
    import hashlib

    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    arm = receipt.arms[0]
    content = receipt.nonce.encode()
    digest = hashlib.sha256(content).hexdigest()
    output = tmp_path / f"receipt.sdk.{digest}.output"
    output.write_bytes(content)
    arm.run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    arm.status = "passed"
    arm.execution_closed = arm.accounting_final = arm.cleanup_complete = True
    arm.charge_usd = "0.10"
    arm.charge_final = arm.answer_verified = True
    arm.output_id, arm.output_sha256, arm.output_file = (
        "out_AAAAAAAAAAAAAAAAAAAAAA",
        digest,
        str(output),
    )
    smoke.validate_evidence(receipt, tmp_path / "receipt.json")
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output evidence differs"):
        smoke.validate_evidence(receipt, tmp_path / "receipt.json")


async def test_cancel_outstanding_never_launches_and_preserves_unknown_intent(
    tmp_path, monkeypatch
):
    receipt = smoke.new_receipt("https://api.example.test", "fixture-org", "a" * 64)
    first, second, _third = receipt.arms
    first.run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    first.status = "running"
    second.status = "launch_pending"
    output = tmp_path / "receipt.json"
    receipt.save(output)
    calls = []

    class Swarms:
        async def get(self, run_id):
            done = bool(calls)
            return SimpleNamespace(
                run_id=run_id,
                execution_closed=done,
                accounting_final=done,
                cleanup_complete=done,
                charge=SimpleNamespace(amount=SimpleNamespace(amount="0.20"), final=True)
                if done
                else None,
                finished=done,
            )

        async def cancel(self, run_id, *, idempotency_key):
            calls.append((run_id, idempotency_key))

    class Client:
        def __init__(self):
            self.swarms = Swarms()
            self.account = SimpleNamespace(get=self.account_get)

        async def account_get(self):
            return SimpleNamespace(org=SimpleNamespace(id="fixture-org"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(smoke, "AsyncSynthClient", Client)
    with pytest.raises(RuntimeError, match="unknown launch"):
        await smoke.cancel_outstanding(receipt, output)
    saved = smoke.Receipt.load(output)
    assert calls == [(first.run_id, first.cancel_key)]
    assert saved.arms[0].status == "failed" and saved.arms[0].cleanup_complete
    assert saved.arms[1].status == "launch_pending" and saved.arms[1].run_id is None
    assert saved.arms[2].status == "not_started"
