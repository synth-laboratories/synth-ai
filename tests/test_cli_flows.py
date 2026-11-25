"""CLI upload, run-wait, device-login, and oversize-file flow tests."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from test_cli import (
    RUN_ID,
    api,  # noqa: F401
    example,
    invoke,
    serve_answer,
    serve_result,
    snapshot_in,
)

from synth_ai import cli
from synth_ai._generated.file_limits import FILE_MAX_BYTES

CHARGE = {"amount": {"amount": "0.02365822", "currency": "USD"}, "final": True}
ENVELOPE = json.loads(
    (Path(__file__).parent / "fixtures" / "core_run_result_1_37_0.json").read_text()
)


def test_binary_upload_uses_prepare_protocol_without_a_format_allowlist(api, capsys, tmp_path):  # noqa: F811
    pdf = tmp_path / "paper.pdf"
    content = b"%PDF-1.4\x00\xff"
    pdf.write_bytes(content)
    _, file = example(api.contract, "uploadFile")
    file.update(
        name="paper.pdf",
        media_type="application/octet-stream",
        size_bytes=len(content),
        digest_sha256=hashlib.sha256(content).hexdigest(),
    )

    def prepare(request):
        assert json.loads(request.content) == {
            "name": "paper.pdf",
            "media_type": "application/octet-stream",
            "size_bytes": len(content),
            "digest_sha256": file["digest_sha256"],
        }
        return httpx.Response(
            201,
            json={"upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA", "state": "committed", "file": file},
        )

    api.overrides["createUpload"] = prepare
    code, out, err = invoke(api, capsys, "upload", str(pdf))
    assert code == cli.EXIT_OK and err == ""
    assert "paper.pdf" in out
    assert [operation for operation, _ in api.calls] == ["createUpload"]


def test_result_without_a_text_answer_says_so(api, capsys, contract):  # noqa: F811
    serve_result(api, contract, None)
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert "answer" in out and "none" in out


def test_result_on_an_unfinished_run_exits_one_with_its_code(api, capsys, contract):  # noqa: F811
    _, running = example(contract, "getRun")
    running.update(state="running", execution_closed=False)
    api.overrides["getRun"] = (200, running)
    api.overrides["getRunResult"] = (
        409,
        {
            "type": "about:blank",
            "title": "refused",
            "detail": "run has not finished",
            "status": 409,
            "instance": "urn:test",
            "code": "result_unavailable",
            "service": "core",
            "stage": "getRunResult",
            "outcome": "not_accepted",
        },
    )
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("result_unavailable: ")


def test_run_wait_prints_the_answer_last(api, capsys, contract):  # noqa: F811
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


def test_upgrade_required_prints_one_line_upgrade_hint(api, capsys, contract):  # noqa: F811
    problem = {
        "type": "https://docs.usesynth.ai/errors/client_upgrade_required",
        "title": "Upgrade Required",
        "status": 426,
        "detail": "synth-ai 0.31.0 is no longer supported",
        "instance": "urn:request",
        "code": "client_upgrade_required",
        "service": "synth-backend",
        "stage": "public_client_version",
        "outcome": "not_accepted",
        "retryable": False,
        "minimum_version": "0.32.0",
        "client_version": "0.31.0",
    }
    api.overrides["getRun"] = (426, problem)
    code, out, err = invoke(api, capsys, "status", RUN_ID)
    assert code == cli.EXIT_SERVER_REFUSAL and out == ""
    lines = err.splitlines()
    assert lines[0] == "client_upgrade_required: synth-ai 0.31.0 is no longer supported"
    assert lines[1].endswith("(minimum 0.32.0); run: pip install -U synth-ai")
    assert len(lines) == 2
    assert len([c for c in api.calls if c[0] == "getRun"]) == 1
    code, out, err = invoke(api, capsys, "status", RUN_ID, "--json")
    assert code == cli.EXIT_SERVER_REFUSAL
    assert json.loads(err)["minimum_version"] == "0.32.0"


def test_result_and_run_wait_end_with_the_run_charge(api, capsys, contract):  # noqa: F811
    serve_answer(api, contract, "The answer is 42.")
    charged = snapshot_in(contract, "completed")
    charged["charge"] = CHARGE
    api.overrides["getRun"] = (200, charged)
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "The answer is 42.\n"
    assert err.splitlines()[-1] == "charge  0.02365822 USD (final)"
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    assert json.loads(out)["charge"] == CHARGE
    code, out, err = invoke(api, capsys, "run", "Goal", "--budget", "1.00", "--wait")
    assert code == cli.EXIT_OK, err
    assert out.splitlines()[-1] == "The answer is 42."
    assert err.splitlines()[-1] == "charge  0.02365822 USD (final)"
    charged["charge"] = {"amount": {"amount": "0.01", "currency": "USD"}, "final": False}
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert err.splitlines()[-1] == "charge  0.01 USD (not final)"


def test_result_without_charge_prints_no_charge_line(api, capsys, contract):  # noqa: F811
    serve_answer(api, contract, "The answer is 42.")
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    assert json.loads(out)["charge"] is None


def test_run_reuses_explicit_launch_and_per_file_keys(api, capsys, contract, tmp_path):  # noqa: F811
    path = tmp_path / "task.md"
    content = b"# Task\n"
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
    argv = ["run", "Goal", "--budget", "4.00", "--file", str(path), "--idempotency-key", "run-key"]
    for _ in range(2):
        code, _, err = invoke(api, capsys, *argv)
        assert code == cli.EXIT_OK, err
    launches = [request for oid, request in api.calls if oid == "createRun"]
    uploads = [request for oid, request in api.calls if oid == "createUpload"]
    assert len(launches) == len(uploads) == 2
    assert [request.headers["Idempotency-Key"] for request in launches] == ["run-key", "run-key"]
    assert launches[0].content == launches[1].content
    assert uploads[0].headers["Idempotency-Key"] == uploads[1].headers["Idempotency-Key"]
    assert uploads[0].headers["Idempotency-Key"] != "run-key"


def test_run_can_reuse_launch_identity_after_lost_response(api, capsys, contract):  # noqa: F811
    sent = []
    seen_request = None

    def launch(request):
        nonlocal seen_request
        sent.append(request.headers["Idempotency-Key"])
        if seen_request is None:
            seen_request = request.content
        assert request.content == seen_request
        if len(sent) == 1:
            raise httpx.ReadError("response lost after admission", request=request)
        status, body = example(contract, "createRun")
        return httpx.Response(status, json=body)

    api.overrides["createRun"] = launch
    argv = ["run", "Goal", "--budget", "4.00", "--idempotency-key", "retained-launch"]
    first, _, _ = invoke(api, capsys, *argv)
    assert first != cli.EXIT_OK
    second, out, err = invoke(api, capsys, *argv)
    assert second == cli.EXIT_OK, err
    assert out.strip()
    assert sent == ["retained-launch", "retained-launch"]


@pytest.mark.parametrize("as_json", [False, True])
def test_upload_failure_prints_recovery_identity(api, capsys, contract, tmp_path, as_json):  # noqa: F811
    path = tmp_path / "task.md"
    content = b"# Task\n"
    path.write_bytes(content)
    _, manifest = example(contract, "uploadFile")
    manifest.update(size_bytes=len(content), digest_sha256=hashlib.sha256(content).hexdigest())
    api.overrides["createUpload"] = (
        201,
        {
            "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
            "state": "pending",
            "file": manifest,
            "put": None,
        },
    )
    argv = ["run", "Goal", "--budget", "4.00", "--file", str(path), "--idempotency-key", "run-key"]
    code, _, err = invoke(api, capsys, *(argv + (["--json"] if as_json else [])))
    assert code == cli.EXIT_CLIENT_ERROR
    key = cli.mutation_key("run-upload", "run-key", {"index": 0})
    if as_json:
        body = json.loads(err)
        assert body["code"] == "upload_incomplete"
        assert body["upload_id"] == "upl_AAAAAAAAAAAAAAAAAAAAAA"
        assert body["idempotency_key"] == key
    else:
        assert "upload_id=upl_AAAAAAAAAAAAAAAAAAAAAA" in err
        assert f"idempotency_key={key}" in err
    assert all(oid != "createRun" for oid, _ in api.calls)


def test_directory_upload_requires_explicit_single_root_and_retry_key(api, capsys, tmp_path):  # noqa: F811
    for arguments in (
        ["upload", "--directory", str(tmp_path)],
        ["upload", "--directory", str(tmp_path), str(tmp_path), "--idempotency-key", "batch"],
    ):
        code, _, error = invoke(api, capsys, *arguments)
        assert code == cli.EXIT_CLIENT_ERROR
        assert "one path and --idempotency-key" in error
        assert api.calls == []


def test_directory_upload_cli_preserves_binary_names_and_retry_identity(api, capsys, tmp_path):  # noqa: F811
    root = tmp_path / "bundle"
    (root / "nested").mkdir(parents=True)
    content = b"\x00\xffbinary"
    (root / "nested" / "policy.bin").write_bytes(content)
    _, file = example(api.contract, "uploadFile")
    file.update(
        name="nested/policy.bin",
        media_type="application/octet-stream",
        size_bytes=len(content),
        digest_sha256=hashlib.sha256(content).hexdigest(),
    )
    api.overrides["createUpload"] = (
        201,
        {
            "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
            "state": "committed",
            "file": file,
        },
    )
    for _ in range(2):
        code, output, error = invoke(
            api,
            capsys,
            "upload",
            "--directory",
            str(root),
            "--idempotency-key",
            "directory-batch",
            "--json",
        )
        assert code == 0, error
        assert json.loads(output)[0]["name"] == "nested/policy.bin"
    assert [operation for operation, _ in api.calls] == ["createUpload", "createUpload"]
    first, second = [request for _, request in api.calls]
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert json.loads(first.content)["digest_sha256"] == hashlib.sha256(content).hexdigest()


def test_directory_upload_cli_reports_partial_custody(api, capsys, tmp_path):  # noqa: F811
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    (root / "b.bin").write_bytes(b"b")
    _, file = example(api.contract, "uploadFile")
    file.update(
        name="a.bin",
        media_type="application/octet-stream",
        size_bytes=1,
        digest_sha256=hashlib.sha256(b"a").hexdigest(),
    )

    def partial(request):
        if json.loads(request.content)["name"] == "a.bin":
            return httpx.Response(
                201,
                json={
                    "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                    "state": "committed",
                    "file": file,
                },
            )
        return httpx.Response(503, json={"code": "unavailable"})

    api.overrides["createUpload"] = partial
    code, output, error = invoke(
        api,
        capsys,
        "upload",
        "--directory",
        str(root),
        "--idempotency-key",
        "partial-batch",
        "--json",
    )
    assert code == cli.EXIT_CLIENT_ERROR
    assert output == ""
    receipt = json.loads(error)
    assert receipt["code"] == "directory_upload_incomplete"
    assert receipt["name"] == "b.bin"
    assert receipt["idempotency_key"] == "partial-batch"
    assert receipt["file_idempotency_key"]
    assert receipt["completed_files"][0]["name"] == "a.bin"


def test_login_device_shows_the_code_and_never_the_device_code(capsys):
    from synth_ai._auth_models import DeviceAuthorization

    cli.show_device_code(
        DeviceAuthorization(
            device_code="synth_odc_v1_secret",
            user_code="BCDF-GHJK",
            verification_uri="https://usesynth.ai/mcp/device",
            verification_uri_complete="https://usesynth.ai/mcp/device?user_code=BCDF-GHJK",
            expires_in=600,
            interval=5,
        )
    )
    err = capsys.readouterr().err
    assert "BCDF-GHJK" in err and "https://usesynth.ai/mcp/device" in err
    assert "synth_odc_v1_secret" not in err


def test_login_device_and_no_browser_are_exclusive(capsys):
    assert cli.main(["login", "--device", "--no-browser"]) != cli.EXIT_OK
    assert "not allowed with argument --device" in capsys.readouterr().err


def test_result_reports_the_run_status_and_final_accounting(api, capsys):  # noqa: F811
    """Contract 1.37.0 (SYN-5637): `synth result` carries run_id, status and accounting."""
    fixture = json.loads(json.dumps(ENVELOPE))
    api.overrides["getRunResult"] = (200, fixture)
    code, out, err = invoke(api, capsys, "result", RUN_ID, "--json")
    assert code == cli.EXIT_OK, err
    assert json.loads(out) == fixture
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert out == "The answer.\n"
    assert "status  completed" in err.splitlines()
    assert "accounting  0.10 USD (final)" in err.splitlines()
    fixture.update(accounting=None, status="partial")
    code, out, err = invoke(api, capsys, "result", RUN_ID)
    assert code == cli.EXIT_OK, err
    assert "status  partial" in err.splitlines()
    assert "accounting  unavailable" in err.splitlines()


@pytest.mark.parametrize("as_json", [False, True])
def test_oversize_upload_surfaces_the_typed_upload_size_exceeded_refusal(
    api,  # noqa: F811
    capsys,
    tmp_path,
    as_json,
):
    """SYN-5666: an oversize file is Core's typed 422, not a generic client_error."""
    path = tmp_path / "big.txt"
    with path.open("wb") as stream:
        stream.truncate(FILE_MAX_BYTES + 1)
    argv = [
        "run",
        "Goal",
        "--budget",
        "1.00",
        "--file",
        str(path),
        *(["--json"] if as_json else []),
    ]
    code, out, err = invoke(api, capsys, *argv)
    assert code == cli.EXIT_SERVER_REFUSAL and out == "", err
    if as_json:
        problem = json.loads(err)
        assert (problem["code"], problem["status"]) == ("upload_size_exceeded", 422)
        assert problem["maximum_size_bytes"] == str(FILE_MAX_BYTES)
    else:
        assert err.startswith("upload_size_exceeded: "), err
    assert api.calls == []
