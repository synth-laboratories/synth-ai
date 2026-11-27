"""Targeted regression tests for Synth Style boundary validation, errors, and redaction."""

from __future__ import annotations

import json
import math
from typing import Literal

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError
from test_cli import RUN_ID, api, invoke  # noqa: F401

from synth_ai import AsyncSynthClient, auth, cli
from synth_ai._cli_view import SavedOutput
from synth_ai._transport import Transport
from synth_ai._wire import WireModel
from synth_ai.errors import (
    DataIntegrityError,
    InvalidRequestError,
    LimitsRevisionConflictError,
    Problem,
    SynthError,
    local_problem,
)
from synth_ai.mcp._arguments import validate_with
from synth_ai.mcp.research_dispatch import dispatch_research
from synth_ai.mcp.server import build_server


@pytest.mark.parametrize("timeout", ["0", "-5", "nan"])
def test_cli_run_wait_rejects_bad_timeout_before_upload(api, capsys, tmp_path, timeout):  # noqa: F811
    attached = tmp_path / "task.md"
    attached.write_text("hello")
    code, out, err = invoke(
        api,
        capsys,
        "run",
        "Goal",
        "--budget",
        "1.00",
        "--file",
        str(attached),
        "--wait",
        "--timeout",
        timeout,
    )
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("client_error: ")
    assert api.calls == []


@pytest.mark.parametrize("bad_key", ["   ", "bad\nkey", "bad\rkey"])
def test_cli_run_rejects_invalid_idempotency_key_before_upload(api, capsys, tmp_path, bad_key):  # noqa: F811
    attached = tmp_path / "task.md"
    attached.write_text("hello")
    code, out, err = invoke(
        api,
        capsys,
        "run",
        "Goal",
        "--budget",
        "1.00",
        "--file",
        str(attached),
        "--idempotency-key",
        bad_key,
    )
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("client_error: ")
    assert api.calls == []


def test_cli_body_and_file_errors_hide_input_and_check_utf8(api, capsys, tmp_path):  # noqa: F811
    bad_json = tmp_path / "body.json"
    bad_json.write_text('{"pack_id": 123456, "secret": "SUPER-SECRET-INPUT"}')
    code, out, err = invoke(api, capsys, "billing", "checkout", "--body", str(bad_json))
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert err.startswith("client_error: ")
    assert "SUPER-SECRET-INPUT" not in err

    bad_utf8 = tmp_path / "art.bin"
    bad_utf8.write_bytes(b"\xff\xfe\xfd")
    code, out, err = invoke(
        api,
        capsys,
        "artifacts",
        "create",
        "--notebook-id",
        "nb_1",
        "--title",
        "t",
        "--kind",
        "html",
        "--summary",
        "s",
        "--file",
        str(bad_utf8),
    )
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert "UnicodeDecodeError" in err
    assert api.calls == []


def test_cli_inspect_wait_update_requires_positive_timeout(api, capsys):  # noqa: F811
    code, out, err = invoke(
        api, capsys, "inspect", RUN_ID, "--view", "wait-update", "--timeout-seconds", "0"
    )
    assert code == cli.EXIT_CLIENT_ERROR and out == ""
    assert "--timeout-seconds must be positive" in err
    assert api.calls == []


def test_cli_main_wraps_unexpected_httpx_error_cleanly(monkeypatch, capsys):
    def boom(_args, _client, _policy):
        raise httpx.ConnectError("raw socket https://user:secret@internal.test/v1")

    monkeypatch.setattr(cli, "execute", boom)
    code = cli.main(["whoami"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_CLIENT_ERROR and captured.out == ""
    assert captured.err.startswith("network_error: ")
    assert "secret@internal.test" not in captured.err


def test_wire_model_and_limits_revision_conflict_validation():
    class SampleWire(WireModel):
        token: str

    with pytest.raises(ValidationError) as exc:
        SampleWire.model_validate({"token": 999999})
    assert "999999" not in str(exc.value)

    with pytest.raises(ValidationError) as saved_exc:
        SavedOutput.model_validate({"output": "SECRET-OUTPUT-PAYLOAD", "path": "/tmp/x"})
    assert "SECRET-OUTPUT-PAYLOAD" not in str(saved_exc.value)

    base = local_problem("limits_revision_conflict", "limits", status=409).model_dump()
    assert (
        LimitsRevisionConflictError(
            Problem.model_validate({**base, "current_revision": 7})
        ).current_revision
        == 7
    )
    assert (
        LimitsRevisionConflictError(
            Problem.model_validate({**base, "current_revision": None})
        ).current_revision
        is None
    )


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "http://api.example.test",
        "https://user:pass@api.example.test",
        "https://api.example.test/v1",
        "https://api.example.test?q=1",
        "ftp://api.example.test",
    ],
)
def test_transport_rejects_invalid_base_url(origin):
    with pytest.raises(ValueError):
        Transport(origin, "fixture")


@pytest.mark.parametrize("token", [None, "", "   ", "bad\rtoken", "bad\ntoken", "bad\0token"])
def test_transport_headers_rejects_bad_tokens_without_leaking(token):
    transport = Transport("https://api.example.test", token)
    with pytest.raises(SynthError) as exc:
        transport.headers()
    if token is not None and token.strip():
        assert token not in str(exc.value)


@pytest.mark.parametrize(
    ("retries", "timeout"),
    [(True, 10), (-1, 10), (6, 10), (1, 0), (1, -1), (1, math.nan), (1, True)],
)
def test_transport_rejects_invalid_retry_or_timeout_bounds(retries, timeout):
    with pytest.raises(ValueError):
        Transport("https://api.example.test", "fixture", retries=retries, timeout=timeout)


async def test_client_rejects_invalid_path_ids_keys_and_artifact_bounds(tmp_path):
    called = False

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"\xff\xfe")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="https://api.example.test", http_client=http
    ) as client:
        for bad_id in ("", "   ", "..", ".", "run\nid"):
            with pytest.raises(ValueError):
                await client.swarms.get(bad_id)
        with pytest.raises(ValueError):
            await client.swarms.cancel(RUN_ID, idempotency_key="bad\nkey")
        with pytest.raises(ValueError):
            await client.research_logs.export_log("nb_1", fmt="xml")
        with pytest.raises(ValueError):
            await client._transport._transfers().read_output(
                RUN_ID, "out_1", revision_id="r1", digest_sha256="a" * 64, max_bytes=0
            )
        assert not called
        with pytest.raises(DataIntegrityError, match="UTF-8"):
            await client.artifacts.get_artifact_content("art_1")


def test_auth_wraps_transport_http_errors_and_bounds_device_polls(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_OAUTH_BASE_URL", "https://api.example.test")

    with httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(503, text="down"))
    ) as client:
        with pytest.raises(SynthError, match="HTTP 503"):
            auth.discover(client, "https://api.example.test")

    with pytest.raises(ValueError, match="timeout"):
        auth.login(
            "https://api.example.test",
            timeout=math.nan,
            open_browser=False,
            redirect_source=lambda u: u,
        )

    meta = {
        "issuer": "https://api.example.test/",
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint": "https://api.example.test/oauth/token",
        "device_authorization_endpoint": "https://api.example.test/oauth/device",
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=meta)
        if request.url.path == "/oauth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "odc_secret",
                    "user_code": "CODE-1234",
                    "verification_uri": "https://api.example.test/device",
                    "expires_in": 60,
                    "interval": 1,
                },
            )
        return httpx.Response(400, json={"error": "authorization_pending"})

    original = httpx.Client
    monkeypatch.setattr(
        auth.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    # Stuck clock (always 0.0) must terminate via bounded max_polls rather than spinning forever.
    with pytest.raises(SynthError, match="maximum poll attempts"):
        auth.login_device(
            "https://api.example.test",
            show=lambda _d: None,
            sleep=lambda _s: None,
            clock=lambda: 0.0,
        )


async def test_mcp_argument_and_error_redaction_contracts():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    with pytest.raises(InvalidRequestError) as exc:
        validate_with(TypeAdapter(Literal["auto"]), "BAD-SECRET-INPUT")
    assert exc.value.code == "tool_arguments_invalid"
    assert "BAD-SECRET-INPUT" not in str(exc.value)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"\xff\xfe"))
    )
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        with pytest.raises(ValueError):
            await dispatch_research(client, "getArtifact", {"artifact_id": "   "})
        with pytest.raises(InvalidRequestError) as res_exc:
            await dispatch_research(
                client, "createArtifact", {"body": {"notebook_id": "BAD-SECRET-INPUT"}}
            )
        assert res_exc.value.code == "tool_arguments_invalid"
        assert "BAD-SECRET-INPUT" not in str(res_exc.value)

        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            blank_result = await session.call_tool("swarms_artifact_get", {"artifact_id": "   "})
            assert blank_result.isError
            blank_text = blank_result.content[0]
            assert isinstance(blank_text, TextContent)
            assert json.loads(blank_text.text)["code"] == "invalid_arguments"

            result = await session.call_tool(
                "swarms_run_output_content",
                {
                    "run_id": RUN_ID,
                    "output_id": "out_" + "A" * 22,
                    "revision_id": "rev_1",
                    "digest_sha256": "0" * 64,
                },
            )
    assert result.isError
    text = result.content[0]
    assert isinstance(text, TextContent)
    body = json.loads(text.text)
    assert body["code"] == "content_integrity_failed"
