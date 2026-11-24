"""Continuation surfaces forward a parent identity and retain typed refusals.

The current source gates intentionally refuse unavailable inherited inputs.
No mock response in this suite claims a successful continuation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient, cli, errors
from synth_ai.mcp.server import call_tool

PARENT = "run_AAAAAAAAAAAAAAAAAAAAAA"


def refusal(code, status):
    return {
        "type": "about:blank",
        "title": code,
        "code": code,
        "status": status,
        "detail": code,
        "instance": "urn:request:continue",
        "service": "synth-backend",
        "stage": "continuation",
        "outcome": "not_accepted",
        "retryable": False,
    }


@pytest.mark.parametrize("synchronous", [False, True])
async def test_launch_forwards_parent_without_rewriting_limits_or_retrying(synchronous):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(409, json=refusal("continue_from_inputs_unavailable", 409))

    async def _launch() -> None:
        if synchronous:
            with SynthClient(
                "fixture", base_url="https://example.test", http_client=http
            ) as client:
                client.swarms.launch(
                    "fixture",
                    budget="1.00",
                    continue_from=PARENT,
                    idempotency_key="continue-once",
                )
        else:
            async with AsyncSynthClient(
                "fixture", base_url="https://example.test", http_client=http
            ) as client:
                await client.swarms.launch(
                    "fixture",
                    budget="1.00",
                    continue_from=PARENT,
                    idempotency_key="continue-once",
                )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(errors.SwarmsAPIError) as caught:
            await _launch()
    assert caught.value.code == "continue_from_inputs_unavailable"
    assert caught.value.outcome == "not_accepted"
    assert len(requests) == 1
    assert requests[0].headers["Idempotency-Key"] == "continue-once"
    body = json.loads(requests[0].content)
    assert body["continue_from"] == PARENT
    assert body["limits"]["maximum_charge"] == {"amount": "1.00", "currency": "USD"}


@pytest.mark.parametrize(
    ("code", "status", "error_name"),
    [
        ("continue_from_not_found", 404, "ContinueFromNotFoundError"),
        ("continue_from_not_terminal", 409, "ContinueFromNotTerminalError"),
        ("continue_from_inputs_unavailable", 409, "ContinueFromInputsUnavailableError"),
    ],
)
def test_continuation_refusals_have_domain_error_types(code, status, error_name):
    response = httpx.Response(
        status,
        json=refusal(code, status),
        request=httpx.Request("POST", "https://example.test/v1/runs"),
    )
    with pytest.raises(getattr(errors, error_name)) as caught:
        errors.raise_for_response(response)
    assert caught.value.code == code
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("parent", ["", "run_short", "00000000-0000-0000-0000-000000000000"])
async def test_invalid_parent_identity_refuses_before_network(parent):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(500)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http,
        AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client,
    ):
        with pytest.raises(errors.LaunchValidationError) as caught:
            await client.swarms.launch("fixture", budget="1.00", continue_from=parent)
    assert caught.value.code == "invalid_launch_request"
    assert not requests


def test_cli_continue_from_uses_the_public_launch_and_typed_refusal(monkeypatch, capsys):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://example.test")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(404, json=refusal("continue_from_not_found", 404))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    status = cli.main(
        [
            "run",
            "fixture",
            "--budget",
            "1.00",
            "--continue-from",
            PARENT,
            "--idempotency-key",
            "cli-continue",
            "--json",
        ],
        http_client=http,
    )
    captured = capsys.readouterr()
    assert status == cli.EXIT_SERVER_REFUSAL
    assert json.loads(captured.err)["code"] == "continue_from_not_found"
    assert captured.out == ""
    assert len(requests) == 1
    assert json.loads(requests[0].content)["continue_from"] == PARENT


async def test_mcp_launch_argument_uses_the_generated_body_and_same_refusal():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(409, json=refusal("continue_from_not_terminal", 409))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(errors.SwarmsAPIError) as caught:
                await call_tool(
                    client,
                    "swarms_run_launch",
                    {
                        "body": {
                            "message": "fixture",
                            "continue_from": PARENT,
                            "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
                        },
                        "idempotency_key": "mcp-continue",
                    },
                )
    assert caught.value.code == "continue_from_not_terminal"
    assert len(requests) == 1
    assert json.loads(requests[0].content)["continue_from"] == PARENT


async def test_mcp_tools_call_advertises_parent_and_retains_typed_refusal():
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(409, json=refusal("continue_from_inputs_unavailable", 409))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            server = build_server(client, upload_descriptor=None)
            async with create_connected_server_and_client_session(server) as session:
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                schema = tools["swarms_run_launch"].inputSchema
                assert "continue_from" in schema["properties"]["body"]["properties"]
                result = await session.call_tool(
                    "swarms_run_launch",
                    {
                        "body": {
                            "message": "fixture",
                            "continue_from": PARENT,
                            "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
                        },
                        "idempotency_key": "mcp-tools-continue",
                    },
                )
    assert result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text)["code"] == "continue_from_inputs_unavailable"
    assert len(requests) == 1
    assert json.loads(requests[0].content)["continue_from"] == PARENT
