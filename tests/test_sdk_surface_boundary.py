"""Adapters must preserve SDK validation and mutation identity.

W11's check_surface_parity.py owns the private-access/static parity detector.
"""

import json

import httpx
import pytest
from test_cli import example

from synth_ai import AsyncSynthClient, cli
from synth_ai._generated.models import Account, RunSnapshot, RunWait
from synth_ai.mcp.server import call_tool


def test_cancel_accepts_explicit_mutation_identity():
    arguments = cli.parser().parse_args(
        ["cancel", "run_AAAAAAAAAAAAAAAAAAAAAA", "--idempotency-key", "cancel-once"]
    )
    assert arguments.idempotency_key == "cancel-once"


@pytest.mark.parametrize(
    "operation, namespace, method, name, arguments, model",
    [
        ("getAccount", "account", "get", "account_get", {}, Account),
        (
            "getRun",
            "swarms",
            "get",
            "swarms_run_get",
            {"run_id": "run_AAAAAAAAAAAAAAAAAAAAAA"},
            RunSnapshot,
        ),
        (
            "waitRun",
            "swarms",
            "wait_update",
            "swarms_run_wait",
            {"run_id": "run_AAAAAAAAAAAAAAAAAAAAAA", "timeout_seconds": 1},
            RunWait,
        ),
    ],
)
async def test_mcp_calls_public_sdk_helper(
    contract, monkeypatch, operation, namespace, method, name, arguments, model
):
    _, body = example(contract, operation)
    expected = model.model_validate(body)
    calls = []

    async def public_helper(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    def refuse_raw_transport(request):
        raise AssertionError("MCP bypassed the public SDK helper")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse_raw_transport)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            monkeypatch.setattr(getattr(client, namespace), method, public_helper)
            result = await call_tool(client, name, arguments)
    assert json.loads(result) == expected.model_dump(mode="json", exclude_unset=True)
    assert len(calls) == 1
