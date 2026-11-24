"""Refusal journeys reach typed public boundaries without OS argv truncation."""

import json
from pathlib import Path

import httpx
import pytest

from synth_ai import AsyncSynthClient, cli
from synth_ai._generated.policy import MESSAGE_MAX_BYTES
from synth_ai.errors import InvalidRequestError


@pytest.mark.parametrize(
    ("message", "optimizer"),
    [
        ("x" * (MESSAGE_MAX_BYTES + 1), None),
        ("safe", {"topology": "no_such_topology", "version": 1}),
    ],
)
async def test_invalid_launch_is_typed_before_network(message, optimizer):
    requests = []

    def handle(request):
        requests.append(request)
        raise AssertionError("invalid launch reached the network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(InvalidRequestError) as caught:
                await client.swarms.launch(message, budget="1", optimizer=optimizer)
    assert caught.value.code == "invalid_launch_request"
    assert caught.value.service == "synth-ai"
    assert caught.value.outcome == "not_accepted"
    assert not requests
    assert message not in caught.value.problem.detail


def test_cli_oversize_task_file_is_refused_before_network(tmp_path, capsys):
    task = tmp_path / "task.txt"
    task.write_text("x" * (MESSAGE_MAX_BYTES + 1))
    code = cli.main(["run", "--task-file", str(task), "--budget", "1", "--json"])
    assert code != 0
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "client_error"
    assert "task" in error["detail"]


def test_journey_cli_large_message_uses_private_file(tmp_path):
    from scripts.acceptance.surfaces import Call, launch_argv

    message = "x" * (MESSAGE_MAX_BYTES + 1)
    args = launch_argv(
        Call(
            "createRun",
            body={"message": message, "limits": {"maximum_charge": {"amount": "1"}}},
            idempotency_key="same-request",
        ),
        tmp_path,
    )
    assert "--task-file" in args
    task = Path(args[args.index("--task-file") + 1])
    assert task.read_text() == message
    assert task.stat().st_mode & 0o077 == 0
    assert args[args.index("--idempotency-key") + 1] == "same-request"
