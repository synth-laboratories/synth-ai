import asyncio
import json
import tomllib
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import __version__
from synth_ai._generated.models import LaunchRequest
from synth_ai._transport import RetryPolicy, Transport
from synth_ai.cli import read_task
from synth_ai.errors import SwarmsAPIError, TransportFailureError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "row", json.loads((ROOT / "contracts/fixtures/message-limits.json").read_text())
)
def test_shared_utf8_boundaries(row, tmp_path):
    text = row["text"] if "text" in row else row["unit"] * row["repeat"]
    path = tmp_path / "task.txt"
    path.write_text(text)
    if row["accepted"]:
        assert LaunchRequest(
            message=text,
            limits={
                "maximum_charge": {"amount": "1.00", "currency": "USD"},
                "wall_time_seconds": 60,
            },
        ).message == read_task(str(path))
    else:
        with pytest.raises(ValidationError) as error:
            LaunchRequest(
                message=text,
                limits={
                    "maximum_charge": {"amount": "1.00", "currency": "USD"},
                    "wall_time_seconds": 60,
                },
            )
        assert any(item["loc"] == ("message",) for item in error.value.errors())
        with pytest.raises(ValueError):
            read_task(str(path))


def test_user_agent_tracks_package_metadata():
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    transport = Transport("https://example.test", "token")
    assert __version__ == expected
    assert transport.headers()["User-Agent"] == "synth-ai/" + expected


@pytest.mark.parametrize(
    "operation,body", [("updateApiKey", {}), ("createRun", {"outcome": "unknown"})]
)
async def test_unverified_or_ambiguous_mutations_are_not_replayed(operation, body):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transport = Transport(
            "https://example.test",
            "token",
            client=client,
            retry_policy=RetryPolicy(backoff_seconds=0, jitter_seconds=0),
        )
        with pytest.raises(SwarmsAPIError):
            await transport.invoke(
                operation, path={"key_id": "key_one"} if operation == "updateApiKey" else None
            )
    assert len(calls) == 1


async def test_overall_deadline_bounds_a_slow_transport():
    calls = []

    async def handle(request):
        calls.append(request)
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transport = Transport(
            "https://example.test",
            "token",
            client=client,
            retry_policy=RetryPolicy(deadline_seconds=0.02),
        )
        with pytest.raises(TransportFailureError) as caught:
            await transport.invoke("getAccount")
        assert isinstance(caught.value.__cause__, TimeoutError)
    assert len(calls) == 1


async def test_retry_after_cannot_extend_deadline():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "5"}, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transport = Transport(
            "https://example.test",
            "token",
            client=client,
            retry_policy=RetryPolicy(deadline_seconds=0.02),
        )
        with pytest.raises(SwarmsAPIError):
            await transport.invoke("getAccount")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"attempts": 0},
        {"attempts": 1.5},
        {"attempts": True},
        {"deadline_seconds": 0},
        {"deadline_seconds": float("nan")},
        {"jitter_seconds": -1},
    ],
)
def test_retry_policy_validates_before_transport(options):
    with pytest.raises(ValueError):
        RetryPolicy(**options)
