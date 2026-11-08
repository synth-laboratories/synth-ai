import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai._transport import retry_delay
from synth_ai.cli import parser, read_task
from synth_ai.errors import ConflictError, TransportFailureError


def snapshot(contract):
    return contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]


async def test_launch_exact_money_and_retry_identity(contract):
    requests = []
    resources = [
        {
            "plugin": "pool",
            "roles": ["worker"],
            "limits": {"spend_micros": 100000, "rollouts": 1},
            "selection": {"pool_ids": ["craftax-control"]},
        }
    ]
    overrides = {"launch": {"preset": "test.codex_pool", "effort": "low"}}

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={})
        return httpx.Response(202, json=snapshot(contract))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            await client.swarms.launch(
                "task",
                budget="0.100001",
                idempotency_key="launch-once",
                resources=resources,
                profile_overrides=overrides,
            )
    assert len(requests) == 2
    assert (
        requests[0].headers["Idempotency-Key"]
        == requests[1].headers["Idempotency-Key"]
        == "launch-once"
    )
    assert requests[0].content == requests[1].content
    body = json.loads(requests[0].content)
    assert body["resources"] == resources
    assert body["profile_overrides"] == overrides
    assert json.loads(requests[0].content)["limits"]["maximum_charge"] == {
        "amount": "0.100001",
        "currency": "USD",
    }


async def test_sync_client_inside_event_loop(contract):
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=snapshot(contract)))
    )
    with SynthClient("fixture", base_url="https://example.test", http_client=http) as client:
        assert (
            client.swarms.get("run_AAAAAAAAAAAAAAAAAAAAAA").run_id == snapshot(contract)["run_id"]
        )
    await http.aclose()


async def test_unknown_outcome_and_problem_fields():
    body = {
        "type": "https://example.test/errors/pending",
        "title": "Pending",
        "status": 409,
        "detail": "Still resolving",
        "instance": "urn:request:one",
        "code": "idempotency_in_progress",
        "service": "synth-backend",
        "stage": "launch",
        "outcome": "unknown",
        "upstream_status": 504,
        "cause": "timeout",
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(409, json=body))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(ConflictError) as caught:
                await client.swarms.cancel("run_AAAAAAAAAAAAAAAAAAAAAA")
    assert caught.value.problem.model_dump(exclude_unset=True) == body
    assert caught.value.outcome == "unknown"


async def test_transport_failure_never_claims_not_accepted():
    def fail(request):
        raise httpx.ReadTimeout("private downstream detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(TransportFailureError) as caught:
                await client.swarms.cancel("run_AAAAAAAAAAAAAAAAAAAAAA")
    assert caught.value.outcome == "unknown"
    assert "private downstream" not in str(caught.value)


async def test_sse_resume_and_close(contract):
    event = {
        "event_id": "evt_AAAAAAAAAAAAAAAAAAAAAA",
        "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
        "sequence": 13,
        "kind": "progress",
        "original_kind": "progress",
        "usage": None,
        "observed_at": "2026-11-04T00:00:00Z",
        "approval_id": None,
    }

    def stream(request):
        if request.headers.get("accept") != "text/event-stream":
            body = snapshot(contract).copy()
            body["execution_closed"] = True
            return httpx.Response(200, json=body)
        assert request.headers["Last-Event-ID"] == "12"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=": keepalive\n\nid: 13\ndata: " + json.dumps(event) + "\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(stream)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = [
                event
                async for event in client.swarms.follow(
                    "run_AAAAAAAAAAAAAAAAAAAAAA", last_event_id="12"
                )
            ]
    assert len(result) == 1


def test_cli_inventory_and_task(tmp_path):
    assert parser().parse_args(["billing", "statements"]).operation == "listStatements"
    assert parser().parse_args(["api-keys", "rotate", "key_x"]).operation == "rotateApiKey"
    (tmp_path / "task.md").write_text("Task text")
    assert read_task(str(tmp_path)) == "Task text"
    (tmp_path / "extra.txt").write_text("Must not silently disappear")
    with pytest.raises(ValueError):
        read_task(str(tmp_path))


def test_retry_wait_is_bounded():
    assert retry_delay("61", 0) is None
    assert retry_delay("0", 0) == 0
    assert retry_delay("garbage", 0) is None
