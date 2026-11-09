"""Public guarantees that require several snapshots, frames, or transfer attempts."""

import asyncio
import hashlib
import json
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import AsyncSynthClient, StreamCheckpoint
from synth_ai._generated.models import LaunchRequest, RoutingConstraints, RunGuidance
from synth_ai._transport import RetryPolicy
from synth_ai.errors import (
    ConflictError,
    ResponseContractError,
    ResultUnavailableError,
    RunFailedError,
    UploadTransferError,
    WaitTimeoutError,
)


def snapshot(contract, **changes):
    value = dict(
        contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    value.update(changes)
    return value


def client(http):
    return AsyncSynthClient(
        "fixture",
        base_url="https://example.test",
        http_client=http,
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0, jitter_seconds=0),
    )


@pytest.mark.parametrize(
    "budget", [None, 1.0, 0, "0", "-1", "NaN", "Infinity", "0.0000001", "92233720369"]
)
async def test_budget_required_and_exact(budget):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: pytest.fail("network called"))
    ) as http:
        async with client(http) as api:
            with pytest.raises((ValueError, ValidationError)):
                await api.swarms.launch("task", budget=budget)
            with pytest.raises(TypeError):
                await api.swarms.launch("task")
            with pytest.raises(TypeError):
                await api.swarms.launch("task", budget="1", maximum_tasks=1)
    with pytest.raises(ValidationError):
        LaunchRequest.model_validate({"limits": {}})


@pytest.mark.parametrize(
    "selector",
    [{}, {"only": []}, {"only": ["openai", "openai"]}, {"only": ["openai"], "exclude": []}],
)
def test_selector_one_mode(selector):
    with pytest.raises(ValidationError):
        RoutingConstraints(providers=selector)


async def test_typed_launch_routes_guidance_and_omits_internal_limits(contract):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot(contract))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            await api.swarms.launch(
                "task",
                budget=Decimal("0.125000"),
                routing=RoutingConstraints(providers={"only": ["openrouter", "azure"]}),
                guidance=[
                    RunGuidance(
                        id="proof", kind="outcome", text="Include evidence", priority="must"
                    )
                ],
            )
    assert requests[0]["limits"] == {"maximum_charge": {"amount": "0.125000", "currency": "USD"}}
    assert requests[0]["routing"] == {"providers": {"only": ["openrouter", "azure"]}}
    assert requests[0]["guidance"][0]["id"] == "proof"


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", " "),
        ("text", "a\0"),
        ("text", "é" * 4097),
        ("evidence", "\0"),
        ("acceptance", "é" * 4097),
    ],
)
def test_guidance_matches_core_byte_and_text_validation(field, value):
    values = dict(id="proof", kind="outcome", text="valid", priority="must")
    values[field] = value
    with pytest.raises(ValidationError):
        RunGuidance(**values)


async def test_finished_waits_for_accounting_and_cleanup(contract):
    states = [
        snapshot(
            contract,
            state="completed",
            execution_closed=True,
            accounting_final=False,
            cleanup_complete=False,
        ),
        snapshot(
            contract,
            state="completed",
            execution_closed=True,
            accounting_final=True,
            cleanup_complete=False,
        ),
        snapshot(
            contract,
            state="completed",
            execution_closed=True,
            accounting_final=True,
            cleanup_complete=True,
        ),
    ]
    calls = []

    def handle(request):
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=("data: " + json.dumps(event(states[0]["run_id"], "A")) + "\n\n").encode(),
            )
        calls.append(request.method)
        return httpx.Response(200, json=states[len(calls) - 1])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            result = await api.swarms.wait(states[0]["run_id"], poll_interval=0.001)
    assert result.cleanup_complete and len(calls) == 3


async def test_failure_and_result_unavailable_preserve_snapshot(contract):
    state = snapshot(
        contract,
        state="failed",
        execution_closed=True,
        accounting_final=True,
        cleanup_complete=True,
        result_available=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (
                httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")
                if r.headers.get("accept") == "text/event-stream"
                else httpx.Response(200, json=state)
            )
        )
    ) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"])).state == "failed"
            for options, kind in [
                ({"raise_on_failure": True}, RunFailedError),
                ({"until": "result_available"}, ResultUnavailableError),
            ]:
                with pytest.raises(kind) as caught:
                    await api.swarms.wait(state["run_id"], **options)
                assert caught.value.snapshot.state == "failed"
                assert caught.value.run_id == state["run_id"]


async def test_wait_timeout_and_cancellation_never_cancel_run(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    calls = []

    def handle(request):
        calls.append(request.method)
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=b": heartbeat\n\n"
            )
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(WaitTimeoutError) as caught:
                await api.swarms.wait(state["run_id"], timeout=0.015, poll_interval=0.1)
            assert caught.value.snapshot.run_id == state["run_id"]
            pending = asyncio.create_task(api.swarms.wait(state["run_id"]))
            await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    assert set(calls) == {"GET"}


async def test_malformed_success_retains_mutation_identity():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(202, json={}, headers={"X-Request-Id": "server-request"})
        )
    ) as http:
        async with client(http) as api:
            with pytest.raises(ResponseContractError) as caught:
                await api.swarms.launch("task", budget="1", idempotency_key="same-launch")
    assert caught.value.idempotency_key == "same-launch"
    assert caught.value.outcome == "unknown"
    assert caught.value.request_id == "server-request"


def event(run_id, suffix):
    return dict(
        event_id="evt_" + suffix * 22,
        run_id=run_id,
        sequence=8,
        kind="progress",
        original_kind="progress",
        usage=None,
        observed_at=None,
        approval_id=None,
    )


class Interrupted(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        yield self.content
        raise httpx.RemoteProtocolError("interrupted frame group")


async def test_stream_reconnect_deduplicates_partial_expanded_sequence(contract):
    state = snapshot(contract, execution_closed=True)
    one, two = event(state["run_id"], "A"), event(state["run_id"], "B")
    first = ("data: " + json.dumps(one) + "\n\n").encode()
    second = first + ("id: 8\ndata: " + json.dumps(two) + "\n\n").encode()
    resumes = []

    def handle(request):
        if request.headers.get("accept") != "text/event-stream":
            return httpx.Response(200, json=state)
        resumes.append(request.headers.get("last-event-id"))
        return (
            httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=Interrupted(first)
            )
            if len(resumes) == 1
            else httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=second)
        )

    checkpoint = StreamCheckpoint(run_id=state["run_id"], last_event_id="7")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            got = [item async for item in api.swarms.follow(state["run_id"], checkpoint=checkpoint)]
    assert [item.event_id for item in got] == [one["event_id"], two["event_id"]]
    assert resumes == ["7", "7"] and checkpoint.last_event_id == "8"
    assert StreamCheckpoint.from_dict(checkpoint.to_dict()) == checkpoint
    with pytest.raises(ValueError):
        checkpoint.bind("run_other")


async def test_stream_error_frame_is_typed(contract):
    problem = dict(
        type="about:blank",
        title="Lost history",
        detail="Lost history",
        status=409,
        instance="urn:test",
        code="event_cursor_expired",
        service="synth-backend",
        stage="events",
        outcome="not_accepted",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text="event: error\ndata: " + json.dumps(problem) + "\n\n",
            )
        )
    ) as http:
        async with client(http) as api:
            with pytest.raises(ConflictError, match="event_cursor_expired"):
                _ = [item async for item in api.swarms.follow(snapshot(contract)["run_id"])]


@pytest.mark.parametrize("payload,max_bytes", [(b"tampered", 100), (b"correct", 2)])
async def test_download_failure_preserves_destination_and_removes_partial(
    tmp_path, payload, max_bytes
):
    target = tmp_path / "report.txt"
    target.write_bytes(b"existing")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload))
    ) as http:
        async with client(http) as api:
            with pytest.raises(ValueError):
                await api.swarms.download_output(
                    "run_a",
                    "out_a",
                    str(target),
                    revision_id="rev_a",
                    digest_sha256=hashlib.sha256(b"correct").hexdigest(),
                    max_bytes=max_bytes,
                )
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


async def test_verified_download_and_memory_read(tmp_path):
    content = b"verified report"
    digest = hashlib.sha256(content).hexdigest()
    target = tmp_path / "report.txt"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=content))
    ) as http:
        async with client(http) as api:
            assert (
                await api.swarms.output_content(
                    "run_a", "out_a", revision_id="rev_a", digest_sha256=digest
                )
                == content
            )
            await api.swarms.download_output(
                "run_a",
                "out_a",
                str(target),
                revision_id="rev_a",
                digest_sha256=digest,
                expected_bytes=len(content),
            )
    assert target.read_bytes() == content


async def test_upload_failure_resumes_same_identity_with_fresh_grant(tmp_path):
    content = b"input"
    source = tmp_path / "task.txt"
    source.write_bytes(content)
    file = dict(
        file_id="file_a",
        revision_id="rev_a",
        name="task.txt",
        media_type="text/plain",
        size_bytes=5,
        digest_sha256=hashlib.sha256(content).hexdigest(),
        created_at="2026-11-08T00:00:00Z",
    )
    upload = dict(
        upload_id="upl_a",
        state="pending",
        file=file,
        put=dict(url="https://storage.test/fresh", headers={}, expires_at="2026-11-08T01:00:00Z"),
    )
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        if request.method == "PUT":
            assert "authorization" not in request.headers
            return httpx.Response(503 if len(calls) == 2 else 204)
        if request.url.path.endswith("/finalize"):
            assert request.headers["Idempotency-Key"] == "same-upload"
            return httpx.Response(200, json={**upload, "state": "committed", "put": None})
        return httpx.Response(200 if request.method == "GET" else 201, json=upload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(UploadTransferError) as caught:
                await api.swarms.upload_file(str(source), idempotency_key="same-upload")
            assert caught.value.upload_id == "upl_a"
            assert caught.value.idempotency_key == "same-upload"
            result = await api.swarms.resume_upload(
                caught.value.upload_id, str(source), idempotency_key=caught.value.idempotency_key
            )
    assert result.file_id == "file_a"
    assert sum(method == "POST" and path.endswith("/uploads") for method, path in calls) == 1


async def test_clean_eof_reconnects_until_execution_closed(contract):
    calls = []
    state = snapshot(contract)

    def handle(request):
        calls.append(request)
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=b": heartbeat\n\n"
            )
        return httpx.Response(200, json={**state, "execution_closed": len(calls) >= 4})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert [item async for item in api.swarms.follow(state["run_id"])] == []
    assert len(calls) == 4


async def test_incomplete_sse_frame_does_not_advance_checkpoint(contract):
    state = snapshot(contract, execution_closed=True)
    checkpoint = StreamCheckpoint(last_event_id="7")

    def handle(request):
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text="id: 8\ndata: " + json.dumps(event(state["run_id"], "A")) + "\n",
            )
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert [
                item async for item in api.swarms.follow(state["run_id"], checkpoint=checkpoint)
            ] == []
    assert checkpoint.last_event_id == "7"


async def test_wait_timeout_bounds_snapshot_request(contract):
    async def handle(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=snapshot(contract))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(WaitTimeoutError) as caught:
                await api.swarms.wait(snapshot(contract)["run_id"], timeout=0.01)
    assert caught.value.snapshot is None
    assert caught.value.run_id == snapshot(contract)["run_id"]


@pytest.mark.parametrize("bad", ["8\nAuthorization: x", "\0", "x" * 1025])
def test_checkpoint_rejects_invalid_cursor(bad):
    with pytest.raises(ValueError):
        StreamCheckpoint(last_event_id=bad)
