"""Run snapshot stream decodes bounded frames and resumes by version token offline."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts import swarms as operations
from synth_ai.contracts.fingerprint import digest_exported_contract, schema_sha256
from synth_ai.contracts.swarms_schema import export_contract
from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    SynthError,
    TransportError,
)
from synth_ai.options import ClientOptions, ObservationOptions
from synth_ai.sdk.swarms.run import AsyncSwarmRun, SwarmRun
from synth_ai.sdk.swarms.watch import SnapshotFrameDecoder

pytestmark = pytest.mark.unit
METADATA = {
    "contract": "swarms.v1",
    "request_id": "watch-fixture",
    "operation_id": None,
    "observed_at": "2026-10-20T20:00:00Z",
}
STREAM_HEADERS = {"X-Synth-Contract": "swarms.v1", "Content-Type": "text/event-stream"}


def snapshot(state="running", run_id="run-1"):
    closed = state == "succeeded"
    return {
        "metadata": METADATA,
        "run_id": run_id,
        "project_id": "project-1",
        "state": state,
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": closed,
        "execution_closed": closed,
        "accounting_final": closed,
        "cleanup_complete": closed,
    }


def snapshot_frame(token, state="running", run_id="run-1"):
    return (
        f"id: {token}\nevent: snapshot\ndata: {json.dumps(snapshot(state, run_id))}\n\n"
    ).encode()


def end_frame(reason):
    return f'event: end\ndata: {{"reason":"{reason}"}}\n\n'.encode()


KEEPALIVE = b": keepalive\n\n"


def stream(*parts):
    return httpx.Response(200, headers=STREAM_HEADERS, content=b"".join(parts))


class Server:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        return response() if callable(response) else response


def sync_run(server, **options):
    client = SynthClient(
        api_key="fixture-key",
        base_url="https://backend.test",
        options=ClientOptions(retry_base_delay_seconds=0, **options),
    )
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(server))
    client._transport._negotiate = lambda *args: None
    client._transport._sleep = lambda seconds: None
    return client, SwarmRun(client._transport, "run-1")


def test_decoder_skips_keepalives_and_reassembles_split_frames():
    decoder = SnapshotFrameDecoder(1 << 16)
    wire = KEEPALIVE + snapshot_frame("rv1.1.1.1.c.s.0123456789abcdef") + end_frame("final")
    frames = [
        frame for index in range(len(wire)) for frame in decoder.feed(wire[index : index + 1])
    ]
    assert [frame.event for frame in frames] == ["snapshot", "end"]
    assert frames[0].event_id == "rv1.1.1.1.c.s.0123456789abcdef" and frames[1].event_id is None
    assert json.loads(frames[0].data)["run_id"] == "run-1"
    assert not decoder.buffer


@pytest.mark.parametrize(
    "frame",
    [
        b"event: snapshot\ndata: {}\n\n",
        b"id: x\nevent: progress\ndata: {}\n\n",
        b"id: x\nevent: snapshot\nretry: 5\ndata: {}\n\n",
        b"event: end\nevent: end\ndata: {}\n\n",
        b"id: a b\nevent: end\ndata: {}\n\n",
    ],
)
def test_decoder_rejects_frames_outside_the_contract(frame):
    with pytest.raises(ContractError):
        list(SnapshotFrameDecoder(1 << 16).feed(frame))


def test_decoder_bounds_frames_without_a_delimiter():
    decoder = SnapshotFrameDecoder(64)
    with pytest.raises(ContractError, match="byte bound"):
        list(decoder.feed(b"data: " + b"x" * 100))


def test_watch_yields_snapshots_and_resumes_with_the_last_event_id_after_rotate():
    server = Server(
        stream(snapshot_frame("v1"), KEEPALIVE, snapshot_frame("v2"), end_frame("rotate")),
        stream(KEEPALIVE, snapshot_frame("v3", "succeeded"), end_frame("final")),
    )
    client, run = sync_run(server)
    with client, run.watch() as watch:
        states = [item.state for item in watch]
    assert states == ["running", "running", "succeeded"]
    assert watch.end_reason == "final" and watch.last_event_id == "v3"
    first, second = server.requests
    assert first.url.path == "/api/v1/swarms/runs/run-1/stream" and first.method == "GET"
    assert first.headers["Accept"] == "text/event-stream"
    assert first.headers["Authorization"] == "Bearer fixture-key"
    assert "Last-Event-ID" not in first.headers
    assert second.headers["Last-Event-ID"] == "v2"


def test_watch_resumes_from_a_caller_token_and_tolerates_a_skipped_initial_snapshot():
    server = Server(stream(KEEPALIVE, end_frame("final")))
    client, run = sync_run(server)
    with client:
        watch = run.watch(last_event_id="rv1.9.1.4.c.s.0123456789abcdef")
        assert list(watch) == []
    assert server.requests[0].headers["Last-Event-ID"] == "rv1.9.1.4.c.s.0123456789abcdef"
    assert watch.end_reason == "final"


def test_watch_reconnects_after_bounded_transient_failures():
    def dropped():
        raise httpx.ReadError("reset")

    server = Server(
        stream(snapshot_frame("v1")),  # EOF without an end frame
        httpx.Response(503, content=b"upstream"),
        dropped,
        stream(snapshot_frame("v2", "succeeded"), end_frame("final")),
    )
    client, run = sync_run(server)
    with client:
        assert [item.state for item in run.watch()] == ["running", "succeeded"]
    assert [request.headers.get("Last-Event-ID") for request in server.requests] == [
        None,
        "v1",
        "v1",
        "v1",
    ]


def test_watch_raises_after_its_retry_budget():
    server = Server(*(httpx.Response(503, content=b"upstream") for _ in range(3)))
    client, run = sync_run(server)
    observation = ObservationOptions(
        timeout_seconds=60,
        idle_timeout_seconds=30,
        reconnect_deadline_seconds=30,
        retry_count_max=2,
        maximum_events=10,
        poll_interval_seconds=1,
    )
    with client, pytest.raises(TransportError):
        list(run.watch(observation=observation))
    assert len(server.requests) == 3


def error_frame(status, code, **fields):
    body = {"status": status, "error": {"code": code, "request_id": "req-1", **fields}}
    return b"event: error\ndata: " + json.dumps(body).encode() + b"\n\n"


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (401, "credential_identity_changed", AuthenticationError),
        (403, "forbidden", PermissionDeniedError),
        (404, "run_not_found", NotFoundError),
        (429, "stream_capacity_exhausted", RateLimitError),
        (408, "request_deadline_exceeded", RequestTimeoutError),
        (503, "service_unavailable", SynthError),
    ],
)
def test_error_frame_raises_the_refusal_its_status_selects(status, code, kind):
    server = Server(stream(snapshot_frame("v1"), error_frame(status, code)))
    client, run = sync_run(server)
    with client, pytest.raises(kind) as raised:
        list(run.watch())
    assert raised.value.code == code and raised.value.request_id == "req-1"
    assert raised.value.status == status
    assert len(server.requests) == 1  # error frames never reconnect


def test_error_frame_preserves_retry_after_and_outcome():
    frame = error_frame(429, "stream_capacity_exhausted", retry_after_seconds=3, outcome="unknown")
    client, run = sync_run(Server(stream(frame)))
    with client, pytest.raises(RateLimitError) as raised:
        list(run.watch())
    assert raised.value.retry_after_seconds == 3


@pytest.mark.parametrize(
    "data",
    [
        b'{"error": {"code": "forbidden", "request_id": null}}',
        b'{"status": 200, "error": {"code": "forbidden", "request_id": null}}',
        b'{"status": "403", "error": {"code": "forbidden", "request_id": null}}',
        b'{"status": 403, "error": {"code": "forbidden"}, "extra": 1}',
        b'{"status": 403, "error": {"request_id": null}}',
    ],
)
def test_error_frame_outside_the_contract_is_a_contract_error(data):
    client, run = sync_run(Server(stream(b"event: error\ndata: " + data + b"\n\n")))
    with client, pytest.raises(ContractError):
        list(run.watch())


def test_http_capacity_refusal_before_open_retries_with_backoff():
    sleeps = []
    body = {"error": {"code": "stream_capacity_exhausted", "request_id": "req-3"}}
    server = Server(
        httpx.Response(
            429, headers={"X-Synth-Contract": "swarms.v1", "Retry-After": "2"}, json=body
        ),
        stream(snapshot_frame("v1", "succeeded"), end_frame("final")),
    )
    client, run = sync_run(server)
    client._transport._sleep = sleeps.append
    with client:
        assert [item.state for item in run.watch()] == ["succeeded"]
    assert len(server.requests) == 2 and sleeps and sleeps[0] >= 2


def test_http_capacity_refusal_exhausts_into_a_typed_rate_limit():
    body = {"error": {"code": "stream_capacity_exhausted", "request_id": "req-4"}}
    server = Server(
        *(
            httpx.Response(429, headers={"X-Synth-Contract": "swarms.v1"}, json=body)
            for _ in range(6)
        )
    )
    client, run = sync_run(server)
    with client, pytest.raises(RateLimitError) as raised:
        list(run.watch())
    assert raised.value.code == "stream_capacity_exhausted" and len(server.requests) == 6


def test_http_refusal_raises_without_reconnecting():
    body = {"error": {"code": "run_not_found", "request_id": "req-2"}}
    server = Server(httpx.Response(404, headers={"X-Synth-Contract": "swarms.v1"}, json=body))
    client, run = sync_run(server)
    with client, pytest.raises(NotFoundError):
        list(run.watch())
    assert len(server.requests) == 1


def test_oversized_frame_is_a_contract_error():
    server = Server(stream(snapshot_frame("v1"), b"id: v2\nevent: snapshot\ndata: " + b"x" * 4096))
    client, run = sync_run(server, maximum_response_bytes=2048)
    with client, pytest.raises(ContractError, match="byte bound"):
        list(run.watch())
    assert len(server.requests) == 1


def test_snapshot_for_another_run_is_a_contract_error():
    server = Server(stream(snapshot_frame("v1", run_id="run-2"), end_frame("final")))
    client, run = sync_run(server)
    with client, pytest.raises(ContractError):
        list(run.watch())


def test_watch_deadline_is_typed_and_does_not_reconnect_forever():
    server = Server(*(stream(end_frame("rotate")) for _ in range(10000)))
    client, run = sync_run(server)
    observation = ObservationOptions(
        timeout_seconds=0.05,
        idle_timeout_seconds=30,
        reconnect_deadline_seconds=30,
        retry_count_max=0,
        maximum_events=10,
        poll_interval_seconds=1,
    )
    with client, pytest.raises(RequestTimeoutError) as raised:
        list(run.watch(observation=observation))
    assert raised.value.code == "run_observation_expired"


def test_async_watch_matches_the_sync_contract():
    async def scenario():
        server = Server(
            stream(snapshot_frame("v1"), end_frame("rotate")),
            httpx.Response(429, content=b"busy"),
            stream(snapshot_frame("v2", "succeeded"), end_frame("final")),
        )
        async with AsyncSynthClient(
            api_key="fixture-key",
            base_url="https://backend.test",
            options=ClientOptions(retry_base_delay_seconds=0),
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(server))

            async def negotiated(*args):
                return None

            async def no_sleep(seconds):
                return None

            client._transport._negotiate = negotiated
            client._transport._sleep = no_sleep
            async with AsyncSwarmRun(client._transport, "run-1").watch() as watch:
                states = [item.state async for item in watch]
        assert states == ["running", "succeeded"] and watch.end_reason == "final"
        assert [request.headers.get("Last-Event-ID") for request in server.requests] == [
            None,
            "v1",
            "v1",
        ]

    asyncio.run(scenario())


def test_swarms_contract_exports_the_stream_operation():
    exported = json.loads(export_contract())
    stream_operation = [item for item in exported["operations"] if item["id"] == "swarms.stream"]
    assert stream_operation == [
        {
            "id": "swarms.stream",
            "method": "GET",
            "path": "/api/v1/swarms/runs/{run_id}/stream",
            "keyed": True,
            "mutation": False,
            "required_scope": "swarms:read",
            "may_charge": False,
            "idempotent": True,
        }
    ]
    ids = [item["id"] for item in exported["operations"]]
    assert len(ids) == len(set(ids))
    assert operations.STREAM.matches_resource_path("/api/v1/swarms/runs/run-1/stream")
    retained = (Path(__file__).parents[1] / "contracts/swarms-v1.json").read_text()
    assert digest_exported_contract(retained) == schema_sha256("swarms.v1")
