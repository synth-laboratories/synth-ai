"""Backend-authored control routes, replay-safe transport and strict receipts.

All requests use httpx MockTransport. No model, provider or live mutation.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from synth_ai.core.errors import ConflictError, SynthError
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.retry import RetryPolicy
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.common import SwarmId
from synth_ai.sdk.research.contracts.swarm_controls import SwarmControlReceipt
from synth_ai.sdk.research.swarms import AsyncSwarmsAPI, SwarmsAPI

RUN = SwarmId("run-control-fixture")


def receipt(kind="steer", **changes):
    return dict(
        run_id=RUN,
        operation_id="operation-fixture",
        kind=kind,
        status="accepted",
        action_id="action_1" if kind == "action_answer" else None,
        interaction_id="interaction-fixture",
        control_seq=7,
        **changes,
    )


def transport(handler):
    result = HttpTransport("https://backend.test", {}, retry_policy=RetryPolicy(2, 0, 0))
    result.client.close()
    result.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://backend.test"
    )
    return result


def test_sync_routes_preserve_text_and_idempotency_bytes():
    requests = []

    def handler(request):
        requests.append(request)
        kind = "action_answer" if "/actions/" in request.url.path else "steer"
        return httpx.Response(202, json=receipt(kind))

    wire = transport(handler)
    try:
        api = SwarmsAPI(wire)
        assert api.steer(RUN, "  steer text  ", idempotency_key=" stable-key ").control_seq == 7
        assert (
            api.answer_action(RUN, "action_1", "answer", idempotency_key="answer-key").action_id
            == "action_1"
        )
        assert [(r.method, r.url.path) for r in requests] == [
            ("POST", f"/smr/runs/{RUN}/steer"),
            ("POST", f"/smr/runs/{RUN}/actions/action_1/answer"),
        ]
        assert json.loads(requests[0].content) == {
            "message": "  steer text  ",
            "idempotency_key": "stable-key",
        }
        assert json.loads(requests[1].content) == {
            "response_text": "answer",
            "idempotency_key": "answer-key",
        }
    finally:
        wire.close()


@pytest.mark.parametrize("key,count", [(None, 1), ("stable-key", 2)])
def test_lost_response_retries_only_with_a_stable_key(key, count):
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            raise httpx.ReadError("lost acknowledgement", request=request)
        body = receipt()
        body["status"] = "duplicate"
        return httpx.Response(202, json=body)

    wire = transport(handler)
    try:
        api = SwarmsAPI(wire)
        if key is None:
            with pytest.raises(SynthError):
                api.steer(RUN, "message")
        else:
            assert api.steer(RUN, "message", idempotency_key=key).status == "duplicate"
        assert len(bodies) == count
        assert all(b == bodies[0] for b in bodies)
    finally:
        wire.close()


def test_conflicting_bytes_remain_a_typed_conflict():
    wire = transport(
        lambda request: httpx.Response(409, json={"detail": {"error_code": "idempotency_conflict"}})
    )
    try:
        with pytest.raises(ConflictError):
            SwarmsAPI(wire).steer(RUN, "changed", idempotency_key="existing")
    finally:
        wire.close()


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "foreign"},
        {"kind": "action_answer"},
        {"control_seq": True},
        {"control_seq": -1},
        {"status": "delivered"},
        {"unknown": "extra"},
    ],
)
def test_bad_acknowledgements_refuse(change):
    body = receipt()
    body.update(change)
    wire = transport(lambda request: httpx.Response(202, json=body))
    try:
        with pytest.raises(ValueError):
            SwarmsAPI(wire).steer(RUN, "message")
    finally:
        wire.close()


def test_input_validation_precedes_any_network_request():
    def forbidden(request):
        raise AssertionError("invalid control reached the transport")

    wire = transport(forbidden)
    try:
        api = SwarmsAPI(wire)
        for text in [" ", "é" * 8193]:
            with pytest.raises(ValueError):
                api.steer(RUN, text)
        with pytest.raises(ValueError):
            api.answer_action(RUN, " ", "answer")
        with pytest.raises(ValueError):
            api.steer(RUN, "message", idempotency_key="x" * 256)
    finally:
        wire.close()


def test_async_has_the_same_routes_and_typed_receipts():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                202, json=receipt("action_answer" if "/actions/" in request.url.path else "steer")
            )

        wire = AsyncHttpTransport("https://backend.test", {})
        await wire.client.aclose()
        wire.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://backend.test"
        )
        try:
            api = AsyncSwarmsAPI(wire)
            assert isinstance(
                await api.steer(RUN, "message", idempotency_key="key"), SwarmControlReceipt
            )
            assert (await api.answer_action(RUN, "action_1", "answer")).kind == "action_answer"
            assert json.loads(requests[1].content) == {"response_text": "answer"}
        finally:
            await wire.close()

    asyncio.run(scenario())


def test_catalog_mirrors_backend_authored_openapi():
    from synth_ai.sdk.research.operations import research_operation

    schema = json.loads((Path(__file__).parents[1] / "openapi/research-v1.json").read_text())
    for name, path in [
        ("steer_run", "/smr/runs/{run_id}/steer"),
        ("answer_run_action", "/smr/runs/{run_id}/actions/{action_id}/answer"),
    ]:
        assert schema["paths"][path]["post"]["operationId"] == name
        operation = research_operation(name)
        assert operation.path_template == path and operation.mutation and operation.idempotent
