"""SYN-4547: run-object approval list/decide keep identity, revision and key offline."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts import swarms as operations
from synth_ai.errors import ContractError, OutcomeUncertainError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import MutationOptions
from synth_ai.sdk.swarms.run import AsyncSwarmRun, SwarmRun
from synth_ai.sdk.swarms.types.approvals import ApprovalDecisionRequest, ApprovalList

pytestmark = pytest.mark.unit
METADATA = {
    "contract": "swarms.v1",
    "request_id": "approval-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}
APPROVAL = {
    "metadata": METADATA,
    "approval_id": "approval-1",
    "run_id": "run-1",
    "revision": 3,
    "state": "awaiting",
    "kind": "tool_call",
    "question": "Allow tool git_push (capability git.push.remote)?",
    "tool": "git_push",
    "capability": "git.push.remote",
    "argument_digest": "sha256:" + "a" * 64,
    "actor_id": "actor-1",
    "attempt_id": "attempt-1",
    "parked_at": "2026-10-29T11:59:00Z",
    "deadline_at": None,
    "on_timeout": None,
    "decidable": True,
}


def listing(run_id="run-1"):
    return {"metadata": METADATA, "run_id": run_id, "items": [APPROVAL]}


def receipt(revision=3, decision="approve"):
    return {
        "metadata": METADATA,
        "run_id": "run-1",
        "approval_id": "approval-1",
        "decision": decision,
        "decided_revision": revision,
        "verdict": "enacted",
    }


def sync_client(respond):
    client = SynthClient(api_key="fixture-key", base_url="https://backend.test")
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    client._transport._negotiate = lambda *args: None
    return client


def test_list_reads_typed_state_and_revision():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=listing())

    with sync_client(respond) as client:
        result = SwarmRun(client._transport, "run-1").approvals()
    assert [r.url.path for r in requests] == ["/api/v1/swarms/runs/run-1/approvals"]
    assert requests[0].method == "GET"
    (item,) = result.items
    assert (item.state, item.revision, item.decidable) == ("awaiting", 3, True)


def test_list_refuses_a_foreign_run_identity():
    def respond(request):
        return httpx.Response(
            200, headers={"X-Synth-Contract": "swarms.v1"}, json=listing("run-2")
        )

    with sync_client(respond) as client, pytest.raises(ContractError):
        SwarmRun(client._transport, "run-1").approvals()


def test_decide_sends_one_keyed_decision_with_expected_revision():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=receipt())

    with sync_client(respond) as client:
        result = SwarmRun(client._transport, "run-1").decide_approval(
            "approval-1",
            decision="approve",
            expected_revision=3,
            mutation=MutationOptions(idempotency_key="decide-1"),
        )
    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/api/v1/swarms/runs/run-1/approvals/approval-1/decision"
    assert request.headers["Idempotency-Key"] == "decide-1"
    assert json.loads(request.content) == {
        "decision": "approve",
        "expected_revision": 3,
        "reason": None,
    }
    assert result.verdict == "enacted" and result.decided_revision == 3


def test_decide_refuses_a_receipt_for_another_revision():
    def respond(request):
        return httpx.Response(
            200, headers={"X-Synth-Contract": "swarms.v1"}, json=receipt(revision=4)
        )

    # A mismatched mutation receipt is never trusted: the caller reconciles.
    with sync_client(respond) as client, pytest.raises(OutcomeUncertainError):
        SwarmRun(client._transport, "run-1").decide_approval(
            "approval-1",
            decision="approve",
            expected_revision=3,
            mutation=MutationOptions(idempotency_key="decide-2"),
        )


def test_async_decide_uses_the_same_contract():
    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200, headers={"X-Synth-Contract": "swarms.v1"}, json=receipt(2, "refuse")
            )

        async with AsyncSynthClient(
            api_key="fixture-key", base_url="https://backend.test"
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(*args):
                return None

            client._transport._negotiate = negotiated
            result = await AsyncSwarmRun(client._transport, "run-1").decide_approval(
                "approval-1",
                decision="refuse",
                expected_revision=2,
                reason="remote not allowed",
                mutation=MutationOptions(idempotency_key="decide-3"),
            )
        assert result.decision == "refuse"
        assert json.loads(requests[0].content)["reason"] == "remote not allowed"

    asyncio.run(run())


def test_request_and_operations_are_closed():
    with pytest.raises(ValidationError):
        ApprovalDecisionRequest(decision="approve", expected_revision=0)
    with pytest.raises(ValidationError):
        ApprovalDecisionRequest.model_validate({"decision": "maybe", "expected_revision": 1})
    assert ApprovalList.model_validate_json(json.dumps(listing())).run_id == "run-1"
    assert operations.APPROVALS.required_scope == "swarms:read"
    assert operations.DECIDE_APPROVAL.mutation and operations.DECIDE_APPROVAL.idempotent
    assert operations.DECIDE_APPROVAL.required_scope == "swarms:control"
    assert BY_NAME["swarms_approvals"].definition is operations.APPROVALS
    assert BY_NAME["swarms_approval_decide"].definition is operations.DECIDE_APPROVAL
