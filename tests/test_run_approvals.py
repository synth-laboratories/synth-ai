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


# ---- SYN-4650: human-in-the-loop waits on the same surface ----

WAIT = {
    "metadata": METADATA,
    "approval_id": "wait-1",
    "run_id": "run-1",
    "revision": 1,
    "state": "awaiting",
    "kind": "wait",
    "question": "Ship the migration?",
    "tool": None,
    "capability": None,
    "argument_digest": None,
    "actor_id": None,
    "attempt_id": None,
    "parked_at": "2026-10-29T11:59:00Z",
    "deadline_at": "2026-10-29T12:29:00Z",
    "on_timeout": "decline",
    "decidable": True,
    "wait_kind": "approval",
    "options": ["accept", "decline"],
    "answer_schema": "orchestra.answer.approval.v1",
    "answer_scope": "once",
    "answerers": ["role:owner"],
    "posture": "pause_actor",
}


def wait_receipt(**overrides):
    value = {
        "metadata": METADATA,
        "run_id": "run-1",
        "approval_id": "wait-1",
        "operation_id": "b" * 64,
        "decision": "accept",
        "answered_revision": 1,
        "revision": 2,
        "answerer": "role:owner",
        "answer_scope": "once",
        "answer_schema": "orchestra.answer.approval.v1",
        "answered_at": "2026-10-29T12:01:00Z",
    }
    value.update(overrides)
    return value


def test_list_carries_waits_beside_tool_calls():
    def respond(request):
        return httpx.Response(
            200,
            headers={"X-Synth-Contract": "swarms.v1"},
            json={"metadata": METADATA, "run_id": "run-1", "items": [APPROVAL, WAIT]},
        )

    with sync_client(respond) as client:
        result = SwarmRun(client._transport, "run-1").approvals()
    tool, wait = result.items
    assert (tool.kind, wait.kind, wait.wait_kind) == ("tool_call", "wait", "approval")
    assert wait.options == ("accept", "decline") and wait.on_timeout == "decline"
    assert wait.deadline_at is not None and wait.tool is None


def test_answer_wait_sends_one_keyed_answer_with_expected_revision():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=wait_receipt())

    with sync_client(respond) as client:
        result = SwarmRun(client._transport, "run-1").answer_wait(
            "wait-1",
            decision="accept",
            expected_revision=1,
            mutation=MutationOptions(idempotency_key="answer-1"),
        )
    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/api/v1/swarms/runs/run-1/approvals/wait-1/answer"
    assert request.headers["Idempotency-Key"] == "answer-1"
    assert json.loads(request.content) == {
        "decision": "accept",
        "expected_revision": 1,
        "answer_scope": None,
        "answer_schema": None,
        "answerer": None,
    }
    assert (result.revision, result.answerer) == (2, "role:owner")


def test_answer_wait_refuses_a_receipt_for_another_answer():
    def respond(request):
        return httpx.Response(
            200,
            headers={"X-Synth-Contract": "swarms.v1"},
            json=wait_receipt(answerer="role:admin"),
        )

    with sync_client(respond) as client, pytest.raises(OutcomeUncertainError):
        SwarmRun(client._transport, "run-1").answer_wait(
            "wait-1",
            decision="accept",
            expected_revision=1,
            answerer="role:owner",
            mutation=MutationOptions(idempotency_key="answer-2"),
        )


def test_async_answer_wait_uses_the_same_contract():
    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "swarms.v1"},
                json=wait_receipt(decision="decline", answer_scope="once"),
            )

        async with AsyncSynthClient(
            api_key="fixture-key", base_url="https://backend.test"
        ) as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(*args):
                return None

            client._transport._negotiate = negotiated
            result = await AsyncSwarmRun(client._transport, "run-1").answer_wait(
                "wait-1",
                decision="decline",
                expected_revision=1,
                answer_scope="once",
                mutation=MutationOptions(idempotency_key="answer-3"),
            )
        assert result.decision == "decline"
        assert json.loads(requests[0].content)["answer_scope"] == "once"

    asyncio.run(run())


def test_wait_answer_contract_and_event_kinds_are_closed():
    from synth_ai.sdk.swarms.types.approvals import WaitAnswerRequest
    from synth_ai.sdk.swarms.types.runs import SwarmEvent

    with pytest.raises(ValidationError):
        WaitAnswerRequest(decision="accept", expected_revision=0)
    with pytest.raises(ValidationError):
        WaitAnswerRequest.model_validate(
            {"decision": "accept", "expected_revision": 1, "answer_scope": "forever"}
        )
    assert operations.ANSWER_WAIT.mutation and operations.ANSWER_WAIT.idempotent
    assert operations.ANSWER_WAIT.required_scope == "swarms:control"
    assert BY_NAME["swarms_wait_answer"].definition is operations.ANSWER_WAIT
    event = SwarmEvent.model_validate(
        {
            "event_id": "run-1.f17.0",
            "run_id": "run-1",
            "sequence": 17,
            "kind": "wait.opened",
            "approval_id": "wait-1",
        }
    )
    assert event.model_dump(mode="json")["approval_id"] == "wait-1"
    with pytest.raises(ValidationError):
        SwarmEvent.model_validate(
            {"event_id": "e", "run_id": "run-1", "sequence": 1, "kind": "approval.parked"}
        )
    plain = SwarmEvent.model_validate(
        {"event_id": "e", "run_id": "run-1", "sequence": 1, "kind": "created"}
    )
    assert "approval_id" not in plain.model_dump(mode="json")
