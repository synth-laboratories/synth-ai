from unittest.mock import AsyncMock, Mock

import pytest

from synth_ai.sdk.research.intern_blockers import (
    InternBlockersAPI,
    AsyncInternBlockersAPI,
    _request_for,
)
from synth_ai.sdk.research.contracts.intern_blockers import (
    InternBlockerOpenSyncRequest,
    InternBlockerResolveRequest,
)


def blocker():
    return dict(blocker_id="b", code="approval", message="review required", retryable=False)


def open_response():
    now = "2026-09-12T00:00:00Z"
    return dict(
        blocker={**blocker(), "sync_session_id": "sync"},
        sync_session=dict(sync_session_id="sync", research_intern_id="intern", org_id="org",
                          objective="review", status="ready", state_generation=0,
                          last_event_sequence=0, binding={}, temporal_workflow_id="wf",
                          execution_mode="fast", created_at=now, updated_at=now),
        handoff_receipt=dict(handoff_receipt_id="h", blocker_id="b", org_id="org",
                             sync_session_id="sync", opened_by_user_id="user", idempotency_key="k",
                             context_digest="a" * 64, created_at=now,
                             context=dict(blocker_id="b", async_assignment_id="async", binding={},
                                          action_schema_version="v1", action_kind="test", action_digest="b" * 64,
                                          summary="review", rationale="approval", preauthorization_rule="operator",
                                          required_operator_capability="approve")),
    )


def test_open_handoff_checks_cross_receipt_identity():
    transport = Mock()
    transport.execute.return_value = open_response()
    api = InternBlockersAPI(transport)
    request = InternBlockerOpenSyncRequest(idempotency_key="k")
    assert api.open_sync("b", request).sync_session.sync_session_id == "sync"
    transport.execute.return_value["handoff_receipt"]["sync_session_id"] = "other"
    with pytest.raises(ValueError, match="identity drifted"):
        api.open_sync("b", request)


@pytest.mark.asyncio
async def test_async_controls_parse_handoff_and_continuation():
    transport = Mock(execute=AsyncMock(return_value=open_response()))
    api = AsyncInternBlockersAPI(transport)
    assert (await api.open_sync("b", InternBlockerOpenSyncRequest(idempotency_key="k"))).blocker.blocker_id == "b"
    transport.execute.return_value = dict(blocker=blocker(), continuation_command=dict(
        schema_version="smr.intern-runtime-command-receipt.v1", command_id="command",
        runtime_kind="async", runtime_id="async", status="received", previous_generation=0,
        state_generation=1, decision_code="accepted", created_at="2026-09-12T00:00:00Z",
    ))
    transport.execute.return_value["blocker"]["resolution_receipt"] = dict(
        idempotency_key="resolve", outcome="denied", comment=None,
        supporting_receipt_ids=[], continuation_command_id="command", sync_session_id="sync",
    )
    transport.execute.return_value["blocker"]["sync_session_id"] = "sync"
    result = await api.resolve("b", InternBlockerResolveRequest(idempotency_key="resolve", outcome="denied"))
    assert result.continuation_command.command_id == "command"
    assert result.continuation_command.status == "received"
    transport.execute.return_value["blocker"]["resolution_receipt"]["outcome"] = "completed"
    with pytest.raises(ValueError, match="identity drifted"):
        await api.resolve("b", InternBlockerResolveRequest(idempotency_key="resolve", outcome="denied"))


def test_blocker_read_preserves_exact_identity():
    transport = Mock()
    transport.execute.return_value = blocker()
    assert InternBlockersAPI(transport).get("b").blocker_id == "b"
    assert transport.execute.call_args.args[0].path == "/smr/research-intern/async/blockers/b"
    transport.execute.return_value["blocker_id"] = "other"
    with pytest.raises(ValueError, match="identity drifted"):
        InternBlockersAPI(transport).get("b")


@pytest.mark.asyncio
async def test_async_blocker_read_uses_async_transport():
    transport = Mock(execute=AsyncMock(return_value=blocker()))
    assert (await AsyncInternBlockersAPI(transport).get("b")).blocker_id == "b"
    transport.execute.assert_awaited_once()


@pytest.mark.parametrize("ids", [("",), ("x", "x"), ("x" * 513,)])
def test_resolution_rejects_invalid_supporting_receipts(ids):
    with pytest.raises(ValueError):
        InternBlockerResolveRequest(
            idempotency_key="k", outcome="completed", supporting_receipt_ids=ids
        )


def test_actions_keep_explicit_idempotency_and_disposition():
    opening = _request_for("b", "open-sync", InternBlockerOpenSyncRequest(idempotency_key="open-1"))
    resolution = _request_for(
        "b", "resolve", InternBlockerResolveRequest(idempotency_key="resolve-1", outcome="denied")
    )
    assert opening.path.endswith("/b/open-sync")
    assert resolution.path.endswith("/b/resolve")
    assert opening.body == {"idempotency_key": "open-1"}
    assert resolution.body["outcome"] == "denied"
    assert resolution.body["idempotency_key"] == "resolve-1"
