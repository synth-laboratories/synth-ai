from __future__ import annotations

import asyncio
from typing import Any

import pytest
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS, research_operation
from synth_ai.sdk.research.research_intern import (
    AsyncResearchInternAPI,
    ResearchInternAPI,
)

NOW = "2026-09-10T12:00:00Z"
SESSION = "sync-1"
LEASE = {
    "schema_version": "smr.intern-sync-presence.v1",
    "lease_id": "lease-1",
    "sync_session_id": SESSION,
    "org_id": "org-1",
    "user_id": "user-1",
    "connection_id": "conn-1",
    "connection_generation": 1,
    "acquired_at": NOW,
    "renewed_at": NOW,
    "expires_at": NOW,
    "interactive_approval_available": True,
}
CARD = {
    "schema_version": "smr.sync-approval-card.v1",
    "approval_id": "approval-1",
    "org_id": "org-1",
    "sync_session_id": SESSION,
    "scope_kind": "sync_session",
    "scope_id": SESSION,
    "decision_state": "pending",
    "application_state": "not_started",
    "title": "Launch rollout",
    "created_at": NOW,
}

RETIRED_LEGACY_SESSION_IDS = (
    "create_research_intern_session",
    "list_research_intern_sessions",
    "get_research_intern_session",
    "append_research_intern_session_event",
    "list_research_intern_session_events",
    "stream_research_intern_session_events",
    "sync_research_intern_session",
    "create_research_intern_session_turn",
    "close_research_intern_session",
    "publish_research_intern_session_trace",
    # Former compatibility aliases.
    "append_research_intern_event",
    "list_research_intern_events",
    "get_research_intern_acceptance_receipt",
)
PRESENCE_APPROVAL_OPERATIONS = {
    "acquire_intern_sync_presence": (
        "PUT",
        "/smr/research-intern/sync-sessions/{sync_session_id}/presence",
    ),
    "release_intern_sync_presence": (
        "POST",
        "/smr/research-intern/sync-sessions/{sync_session_id}/presence/release",
    ),
    "list_intern_sync_approvals": (
        "GET",
        "/smr/research-intern/sync-sessions/{sync_session_id}/approvals",
    ),
    "decide_intern_sync_approval": (
        "POST",
        "/smr/research-intern/sync-approvals/{approval_id}/decision",
    ),
}


def _response(request: HttpRequest) -> Any:
    operation_id = str(request.operation.operation_id)
    if operation_id in {"acquire_intern_sync_presence", "release_intern_sync_presence"}:
        return LEASE
    if operation_id == "list_intern_sync_approvals":
        return [CARD]
    if operation_id == "decide_intern_sync_approval":
        assert request.body is not None
        return {**CARD, "decision_state": "approved", "future_field": "ignored"}
    raise AssertionError(f"unexpected operation: {operation_id}")


class _Transport:
    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        return _response(request)


class _AsyncTransport(_Transport):
    async def execute(self, request: HttpRequest) -> Any:  # type: ignore[override]
        self.requests.append(request)
        return _response(request)


@pytest.mark.parametrize("operation_id", RETIRED_LEGACY_SESSION_IDS)
def test_legacy_session_operations_and_aliases_are_unknown(operation_id: str) -> None:
    with pytest.raises(ValueError, match="unknown Research operation_id"):
        research_operation(operation_id)


def test_presence_and_approval_operations_are_registered() -> None:
    registered = {str(key): value for key, value in RESEARCH_OPERATIONS.items()}
    for operation_id, (method, path) in PRESENCE_APPROVAL_OPERATIONS.items():
        operation = registered[operation_id]
        assert str(operation.method.value).upper() == method
        assert operation.path_template == path


def test_legacy_sessions_namespace_is_gone() -> None:
    api = ResearchInternAPI(_Transport())  # type: ignore[arg-type]
    assert not hasattr(api, "sessions")


def test_sync_presence_and_approval_round_trip() -> None:
    transport = _Transport()
    sync = ResearchInternAPI(transport).sync_  # type: ignore[arg-type]

    lease = sync.presence(SESSION, connection_id="conn-1")
    assert lease.lease_id == "lease-1"
    released = sync.release_presence(SESSION, connection_id="conn-1")
    assert released.sync_session_id == SESSION
    cards = sync.approvals(SESSION)
    assert [card.approval_id for card in cards] == ["approval-1"]
    decided = sync.decide_approval("approval-1", decision="approve", comment="ok")
    assert decided.decision_state == "approved"

    sent = [(str(r.operation.operation_id), r.path, r.body) for r in transport.requests]
    assert sent == [
        (
            "acquire_intern_sync_presence",
            f"/smr/research-intern/sync-sessions/{SESSION}/presence",
            {"connection_id": "conn-1", "connection_generation": 1},
        ),
        (
            "release_intern_sync_presence",
            f"/smr/research-intern/sync-sessions/{SESSION}/presence/release",
            {"connection_id": "conn-1", "connection_generation": 1},
        ),
        (
            "list_intern_sync_approvals",
            f"/smr/research-intern/sync-sessions/{SESSION}/approvals",
            None,
        ),
        (
            "decide_intern_sync_approval",
            "/smr/research-intern/sync-approvals/approval-1/decision",
            {"decision": "approve", "comment": "ok"},
        ),
    ]


def test_presence_lease_for_another_session_fails_closed() -> None:
    sync = ResearchInternAPI(_Transport()).sync_  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="presence lease identity drifted"):
        sync.presence("sync-other", connection_id="conn-1")


def test_async_client_exposes_the_same_presence_and_approval_calls() -> None:
    transport = _AsyncTransport()
    sync = AsyncResearchInternAPI(transport).sync_  # type: ignore[arg-type]

    async def drive() -> None:
        assert (await sync.presence(SESSION, connection_id="conn-1")).lease_id == "lease-1"
        assert len(await sync.approvals(SESSION)) == 1
        decided = await sync.decide_approval("approval-1", decision="deny")
        assert decided.decision_state == "approved"

    asyncio.run(drive())
    assert [str(r.operation.operation_id) for r in transport.requests] == [
        "acquire_intern_sync_presence",
        "list_intern_sync_approvals",
        "decide_intern_sync_approval",
    ]
