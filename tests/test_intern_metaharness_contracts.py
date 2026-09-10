from __future__ import annotations

from typing import Any, cast

from synth_ai.core.http.request import HttpRequest
from synth_ai.mcp.research.tools.research_intern import build_research_intern_tools
from synth_ai.sdk.research.contracts.research_intern import (
    InternCrossMetaThreadMessageCreateRequest,
    InternCrossMetaThreadMessageKind,
)
from synth_ai.sdk.research.research_intern import ResearchInternAPI

NOW = "2026-08-05T12:00:00Z"
SYNC_THREAD = {
    "schema_version": "smr.meta-thread.v1",
    "meta_thread_id": "meta-sync",
    "organization_id": "organization-1",
    "research_intern_id": "intern-1",
    "kind": "sync",
    "lifecycle": "active",
    "head_segment_id": "segment-root",
    "created_at": NOW,
    "updated_at": NOW,
}
ASYNC_THREAD = {**SYNC_THREAD, "meta_thread_id": "meta-async", "kind": "async"}
SEGMENT = {
    "schema_version": "smr.meta-thread-segment.v1",
    "segment_id": "segment-a",
    "meta_thread_id": "meta-sync",
    "parent_segment_id": "segment-root",
    "lane_runtime_id": "session-a",
    "status": "live",
    "opened_at": NOW,
    "is_head": False,
}


class _Transport:
    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        operation_id = str(request.operation.operation_id)
        if operation_id == "list_intern_meta_threads":
            return [SYNC_THREAD, ASYNC_THREAD]
        if operation_id == "list_intern_meta_thread_segments":
            return [SEGMENT]
        if operation_id == "list_intern_meta_thread_messages":
            return []
        if operation_id == "create_intern_meta_thread_message":
            assert request.body is not None
            return {
                "schema_version": "smr.cross-meta-thread-message.v1",
                "message_id": request.body["message_id"],
                "organization_id": "organization-1",
                "research_intern_id": "intern-1",
                "source_meta_thread_id": request.body["source_meta_thread_id"],
                "source_kind": "async",
                "destination_meta_thread_id": request.body[
                    "destination_meta_thread_id"
                ],
                "destination_kind": "sync",
                "kind": request.body["kind"],
                "idempotency_key": request.body["idempotency_key"],
                "payload": request.body.get("payload", {}),
                "created_at": NOW,
            }
        raise AssertionError(f"unexpected operation: {operation_id}")


class _ClientContext:
    def __init__(self, transport: _Transport) -> None:
        self.intern = ResearchInternAPI(cast(Any, transport))

    def __enter__(self) -> _ClientContext:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_sdk_lists_sync_branches_from_meta_thread_projection() -> None:
    transport = _Transport()
    intern = ResearchInternAPI(cast(Any, transport))

    branches = intern.sync_.branches()

    assert len(branches) == 1
    assert branches[0].segment_id == "segment-a"
    assert [str(request.operation.operation_id) for request in transport.requests] == [
        "list_intern_meta_threads",
        "list_intern_meta_thread_segments",
    ]


def test_message_id_has_http_sdk_and_mcp_parity() -> None:
    message_id = "747e0caa-f127-48b1-af40-c0064265fd49"
    transport = _Transport()
    intern = ResearchInternAPI(cast(Any, transport))
    request = InternCrossMetaThreadMessageCreateRequest(
        message_id=message_id,
        source_meta_thread_id="meta-async",
        destination_meta_thread_id="meta-sync",
        kind=InternCrossMetaThreadMessageKind.REQUEST_DECISION,
        idempotency_key="request-1",
        payload={"question": "Continue?"},
    )

    sdk_message = intern.meta_threads.send(request)
    assert sdk_message.message_id == message_id
    assert transport.requests[-1].body is not None
    assert transport.requests[-1].body["message_id"] == message_id

    tools = {
        tool.name: tool
        for tool in build_research_intern_tools(
            lambda _args: cast(Any, _ClientContext(transport))
        )
    }
    mcp_message = tools["intern_meta_send"].handler(
        {
            "message_id": message_id,
            "source_meta_thread_id": "meta-async",
            "destination_meta_thread_id": "meta-sync",
            "kind": "request_decision",
            "idempotency_key": "request-1",
            "payload": {"question": "Continue?"},
        }
    )
    assert mcp_message["message_id"] == message_id
    assert transport.requests[-1].body is not None
    assert transport.requests[-1].body["message_id"] == message_id


def test_mcp_sync_list_is_branch_alias() -> None:
    transport = _Transport()
    tools = {
        tool.name: tool
        for tool in build_research_intern_tools(
            lambda _args: cast(Any, _ClientContext(transport))
        )
    }
    assert tools["intern_sync_list"].handler({}) == tools[
        "intern_sync_branches"
    ].handler({})
    assert tools["intern_sync_list"].handler({})[0]["segment_id"] == "segment-a"
