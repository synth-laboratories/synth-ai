"""Explicit Async blocker controls; see backend packages/intern/contracts.py."""

from .contracts.intern_blockers import (
    InternBlockerOpenSyncRequest,
    InternBlockerOpenSyncResponse,
    InternBlockerResolveRequest,
    InternBlockerResolveResponse,
)
from .contracts.research_intern import InternAsyncBlocker


def _request_for(blocker_id, action=None, request=None):
    from .research_intern import _request

    operations = {
        None: "get_intern_async_blocker",
        "open-sync": "open_intern_async_blocker_sync",
        "resolve": "resolve_intern_async_blocker",
    }
    path = f"/smr/research-intern/async/blockers/{blocker_id}"
    if action:
        path += f"/{action}"
    return _request(
        operations[action], path, **({"body": request.model_dump(mode="json")} if request else {})
    )


def _read(payload, blocker_id):
    result = InternAsyncBlocker.from_wire(payload)
    if result.blocker_id != blocker_id:
        raise ValueError("Async blocker identity drifted")
    return result


def _open(payload, blocker_id, request):
    result = InternBlockerOpenSyncResponse.model_validate(payload)
    receipt = result.handoff_receipt
    if (
        result.blocker.blocker_id != blocker_id
        or receipt.blocker_id != blocker_id
        or receipt.context.blocker_id != blocker_id
        or receipt.context.async_assignment_id != result.blocker.async_assignment_id
        or receipt.idempotency_key != request.idempotency_key
        or receipt.sync_session_id != result.sync_session.sync_session_id
        or result.blocker.sync_session_id != receipt.sync_session_id
        or receipt.org_id != result.sync_session.org_id
    ):
        raise ValueError("Async blocker handoff identity drifted")
    return result


def _resolve(payload, blocker_id, request):
    result = InternBlockerResolveResponse.model_validate(payload)
    receipt = result.blocker.resolution_receipt or {}
    if (
        result.blocker.blocker_id != blocker_id
        or not result.blocker.async_assignment_id
        or result.continuation_command.runtime_id != result.blocker.async_assignment_id
        or receipt.get("idempotency_key") != request.idempotency_key
        or receipt.get("outcome") != request.outcome
        or receipt.get("comment") != request.comment
        or tuple(receipt.get("supporting_receipt_ids") or ()) != request.supporting_receipt_ids
        or receipt.get("continuation_command_id") != result.continuation_command.command_id
        or receipt.get("sync_session_id") != result.blocker.sync_session_id
    ):
        raise ValueError("Async blocker resolution identity drifted")
    return result


class InternBlockersAPI:
    def __init__(self, transport):
        self._transport = transport

    def get(self, blocker_id: str) -> InternAsyncBlocker:
        return _read(self._transport.execute(_request_for(blocker_id)), blocker_id)

    def open_sync(
        self, blocker_id: str, request: InternBlockerOpenSyncRequest
    ) -> InternBlockerOpenSyncResponse:
        """Open the handoff; this does not approve or resolve the blocked action."""
        return _open(
            self._transport.execute(_request_for(blocker_id, "open-sync", request)),
            blocker_id,
            request,
        )

    def resolve(
        self, blocker_id: str, request: InternBlockerResolveRequest
    ) -> InternBlockerResolveResponse:
        """Submit an explicit disposition and return its durable continuation receipt."""
        return _resolve(
            self._transport.execute(_request_for(blocker_id, "resolve", request)),
            blocker_id,
            request,
        )


class AsyncInternBlockersAPI:
    def __init__(self, transport):
        self._transport = transport

    async def get(self, blocker_id: str) -> InternAsyncBlocker:
        return _read(await self._transport.execute(_request_for(blocker_id)), blocker_id)

    async def open_sync(
        self, blocker_id: str, request: InternBlockerOpenSyncRequest
    ) -> InternBlockerOpenSyncResponse:
        """Open the handoff; this does not approve or resolve the blocked action."""
        return _open(
            await self._transport.execute(_request_for(blocker_id, "open-sync", request)),
            blocker_id,
            request,
        )

    async def resolve(
        self, blocker_id: str, request: InternBlockerResolveRequest
    ) -> InternBlockerResolveResponse:
        """Submit an explicit disposition and return its durable continuation receipt."""
        return _resolve(
            await self._transport.execute(_request_for(blocker_id, "resolve", request)),
            blocker_id,
            request,
        )
