"""Explicit Async blocker controls; see backend packages/intern/contracts.py."""

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

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
    """Synchronous read, operator handoff, and explicit resolution of Async blockers.

    ```python
    from synth_ai.sdk.research.intern_blockers import InternBlockersAPI

    from unittest.mock import Mock

    transport = Mock()
    transport.execute.return_value = {
        "blocker_id": "blocker-1", "code": "operator_required",
        "message": "Review the blocked action", "retryable": False,
    }
    api = InternBlockersAPI(transport)
    blocker = api.get("blocker-1")
    assert blocker.blocker_id == "blocker-1"
    ```
    """

    def __init__(self, transport: HttpTransport) -> None:
        """Bind blocker controls to an existing HTTP transport.

        Args:
            transport: Authenticated transport owned by the parent Synth client.
        """
        self._transport = transport

    def get(self, blocker_id: str) -> InternAsyncBlocker:
        """Retrieve an Async blocker and verify its requested identity.

        Args:
            blocker_id: Identity of the Async blocker to address.

        Returns:
            The blocker matching the requested identity.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _read(self._transport.execute(_request_for(blocker_id)), blocker_id)

    def open_sync(
        self, blocker_id: str, request: InternBlockerOpenSyncRequest
    ) -> InternBlockerOpenSyncResponse:
        """Open the handoff; this does not approve or resolve the blocked action.

        Args:
            blocker_id: Identity of the Async blocker to address.
            request: Idempotent handoff-open request.

        Returns:
            The blocker, operator Sync session, and durable handoff receipt.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _open(
            self._transport.execute(_request_for(blocker_id, "open-sync", request)),
            blocker_id,
            request,
        )

    def resolve(
        self, blocker_id: str, request: InternBlockerResolveRequest
    ) -> InternBlockerResolveResponse:
        """Submit an explicit disposition and return its durable continuation receipt.

        Args:
            blocker_id: Identity of the Async blocker to address.
            request: Explicit disposition and supporting resolution receipts.

        Returns:
            The resolved blocker and durable continuation command.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _resolve(
            self._transport.execute(_request_for(blocker_id, "resolve", request)),
            blocker_id,
            request,
        )


class AsyncInternBlockersAPI:
    """Asynchronous read, operator handoff, and explicit resolution of Async blockers.

    ```python
    from synth_ai.sdk.research.intern_blockers import AsyncInternBlockersAPI

    import asyncio
    from unittest.mock import AsyncMock

    transport = AsyncMock()
    transport.execute.return_value = {
        "blocker_id": "blocker-1", "code": "operator_required",
        "message": "Review the blocked action", "retryable": False,
    }
    api = AsyncInternBlockersAPI(transport)
    async def read_blocker():
        return await api.get("blocker-1")

    blocker = asyncio.run(read_blocker())
    assert blocker.blocker_id == "blocker-1"
    ```
    """

    def __init__(self, transport: AsyncHttpTransport) -> None:
        """Bind blocker controls to an existing HTTP transport.

        Args:
            transport: Authenticated transport owned by the parent Synth client.
        """
        self._transport = transport

    async def get(self, blocker_id: str) -> InternAsyncBlocker:
        """Retrieve an Async blocker and verify its requested identity.

        Args:
            blocker_id: Identity of the Async blocker to address.

        Returns:
            The blocker matching the requested identity.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _read(await self._transport.execute(_request_for(blocker_id)), blocker_id)

    async def open_sync(
        self, blocker_id: str, request: InternBlockerOpenSyncRequest
    ) -> InternBlockerOpenSyncResponse:
        """Open the handoff; this does not approve or resolve the blocked action.

        Args:
            blocker_id: Identity of the Async blocker to address.
            request: Idempotent handoff-open request.

        Returns:
            The blocker, operator Sync session, and durable handoff receipt.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _open(
            await self._transport.execute(_request_for(blocker_id, "open-sync", request)),
            blocker_id,
            request,
        )

    async def resolve(
        self, blocker_id: str, request: InternBlockerResolveRequest
    ) -> InternBlockerResolveResponse:
        """Submit an explicit disposition and return its durable continuation receipt.

        Args:
            blocker_id: Identity of the Async blocker to address.
            request: Explicit disposition and supporting resolution receipts.

        Returns:
            The resolved blocker and durable continuation command.

        Raises:
            ValueError: Response identities or receipt fields disagree with the request.
        """
        return _resolve(
            await self._transport.execute(_request_for(blocker_id, "resolve", request)),
            blocker_id,
            request,
        )
