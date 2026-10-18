"""Borrowed job handles never own connector or registration shutdown.

# See: backend/notes/specifications/tanha/current/systems/tunnels/v2_catalog.md
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from .client import AsyncTunnelsClient, TunnelsClient
from .contracts import CatalogCommand, CatalogMutation, ConnectorStatus
from .lease_custody import HandoffOffer


def _verify_handoff(request: HandoffOffer, receipt: dict, current: dict) -> bool:
    if receipt["state"] != "accepted":
        return False
    if (
        receipt["offer_id"] != str(request.offer)
        or receipt["caller_binding"] != str(request.caller_binding)
        or receipt["job_id"] != request.job_id
        or receipt["expected_lease_revision"] != request.expected_lease_revision
        or current["owner_binding"] != str(request.offer)
        or current["revision"] < receipt["lease_revision"]
    ):
        raise ValueError("lease handoff receipt does not match retained custody")
    return True


@dataclass(slots=True)
class BorrowedLeaseHandle:
    connector: UUID
    lease: UUID
    client: TunnelsClient = field(repr=False)
    lease_revision: int = 1
    _handoff_request: Optional[HandoffOffer] = field(default=None, repr=False)
    _detached: bool = False
    _close_command: Optional[CatalogCommand] = field(default=None, repr=False)
    _close_receipt: Optional[CatalogMutation] = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def observer(self) -> LeaseObserver:
        """Borrow a local observation reference without transferring job ownership."""
        return LeaseObserver(self.connector, self.lease, self.client)

    def prepare_handoff(self, job_id: str) -> dict:
        """Retain an exact offer before submission; acceptance must be observed separately."""
        with self._lock:
            if self._close_command is not None or self._detached:
                raise RuntimeError("lease custody already released")
            if self._handoff_request is None:
                current = self.client.lease_status(self.connector, self.lease)
                if current["revision"] != self.lease_revision or current["policy"] != "granted":
                    raise RuntimeError("caller lease custody is stale")
                self._handoff_request = HandoffOffer(
                    offer=uuid4(),
                    expected_lease_revision=self.lease_revision,
                    caller_binding=UUID(current["owner_binding"]),
                    job_id=job_id,
                    deadline_at=datetime.now(UTC) + timedelta(minutes=2),
                )
            if self._handoff_request.job_id != job_id:
                raise ValueError("retained handoff is bound to another job")
            return self.client.offer_lease_handoff(
                self.connector, self.lease, self._handoff_request
            )

    def handoff_status(self) -> dict:
        """Read durable custody; this never acknowledges an unobserved transfer."""
        if self._handoff_request is None:
            raise RuntimeError("lease handoff was not offered")
        return self.client.lease_handoff(self.connector, self.lease, self._handoff_request.offer)

    def _observe_handoff(self) -> bool:
        if self._handoff_request is None:
            return False
        receipt = self.handoff_status()
        current = self.client.lease_status(self.connector, self.lease)
        if _verify_handoff(self._handoff_request, receipt, current):
            self._detached = True
        return self._detached

    def close(self) -> Optional[CatalogMutation]:
        """Commit only this lease's revocation intent, retaining uncertain command identity.

        A returned receipt is policy acceptance. It does not prove running
        origin work stopped or that cleanup finished. Reconcile by command ID.
        """
        with self._lock:
            if self._detached or self._observe_handoff():
                return None
            if self._close_receipt is not None:
                return self._close_receipt
            if self._close_command is None:
                revision = self.client.status(self.connector).revision
                self._close_command = CatalogCommand(
                    command_id=uuid4(),
                    expected_revision=revision,
                    deadline_at=datetime.now(UTC) + timedelta(seconds=30),
                )
            self._close_receipt = self.client.revoke_lease(
                self.connector,
                self.lease,
                command=self._close_command,
                expected_lease_revision=self.lease_revision,
            )
            return self._close_receipt

    def close_status(self) -> CatalogMutation:
        """Observe the retained close command without repeating its side effect."""
        if self._close_command is None:
            raise ValueError("lease close has not been requested")
        return self.client.command_status(self.connector, self._close_command.command_id)


@dataclass(slots=True)
class AsyncBorrowedLeaseHandle:
    connector: UUID
    lease: UUID
    client: AsyncTunnelsClient = field(repr=False)
    lease_revision: int = 1
    _handoff_request: Optional[HandoffOffer] = field(default=None, repr=False)
    _detached: bool = False
    _close_command: Optional[CatalogCommand] = field(default=None, repr=False)
    _close_receipt: Optional[CatalogMutation] = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def observer(self) -> AsyncLeaseObserver:
        """Observe a job without acquiring its revocation responsibility."""
        return AsyncLeaseObserver(self.connector, self.lease, self.client)

    async def prepare_handoff(self, job_id: str) -> dict:
        """Retain an exact offer before submission; acceptance must be observed separately."""
        async with self._lock:
            if self._close_command is not None or self._detached:
                raise RuntimeError("lease custody already released")
            if self._handoff_request is None:
                current = await self.client.lease_status(self.connector, self.lease)
                if current["revision"] != self.lease_revision or current["policy"] != "granted":
                    raise RuntimeError("caller lease custody is stale")
                self._handoff_request = HandoffOffer(
                    offer=uuid4(),
                    expected_lease_revision=self.lease_revision,
                    caller_binding=UUID(current["owner_binding"]),
                    job_id=job_id,
                    deadline_at=datetime.now(UTC) + timedelta(minutes=2),
                )
            if self._handoff_request.job_id != job_id:
                raise ValueError("retained handoff is bound to another job")
            return await self.client.offer_lease_handoff(
                self.connector, self.lease, self._handoff_request
            )

    async def handoff_status(self) -> dict:
        """Read durable custody; this never acknowledges an unobserved transfer."""
        if self._handoff_request is None:
            raise RuntimeError("lease handoff was not offered")
        return await self.client.lease_handoff(
            self.connector, self.lease, self._handoff_request.offer
        )

    async def _observe_handoff(self) -> bool:
        if self._handoff_request is None:
            return False
        receipt = await self.handoff_status()
        current = await self.client.lease_status(self.connector, self.lease)
        if _verify_handoff(self._handoff_request, receipt, current):
            self._detached = True
        return self._detached

    async def close(self) -> Optional[CatalogMutation]:
        """Commit job-only revocation; preserve identity on an uncertain reply."""
        async with self._lock:
            if self._detached or await self._observe_handoff():
                return None
            if self._close_receipt is not None:
                return self._close_receipt
            if self._close_command is None:
                revision = (await self.client.status(self.connector)).revision
                self._close_command = CatalogCommand(
                    command_id=uuid4(),
                    expected_revision=revision,
                    deadline_at=datetime.now(UTC) + timedelta(seconds=30),
                )
            self._close_receipt = await self.client.revoke_lease(
                self.connector,
                self.lease,
                command=self._close_command,
                expected_lease_revision=self.lease_revision,
            )
            return self._close_receipt

    async def close_status(self) -> CatalogMutation:
        """Read command outcome independently of local HTTP transport closure."""
        if self._close_command is None:
            raise ValueError("lease close has not been requested")
        return await self.client.command_status(self.connector, self._close_command.command_id)


@dataclass(slots=True)
class LeaseObserver:
    """A local view of externally owned job authority; close releases only this view.

    # See: backend/notes/specifications/tanha/current/systems/tunnels/v2_catalog.md
    """

    connector: UUID
    lease: UUID
    _client: Optional[TunnelsClient] = field(repr=False)

    def status(self) -> ConnectorStatus:
        """Read connector projection status; it does not prove the job's outcome."""
        if self._client is None:
            raise RuntimeError("lease observer is closed")
        return self._client.status(self.connector)

    def close(self) -> None:
        """Release this local reference without closing transport or revoking policy."""
        self._client = None


@dataclass(slots=True)
class AsyncLeaseObserver:
    """Async local observer with no connector, route or job cleanup ownership."""

    connector: UUID
    lease: UUID
    _client: Optional[AsyncTunnelsClient] = field(repr=False)

    async def status(self) -> ConnectorStatus:
        if self._client is None:
            raise RuntimeError("lease observer is closed")
        return await self._client.status(self.connector)

    async def close(self) -> None:
        self._client = None
