"""Canonical tunnel v2 namespace with explicit connector and job lease authority."""

from .client import AsyncTunnelsClient, TunnelsClient
from .contracts import (
    Capability,
    CatalogCommand,
    CatalogMutation,
    ConnectorStatus,
    LeaseGrant,
    RouteRegistration,
)
from .handles import (
    AsyncBorrowedLeaseHandle,
    AsyncLeaseObserver,
    BorrowedLeaseHandle,
    LeaseObserver,
)
from .lease_custody import HandoffOffer
from .origin_operations import (
    CallerOperationOwner,
    CallerOriginLookup,
    CallerOriginManagement,
    CallerOriginRegistration,
    OriginManagementRevocation,
)
from .session_profile import SessionProfile, SessionProfileState

__all__ = [
    "TunnelsClient",
    "AsyncTunnelsClient",
    "CatalogCommand",
    "CatalogMutation",
    "ConnectorStatus",
    "Capability",
    "RouteRegistration",
    "LeaseGrant",
    "BorrowedLeaseHandle",
    "AsyncBorrowedLeaseHandle",
    "LeaseObserver",
    "AsyncLeaseObserver",
    "HandoffOffer",
    "SessionProfile",
    "SessionProfileState",
    "CallerOperationOwner",
    "CallerOriginLookup",
    "CallerOriginManagement",
    "CallerOriginRegistration",
    "OriginManagementRevocation",
]
