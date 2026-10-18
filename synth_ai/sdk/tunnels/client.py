"""Thin tunnel control facade; business authority remains in the backend.

# See: backend/notes/specifications/tanha/current/systems/tunnels/v2_catalog.md
"""

from __future__ import annotations

from uuid import UUID

from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport

from .contracts import (
    Capability,
    CatalogCommand,
    CatalogMutation,
    ConnectorStatus,
    LeaseGrant,
    RouteRegistration,
)
from .lease_custody import HandoffOffer
from .origin_operations import (
    CallerOriginLookup,
    CallerOriginManagement,
    CallerOriginRegistration,
    OriginManagementRevocation,
)
from .session_operations import SessionOperation
from .session_profile import SessionProfile, SessionProfileState
from .transport import AsyncTunnelTransport, TunnelTransport

PREFIX = "/api/v1/synthtunnel/v2/connectors"


def connector_path(connector: UUID) -> str:
    if connector.int == 0:
        raise ValueError("nil connector identity")
    return f"{PREFIX}/{connector}"


class TunnelsClient:
    def register_origin_operation(
        self, connector: UUID, lease: UUID, request: CallerOriginRegistration
    ) -> dict:
        return self._transport.request_json(
            "POST",
            connector_path(connector) + f"/leases/{lease}/operations",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.operation),
        )

    def origin_operation(
        self, connector: UUID, operation: UUID, request: CallerOriginLookup
    ) -> dict:
        return self._transport.request_json(
            "POST",
            connector_path(connector) + f"/operations/{operation}:lookup",
            json_body=request.model_dump(mode="json"),
            operation_id=str(operation),
        )

    def origin_management_capability(
        self, connector: UUID, operation: UUID, request: CallerOriginManagement
    ) -> Capability:
        return Capability.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/operations/{operation}/management-capability",
                json_body=request.model_dump(mode="json"),
                operation_id=str(operation),
            )
        )

    def revoke_origin_management(
        self, connector: UUID, operation: UUID, request: OriginManagementRevocation
    ) -> dict:
        return self._transport.request_json(
            "POST",
            connector_path(connector) + f"/operations/{operation}/management-revocation",
            json_body=request.model_dump(mode="json"),
            operation_id=str(operation),
        )

    """Explicit registration and job lease operations; closing this client only closes HTTP transport."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = TunnelTransport(transport)

    def close(self) -> None:
        self._transport.close()

    def status(self, connector: UUID) -> ConnectorStatus:
        return ConnectorStatus.model_validate(
            self._transport.request_json("GET", connector_path(connector))
        )

    def register_connector(self, connector: UUID, *, command: CatalogCommand) -> CatalogMutation:
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                PREFIX,
                json_body={**command.model_dump(mode="json"), "connector_id": str(connector)},
                operation_id=str(command.command_id),
            )
        )

    def session_profile(self, connector: UUID) -> SessionProfileState | None:
        """Read scoped enrollment data without changing actor or origin state."""
        value = self._transport.request_json("GET", connector_path(connector) + "/session-profile")
        return SessionProfileState.model_validate(value) if value is not None else None

    def configure_session_profile(
        self, connector: UUID, profile: SessionProfile, *, command: CatalogCommand
    ) -> CatalogMutation:
        """Enroll versioned checkpoint identity; cannot change after lifecycle enrollment."""
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + "/session-profile",
                json_body={
                    **command.model_dump(mode="json"),
                    "profile": profile.model_dump(mode="json"),
                },
                operation_id=str(command.command_id),
            )
        )

    def register_route(
        self, connector: UUID, registration: RouteRegistration, *, command: CatalogCommand
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + "/routes",
                json_body={
                    **command.model_dump(mode="json"),
                    **registration.model_dump(mode="json"),
                },
                operation_id=str(command.command_id),
            )
        )

    def grant_lease(
        self, connector: UUID, grant: LeaseGrant, *, command: CatalogCommand
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + "/leases",
                json_body={**command.model_dump(mode="json"), **grant.wire_payload()},
                operation_id=str(command.command_id),
            )
        )

    def revoke_lease(
        self, connector: UUID, lease: UUID, *, command: CatalogCommand, expected_lease_revision: int
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/revoke",
                json_body={
                    **command.model_dump(mode="json"),
                    "expected_lease_revision": expected_lease_revision,
                },
                operation_id=str(command.command_id),
            )
        )

    def lease_status(self, connector: UUID, lease: UUID) -> dict:
        """Read current scoped custody without returning bearer credentials."""
        return self._transport.request_json("GET", connector_path(connector) + f"/leases/{lease}")

    def offer_lease_handoff(self, connector: UUID, lease: UUID, request: HandoffOffer) -> dict:
        """Offer exact job ownership; an offer is not executor acceptance."""
        return self._transport.request_json(
            "POST",
            connector_path(connector) + f"/leases/{lease}/handoffs",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.offer),
        )

    def lease_handoff(self, connector: UUID, lease: UUID, offer: UUID) -> dict:
        """Reconcile retained custody by lookup without executing another handoff."""
        return self._transport.request_json(
            "GET", connector_path(connector) + f"/leases/{lease}/handoffs/{offer}"
        )

    def command_status(self, connector: UUID, command: UUID) -> CatalogMutation:
        return CatalogMutation.model_validate(
            self._transport.request_json("GET", connector_path(connector) + f"/commands/{command}")
        )

    def attachment_capability(
        self, connector: UUID, client_instance: UUID, *, ttl_seconds: int = 300
    ) -> Capability:
        return Capability.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + "/attach-capability",
                json_body={"client_instance_id": str(client_instance), "ttl_seconds": ttl_seconds},
            )
        )

    def forwarding_capability(
        self, connector: UUID, lease: UUID, *, ttl_seconds: int = 300
    ) -> Capability:
        return Capability.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/forward-capability",
                json_body={"ttl_seconds": ttl_seconds},
            )
        )

    def connector_capability(self, connector: UUID, *, ttl_seconds: int = 300) -> Capability:
        return Capability.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + "/capability",
                json_body={"ttl_seconds": ttl_seconds},
            )
        )

    def handover_route(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
        new_owner: UUID,
    ) -> CatalogMutation:
        """Transfer registration custody under the explicit owner fence."""
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/handover",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                    "new_owner": str(new_owner),
                },
                operation_id=str(command.command_id),
            )
        )

    def unregister_route(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
    ) -> CatalogMutation:
        """Withdraw a registration without allowing a stale owner to delete a replacement."""
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/unregister",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                },
                operation_id=str(command.command_id),
            )
        )

    def change_route_service(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
        service_identity: str,
    ) -> CatalogMutation:
        """Change service identity while historical job bindings remain denied."""
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/service",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                    "service_identity": service_identity,
                },
                operation_id=str(command.command_id),
            )
        )

    def rebind_lease_route(
        self,
        connector: UUID,
        lease: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        expected_lease_revision: int,
    ) -> CatalogMutation:
        """Explicitly accept the route's new service for this job only."""
        return CatalogMutation.model_validate(
            self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/routes/{route}/rebind",
                json_body={
                    **command.model_dump(mode="json"),
                    "expected_lease_revision": expected_lease_revision,
                },
                operation_id=str(command.command_id),
            )
        )

    def register_session_operation(self, connector: UUID, request: SessionOperation) -> dict:
        """Persist session intent before any actor effect; ambiguous registration is lookupable."""
        return self._transport.request_json(
            "POST",
            connector_path(connector) + "/session-operations",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.operation),
        )

    def session_operation(self, connector: UUID, operation: UUID) -> dict:
        return self._transport.request_json(
            "GET", connector_path(connector) + f"/session-operations/{operation}"
        )


class AsyncTunnelsClient:
    async def register_origin_operation(
        self, connector: UUID, lease: UUID, request: CallerOriginRegistration
    ) -> dict:
        return await self._transport.request_json(
            "POST",
            connector_path(connector) + f"/leases/{lease}/operations",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.operation),
        )

    async def origin_operation(
        self, connector: UUID, operation: UUID, request: CallerOriginLookup
    ) -> dict:
        return await self._transport.request_json(
            "POST",
            connector_path(connector) + f"/operations/{operation}:lookup",
            json_body=request.model_dump(mode="json"),
            operation_id=str(operation),
        )

    async def origin_management_capability(
        self, connector: UUID, operation: UUID, request: CallerOriginManagement
    ) -> Capability:
        return Capability.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/operations/{operation}/management-capability",
                json_body=request.model_dump(mode="json"),
                operation_id=str(operation),
            )
        )

    async def revoke_origin_management(
        self, connector: UUID, operation: UUID, request: OriginManagementRevocation
    ) -> dict:
        return await self._transport.request_json(
            "POST",
            connector_path(connector) + f"/operations/{operation}/management-revocation",
            json_body=request.model_dump(mode="json"),
            operation_id=str(operation),
        )

    """Async peer of the same explicit backend-owned commands and capability scopes."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = AsyncTunnelTransport(transport)

    async def close(self) -> None:
        await self._transport.close()

    async def status(self, connector: UUID) -> ConnectorStatus:
        return ConnectorStatus.model_validate(
            await self._transport.request_json("GET", connector_path(connector))
        )

    async def register_connector(
        self, connector: UUID, *, command: CatalogCommand
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                PREFIX,
                json_body={**command.model_dump(mode="json"), "connector_id": str(connector)},
                operation_id=str(command.command_id),
            )
        )

    async def session_profile(self, connector: UUID) -> SessionProfileState | None:
        """Read scoped enrollment data without changing actor or origin state."""
        value = await self._transport.request_json(
            "GET", connector_path(connector) + "/session-profile"
        )
        return SessionProfileState.model_validate(value) if value is not None else None

    async def configure_session_profile(
        self, connector: UUID, profile: SessionProfile, *, command: CatalogCommand
    ) -> CatalogMutation:
        """Enroll versioned checkpoint identity; cannot change after lifecycle enrollment."""
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + "/session-profile",
                json_body={
                    **command.model_dump(mode="json"),
                    "profile": profile.model_dump(mode="json"),
                },
                operation_id=str(command.command_id),
            )
        )

    async def register_route(
        self, connector: UUID, registration: RouteRegistration, *, command: CatalogCommand
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + "/routes",
                json_body={
                    **command.model_dump(mode="json"),
                    **registration.model_dump(mode="json"),
                },
                operation_id=str(command.command_id),
            )
        )

    async def grant_lease(
        self, connector: UUID, grant: LeaseGrant, *, command: CatalogCommand
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + "/leases",
                json_body={**command.model_dump(mode="json"), **grant.wire_payload()},
                operation_id=str(command.command_id),
            )
        )

    async def revoke_lease(
        self, connector: UUID, lease: UUID, *, command: CatalogCommand, expected_lease_revision: int
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/revoke",
                json_body={
                    **command.model_dump(mode="json"),
                    "expected_lease_revision": expected_lease_revision,
                },
                operation_id=str(command.command_id),
            )
        )

    async def lease_status(self, connector: UUID, lease: UUID) -> dict:
        """Read current scoped custody without returning bearer credentials."""
        return await self._transport.request_json(
            "GET", connector_path(connector) + f"/leases/{lease}"
        )

    async def offer_lease_handoff(
        self, connector: UUID, lease: UUID, request: HandoffOffer
    ) -> dict:
        """Offer exact job ownership; an offer is not executor acceptance."""
        return await self._transport.request_json(
            "POST",
            connector_path(connector) + f"/leases/{lease}/handoffs",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.offer),
        )

    async def lease_handoff(self, connector: UUID, lease: UUID, offer: UUID) -> dict:
        """Reconcile retained custody by lookup without executing another handoff."""
        return await self._transport.request_json(
            "GET", connector_path(connector) + f"/leases/{lease}/handoffs/{offer}"
        )

    async def command_status(self, connector: UUID, command: UUID) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "GET", connector_path(connector) + f"/commands/{command}"
            )
        )

    async def attachment_capability(
        self, connector: UUID, client_instance: UUID, *, ttl_seconds: int = 300
    ) -> Capability:
        return Capability.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + "/attach-capability",
                json_body={"client_instance_id": str(client_instance), "ttl_seconds": ttl_seconds},
            )
        )

    async def forwarding_capability(
        self, connector: UUID, lease: UUID, *, ttl_seconds: int = 300
    ) -> Capability:
        return Capability.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/forward-capability",
                json_body={"ttl_seconds": ttl_seconds},
            )
        )

    async def connector_capability(self, connector: UUID, *, ttl_seconds: int = 300) -> Capability:
        return Capability.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + "/capability",
                json_body={"ttl_seconds": ttl_seconds},
            )
        )

    async def handover_route(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
        new_owner: UUID,
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/handover",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                    "new_owner": str(new_owner),
                },
                operation_id=str(command.command_id),
            )
        )

    async def unregister_route(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/unregister",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                },
                operation_id=str(command.command_id),
            )
        )

    async def change_route_service(
        self,
        connector: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        registration_owner: UUID,
        expected_owner_generation: int,
        service_identity: str,
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/routes/{route}/service",
                json_body={
                    **command.model_dump(mode="json"),
                    "registration_owner": str(registration_owner),
                    "expected_owner_generation": expected_owner_generation,
                    "service_identity": service_identity,
                },
                operation_id=str(command.command_id),
            )
        )

    async def rebind_lease_route(
        self,
        connector: UUID,
        lease: UUID,
        route: UUID,
        *,
        command: CatalogCommand,
        expected_lease_revision: int,
    ) -> CatalogMutation:
        return CatalogMutation.model_validate(
            await self._transport.request_json(
                "POST",
                connector_path(connector) + f"/leases/{lease}/routes/{route}/rebind",
                json_body={
                    **command.model_dump(mode="json"),
                    "expected_lease_revision": expected_lease_revision,
                },
                operation_id=str(command.command_id),
            )
        )

    async def register_session_operation(self, connector: UUID, request: SessionOperation) -> dict:
        """Commit canonical session custody; never sends origin work."""
        return await self._transport.request_json(
            "POST",
            connector_path(connector) + "/session-operations",
            json_body=request.model_dump(mode="json"),
            operation_id=str(request.operation),
        )

    async def session_operation(self, connector: UUID, operation: UUID) -> dict:
        return await self._transport.request_json(
            "GET", connector_path(connector) + f"/session-operations/{operation}"
        )
