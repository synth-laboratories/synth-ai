"""Canonical versioned control routes; see backend tunnels/v2_catalog.md.

These routes retain catalog UUID recovery, separately from the negotiated
Index/Swarms customer protocols. They cannot claim those release manifests.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue, RootModel

from synth_ai.errors import UnsupportedOperationError
from synth_ai.options import ClientOptions
from synth_ai.types.tunnels.contracts import Capability, CatalogMutation, ConnectorStatus
from synth_ai.types.tunnels.session_profile import SessionProfileState

from .operations import Operation


class ControlRecord(RootModel[dict[str, JsonValue]]):
    model_config = ConfigDict(allow_inf_nan=False, hide_input_in_errors=True)


class OptionalProfile(RootModel[SessionProfileState | None]):
    pass


ROOT = "/synthtunnel/v2/connectors"
CONNECTOR = ROOT + "/{connector}"
# Names and paths mirror the existing backend router, not inferred endpoints.
_SPECS = (
    ("connector.register", "POST", ROOT, True, CatalogMutation),
    ("connector.status", "GET", CONNECTOR, False, ConnectorStatus),
    ("profile.get", "GET", CONNECTOR + "/session-profile", False, OptionalProfile),
    ("profile.configure", "POST", CONNECTOR + "/session-profile", True, CatalogMutation),
    ("route.register", "POST", CONNECTOR + "/routes", True, CatalogMutation),
    ("lease.grant", "POST", CONNECTOR + "/leases", True, CatalogMutation),
    ("lease.revoke", "POST", CONNECTOR + "/leases/{lease}/revoke", True, CatalogMutation),
    ("lease.status", "GET", CONNECTOR + "/leases/{lease}", False, ControlRecord),
    ("handoff.offer", "POST", CONNECTOR + "/leases/{lease}/handoffs", True, ControlRecord),
    ("handoff.get", "GET", CONNECTOR + "/leases/{lease}/handoffs/{offer}", False, ControlRecord),
    ("command.get", "GET", CONNECTOR + "/commands/{command}", False, CatalogMutation),
    ("capability.attach", "POST", CONNECTOR + "/attach-capability", True, Capability),
    (
        "capability.forward",
        "POST",
        CONNECTOR + "/leases/{lease}/forward-capability",
        True,
        Capability,
    ),
    ("capability.connector", "POST", CONNECTOR + "/capability", True, Capability),
    ("route.handover", "POST", CONNECTOR + "/routes/{route}/handover", True, CatalogMutation),
    ("route.unregister", "POST", CONNECTOR + "/routes/{route}/unregister", True, CatalogMutation),
    ("route.service", "POST", CONNECTOR + "/routes/{route}/service", True, CatalogMutation),
    (
        "lease.rebind",
        "POST",
        CONNECTOR + "/leases/{lease}/routes/{route}/rebind",
        True,
        CatalogMutation,
    ),
    ("session.register", "POST", CONNECTOR + "/session-operations", True, ControlRecord),
    ("session.get", "GET", CONNECTOR + "/session-operations/{operation}", False, ControlRecord),
    ("origin.register", "POST", CONNECTOR + "/leases/{lease}/operations", True, ControlRecord),
    ("origin.get", "POST", CONNECTOR + "/operations/{operation}:lookup", False, ControlRecord),
    (
        "origin.capability",
        "POST",
        CONNECTOR + "/operations/{operation}/management-capability",
        True,
        Capability,
    ),
    (
        "origin.revoke",
        "POST",
        CONNECTOR + "/operations/{operation}/management-revocation",
        True,
        ControlRecord,
    ),
)
CONTROL_OPERATIONS = tuple(
    Operation("tunnels." + name, method, path, True, mutation, "tunnels.v2")
    for name, method, path, mutation, _ in _SPECS
)
RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    operation.id: spec[4] for operation, spec in zip(CONTROL_OPERATIONS, _SPECS, strict=True)
}


def require_control_operation(operation: Operation, options: ClientOptions) -> None:
    if operation not in CONTROL_OPERATIONS:
        raise UnsupportedOperationError("Unknown Tunnel v2 control operation")
    if options.release_manifest_sha256 is not None or options.candidate_manifest_sha256 is not None:
        raise UnsupportedOperationError(
            "Tunnel catalog v2 does not negotiate Index/Swarms release manifests; "
            "use a separate unpinned control client",
            code="tunnel_release_not_qualified",
        )
