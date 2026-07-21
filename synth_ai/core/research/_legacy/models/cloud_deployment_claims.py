"""Typed models for CloudDeployment claims and fencing.

A claim is an advisory, TTL-bounded lease on a CloudDeployment: exactly one
holder at a time, renewed by explicit heartbeat, released idempotently. Every
acquire mints a monotonically increasing integer ``fencing_token``; mutating
deployment operations (deploy, retire) present it via the ``X-Fencing-Token``
header so stale holders are rejected instead of racing.

Wire contract (frozen with the backend claims lane):

- ``POST /smr/v1/deployments/{deployment_id}/claims`` with
  ``{holder, purpose, ttl_seconds}`` returns ``{claim_id, deployment_id,
  holder, purpose, fencing_token, acquired_at, expires_at}``.
- ``POST .../claims/{claim_id}/heartbeat`` returns the refreshed claim and
  repeats the deployment's slot, topology-source, and VM authority.
- ``POST .../claims/{claim_id}/release`` is idempotent and returns claim state.
- ``GET .../claims`` returns the active claim (or explicit none) plus the last
  fencing token issued.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from synth_ai.core.research._legacy.models.cloud_deployments import (
    CloudDeploymentTopologySource,
    cloud_deployment_topology_source_from_wire,
)


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _required_text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    raise ValueError(f"{key} must be null, a datetime, or an ISO-8601 string")


def _required_datetime(payload: Mapping[str, object], key: str, *, label: str) -> datetime:
    value = _optional_datetime(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer when provided")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer when provided") from exc


def _required_int(payload: Mapping[str, object], key: str, *, label: str) -> int:
    value = _optional_int(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_topology_source(
    payload: Mapping[str, object],
) -> CloudDeploymentTopologySource | None:
    value = payload.get("topology_source")
    if value is None:
        return None
    return cloud_deployment_topology_source_from_wire(value)


def _validate_topology_source_binding(
    topology_source: CloudDeploymentTopologySource | None,
    *,
    topology_id: str,
    topology_version: str,
    label: str,
) -> None:
    if topology_source is None:
        return
    if topology_source["topology_id"] != topology_id:
        raise ValueError(f"{label}.topology_source topology_id does not match authority")
    if topology_source["topology_version"] != topology_version:
        raise ValueError(f"{label}.topology_source topology_version does not match authority")


@dataclass(frozen=True, slots=True)
class ClaimAcquireRequest:
    """Request body for ``POST /smr/v1/deployments/{deployment_id}/claims``."""

    holder: str
    purpose: str
    ttl_seconds: int

    def __post_init__(self) -> None:
        if not str(self.holder or "").strip():
            raise ValueError("holder is required")
        if not str(self.purpose or "").strip():
            raise ValueError("purpose is required")
        if isinstance(self.ttl_seconds, bool) or int(self.ttl_seconds) <= 0:
            raise ValueError("ttl_seconds must be a positive integer")

    def to_wire(self) -> dict[str, object]:
        return {
            "holder": str(self.holder).strip(),
            "purpose": str(self.purpose).strip(),
            "ttl_seconds": int(self.ttl_seconds),
        }


@dataclass(frozen=True, slots=True)
class CloudDeploymentClaim:
    """A claim row on a CloudDeployment (acquire and release responses).

    ``fencing_token`` is a monotonically increasing integer minted per acquire;
    pass it to mutating deployment operations so a superseded holder is
    rejected with ``fencing_token_stale`` instead of racing the new holder.
    ``state`` and ``released_at`` are populated on release responses when the
    backend reports them.
    """

    claim_id: str
    deployment_id: str
    holder: str
    purpose: str
    fencing_token: int
    acquired_at: datetime
    expires_at: datetime
    topology_id: str
    topology_version: str
    topology_source: CloudDeploymentTopologySource | None
    vm_name: str | None
    state: str | None = None
    released_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: Mapping[str, object] | object) -> CloudDeploymentClaim:
        mapping = _require_mapping(payload, label="cloud deployment claim")
        topology_id = _required_text(mapping, "topology_id", label="topology_id")
        topology_version = _required_text(
            mapping,
            "topology_version",
            label="topology_version",
        )
        topology_source = _optional_topology_source(mapping)
        _validate_topology_source_binding(
            topology_source,
            topology_id=topology_id,
            topology_version=topology_version,
            label="cloud deployment claim",
        )
        return cls(
            claim_id=_required_text(mapping, "claim_id", label="claim_id"),
            deployment_id=_required_text(mapping, "deployment_id", label="deployment_id"),
            holder=_required_text(mapping, "holder", label="holder"),
            purpose=_required_text(mapping, "purpose", label="purpose"),
            fencing_token=_required_int(mapping, "fencing_token", label="fencing_token"),
            acquired_at=_required_datetime(mapping, "acquired_at", label="acquired_at"),
            expires_at=_required_datetime(mapping, "expires_at", label="expires_at"),
            topology_id=topology_id,
            topology_version=topology_version,
            topology_source=topology_source,
            vm_name=_optional_text(mapping, "vm_name"),
            state=_optional_text(mapping, "state"),
            released_at=_optional_datetime(mapping, "released_at"),
        )


@dataclass(frozen=True, slots=True)
class ClaimHeartbeat:
    """Legacy expiry-only adapter; heartbeat APIs now return ``CloudDeploymentClaim``."""

    expires_at: datetime

    @classmethod
    def from_wire(cls, payload: Mapping[str, object] | object) -> ClaimHeartbeat:
        mapping = _require_mapping(payload, label="claim heartbeat")
        return cls(expires_at=_required_datetime(mapping, "expires_at", label="expires_at"))


@dataclass(frozen=True, slots=True)
class ClaimProjection:
    """Response of ``GET /smr/v1/deployments/{deployment_id}/claims``.

    ``active_claim`` is the currently held claim or ``None`` (explicit none —
    the deployment is claimable). ``last_fencing_token`` is the highest fencing
    token the backend has issued for this deployment, or ``None`` when no claim
    was ever acquired.
    """

    deployment_id: str
    topology_id: str
    topology_version: str
    topology_source: CloudDeploymentTopologySource | None
    vm_name: str | None
    active_claim: CloudDeploymentClaim | None
    last_fencing_token: int | None

    @classmethod
    def from_wire(cls, payload: Mapping[str, object] | object) -> ClaimProjection:
        mapping = _require_mapping(payload, label="claim projection")
        deployment_id = _required_text(mapping, "deployment_id", label="deployment_id")
        topology_id = _required_text(mapping, "topology_id", label="topology_id")
        topology_version = _required_text(
            mapping,
            "topology_version",
            label="topology_version",
        )
        topology_source = _optional_topology_source(mapping)
        vm_name = _optional_text(mapping, "vm_name")
        _validate_topology_source_binding(
            topology_source,
            topology_id=topology_id,
            topology_version=topology_version,
            label="claim projection",
        )
        raw_active = mapping.get("active_claim")
        active_claim = None if raw_active is None else CloudDeploymentClaim.from_wire(raw_active)
        last_fencing_token = _optional_int(mapping, "last_fencing_token")
        if active_claim is not None:
            projection_authority = {
                "deployment_id": deployment_id,
                "topology_id": topology_id,
                "topology_version": topology_version,
                "topology_source": topology_source,
                "vm_name": vm_name,
            }
            for field_name, expected_value in projection_authority.items():
                if getattr(active_claim, field_name) != expected_value:
                    raise ValueError(
                        f"active_claim authority does not match claim projection for {field_name}"
                    )
            if active_claim.fencing_token != last_fencing_token:
                raise ValueError(
                    "active_claim fencing_token does not match claim projection last_fencing_token"
                )
        return cls(
            deployment_id=deployment_id,
            topology_id=topology_id,
            topology_version=topology_version,
            topology_source=topology_source,
            vm_name=vm_name,
            active_claim=active_claim,
            last_fencing_token=last_fencing_token,
        )


__all__ = [
    "ClaimAcquireRequest",
    "ClaimHeartbeat",
    "ClaimProjection",
    "CloudDeploymentClaim",
]
