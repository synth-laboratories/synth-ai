"""Typed checkpoint models for Managed Research checkpointing, persistence, and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class CheckpointCadenceSource(StrEnum):
    MANUAL = "manual"
    PER_TASK = "per_task"
    PER_MILESTONE = "per_milestone"
    INTERVAL = "interval"
    PAUSE = "pause"
    RUNTIME_MESSAGE = "runtime_message"
    WORKFLOW_SIGNAL = "workflow_signal"


class CheckpointScope(StrEnum):
    FULL = "full"


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


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool when provided")
    return value


def _optional_mapping(payload: object, *, label: str) -> dict[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): value for key, value in payload.items()}


def _coerce_cadence_source(value: str | None) -> CheckpointCadenceSource | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    try:
        return CheckpointCadenceSource(normalized)
    except ValueError:
        return None


def _coerce_scope(value: str | None) -> CheckpointScope | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    try:
        return CheckpointScope(normalized)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    project_id: str
    captured_at: datetime
    cadence_source: CheckpointCadenceSource | None = None
    scope: CheckpointScope | None = None
    parent_checkpoint_id: str | None = None
    retained_until: datetime | None = None
    milestone_phase_id: str | None = None
    descriptor_digest: str | None = None
    size_bytes: int | None = None
    state: str = "unknown"
    artifact_id: str | None = None
    artifact_type: str | None = None
    uri: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime | None = None
    error_code: str | None = None
    recoverability: str | None = None
    recoverable: bool | None = None
    run_execution_blocking: bool | None = None
    checkpoint_restore_blocking: bool | None = None
    operator_action: str | None = None
    restorable: bool = False
    branchable: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> Checkpoint:
        mapping = _require_mapping(payload, label="checkpoint")
        captured_at = _optional_datetime(mapping, "captured_at") or _optional_datetime(
            mapping,
            "created_at",
        )
        if captured_at is None:
            raise ValueError("checkpoint.captured_at is required")
        checkpoint_id = _optional_text(mapping, "checkpoint_id") or _optional_text(
            mapping,
            "artifact_id",
        )
        if checkpoint_id is None:
            raise ValueError("checkpoint.checkpoint_id is required")
        return cls(
            checkpoint_id=checkpoint_id,
            run_id=_required_text(mapping, "run_id", label="checkpoint.run_id"),
            project_id=_required_text(
                mapping,
                "project_id",
                label="checkpoint.project_id",
            ),
            captured_at=captured_at,
            cadence_source=_coerce_cadence_source(_optional_text(mapping, "cadence_source")),
            scope=_coerce_scope(_optional_text(mapping, "scope")),
            parent_checkpoint_id=_optional_text(mapping, "parent_checkpoint_id"),
            retained_until=_optional_datetime(mapping, "retained_until"),
            milestone_phase_id=_optional_text(mapping, "milestone_phase_id"),
            descriptor_digest=_optional_text(mapping, "descriptor_digest"),
            size_bytes=_optional_int(mapping, "size_bytes"),
            state=_optional_text(mapping, "state") or "unknown",
            artifact_id=_optional_text(mapping, "artifact_id"),
            artifact_type=_optional_text(mapping, "artifact_type"),
            uri=_optional_text(mapping, "uri"),
            metadata=_optional_mapping(mapping.get("metadata"), label="checkpoint.metadata"),
            created_at=_optional_datetime(mapping, "created_at"),
            error_code=_optional_text(mapping, "error_code"),
            recoverability=_optional_text(mapping, "recoverability"),
            recoverable=_optional_bool(mapping, "recoverable"),
            run_execution_blocking=_optional_bool(mapping, "run_execution_blocking"),
            checkpoint_restore_blocking=_optional_bool(mapping, "checkpoint_restore_blocking"),
            operator_action=_optional_text(mapping, "operator_action"),
            restorable=bool(mapping.get("restorable")),
            branchable=bool(mapping.get("branchable")),
            raw=dict(mapping),
        )

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "captured_at": self.captured_at.isoformat(),
            "state": self.state,
            "metadata": dict(self.metadata),
            "restorable": self.restorable,
            "branchable": self.branchable,
        }
        if self.cadence_source is not None:
            payload["cadence_source"] = self.cadence_source.value
        if self.scope is not None:
            payload["scope"] = self.scope.value
        if self.parent_checkpoint_id is not None:
            payload["parent_checkpoint_id"] = self.parent_checkpoint_id
        if self.retained_until is not None:
            payload["retained_until"] = self.retained_until.isoformat()
        if self.milestone_phase_id is not None:
            payload["milestone_phase_id"] = self.milestone_phase_id
        if self.descriptor_digest is not None:
            payload["descriptor_digest"] = self.descriptor_digest
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        for key in (
            "artifact_id",
            "artifact_type",
            "uri",
            "error_code",
            "recoverability",
            "recoverable",
            "run_execution_blocking",
            "checkpoint_restore_blocking",
            "operator_action",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        return payload


__all__ = [
    "Checkpoint",
    "CheckpointCadenceSource",
    "CheckpointScope",
]
