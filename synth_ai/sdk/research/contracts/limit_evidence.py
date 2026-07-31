"""Typed public projection of durable run-limit evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from synth_ai.sdk.research.contracts._wire import (
    optional_datetime,
    optional_text,
    required_datetime,
    required_text,
)


def _mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): value for key, value in payload.items()}


def _optional_float(payload: Mapping[str, object], name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    return float(value)


def _optional_int(payload: Mapping[str, object], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return value


def _optional_bool(payload: Mapping[str, object], name: str) -> bool | None:
    value = payload.get(name)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean or null")
    return value


def _required_bool(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _string_list(payload: Mapping[str, object], name: str) -> list[str]:
    value = payload.get(name, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings")
        result.append(item)
    return result


def _object(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class SmrRunLimitLaunchCapEvidence:
    dimension: str
    cap_amount: float | None
    cap_revision: int | None
    cap_source: str | None
    policy: dict[str, object]

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitLaunchCapEvidence:
        value = _mapping(payload, label="run limit launch cap evidence")
        return cls(
            dimension=required_text(value, "dimension"),
            cap_amount=_optional_float(value, "cap_amount"),
            cap_revision=_optional_int(value, "cap_revision"),
            cap_source=optional_text(value, "cap_source"),
            policy=_object(value, "policy"),
        )


@dataclass(frozen=True)
class SmrRunLimitLaunchEvidence:
    schema_version: str | None
    status: str | None
    rollout_mode: str | None
    enforcement_enabled: bool | None
    refusal_disposition: str | None
    funding_lane: str | None
    owner_slot_id: str | None
    backend_url: str | None
    enforcement_granularity: dict[str, str]
    strict_inference_admission: dict[str, object]
    limits: list[SmrRunLimitLaunchCapEvidence]

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitLaunchEvidence:
        value = _mapping(payload, label="run limit launch evidence")
        raw_limits = value.get("limits", [])
        if not isinstance(raw_limits, list):
            raise ValueError("limits must be an array")
        granularity = value.get("enforcement_granularity") or {}
        if not isinstance(granularity, Mapping):
            raise ValueError("enforcement_granularity must be an object")
        strict_admission = value.get("strict_inference_admission") or {}
        if not isinstance(strict_admission, Mapping):
            raise ValueError("strict_inference_admission must be an object")
        return cls(
            schema_version=optional_text(value, "schema_version"),
            status=optional_text(value, "status"),
            rollout_mode=optional_text(value, "rollout_mode"),
            enforcement_enabled=_optional_bool(value, "enforcement_enabled"),
            refusal_disposition=optional_text(value, "refusal_disposition"),
            funding_lane=optional_text(value, "funding_lane"),
            owner_slot_id=optional_text(value, "owner_slot_id"),
            backend_url=optional_text(value, "backend_url"),
            enforcement_granularity={str(key): str(item) for key, item in granularity.items()},
            strict_inference_admission={str(key): item for key, item in strict_admission.items()},
            limits=[SmrRunLimitLaunchCapEvidence.from_wire(item) for item in raw_limits],
        )


@dataclass(frozen=True)
class SmrRunLimitDecisionEvidence:
    decision_id: str
    dimension: str
    action: str
    binding: bool
    reason: str
    cap_revision: int | None
    cap_amount: float | None
    used_amount: float | None
    threshold: float | None
    unit: str | None
    remaining_amount: float | None
    accounting_completeness: str
    usage_sources: list[str]
    missing_accounting_sources: list[str]
    trigger_source: str
    enforcement_granularity: str | None
    possible_overshoot: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitDecisionEvidence:
        value = _mapping(payload, label="run limit decision evidence")
        accounting = required_text(value, "accounting_completeness")
        if accounting not in {"complete", "incomplete", "unknown"}:
            raise ValueError("accounting_completeness is not recognized")
        return cls(
            decision_id=required_text(value, "decision_id"),
            dimension=required_text(value, "dimension"),
            action=required_text(value, "action"),
            binding=_required_bool(value, "binding"),
            reason=required_text(value, "reason"),
            cap_revision=_optional_int(value, "cap_revision"),
            cap_amount=_optional_float(value, "cap_amount"),
            used_amount=_optional_float(value, "used_amount"),
            threshold=_optional_float(value, "threshold"),
            unit=optional_text(value, "unit"),
            remaining_amount=_optional_float(value, "remaining_amount"),
            accounting_completeness=accounting,
            usage_sources=_string_list(value, "usage_sources"),
            missing_accounting_sources=_string_list(value, "missing_accounting_sources"),
            trigger_source=required_text(value, "trigger_source"),
            enforcement_granularity=optional_text(value, "enforcement_granularity"),
            possible_overshoot=optional_text(value, "possible_overshoot"),
        )


@dataclass(frozen=True)
class SmrRunLimitWarningDelivery:
    delivery_job_id: str
    status: str
    attempt_count: int
    max_attempts: int
    deliver_after: datetime
    last_error_code: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitWarningDelivery:
        value = _mapping(payload, label="run limit warning delivery")
        return cls(
            delivery_job_id=required_text(value, "delivery_job_id"),
            status=required_text(value, "status"),
            attempt_count=_required_int(value, "attempt_count"),
            max_attempts=_required_int(value, "max_attempts"),
            deliver_after=required_datetime(value, "deliver_after"),
            last_error_code=optional_text(value, "last_error_code"),
        )


@dataclass(frozen=True)
class SmrRunLimitWarningEvidence:
    message_id: str
    decision_id: str | None
    dimension: str | None
    cap_revision: int | None
    threshold: float | None
    cap_amount: float | None
    used_amount: float | None
    remaining_amount: float | None
    unit: str | None
    exhaustion_action: str | None
    audience: str | None
    status: str
    deliveries: list[SmrRunLimitWarningDelivery]

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitWarningEvidence:
        value = _mapping(payload, label="run limit warning evidence")
        deliveries = value.get("deliveries", [])
        if not isinstance(deliveries, list):
            raise ValueError("deliveries must be an array")
        return cls(
            message_id=required_text(value, "message_id"),
            decision_id=optional_text(value, "decision_id"),
            dimension=optional_text(value, "dimension"),
            cap_revision=_optional_int(value, "cap_revision"),
            threshold=_optional_float(value, "threshold"),
            cap_amount=_optional_float(value, "cap_amount"),
            used_amount=_optional_float(value, "used_amount"),
            remaining_amount=_optional_float(value, "remaining_amount"),
            unit=optional_text(value, "unit"),
            exhaustion_action=optional_text(value, "exhaustion_action"),
            audience=optional_text(value, "audience"),
            status=required_text(value, "status"),
            deliveries=[SmrRunLimitWarningDelivery.from_wire(item) for item in deliveries],
        )


@dataclass(frozen=True)
class SmrRunLimitInterruptEvidence:
    operation_id: str
    control_id: str | None
    control_seq: int | None
    signal: str | None
    issued_by: str | None
    status: str
    attempt_count: int
    max_attempts: int
    external_ref: str | None
    failure_code: str | None
    next_retry_at: datetime | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitInterruptEvidence:
        value = _mapping(payload, label="run limit interrupt evidence")
        return cls(
            operation_id=required_text(value, "operation_id"),
            control_id=optional_text(value, "control_id"),
            control_seq=_optional_int(value, "control_seq"),
            signal=optional_text(value, "signal"),
            issued_by=optional_text(value, "issued_by"),
            status=required_text(value, "status"),
            attempt_count=_required_int(value, "attempt_count"),
            max_attempts=_required_int(value, "max_attempts"),
            external_ref=optional_text(value, "external_ref"),
            failure_code=optional_text(value, "failure_code"),
            next_retry_at=optional_datetime(value, "next_retry_at"),
        )


@dataclass(frozen=True)
class SmrRunLimitExtensionEvidence:
    receipt_id: str
    extension_id: str | None
    dimension: str | None
    previous_cap_amount: float | None
    new_cap_amount: float | None
    previous_revision: int | None
    new_revision: int | None
    requested_by: str | None
    reason: str | None
    safe_to_release: bool | None
    resolved_blocker_ids: list[str]
    released_pause_gate_ids: list[str]
    resumed: bool | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitExtensionEvidence:
        value = _mapping(payload, label="run limit extension evidence")
        return cls(
            receipt_id=required_text(value, "receipt_id"),
            extension_id=optional_text(value, "extension_id"),
            dimension=optional_text(value, "dimension"),
            previous_cap_amount=_optional_float(value, "previous_cap_amount"),
            new_cap_amount=_optional_float(value, "new_cap_amount"),
            previous_revision=_optional_int(value, "previous_revision"),
            new_revision=_optional_int(value, "new_revision"),
            requested_by=optional_text(value, "requested_by"),
            reason=optional_text(value, "reason"),
            safe_to_release=_optional_bool(value, "safe_to_release"),
            resolved_blocker_ids=_string_list(value, "resolved_blocker_ids"),
            released_pause_gate_ids=_string_list(value, "released_pause_gate_ids"),
            resumed=_optional_bool(value, "resumed"),
        )


@dataclass(frozen=True)
class SmrRunLimitEvidenceItem:
    evidence_id: str
    kind: str
    occurred_at: datetime
    decision: SmrRunLimitDecisionEvidence | None
    warning: SmrRunLimitWarningEvidence | None
    interrupt: SmrRunLimitInterruptEvidence | None
    extension: SmrRunLimitExtensionEvidence | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitEvidenceItem:
        value = _mapping(payload, label="run limit evidence item")
        kind = required_text(value, "kind")
        if kind not in {"decision", "warning", "interrupt", "extension"}:
            raise ValueError(f"unsupported run limit evidence kind: {kind}")
        decoders = {
            "decision": SmrRunLimitDecisionEvidence,
            "warning": SmrRunLimitWarningEvidence,
            "interrupt": SmrRunLimitInterruptEvidence,
            "extension": SmrRunLimitExtensionEvidence,
        }
        decoded: dict[str, object | None] = {}
        for name, decoder in decoders.items():
            nested = value.get(name)
            decoded[name] = decoder.from_wire(nested) if nested is not None else None
        if decoded[kind] is None:
            raise ValueError(f"{kind} evidence payload is required")
        return cls(
            evidence_id=required_text(value, "evidence_id"),
            kind=kind,
            occurred_at=required_datetime(value, "occurred_at"),
            decision=decoded["decision"],  # type: ignore[arg-type]
            warning=decoded["warning"],  # type: ignore[arg-type]
            interrupt=decoded["interrupt"],  # type: ignore[arg-type]
            extension=decoded["extension"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SmrRunLimitEvidencePage:
    schema_version: str
    org_id: str
    project_id: str
    run_id: str
    launch: SmrRunLimitLaunchEvidence | None
    items: list[SmrRunLimitEvidenceItem]
    next_cursor: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunLimitEvidencePage:
        value = _mapping(payload, label="run limit evidence page")
        schema_version = required_text(value, "schema_version")
        if schema_version != "smr.run_limit_evidence.v1":
            raise ValueError(f"unsupported run limit evidence schema: {schema_version}")
        items = value.get("items", [])
        if not isinstance(items, list):
            raise ValueError("items must be an array")
        launch = value.get("launch")
        return cls(
            schema_version=schema_version,
            org_id=required_text(value, "org_id"),
            project_id=required_text(value, "project_id"),
            run_id=required_text(value, "run_id"),
            launch=(SmrRunLimitLaunchEvidence.from_wire(launch) if launch is not None else None),
            items=[SmrRunLimitEvidenceItem.from_wire(item) for item in items],
            next_cursor=optional_text(value, "next_cursor"),
        )


__all__ = [
    "SmrRunLimitDecisionEvidence",
    "SmrRunLimitEvidenceItem",
    "SmrRunLimitEvidencePage",
    "SmrRunLimitExtensionEvidence",
    "SmrRunLimitInterruptEvidence",
    "SmrRunLimitLaunchCapEvidence",
    "SmrRunLimitLaunchEvidence",
    "SmrRunLimitWarningDelivery",
    "SmrRunLimitWarningEvidence",
]
