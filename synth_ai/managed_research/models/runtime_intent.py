"""Typed public runtime-intent models for operator steering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from synth_ai.managed_research.models.run_state import _optional_string, _require_mapping


class RuntimeIntentKind(StrEnum):
    SET_TASK_STATE = "set_task_state"
    SET_RUN_STATE = "set_run_state"
    CREATE_APPROVAL = "create_approval"
    CREATE_QUESTION = "create_question"
    RESOLVE_APPROVAL = "resolve_approval"
    ANSWER_QUESTION = "answer_question"
    RECORD_SPEND = "record_spend"
    PLAN_TASKS = "plan_tasks"
    WRITE_PROJECT_MILESTONES = "write_project_milestones"


class RuntimeIntentStatus(StrEnum):
    QUEUED = "queued"
    APPLIED = "applied"
    REJECTED = "rejected"
    IGNORED = "ignored"


class RuntimeMessageMode(StrEnum):
    QUEUE = "queue"
    STEER = "steer"
    CONTROL = "control"


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


def _normalized_payload(payload: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("runtime intent payload must be a mapping")
    return dict(payload)


def _mapping_list(values: Iterable[Mapping[str, Any] | dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("runtime intent payload entries must be mappings")
        out.append(dict(value))
    return out


@dataclass(frozen=True)
class RuntimeIntent:
    kind: RuntimeIntentKind
    payload: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "payload": dict(self.payload)}

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeIntent:
        mapping = _require_mapping(payload, label="runtime intent")
        kind = RuntimeIntentKind(str(mapping.get("kind") or "").strip())
        body = mapping.get("payload")
        return cls(
            kind=kind, payload=_normalized_payload(body if isinstance(body, Mapping) else None)
        )

    @classmethod
    def set_task_state(
        cls,
        *,
        task_id: str,
        state: str,
        reason: str | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {"task_id": task_id, "state": state, "requested_by_role": requested_by_role}
        if reason:
            payload["reason"] = reason
        return cls(RuntimeIntentKind.SET_TASK_STATE, payload)

    @classmethod
    def set_run_state(
        cls,
        *,
        state: str,
        stop_reason: str | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {"state": state, "requested_by_role": requested_by_role}
        if stop_reason:
            payload["stop_reason"] = stop_reason
        return cls(RuntimeIntentKind.SET_RUN_STATE, payload)

    @classmethod
    def create_approval(
        cls,
        *,
        kind_requested: str,
        approval_id: str | None = None,
        title: str | None = None,
        body: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {
            "kind_requested": kind_requested,
            "requested_by_role": requested_by_role,
        }
        for key, value in (
            ("approval_id", approval_id),
            ("title", title),
            ("body", body),
        ):
            if value:
                payload[key] = value
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return cls(RuntimeIntentKind.CREATE_APPROVAL, payload)

    @classmethod
    def create_question(
        cls,
        *,
        prompt: str,
        question_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {"prompt": prompt, "requested_by_role": requested_by_role}
        if question_id:
            payload["question_id"] = question_id
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return cls(RuntimeIntentKind.CREATE_QUESTION, payload)

    @classmethod
    def resolve_approval(
        cls,
        *,
        approval_id: str,
        user_id: str,
        decision: str,
        comment: str | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {
            "approval_id": approval_id,
            "user_id": user_id,
            "decision": decision,
            "requested_by_role": requested_by_role,
        }
        if comment is not None:
            payload["comment"] = comment
        return cls(RuntimeIntentKind.RESOLVE_APPROVAL, payload)

    @classmethod
    def answer_question(
        cls,
        *,
        question_id: str,
        user_id: str,
        response_text: str,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        return cls(
            RuntimeIntentKind.ANSWER_QUESTION,
            {
                "question_id": question_id,
                "user_id": user_id,
                "response_text": response_text,
                "requested_by_role": requested_by_role,
            },
        )

    @classmethod
    def record_spend(
        cls,
        *,
        provider: str,
        meter_kind: str,
        quantity: float,
        funding_source: str,
        model: str | None = None,
        cost_cents: int | None = None,
        episode_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        worker_id: str | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload: dict[str, Any] = {
            "provider": provider,
            "meter_kind": meter_kind,
            "quantity": quantity,
            "funding_source": funding_source,
            "requested_by_role": requested_by_role,
        }
        for key, value in (
            ("model", model),
            ("cost_cents", cost_cents),
            ("episode_id", episode_id),
            ("worker_id", worker_id),
        ):
            if value is not None:
                payload[key] = value
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return cls(RuntimeIntentKind.RECORD_SPEND, payload)

    @classmethod
    def plan_tasks(
        cls,
        *,
        tasks: Iterable[Mapping[str, Any] | dict[str, Any]],
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        return cls(
            RuntimeIntentKind.PLAN_TASKS,
            {"tasks": _mapping_list(tasks), "requested_by_role": requested_by_role},
        )

    @classmethod
    def write_project_milestones(
        cls,
        *,
        milestones: Iterable[Mapping[str, Any] | dict[str, Any]],
        mode: str,
        rationale: str | None = None,
        requested_by_role: str = "human",
    ) -> RuntimeIntent:
        payload = {
            "milestones": _mapping_list(milestones),
            "mode": mode,
            "requested_by_role": requested_by_role,
        }
        if rationale:
            payload["rationale"] = rationale
        return cls(RuntimeIntentKind.WRITE_PROJECT_MILESTONES, payload)


@dataclass(frozen=True)
class RuntimeIntentReceipt:
    runtime_intent_id: str
    runtime_intent_status: RuntimeIntentStatus
    runtime_intent_ack_at: datetime | None
    run_id: str
    intent_kind: RuntimeIntentKind
    mode: RuntimeMessageMode
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeIntentReceipt:
        mapping = _require_mapping(payload, label="runtime intent receipt")
        return cls(
            runtime_intent_id=str(mapping.get("runtime_intent_id") or ""),
            runtime_intent_status=RuntimeIntentStatus(
                str(mapping.get("runtime_intent_status") or "")
            ),
            runtime_intent_ack_at=_optional_datetime(mapping, "runtime_intent_ack_at"),
            run_id=str(mapping.get("run_id") or ""),
            intent_kind=RuntimeIntentKind(str(mapping.get("intent_kind") or "")),
            mode=RuntimeMessageMode(str(mapping.get("mode") or "")),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class RuntimeIntentView(RuntimeIntentReceipt):
    message_id: str = ""
    seq: int = 0
    action: str = ""
    topic: str | None = None
    causation_id: str | None = None
    sender: str | None = None
    target: str | None = None
    body: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    requested_by: str | None = None
    requested_by_role: str | None = None
    resolved_at: datetime | None = None
    error_code: str | None = None
    error_detail: str | None = None
    retryable: bool = False
    applied_mode: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeIntentView:
        mapping = _require_mapping(payload, label="runtime intent view")
        receipt = RuntimeIntentReceipt.from_wire(mapping)
        raw_payload = mapping.get("payload")
        return cls(
            runtime_intent_id=receipt.runtime_intent_id,
            runtime_intent_status=receipt.runtime_intent_status,
            runtime_intent_ack_at=receipt.runtime_intent_ack_at,
            run_id=receipt.run_id,
            intent_kind=receipt.intent_kind,
            mode=receipt.mode,
            raw=dict(mapping),
            message_id=str(mapping.get("message_id") or receipt.runtime_intent_id),
            seq=int(mapping.get("seq") or 0),
            action=str(mapping.get("action") or ""),
            topic=_optional_string(mapping, "topic"),
            causation_id=_optional_string(mapping, "causation_id"),
            sender=_optional_string(mapping, "sender"),
            target=_optional_string(mapping, "target"),
            body=_optional_string(mapping, "body"),
            payload=_normalized_payload(raw_payload if isinstance(raw_payload, Mapping) else None),
            requested_by=_optional_string(mapping, "requested_by"),
            requested_by_role=_optional_string(mapping, "requested_by_role"),
            resolved_at=_optional_datetime(mapping, "resolved_at"),
            error_code=_optional_string(mapping, "error_code"),
            error_detail=_optional_string(mapping, "error_detail"),
            retryable=bool(mapping.get("retryable", False)),
            applied_mode=_optional_string(mapping, "applied_mode"),
        )


__all__ = [
    "RuntimeIntent",
    "RuntimeIntentKind",
    "RuntimeIntentReceipt",
    "RuntimeIntentStatus",
    "RuntimeIntentView",
    "RuntimeMessageMode",
]
