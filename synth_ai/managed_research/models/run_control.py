"""Typed public run-control acknowledgement mirrored from the backend contract.

Returned by pause / resume / stop to let callers correlate the control-plane
mutation with its durable runtime-intent message id and ack timestamp, so
replay/idempotence logic can target a specific intent instead of polling
run state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models.run_state import (
    ManagedResearchRun,
    _optional_string,
    _require_mapping,
    _require_string,
)


class ManagedResearchRunControlEnqueueStatus(StrEnum):
    ACCEPTED = "accepted"
    NOOP = "noop"
    TERMINAL_SYNC = "terminal_sync"


class ManagedResearchActorControlAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    INTERRUPT = "interrupt"


class ManagedResearchActorControlActorType(StrEnum):
    WORKER = "worker"
    ORCHESTRATOR = "orchestrator"


class RunLifecycleControlErrorCode(StrEnum):
    ALREADY_IN_STATE = "already_in_state"
    TERMINAL_RUN = "terminal_run"
    RUNTIME_NOT_LIVE = "runtime_not_live"
    RUN_NOT_FOUND = "run_not_found"
    PROJECT_ARCHIVED = "project_archived"
    LOCK_PRESSURE = "lock_pressure"


class ManagedResearchRunControlError(SmrApiError):
    """Raised when the backend rejects a pause/resume/stop with HTTP 409.

    The backend returns ``detail`` as a mapping with keys
    ``error_code``, ``message``, ``retryable``, ``current_state`` and
    ``run_id``. We surface each as a typed attribute so callers can
    discriminate auth vs. config vs. transient failure modes without
    string-sniffing.
    """

    def __init__(
        self,
        *,
        error_code: RunLifecycleControlErrorCode,
        message: str,
        retryable: bool,
        current_state: str | None,
        run_id: str | None,
        status_code: int | None = 409,
        response_text: str | None = None,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_text=response_text,
        )
        self.error_code = error_code
        self.retryable = retryable
        self.current_state = current_state
        self.run_id = run_id
        self.detail: dict[str, object] = dict(detail) if detail else {}

    @classmethod
    def from_response(
        cls,
        *,
        payload: object,
        status_code: int | None,
        response_text: str | None,
    ) -> ManagedResearchRunControlError:
        """Build the typed error from a 409 JSON body.

        Raises ``ValueError`` if the body does not match the documented
        contract. This is intentional: a 409 from these endpoints that
        lacks the expected structure is a contract drift, not a generic
        API failure, and collapsing it into a plain ``SmrApiError``
        would mask that.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("run control 409 body must be a JSON object with a 'detail' mapping")
        detail = payload.get("detail")
        if detail == RunLifecycleControlErrorCode.PROJECT_ARCHIVED.value:
            return cls(
                error_code=RunLifecycleControlErrorCode.PROJECT_ARCHIVED,
                message=(
                    "Project is archived; this backend does not allow the requested "
                    "run control through the project mutation guard."
                ),
                retryable=False,
                current_state=None,
                run_id=None,
                status_code=status_code,
                response_text=response_text,
                detail={"legacy_detail": detail},
            )
        if not isinstance(detail, Mapping):
            raise ValueError(
                "run control 409 body missing mapping 'detail' with error_code/message/retryable/current_state/run_id"
            )
        code_raw = detail.get("error_code")
        if not isinstance(code_raw, str) or not code_raw.strip():
            raise ValueError("run control 409 detail.error_code must be a non-empty string")
        try:
            error_code = RunLifecycleControlErrorCode(code_raw.strip())
        except ValueError as exc:
            raise ValueError(
                f"run control 409 detail.error_code {code_raw!r} is not a known RunLifecycleControlErrorCode"
            ) from exc
        message = detail.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("run control 409 detail.message must be a non-empty string")
        retryable = detail.get("retryable")
        if not isinstance(retryable, bool):
            raise ValueError("run control 409 detail.retryable must be a bool")
        current_state = detail.get("current_state")
        if not isinstance(current_state, str) or not current_state.strip():
            raise ValueError("run control 409 detail.current_state must be a non-empty string")
        run_id = detail.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run control 409 detail.run_id must be a non-empty string")
        return cls(
            error_code=error_code,
            message=message,
            retryable=retryable,
            current_state=current_state,
            run_id=run_id,
            status_code=status_code,
            response_text=response_text,
            detail=detail,
        )


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
        # fromisoformat accepts the `+00:00` Z-suffix-equivalent the backend emits
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    raise ValueError(f"{key} must be null, a datetime, or an ISO-8601 string")


def _require_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _actor_control_action(
    payload: Mapping[str, object], key: str
) -> ManagedResearchActorControlAction:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{key} must be a non-empty string")
    try:
        return ManagedResearchActorControlAction(value)
    except ValueError as exc:
        raise ValueError(
            f"{key} {value!r} is not a known ManagedResearchActorControlAction"
        ) from exc


def _actor_control_actor_type(
    payload: Mapping[str, object], key: str
) -> ManagedResearchActorControlActorType:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{key} must be a non-empty string")
    try:
        return ManagedResearchActorControlActorType(value)
    except ValueError as exc:
        raise ValueError(
            f"{key} {value!r} is not a known ManagedResearchActorControlActorType"
        ) from exc


@dataclass(frozen=True)
class ManagedResearchRunControlAck:
    """Result of a pause/resume/stop call.

    `control_intent_id` is the durable runtime-intent message id; replaying
    the same control with the same intent id is a no-op on the backend.
    `control_intent_ack_at` is when the backend enqueued the intent — not
    when the run actually transitioned.
    """

    run: ManagedResearchRun
    control_intent_id: str | None
    control_intent_ack_at: datetime | None
    enqueue_status: ManagedResearchRunControlEnqueueStatus | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunControlAck:
        mapping = _require_mapping(payload, label="run control ack")
        return cls(
            run=ManagedResearchRun.from_wire(mapping),
            control_intent_id=_optional_string(mapping, "control_intent_id"),
            control_intent_ack_at=_optional_datetime(mapping, "control_intent_ack_at"),
            enqueue_status=(
                ManagedResearchRunControlEnqueueStatus(value)
                if (value := _optional_string(mapping, "enqueue_status")) is not None
                else None
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class ManagedResearchActorControlAck:
    """Result of a project-run actor pause/resume control request."""

    accepted: bool
    actor_id: str
    actor_type: ManagedResearchActorControlActorType
    run_id: str
    requested_action: ManagedResearchActorControlAction
    previous_state: str
    target_state: str
    receipt_id: str | None = None
    requested_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchActorControlAck:
        mapping = _require_mapping(payload, label="actor control ack")
        return cls(
            accepted=_require_bool(mapping, "accepted"),
            actor_id=_require_string(mapping, "actor_id", label="actor_control.actor_id"),
            actor_type=_actor_control_actor_type(mapping, "actor_type"),
            run_id=_require_string(mapping, "run_id", label="actor_control.run_id"),
            requested_action=_actor_control_action(mapping, "requested_action"),
            previous_state=_require_string(
                mapping, "previous_state", label="actor_control.previous_state"
            ),
            target_state=_require_string(
                mapping, "target_state", label="actor_control.target_state"
            ),
            receipt_id=_optional_string(mapping, "receipt_id"),
            requested_at=_optional_datetime(mapping, "requested_at"),
            raw=dict(mapping),
        )


__all__ = [
    "ManagedResearchActorControlAction",
    "ManagedResearchActorControlActorType",
    "ManagedResearchActorControlAck",
    "ManagedResearchRunControlAck",
    "ManagedResearchRunControlEnqueueStatus",
    "ManagedResearchRunControlError",
    "RunLifecycleControlErrorCode",
]
