"""Typed contracts for the champion-free Factory Result authority.

A Factory produces Results. Some Factories also *optimize* something; those
declare an evaluation lens and get a derived best-so-far. Nothing here is
required to operate a Factory — a monitoring, reporting, or maintenance Factory
uses none of it, and that is the ordinary case rather than a degraded one.

The two shapes worth reading closely:

``FactoryBestResult``
    Total. ``outcome`` always says which of the three answers this is, and
    ``result_id`` is populated only for ``BEST``. A caller cannot dereference a
    winner that does not exist.

``FactoryResultEvaluation``
    Carries ``attempt_key``. Retries are idempotent under the same key; a
    correction uses a new one, so the record of what was believed when survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import (
    array_value,
    object_value,
    optional_datetime,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.sdk.research.contracts.common import FactoryId, require_text


class FactoryLensDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class FactoryLensMissingPolicy(StrEnum):
    """How a lens treats a Result with no usable score."""

    #: Unscored Results are not ranked at all.
    INELIGIBLE = "ineligible"
    #: Unscored Results rank, always behind every scored one.
    WORST = "worst"


class FactoryLensTieBreak(StrEnum):
    EARLIEST_RESULT = "earliest_result"
    LATEST_RESULT = "latest_result"
    LOWEST_RESULT_ID = "lowest_result_id"


class FactoryResultKind(StrEnum):
    REPORT = "report"
    CODE_CHANGE = "code_change"
    DATASET = "dataset"
    MODEL = "model"
    PROMPT = "prompt"
    POLICY = "policy"
    ARTIFACT = "artifact"


class FactoryEvaluationStatus(StrEnum):
    PENDING = "pending"
    #: Completed with a usable score. There is no pass threshold under a lens;
    #: the score and the ordering are the whole verdict.
    EVALUATED = "evaluated"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FactoryBestOutcome(StrEnum):
    BEST = "best"
    #: No live Result matches the lens eligibility. "The lens does not apply
    #: here" — distinct from the Factory having produced nothing.
    NO_ELIGIBLE_RESULTS = "no_eligible_results"
    #: Eligible Results exist but none carries a completed evaluation.
    NO_SCORED_RESULTS = "no_scored_results"


class FactoryPreferenceAction(StrEnum):
    PREFER = "prefer"
    RETRACT = "retract"


def _optional_float(payload: JsonObject, name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number when present")
    return float(value)


def _required_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _string_tuple(payload: JsonObject, name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class FactoryLensSpec:
    """Declare how a Factory compares its Results.

    Submitting a spec for an existing ``lens_key`` appends a new immutable
    version rather than editing the old one, so a best-so-far answer computed
    under an earlier version stays reproducible.
    """

    lens_key: str
    direction: FactoryLensDirection
    objective: str | None = None
    missing_policy: FactoryLensMissingPolicy = FactoryLensMissingPolicy.INELIGIBLE
    tie_break: FactoryLensTieBreak = FactoryLensTieBreak.EARLIEST_RESULT
    eligible_result_kinds: tuple[FactoryResultKind, ...] = ()
    definition: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.lens_key, field_name="lens_key")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "lens_key": self.lens_key,
            "direction": self.direction.value,
            "missing_policy": self.missing_policy.value,
            "tie_break": self.tie_break.value,
            "eligible_result_kinds": [kind.value for kind in self.eligible_result_kinds],
            "definition": dict(self.definition),
        }
        if self.objective is not None:
            value["objective"] = self.objective
        return value


@dataclass(frozen=True, slots=True)
class FactoryEvaluationLens:
    """One immutable lens version as stored by the backend."""

    lens_id: str
    factory_id: FactoryId
    lens_key: str
    lens_version: int
    direction: FactoryLensDirection
    missing_policy: FactoryLensMissingPolicy
    tie_break: FactoryLensTieBreak
    status: str
    created_at: datetime
    objective: str | None = None
    eligible_result_kinds: tuple[str, ...] = ()
    definition: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryEvaluationLens:
        value = object_value(payload, operation_id="decode_factory_evaluation_lens")
        definition = value.get("definition")
        return cls(
            lens_id=required_text(value, "lens_id"),
            factory_id=FactoryId(required_text(value, "factory_id")),
            lens_key=required_text(value, "lens_key"),
            lens_version=_required_int(value, "lens_version"),
            direction=FactoryLensDirection(required_text(value, "direction")),
            missing_policy=FactoryLensMissingPolicy(required_text(value, "missing_policy")),
            tie_break=FactoryLensTieBreak(required_text(value, "tie_break")),
            status=required_text(value, "status"),
            created_at=required_datetime(value, "created_at"),
            objective=optional_text(value, "objective"),
            eligible_result_kinds=_string_tuple(value, "eligible_result_kinds"),
            definition=dict(definition) if isinstance(definition, dict) else {},
        )


@dataclass(frozen=True, slots=True)
class FactoryBestResult:
    """Derived best-so-far under one lens version.

    Read ``outcome`` before ``result_id``. The counts are reported for every
    outcome because knowing that 40 Results were considered and 0 scored is a
    different operational situation from 0 Results existing.
    """

    lens_key: str
    lens_version: int
    direction: FactoryLensDirection
    outcome: FactoryBestOutcome
    reason: str
    considered: int
    eligible: int
    scored: int
    objective: str | None = None
    result_id: str | None = None
    score: float | None = None

    @property
    def has_best(self) -> bool:
        return self.outcome is FactoryBestOutcome.BEST and self.result_id is not None

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryBestResult:
        value = object_value(payload, operation_id="decode_factory_best_result")
        return cls(
            lens_key=required_text(value, "lens_key"),
            lens_version=_required_int(value, "lens_version"),
            direction=FactoryLensDirection(required_text(value, "direction")),
            outcome=FactoryBestOutcome(required_text(value, "outcome")),
            reason=required_text(value, "reason"),
            considered=_required_int(value, "considered"),
            eligible=_required_int(value, "eligible"),
            scored=_required_int(value, "scored"),
            objective=optional_text(value, "objective"),
            result_id=optional_text(value, "result_id"),
            score=_optional_float(value, "score"),
        )


@dataclass(frozen=True, slots=True)
class FactoryBestResults:
    """Best-so-far across every lens a Factory declares.

    ``optimizes is False`` is a valid steady state: the Factory does useful work
    and hillclimbs nothing. Render it differently from "no results yet".
    """

    factory_id: FactoryId
    optimizes: bool
    lenses: tuple[FactoryBestResult, ...] = ()

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryBestResults:
        value = object_value(payload, operation_id="decode_factory_best_results")
        optimizes = value.get("optimizes")
        if not isinstance(optimizes, bool):
            raise ValueError("factory best results optimizes must be a boolean")
        return cls(
            factory_id=FactoryId(required_text(value, "factory_id")),
            optimizes=optimizes,
            lenses=tuple(
                FactoryBestResult.from_wire(item)
                for item in array_value(
                    value.get("lenses", []),
                    operation_id="retrieve_factory_best_results",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class FactoryResultEvaluationRequest:
    """Post one externally owned verdict for a Result under a lens."""

    lens_key: str
    attempt_key: str
    status: FactoryEvaluationStatus
    score: float | None = None
    baseline_score: float | None = None
    evaluator: str | None = None
    record: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.lens_key, field_name="lens_key")
        require_text(self.attempt_key, field_name="attempt_key")
        if self.status is FactoryEvaluationStatus.EVALUATED and self.score is None:
            raise ValueError("evaluation_score_required: an 'evaluated' verdict must carry a score")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "lens_key": self.lens_key,
            "attempt_key": self.attempt_key,
            "status": self.status.value,
            "record": dict(self.record),
        }
        if self.score is not None:
            value["score"] = float(self.score)
        if self.baseline_score is not None:
            value["baseline_score"] = float(self.baseline_score)
        if self.evaluator is not None:
            value["evaluator"] = self.evaluator
        return value


@dataclass(frozen=True, slots=True)
class FactoryResultEvaluation:
    """One stored verdict."""

    evaluation_id: str
    result_id: str
    lens_id: str
    lens_key: str
    lens_version: int
    attempt_key: str
    status: FactoryEvaluationStatus
    created_at: datetime
    score: float | None = None
    baseline_score: float | None = None
    evaluator: str | None = None
    record: JsonObject = field(default_factory=dict)
    evaluated_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryResultEvaluation:
        value = object_value(payload, operation_id="decode_factory_result_evaluation")
        record = value.get("record")
        return cls(
            evaluation_id=required_text(value, "evaluation_id"),
            result_id=required_text(value, "result_id"),
            lens_id=required_text(value, "lens_id"),
            lens_key=required_text(value, "lens_key"),
            lens_version=_required_int(value, "lens_version"),
            attempt_key=required_text(value, "attempt_key"),
            status=FactoryEvaluationStatus(required_text(value, "status")),
            created_at=required_datetime(value, "created_at"),
            score=_optional_float(value, "score"),
            baseline_score=_optional_float(value, "baseline_score"),
            evaluator=optional_text(value, "evaluator"),
            record=dict(record) if isinstance(record, dict) else {},
            evaluated_at=optional_datetime(value, "evaluated_at"),
        )


@dataclass(frozen=True, slots=True)
class FactoryPreferenceRequest:
    """Record a human preference beside the derived best, not instead of it."""

    idempotency_key: str
    reason: str
    action: FactoryPreferenceAction = FactoryPreferenceAction.PREFER
    result_id: str | None = None
    lens_key: str | None = None

    def __post_init__(self) -> None:
        require_text(self.idempotency_key, field_name="idempotency_key")
        require_text(self.reason, field_name="reason")
        if self.action is FactoryPreferenceAction.PREFER and not self.result_id:
            raise ValueError("preference_result_required: a 'prefer' event must name a Result")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "action": self.action.value,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
        }
        if self.result_id is not None:
            value["result_id"] = self.result_id
        if self.lens_key is not None:
            value["lens_key"] = self.lens_key
        return value


@dataclass(frozen=True, slots=True)
class FactoryPreferenceEvent:
    """One immutable preference event."""

    event_id: str
    factory_id: FactoryId
    action: FactoryPreferenceAction
    actor_kind: str
    idempotency_key: str
    created_at: datetime
    result_id: str | None = None
    lens_id: str | None = None
    actor_id: str | None = None
    reason: str | None = None

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryPreferenceEvent:
        value = object_value(payload, operation_id="decode_factory_preference_event")
        return cls(
            event_id=required_text(value, "event_id"),
            factory_id=FactoryId(required_text(value, "factory_id")),
            action=FactoryPreferenceAction(required_text(value, "action")),
            actor_kind=required_text(value, "actor_kind"),
            idempotency_key=required_text(value, "idempotency_key"),
            created_at=required_datetime(value, "created_at"),
            result_id=optional_text(value, "result_id"),
            lens_id=optional_text(value, "lens_id"),
            actor_id=optional_text(value, "actor_id"),
            reason=optional_text(value, "reason"),
        )


__all__ = [
    "FactoryBestOutcome",
    "FactoryBestResult",
    "FactoryBestResults",
    "FactoryEvaluationLens",
    "FactoryEvaluationStatus",
    "FactoryLensDirection",
    "FactoryLensMissingPolicy",
    "FactoryLensSpec",
    "FactoryLensTieBreak",
    "FactoryPreferenceAction",
    "FactoryPreferenceEvent",
    "FactoryPreferenceRequest",
    "FactoryResultEvaluation",
    "FactoryResultEvaluationRequest",
    "FactoryResultKind",
]
