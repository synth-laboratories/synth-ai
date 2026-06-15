"""Typed public launch, read, output, interrupt, and stream models."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Self, TypeAlias

from synth_ai.managed_research.models.local_execution_profile import LocalExecutionProfile
from synth_ai.managed_research.models.run_events import (
    RunRuntimeStreamEvent,
    RunRuntimeStreamEventKind,
    TranscriptEvent,
)
from synth_ai.managed_research.models.run_observability import RunObservabilitySnapshot
from synth_ai.managed_research.models.runtime_image import (
    EnvironmentSpec,
    RuntimeImage,
    SandboxOverride,
    align_execution_profile_for_runtime_image,
    runtime_image_launch_patches,
)
from synth_ai.managed_research.models.smr_actor_models import SmrActorModelAssignment
from synth_ai.managed_research.models.smr_agent_harnesses import SmrAgentHarness
from synth_ai.managed_research.models.smr_agent_kinds import SmrAgentKind
from synth_ai.managed_research.models.smr_agent_models import SmrAgentModel
from synth_ai.managed_research.models.smr_horizons import SmrIntendedHorizonHours
from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind
from synth_ai.managed_research.models.smr_providers import ProviderBinding, UsageLimit
from synth_ai.managed_research.models.smr_roles import SmrRoleBindings
from synth_ai.managed_research.models.smr_run_policy import SmrRunPolicy
from synth_ai.managed_research.models.smr_runbooks import SmrRunbookKind
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode
from synth_ai.managed_research.models.types import (
    KickoffContract,
    KickoffContractFile,
    RunResourceBindings,
)

WirePayload: TypeAlias = dict[str, object]
WireMapping: TypeAlias = Mapping[str, object]


class Request:
    """Base class for public SDK request objects."""


class CommandRequest(Request):
    """Base class for requests that ask Managed Research to perform work."""


class ReadRequest(Request):
    """Base class for requests that read Managed Research state."""


class Ref:
    """Base class for stable Managed Research references."""


@dataclass(frozen=True, slots=True)
class RunRef(Ref):
    project_id: str
    run_id: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.project_id, field_name="project_id")
        _require_non_empty_text(self.run_id, field_name="run_id")

    def to_wire(self) -> WirePayload:
        return {"project_id": self.project_id, "run_id": self.run_id}

    @classmethod
    def from_wire(cls, payload: WireMapping) -> Self:
        return cls(
            project_id=_required_text(payload, "project_id"),
            run_id=_required_text(payload, "run_id"),
        )


@dataclass(frozen=True, slots=True)
class Kickoff:
    initial_runtime_messages: Sequence[WireMapping] = ()
    contract: KickoffContract | WireMapping | None = None
    instructions: str | None = None
    objective: str | None = None
    task_briefs: Sequence[str] = ()
    model_visible_files: Sequence[KickoffContractFile | WireMapping] = ()

    def __post_init__(self) -> None:
        if self.instructions is not None:
            _require_non_empty_text(self.instructions, field_name="instructions")
        if self.objective is not None:
            _require_non_empty_text(self.objective, field_name="objective")
        for index, message in enumerate(self.initial_runtime_messages):
            _require_mapping(message, label=f"initial_runtime_messages[{index}]")
        for index, brief in enumerate(self.task_briefs):
            _require_non_empty_text(brief, field_name=f"task_briefs[{index}]")
        for index, file in enumerate(self.model_visible_files):
            _coerce_kickoff_contract_file(file, label=f"model_visible_files[{index}]")

    @classmethod
    def from_messages(
        cls,
        messages: Iterable[WireMapping],
        *,
        contract: KickoffContract | WireMapping | None = None,
        instructions: str | None = None,
        objective: str | None = None,
        task_briefs: Sequence[str] = (),
        model_visible_files: Sequence[KickoffContractFile | WireMapping] = (),
    ) -> Self:
        return cls(
            initial_runtime_messages=tuple(dict(message) for message in messages),
            contract=contract,
            instructions=instructions,
            objective=objective,
            task_briefs=tuple(task_briefs),
            model_visible_files=tuple(model_visible_files),
        )

    def to_client_kwargs(self) -> WirePayload:
        payload: WirePayload = {}
        messages = [dict(message) for message in self.initial_runtime_messages]
        instruction_messages = self._instruction_messages()
        if instruction_messages:
            messages = [*instruction_messages, *messages]
        if messages:
            payload["initial_runtime_messages"] = messages

        contract = self._contract_payload()
        if contract is not None:
            payload["kickoff_contract"] = contract
        return payload

    def to_wire(self) -> WirePayload:
        return dict(self.to_client_kwargs())

    def _instruction_messages(self) -> list[WirePayload]:
        messages: list[WirePayload] = []
        if self.objective is not None:
            messages.append(
                {
                    "role": "user",
                    "content": self.objective.strip(),
                    "kind": "objective",
                }
            )
        if self.instructions is not None:
            messages.append(
                {
                    "role": "user",
                    "content": self.instructions.strip(),
                    "kind": "instructions",
                }
            )
        return messages

    def _contract_payload(self) -> WirePayload | None:
        if self.contract is None:
            if not self.task_briefs and not self.model_visible_files:
                return None
            raise ValueError("task_briefs and model_visible_files require a kickoff contract")

        if isinstance(self.contract, KickoffContract):
            payload = self.contract.to_wire()
        else:
            payload = dict(_require_mapping(self.contract, label="kickoff contract"))

        if self.task_briefs:
            if payload.get("task_briefs"):
                raise ValueError("kickoff contract already defines task_briefs")
            payload["task_briefs"] = [brief.strip() for brief in self.task_briefs]
        if self.model_visible_files:
            if payload.get("model_visible_contract_files"):
                raise ValueError("kickoff contract already defines model_visible_contract_files")
            payload["model_visible_contract_files"] = [
                _coerce_kickoff_contract_file(file, label="model_visible_files").to_wire()
                for file in self.model_visible_files
            ]
        return payload


@dataclass(frozen=True, slots=True)
class RunLaunchRequest(CommandRequest):
    host_kind: SmrHostKind | str | None = None
    work_mode: SmrWorkMode | str | None = None
    mode: SmrWorkMode | str | None = None
    intended_horizon_hours: SmrIntendedHorizonHours | int | None = None
    providers: Sequence[ProviderBinding | str | WireMapping] = ()
    limit: UsageLimit | WireMapping | None = None
    worker_pool_id: str | None = None
    runbook: SmrRunbookKind | str | None = None
    runbook_preset: str | None = None
    runbook_config_id: str | None = None
    local_execution: WireMapping | None = None
    execution_profile: LocalExecutionProfile | WireMapping | None = None
    timebox_seconds: int | None = None
    agent_profile: str | None = None
    agent_model: SmrAgentModel | str | None = None
    agent_harness: SmrAgentHarness | None = None
    agent_kind: SmrAgentKind | None = None
    agent_model_params: WireMapping | None = None
    actor_model_overrides: Sequence[SmrActorModelAssignment | WireMapping] = ()
    roles: SmrRoleBindings | WireMapping | None = None
    kickoff: Kickoff | None = None
    open_ended_question: WireMapping | None = None
    directed_effort_outcome: WireMapping | None = None
    required_work_products: Sequence[WireMapping] | None = None
    require_report: bool | None = None
    initial_runtime_messages: Sequence[WireMapping] = ()
    workflow: WireMapping | None = None
    runtime_image: RuntimeImage | None = None
    sandbox_override: SandboxOverride | WireMapping | None = None
    environment: EnvironmentSpec | WireMapping | None = None
    run_policy: SmrRunPolicy | WireMapping | None = None
    kickoff_contract: KickoffContract | WireMapping | None = None
    resource_bindings: RunResourceBindings | WireMapping | None = None
    ai_cache: WireMapping | None = None
    primary_objective_id: str | None = None
    primary_objective_kind: str | None = None
    primary_parent_ref: WireMapping | None = None
    primary_parent: WireMapping | None = None
    idempotency_key_run_create: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        _validate_launch_text(self.worker_pool_id, field_name="worker_pool_id")
        _validate_launch_text(self.runbook_preset, field_name="runbook_preset")
        _validate_launch_text(self.runbook_config_id, field_name="runbook_config_id")
        _validate_launch_text(self.agent_profile, field_name="agent_profile")
        _validate_launch_text(self.primary_objective_id, field_name="primary_objective_id")
        _validate_launch_text(
            self.primary_objective_kind,
            field_name="primary_objective_kind",
        )
        _validate_launch_text(
            self.idempotency_key_run_create,
            field_name="idempotency_key_run_create",
        )
        _validate_launch_text(self.idempotency_key, field_name="idempotency_key")
        _validate_positive_int(self.timebox_seconds, field_name="timebox_seconds")
        _validate_launch_sequences(self)
        _validate_launch_mappings(self)
        _validate_launch_combinations(self)

    def to_client_kwargs(self) -> WirePayload:
        payload = _non_empty_kwargs(
            host_kind=self.host_kind,
            work_mode=self.work_mode,
            mode=self.mode,
            intended_horizon_hours=self.intended_horizon_hours,
            providers=tuple(self.providers) if self.providers else None,
            limit=self.limit,
            worker_pool_id=self.worker_pool_id,
            runbook=self.runbook,
            runbook_preset=self.runbook_preset,
            runbook_config_id=self.runbook_config_id,
            local_execution=self.local_execution,
            execution_profile=self.execution_profile,
            timebox_seconds=self.timebox_seconds,
            agent_profile=self.agent_profile,
            agent_model=self.agent_model,
            agent_harness=self.agent_harness,
            agent_kind=self.agent_kind,
            agent_model_params=self.agent_model_params,
            actor_model_overrides=tuple(self.actor_model_overrides)
            if self.actor_model_overrides
            else None,
            roles=self.roles,
            open_ended_question=self.open_ended_question,
            directed_effort_outcome=self.directed_effort_outcome,
            required_work_products=tuple(self.required_work_products)
            if self.required_work_products
            else None,
            require_report=self.require_report,
            initial_runtime_messages=tuple(self.initial_runtime_messages)
            if self.initial_runtime_messages
            else None,
            workflow=self.workflow,
            run_policy=self.run_policy,
            kickoff_contract=self.kickoff_contract,
            resource_bindings=self.resource_bindings,
            ai_cache=self.ai_cache,
            primary_objective_id=self.primary_objective_id,
            primary_objective_kind=self.primary_objective_kind,
            primary_parent_ref=self.primary_parent_ref,
            primary_parent=self.primary_parent,
            idempotency_key_run_create=self.idempotency_key_run_create,
            idempotency_key=self.idempotency_key,
        )
        if self.kickoff is not None:
            payload.update(self.kickoff.to_client_kwargs())

        environment_payload, sandbox_payload = runtime_image_launch_patches(
            self.runtime_image,
            environment=self.environment,
            sandbox_override=self.sandbox_override,
        )
        if environment_payload is not None:
            payload["environment"] = environment_payload
        if sandbox_payload is not None:
            payload["sandbox_override"] = sandbox_payload
        if self.runtime_image is not None:
            execution_profile = payload.get("execution_profile")
            if isinstance(execution_profile, Mapping):
                payload["execution_profile"] = align_execution_profile_for_runtime_image(
                    self.runtime_image,
                    host_kind=payload.get("host_kind", self.host_kind),
                    execution_profile=execution_profile,
                )
        return payload

    def to_wire(self) -> WirePayload:
        from synth_ai.managed_research.sdk.client import _build_project_run_payload

        return _build_project_run_payload(**self.to_client_kwargs())

    @classmethod
    def from_client_kwargs(cls, values: Mapping[str, object]) -> Self:
        return cls(**_run_launch_init_kwargs(values))


LaunchPreflightRequest = RunLaunchRequest


@dataclass(frozen=True, slots=True)
class RunLaunchResult:
    run_ref: RunRef
    payload: WireMapping = field(default_factory=dict)

    @classmethod
    def from_wire(cls, *, project_id: str, payload: WireMapping) -> Self:
        run_id = _first_required_text(payload, ("run_id", "id"), label="run_id")
        return cls(
            run_ref=RunRef(project_id=project_id, run_id=run_id),
            payload=dict(payload),
        )

    @property
    def project_id(self) -> str:
        return self.run_ref.project_id

    @property
    def run_id(self) -> str:
        return self.run_ref.run_id

    def to_wire(self) -> WirePayload:
        payload = dict(self.payload)
        payload["project_id"] = self.project_id
        payload["run_id"] = self.run_id
        return payload


@dataclass(frozen=True, slots=True)
class RunReadRequest(ReadRequest):
    run_ref: RunRef
    event_limit: int | None = None
    actor_limit: int | None = None
    task_limit: int | None = None
    question_limit: int | None = None
    timeline_limit: int | None = None
    message_limit: int | None = None
    include_raw_diagnostics: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "event_limit",
            "actor_limit",
            "task_limit",
            "question_limit",
            "timeline_limit",
            "message_limit",
        ):
            _validate_positive_int(getattr(self, field_name), field_name=field_name)

    def to_query(self) -> WirePayload:
        return _non_empty_kwargs(
            event_limit=self.event_limit,
            actor_limit=self.actor_limit,
            task_limit=self.task_limit,
            question_limit=self.question_limit,
            timeline_limit=self.timeline_limit,
            message_limit=self.message_limit,
            include_raw_diagnostics=self.include_raw_diagnostics,
        )


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_ref: RunRef
    snapshot: RunObservabilitySnapshot | None = None
    generated_at: datetime | None = None
    cursor: str | None = None
    outputs: Sequence[Output] = ()

    @classmethod
    def from_observability(
        cls,
        *,
        run_ref: RunRef,
        snapshot: RunObservabilitySnapshot,
    ) -> Self:
        return cls(run_ref=run_ref, snapshot=snapshot)


class OutputKind(StrEnum):
    REPORT = "report"
    MODEL = "model"
    TRAINED_MODEL = "trained_model"
    CONTAINER_EVAL = "container_eval"
    CONTAINER_EVALUATION = "container_evaluation"
    EXPERIMENT_RESULT = "experiment_result"
    VISUAL = "visual"
    WORKSPACE_ARCHIVE = "workspace_archive"
    ARTIFACT_MANIFEST = "artifact_manifest"


@dataclass(frozen=True, slots=True)
class Output:
    output_id: str
    kind: OutputKind
    title: str | None = None
    summary: str | None = None
    run_ref: RunRef | None = None
    created_at: datetime | None = None
    payload: WireMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.output_id, field_name="output_id")

    @classmethod
    def from_wire(cls, payload: WireMapping, *, run_ref: RunRef | None = None) -> Self:
        output_id = _first_required_text(
            payload,
            ("output_id", "work_product_id", "artifact_id", "id"),
            label="output_id",
        )
        kind = _output_kind_from_payload(payload)
        payload_run_ref = _run_ref_from_payload(payload)
        return cls(
            output_id=output_id,
            kind=kind,
            title=_optional_text(payload, "title"),
            summary=_optional_text(payload, "summary") or _optional_text(payload, "description"),
            run_ref=payload_run_ref or run_ref,
            created_at=_optional_datetime(payload, "created_at"),
            payload=dict(payload),
        )

    def to_wire(self) -> WirePayload:
        payload = _non_empty_kwargs(
            output_id=self.output_id,
            kind=self.kind.value,
            title=self.title,
            summary=self.summary,
            run_ref=None if self.run_ref is None else self.run_ref.to_wire(),
            created_at=None if self.created_at is None else self.created_at.isoformat(),
        )
        payload.update(dict(self.payload))
        return payload


@dataclass(frozen=True, slots=True)
class ReportOutput(Output):
    kind: OutputKind = OutputKind.REPORT


@dataclass(frozen=True, slots=True)
class OutputListRequest(ReadRequest):
    project_id: str | None = None
    run_ref: RunRef | None = None
    kinds: Sequence[OutputKind] = ()
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.project_id is None and self.run_ref is None:
            raise ValueError("project_id or run_ref is required")
        if self.project_id is not None and self.run_ref is not None:
            raise ValueError("project_id and run_ref cannot both be provided")
        _validate_launch_text(self.project_id, field_name="project_id")
        _validate_positive_int(self.limit, field_name="limit")

    def to_query(self) -> WirePayload:
        return _non_empty_kwargs(
            kinds=[kind.value for kind in self.kinds] if self.kinds else None,
            limit=self.limit,
        )


@dataclass(frozen=True, slots=True)
class Interrupt:
    interrupt_id: str
    run_ref: RunRef
    reason: str | None = None
    payload: WireMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.interrupt_id, field_name="interrupt_id")
        _validate_launch_text(self.reason, field_name="reason")

    @classmethod
    def from_wire(cls, payload: WireMapping) -> Self:
        return cls(
            interrupt_id=_first_required_text(
                payload,
                ("interrupt_id", "approval_id", "id"),
                label="interrupt_id",
            ),
            run_ref=RunRef.from_wire(_required_mapping(payload, "run_ref")),
            reason=_optional_text(payload, "reason"),
            payload=dict(payload),
        )


class EventKind(StrEnum):
    RUN_STATE_CHANGED = "run.state.changed"
    ACTOR_STATE_CHANGED = "actor.state.changed"
    TASK_STATE_CHANGED = "task.state.changed"
    RUNTIME_MESSAGE = "runtime.message"
    OUTPUT_PUBLISHED = "output.published"
    WORK_PRODUCT_PUBLISHED = "work_product.published"
    CHECKPOINT_CREATED = "checkpoint.created"
    INTERRUPT_CREATED = "interrupt.created"
    DIAGNOSTIC_RECORDED = "diagnostic.recorded"
    HEARTBEAT = "heartbeat"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Event:
    event_kind: EventKind
    run_ref: RunRef | None
    event_id: str | None = None
    sequence: str | None = None
    created_at: datetime | None = None
    payload: object = None
    raw_event: RunRuntimeStreamEvent | None = None

    @classmethod
    def from_runtime_event(
        cls,
        event: RunRuntimeStreamEvent,
        *,
        run_ref: RunRef | None = None,
    ) -> Event:
        parser = _EVENT_PARSERS.get(event.kind)
        if parser is not None:
            return parser(event, run_ref=run_ref)
        return UnknownEvent(
            event_kind=EventKind.UNKNOWN,
            run_ref=_event_run_ref(event, run_ref),
            event_id=event.event_id,
            sequence=event.sequence,
            created_at=event.occurred_at,
            payload=event.payload,
            raw_event=event,
        )


@dataclass(frozen=True, slots=True)
class RunEvent(Event):
    state: str | None = None


@dataclass(frozen=True, slots=True)
class RunStateChangedEvent(RunEvent):
    pass


@dataclass(frozen=True, slots=True)
class ActorStateChangedEvent(Event):
    actor_id: str | None = None
    actor_key: str | None = None
    state: str | None = None


@dataclass(frozen=True, slots=True)
class TaskEvent(Event):
    task_key: str | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskStateChangedEvent(TaskEvent):
    state: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeMessageEvent(Event):
    transcript_event: TranscriptEvent | None = None
    message_kind: str | None = None
    participant_role: str | None = None


@dataclass(frozen=True, slots=True)
class OutputPublishedEvent(Event):
    output_id: str | None = None
    output: Output | None = None


@dataclass(frozen=True, slots=True)
class WorkProductPublishedEvent(OutputPublishedEvent):
    work_product_id: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointCreatedEvent(Event):
    checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class InterruptCreatedEvent(Event):
    interrupt_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticEvent(Event):
    code: str | None = None
    severity: str | None = None


@dataclass(frozen=True, slots=True)
class HeartbeatEvent(Event):
    @classmethod
    def from_runtime_event(
        cls,
        event: RunRuntimeStreamEvent,
        *,
        run_ref: RunRef | None = None,
    ) -> HeartbeatEvent:
        return cls(
            event_kind=EventKind.HEARTBEAT,
            run_ref=_event_run_ref(event, run_ref),
            event_id=event.event_id,
            sequence=event.sequence,
            created_at=event.occurred_at,
            payload=event.payload,
            raw_event=event,
        )


@dataclass(frozen=True, slots=True)
class UnknownEvent(Event):
    pass


class EventStreamReplayPolicy(StrEnum):
    RESUME_AFTER_LAST_EVENT = "resume_after_last_event"
    REPLAY_FROM_CURSOR = "replay_from_cursor"


class EventStreamPhase(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class EventStreamRequest(ReadRequest):
    run_ref: RunRef
    event_kinds: Sequence[EventKind] = ()
    last_event_id: str | None = None
    transcript_cursor: str | None = None
    replay_policy: EventStreamReplayPolicy = EventStreamReplayPolicy.RESUME_AFTER_LAST_EVENT
    heartbeat_timeout_seconds: int = 60
    max_replay_events: int | None = None
    actor_id: str | None = None
    task_key: str | None = None

    def __post_init__(self) -> None:
        _validate_launch_text(self.last_event_id, field_name="last_event_id")
        _validate_launch_text(self.transcript_cursor, field_name="transcript_cursor")
        _validate_positive_int(
            self.heartbeat_timeout_seconds,
            field_name="heartbeat_timeout_seconds",
        )
        _validate_positive_int(self.max_replay_events, field_name="max_replay_events")
        _validate_launch_text(self.actor_id, field_name="actor_id")
        _validate_launch_text(self.task_key, field_name="task_key")


@dataclass(slots=True)
class EventStreamState:
    run_ref: RunRef
    phase: EventStreamPhase = EventStreamPhase.OPEN
    last_event_id: str | None = None
    last_sequence: str | None = None
    transcript_cursor: str | None = None
    terminal_event: Event | None = None
    events_seen: int = 0
    output_refs: list[Output] = field(default_factory=list)
    snapshot_payload: WirePayload = field(default_factory=dict)

    def accumulate(self, event: Event) -> None:
        if self.phase is not EventStreamPhase.OPEN:
            raise ValueError(f"cannot accumulate event while stream is {self.phase.value}")
        if event.run_ref is not None:
            assert event.run_ref.run_id == self.run_ref.run_id
        self._assert_sequence_order(event)
        self.events_seen += 1
        self.last_event_id = event.event_id or self.last_event_id
        self.last_sequence = event.sequence or self.last_sequence
        if isinstance(event, OutputPublishedEvent) and event.output is not None:
            self.output_refs.append(event.output)
        if isinstance(event, RunStateChangedEvent) and _is_terminal_state(event.state):
            self.terminal_event = event
        raw = event.raw_event
        if raw is not None:
            self.transcript_cursor = raw.transcript_cursor or self.transcript_cursor
            if raw.kind == RunRuntimeStreamEventKind.SNAPSHOT.value:
                self.snapshot_payload = dict(_required_event_payload_mapping(raw))

    def get_outputs(self) -> tuple[Output, ...]:
        return tuple(self.output_refs)

    def transition(self, phase: EventStreamPhase) -> None:
        valid = {
            EventStreamPhase.OPEN: {
                EventStreamPhase.CLOSED,
                EventStreamPhase.EXHAUSTED,
            },
            EventStreamPhase.CLOSED: set(),
            EventStreamPhase.EXHAUSTED: set(),
        }
        if phase not in valid[self.phase]:
            raise ValueError(
                f"invalid event stream transition: {self.phase.value} -> {phase.value}"
            )
        self.phase = phase

    def _assert_sequence_order(self, event: Event) -> None:
        if self.last_sequence is None or event.sequence is None:
            return
        if event.sequence == self.last_sequence:
            raise ValueError(f"duplicate stream sequence: {event.sequence}")


class EventStream(Iterator[Event]):
    def __init__(
        self,
        events: Iterable[RunRuntimeStreamEvent],
        *,
        request: EventStreamRequest,
    ) -> None:
        self._iterator = iter(events)
        self.request = request
        self.state = EventStreamState(run_ref=request.run_ref)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __iter__(self) -> EventStream:
        return self

    def __next__(self) -> Event:
        if self.state.phase is not EventStreamPhase.OPEN:
            raise StopIteration
        try:
            raw_event = next(self._iterator)
        except StopIteration:
            self.state.transition(EventStreamPhase.EXHAUSTED)
            raise
        event = Event.from_runtime_event(
            raw_event,
            run_ref=self.request.run_ref,
        )
        if self.request.event_kinds and event.event_kind not in self.request.event_kinds:
            return self.__next__()
        self.state.accumulate(event)
        return event

    def close(self) -> None:
        if self.state.phase is EventStreamPhase.OPEN:
            self.state.transition(EventStreamPhase.CLOSED)
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def until_done(self) -> None:
        for _event in self:
            if self.state.terminal_event is not None:
                return

    def get_outputs(self) -> tuple[Output, ...]:
        return self.state.get_outputs()

    def get_final_snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            run_ref=self.request.run_ref,
            cursor=self.state.transcript_cursor,
            outputs=self.state.get_outputs(),
        )


class AsyncEventStream(AsyncIterator[Event]):
    def __init__(
        self,
        events: AsyncIterable[RunRuntimeStreamEvent],
        *,
        request: EventStreamRequest,
    ) -> None:
        self._iterator = events.__aiter__()
        self.request = request
        self.state = EventStreamState(run_ref=request.run_ref)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def __aiter__(self) -> AsyncEventStream:
        return self

    async def __anext__(self) -> Event:
        if self.state.phase is not EventStreamPhase.OPEN:
            raise StopAsyncIteration
        try:
            raw_event = await self._iterator.__anext__()
        except StopAsyncIteration:
            self.state.transition(EventStreamPhase.EXHAUSTED)
            raise
        event = Event.from_runtime_event(
            raw_event,
            run_ref=self.request.run_ref,
        )
        if self.request.event_kinds and event.event_kind not in self.request.event_kinds:
            return await self.__anext__()
        self.state.accumulate(event)
        return event

    async def close(self) -> None:
        if self.state.phase is EventStreamPhase.OPEN:
            self.state.transition(EventStreamPhase.CLOSED)
        close = getattr(self._iterator, "aclose", None)
        if callable(close):
            await close()

    async def until_done(self) -> None:
        async for _event in self:
            if self.state.terminal_event is not None:
                return

    def get_outputs(self) -> tuple[Output, ...]:
        return self.state.get_outputs()

    def get_final_snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            run_ref=self.request.run_ref,
            cursor=self.state.transcript_cursor,
            outputs=self.state.get_outputs(),
        )


def _parse_snapshot_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    timeline = _optional_mapping(payload, "timeline")
    state = None
    if timeline is not None:
        state = _optional_text(timeline, "authority_state")
    return RunStateChangedEvent(
        event_kind=EventKind.RUN_STATE_CHANGED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        state=state,
    )


def _parse_transcript_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    transcript_event = event.transcript_event
    if transcript_event is None:
        raise ValueError("transcript stream event requires transcript payload")
    return RuntimeMessageEvent(
        event_kind=EventKind.RUNTIME_MESSAGE,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=event.payload,
        raw_event=event,
        transcript_event=transcript_event,
        message_kind=transcript_event.kind,
        participant_role=transcript_event.participant_role,
    )


def _parse_heartbeat_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    return HeartbeatEvent.from_runtime_event(event, run_ref=run_ref)


def _parse_run_state_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return RunStateChangedEvent(
        event_kind=EventKind.RUN_STATE_CHANGED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        state=_optional_text(payload, "state"),
    )


def _parse_actor_state_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return ActorStateChangedEvent(
        event_kind=EventKind.ACTOR_STATE_CHANGED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        actor_id=_optional_text(payload, "actor_id"),
        actor_key=_optional_text(payload, "actor_key"),
        state=_optional_text(payload, "state"),
    )


def _parse_task_state_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return TaskStateChangedEvent(
        event_kind=EventKind.TASK_STATE_CHANGED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        task_key=_optional_text(payload, "task_key"),
        task_id=_optional_text(payload, "task_id"),
        state=_optional_text(payload, "state"),
    )


def _parse_output_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    event_run_ref = _event_run_ref(event, run_ref)
    output = Output.from_wire(payload, run_ref=event_run_ref)
    return OutputPublishedEvent(
        event_kind=EventKind.OUTPUT_PUBLISHED,
        run_ref=event_run_ref,
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        output_id=output.output_id,
        output=output,
    )


def _parse_work_product_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    event_run_ref = _event_run_ref(event, run_ref)
    output = Output.from_wire(payload, run_ref=event_run_ref)
    return WorkProductPublishedEvent(
        event_kind=EventKind.WORK_PRODUCT_PUBLISHED,
        run_ref=event_run_ref,
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        output_id=output.output_id,
        output=output,
        work_product_id=_optional_text(payload, "work_product_id"),
    )


def _parse_checkpoint_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return CheckpointCreatedEvent(
        event_kind=EventKind.CHECKPOINT_CREATED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        checkpoint_id=_optional_text(payload, "checkpoint_id"),
    )


def _parse_interrupt_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return InterruptCreatedEvent(
        event_kind=EventKind.INTERRUPT_CREATED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        interrupt_id=_optional_text(payload, "interrupt_id"),
    )


def _parse_diagnostic_event(
    event: RunRuntimeStreamEvent,
    *,
    run_ref: RunRef | None,
) -> Event:
    payload = _required_event_payload_mapping(event)
    return DiagnosticEvent(
        event_kind=EventKind.DIAGNOSTIC_RECORDED,
        run_ref=_event_run_ref(event, run_ref),
        event_id=event.event_id,
        sequence=event.sequence,
        created_at=event.occurred_at,
        payload=dict(payload),
        raw_event=event,
        code=_optional_text(payload, "code"),
        severity=_optional_text(payload, "severity"),
    )


_EVENT_PARSERS = {
    RunRuntimeStreamEventKind.SNAPSHOT.value: _parse_snapshot_event,
    RunRuntimeStreamEventKind.TRANSCRIPT.value: _parse_transcript_event,
    RunRuntimeStreamEventKind.HEARTBEAT.value: _parse_heartbeat_event,
    EventKind.RUN_STATE_CHANGED.value: _parse_run_state_event,
    EventKind.ACTOR_STATE_CHANGED.value: _parse_actor_state_event,
    EventKind.TASK_STATE_CHANGED.value: _parse_task_state_event,
    EventKind.OUTPUT_PUBLISHED.value: _parse_output_event,
    EventKind.WORK_PRODUCT_PUBLISHED.value: _parse_work_product_event,
    EventKind.CHECKPOINT_CREATED.value: _parse_checkpoint_event,
    EventKind.INTERRUPT_CREATED.value: _parse_interrupt_event,
    EventKind.DIAGNOSTIC_RECORDED.value: _parse_diagnostic_event,
}


def _non_empty_kwargs(**values: object) -> WirePayload:
    return {key: value for key, value in values.items() if value is not None}


def _required_mapping(payload: WireMapping, key: str) -> WireMapping:
    value = payload.get(key)
    return _require_mapping(value, label=key)


def _optional_mapping(payload: WireMapping, key: str) -> WireMapping | None:
    value = payload.get(key)
    if value is None:
        return None
    return _require_mapping(value, label=key)


def _require_mapping(payload: object, *, label: str) -> WireMapping:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _required_event_payload_mapping(event: RunRuntimeStreamEvent) -> WireMapping:
    return _require_mapping(event.payload, label=f"{event.kind} event payload")


def _required_text(payload: WireMapping, key: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _first_required_text(
    payload: WireMapping,
    keys: Sequence[str],
    *,
    label: str,
) -> str:
    for key in keys:
        value = _optional_text(payload, key)
        if value is not None:
            return value
    raise ValueError(f"{label} is required")


def _optional_text(payload: WireMapping, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    text = value.strip()
    return text or None


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} is required")


def _validate_launch_text(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_non_empty_text(value, field_name=field_name)


def _validate_positive_int(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _optional_datetime(payload: WireMapping, key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO-8601 datetime")
    text = value.strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _coerce_kickoff_contract_file(
    value: KickoffContractFile | WireMapping,
    *,
    label: str,
) -> KickoffContractFile:
    if isinstance(value, KickoffContractFile):
        return value
    return KickoffContractFile.from_wire(_require_mapping(value, label=label))


def _validate_launch_sequences(request: RunLaunchRequest) -> None:
    for index, provider in enumerate(request.providers):
        if not isinstance(provider, (ProviderBinding, str, Mapping)):
            raise ValueError(f"providers[{index}] must be a provider binding")
    for index, message in enumerate(request.initial_runtime_messages):
        _require_mapping(message, label=f"initial_runtime_messages[{index}]")
    for index, override in enumerate(request.actor_model_overrides):
        if not isinstance(override, (SmrActorModelAssignment, Mapping)):
            raise ValueError(f"actor_model_overrides[{index}] must be an actor model assignment")
    for index, spec in enumerate(request.required_work_products or ()):
        _require_mapping(spec, label=f"required_work_products[{index}]")


def _validate_launch_mappings(request: RunLaunchRequest) -> None:
    mapping_fields = (
        "local_execution",
        "agent_model_params",
        "workflow",
        "primary_parent_ref",
        "primary_parent",
        "ai_cache",
        "open_ended_question",
        "directed_effort_outcome",
    )
    for field_name in mapping_fields:
        value = getattr(request, field_name)
        if value is not None:
            _require_mapping(value, label=field_name)
    if request.environment is not None and not isinstance(request.environment, EnvironmentSpec):
        _require_mapping(request.environment, label="environment")
    if request.sandbox_override is not None and not isinstance(
        request.sandbox_override, SandboxOverride
    ):
        _require_mapping(request.sandbox_override, label="sandbox_override")


def _validate_launch_combinations(request: RunLaunchRequest) -> None:
    if (
        request.work_mode is not None
        and request.mode is not None
        and str(request.work_mode) != str(request.mode)
    ):
        raise ValueError("work_mode and mode cannot both be provided with different values")
    if request.runbook_preset is not None and request.runbook_config_id is not None:
        raise ValueError("runbook_preset and runbook_config_id cannot both be provided")
    if request.open_ended_question is not None and request.directed_effort_outcome is not None:
        raise ValueError("pass either open_ended_question or directed_effort_outcome, not both")
    if request.roles is not None and any(
        value is not None
        for value in (
            request.agent_profile,
            request.agent_model,
            request.agent_harness,
            request.agent_kind,
            request.agent_model_params,
        )
    ):
        raise ValueError("roles cannot be combined with top-level agent fields")
    if request.primary_objective_id is not None and (
        request.primary_parent_ref is not None or request.primary_parent is not None
    ):
        raise ValueError("primary_objective_id cannot be combined with primary parent refs")
    if request.primary_parent_ref is not None and request.primary_parent is not None:
        raise ValueError("primary_parent_ref and primary_parent cannot both be provided")
    if request.kickoff is not None and (
        request.initial_runtime_messages or request.kickoff_contract is not None
    ):
        raise ValueError(
            "kickoff cannot be combined with initial_runtime_messages or kickoff_contract"
        )
    if _requires_explicit_launch_axes(request):
        missing = []
        if request.host_kind is None:
            missing.append("host_kind")
        if request.work_mode is None and request.mode is None:
            missing.append("work_mode")
        if not request.providers:
            missing.append("providers")
        if missing:
            raise ValueError(
                "Provide intended_horizon_hours, runbook_preset, runbook_config_id, "
                "or explicit launch fields: " + ", ".join(missing)
            )


def _requires_explicit_launch_axes(request: RunLaunchRequest) -> bool:
    return (
        request.runbook_preset is None
        and request.runbook_config_id is None
        and request.intended_horizon_hours is None
    )


def _run_launch_init_kwargs(values: Mapping[str, object]) -> dict[str, object]:
    allowed = set(RunLaunchRequest.__dataclass_fields__)
    unknown = sorted(
        key for key, value in values.items() if key not in allowed and value is not None
    )
    if unknown:
        raise ValueError("unknown RunLaunchRequest fields: " + ", ".join(unknown))
    return {key: value for key, value in values.items() if key in allowed and value is not None}


def _output_kind_from_payload(payload: WireMapping) -> OutputKind:
    raw = (
        _optional_text(payload, "kind")
        or _optional_text(payload, "output_kind")
        or _optional_text(payload, "artifact_type")
    )
    if raw is None:
        raise ValueError("output kind is required")
    try:
        return OutputKind(raw)
    except ValueError as exc:
        raise ValueError(f"unknown output kind: {raw}") from exc


def _run_ref_from_payload(payload: WireMapping) -> RunRef | None:
    run_ref_payload = payload.get("run_ref")
    if isinstance(run_ref_payload, Mapping):
        return RunRef.from_wire(run_ref_payload)
    project_id = _optional_text(payload, "project_id")
    run_id = _optional_text(payload, "run_id")
    if project_id is None and run_id is None:
        return None
    if project_id is None or run_id is None:
        raise ValueError("project_id and run_id must both be present")
    return RunRef(project_id=project_id, run_id=run_id)


def _event_run_ref(
    event: RunRuntimeStreamEvent,
    explicit_run_ref: RunRef | None,
) -> RunRef | None:
    if explicit_run_ref is not None:
        if event.run_id is not None and event.run_id != explicit_run_ref.run_id:
            raise ValueError("stream event run_id does not match EventStreamRequest")
        return explicit_run_ref
    return None


def _is_terminal_state(state: str | None) -> bool:
    return state in {"completed", "succeeded", "done", "failed", "cancelled", "canceled"}


__all__ = [
    "AsyncEventStream",
    "ActorStateChangedEvent",
    "CheckpointCreatedEvent",
    "CommandRequest",
    "DiagnosticEvent",
    "Event",
    "EventKind",
    "EventStream",
    "EventStreamPhase",
    "EventStreamReplayPolicy",
    "EventStreamRequest",
    "EventStreamState",
    "HeartbeatEvent",
    "Interrupt",
    "InterruptCreatedEvent",
    "Kickoff",
    "LaunchPreflightRequest",
    "Output",
    "OutputKind",
    "OutputListRequest",
    "OutputPublishedEvent",
    "ReadRequest",
    "Ref",
    "ReportOutput",
    "Request",
    "RunEvent",
    "RunLaunchRequest",
    "RunLaunchResult",
    "RunReadRequest",
    "RunRef",
    "RunSnapshot",
    "RunStateChangedEvent",
    "RuntimeMessageEvent",
    "TaskEvent",
    "TaskStateChangedEvent",
    "UnknownEvent",
    "WireMapping",
    "WirePayload",
    "WorkProductPublishedEvent",
]
