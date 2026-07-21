"""Typed swarm launch and lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    object_value,
    optional_bool,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    EffortId,
    OrganizationId,
    ProjectId,
    SwarmId,
    require_text,
)


class ActorHarness(StrEnum):
    CODEX = "codex"
    CURSOR = "cursor"
    OPENCODE_SDK = "opencode_sdk"


class ActorModel(StrEnum):
    GPT_5_CODEX = "gpt-5-codex"
    GPT_5_3_CODEX = "gpt-5.3-codex"
    GPT_5_3_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_5 = "gpt-5.5"
    GPT_5_6_LUNA = "gpt-5.6-luna"
    NEMOTRON_SUPER = "nemotron-super"
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
    DEEPSEEK_V4_FLASH_DIRECT = "deepseek/deepseek-v4-flash-direct"
    DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"
    DEEPSEEK_V4_PRO_DIRECT = "deepseek/deepseek-v4-pro-direct"
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    DEEPSEEK_REASONER = "deepseek/deepseek-reasoner"
    CURSOR_COMPOSER_2_5 = "cursor/composer-2.5"
    CURSOR_GPT_5 = "cursor/gpt-5"
    CURSOR_SONNET_4 = "cursor/sonnet-4"
    GROK_4_3 = "x-ai/grok-4.3"
    GROK_BUILD = "x-ai/grok-build"
    KIMI_K2_6 = "moonshotai/kimi-k2.6"
    GLM_5_2 = "baseten/zai-org/GLM-5.2"
    MODAL_GLM_5_2_FP8 = "modal/zai-org/GLM-5.2-FP8"


class ActorType(StrEnum):
    ORCHESTRATOR = "orchestrator"
    REVIEWER = "reviewer"
    WORKER = "worker"


class ActorSubtype(StrEnum):
    MAIN = "main"
    ENGINEER = "engineer"
    RESEARCHER = "researcher"
    ARTIFACT_BUILDER = "artifact_builder"
    ARTIFACT_REVIEWER = "artifact_reviewer"
    TASK_COMPLETION = "task_completion"
    RUN_COMPLETION = "run_completion"
    SAFETY = "safety"
    OBJECTIVE = "objective"
    SERAPH = "seraph"
    GARDENER = "gardener"


class WorkMode(StrEnum):
    GENERAL = "general"
    OPEN_ENDED_DISCOVERY = "open_ended_discovery"
    DIRECTED_EFFORT = "directed_effort"


class Runbook(StrEnum):
    LITE = "lite"
    STANDARD = "standard"
    HEAVY = "heavy"
    OVERNIGHT = "overnight"
    CONTINUOUS = "continuous"


class HostKind(StrEnum):
    DOCKER = "docker"
    DAYTONA = "daytona"


class ResourceProvider(StrEnum):
    OPENROUTER = "openrouter"
    TINKER = "tinker"
    SYNTH_AI = "synth_ai"
    CURSOR = "cursor"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    MODAL = "modal"
    OPENAI_CHATGPT = "openai_chatgpt"
    BASETEN = "baseten"


class KickoffMessageMode(StrEnum):
    QUEUE = "queue"
    INTERRUPT = "interrupt"
    STEER = "steer"


class ResearchSwarmState(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    BLOCKED = "blocked"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    DONE = "done"
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELED = "canceled"
    ACTIVE = "active"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ResearchSwarmState.DONE,
            ResearchSwarmState.PARTIAL,
            ResearchSwarmState.FAILED,
            ResearchSwarmState.STOPPED,
            ResearchSwarmState.CANCELED,
        }


class BranchMode(StrEnum):
    EXACT = "exact"
    WITH_MESSAGE = "with_message"


@dataclass(frozen=True, slots=True)
class ActorModelAssignment:
    actor_type: ActorType
    actor_subtype: ActorSubtype
    model: ActorModel
    harness: ActorHarness | None = None

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "actor_type": self.actor_type.value,
            "actor_subtype": self.actor_subtype.value,
            "agent_model": self.model.value,
        }
        if self.harness is not None:
            payload["agent_harness"] = self.harness.value
        return payload


@dataclass(frozen=True, slots=True)
class ResourceLimit:
    max_spend_usd: float | None = None
    max_tokens: int | None = None
    max_wallclock_seconds: int | None = None
    max_gpu_hours: float | None = None
    max_concurrent_actors: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_spend_usd", self.max_spend_usd),
            ("max_tokens", self.max_tokens),
            ("max_wallclock_seconds", self.max_wallclock_seconds),
            ("max_gpu_hours", self.max_gpu_hours),
            ("max_concurrent_actors", self.max_concurrent_actors),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_wire(self) -> JsonObject:
        return {
            name: value
            for name, value in (
                ("max_spend_usd", self.max_spend_usd),
                ("max_tokens", self.max_tokens),
                ("max_wallclock_seconds", self.max_wallclock_seconds),
                ("max_gpu_hours", self.max_gpu_hours),
                ("max_concurrent_actors", self.max_concurrent_actors),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    provider: ResourceProvider
    limit: ResourceLimit | None = None

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"provider": self.provider.value}
        if self.limit is not None:
            payload["limit"] = self.limit.to_wire()
        return payload


@dataclass(frozen=True, slots=True)
class EnvironmentVariable:
    name: str
    value: str

    def __post_init__(self) -> None:
        require_text(self.name, field_name="environment variable name")


@dataclass(frozen=True, slots=True)
class LocalExecution:
    slot_id: str
    runtime_id: str
    dispatch_pool: str
    host_kind: HostKind
    environment: tuple[EnvironmentVariable, ...] = ()
    unset_environment: tuple[str, ...] = ()
    requires_hosted_capacity: bool = False

    def __post_init__(self) -> None:
        require_text(self.slot_id, field_name="slot_id")
        require_text(self.runtime_id, field_name="runtime_id")
        require_text(self.dispatch_pool, field_name="dispatch_pool")

    def to_wire(self) -> JsonObject:
        return {
            "slot_id": self.slot_id,
            "runtime_id": self.runtime_id,
            "dispatch_pool": self.dispatch_pool,
            "host_kind": self.host_kind.value,
            "env": {variable.name: variable.value for variable in self.environment},
            "unset_env": list(self.unset_environment),
            "requires_hosted_capacity": self.requires_hosted_capacity,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCapability:
    name: str
    enabled: bool

    def __post_init__(self) -> None:
        require_text(self.name, field_name="execution capability name")


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    schema_version: str
    profile_id: str
    product: str
    host_kind: HostKind
    required_runtime_kind: str
    capabilities: tuple[ExecutionCapability, ...] = ()
    docker_image: str | None = None
    daytona_snapshot: str | None = None
    required_product: str | None = None
    required_repo: str | None = None
    local_source_kind: str | None = None
    source_binding_kind: str = "tool_repo"

    def __post_init__(self) -> None:
        for name, value in (
            ("schema_version", self.schema_version),
            ("profile_id", self.profile_id),
            ("product", self.product),
            ("required_runtime_kind", self.required_runtime_kind),
            ("source_binding_kind", self.source_binding_kind),
        ):
            require_text(value, field_name=name)

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "product": self.product,
            "host_kind": self.host_kind.value,
            "required_runtime_kind": self.required_runtime_kind,
            "source_binding_kind": self.source_binding_kind,
            "capabilities": {
                capability.name: capability.enabled for capability in self.capabilities
            },
        }
        for name, value in (
            ("docker_image", self.docker_image),
            ("daytona_snapshot", self.daytona_snapshot),
            ("required_product", self.required_product),
            ("required_repo", self.required_repo),
            ("local_source_kind", self.local_source_kind),
        ):
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True, slots=True)
class KickoffMessage:
    body: str
    mode: KickoffMessageMode = KickoffMessageMode.QUEUE
    sender: str | None = None
    topic: str | None = None
    action: str | None = None
    causation_id: str | None = None
    payload: JsonValue = None

    def __post_init__(self) -> None:
        require_text(self.body, field_name="kickoff message body")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {"body": self.body, "mode": self.mode.value}
        for name, field_value in (
            ("sender", self.sender),
            ("topic", self.topic),
            ("action", self.action),
            ("causation_id", self.causation_id),
        ):
            if field_value is not None:
                value[name] = field_value
        if self.payload is not None:
            value["payload"] = self.payload
        return value


@dataclass(frozen=True, slots=True)
class ResearchSwarmLaunchRequest:
    objective: str
    work_mode: WorkMode | None = None
    runbook: Runbook | None = None
    runbook_preset: str | None = None
    intended_horizon_hours: int | None = None
    timebox_seconds: int | None = None
    host_kind: HostKind | None = None
    agent_model: ActorModel | None = None
    agent_harness: ActorHarness | None = None
    actor_model_assignments: tuple[ActorModelAssignment, ...] = ()
    providers: tuple[ProviderBinding, ...] = ()
    limit: ResourceLimit | None = None
    kickoff_messages: tuple[KickoffMessage, ...] = ()
    local_execution: LocalExecution | None = None
    execution_profile: ExecutionProfile | None = None
    worker_pool_id: str | None = None
    dev_environment_id: str | None = None
    effort_id: EffortId | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        require_text(self.objective, field_name="objective")
        if self.intended_horizon_hours not in {None, 1, 4, 8, 24, 168}:
            raise ValueError("intended_horizon_hours must be one of 1, 4, 8, 24, or 168")
        if self.timebox_seconds is not None and self.timebox_seconds <= 0:
            raise ValueError("timebox_seconds must be positive")
        for name, value in (
            ("runbook_preset", self.runbook_preset),
            ("worker_pool_id", self.worker_pool_id),
            ("dev_environment_id", self.dev_environment_id),
            ("effort_id", self.effort_id),
            ("idempotency_key", self.idempotency_key),
        ):
            if value is not None:
                require_text(value, field_name=name)

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"objective": self.objective}
        enum_values = (
            ("work_mode", self.work_mode),
            ("runbook", self.runbook),
            ("host_kind", self.host_kind),
            ("agent_model", self.agent_model),
            ("agent_harness", self.agent_harness),
        )
        for name, value in enum_values:
            if value is not None:
                payload[name] = value.value
        scalar_values = (
            ("runbook_preset", self.runbook_preset),
            ("intended_horizon_hours", self.intended_horizon_hours),
            ("timebox_seconds", self.timebox_seconds),
            ("worker_pool_id", self.worker_pool_id),
            ("dev_environment_id", self.dev_environment_id),
            ("effort_id", self.effort_id),
            ("idempotency_key", self.idempotency_key),
        )
        for name, value in scalar_values:
            if value is not None:
                payload[name] = value
        if self.actor_model_assignments:
            payload["actor_model_overrides"] = [
                assignment.to_wire() for assignment in self.actor_model_assignments
            ]
        if self.providers:
            payload["providers"] = [provider.to_wire() for provider in self.providers]
        if self.limit is not None:
            payload["limit"] = self.limit.to_wire()
        if self.kickoff_messages:
            payload["initial_runtime_messages"] = [
                message.to_wire() for message in self.kickoff_messages
            ]
        if self.local_execution is not None:
            payload["local_execution"] = self.local_execution.to_wire()
        if self.execution_profile is not None:
            payload["execution_profile"] = self.execution_profile.to_wire()
        return payload


@dataclass(frozen=True, slots=True)
class ResearchSwarm:
    swarm_id: SwarmId
    project_id: ProjectId
    organization_id: OrganizationId
    state: ResearchSwarmState
    runbook: str
    trigger: str
    created_at: datetime
    updated_at: datetime
    work_mode: WorkMode | None = None
    effort_id: EffortId | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    terminal_outcome: str | None = None
    work_completed: bool = False

    @classmethod
    def from_wire(cls, value: JsonValue) -> ResearchSwarm:
        payload = object_value(value, operation_id="swarm")
        work_mode = optional_text(payload, "work_mode")
        effort_id = optional_text(payload, "effort_id")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            organization_id=OrganizationId(required_text(payload, "org_id")),
            state=ResearchSwarmState(required_text(payload, "public_state")),
            runbook=required_text(payload, "runbook"),
            trigger=required_text(payload, "trigger"),
            created_at=required_datetime(payload, "created_at"),
            updated_at=required_datetime(payload, "updated_at"),
            work_mode=WorkMode(work_mode) if work_mode is not None else None,
            effort_id=EffortId(effort_id) if effort_id is not None else None,
            started_at=_optional_datetime(payload, "started_at"),
            finished_at=_optional_datetime(payload, "finished_at"),
            terminal_outcome=optional_text(payload, "terminal_outcome"),
            work_completed=optional_bool(payload, "work_completed"),
        )


@dataclass(frozen=True, slots=True)
class ResearchSwarmPreflight:
    project_id: ProjectId
    clear_to_trigger: bool
    blockers: tuple[str, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ResearchSwarmPreflight:
        payload = object_value(value, operation_id="swarm preflight")
        raw_blockers = payload.get("blockers", [])
        if not isinstance(raw_blockers, list):
            raise ValueError("preflight blockers must be an array")
        blockers: list[str] = []
        for blocker in raw_blockers:
            if isinstance(blocker, str):
                blockers.append(blocker)
            elif isinstance(blocker, dict):
                message = blocker.get("message") or blocker.get("detail") or blocker.get("code")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("preflight blocker must include message, detail, or code")
                blockers.append(message.strip())
            else:
                raise ValueError("preflight blocker must be a string or object")
        return cls(
            ProjectId(required_text(payload, "project_id")),
            required_bool(payload, "clear_to_trigger"),
            tuple(blockers),
        )


@dataclass(frozen=True, slots=True)
class ResearchSwarmBranchRequest:
    checkpoint_id: str | None = None
    checkpoint_record_id: str | None = None
    checkpoint_uri: str | None = None
    source_node_id: str | None = None
    mode: BranchMode = BranchMode.EXACT
    message: str | None = None
    reason: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        if not any((self.checkpoint_id, self.checkpoint_record_id, self.checkpoint_uri)):
            raise ValueError("branch request requires one checkpoint reference")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"mode": self.mode.value}
        for name, value in (
            ("checkpoint_id", self.checkpoint_id),
            ("checkpoint_record_id", self.checkpoint_record_id),
            ("checkpoint_uri", self.checkpoint_uri),
            ("source_node_id", self.source_node_id),
            ("message", self.message),
            ("reason", self.reason),
            ("title", self.title),
        ):
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True, slots=True)
class ResearchSwarmBranchResult:
    accepted: bool
    parent_swarm_id: SwarmId
    child_swarm_id: SwarmId
    source_checkpoint_id: str
    created_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> ResearchSwarmBranchResult:
        payload = object_value(value, operation_id="branch swarm")
        return cls(
            required_bool(payload, "accepted"),
            SwarmId(required_text(payload, "parent_run_id")),
            SwarmId(required_text(payload, "child_run_id")),
            required_text(payload, "source_checkpoint_id"),
            required_datetime(payload, "created_at"),
        )


def _optional_datetime(payload: JsonObject, name: str) -> datetime | None:
    value = optional_text(payload, name)
    if value is None:
        return None
    probe: JsonObject = {name: value}
    return required_datetime(probe, name)


__all__ = [
    "ActorHarness",
    "ActorModel",
    "ActorModelAssignment",
    "ActorSubtype",
    "ActorType",
    "BranchMode",
    "HostKind",
    "EnvironmentVariable",
    "ExecutionCapability",
    "ExecutionProfile",
    "KickoffMessage",
    "KickoffMessageMode",
    "LocalExecution",
    "ProviderBinding",
    "ResearchSwarm",
    "ResearchSwarmBranchRequest",
    "ResearchSwarmBranchResult",
    "ResearchSwarmLaunchRequest",
    "ResearchSwarmPreflight",
    "ResearchSwarmState",
    "ResourceLimit",
    "ResourceProvider",
    "Runbook",
    "WorkMode",
]
