"""Typed swarm launch and lifecycle contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

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
    ConfigurationVersionId,
    EffortId,
    OrganizationId,
    ProjectId,
    SwarmId,
    require_text,
)


FrozenJsonScalar: TypeAlias = str | int | float | bool | None
FrozenJsonValue: TypeAlias = (
    FrozenJsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
)


def _freeze_json(value: JsonValue) -> FrozenJsonValue:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: FrozenJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


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


class SwarmState(StrEnum):
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
            SwarmState.DONE,
            SwarmState.PARTIAL,
            SwarmState.FAILED,
            SwarmState.STOPPED,
            SwarmState.CANCELED,
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
class RoleBinding:
    """One orchestrator or reviewer role binding."""

    model: ActorModel
    params: Mapping[str, FrozenJsonValue] = MappingProxyType({})
    agent_harness: ActorHarness | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ActorModel):
            raise ValueError("role binding model must be ActorModel")
        if self.agent_harness is not None and not isinstance(
            self.agent_harness, ActorHarness
        ):
            raise ValueError("role binding agent_harness must be ActorHarness")
        frozen = _freeze_json(dict(self.params) if self.params is not None else {})
        if not isinstance(frozen, Mapping):
            raise ValueError("role binding params must be an object")
        object.__setattr__(self, "params", frozen)

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "model": self.model.value,
            "params": _thaw_json(self.params),
        }
        if self.agent_harness is not None:
            payload["agent_harness"] = self.agent_harness.value
        return payload


@dataclass(frozen=True, slots=True)
class WorkerRolePalette:
    """Worker model palette with optional subtype overrides."""

    permitted_models: tuple[ActorModel, ...]
    default_model: ActorModel
    default_params: Mapping[str, FrozenJsonValue] = MappingProxyType({})
    agent_harness: ActorHarness | None = None
    subtypes: Mapping[str, RoleBinding] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.permitted_models:
            raise ValueError("worker permitted_models must be non-empty")
        if any(not isinstance(model, ActorModel) for model in self.permitted_models):
            raise ValueError("worker permitted_models must be ActorModel values")
        if self.default_model not in self.permitted_models:
            raise ValueError("worker default_model must be in permitted_models")
        if self.agent_harness is not None and not isinstance(
            self.agent_harness, ActorHarness
        ):
            raise ValueError("worker agent_harness must be ActorHarness")
        params = _freeze_json(
            dict(self.default_params) if self.default_params is not None else {}
        )
        if not isinstance(params, Mapping):
            raise ValueError("worker default_params must be an object")
        object.__setattr__(self, "default_params", params)
        normalized_subtypes: dict[str, RoleBinding] = {}
        for name, binding in dict(self.subtypes or {}).items():
            if not isinstance(binding, RoleBinding):
                raise ValueError("worker subtype bindings must be RoleBinding")
            normalized_subtypes[require_text(name, field_name="worker subtype")] = (
                binding
            )
        object.__setattr__(self, "subtypes", MappingProxyType(normalized_subtypes))

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "permitted_models": [model.value for model in self.permitted_models],
            "default_model": self.default_model.value,
            "default_params": _thaw_json(self.default_params),
            "subtypes": {
                name: binding.to_wire() for name, binding in self.subtypes.items()
            },
        }
        if self.agent_harness is not None:
            payload["agent_harness"] = self.agent_harness.value
        return payload


@dataclass(frozen=True, slots=True)
class RoleBindings:
    """Closed orchestrator/reviewer/worker launch policy."""

    orchestrator: RoleBinding
    reviewer: RoleBinding
    worker: WorkerRolePalette
    reviewer_subtypes: Mapping[str, RoleBinding] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.orchestrator, RoleBinding):
            raise ValueError("roles.orchestrator must be RoleBinding")
        if not isinstance(self.reviewer, RoleBinding):
            raise ValueError("roles.reviewer must be RoleBinding")
        if not isinstance(self.worker, WorkerRolePalette):
            raise ValueError("roles.worker must be WorkerRolePalette")
        normalized_subtypes: dict[str, RoleBinding] = {}
        for name, binding in dict(self.reviewer_subtypes or {}).items():
            if not isinstance(binding, RoleBinding):
                raise ValueError("reviewer subtype bindings must be RoleBinding")
            normalized_subtypes[
                require_text(name, field_name="reviewer subtype")
            ] = binding
        object.__setattr__(self, "reviewer_subtypes", MappingProxyType(normalized_subtypes))

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "orchestrator": self.orchestrator.to_wire(),
            "reviewer": self.reviewer.to_wire(),
            "worker": self.worker.to_wire(),
        }
        if self.reviewer_subtypes:
            payload["reviewer_subtypes"] = {
                name: binding.to_wire()
                for name, binding in self.reviewer_subtypes.items()
            }
        return payload


@dataclass(frozen=True, slots=True)
class PlatformResolvedExecutionTarget:
    """Stable placement target that delegates substrate choice to the platform."""

    kind: str = "platform_resolved"

    def __post_init__(self) -> None:
        if self.kind != "platform_resolved":
            raise ValueError("stable execution_target.kind must be platform_resolved")

    def to_wire(self) -> JsonObject:
        return {"kind": "platform_resolved"}


@dataclass(frozen=True, slots=True)
class ActorImageBinding:
    """Admitted runtime image-release binding for one actor role."""

    release_id: str
    reason: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        require_text(self.release_id, field_name="actor image release_id")
        for name, value in (("reason", self.reason), ("notes", self.notes)):
            if value is not None:
                require_text(value, field_name=name)

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"release_id": self.release_id}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True, slots=True)
class KickoffArtifact:
    """Workspace-staged kickoff contract reference.

    Stable launches stage kickoff bodies as project workspace artifacts and
    reference them here. Unclosed kickoff dictionaries remain advanced-only.
    """

    path: str
    contract_kind: str = "staged_smr_kickoff_contract"

    def __post_init__(self) -> None:
        require_text(self.path, field_name="kickoff artifact path")
        require_text(self.contract_kind, field_name="kickoff contract_kind")

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": 1,
            "contract_kind": self.contract_kind,
            "kickoff_contract_file": self.path,
        }


@dataclass(frozen=True, slots=True)
class SwarmSpec:
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
    roles: RoleBindings | None = None
    providers: tuple[ProviderBinding, ...] = ()
    limit: ResourceLimit | None = None
    required_capabilities: tuple[str, ...] = ()
    kickoff_messages: tuple[KickoffMessage, ...] = ()
    kickoff_artifact: KickoffArtifact | None = None
    execution_target: PlatformResolvedExecutionTarget | None = None
    actor_image_overrides: Mapping[str, ActorImageBinding] = MappingProxyType({})
    local_execution: LocalExecution | None = None
    execution_profile: ExecutionProfile | None = None
    environment_name: str | None = None
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
        if self.roles is not None and not isinstance(self.roles, RoleBindings):
            raise ValueError("roles must be RoleBindings")
        if self.execution_target is not None and not isinstance(
            self.execution_target,
            PlatformResolvedExecutionTarget,
        ):
            raise ValueError(
                "stable execution_target must be PlatformResolvedExecutionTarget"
            )
        if self.kickoff_artifact is not None and not isinstance(
            self.kickoff_artifact,
            KickoffArtifact,
        ):
            raise ValueError("kickoff_artifact must be KickoffArtifact")
        if self.actor_model_assignments and self.roles is not None:
            raise ValueError("roles cannot be combined with actor_model_assignments")
        if self.execution_target is not None and any(
            value is not None
            for value in (
                self.host_kind,
                self.local_execution,
                self.execution_profile,
                self.worker_pool_id,
            )
        ):
            raise ValueError(
                "execution_target cannot be combined with legacy placement fields"
            )
        normalized_images: dict[str, ActorImageBinding] = {}
        for role, binding in dict(self.actor_image_overrides or {}).items():
            if not isinstance(binding, ActorImageBinding):
                raise ValueError(
                    "actor_image_overrides values must be ActorImageBinding"
                )
            normalized_images[require_text(role, field_name="actor image role")] = (
                binding
            )
        object.__setattr__(
            self, "actor_image_overrides", MappingProxyType(normalized_images)
        )
        capabilities = tuple(
            require_text(item, field_name="required_capabilities")
            for item in self.required_capabilities
        )
        object.__setattr__(self, "required_capabilities", capabilities)
        for name, value in (
            ("runbook_preset", self.runbook_preset),
            ("environment_name", self.environment_name),
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
        )
        for name, value in scalar_values:
            if value is not None:
                payload[name] = value
        if self.idempotency_key is not None:
            payload["idempotency_key_run_create"] = self.idempotency_key
        if self.environment_name is not None:
            payload["environment"] = {
                "schema_version": "2026-05-14-environment-v1",
                "name": self.environment_name,
            }
        if self.roles is not None:
            payload["roles"] = self.roles.to_wire()
        if self.actor_model_assignments:
            payload["actor_model_overrides"] = [
                assignment.to_wire() for assignment in self.actor_model_assignments
            ]
        if self.providers:
            payload["providers"] = [provider.to_wire() for provider in self.providers]
        if self.limit is not None:
            payload["limit"] = self.limit.to_wire()
        if self.required_capabilities:
            payload["required_capabilities"] = list(self.required_capabilities)
        if self.kickoff_messages:
            payload["initial_runtime_messages"] = [
                message.to_wire() for message in self.kickoff_messages
            ]
        if self.kickoff_artifact is not None:
            payload["kickoff_contract"] = self.kickoff_artifact.to_wire()
        if self.execution_target is not None:
            payload["execution_target"] = self.execution_target.to_wire()
        if self.actor_image_overrides:
            payload["actor_image_overrides"] = {
                role: binding.to_wire()
                for role, binding in self.actor_image_overrides.items()
            }
        if self.local_execution is not None:
            payload["local_execution"] = self.local_execution.to_wire()
        if self.execution_profile is not None:
            payload["execution_profile"] = self.execution_profile.to_wire()
        return payload


@dataclass(frozen=True, slots=True)
class Swarm:
    swarm_id: SwarmId
    project_id: ProjectId
    organization_id: OrganizationId
    state: SwarmState
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
    def from_wire(cls, value: JsonValue) -> Swarm:
        payload = object_value(value, operation_id="swarm")
        work_mode = optional_text(payload, "work_mode")
        effort_id = optional_text(payload, "effort_id")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            organization_id=OrganizationId(required_text(payload, "org_id")),
            state=SwarmState(required_text(payload, "public_state")),
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
class ResolvedSwarmConfiguration:
    """Immutable, replayable configuration snapshot resolved for one swarm."""

    swarm_id: SwarmId
    project_id: ProjectId
    config_version_id: ConfigurationVersionId
    snapshot_sha256: str
    snapshot: Mapping[str, FrozenJsonValue]
    schema_version: str = "synth.research.resolved-swarm-configuration.v1"

    @classmethod
    def from_wire(cls, value: JsonValue) -> ResolvedSwarmConfiguration:
        payload = object_value(value, operation_id="retrieve_swarm_configuration")
        schema_version = required_text(payload, "schema_version")
        if schema_version != "synth.research.resolved-swarm-configuration.v1":
            raise ValueError(
                "unsupported resolved swarm configuration schema "
                f"{schema_version!r}"
            )
        digest = required_text(payload, "snapshot_sha256").lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("snapshot_sha256 must be a 64-character hexadecimal digest")
        snapshot = object_value(
            payload.get("snapshot"),
            operation_id="resolved swarm configuration snapshot",
        )
        frozen_snapshot = _freeze_json(snapshot)
        if not isinstance(frozen_snapshot, Mapping):
            raise ValueError("resolved swarm configuration snapshot must be an object")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            config_version_id=ConfigurationVersionId(
                required_text(payload, "config_version_id")
            ),
            snapshot_sha256=digest,
            snapshot=frozen_snapshot,
            schema_version=schema_version,
        )

    def to_wire(self) -> JsonObject:
        """Return a JSON-serializable copy suitable for CLI and MCP adapters."""
        snapshot = _thaw_json(self.snapshot)
        if not isinstance(snapshot, dict):
            raise ValueError("resolved swarm configuration snapshot must be an object")
        return {
            "schema_version": self.schema_version,
            "run_id": self.swarm_id,
            "project_id": self.project_id,
            "config_version_id": self.config_version_id,
            "snapshot_sha256": self.snapshot_sha256,
            "snapshot": snapshot,
        }


@dataclass(frozen=True, slots=True)
class SwarmPreflight:
    project_id: ProjectId
    clear_to_trigger: bool
    blockers: tuple[str, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmPreflight:
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
class BranchSpec:
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
class BranchResult:
    accepted: bool
    parent_swarm_id: SwarmId
    child_swarm_id: SwarmId
    source_checkpoint_id: str
    created_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> BranchResult:
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


ResearchSwarm = Swarm
ResearchSwarmBranchRequest = BranchSpec
ResearchSwarmBranchResult = BranchResult
ResearchSwarmLaunchRequest = SwarmSpec
ResearchSwarmPreflight = SwarmPreflight
ResearchSwarmState = SwarmState


__all__ = [
    "ActorHarness",
    "ActorImageBinding",
    "ActorModel",
    "ActorModelAssignment",
    "ActorSubtype",
    "ActorType",
    "BranchMode",
    "HostKind",
    "EnvironmentVariable",
    "ExecutionCapability",
    "ExecutionProfile",
    "KickoffArtifact",
    "KickoffMessage",
    "KickoffMessageMode",
    "LocalExecution",
    "PlatformResolvedExecutionTarget",
    "ProviderBinding",
    "ResolvedSwarmConfiguration",
    "RoleBinding",
    "RoleBindings",
    "Swarm",
    "BranchSpec",
    "BranchResult",
    "SwarmSpec",
    "SwarmPreflight",
    "SwarmState",
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
    "WorkerRolePalette",
]
