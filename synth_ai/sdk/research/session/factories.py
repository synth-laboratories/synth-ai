"""Factory and Effort SDK namespaces."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, cast

from synth_ai.sdk.research.contracts.factories import EffortRecurrence
from synth_ai.sdk.research.contracts.factory_lenses import (
    FactoryBestResults,
    FactoryEvaluationLens,
    FactoryLensSpec,
    FactoryPreferenceEvent,
    FactoryPreferenceRequest,
    FactoryResultEvaluation,
    FactoryResultEvaluationRequest,
)
from synth_ai.sdk.research.contracts.factory_operations import (
    AuthorizationPolicy,
    Effort,
    EffortCreateRequest,
    EffortFromRunsRequest,
    EffortPatchRequest,
    EffortStatus,
    EffortType,
    ExperimentBundle,
    ExperimentComparison,
    ExperimentHistory,
    Factory,
    FactoryActorOutput,
    FactoryActorOutputCreateRequest,
    FactoryActorOutputKind,
    FactoryActorOutputPatchRequest,
    FactoryActorOutputStatus,
    FactoryActorRole,
    FactoryCandidate,
    FactoryCandidateGradingRequest,
    FactoryCandidateGradingStatus,
    FactoryChampionDecision,
    FactoryChampionEvent,
    FactoryChampionRollbackRequest,
    FactoryChampionSelectRequest,
    FactoryCreateRequest,
    FactoryIdea,
    FactoryIdeaCreateRequest,
    FactoryIdeaPatchRequest,
    FactoryIdeaSource,
    FactoryIdeaStatus,
    FactoryPatchRequest,
    FactoryProjectLink,
    FactoryProjectLinkRequest,
    FactoryProjectPatchRequest,
    FactoryProjectRole,
    FactoryProjectStatus,
    FactoryResult,
    FactoryResultEvaluateRequest,
    FactoryResultRestoreRequest,
    FactoryResultSelectionDecision,
    FactoryResultSelectionEvent,
    FactoryResultSelectRequest,
    FactoryStatus,
    FactoryTransitionRequest,
    FactoryTransitionResponse,
    FactoryWakeDueRequest,
    FactoryWakeDueResult,
    FactoryWorkspace,
    GraduationProposal,
    RecurrencePolicy,
)
from synth_ai.sdk.research.contracts.types import SmrRunnableProjectRequest
from synth_ai.sdk.research.session._base import _ClientNamespace


def _enum_query_value(value: object | None) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _wire_mapping_payload(value: object, *, field_name: str) -> dict[str, Any]:
    to_wire = getattr(value, "to_wire", None)
    wire_value = to_wire() if callable(to_wire) else value
    if not isinstance(wire_value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or support to_wire()")
    return dict(cast(Mapping[str, Any], wire_value))


def _standup_mapping(value: object, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _standup_required_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _standup_required_string(parent: Mapping[str, Any], key: str) -> str:
    value = str(parent.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _standup_optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _factory_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    factory = _standup_required_mapping(plan, "factory")
    return {
        "name": _standup_required_string(factory, "name"),
        "kind": str(factory.get("kind") or "customer"),
        "description": _standup_optional_string(factory.get("description")),
        "budget_policy": _standup_mapping(
            factory.get("budget_policy"),
            field="factory.budget_policy",
        ),
        "cap_policy": _standup_mapping(factory.get("cap_policy"), field="factory.cap_policy"),
        "publication_policy": _standup_mapping(
            factory.get("publication_policy"),
            field="factory.publication_policy",
        ),
        "authorization_policy": _standup_mapping(
            factory.get("authorization_policy"),
            field="factory.authorization_policy",
        ),
        "metadata": _standup_mapping(factory.get("metadata"), field="factory.metadata"),
    }


def _project_link_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    project = _standup_mapping(plan.get("project"), field="project")
    return {
        "role": str(project.get("role") or "canonical"),
        "status": str(project.get("status") or "active"),
        "display_name": _standup_optional_string(project.get("display_name")),
        "description": _standup_optional_string(project.get("description")),
        "workspace_policy": _standup_mapping(
            project.get("workspace_policy"),
            field="project.workspace_policy",
        ),
        "resource_bindings": _standup_mapping(
            project.get("resource_bindings"),
            field="project.resource_bindings",
        ),
        "feed_health": _standup_mapping(project.get("feed_health"), field="project.feed_health"),
        "default_launch_profile": _standup_mapping(
            project.get("default_launch_profile"),
            field="project.default_launch_profile",
        ),
        "metadata": _standup_mapping(project.get("metadata"), field="project.metadata"),
    }


def _effort_payloads(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    efforts = plan.get("efforts")
    if not isinstance(efforts, list) or not efforts:
        raise ValueError("efforts must be a non-empty JSON array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(efforts):
        if not isinstance(item, Mapping):
            raise ValueError(f"efforts[{index}] must be a JSON object")
        result.append({str(item_key): item_value for item_key, item_value in item.items()})
    return result


def _effort_kwargs(effort: Mapping[str, Any], *, default_project_id: str) -> dict[str, Any]:
    return {
        "name": _standup_required_string(effort, "name"),
        "project_id": _standup_optional_string(effort.get("project_id")) or default_project_id,
        "hypothesis_or_topic": _standup_optional_string(
            effort.get("hypothesis_or_topic") or effort.get("topic")
        ),
        "effort_type": str(effort.get("effort_type") or effort.get("type") or "research"),
        "status": str(effort.get("status") or "active"),
        "recurrence_policy": _standup_mapping(
            effort.get("recurrence_policy"),
            field="effort.recurrence_policy",
        ),
        "next_wake_at": _standup_optional_string(effort.get("next_wake_at")),
        "latest_run_id": _standup_optional_string(effort.get("latest_run_id")),
        "latest_report_id": _standup_optional_string(effort.get("latest_report_id")),
        "latest_work_product_id": _standup_optional_string(effort.get("latest_work_product_id")),
        "decision_needed": bool(effort.get("decision_needed") or False),
        "decision_note": _standup_optional_string(effort.get("decision_note")),
        "budget_policy": _standup_mapping(
            effort.get("budget_policy"),
            field="effort.budget_policy",
        ),
        "publication_policy": _standup_mapping(
            effort.get("publication_policy"),
            field="effort.publication_policy",
        ),
        "authorization_policy": _standup_mapping(
            effort.get("authorization_policy"),
            field="effort.authorization_policy",
        ),
        "actor_notes": _standup_mapping(effort.get("actor_notes"), field="effort.actor_notes"),
        "metadata": _standup_mapping(effort.get("metadata"), field="effort.metadata"),
    }


def _wake_due_preview_kwargs(plan: Mapping[str, Any]) -> dict[str, Any]:
    wake_due = _standup_mapping(plan.get("wake_due"), field="wake_due")
    return {
        "launch_request": _standup_mapping(
            wake_due.get("launch_request"),
            field="wake_due.launch_request",
        )
        or None,
        "effort_ids": tuple(str(value) for value in wake_due.get("effort_ids") or []),
        "limit": int(wake_due.get("limit") or 10),
        "allow_overlap": bool(wake_due.get("allow_overlap") or False),
        "continue_on_error": bool(wake_due.get("continue_on_error", True)),
        "dry_run": True,
    }


def _project_id_from_response(payload: Mapping[str, Any]) -> str:
    project_id = _standup_optional_string(payload.get("project_id") or payload.get("id"))
    if project_id:
        return project_id
    project = payload.get("project")
    if isinstance(project, Mapping):
        project_id = _standup_optional_string(project.get("project_id") or project.get("id"))
        if project_id:
            return project_id
    raise ValueError("create_project response did not include project_id")


@dataclass(frozen=True, slots=True)
class FactoryStandupPlan:
    """The resolved request payloads a stand-up would send, before it sends them."""

    factory: dict[str, Any]
    project_link: dict[str, Any]
    efforts: list[dict[str, Any]]
    project_id: str | None = None
    create_project: dict[str, Any] | None = None
    wake_due: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FactoryStandupResult:
    """What a stand-up created, with the plan it created it from."""

    plan: FactoryStandupPlan
    factory: Factory
    project_id: str
    link: FactoryProjectLink
    efforts: list[Effort]
    status: FactoryStatus
    created_project: dict[str, Any] | None = None
    wake_due_preview: FactoryWakeDueResult | None = None
    wake_due_result: FactoryWakeDueResult | None = None

    @property
    def factory_id(self) -> str:
        return self.factory.factory_id

    @property
    def effort_ids(self) -> list[str]:
        return [effort.effort_id for effort in self.efforts]


def _effort_recurrence_payload(
    *,
    recurrence: EffortRecurrence | None,
    recurrence_policy: (
        EffortRecurrence | RecurrencePolicy | Mapping[str, Any] | dict[str, Any] | None
    ),
) -> dict[str, Any]:
    if recurrence is not None and recurrence_policy is not None:
        raise ValueError("recurrence cannot be combined with recurrence_policy")
    selected = recurrence if recurrence is not None else recurrence_policy
    if selected is None:
        return {}
    return _wire_mapping_payload(selected, field_name="recurrence")


class FactoriesAPI(_ClientNamespace):
    def create(
        self,
        request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.create_factory(request))

    def list(self, *, include_archived: bool = False) -> List[Factory]:
        return [
            Factory.from_wire(item)
            for item in self._client.list_factories(include_archived=include_archived)
        ]

    def get(self, factory_id: str) -> Factory:
        return Factory.from_wire(self._client.get_factory(factory_id))

    def patch(
        self,
        factory_id: str,
        request: FactoryPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.patch_factory(factory_id, request))

    def list_candidates(
        self,
        factory_id: str,
        *,
        grading_status: FactoryCandidateGradingStatus | str | None = None,
        effort_id: str | None = None,
        limit: int = 200,
    ) -> List[FactoryCandidate]:
        return [
            FactoryCandidate.from_wire(item)
            for item in self._client.list_factory_candidates(
                factory_id,
                grading_status=_enum_query_value(grading_status),
                effort_id=effort_id,
                limit=limit,
            )
        ]

    def record_candidate_grading(
        self,
        factory_id: str,
        candidate_id: str,
        request: FactoryCandidateGradingRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryCandidate:
        return FactoryCandidate.from_wire(
            self._client.record_factory_candidate_grading(
                factory_id,
                candidate_id,
                request,
            )
        )

    def select_champion(
        self,
        factory_id: str,
        request: FactoryChampionSelectRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryChampionDecision:
        return FactoryChampionDecision.from_wire(
            self._client.select_factory_champion(factory_id, request)
        )

    def rollback_champion(
        self,
        factory_id: str,
        request: FactoryChampionRollbackRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryChampionDecision:
        return FactoryChampionDecision.from_wire(
            self._client.rollback_factory_champion(factory_id, request)
        )

    def list_champion_events(
        self,
        factory_id: str,
        *,
        limit: int = 100,
    ) -> List[FactoryChampionEvent]:
        return [
            FactoryChampionEvent.from_wire(item)
            for item in self._client.list_factory_champion_events(
                factory_id,
                limit=limit,
            )
        ]

    @property
    def results(self) -> FactoryResultsAPI:
        """Public Result surface: ``research.factories.results.list(factory_id)``."""

        return FactoryResultsAPI(self._client)

    @property
    def lenses(self) -> FactoryLensesAPI:
        """Optional optimization lens: ``research.factories.lenses``.

        Absent for the majority of Factories, which optimize nothing.
        """

        return FactoryLensesAPI(self._client)

    def link_project(
        self,
        factory_id: str,
        project_id: str,
        *,
        role: FactoryProjectRole | str = FactoryProjectRole.CANONICAL,
        status: FactoryProjectStatus | str = FactoryProjectStatus.ACTIVE,
        display_name: str | None = None,
        description: str | None = None,
        workspace_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        resource_bindings: Mapping[str, Any] | dict[str, Any] | None = None,
        feed_health: Mapping[str, Any] | dict[str, Any] | None = None,
        default_launch_profile: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryProjectLink:
        return FactoryProjectLink.from_wire(
            self._client.link_factory_project(
                factory_id,
                FactoryProjectLinkRequest(
                    project_id=project_id,
                    role=role,
                    status=status,
                    display_name=display_name,
                    description=description,
                    workspace_policy=dict(workspace_policy or {}),
                    resource_bindings=dict(resource_bindings or {}),
                    feed_health=dict(feed_health or {}),
                    default_launch_profile=dict(default_launch_profile or {}),
                    metadata=dict(metadata or {}),
                ),
            )
        )

    def link_workspace_project(
        self,
        factory_id: str,
        project_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        workspace_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        resource_bindings: Mapping[str, Any] | dict[str, Any] | None = None,
        feed_health: Mapping[str, Any] | dict[str, Any] | None = None,
        default_launch_profile: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryProjectLink:
        """Link or replace the canonical workspace Project for a Factory."""

        return self.link_project(
            factory_id,
            project_id,
            role=FactoryProjectRole.CANONICAL,
            status=FactoryProjectStatus.ACTIVE,
            display_name=display_name,
            description=description,
            workspace_policy=workspace_policy,
            resource_bindings=resource_bindings,
            feed_health=feed_health,
            default_launch_profile=default_launch_profile,
            metadata=metadata,
        )

    def link_auxiliary_project(
        self,
        factory_id: str,
        project_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        workspace_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        resource_bindings: Mapping[str, Any] | dict[str, Any] | None = None,
        feed_health: Mapping[str, Any] | dict[str, Any] | None = None,
        default_launch_profile: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryProjectLink:
        """Link an additional active Project to the Factory."""

        return self.link_project(
            factory_id,
            project_id,
            role=FactoryProjectRole.AUXILIARY,
            status=FactoryProjectStatus.ACTIVE,
            display_name=display_name,
            description=description,
            workspace_policy=workspace_policy,
            resource_bindings=resource_bindings,
            feed_health=feed_health,
            default_launch_profile=default_launch_profile,
            metadata=metadata,
        )

    def list_projects(
        self,
        factory_id: str,
        *,
        include_archived: bool = False,
    ) -> List[FactoryProjectLink]:
        return [
            FactoryProjectLink.from_wire(item)
            for item in self._client.list_factory_projects(
                factory_id,
                include_archived=include_archived,
            )
        ]

    def get_project(self, factory_id: str, project_id: str) -> FactoryProjectLink:
        return FactoryProjectLink.from_wire(
            self._client.get_factory_project(factory_id, project_id)
        )

    def patch_project(
        self,
        factory_id: str,
        project_id: str,
        request: FactoryProjectPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryProjectLink:
        return FactoryProjectLink.from_wire(
            self._client.patch_factory_project(factory_id, project_id, request)
        )

    def pause_project(self, factory_id: str, project_id: str) -> FactoryProjectLink:
        return self.patch_project(
            factory_id,
            project_id,
            FactoryProjectPatchRequest(status=FactoryProjectStatus.PAUSED),
        )

    def resume_project(self, factory_id: str, project_id: str) -> FactoryProjectLink:
        return self.patch_project(
            factory_id,
            project_id,
            FactoryProjectPatchRequest(status=FactoryProjectStatus.ACTIVE),
        )

    def archive_project(self, factory_id: str, project_id: str) -> FactoryProjectLink:
        return self.patch_project(
            factory_id,
            project_id,
            FactoryProjectPatchRequest(
                role=FactoryProjectRole.ARCHIVED_REFERENCE,
                status=FactoryProjectStatus.ARCHIVED,
            ),
        )

    def workspace(
        self,
        factory_id: str,
        *,
        include_archived: bool = False,
    ) -> FactoryWorkspace:
        return FactoryWorkspace.from_wire(
            self._client.get_factory_workspace(
                factory_id,
                include_archived=include_archived,
            )
        )

    def get_workspace(
        self,
        factory_id: str,
        *,
        include_archived: bool = False,
    ) -> FactoryWorkspace:
        return self.workspace(factory_id, include_archived=include_archived)

    def get_workspace_project(self, factory_id: str) -> FactoryProjectLink | None:
        workspace = self.workspace(factory_id)
        if workspace.canonical_project is not None:
            return workspace.canonical_project
        if workspace.project is not None:
            return workspace.project
        return self.canonical_project(factory_id)

    def archive_workspace_project(self, factory_id: str) -> FactoryProjectLink | None:
        workspace_project = self.get_workspace_project(factory_id)
        if workspace_project is None:
            return None
        return self.patch_project(
            factory_id,
            workspace_project.project_id,
            FactoryProjectPatchRequest(
                role=FactoryProjectRole.ARCHIVED_REFERENCE,
                status=FactoryProjectStatus.ARCHIVED,
            ),
        )

    def canonical_project(self, factory_id: str) -> FactoryProjectLink | None:
        for link in self.list_projects(factory_id):
            if (
                link.role == FactoryProjectRole.CANONICAL
                and link.status != FactoryProjectStatus.ARCHIVED
            ):
                return link
        return None

    def start(
        self,
        factory_id: str,
        *,
        reason: str | None = None,
        dry_run: bool = False,
        request: FactoryTransitionRequest | Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryTransitionResponse:
        return FactoryTransitionResponse.from_wire(
            self._client.start_factory(
                factory_id,
                request,
                reason=reason,
                dry_run=dry_run,
            )
        )

    def pause(
        self,
        factory_id: str,
        *,
        reason: str | None = None,
        dry_run: bool = False,
        request: FactoryTransitionRequest | Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryTransitionResponse:
        return FactoryTransitionResponse.from_wire(
            self._client.pause_factory(
                factory_id,
                request,
                reason=reason,
                dry_run=dry_run,
            )
        )

    def resume(
        self,
        factory_id: str,
        *,
        reason: str | None = None,
        dry_run: bool = False,
        request: FactoryTransitionRequest | Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryTransitionResponse:
        return FactoryTransitionResponse.from_wire(
            self._client.resume_factory(
                factory_id,
                request,
                reason=reason,
                dry_run=dry_run,
            )
        )

    def archive(
        self,
        factory_id: str,
        *,
        reason: str | None = None,
        dry_run: bool = False,
        request: FactoryTransitionRequest | Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryTransitionResponse:
        return FactoryTransitionResponse.from_wire(
            self._client.archive_factory(
                factory_id,
                request,
                reason=reason,
                dry_run=dry_run,
            )
        )

    def status(self, factory_id: str) -> FactoryStatus:
        return FactoryStatus.from_wire(self._client.get_factory_status(factory_id))

    def experiment_bundle(
        self,
        project_id: str,
        experiment_id: str,
    ) -> ExperimentBundle:
        """Read the backend-owned experiment observability projection."""

        return ExperimentBundle.from_wire(
            self._client.get_experiment_bundle(project_id, experiment_id)
        )

    def experiment_history(
        self,
        project_id: str,
        *,
        limit: int = 50,
    ) -> ExperimentHistory:
        return ExperimentHistory.from_wire(
            self._client.get_experiment_history(project_id, limit=limit)
        )

    def compare_experiments(
        self,
        project_id: str,
        experiment_ids: tuple[str, ...] | List[str],
    ) -> ExperimentComparison:
        return ExperimentComparison.from_wire(
            self._client.compare_experiments(project_id, experiment_ids)
        )

    def create_idea(
        self,
        factory_id: str,
        *,
        title: str,
        body: str | None = None,
        status: FactoryIdeaStatus | str = FactoryIdeaStatus.OPEN,
        source: FactoryIdeaSource | str = FactoryIdeaSource.HUMAN,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        priority: str | None = None,
        tags: tuple[str, ...] = (),
        promotion_target: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryIdea:
        return FactoryIdea.from_wire(
            self._client.create_factory_idea(
                factory_id,
                FactoryIdeaCreateRequest(
                    title=title,
                    body=body,
                    status=status,
                    source=source,
                    project_id=project_id,
                    effort_id=effort_id,
                    run_id=run_id,
                    priority=priority,
                    tags=tags,
                    promotion_target=dict(promotion_target or {}),
                    metadata=dict(metadata or {}),
                ),
            )
        )

    def list_ideas(
        self,
        factory_id: str,
        *,
        status: FactoryIdeaStatus | str | None = None,
        source: FactoryIdeaSource | str | None = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> List[FactoryIdea]:
        return [
            FactoryIdea.from_wire(item)
            for item in self._client.list_factory_ideas(
                factory_id,
                status=_enum_query_value(status),
                source=_enum_query_value(source),
                include_archived=include_archived,
                limit=limit,
            )
        ]

    def get_idea(self, factory_id: str, idea_id: str) -> FactoryIdea:
        return FactoryIdea.from_wire(self._client.get_factory_idea(factory_id, idea_id))

    def patch_idea(
        self,
        factory_id: str,
        idea_id: str,
        request: FactoryIdeaPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryIdea:
        return FactoryIdea.from_wire(self._client.patch_factory_idea(factory_id, idea_id, request))

    def promote_idea(
        self,
        factory_id: str,
        idea_id: str,
        *,
        promotion_target: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryIdea:
        return self.patch_idea(
            factory_id,
            idea_id,
            FactoryIdeaPatchRequest(
                status=FactoryIdeaStatus.PROMOTED,
                promotion_target=dict(promotion_target or {}),
            ),
        )

    def pause_idea(self, factory_id: str, idea_id: str) -> FactoryIdea:
        return self.patch_idea(
            factory_id,
            idea_id,
            FactoryIdeaPatchRequest(status=FactoryIdeaStatus.PAUSED),
        )

    def archive_idea(self, factory_id: str, idea_id: str) -> FactoryIdea:
        return self.patch_idea(
            factory_id,
            idea_id,
            FactoryIdeaPatchRequest(status=FactoryIdeaStatus.ARCHIVED),
        )

    def create_actor_output(
        self,
        factory_id: str,
        *,
        actor_role: FactoryActorRole | str,
        kind: FactoryActorOutputKind | str,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.DRAFT,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        report_id: str | None = None,
        work_product_id: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryActorOutput:
        return FactoryActorOutput.from_wire(
            self._client.create_factory_actor_output(
                factory_id,
                FactoryActorOutputCreateRequest(
                    actor_role=actor_role,
                    kind=kind,
                    title=title,
                    summary=summary,
                    status=status,
                    project_id=project_id,
                    effort_id=effort_id,
                    run_id=run_id,
                    report_id=report_id,
                    work_product_id=work_product_id,
                    payload=dict(payload or {}),
                    metadata=dict(metadata or {}),
                ),
            )
        )

    def list_actor_outputs(
        self,
        factory_id: str,
        *,
        actor_role: FactoryActorRole | str | None = None,
        kind: FactoryActorOutputKind | str | None = None,
        status: FactoryActorOutputStatus | str | None = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> List[FactoryActorOutput]:
        return [
            FactoryActorOutput.from_wire(item)
            for item in self._client.list_factory_actor_outputs(
                factory_id,
                actor_role=_enum_query_value(actor_role),
                kind=_enum_query_value(kind),
                status=_enum_query_value(status),
                include_archived=include_archived,
                limit=limit,
            )
        ]

    def get_actor_output(
        self,
        factory_id: str,
        actor_output_id: str,
    ) -> FactoryActorOutput:
        return FactoryActorOutput.from_wire(
            self._client.get_factory_actor_output(factory_id, actor_output_id)
        )

    def patch_actor_output(
        self,
        factory_id: str,
        actor_output_id: str,
        request: FactoryActorOutputPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryActorOutput:
        return FactoryActorOutput.from_wire(
            self._client.patch_factory_actor_output(factory_id, actor_output_id, request)
        )

    def record_adjudicator_brief(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.DRAFT,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        report_id: str | None = None,
        work_product_id: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryActorOutput:
        return self.create_actor_output(
            factory_id,
            actor_role=FactoryActorRole.ADJUDICATOR,
            kind=FactoryActorOutputKind.ADJUDICATOR_BRIEF,
            title=title,
            summary=summary,
            status=status,
            project_id=project_id,
            effort_id=effort_id,
            run_id=run_id,
            report_id=report_id,
            work_product_id=work_product_id,
            payload=payload,
            metadata=metadata,
        )

    def record_gardener_digest(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.DRAFT,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        report_id: str | None = None,
        work_product_id: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryActorOutput:
        return self.create_actor_output(
            factory_id,
            actor_role=FactoryActorRole.GARDENER,
            kind=FactoryActorOutputKind.GARDENER_DIGEST,
            title=title,
            summary=summary,
            status=status,
            project_id=project_id,
            effort_id=effort_id,
            run_id=run_id,
            report_id=report_id,
            work_product_id=work_product_id,
            payload=payload,
            metadata=metadata,
        )

    def record_architect_feed_health(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.DRAFT,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        report_id: str | None = None,
        work_product_id: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryActorOutput:
        return self.create_actor_output(
            factory_id,
            actor_role=FactoryActorRole.ARCHITECT,
            kind=FactoryActorOutputKind.ARCHITECT_FEED_HEALTH,
            title=title,
            summary=summary,
            status=status,
            project_id=project_id,
            effort_id=effort_id,
            run_id=run_id,
            report_id=report_id,
            work_product_id=work_product_id,
            payload=payload,
            metadata=metadata,
        )

    def watch_status(
        self,
        factory_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
        stop_when_idle: bool = False,
    ) -> Iterator[FactoryStatus]:
        start = time.monotonic()
        while True:
            status = self.status(factory_id)
            yield status
            if stop_when_idle and not status.latest_runs and not status.open_decisions:
                return
            if timeout is not None and time.monotonic() - start >= timeout:
                return
            time.sleep(max(poll_interval, 0.1))

    def list_efforts(self, factory_id: str) -> List[Effort]:
        return [
            Effort.from_wire(item) for item in self._client.list_efforts_for_factory(factory_id)
        ]

    def create_effort(
        self,
        factory_id: str,
        *,
        name: str,
        project_id: str | None = None,
        hypothesis_or_topic: str | None = None,
        effort_type: EffortType | str = EffortType.RESEARCH,
        status: EffortStatus | str = EffortStatus.ACTIVE,
        recurrence: EffortRecurrence | None = None,
        recurrence_policy: (
            EffortRecurrence | RecurrencePolicy | Mapping[str, Any] | dict[str, Any] | None
        ) = None,
        next_wake_at: datetime | str | None = None,
        latest_run_id: str | None = None,
        latest_report_id: str | None = None,
        latest_work_product_id: str | None = None,
        decision_needed: bool = False,
        decision_note: str | None = None,
        budget_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        publication_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        authorization_policy: (
            AuthorizationPolicy | Mapping[str, Any] | dict[str, Any] | None
        ) = None,
        actor_notes: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> Effort:
        workspace_project_id = project_id
        if workspace_project_id is None:
            workspace_project = self.get_workspace_project(factory_id)
            if workspace_project is None:
                raise ValueError(
                    "Factory has no workspace Project. Link one with "
                    "factories.link_workspace_project before creating Efforts."
                )
            workspace_project_id = workspace_project.project_id

        policy_payload = _effort_recurrence_payload(
            recurrence=recurrence,
            recurrence_policy=recurrence_policy,
        )

        return Effort.from_wire(
            self._client.create_effort(
                EffortCreateRequest(
                    factory_id=factory_id,
                    project_id=workspace_project_id,
                    name=name,
                    allow_implicit_project_link=False,
                    hypothesis_or_topic=hypothesis_or_topic,
                    status=status,
                    effort_type=effort_type,
                    recurrence_policy=policy_payload,
                    next_wake_at=next_wake_at,
                    latest_run_id=latest_run_id,
                    latest_report_id=latest_report_id,
                    latest_work_product_id=latest_work_product_id,
                    decision_needed=decision_needed,
                    decision_note=decision_note,
                    budget_policy=dict(budget_policy or {}),
                    publication_policy=dict(publication_policy or {}),
                    authorization_policy=(
                        authorization_policy
                        if isinstance(authorization_policy, AuthorizationPolicy)
                        else dict(authorization_policy or {})
                    ),
                    actor_notes=dict(actor_notes or {}),
                    metadata=dict(metadata or {}),
                )
            )
        )

    def list_open_decisions(self, factory_id: str) -> List[Effort]:
        return list(self.status(factory_id).open_decisions)

    @staticmethod
    def plan_standup(plan: Mapping[str, Any]) -> FactoryStandupPlan:
        """Resolve a stand-up plan into request payloads without sending anything.

        This is the dry run: every validation error a real ``standup`` would
        raise, raised here first.
        """

        project = _standup_mapping(plan.get("project"), field="project")
        create_project = _standup_mapping(plan.get("create_project"), field="create_project")
        project_id = _standup_optional_string(project.get("project_id"))
        if project_id is None and not create_project:
            raise ValueError("project.project_id or create_project is required")
        if create_project:
            SmrRunnableProjectRequest.from_wire(create_project)
        return FactoryStandupPlan(
            factory=_factory_payload(plan),
            project_link=_project_link_payload(plan),
            efforts=_effort_payloads(plan),
            project_id=project_id,
            create_project=create_project or None,
            wake_due=(_wake_due_preview_kwargs(plan) if plan.get("wake_due") else None),
        )

    def standup(
        self,
        plan: Mapping[str, Any],
        *,
        wake_due: bool = False,
        wake_due_launch: bool = False,
    ) -> FactoryStandupResult:
        """Create a Factory, link its project, and seed its efforts from one plan.

        ``wake_due`` previews due work; ``wake_due_launch`` additionally confirms
        that exact preview, which is the only way to actually launch runs.
        """

        resolved = self.plan_standup(plan)
        should_wake = wake_due or wake_due_launch or resolved.wake_due is not None

        created_project: dict[str, Any] | None = None
        project_id = resolved.project_id
        if project_id is None:
            created_project = self._client.create_runnable_project(resolved.create_project)
            project_id = _project_id_from_response(created_project)

        factory = self.create(resolved.factory)
        link = self.link_project(factory.factory_id, project_id, **resolved.project_link)
        efforts = [
            self.create_effort(
                factory.factory_id,
                **_effort_kwargs(effort, default_project_id=project_id),
            )
            for effort in resolved.efforts
        ]

        preview: FactoryWakeDueResult | None = None
        result: FactoryWakeDueResult | None = None
        if should_wake:
            preview = self.wake_due(
                factory.factory_id,
                **(resolved.wake_due or _wake_due_preview_kwargs({})),
            )
            if wake_due_launch and preview.ready > 0 and not preview.confirmation_required:
                raise RuntimeError("wake preview has ready work but is not confirmation-ready")
            result = (
                self._confirm_wake_preview(factory_id=factory.factory_id, preview=preview)
                if wake_due_launch and preview.confirmation_required
                else preview
            )

        return FactoryStandupResult(
            plan=resolved,
            factory=factory,
            project_id=project_id,
            link=link,
            efforts=efforts,
            status=self.status(factory.factory_id),
            created_project=created_project,
            wake_due_preview=preview if wake_due_launch else None,
            wake_due_result=result,
        )

    def _confirm_wake_preview(
        self,
        *,
        factory_id: str,
        preview: FactoryWakeDueResult,
    ) -> FactoryWakeDueResult:
        if preview.factory_id != factory_id:
            raise RuntimeError("wake preview factory_id does not match the created Factory")
        if not preview.dry_run or not preview.confirmation_required:
            raise RuntimeError("wake preview is not confirmation-ready")
        if preview.preview_id is None or preview.preview_token is None:
            raise RuntimeError("wake preview omitted its preview_id or preview_token")
        contract = preview.request_contract
        if contract is None:
            raise RuntimeError("wake preview omitted its resolved request_contract")
        if contract.confirmed_preview_token is not None:
            raise RuntimeError("wake preview request_contract is not confirmation-ready")
        result = self.wake_due(
            factory_id,
            launch_request=contract.launch_request,
            limit=contract.limit,
            allow_overlap=contract.allow_overlap,
            dry_run=False,
            continue_on_error=contract.continue_on_error,
            confirmed_preview_id=preview.preview_id,
            confirmed_preview_token=preview.preview_token,
        )
        if result.confirmed_preview_id != preview.preview_id or result.receipt_id is None:
            raise RuntimeError("wake receipt is not durably bound to the confirmed preview")
        return result

    def wake_due(
        self,
        factory_id: str,
        *,
        launch_request: Mapping[str, Any] | dict[str, Any] | None = None,
        effort_ids: tuple[str, ...] = (),
        limit: int = 10,
        allow_overlap: bool = False,
        dry_run: bool = False,
        continue_on_error: bool = True,
        confirmed_preview_id: str | None = None,
        confirmed_preview_token: str | None = None,
    ) -> FactoryWakeDueResult:
        """Preview due work or execute it with the corresponding signed token."""
        if dry_run and (confirmed_preview_id is not None or confirmed_preview_token is not None):
            raise ValueError("Factory wake previews do not accept confirmation fields")
        if not dry_run and (confirmed_preview_id is None or confirmed_preview_token is None):
            raise ValueError(
                "Factory wake execution requires the preview_id and preview_token "
                "returned by a dry-run preview"
            )
        return FactoryWakeDueResult.from_wire(
            self._client.wake_due_factory_efforts(
                factory_id,
                FactoryWakeDueRequest(
                    launch_request=dict(launch_request) if launch_request else None,
                    effort_ids=effort_ids,
                    limit=limit,
                    allow_overlap=allow_overlap,
                    dry_run=dry_run,
                    continue_on_error=continue_on_error,
                    confirmed_preview_id=confirmed_preview_id,
                    confirmed_preview_token=confirmed_preview_token,
                ),
            )
        )


class FactoryResultsAPI(_ClientNamespace):
    """Public Result surface, accessed as ``research.factories.results``.

    A Result is the canonical public object a Factory produces. Evaluation and
    current-best selection are optional; ordinary Results carry neither. These
    methods resolve to the same backend authority the legacy candidate/champion
    methods on ``FactoriesAPI`` use — never a second source of truth.
    """

    def list(
        self,
        factory_id: str,
        *,
        effort_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        readiness: str | None = None,
        evaluation_status: str | None = None,
        current_best: bool | None = None,
        limit: int = 100,
    ) -> List[FactoryResult]:
        return [
            FactoryResult.from_wire(item)
            for item in self._client.list_factory_results(
                factory_id,
                effort_id=effort_id,
                run_id=run_id,
                kind=kind,
                readiness=readiness,
                evaluation_status=evaluation_status,
                current_best=current_best,
                limit=limit,
            )
        ]

    def get(self, factory_id: str, result_id: str) -> FactoryResult:
        return FactoryResult.from_wire(self._client.get_factory_result(factory_id, result_id))

    def evaluate(
        self,
        factory_id: str,
        result_id: str,
        *,
        evaluation: Mapping[str, Any] | dict[str, Any],
    ) -> FactoryResult:
        return FactoryResult.from_wire(
            self._client.evaluate_factory_result(
                factory_id,
                result_id,
                FactoryResultEvaluateRequest(evaluation=dict(evaluation)),
            )
        )

    def select_current_best(
        self,
        factory_id: str,
        *,
        result_id: str,
        reason: str,
        scope: str | None = None,
        effort_id: str | None = None,
    ) -> FactoryResultSelectionDecision:
        return FactoryResultSelectionDecision.from_wire(
            self._client.select_factory_result_current_best(
                factory_id,
                FactoryResultSelectRequest(
                    result_id=result_id,
                    reason=reason,
                    scope=scope,
                    effort_id=effort_id,
                ),
            )
        )

    def restore_current_best(
        self,
        factory_id: str,
        *,
        result_id: str,
        reason: str,
        scope: str | None = None,
        effort_id: str | None = None,
    ) -> FactoryResultSelectionDecision:
        return FactoryResultSelectionDecision.from_wire(
            self._client.restore_factory_result_current_best(
                factory_id,
                FactoryResultRestoreRequest(
                    result_id=result_id,
                    reason=reason,
                    scope=scope,
                    effort_id=effort_id,
                ),
            )
        )

    def selection_events(
        self,
        factory_id: str,
        *,
        limit: int = 100,
    ) -> List[FactoryResultSelectionEvent]:
        return [
            FactoryResultSelectionEvent.from_wire(item)
            for item in self._client.list_factory_result_selection_events(
                factory_id,
                limit=limit,
            )
        ]


class EffortsAPI(_ClientNamespace):
    def create(
        self,
        request: EffortCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Effort:
        return Effort.from_wire(self._client.create_effort(request))

    def get(self, effort_id: str) -> Effort:
        return Effort.from_wire(self._client.get_effort(effort_id))

    def patch(
        self,
        effort_id: str,
        request: EffortPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Effort:
        return Effort.from_wire(self._client.patch_effort(effort_id, request))

    def list_graduation_proposals(self, project_id: str) -> List[GraduationProposal]:
        return [
            GraduationProposal.from_wire(item)
            for item in self._client.list_graduation_proposals(project_id)
        ]

    def from_runs(
        self,
        *,
        project_id: str,
        name: str,
        run_ids: Iterable[str],
        factory_id: str | None = None,
    ) -> Effort:
        return Effort.from_wire(
            self._client.create_effort_from_runs(
                EffortFromRunsRequest(
                    project_id=project_id,
                    name=name,
                    run_ids=tuple(run_ids),
                    factory_id=factory_id,
                )
            )
        )

    def list_runs(self, effort_id: str) -> List[dict[str, Any]]:
        return self._client.list_runs_for_effort(effort_id)

    def pause(self, effort_id: str) -> Effort:
        return self.patch(effort_id, EffortPatchRequest(status=EffortStatus.PAUSED))

    def resume(self, effort_id: str) -> Effort:
        return self.patch(effort_id, EffortPatchRequest(status=EffortStatus.ACTIVE))

    def mark_waiting(
        self,
        effort_id: str,
        *,
        next_wake_at: datetime | str | None = None,
        note: str | None = None,
    ) -> Effort:
        return self.patch(
            effort_id,
            EffortPatchRequest(
                status=EffortStatus.WAITING,
                next_wake_at=next_wake_at,
                decision_note=note,
            ),
        )

    def mark_blocked(self, effort_id: str, *, note: str | None = None) -> Effort:
        return self.patch(
            effort_id,
            EffortPatchRequest(status=EffortStatus.BLOCKED, decision_note=note),
        )

    def mark_ready_for_review(self, effort_id: str, *, note: str | None = None) -> Effort:
        return self.patch(
            effort_id,
            EffortPatchRequest(status=EffortStatus.READY_FOR_REVIEW, decision_note=note),
        )

    def archive_reference(self, effort_id: str) -> Effort:
        return self.patch(
            effort_id,
            EffortPatchRequest(status=EffortStatus.ARCHIVED_REFERENCE),
        )

    def set_next_wake(self, effort_id: str, next_wake_at: datetime | str | None) -> Effort:
        return self.patch(effort_id, EffortPatchRequest(next_wake_at=next_wake_at))

    def schedule(
        self,
        effort_id: str,
        *,
        next_wake_at: datetime | str,
        recurrence_policy: (EffortRecurrence | Mapping[str, Any] | dict[str, Any] | None) = None,
        launch_request: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> Effort:
        policy: dict[str, Any] = {}
        if recurrence_policy is not None:
            policy.update(
                _wire_mapping_payload(
                    recurrence_policy,
                    field_name="recurrence_policy",
                )
            )
        if launch_request is not None:
            policy["launch_request"] = dict(launch_request)
        return self.patch(
            effort_id,
            EffortPatchRequest(
                status=EffortStatus.WAITING,
                recurrence_policy=policy,
                next_wake_at=next_wake_at,
            ),
        )

    def resolve_decision(self, effort_id: str, *, note: str | None = None) -> Effort:
        return self.patch(
            effort_id,
            EffortPatchRequest(decision_needed=False, decision_note=note),
        )

    def launch(
        self,
        effort_id: str,
        objective: str | None = None,
        *,
        run_kind: str = "research",
        **kwargs: Any,
    ):
        effort = self.get(effort_id)
        objective_text = (
            str(effort.hypothesis_or_topic or effort.name or "").strip()
            if objective is None
            else str(objective).strip()
        )
        return self._client.runs.start(
            objective_text,
            project_id=effort.project_id,
            effort_id=effort.effort_id,
            run_kind=run_kind,
            **kwargs,
        )

    def launch_maintenance(
        self,
        effort_id: str,
        *,
        objective: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Keep a migration stub so callers get a clear local refusal instead
        # of an HTTP 400 from a forbidden client run_kind=maintenance start.
        _ = (objective, kwargs)
        raise ValueError(
            f"launch_maintenance({effort_id!r}) is not supported; maintenance "
            "is Factory wake-due owned. Use factories.preview_wake / "
            "factories.wake_due instead of efforts.launch(..., "
            "run_kind='maintenance')"
        )


class FactoryLensesAPI(_ClientNamespace):
    """Evaluation lenses, derived best-so-far, and human preference.

    Requires the Factory to be on the champion-free Result authority; the
    backend answers a typed 409 otherwise rather than storing a lens that could
    never take effect.
    """

    def define(self, factory_id: str, spec: FactoryLensSpec) -> FactoryEvaluationLens:
        """Declare how this Factory compares Results, appending a new version."""

        return FactoryEvaluationLens.from_wire(
            self._client.define_factory_evaluation_lens(factory_id, spec.to_wire())
        )

    def list(
        self,
        factory_id: str,
        *,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> List[FactoryEvaluationLens]:
        """List lens versions, newest per key unless superseded are requested."""

        return [
            FactoryEvaluationLens.from_wire(item)
            for item in self._client.list_factory_evaluation_lenses(
                factory_id,
                include_superseded=include_superseded,
                limit=limit,
            )
        ]

    def best_so_far(self, factory_id: str) -> FactoryBestResults:
        """Derived best-so-far per lens.

        ``optimizes is False`` means the Factory hillclimbs nothing — a valid
        steady state, not an empty result set.
        """

        return FactoryBestResults.from_wire(self._client.get_factory_best_results(factory_id))

    def record_evaluation(
        self,
        factory_id: str,
        result_id: str,
        request: FactoryResultEvaluationRequest,
    ) -> FactoryResultEvaluation:
        """Store one externally owned verdict, idempotent under attempt_key."""

        return FactoryResultEvaluation.from_wire(
            self._client.record_factory_result_evaluation(factory_id, result_id, request.to_wire())
        )

    def prefer(self, factory_id: str, request: FactoryPreferenceRequest) -> FactoryPreferenceEvent:
        """Append an immutable preference event beside the derived best."""

        return FactoryPreferenceEvent.from_wire(
            self._client.record_factory_result_preference(factory_id, request.to_wire())
        )


__all__ = [
    "EffortsAPI",
    "FactoriesAPI",
    "FactoryResultsAPI",
    "FactoryStandupPlan",
    "FactoryStandupResult",
]
