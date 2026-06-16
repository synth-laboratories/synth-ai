"""Factory and Effort SDK namespaces."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any, List, cast

from synth_ai.managed_research.models.factories import (
    AuthorizationPolicy,
    Effort,
    EffortCreateRequest,
    EffortPatchRequest,
    EffortStatus,
    EffortType,
    Factory,
    FactoryActorOutput,
    FactoryActorOutputCreateRequest,
    FactoryActorOutputKind,
    FactoryActorOutputPatchRequest,
    FactoryActorOutputStatus,
    FactoryActorRole,
    FactoryCreateRequest,
    FactoryIdea,
    FactoryIdeaCreateRequest,
    FactoryIdeaPatchRequest,
    FactoryIdeaSource,
    FactoryIdeaStatus,
    FactoryLifecycleState,
    FactoryPatchRequest,
    FactoryProjectLink,
    FactoryProjectLinkRequest,
    FactoryProjectPatchRequest,
    FactoryProjectRole,
    FactoryProjectStatus,
    FactoryStatus,
    FactoryWakeDueRequest,
    FactoryWakeDueResult,
    FactoryWorkspace,
    RecurrencePolicy,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


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

    def pause(self, factory_id: str) -> Factory:
        return self.patch(factory_id, FactoryPatchRequest(status=FactoryLifecycleState.PAUSED))

    def resume(self, factory_id: str) -> Factory:
        return self.patch(factory_id, FactoryPatchRequest(status=FactoryLifecycleState.ACTIVE))

    def archive(self, factory_id: str) -> Factory:
        return self.patch(factory_id, FactoryPatchRequest(status=FactoryLifecycleState.ARCHIVED))

    def status(self, factory_id: str) -> FactoryStatus:
        return FactoryStatus.from_wire(self._client.get_factory_status(factory_id))

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

    def record_seraph_brief(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.ACCEPTED,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> FactoryActorOutput:
        return self.create_actor_output(
            factory_id,
            actor_role=FactoryActorRole.SERAPH,
            kind=FactoryActorOutputKind.SERAPH_BRIEF,
            title=title,
            summary=summary,
            status=status,
            project_id=project_id,
            effort_id=effort_id,
            run_id=run_id,
            payload=payload,
            metadata=metadata,
        )

    def record_gardener_digest(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.ACCEPTED,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
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
            payload=payload,
            metadata=metadata,
        )

    def record_architect_feed_health(
        self,
        factory_id: str,
        *,
        title: str,
        summary: str | None = None,
        status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.ACCEPTED,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
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
        recurrence_policy: RecurrencePolicy | Mapping[str, Any] | dict[str, Any] | None = None,
        next_wake_at: datetime | str | None = None,
        latest_run_id: str | None = None,
        latest_report_id: str | None = None,
        latest_work_product_id: str | None = None,
        decision_needed: bool = False,
        decision_note: str | None = None,
        budget_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        publication_policy: Mapping[str, Any] | dict[str, Any] | None = None,
        authorization_policy: AuthorizationPolicy
        | Mapping[str, Any]
        | dict[str, Any]
        | None = None,
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

        policy_payload: dict[str, Any] = {}
        if recurrence_policy is not None:
            policy_payload.update(
                _wire_mapping_payload(
                    recurrence_policy,
                    field_name="recurrence_policy",
                )
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

    def wake_due(
        self,
        factory_id: str,
        *,
        launch_request: Mapping[str, Any] | dict[str, Any] | None = None,
        limit: int = 10,
        allow_overlap: bool = False,
        dry_run: bool = False,
        continue_on_error: bool = True,
    ) -> FactoryWakeDueResult:
        return FactoryWakeDueResult.from_wire(
            self._client.wake_due_factory_efforts(
                factory_id,
                FactoryWakeDueRequest(
                    launch_request=dict(launch_request) if launch_request else None,
                    limit=limit,
                    allow_overlap=allow_overlap,
                    dry_run=dry_run,
                    continue_on_error=continue_on_error,
                ),
            )
        )


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
        recurrence_policy: RecurrencePolicy | Mapping[str, Any] | dict[str, Any] | None = None,
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

    def launch(self, effort_id: str, objective: str | None = None, **kwargs: Any):
        from synth_ai.managed_research.models.run_state import ManagedResearchRun
        from synth_ai.managed_research.sdk.runs import RunHandle

        effort = self.get(effort_id)
        if objective is not None:
            return self._client.runs.start(
                objective,
                project_id=effort.project_id,
                effort_id=effort.effort_id,
                **kwargs,
            )
        wire = self._client.trigger_run(
            effort.project_id,
            effort_id=effort.effort_id,
            **kwargs,
        )
        run = ManagedResearchRun.from_wire(wire)
        return RunHandle(self._client, run.project_id, run.run_id)


__all__ = ["EffortsAPI", "FactoriesAPI"]
