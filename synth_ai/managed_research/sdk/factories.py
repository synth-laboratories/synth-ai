"""Factory and Effort SDK namespaces."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any

from synth_ai.managed_research.models.factories import (
    Effort,
    EffortCreateRequest,
    EffortPatchRequest,
    EffortStatus,
    EffortType,
    Factory,
    FactoryCreateRequest,
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


class FactoriesAPI(_ClientNamespace):
    def create(
        self,
        request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.create_factory(request))

    def list(self, *, include_archived: bool = False) -> list[Factory]:
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
        """Link the V1 singular workspace Project for a Factory."""

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

    def list_projects(
        self,
        factory_id: str,
        *,
        include_archived: bool = False,
    ) -> list[FactoryProjectLink]:
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
            FactoryProjectPatchRequest(status=FactoryProjectStatus.ARCHIVED),
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
        if workspace.project is not None:
            return workspace.project
        if workspace.canonical_project is not None:
            return workspace.canonical_project
        return self.canonical_project(factory_id)

    def archive_workspace_project(self, factory_id: str) -> FactoryProjectLink | None:
        workspace_project = self.get_workspace_project(factory_id)
        if workspace_project is None:
            return None
        return self.patch_project(
            factory_id,
            workspace_project.project_id,
            FactoryProjectPatchRequest(status=FactoryProjectStatus.ARCHIVED),
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

    def list_efforts(self, factory_id: str) -> list[Effort]:
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
            to_wire = getattr(recurrence_policy, "to_wire", None)
            policy_payload.update(dict(to_wire() if callable(to_wire) else recurrence_policy))

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
                    actor_notes=dict(actor_notes or {}),
                    metadata=dict(metadata or {}),
                )
            )
        )

    def list_open_decisions(self, factory_id: str) -> list[Effort]:
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
            to_wire = getattr(recurrence_policy, "to_wire", None)
            policy.update(dict(to_wire() if callable(to_wire) else recurrence_policy))
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
