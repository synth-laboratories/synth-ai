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
    Factory,
    FactoryCreateRequest,
    FactoryLifecycleState,
    FactoryPatchRequest,
    FactoryStatus,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class FactoriesAPI(_ClientNamespace):
    def create(
        self,
        request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.create_factory(request))

    def list(self) -> list[Factory]:
        return [Factory.from_wire(item) for item in self._client.list_factories()]

    def get(self, factory_id: str) -> Factory:
        return Factory.from_wire(self._client.get_factory(factory_id))

    def patch(
        self,
        factory_id: str,
        request: FactoryPatchRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Factory:
        return Factory.from_wire(self._client.patch_factory(factory_id, request))

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

    def list_open_decisions(self, factory_id: str) -> list[Effort]:
        return list(self.status(factory_id).open_decisions)


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
