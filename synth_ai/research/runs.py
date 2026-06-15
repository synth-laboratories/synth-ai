"""``client.research.runs`` — alpha run surface."""

from __future__ import annotations

from typing import Any, List

from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrRunUsage,
)
from synth_ai.managed_research.models.run_diagnostics import (
    SmrRunActorUsage,
    SmrRunCostSummary,
)
from synth_ai.managed_research.models.run_observability import (
    ManagedResearchRunContract,
    RunObservabilitySnapshot,
)
from synth_ai.managed_research.models.types import RunArtifact, RunArtifactManifest
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.runs import ProjectSelector, RunHandle
from synth_ai.research.models import ResearchRun, ResearchRunbookPreset, ResearchWorkProduct


class ResearchRunHandle:
    """Handle for one Research run (wait, readback, workspace archive)."""

    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle

    @property
    def project_id(self) -> str:
        return self._handle.project_id

    @property
    def run_id(self) -> str:
        return self._handle.run_id

    def get(self) -> ResearchRun:
        return self._handle.get()

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
    ) -> ResearchRun:
        return self._handle.wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )

    def contract(self) -> ManagedResearchRunContract:
        return self._handle.contract()

    def progress(
        self,
        *,
        detail_level: str = "control",
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> RunObservabilitySnapshot:
        return self._handle._client.get_run_observability_snapshot(
            self.project_id,
            self.run_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def full_progress(self) -> RunObservabilitySnapshot:
        return self._handle._client.get_run_observability_snapshot_full(
            self.project_id,
            self.run_id,
        )

    def work_products(self) -> list[ResearchWorkProduct]:
        return self._handle.work_products()

    def reports(self) -> list[ResearchWorkProduct]:
        return self._handle.reports()

    def final_report(self) -> ResearchWorkProduct | None:
        return self._handle.final_report()

    def report_text(self, work_product_id: str | None = None) -> str | None:
        return self._handle.report_text(work_product_id)

    def work_product_content(
        self,
        work_product_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        return self._handle._client.work_products.content(
            work_product_id,
            as_text=as_text,
        )

    def usage(self) -> SmrRunUsage:
        return self._handle._client.get_run_usage(self.run_id)

    def actor_usage(self) -> SmrRunActorUsage:
        return self._handle.actor_usage()

    def cost_summary(self) -> SmrRunCostSummary:
        return self._handle.cost_summary()

    def resource_limits(self) -> SmrResourceLimits:
        return self._handle.resource_limits()

    def progress_toward_resource_limits(self) -> SmrResourceLimitProgress:
        return self._handle.progress_toward_resource_limits()

    def artifact_manifest(self) -> RunArtifactManifest:
        return self._handle.artifact_manifest()

    def artifacts(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[RunArtifact]:
        return self._handle.artifacts(
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )

    def download_workspace_archive(
        self,
        destination: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._handle._client.download_run_workspace_archive(
            self.project_id,
            self.run_id,
            destination,
            timeout_seconds=timeout_seconds,
        )

    def list_artifacts(self) -> List[dict[str, Any]]:
        return self._handle._client.list_run_artifacts(self.project_id, self.run_id)


class ResearchRunsAPI:
    """Public Research run methods (alpha must-have)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def runbook_presets(self) -> tuple[ResearchRunbookPreset, ...]:
        return self._session.runs.runbook_presets()

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchRunHandle:
        """Start a run with a primary objective message (canonical name)."""
        handle = self._session.runs.start(
            objective,
            project_id=project_id,
            project=project,
            **kwargs,
        )
        return ResearchRunHandle(handle)

    def trigger(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compatibility alias for existing ReportBench drivers."""
        return self._session.runs.trigger(
            project_id,
            project=project,
            **kwargs,
        )

    def start_run(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._session.runs.start_run(
            project_id,
            project=project,
            **kwargs,
        )

    def get(self, project_id: str, run_id: str) -> ResearchRunHandle:
        return ResearchRunHandle(self._session.run(project_id, run_id))

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._session.runs.list(project_id, active_only=active_only, **kwargs)

    def wait(
        self,
        project_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
    ) -> ResearchRun:
        return self.get(project_id, run_id).wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )


__all__ = ["ResearchRunHandle", "ResearchRunsAPI"]
