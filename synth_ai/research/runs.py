"""``client.research.runs`` — alpha run surface."""

from __future__ import annotations

from typing import Any

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

    def work_products(self) -> list[ResearchWorkProduct]:
        return self._handle._client.work_products.list_for_run(
            self.project_id,
            self.run_id,
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

    def list_artifacts(self) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
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
