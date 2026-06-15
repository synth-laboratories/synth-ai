"""``SynthClient().research`` namespace (alpha)."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from synth_ai.research.control import ResearchControlClient
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.runs import ResearchRunsAPI


class ResearchClient:
    """Public Research API under ``SynthClient``.

    Alpha bootstrap: one ``ResearchControlClient`` session backs projects, runs,
    limits, and the control-plane driver used by ReportBench README smoke.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._session: ResearchControlClient | None = None
        self._projects: ResearchProjectsAPI | None = None
        self._runs: ResearchRunsAPI | None = None

    def _impl(self) -> ResearchControlClient:
        if self._session is None:
            self._session = ResearchControlClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._session

    @property
    def projects(self) -> ResearchProjectsAPI:
        if self._projects is None:
            self._projects = ResearchProjectsAPI(self._impl())
        return self._projects

    @property
    def runs(self) -> ResearchRunsAPI:
        if self._runs is None:
            self._runs = ResearchRunsAPI(self._impl())
        return self._runs

    def get_limits(self) -> dict[str, Any]:
        return self._impl().get_limits()

    def control(self, *, timeout_seconds: float | None = None) -> ResearchControlClient:
        """Low-level control plane (ReportBench / full SMR launch path)."""
        if timeout_seconds is not None and timeout_seconds != self.timeout_seconds:
            return ResearchControlClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout_seconds=timeout_seconds,
            )
        return self._impl()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._projects = None
        self._runs = None

    def __enter__(self) -> Self:
        self._impl().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["ResearchClient"]
