"""Factory- and Effort-scoped handles for ``client.research.factories``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, List

from synth_ai.managed_research.models.factories import (
    Effort,
    Factory,
    FactoryStatus,
    FactoryWakeDueResult,
)
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.factory_usage import (
    FactoryEventsPage,
    FactoryUsage,
    fetch_factory_events,
    fetch_factory_usage,
)

if TYPE_CHECKING:
    from synth_ai.research.factories import ResearchFactoriesAPI


class ResearchEffortHandle:
    """Handle bound to one Effort id."""

    def __init__(self, session: ManagedResearchClient, effort_id: str) -> None:
        self._session = session
        self.effort_id = effort_id

    def get(self) -> Effort:
        """Fetch the Effort. Backend route: ``GET /smr/efforts/{effort_id}``."""
        return self._session.efforts.get(self.effort_id)

    def pause(self) -> Effort:
        """Pause the Effort. Backend route: ``PATCH /smr/efforts/{effort_id}``."""
        return self._session.efforts.pause(self.effort_id)

    def resume(self) -> Effort:
        """Resume the Effort. Backend route: ``PATCH /smr/efforts/{effort_id}``."""
        return self._session.efforts.resume(self.effort_id)


class ResearchFactoryEffortsAPI:
    """Efforts namespace scoped to one Factory."""

    def __init__(self, session: ManagedResearchClient, factory_id: str) -> None:
        self._session = session
        self._factory_id = factory_id

    def list(self) -> List[Effort]:
        """List the Factory's Efforts.

        Backend route: ``GET /smr/factories/{factory_id}/efforts``.
        """
        return self._session.factories.list_efforts(self._factory_id)

    def open(self, effort_id: str) -> ResearchEffortHandle:
        """Open a handle bound to one Effort id (no network call)."""
        return ResearchEffortHandle(self._session, effort_id)


class ResearchFactoryHandle:
    """Handle bound to one Factory id.

    Lifecycle and wake methods delegate to the same session/facade bindings as
    ``client.research.factories``; the handle only fixes the ``factory_id``.
    """

    def __init__(
        self,
        session: ManagedResearchClient,
        factories: ResearchFactoriesAPI,
        factory_id: str,
    ) -> None:
        self._session = session
        self._factories = factories
        self.factory_id = factory_id
        self._efforts: ResearchFactoryEffortsAPI | None = None

    @property
    def efforts(self) -> ResearchFactoryEffortsAPI:
        """Efforts owned by this Factory."""
        if self._efforts is None:
            self._efforts = ResearchFactoryEffortsAPI(self._session, self.factory_id)
        return self._efforts

    def status(self) -> FactoryStatus:
        """Read the Factory workflow projection.

        Backend route: ``GET /smr/factories/{factory_id}/status``.
        """
        return self._session.factories.status(self.factory_id)

    def pause(self) -> Factory:
        """Pause the Factory. Backend route: ``PATCH /smr/factories/{factory_id}``."""
        return self._session.factories.pause(self.factory_id)

    def resume(self) -> Factory:
        """Resume the Factory. Backend route: ``PATCH /smr/factories/{factory_id}``."""
        return self._session.factories.resume(self.factory_id)

    def archive(self) -> Factory:
        """Archive the Factory. Backend route: ``PATCH /smr/factories/{factory_id}``."""
        return self._session.factories.archive(self.factory_id)

    def usage(self, *, window: str = "month_to_date") -> FactoryUsage:
        """Read the factory usage aggregate.

        Backend route: ``GET /smr/factories/{factory_id}/usage``
        (query ``window``: ``month_to_date`` | ``last_7_days``).
        """
        return fetch_factory_usage(self._session, self.factory_id, window=window)

    def events(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> FactoryEventsPage:
        """Read one newest-first page of durable factory events.

        Backend route: ``GET /smr/factories/{factory_id}/events``
        (query ``limit`` 1-500, ``cursor``).
        """
        return fetch_factory_events(
            self._session,
            self.factory_id,
            limit=limit,
            cursor=cursor,
        )

    def watch_status(
        self,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
        stop_when_idle: bool = False,
    ) -> Iterator[FactoryStatus]:
        """Poll the status projection until idle or timeout.

        Backend route: repeated ``GET /smr/factories/{factory_id}/status``.
        """
        return self._session.factories.watch_status(
            self.factory_id,
            poll_interval=poll_interval,
            timeout=timeout,
            stop_when_idle=stop_when_idle,
        )

    def preview_wake(
        self,
        *,
        launch_request: Mapping[str, Any] | dict[str, Any] | None = None,
        limit: int = 10,
        allow_overlap: bool = False,
        continue_on_error: bool = True,
    ) -> FactoryWakeDueResult:
        """Preview due experiments without launching.

        Backend route: ``POST /smr/factories/{factory_id}/wake-due`` (dry run).
        """
        return self._factories.preview_wake(
            self.factory_id,
            launch_request=launch_request,
            limit=limit,
            allow_overlap=allow_overlap,
            continue_on_error=continue_on_error,
        )

    def wake_due(self, *, preview: FactoryWakeDueResult) -> FactoryWakeDueResult:
        """Launch exactly the due experiments bound to a reviewed preview.

        Backend route: ``POST /smr/factories/{factory_id}/wake-due`` (confirmed).
        """
        return self._factories.wake_due(self.factory_id, preview=preview)


__all__ = [
    "ResearchEffortHandle",
    "ResearchFactoryEffortsAPI",
    "ResearchFactoryHandle",
]
