"""Post-747 bindings tests: fake sessions only, no network.

Covers factory usage/events, account readiness, and the trigger_result
launch-request regression.
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.models.run_launch import RunLaunchRequest, RunLaunchResult
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.runs import RunsAPI
from synth_ai.research.account import AccountReadiness, ResearchAccountAPI
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.factory_usage import (
    FactoryEventsPage,
    FactoryUsage,
)


class _RecordingSession:
    """Fake session client capturing every ``_request_json`` call."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = responses or {}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
            }
        )
        return self._responses.get(f"{method} {path}", {})


def _sample_usage_payload() -> dict[str, Any]:
    return {
        "factory_id": "fac_1",
        "org_id": "org_1",
        "window": "month_to_date",
        "window_start": "2026-07-01T00:00:00Z",
        "window_end": "2026-07-20T00:00:00Z",
        "cost": {"total_cents": 1234, "total_pico_usd": 12340000000000, "total_usd": 12.34},
        "budget": {
            "run_limit": 10,
            "run_count": 4,
            "remaining_runs": 6,
            "limit_usd": 100.0,
            "used_usd": 12.34,
            "remaining_usd": 87.66,
        },
        "run_count": 4,
        "effort_count": 2,
        "by_effort": [
            {
                "effort_id": "eff_1",
                "name": "Effort One",
                "cost_microcents": 123400,
                "cost_cents": 1,
                "cost_usd": 0.01234,
                "debit_count": 3,
            }
        ],
        "updated_at": "2026-07-20T12:00:00Z",
        "brand_new_backend_key": True,
    }


def test_factory_usage_path_and_decode() -> None:
    session = _RecordingSession(
        responses={"GET /smr/factories/fac_1/usage": _sample_usage_payload()}
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]

    usage = api.usage("fac_1", window="last_7_days")
    assert isinstance(usage, FactoryUsage)
    assert usage.factory_id == "fac_1"
    assert usage.org_id == "org_1"
    assert usage.window == "month_to_date"
    assert usage.window_start is not None
    assert usage.cost.total_cents == 1234
    assert usage.cost.total_pico_usd == 12340000000000
    assert usage.cost.total_usd == 12.34
    assert usage.budget is not None
    assert usage.budget.run_limit == 10
    assert usage.budget.remaining_usd == 87.66
    assert usage.run_count == 4
    assert usage.effort_count == 2
    assert usage.by_effort[0].effort_id == "eff_1"
    assert usage.by_effort[0].debit_count == 3
    assert usage.updated_at is not None
    assert usage.raw["brand_new_backend_key"] is True

    assert session.calls == [
        {
            "method": "GET",
            "path": "/smr/factories/fac_1/usage",
            "params": {"window": "last_7_days"},
            "json_body": None,
        }
    ]


def test_factory_usage_null_budget() -> None:
    payload = _sample_usage_payload()
    payload["budget"] = None
    usage = FactoryUsage.from_wire(payload)
    assert usage.budget is None


def test_factory_events_path_and_decode() -> None:
    session = _RecordingSession(
        responses={
            "GET /smr/factories/fac_1/events": {
                "factory_id": "fac_1",
                "events": [
                    {
                        "event_id": "evt_1",
                        "occurred_at": "2026-07-20T11:00:00Z",
                        "kind": "champion_selected",
                        "source": "champion_event",
                        "payload": {"candidate_id": "cand_1"},
                    }
                ],
                "next_cursor": "cur_2",
            }
        }
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]

    page = api.events("fac_1", limit=25, cursor="cur_1")
    assert isinstance(page, FactoryEventsPage)
    assert page.factory_id == "fac_1"
    assert page.next_cursor == "cur_2"
    assert page.events[0].event_id == "evt_1"
    assert page.events[0].kind == "champion_selected"
    assert page.events[0].source == "champion_event"
    assert page.events[0].payload == {"candidate_id": "cand_1"}
    assert page.events[0].occurred_at is not None

    assert session.calls == [
        {
            "method": "GET",
            "path": "/smr/factories/fac_1/events",
            "params": {"limit": 25, "cursor": "cur_1"},
            "json_body": None,
        }
    ]


def test_factory_events_default_params_omitted() -> None:
    session = _RecordingSession(
        responses={"GET /smr/factories/fac_1/events": {"factory_id": "fac_1", "events": []}}
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    page = api.events("fac_1")
    assert page.events == ()
    assert page.next_cursor is None
    assert session.calls[0]["params"] is None


class _PagingSession:
    """Fake session serving a fixed cursor sequence, including a repeat."""

    def __init__(self, pages: dict[str | None, dict[str, Any]]) -> None:
        self.cursors_seen: list[str | None] = []
        self._pages = pages

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        cursor = (params or {}).get("cursor")
        self.cursors_seen.append(cursor)
        return self._pages[cursor]


def _event(event_id: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "occurred_at": "2026-07-20T11:00:00Z",
        "kind": "k",
        "source": "project_event",
        "payload": {},
    }


def test_iter_events_pages_until_cursor_drains() -> None:
    session = _PagingSession(
        {
            None: {"factory_id": "fac_1", "events": [_event("e1")], "next_cursor": "c1"},
            "c1": {"factory_id": "fac_1", "events": [_event("e2")], "next_cursor": "c2"},
            "c2": {"factory_id": "fac_1", "events": [_event("e3")], "next_cursor": None},
        }
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    ids = [event.event_id for event in api.iter_events("fac_1")]
    assert ids == ["e1", "e2", "e3"]
    assert session.cursors_seen == [None, "c1", "c2"]


def test_iter_events_stops_on_repeated_cursor() -> None:
    session = _PagingSession(
        {
            None: {"factory_id": "fac_1", "events": [_event("e1")], "next_cursor": "c1"},
            "c1": {"factory_id": "fac_1", "events": [_event("e2")], "next_cursor": "c1"},
        }
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    ids = [event.event_id for event in api.iter_events("fac_1")]
    assert ids == ["e1", "e2"]
    assert session.cursors_seen == [None, "c1"]


def test_factory_handle_usage_and_events_paths() -> None:
    session = _RecordingSession(
        responses={
            "GET /smr/factories/fac_1/usage": _sample_usage_payload(),
            "GET /smr/factories/fac_1/events": {"factory_id": "fac_1", "events": []},
        }
    )
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    handle = api.open("fac_1")

    usage = handle.usage(window="month_to_date")
    assert usage.factory_id == "fac_1"
    page = handle.events(limit=5)
    assert page.factory_id == "fac_1"

    observed = [(call["method"], call["path"]) for call in session.calls]
    assert observed == [
        ("GET", "/smr/factories/fac_1/usage"),
        ("GET", "/smr/factories/fac_1/events"),
    ]
    assert session.calls[0]["params"] == {"window": "month_to_date"}
    assert session.calls[1]["params"] == {"limit": 5}


def test_account_readiness_path_and_decode() -> None:
    session = _RecordingSession(
        responses={
            "GET /api/v1/account/readiness": {
                "ready": False,
                "checks": [
                    {
                        "id": "balance_positive",
                        "ok": False,
                        "label": "Credit balance is positive",
                        "cta": {"label": "Add credits", "href": "/settings/billing"},
                        "metadata": {"balance_cents": 0, "balance_dollars": 0.0},
                    },
                    {
                        "id": "byok_valid",
                        "ok": True,
                        "label": "Provider keys stored",
                        "cta": None,
                        "metadata": {},
                    },
                ],
            }
        }
    )
    account = ResearchAccountAPI(session)  # type: ignore[arg-type]

    readiness = account.readiness()
    assert isinstance(readiness, AccountReadiness)
    assert readiness.ready is False
    assert len(readiness.checks) == 2
    first = readiness.checks[0]
    assert first.id == "balance_positive"
    assert first.ok is False
    assert first.cta is not None
    assert first.cta.label == "Add credits"
    assert first.cta.href == "/settings/billing"
    assert first.metadata["balance_cents"] == 0
    second = readiness.checks[1]
    assert second.ok is True
    assert second.cta is None

    assert session.calls == [
        {
            "method": "GET",
            "path": "/api/v1/account/readiness",
            "params": None,
            "json_body": None,
        }
    ]


class _FakeLaunchClient:
    """Fake client using the real ``trigger_run_result`` over a recording ``trigger_run``."""

    trigger_run_result = ManagedResearchClient.trigger_run_result

    def __init__(self) -> None:
        self.trigger_calls: list[tuple[str, dict[str, Any]]] = []

    def trigger_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.trigger_calls.append((project_id, kwargs))
        return {"run_id": "run_1"}


def test_trigger_run_result_expands_launch_request() -> None:
    """Regression: trigger_run has no ``request`` kwarg; the request must be expanded."""
    client = _FakeLaunchClient()
    request = RunLaunchRequest(runbook_preset="smoke", timebox_seconds=60)

    result = client.trigger_run_result("proj_1", request=request)

    assert isinstance(result, RunLaunchResult)
    assert result.project_id == "proj_1"
    assert result.run_id == "run_1"
    assert len(client.trigger_calls) == 1
    project_id, kwargs = client.trigger_calls[0]
    assert project_id == "proj_1"
    assert "request" not in kwargs
    assert kwargs["runbook_preset"] == "smoke"
    assert kwargs["timebox_seconds"] == 60


def test_runs_api_trigger_result_uses_expanded_request() -> None:
    client = _FakeLaunchClient()
    api = RunsAPI(client)  # type: ignore[arg-type]
    request = RunLaunchRequest(runbook_preset="smoke", timebox_seconds=30)

    result = api.trigger_result("proj_1", request=request)

    assert result.run_id == "run_1"
    project_id, kwargs = client.trigger_calls[0]
    assert project_id == "proj_1"
    assert "request" not in kwargs
    assert kwargs["timebox_seconds"] == 30
