"""The per-scope limit CRUD client sends the backend's exact route shapes."""

from __future__ import annotations

from typing import Any

import pytest
from synth_ai.sdk.research.public import LimitScope, ScopeLimit
from synth_ai.sdk.research.session.scope_limits import KEEP_CAP, ScopeLimitsAPI

_ROW = {
    "scope_kind": "run",
    "scope_id": "run-1",
    "dimension": "spend_usd",
    "cap_amount": 25,
    "used_amount": 3.5,
    "remaining_amount": 21.5,
    "fraction_used": 0.14,
    "unit": "usd",
    "cap_revision": 2,
    "cap_source": "api",
    "alert_at_fraction": 0.9,
    "enforce_at_fraction": 1.0,
    "exhaustion_action": "pause",
    "notify_audience": "orchestrator",
    "policy": {"fail_closed_on_unpriced": True},
    "last_used_writer": "spend_ledger",
    "updated_at": "2026-09-23T12:00:00+00:00",
}


class _Recorder:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _request_json(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append((method, path, json_body))
        return self.response


def _api(response: Any) -> tuple[ScopeLimitsAPI, _Recorder]:
    recorder = _Recorder(response)
    return ScopeLimitsAPI(recorder), recorder  # type: ignore[arg-type]


_BASE = "/smr/projects/p-1/runs/run-1/limits"


def test_rows_parse_into_scope_limits() -> None:
    api, recorder = _api([_ROW])
    (row,) = api.list("p-1", LimitScope.RUN, "run-1")
    assert recorder.calls == [("GET", _BASE, None)]
    assert isinstance(row, ScopeLimit)
    assert (row.dimension, row.cap_amount, row.cap_revision) == ("spend_usd", 25.0, 2)
    assert row.policy == {"fail_closed_on_unpriced": True}
    assert row.updated_at is not None


def test_create_update_delete_paths_and_bodies() -> None:
    api, recorder = _api(_ROW)
    api.create("p-1", LimitScope.RUN, "run-1", dimension="tokens", cap_amount=1000)
    api.update("p-1", LimitScope.RUN, "run-1", "spend_usd", cap_amount=30)
    api.delete("p-1", LimitScope.RUN, "run-1", "spend_usd")
    assert recorder.calls == [
        ("POST", _BASE, {"dimension": "tokens", "cap_amount": 1000}),
        ("PATCH", f"{_BASE}/spend_usd", {"cap_amount": 30}),
        ("DELETE", f"{_BASE}/spend_usd", None),
    ]


def test_update_keeps_or_removes_the_cap_explicitly() -> None:
    api, recorder = _api(_ROW)
    api.update("p-1", LimitScope.RUN, "run-1", "spend_usd", policy={"exhaustion_action": "stop"})
    api.update("p-1", LimitScope.RUN, "run-1", "spend_usd", cap_amount=None)
    assert recorder.calls[0][2] == {"policy": {"exhaustion_action": "stop"}}
    assert recorder.calls[1][2] == {"cap_amount": None}
    assert KEEP_CAP.value == "keep"


def test_objective_scope_uses_its_own_path_segment() -> None:
    api, recorder = _api([])
    assert api.list("p-1", LimitScope.OBJECTIVE, "obj-1") == []
    assert recorder.calls[0][1] == "/smr/projects/p-1/objectives/obj-1/limits"
    with pytest.raises(ValueError):
        api.list("p-1", "project", "p-1")  # type: ignore[arg-type]
