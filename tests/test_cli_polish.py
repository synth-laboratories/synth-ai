"""Unit tests for CLI pagination cursors, --created-after, and billing/usage/limits views."""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from synth_ai import _cli_view as view
from synth_ai import cli
from synth_ai._generated.models import (
    Balance,
    BillingHistoryPage,
    BillingStatementPage,
    EventPage,
    LimitList,
    OutputPage,
    RunMessagePage,
    RunPage,
    RunTaskPage,
    UsageRecordPage,
    UsageSummary,
)

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"


def _example(contract: dict[str, object], path: str, method: str = "get") -> dict[str, object]:
    paths = contract["paths"]
    assert isinstance(paths, dict)
    return copy.deepcopy(
        paths[path][method]["responses"]["200"]["content"]["application/json"]["example"]
    )


@pytest.mark.parametrize("argv", [["ls"], ["runs", "list"]])
def test_cli_ls_and_runs_list_pass_created_after_and_render_next_cursor(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    seen_params: list[dict[str, str]] = []
    page_raw = _example(contract, "/v1/runs")
    page_raw["has_more"] = True
    page_raw["next_cursor"] = "cur_next_123"

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/runs":
            seen_params.append(dict(request.url.params))
            return httpx.Response(200, json=page_raw)
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code = cli.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            *argv,
            "--created-after",
            "2026-11-28T00:00:00Z",
        ],
        http_client=http,
    )
    assert code == 0
    assert seen_params == [{"limit": "20", "created_after": "2026-11-28T00:00:00Z"}]
    out = capsys.readouterr().out
    assert "next cursor  cur_next_123" in out


def test_cli_watch_once_renders_next_cursor_in_human_mode_only(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    snap = _example(contract, "/v1/runs/{run_id}")
    snap["run_id"] = RUN_ID
    snap["state"] = "running"
    snap["execution_closed"] = False
    snap["accounting_final"] = False
    snap["cleanup_complete"] = False

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}/events":
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "has_more": True,
                    "next_cursor": "ev_cur_42",
                    "resume_cursor": "ev_cur_42",
                },
            )
        if request.method == "GET" and request.url.path == f"/v1/runs/{RUN_ID}":
            return httpx.Response(200, json=snap)
        return httpx.Response(404, json={"code": "not_found", "detail": "not found"})

    http_human = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code_human = cli.main(
        ["--base-url", "http://127.0.0.1:9", "watch", RUN_ID, "--once"],
        http_client=http_human,
    )
    assert code_human == 0
    assert "next cursor  ev_cur_42" in capsys.readouterr().out

    http_json = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code_json = cli.main(
        ["--base-url", "http://127.0.0.1:9", "watch", RUN_ID, "--once", "--json"],
        http_client=http_json,
    )
    assert code_json == 0
    assert capsys.readouterr().out.strip() == ""


def test_paginated_views_show_next_cursor_when_has_more(contract: dict[str, object]) -> None:
    cases = [
        (lambda raw: view.runs(RunPage.model_validate(raw)), "/v1/runs"),
        (lambda raw: view.events(EventPage.model_validate(raw)), "/v1/runs/{run_id}/events"),
        (lambda raw: view.tasks(RunTaskPage.model_validate(raw)), "/v1/runs/{run_id}/tasks"),
        (lambda raw: view.outputs(OutputPage.model_validate(raw)), "/v1/runs/{run_id}/outputs"),
        (
            lambda raw: view.messages(RunMessagePage.model_validate(raw)),
            "/v1/runs/{run_id}/messages",
        ),
        (
            lambda raw: view.usage_records(UsageRecordPage.model_validate(raw)),
            "/v1/usage/records",
        ),
        (
            lambda raw: view.billing_history(BillingHistoryPage.model_validate(raw)),
            "/v1/billing/history",
        ),
        (
            lambda raw: view.billing_statements(BillingStatementPage.model_validate(raw)),
            "/v1/billing/statements",
        ),
    ]
    for render, path in cases:
        raw_more = _example(contract, path)
        if "availability" in raw_more:
            raw_more["availability"] = {"kind": "available"}
        raw_more["has_more"] = True
        raw_more["next_cursor"] = "cur_page_2"
        rendered_more = render(raw_more)
        assert "next cursor  cur_page_2" in rendered_more

        raw_end = _example(contract, path)
        if "availability" in raw_end:
            raw_end["availability"] = {"kind": "available"}
        raw_end["has_more"] = False
        raw_end["next_cursor"] = None
        rendered_end = render(raw_end)
        assert "next cursor" not in rendered_end


def test_balance_view_shows_operator_credits_and_debt(contract: dict[str, object]) -> None:
    raw = _example(contract, "/v1/billing/balance")
    raw["operator_credits"] = {"amount": "12.50", "currency": "USD"}
    raw["debt"] = {"amount": "1.25", "currency": "USD"}
    rendered = view.balance(Balance.model_validate(raw))
    assert "operator credits" in rendered and "12.50 USD" in rendered
    assert "debt" in rendered and "1.25 USD" in rendered


def test_usage_summary_view_shows_settling_and_data_through(contract: dict[str, object]) -> None:
    raw = _example(contract, "/v1/usage/summary")
    raw["data_through"] = "2026-11-28T10:00:00Z"
    raw["settling"] = {"count": 3, "amount": {"amount": "0.75", "currency": "USD"}}
    rendered = view.usage_summary(UsageSummary.model_validate(raw))
    assert "data through" in rendered and "2026-11-28T10:00:00Z" in rendered
    assert "settling" in rendered and "0.75 USD (3)" in rendered


def test_limits_view_includes_reset_at_column(contract: dict[str, object]) -> None:
    raw = _example(contract, "/v1/limits")
    raw["items"] = [
        {
            "resource": "swarm_run",
            "metric": "spend",
            "window": "daily",
            "cap": 50.0,
            "current": 12.5,
            "reset_at": "2026-11-29T00:00:00Z",
        },
        {
            "resource": "swarm_run",
            "metric": "concurrency",
            "window": "instant",
            "cap": 5.0,
            "current": 1.0,
            "reset_at": None,
        },
    ]
    rendered = view.limits(LimitList.model_validate(raw))
    lines = rendered.splitlines()
    header_line = next(line for line in lines if "RESOURCE" in line)
    assert "RESET AT" in header_line
    assert "2026-11-29T00:00:00Z" in rendered
    assert json.dumps(raw)  # sanity check fixture is JSON-serializable
