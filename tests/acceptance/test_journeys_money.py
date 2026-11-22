"""SYN-5695 J2 money steps: limits, statement, pause/resume, queued launch, Stripe test checkout.

Shares the loopback harness in journey_loopback.py; split from test_journeys.py
so each file stays under the 120 s pytest cap.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from journey_loopback import (
    MONEY,
    PURCHASE_GRANT,
    SURFACES,
    checkout_responder,
    example,
    ledger_entry,
    limits_api,
    limits_view,
    run,
)

from scripts.acceptance.journeys import (
    Step,
    Verdict,
)
from scripts.acceptance.surfaces import SurfaceName

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("surface", [SurfaceName.CLI, SurfaceName.MCP_STDIO])
def test_raise_limit_is_seen_in_the_next_budget_view(contract, tmp_path, surface):
    result, requests = run(
        surface, limits_api(contract), tmp_path, Step.RAISE_LIMIT, run_id="run_" + "R" * 22
    )
    assert result.verdict == Verdict.PASS, result
    assert [method for method, _, _ in requests] == ["GET", "POST", "GET"]


def four_limits_api(contract, retention_ms):
    """Staging P2 shape: retention already at the contract maximum (86400 s)."""
    state = {
        "revision": 0,
        "posts": [],
        "caps": {
            "spend_micros": 1_000_000,
            "tokens": 1000,
            "execution_ms": 3_600_000,
            "pause_retention_ms": retention_ms,
        },
    }
    fields = {
        "maximum_charge": "spend_micros",
        "maximum_tokens": "tokens",
        "wall_time_seconds": "execution_ms",
        "pause_retention_seconds": "pause_retention_ms",
    }

    def responder(method, path, request):
        if method == "POST" and path.endswith("/limits"):
            field, value = next(iter(request["body"]["limits"].items()))
            state["posts"].append(field)
            state["caps"][fields[field]] = (
                int(Decimal(value["amount"]) * 1_000_000)
                if field == "maximum_charge"
                else value
                if field == "maximum_tokens"
                else value * 1000
            )
            state["revision"] += 1
        view = limits_view(contract, cap=0, revision=state["revision"])
        view["run_id"] = path.split("/")[3]
        view["primary"] = [
            {"dimension": dimension, "effective_cap": cap, "state": "capped"}
            for dimension, cap in state["caps"].items()
        ]
        return 200, view, {}

    return responder, state


def test_sdk_raise_holds_retention_at_contract_maximum_and_passes(contract, tmp_path):
    responder, state = four_limits_api(contract, 86_400_000)
    result, _ = run(
        SurfaceName.SDK, responder, tmp_path, Step.RAISE_LIMIT, run_id="run_" + "R" * 22
    )
    assert result.verdict == Verdict.PASS, result
    assert state["posts"] == ["maximum_charge", "maximum_tokens", "wall_time_seconds"]
    assert state["caps"] == {
        "spend_micros": 2_000_000,
        "tokens": 2000,
        "execution_ms": 7_200_000,
        "pause_retention_ms": 86_400_000,
    }
    assert result.evidence[-1]["held_at_contract_maximum"] == ["pause_retention_ms"]


def test_sdk_raise_caps_retention_at_contract_maximum(contract, tmp_path):
    responder, state = four_limits_api(contract, 50_000_000)  # 2x would be 100000 s > 86400.
    result, _ = run(
        SurfaceName.SDK, responder, tmp_path, Step.RAISE_LIMIT, run_id="run_" + "R" * 22
    )
    assert result.verdict == Verdict.PASS, result
    assert state["posts"][-1] == "pause_retention_seconds"
    assert state["caps"]["pause_retention_ms"] == 86_400_000
    assert result.evidence[-1]["held_at_contract_maximum"] == []


@pytest.mark.parametrize("surface", SURFACES)
def test_raise_limit_on_an_ended_run_is_inconclusive_not_failed(contract, tmp_path, surface):
    result, requests = run(
        surface,
        limits_api(contract, status="completed"),
        tmp_path,
        Step.RAISE_LIMIT,
        run_id="run_" + "R" * 22,
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "precondition_not_met:run_not_running"
    assert all(method == "GET" for method, _, _ in requests)


@pytest.mark.parametrize("surface", SURFACES)
def test_current_month_statement_passes(contract, tmp_path, surface):
    month = datetime.now(UTC).strftime("%Y-%m")
    item = {
        name: MONEY
        for name in (
            "opening_ledger purchases allowance credits conversion charges refunds "
            "adjustments expirations closing_ledger unspent_returned usage_charge"
        ).split()
    }
    item.update(
        month=month,
        period_start=month + "-01T00:00:00Z",
        period_end=month + "-28T00:00:00Z",
        entries=1,
        usage_input_tokens=0,
        usage_output_tokens=0,
    )

    def responder(method, path, _):
        if path.startswith("/v1/billing/statements"):
            return 200, {"has_more": False, "items": [item]}, {}
        return 200, example(contract, "/v1/billing/balance"), {}

    result, _ = run(surface, responder, tmp_path, Step.STATEMENT)
    assert result.verdict == Verdict.PASS, result


@pytest.mark.parametrize("surface", SURFACES)
def test_pause_on_limit_then_resume(contract, tmp_path, surface):
    snapshot = example(contract, "/v1/runs/{run_id}")
    limits = limits_api(contract)
    resumed = {"done": False}
    pause = {
        "cause": "limit",
        "operation_id": "op_fixture",
        "limit": None,
        "requested_at_ms": 1,
        "drained_at_ms": None,
        "mode": "cold",
    }

    def responder(method, path, request):
        if path.endswith("/limits"):
            if method == "POST":
                assert request["body"]["resume"] is True
                resumed["done"] = True
            return limits(method, path, request)
        body = copy.deepcopy(snapshot)
        if resumed["done"]:
            body.update(state="running", pause=None)
        else:
            body.update(state="paused", pause=pause)
        return 200, body, {}

    run_id = "run_" + "L" * 22
    paused, _ = run(surface, responder, tmp_path / "pause", Step.PAUSE_LIMIT, limit_run_id=run_id)
    assert paused.verdict == Verdict.PASS, paused
    result, _ = run(surface, responder, tmp_path / "resume", Step.RESUME, limit_run_id=run_id)
    assert result.verdict == Verdict.PASS, result


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["running", "completed"], (Verdict.PASS, "")),
        (["completed"], (Verdict.INCONCLUSIVE, "precondition_not_met:run_already_terminal")),
        (["foreign_running", "completed"], (Verdict.FAIL, "queue_observation_run_mismatch")),
    ],
)
def test_queued_launch_follows_the_harness_launch(contract, tmp_path, surface, states, expected):
    run_id = "run_" + "Q" * 22
    snapshot = example(contract, "/v1/runs/{run_id}")
    snapshot.update(run_id=run_id, state="queued")
    snapshot["queue"] = {
        "position": 2,
        "ahead": 1,
        "estimated_start_at": "2026-11-18T05:00:00Z",
        "expires_at": "2026-11-18T05:10:00Z",
    }
    client = "mcp" if surface == SurfaceName.MCP_STDIO else surface.value
    folder = tmp_path / f"public-client-{client}"
    folder.mkdir()
    (folder / "launch-row.json").write_text(
        json.dumps({"status": 202, "client": client, "run_id": run_id, "body": snapshot})
    )
    remaining = list(states)

    def responder(method, path, _):
        body = copy.deepcopy(snapshot)
        state = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        body.update(state=state, queue=None)
        if state == "foreign_running":
            body.update(state="running", run_id="run_" + "X" * 22)
        return 200, body, {}

    result, _ = run(
        surface,
        responder,
        tmp_path / "run",
        Step.QUEUED_LAUNCH,
        run_id=run_id,
        queued_journal=folder,
    )
    assert (result.verdict, result.cause) == expected, result


@pytest.mark.parametrize(
    ("field", "value"), [("status", 422), ("client", "cli"), ("selected_run", "run_other")]
)
def test_queue_rejects_unbound_launch_before_polling(contract, tmp_path, field, value):
    snapshot = example(contract, "/v1/runs/{run_id}")
    run_id = snapshot["run_id"]
    snapshot.update(state="queued", queue={"position": 1})
    row = {"body": snapshot, "client": "sdk", "status": 202, "run_id": run_id}
    if field != "selected_run":
        row[field] = value
    (tmp_path / "launch-row.json").write_text(json.dumps(row))

    def responder(method, path, request):
        return 200, {**snapshot, "state": "completed", "queue": None}, {}

    result, requests = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.QUEUED_LAUNCH,
        run_id=value if field == "selected_run" else run_id,
        queued_journal=tmp_path,
    )
    assert result.verdict == Verdict.FAIL
    assert requests == []


@pytest.mark.parametrize("surface", SURFACES)
def test_buy_credits_completes_stripe_test_checkout_through_hook(contract, tmp_path, surface):
    completed: list[str] = []

    def pay(url):
        completed.append(url)
        return None

    responder, url = checkout_responder(contract, completed)
    result, requests = run(
        surface, responder, tmp_path, Step.BUY_CREDITS_TEST, complete_checkout=pay
    )
    assert completed == [url]
    assert result.verdict == Verdict.PASS, result
    paths = [(method, path.split("?")[0]) for method, path, _ in requests]
    assert paths.index(("POST", "/v1/billing/checkout")) < paths.index(
        ("GET", "/v1/billing/checkouts/chk_AAAAAAAAQACAAAAAAAAAAQ")
    )


def buy(contract, tmp_path, surface=SurfaceName.SDK, **fixture):
    completed: list[str] = []

    def pay(url):
        completed.append(url)
        return None

    responder, _ = checkout_responder(contract, completed, **fixture)
    result, _ = run(surface, responder, tmp_path, Step.BUY_CREDITS_TEST, complete_checkout=pay)
    return result, responder.state


def notes_of(result):
    return result.evidence[-1]


# 08:06Z staging shape: a live run settled 0.06426823 during the mcp_stdio buy.
SETTLEMENT = ledger_entry("led_settle", "settle", "-0.06426823", "2026-11-19T08:06:01Z")


@pytest.mark.parametrize("surface", SURFACES)
def test_buy_credits_reconciles_a_concurrent_settlement_exactly(contract, tmp_path, surface):
    result, state = buy(
        contract, tmp_path, surface, gain="9.93573177", entries=[PURCHASE_GRANT, SETTLEMENT]
    )
    assert result.verdict == Verdict.PASS, result
    notes = notes_of(result)
    assert [row["entry_id"] for row in notes["entries"]] == ["led_purchase", "led_settle"]
    assert notes["purchase_grant_entry_ids"] == ["led_purchase"]
    assert notes["interval_reads"][-1]["interval_sum"] == "9.93573177"
    # Every page was read on this driver's own surface, with the interval bounds.
    assert len(state["history"]) == 2 and state["history"][1]["cursor"] == "1"
    assert all({"since", "until"} <= set(query) for query in state["history"])


def test_same_gain_without_the_debit_entry_fails(contract, tmp_path):
    result, _ = buy(contract, tmp_path, gain="9.93573177")
    assert result.verdict == Verdict.FAIL
    assert result.cause == "ledger_interval_unreconciled"
    assert len(notes_of(result)["interval_reads"]) == 2  # One consistent re-read, then Fail.


def test_purchase_grant_of_9_99_fails(contract, tmp_path):
    grant = ledger_entry(
        "led_short", "grant", "9.99", "2026-11-19T08:06:09Z", funding_class="purchased"
    )
    result, _ = buy(contract, tmp_path, gain="9.99", entries=[grant])
    assert result.verdict == Verdict.FAIL
    assert result.cause == "balance_not_credited"


def test_two_purchase_grants_in_the_window_fail(contract, tmp_path):
    second = dict(PURCHASE_GRANT, entry_id="led_purchase_2", created_at="2026-11-19T08:06:10Z")
    result, _ = buy(contract, tmp_path, gain="20", entries=[PURCHASE_GRANT, second])
    assert result.verdict == Verdict.FAIL
    assert result.cause == "purchase_grant_ambiguous"


def test_purchase_grant_outside_the_checkout_window_fails(contract, tmp_path):
    early = dict(PURCHASE_GRANT, created_at="2026-11-19T08:04:59Z")
    result, _ = buy(contract, tmp_path, entries=[early])
    assert result.verdict == Verdict.FAIL
    assert result.cause == "balance_not_credited"


def test_unattributed_grant_class_fails(contract, tmp_path):
    operator = dict(PURCHASE_GRANT, funding_class="operator")
    result, _ = buy(contract, tmp_path, entries=[operator])
    assert result.cause == "balance_not_credited"


def test_no_concurrent_debits_and_gain_10_passes(contract, tmp_path):
    result, _ = buy(contract, tmp_path)
    assert result.verdict == Verdict.PASS, result
    assert notes_of(result)["ledger_gain_usd"] == "10"
    assert len(notes_of(result)["interval_reads"]) == 1


def test_closing_edge_race_is_resolved_by_one_consistent_reread(contract, tmp_path):
    late = ledger_entry("led_late", "settle", "-0.5", "2026-11-19T08:06:20Z")
    reads = [("10", [PURCHASE_GRANT, late]), ("9.5", [PURCHASE_GRANT, late])]
    result, _ = buy(contract, tmp_path, reads=reads)
    assert result.verdict == Verdict.PASS, result
    assert [read["interval_sum"] for read in notes_of(result)["interval_reads"]] == ["9.5", "9.5"]
    assert [read["ledger_after"] for read in notes_of(result)["interval_reads"]] == [
        "310",
        "309.5",
    ]


def test_buy_credits_without_hook_stays_inconclusive(contract, tmp_path):
    responder, _ = checkout_responder(contract, [])
    result, _ = run(SurfaceName.SDK, responder, tmp_path, Step.BUY_CREDITS_TEST)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "checkout_not_completed:stripe_test_completion_unavailable"


def test_buy_credits_typed_hook_failure_is_inconclusive(contract, tmp_path):
    responder, _ = checkout_responder(contract, [])
    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.BUY_CREDITS_TEST,
        complete_checkout=lambda _url: "stripe_test_checkout_not_redirected",
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "checkout_not_completed:stripe_test_checkout_not_redirected"


def test_stripe_completer_refuses_live_or_foreign_pages():
    from scripts.acceptance.stripe_checkout import complete_test_checkout, test_mode_url

    assert test_mode_url("https://checkout.stripe.com/c/pay/cs_test_abc#frag")
    for url in (
        "https://checkout.stripe.com/c/pay/cs_live_abc",
        "http://checkout.stripe.com/c/pay/cs_test_abc",
        "https://evil.example/c/pay/cs_test_abc",
    ):
        assert complete_test_checkout(url) == "stripe_checkout_not_test_mode"
