"""J2 money journey steps for SYN-5695."""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from ._journeys_common import (
    RESUMED,
    TERMINAL,
    JourneyContext,
    Step,
    StepReceipt,
    Verdict,
    items,
    key,
    missing,
    probe_launch,
    receipt,
    wait_state,
)
from .surfaces import Call, Outcome, SDKSurface, obj

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

CHECKOUT_COMPLETION_WAIT_SECONDS = 60.0
CHECKOUT_POLL_SECONDS = 2.0
TERMINAL_CHECKOUT = frozenset({"completed", "failed", "expired"})

# The credits.10 pack. The ledger proof (coordinator ruling 2026-11-19, option B):
# Core exposes no checkout id on ledger entries, so the purchase is the one
# purchased grant of exactly this amount inside the checkout's own window, and
# the whole interval must reconcile entry by entry.
PURCHASE_USD = Decimal("10")
HISTORY_PAGE_LIMIT = 100
HISTORY_MAX_PAGES = 50


def checkout_state(outcome: Outcome) -> Optional[str]:
    state = outcome.value.get("state") if isinstance(outcome.value, dict) else None
    return state if isinstance(state, str) else None


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def instant(value: JsonValue) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def usd(value: JsonValue) -> Optional[Decimal]:
    """Exact USD amount of a Money object, or None (never a float, never another currency)."""
    if not isinstance(value, dict) or value.get("currency") != "USD":
        return None
    try:
        amount = Decimal(str(value.get("amount")))
    except ArithmeticError:
        return None
    return amount if amount.is_finite() else None


def history_interval(
    context: JourneyContext, since: str, until: str
) -> tuple[Optional[list[dict[str, JsonValue]]], Optional[Outcome]]:
    """Every ledger entry with since <= created_at < until, all pages, on this driver's surface."""
    entries: list[dict[str, JsonValue]] = []
    cursor: Optional[str] = None
    page: Optional[Outcome] = None
    for _ in range(HISTORY_MAX_PAGES):
        query: dict[str, JsonValue] = {"since": since, "until": until, "limit": HISTORY_PAGE_LIMIT}
        if cursor is not None:
            query["cursor"] = cursor
        page = context.surface.call(Call("listBillingHistory", query=query))
        value = page.value
        if not page.accepted or not isinstance(value, dict):
            return None, page
        entries += [entry for entry in items(value.get("items")) if isinstance(entry, dict)]
        if value.get("has_more") is not True:
            return entries, page
        next_cursor = value.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return None, page
        cursor = next_cursor
    return None, page


def entry_row(entry: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "entry_id": entry.get("entry_id"),
        "cause": entry.get("cause"),
        "funding_class": entry.get("funding_class"),
        "delta": entry.get("delta"),
        "created_at": entry.get("created_at"),
    }


def buy_credits(context: JourneyContext) -> StepReceipt:
    step = Step.BUY_CREDITS_TEST
    # The interval opens before the first balance read and before the checkout exists.
    since = utc_stamp()
    before = context.surface.call(Call("getBalance"))
    created = context.surface.call(
        Call("createCheckout", body={"pack_id": "credits.10"}, idempotency_key=key("checkout"))
    )
    if not created.accepted or not isinstance(created.value, dict):
        return receipt(context, step, Verdict.FAIL, "checkout_not_created", before, created)
    checkout = Call("getCheckout", path={"id": str(created.value["checkout_id"])})
    url = created.value.get("url")
    if context.complete_checkout is None or not isinstance(url, str):
        status = context.surface.call(checkout)
        state = checkout_state(status)
        cause = "checkout_not_completed:stripe_test_completion_unavailable"
    else:
        failure = context.complete_checkout(url)
        # The webhook lands after Stripe's redirect; poll a bounded window for it.
        deadline = time.monotonic() + CHECKOUT_COMPLETION_WAIT_SECONDS
        while True:
            status = context.surface.call(checkout)
            state = checkout_state(status)
            if failure is not None or state in TERMINAL_CHECKOUT or time.monotonic() >= deadline:
                break
            time.sleep(CHECKOUT_POLL_SECONDS)
        cause = "checkout_not_completed:" + (failure or "webhook_not_observed")
    if state != "completed":
        # Never assume credit: completion is Core's webhook-confirmed state.
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            cause,
            before,
            created,
            status,
            notes={"checkout_state": state},
        )
    return reconcile_purchase(context, step, since, before, created, status)


def reconcile_purchase(
    context: JourneyContext,
    step: Step,
    since: str,
    before: Outcome,
    created: Outcome,
    status: Outcome,
) -> StepReceipt:
    """Pass only if the interval reconciles exactly and holds exactly one purchase grant.

    `ledger` (Balance) moves only by entry `delta`; `held_delta` moves held value
    and is not summed. An entry landing between the closing balance read and the
    `until` stamp is a read race, not a discrepancy: on a mismatch the closing
    edge is re-read ONCE (fresh balance, then a later `until`) and the same exact
    equality must hold. The opening edge (since stamped, then balance read) has
    no re-read: a mismatch there Fails.
    """
    view = status.value if isinstance(status.value, dict) else {}
    opened, closed = instant(view.get("created_at")), instant(view.get("completed_at"))
    window = {
        "checkout_created_at": view.get("created_at"),
        "completed_at": view.get("completed_at"),
    }
    if opened is None or closed is None:
        return receipt(
            context, step, Verdict.FAIL, "checkout_window_unavailable", before, created, status
        )
    ledger_before = usd(obj(before.value).get("ledger")) if before.accepted else None
    reads: list[Mapping[str, JsonValue]] = []
    entries: list[dict[str, JsonValue]] = []
    after: Optional[Outcome] = None
    gain = interval = None
    for _ in range(2):
        after = context.surface.call(Call("getBalance"))
        until = utc_stamp()
        listed, page = history_interval(context, since, until)
        ledger_after = usd(obj(after.value).get("ledger")) if after.accepted else None
        if listed is None or ledger_before is None or ledger_after is None:
            outcomes = (before, created, status, after) + ((page,) if page is not None else ())
            return receipt(context, step, Verdict.FAIL, "ledger_interval_unreadable", *outcomes)
        deltas = [usd(entry.get("delta")) for entry in listed]
        if any(delta is None for delta in deltas):
            return receipt(context, step, Verdict.FAIL, "ledger_entry_unreadable", before, after)
        entries = listed
        gain = ledger_after - ledger_before
        interval = sum((delta for delta in deltas if delta is not None), Decimal(0))
        reads.append(
            {
                "until": until,
                "ledger_after": str(ledger_after),
                "entries": len(listed),
                "interval_sum": str(interval),
            }
        )
        if gain == interval:
            break
    grants = [
        entry
        for entry in entries
        if entry.get("cause") == "grant"
        and entry.get("funding_class") == "purchased"
        and usd(entry.get("delta")) == PURCHASE_USD
        and (stamp := instant(entry.get("created_at"))) is not None
        and opened <= stamp <= closed
    ]
    if gain != interval:
        verdict, cause = Verdict.FAIL, "ledger_interval_unreconciled"
    elif not grants:
        verdict, cause = Verdict.FAIL, "balance_not_credited"
    elif len(grants) > 1:
        verdict, cause = Verdict.FAIL, "purchase_grant_ambiguous"
    else:
        verdict, cause = Verdict.PASS, ""
    assert after is not None
    interval_reads: list[JsonValue] = [dict(row) for row in reads]
    entry_rows: list[JsonValue] = [dict(entry_row(entry)) for entry in entries]
    grant_ids: list[JsonValue] = [str(entry.get("entry_id")) for entry in grants]
    return receipt(
        context,
        step,
        verdict,
        cause,
        before,
        created,
        status,
        after,
        notes={
            "ledger_field": "Balance.ledger vs BillingHistoryEntry.delta (held_delta excluded)",
            "since": since,
            "ledger_before": str(ledger_before),
            "ledger_gain_usd": str(gain),
            "interval_reads": interval_reads,
            "entries": entry_rows,
            "purchase_grant_entry_ids": grant_ids,
            "checkout_window": window,
            "purchase_usd": str(PURCHASE_USD),
        },
    )


def spend_cap(view: Outcome) -> Optional[int]:
    if not isinstance(view.value, dict):
        return None
    for row in items(view.value.get("primary")):
        if isinstance(row, dict) and row.get("dimension") == "spend_micros":
            cap = row.get("effective_cap")
            return cap if isinstance(cap, int) else None
    return None


def raise_limit(context: JourneyContext) -> StepReceipt:
    step = Step.RAISE_LIMIT
    if context.run_id is None:
        return missing(context, step, "running_run")
    if isinstance(context.surface, SDKSurface):
        return raise_four(context)
    path = {"run_id": context.run_id}
    before = context.surface.call(Call("getRunLimits", path=path))
    if before.accepted and obj(before.value).get("status") != "running":
        # The batch's run ended before this driver's turn: not this driver's fault.
        return receipt(
            context, step, Verdict.INCONCLUSIVE, "precondition_not_met:run_not_running", before
        )
    cap = spend_cap(before)
    if cap is None or not isinstance(before.value, dict):
        return receipt(context, step, Verdict.FAIL, "budget_view_unavailable", before)
    target = Decimal(cap * 2) / Decimal(1_000_000)
    body: dict[str, JsonValue] = {
        "expected_revision": before.value["revision"],
        "limits": {"maximum_charge": {"amount": str(target), "currency": "USD"}},
        "reason": "SYN-5695 J2 raise",
    }
    updated = context.surface.call(
        Call("updateRunLimits", path=path, body=body, idempotency_key=key("raise"))
    )
    after = context.surface.call(Call("getRunLimits", path=path))
    observed = spend_cap(after)
    verdict = Verdict.PASS if updated.accepted and observed == cap * 2 else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "limit_raise_not_observed",
        before,
        updated,
        after,
        notes={"spend_cap_before": cap, "spend_cap_after": observed},
    )


def raise_four(context: JourneyContext) -> StepReceipt:
    """SDK: the merged SYN-5663 helper, four raises each seen in the next view."""
    from .live_limits import (
        Dimension,
        LimitTargets,
        LiveLimitsError,
        caps,
        contract_maximum,
        raise_live_limits,
    )

    assert isinstance(context.surface, SDKSurface)
    assert context.run_id is not None
    client = context.surface.client
    try:
        current = caps(client.swarms.limits(context.run_id), context.run_id)
    except LiveLimitsError as error:
        verdict = Verdict.INCONCLUSIVE if error.code == "run_not_running" else Verdict.FAIL
        cause = (
            "precondition_not_met:run_not_running"
            if verdict == Verdict.INCONCLUSIVE
            else error.code
        )
        return receipt(context, Step.RAISE_LIMIT, verdict, cause)

    def seconds_target(dimension: Dimension, field: str) -> Optional[int]:
        # Double, but never past the contract maximum; None (held, asserted
        # unchanged) only when the cap already sits at that maximum.
        maximum = contract_maximum(field)
        current_seconds = current[dimension] // 1000
        return None if current_seconds >= maximum else min(current_seconds * 2, maximum)

    targets = LimitTargets(
        budget_usd=str(Decimal(current[Dimension.SPEND] * 2) / Decimal(1_000_000)),
        maximum_tokens=current[Dimension.TOKENS] * 2,
        wall_time_seconds=seconds_target(Dimension.EXECUTION, "wall_time_seconds"),
        pause_retention_seconds=seconds_target(Dimension.RETENTION, "pause_retention_seconds"),
    )
    directory = context.receipt_dir / "live-limits"
    try:
        views = raise_live_limits(
            client,
            run_id=context.run_id,
            targets=targets,
            mutation_prefix=key("live-limits"),
            receipt_directory=directory,
        )
    except LiveLimitsError as error:
        return receipt(context, Step.RAISE_LIMIT, Verdict.FAIL, error.code)
    return receipt(
        context,
        Step.RAISE_LIMIT,
        Verdict.PASS,
        "",
        notes={
            "helper": "scripts/acceptance/live_limits.py",
            "receipt_directory": str(directory),
            "revisions": [view.revision for view in views],
            "held_at_contract_maximum": [dimension.value for dimension in targets.held()],
        },
    )


def sku_refusal(context: JourneyContext) -> StepReceipt:
    from ._journeys_common import launch_body

    # Reachable whitelist refusal on a cached slot: fork_flush is a public
    # topology the release environment does not admit (run_profile_deferred).
    body = launch_body(
        "SYN-5695 whitelist probe", optimizer={"topology": "fork_flush", "version": 1}
    )
    return probe_launch(
        context, Step.SKU_REFUSAL, body, frozenset({"optimizer_topology_not_allowed"})
    )


def limit_paused(value: Mapping[str, JsonValue]) -> bool:
    pause = value.get("pause")
    return (
        value.get("state") == "paused" and isinstance(pause, dict) and pause.get("cause") == "limit"
    )


def pause_on_limit(context: JourneyContext) -> StepReceipt:
    step = Step.PAUSE_LIMIT
    if context.limit_run_id is None:
        return missing(context, step, "limit_run")
    outcome, seen = wait_state(
        context, context.limit_run_id, lambda v: limit_paused(v) or v.get("state") in TERMINAL
    )
    value = outcome.value if isinstance(outcome.value, dict) else {}
    limits = context.surface.call(Call("getRunLimits", path={"run_id": context.limit_run_id}))
    verdict = Verdict.PASS if limit_paused(value) and limits.accepted else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "run_not_paused_by_limit",
        outcome,
        limits,
        notes={"states": list(seen), "pause": value.get("pause")},
    )


def resume(context: JourneyContext) -> StepReceipt:
    step = Step.RESUME
    if context.limit_run_id is None:
        return missing(context, step, "limit_run")
    path = {"run_id": context.limit_run_id}
    view = context.surface.call(Call("getRunLimits", path=path))
    cap = spend_cap(view)
    if cap is None or not isinstance(view.value, dict):
        return receipt(context, step, Verdict.FAIL, "budget_view_unavailable", view)
    body: dict[str, JsonValue] = {
        "expected_revision": view.value["revision"],
        "limits": {
            "maximum_charge": {"amount": str(Decimal(cap * 2) / 1_000_000), "currency": "USD"}
        },
        "resume": True,
        "reason": "SYN-5695 J2 resume after limit",
    }
    updated = context.surface.call(
        Call("updateRunLimits", path=path, body=body, idempotency_key=key("resume"))
    )
    outcome, seen = wait_state(context, context.limit_run_id, lambda v: v.get("state") != "paused")
    state = outcome.value.get("state") if isinstance(outcome.value, dict) else None
    verdict = Verdict.PASS if updated.accepted and state in RESUMED else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "run_not_resumed",
        view,
        updated,
        outcome,
        notes={"states": list(seen)},
    )


def statement(context: JourneyContext) -> StepReceipt:
    page = context.surface.call(Call("listStatements"))
    balance = context.surface.call(Call("getBalance"))
    rows = items(obj(page.value).get("items"))
    month = datetime.now(UTC).strftime("%Y-%m")
    current = [item for item in rows if isinstance(item, dict) and item.get("month") == month]
    verdict = Verdict.PASS if current and balance.accepted else Verdict.FAIL
    return receipt(
        context,
        Step.STATEMENT,
        verdict,
        "" if verdict == Verdict.PASS else "current_statement_absent",
        page,
        balance,
        notes={"month": month, "statements": len(rows)},
    )
