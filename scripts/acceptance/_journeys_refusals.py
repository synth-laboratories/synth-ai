"""J-refusals, overload, rate window, and queued launch steps for SYN-5695."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING

import httpx

from ._journeys_common import (
    API_KEY_INVALID,
    OVERLOAD_CODES,
    TERMINAL,
    JourneyContext,
    Secret,
    Step,
    StepReceipt,
    Verdict,
    credential_works,
    expect_refusal,
    key,
    launch_body,
    missing,
    probe_launch,
    receipt,
    wait_state,
)
from .surfaces import (
    CLI_ARGV_TOO_LONG,
    Call,
    CLISurface,
    MCPSurface,
    Outcome,
    SurfaceName,
    comparable,
    obj,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

CLIENT_REFUSALS = frozenset(
    {"launch_request_too_large", "invalid_launch_request", "client_error", "tool_arguments_invalid"}
)
ADMISSION_QUEUE_FULL = "admission_queue_full"
ADMISSION_PROBE_CHARGE = "2.50"

# Core's per-organization launch window (core public/operation_limits.go,
# organizationLaunchesPerMinute = 10). Every authenticated POST /v1/runs counts
# before the handler runs, so empty bodies (422 before any effect) fill it.
LAUNCH_WINDOW_FILL = 12
VERSION_REFUSALS = frozenset({"optimizer_version_unavailable", "optimizer_topology_unavailable"})


def oversize(context: JourneyContext) -> StepReceipt:
    body = launch_body("x" * (1_048_576 + 1))
    if isinstance(context.surface, CLISurface):
        outcome = context.surface.call(Call("createRun", body=body, idempotency_key=key("cli")))
        if outcome.cause == CLI_ARGV_TOO_LONG:
            return receipt(
                context,
                Step.OVERSIZE,
                Verdict.INCONCLUSIVE,
                "capability_missing:cli_goal_only_argv",
                outcome,
            )
        return expect_refusal(context, Step.OVERSIZE, outcome, codes=CLIENT_REFUSALS)
    return probe_launch(
        context,
        Step.OVERSIZE,
        body,
        frozenset(
            {
                "launch_request_too_large",
                "invalid_launch_request",
                "client_error",
                "tool_arguments_invalid",
            }
        ),
    )


def topology(context: JourneyContext) -> StepReceipt:
    body = launch_body(
        "SYN-5695 topology probe", optimizer={"topology": "no_such_topology", "version": 1}
    )
    return probe_launch(
        context,
        Step.TOPOLOGY,
        body,
        frozenset(
            {
                "optimizer_topology_unavailable",
                "invalid_launch_request",
                "client_error",
                "tool_arguments_invalid",
            }
        ),
    )


def topology_version(context: JourneyContext) -> StepReceipt:
    body = launch_body(
        "SYN-5695 version probe", optimizer={"topology": "swarms_v2", "version": 999}
    )
    return probe_launch(
        context,
        Step.TOPOLOGY_VERSION,
        body,
        frozenset({"optimizer_version_unavailable", "optimizer_topology_unavailable"}),
    )


def index(context: JourneyContext) -> StepReceipt:
    """Index is not a Swarms surface; each client must refuse it readably."""
    step = Step.INDEX
    if context.driver == SurfaceName.SDK:
        # The SDK has no Index method; the route itself must refuse with a Problem.
        key_ = context.secrets.get("SYNTH_API_KEY", "")
        with httpx.Client(timeout=30) as client:
            response = client.get(
                context.origin + "/v1/index/search",
                params={"q": "syn5695"},
                headers={"Authorization": "Bearer " + key_},
            )
        try:
            problem = response.json()
        except ValueError:
            problem = {}
        code = problem.get("code") if isinstance(problem, dict) else None
        outcome = Outcome(
            "index_search",
            response.is_success,
            status=response.status_code,
            code=code if isinstance(code, str) else None,
            problem=comparable(problem) if isinstance(problem, dict) else None,
        )
        return expect_refusal(
            context, step, outcome, codes=frozenset({"index_not_available"}), status=404
        )
    if isinstance(context.surface, CLISurface):
        outcome = context.surface.run("index_search", ["index", "search", "syn5695"])
        return expect_refusal(context, step, outcome, codes=frozenset({"index_not_available"}))
    assert isinstance(context.surface, MCPSurface)
    listed = "index_search" in context.surface.available
    outcome = context.surface.caller.call("index_search", {"query": "syn5695"})
    ok = outcome.is_error and not listed
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "index_tool_answered",
        notes={"listed": listed, "is_error": outcome.is_error},
    )


def expired_key(context: JourneyContext) -> StepReceipt:
    expired = context.secrets.get(Secret.EXPIRED)
    if expired is None:
        return missing(context, Step.EXPIRED_KEY, "expired_api_key")
    outcome = credential_works(context, expired)
    return expect_refusal(
        context, Step.EXPIRED_KEY, outcome, codes=frozenset({API_KEY_INVALID}), status=401
    )


def admission_overload_refusal(context: JourneyContext) -> StepReceipt:
    """Hosted overload: the org's admission queue is full (running cap 0 and
    every waiting place taken), so a valid launch must be refused 429
    run_queue_full with retry advice and no accepted effect. The org launch
    window's 429 rate_limit_exceeded is a different refusal and does not pass.
    """
    step = Step.OVERLOAD_REFUSAL
    # A valid budget (the J1 producer default) so Core reaches admission;
    # PROBE_CHARGE is refused run_budget_below_call_bound before the queue.
    body: dict[str, JsonValue] = {
        "message": "SYN-5696 hosted overload probe (refusal expected)",
        "limits": {"maximum_charge": {"amount": ADMISSION_PROBE_CHARGE, "currency": "USD"}},
    }
    saturated = context.secrets.get(Secret.OVERLOAD_API_KEY)
    with ExitStack() as stack:
        surface = (
            context.surface if saturated is None else stack.enter_context(context.make(saturated))
        )
        outcome = surface.call(Call("createRun", body=body, idempotency_key=key("overload")))
        if outcome.accepted:
            run = outcome.value.get("run_id") if isinstance(outcome.value, dict) else None
            cancelled = (
                surface.call(
                    Call("cancelRun", path={"run_id": str(run)}, idempotency_key=key("cancel"))
                )
                if isinstance(run, str)
                else Outcome("cancelRun", False, cause="accepted run id absent")
            )
            return receipt(
                context, step, Verdict.FAIL, "refusal_expected_but_accepted", outcome, cancelled
            )
    result = expect_refusal(
        context, step, outcome, codes=frozenset({"run_queue_full"}), status=None
    )
    if result.verdict != Verdict.PASS:
        return result
    if outcome.status not in (None, 429):
        return receipt(context, step, Verdict.FAIL, "refusal_status_unexpected", outcome)
    if outcome.retryable is not True and outcome.retry_after is None:
        return receipt(context, step, Verdict.FAIL, "retry_advice_absent", outcome)
    return result


def overload_refusal(context: JourneyContext) -> StepReceipt:
    if context.secrets.get(Secret.OVERLOAD_MODE) == ADMISSION_QUEUE_FULL:
        return admission_overload_refusal(context)
    # The orchestrator holds the inbound bulkhead for longer than the client's
    # retry budget, so the client must surface a typed retryable refusal.
    outcome = context.surface.call(Call("getAccount"))
    result = expect_refusal(context, Step.OVERLOAD_REFUSAL, outcome, codes=OVERLOAD_CODES)
    if result.verdict == Verdict.PASS and (
        outcome.retry_after is None and outcome.retryable is not True
    ):
        return receipt(context, Step.OVERLOAD_REFUSAL, Verdict.FAIL, "retry_after_absent", outcome)
    return result


def rate_limit_refusal(context: JourneyContext) -> StepReceipt:
    """SYN-5723 retryable 429: fill the org launch window, then launch through the client.

    The probe launch reaches Core but is refused before any effect (unknown
    optimizer version), so a client that honours Retry-After and retries ends
    on that refusal no sooner than Retry-After; a client whose budget is shorter
    surfaces the typed retryable 429. Either is a Pass; an early retry is not.
    """
    step = Step.RATE_LIMIT_REFUSAL
    api_key = context.secrets.get("SYNTH_API_KEY", "")
    statuses: list[int] = []
    retry_after: int | None = None
    with httpx.Client(timeout=30) as client:
        for _ in range(LAUNCH_WINDOW_FILL):
            response = client.post(
                context.origin + "/v1/runs",
                json={},
                headers={"Authorization": "Bearer " + api_key, "Idempotency-Key": key("fill")},
            )
            statuses.append(response.status_code)
            if response.status_code == 429:
                value = response.headers.get("Retry-After", "")
                retry_after = int(value) if value.isdigit() else None
                break
    notes: dict[str, JsonValue] = {"fill_statuses": list(statuses), "retry_after": retry_after}
    if statuses[-1:] != [429] or retry_after is None:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:launch_window_not_saturated",
            notes=notes,
        )
    body = launch_body("SYN-5723 rate probe", optimizer={"topology": "swarms_v2", "version": 999})
    started = time.monotonic()
    outcome = context.surface.call(Call("createRun", body=body, idempotency_key=key("rate")))
    notes["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if outcome.accepted:
        # An accepted probe is a defect; stop it before it spends more.
        run = outcome.value.get("run_id") if isinstance(outcome.value, dict) else None
        cancelled = (
            context.surface.call(
                Call("cancelRun", path={"run_id": str(run)}, idempotency_key=key("cancel"))
            )
            if isinstance(run, str)
            else Outcome("cancelRun", False, cause="accepted run id absent")
        )
        return receipt(
            context,
            step,
            Verdict.FAIL,
            "refusal_expected_but_accepted",
            outcome,
            cancelled,
            notes=notes,
        )
    if outcome.code == "rate_limit_exceeded":
        ok = outcome.status == 429 and outcome.retryable is True and outcome.retry_after is not None
        return receipt(
            context,
            step,
            Verdict.PASS if ok else Verdict.FAIL,
            "" if ok else "rate_limit_not_retryable",
            outcome,
            notes=notes,
        )
    if outcome.code in VERSION_REFUSALS:
        # Retry-After was read before this call; the window cannot reopen sooner than one
        # second before it, so an earlier answer means the client did not wait.
        waited = time.monotonic() - started >= retry_after - 1
        return receipt(
            context,
            step,
            Verdict.PASS if waited else Verdict.FAIL,
            "" if waited else "retried_before_retry_after",
            outcome,
            notes=notes,
        )
    return receipt(context, step, Verdict.FAIL, "refusal_code_unexpected", outcome, notes=notes)


def overload_recovery(context: JourneyContext) -> StepReceipt:
    outcome = context.surface.call(Call("getAccount"))
    verdict = Verdict.PASS if outcome.accepted else Verdict.FAIL
    return receipt(
        context,
        Step.OVERLOAD_RECOVERY,
        verdict,
        "" if outcome.accepted else "not_recovered",
        outcome,
    )


def queued_launch(context: JourneyContext) -> StepReceipt:
    """Follow a run the cached harness launched (public-client-<kind>/launch-row.json)."""
    step = Step.QUEUED_LAUNCH
    if context.queued_journal is None:
        return missing(context, step, "cached_harness_queued_launch")
    row_path = context.queued_journal / "launch-row.json"
    if not row_path.is_file():
        return missing(context, step, "launch_row")
    row = obj(json.loads(row_path.read_text()))
    snapshot = obj(row.get("body"))
    run = row.get("run_id")
    queue = obj(snapshot.get("queue"))
    client = "mcp" if context.driver == SurfaceName.MCP_STDIO else context.driver.value
    if row.get("status") != 202 or row.get("client") != client:
        return receipt(context, step, Verdict.FAIL, "launch_row_client_or_status_mismatch")
    if context.run_id is None:
        return missing(context, step, "selected_queued_run")
    if (
        not isinstance(run, str)
        or not run
        or run != context.run_id
        or snapshot.get("run_id") != run
    ):
        return receipt(context, step, Verdict.FAIL, "launch_row_run_mismatch")
    if snapshot.get("state") != "queued" or not queue:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:capacity_not_saturated",
            notes={"acknowledged_state": snapshot.get("state")},
        )
    outcome, seen = wait_state(
        context, run, lambda view: view.get("run_id") != run or view.get("state") in TERMINAL
    )
    if not outcome.accepted or obj(outcome.value).get("run_id") != run:
        return receipt(context, step, Verdict.FAIL, "queue_observation_run_mismatch", outcome)
    position = queue.get("position")
    notes: dict[str, JsonValue] = {"run_id": run, "queue": queue, "states": list(seen)}
    if type(position) is not int or position < 1:
        return receipt(context, step, Verdict.FAIL, "queue_position_invalid", outcome, notes=notes)
    estimate = queue.get("estimated_start_at")
    if not isinstance(estimate, str) or not estimate:
        return receipt(
            context, step, Verdict.INCONCLUSIVE, "queue_estimate_unobserved", outcome, notes=notes
        )
    if seen[-1:] and seen[-1] in TERMINAL and "running" in seen:
        return receipt(context, step, Verdict.PASS, "", outcome, notes=notes)
    if seen and seen[0] in TERMINAL:
        # This driver's turn came after the run ended; its progression is unseen.
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:run_already_terminal",
            outcome,
            notes=notes,
        )
    return receipt(
        context, step, Verdict.FAIL, "queued_progression_unobserved", outcome, notes=notes
    )
