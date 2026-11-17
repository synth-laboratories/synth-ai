"""Recorded mid-run message steps for the existing SYN-5695 journey framework.

Fixture text stays in its private input file and HTTP transport. Receipts retain
identities, hashes and transition positions only. This module launches no run,
changes no limits and never cancels work when a local wait expires.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Named Optional at acceptance boundaries.
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

from pydantic import JsonValue

from .journeys import (
    POLL_SECONDS,
    JourneyContext,
    Secret,
    Step,
    StepReceipt,
    Verdict,
    expect_refusal,
    missing,
    receipt,
)
from .message_fixture import MessageFixture, sha256
from .surfaces import Call, Outcome, Surface, comparable, obj

WAIT_SECONDS_MAX = 120
EVENT_PAGES_MAX = 256
MESSAGE_PAGES_MAX = 128
PAGE_ITEMS = 100
TERMINAL = frozenset({"completed", "partial", "failed", "cancelled"})


def fixture(context: JourneyContext, step: Step) -> MessageFixture | StepReceipt:
    if (
        context.message_fixture is None
        or context.message_recording_sha256 is None
        or context.message_fixture_sha256 is None
    ):
        return missing(context, step, "message_fixture_and_recording_identity")
    try:
        value = MessageFixture.read(context.message_fixture, context.message_fixture_sha256)
    except (OSError, ValueError, TypeError):
        return receipt(context, step, Verdict.FAIL, "message_fixture_invalid")
    if value.run_id != context.run_id:
        return receipt(context, step, Verdict.FAIL, "message_fixture_run_mismatch")
    if value.recording_sha256 != context.message_recording_sha256:
        return receipt(context, step, Verdict.FAIL, "message_recording_identity_mismatch")
    return value


def key(value: MessageFixture, case: str) -> str:
    # Stable across replay. The public idempotency scope includes the run path.
    return f"syn5747-{value.recording_sha256}-{case}"


def question_call(value: MessageFixture, run_id: str, case: str) -> Call:
    return Call(
        "sendRunMessage",
        path={"run_id": run_id},
        body={"kind": "question", "text": value.question},
        idempotency_key=key(value, case),
    )


@dataclass(frozen=True, slots=True)
class QuestionSent:
    message_id: str
    accepted: Outcome
    repeated: Outcome


@dataclass(frozen=True, slots=True)
class DeliveryObserved:
    decision_step: int


def message_question(context: JourneyContext) -> StepReceipt:
    step = Step.MESSAGE_QUESTION
    value = fixture(context, step)
    if isinstance(value, StepReceipt):
        return value
    snapshot = context.surface.call(Call("getRun", path={"run_id": value.run_id}))
    if not snapshot.accepted or obj(snapshot.value).get("state") != "running":
        return missing(context, step, "running_recorded_message_run")
    initial = customer_messages(context.surface, value.run_id)
    if isinstance(initial, str):
        return receipt(context, step, Verdict.FAIL, initial)
    if any(obj(item.get("sender")).get("kind") == "customer" for item in initial):
        return missing(context, step, "fresh_run_before_recorded_customer_input")
    sent = _send_question(context, value)
    if isinstance(sent, StepReceipt):
        return sent
    observed = _follow_answer(context, value, sent)
    if isinstance(observed, StepReceipt):
        return observed
    return _read_answer(context, value, sent, observed)


def _send_question(context: JourneyContext, value: MessageFixture) -> QuestionSent | StepReceipt:
    step = Step.MESSAGE_QUESTION
    accepted = context.surface.call(question_call(value, value.run_id, "question"))
    accepted_message = obj(accepted.value)
    message_id = accepted_message.get("message_id")
    if not accepted.accepted or not isinstance(message_id, str) or not message_id:
        return receipt(context, step, Verdict.FAIL, "message_not_accepted", accepted)
    if accepted_message.get("run_id") != value.run_id or accepted_message.get("kind") != "question":
        return receipt(context, step, Verdict.FAIL, "message_identity_or_kind_changed", accepted)
    context.state["message_id"] = message_id
    repeated = context.surface.call(question_call(value, value.run_id, "question"))
    if not repeated.accepted or obj(repeated.value).get("message_id") != message_id:
        return receipt(
            context, step, Verdict.FAIL, "message_idempotency_changed", accepted, repeated
        )
    return QuestionSent(message_id, accepted, repeated)


def _follow_answer(
    context: JourneyContext, value: MessageFixture, sent: QuestionSent
) -> DeliveryObserved | StepReceipt:
    step = Step.MESSAGE_QUESTION
    message_id, accepted = sent.message_id, sent.accepted
    cursor = value.event_cursor
    deadline = time.monotonic() + WAIT_SECONDS_MAX
    found = False
    for _ in range(EVENT_PAGES_MAX):
        if time.monotonic() >= deadline:
            break
        events = context.surface.call(
            Call(
                "listRunEvents",
                path={"run_id": value.run_id},
                query={"cursor": cursor, "limit": PAGE_ITEMS},
            )
        )
        page = obj(events.value)
        items = page.get("items")
        if (
            not events.accepted
            or not isinstance(items, list)
            or type(page.get("has_more")) is not bool
        ):
            return receipt(context, step, Verdict.FAIL, "message_events_unreadable", events)
        for event in items:
            row = obj(event)
            payload = obj(row.get("payload"))
            if row.get("kind") != "run.message.disposed" or payload.get("message_id") != message_id:
                continue
            state = obj(payload.get("state"))
            if (
                state.get("kind") != "disposed"
                or obj(state.get("disposition")).get("kind") != "answered"
            ):
                return receipt(context, step, Verdict.FAIL, "question_not_answered", events)
            if (
                type(state.get("decision_step")) is not int
                or state.get("decision_step") != value.delivery_decision_step
            ):
                return receipt(
                    context, step, Verdict.FAIL, "message_delivery_position_changed", events
                )
            found = True
        if found:
            break
        next_cursor = (
            page.get("next_cursor") if page.get("has_more") is True else page.get("resume_cursor")
        )
        if not isinstance(next_cursor, str) or not next_cursor:
            return receipt(context, step, Verdict.FAIL, "message_event_cursor_absent", events)
        if page.get("has_more") is True and next_cursor == cursor:
            return receipt(context, step, Verdict.FAIL, "message_event_cursor_repeated", events)
        cursor = next_cursor
        if page.get("has_more") is False:
            time.sleep(POLL_SECONDS)
    if not found:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "message_reply_timeout",
            accepted,
            notes={"message_id": message_id, "recording_sha256": value.recording_sha256},
        )
    return DeliveryObserved(value.delivery_decision_step)


def _read_answer(
    context: JourneyContext, value: MessageFixture, sent: QuestionSent, observed: DeliveryObserved
) -> StepReceipt:
    step = Step.MESSAGE_QUESTION
    message_id, accepted, repeated = sent.message_id, sent.accepted, sent.repeated
    read = context.surface.call(
        Call("getRunMessage", path={"run_id": value.run_id, "message_id": message_id})
    )
    read_message = obj(read.value)
    read_state = obj(read_message.get("state"))
    if (
        not read.accepted
        or read_message.get("message_id") != message_id
        or read_message.get("run_id") != value.run_id
        or read_message.get("thread_id") != value.customer_thread_id
        or read_state.get("kind") != "disposed"
        or read_state.get("decision_step") != observed.decision_step
        or obj(read_state.get("disposition")).get("kind") != "answered"
    ):
        return receipt(context, step, Verdict.FAIL, "message_read_does_not_match_disposition", read)
    page = customer_messages(context.surface, value.run_id)
    if isinstance(page, str):
        return receipt(context, step, Verdict.FAIL, page)
    replies = [item for item in page if item.get("reply_to_message_id") == message_id]
    if len(replies) != 1 or obj(replies[0].get("sender")).get("kind") != "manager":
        return receipt(context, step, Verdict.FAIL, "manager_reply_missing_or_duplicated")
    body = obj(replies[0].get("body"))
    if body.get("kind") == "unavailable" and body.get("reason") == "retention":
        return missing(context, step, "recorded_reply_retained")
    text = body.get("text")
    if (
        body.get("kind") != "available"
        or not isinstance(text, str)
        or sha256(text) != value.answer_sha256
    ):
        return receipt(context, step, Verdict.FAIL, "recorded_reply_body_mismatch")
    return receipt(
        context,
        step,
        Verdict.PASS,
        "",
        accepted,
        repeated,
        read,
        notes={
            "message_id": message_id,
            "recording_sha256": value.recording_sha256,
            "fixture_sha256": context.message_fixture_sha256,
            "question_sha256": value.question_sha256,
            "answer_sha256": value.answer_sha256,
            "delivery_decision_step": value.delivery_decision_step,
        },
    )


def customer_messages(surface: Surface, run_id: str) -> list[dict[str, JsonValue]] | str:
    items: list[dict[str, JsonValue]] = []
    cursor: Optional[str] = None
    for _ in range(MESSAGE_PAGES_MAX):
        query: dict[str, JsonValue] = {"limit": PAGE_ITEMS}
        if cursor is not None:
            query["cursor"] = cursor
        result = surface.call(Call("listRunMessages", path={"run_id": run_id}, query=query))
        page = obj(result.value)
        rows = page.get("items")
        if (
            not result.accepted
            or not isinstance(rows, list)
            or type(page.get("has_more")) is not bool
        ):
            return "message_list_unreadable"
        items.extend(obj(item) for item in rows)
        if page.get("has_more") is False:
            return items
        next_cursor = page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            return "message_list_cursor_invalid"
        cursor = next_cursor
    return "message_list_page_bound"


def message_terminal(context: JourneyContext) -> StepReceipt:
    step = Step.MESSAGE_TERMINAL
    value = fixture(context, step)
    if isinstance(value, StepReceipt):
        return value
    if value.terminal_run_id is None:
        return missing(context, step, "terminal_message_run")
    state = context.surface.call(Call("getRun", path={"run_id": value.terminal_run_id}))
    if not state.accepted or obj(state.value).get("state") not in TERMINAL:
        return missing(context, step, "terminal_message_run")
    outcome = context.surface.call(question_call(value, value.terminal_run_id, "terminal"))
    return expect_refusal(
        context, step, outcome, codes=frozenset({"run_not_accepting_messages"}), status=409
    )


def message_oversize(context: JourneyContext) -> StepReceipt:
    step = Step.MESSAGE_OVERSIZE
    value = fixture(context, step)
    if isinstance(value, StepReceipt):
        return value
    # More than8KiB in UTF-8, below8192 characters: verifies the byte limit.
    outcome = context.surface.call(
        Call(
            "sendRunMessage",
            path={"run_id": value.run_id},
            body={"kind": "question", "text": "🙂" * 2049},
            idempotency_key=key(value, "oversize"),
        )
    )
    return expect_refusal(
        context, step, outcome, codes=frozenset({"message_too_large"}), status=422
    )


def message_rate(context: JourneyContext) -> StepReceipt:
    step = Step.MESSAGE_RATE
    value = fixture(context, step)
    if isinstance(value, StepReceipt):
        return value
    if value.rate_limited_run_id is None:
        return missing(context, step, "recorded_rate_limited_message_run")
    outcome = context.surface.call(question_call(value, value.rate_limited_run_id, "rate"))
    checked = expect_refusal(
        context, step, outcome, codes=frozenset({"run_message_rate_limited"}), status=429
    )
    if checked.verdict != Verdict.PASS:
        return checked
    if outcome.retryable is not True or (outcome.problem or {}).get("outcome") != "not_accepted":
        return receipt(context, step, Verdict.FAIL, "message_rate_not_retryable", outcome)
    overload = obj((outcome.problem or {}).get("overload"))
    delay = overload.get("retry_after")
    if (
        overload.get("code") != "RateLimited"
        or overload.get("limit") != "run_messages_per_hour"
        or overload.get("retryable") is not True
        or not isinstance(delay, int)
        or isinstance(delay, bool)
        or delay <= 0
    ):
        return receipt(context, step, Verdict.FAIL, "message_rate_overload_shape_changed", outcome)
    if context.driver.value in {"sdk", "cli"} and (
        outcome.retry_after is None or outcome.retry_after <= 0
    ):
        return receipt(context, step, Verdict.FAIL, "message_rate_retry_after_absent", outcome)
    if context.driver.value in {"sdk", "cli"} and outcome.retry_after != delay:
        return receipt(context, step, Verdict.FAIL, "message_rate_retry_after_mismatch", outcome)
    return checked


def message_isolation(context: JourneyContext) -> StepReceipt:
    step = Step.MESSAGE_ISOLATION
    value = fixture(context, step)
    if isinstance(value, StepReceipt):
        return value
    message_id = context.state.get("message_id")
    foreign = context.secrets.get(Secret.FOREIGN)
    if (
        message_id is None
        or message_id == ""
        or foreign is None
        or foreign == ""
        or not value.internal_message_ids
        or value.absent_run_id is None
    ):
        return missing(context, step, "foreign_key_and_current_internal_message_ids")
    outcomes: list[Outcome] = []
    with context.make(foreign) as surface:
        for call in (
            question_call(value, value.run_id, "foreign"),
            Call("listRunMessages", path={"run_id": value.run_id}),
            Call("getRunMessage", path={"run_id": value.run_id, "message_id": message_id}),
        ):
            outcome = surface.call(call)
            outcomes.append(outcome)
            if not outcome.typed or outcome.status != 404:
                return receipt(
                    context, step, Verdict.FAIL, "message_cross_org_not_hidden", *outcomes
                )
            absent = Call(
                call.operation,
                path={**call.path, "run_id": value.absent_run_id},
                body=call.body,
                query=call.query,
                idempotency_key=call.idempotency_key,
            )
            unknown = surface.call(absent)
            outcomes.append(unknown)
            if (
                not unknown.typed
                or unknown.status != 404
                or outcome.problem is None
                or unknown.problem is None
                or comparable(outcome.problem) != comparable(unknown.problem)
            ):
                return receipt(
                    context, step, Verdict.FAIL, "message_cross_org_existence_leaked", *outcomes
                )
    messages = customer_messages(context.surface, value.run_id)
    if isinstance(messages, str):
        return receipt(context, step, Verdict.FAIL, messages, *outcomes)
    if any(
        item.get("message_id") in value.internal_message_ids
        or item.get("visibility") == "internal"
        or item.get("thread_id") != value.customer_thread_id
        or obj(item.get("sender")).get("kind") not in {"customer", "manager", "actor"}
        for item in messages
    ):
        return receipt(context, step, Verdict.FAIL, "internal_thread_message_leaked", *outcomes)
    return receipt(
        context,
        step,
        Verdict.PASS,
        "",
        *outcomes,
        notes={"excluded_internal_ids": len(value.internal_message_ids)},
    )


MESSAGE_OPERATIONS: Mapping[Step, tuple[str, ...]] = {
    Step.MESSAGE_QUESTION: (
        "getRun",
        "sendRunMessage",
        "listRunEvents",
        "getRunMessage",
        "listRunMessages",
    ),
    Step.MESSAGE_TERMINAL: ("getRun", "sendRunMessage"),
    Step.MESSAGE_OVERSIZE: ("sendRunMessage",),
    Step.MESSAGE_RATE: ("sendRunMessage",),
    Step.MESSAGE_ISOLATION: ("sendRunMessage", "listRunMessages", "getRunMessage"),
}
MESSAGE_STEPS: Mapping[Step, Callable[[JourneyContext], StepReceipt]] = {
    Step.MESSAGE_QUESTION: message_question,
    Step.MESSAGE_TERMINAL: message_terminal,
    Step.MESSAGE_OVERSIZE: message_oversize,
    Step.MESSAGE_RATE: message_rate,
    Step.MESSAGE_ISOLATION: message_isolation,
}
