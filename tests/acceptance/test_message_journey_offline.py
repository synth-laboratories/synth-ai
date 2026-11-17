"""Scripted-surface checks only: no SDK transport, children, browser or slot.

These receipts are component evidence and must never be installed gate inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager

import pytest

from scripts.acceptance import message_journey as messages
from scripts.acceptance.journeys import JourneyContext, Secret, Step, Verdict, run_step
from scripts.acceptance.message_fixture import MESSAGE_BYTES_MAX
from scripts.acceptance.surfaces import CLI_ARGV, SDK_CALLS, Call, Outcome, SurfaceName

QUESTION = "Which artifact was selected in the recorded fixture?"
ANSWER = "The recorded fixture selected artifact alpha."
RUN = "run_fixture"
MESSAGE = "message_fixture"
RECORDING = "a" * 64


class ScriptedSurface:
    name = SurfaceName.SDK

    def __init__(self, *, foreign=False):
        self.foreign = foreign
        self.calls = []
        self.sends = 0
        self.changed_id = False
        self.position = 17
        self.reply_body = {"kind": "available", "text": ANSWER}
        self.leak_internal = False
        self.leak_existence = False
        self.retry_after = 120
        self.rate_limit = "run_messages_per_hour"
        self.refusal_outcome = "not_accepted"
        self.preexisting_question = False

    def problem(self, operation, code, status, run_id, **extra):
        body = {
            "status": status,
            "code": code,
            "instance": f"/runs/{run_id}",
            "outcome": self.refusal_outcome,
            **extra,
        }
        return Outcome(
            operation,
            False,
            status=status,
            code=code,
            problem=body,
            retryable=extra.get("retryable"),
            retry_after=self.retry_after,
        )

    def message(self, *, queued=False):
        return {
            "message_id": MESSAGE,
            "run_id": RUN,
            "kind": "question",
            "sender": {"kind": "customer"},
            "body": {"kind": "available", "text": QUESTION},
            "state": {"kind": "queued"}
            if queued
            else {
                "kind": "disposed",
                "decision_step": self.position,
                "disposition": {"kind": "answered"},
            },
        }

    def call(self, call):
        self.calls.append(call)
        run = call.path.get("run_id")
        if self.foreign:
            code = (
                "resource_not_found"
                if self.leak_existence and run == "run_absent"
                else "run_not_found"
            )
            return self.problem(call.operation, code, 404, run)
        if call.operation == "getRun":
            return Outcome(
                call.operation,
                True,
                {"run_id": run, "state": "completed" if run == "run_terminal" else "running"},
            )
        if call.operation == "sendRunMessage":
            if run == "run_terminal":
                return self.problem(call.operation, "run_not_accepting_messages", 409, run)
            if len(str(call.body["body"]).encode()) > MESSAGE_BYTES_MAX:
                return self.problem(call.operation, "run_message_too_large", 422, run)
            if run == "run_rate":
                return self.problem(
                    call.operation,
                    "rate_limit_exceeded",
                    429,
                    run,
                    retryable=True,
                    overload={
                        "code": "RateLimited",
                        "limit": self.rate_limit,
                        "service": "core-api",
                        "retryable": True,
                        "retry_after": 120,
                    },
                )
            self.sends += 1
            value = self.message(queued=True)
            if self.changed_id and self.sends > 1:
                value["message_id"] = "another_message"
            return Outcome(call.operation, True, value)
        if call.operation == "listRunEvents":
            return Outcome(
                call.operation,
                True,
                {
                    "has_more": False,
                    "resume_cursor": "next",
                    "items": [
                        {
                            "kind": "run.message.disposed",
                            "payload": {"message_id": MESSAGE, "state": self.message()["state"]},
                        }
                    ],
                },
            )
        if call.operation == "getRunMessage":
            return Outcome(call.operation, True, self.message())
        if call.operation == "listRunMessages":
            if self.sends == 0 and not self.preexisting_question:
                return Outcome(call.operation, True, {"items": [], "has_more": False})
            items = [
                self.message(),
                {
                    "message_id": "reply_fixture",
                    "reply_to_message_id": MESSAGE,
                    "sender": {"kind": "manager"},
                    "body": self.reply_body,
                },
            ]
            if self.leak_internal:
                items.append({"message_id": "internal_fixture", "sender": {"kind": "actor"}})
            return Outcome(call.operation, True, {"items": items, "has_more": False})
        raise AssertionError(call.operation)


@pytest.fixture
def context(tmp_path):
    surface = ScriptedSurface()
    foreign = ScriptedSurface(foreign=True)

    @contextmanager
    def make(_credential):
        yield foreign

    path = tmp_path / "message-fixture.json"
    value = {
        "schema": "synth.journeys.message-fixture.v1",
        "run_id": RUN,
        "recording_sha256": RECORDING,
        "question": QUESTION,
        "question_sha256": messages.sha256(QUESTION),
        "answer_sha256": messages.sha256(ANSWER),
        "delivery_decision_step": 17,
        "event_cursor": "recorded-ready-cursor",
        "internal_message_ids": ["internal_fixture"],
        "terminal_run_id": "run_terminal",
        "rate_limited_run_id": "run_rate",
        "absent_run_id": "run_absent",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(value, output)
    ctx = JourneyContext(
        surface,
        SurfaceName.SDK,
        "http://127.0.0.1:1",
        make,
        tmp_path,
        run_id=RUN,
        secrets={Secret.FOREIGN: "fixture-foreign"},
        message_fixture=path,
        message_recording_sha256=RECORDING,
        message_fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return ctx, surface, foreign


def test_recorded_question_checks_position_idempotency_reply_and_redacts_text(context):
    ctx, surface, _ = context
    result = messages.message_question(ctx)
    assert result.verdict == Verdict.PASS
    sends = [call for call in surface.calls if call.operation == "sendRunMessage"]
    assert len(sends) == 2 and sends[0].idempotency_key == sends[1].idempotency_key
    event = next(call for call in surface.calls if call.operation == "listRunEvents")
    assert event.query["cursor"] == "recorded-ready-cursor"
    retained = json.dumps(result.to_json())
    assert QUESTION not in retained and ANSWER not in retained
    assert "delivery_decision_step" in retained and RECORDING in retained


@pytest.mark.parametrize(
    "change,cause",
    [
        ("position", "message_delivery_position_changed"),
        ("changed_id", "message_idempotency_changed"),
    ],
)
def test_changed_recording_position_or_idempotency_cannot_pass(context, change, cause):
    ctx, surface, _ = context
    setattr(surface, change, 18 if change == "position" else True)
    result = messages.message_question(ctx)
    assert result.verdict == Verdict.FAIL and result.cause == cause


@pytest.mark.parametrize(
    "body,verdict",
    [(None, Verdict.FAIL), ({"kind": "unavailable", "reason": "retention"}, Verdict.INCONCLUSIVE)],
)
def test_missing_body_never_becomes_an_empty_successful_answer(context, body, verdict):
    ctx, surface, _ = context
    surface.reply_body = body
    assert messages.message_question(ctx).verdict == verdict


def test_local_timeout_does_not_cancel_and_retains_accepted_identity(context, monkeypatch):
    ctx, surface, _ = context
    monkeypatch.setattr(messages, "WAIT_SECONDS_MAX", 0)
    result = messages.message_question(ctx)
    assert result.verdict == Verdict.INCONCLUSIVE and result.cause == "message_reply_timeout"
    assert ctx.state["message_id"] == MESSAGE
    assert not any(call.operation == "cancelRun" for call in surface.calls)


def test_recording_mismatch_or_missing_capability_stops_before_mutation(context, monkeypatch):
    ctx, surface, _ = context
    ctx.message_recording_sha256 = "b" * 64
    assert messages.message_question(ctx).cause == "message_recording_identity_mismatch"
    assert surface.calls == []

    monkeypatch.delitem(SDK_CALLS, "sendRunMessage", raising=False)
    assert run_step(ctx, Step.MESSAGE_QUESTION).verdict == Verdict.INCONCLUSIVE
    assert surface.calls == []


def test_fixture_cannot_change_expected_position_after_identity_capture(context):
    ctx, surface, _ = context
    value = json.loads(ctx.message_fixture.read_text())
    value["delivery_decision_step"] = 18
    ctx.message_fixture.write_text(json.dumps(value))
    surface.position = 18
    result = messages.message_question(ctx)
    assert result.verdict == Verdict.FAIL and result.cause == "message_fixture_invalid"
    assert surface.calls == []


def test_preexisting_answer_cannot_stand_in_for_a_new_mid_run_send(context):
    ctx, surface, _ = context
    surface.preexisting_question = True
    result = messages.message_question(ctx)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert not any(call.operation == "sendRunMessage" for call in surface.calls)


def test_terminal_and_utf8_byte_limit_refusals(context):
    ctx, surface, _ = context
    assert messages.message_terminal(ctx).verdict == Verdict.PASS
    assert messages.message_oversize(ctx).verdict == Verdict.PASS
    oversized = surface.calls[-1].body["body"]
    assert len(oversized) < 8192 < len(oversized.encode())


def test_rate_refusal_requires_the_message_bucket_and_matching_retry_after(context):
    ctx, surface, _ = context
    assert messages.message_rate(ctx).verdict == Verdict.PASS
    surface.retry_after = 119
    assert messages.message_rate(ctx).cause == "message_rate_retry_after_mismatch"
    surface.rate_limit = "unrelated_global_rate"
    assert messages.message_rate(ctx).cause == "message_rate_overload_shape_changed"
    surface.rate_limit = "run_messages_per_hour"
    surface.retry_after = 120
    surface.refusal_outcome = "unknown"
    assert messages.message_rate(ctx).cause == "message_rate_not_retryable"


def test_cross_org_is_compared_to_absence_and_internal_threads_never_leak(context):
    ctx, surface, foreign = context
    assert messages.message_question(ctx).verdict == Verdict.PASS
    assert messages.message_isolation(ctx).verdict == Verdict.PASS
    foreign.leak_existence = True
    assert messages.message_isolation(ctx).cause == "message_cross_org_existence_leaked"
    foreign.leak_existence = False
    surface.leak_internal = True
    assert messages.message_isolation(ctx).cause == "internal_thread_message_leaked"


def test_event_surface_adapters_keep_cursor_and_limit():
    calls = []

    class Client:
        @property
        def swarms(self):
            return self

        def events(self, run_id, **query):
            calls.append((run_id, query))

    call = Call("listRunEvents", path={"run_id": RUN}, query={"cursor": "checkpoint", "limit": 100})
    SDK_CALLS["listRunEvents"](Client(), call)
    assert calls == [(RUN, {"cursor": "checkpoint", "limit": 100})]
    assert CLI_ARGV["listRunEvents"](call, None) == [
        "logs",
        RUN,
        "--limit",
        "100",
        "--cursor",
        "checkpoint",
    ]
