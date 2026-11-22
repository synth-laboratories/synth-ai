"""Scripted-surface checks of the fixture writer: no SDK transport, no slot.

Component evidence only; never an installed gate input.
"""

from __future__ import annotations

import hashlib
import json
import stat
from contextlib import contextmanager

import pytest
from test_message_journey_offline import (
    ANSWER,
    MESSAGE,
    QUESTION,
    RECORDING,
    RUN,
    THREAD,
    ScriptedSurface,
)

from scripts.acceptance import message_journey as messages
from scripts.acceptance.journeys import JourneyContext, Secret, Verdict
from scripts.acceptance.message_fixture import MessageFixture
from scripts.acceptance.message_fixture_writer import (
    FixtureRefused,
    recorded_exchange,
    write_fixture,
)
from scripts.acceptance.surfaces import Outcome, SurfaceName

RECORD_RUN = "run_record"
RECORD_THREAD = "customer_thread_record"


class RecordingSurface:
    """The record run after its exchange, and the fresh replay run beside it."""

    name = SurfaceName.SDK

    def __init__(self):
        self.calls = []
        self.disposition = "answered"
        self.position = 17
        self.replies = 1
        self.reply_body = {"kind": "available", "text": ANSWER}
        self.replay_event_kinds = ["created", "objective.created"]
        self.event_pages = 2

    def history(self):
        question = {
            "message_id": "message_record",
            "run_id": RECORD_RUN,
            "thread_id": RECORD_THREAD,
            "kind": "question",
            "author": "customer",
            "body": {"kind": "available", "text": QUESTION},
            "state": {
                "kind": "disposed",
                "decision_step": self.position,
                "disposition": {"kind": self.disposition},
            },
        }
        other = {
            **question,
            "message_id": "message_other",
            "body": {"kind": "available", "text": "another"},
        }
        reply = {
            "message_id": "reply_record",
            "run_id": RECORD_RUN,
            "thread_id": RECORD_THREAD,
            "reply_to_message_id": "message_record",
            "author": "manager",
            "kind": "reply",
            "state": {"kind": "posted"},
            "body": self.reply_body,
        }
        return [
            other,
            question,
            *[{**reply, "message_id": f"reply_{n}"} for n in range(self.replies)],
        ]

    def call(self, call):
        self.calls.append(call)
        run = call.path.get("run_id")
        if call.operation == "listRunMessages" and run == RECORD_RUN:
            # Two pages, so the reader must follow next_cursor to find the reply.
            rows = self.history()
            first = call.query.get("cursor") is None
            return Outcome(
                call.operation,
                True,
                {
                    "items": rows[:1] if first else rows[1:],
                    "has_more": first,
                    "next_cursor": "page-2" if first else None,
                    "availability": {"kind": "available"},
                },
            )
        if call.operation == "listRunEvents" and run == RUN:
            page = (
                1
                if call.query.get("cursor") is None
                else int(str(call.query["cursor"]).split("-")[1]) + 1
            )
            last = page >= self.event_pages
            body = {
                "items": [{"kind": kind} for kind in self.replay_event_kinds],
                "has_more": not last,
                "next_cursor": None if last else f"cursor-{page}",
                "resume_cursor": "replay-head-cursor" if last else None,
            }
            return Outcome(call.operation, True, body)
        raise AssertionError((call.operation, run))


def seal_default(surface, path, **changes):
    arguments = {
        "record_run_id": RECORD_RUN,
        "question": QUESTION,
        "run_id": RUN,
        "customer_thread_id": THREAD,
        "recording_sha256": RECORDING,
        "internal_message_ids": ["internal_fixture"],
        "terminal_run_id": "run_terminal",
        "rate_limited_run_id": "run_rate",
        "absent_run_id": "run_absent",
    }
    return write_fixture(surface, path, **{**arguments, **changes})


def test_sealed_fixture_binds_the_recorded_exchange_and_the_journey_accepts_it(tmp_path):
    path = tmp_path / "message-fixture.json"
    surface = RecordingSurface()
    sealed, exchange = seal_default(surface, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sealed == hashlib.sha256(path.read_bytes()).hexdigest()
    fixture = MessageFixture.read(path, sealed)
    assert fixture.run_id == RUN and fixture.customer_thread_id == THREAD
    assert fixture.recording_sha256 == RECORDING and fixture.question == QUESTION
    assert fixture.answer_sha256 == messages.sha256(ANSWER) == exchange.answer_sha256
    assert fixture.delivery_decision_step == 17
    assert fixture.event_cursor == "replay-head-cursor"
    assert fixture.internal_message_ids == ("internal_fixture",)
    # The writer only reads: it sends nothing and controls nothing.
    assert {call.operation for call in surface.calls} == {"listRunMessages", "listRunEvents"}
    # The reply text is never written; the question is, as the journey needs it.
    assert ANSWER not in path.read_text()

    replay = ScriptedSurface()

    @contextmanager
    def make(_credential):
        yield ScriptedSurface(foreign=True)

    context = JourneyContext(
        replay,
        SurfaceName.SDK,
        "http://127.0.0.1:1",
        make,
        tmp_path,
        run_id=RUN,
        secrets={Secret.FOREIGN: "fixture-foreign"},
        message_fixture=path,
        message_recording_sha256=RECORDING,
        message_fixture_sha256=sealed,
    )
    result = messages.message_question(context)
    assert result.verdict == Verdict.PASS, result.cause
    followed = next(call for call in replay.calls if call.operation == "listRunEvents")
    assert followed.query["cursor"] == "replay-head-cursor"
    assert MESSAGE in json.dumps(result.to_json())


@pytest.mark.parametrize(
    "change,value,code",
    [
        ("disposition", "acknowledged", "message_recording_question_not_answered"),
        ("replies", 0, "message_recording_reply_missing_or_duplicated"),
        ("replies", 2, "message_recording_reply_missing_or_duplicated"),
        ("position", 0, "message_recording_position_invalid"),
        (
            "reply_body",
            {"kind": "unavailable", "reason": "retention"},
            "message_recording_reply_body_unavailable",
        ),
        (
            "replay_event_kinds",
            ["run.message.accepted"],
            "message_fixture_run_already_has_messages",
        ),
    ],
)
def test_a_recording_without_one_answered_exchange_is_never_sealed(tmp_path, change, value, code):
    path = tmp_path / "message-fixture.json"
    surface = RecordingSurface()
    setattr(surface, change, value)
    with pytest.raises(FixtureRefused) as refused:
        seal_default(surface, path)
    assert refused.value.code == code
    assert not path.exists()


def test_question_must_be_the_recorded_one_and_the_replay_run_a_different_run(tmp_path):
    path = tmp_path / "message-fixture.json"
    with pytest.raises(FixtureRefused) as refused:
        recorded_exchange(RecordingSurface(), RECORD_RUN, "a question nobody asked")
    assert refused.value.code == "message_recording_question_not_unique"
    with pytest.raises(FixtureRefused) as refused:
        seal_default(RecordingSurface(), path, run_id=RECORD_RUN)
    assert refused.value.code == "message_fixture_replay_run_is_the_recording"
    assert not path.exists()


def test_a_sealed_fixture_is_never_replaced_and_bad_identities_leave_no_file(tmp_path):
    path = tmp_path / "message-fixture.json"
    sealed, _ = seal_default(RecordingSurface(), path)
    moved = RecordingSurface()
    moved.position = 18
    with pytest.raises(FixtureRefused) as refused:
        seal_default(moved, path)
    assert refused.value.code == "message_fixture_already_sealed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sealed
    other = tmp_path / "other.json"
    with pytest.raises(FixtureRefused) as refused:
        seal_default(RecordingSurface(), other, recording_sha256="not-a-digest")
    assert refused.value.code == "message_fixture_digest_invalid"
    with pytest.raises(FixtureRefused) as refused:
        seal_default(RecordingSurface(), other, customer_thread_id="")
    assert refused.value.code == "message_fixture_identity_invalid"
    assert not other.exists()
