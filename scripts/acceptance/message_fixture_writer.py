"""Seal the private message fixture that `message_journey` replays (SYN-5745/5747).

The fixture binds one fresh replay run to one recorded exchange: the question a
customer sent to the record run, the decision step at which the manager took it,
and the digest of the manager's reply. The recorded facts are read from the
record run through the same public surface the journeys use, never typed in by
hand, so a recording without an answered question cannot be sealed.

    python -m scripts.acceptance.message_fixture_writer --driver sdk \
      --record-run-id "$RECORD_RUN" --question-file "$PRIVATE_QUESTION" \
      --run-id "$REPLAY_RUN" --customer-thread-id "$REPLAY_THREAD" \
      --recording-sha256 "$RECORDING_SHA256" \
      --internal-message-id "$INTERNAL_ID" --out "$PRIVATE_FIXTURE"

It prints only `{"fixture_sha256", "delivery_decision_step", ...}`; question and
reply text stay in the mode 0600 files. It sends no message, launches no run and
changes no limits. Environment: SYNTH_API_KEY, SYNTH_LOCAL_API_BASE_URL.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Named Optional at acceptance boundaries.
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .message_fixture import FIXTURE_BYTES_MAX, MESSAGE_BYTES_MAX, MessageFixture, digest, sha256
from .surfaces import Call, Surface, SurfaceName, obj

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

SCHEMA = "synth.journeys.message-fixture.v1"
PAGES_MAX = 256
PAGE_ITEMS = 100
LOOPBACK = ("http://127.0.0.1:", "http://localhost:")


class FixtureRefused(ValueError):
    """A typed reason the fixture was not sealed; `code` is safe to print."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class RecordedExchange:
    """What the record run retained about one answered customer question."""

    question: str
    answer_sha256: str
    delivery_decision_step: int


def _messages(surface: Surface, run_id: str) -> list[dict[str, JsonValue]]:
    items: list[dict[str, JsonValue]] = []
    cursor: Optional[str] = None
    for _ in range(PAGES_MAX):
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
            raise FixtureRefused("message_recording_history_unreadable")
        if obj(page.get("availability")) != {"kind": "available"}:
            raise FixtureRefused("message_recording_history_unavailable")
        items.extend(obj(item) for item in rows)
        if page.get("has_more") is False:
            return items
        next_cursor = page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            raise FixtureRefused("message_recording_history_cursor_invalid")
        cursor = next_cursor
    raise FixtureRefused("message_recording_history_unbounded")


def recorded_exchange(surface: Surface, record_run_id: str, question: str) -> RecordedExchange:
    """Read the one answered exchange for `question` from the record run.

    Refuses unless exactly one customer question carries that text, the manager
    disposed it as answered at a positive decision step, and exactly one posted
    manager reply with an available body points at it.
    """
    if not 1 <= len(question.encode("utf-8")) <= MESSAGE_BYTES_MAX:
        raise FixtureRefused("message_fixture_question_invalid")
    history = _messages(surface, record_run_id)
    asked = [
        item
        for item in history
        if item.get("author") == "customer"
        and item.get("kind") == "question"
        and obj(item.get("body")).get("text") == question
    ]
    if len(asked) != 1:
        raise FixtureRefused("message_recording_question_not_unique")
    message_id = asked[0].get("message_id")
    state = obj(asked[0].get("state"))
    if not isinstance(message_id, str) or not message_id:
        raise FixtureRefused("message_recording_question_identity_invalid")
    if state.get("kind") != "disposed" or obj(state.get("disposition")).get("kind") != "answered":
        raise FixtureRefused("message_recording_question_not_answered")
    step = state.get("decision_step")
    if type(step) is not int or step <= 0:
        raise FixtureRefused("message_recording_position_invalid")
    replies = [item for item in history if item.get("reply_to_message_id") == message_id]
    if (
        len(replies) != 1
        or replies[0].get("author") != "manager"
        or replies[0].get("kind") != "reply"
        or obj(replies[0].get("state")) != {"kind": "posted"}
    ):
        raise FixtureRefused("message_recording_reply_missing_or_duplicated")
    body = obj(replies[0].get("body"))
    text = body.get("text")
    if body.get("kind") != "available" or not isinstance(text, str) or not text:
        raise FixtureRefused("message_recording_reply_body_unavailable")
    return RecordedExchange(question, sha256(text), step)


def current_event_cursor(surface: Surface, run_id: str) -> str:
    """The replay run's event cursor at its present head, before any message."""
    cursor: Optional[str] = None
    for _ in range(PAGES_MAX):
        query: dict[str, JsonValue] = {"limit": PAGE_ITEMS}
        if cursor is not None:
            query["cursor"] = cursor
        result = surface.call(Call("listRunEvents", path={"run_id": run_id}, query=query))
        page = obj(result.value)
        items = page.get("items")
        if (
            not result.accepted
            or not isinstance(items, list)
            or type(page.get("has_more")) is not bool
        ):
            raise FixtureRefused("message_fixture_events_unreadable")
        if any(str(obj(item).get("kind", "")).startswith("run.message.") for item in items):
            raise FixtureRefused("message_fixture_run_already_has_messages")
        more = page.get("has_more") is True
        following = page.get("next_cursor") if more else page.get("resume_cursor")
        if not isinstance(following, str) or not following or (more and following == cursor):
            raise FixtureRefused("message_fixture_event_cursor_absent")
        cursor = following
        if not more:
            return cursor
    raise FixtureRefused("message_fixture_events_unbounded")


def seal(
    path: Path,
    *,
    run_id: str,
    customer_thread_id: str,
    recording_sha256: str,
    exchange: RecordedExchange,
    event_cursor: str,
    internal_message_ids: Sequence[str],
    terminal_run_id: Optional[str] = None,
    rate_limited_run_id: Optional[str] = None,
    absent_run_id: Optional[str] = None,
) -> str:
    """Write the fixture once, mode 0600, and return its SHA-256.

    The written bytes are read back through `MessageFixture.read`, so a fixture
    this returns is one the journey accepts. An existing file is never replaced:
    changing an expected delivery position after capture is a failure.
    """
    if not digest(recording_sha256) or not digest(exchange.answer_sha256):
        raise FixtureRefused("message_fixture_digest_invalid")
    value: dict[str, JsonValue] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "customer_thread_id": customer_thread_id,
        "recording_sha256": recording_sha256,
        "question": exchange.question,
        "question_sha256": sha256(exchange.question),
        "answer_sha256": exchange.answer_sha256,
        "delivery_decision_step": exchange.delivery_decision_step,
        "event_cursor": event_cursor,
        "internal_message_ids": list(internal_message_ids),
    }
    for name, target in (
        ("terminal_run_id", terminal_run_id),
        ("rate_limited_run_id", rate_limited_run_id),
        ("absent_run_id", absent_run_id),
    ):
        if target is not None:
            value[name] = target
    raw = json.dumps(value, sort_keys=True).encode("utf-8")
    if len(raw) > FIXTURE_BYTES_MAX:
        raise FixtureRefused("message_fixture_not_private_or_bounded")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FixtureRefused("message_fixture_already_sealed") from None
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw)
    sealed = hashlib.sha256(raw).hexdigest()
    try:
        MessageFixture.read(path, sealed)
    except (OSError, ValueError, TypeError) as error:
        path.unlink()
        code = str(error) if isinstance(error, ValueError) else "message_fixture_unreadable"
        raise FixtureRefused(code) from None
    return sealed


def write_fixture(
    surface: Surface,
    path: Path,
    *,
    record_run_id: str,
    question: str,
    run_id: str,
    customer_thread_id: str,
    recording_sha256: str,
    internal_message_ids: Sequence[str],
    terminal_run_id: Optional[str] = None,
    rate_limited_run_id: Optional[str] = None,
    absent_run_id: Optional[str] = None,
) -> tuple[str, RecordedExchange]:
    """Read the recording and the replay run's cursor, then seal the fixture."""
    if record_run_id == run_id:
        raise FixtureRefused("message_fixture_replay_run_is_the_recording")
    exchange = recorded_exchange(surface, record_run_id, question)
    cursor = current_event_cursor(surface, run_id)
    sealed = seal(
        path,
        run_id=run_id,
        customer_thread_id=customer_thread_id,
        recording_sha256=recording_sha256,
        exchange=exchange,
        event_cursor=cursor,
        internal_message_ids=internal_message_ids,
        terminal_run_id=terminal_run_id,
        rate_limited_run_id=rate_limited_run_id,
        absent_run_id=absent_run_id,
    )
    return sealed, exchange


def _private_text(path: Path) -> str:
    if path.stat().st_mode & 0o077 or path.stat().st_size > MESSAGE_BYTES_MAX:
        raise FixtureRefused("message_fixture_question_not_private_or_bounded")
    return path.read_text(encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    from .journey_main import surface_factory

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--driver", type=SurfaceName, default=SurfaceName.SDK)
    parser.add_argument("--record-run-id", required=True)
    parser.add_argument("--question-file", type=Path, required=True, help="mode 0600")
    parser.add_argument("--run-id", required=True, help="the fresh replay run to bind")
    parser.add_argument("--customer-thread-id", required=True)
    parser.add_argument("--recording-sha256", required=True)
    parser.add_argument("--internal-message-id", action="append", default=[])
    parser.add_argument("--terminal-run-id")
    parser.add_argument("--rate-limited-run-id")
    parser.add_argument("--absent-run-id")
    parser.add_argument("--mcp-url")
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args(argv)
    origin = os.environ["SYNTH_LOCAL_API_BASE_URL"].rstrip("/")
    if not origin.startswith(LOOPBACK):
        parser.error("the fixture writer runs only against the leased loopback slot origin")
    make = surface_factory(
        options.driver,
        origin=origin,
        work_directory=options.out.parent,
        mcp_url=options.mcp_url,
        profile="local",
    )
    try:
        question = _private_text(options.question_file)
        with make(os.environ["SYNTH_API_KEY"]) as surface:
            sealed, exchange = write_fixture(
                surface,
                options.out,
                record_run_id=options.record_run_id,
                question=question,
                run_id=options.run_id,
                customer_thread_id=options.customer_thread_id,
                recording_sha256=options.recording_sha256,
                internal_message_ids=options.internal_message_id,
                terminal_run_id=options.terminal_run_id,
                rate_limited_run_id=options.rate_limited_run_id,
                absent_run_id=options.absent_run_id,
            )
    except FixtureRefused as refused:
        print(json.dumps({"sealed": False, "code": refused.code}))
        return 1
    print(
        json.dumps(
            {
                "sealed": True,
                "fixture_sha256": sealed,
                "question_sha256": sha256(exchange.question),
                "answer_sha256": exchange.answer_sha256,
                "delivery_decision_step": exchange.delivery_decision_step,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
