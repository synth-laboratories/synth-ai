"""Immutable private binding from a driver run to its recorded customer input."""

from __future__ import annotations

# ruff: noqa: UP045 -- Named Optional at acceptance boundaries.
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FIXTURE_BYTES_MAX = 65536
MESSAGE_BYTES_MAX = 8192


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class MessageFixture:
    run_id: str
    customer_thread_id: str
    recording_sha256: str
    question: str
    question_sha256: str
    answer_sha256: str
    delivery_decision_step: int
    event_cursor: str
    internal_message_ids: tuple[str, ...]
    terminal_run_id: Optional[str] = None
    rate_limited_run_id: Optional[str] = None
    absent_run_id: Optional[str] = None

    @classmethod
    def read(cls, path: Path, expected_sha256: str) -> MessageFixture:
        if path.stat().st_mode & 0o077 or path.stat().st_size > FIXTURE_BYTES_MAX:
            raise ValueError("message_fixture_not_private_or_bounded")
        raw = path.read_bytes()
        if not digest(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("message_fixture_changed")
        value = json.loads(raw)
        required = {
            "schema",
            "run_id",
            "customer_thread_id",
            "recording_sha256",
            "question",
            "question_sha256",
            "answer_sha256",
            "delivery_decision_step",
            "event_cursor",
            "internal_message_ids",
        }
        optional = {"terminal_run_id", "rate_limited_run_id", "absent_run_id"}
        if (
            not isinstance(value, dict)
            or not required <= value.keys()
            or value.keys() - required - optional
        ):
            raise ValueError("message_fixture_shape_invalid")
        if value.pop("schema") != "synth.journeys.message-fixture.v1":
            raise ValueError("message_fixture_schema_unknown")
        for name in ("recording_sha256", "question_sha256", "answer_sha256"):
            if not digest(value[name]):
                raise ValueError("message_fixture_digest_invalid")
        question = value["question"]
        if (
            not isinstance(question, str)
            or not 1 <= len(question.encode("utf-8")) <= MESSAGE_BYTES_MAX
        ):
            raise ValueError("message_fixture_question_invalid")
        if sha256(question) != value["question_sha256"]:
            raise ValueError("message_fixture_question_changed")
        position = value["delivery_decision_step"]
        if type(position) is not int or position <= 0:
            raise ValueError("message_fixture_position_invalid")
        for name, maximum in (("run_id", 128), ("customer_thread_id", 128), ("event_cursor", 2048)):
            if not isinstance(value[name], str) or not 1 <= len(value[name]) <= maximum:
                raise ValueError("message_fixture_identity_invalid")
        ids = value["internal_message_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) > 128
            or any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in ids)
        ):
            raise ValueError("message_fixture_internal_ids_invalid")
        for name in optional:
            target = value.get(name)
            if target is not None and (not isinstance(target, str) or not 1 <= len(target) <= 128):
                raise ValueError("message_fixture_target_invalid")
        value["internal_message_ids"] = tuple(ids)
        return cls(**value)
