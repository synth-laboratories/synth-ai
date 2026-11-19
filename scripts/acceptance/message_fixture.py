"""Immutable private binding from a driver run to its recorded customer input."""

from __future__ import annotations

# ruff: noqa: UP045 -- Named Optional at acceptance boundaries.
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import JsonValue

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
class PriorGuidance:
    message_id: str
    kind: str
    body_sha256: str
    decision_step: int

    @classmethod
    def read(cls, value: object) -> PriorGuidance:
        if (not isinstance(value, dict) or set(value) != {"message_id", "kind", "body_sha256", "decision_step"}
                or not isinstance(value["message_id"], str) or not 1 <= len(value["message_id"]) <= 128
                or value["kind"] != "guidance" or not digest(value["body_sha256"])
                or type(value["decision_step"]) is not int or value["decision_step"] < 1):
            raise ValueError("message_fixture_prior_guidance_invalid")
        return cls(**value)

    def matches(self, row: Mapping[str, JsonValue], *, run_id: str, thread_id: str,
                match_message_id: bool = True) -> bool:
        body, state = row.get("body"), row.get("state")
        return (
            row.get("run_id") == run_id and row.get("thread_id") == thread_id
            and row.get("author") == "customer" and row.get("kind") == self.kind
            and (not match_message_id or row.get("message_id") == self.message_id)
            and isinstance(body, dict) and body.get("kind") == "available"
            and isinstance(body.get("text"), str) and sha256(body["text"]) == self.body_sha256
            and isinstance(state, dict) and state.get("kind") in ("delivered", "disposed")
            and type(state.get("decision_step")) is int and state["decision_step"] == self.decision_step
        )


def accepts_customer_prefix(rows: Sequence[Mapping[str, JsonValue]], *, run_id: str, thread_id: str,
                            guidance: Optional[PriorGuidance]) -> bool:
    customers = [row for row in rows if row.get("author") == "customer"]
    if guidance is None:
        return not customers
    return len(customers) == 1 and guidance.matches(customers[0], run_id=run_id, thread_id=thread_id)


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
    prior_guidance: Optional[PriorGuidance] = None

    def accepts_customer_prefix(self, rows: Sequence[Mapping[str, JsonValue]]) -> bool:
        return accepts_customer_prefix(rows, run_id=self.run_id, thread_id=self.customer_thread_id,
                                       guidance=self.prior_guidance)

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
            or value.keys() - required - optional - {"prior_guidance"}
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
        if "prior_guidance" in value:
            guidance = PriorGuidance.read(value["prior_guidance"])
            if guidance.decision_step >= position:
                raise ValueError("message_fixture_prior_guidance_invalid")
            value["prior_guidance"] = guidance
        value["internal_message_ids"] = tuple(ids)
        return cls(**value)
