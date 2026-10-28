"""Capture policy and failure/retry behavior for Swarm actor research journal appends.

A Swarm actor appends `log_entry` records to an exact research log through the
admitted `forge_record_write` path. This module is the producer-side discipline
for those appends:

Capture policy (`CAPTURE_POLICY`)
    * An entry is an intentional, authored note: one of the six Forge
      categories, an aware `occurred_at` and the exact log reference.
      `failure` and null/negative results are first-class entries.
    * Text is screened before it leaves the actor: secret-looking strings, raw
      role-prefixed transcripts and hidden-reasoning markup are refused with the
      matched rule name (the matched text is never echoed). Over-limit text is
      refused, never truncated.
    * The producer identity is never chosen here. Forge stamps the native actor,
      task/run/attempt and delegation from the admitted grant.

Failure and retry
    * Every append returns one typed `AppendResult`; nothing is dropped
      silently. Outcomes: `committed`, `refused_by_policy`, `denied`,
      `conflict`, `rejected`, `unknown`.
    * Only outcome-uncertain failures (transport loss, timeouts, rate limits,
      5xx, `retryable=True`) are retried, at most `RETRY_ATTEMPTS` times, with
      the identical record ID, idempotency key and payload bytes. A replay of a
      committed write returns the original receipt.
    * When retries are exhausted the outcome is `unknown` and the intent stays
      pending; `reconcile()` resends the identical intent later. Denials,
      conflicts and contract rejections are terminal and reported as gaps.
    * With an outbox path, the intent is written ahead of the request and the
      outcome after it, so a process interruption leaves the intent pending on
      disk and the next `ActorJournal` on that outbox reconciles it.
    * `completeness()` reports counts and a status: `complete` only when every
      intent committed.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import ValidationError

from synth_ai.errors import (
    AuthenticationError,
    ConflictError,
    OutcomeUncertainError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    SynthError,
    TransportError,
)
from synth_ai.options import MutationOptions

from .contracts.records import LogEntry

CAPTURE_POLICY = "synth.forge.actor-journal-capture.v1"
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)
TEXT_LIMIT = 8000
CATEGORIES = ("observation", "rationale", "attempt", "failure", "decision", "correction")

SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bA(?:KIA|SIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("api_key_prefix", re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{20,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*\S{8,}"
        ),
    ),
)
TRANSCRIPT_TURN = re.compile(r"(?im)^\s*(?:user|human|assistant|system|ai|claude|codex)\s*:\s")
HIDDEN_REASONING = re.compile(r"(?i)<\s*/?\s*(?:thinking|reasoning|scratchpad|antml:[a-z_]+)\b")


class AppendOutcome(StrEnum):
    COMMITTED = "committed"
    REFUSED_BY_POLICY = "refused_by_policy"
    DENIED = "denied"
    CONFLICT = "conflict"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class CaptureRefusedError(ValueError):
    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(message)


class RecordWriter(Protocol):
    def write(
        self,
        project_id: str,
        record_id: str,
        payload: dict[str, Any],
        *,
        expected_revision: int = 0,
        mutation: MutationOptions | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class AppendIntent:
    """The exact bytes an append sends; retries resend this unchanged."""

    project_id: str
    record_id: str
    idempotency_key: str
    payload: dict[str, Any]

    def as_json(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "record_id": self.record_id,
            "idempotency_key": self.idempotency_key,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class AppendResult:
    outcome: AppendOutcome
    intent: AppendIntent | None
    attempts: int
    receipt: Any = None
    rule: str | None = None
    error_code: str | None = None
    status: int | None = None
    detail: str | None = None

    @property
    def pending(self) -> bool:
        return self.outcome is AppendOutcome.UNKNOWN


def screen(text: str) -> None:
    """Refuse secrets, raw transcripts, hidden reasoning and over-limit text."""
    if len(text) > TEXT_LIMIT:
        raise CaptureRefusedError(
            "over_limit",
            f"entry text is {len(text)} characters; the limit is {TEXT_LIMIT} and text is never truncated",
        )
    for name, pattern in SECRET_RULES:
        if pattern.search(text):
            raise CaptureRefusedError(
                "secret_like",
                f"entry contains a secret-looking string (rule {name}); cite redacted evidence instead",
            )
    if HIDDEN_REASONING.search(text):
        raise CaptureRefusedError(
            "hidden_reasoning", "entry contains hidden-reasoning markup; write a summary"
        )
    turns = len(TRANSCRIPT_TURN.findall(text))
    if turns >= 2:
        raise CaptureRefusedError(
            "raw_transcript",
            f"entry has {turns} role-prefixed conversation turns; summarize what was learned instead",
        )


def entry_payload(
    *,
    log: dict[str, Any],
    category: str,
    text: str,
    occurred_at: datetime | None = None,
    sources: tuple[dict[str, Any], ...] = (),
    correction_of: dict[str, Any] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Build one policy-conformant `log_entry` payload or raise `CaptureRefusedError`."""
    if category not in CATEGORIES:
        raise CaptureRefusedError(
            "invalid_category", f"category must be one of {', '.join(CATEGORIES)}"
        )
    if not isinstance(text, str) or not text.strip():
        raise CaptureRefusedError("empty_text", "entry text is empty")
    screen(text)
    selected = now() if occurred_at is None else occurred_at
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise CaptureRefusedError("naive_occurred_at", "occurred_at must carry a timezone offset")
    payload: dict[str, Any] = {
        "kind": "log_entry",
        "log": log,
        "category": category,
        "text": text,
        "occurred_at": selected.isoformat(),
        "sources": list(sources),
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    try:
        return LogEntry.model_validate(payload).model_dump(mode="json", exclude_none=True)
    except ValidationError as error:
        raise CaptureRefusedError("invalid_entry", str(error.errors()[0]["msg"])) from None


def classify(error: SynthError) -> AppendOutcome | None:
    """Terminal outcome for an error, or None when the outcome is uncertain and retryable."""
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return AppendOutcome.DENIED
    if isinstance(error, ConflictError):
        return AppendOutcome.CONFLICT
    if isinstance(
        error, (TransportError, RequestTimeoutError, OutcomeUncertainError, RateLimitError)
    ):
        return None
    if error.retryable is True or (error.status is not None and error.status >= 500):
        return None
    return AppendOutcome.REJECTED


@dataclass
class ActorJournal:
    """Append entries for one actor to one exact research log with typed outcomes."""

    records: RecordWriter
    project_id: str
    log: dict[str, Any]
    outbox: Path | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    results: list[AppendResult] = field(default_factory=list)
    _pending: dict[str, AppendIntent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outbox is not None and self.outbox.exists():
            for intent in pending_in_outbox(self.outbox):
                self._pending[intent.idempotency_key] = intent

    def append(
        self,
        category: str,
        text: str,
        *,
        occurred_at: datetime | None = None,
        sources: tuple[dict[str, Any], ...] = (),
        correction_of: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> AppendResult:
        try:
            payload = entry_payload(
                log=self.log,
                category=category,
                text=text,
                occurred_at=occurred_at,
                sources=sources,
                correction_of=correction_of,
                now=self.now,
            )
        except CaptureRefusedError as refused:
            result = AppendResult(
                AppendOutcome.REFUSED_BY_POLICY, None, 0, rule=refused.rule, detail=str(refused)
            )
            self.results.append(result)
            return result
        record_id = entry_id or "entry-" + uuid4().hex
        intent = AppendIntent(self.project_id, record_id, "journal-" + record_id, payload)
        return self._send(intent)

    def reconcile(self) -> list[AppendResult]:
        """Resend every pending intent unchanged; returns the new outcomes."""
        return [self._send(intent) for intent in list(self._pending.values())]

    @property
    def pending(self) -> tuple[AppendIntent, ...]:
        return tuple(self._pending.values())

    def completeness(self) -> dict[str, Any]:
        latest: dict[str, AppendResult] = {}
        refused = 0
        for result in self.results:
            if result.intent is None:
                refused += 1
            else:
                latest[result.intent.idempotency_key] = result
        counts = {outcome.value: 0 for outcome in AppendOutcome}
        counts[AppendOutcome.REFUSED_BY_POLICY.value] = refused
        for result in latest.values():
            counts[result.outcome.value] += 1
        for key in self._pending:
            if key not in latest:
                counts[AppendOutcome.UNKNOWN.value] += 1
        unresolved = counts["unknown"]
        gaps = (
            counts["denied"] + counts["conflict"] + counts["rejected"] + counts["refused_by_policy"]
        )
        status = "incomplete" if unresolved else ("gaps" if gaps else "complete")
        return {
            "policy": CAPTURE_POLICY,
            "status": status,
            "counts": counts,
            "pending_keys": sorted(self._pending),
            "cost": "unpaid",
        }

    def _send(self, intent: AppendIntent) -> AppendResult:
        self._pending[intent.idempotency_key] = intent
        self._record({"event": "intent", **intent.as_json()})
        attempts = 0
        last: SynthError | None = None
        while attempts < RETRY_ATTEMPTS:
            attempts += 1
            try:
                receipt = self.records.write(
                    intent.project_id,
                    intent.record_id,
                    intent.payload,
                    expected_revision=0,
                    mutation=MutationOptions(idempotency_key=intent.idempotency_key),
                )
            except SynthError as error:
                terminal = classify(error)
                if terminal is not None:
                    return self._finish(
                        AppendResult(
                            terminal,
                            intent,
                            attempts,
                            error_code=error.code,
                            status=error.status,
                            detail=str(error),
                        )
                    )
                last = error
                if attempts < RETRY_ATTEMPTS:
                    delay = RETRY_BACKOFF_SECONDS[min(attempts - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                    if error.retry_after_seconds is not None:
                        delay = max(delay, error.retry_after_seconds)
                    self.sleep(delay)
                continue
            return self._finish(
                AppendResult(AppendOutcome.COMMITTED, intent, attempts, receipt=receipt)
            )
        assert last is not None
        result = AppendResult(
            AppendOutcome.UNKNOWN,
            intent,
            attempts,
            error_code=last.code,
            status=last.status,
            detail=str(last),
        )
        self.results.append(result)
        self._record(
            {"event": "outcome", "idempotency_key": intent.idempotency_key, "outcome": "unknown"}
        )
        return result

    def _finish(self, result: AppendResult) -> AppendResult:
        assert result.intent is not None
        self._pending.pop(result.intent.idempotency_key, None)
        self.results.append(result)
        self._record(
            {
                "event": "outcome",
                "idempotency_key": result.intent.idempotency_key,
                "outcome": result.outcome.value,
            }
        )
        return result

    def _record(self, line: dict[str, Any]) -> None:
        if self.outbox is None:
            return
        self.outbox.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()


def pending_in_outbox(path: Path) -> list[AppendIntent]:
    """Intents whose last recorded outcome is missing or `unknown`."""
    intents: dict[str, AppendIntent] = {}
    settled: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        key = line["idempotency_key"]
        if line["event"] == "intent":
            intents[key] = AppendIntent(line["project_id"], line["record_id"], key, line["payload"])
            settled.discard(key)
        elif line["outcome"] != "unknown":
            settled.add(key)
    return [intent for key, intent in intents.items() if key not in settled]
