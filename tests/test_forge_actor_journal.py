"""Swarm actor journal capture policy and failure/retry behavior, with a fake writer."""

import json
from datetime import UTC, datetime

import pytest
from synth_ai.errors import (
    ConflictError,
    PermissionDeniedError,
    RateLimitError,
    RefusedError,
    SynthError,
    TransportError,
)
from synth_ai.sdk.forge.journal import (
    CAPTURE_POLICY,
    RETRY_ATTEMPTS,
    ActorJournal,
    AppendOutcome,
    pending_in_outbox,
)

PROJECT = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
LOG = {
    "authority": "forge",
    "kind": "research_log",
    "record_id": "log",
    "revision": "1",
    "digest_sha256": "a" * 64,
}
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


class FakeRecords:
    """Scripted writer: each call pops the next step (an exception or 'ok')."""

    def __init__(self, steps=()):
        self.steps = list(steps)
        self.calls = []
        self.committed = {}

    def write(self, project_id, record_id, payload, *, expected_revision=0, mutation=None):
        key = mutation.idempotency_key
        self.calls.append((project_id, record_id, json.dumps(payload, sort_keys=True), key))
        step = self.steps.pop(0) if self.steps else "ok"
        if step == "lost_after_commit":
            self.committed.setdefault(key, {"receipt": key})
            raise TransportError("response lost", reason="response_lost")
        if isinstance(step, BaseException):
            raise step
        return self.committed.setdefault(key, {"receipt": key})


def journal(records, **kwargs):
    sleeps = []
    built = ActorJournal(records, PROJECT, LOG, sleep=sleeps.append, now=lambda: NOW, **kwargs)
    return built, sleeps


def test_committed_entry_carries_exact_log_and_aware_time():
    records = FakeRecords()
    actor, _ = journal(records)
    result = actor.append("failure", "Seed 3 diverged at step 40; null result.", entry_id="e1")
    assert result.outcome is AppendOutcome.COMMITTED and result.attempts == 1
    payload = json.loads(records.calls[0][2])
    assert payload["log"] == LOG and payload["category"] == "failure"
    assert payload["occurred_at"].startswith("2026-10-28T12:00:00")
    assert records.calls[0][3] == "journal-e1"
    assert actor.completeness()["status"] == "complete"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("token ghp_" + "a" * 36, "secret_like"),
        ("user: hi\nassistant: hello\n", "raw_transcript"),
        ("<thinking>private</thinking>", "hidden_reasoning"),
        ("x" * 8001, "over_limit"),
    ],
)
def test_policy_refusals_are_typed_never_sent_and_counted(text, rule):
    records = FakeRecords()
    actor, _ = journal(records)
    result = actor.append("observation", text)
    assert result.outcome is AppendOutcome.REFUSED_BY_POLICY and result.rule == rule
    assert records.calls == []
    report = actor.completeness()
    assert report["status"] == "gaps" and report["counts"]["refused_by_policy"] == 1
    assert report["policy"] == CAPTURE_POLICY


def test_correction_requires_exact_prior_entry_and_naive_time_is_refused():
    actor, _ = journal(FakeRecords())
    assert actor.append("correction", "fix").rule == "invalid_entry"
    naive = actor.append("observation", "ok", occurred_at=datetime(2026, 10, 28))
    assert naive.rule == "naive_occurred_at"


def test_transient_failures_retry_identical_bytes_and_replay_returns_original_receipt():
    records = FakeRecords(["lost_after_commit", RateLimitError("slow", retry_after_seconds=3.0)])
    actor, sleeps = journal(records)
    result = actor.append("attempt", "Ran arm B with lr=3e-4.", entry_id="e2")
    assert result.outcome is AppendOutcome.COMMITTED and result.attempts == 3
    assert result.receipt == {"receipt": "journal-e2"}
    assert len(set(records.calls)) == 1
    assert sleeps == [0.5, 3.0]


def test_exhausted_retries_keep_the_intent_pending_and_reconcile_sends_it_unchanged():
    records = FakeRecords([TransportError("down", reason="request_not_sent")] * RETRY_ATTEMPTS)
    actor, _ = journal(records)
    result = actor.append("observation", "Loss plateaued at 0.41.", entry_id="e3")
    assert result.outcome is AppendOutcome.UNKNOWN and result.pending
    assert actor.completeness()["status"] == "incomplete"
    assert actor.completeness()["pending_keys"] == ["journal-e3"]
    (again,) = actor.reconcile()
    assert again.outcome is AppendOutcome.COMMITTED
    assert len(set(records.calls)) == 1
    assert actor.pending == () and actor.completeness()["status"] == "complete"


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (PermissionDeniedError("no grant", status=403), AppendOutcome.DENIED),
        (ConflictError("different payload", status=409), AppendOutcome.CONFLICT),
        (RefusedError("contract", status=422), AppendOutcome.REJECTED),
    ],
)
def test_terminal_errors_are_not_retried_and_are_reported_as_gaps(error, outcome):
    records = FakeRecords([error])
    actor, sleeps = journal(records)
    result = actor.append("decision", "Stop arm C.")
    assert result.outcome is outcome and result.attempts == 1 and sleeps == []
    assert actor.completeness()["status"] == "gaps"
    assert actor.pending == ()


def test_server_errors_are_retried():
    records = FakeRecords([SynthError("bad gateway", status=502)])
    actor, _ = journal(records)
    assert actor.append("observation", "ok").attempts == 2


def test_interrupted_append_survives_in_the_outbox_and_the_next_journal_reconciles(tmp_path):
    outbox = tmp_path / "journal-outbox.jsonl"
    records = FakeRecords([KeyboardInterrupt()])
    actor, _ = journal(records, outbox=outbox)
    with pytest.raises(KeyboardInterrupt):
        actor.append("attempt", "Started sweep 2.", entry_id="e4")
    (pending,) = pending_in_outbox(outbox)
    assert pending.idempotency_key == "journal-e4"
    resumed, _ = journal(records, outbox=outbox)
    assert resumed.completeness()["status"] == "incomplete"
    (result,) = resumed.reconcile()
    assert result.outcome is AppendOutcome.COMMITTED
    assert records.calls[0] == records.calls[1]
    assert pending_in_outbox(outbox) == []
