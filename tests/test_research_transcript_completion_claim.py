"""The SDK decodes task.completion_claimed transcript events."""

from __future__ import annotations

import pytest

from synth_ai.sdk.research.contracts.transcript import (
    SwarmTranscriptEvent,
    TaskCompletionClaimedEvent,
)


def _wire(**payload_overrides):
    payload = {
        "schema_version": "smr.task-completion-claimed.v1",
        "claim_id": "a48dd3a6-dd30-405b-92c6-59462f2e5132",
        "task_id": "2d73dc80-1f29-50ad-8415-7774c4c15008",
        "task_key": "workshop-left-05b8f11599b94aca86344ca3f3a8fcd6",
        "participant_id": "workshop-left-05b8f11599b94aca86344ca3f3a8fcd6",
        "worker_actor_key": "workshop-left-05b8f11599b94aca86344ca3f3a8fcd6",
        "worker_actor_id": "e025cb06-0dc0-5b87-8370-95a17d4e8e64",
        "claimed_state": "done",
        "claim_kind": "completed_with_evidence",
        "status": "accepted_as_claim",
        "source_intent_id": "e99ad725-afc6-4b99-9b94-0b3c897bffc0",
        "text": "Computed 1564 × 7 = 10948.",
        "text_truncated": False,
        "text_source": "claim_notes",
        "summary": "Worker requested task state done.",
        "summary_truncated": False,
        "recorded_at": "2026-09-13T15:51:46.926877+00:00",
        "turn_id_source": "durable_transcript_turn",
        **payload_overrides,
    }
    return {
        "schema_version": "synth.research.transcript-event.v1",
        "event_id": "4c5b3a3e-7f7a-5c61-9d0c-1f2a3b4c5d6e",
        "live_cursor": None,
        "run_id": "50b06173-f15f-42c0-95d2-1fd5cecb7af9",
        "participant_session_id": "run-local:worker:50b06173:2d73dc80",
        "participant_role": "worker",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "occurred_at": "2026-09-13T15:51:46.926877+00:00",
        "kind": "task.completion_claimed",
        "payload": payload,
        "semantic_scope": "participant",
        "run_terminal": True,
        "redaction_profile": "public_default",
        "visibility_decision": "stream_redacted",
        "payload_classification": "public_safe",
    }


def test_completion_claim_event_decodes_attributed_contribution() -> None:
    event = SwarmTranscriptEvent.from_wire(_wire())
    claim = TaskCompletionClaimedEvent.from_transcript_event(event)
    assert claim.participant_id == "workshop-left-05b8f11599b94aca86344ca3f3a8fcd6"
    assert claim.text == "Computed 1564 × 7 = 10948."
    assert claim.turn_id == "turn-1"
    assert claim.event_id == "4c5b3a3e-7f7a-5c61-9d0c-1f2a3b4c5d6e"


def test_completion_claim_rejects_other_kinds_and_schemas() -> None:
    other = dict(_wire(), kind="message.completed")
    with pytest.raises(ValueError, match="not a task.completion_claimed"):
        TaskCompletionClaimedEvent.from_transcript_event(SwarmTranscriptEvent.from_wire(other))
    with pytest.raises(ValueError, match="schema_version"):
        TaskCompletionClaimedEvent.from_transcript_event(
            SwarmTranscriptEvent.from_wire(_wire(schema_version="v0"))
        )
