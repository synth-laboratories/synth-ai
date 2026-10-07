"""SwarmStatus decodes backend ``pending_questions`` (backend c4763f816, SYN-3850)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest
from synth_ai.sdk.research.contracts.status import SwarmStatus, SwarmStatusPendingQuestion

LEGACY_PAYLOAD = {
    "schema_version": 1,
    "run_id": "run-1",
    "project_id": "proj-1",
    "state": {"public_state": "running", "reason": None, "blocker_code": None},
    "liveness": {
        "phase": "active",
        "task_counts": {"running": 1},
        "participant_queue_counts": {},
    },
    "terminal": {
        "run_state": "running",
        "run_state_is_terminal": False,
        "contract_status": None,
        "contract_reason": None,
        "outcome": None,
    },
    "finalization": {
        "status": None,
        "proof_status": None,
        "proof_satisfied": False,
        "blocker_code": None,
        "reason": None,
    },
    "recovery": {
        "blocked": False,
        "codes": [],
        "incident_count": 0,
        "latest_recorded_at": None,
    },
    "progress": {"last_progress_at": None, "last_progress_kind": None},
    "issues": {"issues": [], "invariants": []},
    "failure": {"code": None, "source": None, "classification": None, "terminal": False},
    "freshness": {
        "projection_authority": "smr_run_status_projection.v1",
        "last_authoritative_update_at": "2026-10-05T12:00:00+00:00",
        "generated_at": "2026-10-05T12:00:01+00:00",
    },
}

REF = {"artifact_id": "art-1", "media_type": "text/plain", "bytes": 42}


def _with_questions(questions: object) -> dict[str, object]:
    payload = copy.deepcopy(LEGACY_PAYLOAD)
    payload["pending_questions"] = questions
    return payload


def test_legacy_payload_without_pending_questions_decodes() -> None:
    status = SwarmStatus.from_wire(copy.deepcopy(LEGACY_PAYLOAD))
    assert status.swarm_id == "run-1"
    assert status.pending_questions == ()


def test_empty_pending_questions_decodes() -> None:
    assert SwarmStatus.from_wire(_with_questions([])).pending_questions == ()


def test_backend_pending_questions_decode() -> None:
    # Exact shape the backend dumps (model_dump(mode="json") in c4763f816's test).
    status = SwarmStatus.from_wire(
        _with_questions(
            [
                {
                    "action_id": "act-1",
                    "execution_id": "ex-1",
                    "question_text_ref": REF,
                    "question_text": None,
                    "asked_at": "2026-10-05T11:59:00Z",
                },
                {"action_id": "act-2", "execution_id": "ex-2"},
            ]
        )
    )
    assert status.pending_questions == (
        SwarmStatusPendingQuestion(
            action_id="act-1",
            execution_id="ex-1",
            question_text_ref=REF,
            question_text=None,
            asked_at=datetime(2026, 10, 5, 11, 59, tzinfo=UTC),
        ),
        SwarmStatusPendingQuestion(action_id="act-2", execution_id="ex-2"),
    )
    wire = status.to_wire()
    assert SwarmStatus.from_wire(wire) == status
    assert wire["pending_questions"][1] == {
        "action_id": "act-2",
        "execution_id": "ex-2",
        "question_text_ref": None,
        "question_text": None,
        "asked_at": None,
    }


def test_other_unknown_top_level_keys_still_rejected() -> None:
    payload = _with_questions([])
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="extra=\\['surprise'\\]"):
        SwarmStatus.from_wire(payload)


def test_missing_required_top_level_key_still_rejected() -> None:
    payload = _with_questions([])
    del payload["freshness"]
    with pytest.raises(ValueError, match="missing=\\['freshness'\\]"):
        SwarmStatus.from_wire(payload)


@pytest.mark.parametrize(
    "question",
    [
        {"action_id": "act-1", "execution_id": "ex-1", "extra": 1},
        {"action_id": "act-1"},
        {"action_id": "", "execution_id": "ex-1"},
        {"action_id": "act-1", "execution_id": "ex-1", "question_text_ref": "ref"},
        {"action_id": "act-1", "execution_id": "ex-1", "asked_at": "2026-10-05T11:59:00"},
    ],
)
def test_malformed_pending_question_rejected(question: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SwarmStatus.from_wire(_with_questions([question]))


def test_pending_questions_must_be_array() -> None:
    with pytest.raises(ValueError):
        SwarmStatus.from_wire(_with_questions({"action_id": "act-1"}))
