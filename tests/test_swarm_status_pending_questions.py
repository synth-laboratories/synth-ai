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


def test_unknown_additive_top_level_keys_are_recorded_not_fatal() -> None:
    # Pressure 6204833e: an additive server key must not break installed clients.
    payload = _with_questions([])
    payload["surprise"] = {"nested": [1, 2]}
    status = SwarmStatus.from_wire(payload)
    assert status.unrecognized_fields == frozenset({"surprise"})
    known = SwarmStatus.from_wire(_with_questions([]))
    assert status.to_wire() == known.to_wire()
    assert "surprise" not in status.to_wire()


AUTHORITY_REF = {
    "owner": "backend",
    "schema": "synth.orchestra-constructor-terminal.v5",
    "reference": "ref-1",
    "sha256": "a" * 64,
}


def test_terminal_adapter_keys_decode_strictly() -> None:
    payload = _with_questions([])
    payload.update(
        active=False,
        authority_ref=AUTHORITY_REF,
        finished_at="2026-10-05T12:00:00Z",
    )
    status = SwarmStatus.from_wire(payload)
    assert status.active is False
    assert status.authority_ref is not None
    assert status.authority_ref.to_wire() == AUTHORITY_REF
    assert status.finished_at == datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    assert status.unrecognized_fields == frozenset()
    assert SwarmStatus.from_wire(status.to_wire()) == status


def test_terminal_adapter_keys_absent_or_null_decode_as_none() -> None:
    absent = SwarmStatus.from_wire(copy.deepcopy(LEGACY_PAYLOAD))
    payload = _with_questions([])
    payload.update(active=None, authority_ref=None, finished_at=None)
    explicit = SwarmStatus.from_wire(payload)
    for status in (absent, explicit):
        assert (status.active, status.authority_ref, status.finished_at) == (None, None, None)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("active", "false"),
        ("authority_ref", {**AUTHORITY_REF, "extra": 1}),
        ("authority_ref", {"owner": "backend"}),
        ("authority_ref", "ref-1"),
        ("finished_at", "2026-10-05T12:00:00"),
        ("finished_at", 1),
    ],
)
def test_malformed_known_optional_keys_still_rejected(key: str, value: object) -> None:
    payload = _with_questions([])
    payload[key] = value
    with pytest.raises(ValueError):
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
