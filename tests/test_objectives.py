from synth_ai._generated.models import (
    ObjectiveClaimedPayload,
    ObjectiveProgressPayload,
    RunEvent,
    RunSnapshot,
)

SNAPSHOT = {
    "metadata": {},
    "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
    "labels": {},
    "labels_revision": 1,
    "guidance_coverage": [],
    "state": "running",
    "model": "example",
    "provider": "example",
    "result_available": False,
    "execution_closed": False,
    "accounting_final": False,
    "cleanup_complete": False,
    "problems": [],
    "problems_observed": True,
    "created_at": "2026-11-06T00:00:00Z",
    "started_at": None,
    "closed_at": None,
}
OBJECTIVE = {
    "id": "goal",
    "kind": "deo",
    "title": "Reach the score",
    "statement": "Score at least 0.5",
    "required": True,
    "criteria": ["score >= 0.5"],
    "state": "claimed",
    "metric": {
        "name": "score",
        "direction": "maximize",
        "target": 0.5,
        "source": "rollout_score",
    },
    "metric_state": {
        "current": 0.125,
        "best": 0.25,
        "best_rollout_id": "rollout_a",
        "best_policy_sha256": None,
        "best_receipt": {
            "authority": "rhodes",
            "kind": "pool_rollout_result",
            "record_id": "rollout_a",
            "revision": "1",
            "digest_sha256": "d" * 64,
        },
        "samples": 2,
        "met": False,
        "percent": 50,
    },
    "claims": [
        {
            "id": "c1",
            "kind": "progress",
            "percent_complete": 50,
            "summary": "halfway",
            "by": "worker-1",
            "observed_at": None,
        }
    ],
    "verdict": None,
}
EVENT = {
    "event_id": "evt_AAAAAAAAAAAAAAAAAAAAAA",
    "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
    "sequence": 3,
    "kind": "objective.progress",
    "original_kind": "observation",
    "usage": None,
    "observed_at": None,
    "approval_id": None,
}


def test_filled_snapshot_reads_the_objective_shape():
    snapshot = RunSnapshot.model_validate(
        {**SNAPSHOT, "objectives": [OBJECTIVE], "objectives_read": "ok"}
    )
    objective = snapshot.objectives[0]
    assert snapshot.objectives_read == "ok"
    assert objective.metric.target == 0.5
    assert objective.metric_state.best == 0.25
    assert objective.metric_state.best_receipt.record_id == "rollout_a"
    assert objective.claims[0].percent_complete == 50
    assert objective.verdict is None


def test_dumped_objectives_keep_the_launcher_claim_fields():
    snapshot = RunSnapshot.model_validate(
        {**SNAPSHOT, "objectives": [OBJECTIVE], "objectives_read": "ok"}
    )
    claim = snapshot.model_dump(mode="json")["objectives"][0]["claims"][0]
    assert {key: claim[key] for key in ("id", "kind", "percent_complete", "summary", "by")} == {
        "id": "c1",
        "kind": "progress",
        "percent_complete": 50,
        "summary": "halfway",
        "by": "worker-1",
    }


def test_unavailable_read_is_distinct_from_no_objectives():
    snapshot = RunSnapshot.model_validate(
        {**SNAPSHOT, "objectives": [], "objectives_read": "unavailable"}
    )
    assert snapshot.objectives == []
    assert snapshot.objectives_read == "unavailable"


def test_a_server_without_the_marker_still_validates():
    assert RunSnapshot.model_validate({**SNAPSHOT, "objectives": []}).objectives_read is None


def test_event_payloads_parse_by_shape_and_are_optional():
    progress = RunEvent.model_validate(
        {
            **EVENT,
            "payload": {
                "objective_id": "goal",
                "name": "score",
                "current": 0.125,
                "best": 0.25,
                "target": 0.5,
                "direction": "maximize",
                "met": False,
                "percent": 50,
                "samples": 2,
                "rollout_id": "rollout_a",
                "receipt_digest": "d" * 64,
            },
        }
    )
    assert isinstance(progress.payload, ObjectiveProgressPayload)
    claimed = RunEvent.model_validate(
        {
            **EVENT,
            "kind": "objective.claimed",
            "payload": {
                "objective_id": "goal",
                "claim_id": "c1",
                "percent_complete": 50,
                "kind": "progress",
                "summary": "halfway",
            },
        }
    )
    assert isinstance(claimed.payload, ObjectiveClaimedPayload)
    assert RunEvent.model_validate({**EVENT, "payload": None}).payload is None
    assert RunEvent.model_validate(EVENT).payload is None


# Staging 2026-11-11: Core sent a body on the limit kinds while the contract
# declared none, so `events()` refused every run at its first event.
LIMIT_EVENTS = [
    ("limits.revised", {"revision": 1, "at_ms": 1794391592930, "pause": None}),
    (
        "run.paused",
        {
            "revision": 2,
            "at_ms": 1794391593000,
            "pause": {
                "cause": "customer",
                "operation_id": "op-1",
                "limit": None,
                "requested_at_ms": 1794391593000,
                "drained_at_ms": None,
                "mode": "warm",
            },
        },
    ),
    (
        "limit.warning",
        {
            "revision": 2,
            "at_ms": 1794391594000,
            "scope": "run",
            "dimension": "spend_micros",
            "cap": 1000,
            "used": 800,
            "reserved": 10,
            "unknown_reserved": 0,
        },
    ),
    (
        "actor.parked",
        {
            "actor_id": "worker",
            "assignment": "worker/task",
            "dimension": "spend_micros",
            "at_ms": 1794391594000,
        },
    ),
    ("actor.unparked", {"actor_id": "worker", "assignment": None, "at_ms": 1794391595000}),
]


def test_limit_events_read_their_declared_payloads():
    from synth_ai._generated.models import EventPage

    page = EventPage.model_validate(
        {
            "items": [
                {**EVENT, "sequence": index, "kind": kind, "original_kind": kind, "payload": body}
                for index, (kind, body) in enumerate(LIMIT_EVENTS)
            ],
            "has_more": False,
            "resume_cursor": None,
        }
    )
    assert page.items[0].payload.revision == 1 and page.items[0].payload.pause is None
    assert page.items[1].payload.pause.mode == "warm"
    assert page.items[2].payload.cap == 1000
    assert page.items[3].payload.actor_id == "worker"
    assert page.items[4].payload.assignment is None


def test_a_payload_on_a_kind_that_declares_none_is_still_refused():
    import pytest
    from pydantic import ValidationError

    from synth_ai._generated.models import RunEvent

    with pytest.raises(ValidationError):
        RunEvent.model_validate({**EVENT, "kind": "running", "payload": {"anything": 1}})
