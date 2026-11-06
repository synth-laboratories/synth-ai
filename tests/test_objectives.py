from synth_ai._generated.models import (
    ObjectiveClaimedPayload,
    ObjectiveProgressPayload,
    RunEvent,
    RunSnapshot,
)

SNAPSHOT = {
    "metadata": {},
    "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
    "project_id": "proj_AAAAAAAAAAAAAAAAAAAAAA",
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
