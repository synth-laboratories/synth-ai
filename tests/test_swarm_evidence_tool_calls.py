"""Swarm evidence carries the backend's durable tool-call outcomes.

Found live on slot3 (rc1): retrieve_swarm_evidence failed with
"fields drifted: extra=['tool_calls']" because the backend added
SmrSwarmEvidenceResponse.tool_calls and the SDK's exact-field parser did not
know it.
"""

from __future__ import annotations

import pytest

from synth_ai.sdk.research.contracts.evidence import SwarmEvidence, SwarmToolCallStatus

_CALL = {
    "tool_call_id": "call-1",
    "stable_actor_key": None,
    "actor_role": "orchestrator",
    "correlation_id": None,
    "bundle_name": "pool_control",
    "tool_name": "create_pool_rollout",
    "arguments_digest": "a" * 64,
    "status": "succeeded",
    "error_code": None,
    "retryable": False,
    "duration_ms": 120,
    "occurred_at": "2026-09-12T01:00:00+00:00",
}


def _evidence(**overrides):
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "project_id": "proj-1",
        "artifacts": [],
        "work_products": [],
        "selected_artifact_contents": [],
        "trace_publications": [],
        "tool_calls": [_CALL],
        # The backend's freshness block also counts the tool calls (found live on
        # slot3, rc1 E01 attempt 8: "freshness fields drifted: extra=['tool_call_count']").
        "freshness": {
            "artifact_count": 0,
            "work_product_count": 0,
            "tool_call_count": 1,
            "generated_at": "2026-09-12T01:00:00+00:00",
            "run_is_terminal": True,
        },
    }
    payload.update(overrides)
    return payload


def test_tool_calls_parse_into_typed_outcomes() -> None:
    evidence = SwarmEvidence.from_wire(_evidence())

    (call,) = evidence.tool_calls
    assert call.tool_name == "create_pool_rollout"
    assert call.status is SwarmToolCallStatus.SUCCEEDED
    assert evidence.to_wire()["tool_calls"][0]["arguments_digest"] == "a" * 64


def test_freshness_carries_the_backend_tool_call_count() -> None:
    evidence = SwarmEvidence.from_wire(_evidence())

    assert evidence.freshness.tool_call_count == 1
    assert evidence.to_wire()["freshness"]["tool_call_count"] == 1


def test_freshness_without_tool_call_count_still_parses() -> None:
    legacy = _evidence()
    legacy["freshness"] = {k: v for k, v in legacy["freshness"].items() if k != "tool_call_count"}

    assert SwarmEvidence.from_wire(legacy).freshness.tool_call_count == 0


def test_freshness_drift_is_still_refused() -> None:
    drifted = _evidence()
    drifted["freshness"] = {**drifted["freshness"], "unexpected": 1}

    with pytest.raises(ValueError, match="freshness fields drifted"):
        SwarmEvidence.from_wire(drifted)


def test_tool_call_drift_is_still_refused() -> None:
    with pytest.raises(ValueError, match="drifted"):
        SwarmEvidence.from_wire(_evidence(tool_calls=[{**_CALL, "arguments": {"secret": 1}}]))
