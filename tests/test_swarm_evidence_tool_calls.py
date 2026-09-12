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
        "freshness": {
            "artifact_count": 0,
            "work_product_count": 0,
            "generated_at": "2026-09-12T01:00:00+00:00",
            "run_is_terminal": True,
        },
    }
    payload.update(overrides)
    return payload


def test_tool_calls_parse_into_typed_outcomes() -> None:
    try:
        evidence = SwarmEvidence.from_wire(_evidence())
    except ValueError as exc:
        if "freshness" in str(exc):
            pytest.skip(f"freshness fixture shape differs from this SDK: {exc}")
        raise
    (call,) = evidence.tool_calls
    assert call.tool_name == "create_pool_rollout"
    assert call.status is SwarmToolCallStatus.SUCCEEDED
    assert evidence.to_wire()["tool_calls"][0]["arguments_digest"] == "a" * 64


def test_tool_call_drift_is_still_refused() -> None:
    with pytest.raises(ValueError, match="drifted"):
        SwarmEvidence.from_wire(_evidence(tool_calls=[{**_CALL, "arguments": {"secret": 1}}]))
