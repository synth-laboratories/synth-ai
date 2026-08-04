from __future__ import annotations

from datetime import UTC, datetime

from synth_ai.sdk.research.contracts.factory_operations import FactoryWakeDueRequest
from synth_ai.sdk.research.contracts.research_intern import (
    InternRuntimeOutcome,
    InternSyncSession,
)


def test_sync_projection_accepts_integrated_backend_fields() -> None:
    now = datetime.now(UTC).isoformat()
    session = InternSyncSession.from_wire(
        {
            "schema_version": "smr.intern-sync-session.v1",
            "sync_session_id": "sync-1",
            "research_intern_id": "intern-1",
            "org_id": "org-1",
            "objective": "Improve Craftax",
            "status": "closed",
            "state_generation": 3,
            "last_event_sequence": 3,
            "binding": {"project_id": "project-1"},
            "metadata": {},
            "objective_bounds": {"task_id": "craftax-1", "candidate_timeout_seconds": 900},
            "outcome": "stopped",
            "temporal_workflow_id": "workflow-1",
            "execution_mode": "standard",
            "execution_profile_id": "intern_sync",
            "visuals": [],
            "kit_state_receipts": [],
            "experiments": [],
            "harness_bundle_available": True,
            "created_at": now,
            "updated_at": now,
            "closed_at": now,
        }
    )
    assert session.outcome is InternRuntimeOutcome.STOPPED
    assert session.harness_bundle_available is True


def test_factory_wake_effort_scope_round_trips_in_signed_contract() -> None:
    request = FactoryWakeDueRequest(effort_ids=("effort-a", "effort-b"), dry_run=True)
    wire = request.to_contract_wire()
    replay = FactoryWakeDueRequest.from_contract_wire(wire)
    assert replay.effort_ids == ("effort-a", "effort-b")
