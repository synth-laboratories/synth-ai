from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from synth_ai.sdk.research.contracts.factory_operations import FactoryWakeDueRequest
from synth_ai.sdk.research.contracts.research_intern import (
    InternAsyncRuntime,
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
            "workspace_run_receipts": [],
            "experiments": [],
            "harness_bundle_available": True,
            "created_at": now,
            "updated_at": now,
            "closed_at": now,
        }
    )
    assert session.outcome is InternRuntimeOutcome.STOPPED
    assert session.harness_bundle_available is True
    assert session.workspace_run_receipts == ()
    with pytest.raises(ValidationError, match="kit_state_receipts"):
        InternSyncSession.from_wire({**session.to_wire(), "kit_state_receipts": []})


def test_factory_wake_effort_scope_round_trips_in_signed_contract() -> None:
    request = FactoryWakeDueRequest(effort_ids=("effort-a", "effort-b"), dry_run=True)
    wire = request.to_contract_wire()
    replay = FactoryWakeDueRequest.from_contract_wire(wire)
    assert replay.effort_ids == ("effort-a", "effort-b")


def test_async_projection_preserves_actor_reply_wait_identity() -> None:
    now = datetime.now(UTC).isoformat()
    runtime = InternAsyncRuntime.from_wire(
        {
            "schema_version": "smr.intern-async-runtime.v1",
            "async_runtime_id": "async-1",
            "async_assignment_id": "async-1",
            "cardinality": "one_per_organization",
            "instance_kind": "organization_async_intern",
            "research_intern_id": "intern-1",
            "org_id": "org-1",
            "objective": "Wait for the actor reply",
            "status": "reconciling",
            "state_generation": 4,
            "last_event_sequence": 4,
            "cycle_number": 1,
            "plan": {},
            "awaiting_actor_reply_message_id": "mq-1",
            "awaiting_actor_reply_thread_id": "thread-1",
            "pending_instruction_count": 0,
            "open_judgment_items": [
                {
                    "schema_version": "smr.intern-async-judgment.v1",
                    "interaction_id": "judgment-1",
                    "effort_id": "effort-1",
                    "prompt": "Confirm the next experiment?",
                    "created_generation": 3,
                    "context": {},
                }
            ],
            "effort_work": [
                {
                    "effort_id": "effort-1",
                    "status": "awaiting_input",
                    "open_interaction_id": "judgment-1",
                }
            ],
            "binding": {},
            "external_execution_status": "active",
            "evidence_readiness": "pending",
            "budget": {
                "maximum_concurrent_runs": 1,
                "maximum_daily_cost_cents": 5000,
                "maximum_monthly_cost_cents": 50000,
            },
            "temporal_workflow_id": "intern-async:org-1:async-1",
            "leave_safe": True,
            "created_at": now,
            "updated_at": now,
        }
    )

    assert runtime.awaiting_actor_reply_message_id == "mq-1"
    assert runtime.awaiting_actor_reply_thread_id == "thread-1"
    assert len(runtime.open_judgment_items) == 1
    assert runtime.open_judgment_items[0].effort_id == "effort-1"
    assert runtime.effort_work[0].status == "awaiting_input"
    assert runtime.budget.maximum_daily_cost_cents == 5000
    assert runtime.budget.maximum_monthly_cost_cents == 50000


def test_sync_create_allows_send_first_and_preserves_operator_approval_default():
    from synth_ai.sdk.research.contracts.research_intern import InternSyncSessionCreateRequest

    request = InternSyncSessionCreateRequest(idempotency_key="send-first")
    assert request.objective == ""
    assert request.to_wire()["require_operator_approval"] is True
    unattended = InternSyncSessionCreateRequest(
        idempotency_key="unattended", require_operator_approval=False
    )
    assert unattended.to_wire()["require_operator_approval"] is False
