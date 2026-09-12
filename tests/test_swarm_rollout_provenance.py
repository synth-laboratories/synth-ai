"""Swarm rollout reads decode run → pool/task → intent → launch/lease → image → artifact.

Both transports, legacy unknowns, the explicit ``lease: null`` disposition and
field lockstep with the vendored backend OpenAPI contract.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.contracts import swarm_rollouts as contracts
from synth_ai.sdk.research.contracts.common import SwarmId
from synth_ai.sdk.research.swarms import AsyncSwarmsAPI, SwarmsAPI

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "openapi" / "research-v1.json").read_text()
)
DIGEST = "sha256:" + "a" * 64
ROLLOUT: dict[str, Any] = {
    "rollout_id": "rollout-1",
    "pool_id": "pool-1",
    "task_id": "task-1",
    "adapter": "harbor",
    "status": "completed",
    "seed": 3,
    "trace_correlation_id": "trace-1",
    "budget_parent_run_id": "run-1",
    "release_coordinates": None,
    "logical_intent": {
        "intent_ref": "rintent_1",
        "source": "derived",
        "intent_id": None,
        "scope_kind": "smr_run",
        "accepted_submissions": 2,
    },
    "execution": {
        "recorded_by": "rhodes_worker",
        "interface": "harbor",
        "interface_mode": "harbor_sandbox",
        "provider": "docker",
        "launch_id": "launch_1",
        "lease": None,
        "lease_disposition": {
            "status": "not_applicable",
            "reason": "harbor_rollout_launches_ephemeral_sandboxes",
        },
        "deployment_binding": "ephemeral_per_rollout",
        "executed_image_digest": DIGEST,
        "image_attestation": "provider_image_id",
        "executed_source_digest": None,
        "launches": [
            {
                "sequence": 0,
                "role": "agent",
                "provider": "docker",
                "launch_id": "launch_1",
                "executed_image_digest": DIGEST,
                "image_attestation": "provider_image_id",
                "recorded_at": "2026-09-12T12:00:00+00:00",
            }
        ],
        "recorded_at": "2026-09-12T12:00:00+00:00",
    },
    "artifacts": [
        {
            "artifact_id": "artifact-1",
            "artifact_type": "result",
            "content_type": "application/json",
            "size_bytes": 12,
            "created_at": "2026-09-12T12:01:00Z",
        }
    ],
    "success": True,
    "score": 1,
    "error": None,
    "created_at": "2026-09-12T11:59:00Z",
    "started_at": None,
    "completed_at": None,
    "cancelled_at": None,
}


def _page(*items: dict[str, Any]) -> dict[str, Any]:
    return {"run_id": "run-1", "items": list(items), "limit": 100}


class _Transport:
    def __init__(self, page: dict[str, Any]) -> None:
        self.page = page
        self.requests: list[HttpRequest] = []

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        return self.page


class _AsyncTransport(_Transport):
    async def execute(self, request: HttpRequest) -> Any:  # type: ignore[override]
        return super().execute(request)


def _assert_chain(rollout: contracts.SwarmRollout) -> None:
    assert (rollout.budget_parent_run_id, rollout.pool_id, rollout.task_id) == ("run-1", "pool-1", "task-1")
    assert rollout.logical_intent is not None
    assert (rollout.logical_intent.source, rollout.logical_intent.accepted_submissions) == ("derived", 2)
    execution = rollout.execution
    assert execution is not None and execution.lease is None
    assert execution.lease_disposition.status == "not_applicable"
    assert execution.executed_image_digest == DIGEST
    assert execution.launches[0].role == "agent"
    assert [artifact.artifact_id for artifact in rollout.artifacts] == ["artifact-1"]


def test_sync_transport_decodes_the_provenance_chain() -> None:
    transport = _Transport(_page(ROLLOUT))
    (rollout,) = SwarmsAPI(transport).rollouts(SwarmId("run-1"))  # type: ignore[arg-type]
    _assert_chain(rollout)
    assert transport.requests[0].path == "/smr/runs/run-1/rollouts"


def test_async_transport_decodes_the_same_chain() -> None:
    transport = _AsyncTransport(_page(ROLLOUT))
    api = AsyncSwarmsAPI(transport)  # type: ignore[arg-type]
    (rollout,) = asyncio.run(api.rollouts(SwarmId("run-1"), limit=5))
    _assert_chain(rollout)
    assert transport.requests[0].path == "/smr/runs/run-1/rollouts"
    assert transport.requests[0].query == {"limit": 5}


def test_legacy_rows_decode_as_unknown_not_as_absent_execution() -> None:
    legacy = {k: v for k, v in ROLLOUT.items() if k not in {"logical_intent", "execution", "artifacts"}}
    (rollout,) = SwarmsAPI(_Transport(_page(legacy))).rollouts(SwarmId("run-1"))  # type: ignore[arg-type]
    assert rollout.logical_intent is None and rollout.execution is None and rollout.artifacts == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["execution"].pop("lease"),  # a lease must be stated, even as null
        lambda r: r["execution"]["lease_disposition"].update(status="recorded"),
        lambda r: r["execution"].update(lease={"lease_id": "nearby-lease"}),
        lambda r: r["execution"].update(executed_image_digest="latest"),
        lambda r: r["logical_intent"].update(accepted_submissions=0),
        lambda r: r["logical_intent"].update(intent_id="derived-cannot-name-one"),
        lambda r: r["logical_intent"].update(source="caller"),
        lambda r: r.update(artifacts={"artifact_id": "a"}),
    ],
)
def test_malformed_provenance_is_refused(mutate) -> None:
    rollout = copy.deepcopy(ROLLOUT)
    mutate(rollout)
    with pytest.raises(ValueError):
        SwarmsAPI(_Transport(_page(rollout))).rollouts(SwarmId("run-1"))  # type: ignore[arg-type]


def test_recorded_lease_decodes_when_the_interface_uses_one() -> None:
    rollout = copy.deepcopy(ROLLOUT)
    rollout["execution"]["lease"] = {"lease_id": "lease-1"}
    rollout["execution"]["lease_disposition"] = {"status": "recorded", "reason": "interactive_lease"}
    (decoded,) = SwarmsAPI(_Transport(_page(rollout))).rollouts(SwarmId("run-1"))  # type: ignore[arg-type]
    assert decoded.execution is not None and decoded.execution.lease == contracts.SwarmRolloutLease("lease-1")


@pytest.mark.parametrize(
    ("mirror", "schema"),
    [
        (contracts.SwarmRollout, "SwarmRolloutSummary"),
        (contracts.SwarmRolloutLogicalIntent, "SwarmRolloutLogicalIntent"),
        (contracts.SwarmRolloutExecution, "SwarmRolloutExecution"),
        (contracts.SwarmRolloutExecutionLaunch, "SwarmRolloutExecutionLaunch"),
        (contracts.SwarmRolloutLease, "SwarmRolloutLease"),
        (contracts.SwarmRolloutLeaseDisposition, "SwarmRolloutLeaseDisposition"),
        (contracts.SwarmRolloutArtifact, "SwarmRolloutArtifact"),
        (contracts.SwarmRolloutReleaseCoordinates, "SwarmRolloutReleaseCoordinates"),
    ],
)
def test_mirrors_hold_lockstep_with_the_backend_contract(mirror, schema) -> None:
    properties = CONTRACT["components"]["schemas"][schema]["properties"]
    assert {field.name for field in dataclasses.fields(mirror)} == set(properties)
