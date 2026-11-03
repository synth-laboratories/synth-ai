import json

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.resources import ResourceGrantSet, ResourceRequest, TraceSelection

PROJECT = "00000000-0000-0000-0000-00000000000f"
OTHER_PROJECT = "00000000-0000-0000-0000-000000000010"


def declaration():
    return {
        "plugin": "mnemosyne.trace",
        "roles": ["worker", "reviewer"],
        "limits": {"spend_micros": 0, "calls": 10},
        "selection": {
            "project_ids": [PROJECT],
            "operations": ["read", "retain", "compare"],
            "role_operations": {"worker": ["read", "retain"], "reviewer": ["read", "compare"]},
        },
    }


def test_trace_selection_round_trips_without_implying_admission():
    request = ResourceRequest.model_validate_json(json.dumps(declaration()))
    assert isinstance(request.selection, TraceSelection)
    assert request.model_dump(mode="json")["selection"] == declaration()["selection"]
    grant = {
        **request.model_dump(mode="json"),
        "funding_lane": "platform",
        "registry_digest": "a" * 64,
        "grant_id": "trace-grant",
    }
    result = ResourceGrantSet.model_validate_json(json.dumps({"grants": [grant]}))
    assert isinstance(result.grants[0].selection, TraceSelection)


@pytest.mark.parametrize(
    "change", ["missing_role", "extra_role", "widen_role", "duplicate", "project"]
)
def test_trace_selection_refuses_widening_and_ambiguous_scope(change):
    value = declaration()
    selection = value["selection"]
    if change == "missing_role":
        selection["role_operations"].pop("reviewer")
    elif change == "extra_role":
        selection["role_operations"]["planner"] = ["read"]
    elif change == "widen_role":
        selection["operations"] = ["read"]
    elif change == "duplicate":
        selection["role_operations"]["worker"] = ["read", "read"]
    else:
        selection["project_ids"] = [PROJECT, OTHER_PROJECT]
    with pytest.raises(ValidationError):
        ResourceRequest.model_validate_json(json.dumps(value))


def test_trace_project_must_equal_launch_project():
    launch = {
        "message": "run",
        "project_id": OTHER_PROJECT,
        "limits": {
            "maximum_charge": None,
            "wall_time_seconds": 1,
            "maximum_tasks": 1,
            "maximum_parallel_tasks": 1,
        },
        "resources": [declaration()],
    }
    with pytest.raises(ValidationError, match="Trace resource project"):
        LaunchRequest.model_validate_json(json.dumps(launch))
    launch["project_id"] = PROJECT
    assert LaunchRequest.model_validate_json(json.dumps(launch)).project_id == PROJECT
