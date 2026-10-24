import json

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.resources import ProjectSelection, ResourceGrantSet, ResourceRequest

PROJECT = "00000000-0000-0000-0000-00000000000f"


def request(**updates):
    value = {
        "plugin": "forge.research",
        "roles": ["worker"],
        "limits": {"spend_micros": 0, "calls": 64},
        "selection": {"project_ids": [PROJECT]},
    }
    return ResourceRequest.model_validate_json(json.dumps({**value, **updates}))


def test_project_selection_round_trips_as_known_plugin():
    item = request()
    assert isinstance(item.selection, ProjectSelection)
    assert item.model_dump(mode="json")["selection"] == {"project_ids": [PROJECT]}
    grant = {
        **item.model_dump(mode="json"),
        "funding_lane": "platform",
        "registry_digest": "a" * 64,
        "grant_id": "forge-grant",
    }
    result = ResourceGrantSet.model_validate_json(json.dumps({"grants": [grant]}))
    assert result.grants[0].plugin == "forge.research"
    assert result.grants[0].selection.project_ids == (PROJECT,)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {"project_ids": []},
        {"project_ids": [PROJECT, PROJECT]},
        {"pool_ids": ["pool-a"]},
        {"project_ids": ["invalid"]},
        {"project_ids": [PROJECT], "pool_ids": ["pool-a"]},
    ],
)
def test_project_selection_is_closed_and_distinct(selection):
    with pytest.raises(ValidationError):
        request(selection=selection)


@pytest.mark.parametrize("plugin", ["rhodes.pool", "index.search", "openrouter.inference"])
def test_other_plugins_refuse_forge_project_selection(plugin):
    with pytest.raises(ValidationError):
        request(plugin=plugin)


def test_launch_admits_four_resource_requests():
    assert 4 in [
        getattr(item, "max_length", None)
        for item in LaunchRequest.model_fields["resources"].metadata
    ]
