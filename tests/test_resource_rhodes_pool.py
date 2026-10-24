import json

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.types.resources import ResourceRequest


def grant(**overrides):
    value = {
        "plugin": "rhodes.pool",
        "roles": ["worker"],
        "limits": {"rollouts": 4, "spend_micros": 20_000_000},
        "selection": {"pool_ids": ["pool_da4285b35553"]},
    }
    value.update(overrides)
    return value


def test_rhodes_pool_grant_round_trips_with_its_pool_selection():
    request = ResourceRequest.model_validate_json(json.dumps(grant()))
    assert request.model_dump(mode="json")["selection"] == {"pool_ids": ["pool_da4285b35553"]}


@pytest.mark.parametrize(
    "overrides",
    [
        {"selection": None},
        {"selection": {"pool_ids": []}},
        {"selection": {"pool_ids": ["pool_a", "pool_a"]}},
        {"selection": {"pool_ids": ["pool a"]}},
        {"collections": ["c"]},
    ],
)
def test_rhodes_pool_grant_refuses_bad_selection(overrides):
    with pytest.raises(ValidationError):
        ResourceRequest.model_validate_json(json.dumps(grant(**overrides)))


def test_other_plugins_cannot_carry_a_pool_selection():
    with pytest.raises(ValidationError):
        ResourceRequest.model_validate_json(
            json.dumps(
                grant(plugin="index.search", limits={"index_searches": 1}, collections=["c"])
            )
        )
