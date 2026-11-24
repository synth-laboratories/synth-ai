"""Index limits on a launch: SYN-5492."""

import pytest
from pydantic import ValidationError

from synth_ai.models import OwnerResourceRequest

REQUEST = {
    "plugin": "index",
    "roles": ["planner", "worker"],
    "limits": {"spend_micros": 0, "index_searches": 20, "calls": 60},
    "selection": {},
}


def test_index_limits_keep_their_wire_shape():
    resource = OwnerResourceRequest.model_validate(REQUEST)
    assert resource.model_dump(mode="json", exclude_unset=True) == REQUEST


def test_index_opt_out_is_expressible():
    limits = {"spend_micros": 0, "index_searches": 0}
    resource = OwnerResourceRequest.model_validate({**REQUEST, "limits": limits})
    assert resource.limits["index_searches"] == 0


@pytest.mark.parametrize("change", [{"plugin": "forge"}, {"collections": ["private"]}])
def test_unknown_plugins_and_fields_are_refused(change):
    with pytest.raises(ValidationError):
        OwnerResourceRequest.model_validate({**REQUEST, **change})
