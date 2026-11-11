import pytest
from pydantic import ValidationError

from synth_ai.models import OwnerResourceRequest


def request(templates):
    return {
        "plugin": "pool",
        "roles": ["planner", "worker"],
        "limits": {"spend_micros": 1000, "rollouts": 3},
        "selection": {},
        "evaluation_templates": templates,
    }


def test_registered_template_reaches_launch_wire():
    revision = "sha256:" + "a" * 64
    resource = OwnerResourceRequest.model_validate(request([revision]))
    wire = resource.model_dump(exclude_unset=True)
    assert wire["evaluation_templates"] == [revision]
    assert wire["selection"] == {}


@pytest.mark.parametrize(
    "templates", [[], ["latest"], ["sha256:" + "A" * 64], ["sha256:" + "a" * 64] * 2]
)
def test_invalid_template_revisions_refuse_locally(templates):
    with pytest.raises(ValidationError):
        OwnerResourceRequest.model_validate(request(templates))
