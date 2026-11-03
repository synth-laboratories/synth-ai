"""Current backend additions remain expressible without changing older launch bytes."""
import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.resources import ResourceRequest


def body(**extra):
    return dict(project_id="72000000-0000-4000-8000-000000000002", message="run",
        limits=dict(maximum_charge=dict(coefficient=1, scale=0), wall_time_seconds=10,
                    maximum_tasks=2, maximum_parallel_tasks=1), **extra)


def test_parent_hold_reference_is_exact_and_absence_omitted():
    assert "parent_hold" not in LaunchRequest(**body()).model_dump(mode="json")
    reference = "rsv_" + "0" * 26
    assert LaunchRequest(**body(parent_hold={"reservation_ref": reference})).parent_hold == {"reservation_ref": reference}


@pytest.mark.parametrize("hold", [{"reservation_ref":"invalid"}, {"reservation_ref":"rsv_"+"0"*26,"amount":"1"}, {}])
def test_parent_hold_rejects_nonreference_authority(hold):
    with pytest.raises(ValidationError):
        LaunchRequest(**body(parent_hold=hold))


def test_modal_resource_dimensions_match_current_backend():
    resource = ResourceRequest(plugin="modal.compute", roles=("worker",), limits={"gpu_seconds":1,"cpu_seconds":2})
    assert resource.limits == {"gpu_seconds":1,"cpu_seconds":2}
