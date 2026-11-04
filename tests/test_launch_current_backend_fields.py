"""Current backend additions remain expressible without changing older launch bytes."""
import asyncio
import inspect

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.client import AsyncSwarmsClient, SwarmsClient
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.resources import ResourceRequest
from synth_ai.types.usage import ResourceUsage


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


@pytest.mark.parametrize("client_type", [SwarmsClient, AsyncSwarmsClient])
@pytest.mark.parametrize("operation", ["preflight", "create"])
def test_parent_hold_reaches_each_launch_surface_without_reauthoring(client_type, operation):
    class Captured(Exception):
        pass

    sent = []

    class Transport:
        def request(self, _operation, _model, **kwargs):
            sent.append(kwargs["body"].model_dump(mode="json"))
            raise Captured

    class AsyncTransport(Transport):
        async def request(self, *args, **kwargs):
            return super().request(*args, **kwargs)

    hold = {"reservation_ref": "rsv_" + "0" * 26}
    request = LaunchRequest(**body(parent_hold=hold))
    client = client_type(AsyncTransport() if client_type is AsyncSwarmsClient else Transport())
    with pytest.raises(Captured):
        value = getattr(client, operation)(
            message=request.message, project_id=request.project_id,
            limits=request.limits, parent_hold=hold,
        )
        if inspect.isawaitable(value):
            asyncio.run(value)
    assert sent == [request.model_dump(mode="json")]


@pytest.mark.parametrize("dimension", ["cpu_seconds", "gpu_seconds"])
def test_modal_usage_decodes_without_turning_unknown_amount_into_zero(dimension):
    row = ResourceUsage(plugin="modal.compute", dimension=dimension, usage_class="tool",
                        role="worker", state="measured", amount=2)
    assert row.dimension == dimension and row.amount == 2
    unknown = ResourceUsage(plugin="modal.compute", dimension=dimension, usage_class="tool",
                            role="worker", state="unknown", amount=None)
    assert unknown.amount is None
    with pytest.raises(ValidationError):
        ResourceUsage(plugin="modal.compute", dimension=dimension, usage_class="tool",
                      role="worker", state="unknown", amount=0)
