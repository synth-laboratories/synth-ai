"""SYN-4845: a research binding may select one exact Forge method release."""

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms import ResearchBinding, ResearchMethodChoice

FORGE = "63000000-0000-4000-8000-000000000001"


def test_binding_without_method_keeps_its_bytes():
    binding = ResearchBinding(forge_project_id=FORGE, record_protocol="forge.records@1")
    assert binding.model_dump_json() == (
        '{"forge_project_id":"' + FORGE + '","record_protocol":"forge.records@1"}'
    )


def test_method_choice_round_trips_with_typed_slots():
    body = (
        '{"forge_project_id":"' + FORGE + '","record_protocol":"forge.records@1",'
        '"method":{"method_id":"classic","revision":"1",'
        '"parameters":{"authoring":"manual","preregistered":false,"replicates":3}}}'
    )
    binding = ResearchBinding.model_validate_json(body)
    assert isinstance(binding.method, ResearchMethodChoice)
    assert binding.model_dump_json() == body


@pytest.mark.parametrize(
    "method",
    [
        {"method_id": "classic"},
        {"method_id": "classic", "revision": "1", "parameters": {"x": 1.5}},
        {"method_id": "classic", "revision": "1", "extra": True},
        {"method_id": "-bad", "revision": "1"},
    ],
)
def test_malformed_method_choice_refuses(method):
    with pytest.raises(ValidationError):
        ResearchBinding.model_validate(
            {"forge_project_id": FORGE, "record_protocol": "p", "method": method}
        )
