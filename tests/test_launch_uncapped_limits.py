"""Explicit null maximum_charge requests operator-funded local execution."""

import json

import pytest
from pydantic import ValidationError
from synth_ai.errors import ContractError
from synth_ai.sdk.swarms import Limits
from synth_ai.sdk.swarms.client import launch_request, validate_preflight
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.money import Money
from tests.http_fixtures import METADATA

FINITE = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
UNCAPPED = FINITE.model_copy(update={"maximum_charge": None})


def _preflight(resolved: Limits) -> LaunchPreflight:
    configuration = {
        "metadata": METADATA,
        "run_id": None,
        "project_id": "p",
        "configuration_id": "cfg",
        "digest_sha256": "d" * 64,
        "profile_id": "prof",
        "runtime_image_digest": "e" * 64,
        "limits": resolved.model_dump(mode="json"),
        "inputs": {"files": []},
        "providers": {"default": "openai"},
        "models": {"default": "gpt-6-luna"},
    }
    return LaunchPreflight.model_validate_json(
        json.dumps(
            {
                "metadata": METADATA,
                "ready": True,
                "blockers": [],
                "configuration": configuration,
                "expires_at": "2026-10-20T00:00:00Z",
            }
        )
    )


def test_explicit_null_survives_serialization_even_when_nones_are_dropped():
    request = launch_request(None, "p", UNCAPPED, None, None, None, message="m")
    for body in (
        json.loads(request.model_dump_json()),
        json.loads(request.model_dump_json(exclude_none=True)),
    ):
        assert "maximum_charge" in body["limits"]
        assert body["limits"]["maximum_charge"] is None


def test_maximum_charge_stays_required():
    with pytest.raises(ValidationError):
        Limits(wall_time_seconds=60, maximum_tasks=1, maximum_parallel_tasks=1)


def test_negative_finite_charge_still_refuses():
    with pytest.raises(ValidationError):
        FINITE.model_validate(
            {
                **FINITE.model_dump(mode="json"),
                "maximum_charge": {"coefficient": -1, "scale": 0, "currency": "USD"},
            }
        )


def test_preflight_cannot_widen_a_finite_ceiling_to_uncapped():
    request = launch_request(None, "p", FINITE, None, None, None, message="m")
    validate_preflight(_preflight(FINITE), request)
    with pytest.raises(ContractError):
        validate_preflight(_preflight(UNCAPPED), request)


def test_uncapped_request_accepts_whatever_the_backend_resolves():
    request = launch_request(None, "p", UNCAPPED, None, None, None, message="m")
    assert (
        validate_preflight(_preflight(UNCAPPED), request).configuration.limits.maximum_charge
        is None
    )
    validate_preflight(_preflight(FINITE), request)
