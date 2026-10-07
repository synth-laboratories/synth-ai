"""Correct contracts for the 2026-10-06 audit; expected failures carry finding IDs."""

import dataclasses
import json
from pathlib import Path

import pytest
from synth_ai.sdk.research.contracts.types import RunResourceBindings

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
SPEC = json.loads((FIXTURES / "research_openapi.generated.json").read_text())
FULL = json.loads((FIXTURES / "backend_full_openapi.generated.json").read_text())


@pytest.mark.parametrize(
    "schema",
    [
        "AsyncRuntimeEnsureRequest",
        "SmrRunResourceBindingsRequest",
        "SmrRunnableProjectCreateRequest",
        "SmrAgentModel",
        "SmrProjectTriggerRequest",
    ],
)
def test_vendored_schema_matches_generated__MX04(schema):
    vendored = json.loads((ROOT / "openapi/research-v1.json").read_text())
    assert vendored["components"]["schemas"][schema] == SPEC["components"]["schemas"][schema], (
        f"MX-04: stale vendored {schema}"
    )


def test_explicit_empty_resource_inventories_round_trip__PR02():
    names = {f.name for f in dataclasses.fields(RunResourceBindings)}
    assert "model_file_ids" in names, "PR-02: typed bindings omit model_file_ids"
    bindings = RunResourceBindings(
        model_file_ids=[], external_repository_ids=[], credential_ref_ids=[]
    )
    assert bindings.to_wire() == {
        "model_file_ids": [],
        "external_repository_ids": [],
        "credential_ref_ids": [],
    }, "PR-02: explicit empty inventories must survive serialization"


@pytest.mark.parametrize(
    "code,expected_retryable",
    [
        ("scientific_writer_transferred", False),
        ("transfer_fence_active", True),
        ("provenance_unbound", False),
        ("run_deployment_pins_missing", False),
    ],
)
def test_authority_errors_are_first_class__RW08(code, expected_retryable):
    import httpx
    from synth_ai.sdk.research.errors import ResearchApiError, ResearchStructuredDenialError
    from synth_ai.sdk.research.transport.http import _raise_for_error_response

    response = httpx.Response(
        409 if code in {"scientific_writer_transferred", "transfer_fence_active"} else 422,
        request=httpx.Request("POST", "http://offline.invalid/smr/projects/p/trigger"),
        json={
            "detail": {
                "error_code": code,
                "message": "launch refused",
                "operation_id": "intent",
                "retry_after": "fence_release" if expected_retryable else None,
            }
        },
    )
    with pytest.raises(ResearchApiError) as refused:
        _raise_for_error_response(response, operation_id="intent")
    assert type(refused.value) not in {ResearchStructuredDenialError, ResearchApiError}, (
        f"RW-08: {code} degrades to generic denial"
    )
    assert refused.value.retryable is expected_retryable, f"RW-08: {code} retryability lost"


@pytest.mark.parametrize("field", ["resource_readiness", "runtime_readiness", "checks"])
def test_preflight_keeps_typed_readiness__PR08(field):
    from synth_ai.sdk.research.contracts.types import SmrLaunchPreflight

    assert field in {member.name for member in dataclasses.fields(SmrLaunchPreflight)}, (
        f"PR-08: typed preflight loses {field}"
    )


def test_full_vendored_snapshot_matches_backend__MX04():
    def without_generator_noise(specification):
        specification = json.loads(json.dumps(specification))
        schemas = specification["components"]["schemas"]
        for name in list(schemas):
            if name.startswith("Body_publish_") and name.endswith("_visual"):
                del schemas[name]
        for name in ("ValidationError",):
            schema = schemas.get(name, {})
            for field in ("ctx", "input"):
                schema.get("properties", {}).pop(field, None)
        return specification

    vendored = json.loads((ROOT / "openapi/research-v1.json").read_text())
    assert without_generator_noise(vendored) == without_generator_noise(SPEC), (
        "MX-04: SDK snapshot differs from authoritative generated backend surface"
    )
