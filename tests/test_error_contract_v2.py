"""SYN-4551: the SDK asks for synth.errors.v2 and decodes both error shapes."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from synth_ai.contracts.swarms import CREATE_PROJECT
from synth_ai.core.http.response import (
    ACCEPT_ERRORS_HEADER,
    ERROR_CONTRACT_HEADER,
    ERRORS_V2,
    decode_response,
)
from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    OutcomeUncertainError,
    SynthError,
)
from synth_ai.types.base import ClientTiming, ResponseModel

pytestmark = pytest.mark.unit

CONTRACT = CREATE_PROJECT.contract
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "errors_v2_rolling_compat.json").read_text()
)
CASES = {case["name"]: case for case in FIXTURE["cases"]}


def _decode(status: int, body: object, headers: dict[str, str], *, mutation: bool = False):
    response = httpx.Response(
        status,
        headers={"X-Synth-Contract": CONTRACT, "Content-Type": "application/json", **headers},
        content=json.dumps(body).encode(),
    )
    with pytest.raises(SynthError) as caught:
        decode_response(
            response,
            response.content,
            ResponseModel,
            ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
            None,
            mutation,
            CONTRACT,
        )
    return caught.value


def test_fixture_names_the_wire_constants():
    assert (FIXTURE["accept_header"], FIXTURE["response_header"], FIXTURE["version"]) == (
        ACCEPT_ERRORS_HEADER,
        ERROR_CONTRACT_HEADER,
        ERRORS_V2,
    )


def test_request_advertises_errors_v2():
    from synth_ai.core.http import request as request_module

    source = Path(request_module.__file__).read_text()
    assert f'request.headers["{ACCEPT_ERRORS_HEADER}"] = "{ERRORS_V2}"' in source


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_body_decodes_into_typed_exception_fields(name):
    case = CASES[name]
    error = _decode(case["status"], case["v2"], {ERROR_CONTRACT_HEADER: ERRORS_V2})
    detail = case["v2"]["error"]
    assert error.code == detail["code"]
    assert error.request_id == detail["request_id"]
    assert error.service == detail.get("service")
    assert error.stage == detail.get("stage")
    assert error.cause == detail.get("cause_code")
    assert error.upstream_status == detail.get("upstream_status")


@pytest.mark.parametrize("name", sorted(CASES))
def test_v1_body_falls_back_with_empty_cause_fields(name):
    case = CASES[name]
    error = _decode(case["status"], case["v1"], {})
    assert error.code == case["v1"]["error"]["code"]
    assert (error.service, error.stage, error.cause, error.upstream_status) == (
        None,
        None,
        None,
        None,
    )


def test_v2_message_names_service_stage_cause_and_upstream_status():
    case = CASES["upstream_refusal_with_status"]
    error = _decode(case["status"], case["v2"], {ERROR_CONTRACT_HEADER: ERRORS_V2})
    assert str(error) == (
        "swarms.v1 refused: upstream_unavailable [code=upstream_unavailable status=502 "
        "request_id=req-2 service=orchestra stage=runtime_binding_place "
        "cause=port_owner_unreachable upstream_status=503]"
    )


def test_typed_kind_preserved_with_v2_fields():
    case = CASES["bare_refusal_omits_absent_fields"]
    error = _decode(case["status"], case["v2"], {ERROR_CONTRACT_HEADER: ERRORS_V2})
    assert isinstance(error, AuthenticationError)
    assert error.service == "backend"


def test_mutation_uncertain_error_carries_v2_fields():
    case = CASES["upstream_refusal_with_status"]
    body = {"error": {**case["v2"]["error"], "outcome": "unknown"}}
    error = _decode(case["status"], body, {ERROR_CONTRACT_HEADER: ERRORS_V2}, mutation=True)
    assert isinstance(error, OutcomeUncertainError)
    assert (error.service, error.cause, error.upstream_status) == (
        "orchestra",
        "port_owner_unreachable",
        503,
    )


def test_unknown_error_contract_version_is_a_contract_error():
    # 409: a non-transient status, so the decoder (not the infrastructure
    # classifier) judges the body.
    case = CASES["customer_refusal_with_stage_and_cause"]
    error = _decode(409, case["v2"], {ERROR_CONTRACT_HEADER: "synth.errors.v3"})
    assert isinstance(error, ContractError)


def test_v2_body_without_marker_still_decodes_as_v1_view():
    # ResponseBase ignores unknown keys, so an unmarked body never crashes; the
    # cause fields are only trusted when the backend marks the body v2.
    case = CASES["customer_refusal_with_stage_and_cause"]
    error = _decode(case["status"], case["v2"], {})
    assert error.code == "service_unavailable"
    assert error.service is None and error.cause is None


def test_malformed_v2_token_is_a_contract_error():
    body = {"error": {**CASES["customer_refusal_with_stage_and_cause"]["v2"]["error"]}}
    body["error"]["cause_code"] = "free text with spaces"
    error = _decode(409, body, {ERROR_CONTRACT_HEADER: ERRORS_V2})
    assert isinstance(error, ContractError)
