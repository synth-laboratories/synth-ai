"""SMR quota refusals decode into the typed QuotaExceededError (SYN-4844).

Fixtures in ``tests/fixtures/smr_quota_refusals_v1.json`` are the backend's
real wire bodies, recorded from ``http_exception_structured_409_handler`` on
backend#3588 (see the fixture's ``source``).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from synth_ai.contracts.swarms import CREATE_PROJECT
from synth_ai.core.http.quota import decode_quota_refusal
from synth_ai.core.http.response import decode_response
from synth_ai.errors import ContractError, QuotaExceededError, RefusedError
from synth_ai.types.base import ClientTiming, ResponseModel
from synth_ai.types.identity import MutationIdentity

pytestmark = pytest.mark.unit

FIXTURES = json.loads((Path(__file__).parent / "fixtures/smr_quota_refusals_v1.json").read_text())[
    "cases"
]
CONTRACT = CREATE_PROJECT.contract


def _case(name: str) -> tuple[int, bytes]:
    case = FIXTURES[name]
    return case["status"], json.dumps(case["body"]).encode()


def _decode(status: int, body: bytes, *, mutation: bool = True) -> None:
    response = httpx.Response(
        status,
        headers={"X-Synth-Contract": CONTRACT, "Content-Type": "application/json"},
        content=body,
    )
    decode_response(
        response,
        response.content,
        ResponseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        MutationIdentity.create("op", b"{}", "/api/v1/x", key="k"),
        mutation,
        CONTRACT,
        operation="smr.projects.create",
    )


@pytest.mark.parametrize(
    ("name", "code", "scope", "limit", "used"),
    [
        ("plan_active_limit_403", "smr_project_plan_active_limit_reached", "organization", 4, 4),
        ("active_cap_409", "smr_project_active_limit_reached", "organization", 10, 10),
        ("nonterminal_run_cap_409", "smr_project_nonterminal_run_limit_reached", "project", 10, 3),
    ],
)
def test_recorded_quota_bodies_decode_into_typed_quota_error(name, code, scope, limit, used):
    status, body = _case(name)
    with pytest.raises(QuotaExceededError) as caught:
        _decode(status, body)
    error = caught.value
    assert isinstance(error, RefusedError)
    assert (error.status, error.code, error.retryable) == (status, code, False)
    assert (error.scope, error.limit, error.used) == (scope, limit, used)
    assert (error.service, error.stage) == ("smr", FIXTURES[name]["body"]["detail"]["stage"])
    assert error.remedy and error.cause
    assert "smr.projects.create refused" in str(error)


def test_read_path_decodes_the_same_typed_error():
    status, body = _case("plan_active_limit_403")
    with pytest.raises(QuotaExceededError):
        _decode(status, body, mutation=False)


def test_null_limit_is_preserved():
    status, body = _case("plan_active_limit_403")
    payload = json.loads(body)
    payload["detail"]["limit"] = None
    error = decode_quota_refusal(status, json.dumps(payload).encode())
    assert (error.limit, error.used) == (None, 4)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("used"),
        lambda d: d.update(limit="ten"),
        lambda d: d.update(scope="galaxy"),
        lambda d: d.update(used=-1),
    ],
)
def test_quota_code_with_unknown_body_shape_is_a_contract_error(mutate):
    status, body = _case("active_cap_409")
    payload = json.loads(body)
    mutate(payload["detail"])
    with pytest.raises(ContractError):
        _decode(status, json.dumps(payload).encode(), mutation=False)


def test_legacy_string_detail_is_still_a_contract_error():
    with pytest.raises(ContractError):
        _decode(403, b'{"detail": "Active project limit reached for your plan."}', mutation=False)


def test_decoder_refuses_unknown_codes_and_statuses():
    status, body = _case("active_cap_409")
    with pytest.raises(ContractError):
        decode_quota_refusal(500, body)
    payload = json.loads(body)
    payload["detail"]["error_code"] = "something_else"
    with pytest.raises(ContractError):
        decode_quota_refusal(status, json.dumps(payload).encode())
