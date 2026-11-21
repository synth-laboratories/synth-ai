"""SYN-5695 J-refusal steps: whitelist, contract-invalid launches, expired and missing keys, overload, Index, rate window.

Shares the loopback harness in journey_loopback.py; split from test_journeys.py
so each file stays under the 120 s pytest cap.
"""

from __future__ import annotations

import pytest
from journey_loopback import (
    SURFACES,
    api,
    example,
    launch_window,
    problem,
    run,
)

from scripts.acceptance import journey_main, journeys
from scripts.acceptance.journeys import (
    Secret,
    Step,
    Verdict,
    capabilities,
    matrix_operations,
)
from scripts.acceptance.surfaces import CLI_ARGV, SDK_CALLS, Call, SurfaceName

pytestmark = pytest.mark.acceptance


def test_capabilities_name_only_contract_operations(contract):
    operations = {
        operation["operationId"]
        for methods in contract["paths"].values()
        for method, operation in methods.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }
    assert set(SDK_CALLS) <= operations
    assert set(CLI_ARGV) <= operations
    document = capabilities()
    assert set(document["steps"]) == {step.value for step in Step}
    matrix = {row[0] for row in matrix_operations()}
    assert {"getRun", "getFile", "revokeApiKey", "getCheckout"} <= matrix
    assert "getAccount" not in matrix


@pytest.mark.parametrize("surface", SURFACES)
def test_whitelist_refusal_is_typed_and_launch_not_accepted(tmp_path, surface):
    def responder(method, path, _):
        return 403, problem("optimizer_topology_not_allowed", 403), {}

    result, requests = run(surface, responder, tmp_path, Step.SKU_REFUSAL)
    assert result.verdict == Verdict.PASS, result
    assert [(method, path) for method, path, _ in requests] == [("POST", "/v1/runs")]


@pytest.mark.parametrize("surface", SURFACES)
def test_accepted_probe_is_cancelled_and_fails(contract, tmp_path, surface):
    snapshot = example(contract, "/v1/runs/{run_id}")

    def responder(method, path, _):
        return (202 if path == "/v1/runs" else 200), snapshot, {}

    result, requests = run(surface, responder, tmp_path, Step.TOPOLOGY_VERSION)
    assert result.verdict == Verdict.FAIL
    assert result.cause == "refusal_expected_but_accepted"
    assert any(path.endswith("/cancel") for _, path, _ in requests)


# The SDK refuses a contract-invalid launch with a raw pydantic error that has
# no public code (reported on SYN-5695/SYN-5666). CLI and MCP refuse it typed.


CLIENT_SIDE = {
    SurfaceName.SDK: (Verdict.PASS, ""),
    SurfaceName.CLI: (Verdict.PASS, ""),
    SurfaceName.MCP_STDIO: (Verdict.PASS, ""),
}


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("step", [Step.OVERSIZE, Step.TOPOLOGY])
def test_contract_invalid_launch_never_reaches_core(tmp_path, surface, step):
    def responder(method, path, _):
        raise AssertionError("an invalid launch must not reach Core")

    result, requests = run(surface, responder, tmp_path, step)
    assert requests == []
    expected = CLIENT_SIDE[surface]
    assert (result.verdict, result.cause) == expected, result


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("status", "code", "verdict"),
    [
        (401, "api_key_invalid", Verdict.PASS),
        (401, "authentication_required", Verdict.FAIL),
        (401, "unauthenticated", Verdict.FAIL),
        (403, "api_key_invalid", Verdict.FAIL),
    ],
)
def test_expired_key_requires_the_exact_problem(tmp_path, surface, status, code, verdict):
    def responder(method, path, headers):
        assert headers["bearer"] == "fixture-expired"
        return status, problem(code, status), {}

    result, _ = run(
        surface,
        responder,
        tmp_path,
        Step.EXPIRED_KEY,
        secrets={Secret.EXPIRED.value: "fixture-expired"},
    )
    assert result.verdict == verdict, result


@pytest.mark.parametrize("surface", SURFACES)
def test_missing_credential_is_distinct_from_a_rejected_api_key(tmp_path, monkeypatch, surface):
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "absent.json"))

    def responder(method, path, request):
        pytest.fail("a locally missing credential reached the wire")

    with api(responder) as (origin, requests):
        make = journey_main.surface_factory(
            surface, origin=origin, work_directory=tmp_path, mcp_url=None
        )
        with make("") as driver:
            response = driver.call(Call("getAccount"))
    assert not response.accepted
    assert requests == []
    assert response.status is None
    assert response.code != "api_key_invalid"


@pytest.mark.parametrize("surface", SURFACES)
def test_overload_surfaces_retryable_after_bounded_retries(tmp_path, surface):
    def responder(method, path, _):
        return 503, problem("overloaded", 503, retryable=True), {"Retry-After": "1"}

    result, requests = run(surface, responder, tmp_path, Step.OVERLOAD_REFUSAL)
    assert result.verdict == Verdict.PASS, result
    assert len(requests) >= 2  # The client retried before surfacing the refusal.


ADMISSION_FULL = {"secrets": {"SYNTH_JOURNEY_OVERLOAD_MODE": "admission_queue_full"}}


@pytest.mark.parametrize("surface", SURFACES)
def test_hosted_overload_launch_refused_run_queue_full(tmp_path, surface):
    def responder(method, path, request):
        assert (method, path) == ("POST", "/v1/runs")
        # A budget below Core's call bound is refused 422 before admission.
        assert request["body"]["limits"]["maximum_charge"]["amount"] == "2.50"
        return 429, problem("run_queue_full", 429, retryable=True), {"Retry-After": "1"}

    result, requests = run(surface, responder, tmp_path, Step.OVERLOAD_REFUSAL, **ADMISSION_FULL)
    assert result.verdict == Verdict.PASS, result
    assert all(method == "POST" for method, _, _ in requests)


@pytest.mark.parametrize("surface", SURFACES)
def test_hosted_overload_uses_the_saturated_org_key(tmp_path, surface):
    def responder(method, path, request):
        assert request["bearer"] == "fixture-saturated"
        return 429, problem("run_queue_full", 429, retryable=True), {"Retry-After": "1"}

    options = {
        "secrets": {
            **ADMISSION_FULL["secrets"],
            "SYNTH_JOURNEY_OVERLOAD_API_KEY": "fixture-saturated",
        }
    }
    result, requests = run(surface, responder, tmp_path, Step.OVERLOAD_REFUSAL, **options)
    assert result.verdict == Verdict.PASS, result
    assert requests and {bearer for _, _, bearer in requests} == {"fixture-saturated"}


@pytest.mark.parametrize("surface", SURFACES)
def test_hosted_overload_rate_window_429_does_not_pass(tmp_path, surface):
    def responder(method, path, _):
        return 429, problem("rate_limit_exceeded", 429, retryable=True), {"Retry-After": "1"}

    result, _ = run(surface, responder, tmp_path, Step.OVERLOAD_REFUSAL, **ADMISSION_FULL)
    assert result.verdict == Verdict.FAIL, result
    assert result.cause == "refusal_code_unexpected"


@pytest.mark.parametrize("surface", SURFACES)
def test_hosted_overload_accepted_launch_fails_and_cancels(tmp_path, surface):
    contract = journeys.contract()
    launched = example(contract, "/v1/runs", "post", "202")
    cancel_path = next(p for p in contract["paths"] if p.endswith("/cancel") and "/runs/" in p)
    cancelled = example(contract, cancel_path, "post", "202")

    def responder(method, path, _):
        if path.endswith("/cancel"):
            return 202, cancelled, {}
        return 202, launched, {}

    result, requests = run(surface, responder, tmp_path, Step.OVERLOAD_REFUSAL, **ADMISSION_FULL)
    assert result.verdict == Verdict.FAIL, result
    assert result.cause == "refusal_expected_but_accepted"
    assert any(method == "POST" and path.endswith("/cancel") for method, path, _ in requests)


@pytest.mark.parametrize("surface", SURFACES)
def test_index_is_refused_on_every_surface(tmp_path, surface):
    def responder(method, path, _):
        assert path.startswith("/v1/index/")
        return 404, problem("index_not_available", 404), {}

    result, _ = run(surface, responder, tmp_path, Step.INDEX)
    assert result.verdict == Verdict.PASS, result


def test_generic_route_miss_for_index_is_not_a_typed_refusal(tmp_path):
    def responder(method, path, _):
        return 404, problem("route_not_found", 404), {}

    result, _ = run(SurfaceName.SDK, responder, tmp_path, Step.INDEX)
    assert result.verdict == Verdict.FAIL, result


def test_missing_precondition_is_inconclusive_without_requests(tmp_path):
    def responder(method, path, _):
        raise AssertionError("no request without its precondition")

    for step in (Step.MINT_KEY, Step.CLI_OAUTH, Step.RAISE_LIMIT, Step.CROSS_ORG_404):
        result, requests = run(SurfaceName.SDK, responder, tmp_path / step, step)
        assert result.verdict == Verdict.INCONCLUSIVE
        assert result.cause.startswith("precondition_missing:")
        assert requests == []


@pytest.mark.parametrize("surface", SURFACES)
def test_rate_limit_refusal_waits_retry_after_then_ends_on_pre_effect_refusal(tmp_path, surface):
    responder, state = launch_window(
        3, 2, then=(422, problem("optimizer_version_unavailable", 422), {})
    )
    result, _ = run(surface, responder, tmp_path, Step.RATE_LIMIT_REFUSAL)
    assert (result.verdict, result.cause) == (Verdict.PASS, ""), result
    probes = [at for at, body in state["posts"] if body != {}]
    assert len(probes) >= 2 and probes[-1] >= state["reopens"]  # retried only after reopening
    assert result.evidence[-1]["fill_statuses"] == [422, 422, 422, 429]


@pytest.mark.parametrize("surface", SURFACES)
def test_rate_limit_refusal_that_never_reopens_surfaces_typed_retryable_429(tmp_path, surface):
    responder, _ = launch_window(1, 600, then=(500, {}, {}))
    result, _ = run(surface, responder, tmp_path, Step.RATE_LIMIT_REFUSAL)
    assert result.verdict == Verdict.PASS, result
    assert result.evidence[0]["code"] == "rate_limit_exceeded"


def test_rate_limit_refusal_without_a_saturated_window_is_inconclusive(tmp_path):
    def responder(method, path, _):
        return 422, problem("invalid_launch_request", 422), {}

    result, requests = run(SurfaceName.SDK, responder, tmp_path, Step.RATE_LIMIT_REFUSAL)
    assert (result.verdict, result.cause) == (
        Verdict.INCONCLUSIVE,
        "precondition_not_met:launch_window_not_saturated",
    )
    assert len(requests) == 12


@pytest.mark.parametrize("surface", SURFACES)
def test_refusal_cells_keep_the_problem_detail_and_guide_link(tmp_path, surface):
    """SYN-5666: the J-refusals reducer compares each kept refusal body with its guide entry."""
    guide = "https://docs.usesynth.ai/errors/optimizer_topology_not_allowed"
    detail = "The swarms_v2 topology is not allowed for this organization."

    def responder(method, path, _):
        body = problem("optimizer_topology_not_allowed", 403, type=guide, detail=detail)
        return 403, body, {}

    result, _ = run(surface, responder, tmp_path, Step.SKU_REFUSAL)
    assert result.verdict == Verdict.PASS, result
    refusals = [item for item in result.evidence if item.get("accepted") is False]
    assert refusals, result
    assert all((item["detail"], item["guide"]) == (detail, guide) for item in refusals), refusals


def test_local_refusal_cell_keeps_its_detail_without_a_guide_link():
    from scripts.acceptance.surfaces import Outcome

    cell = Outcome("createRun", False, code="client_error", detail="unreadable file").evidence()
    assert (cell["detail"], cell["guide"]) == ("unreadable file", None)
    assert "detail" not in Outcome("getRun", True, {}).evidence()
