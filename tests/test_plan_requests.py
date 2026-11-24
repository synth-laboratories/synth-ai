"""Special Free plan requests through the SDK, CLI and MCP dispatch (SYN-5754).

Every case runs against an injected in-memory HTTP transport. The use case is
customer content, so each path asserts a sentinel never reaches a log or stream.
"""

import json
import logging

import httpx
import pytest
from pydantic import JsonValue
from test_cli import api, example, invoke  # noqa: F401  (api is the CLI fixture)

from synth_ai import AsyncSynthClient, RetryPolicy, SynthClient, cli
from synth_ai._generated.models import PlanRequestState
from synth_ai.diagnostics import OperationDiagnostic
from synth_ai.errors import (
    ConflictError,
    InvalidRequestError,
    RateLimitedError,
    SwarmsAPIError,
    local_problem,
)
from synth_ai.mcp.dispatch import dispatch

SENTINEL = "SENTINEL-USE-CASE-7f3a climbing agents for a robotics lab"
LINK = "https://example.test/SENTINEL-LINK-7f3a"
REQUEST_ID = "00000000-0000-4000-8000-000000000054"
COOLDOWN_SECONDS = 1209600  # Core's 14-day denial cooldown.
REFUSALS = [
    ("requestSpecialFree", "plan_request_pending_exists", 409, ConflictError, None),
    ("requestSpecialFree", "plan_request_cooldown", 429, RateLimitedError, COOLDOWN_SECONDS),
    ("requestSpecialFree", "plan_request_not_needed", 409, ConflictError, None),
    ("requestSpecialFree", "plan_request_too_large", 422, InvalidRequestError, None),
    ("withdrawPlanRequest", "plan_request_not_pending", 409, ConflictError, None),
]


class FakeCore:
    """An in-memory Core: contract examples, or one typed refusal."""

    def __init__(self, contract, refusal=None):
        self.contract = contract
        self.refusal = refusal
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.refusal is not None:
            code, status, retry_after = self.refusal
            problem = local_problem(code, "plan_request", status=status).model_dump(mode="json")
            headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
            return httpx.Response(status, headers=headers, json=problem)
        if request.method == "POST":
            status, body = example(self.contract, "requestSpecialFree")
        elif request.method == "DELETE":
            status, body = example(self.contract, "withdrawPlanRequest")
        else:
            status, body = example(self.contract, "listPlanRequests")
        return httpx.Response(status, json=body)


def client_for(core, diagnostics=None):
    return AsyncSynthClient(
        "fixture",
        base_url="https://example.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(core)),
        retry_policy=RetryPolicy(attempts=3, backoff_seconds=0, jitter_seconds=0),
        diagnostic_hook=None if diagnostics is None else diagnostics.append,
    )


async def call(client, operation):
    if operation == "requestSpecialFree":
        return await client.account.request_special_free(use_case=SENTINEL, links=[LINK])
    return await client.account.withdraw_plan_request(REQUEST_ID)


def leaked(*texts):
    return [text for text in texts if "SENTINEL" in text]


async def test_request_sends_the_use_case_once_and_never_logs_it(contract, caplog, capsys):
    core, diagnostics = FakeCore(contract), []
    caplog.set_level(logging.DEBUG)
    async with client_for(core, diagnostics) as client:
        request = await client.account.request_special_free(
            use_case=SENTINEL, links=[LINK], idempotency_key="plan-request-1"
        )
    assert request.request_id == REQUEST_ID and request.state is PlanRequestState.PENDING
    (sent,) = core.requests
    assert (sent.method, sent.url.path) == ("POST", "/v1/plan-requests")
    assert json.loads(sent.content) == {"use_case": SENTINEL, "links": [LINK]}
    assert sent.headers["Idempotency-Key"] == "plan-request-1"
    assert diagnostics and all(isinstance(event, OperationDiagnostic) for event in diagnostics)
    captured = capsys.readouterr()
    assert not leaked(caplog.text, captured.out, captured.err, repr(diagnostics), repr(request))


async def test_request_without_links_sends_an_empty_list_and_a_durable_key(contract):
    core = FakeCore(contract)
    async with client_for(core) as client:
        await client.account.request_special_free(use_case=SENTINEL)
    (sent,) = core.requests
    assert json.loads(sent.content) == {"use_case": SENTINEL, "links": []}
    assert sent.headers["Idempotency-Key"]


async def test_list_and_withdraw_return_status_projections(contract):
    core = FakeCore(contract)
    async with client_for(core) as client:
        listed = await client.account.plan_requests()
        withdrawn = await client.account.withdraw_plan_request(REQUEST_ID)
    assert [item.state for item in listed.items] == [PlanRequestState.DENIED]
    assert listed.limit == 100 and withdrawn.state is PlanRequestState.WITHDRAWN
    assert [(r.method, r.url.path) for r in core.requests] == [
        ("GET", "/v1/plan-requests"),
        ("DELETE", "/v1/plan-requests/" + REQUEST_ID),
    ]


@pytest.mark.parametrize(("operation", "code", "status", "kind", "retry_after"), REFUSALS)
async def test_each_refusal_is_typed_unretried_and_silent(
    contract, caplog, capsys, operation, code, status, kind, retry_after
):
    core, diagnostics = FakeCore(contract, refusal=(code, status, retry_after)), []
    caplog.set_level(logging.DEBUG)
    async with client_for(core, diagnostics) as client:
        with pytest.raises(SwarmsAPIError) as caught:
            await call(client, operation)
    error = caught.value
    assert type(error) is kind
    assert (error.code, error.status) == (code, status)
    assert error.retry_after_seconds == retry_after
    assert len(core.requests) == 1  # A 14-day cooldown must never be slept on or replayed.
    captured = capsys.readouterr()
    assert not leaked(str(error), repr(error.problem), caplog.text, captured.out, captured.err)
    assert not leaked(repr(diagnostics))


async def test_cooldown_marked_retryable_is_still_never_slept_on(contract):
    problem = local_problem("plan_request_cooldown", "plan_request", status=429).model_dump(
        mode="json"
    )
    problem["retryable"] = True
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": str(COOLDOWN_SECONDS)}, json=problem)

    async with client_for(handle) as client:
        with pytest.raises(RateLimitedError) as caught:
            await client.account.request_special_free(use_case=SENTINEL)
    assert caught.value.retry_after_seconds == COOLDOWN_SECONDS
    assert len(requests) == 1  # The wait exceeds the operation deadline, so no retry is scheduled.


@pytest.mark.parametrize("use_case", ["", 7])
async def test_local_validation_refuses_without_echoing_input_or_calling(contract, use_case):
    core = FakeCore(contract)
    async with client_for(core) as client:
        with pytest.raises(ValueError) as caught:
            await client.account.request_special_free(use_case=use_case, links=[LINK])
        with pytest.raises(ValueError) as bad_links:
            await client.account.request_special_free(use_case=SENTINEL, links=[7])
    assert core.requests == []
    assert not leaked(str(caught.value), str(bad_links.value))
    assert bad_links.value.__cause__ is None and bad_links.value.__suppress_context__


def test_sync_client_delegates_plan_requests(contract):
    core = FakeCore(contract, refusal=("plan_request_cooldown", 429, COOLDOWN_SECONDS))
    with (
        SynthClient(
            "fixture",
            base_url="https://example.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(core)),
        ) as client,
        pytest.raises(RateLimitedError) as caught,
    ):
        client.account.request_special_free(use_case=SENTINEL)
    assert caught.value.retry_after_seconds == COOLDOWN_SECONDS and len(core.requests) == 1


async def test_mcp_dispatch_reaches_plan_requests_through_the_sdk(contract):
    core = FakeCore(contract)
    async with client_for(core) as client:
        body: dict[str, JsonValue] = {"use_case": SENTINEL, "links": [LINK]}
        created = await dispatch(client, "requestSpecialFree", {"body": body})
        listed = await dispatch(client, "listPlanRequests", {})
        withdrawn = await dispatch(client, "withdrawPlanRequest", {"request_id": REQUEST_ID})
    assert created is not None and listed is not None and withdrawn is not None
    assert [r.method for r in core.requests] == ["POST", "GET", "DELETE"]
    assert json.loads(core.requests[0].content) == body


def test_cli_request_list_and_withdraw(api, capsys):  # noqa: F811
    code, out, err = invoke(
        api, capsys, "plan", "request", "--use-case", SENTINEL, "--link", LINK, "--json"
    )
    assert code == cli.EXIT_OK, err
    assert json.loads(out)["state"] == "pending" and not leaked(out, err)
    oid, sent = api.calls[-1]
    assert oid == "requestSpecialFree"
    assert json.loads(sent.content) == {"use_case": SENTINEL, "links": [LINK]}
    code, out, err = invoke(api, capsys, "plan", "request", "--use-case", SENTINEL)
    assert code == cli.EXIT_OK and REQUEST_ID in out and not leaked(out, err)
    code, out, err = invoke(api, capsys, "plan", "requests")
    assert code == cli.EXIT_OK and "denied" in out and "not_a_fit" in out, err
    code, out, err = invoke(api, capsys, "plan", "withdraw", REQUEST_ID, "--json")
    assert code == cli.EXIT_OK and json.loads(out)["state"] == "withdrawn", err
    assert [oid for oid, _ in api.calls][-2:] == ["listPlanRequests", "withdrawPlanRequest"]


@pytest.mark.parametrize(("operation", "code", "status", "kind", "retry_after"), REFUSALS)
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_refusals_exit_four_and_surface_retry_after_without_the_use_case(
    api,  # noqa: F811
    capsys,
    operation,
    code,
    status,
    kind,
    retry_after,
    as_json,
):
    problem = local_problem(code, "plan_request", status=status).model_dump(mode="json")
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    api.overrides[operation] = lambda request: httpx.Response(status, headers=headers, json=problem)
    argv = (
        ["plan", "request", "--use-case", SENTINEL, "--link", LINK]
        if operation == "requestSpecialFree"
        else ["plan", "withdraw", REQUEST_ID]
    )
    exit_code, out, err = invoke(api, capsys, *argv, *(["--json"] if as_json else []))
    assert exit_code == cli.EXIT_SERVER_REFUSAL and out == ""
    assert code in err and not leaked(err)
    assert len(api.calls) == 1
    if as_json:
        document = json.loads(err)
        assert document["code"] == code and document["status"] == status
        assert document.get("retry_after_seconds") == retry_after
    elif retry_after is not None:
        assert f"retry after {retry_after} seconds" in err


def test_cli_usage_errors_do_not_echo_the_use_case(api, capsys):  # noqa: F811
    code, out, err = invoke(api, capsys, "plan", "request", "--use-case", SENTINEL, "--bogus")
    assert code == cli.EXIT_CLIENT_ERROR and api.calls == [] and not leaked(out, err)
    code, out, err = invoke(api, capsys, "plan", "request")
    assert code == cli.EXIT_CLIENT_ERROR and "--use-case" in err
