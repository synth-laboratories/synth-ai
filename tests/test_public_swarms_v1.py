"""Public Swarms v1 client, CLI and MCP catalog against an httpx mock of the contract."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest
from synth_ai import cli_public
from synth_ai.client import AsyncSynthClient, SynthClient
from synth_ai.products import PUBLIC_PRODUCTS
from synth_ai.public import errors
from synth_ai.public.models import Money

BASE = "https://api.test.usesynth.ai"
KEY = "sk_synth_user_" + "a" * 48

RUN = {
    "run_id": "run_1",
    "project_id": "proj_1",
    "state": "queued",
    "model": "gpt-6-luna",
    "provider": "synth",
    "result_available": False,
    "execution_closed": False,
    "accounting_final": False,
    "cleanup_complete": False,
}
ACCOUNT = {
    "org": {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Org",
        "personal": True,
        "plan": "free",
        "access_tier": "standard",
        "created_at": "2026-11-01T00:00:00Z",
    },
    "principal": {
        "org_id": "00000000-0000-0000-0000-000000000001",
        "credential_kind": "api_key",
        "credential_purpose": "personal",
        "principal_class": "user",
        "scopes": ["swarms:run"],
        "permissions": [],
        "authorization_version": 1,
        "access_tier": "standard",
    },
    "limits": [{"resource": "swarms", "metric": "runs", "window": "day", "cap": -1}],
}


class Contract:
    """In-memory server for the Swarms v1 paths."""

    def __init__(self, polls_until_done: int = 2, final_state: str = "completed") -> None:
        self.requests: list[httpx.Request] = []
        self.polls = polls_until_done
        self.final_state = final_state

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/api/v1/swarms/runs" and method == "POST":
            assert request.headers["Idempotency-Key"]
            assert request.headers["Authorization"] == f"Bearer {KEY}"
            return httpx.Response(201, json=RUN)
        if path == "/api/v1/swarms/runs/run_1" and method == "GET":
            self.polls -= 1
            if self.polls > 0:
                return httpx.Response(200, json={**RUN, "state": "running"})
            closed = {**RUN, "state": self.final_state, "execution_closed": True}
            if self.final_state == "failed":
                closed["failure"] = {"code": "task_failed", "retryable": False}
            return httpx.Response(200, json=closed)
        if path == "/api/v1/swarms/runs/run_1/outputs":
            return httpx.Response(
                200, json={"outputs": [{"output_id": "o1", "kind": "report", "size_bytes": 3}]}
            )
        if path == "/api/v1/account":
            return httpx.Response(200, json=ACCOUNT)
        if path == "/v1/billing/plans":
            assert "Authorization" not in request.headers
            return httpx.Response(200, json={"policy_version": "v023.20261014.weekly-signup",
                "plans": [{"plan_id": "standard.monthly", "name": "Standard", "kind": "recurring",
                           "price_cents": 2000, "interval": "month",
                           "weekly_included_microcents": 6_000_000_000}]})
        if path == "/v1/billing/balance":
            assert request.headers["Authorization"] == f"Bearer {KEY}"
            return httpx.Response(200, json={"org_id": "o", "ledger_microcents": 10,
                "held_microcents": 2, "headroom_microcents": 0, "available_microcents": 8,
                "available_usd_cents": 0})
        if path == "/v1/billing/checkout" and method == "POST":
            body = json.loads(request.content)
            assert body in ({"plan_id": "standard.monthly"}, {"credit_cents": 500})
            return httpx.Response(201, json={"session_id": "cs_1",
                                             "url": "https://checkout.stripe.com/c/cs_1"})
        return httpx.Response(404, json={"error": {"code": "not_found", "service": "synth-backend",
                                                    "stage": "route"}})


def make_client(handler) -> SynthClient:
    return SynthClient(
        api_key=KEY,
        base_url=BASE,
        project_id="proj_1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_run_one_call_happy_path() -> None:
    server = Contract()
    with make_client(server) as client:
        result = client.swarms.run("Compare retrieval methods", "2.50", poll_seconds=0)
    assert result.run.state == "completed"
    assert [o.output_id for o in result.outputs] == ["o1"]
    launch = json.loads(server.requests[0].content)
    assert launch["project_id"] == "proj_1"
    assert launch["message"] == "Compare retrieval methods"
    assert launch["limits"]["maximum_charge"] == {"currency": "USD", "coefficient": 25,
                                                  "exponent": -1}


def test_run_failed_raises_typed_error() -> None:
    with make_client(Contract(final_state="failed")) as client:
        with pytest.raises(errors.RunFailedError) as caught:
            client.swarms.run("x", 1, poll_seconds=0)
    assert caught.value.error.code == "task_failed"


def test_money_usd_exact() -> None:
    assert Money.usd("100").model_dump() == {"currency": "USD", "coefficient": 1, "exponent": 2}
    assert Money.usd(0).coefficient == 0
    with pytest.raises(ValueError):
        Money.usd("-1")


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, errors.InvalidRequestError),
        (401, errors.UnauthenticatedError),
        (402, errors.PaymentRequiredError),
        (403, errors.ForbiddenError),
        (404, errors.NotFoundError),
        (409, errors.ConflictError),
        (422, errors.InvalidRequestError),
        (429, errors.RateLimitedError),
        (500, errors.ServerError),
        (502, errors.UpstreamUnavailableError),
        (503, errors.UpstreamUnavailableError),
        (504, errors.UpstreamUnavailableError),
    ],
)
def test_error_code_mapping(status: int, kind: type[errors.SwarmsAPIError]) -> None:
    body = {"error": {"code": f"code_{status}", "message": "Refused.", "service": "synth-backend",
                      "stage": "admission", "upstream_status": 503, "cause": "typed_cause"}}
    with make_client(lambda _r: httpx.Response(status, json=body)) as client:
        with pytest.raises(kind) as caught:
            client.swarms.get("run_1")
    error = caught.value
    assert isinstance(error, errors.SwarmsAPIError)
    assert error.http_status == status
    assert (error.error.code, error.error.service, error.error.stage) == (
        f"code_{status}", "synth-backend", "admission")
    assert (error.error.upstream_status, error.error.cause) == (503, "typed_cause")
    assert error.message == "Refused."


def test_malformed_error_body() -> None:
    with make_client(lambda _r: httpx.Response(500, text="<html>")) as client:
        with pytest.raises(errors.MalformedErrorResponseError) as caught:
            client.account.get()
    assert caught.value.error.upstream_status == 500


def test_transport_failure() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with make_client(boom) as client:
        with pytest.raises(errors.TransportFailureError) as caught:
            client.swarms.list()
    assert caught.value.error.code == "transport_failed"


def test_account_and_billing() -> None:
    with make_client(Contract()) as client:
        assert client.account.get().org.plan == "free"
        assert client.billing.plans().plans[0].plan_id == "standard.monthly"
        assert client.billing.balance().available_microcents == 8
        assert client.billing.checkout(plan_id="standard.monthly").session_id == "cs_1"
        assert client.billing.checkout(credit_cents=500).url.startswith("https://")
        with pytest.raises(ValueError):
            client.billing.checkout(plan_id="standard.monthly", credit_cents=500)
        with pytest.raises(ValueError):
            client.billing.checkout(credit_cents=499)


def test_public_client_surface_only() -> None:
    with make_client(Contract()) as client:
        public = {name for name in vars(client) if not name.startswith("_")}
        assert public == {"swarms", "account", "billing", "repos", "links", "artifacts"}
        assert not hasattr(SynthClient, "anonymous")
        assert type(client.hidden("index")).__name__ == "IndexClient"


def test_base_url_config() -> None:
    script = "from synth_ai.client import resolve_url; print(resolve_url(None))"
    clean = {k: v for k, v in os.environ.items()
             if k not in {"SYNTH_API_BASE_URL", "SYNTH_API_URL", "SYNTH_BACKEND_URL"}}
    default = subprocess.run([sys.executable, "-c", script], env=clean, check=True,
                             capture_output=True, text=True).stdout.strip()
    assert default == "https://api.usesynth.ai"
    override = subprocess.run(
        [sys.executable, "-c", script],
        env={**clean, "SYNTH_API_BASE_URL": "https://core-api-staging.example.com"},
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert override == "https://core-api-staging.example.com"


def test_cli_registers_only_public_products() -> None:
    assert PUBLIC_PRODUCTS == ("swarms", "account", "billing")
    assert cli_public.registered_commands() == {
        "swarms", "account", "billing", "login", "logout", "auth"}


def test_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    server = Contract(polls_until_done=1)

    def factory() -> SynthClient:
        return make_client(server)

    assert cli_public.main(["swarms", "run", "hello", "--budget", "1"],
                           client_factory=factory) == 0
    assert '"state": "completed"' in capsys.readouterr().out
    assert cli_public.main(["account"], client_factory=factory) == 0
    assert '"plan": "free"' in capsys.readouterr().out
    assert cli_public.main(["billing", "plans"], client_factory=factory) == 0
    assert cli_public.main(["swarms", "get", "missing"], client_factory=factory) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "not_found"


def test_mcp_catalog_public_only() -> None:
    from synth_ai.mcp.catalog import ALL_BY_NAME, ALL_TOOLS, BY_NAME, TOOLS
    from synth_ai.products import tool_product

    assert [tool.name for tool in TOOLS] == [
        "swarms_run", "swarms_launch", "swarms_get", "swarms_wait", "swarms_events",
        "swarms_cancel", "swarms_outputs", "swarms_list", "account_get", "billing_balance",
        "billing_plans",
    ]
    assert {tool_product(tool.name) for tool in TOOLS} <= set(PUBLIC_PRODUCTS)
    assert len(ALL_TOOLS) == 95
    assert "index_search" not in BY_NAME and "forge_project_get" not in BY_NAME
    assert "swarms_create" not in BY_NAME and "swarms_create" in ALL_BY_NAME
    assert BY_NAME["swarms_run"].paid and BY_NAME["swarms_run"].mutates
    assert not BY_NAME["billing_plans"].keyed


def test_hosted_scopes_are_the_contract_scopes() -> None:
    from synth_ai.mcp.hosted import AUTHORIZATION_SCOPES

    assert AUTHORIZATION_SCOPES == (
        "account:billing:read", "account:read", "account:usage:read", "swarms:control",
        "swarms:read", "swarms:run", "swarms:write",
    )


def make_async_client(handler) -> AsyncSynthClient:
    return AsyncSynthClient(
        api_key=KEY,
        base_url=BASE,
        project_id="proj_1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_async_run_and_namespaces() -> None:
    async def scenario() -> None:
        async with make_async_client(Contract()) as client:
            result = await client.swarms.run("Compare", "2.50", poll_seconds=0)
            assert result.run.state == "completed"
            assert [o.output_id for o in result.outputs] == ["o1"]
            assert (await client.account.get()).org.plan == "free"
            assert (await client.billing.plans()).plans[0].plan_id == "standard.monthly"
            assert (await client.billing.balance()).available_microcents == 8
            with pytest.raises(errors.NotFoundError):
                await client.swarms.get("missing")

    asyncio.run(scenario())


def test_mcp_swarms_run_through_public_client() -> None:
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
    from synth_ai.mcp.catalog import TOOLS
    from synth_ai.mcp.runtime import SDKExecutor
    from synth_ai.mcp.server import create_server

    server_state = Contract(polls_until_done=1)

    async def scenario() -> None:
        async with make_async_client(server_state) as client:
            executor = SDKExecutor(client, client, frozenset(tool.name for tool in TOOLS))
            server = create_server(executor)
            listed = (await server.request_handlers[ListToolsRequest](
                ListToolsRequest(method="tools/list"))).root
            assert {tool.name for tool in listed.tools} == {tool.name for tool in TOOLS}
            request = CallToolRequest(method="tools/call", params=CallToolRequestParams(
                name="swarms_run", arguments={"message": "Compare", "budget_usd": "2.50"}))
            result = (await server.request_handlers[CallToolRequest](request)).root
            assert not result.isError
            assert result.structuredContent["run"]["state"] == "completed"
            missing = CallToolRequest(method="tools/call", params=CallToolRequestParams(
                name="swarms_get", arguments={"run_id": "missing"}))
            refused = (await server.request_handlers[CallToolRequest](missing)).root
            assert refused.isError

    asyncio.run(scenario())
    # No capability negotiation: every request is a contract path.
    assert all(r.url.path.startswith(("/api/v1/", "/v1/")) for r in server_state.requests)
    assert not any("capabilities" in r.url.path for r in server_state.requests)
