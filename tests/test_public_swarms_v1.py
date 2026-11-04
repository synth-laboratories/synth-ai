"""Public Swarms v1 client, CLI and MCP catalog against an httpx mock of the contract."""

from __future__ import annotations

import argparse
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
    "metadata": {},
    "labels": None,
    "labels_revision": 1,
    "objectives": None,
    "guidance_coverage": None,
    "problems": None,
    "problems_observed": False,
}
PAGE = {"metadata": {}}


def event(sequence: int) -> dict:
    return {"event_id": f"e{sequence}", "run_id": "run_1", "sequence": sequence,
            "kind": "task.started", "original_kind": "task.started", "usage": None,
            "observed_at": "2026-11-04T09:00:00Z", "approval_id": None}


OUTPUT = {"run_id": "run_1", "output_id": "o1", "state": "available",
          "reference": {"content_ref": "ref"}, "filename": "report.md"}
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
            return httpx.Response(200, json=RUN)
        if path == "/api/v1/swarms/runs/run_1" and method == "GET":
            self.polls -= 1
            if self.polls > 0:
                return httpx.Response(200, json={**RUN, "state": "running"})
            closed = {**RUN, "state": self.final_state, "execution_closed": True}
            if self.final_state == "failed":
                closed["failure"] = {"code": "task_failed", "retryable": False}
                closed["control"] = {"operation_id": "op", "command": "cancel", "state": "applied"}
            return httpx.Response(200, json=closed)
        if path == "/api/v1/swarms/runs/run_1/outputs":
            return httpx.Response(
                200, json={**PAGE, "next_cursor": None, "complete": True, "items": [OUTPUT]}
            )
        if path == "/api/v1/swarms/runs/run_1/outputs/o1":
            return httpx.Response(200, json=OUTPUT)
        if path == "/api/v1/swarms/runs" and method == "GET":
            cursor = request.url.params.get("cursor")
            assert request.url.params.get("project_id") in (None, "proj_1")
            if cursor is None:
                return httpx.Response(200, json={**PAGE, "next_cursor": "c1", "complete": False,
                                                 "items": [RUN]})
            return httpx.Response(200, json={**PAGE, "next_cursor": None, "complete": True,
                                             "items": []})
        if path == "/api/v1/swarms/runs/run_1/events":
            cursor = request.url.params.get("cursor")
            if cursor is None:
                return httpx.Response(200, json={**PAGE, "next_cursor": "s1", "complete": False,
                                                 "resume_cursor": "s1", "items": [event(0)]})
            if cursor == "s1":
                return httpx.Response(200, json={**PAGE, "next_cursor": None, "complete": True,
                                                 "resume_cursor": "s2", "items": [event(1)]})
            return httpx.Response(200, json={**PAGE, "next_cursor": None, "complete": True,
                                             "resume_cursor": cursor, "items": []})
        if path == "/api/v1/swarms/runs/legacy_1/events":
            return httpx.Response(409, json={"error": {"code": "run_not_orchestra_authority",
                "service": "synth-backend", "stage": "swarms"}, "outcome": "not_accepted"})
        if path == "/api/v1/swarms/runs/run_1/cancel" and method == "POST":
            assert request.headers["Idempotency-Key"] and not request.content
            return httpx.Response(200, json={"operation_id": "op", "resource_id": "run_1",
                "state": "accepted", "command": "cancel", "reason": "requested",
                "execution_closed": False, "accounting_final": False, "cleanup_complete": False})
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
        if path == "/v1/billing/history":
            if request.url.params.get("cursor") == "bad":
                return httpx.Response(422, json={"error": {"code": "invalid_cursor",
                    "service": "synth-backend", "stage": "billing_history"}})
            return httpx.Response(200, json={"next_cursor": None, "items": [{
                "entry_id": "le_1", "cause": "grant", "delta_microcents": 1_000_000_000,
                "held_delta_microcents": 0, "funding_class": "included", "hold_kind": None,
                "expires_at": None, "created_at": "2026-11-04T09:00:00Z"}]})
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
    with make_client(Contract(final_state="failed")) as client, pytest.raises(errors.RunFailedError) as caught:
        client.swarms.run("x", 1, poll_seconds=0)
    assert caught.value.error.code == "task_failed"


def test_runs_pages_events_outputs_and_cancel() -> None:
    with make_client(Contract()) as client:
        first = client.swarms.list(limit=1, project_id="proj_1")
        assert ([r.run_id for r in first.items], first.next_cursor, first.complete) == (
            ["run_1"], "c1", False)
        last = client.swarms.list(cursor=first.next_cursor)
        assert (last.items, last.next_cursor, last.complete) == ([], None, True)
        page = client.swarms.events("run_1")
        assert ([e.sequence for e in page.items], page.resume_cursor) == ([0], "s1")
        assert [e.event_id for e in client.swarms.iter_events("run_1")] == ["e0", "e1"]
        later = client.swarms.events("run_1", cursor="s2")
        assert (later.items, later.resume_cursor, later.complete) == ([], "s2", True)
        outputs = client.swarms.outputs("run_1")
        assert outputs.complete and outputs.items[0].filename == "report.md"
        record = client.swarms.output("run_1", "o1")
        assert (record.state, record.reference) == ("available", {"content_ref": "ref"})
        receipt = client.swarms.cancel("run_1")
        assert (receipt.resource_id, receipt.state, receipt.command) == (
            "run_1", "accepted", "cancel")


def test_legacy_run_refuses_events_typed() -> None:
    with make_client(Contract()) as client, pytest.raises(errors.RunNotOrchestraAuthorityError) as caught:
        client.swarms.events("legacy_1")
    assert isinstance(caught.value, errors.ConflictError)
    assert (caught.value.http_status, caught.value.outcome) == (409, "not_accepted")


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
    with make_client(lambda _r: httpx.Response(status, json=body)) as client, pytest.raises(kind) as caught:
        client.swarms.get("run_1")
    error = caught.value
    assert isinstance(error, errors.SwarmsAPIError)
    assert error.http_status == status
    assert (error.error.code, error.error.service, error.error.stage) == (
        f"code_{status}", "synth-backend", "admission")
    assert (error.error.upstream_status, error.error.cause) == (503, "typed_cause")
    assert error.message == "Refused."


def test_error_body_with_outcome_is_typed() -> None:
    body = {"error": {"code": "invalid_launch_request", "service": "synth-backend",
                      "stage": "swarms"}, "outcome": "not_accepted"}
    with make_client(lambda _r: httpx.Response(422, json=body)) as client, pytest.raises(errors.InvalidRequestError) as caught:
        client.swarms.launch("x", 1)
    assert caught.value.error.code == "invalid_launch_request"
    assert caught.value.outcome == "not_accepted"


def test_malformed_error_body() -> None:
    with make_client(lambda _r: httpx.Response(500, text="<html>")) as client, pytest.raises(errors.MalformedErrorResponseError) as caught:
        client.account.get()
    assert caught.value.error.upstream_status == 500


def test_transport_failure() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with make_client(boom) as client, pytest.raises(errors.TransportFailureError) as caught:
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


def test_cli_verbs_and_mcp_tools_are_one_to_one() -> None:
    from synth_ai.mcp.catalog import TOOLS
    from synth_ai.surface import VERBS

    assert cli_public.registered_verbs() == {verb.cli for verb in VERBS}
    assert [verb.tool for verb in VERBS] == [tool.name for tool in TOOLS]
    hosted_on_staging = {  # tools/list on staging /mcp, 2026-11-04
        "account_billing_balance", "account_get", "swarms_run_cancel", "swarms_run_events",
        "swarms_run_get", "swarms_run_launch", "swarms_run_output_get", "swarms_run_outputs",
        "swarms_runs_list"}
    assert {verb.tool for verb in VERBS if verb.hosted} == hosted_on_staging


def test_catalog_doc_is_generated() -> None:
    from pathlib import Path

    from synth_ai.mcp.catalog_doc import render

    path = Path(__file__).resolve().parents[1] / "docs/mcp-tools.md"
    assert path.read_text() == render()


def test_billing_history_and_single_attempt_on_4xx(capsys: pytest.CaptureFixture[str]) -> None:
    server = Contract()

    def factory() -> SynthClient:
        return make_client(server)

    assert cli_public.main(["billing", "history", "--limit", "5"], client_factory=factory) == 0
    page = json.loads(capsys.readouterr().out)
    assert page["items"][0]["cause"] == "grant"
    before = len(server.requests)
    assert cli_public.main(["billing", "history", "--cursor", "bad"],
                           client_factory=factory) == 1
    assert len(server.requests) == before + 1  # no retry on 4xx
    refusal = json.loads(capsys.readouterr().err)
    assert (refusal["error"]["code"], refusal["status"], refusal["outcome"]) == (
        "invalid_cursor", 422, None)


def test_cli_error_display_carries_outcome(capsys: pytest.CaptureFixture[str]) -> None:
    body = {"error": {"code": "zero_charge_launch_unavailable", "service": "synth-backend",
                      "stage": "swarms"}, "outcome": "not_accepted"}

    def factory() -> SynthClient:
        return make_client(lambda _r: httpx.Response(409, json=body))

    assert cli_public.main(["swarms", "launch", "x", "--budget", "0"],
                           client_factory=factory) == 1
    refusal = json.loads(capsys.readouterr().err)
    assert (refusal["error"]["code"], refusal["outcome"], refusal["status"]) == (
        "zero_charge_launch_unavailable", "not_accepted", 409)


def test_cli_run_streams_events_with_resume_and_saves_outputs(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = Contract(polls_until_done=1)
    task = tmp_path / "bundle"
    task.mkdir()
    (task / "task.md").write_text("Compare retrieval methods")
    out = tmp_path / "out"
    lines: list[str] = []

    def factory() -> SynthClient:
        return make_client(server)

    code = cli_public.main(
        ["swarms", "run", "--program", "swarms.general@1", "--task", str(task),
         "--budget", "2.50", "--out", str(out)],
        client_factory=factory, emit=lines.append,
    )
    assert code == 0, capsys.readouterr().err
    launch = json.loads(server.requests[0].content)
    assert launch["program"] == {"family": {"id": "swarms.general", "version": 1}}
    assert launch["message"] == "Compare retrieval methods"
    records = [json.loads(line) for line in lines]
    assert records[0] == {"run_id": "run_1", "state": "queued"}
    assert [r["event_id"] for r in records if "event_id" in r] == ["e0", "e1"]
    assert [r["resume_cursor"] for r in records if "resume_cursor" in r] == ["s1", "s2", "s2"]
    cursors = [r.url.params.get("cursor") for r in server.requests if r.url.path.endswith("/events")]
    assert cursors == [None, "s1", "s2"]
    assert json.loads((out / "o1.json").read_text())["filename"] == "report.md"
    assert records[-1]["state"] == "completed"


def test_cli_run_attach_resumes_from_cursor() -> None:
    server = Contract(polls_until_done=1)
    lines: list[str] = []
    assert cli_public.main(["swarms", "run", "--attach", "run_1", "--cursor", "s2"],
                           client_factory=lambda: make_client(server), emit=lines.append) == 0
    assert not any(r.method == "POST" for r in server.requests)
    cursors = [r.url.params.get("cursor") for r in server.requests if r.url.path.endswith("/events")]
    assert cursors == ["s2", "s2"]


def test_task_bundle_and_program_parsing(tmp_path) -> None:
    (tmp_path / "task.md").write_text("Do it")
    assert cli_public.read_task(str(tmp_path)) == "Do it"
    (tmp_path / "data.csv").write_text("a,b")
    with pytest.raises(errors.SwarmsAPIError) as caught:
        cli_public.read_task(str(tmp_path))
    assert caught.value.error.code == "task_inputs_unsupported"
    with pytest.raises(errors.SwarmsAPIError) as missing:
        cli_public.read_task(str(tmp_path / "data.csv.missing"))
    assert missing.value.error.code == "task_unreadable"
    with pytest.raises(argparse.ArgumentTypeError):
        cli_public.parse_program("swarms.general")


def test_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    server = Contract(polls_until_done=1)

    def factory() -> SynthClient:
        return make_client(server)

    assert cli_public.main(["swarms", "run", "hello", "--budget", "1"],
                           client_factory=factory) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["state"] == "completed"
    assert cli_public.main(["account"], client_factory=factory) == 0
    assert json.loads(capsys.readouterr().out)["org"]["plan"] == "free"
    assert cli_public.main(["billing", "plans"], client_factory=factory) == 0
    assert cli_public.main(["swarms", "status", "missing"], client_factory=factory) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "not_found"


def test_mcp_catalog_public_only() -> None:
    from synth_ai.mcp.catalog import ALL_BY_NAME, ALL_TOOLS, BY_NAME, TOOLS
    from synth_ai.products import tool_product

    assert [tool.name for tool in TOOLS] == [
        "swarms_run", "swarms_run_launch", "swarms_run_get", "swarms_run_wait",
        "swarms_run_events", "swarms_run_cancel", "swarms_run_outputs", "swarms_run_output_get",
        "swarms_runs_list", "account_get", "account_billing_balance", "account_billing_history",
        "account_billing_plans",
    ]
    assert {tool_product(tool.name) for tool in TOOLS} <= set(PUBLIC_PRODUCTS)
    assert len(ALL_TOOLS) == 95
    assert "index_search" not in BY_NAME and "forge_project_get" not in BY_NAME
    assert "swarms_create" not in BY_NAME and "swarms_create" in ALL_BY_NAME
    assert BY_NAME["swarms_run"].paid and BY_NAME["swarms_run"].mutates
    assert not BY_NAME["account_billing_plans"].keyed


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
                name="swarms_run_get", arguments={"run_id": "missing"}))
            refused = (await server.request_handlers[CallToolRequest](missing)).root
            assert refused.isError

    asyncio.run(scenario())
    # No capability negotiation: every request is a contract path.
    assert all(r.url.path.startswith(("/api/v1/", "/v1/")) for r in server_state.requests)
    assert not any("capabilities" in r.url.path for r in server_state.requests)
