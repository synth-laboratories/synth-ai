"""SYN-5695 J2/J3/J-refusal steps on controlled loopback HTTP.

Real SDK, CLI subprocess and MCP stdio clients answer one scripted server.
These fixtures prove the step logic and client typing; they are not cached-slot
or live acceptance receipts.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acceptance import journey_main, journeys
from scripts.acceptance.journeys import (
    JourneyContext,
    Secret,
    Step,
    Verdict,
    capabilities,
    matrix_operations,
    run_step,
)
from scripts.acceptance.surfaces import CLI_ARGV, SDK_CALLS, Call, SurfaceName
from synth_ai.errors import local_problem

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).parents[2]
SURFACES = [SurfaceName.SDK, SurfaceName.CLI, SurfaceName.MCP_STDIO]
Responder = Callable[[str, str, dict[str, str]], tuple[int, object, dict[str, str]]]


def problem(code, status, **extra):
    document = local_problem(code, "fixture", status=status).model_dump(mode="json")
    document.update(outcome="not_accepted", **extra)
    return document


def example(contract, path, method="get", status="200"):
    response = contract["paths"][path][method]["responses"][status]
    return copy.deepcopy(response["content"]["application/json"]["example"])


@contextmanager
def api(responder: Responder):
    requests: list[tuple[str, str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length) if length else b""
            bearer = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
            requests.append((self.command, self.path, bearer))
            request = {"bearer": bearer, "body": json.loads(raw_body) if raw_body else None}
            status, payload, headers = responder(self.command, self.path, request)
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = do_DELETE = do_PATCH = do_PUT = answer

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def context_for(surface, origin, tmp_path, **options):
    make = journey_main.surface_factory(
        surface, origin=origin, work_directory=tmp_path, mcp_url=None
    )
    secrets = {"SYNTH_API_KEY": "fixture-admin", **options.pop("secrets", {})}
    with make("fixture-admin") as opened:
        yield JourneyContext(
            surface=opened,
            driver=surface,
            origin=origin,
            make=make,
            receipt_dir=tmp_path,
            secrets=secrets,
            **options,
        )


def run(surface, responder, tmp_path, step, **options):
    with api(responder) as (origin, requests):
        with context_for(surface, origin, tmp_path, **options) as context:
            return run_step(context, step), requests


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
def test_member_is_refused_billing_admin_change(contract, tmp_path, surface):
    account = example(contract, "/v1/account")

    def responder(method, path, headers):
        if path == "/v1/account":
            body = copy.deepcopy(account)
            role = "member" if headers["bearer"] == "fixture-member" else "admin"
            body["principal"]["org_role"] = role
            return 200, body, {}
        assert headers["bearer"] == "fixture-member"
        return 403, problem("billing_admin_required", 403), {}

    result, requests = run(
        surface,
        responder,
        tmp_path,
        Step.MEMBER_ADMIN,
        secrets={Secret.MEMBER.value: "fixture-member"},
    )
    assert result.verdict == Verdict.PASS, result
    assert ("POST", "/v1/billing/subscription/cancel", "fixture-member") in requests


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


def matrix_ids():
    return {
        "run_id": "run_" + "F" * 22,
        "output_id": "out_" + "F" * 22,
        "file_id": "file_" + "F" * 22,
        "upload_id": "upl_" + "F" * 22,
        "export_id": "exp_" + "F" * 22,
        "id": "chk_" + "F" * 22,
        "request_id": "11111111-1111-4111-8111-111111111111",
        "revision_id": "rev_" + "F" * 22,
        "key_id": "key_" + "F" * 22,
        "grant_id": "22222222-2222-4222-8222-222222222222",
    }


def test_cross_org_matrix_identical_404s_are_not_a_full_pass(tmp_path):
    def responder(method, path, headers):
        assert headers["bearer"] == "fixture-org-b"
        return 404, problem("resource_not_found", 404), {}

    result, requests = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    rows = [row for row in result.evidence if row.get("result") in {"pass", "leak", "differs"}]
    assert rows and all(row["result"] == "pass" for row in rows)
    # Operations the SDK has no public method for stay uncovered: Inconclusive.
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause.startswith("uncovered=")
    assert requests


def test_cross_org_leak_fails(contract, tmp_path):
    snapshot = example(contract, "/v1/runs/{run_id}")
    foreign = matrix_ids()["run_id"]

    def responder(method, path, _):
        if method == "GET" and path == f"/v1/runs/{foreign}":
            return 200, snapshot, {}
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    assert result.verdict == Verdict.FAIL
    assert any(
        row.get("operation") == "getRun" and row["result"] == "leak" for row in result.evidence
    )


def test_entrypoint_writes_exclusive_step_receipts(tmp_path, monkeypatch, capsys):
    def responder(method, path, _):
        return 403, problem("optimizer_topology_not_allowed", 403), {}

    with api(responder) as (origin, _):
        monkeypatch.setenv("SYNTH_LOCAL_API_BASE_URL", origin)
        monkeypatch.setenv("SYNTH_API_KEY", "fixture-admin")
        for name in Secret:
            monkeypatch.delenv(name.value, raising=False)
        code = journey_main.main(
            [
                "--driver",
                "sdk",
                "--steps",
                f"{Step.SKU_REFUSAL},{Step.MINT_KEY}",
                "--receipt-dir",
                str(tmp_path / "receipts"),
            ]
        )
    assert code == 2  # Pass + Inconclusive (no session) is not a Pass.
    document = json.loads((tmp_path / "receipts/steps.json").read_text())
    assert [step["verdict"] for step in document["steps"]] == ["Pass", "Inconclusive"]
    assert "fixture-admin" not in (tmp_path / "receipts/steps.json").read_text()
    assert "fixture-admin" not in capsys.readouterr().out
    assert os.stat(tmp_path / "receipts").st_mode & 0o777 == 0o700
    assert sys.executable


MONEY = {"amount": "0.00", "currency": "USD"}


def limits_view(contract, *, cap, revision, status="running"):
    view = example(contract, "/v1/runs/{run_id}/limits")
    view.update(revision=revision, status=status)
    view["primary"] = [{"dimension": "spend_micros", "effective_cap": cap, "state": "capped"}]
    return view


def limits_api(contract, status="running"):
    state = {"cap": 1_000_000, "revision": 0}

    def responder(method, path, request):
        if method == "POST" and path.endswith("/limits"):
            amount = request["body"]["limits"]["maximum_charge"]["amount"]
            state["cap"] = int(Decimal(amount) * 1_000_000)
            state["revision"] += 1
        return (
            200,
            limits_view(contract, cap=state["cap"], revision=state["revision"], status=status),
            {},
        )

    return responder


@pytest.mark.parametrize("surface", [SurfaceName.CLI, SurfaceName.MCP_STDIO])
def test_raise_limit_is_seen_in_the_next_budget_view(contract, tmp_path, surface):
    result, requests = run(
        surface, limits_api(contract), tmp_path, Step.RAISE_LIMIT, run_id="run_" + "R" * 22
    )
    assert result.verdict == Verdict.PASS, result
    assert [method for method, _, _ in requests] == ["GET", "POST", "GET"]


@pytest.mark.parametrize("surface", SURFACES)
def test_raise_limit_on_an_ended_run_is_inconclusive_not_failed(contract, tmp_path, surface):
    result, requests = run(
        surface,
        limits_api(contract, status="completed"),
        tmp_path,
        Step.RAISE_LIMIT,
        run_id="run_" + "R" * 22,
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "precondition_not_met:run_not_running"
    assert all(method == "GET" for method, _, _ in requests)


@pytest.mark.parametrize("surface", SURFACES)
def test_current_month_statement_passes(contract, tmp_path, surface):
    month = datetime.now(UTC).strftime("%Y-%m")
    item = {
        name: MONEY
        for name in (
            "opening_ledger purchases allowance credits conversion charges refunds "
            "adjustments expirations closing_ledger unspent_returned usage_charge"
        ).split()
    }
    item.update(
        month=month,
        period_start=month + "-01T00:00:00Z",
        period_end=month + "-28T00:00:00Z",
        entries=1,
        usage_input_tokens=0,
        usage_output_tokens=0,
    )

    def responder(method, path, _):
        if path.startswith("/v1/billing/statements"):
            return 200, {"has_more": False, "items": [item]}, {}
        return 200, example(contract, "/v1/billing/balance"), {}

    result, _ = run(surface, responder, tmp_path, Step.STATEMENT)
    assert result.verdict == Verdict.PASS, result


@pytest.mark.parametrize("surface", SURFACES)
def test_pause_on_limit_then_resume(contract, tmp_path, surface):
    snapshot = example(contract, "/v1/runs/{run_id}")
    limits = limits_api(contract)
    resumed = {"done": False}
    pause = {
        "cause": "limit",
        "operation_id": "op_fixture",
        "limit": None,
        "requested_at_ms": 1,
        "drained_at_ms": None,
        "mode": "cold",
    }

    def responder(method, path, request):
        if path.endswith("/limits"):
            if method == "POST":
                assert request["body"]["resume"] is True
                resumed["done"] = True
            return limits(method, path, request)
        body = copy.deepcopy(snapshot)
        if resumed["done"]:
            body.update(state="running", pause=None)
        else:
            body.update(state="paused", pause=pause)
        return 200, body, {}

    run_id = "run_" + "L" * 22
    paused, _ = run(surface, responder, tmp_path / "pause", Step.PAUSE_LIMIT, limit_run_id=run_id)
    assert paused.verdict == Verdict.PASS, paused
    result, _ = run(surface, responder, tmp_path / "resume", Step.RESUME, limit_run_id=run_id)
    assert result.verdict == Verdict.PASS, result


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("denial_status", "denial_code", "verdict"),
    [
        (401, "api_key_invalid", Verdict.PASS),
        (403, "api_key_invalid", Verdict.FAIL),
        (401, "authentication_required", Verdict.FAIL),
    ],
)
def test_key_mint_rotate_revoke_with_a_session(
    contract, tmp_path, surface, denial_status, denial_code, verdict
):
    minted = example(contract, "/v1/api-keys", method="post", status="201")
    account = example(contract, "/v1/account")
    live: set[str] = set()
    counter = {"n": 0}

    def issue():
        counter["n"] += 1
        body = copy.deepcopy(minted)
        body["key_id"] = "key_" + str(counter["n"]) * 22
        body["key"] = "sk_test_" + str(counter["n"]) * 48
        live.add(body["key"])
        return body

    def responder(method, path, request):
        bearer = request["bearer"]
        if path == "/v1/account":
            if bearer in live:
                return 200, account, {}
            return denial_status, problem(denial_code, denial_status), {}
        assert bearer == "fixture-session"
        if method == "POST" and path == "/v1/api-keys":
            return 201, issue(), {}
        if path.endswith("/rotate"):
            live.discard("sk_test_" + path.split("/")[3][-1] * 48)
            return 200, issue(), {}
        live.discard("sk_test_" + path.split("/")[3][-1] * 48)
        return 204, None, {}

    session = {Secret.SESSION.value: "fixture-session"}
    with api(responder) as (origin, _):
        with context_for(surface, origin, tmp_path, secrets=session) as context:
            verdicts = [
                run_step(context, step).verdict
                for step in (Step.MINT_KEY, Step.ROTATE_KEY, Step.REVOKE_KEY)
            ]
    assert verdicts == [Verdict.PASS, verdict, verdict]
    assert live == set()


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["running", "completed"], (Verdict.PASS, "")),
        (["completed"], (Verdict.INCONCLUSIVE, "precondition_not_met:run_already_terminal")),
    ],
)
def test_queued_launch_follows_the_harness_launch(contract, tmp_path, surface, states, expected):
    run_id = "run_" + "Q" * 22
    snapshot = example(contract, "/v1/runs/{run_id}")
    snapshot.update(run_id=run_id, state="queued")
    snapshot["queue"] = {"position": 2, "ahead": 1, "estimated_start_at": None, "expires_at": None}
    folder = tmp_path / "public-client-sdk"
    folder.mkdir()
    (folder / "launch-row.json").write_text(json.dumps({"run_id": run_id, "body": snapshot}))
    remaining = list(states)

    def responder(method, path, _):
        body = copy.deepcopy(snapshot)
        body.update(state=remaining.pop(0) if len(remaining) > 1 else remaining[0], queue=None)
        return 200, body, {}

    result, _ = run(surface, responder, tmp_path / "run", Step.QUEUED_LAUNCH, queued_journal=folder)
    assert (result.verdict, result.cause) == expected, result


@pytest.mark.parametrize("surface", SURFACES)
def test_cross_org_matrix_classifies_every_surface(tmp_path, surface):
    ids = matrix_ids()

    def responder(method, path, _):
        if path.startswith("/v1/api-keys"):
            return 403, problem("insufficient_scope", 403), {}  # Session-only routes.
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        surface,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=ids,
    )
    results = {row.get("operation"): row.get("result") for row in result.evidence}
    assert "leak" not in results.values() and "differs" not in results.values()
    assert results["getRun"] == "pass"
    assert result.verdict == Verdict.INCONCLUSIVE
    if surface != SurfaceName.CLI:
        assert results["revokeApiKey"] == "unproven:uniform_403"


def test_cross_org_existence_signal_fails(tmp_path):
    foreign = matrix_ids()["file_id"]

    def responder(method, path, _):
        detail = "forbidden" if path.endswith(foreign) else "absent"
        return 404, {**problem("resource_not_found", 404), "detail": detail}, {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    assert result.verdict == Verdict.FAIL
    assert any(
        row.get("operation") == "getFile" and row["result"] == "differs" for row in result.evidence
    )


def test_a_step_exception_keeps_earlier_receipts(tmp_path, monkeypatch):
    def responder(method, path, _):
        return 403, problem("optimizer_topology_not_allowed", 403), {}

    def explode(context):
        raise TimeoutError("fixture")

    from scripts.acceptance import journeys

    _function, drivers = journeys.STEPS[Step.STATEMENT]
    monkeypatch.setitem(journeys.STEPS, Step.STATEMENT, (explode, drivers))
    with api(responder) as (origin, _):
        monkeypatch.setenv("SYNTH_LOCAL_API_BASE_URL", origin)
        monkeypatch.setenv("SYNTH_API_KEY", "fixture-admin")
        code = journey_main.main(
            [
                "--driver",
                "sdk",
                "--steps",
                f"{Step.SKU_REFUSAL},{Step.STATEMENT}",
                "--receipt-dir",
                str(tmp_path / "r"),
            ]
        )
    steps = json.loads((tmp_path / "r/steps.json").read_text())["steps"]
    assert code == 2
    assert [(s["verdict"], s["cause"]) for s in steps] == [
        ("Pass", ""),
        ("Inconclusive", "step_outcome_unknown:TimeoutError"),
    ]


def launch_window(cap, retry_after, *, then):
    """Core's org launch window: `cap` counted launches, then 429 until it reopens."""
    import time

    state = {"count": 0, "reopens": None, "posts": []}

    def responder(method, path, request):
        if method != "POST" or path != "/v1/runs":
            raise AssertionError(f"unexpected {method} {path}")
        now = time.monotonic()
        state["posts"].append((now, request["body"]))
        if state["reopens"] is not None and now >= state["reopens"]:
            return then
        state["count"] += 1
        if state["count"] <= cap:
            return 422, problem("invalid_launch_request", 422), {}
        if state["reopens"] is None:
            state["reopens"] = now + retry_after
        return (
            429,
            problem("rate_limit_exceeded", 429, retryable=True),
            {"Retry-After": str(retry_after)},
        )

    return responder, state


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


def test_unshipped_message_routes_are_typed_not_in_release_for_shipped_contract(tmp_path):
    def responder(method, path, _):
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    results = {row.get("operation"): row.get("result") for row in result.evidence}
    assert journeys.contract_version() == "1.35.1"
    for operation in ("sendRunMessage", "listRunMessages", "getRunMessage"):
        assert results[operation] == "not_in_release:1.35.1"
    # A not_in_release cell is never a pass and never counted as uncovered.
    uncovered = sum(
        1 for value in results.values() if value.startswith(("uncovered:", "unproven:"))
    )
    assert result.cause == f"uncovered={uncovered}"


def test_unlisted_contract_version_keeps_message_routes_uncovered(tmp_path, monkeypatch):
    monkeypatch.setattr(journeys, "contract_version", lambda: "1.36.0")

    def responder(method, path, _):
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    results = {row.get("operation"): row.get("result") for row in result.evidence}
    assert results["sendRunMessage"] == "uncovered:contract_operation_missing"


def test_session_only_key_routes_are_probed_with_the_foreign_session(tmp_path):
    def responder(method, path, headers):
        if path.startswith("/v1/api-keys"):
            if headers["bearer"] != "fixture-org-b-session":
                return 403, problem("session_required", 403), {}
            return 404, problem("key_not_found", 404), {}
        assert headers["bearer"] == "fixture-org-b"
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={
            Secret.FOREIGN.value: "fixture-org-b",
            Secret.FOREIGN_SESSION.value: "fixture-org-b-session",
        },
        foreign_ids=matrix_ids(),
    )
    rows = {row.get("operation"): row for row in result.evidence}
    assert rows["revokeApiKey"]["result"] == "pass"
    assert rows["revokeApiKey"]["actor_credential"] == "clerk_session"
    assert rows["getRun"]["actor_credential"] == "api_key"


def test_session_only_key_routes_without_a_session_stay_unproven(tmp_path):
    def responder(method, path, headers):
        if path.startswith("/v1/api-keys"):
            return 403, problem("session_required", 403), {}
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    rows = {row.get("operation"): row for row in result.evidence}
    assert rows["revokeApiKey"]["result"] == "unproven:uniform_403"


def checkout_responder(contract, completed):
    """Core billing fixture whose checkout completes once the hook has paid it."""
    url = "https://checkout.stripe.com/c/pay/cs_test_fixture"

    def responder(method, path, _):
        if path.startswith("/v1/billing/balance"):
            balance = example(contract, "/v1/billing/balance")
            amount = "310.00" if completed else "300.00"
            balance["ledger"] = {"amount": amount, "currency": "USD"}
            return 200, balance, {}
        if method == "POST" and path.startswith("/v1/billing/checkout"):
            created = example(contract, "/v1/billing/checkout", "post", "201")
            created.update(session_id="cs_test_fixture", url=url)
            return 201, created, {}
        if path.startswith("/v1/billing/checkouts/"):
            view = example(contract, "/v1/billing/checkouts/{id}")
            view.update(kind="credits", url=url, state="completed" if completed else "open")
            view["amount"] = {"amount": "10.00", "currency": "USD"}
            return 200, view, {}
        return 404, problem("not_found", 404), {}

    return responder, url


@pytest.mark.parametrize("surface", SURFACES)
def test_buy_credits_completes_stripe_test_checkout_through_hook(contract, tmp_path, surface):
    completed: list[str] = []

    def pay(url):
        completed.append(url)
        return None

    responder, url = checkout_responder(contract, completed)
    result, requests = run(
        surface, responder, tmp_path, Step.BUY_CREDITS_TEST, complete_checkout=pay
    )
    assert completed == [url]
    assert result.verdict == Verdict.PASS, result
    paths = [(method, path.split("?")[0]) for method, path, _ in requests]
    assert paths.index(("POST", "/v1/billing/checkout")) < paths.index(
        ("GET", "/v1/billing/checkouts/chk_AAAAAAAAQACAAAAAAAAAAQ")
    )


def test_buy_credits_without_hook_stays_inconclusive(contract, tmp_path):
    responder, _ = checkout_responder(contract, [])
    result, _ = run(SurfaceName.SDK, responder, tmp_path, Step.BUY_CREDITS_TEST)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "checkout_not_completed:stripe_test_completion_unavailable"


def test_buy_credits_typed_hook_failure_is_inconclusive(contract, tmp_path):
    responder, _ = checkout_responder(contract, [])
    result, _ = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.BUY_CREDITS_TEST,
        complete_checkout=lambda _url: "stripe_test_checkout_not_redirected",
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.cause == "checkout_not_completed:stripe_test_checkout_not_redirected"


def test_stripe_completer_refuses_live_or_foreign_pages():
    from scripts.acceptance.stripe_checkout import complete_test_checkout, test_mode_url

    assert test_mode_url("https://checkout.stripe.com/c/pay/cs_test_abc#frag")
    for url in (
        "https://checkout.stripe.com/c/pay/cs_live_abc",
        "http://checkout.stripe.com/c/pay/cs_test_abc",
        "https://evil.example/c/pay/cs_test_abc",
    ):
        assert complete_test_checkout(url) == "stripe_checkout_not_test_mode"
