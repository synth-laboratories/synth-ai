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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acceptance import journey_main
from scripts.acceptance.journeys import (
    JourneyContext,
    Secret,
    Step,
    Verdict,
    capabilities,
    matrix_operations,
    run_step,
)
from scripts.acceptance.surfaces import CLI_ARGV, SDK_CALLS, SurfaceName
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
            if length:
                self.rfile.read(length)
            bearer = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
            requests.append((self.command, self.path, bearer))
            status, payload, headers = responder(self.command, self.path, {"bearer": bearer})
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
    SurfaceName.SDK: (Verdict.FAIL, "refusal_untyped"),
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
    if (surface, step) == (SurfaceName.CLI, Step.OVERSIZE):
        # `synth run` accepts the goal only as argv, so the OS refuses it first.
        expected = (Verdict.INCONCLUSIVE, "capability_missing:cli_goal_only_argv")
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
def test_expired_key_is_typed_401(tmp_path, surface):
    def responder(method, path, headers):
        assert headers["bearer"] == "fixture-expired"
        return 401, problem("authentication_required", 401), {}

    result, _ = run(
        surface,
        responder,
        tmp_path,
        Step.EXPIRED_KEY,
        secrets={Secret.EXPIRED.value: "fixture-expired"},
    )
    assert result.verdict == Verdict.PASS, result


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
        return 404, problem("route_not_found", 404), {}

    result, _ = run(surface, responder, tmp_path, Step.INDEX)
    assert result.verdict == Verdict.PASS, result


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
        "evaluation_id": "eval_" + "F" * 22,
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
