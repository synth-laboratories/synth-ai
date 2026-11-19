"""Loopback HTTP harness shared by the SYN-5695 journey step tests.

Real SDK, CLI subprocess and MCP stdio clients answer one scripted server.
These fixtures prove the step logic and client typing; they are not cached-slot
or live acceptance receipts.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable
from contextlib import contextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.acceptance import journey_main, journeys
from scripts.acceptance.journeys import (
    JourneyContext,
    Step,
    run_step,
)
from scripts.acceptance.surfaces import SurfaceName
from synth_ai.errors import local_problem

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


OWNER_BEARERS = frozenset({"fixture-admin", "fixture-session-a"})


def organization_accounts(responder, binding, foreign_scopes=()):
    contract = journeys.contract()

    def answer(method, path, request):
        if path != "/v1/account":
            return responder(method, path, request)
        account = example(contract, path)
        owner = request["bearer"] in OWNER_BEARERS
        org = (
            "00000000-0000-4000-8000-000000000001"
            if owner or binding == "same_org"
            else "00000000-0000-4000-8000-000000000002"
        )
        user = "user_owner" if owner or binding == "same_user" else "user_other"
        if binding == "wrong_session" and request["bearer"] == "fixture-session-b":
            user = "user_third"
        account["org"]["id"] = org
        account["principal"].update(org_id=org, user_id=user)
        if not owner:
            account["principal"]["scopes"] = list(foreign_scopes)
        return 200, account, {}

    return answer


def run(surface, responder, tmp_path, step, **options):
    if step == Step.CROSS_ORG_404:
        responder = organization_accounts(
            responder, options.pop("binding", "distinct"), options.pop("foreign_scopes", ())
        )
    with api(responder) as (origin, requests):
        with context_for(surface, origin, tmp_path, **options) as context:
            return run_step(context, step), requests


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
