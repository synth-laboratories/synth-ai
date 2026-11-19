"""SYN-5695 J3 access steps: member versus admin, key mint/rotate/revoke.

Shares the loopback harness in journey_loopback.py; split from test_journeys.py
so each file stays under the 120 s pytest cap.
"""

from __future__ import annotations

import copy

import pytest
from journey_loopback import (
    SURFACES,
    api,
    context_for,
    example,
    problem,
    run,
)

from scripts.acceptance.journeys import (
    Secret,
    Step,
    Verdict,
    run_step,
)

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("binding", ["distinct", "same_user", "other_org", "admin_credential"])
def test_member_is_refused_billing_admin_change(contract, tmp_path, surface, binding):
    account = example(contract, "/v1/account")

    def responder(method, path, headers):
        if path == "/v1/account":
            body = copy.deepcopy(account)
            role = "member" if headers["bearer"] == "fixture-member" else "admin"
            body["principal"]["user_id"] = "user_" + (role if binding != "same_user" else "one")
            body["principal"]["org_role"] = "admin" if binding == "admin_credential" else role
            if role == "member" and binding == "other_org":
                body["org"]["id"] = body["principal"]["org_id"] = (
                    "00000000-0000-4000-8000-000000000002"
                )
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
    if binding == "distinct":
        assert result.verdict == Verdict.PASS, result
        assert ("POST", "/v1/billing/subscription/cancel", "fixture-member") in requests
    else:
        assert result.verdict == Verdict.FAIL
        assert all(method == "GET" for method, _, _ in requests)


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
