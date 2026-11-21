"""SYN-5695 cross-org isolation steps and the journey entrypoint receipts.

Shares the loopback harness in journey_loopback.py; split from test_journeys.py
so each file stays under the 120 s pytest cap.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.parse import parse_qs, urlsplit

import pytest
from journey_loopback import (
    SURFACES,
    api,
    example,
    matrix_ids,
    problem,
    run,
)

from scripts.acceptance import journey_main, journeys
from scripts.acceptance.journeys import (
    Secret,
    Step,
    Verdict,
)
from scripts.acceptance.surfaces import SurfaceName

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("binding", ["same_org", "same_user", "wrong_session"])
def test_cross_org_refuses_unbound_fixture_before_resource_operations(tmp_path, binding):
    def responder(method, path, request):
        return 404, problem("resource_not_found", 404), {}

    result, requests = run(
        SurfaceName.SDK,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        binding=binding,
        secrets={Secret.FOREIGN: "fixture-org-b", Secret.FOREIGN_SESSION: "fixture-session-b"},
        foreign_ids=matrix_ids(),
    )
    assert result.verdict == Verdict.FAIL
    assert all(method == "GET" and path == "/v1/account" for method, path, _ in requests)


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
    if surface == SurfaceName.MCP_STDIO:
        assert results["getRunOutputContent"] == "uncovered:output_pin_missing_or_invalid"


def test_mcp_isolation_label_update_reaches_core(tmp_path):
    calls = []

    def responder(method, path, request):
        if method == "PATCH" and path.endswith("/labels"):
            if request["bearer"] == "fixture-admin":  # Org A's positive control.
                return 200, example(journeys.contract(), "/v1/runs/{run_id}/labels", "patch"), {}
            calls.append(request["body"])
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        SurfaceName.MCP_STDIO,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids=matrix_ids(),
    )
    rows = {row["operation"]: row["result"] for row in result.evidence}
    assert calls == [{"expected_revision": 1, "labels": {}}] * 2
    assert rows["updateRunLabels"] == "pass"


def test_mcp_isolation_output_content_preserves_the_output_pin(tmp_path):
    output_pin = {"revision_id": "fixture-output-revision", "digest_sha256": "a" * 64}
    calls = []

    def responder(method, path, _):
        url = urlsplit(path)
        if method == "GET" and url.path.endswith("/content"):
            calls.append((url.path, parse_qs(url.query)))
        return 404, problem("resource_not_found", 404), {}

    ids = matrix_ids()
    result, _ = run(
        SurfaceName.MCP_STDIO,
        responder,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN.value: "fixture-org-b"},
        foreign_ids={
            **ids,
            "output_revision_id": output_pin["revision_id"],
            "output_digest_sha256": output_pin["digest_sha256"],
        },
    )
    rows = {row["operation"]: row["result"] for row in result.evidence}
    assert len(calls) == 2
    assert calls[0][0] == f"/v1/runs/{ids['run_id']}/outputs/{ids['output_id']}/content"
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] == calls[1][1] == {name: [value] for name, value in output_pin.items()}
    assert rows["getRunOutputContent"] == "pass"


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


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.35.1", "not_in_release:1.35.1"),
        ("1.36.0", "uncovered:contract_operation_missing"),
    ],
)
def test_absent_message_routes_follow_the_selected_release_contract(
    tmp_path, monkeypatch, version, expected
):
    document = journeys.contract()
    document["info"]["version"] = version
    for path, methods in document["paths"].items():
        document["paths"][path] = {
            method: operation
            for method, operation in methods.items()
            if operation.get("operationId") not in journeys.REQUIRED_MESSAGE_OPERATIONS
        }
    monkeypatch.setattr(journeys, "contract", lambda: document)

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
    for operation in journeys.REQUIRED_MESSAGE_OPERATIONS:
        assert results[operation] == expected
    uncovered = sum(
        1 for value in results.values() if value.startswith(("uncovered:", "unproven:"))
    )
    assert result.cause == f"uncovered={uncovered}"


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


REMAINING_BINDINGS = {
    SurfaceName.SDK: {
        "finalizeUpload",
        "getRunOutputContent",
        "getUpload",
        "updateApiKey",
        "updateRunLabels",
        "waitRun",
        "withdrawPlanRequest",
    },
    SurfaceName.CLI: {
        "finalizeUpload",
        "getRunOutput",
        "getRunOutputContent",
        "getUpload",
        "listRunOutputs",
        "updateApiKey",
        "updateRunLabels",
        "waitRun",
        "withdrawPlanRequest",
    },
    SurfaceName.MCP_STDIO: {"getRunOutputContent", "withdrawPlanRequest"},
}


def remaining_ids():
    return {
        **matrix_ids(),
        "owner_org_id": "00000000-0000-4000-8000-000000000001",
        "owner_user_id": "user_owner",
        "foreign_org_id": "00000000-0000-4000-8000-000000000002",
        "foreign_user_id": "user_other",
        "output_revision_id": "rev_fixture",
        "output_digest_sha256": "a" * 64,
    }


@pytest.mark.parametrize("surface", SURFACES)
def test_remaining_isolation_dispatches_only_selected_operations(surface, tmp_path):
    ids = remaining_ids()
    selected = REMAINING_BINDINGS[surface]
    calls = []

    def respond(method, path, request):
        calls.append((method, path, request))
        if path.startswith("/v1/api-keys/"):
            assert request["bearer"] == "fixture-session-b"
            assert request["body"]["expected_version"] == 12
        else:
            assert request["bearer"] == "fixture-org-b"
        if path.endswith("/labels"):
            assert request["body"]["expected_revision"] == 7
        if "/content?" in path:
            assert parse_qs(urlsplit(path).query) == {
                "revision_id": [ids["output_revision_id"]],
                "digest_sha256": [ids["output_digest_sha256"]],
            }
        if "/wait?" in path:
            assert parse_qs(urlsplit(path).query) == {"timeout_seconds": ["1"]}
        return 404, problem("resource_not_found", 404), {}

    result, _ = run(
        surface,
        respond,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN: "fixture-org-b", Secret.FOREIGN_SESSION: "fixture-session-b"},
        foreign_ids=ids,
        isolation_operations=frozenset(selected),
        isolation_bodies={
            "updateApiKey": {"expected_version": 12, "scopes": ["account:read"]},
            "updateRunLabels": {"expected_revision": 7, "labels": {"test": "fixture"}},
        },
    )
    assert result.verdict == Verdict.PASS, result.evidence
    assert {row["operation"] for row in result.evidence} == selected
    assert len(calls) == 2 * len(selected)


def test_remaining_isolation_rejects_a_different_driver_principal(tmp_path):
    ids = {**remaining_ids(), "foreign_user_id": "different_bound_user"}

    def respond(*_):
        pytest.fail("no resource probe may use other principals")

    result, requests = run(
        SurfaceName.SDK,
        respond,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN: "fixture-org-b"},
        foreign_ids=ids,
        isolation_operations=frozenset({"getUpload"}),
    )
    assert result.cause == "designated_fixture_principals_mismatch"
    assert all(path == "/v1/account" for _, path, _ in requests)


def test_remaining_isolation_stops_after_a_foreign_acceptance(contract, tmp_path):
    ids = remaining_ids()

    def respond(method, path, _):
        assert urlsplit(path).path == f"/v1/runs/{ids['run_id']}/outputs"
        return 200, example(contract, "/v1/runs/{run_id}/outputs"), {}

    result, requests = run(
        SurfaceName.SDK,
        respond,
        tmp_path,
        Step.CROSS_ORG_404,
        secrets={Secret.FOREIGN: "fixture-org-b"},
        foreign_ids=ids,
        isolation_operations=frozenset({"listRunOutputs", "getRunOutput"}),
    )
    assert result.verdict == Verdict.FAIL
    assert [row["result"] for row in result.evidence] == ["leak"]
    assert sum(path != "/v1/account" for _, path, _ in requests) == 1
