"""SYN-5737 owner-mutation and grant-revocation cells of the cross-org matrix.

Shares the loopback harness in journey_loopback.py; kept apart from
test_journeys_isolation.py so each file stays under the 120 s pytest cap.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest
from journey_loopback import SURFACES, example, matrix_ids, problem, run

from scripts.acceptance import journeys
from scripts.acceptance.journeys import Secret, Step, Verdict
from scripts.acceptance.surfaces import SurfaceName

pytestmark = pytest.mark.acceptance

# SYN-5737: the standard matrix covers the owner mutations on every driver. Org B's
# refusal counts only next to org A's own call (the positive control), and a refused
# API key is an observation, never a pass of the contract's apiKey scheme.
OWNER_MUTATION_ROUTES = {
    "updateApiKey": ("PATCH", "/v1/api-keys/{key_id}", "patch"),
    "updateRunLabels": ("PATCH", "/v1/runs/{run_id}/labels", "patch"),
    "finalizeUpload": ("POST", "/v1/uploads/{upload_id}/finalize", "post"),
    "withdrawPlanRequest": ("DELETE", "/v1/plan-requests/{request_id}", "delete"),
}
MATRIX_SECRETS = {
    Secret.FOREIGN.value: "fixture-org-b",
    Secret.FOREIGN_SESSION.value: "fixture-session-b",
    Secret.SESSION.value: "fixture-session-a",
}


def full_matrix_ids():
    return {
        **matrix_ids(),
        "message_id": "msg_" + "F" * 22,
        "output_revision_id": "rev_fixture",
        "output_digest_sha256": "a" * 64,
    }


def owner_mutation(contract, ids, method, path):
    for operation, (verb, template, _) in OWNER_MUTATION_ROUTES.items():
        if verb == method and urlsplit(path).path == template.format(**ids):
            return operation
    return None


def staging_like(contract, ids, owner_calls, *, owner_answer=None):
    """Org B: 404s by session, 403 session_required by key on session-gated routes."""

    def responder(method, path, request):
        bearer = request["bearer"]
        operation = owner_mutation(contract, ids, method, path)
        if bearer in {"fixture-admin", "fixture-session-a"}:
            assert operation is not None, f"owner called {method} {path}"
            owner_calls.append((operation, bearer, request["body"]))
            if owner_answer is not None:
                return owner_answer
            _, template, verb = OWNER_MUTATION_ROUTES[operation]
            return 200, example(contract, template, verb), {}
        if bearer == "fixture-org-b" and path.startswith(("/v1/api-keys", "/mcp/oauth/grants")):
            return 403, problem("session_required", 403), {}
        return 404, problem("resource_not_found", 404), {}

    return responder


@pytest.mark.parametrize("surface", SURFACES)
def test_owner_mutations_are_covered_with_the_owner_positive_control(contract, tmp_path, surface):
    ids = {**full_matrix_ids(), "key_version": "4", "labels_revision": "9"}
    owner_calls = []
    result, _ = run(
        surface,
        staging_like(contract, ids, owner_calls),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=MATRIX_SECRETS,
        foreign_ids=ids,
    )
    rows = {row["operation"]: row for row in result.evidence}
    for operation in OWNER_MUTATION_ROUTES:
        assert rows[operation]["result"] == "pass", rows[operation]
        assert rows[operation]["owner_control"]["result"] == "accepted"
        assert rows[operation]["foreign"]["status"] == 404
    assert rows["updateApiKey"]["owner_control"]["credential"] == "clerk_session"
    assert rows["finalizeUpload"]["owner_control"]["credential"] == "api_key"
    # Org A's control runs once per mutation, with the pinned concurrency values.
    assert sorted(operation for operation, _, _ in owner_calls) == sorted(OWNER_MUTATION_ROUTES)
    bodies = {operation: (bearer, body) for operation, bearer, body in owner_calls}
    assert bodies["updateApiKey"][0] == "fixture-session-a"
    assert bodies["updateApiKey"][1]["expected_version"] == 4
    assert bodies["updateRunLabels"][1]["expected_revision"] == 9
    # Denominator: every ID-bearing contract operation passes on every driver.
    results = [row["result"] for row in result.evidence]
    assert results.count("pass") == len(journeys.matrix_operations()) == 55, results
    assert result.verdict == Verdict.PASS, result.cause


@pytest.mark.parametrize("surface", [SurfaceName.SDK, SurfaceName.CLI])
def test_an_owner_refusal_leaves_the_mutation_cell_unproven(contract, tmp_path, surface):
    ids = full_matrix_ids()
    result, _ = run(
        surface,
        staging_like(contract, ids, [], owner_answer=(404, problem("resource_not_found", 404), {})),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=MATRIX_SECRETS,
        foreign_ids=ids,
    )
    rows = {row["operation"]: row for row in result.evidence}
    for operation in OWNER_MUTATION_ROUTES:
        assert rows[operation]["result"] == "unproven:owner_control_resource_not_found"
        assert rows[operation]["owner_control"]["result"] == "refused"
    assert result.verdict == Verdict.INCONCLUSIVE and result.cause == "uncovered=4"


def test_a_consumed_owner_fixture_is_a_resolved_conflict(contract, tmp_path):
    ids = full_matrix_ids()
    result, _ = run(
        SurfaceName.SDK,
        staging_like(
            contract, ids, [], owner_answer=(409, problem("upload_session_closed", 409), {})
        ),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=MATRIX_SECRETS,
        foreign_ids=ids,
    )
    rows = {row["operation"]: row for row in result.evidence}
    assert rows["finalizeUpload"]["result"] == "pass"
    control = rows["finalizeUpload"]["owner_control"]
    assert control["result"] == "resolved_conflict:upload_session_closed"
    assert control["status"] == 409


def test_session_only_owner_control_without_an_owner_session_is_unproven(contract, tmp_path):
    ids = full_matrix_ids()
    secrets = {name: value for name, value in MATRIX_SECRETS.items() if name != Secret.SESSION}
    result, _ = run(
        SurfaceName.SDK,
        staging_like(contract, ids, []),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=secrets,
        foreign_ids=ids,
    )
    rows = {row["operation"]: row for row in result.evidence}
    assert rows["updateApiKey"]["result"] == "unproven:owner_session_absent"
    assert rows["updateRunLabels"]["result"] == "pass"
    assert result.verdict == Verdict.INCONCLUSIVE


@pytest.mark.parametrize(
    ("surface", "foreign_scopes", "finding"),
    [
        (SurfaceName.SDK, (), "key_lacks_required_scope"),
        (SurfaceName.CLI, (), "key_lacks_required_scope"),
        (SurfaceName.MCP_STDIO, (), "key_lacks_required_scope"),
        (
            SurfaceName.SDK,
            ("account:write",),
            "contract_deviation:scoped_user_bound_api_key_refused",
        ),
    ],
)
def test_grant_revocation_is_proven_by_session_and_the_key_refusal_is_recorded(
    contract, tmp_path, surface, foreign_scopes, finding
):
    ids = full_matrix_ids()
    result, _ = run(
        surface,
        staging_like(contract, ids, []),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=MATRIX_SECRETS,
        foreign_ids=ids,
        foreign_scopes=foreign_scopes,
    )
    row = {row["operation"]: row for row in result.evidence}["revokeOAuthGrant"]
    assert row["result"] == "pass" and row["actor_credential"] == "clerk_session"
    assert row["foreign"]["status"] == 404
    assert row["api_key_observation"] == {
        "status": 403,
        "code": "session_required",
        "required_scope": "account:write",
        "key_holds_required_scope": bool(foreign_scopes),
        "finding": finding,
        "api_key_scheme": "not_proven",
    }


def test_grant_revocation_refused_to_a_key_alone_never_passes(contract, tmp_path):
    ids = full_matrix_ids()
    secrets = {Secret.FOREIGN.value: "fixture-org-b"}
    result, _ = run(
        SurfaceName.SDK,
        staging_like(contract, ids, []),
        tmp_path,
        Step.CROSS_ORG_404,
        secrets=secrets,
        foreign_ids=ids,
    )
    row = {row["operation"]: row for row in result.evidence}["revokeOAuthGrant"]
    assert row["result"] == "unproven:uniform_403" and "api_key_observation" not in row
    assert result.verdict == Verdict.INCONCLUSIVE
