"""SDK and stdio MCP scoped report-version shares (SYN-5157)."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import OutcomeUncertainError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.sdk.forge.types import ReportShareIntent
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

ORG = "81000000-0000-4000-8000-000000000001"
PROJECT = "81000000-0000-4000-8000-000000000002"
INVITED = "81000000-0000-4000-8000-000000000003"
DIGEST = "a" * 64
OTHER = "b" * 64
EXPIRES = datetime(2030, 1, 1, tzinfo=UTC)
SHARE_TOOLS = (
    "forge_report_share",
    "forge_report_share_get",
    "forge_report_shares_list",
    "forge_report_share_revoke",
)
OPERATIONS = (
    "forge.report_shares.put",
    "forge.report_shares.get",
    "forge.report_shares.list",
    "forge.report_shares.revoke",
)


def share(digest=DIGEST, *, audience="invited", expired=False):
    return {
        "schema_version": "synth.forge.report-share.v1",
        "scope": {"organization_id": ORG, "project_id": PROJECT},
        "version_sha256": digest,
        "audience": audience,
        "invited": ["user:" + INVITED] if audience == "invited" else [],
        "policy_version": 2,
        "publication_id": "pub-1",
        "expires_at": EXPIRES.isoformat(),
        "expired": expired,
    }


def wire(changed=False):
    capability, headers = receipt_and_headers()
    for name in OPERATIONS:
        capability["operations"].append(
            {
                "operation_id": name,
                "contract": "core.v1",
                "state": "enabled",
                "surfaces": ["http", "mcp"],
                "reason": None,
            }
        )
    capability["operations"].sort(key=lambda item: (item["contract"], item["operation_id"]))
    snapshot = {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
    digest = snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(snapshot)))
    capability["capability_sha256"] = headers["X-Synth-Capabilities-SHA256"] = digest
    calls = []

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        metadata = {**capability["metadata"], "contract": "core.v1"}
        if request.url.path.endswith("/report-shares"):
            payload = {"shares": [share(), share(OTHER, audience="organization", expired=True)]}
        else:
            payload = {"share": share(OTHER if changed else DIGEST)}
        return httpx.Response(200, headers=headers, json={"metadata": metadata, **payload})

    return httpx.MockTransport(respond), calls


def client_for(transport):
    network = httpx.Client(transport=transport)
    return network, SynthClient(
        api_key="fixture",
        base_url="https://backend.test",
        options=ClientOptions(http_client=network),
    )


def test_sync_share_get_list_revoke_round_trip_with_expiry():
    transport, calls = wire()
    network, client = client_for(transport)
    with network, client:
        created = client.forge.reports.share(
            PROJECT, DIGEST, audience="invited", invited=[INVITED], expires_at=EXPIRES
        )
        assert created.share.expires_at == EXPIRES and created.share.expired is False
        assert client.forge.reports.get_share(PROJECT, DIGEST).share.audience == "invited"
        page = client.forge.reports.list_shares(PROJECT, limit=10)
        assert [item.expired for item in page.shares] == [False, True]
        assert client.forge.reports.revoke_share(PROJECT, DIGEST).share.version_sha256 == DIGEST
    put, get, listed, revoke = calls
    assert (put.method, put.url.path) == (
        "PUT",
        f"/api/v1/forge/projects/{PROJECT}/report-shares/{DIGEST}",
    )
    body = json.loads(put.content)
    assert body["operation_id"] == put.headers["Idempotency-Key"]
    assert body["audience"] == "invited" and body["invited"] == [INVITED]
    assert datetime.fromisoformat(body["expires_at"]) == EXPIRES
    assert get.method == "GET" and get.url.path.endswith(f"/report-shares/{DIGEST}")
    assert listed.url.params["limit"] == "10"
    assert (revoke.method, revoke.url.path) == (
        "POST",
        f"/api/v1/forge/projects/{PROJECT}/report-shares/{DIGEST}/revoke",
    )


def test_a_different_share_in_the_response_is_uncertain():
    transport, _ = wire(changed=True)
    network, client = client_for(transport)
    with network, client:
        with pytest.raises(OutcomeUncertainError):
            client.forge.reports.share(PROJECT, DIGEST, audience="organization")
        with pytest.raises(OutcomeUncertainError):
            client.forge.reports.revoke_share(PROJECT, DIGEST)


@pytest.mark.parametrize("audience", ["public", "unlisted", "link", "anyone"])
def test_no_public_or_unlisted_audience_exists_in_the_client(audience):
    with pytest.raises(ValueError):
        ReportShareIntent(operation_id="k", audience=audience)
    with pytest.raises(ValueError):
        BY_NAME["forge_report_share"].validate_arguments(
            {"project_id": PROJECT, "digest": DIGEST, "audience": audience, "idempotency_key": "k"}
        )


def test_expiry_must_be_timezone_aware():
    transport, calls = wire()
    network, client = client_for(transport)
    with network, client:
        with pytest.raises(ValueError):
            client.forge.reports.share(
                PROJECT, DIGEST, audience="organization", expires_at=datetime(2030, 1, 1)
            )
    assert calls == []
    with pytest.raises(ValueError):
        BY_NAME["forge_report_share"].validate_arguments(
            {
                "project_id": PROJECT,
                "digest": DIGEST,
                "audience": "organization",
                "expires_at": "2030-01-01T00:00:00",
                "idempotency_key": "k",
            }
        )


def test_stdio_mcp_share_tools_call_the_public_routes():
    transport, calls = wire()
    later = (datetime.now(UTC) + timedelta(days=1)).isoformat()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            base = {"project_id": PROJECT, "digest": DIGEST}
            created = await BY_NAME["forge_report_share"].invoke(
                client,
                {**base, "audience": "organization", "expires_at": later, "idempotency_key": "s1"},
            )
            got = await BY_NAME["forge_report_share_get"].invoke(client, base)
            listed = await BY_NAME["forge_report_shares_list"].invoke(
                client, {"project_id": PROJECT}
            )
            revoked = await BY_NAME["forge_report_share_revoke"].invoke(
                client, {**base, "idempotency_key": "r1"}
            )
            return created, got, listed, revoked

    created, got, listed, revoked = asyncio.run(run())
    assert created["share"]["version_sha256"] == DIGEST and "expired" in created["share"]
    assert got["share"]["audience"] == "invited"
    assert len(listed["shares"]) == 2
    assert revoked["share"]["version_sha256"] == DIGEST
    assert [call.headers.get("Idempotency-Key") for call in calls] == ["s1", None, None, "r1"]


def test_share_tools_carry_no_authority_fields():
    for name, scope, mutates in zip(
        SHARE_TOOLS,
        ("forge:write", "forge:read", "forge:read", "forge:write"),
        (True, False, False, True),
        strict=True,
    ):
        tool = BY_NAME[name]
        assert (tool.scope, tool.mutates, tool.paid) == (scope, mutates, False)
    base = {"project_id": PROJECT, "digest": DIGEST, "idempotency_key": "k"}
    for field in ("principal", "owner", "authorization", "acting_credential", "visibility"):
        with pytest.raises(ValueError):
            BY_NAME["forge_report_share_revoke"].validate_arguments({**base, field: "invented"})
