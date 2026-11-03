"""SDK and stdio MCP report publish / revoke-inactive (SYN-5157)."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import OutcomeUncertainError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.sdk.forge.contracts.contracts import ExactReference, Producer, Scope, contract_digest
from synth_ai.sdk.forge.contracts.operations import RecordRevision
from synth_ai.sdk.forge.contracts.records import Report
from synth_ai.sdk.forge.contracts.report_manifests import (
    manifest_from_report,
    publish_initial,
    publish_next,
    version_digest,
)
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

ORG = "81000000-0000-4000-8000-000000000001"
PROJECT = "81000000-0000-4000-8000-000000000002"
AT = datetime(2026, 10, 28, 12, tzinfo=UTC)


def versions():
    payload = Report(title="Report", body="Body.", methods="Measured once.")
    record = RecordRevision(
        scope=Scope(organization_id=ORG, project_id=PROJECT),
        reference=ExactReference(
            authority="forge",
            kind="report",
            record_id="report-1",
            revision="1",
            digest_sha256=contract_digest(payload),
        ),
        payload=payload,
        producer=Producer(authority="human", subject_id="user-1"),
        operation_id="op-report-1",
        recorded_at=AT,
    )
    first = publish_initial(manifest_from_report(record), AT)
    return first, publish_next(
        first, disposition="retraction", reason="Withdrawn.", published_at=AT.replace(hour=13)
    )


FIRST, RETRACTION = versions()


def wire(changed=False):
    capability, headers = receipt_and_headers()
    for name in ("forge.report_versions.publish", "forge.report_shares.revoke_inactive"):
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
        body = json.loads(request.content)
        if request.url.path.endswith("/report-versions"):
            digest = "0" * 64 if changed else version_digest(RETRACTION)
            payload = {"publication": {"version_sha256": digest, "revocation": None}}
        else:
            payload = {
                "revocation": {
                    "scope": {"organization_id": ORG, "project_id": PROJECT},
                    "lineage_id": "other" if changed else body["lineage_id"],
                    "head_sha256": version_digest(RETRACTION),
                    "revocations": [
                        {
                            "version_sha256": version_digest(FIRST),
                            "version": 1,
                            "status": "pending",
                            "refusal_code": "evidence_unavailable",
                            "retryable": True,
                        }
                    ],
                }
            }
        return httpx.Response(200, headers=headers, json={"metadata": metadata, **payload})

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("changed", [False, True])
def test_sync_client_treats_a_different_digest_or_lineage_as_uncertain(changed):
    transport, calls = wire(changed)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
    ):
        if changed:
            with pytest.raises(OutcomeUncertainError):
                client.forge.reports.publish(PROJECT, RETRACTION)
            with pytest.raises(OutcomeUncertainError):
                client.forge.reports.revoke_inactive(PROJECT, FIRST.lineage_id)
        else:
            result = client.forge.reports.publish(PROJECT, RETRACTION)
            assert result.publication.version_sha256 == version_digest(RETRACTION)
            report = client.forge.reports.revoke_inactive(PROJECT, FIRST.lineage_id)
            assert report.revocation.revocations[0].status == "pending"
    publish = calls[0]
    assert publish.url.path == f"/api/v1/forge/projects/{PROJECT}/report-versions"
    body = json.loads(publish.content)
    assert body["operation_id"] == publish.headers["Idempotency-Key"]
    assert set(body) == {"schema_version", "operation_id", "version"}
    assert calls[1].url.path.endswith("/report-shares/revoke-inactive")


def test_stdio_mcp_tools_call_the_public_routes_with_stable_keys():
    transport, calls = wire()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            published = await BY_NAME["forge_report_version_publish"].invoke(
                client,
                {
                    "project_id": PROJECT,
                    "version": RETRACTION.model_dump(mode="json"),
                    "idempotency_key": "publish-1",
                },
            )
            revoked = await BY_NAME["forge_report_shares_revoke_inactive"].invoke(
                client,
                {"project_id": PROJECT, "lineage_id": FIRST.lineage_id, "idempotency_key": "rev-1"},
            )
            return published, revoked

    published, revoked = asyncio.run(run())
    assert published["publication"]["version_sha256"] == version_digest(RETRACTION)
    assert revoked["revocation"]["lineage_id"] == FIRST.lineage_id
    assert [call.headers["Idempotency-Key"] for call in calls] == ["publish-1", "rev-1"]


def test_mcp_report_tools_cannot_supply_authority_or_producer():
    for name in ("forge_report_version_publish", "forge_report_shares_revoke_inactive"):
        tool = BY_NAME[name]
        assert tool.scope == "forge:write" and tool.mutates and tool.keyed and not tool.paid
    base = {"project_id": PROJECT, "lineage_id": FIRST.lineage_id, "idempotency_key": "k"}
    for field in ("principal", "producer", "authorization", "acting_credential"):
        with pytest.raises(ValueError):
            BY_NAME["forge_report_shares_revoke_inactive"].validate_arguments(
                {**base, field: "invented"}
            )
