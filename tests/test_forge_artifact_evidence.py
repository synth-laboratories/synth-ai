"""Both SDK clients and MCP preserve exact revision evidence and absent audits."""

import asyncio
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

PROJECT = "81000000-0000-4000-8000-000000000002"


def wire(changed=False):
    capability, headers = receipt_and_headers()
    capability["operations"].append(
        {
            "operation_id": "forge.records.artifact_evidence",
            "contract": "core.v1",
            "state": "enabled",
            "surfaces": ["http", "mcp"],
            "reason": None,
        }
    )
    snapshot = {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
    digest = snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(snapshot)))
    capability["capability_sha256"] = headers["X-Synth-Capabilities-SHA256"] = digest
    calls = []

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        return httpx.Response(
            200,
            headers=headers,
            json={
                "metadata": {**capability["metadata"], "contract": "core.v1"},
                "artifact_evidence": {
                    "schema_version": "forge.workspace-artifact-view.v1",
                    "scope": {
                        "organization_id": "81000000-0000-4000-8000-000000000001",
                        "project_id": PROJECT,
                    },
                    "revision": {
                        "schema_version": "forge.revision-reference.v1",
                        "reference": {
                            "authority": "forge",
                            "kind": "artifact",
                            "record_id": "other" if changed else "material",
                            "revision": "1",
                            "digest_sha256": "a" * 64,
                        },
                        "document_digest_sha256": "b" * 64,
                    },
                    "evidence": None,
                    "verification_reference": None,
                },
            },
        )

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_artifact_evidence_exact_identity_and_legacy_absence(async_mode, changed):
    transport, calls = wire(changed)

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            if changed:
                with pytest.raises(ContractError):
                    await client.forge.records.artifact_evidence(PROJECT, "material", revision=1)
            else:
                tool = BY_NAME["forge_artifact_evidence_get"]
                result = await tool.invoke(
                    client, {"project_id": PROJECT, "record_id": "material", "revision": 1}
                )
                assert result["artifact_evidence"]["evidence"] is None

    if async_mode:
        asyncio.run(run())
    else:
        with (
            httpx.Client(transport=transport) as network,
            SynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(http_client=network),
            ) as client,
        ):
            if changed:
                with pytest.raises(ContractError):
                    client.forge.records.artifact_evidence(PROJECT, "material", revision=1)
            else:
                result = client.forge.records.artifact_evidence(PROJECT, "material", revision=1)
                assert result.artifact_evidence.verification_reference is None
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/records/material/artifact-evidence")
    assert calls[0].url.params["revision"] == "1"
    assert calls[0].method == "GET"


def test_mcp_evidence_read_cannot_supply_authority_or_past_receipt():
    tool = BY_NAME["forge_artifact_evidence_get"]
    assert tool.scope == "forge:read" and not tool.paid
    for field in ("principal", "authorization", "evidence", "credential_id"):
        with pytest.raises(ValueError):
            tool.validate_arguments(
                {"project_id": PROJECT, "record_id": "material", field: "invented"}
            )


def test_generated_archive_decoder_preserves_native_revision_provenance():
    from datetime import UTC, datetime
    from hashlib import sha256

    from synth_ai.sdk.forge.contracts.contracts import (
        ExactReference,
        Producer,
        Scope,
        contract_digest,
    )
    from synth_ai.sdk.forge.contracts.freezing import (
        ARCHIVE_V3,
        ARCHIVE_V4,
        scientific_archive,
        scientific_archive_sources,
        verify_scientific_archive,
    )
    from synth_ai.sdk.forge.contracts.operations import revision_reference
    from synth_ai.sdk.forge.contracts.operations_v2 import RecordRevisionV2
    from synth_ai.sdk.forge.contracts.records import ResearchLog, Snapshot

    scope = Scope(organization_id="81000000-0000-4000-8000-000000000001", project_id=PROJECT)
    payload = ResearchLog(title="A retained observation")
    now = datetime.now(UTC)
    record = RecordRevisionV2(
        scope=scope,
        payload=payload,
        reference=ExactReference(
            authority="forge",
            kind="research_log",
            record_id="log",
            revision="1",
            digest_sha256=contract_digest(payload),
        ),
        producer=Producer(authority="human", subject_id="human"),
        operation_id="original",
        recorded_at=now,
    )
    content = scientific_archive(
        scope, (record,), cutoff_at=now, gaps=("No execution.",), schema_version=ARCHIVE_V4
    )
    snapshot = Snapshot(
        cutoff_at=now,
        sources=scientific_archive_sources((record,), schema_version=ARCHIVE_V4),
        archive=ExactReference(
            authority="artifact-platform",
            kind="archive",
            record_id=PROJECT,
            revision="1",
            digest_sha256=sha256(content).hexdigest(),
        ),
        capture_completeness="partial",
        gaps=("No execution.",),
    )
    assert verify_scientific_archive(
        content,
        scope,
        snapshot,
        retained=lambda scope, reference: revision_reference(record),
        require_revisions=True,
    ) == (record,)
    with pytest.raises(ValueError):
        scientific_archive(
            scope, (record,), cutoff_at=now, gaps=("No execution.",), schema_version=ARCHIVE_V3
        )
