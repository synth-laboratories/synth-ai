"""Public upload clients/MCP against synthetic wire custody, with no byte service."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.runtime import SDKExecutor
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.forge.index_types import ForgeIndexDraftUploadResponse
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_index_drafts import OPERATION, PIN, PROJECT
from tests.test_forge_workspaces import receipt_and_headers

DRAFT = "original-draft"
DESCRIPTOR = {
    "kind": "engineering_note",
    "title": "Fixture",
    "abstract": "No execution claim.",
    "research_areas": ["systems"],
    "workflow_stages": ["research_operations"],
    "card": dict.fromkeys(
        ("reusable", "how_to_use", "observed", "limitations", "resources"), "Fixture only."
    ),
    "assets": [{"asset_id": "report", "role": "report", "logical_path": "report.txt"}],
    "claims": [
        {
            "claim_id": "claim",
            "kind": "diagnostic",
            "statement": "Fixture only.",
            "scope": "Fixture.",
            "author_assessment": "not_tested",
            "missing_evidence_reason": "No execution.",
        }
    ],
    "reproduction": {
        "level": "inspectable",
        "instructions": "Inspect.",
        "expected_outputs": "Fixture.",
        "restrictions": "No run.",
    },
}


def fixture_transport(wrong_identity=False):
    capability, headers = receipt_and_headers()
    for operation in capability["operations"]:
        operation["operation_id"] = (
            operation["operation_id"]
            .replace("forge.projects.apply", "forge.index_drafts.upload")
            .replace("forge.projects.get", "forge.index_drafts.upload_get")
        )
    capability["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(
            json.dumps(
                {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
            )
        )
    )
    headers["X-Synth-Capabilities-SHA256"] = capability["capability_sha256"]
    calls = []

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        assert request.url.path.startswith(
            f"/api/v1/index/forge/projects/{PROJECT}/drafts/{DRAFT}/uploads"
        )
        if request.method == "POST":
            assert request.headers["Idempotency-Key"] == "original-upload"
            assert (
                request.headers["X-Synth-Intent-SHA256"]
                == hashlib.sha256(request.content).hexdigest()
            )
            assert json.loads(request.content)["descriptor"]["assets"] == DESCRIPTOR["assets"]
        result = ForgeIndexDraftUploadResponse.model_validate_json(
            json.dumps(
                {
                    "metadata": {
                        **capability["metadata"],
                        "contract": "core.v1",
                        "operation_id": OPERATION,
                    },
                    "project_id": PROJECT,
                    "draft_operation_id": "foreign" if wrong_identity else DRAFT,
                    "export_revision": PIN,
                    "upload_id": "original-upload-resource",
                    "contribution_id": "allocated",
                    "revision_id": "revision",
                    "manifest": {
                        "byte_size": 100,
                        "digest_sha256": "a" * 64,
                        "media_type": "application/zip",
                    },
                    "package": {
                        "contribution_id": "allocated",
                        "revision_id": "revision",
                        "package_id": "package",
                        "digest_sha256": "a" * 64,
                    },
                    "state": "finalized",
                }
            )
        )
        return httpx.Response(200, headers=headers, json=result.model_dump(mode="json"))

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
def test_upload_retry_and_get_preserve_custody(async_mode):
    transport, calls = fixture_transport()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            first = await client.forge.index_drafts.upload(
                PROJECT,
                DRAFT,
                DESCRIPTOR,
                mutation=MutationOptions(idempotency_key="original-upload"),
            )
            again = await client.forge.index_drafts.upload(
                PROJECT,
                DRAFT,
                DESCRIPTOR,
                mutation=MutationOptions(idempotency_key="original-upload"),
            )
            found = await client.forge.index_drafts.upload_get(PROJECT, DRAFT, OPERATION)
            assert first.package == again.package == found.package

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
            first = client.forge.index_drafts.upload(
                PROJECT,
                DRAFT,
                DESCRIPTOR,
                mutation=MutationOptions(idempotency_key="original-upload"),
            )
            again = client.forge.index_drafts.upload(
                PROJECT,
                DRAFT,
                DESCRIPTOR,
                mutation=MutationOptions(idempotency_key="original-upload"),
            )
            found = client.forge.index_drafts.upload_get(PROJECT, DRAFT, OPERATION)
            assert first.package == again.package == found.package
    assert len(calls) == 3 and calls[0].content == calls[1].content


def test_upload_mcp_grants_and_author_only_contract():
    tool = BY_NAME["forge_index_draft_upload"]
    args = {
        "project_id": PROJECT,
        "draft_operation_id": DRAFT,
        "descriptor": DESCRIPTOR,
        "idempotency_key": "original-upload",
    }
    tool.validate_arguments(args)
    for field in ("rights_attested", "contribution_id", "provenance", "transfer_url"):
        with pytest.raises(ValueError):
            tool.validate_arguments({**args, "descriptor": {**DESCRIPTOR, field: "injected"}})
    with pytest.raises(ValueError):
        tool.validate_arguments(
            {
                **args,
                "descriptor": {
                    **DESCRIPTOR,
                    "assets": [{**DESCRIPTOR["assets"][0], "digest_sha256": "b" * 64}],
                },
            }
        )
    executor = SDKExecutor.__new__(SDKExecutor)
    executor._enabled = frozenset({tool.name})
    assert not executor._permitted(tool, SimpleNamespace(scopes=("forge:read",)), tool.scope)
    assert executor._permitted(
        tool, SimpleNamespace(scopes=("forge:read", "index:intake")), tool.scope
    )
    props = tool.output_schema()["properties"]
    assert "transfer_url" not in props and "transfer_headers" not in props


def test_upload_receipt_rejects_foreign_draft():
    transport, _ = fixture_transport(wrong_identity=True)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(ContractError, match="different private upload receipt"),
    ):
        client.forge.index_drafts.upload_get(PROJECT, DRAFT, OPERATION)
