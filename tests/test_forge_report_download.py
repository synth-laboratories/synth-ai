"""SDK + stdio MCP reader-scoped report artifact download (SYN-5158 gap 3)."""

import asyncio
import base64
import hashlib
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient
from synth_ai.errors import ContractError, NotFoundError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.sdk.forge.contracts.contracts import ExactReference
from synth_ai.sdk.forge.contracts.report_manifests import (
    ManifestArtifact,
    publish_initial,
    version_digest,
)
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_report_reader import SCOPE, b64, document
from tests.test_forge_report_shares import client_for
from tests.test_forge_reports import AT, PROJECT, versions
from tests.test_forge_workspaces import receipt_and_headers

CURVE = b'{"loss": [0.9, 0.5, 0.31]}'
CURVE_SHA = hashlib.sha256(CURVE).hexdigest()
FIRST, _ = versions()
PINNED = publish_initial(
    FIRST.manifest.model_copy(
        update={
            "artifacts": (
                ManifestArtifact(
                    artifact_id="curve",
                    object=ExactReference(
                        authority="artifact-platform",
                        kind="object",
                        record_id="pub-curve",
                        revision="1",
                        digest_sha256=CURVE_SHA,
                    ),
                    content_sha256=CURVE_SHA,
                    role="evidence",
                    media_type="application/json",
                    byte_count=len(CURVE),
                ),
            )
        }
    ),
    AT,
)
DIGEST = version_digest(PINNED)
OPERATIONS = ("forge.report_reader.open", "forge.report_reader.download")


def wire(*, body=CURVE, refuse=None):
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
        if request.url.path.endswith("/content"):
            if refuse is not None:
                refusal = {"error": {"code": refuse, "request_id": "4" * 32}}
                return httpx.Response(
                    404,
                    headers={**headers, "Content-Type": "application/json"},
                    stream=httpx.ByteStream(json.dumps(refusal).encode()),
                )
            return httpx.Response(
                200,
                headers={
                    **headers,
                    "Content-Type": "application/json",
                    "Content-Length": str(len(CURVE)),
                    "X-Synth-Content-SHA256": CURVE_SHA,
                    "X-Forge-Report-Version-SHA256": DIGEST,
                    "X-Forge-Report-Artifact-Id": "curve",
                    "X-Forge-Report-Access": "invited",
                },
                stream=httpx.ByteStream(body),
            )
        opened = document(PINNED)
        return httpx.Response(200, headers=headers, json={"metadata": metadata, "document": opened})

    return httpx.MockTransport(respond), calls


def test_download_writes_verified_bytes_and_receipt(tmp_path):
    transport, calls = wire()
    network, client = client_for(transport)
    destination = tmp_path / "curve.json"
    with network, client:
        receipt = client.forge.reports.download(PROJECT, DIGEST, "curve", destination)
    assert destination.read_bytes() == CURVE
    assert (receipt.digest_sha256, receipt.byte_size, receipt.access) == (
        CURVE_SHA,
        len(CURVE),
        "invited",
    )
    assert receipt.role == "evidence" and receipt.scope == SCOPE
    opened, content = calls
    assert opened.url.path.endswith(f"/report-reader/versions/{DIGEST}")
    assert content.url.path == (
        f"/api/v1/forge/projects/{PROJECT}/report-reader/versions/{DIGEST}/artifacts/curve/content"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["curve.json"]


def test_artifact_not_in_manifest_is_refused_before_any_byte_request(tmp_path):
    transport, calls = wire()
    network, client = client_for(transport)
    with network, client, pytest.raises(ContractError) as error:
        client.forge.reports.download(PROJECT, DIGEST, "other", tmp_path / "x")
    assert error.value.code == "not_in_manifest"
    assert len(calls) == 1 and not (tmp_path / "x").exists()


@pytest.mark.parametrize(
    ("body", "code"),
    [(CURVE[:-1], "report_artifact_truncated"), (CURVE.upper(), "digest_mismatch")],
    ids=["truncated", "tampered"],
)
def test_short_or_tampered_body_publishes_nothing(tmp_path, body, code):
    transport, _ = wire(body=body)
    network, client = client_for(transport)
    with network, client, pytest.raises(ContractError) as error:
        client.forge.reports.download(PROJECT, DIGEST, "curve", tmp_path / "curve")
    assert error.value.code == code
    assert list(tmp_path.iterdir()) == []


def test_typed_server_refusal_surfaces(tmp_path):
    transport, _ = wire(refuse="forge_share_expired")
    network, client = client_for(transport)
    with network, client, pytest.raises(NotFoundError) as error:
        client.forge.reports.download(PROJECT, DIGEST, "curve", tmp_path / "curve")
    assert error.value.code == "forge_share_expired"


def test_stdio_mcp_artifact_tool_returns_bounded_data():
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
            return await BY_NAME["forge_report_artifact"].invoke(
                client, {"project_id": PROJECT, "digest": DIGEST, "artifact_id": "curve"}
            )

    result = asyncio.run(run())
    artifact = result["artifact"]
    assert base64.b64decode(artifact["content_base64"]) == CURVE
    assert artifact["rendering"] == "plain_text" and artifact["index_contribution"] == "none"
    assert artifact["content_sha256"] == CURVE_SHA and artifact["access"] == "invited"
    assert len(calls) == 2
    tool = BY_NAME["forge_report_artifact"]
    assert (tool.scope, tool.mutates, tool.paid) == ("forge:read", False, False)
    assert b64(CURVE) == artifact["content_base64"]
