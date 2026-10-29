"""SDK and stdio MCP report reader: browse/open/compare/export (SYN-5158)."""

import asyncio
import base64
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.options import ClientOptions
from synth_ai.sdk.forge.contracts.contracts import Scope, canonical_bytes, contract_digest
from synth_ai.sdk.forge.contracts.report_manifests import (
    publish_next,
    version_bytes,
    version_digest,
)
from synth_ai.sdk.forge.report_reader import EXPORT_MANIFEST_PATH, export_files
from synth_ai.sdk.forge.types import (
    ReportExportBundle,
    ReportExportEntry,
    ReportExportFile,
    ReportExportManifest,
    compare_report_versions,
    report_export_path,
)
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_report_shares import client_for
from tests.test_forge_reports import AT, ORG, PROJECT, versions
from tests.test_forge_workspaces import receipt_and_headers

SCOPE = Scope(organization_id=ORG, project_id=PROJECT)
FIRST, _ = versions()
CORRECTION = publish_next(
    FIRST,
    disposition="correction",
    reason="Second measurement.",
    published_at=AT.replace(hour=14),
    manifest=FIRST.manifest.model_copy(
        update={"claims": ("Gain is 3%.",), "limitations": ("One seed.",)}
    ),
)
BASE, HEAD = version_digest(FIRST), version_digest(CORRECTION)
READER_TOOLS = (
    "forge_reports_browse",
    "forge_report_open",
    "forge_report_compare",
    "forge_report_export",
)
OPERATIONS = (
    "forge.report_reader.browse",
    "forge.report_reader.open",
    "forge.report_reader.compare",
    "forge.report_reader.export",
)


def b64(content: bytes) -> str:
    return base64.b64encode(content).decode()


def document(version, *, tamper=False):
    content = version_bytes(version)
    digest = version_digest(version)
    if tamper:
        content = content.replace(b"Measured once.", b"Measured twice")
    return {
        "scope": SCOPE.model_dump(mode="json"),
        "version_sha256": digest,
        "access": "invited",
        "size_bytes": len(content),
        "content_base64": b64(content),
        "version": version.model_dump(mode="json"),
    }


def bundle(versions_, *, tamper=False) -> ReportExportBundle:
    entries = tuple(
        ReportExportEntry(
            path=report_export_path(version_digest(item)),
            version_sha256=version_digest(item),
            lineage_id=item.lineage_id,
            version=item.version,
            disposition=item.disposition,
            size_bytes=len(version_bytes(item)),
            access="project",
        )
        for item in versions_
    )
    manifest = ReportExportManifest(
        scope=SCOPE,
        exported_by="user:reader",
        exported_at=AT,
        selection=tuple(entry.version_sha256 for entry in entries),
        entries=entries,
    )
    return ReportExportBundle(
        manifest=manifest,
        manifest_sha256="0" * 64 if tamper else contract_digest(manifest),
        files=tuple(
            ReportExportFile(
                path=entry.path,
                version_sha256=entry.version_sha256,
                content_base64=b64(version_bytes(item)),
            )
            for entry, item in zip(entries, versions_, strict=True)
        ),
    )


def wire(*, tamper=False):
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
        path = request.url.path
        if path.endswith("/report-reader/versions"):
            payload = {
                "versions": [
                    {
                        "scope": SCOPE.model_dump(mode="json"),
                        "version_sha256": digest_,
                        "access": access,
                        "publication_id": "pub-" + access,
                    }
                    for digest_, access in sorted(((BASE, "owner"), (HEAD, "organization")))
                ]
            }
        elif "/report-reader/compare/" in path:
            payload = {
                "comparison": compare_report_versions(
                    FIRST, BASE, "project", CORRECTION, HEAD, "invited"
                ).model_dump(mode="json")
            }
        elif path.endswith("/report-reader/export"):
            selected = request.url.params.get_list("digest")
            chosen = [{BASE: FIRST, HEAD: CORRECTION}[item] for item in selected]
            payload = {"export": bundle(chosen, tamper=tamper).model_dump(mode="json")}
        else:
            payload = {"document": document(FIRST, tamper=tamper)}
        return httpx.Response(200, headers=headers, json={"metadata": metadata, **payload})

    return httpx.MockTransport(respond), calls


def test_compare_reports_successor_and_structured_changes():
    comparison = compare_report_versions(FIRST, BASE, "project", CORRECTION, HEAD, "invited")
    assert comparison.relation == "successor" and comparison.manifest_changed
    assert comparison.claims.added == ("Gain is 3%.",)
    assert comparison.limitations.added == ("One seed.",)
    assert comparison.fields == () and comparison.artifacts == ()
    reverse = compare_report_versions(CORRECTION, HEAD, "invited", FIRST, BASE, "project")
    assert reverse.relation == "predecessor"
    assert reverse.claims.removed == ("Gain is 3%.",)
    same = compare_report_versions(FIRST, BASE, "project", FIRST, BASE, "project")
    assert same.relation == "identical" and not same.manifest_changed


def test_sync_browse_open_compare_export_round_trip():
    transport, calls = wire()
    network, client = client_for(transport)
    with network, client:
        page = client.forge.reports.browse(PROJECT, limit=5)
        assert [entry.access for entry in page.versions] == [
            {BASE: "owner", HEAD: "organization"}[item] for item in sorted((BASE, HEAD))
        ]
        opened = client.forge.reports.open(PROJECT, BASE).document
        assert opened.rendering == "plain_text" and opened.index_contribution == "none"
        assert opened.version.manifest.provenance.producer.subject_id == "user-1"
        compared = client.forge.reports.compare(PROJECT, BASE, HEAD).comparison
        assert compared.relation == "successor"
        exported = client.forge.reports.export(PROJECT, [HEAD, BASE]).export
    files = export_files(exported)
    assert set(files) == {
        EXPORT_MANIFEST_PATH,
        report_export_path(BASE),
        report_export_path(HEAD),
    }
    assert files[EXPORT_MANIFEST_PATH] == canonical_bytes(exported.manifest)
    assert files[report_export_path(HEAD)] == version_bytes(CORRECTION)
    browse, opened_call, compare, export = calls
    assert browse.url.path == f"/api/v1/forge/projects/{PROJECT}/report-reader/versions"
    assert browse.url.params["limit"] == "5"
    assert opened_call.url.path.endswith(f"/report-reader/versions/{BASE}")
    assert compare.url.path.endswith(f"/report-reader/compare/{BASE}/{HEAD}")
    assert export.url.params.get_list("digest") == [HEAD, BASE]
    assert {call.method for call in calls} == {"GET"}
    assert all("Idempotency-Key" not in call.headers for call in calls)


def test_tampered_version_bytes_or_export_manifest_are_refused():
    transport, _ = wire(tamper=True)
    network, client = client_for(transport)
    with network, client:
        with pytest.raises(ContractError):
            client.forge.reports.open(PROJECT, BASE)
        with pytest.raises(ContractError):
            client.forge.reports.export(PROJECT, [BASE])


def test_export_selection_is_explicit_and_bounded():
    transport, calls = wire()
    network, client = client_for(transport)
    with network, client:
        for selection in ([], [BASE, BASE], [BASE] * 33, BASE, ["not-a-digest"]):
            with pytest.raises(ValueError):
                client.forge.reports.export(PROJECT, selection)
    assert calls == []


def test_stdio_mcp_reader_tools_call_the_public_routes():
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
            base = {"project_id": PROJECT, "digest": BASE}
            return (
                await BY_NAME["forge_reports_browse"].invoke(client, {"project_id": PROJECT}),
                await BY_NAME["forge_report_open"].invoke(client, base),
                await BY_NAME["forge_report_compare"].invoke(
                    client, {**base, "other_digest": HEAD}
                ),
                await BY_NAME["forge_report_export"].invoke(
                    client, {"project_id": PROJECT, "digests": [BASE]}
                ),
            )

    browsed, opened, compared, exported = asyncio.run(run())
    assert len(browsed["versions"]) == 2
    assert opened["document"]["media_type"] == (
        "application/vnd.synth.forge.report-version.v1+json"
    )
    assert compared["comparison"]["claims"]["added"] == ["Gain is 3%."]
    assert exported["export"]["manifest"]["selection"] == [BASE]
    assert len(calls) == 4


def test_reader_tools_are_reads_without_authority_fields():
    for name in READER_TOOLS:
        tool = BY_NAME[name]
        assert (tool.scope, tool.mutates, tool.paid) == ("forge:read", False, False)
    base = {"project_id": PROJECT, "digest": BASE}
    for field in ("principal", "access", "authorization", "acting_credential", "audience"):
        with pytest.raises(ValueError):
            BY_NAME["forge_report_open"].validate_arguments({**base, field: "invented"})
    with pytest.raises(ValueError):
        BY_NAME["forge_report_export"].validate_arguments(
            {"project_id": PROJECT, "digests": [BASE] * 33}
        )
