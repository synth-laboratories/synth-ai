from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from synth_ai.core.research.contracts.traces import (
    TraceBundleDownload,
    TraceBundleDownloadObject,
    TraceBundleObjectKind,
    TraceCatalogProvider,
    TraceDownload,
    TraceStoreAccessReceipt,
    TraceStoreDescriptor,
    TraceStoreLifecycleReceipt,
)
from synth_ai.core.research.operations import RESEARCH_OPERATIONS
from synth_ai.core.research.traces import (
    FactoryTraceStoreAPI,
    _materialize_download,
    _transfer_timeout_seconds,
)


TRACE_OPERATION_IDS = {
    "get_factory_trace_store",
    "provision_factory_trace_store",
    "rotate_factory_trace_store_credential",
    "invalidate_factory_trace_store_credential",
    "tombstone_factory_trace_store",
    "rebuild_factory_trace_store",
    "prepare_factory_trace_bundle",
    "finalize_factory_trace_bundle",
    "list_factory_traces",
    "create_factory_trace_download",
    "download_factory_trace_bundle",
}


def _access_receipt(action: str) -> TraceStoreAccessReceipt:
    return TraceStoreAccessReceipt(
        action=action,
        store_id="store-a",
        factory_id="factory-a",
        org_id="org-a",
        principal_id="api-key:key-a",
        request_digest="sha256:" + "1" * 64,
        result_digest="sha256:" + "2" * 64,
        occurred_at=datetime.now(UTC),
        content_digest="sha256:" + "3" * 64,
        receipt_uri="s3://traces/receipts/access.json",
        details={},
    )


def test_trace_operation_registry_has_backend_parity() -> None:
    assert TRACE_OPERATION_IDS <= {str(item) for item in RESEARCH_OPERATIONS}
    assert (
        RESEARCH_OPERATIONS["download_factory_trace_bundle"].path_template
        == "/smr/factories/{factory_id}/trace-bundles/{publication_id}:download"
    )
    assert RESEARCH_OPERATIONS["prepare_factory_trace_bundle"].idempotent is True
    assert RESEARCH_OPERATIONS["finalize_factory_trace_bundle"].idempotent is True
    for operation_id in (
        "provision_factory_trace_store",
        "rotate_factory_trace_store_credential",
        "invalidate_factory_trace_store_credential",
        "tombstone_factory_trace_store",
        "rebuild_factory_trace_store",
    ):
        assert RESEARCH_OPERATIONS[operation_id].idempotent is False


def test_trace_transfer_timeout_is_explicit_and_bounded() -> None:
    assert _transfer_timeout_seconds(600) == 600.0
    for invalid in (True, 0, 7201):
        with pytest.raises(ValueError, match="between 1 and 7200"):
            _transfer_timeout_seconds(invalid)  # type: ignore[arg-type]


def test_trace_descriptor_is_strict_and_keeps_no_credentials() -> None:
    payload = {
        "store_id": "store-a",
        "org_id": "org-a",
        "factory_id": "factory-a",
        "store_schema_version": "synth.trace-store.v1",
        "bundle_schema_version": "synth.trace-bundle.v1",
        "blob_provider": "s3",
        "blob_bucket": "traces",
        "blob_prefix": "factories/factory-a",
        "blob_region": None,
        "blob_endpoint_profile": None,
        "catalog_provider": "turso",
        "catalog_database_name": "factory-a",
        "catalog_group_name": "trace",
        "catalog_region": "iad",
        "provisioning_status": "ready",
        "provisioning_version": 1,
        "migration_generation": 1,
        "retention_policy": {},
        "visibility_policy": {},
        "encryption_policy": {},
        "created_at": "2026-07-25T00:00:00Z",
        "updated_at": "2026-07-25T00:00:00Z",
    }
    descriptor = TraceStoreDescriptor.from_wire(payload)
    assert descriptor.catalog_provider is TraceCatalogProvider.TURSO
    with pytest.raises(ValidationError):
        TraceStoreDescriptor.from_wire({**payload, "database_token": "secret"})


def test_trace_response_digests_are_strictly_typed() -> None:
    with pytest.raises(ValidationError):
        TraceBundleDownloadObject(
            path="traces/a.json",
            digest="not-a-digest",
            size_bytes=1,
            media_type="application/json",
            kind=TraceBundleObjectKind.TRACE,
            download_url="https://s3.example/object",
            expires_at=datetime.now(UTC),
        )
    with pytest.raises(ValidationError):
        TraceStoreAccessReceipt(
            action="query",
            store_id="store-a",
            factory_id="factory-a",
            org_id="org-a",
            request_digest="not-a-digest",
            result_digest="sha256:" + "2" * 64,
            occurred_at=datetime.now(UTC),
            content_digest="sha256:" + "3" * 64,
            receipt_uri="s3://traces/receipts/access.json",
        )
    with pytest.raises(ValidationError):
        TraceDownload(
            trace_digest="sha256:" + "1" * 64,
            bytes_digest="not-a-digest",
            size_bytes=1,
            media_type="application/json",
            s3_uri="s3://traces/blobs/trace.json",
            download_url="https://s3.example/object",
            expires_at=datetime.now(UTC),
            receipt=_access_receipt("download_trace"),
        )


def test_lifecycle_receipt_carries_redacted_principal_and_org_authority() -> None:
    receipt = TraceStoreLifecycleReceipt(
        action="rotate_catalog_credential",
        store_id="store-a",
        factory_id="factory-a",
        org_id="org-a",
        principal_id="api-key:key-a",
        generation=2,
        occurred_at=datetime.now(UTC),
        content_digest="sha256:" + "4" * 64,
        receipt_uri="s3://traces/receipts/lifecycle.json",
    )

    assert receipt.principal_id == "api-key:key-a"
    assert "token" not in receipt.to_wire()


class RecordingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[Any] = []

    def execute(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self.response


def test_factory_trace_query_sends_all_typed_filters() -> None:
    transport = RecordingTransport(
        {
            "factory_id": "factory-a",
            "count": 0,
            "traces": [],
            "receipt": _access_receipt("query").to_wire(),
        }
    )
    api = FactoryTraceStoreAPI(transport, "factory-a")  # type: ignore[arg-type]

    result = api.query(
        project_id="project-a",
        effort_id="effort-a",
        run_id="run-a",
        correlation_id="correlation-a",
        actor_id="actor-a",
        session_id="session-a",
        criterion_id="criterion-a",
        annotation_label="needs-review",
        reward_id="reward-a",
        reward_min=0.25,
        reward_max=0.75,
        workflow_address="map/child/0",
        limit=17,
    )

    assert result.count == 0
    request = transport.requests[0]
    assert request.operation.operation_id == "list_factory_traces"
    assert request.query == {
        "project_id": "project-a",
        "effort_id": "effort-a",
        "run_id": "run-a",
        "correlation_id": "correlation-a",
        "actor_id": "actor-a",
        "session_id": "session-a",
        "criterion_id": "criterion-a",
        "annotation_label": "needs-review",
        "reward_id": "reward-a",
        "reward_min": 0.25,
        "reward_max": 0.75,
        "workflow_address": "map/child/0",
        "limit": 17,
    }


def test_exact_cloud_bundle_materialization_preserves_object_bytes(
    tmp_path: Path,
) -> None:
    LocalTraceBundle = pytest.importorskip(
        "synth_containers.tracing.store.bundle"
    ).LocalTraceBundle
    source = LocalTraceBundle(tmp_path / "source")
    receipt = source.write_receipt("acceptance", {"ok": True})
    manifest = source.write_manifest()
    item = manifest.objects[0]
    payload = receipt.read_bytes()
    descriptor = TraceBundleDownload(
        publication_id="publication-a",
        bundle_id=manifest.bundle_id,
        manifest_digest=manifest.content_digest,
        manifest=manifest.to_dict(),
        objects=[
            TraceBundleDownloadObject(
                path=item.path,
                digest=item.bytes_digest,
                size_bytes=item.byte_size,
                media_type=item.media_type,
                kind=TraceBundleObjectKind(item.kind),
                download_url="https://s3.example/object",
                expires_at=datetime.now(UTC),
            )
        ],
        receipt=_access_receipt("download_bundle"),
    )

    target = _materialize_download(
        descriptor,
        tmp_path / "downloaded",
        {item.path: payload},
    )

    restored = LocalTraceBundle(target)
    assert restored.verify_self_contained() == (True, ())
    assert (target / item.path).read_bytes() == payload
    assert restored.read_manifest()["content_digest"] == manifest.content_digest
