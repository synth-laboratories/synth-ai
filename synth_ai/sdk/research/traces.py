"""Factory-scoped managed Trace V5 storage and exact bundle transfer."""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.traces import (
    TraceBundleDownload,
    TraceBundleObjectDeclaration,
    TraceBundlePrepareRequest,
    TraceBundlePublication,
    TraceCatalogProvider,
    TraceDownload,
    TracePromotionReceipt,
    TraceQueryResult,
    TraceStoreDescriptor,
    TraceStoreLifecycleReceipt,
    TraceStorePreflightRequest,
    TraceStorePreflightResponse,
    TraceStoreProvisionResult,
)
from synth_ai.sdk.research.operations import research_operation

DEFAULT_TRACE_TRANSFER_TIMEOUT_SECONDS = 600.0


def _transfer_timeout_seconds(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 1 <= float(value) <= 7200
    ):
        raise ValueError("transfer_timeout_seconds must be between 1 and 7200")
    return float(value)


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
    headers: dict[str, str] | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
        headers=headers or {},
    )


def _query(**values: JsonValue) -> JsonObject:
    return {name: value for name, value in values.items() if value is not None}


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _containers_bundle_api() -> tuple[Any, Any, Any]:
    try:
        interchange = importlib.import_module("synth_containers.tracing.interchange")
        bundle_module = importlib.import_module("synth_containers.tracing.store.bundle")
        canonical = importlib.import_module("synth_containers.tracing.canonical")
    except ImportError as error:
        raise RuntimeError(
            "Local Trace V5 bundle transfer is owned by synth-containers; "
            "install a compatible synth-containers release explicitly"
        ) from error
    return (
        interchange.load_bundle_manifest,
        bundle_module.LocalTraceBundle,
        canonical.canonical_bytes,
    )


def _bundle_prepare_request(
    root: Path,
    *,
    project_id: str | None,
    run_id: str | None,
    effort_id: str | None,
    metadata: dict[str, Any] | None,
) -> tuple[TraceBundlePrepareRequest, dict[str, Path]]:
    load_manifest, local_bundle_type, canonical_bytes = _containers_bundle_api()
    bundle = local_bundle_type(root)
    ok, errors = bundle.verify_self_contained()
    if not ok:
        raise ValueError(f"Trace V5 bundle is not self-contained: {errors!r}")
    manifest = bundle.read_manifest()
    loaded = load_manifest(canonical_bytes(manifest))
    manifest_payload = loaded.to_dict()
    declarations: list[TraceBundleObjectDeclaration] = []
    object_paths: dict[str, Path] = {}
    for item in loaded.objects:
        declaration = TraceBundleObjectDeclaration(
            digest=item.bytes_digest,
            path=item.path,
            size_bytes=item.byte_size,
            media_type=item.media_type,
            kind=item.kind,
        )
        declarations.append(declaration)
        object_paths.setdefault(item.bytes_digest, root / item.path)
    if not declarations:
        raise ValueError("Trace V5 cloud promotion requires manifest.objects")
    request = TraceBundlePrepareRequest(
        bundle_id=loaded.bundle_id,
        manifest_digest=loaded.content_digest,
        manifest=manifest_payload,
        objects=declarations,
        project_id=project_id,
        run_id=run_id,
        effort_id=effort_id,
        metadata=dict(metadata or {}),
    )
    return request, object_paths


def _safe_target(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe Trace V5 bundle object path {relative_path!r}")
    target = root.joinpath(*path.parts)
    if not target.is_relative_to(root):
        raise ValueError(f"Trace V5 bundle object escapes target root: {relative_path!r}")
    return target


def _write_exact(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"Trace V5 target contains different bytes: {target}")
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def _materialize_download(
    descriptor: TraceBundleDownload,
    target: Path,
    object_bodies: dict[str, bytes],
) -> Path:
    load_manifest, local_bundle_type, canonical_bytes = _containers_bundle_api()
    manifest_bytes = canonical_bytes(descriptor.manifest)
    manifest = load_manifest(manifest_bytes)
    if manifest.content_digest != descriptor.manifest_digest:
        raise ValueError(
            "downloaded Trace V5 manifest digest differs from the publication descriptor"
        )
    if target.exists():
        raise FileExistsError(f"Trace V5 bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.",
        dir=target.parent,
    ) as staging_name:
        staging = Path(staging_name)
        _write_exact(staging / "manifest.json", manifest_bytes)
        for item in descriptor.objects:
            payload = object_bodies[item.path]
            actual = _sha256(payload)
            if actual != item.digest:
                raise ValueError(
                    f"Trace V5 object {item.path!r} digest mismatch: "
                    f"expected {item.digest}, found {actual}"
                )
            if len(payload) != item.size_bytes:
                raise ValueError(
                    f"Trace V5 object {item.path!r} size mismatch: "
                    f"expected {item.size_bytes}, found {len(payload)}"
                )
            _write_exact(_safe_target(staging, item.path), payload)
        bundle = local_bundle_type(staging)
        ok, errors = bundle.verify_self_contained()
        if not ok:
            raise ValueError(f"downloaded Trace V5 bundle failed verification: {errors!r}")
        os.replace(staging, target)
    return target


class FactoryTraceStoreAPI:
    """Managed Trace V5 operations bound to one Factory."""

    def __init__(self, transport: HttpTransport, factory_id: str) -> None:
        self._transport = transport
        self.factory_id = factory_id

    @property
    def _base(self) -> str:
        return f"/smr/factories/{self.factory_id}"

    def descriptor(self) -> TraceStoreDescriptor:
        value = self._transport.execute(
            _request("get_factory_trace_store", f"{self._base}/trace-store")
        )
        return TraceStoreDescriptor.from_wire(value)

    def provision(
        self,
        *,
        catalog_provider: TraceCatalogProvider = TraceCatalogProvider.NONE,
    ) -> TraceStoreProvisionResult:
        value = self._transport.execute(
            _request(
                "provision_factory_trace_store",
                f"{self._base}/trace-store:provision",
                body={"catalog_provider": catalog_provider.value},
            )
        )
        return TraceStoreProvisionResult.from_wire(value)

    def preflight(
        self,
        request: TraceStorePreflightRequest | None = None,
    ) -> TraceStorePreflightResponse:
        """Provision-if-missing and health-check the trace store before launch.

        Sealed runners call this before creating paid/live resources and embed
        the returned ``run_envelope_identity`` in the run envelope so
        post-grade trace inventory can never 404 on an unprovisioned store.
        """
        value = self._transport.execute(
            _request(
                "preflight_factory_trace_store",
                f"{self._base}/trace-store:preflight",
                body=(request or TraceStorePreflightRequest()).to_wire(),
            )
        )
        response = TraceStorePreflightResponse.from_wire(value)
        if (
            response.descriptor.factory_id != self.factory_id
            or response.run_envelope_identity.factory_id != self.factory_id
        ):
            raise ValueError("trace store preflight identity drifted")
        return response

    def rotate_credential(self) -> TraceStoreLifecycleReceipt:
        return self._lifecycle(
            "rotate_factory_trace_store_credential",
            "rotate-credential",
        )

    def invalidate_credential(self) -> TraceStoreLifecycleReceipt:
        return self._lifecycle(
            "invalidate_factory_trace_store_credential",
            "invalidate-credential",
        )

    def tombstone(self) -> TraceStoreLifecycleReceipt:
        return self._lifecycle("tombstone_factory_trace_store", "tombstone")

    def rebuild(self) -> TraceStoreLifecycleReceipt:
        return self._lifecycle("rebuild_factory_trace_store", "rebuild")

    def _lifecycle(self, operation_id: str, action: str) -> TraceStoreLifecycleReceipt:
        value = self._transport.execute(
            _request(operation_id, f"{self._base}/trace-store:{action}")
        )
        return TraceStoreLifecycleReceipt.from_wire(value)

    def prepare(self, request: TraceBundlePrepareRequest) -> TraceBundlePublication:
        value = self._transport.execute(
            _request(
                "prepare_factory_trace_bundle",
                f"{self._base}/trace-bundles:prepare-upload",
                body=request.to_wire(),
                headers={"Idempotency-Key": (f"trace-bundle-prepare:{request.manifest_digest}")},
            )
        )
        return TraceBundlePublication.from_wire(value)

    def finalize(self, publication_id: str) -> TracePromotionReceipt:
        value = self._transport.execute(
            _request(
                "finalize_factory_trace_bundle",
                f"{self._base}/trace-bundles:finalize",
                body={"publication_id": publication_id},
                headers={"Idempotency-Key": f"trace-bundle-finalize:{publication_id}"},
            )
        )
        return TracePromotionReceipt.from_wire(value)

    def upload_bundle(
        self,
        root: str | Path,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        effort_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        transfer_timeout_seconds: float = DEFAULT_TRACE_TRANSFER_TIMEOUT_SECONDS,
    ) -> TracePromotionReceipt:
        transfer_timeout = _transfer_timeout_seconds(transfer_timeout_seconds)
        bundle_root = Path(root)
        request, object_paths = _bundle_prepare_request(
            bundle_root,
            project_id=project_id,
            run_id=run_id,
            effort_id=effort_id,
            metadata=metadata,
        )
        publication = self.prepare(request)
        with httpx.Client(
            timeout=transfer_timeout,
            follow_redirects=False,
        ) as client:
            uploaded: set[str] = set()
            for upload in publication.upload_objects:
                if upload.already_present or upload.digest in uploaded:
                    continue
                if upload.upload_url is None:
                    raise ValueError(
                        f"backend omitted upload URL for missing object {upload.digest}"
                    )
                payload = object_paths[upload.digest].read_bytes()
                if _sha256(payload) != upload.digest:
                    raise ValueError(f"local Trace V5 object changed: {upload.digest}")
                upload_error: httpx.TransportError | None = None
                for _attempt in range(2):
                    try:
                        response = client.put(
                            upload.upload_url,
                            content=payload,
                            headers=upload.required_headers,
                        )
                        upload_error = None
                        break
                    except httpx.TransportError as error:
                        upload_error = error
                else:
                    raise RuntimeError(
                        f"Trace V5 immutable object upload outcome is uncertain for {upload.digest}"
                    ) from upload_error
                if response.status_code not in {409, 412}:
                    response.raise_for_status()
                uploaded.add(upload.digest)
        return self.finalize(publication.publication_id)

    def query(
        self,
        *,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
        trace_kind: str | None = None,
        task_id: str | None = None,
        actor_id: str | None = None,
        session_id: str | None = None,
        criterion_id: str | None = None,
        annotation_label: str | None = None,
        reward_id: str | None = None,
        reward_min: float | None = None,
        reward_max: float | None = None,
        workflow_address: str | None = None,
        limit: int = 100,
    ) -> TraceQueryResult:
        value = self._transport.execute(
            _request(
                "list_factory_traces",
                f"{self._base}/traces",
                query=_query(
                    project_id=project_id,
                    effort_id=effort_id,
                    run_id=run_id,
                    correlation_id=correlation_id,
                    trace_kind=trace_kind,
                    task_id=task_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    criterion_id=criterion_id,
                    annotation_label=annotation_label,
                    reward_id=reward_id,
                    reward_min=reward_min,
                    reward_max=reward_max,
                    workflow_address=workflow_address,
                    limit=limit,
                ),
            )
        )
        return TraceQueryResult.from_wire(value)

    def trace_download(
        self,
        trace_digest: str,
        *,
        expires_in_seconds: int = 3600,
    ) -> TraceDownload:
        value = self._transport.execute(
            _request(
                "create_factory_trace_download",
                f"{self._base}/traces/{trace_digest}:download-url",
                query={"expires_in_seconds": expires_in_seconds},
            )
        )
        return TraceDownload.from_wire(value)

    def bundle_download(
        self,
        publication_id: str,
        *,
        expires_in_seconds: int = 3600,
    ) -> TraceBundleDownload:
        value = self._transport.execute(
            _request(
                "download_factory_trace_bundle",
                f"{self._base}/trace-bundles/{publication_id}:download",
                query={"expires_in_seconds": expires_in_seconds},
            )
        )
        return TraceBundleDownload.from_wire(value)

    def download_bundle(
        self,
        publication_id: str,
        target: str | Path,
        *,
        expires_in_seconds: int = 3600,
        transfer_timeout_seconds: float = DEFAULT_TRACE_TRANSFER_TIMEOUT_SECONDS,
    ) -> Path:
        transfer_timeout = _transfer_timeout_seconds(transfer_timeout_seconds)
        descriptor = self.bundle_download(
            publication_id,
            expires_in_seconds=expires_in_seconds,
        )
        destination = Path(target)
        if destination.exists():
            raise FileExistsError(f"Trace V5 bundle target already exists: {destination}")
        load_manifest, local_bundle_type, canonical_bytes = _containers_bundle_api()
        manifest_bytes = canonical_bytes(descriptor.manifest)
        manifest = load_manifest(manifest_bytes)
        if manifest.content_digest != descriptor.manifest_digest:
            raise ValueError("downloaded Trace V5 manifest digest differs from publication")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            httpx.Client(
                timeout=transfer_timeout,
                follow_redirects=True,
            ) as client,
            tempfile.TemporaryDirectory(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            ) as staging_name,
        ):
            staging = Path(staging_name)
            _write_exact(staging / "manifest.json", manifest_bytes)
            for item in descriptor.objects:
                output = _safe_target(staging, item.path)
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size_bytes = 0
                with client.stream("GET", item.download_url) as response:
                    response.raise_for_status()
                    with output.open("xb") as sink:
                        for chunk in response.iter_bytes():
                            digest.update(chunk)
                            size_bytes += len(chunk)
                            sink.write(chunk)
                actual = f"sha256:{digest.hexdigest()}"
                if actual != item.digest or size_bytes != item.size_bytes:
                    raise ValueError(
                        f"Trace V5 object {item.path!r} failed streamed integrity check"
                    )
            bundle = local_bundle_type(staging)
            ok, errors = bundle.verify_self_contained()
            if not ok:
                raise ValueError(f"downloaded Trace V5 bundle failed verification: {errors!r}")
            os.replace(staging, destination)
        return destination


class ResearchTracesAPI:
    """Factory-scoped Trace V5 storage namespace."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def for_factory(self, factory_id: str) -> FactoryTraceStoreAPI:
        return FactoryTraceStoreAPI(self._transport, factory_id)


class AsyncFactoryTraceStoreAPI:
    """Native asynchronous managed Trace V5 operations bound to one Factory."""

    def __init__(self, transport: AsyncHttpTransport, factory_id: str) -> None:
        self._transport = transport
        self.factory_id = factory_id

    @property
    def _base(self) -> str:
        return f"/smr/factories/{self.factory_id}"

    async def descriptor(self) -> TraceStoreDescriptor:
        value = await self._transport.execute(
            _request("get_factory_trace_store", f"{self._base}/trace-store")
        )
        return TraceStoreDescriptor.from_wire(value)

    async def provision(
        self,
        *,
        catalog_provider: TraceCatalogProvider = TraceCatalogProvider.NONE,
    ) -> TraceStoreProvisionResult:
        value = await self._transport.execute(
            _request(
                "provision_factory_trace_store",
                f"{self._base}/trace-store:provision",
                body={"catalog_provider": catalog_provider.value},
            )
        )
        return TraceStoreProvisionResult.from_wire(value)

    async def preflight(
        self,
        request: TraceStorePreflightRequest | None = None,
    ) -> TraceStorePreflightResponse:
        """Provision-if-missing and health-check the trace store before launch."""
        value = await self._transport.execute(
            _request(
                "preflight_factory_trace_store",
                f"{self._base}/trace-store:preflight",
                body=(request or TraceStorePreflightRequest()).to_wire(),
            )
        )
        response = TraceStorePreflightResponse.from_wire(value)
        if (
            response.descriptor.factory_id != self.factory_id
            or response.run_envelope_identity.factory_id != self.factory_id
        ):
            raise ValueError("trace store preflight identity drifted")
        return response

    async def rotate_credential(self) -> TraceStoreLifecycleReceipt:
        return await self._lifecycle(
            "rotate_factory_trace_store_credential",
            "rotate-credential",
        )

    async def invalidate_credential(self) -> TraceStoreLifecycleReceipt:
        return await self._lifecycle(
            "invalidate_factory_trace_store_credential",
            "invalidate-credential",
        )

    async def tombstone(self) -> TraceStoreLifecycleReceipt:
        return await self._lifecycle("tombstone_factory_trace_store", "tombstone")

    async def rebuild(self) -> TraceStoreLifecycleReceipt:
        return await self._lifecycle("rebuild_factory_trace_store", "rebuild")

    async def _lifecycle(
        self,
        operation_id: str,
        action: str,
    ) -> TraceStoreLifecycleReceipt:
        value = await self._transport.execute(
            _request(operation_id, f"{self._base}/trace-store:{action}")
        )
        return TraceStoreLifecycleReceipt.from_wire(value)

    async def prepare(self, request: TraceBundlePrepareRequest) -> TraceBundlePublication:
        value = await self._transport.execute(
            _request(
                "prepare_factory_trace_bundle",
                f"{self._base}/trace-bundles:prepare-upload",
                body=request.to_wire(),
                headers={"Idempotency-Key": (f"trace-bundle-prepare:{request.manifest_digest}")},
            )
        )
        return TraceBundlePublication.from_wire(value)

    async def finalize(self, publication_id: str) -> TracePromotionReceipt:
        value = await self._transport.execute(
            _request(
                "finalize_factory_trace_bundle",
                f"{self._base}/trace-bundles:finalize",
                body={"publication_id": publication_id},
                headers={"Idempotency-Key": f"trace-bundle-finalize:{publication_id}"},
            )
        )
        return TracePromotionReceipt.from_wire(value)

    async def upload_bundle(
        self,
        root: str | Path,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        effort_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        transfer_timeout_seconds: float = DEFAULT_TRACE_TRANSFER_TIMEOUT_SECONDS,
    ) -> TracePromotionReceipt:
        transfer_timeout = _transfer_timeout_seconds(transfer_timeout_seconds)
        request, object_paths = _bundle_prepare_request(
            Path(root),
            project_id=project_id,
            run_id=run_id,
            effort_id=effort_id,
            metadata=metadata,
        )
        publication = await self.prepare(request)
        async with httpx.AsyncClient(
            timeout=transfer_timeout,
            follow_redirects=False,
        ) as client:
            uploaded: set[str] = set()
            for upload in publication.upload_objects:
                if upload.already_present or upload.digest in uploaded:
                    continue
                if upload.upload_url is None:
                    raise ValueError(
                        f"backend omitted upload URL for missing object {upload.digest}"
                    )
                payload = object_paths[upload.digest].read_bytes()
                if _sha256(payload) != upload.digest:
                    raise ValueError(f"local Trace V5 object changed: {upload.digest}")
                upload_error: httpx.TransportError | None = None
                for _attempt in range(2):
                    try:
                        response = await client.put(
                            upload.upload_url,
                            content=payload,
                            headers=upload.required_headers,
                        )
                        upload_error = None
                        break
                    except httpx.TransportError as error:
                        upload_error = error
                else:
                    raise RuntimeError(
                        f"Trace V5 immutable object upload outcome is uncertain for {upload.digest}"
                    ) from upload_error
                if response.status_code not in {409, 412}:
                    response.raise_for_status()
                uploaded.add(upload.digest)
        return await self.finalize(publication.publication_id)

    async def query(
        self,
        *,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
        trace_kind: str | None = None,
        task_id: str | None = None,
        actor_id: str | None = None,
        session_id: str | None = None,
        criterion_id: str | None = None,
        annotation_label: str | None = None,
        reward_id: str | None = None,
        reward_min: float | None = None,
        reward_max: float | None = None,
        workflow_address: str | None = None,
        limit: int = 100,
    ) -> TraceQueryResult:
        value = await self._transport.execute(
            _request(
                "list_factory_traces",
                f"{self._base}/traces",
                query=_query(
                    project_id=project_id,
                    effort_id=effort_id,
                    run_id=run_id,
                    correlation_id=correlation_id,
                    trace_kind=trace_kind,
                    task_id=task_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    criterion_id=criterion_id,
                    annotation_label=annotation_label,
                    reward_id=reward_id,
                    reward_min=reward_min,
                    reward_max=reward_max,
                    workflow_address=workflow_address,
                    limit=limit,
                ),
            )
        )
        return TraceQueryResult.from_wire(value)

    async def trace_download(
        self,
        trace_digest: str,
        *,
        expires_in_seconds: int = 3600,
    ) -> TraceDownload:
        value = await self._transport.execute(
            _request(
                "create_factory_trace_download",
                f"{self._base}/traces/{trace_digest}:download-url",
                query={"expires_in_seconds": expires_in_seconds},
            )
        )
        return TraceDownload.from_wire(value)

    async def bundle_download(
        self,
        publication_id: str,
        *,
        expires_in_seconds: int = 3600,
    ) -> TraceBundleDownload:
        value = await self._transport.execute(
            _request(
                "download_factory_trace_bundle",
                f"{self._base}/trace-bundles/{publication_id}:download",
                query={"expires_in_seconds": expires_in_seconds},
            )
        )
        return TraceBundleDownload.from_wire(value)

    async def download_bundle(
        self,
        publication_id: str,
        target: str | Path,
        *,
        expires_in_seconds: int = 3600,
        transfer_timeout_seconds: float = DEFAULT_TRACE_TRANSFER_TIMEOUT_SECONDS,
    ) -> Path:
        transfer_timeout = _transfer_timeout_seconds(transfer_timeout_seconds)
        descriptor = await self.bundle_download(
            publication_id,
            expires_in_seconds=expires_in_seconds,
        )
        destination = Path(target)
        if destination.exists():
            raise FileExistsError(f"Trace V5 bundle target already exists: {destination}")
        load_manifest, local_bundle_type, canonical_bytes = _containers_bundle_api()
        manifest_bytes = canonical_bytes(descriptor.manifest)
        manifest = load_manifest(manifest_bytes)
        if manifest.content_digest != descriptor.manifest_digest:
            raise ValueError("downloaded Trace V5 manifest digest differs from publication")
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(
            timeout=transfer_timeout,
            follow_redirects=True,
        ) as client:
            with tempfile.TemporaryDirectory(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            ) as staging_name:
                staging = Path(staging_name)
                _write_exact(staging / "manifest.json", manifest_bytes)
                for item in descriptor.objects:
                    output = _safe_target(staging, item.path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size_bytes = 0
                    async with client.stream("GET", item.download_url) as response:
                        response.raise_for_status()
                        with output.open("xb") as sink:
                            async for chunk in response.aiter_bytes():
                                digest.update(chunk)
                                size_bytes += len(chunk)
                                sink.write(chunk)
                    actual = f"sha256:{digest.hexdigest()}"
                    if actual != item.digest or size_bytes != item.size_bytes:
                        raise ValueError(
                            f"Trace V5 object {item.path!r} failed streamed integrity check"
                        )
                bundle = local_bundle_type(staging)
                ok, errors = bundle.verify_self_contained()
                if not ok:
                    raise ValueError(f"downloaded Trace V5 bundle failed verification: {errors!r}")
                os.replace(staging, destination)
        return destination


class AsyncResearchTracesAPI:
    """Native asynchronous Factory-scoped Trace V5 storage namespace."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    def for_factory(self, factory_id: str) -> AsyncFactoryTraceStoreAPI:
        return AsyncFactoryTraceStoreAPI(self._transport, factory_id)


__all__ = [
    "AsyncFactoryTraceStoreAPI",
    "AsyncResearchTracesAPI",
    "FactoryTraceStoreAPI",
    "ResearchTracesAPI",
]
