"""Exact native resource reads with typed sync/async parity.

# See: testing-resources/specifications/sdk/forge_resource_reads.md
Only authenticated backend routes are called; native custody remains in SMR.
"""

from __future__ import annotations

import hashlib
import json
from typing import cast
from urllib.parse import quote

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.dataset_revisions import (
    DatasetRevisionContent,
    SealedDatasetRevision,
)
from synth_ai.sdk.research.contracts.resource_reads import (
    DataPoolReadReference,
    DatasetRevisionReadReference,
    ProjectDataPoolDescriptor,
    ProjectDataPoolInventory,
)
from synth_ai.sdk.research.contracts.traces import validate_sha256_digest

RESOURCE_READ_BYTES_MAX = 64 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _project_root(project_id: str) -> str:
    if not project_id:
        raise ValueError("project_id is required")
    return "/smr/projects/" + quote(project_id, safe="")


def _revision_root(reference: DatasetRevisionReadReference) -> str:
    return (
        _project_root(reference.project_id)
        + f"/data-bindings/{reference.data_binding_id}/revisions/{reference.dataset_revision_id}"
    )


def _revision(value: object, reference: DatasetRevisionReadReference) -> SealedDatasetRevision:
    """Verify requested identity and native canonical digests at the read boundary.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    This is a consumer integrity check, not current authorization or receipt minting.
    """
    revision = SealedDatasetRevision.model_validate(value)
    if (
        revision.org_id != reference.org_id
        or revision.project_id != reference.project_id
        or revision.binding_id != reference.data_binding_id
        or revision.dataset_revision_id != reference.dataset_revision_id
        or revision.revision_digest != reference.revision_digest
    ):
        raise ValueError("DatasetRevision response differs from requested exact scope/identity")
    paths = [item.logical_path for item in revision.contents]
    if len(paths) != len(set(paths)):
        raise ValueError("DatasetRevision repeats a logical object path")
    contents = [
        item.model_dump(mode="json")
        for item in sorted(revision.contents, key=lambda item: item.logical_path)
    ]
    payload = revision.model_dump(
        mode="json", exclude={"revision_digest", "sealed_at", "sealed_by_principal_id"}
    )
    payload["contents"] = contents
    payload["lineage"]["sources"] = sorted(
        payload["lineage"]["sources"],
        key=lambda item: (item["kind"], item["authority_id"], item["authority_version"]),
    )
    if (
        revision.content_digest != _digest(_canonical_bytes(contents))
        or revision.manifest_digest != _digest(_manifest(revision))
        or revision.revision_digest != _digest(_canonical_bytes(payload))
    ):
        raise ValueError("DatasetRevision canonical declarations differ from retained digests")
    return revision


def _manifest(revision: SealedDatasetRevision) -> bytes:
    """Encode the native canonical manifest solely for consumer verification.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    """
    return _canonical_bytes(
        {
            "dataset_schema_version": revision.dataset_schema_version,
            "contents": [
                item.model_dump(mode="json")
                for item in sorted(revision.contents, key=lambda item: item.logical_path)
            ],
        }
    )


def _bounded(size_bytes: int, max_bytes: int) -> None:
    if type(max_bytes) is not int or not 0 <= max_bytes <= RESOURCE_READ_BYTES_MAX:
        raise ValueError("max_bytes must be between zero and the native 64 MiB delivery bound")
    if size_bytes > max_bytes:
        raise ValueError("Selected resource exceeds the requested byte delivery bound")


def _verified(content: bytes, *, size_bytes: int, content_digest: str, max_bytes: int) -> bytes:
    """Bind exact delivered bytes before callers or MCP receive them.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    """
    _bounded(size_bytes, max_bytes)
    if len(content) != size_bytes or _digest(content) != content_digest:
        raise ValueError("Resource bytes differ from the requested native declaration")
    return content


def _declared_object(revision: SealedDatasetRevision, logical_path: str) -> DatasetRevisionContent:
    """Select a retained declaration without turning caller paths into locators.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    """
    for item in revision.contents:
        if item.logical_path == logical_path:
            return item
    raise ValueError("Logical path is absent from the exact DatasetRevision")


def _pools(value: object) -> tuple[ProjectDataPoolDescriptor, ...]:
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("Native data-pool descriptors must be a bounded array")
    descriptors = tuple(ProjectDataPoolDescriptor.model_validate(item) for item in value)
    ids = tuple(item.pool_id for item in descriptors)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("Native data-pool descriptors repeat or reorder pool identities")
    return descriptors


def _inventory(value: object, reference: DataPoolReadReference) -> ProjectDataPoolInventory:
    """Verify complete declarations and their native scope/membership/digest.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    """
    inventory = ProjectDataPoolInventory.model_validate(value)
    if (
        inventory.project_id != reference.project_id
        or inventory.descriptor.pool_id != reference.pool_id
    ):
        raise ValueError("Data-pool inventory differs from requested project/pool")
    declarations = []
    for row in inventory.files:
        membership = row.metadata.get("data_pool")
        if not isinstance(membership, dict):
            raise ValueError("Native data-pool file lacks its pool membership declaration")
        selected_pool_id = cast(dict[str, object], membership).get("pool_id")
        if (
            row.org_id != reference.org_id
            or row.project_id != reference.project_id
            or row.run_id is not None
            or row.scope_kind != "project"
            or not str(row.path).startswith(f"data-pools/{reference.pool_id}/")
            or selected_pool_id != reference.pool_id
            or type(row.content_bytes) is not int
            or row.content_bytes < 0
            or not row.content_sha256
        ):
            raise ValueError(
                "Native data-pool file crossed scope or lacks membership/byte identity"
            )
        digest = "sha256:" + row.content_sha256.removeprefix("sha256:")
        validate_sha256_digest(digest)
        declarations.append(
            {
                "file_id": row.file_id,
                "path": row.path,
                "content_digest": digest,
                "size_bytes": row.content_bytes,
                "media_type": row.content_type or "application/octet-stream",
                "visibility": row.visibility,
                "storage_backend": row.storage_backend,
                "encoding": row.encoding,
                "metadata": row.metadata,
            }
        )
    if (
        inventory.file_count != len(inventory.files)
        or len({row.file_id for row in inventory.files}) != inventory.file_count
        or inventory.size_bytes != sum(item["size_bytes"] for item in declarations)
        or inventory.inventory_digest
        != _digest(
            _canonical_bytes(sorted(declarations, key=lambda item: (item["path"], item["file_id"])))
        )
    ):
        raise ValueError("Native data-pool inventory count, bytes or declaration digest drifted")
    return inventory


def _pool_file(
    inventory: ProjectDataPoolInventory, file_id: str, content_digest: str
) -> tuple[int, str]:
    """Require the caller's exact file identity and return its verified declaration.

    # See: testing-resources/specifications/sdk/forge_resource_reads.md
    """
    digest = validate_sha256_digest("sha256:" + content_digest.removeprefix("sha256:"))
    row = next((item for item in inventory.files if item.file_id == file_id), None)
    if row is None or "sha256:" + str(row.content_sha256).removeprefix("sha256:") != digest:
        raise ValueError("Selected pool file is absent or differs from the requested digest")
    if type(row.content_bytes) is not int or row.content_bytes < 0:
        raise ValueError("Selected pool file lacks an exact byte-count declaration")
    return row.content_bytes, digest


class DatasetRevisionsAPI:
    """Exact immutable revisions and original bytes through existing backend custody."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, reference: DatasetRevisionReadReference) -> SealedDatasetRevision:
        """Retrieve exact retained metadata. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _revision(
            self._transport.request_json(
                "GET",
                _revision_root(reference),
                params={"revision_digest": reference.revision_digest},
                operation_id="getExactDatasetRevision",
            ),
            reference,
        )

    def download_manifest(
        self, reference: DatasetRevisionReadReference, *, max_bytes: int = RESOURCE_READ_BYTES_MAX
    ) -> bytes:
        """Verify original manifest bytes. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        revision = self.get(reference)
        expected = _manifest(revision)
        _bounded(len(expected), max_bytes)
        content = self._transport.request_bytes(
            "GET",
            _revision_root(reference) + "/manifest",
            params={"revision_digest": reference.revision_digest},
            operation_id="getExactDatasetRevisionManifest",
        )
        return _verified(
            content,
            size_bytes=len(expected),
            content_digest=revision.manifest_digest,
            max_bytes=max_bytes,
        )

    def download_content(
        self,
        reference: DatasetRevisionReadReference,
        logical_path: str,
        *,
        max_bytes: int = RESOURCE_READ_BYTES_MAX,
    ) -> bytes:
        """Verify an original declared object. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        declaration = _declared_object(self.get(reference), logical_path)
        _bounded(declaration.size_bytes, max_bytes)
        content = self._transport.request_bytes(
            "GET",
            _revision_root(reference) + "/content",
            params={"revision_digest": reference.revision_digest, "logical_path": logical_path},
            operation_id="getExactDatasetRevisionContent",
        )
        return _verified(
            content,
            size_bytes=declaration.size_bytes,
            content_digest=declaration.content_digest,
            max_bytes=max_bytes,
        )


class AsyncDatasetRevisionsAPI:
    """Native-async peer of DatasetRevisionsAPI."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, reference: DatasetRevisionReadReference) -> SealedDatasetRevision:
        """Retrieve exact retained metadata. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _revision(
            await self._transport.request_json(
                "GET",
                _revision_root(reference),
                params={"revision_digest": reference.revision_digest},
                operation_id="getExactDatasetRevision",
            ),
            reference,
        )

    async def download_manifest(
        self, reference: DatasetRevisionReadReference, *, max_bytes: int = RESOURCE_READ_BYTES_MAX
    ) -> bytes:
        """Verify original manifest bytes. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        revision = await self.get(reference)
        expected = _manifest(revision)
        _bounded(len(expected), max_bytes)
        content = await self._transport.request_bytes(
            "GET",
            _revision_root(reference) + "/manifest",
            params={"revision_digest": reference.revision_digest},
            operation_id="getExactDatasetRevisionManifest",
        )
        return _verified(
            content,
            size_bytes=len(expected),
            content_digest=revision.manifest_digest,
            max_bytes=max_bytes,
        )

    async def download_content(
        self,
        reference: DatasetRevisionReadReference,
        logical_path: str,
        *,
        max_bytes: int = RESOURCE_READ_BYTES_MAX,
    ) -> bytes:
        """Verify an original declared object. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        declaration = _declared_object(await self.get(reference), logical_path)
        _bounded(declaration.size_bytes, max_bytes)
        content = await self._transport.request_bytes(
            "GET",
            _revision_root(reference) + "/content",
            params={"revision_digest": reference.revision_digest, "logical_path": logical_path},
            operation_id="getExactDatasetRevisionContent",
        )
        return _verified(
            content,
            size_bytes=declaration.size_bytes,
            content_digest=declaration.content_digest,
            max_bytes=max_bytes,
        )


class ProjectDataPoolsAPI:
    """Complete native pool declarations and exact selected bytes."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, project_id: str) -> tuple[ProjectDataPoolDescriptor, ...]:
        """Read native descriptors. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _pools(
            self._transport.request_json(
                "GET",
                _project_root(project_id) + "/data-pools",
                operation_id="listProjectDataPools",
            )
        )

    def get(self, reference: DataPoolReadReference) -> ProjectDataPoolInventory:
        """Verify complete scoped declarations. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _inventory(
            self._transport.request_json(
                "GET",
                _project_root(reference.project_id)
                + "/data-pools/"
                + quote(reference.pool_id, safe=""),
                operation_id="getProjectDataPoolInventory",
            ),
            reference,
        )

    def download_file(
        self,
        reference: DataPoolReadReference,
        file_id: str,
        *,
        content_digest: str,
        max_bytes: int = RESOURCE_READ_BYTES_MAX,
    ) -> bytes:
        """Verify digest-pinned native bytes. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        size_bytes, digest = _pool_file(self.get(reference), file_id, content_digest)
        _bounded(size_bytes, max_bytes)
        content = self._transport.request_bytes(
            "GET",
            _project_root(reference.project_id)
            + "/data-pools/"
            + quote(reference.pool_id, safe="")
            + "/files/"
            + quote(file_id, safe="")
            + "/content",
            params={"content_digest": content_digest},
            operation_id="getPinnedProjectDataPoolFile",
        )
        return _verified(content, size_bytes=size_bytes, content_digest=digest, max_bytes=max_bytes)


class AsyncProjectDataPoolsAPI:
    """Native-async peer of ProjectDataPoolsAPI."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, project_id: str) -> tuple[ProjectDataPoolDescriptor, ...]:
        """Read native descriptors. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _pools(
            await self._transport.request_json(
                "GET",
                _project_root(project_id) + "/data-pools",
                operation_id="listProjectDataPools",
            )
        )

    async def get(self, reference: DataPoolReadReference) -> ProjectDataPoolInventory:
        """Verify complete scoped declarations. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        return _inventory(
            await self._transport.request_json(
                "GET",
                _project_root(reference.project_id)
                + "/data-pools/"
                + quote(reference.pool_id, safe=""),
                operation_id="getProjectDataPoolInventory",
            ),
            reference,
        )

    async def download_file(
        self,
        reference: DataPoolReadReference,
        file_id: str,
        *,
        content_digest: str,
        max_bytes: int = RESOURCE_READ_BYTES_MAX,
    ) -> bytes:
        """Verify digest-pinned native bytes. See testing-resources/specifications/sdk/forge_resource_reads.md."""
        size_bytes, digest = _pool_file(await self.get(reference), file_id, content_digest)
        _bounded(size_bytes, max_bytes)
        content = await self._transport.request_bytes(
            "GET",
            _project_root(reference.project_id)
            + "/data-pools/"
            + quote(reference.pool_id, safe="")
            + "/files/"
            + quote(file_id, safe="")
            + "/content",
            params={"content_digest": content_digest},
            operation_id="getPinnedProjectDataPoolFile",
        )
        return _verified(content, size_bytes=size_bytes, content_digest=digest, max_bytes=max_bytes)


__all__ = [
    "AsyncDatasetRevisionsAPI",
    "AsyncProjectDataPoolsAPI",
    "DatasetRevisionsAPI",
    "ProjectDataPoolsAPI",
]
