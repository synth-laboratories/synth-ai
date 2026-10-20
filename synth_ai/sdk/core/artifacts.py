"""Artifact uploads, exact reads, labels and aliases under current project grants."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import TypeAdapter

from synth_ai.contracts import core as operations
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.core.transfer import transfer as transfer_bytes, async_transfer
from synth_ai.options import ReadOptions, MutationOptions
from synth_ai.errors import RequestTimeoutError
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.refs import Ref, Relation, ReferenceIdentifier

from .artifact_types import (
    Artifact,
    ArtifactAlias,
    ArtifactAliasSet,
    ArtifactLabelsUpdate,
    ArtifactSource,
    ArtifactUploadManifest,
    ArtifactUploadRequest,
    ArtifactUploadStatus,
    ArtifactUploadReceipt,
    ArtifactDownloadReceipt,
    ArtifactKind,
    Labels,
)
from .artifact_download import (
    ArtifactDownloadOptions,
    ArtifactTransferOptions,
    download_object,
    download_object_async,
)


def selection(project_id, artifact, revision):
    project_id = identifier(project_id)
    reference = artifact if isinstance(artifact, Ref) else None
    if reference is not None:
        if reference.authority != "artifacts" or reference.kind != "artifact":
            raise ValueError("Expected an exact artifact Ref")
        Ref.artifact(reference.record_id, reference.revision, reference.digest_sha256)
        artifact_id, revision = reference.record_id, reference.revision
    else:
        artifact_id = identifier(artifact)
    if revision != "latest" and (
        not isinstance(revision, (str, int))
        or isinstance(revision, bool)
        or not str(revision).isascii()
        or not str(revision).isdigit()
        or not 1 <= int(revision) <= 2_147_483_647
    ):
        raise ValueError("Select an exact positive artifact revision or the explicit latest lookup")
    query = {"revision": str(revision)}
    if reference is not None:
        query["digest_sha256"] = reference.digest_sha256
    return project_id, artifact_id, reference, query


def checked_artifact(value, project_id, artifact_id, reference=None, revision="latest"):
    return verified(
        value,
        value.project_id == project_id
        and value.artifact_id == artifact_id
        and (reference is None or value.ref == reference)
        and (revision == "latest" or value.ref.revision == str(int(revision))),
        "Artifact changed the selected project or exact publication",
    )


def list_query(limit, cursor, kind, label, linked):
    query = page_parameters(limit, cursor)
    if kind is not None:
        TypeAdapter(ArtifactKind).validate_python(kind)
        query["kind"] = kind
    if label is not None:
        selected = TypeAdapter(Labels).validate_python(label)
        if len(selected) > 4:
            raise ValueError("Artifact list permits at most4 label filters")
        query["label"] = tuple(key + "=" + value for key, value in sorted(selected.items()))
    if linked is not None:
        relation, reference = linked
        if not isinstance(reference, Ref):
            raise ValueError("Linked filter needs an exact Ref")
        query.update(
            linked_relation=Relation(relation).value,
            linked_subject=":".join(
                (reference.authority, reference.kind, reference.record_id, reference.revision)
            ),
            linked_subject_digest_sha256=reference.digest_sha256,
        )
    return query


def checked_upload(value, previous):
    return verified(
        value,
        value.upload_id == previous.upload_id
        and value.project_id == previous.project_id
        and value.artifact_id == previous.artifact_id
        and value.revision == previous.revision
        and value.manifest == previous.manifest,
        "Artifact upload changed its immutable publication selection",
    )


class ArtifactUpload:
    def __init__(self, transport, snapshot):
        self._transport, self.snapshot = transport, snapshot
        self.upload_id = snapshot.upload_id

    def status(self, *, request=None) -> ArtifactUploadStatus:
        return self._transport.request(
            operations.ARTIFACT_UPLOAD,
            ArtifactUploadStatus,
            path=f"/artifacts/uploads/{identifier(self.upload_id)}",
            options=request,
            validate=lambda value: checked_upload(value, self.snapshot),
        )

    def transfer(self, source: ArtifactSource, *, transfer: ArtifactTransferOptions):
        deadline = time.monotonic() + transfer.timeout_seconds
        status = self.status(
            request=ReadOptions(timeout_seconds=transfer.timeout_seconds, read_retry_count=0)
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Artifact transfer deadline expired")
        return transfer_bytes(
            status,
            source,
            transfer.model_copy(update={"timeout_seconds": remaining}),
            self._transport.transfer_hosts,
        )

    def finalize(self, *, mutation: MutationOptions | None = None) -> ArtifactUploadReceipt:
        return self._transport.request(
            operations.ARTIFACT_FINALIZE,
            ArtifactUploadReceipt,
            path=f"/artifacts/uploads/{identifier(self.upload_id)}/finalize",
            options=mutation,
            validate=lambda result: verified(
                result,
                result.upload_id == self.upload_id
                and result.artifact.project_id == self.snapshot.project_id
                and result.artifact.artifact_id == self.snapshot.artifact_id
                and result.artifact.ref.revision == str(self.snapshot.revision)
                and result.artifact.manifest.objects[0].digest_sha256
                == self.snapshot.manifest.digest_sha256,
                "Final artifact changed upload identity or content",
            ),
        )


class AsyncArtifactUpload:
    def __init__(self, transport, snapshot):
        self._transport, self.snapshot = transport, snapshot
        self.upload_id = snapshot.upload_id

    async def status(self, *, request=None) -> ArtifactUploadStatus:
        return await self._transport.request(
            operations.ARTIFACT_UPLOAD,
            ArtifactUploadStatus,
            path=f"/artifacts/uploads/{identifier(self.upload_id)}",
            options=request,
            validate=lambda value: checked_upload(value, self.snapshot),
        )

    async def transfer(self, source: ArtifactSource, *, transfer: ArtifactTransferOptions):
        deadline = time.monotonic() + transfer.timeout_seconds
        status = await self.status(
            request=ReadOptions(timeout_seconds=transfer.timeout_seconds, read_retry_count=0)
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Artifact transfer deadline expired")
        return await async_transfer(
            status,
            source,
            transfer.model_copy(update={"timeout_seconds": remaining}),
            self._transport.transfer_hosts,
        )

    async def finalize(self, *, mutation: MutationOptions | None = None) -> ArtifactUploadReceipt:
        return await self._transport.request(
            operations.ARTIFACT_FINALIZE,
            ArtifactUploadReceipt,
            path=f"/artifacts/uploads/{identifier(self.upload_id)}/finalize",
            options=mutation,
            validate=lambda result: verified(
                result,
                result.upload_id == self.upload_id
                and result.artifact.project_id == self.snapshot.project_id
                and result.artifact.artifact_id == self.snapshot.artifact_id
                and result.artifact.ref.revision == str(self.snapshot.revision)
                and result.artifact.manifest.objects[0].digest_sha256
                == self.snapshot.manifest.digest_sha256,
                "Final artifact changed upload identity or content",
            ),
        )


class ArtifactUploads:
    def __init__(self, transport):
        self._transport = transport

    def prepare(
        self,
        project_id,
        *,
        manifest: ArtifactUploadManifest,
        kind: ArtifactKind = "file",
        labels=None,
        artifact_id=None,
        mutation=None,
    ) -> ArtifactUpload:
        project_id = identifier(project_id)
        body = ArtifactUploadRequest(
            manifest=manifest, kind=kind, labels=labels or {}, artifact_id=artifact_id
        )
        result = self._transport.request(
            operations.ARTIFACT_PREPARE,
            ArtifactUploadStatus,
            path=f"/projects/{project_id}/artifacts/uploads",
            body=body,
            options=mutation,
            validate=lambda value: verified(
                value,
                value.project_id == project_id
                and value.manifest == manifest
                and value.kind == kind
                and (artifact_id is None or value.artifact_id == artifact_id),
                "Artifact upload preparation changed intent",
            ),
        )
        return ArtifactUpload(self._transport, result)

    def get(self, upload_id, *, request=None) -> ArtifactUpload:
        upload_id = identifier(upload_id)
        result = self._transport.request(
            operations.ARTIFACT_UPLOAD,
            ArtifactUploadStatus,
            path=f"/artifacts/uploads/{upload_id}",
            options=request,
            validate=lambda value: verified(
                value, value.upload_id == upload_id, "Artifact upload reattachment changed identity"
            ),
        )
        return ArtifactUpload(self._transport, result)


class AsyncArtifactUploads:
    def __init__(self, transport):
        self._transport = transport

    async def prepare(
        self,
        project_id,
        *,
        manifest: ArtifactUploadManifest,
        kind: ArtifactKind = "file",
        labels=None,
        artifact_id=None,
        mutation=None,
    ) -> AsyncArtifactUpload:
        project_id = identifier(project_id)
        body = ArtifactUploadRequest(
            manifest=manifest, kind=kind, labels=labels or {}, artifact_id=artifact_id
        )
        result = await self._transport.request(
            operations.ARTIFACT_PREPARE,
            ArtifactUploadStatus,
            path=f"/projects/{project_id}/artifacts/uploads",
            body=body,
            options=mutation,
            validate=lambda value: verified(
                value,
                value.project_id == project_id
                and value.manifest == manifest
                and value.kind == kind
                and (artifact_id is None or value.artifact_id == artifact_id),
                "Artifact upload preparation changed intent",
            ),
        )
        return AsyncArtifactUpload(self._transport, result)

    async def get(self, upload_id, *, request=None) -> AsyncArtifactUpload:
        upload_id = identifier(upload_id)
        result = await self._transport.request(
            operations.ARTIFACT_UPLOAD,
            ArtifactUploadStatus,
            path=f"/artifacts/uploads/{upload_id}",
            options=request,
            validate=lambda value: verified(
                value, value.upload_id == upload_id, "Artifact upload reattachment changed identity"
            ),
        )
        return AsyncArtifactUpload(self._transport, result)


class Artifacts:
    def __init__(self, transport):
        self._transport = transport
        self.uploads = ArtifactUploads(transport)
        self.labels = ArtifactLabels(self)
        self.aliases = ArtifactAliases(self)

    def get(self, project_id, artifact: Ref | str, *, revision="latest", request=None) -> Artifact:
        project_id, artifact_id, reference, query = selection(project_id, artifact, revision)
        return self._transport.request(
            operations.ARTIFACT_GET,
            Artifact,
            path=f"/projects/{project_id}/artifacts/{artifact_id}",
            query=query,
            options=request,
            validate=lambda value: checked_artifact(
                value, project_id, artifact_id, reference, query["revision"]
            ),
        )

    def list(
        self,
        project_id,
        *,
        kind=None,
        label=None,
        linked=None,
        limit=200,
        cursor: PageCursor | None = None,
        request=None,
    ) -> Page[Artifact]:
        project_id = identifier(project_id)
        query = list_query(limit, cursor, kind, label, linked)
        return self._transport.request(
            operations.ARTIFACT_LIST,
            Page[Artifact],
            path=f"/projects/{project_id}/artifacts",
            query=query,
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(
                    item.project_id == project_id
                    and (kind is None or item.kind == kind)
                    and all(item.labels.get(key) == value for key, value in (label or {}).items())
                    for item in result.items
                ),
                "Artifact list changed its selection",
            ),
        )

    def download(
        self,
        project_id,
        artifact: Ref | str,
        destination: Path,
        *,
        revision="latest",
        path=None,
        options=None,
    ) -> ArtifactDownloadReceipt:
        options = options or ArtifactDownloadOptions()
        deadline = time.monotonic() + options.timeout_seconds
        from .repo_archive import archive_destination

        destination = archive_destination(destination)
        selected = self.get(
            project_id,
            artifact,
            revision=revision,
            request=ReadOptions(timeout_seconds=options.timeout_seconds, read_retry_count=0),
        )
        return download_object(
            self._transport, selected, destination, path=path, options=options, deadline=deadline
        )

    def upload(
        self,
        project_id,
        source: ArtifactSource,
        *,
        kind="file",
        labels=None,
        artifact_id=None,
        prepare_mutation=None,
        finalize_mutation=None,
        transfer=None,
    ) -> Artifact:
        transfer = transfer or ArtifactTransferOptions(
            timeout_seconds=900,
            maximum_bytes=64 * 1024 * 1024,
            chunk_bytes=65536,
            retry_count_max=0,
        )
        upload = self.uploads.prepare(
            project_id,
            manifest=source.manifest,
            kind=kind,
            labels=labels,
            artifact_id=artifact_id,
            mutation=prepare_mutation,
        )
        if upload.snapshot.state == "prepared":
            upload.transfer(source, transfer=transfer)
        return upload.finalize(mutation=finalize_mutation).artifact

    def update_labels(
        self, project_id, artifact_id, *, set=None, remove=(), expected_revision, mutation=None
    ) -> Artifact:
        project_id, artifact_id = identifier(project_id), identifier(artifact_id)
        return self._transport.request(
            operations.ARTIFACT_LABELS,
            Artifact,
            path=f"/projects/{project_id}/artifacts/{artifact_id}/labels",
            body=ArtifactLabelsUpdate(
                set=set or {}, remove=tuple(remove), expected_revision=expected_revision
            ),
            options=mutation,
            validate=lambda value: checked_artifact(value, project_id, artifact_id),
        )

    def set_alias(
        self, project_id, name, reference: Ref, *, expected_revision=0, mutation=None
    ) -> ArtifactAlias:
        project_id = identifier(project_id)
        name = TypeAdapter(ReferenceIdentifier).validate_python(name)
        return self._transport.request(
            operations.ARTIFACT_ALIAS,
            ArtifactAlias,
            path=f"/projects/{project_id}/artifacts/aliases/{name}",
            body=ArtifactAliasSet(ref=reference, expected_revision=expected_revision),
            options=mutation,
            validate=lambda value: verified(
                value,
                value.project_id == project_id and value.name == name and value.ref == reference,
                "Artifact alias changed exact selection",
            ),
        )

    def get_alias(self, project_id, name, *, request=None) -> ArtifactAlias:
        project_id = identifier(project_id)
        name = TypeAdapter(ReferenceIdentifier).validate_python(name)
        return self._transport.request(
            operations.ARTIFACT_ALIAS_GET,
            ArtifactAlias,
            path=f"/projects/{project_id}/artifacts/aliases/{name}",
            options=request,
            validate=lambda value: verified(
                value,
                value.project_id == project_id and value.name == name,
                "Artifact alias changed selected project/name",
            ),
        )


class AsyncArtifacts:
    def __init__(self, transport):
        self._transport = transport
        self.uploads = AsyncArtifactUploads(transport)
        self.labels = AsyncArtifactLabels(self)
        self.aliases = AsyncArtifactAliases(self)

    async def get(
        self, project_id, artifact: Ref | str, *, revision="latest", request=None
    ) -> Artifact:
        project_id, artifact_id, reference, query = selection(project_id, artifact, revision)
        return await self._transport.request(
            operations.ARTIFACT_GET,
            Artifact,
            path=f"/projects/{project_id}/artifacts/{artifact_id}",
            query=query,
            options=request,
            validate=lambda value: checked_artifact(
                value, project_id, artifact_id, reference, query["revision"]
            ),
        )

    async def list(
        self,
        project_id,
        *,
        kind=None,
        label=None,
        linked=None,
        limit=200,
        cursor: PageCursor | None = None,
        request=None,
    ) -> Page[Artifact]:
        project_id = identifier(project_id)
        query = list_query(limit, cursor, kind, label, linked)
        return await self._transport.request(
            operations.ARTIFACT_LIST,
            Page[Artifact],
            path=f"/projects/{project_id}/artifacts",
            query=query,
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(
                    item.project_id == project_id
                    and (kind is None or item.kind == kind)
                    and all(item.labels.get(key) == value for key, value in (label or {}).items())
                    for item in result.items
                ),
                "Artifact list changed its selection",
            ),
        )

    async def download(
        self,
        project_id,
        artifact: Ref | str,
        destination: Path,
        *,
        revision="latest",
        path=None,
        options=None,
    ) -> ArtifactDownloadReceipt:
        options = options or ArtifactDownloadOptions()
        deadline = time.monotonic() + options.timeout_seconds
        from .repo_archive import archive_destination

        destination = archive_destination(destination)
        selected = await self.get(
            project_id,
            artifact,
            revision=revision,
            request=ReadOptions(timeout_seconds=options.timeout_seconds, read_retry_count=0),
        )
        return await download_object_async(
            self._transport, selected, destination, path=path, options=options, deadline=deadline
        )

    async def upload(
        self,
        project_id,
        source: ArtifactSource,
        *,
        kind="file",
        labels=None,
        artifact_id=None,
        prepare_mutation=None,
        finalize_mutation=None,
        transfer=None,
    ) -> Artifact:
        transfer = transfer or ArtifactTransferOptions(
            timeout_seconds=900,
            maximum_bytes=64 * 1024 * 1024,
            chunk_bytes=65536,
            retry_count_max=0,
        )
        upload = await self.uploads.prepare(
            project_id,
            manifest=source.manifest,
            kind=kind,
            labels=labels,
            artifact_id=artifact_id,
            mutation=prepare_mutation,
        )
        if upload.snapshot.state == "prepared":
            await upload.transfer(source, transfer=transfer)
        return (await upload.finalize(mutation=finalize_mutation)).artifact

    async def update_labels(
        self, project_id, artifact_id, *, set=None, remove=(), expected_revision, mutation=None
    ) -> Artifact:
        project_id, artifact_id = identifier(project_id), identifier(artifact_id)
        return await self._transport.request(
            operations.ARTIFACT_LABELS,
            Artifact,
            path=f"/projects/{project_id}/artifacts/{artifact_id}/labels",
            body=ArtifactLabelsUpdate(
                set=set or {}, remove=tuple(remove), expected_revision=expected_revision
            ),
            options=mutation,
            validate=lambda value: checked_artifact(value, project_id, artifact_id),
        )

    async def set_alias(
        self, project_id, name, reference: Ref, *, expected_revision=0, mutation=None
    ) -> ArtifactAlias:
        project_id = identifier(project_id)
        name = TypeAdapter(ReferenceIdentifier).validate_python(name)
        return await self._transport.request(
            operations.ARTIFACT_ALIAS,
            ArtifactAlias,
            path=f"/projects/{project_id}/artifacts/aliases/{name}",
            body=ArtifactAliasSet(ref=reference, expected_revision=expected_revision),
            options=mutation,
            validate=lambda value: verified(
                value,
                value.project_id == project_id and value.name == name and value.ref == reference,
                "Artifact alias changed exact selection",
            ),
        )

    async def get_alias(self, project_id, name, *, request=None) -> ArtifactAlias:
        project_id = identifier(project_id)
        name = TypeAdapter(ReferenceIdentifier).validate_python(name)
        return await self._transport.request(
            operations.ARTIFACT_ALIAS_GET,
            ArtifactAlias,
            path=f"/projects/{project_id}/artifacts/aliases/{name}",
            options=request,
            validate=lambda value: verified(
                value,
                value.project_id == project_id and value.name == name,
                "Artifact alias changed selected project/name",
            ),
        )


class ArtifactLabels:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    def update(self, project_id, artifact_id, **options) -> Artifact:
        return self._artifacts.update_labels(project_id, artifact_id, **options)


class AsyncArtifactLabels:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    async def update(self, project_id, artifact_id, **options) -> Artifact:
        return await self._artifacts.update_labels(project_id, artifact_id, **options)


class ArtifactAliases:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    def set(self, project_id, name, reference, **options) -> ArtifactAlias:
        return self._artifacts.set_alias(project_id, name, reference, **options)

    def get(self, project_id, name, **options) -> ArtifactAlias:
        return self._artifacts.get_alias(project_id, name, **options)


class AsyncArtifactAliases:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    async def set(self, project_id, name, reference, **options) -> ArtifactAlias:
        return await self._artifacts.set_alias(project_id, name, reference, **options)

    async def get(self, project_id, name, **options) -> ArtifactAlias:
        return await self._artifacts.get_alias(project_id, name, **options)
