"""Allocate private Index custody for an exact frozen Forge export."""

import json

from synth_ai.contracts import forge as operations
from synth_ai.core.parameters import identifier, verified

from .index_types import (
    ForgeIndexDraftDescriptor,
    ForgeIndexDraftRequest,
    ForgeIndexDraftResponse,
    ForgeIndexDraftUploadRequest,
    ForgeIndexDraftUploadResponse,
)
from .records import project_identity


def checked(result, project_id, *, reference=None, operation_id=None):
    return verified(
        result,
        result.project_id == project_id
        and result.metadata.operation_id is not None
        and (reference is None or result.export_revision == reference)
        and (operation_id is None or result.metadata.operation_id == operation_id),
        "Forge returned a different private draft receipt",
    )


def checked_upload(result, project_id, draft_operation_id, *, operation_id=None):
    return verified(
        result,
        result.project_id == project_id
        and result.draft_operation_id == draft_operation_id
        and result.metadata.operation_id is not None
        and (operation_id is None or result.metadata.operation_id == operation_id),
        "Forge returned a different private upload receipt",
    )


class IndexDrafts:
    def __init__(self, transport):
        self._transport = transport

    def create(self, project_id, export_revision, *, mutation=None):
        project_id = project_identity(project_id)
        body = ForgeIndexDraftRequest(export_revision=export_revision)
        return self._transport.request(
            operations.INDEX_DRAFT_CREATE,
            ForgeIndexDraftResponse,
            path=operations.INDEX_DRAFT_CREATE.path_for(project_id=project_id),
            body=body,
            options=mutation,
            validate=lambda result: checked(result, project_id, reference=body.export_revision),
        )

    def get(self, project_id, operation_id, *, request=None):
        project_id, operation_id = project_identity(project_id), identifier(operation_id)
        return self._transport.request(
            operations.INDEX_DRAFT_GET,
            ForgeIndexDraftResponse,
            path=operations.INDEX_DRAFT_GET.path_for(
                project_id=project_id, operation_id=operation_id
            ),
            options=request,
            validate=lambda result: checked(result, project_id, operation_id=operation_id),
        )

    def upload(self, project_id, draft_operation_id, descriptor, *, mutation=None):
        project_id = project_identity(project_id)
        draft_operation_id = identifier(draft_operation_id)
        descriptor = (
            descriptor
            if isinstance(descriptor, ForgeIndexDraftDescriptor)
            else ForgeIndexDraftDescriptor.model_validate_json(json.dumps(descriptor))
        )
        body = ForgeIndexDraftUploadRequest(descriptor=descriptor)
        return self._transport.request(
            operations.INDEX_DRAFT_UPLOAD,
            ForgeIndexDraftUploadResponse,
            path=operations.INDEX_DRAFT_UPLOAD.path_for(
                project_id=project_id, draft_operation_id=draft_operation_id
            ),
            body=body,
            options=mutation,
            validate=lambda result: checked_upload(result, project_id, draft_operation_id),
        )

    def upload_get(self, project_id, draft_operation_id, operation_id, *, request=None):
        project_id = project_identity(project_id)
        draft_operation_id, operation_id = identifier(draft_operation_id), identifier(operation_id)
        return self._transport.request(
            operations.INDEX_DRAFT_UPLOAD_GET,
            ForgeIndexDraftUploadResponse,
            path=operations.INDEX_DRAFT_UPLOAD_GET.path_for(
                project_id=project_id,
                draft_operation_id=draft_operation_id,
                operation_id=operation_id,
            ),
            options=request,
            validate=lambda result: checked_upload(
                result, project_id, draft_operation_id, operation_id=operation_id
            ),
        )


class AsyncIndexDrafts:
    def __init__(self, transport):
        self._transport = transport

    async def create(self, project_id, export_revision, *, mutation=None):
        project_id = project_identity(project_id)
        body = ForgeIndexDraftRequest(export_revision=export_revision)
        return await self._transport.request(
            operations.INDEX_DRAFT_CREATE,
            ForgeIndexDraftResponse,
            path=operations.INDEX_DRAFT_CREATE.path_for(project_id=project_id),
            body=body,
            options=mutation,
            validate=lambda result: checked(result, project_id, reference=body.export_revision),
        )

    async def get(self, project_id, operation_id, *, request=None):
        project_id, operation_id = project_identity(project_id), identifier(operation_id)
        return await self._transport.request(
            operations.INDEX_DRAFT_GET,
            ForgeIndexDraftResponse,
            path=operations.INDEX_DRAFT_GET.path_for(
                project_id=project_id, operation_id=operation_id
            ),
            options=request,
            validate=lambda result: checked(result, project_id, operation_id=operation_id),
        )

    async def upload(self, project_id, draft_operation_id, descriptor, *, mutation=None):
        project_id = project_identity(project_id)
        draft_operation_id = identifier(draft_operation_id)
        descriptor = (
            descriptor
            if isinstance(descriptor, ForgeIndexDraftDescriptor)
            else ForgeIndexDraftDescriptor.model_validate_json(json.dumps(descriptor))
        )
        body = ForgeIndexDraftUploadRequest(descriptor=descriptor)
        return await self._transport.request(
            operations.INDEX_DRAFT_UPLOAD,
            ForgeIndexDraftUploadResponse,
            path=operations.INDEX_DRAFT_UPLOAD.path_for(
                project_id=project_id, draft_operation_id=draft_operation_id
            ),
            body=body,
            options=mutation,
            validate=lambda result: checked_upload(result, project_id, draft_operation_id),
        )

    async def upload_get(self, project_id, draft_operation_id, operation_id, *, request=None):
        project_id = project_identity(project_id)
        draft_operation_id, operation_id = identifier(draft_operation_id), identifier(operation_id)
        return await self._transport.request(
            operations.INDEX_DRAFT_UPLOAD_GET,
            ForgeIndexDraftUploadResponse,
            path=operations.INDEX_DRAFT_UPLOAD_GET.path_for(
                project_id=project_id,
                draft_operation_id=draft_operation_id,
                operation_id=operation_id,
            ),
            options=request,
            validate=lambda result: checked_upload(
                result, project_id, draft_operation_id, operation_id=operation_id
            ),
        )
