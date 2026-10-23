"""Allocate private Index custody for an exact frozen Forge export."""

from synth_ai.contracts import forge as operations
from synth_ai.core.parameters import identifier, verified

from .index_types import ForgeIndexDraftRequest, ForgeIndexDraftResponse
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
