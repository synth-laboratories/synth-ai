"""Typed workspace tools; host identity and authority never enter tool arguments."""

from typing import Annotated

from pydantic import Field

from synth_ai.options import MutationOptions
from synth_ai.sdk.forge.contracts.projects import ProjectId, ProjectMember
from synth_ai.sdk.forge.contracts.scientific_v2 import RecordV2
from synth_ai.sdk.forge.contracts.workspace_science import WorkspaceRecord
from synth_ai.sdk.forge.index_types import ForgeIndexDraftRequest, ForgeIndexDraftUploadRequest
from synth_ai.types.base import Identifier, Model


class ForgeProjectIdentity(Model):
    project_id: ProjectId


class ForgeProjectCreate(Model):
    project_id: ProjectId | None = None
    title: Annotated[str, Field(min_length=1, max_length=8000)]
    objective: Annotated[str, Field(min_length=1, max_length=8000)]
    idempotency_key: Identifier


class ForgeProjectMutation(ForgeProjectIdentity):
    expected_revision: Annotated[int, Field(strict=True, ge=1, le=2147483646)]
    idempotency_key: Identifier


class ForgeProjectUpdate(ForgeProjectMutation):
    title: Annotated[str, Field(min_length=1, max_length=8000)]
    objective: Annotated[str, Field(min_length=1, max_length=8000)]


class ForgeProjectArchive(ForgeProjectMutation):
    archived: bool


class ForgeProjectMembers(ForgeProjectMutation):
    members: Annotated[tuple[ProjectMember, ...], Field(min_length=1, max_length=256)]


class ForgeRecordRead(ForgeProjectIdentity):
    record_id: Identifier
    revision: Annotated[int, Field(strict=True, ge=1, le=2147483647)] | None = None


class ForgeRecordWrite(ForgeProjectIdentity):
    record_id: Identifier
    expected_revision: Annotated[int, Field(strict=True, ge=0, le=2147483646)]
    idempotency_key: Identifier
    payload: WorkspaceRecord | RecordV2


class ForgeReceiptRead(ForgeProjectIdentity):
    operation_id: Identifier


class ForgeIndexDraftCreate(ForgeIndexDraftRequest, ForgeProjectIdentity):
    idempotency_key: Identifier


async def index_draft_create(client, args):
    return await client.forge.index_drafts.create(
        args.project_id,
        args.export_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def index_draft_get(client, args):
    return await client.forge.index_drafts.get(args.project_id, args.operation_id)


async def record_get(client, args):
    return await client.forge.records.get(args.project_id, args.record_id, revision=args.revision)


async def artifact_evidence_get(client, args):
    return await client.forge.records.artifact_evidence(
        args.project_id, args.record_id, revision=args.revision
    )


async def record_write(client, args):
    return await client.forge.records.write(
        args.project_id,
        args.record_id,
        args.payload,
        expected_revision=args.expected_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def receipt_get(client, args):
    return await client.forge.receipts.get(args.project_id, args.operation_id)


async def get(client, args):
    return await client.forge.projects.get(args.project_id)


async def create(client, args):
    return await client.forge.projects.create(
        project_id=args.project_id,
        title=args.title,
        objective=args.objective,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def update(client, args):
    return await client.forge.projects.update(
        args.project_id,
        title=args.title,
        objective=args.objective,
        expected_revision=args.expected_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def archive(client, args):
    return await client.forge.projects.archive(
        args.project_id,
        archived=args.archived,
        expected_revision=args.expected_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def members(client, args):
    return await client.forge.projects.members(
        args.project_id,
        members=args.members,
        expected_revision=args.expected_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


class ForgeIndexDraftUpload(ForgeIndexDraftUploadRequest, ForgeProjectIdentity):
    draft_operation_id: Identifier
    idempotency_key: Identifier


class ForgeIndexDraftUploadRead(ForgeReceiptRead):
    draft_operation_id: Identifier


async def index_draft_upload(client, args):
    return await client.forge.index_drafts.upload(
        args.project_id,
        args.draft_operation_id,
        args.descriptor,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def index_draft_upload_get(client, args):
    return await client.forge.index_drafts.upload_get(
        args.project_id, args.draft_operation_id, args.operation_id
    )
