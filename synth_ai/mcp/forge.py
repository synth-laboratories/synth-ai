"""Typed workspace tools; host identity and authority never enter tool arguments."""

from typing import Annotated

from pydantic import Field

from synth_ai.options import MutationOptions
from synth_ai.sdk.forge.contracts.projects import ProjectMember
from synth_ai.types.base import Identifier, Model


class ForgeProjectIdentity(Model):
    project_id: Identifier


class ForgeProjectCreate(Model):
    project_id: Identifier | None = None
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
