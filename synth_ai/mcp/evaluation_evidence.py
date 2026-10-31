"""Customer project evidence tools, gated by backend capabilities and authorization."""

from pydantic import Field, StrictInt

from synth_ai.sdk.forge.contracts.projects import ProjectId
from synth_ai.types.base import Identifier, Model


class Project(Model):
    project_id: ProjectId


class Page(Project):
    after: Identifier | None = None
    limit: StrictInt = Field(default=50, ge=1, le=100)


class Identity(Project):
    evidence_id: Identifier


async def capabilities(client, args):
    return await client.mnemosyne.evaluation_evidence.capabilities(args.project_id)


async def list_evidence(client, args):
    return await client.mnemosyne.evaluation_evidence.list(
        args.project_id, after=args.after, limit=args.limit
    )


async def get(client, args):
    return await client.mnemosyne.evaluation_evidence.get(args.project_id, args.evidence_id)


async def measurements(client, args):
    return await client.mnemosyne.evaluation_evidence.measurements(
        args.project_id, args.evidence_id
    )
