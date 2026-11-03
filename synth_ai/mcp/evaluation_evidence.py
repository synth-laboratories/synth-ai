"""Customer project evidence tools, gated by backend capabilities and authorization."""

from pydantic import Field, StrictInt, model_validator

from synth_ai.sdk.forge.contracts.projects import ProjectId
from synth_ai.types.base import Identifier, Model


class Project(Model):
    project_id: ProjectId


class Page(Project):
    after: Identifier | None = None
    limit: StrictInt = Field(default=50, ge=1, le=100)


class Identity(Project):
    evidence_id: Identifier


class Comparison(Project):
    evidence_ids: list[Identifier] = Field(
        min_length=2, max_length=8, json_schema_extra={"uniqueItems": True}
    )

    @model_validator(mode="after")
    def distinct_ids(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence IDs must be distinct")
        return self


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


async def compare(client, args):
    return await client.mnemosyne.evaluation_evidence.compare(args.project_id, args.evidence_ids)
