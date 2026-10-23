"""Explicit native Index private-owner tools. See SDK_0230_MCP.md."""

from synth_ai.options import MutationOptions
from synth_ai.sdk.index.types.contributions import SubmissionRequest
from synth_ai.types.base import Identifier

from .contracts import ContributionIdentity


class ContributionSubmit(SubmissionRequest, ContributionIdentity):
    idempotency_key: Identifier


async def submit(client, args):
    return await client.index.contributions.submit(
        args.contribution_id,
        revision=args.revision_id,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def get(client, args):
    return await client.index.contributions.get_owner(args.contribution_id)
