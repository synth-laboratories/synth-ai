"""Explicit native Index private-owner tools. See SDK_0230_MCP.md."""

from synth_ai.options import MutationOptions
from synth_ai.sdk.index.types.contributions import (
    ContributionDraft,
    ResearchCorrection,
    RightsAttestation,
    SubmissionRequest,
)
from synth_ai.types.base import Identifier, Model

from .contracts import ContributionIdentity


class ContributionSubmit(SubmissionRequest, ContributionIdentity):
    idempotency_key: Identifier


class ContributionCreate(Model):
    idempotency_key: Identifier


class ContributionUpdate(ContributionDraft, ContributionIdentity):
    idempotency_key: Identifier


class ContributionAttest(RightsAttestation, ContributionIdentity):
    idempotency_key: Identifier


class ContributionCorrect(ResearchCorrection, ContributionIdentity):
    idempotency_key: Identifier


async def create(client, args):
    return await client.index.contributions.create(
        mutation=MutationOptions(idempotency_key=args.idempotency_key)
    )


def intent(args, model):
    return model.model_validate(args.model_dump(exclude={"contribution_id", "idempotency_key"}))


async def update(client, args):
    return await client.index.contributions.update(
        args.contribution_id,
        draft=intent(args, ContributionDraft),
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def attest(client, args):
    return await client.index.contributions.attest(
        args.contribution_id,
        attestation=intent(args, RightsAttestation),
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def correct(client, args):
    return await client.index.contributions.correct(
        args.contribution_id,
        correction=intent(args, ResearchCorrection),
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def submit(client, args):
    return await client.index.contributions.submit(
        args.contribution_id,
        revision=args.revision_id,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def get(client, args):
    return await client.index.contributions.get_owner(args.contribution_id)
