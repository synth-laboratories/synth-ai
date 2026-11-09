"""Small workflows using the public SDK; each is exercised by test_examples.py."""

from collections.abc import Awaitable, Callable
from contextlib import aclosing
from pathlib import Path

from synth_ai import AsyncSynthClient, CompletionMilestone, StreamCheckpoint
from synth_ai.errors import UploadTransferError, WaitTimeoutError
from synth_ai.models import (
    Exclude,
    File,
    Only,
    RoutingConstraints,
    RunEvent,
    RunSnapshot,
    UsageSummary,
)


async def required_budget(client: AsyncSynthClient, message: str) -> RunSnapshot:
    return await client.swarms.launch(message, budget="0.10", wall_time_seconds=60)


async def select_profile(
    client: AsyncSynthClient, message: str, *, research_task: str | None = None
) -> RunSnapshot:
    catalog = await client.swarms.profiles(research_task=research_task)
    selected = next(
        (
            p
            for p in catalog.profiles
            if p.available and p.requires_research_task == (research_task is not None)
        ),
        None,
    )
    if selected is None:
        raise ValueError("no matching profile is currently available")
    return await client.swarms.launch(
        message, budget="0.10", profile=selected.id, research_task=research_task
    )


async def constrained_routing(client: AsyncSynthClient, message: str) -> RunSnapshot:
    return await client.swarms.launch(
        message,
        budget="0.10",
        routing=RoutingConstraints(
            providers=Only(only=["openai"]), models=Exclude(exclude=["excluded-model"])
        ),
    )


async def wait_result_then_finished(client: AsyncSynthClient, run_id: str) -> RunSnapshot:
    # A result can exist before final accounting and resource cleanup.
    await client.swarms.wait(run_id, until=CompletionMilestone.RESULT_AVAILABLE)
    return await client.swarms.wait(run_id, until=CompletionMilestone.FINISHED)


async def cancel_after_timeout(
    client: AsyncSynthClient, run_id: str, *, timeout: float, cancel_key: str
) -> RunSnapshot:
    try:
        return await client.swarms.wait(run_id, timeout=timeout)
    except WaitTimeoutError:
        # This is an explicit application choice; observation timeout alone
        # never sends a cancellation request.
        await client.swarms.cancel(run_id, idempotency_key=cancel_key)
        return await client.swarms.wait(run_id)


async def restartable_stream(
    client: AsyncSynthClient,
    run_id: str,
    checkpoint: StreamCheckpoint,
    apply: Callable[[RunEvent], Awaitable[None]],
    save: Callable[[StreamCheckpoint], None],
) -> None:
    async with aclosing(client.swarms.follow(run_id, checkpoint=checkpoint)) as events:
        async for event in events:
            await apply(event)
            # Save only after application. Applications needing atomic durability
            # must commit their state and this checkpoint in one transaction.
            save(checkpoint)


async def recover_upload(client: AsyncSynthClient, filename: str, *, upload_key: str) -> File:
    try:
        return await client.swarms.upload_file(filename, idempotency_key=upload_key)
    except UploadTransferError as error:
        return await client.swarms.resume_upload(
            error.upload_id, filename, idempotency_key=error.idempotency_key
        )


async def verified_artifact(
    client: AsyncSynthClient, run_id: str, output_id: str, destination: str
) -> Path:
    output = await client.swarms.output(run_id, output_id)
    if output.reference is None:
        raise ValueError("output content is not available")
    return await client.swarms.download_output(
        run_id,
        output_id,
        destination,
        revision_id=output.reference["revision_id"],
        digest_sha256=output.reference["digest_sha256"],
    )


async def usage_readback(client: AsyncSynthClient, *, since: str, until: str) -> UsageSummary:
    # These are Core's settled usage facts, not an SDK billing estimate.
    return await client.account.usage_summary(since=since, until=until)
