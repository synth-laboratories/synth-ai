"""Type-only consumer contracts. These functions are checked, never executed."""

from collections.abc import AsyncGenerator, Iterator
from typing import assert_type

from pydantic import JsonValue

from synth_ai import AsyncSynthClient, CompletionMilestone, RunState, SynthClient
from synth_ai._generated.models import Balance, File, RunEvent, RunSnapshot
from synth_ai.errors import WaitTimeoutError
from synth_ai.models import InputFile, RunInputs, RunProfileCatalog, StreamCheckpoint


def synchronous(client: SynthClient) -> None:
    inputs = RunInputs(
        files=[InputFile(file_id="file", revision_id="revision", digest_sha256="a" * 64)]
    )
    assert inputs.files is not None
    assert_type(inputs.files[0], InputFile)
    run = client.swarms.run(
        "Solve the task", budget="1", inputs=inputs, until=CompletionMilestone.FINISHED
    )
    assert_type(run, RunSnapshot)
    assert_type(run.known_state, RunState | None)
    assert_type(run.metadata, dict[str, JsonValue])
    assert_type(client.swarms.profiles(), RunProfileCatalog)
    assert_type(client.swarms.upload_text("task"), File)
    assert_type(client.swarms.iter_events(run.run_id), Iterator[RunEvent])
    assert_type(client.swarms.follow(run.run_id, checkpoint=StreamCheckpoint()), Iterator[RunEvent])
    try:
        client.swarms.wait(run.run_id)
    except WaitTimeoutError as error:
        assert_type(error.snapshot, RunSnapshot | None)


async def asynchronous(client: AsyncSynthClient) -> None:
    assert_type(await client.swarms.launch("Solve the task", budget="1"), RunSnapshot)
    assert_type(await client.swarms.run("Solve the task", budget="1"), RunSnapshot)
    assert_type(await client.swarms.profiles(), RunProfileCatalog)
    assert_type(await client.billing.balance(), Balance)
    assert_type(client.swarms.follow("run"), AsyncGenerator[RunEvent, None])
    async for event in client.swarms.iter_events("run"):
        assert_type(event, RunEvent)
