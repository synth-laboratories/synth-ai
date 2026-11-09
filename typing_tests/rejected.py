"""Expected failures; each marked line must produce precisely its declared error."""

from synth_ai import AsyncSynthClient, SynthClient


def invalid_calls(client: SynthClient) -> None:
    client.swarms.launch("task")  # type-error: missing-argument
    client.swarms.launch("task", budget=1.5)  # type-error: invalid-argument-type
    client.swarms.launch(
        "task",
        budget="1",
        wall_time_seconds="60",  # type-error: invalid-argument-type
    )
    client.swarms.launch(
        "task",
        budget="1",
        inputs={"files": []},  # type-error: invalid-argument-type
    )
    client.swarms.run("task", budget="1", until="typo")  # type-error: invalid-argument-type
    client.swarms.launch("task", budget="1", profile_overrides={})  # type-error: unknown-argument


async def invalid_async(client: AsyncSynthClient) -> None:
    await client.swarms.launch("task", budget=1.5)  # type-error: invalid-argument-type
    await client.swarms.run("task")  # type-error: missing-argument


def invalid_result(client: SynthClient) -> int:
    return client.swarms.get("run")  # type-error: invalid-return-type


def invalid_hook(value: int) -> None:
    pass


def invalid_diagnostics() -> None:
    AsyncSynthClient("fixture", diagnostic_hook=invalid_hook)  # type-error: invalid-argument-type
