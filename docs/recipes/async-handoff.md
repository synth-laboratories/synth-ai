# Async Research handoff

The async client is native `httpx.AsyncClient` I/O with the same contracts and
operation registry as the synchronous client. It does not proxy through threads
or dynamically typed attributes.

Start work, persist its durable identity, and let the process exit:

```python
import asyncio
from pathlib import Path

from synth_ai import AsyncSynthClient
from synth_ai.research import SwarmSpec


async def start_and_leave() -> None:
    async with AsyncSynthClient() as client:
        handle = await client.research.swarms.create(
            SwarmSpec(
                objective="Produce a bounded handoff report for the next operator.",
                timebox_seconds=1800,
            )
        )
        Path("research-swarm-id.txt").write_text(
            f"{handle.swarm_id}\n",
            encoding="utf-8",
        )
        print(handle.swarm_id)


asyncio.run(start_and_leave())
```

Reattach from another process and wait using the same native async transport:

```python
import asyncio
from pathlib import Path

from synth_ai import AsyncSynthClient
from synth_ai.research import Error, SwarmId


async def return_by_id() -> None:
    swarm_id = SwarmId(
        Path("research-swarm-id.txt").read_text(encoding="utf-8").strip()
    )
    async with AsyncSynthClient() as client:
        try:
            resolved = await client.research.swarms.configuration(swarm_id)
            print(resolved.config_version_id, resolved.snapshot_sha256)
            usage = await client.research.swarms.usage(swarm_id)
            print(usage.freshness.source, usage.freshness.as_of)
            terminal = await client.research.swarms.wait(
                swarm_id,
                timeout_seconds=1800,
                poll_interval_seconds=2,
            )
            print(terminal.state)
        except Error as error:
            if error.failure is not None:
                print(error.failure.code, error.failure.request_id)
            raise


asyncio.run(return_by_id())
```

Cancellation propagates normally: cancelling the Python task interrupts the
local wait; it does not claim that server work stopped. Request server-side
cancellation explicitly with `await client.research.swarms.cancel(swarm_id)`.

Async Factory and Effort operations have the same method and model names under
`client.research.factories` and `client.research.factories.efforts`.
