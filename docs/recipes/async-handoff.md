# Async handoff

**Prerequisites:** `synth-ai[research]`, `SYNTH_API_KEY`, and
`SYNTH_RESEARCH_PROJECT_ID` for an existing prepared project. This uses the
public async adapter over the same `SynthClient().research` session.

**Duration:** About 5 minutes to integrate; the example waits at most 30 minutes.
**Cost:** The handed-off run is billable while it executes. Read the canonical
plan and drawdown rather than assuming free use.

```python
import asyncio
import os

from synth_ai import SynthClient
from synth_ai.research import AsyncResearchClient, ResearchWorkMode


async def main() -> None:
    project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]
    research = SynthClient().research
    async_research = AsyncResearchClient(research)
    work_mode = ResearchWorkMode.DIRECTED_EFFORT

    handle = None
    try:
        plan = await async_research.economics.plan()
        preflight = await async_research.runs.check_preflight(
            project_id, work_mode=work_mode
        )
        clear_to_trigger = preflight.get(
            "clear_to_trigger", preflight.get("allowed")
        )
        if plan.blocked or clear_to_trigger is not True:
            raise RuntimeError(
                f"handoff denied: {plan.blocked_detail}; {preflight.get('blockers')}"
            )

        handle = await async_research.runs.create(
            project_id,
            objective="Produce a bounded handoff report for the next operator.",
            work_mode=work_mode,
        )
        final = await async_research.runs.wait(
            project_id, handle.run_id, timeout=1800, raise_if_failed=True
        )
        drawdown = await async_research.economics.run_drawdown(handle.run_id)
        print(final.public_state.value, drawdown)
    except (asyncio.CancelledError, TimeoutError):
        if handle is not None:
            await async_research.runs.stop(handle.run_id, project_id=project_id)
        raise
    finally:
        await async_research.close()


asyncio.run(main())
```

**Output:** A terminal typed run state and canonical billing drawdown available
to the awaiting coroutine. **Recovery:** Cancellation requests a graceful stop;
inspect the run through a new `SynthClient().research.runs.get(...)` handle before
deciding to resume. **Cleanup:** The async adapter closes the shared research
session, while the pre-existing project remains intact.
