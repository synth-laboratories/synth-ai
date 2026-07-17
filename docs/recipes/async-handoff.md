# Async research handoff

**Prerequisites:** Install an exact published release with
`uv add "synth-ai[research]==<version>"`, then set `SYNTH_AI_VERSION` to that
same version. Set `SYNTH_API_KEY` and `SYNTH_RESEARCH_PROJECT_ID` for a prepared
project with a positive backend-owned per-run cost cap and server-side task
deadline. This recipe deliberately uses two processes: start-and-leave, then
return-by-ID.

**Duration:** About 5 minutes to integrate. The creating process does not wait;
the returning process may wait up to 30 minutes. The server-side project
deadline remains runtime authority. **Cost:** Work is billable after handoff;
both processes fail early when the project has no positive per-run cap.

Start the run, persist its ID, and close the client:

```python
import asyncio
import os
from importlib.metadata import version
from pathlib import Path

from synth_ai import SynthClient
from synth_ai.research import AsyncResearchClient, ResearchWorkMode


async def start_and_leave() -> None:
    if version("synth-ai") != os.environ["SYNTH_AI_VERSION"]:
        raise RuntimeError("installed synth-ai does not match SYNTH_AI_VERSION")

    project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]
    async_research = AsyncResearchClient(SynthClient().research)
    handle = None
    try:
        economics = await async_research.economics.project(project_id)
        run_cap_cents = next(
            (
                economics.budgets.get(key)
                for key in (
                    "run_usd_cents",
                    "per_run_usd_cents",
                    "per_run_max_usd_cents",
                )
                if economics.budgets.get(key) is not None
            ),
            None,
        )
        if (
            isinstance(run_cap_cents, bool)
            or not isinstance(run_cap_cents, int)
            or run_cap_cents <= 0
        ):
            raise RuntimeError("project has no positive backend-owned per-run cost cap")

        plan = await async_research.economics.plan()
        preflight = await async_research.runs.check_preflight(
            project_id, work_mode=ResearchWorkMode.DIRECTED_EFFORT
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
            work_mode=ResearchWorkMode.DIRECTED_EFFORT,
        )
        limits = await asyncio.to_thread(handle.resource_limits)
        wallclock = next(
            (item for item in limits.items if item.metric == "wallclock_seconds"),
            None,
        )
        if (
            wallclock is None
            or not wallclock.blocks_at_limit
            or wallclock.limit_value is None
            or not 0 < wallclock.limit_value <= 1800
        ):
            raise RuntimeError("run has no blocking server wallclock limit <= 1800 seconds")
        Path("research-run-id.txt").write_text(handle.run_id + "\n", encoding="utf-8")
        print(handle.run_id, run_cap_cents, wallclock)
    except BaseException:
        if handle is not None:
            await async_research.runs.stop(handle.run_id, project_id=project_id)
        raise
    finally:
        await async_research.close()


asyncio.run(start_and_leave())
```

Later, in a new process, reopen the run by ID and explicitly wait or stop:

```python
import asyncio
import os
from importlib.metadata import version
from pathlib import Path

from synth_ai import SynthClient
from synth_ai.research import AsyncResearchClient


async def return_by_id() -> None:
    if version("synth-ai") != os.environ["SYNTH_AI_VERSION"]:
        raise RuntimeError("installed synth-ai does not match SYNTH_AI_VERSION")

    project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]
    run_id = Path("research-run-id.txt").read_text(encoding="utf-8").strip()
    action = os.environ.get("SYNTH_HANDOFF_ACTION", "wait")
    if action not in {"wait", "stop"}:
        raise ValueError("SYNTH_HANDOFF_ACTION must be wait or stop")

    async_research = AsyncResearchClient(SynthClient().research)
    handle = None
    try:
        economics = await async_research.economics.project(project_id)
        run_cap_cents = next(
            (
                economics.budgets.get(key)
                for key in (
                    "run_usd_cents",
                    "per_run_usd_cents",
                    "per_run_max_usd_cents",
                )
                if economics.budgets.get(key) is not None
            ),
            None,
        )
        if (
            isinstance(run_cap_cents, bool)
            or not isinstance(run_cap_cents, int)
            or run_cap_cents <= 0
        ):
            raise RuntimeError("project has no positive backend-owned per-run cost cap")

        handle = await async_research.runs.open(project_id, run_id)
        limits = await asyncio.to_thread(handle.resource_limits)
        wallclock = next(
            (item for item in limits.items if item.metric == "wallclock_seconds"),
            None,
        )
        if (
            wallclock is None
            or not wallclock.blocks_at_limit
            or wallclock.limit_value is None
            or not 0 < wallclock.limit_value <= 1800
        ):
            raise RuntimeError("run has no blocking server wallclock limit <= 1800 seconds")
        progress = await asyncio.to_thread(handle.progress.get_typed)
        print(progress.public_state, run_cap_cents, wallclock)
        if action == "stop":
            await async_research.runs.stop(run_id, project_id=project_id)
            return

        final = await async_research.runs.wait(
            project_id, run_id, timeout=1800, raise_if_failed=True
        )
        work_products = await asyncio.to_thread(handle.work_products.list)
        if not work_products:
            raise RuntimeError("terminal handoff run published no WorkProduct")
        reports = [
            item
            for item in work_products
            if item.kind in {"report", "research_report"}
        ]
        if not reports:
            raise RuntimeError("terminal handoff run published no report WorkProduct")
        report = reports[0]
        report_text = await asyncio.to_thread(
            handle.work_products.content.get, report.work_product_id, as_text=True
        )
        drawdown = await async_research.economics.run_drawdown(run_id)
        print(final.public_state.value)
        print(report_text)
        print(drawdown)
    except BaseException:
        if handle is not None:
            await async_research.runs.stop(run_id, project_id=project_id)
        raise
    finally:
        await async_research.close()


asyncio.run(return_by_id())
```

**Output:** A persisted run ID and `SmrResourceLimit` from the first process,
then `ResearchRunProgress`, an explicit stop-or-wait decision,
`ResearchWorkProduct`, readable `str` content, the exact integer cost cap, and
`ResearchBillingDrawdown` from the second. **Recovery:** Re-run only the return step with the same
ID. Any failure after reopening requests stop before it is re-raised.
**Cleanup:** `stop` is explicit; successful `wait` reaches terminal state. The
client sessions close in both processes, while the prepared project is retained.
