# First bounded research run

**Prerequisites:** Install an exact published release with
`uv add "synth-ai[research]==<version>"`, then set `SYNTH_AI_VERSION` to that
same version. Set `SYNTH_API_KEY` and `SYNTH_RESEARCH_PROJECT_ID` for an
existing prepared README/ReportBench-style project. The project must have a
backend-owned per-run cost cap and a server-side task deadline.

**Duration:** About 5 minutes of setup; the server-side project deadline is the
runtime authority. The 15-minute client wait below does not terminate server
work by itself. **Cost:** This launches billable work and fails early unless the
project economics response contains a positive per-run USD-cent cap.

```python
import os
from importlib.metadata import version

from synth_ai import SynthClient
from synth_ai.research import ResearchWorkMode

expected_version = os.environ["SYNTH_AI_VERSION"]
if version("synth-ai") != expected_version:
    raise RuntimeError("installed synth-ai does not match SYNTH_AI_VERSION")

project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]
research = SynthClient().research
work_mode = ResearchWorkMode.DIRECTED_EFFORT

handle = None
try:
    economics = research.economics.project(project_id)
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

    plan = research.economics.plan()
    preflight = research.runs.check_preflight(project_id, work_mode=work_mode)
    clear_to_trigger = preflight.get("clear_to_trigger", preflight.get("allowed"))
    if plan.blocked or clear_to_trigger is not True:
        raise RuntimeError(
            f"launch denied: {plan.blocked_detail}; {preflight.get('blockers')}"
        )

    handle = research.runs.create(
        project_id,
        objective="Inspect the repository and publish one bounded findings report.",
        work_mode=work_mode,
    )
    limits = handle.resource_limits()
    wallclock = next(
        (item for item in limits.items if item.metric == "wallclock_seconds"),
        None,
    )
    if (
        wallclock is None
        or not wallclock.blocks_at_limit
        or wallclock.limit_value is None
        or not 0 < wallclock.limit_value <= 900
    ):
        raise RuntimeError("run has no blocking server wallclock limit <= 900 seconds")
    final = research.runs.wait(
        project_id, handle.run_id, timeout=900, raise_if_failed=True
    )
    work_products = handle.work_products.list()
    if not work_products:
        raise RuntimeError("terminal run published no WorkProduct")
    reports = [
        item for item in work_products if item.kind in {"report", "research_report"}
    ]
    if not reports:
        raise RuntimeError("terminal run published no report WorkProduct")
    report = reports[0]
    report_text = handle.work_products.content.get(report.work_product_id, as_text=True)
    print(final.public_state.value)
    print(report_text)
    print(run_cap_cents, wallclock)
    print(research.economics.run_drawdown(handle.run_id))
except BaseException:
    if handle is not None:
        research.runs.stop(handle.run_id, project_id=project_id)
    raise
finally:
    research.close()
```

**Output:** `ResearchRun`, `ResearchWorkProduct`, readable `str` content,
`SmrResourceLimit`, the exact integer cost cap, and `ResearchBillingDrawdown`.
**Recovery:** Correct preflight or budget blockers before
retrying. Any exception after launch requests a graceful stop before it is
re-raised; inspect the run by ID if that stop itself fails. **Cleanup:** The run
is terminal on success. Failure paths request stop, and the pre-existing project
is intentionally retained.
