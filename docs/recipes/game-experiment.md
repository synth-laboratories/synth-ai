# Game experiment

**Prerequisites:** `synth-ai[research]`, `SYNTH_API_KEY`, and
`SYNTH_GAME_PROJECT_ID` for a prepared project whose workspace already contains
the game environment and an evaluation harness bounded to finish within 30 minutes.

**Duration:** About 10 minutes of setup; experiment runtime depends on the
harness and rollout budget. **Cost:** Game rollouts and model calls are billable.
Check the canonical plan and run drawdown; do not assume an allowance makes the
experiment free.

```python
import os

from synth_ai import SynthClient
from synth_ai.research import ResearchWorkMode

project_id = os.environ["SYNTH_GAME_PROJECT_ID"]
research = SynthClient().research
work_mode = ResearchWorkMode.DIRECTED_EFFORT

handle = None
try:
    plan = research.economics.plan()
    preflight = research.runs.check_preflight(project_id, work_mode=work_mode)
    clear_to_trigger = preflight.get("clear_to_trigger", preflight.get("allowed"))
    if plan.blocked or clear_to_trigger is not True:
        raise RuntimeError(
            f"experiment denied: {plan.blocked_detail}; {preflight.get('blockers')}"
        )

    handle = research.runs.create(
        project_id,
        objective=(
            "Run the prepared game evaluation with its existing bounded rollout budget, "
            "compare outcomes, and publish the result artifact."
        ),
        work_mode=work_mode,
    )
    research.runs.wait(project_id, handle.run_id, timeout=1800, raise_if_failed=True)
    print(handle.artifacts.list())
    print(research.economics.run_drawdown(handle.run_id))
except (KeyboardInterrupt, TimeoutError):
    if handle is not None:
        research.runs.stop(handle.run_id, project_id=project_id)
    raise
finally:
    research.close()
```

**Output:** Typed artifact metadata plus the canonical run drawdown. **Recovery:**
Use `handle.progress.get_typed()` and artifact metadata to diagnose an incomplete
run; change the prepared harness or budget explicitly before retrying.
**Cleanup:** The exception path requests a graceful stop. The recipe does not
delete the reusable game project or its evidence.
