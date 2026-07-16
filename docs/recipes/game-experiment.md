# Game experiment

**Prerequisites:** `synth-ai[research]`, `SYNTH_API_KEY`, and
`SYNTH_GAME_PROJECT_ID` for a prepared project whose workspace already contains
the game environment and evaluation harness.

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

plan = research.economics.plan()
preflight = research.runs.check_preflight(project_id, work_mode=work_mode)
if plan.blocked or preflight.clear_to_trigger is not True:
    raise RuntimeError(f"experiment denied: {plan.blocked_detail}; {preflight.blockers}")

handle = research.runs.create(
    project_id,
    objective=(
        "Run the prepared game evaluation with its existing bounded rollout budget, "
        "compare outcomes, and publish the result artifact."
    ),
    work_mode=work_mode,
)
try:
    research.runs.wait(project_id, handle.run_id, timeout=3600, raise_if_failed=True)
    print(handle.artifacts.list())
    print(research.economics.run_drawdown(handle.run_id))
except (KeyboardInterrupt, TimeoutError):
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
