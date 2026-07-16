# First bounded research run

**Prerequisites:** `synth-ai[research]`, `SYNTH_API_KEY`, and
`SYNTH_RESEARCH_PROJECT_ID` for an existing prepared project.

**Duration:** About 5 minutes of setup; the 15-minute wait bound below is not a
runtime guarantee. **Cost:** This launches billable work. Review the returned
plan and preflight before continuing; neither is a price quote.

```python
import os

from synth_ai import SynthClient
from synth_ai.research import ResearchWorkMode

project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]
research = SynthClient().research
work_mode = ResearchWorkMode.DIRECTED_EFFORT

plan = research.economics.plan()
preflight = research.runs.check_preflight(project_id, work_mode=work_mode)
if plan.blocked or preflight.clear_to_trigger is not True:
    raise RuntimeError(f"launch denied: {plan.blocked_detail}; {preflight.blockers}")

handle = research.runs.create(
    project_id,
    objective="Inspect the repository and produce a bounded findings report.",
    work_mode=work_mode,
)
try:
    final = research.runs.wait(
        project_id, handle.run_id, timeout=900, raise_if_failed=True
    )
    print(final.public_state.value)
    print(handle.work_products.list())
    print(research.economics.run_drawdown(handle.run_id))
except (KeyboardInterrupt, TimeoutError):
    research.runs.stop(handle.run_id, project_id=project_id)
    raise
finally:
    research.close()
```

**Output:** Final public run state, typed WorkProduct metadata, and canonical
billing drawdown. **Recovery:** Correct preflight blockers before retrying; on a
timeout, inspect `handle.progress.get_typed()` and resume or launch deliberately.
**Cleanup:** The exception path requests a graceful stop. The pre-existing
project is intentionally left intact.
