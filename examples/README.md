# SDK recipes

[`sdk_workflows.py`](sdk_workflows.py) contains small async functions that accept
an existing `AsyncSynthClient`. Importing it performs no network operations.
The synchronous facade exposes the same operations and typed lazy iterators.
Run every recipe against deterministic wire fixtures with:

```sh
uv run pytest tests/test_examples.py
```

Use an explicit API key or the SDK's configured environment for live calls:

```python
from synth_ai import AsyncSynthClient
from examples.sdk_workflows import required_budget

async with AsyncSynthClient() as client:
    run = await required_budget(client, "Summarize the supplied evidence")
```

| Function | Behavior |
| --- | --- |
| `required_budget` | Explicit $0.10 cap and 60-second execution limit |
| `select_profile` | Discover availability, select a matching published profile |
| `constrained_routing` | Hard provider only-list and model exclude-list |
| `wait_result_then_finished` | Result availability followed by final accounting/cleanup |
| `cancel_after_timeout` | Explicit application cancellation after observation timeout |
| `restartable_stream` | Apply each event before persisting its received checkpoint |
| `recover_upload` | Reuse upload ID and idempotency key after uncertain transfer |
| `verified_artifact` | Pin revision/digest and atomically complete the download |
| `usage_readback` | Read Core's settled usage, pending and settling amounts |

For event processing that must survive crashes atomically, store the application
state and checkpoint in one transaction. A checkpoint is receipt state; it is
not an acknowledgement that user code durably processed an event. Handlers must
tolerate replay after application succeeds but checkpoint persistence fails.

Migration: remove `program`, `method`, and `profile_overrides`; select `profile`
from discovery and supply `research_task` only when it is required. Remove
`maximum_tasks` and `maximum_parallel_tasks`; supply `budget` and optional
`wall_time_seconds` and `parallel_workers`. Use `RunSnapshot.finished` for finality and
`result_available` for output availability. A timeout does not cancel a run.
Unknown future profile IDs and run-state strings stay representable.

Live calls require the compatible Core cohort pinned in `contracts/SOURCE.toml`.
The 0.31 cohort exposes a 32 KiB UTF-8 message limit; older server limits are not
inferred from the Python package version. Publication and live release
acceptance are separate from these offline recipes.
