---
name: synth-smr-control
description: Control Synth Managed Research (SMR) projects and runs through Synth API-compatible SDK patterns. Use when asked to create or configure SMR projects, run onboarding and dry runs, trigger/pause/resume/stop runs, answer questions, resolve approvals, inspect artifacts/usage/ops status, or automate these flows from Python.
---

# Synth SMR Control

## Overview

Use this skill to operate SMR from code with predictable, scriptable API calls.
Prefer the SDK client in `synth_ai.sdk.managed_research` for deterministic behavior and route compatibility fallbacks.

## Workflow

1. Read `references/api-surface.md` for the route contract and known compatibility drifts.
2. Use `synth_ai.sdk.managed_research.SmrControlClient` as the default client surface instead of hand-writing ad-hoc curl commands.
3. Start from project-level operations (`create/get/patch/pause/resume/trigger`) and then move to run-level operations.
4. For run-scoped reads, try project-scoped routes first and fallback to canonical run routes when the backend does not expose aliases.
5. When uploading provider keys, prefer encrypted payloads if accepted; fallback to plaintext `api_key` only when required by the backend contract.
6. For data-bound runs, upload files through `starting-data/upload-urls` (via SDK `upload_starting_data_files` helpers) and set `execution.input_spec` explicitly before triggering.

## Quick Start

```python
import os

from synth_ai.sdk.managed_research import SmrControlClient

project_id = "<project-id>"

with SmrControlClient(api_key=os.environ["SYNTH_API_KEY"]) as client:
    project = client.get_project(project_id)
    status = client.get_project_status(project_id)
    print(project["name"], status["state"])
```

## Common Tasks

### Trigger a run

```python
run = client.trigger_run(project_id, timebox_seconds=8 * 60 * 60)
print(run["run_id"], run["state"])
```

### Upload Banking77 starting data

```python
client.upload_starting_data_directory(
    project_id,
    "examples/managed_research/banking77_starting_data",
    dataset_ref="starting-data/banking77",
)
```

### List active runs (compat-safe)

```python
active = client.list_active_runs(project_id)
for run in active:
    print(run["run_id"], run.get("state"))
```

### Resolve approvals and questions

```python
approvals = client.list_project_approvals(project_id, status_filter="pending")
questions = client.list_project_questions(project_id, status_filter="pending")

# Example decisions
# client.approve(run_id, approval_id, comment="Approved for launch")
# client.respond_question(run_id, question_id, response_text="Use dataset v2")
```

## Output Expectations

When using this skill, return:

- the exact project/run IDs touched,
- endpoint-level actions taken,
- any fallback behavior used (project-scoped route -> canonical route),
- and unresolved blockers (auth, onboarding, missing secrets, entitlement, or route mismatch).

## Safety Rules

- Never log plaintext provider keys.
- Do not call internal worker-only endpoints (`/smr/internal/*`) for customer/operator control tasks.
- Treat `pause`, `resume`, `stop`, `approve`, and `deny` as side-effecting operations; confirm target IDs before execution.

## Resources

- API route contract and drift notes: `references/api-surface.md`
- Python wrapper: `scripts/smr_control.py`
- SDK client: `synth_ai.sdk.managed_research`
