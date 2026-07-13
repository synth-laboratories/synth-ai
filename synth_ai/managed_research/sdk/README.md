# SDK

This subtree owns the Python control-plane client and the typed namespace wrappers built on top of it.

Surface note: this SDK targets the authenticated private-beta Managed Research
API. Managed Research beta access is an account/org entitlement enforced by the
backend through entitlement checks and launch preflight rather than by a
separate SDK client fork.

**Hosted inference (prod):** Synth defaults to **Zero Data Retention (ZDR)** on
all provider routes, **Western-domiciled** inference (**US at launch**; Canada
and EU later), and **`(provider, model)`** pricing. Orgs may opt out of ZDR when
using models that do not support it. **Local testing and evals** may use routes
not allowed in prod (e.g. direct DeepSeek API). See the root
[`README.md`](../../../../README.md) § Hosted inference policy.

Ownership:
- `client.py` owns transport-facing request building and raw backend interaction
- namespace modules such as `progress.py`, `runs.py`, and `workspace_inputs.py` own higher-level typed return surfaces

Guidelines:
- parse and validate request inputs before dispatch
- keep request models typed until the final JSON serialization edge
- avoid heuristic payload-shape probing
- when a backend route has a stable response concept, return a typed model from the namespace API when practical

Current typed namespace returns:
- [`project.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/sdk/project.py)
  - bound project setup via `client.project(id).setup.get()` and
    `client.project(id).setup.prepare()`
  - bound project launch preflight via `client.project(id).runs.preflight(...)`
  - typed run reads via `client.project(id).runs.get(run_id)`
  - trained-model result helpers via `client.project(id).models.*`
  - `ProjectWorkspaceProjection` via `client.project(id).workspace()`, including
    actor/event/context-pack/changeset/canon-change/next-action readouts
  - review-gated project ChangeSets via `client.project(id).changesets.*`
  - project-run actor controls via `client.project(id).runs.pause_actor(...)`,
    `resume_actor(...)`, and `interrupt_actor(...)`
- [`progress.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/sdk/progress.py)
  - `ProjectSetupAuthority` via `get_project_setup_authority(...)`
  - `LaunchPreflight` via `get_launch_preflight(...)`
- [`runs.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/sdk/runs.py)
  - `ManagedResearchRun` via `get(run_id, project_id=...)`
  - `SmrLogicalTimeline` via `get_logical_timeline(project_id, run_id)`
  - `SmrRunBranchResponse` via `branch_from_checkpoint(...)`
- [`workspace_inputs.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/sdk/workspace_inputs.py)
  - `WorkspaceInputsState`
  - `WorkspaceUploadResult`

Noun-first inspection belongs on the run/project namespaces and direct client
reads such as `get_run(...)`, `get_project_workspace(...)`,
`list_objectives(...)`, `list_run_objective_events(...)`,
`get_run_work_graph(...)`, and `list_run_questions(...)`.

Wire-shaped helpers remain on `SmrControlClient` where MCP and lower-level callers need backend-shaped payloads.

Noun-first namespaces now mirror the customer surface:

- org-scoped setup: `client.github`, `client.credentials`, `client.exports`
- project-scoped work/results/status: `client.project(id).repos`, `.datasets`,
  `.files`, `.prs`, `.models`, `.outputs`, and `.readiness()`

Contract posture:

- backend route and schema shape stay authoritative through
  `/Users/joshpurtell/Documents/GitHub/backend/smr_openapi.yaml`
- backend-to-SDK drift is checked with
  `/Users/joshpurtell/Documents/GitHub/backend/scripts/validate_smr_openapi.py`
- older names may remain as wrappers, but new noun behavior belongs only on the
  flat namespaces above

Examples:

```python
timeline = client.runs.get_logical_timeline("proj_123", "run_123")

exact_branch = client.runs.branch_from_checkpoint(
    "run_123",
    project_id="proj_123",
    checkpoint_id="ckpt_123",
)

seeded_branch = client.runs.branch_from_checkpoint(
    "run_123",
    project_id="proj_123",
    checkpoint_id="ckpt_123",
    mode="with_message",
    message="Retry this from the checkpoint, but explain the regression first.",
)
```

Four-model launch picker:

```python
from synth_ai import SynthClient
from synth_ai.managed_research import (
    RoleBinding,
    SmrAgentModel,
    SmrRoleBindings,
    WorkerRolePalette,
)

client = SynthClient()

models = client.research.projects.get_agent_models()
available = {model["id"]: model for model in models["models"]}

selection = SmrAgentModel.GPT_5_4_MINI
params = {"reasoning_effort": "medium"}

run = client.research.runs.start(
    project_id="proj_123",
    agent_model=selection,
    agent_model_params=params,
    roles=SmrRoleBindings(
        orchestrator=RoleBinding(model=selection, params=params),
        reviewer=RoleBinding(model=selection, params=params),
        worker=WorkerRolePalette(
            default=RoleBinding(model=selection, params=params),
        ),
    ),
)
```

The launch UI exposes the backend-owned public subset:
`baseten/zai-org/GLM-5.2`, `gpt-5.4-mini` with medium or high reasoning, and
`gpt-5.5` with low reasoning when the backend catalog reports that effort.
