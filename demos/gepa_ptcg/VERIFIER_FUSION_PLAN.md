# Verifier Fusion Plan: PTCG Gameplay Eval + Rubrics Consolidation Refactor

This document covers **two related refactors**:

1. **Part 1**: Verifier fusion for PTCG gameplay evals (task reward + zero-shot verifier + fused reward)
2. **Part 2**: Rubrics consolidation (delete `TaskInfo.rubric`, single source via `/info`)

Both changes tie together because they clean up how rubrics flow from task apps to the backend verifier.

---

## Part 1: Verifier Fusion for PTCG Gameplay Eval

### Goal

Run **headless gameplay evals** for `demos/gepa_ptcg` where:

- The **task app** computes the canonical task reward (win/loss or outcome reward).
- The Synth backend runs a **zero-shot verifier** that evaluates **gameplay quality** from the hydrated trace against a rubric.
- The backend **fuses** task reward + verifier reward into a single per-seed `reward`.

This is intended to work with the existing **eval job** flow (not prompt-learning/GEPA optimization yet).

---

### Goal / Non-goals

- **Goal**: In eval jobs, record *both*:
  - `outcome_reward` (task app, who won)
  - `verifier_reward` (zero-shot rubric, gameplay quality)
  - `reward` (fused)
- **Goal**: Reuse **interceptor hydration** for traces (avoid building v3 traces in task app).
- **Non-goal**: Modify `monorepo/specs` (explicitly forbidden).
- **Non-goal**: Embed LLM calls into Rust servers / UI harness. This is headless LocalAPI.

---

### Current state (what already exists)

#### 1) Eval job pipeline supports verifier evaluation

The backend eval job service supports an optional `verifier_config` in the eval request.
If present and enabled, it computes a per-seed `verifier_reward` and then fuses it into the final `reward`.

#### 2) Verifier endpoint exists and is zero-shot capable

Backend route: `POST /api/graphs/verifiers/completions`

It supports built-in zero-shot verifier graph IDs:
- `zero_shot_verifier` (auto routing)
- `zero_shot_verifier_rubric_single`, `..._mapreduce`, `..._rlm`, etc.

#### 3) Trace hydration exists for eval jobs

Eval jobs already hydrate v3 traces from the interceptor store and normalize rollouts to v3 traces for verifier evaluation.
This means task apps do **not** need to build tracing-v3 `event_history` manually, as long as:
- LLM calls go through the interceptor (which we already do in `gepa_ptcg`).

---

### Proposed reward semantics

We want:

- **Task reward** (from task app / LocalAPI): \( r_{local} \in [0, 1] \)
  - Example: win=1, loss=0, draw=0.5, etc.
- **Verifier reward** (from backend verifier): \( r_{verifier} \in [0, 1] \)
  - A rubric-based "gameplay quality" reward independent of win/loss
- **Fused reward**:

\[
r_{final} = w_{local}\cdot r_{local} + w_{verifier}\cdot r_{verifier}
\]

Example weights:
- `w_env=0.5`, `w_verifier=0.5` (simple balanced)

---

### Reward semantics: double-counting risk (CRITICAL)

#### The problem

The verifier pipeline can optionally accept the task app reward as input (we will call it `local_api_reward`):

```python
# backend/app/routes/eval/scoring.py (current)
verifier_result = await verifier.reward_trajectory(
    ...,
    local_api_reward=local_api_rewards.get(seed),  # task app reward passed in
)
```

Depending on the verifier graph implementation, the verifier may:
- **Ignore** `local_api_reward` and evaluate purely on trace + rubric (desired for independent fusion)
- **Incorporate** `local_api_reward` into its output (causes double-counting when fused again)

If the verifier incorporates `env_reward`, and then the eval job service fuses:

```
reward = w_local * local_api_reward + w_verifier * verifier_reward
```

…the task reward gets counted **twice**: once inside `verifier_reward`, once in the outer fusion.

#### The fix (required monorepo change)

To guarantee no double-counting, we change the eval verifier evaluation call to:

```python
# backend/app/routes/eval/scoring.py (proposed)
verifier_result = await verifier.reward_trajectory(
    ...,
    local_api_reward=None,  # <-- do NOT pass task reward into verifier
)
```

This ensures the verifier evaluates only the trace + rubric, and fusion happens exactly once in the eval job service.

#### Code change

Single line change in `backend/app/routes/eval/scoring.py`:

```diff
- local_api_reward=local_api_rewards.get(seed),
+ local_api_reward=None,
```

---

### Gameplay-quality rubric (initial draft)

We want a generic rubric that works for "agent plays a turn-based game":

- **Event criteria** (local action quality):
  - legality / prompt following (choose only allowed actions)
  - progress / avoid stalling
  - attack / advance board when beneficial
  - resource management (energy attach once per turn, don't waste)
- **Outcome criteria** (global quality):
  - win or create advantage
  - avoid obvious blunders

These criteria must be expressed in the backend's expected rubric format:
- `rubric.event`: list of dict criteria
- `rubric.outcome`: list of dict criteria

---

### Verifier configuration (eval request)

We will pass `verifier_config` in the eval job request (SDK: `EvalJobConfig.verifier_config`).

Example:

```json
{
  "enabled": true,
  "reward_source": "fused",
  "backend_base": "http://localhost:8000",
  "backend_api_key": "<SYNTH_API_KEY>",
  "verifier_graph_id": "zero_shot_verifier_rubric_single",
  "backend_provider": "openai",
  "backend_model": "gpt-4.1-mini",
  "backend_event_enabled": true,
  "backend_outcome_enabled": true,
  "concurrency": 4,
  "weight_env": 0.5,
  "weight_event": 0.0,
  "weight_outcome": 0.5
}
```

Notes:
- We fuse only outcome-level verifier reward into final reward initially (keep it simple).
- If event-level evaluation returns per-event totals, we can later add `weight_event > 0`.

---

### Implementation steps (Part 1)

#### A) Task app: expose a gameplay-quality rubric

- Add a `RubricBundle` to `demos/gepa_ptcg/localapi_ptcg.py` via `LocalAPIConfig(rubrics=...)`
- Ensure `/info` returns `rubrics` for the backend to use.

#### B) Eval runner: pass verifier_config

- Update `demos/gepa_ptcg/run_demo.py` to populate `EvalJobConfig.verifier_config` with:
  - `verifier_graph_id="zero_shot_verifier_rubric_single"` (or `zero_shot_verifier`)
  - `reward_source="fused"`
  - weights as above

#### C) Backend: enforce independence / prevent double-counting

- Make the single-line change in `backend/app/routes/eval/scoring.py` (`env_reward=None`).

---

### Validation / acceptance criteria (Part 1)

For a small local run (e.g. 5 seeds) we should see:
- Per seed result row contains:
  - `local_api_reward` (task app)
  - `verifier_reward` (non-null for most seeds)
  - `reward` differs from both and matches the configured fusion weights
- Backend logs show:
  - trace hydration succeeded
  - verifier endpoint calls succeeded (200)

Failure modes and what they mean:
- **verifier_reward is null**: rubric missing/unparseable, verifier_graph_id wrong, trace hydration missing, or verifier endpoint errors.
- **verifier_reward correlates too strongly with win**: likely env_reward leaked into verifier evaluation or rubric is too outcome-focused.

Note: this doc uses `local_api_reward` / `verifier_reward` naming. Existing persisted fields in the backend may still be
named `outcome_reward` / `verifier_reward` until the backend refactor lands; the intent is:
- `outcome_reward` == `local_api_reward`
- `verifier_reward` == `verifier_reward` (canonical naming)

---

### Open questions (Part 1)

1) **Verifier graph id**: confirm which one to use:
   - `zero_shot_verifier_rubric_single` (fast)
   - `zero_shot_verifier_rubric_mapreduce` (slower, potentially higher quality)
   - `zero_shot_verifier` (auto routes)
2) **Judge model** for the verifier:
   - keep cheap (`gpt-4.1-nano` / `gpt-5-nano`) vs better (`gpt-4.1-mini`)
3) **Fusion weights**:
   - start 50/50 or bias toward win (e.g. 0.7 env / 0.3 verifier)?

---

## Part 2: Rubrics Consolidation Refactor (Breaking Change)

### Background

The SDK currently has **two ways** for task apps to advertise rubrics:

| Path | Location | Status | How backend consumes |
|------|----------|--------|---------------------|
| **Legacy** | `TaskInfo.rubric` (returned per-seed from `/task_info?seed=...`) | `[DEPRECATED]` in contracts | Backend fetches `/task_info`, extracts `.rubric`, normalizes into verifier payload |
| **Modern** | `LocalAPIConfig.rubrics` (exposed via `GET /info` as `rubrics.outcome` / `rubrics.events`) | Canonical, preferred | Backend fetches `/info`, reads `rubrics` bundle, uses directly |

The legacy path exists because early task apps populated `TaskInfo.rubric`. The modern path was added to decouple per-instance metadata from global rubric definitions.

**Problem**: Having two sources creates:
- Maintenance burden (backend must merge/fallback)
- Confusion (which one is authoritative?)
- Fragile normalization code (handles lists, dicts, Pydantic models, etc.)

### Decision: Single Source via `/info`

**Canonical source**: `GET /info` → `rubrics` field (a `RubricBundle` with `.outcome` and `.events`).

**Remove**: `TaskInfo.rubric` field entirely from the SDK contracts and backend consumption logic.

**No grace period. No fallback. The deprecated field has been marked long enough — time to rip off the bandaid.**

---

### Canonical rubric data model (synth-ai SDK)

The rubric data model lives in `synth_ai.sdk.task.rubrics`:

```python
# synth_ai/sdk/task/rubrics/models.py

class Criterion(BaseModel):
    id: str                    # unique criterion identifier
    description: str           # what this criterion evaluates
    weight: float = 1.0        # relative importance (must be > 0)
    required: bool = False     # if True, failing this criterion fails the rubric

class Rubric(BaseModel):
    version: str = "1.0"
    goal_text: str | None = None  # high-level goal description (optional)
    criteria: list[Criterion] = []  # list of evaluation criteria
    aggregation: str = "weighted_sum"  # how to combine criterion rewards (options: "sum", "weighted_sum", "custom", "inherit")

# synth_ai/sdk/task/server.py

class RubricBundle(BaseModel):  # MUST be BaseModel (not @dataclass) for OpenAPI compatibility
    """Rubric bundle exposed via /info endpoint. Must be Pydantic BaseModel for OpenAPI schema generation."""
    outcome: Rubric | None = None   # outcome-level rubric (end of rollout)
    events: Rubric | None = None    # event-level rubric (per action/step)
```

Task apps expose rubrics via `LocalAPIConfig(rubrics=RubricBundle(...))`, which the SDK serves from `GET /info` as:

```json
{
  "rubrics": {
    "outcome": { "version": "1.0", "goal_text": "...", "criteria": [...] },
    "events": { "version": "1.0", "goal_text": "...", "criteria": [...] }
  }
}
```

#### How to assemble a rubric (complete example)

```python
from synth_ai.sdk.task.rubrics import Criterion, Rubric
from synth_ai.sdk.task.server import LocalAPIConfig, RubricBundle

# Step 1: Define criteria
legal_action_criterion = Criterion(
    id="legal_actions",
    description="Agent only chooses legal actions from available_actions",
    weight=1.0,
    required=True,  # Fail fast on illegal actions
)

strategic_play_criterion = Criterion(
    id="strategic_play",
    description="Agent makes strategic decisions (attacks when beneficial, manages energy)",
    weight=0.8,
    required=False,
)

avoid_stalling_criterion = Criterion(
    id="avoid_stalling",
    description="Agent progresses the game state and avoids infinite loops",
    weight=0.6,
    required=False,
)

# Step 2: Create a Rubric with criteria
gameplay_rubric = Rubric(
    version="1.0",
    goal_text="Evaluate Pokemon TCG gameplay quality",
    criteria=[
        legal_action_criterion,
        strategic_play_criterion,
        avoid_stalling_criterion,
    ],
    aggregation="weighted_sum",  # Options: "sum", "weighted_sum", "custom", "inherit"
)

# Step 3: Create a RubricBundle (can include both outcome and events rubrics)
rubric_bundle = RubricBundle(
    outcome=gameplay_rubric,  # Evaluated at end of rollout
    events=None,  # Optional: per-action rubric for step-wise evaluation
)

# Step 4: Pass to LocalAPIConfig
def build_config() -> LocalAPIConfig:
    return LocalAPIConfig(
        app_id="gepa_ptcg",
        name="Pokemon TCG Gameplay",
        description="Headless Pokemon TCG gameplay evaluation",
        provide_taskset_description=lambda: {"splits": ["train"]},
        provide_task_instances=lambda seeds: [...],
        rollout=run_rollout,
        rubrics=rubric_bundle,  # <-- Rubrics go here
    )
```

**Notes:**
- **`Criterion`**: Each criterion has an `id` (unique), `description` (what it evaluates), `weight` (relative importance, must be > 0), and `required` (if True, failing this fails the entire rubric).
- **`Rubric`**: Contains a list of criteria, optional `goal_text` (high-level description), and `aggregation` method (how to combine criterion rewards).
- **`RubricBundle`**: Contains `outcome` (evaluated at end of rollout) and/or `events` (evaluated per action/step). At least one must be provided.
- **Weights**: Don't need to sum to 1.0 (flexible model). The verifier will normalize as needed.
- **Required criteria**: If `required=True`, failing that criterion causes the entire rubric to fail (useful for "must be legal" checks).

**OpenAPI compatibility (REQUIRED for Rust/non-Python clients):**

All contract/boundary data classes must be OpenAPI-compatible so users can generate clients in Rust, TypeScript, etc.

- ✅ **`Criterion` and `Rubric`**: Already Pydantic `BaseModel` → OpenAPI-compatible
- ❌ **`RubricBundle`**: Currently a `@dataclass` → **MUST be converted to Pydantic `BaseModel`**
- ❌ **`/info` endpoint**: Currently returns `Mapping[str, Any]` → **MUST have typed response model**

**Required changes:**
1. Convert `RubricBundle` from `@dataclass` to Pydantic `BaseModel` (enables automatic OpenAPI schema generation)
2. Create `InfoResponse` Pydantic model for `/info` endpoint with typed `rubrics` field
3. Update `/info` endpoint to use `response_model=InfoResponse`

This ensures FastAPI generates complete OpenAPI schema that Rust/TypeScript clients can consume.

---

### Scope of changes

#### Inventory (audit of current references)

This is an explicit audit so we don’t miss any callers during the refactor. As of 2026-01-13:

**synth-ai** (task-app side)
- `demos/gepa_ptcg/localapi_ptcg.py`: currently constructs `RubricInfo` and attaches it to `TaskInfo(rubric=...)`.
- `demos/gepa_crafter_vlm/demo_crafter_react.py`: sets `TaskInfo(..., rubric={...})` (legacy pattern).
- `demos/web-design/web_design_task_app.py`: uses `RubricInfo` in `TaskInfo(rubric=RubricInfo(...))` (line 239).
- `demos/web-design/run_demo.py`: imports and uses `RubricInfo`, `RubricCriterion`, `RubricSection` (lines 54, 57, 567).
- `synth_ai/sdk/task/contracts.py`: defines `RubricInfo` and `TaskInfo.rubric` (deprecated).
- `synth_ai/sdk/task/__init__.py`: re-exports `RubricInfo` (and related legacy symbols like `RubricCriterion`, `RubricSection`).
- `synth_ai/sdk/task/localapi_template.py`: template yields `TaskInfo(...)` (ensure no rubric fields).
- `synth_ai/cli/lib/apps/task_app.py`: validates `/info` and `/task_info` and already understands `/info.rubrics`.

**monorepo** (backend side)
- `backend/app/routes/eval/job_service.py`: captures task app reward, stores `outcome_reward`, calls verifier, then fuses into `reward`.
- `backend/app/routes/eval/scoring.py`: currently passes task reward into verifier (`env_reward=...`) (double-counting risk).
- `backend/app/routes/eval/models.py` and `backend/app/routes/eval/routes.py`: expose `outcome_reward` and `verifier_reward`.
- `backend/app/routes/prompt_learning/core/verifying.py`: passes `env_reward` into `RubricPipeline.reward(...)`.
- `backend/app/routes/prompt_learning/core/rubric_pipeline.py`: consumes `task_info["rubric"]` and has fallback merge-from-`/info`; also explicitly computes an env component in the verifier reward.
- `backend/app/routes/prompt_learning/routes_online.py`: builds rubric payload from TaskInfo (`_build_rubric_payload(...)`).
- Prompt-learning optimizers rely on outcome keys:
  - `backend/app/routes/prompt_learning/algorithm/mipro/optimizer/optimizer.py`: prefers `outcome_reward`.
  - `backend/app/routes/prompt_learning/algorithm/gepa/optimizer.py`: reads/writes `verifier_reward` and `outcome_reward`.
- GEPA/GraphGen integration has separate “rubric” concepts (not LocalAPI rubrics):
  - `backend/graphs/gepa_integration/graph_evolve_job.py`: `task_metadata.get("rubric")`, `task.get("rubric")` (GraphGen tasks).
  - `backend/app/routes/graphgen/*`: uses `task.rubric` (GraphGen rubric).

Note: GraphGen’s `task.rubric` is *not* the same contract as LocalAPI `/info.rubrics`. This plan does not unify those
schemas, but we list them so reviewers know what is and isn’t being changed.

#### synth-ai SDK (breaking)

| File | Change |
|------|--------|
| `synth_ai/sdk/task/contracts.py` | **Delete** `RubricInfo` and **delete** `TaskInfo.rubric` entirely (breaking) |
| `synth_ai/sdk/task/__init__.py` | **Remove** legacy exports (`RubricInfo`, anything only supporting `TaskInfo.rubric`) |
| `synth_ai/sdk/task/server.py` | **REQUIRED for OpenAPI**: Convert `RubricBundle` from `@dataclass` to Pydantic `BaseModel`. Create `InfoResponse` Pydantic model for `/info` endpoint with typed `rubrics: RubricBundle | None` field. Update `/info` endpoint to use `response_model=InfoResponse`. This enables OpenAPI schema generation for Rust/TypeScript clients. |
| `synth_ai/sdk/task/localapi_template.py` | Ensure template does **not** set or mention `TaskInfo.rubric` |
| `synth_ai/cli/lib/apps/task_app.py` | Confirm validators treat `/info.rubrics` as canonical and do not require `/task_info` to contain rubric |
| `synth_ai/sdk/graphs/completions.py` | No change required (this is *verifier API input* rubrics, separate from LocalAPI advertising) |
| All `synth-ai/demos/*` task apps | Remove any `TaskInfo(rubric=...)` usage; move to `LocalAPIConfig(rubrics=RubricBundle(...))` |

#### monorepo backend (breaking)

| File | Change |
|------|--------|
| `backend/app/routes/prompt_learning/core/rubric_pipeline.py` | **Delete** all reads of `task_info["rubric"]` and **delete** the fallback merge-from-`/info` logic (because `/info` is now the only source). Also remove/adjust env-component logic so verifier reward can be independent when desired. |
| `backend/app/routes/prompt_learning/core/verifying.py` | Rename `env_reward` plumbing to `local_api_reward` (task app reward) and align internal naming to `verifier_reward`. |
| `backend/app/routes/prompt_learning/routes_online.py` | Remove `_build_rubric_payload(task_info)` path; fetch rubrics from `/info` only. |
| `backend/app/routes/eval/scoring.py` | **Fix double counting**: pass `local_api_reward=None` into verifier evaluation for eval jobs. |
| `backend/app/routes/eval/job_service.py` | Rename internal variables to `local_api_reward` / `verifier_reward` and ensure persisted results store both separately. |
| `backend/app/routes/eval/models.py` | Add canonical fields (or aliases) so API exposes `local_api_reward` and `verifier_reward` clearly. |
| `backend/app/routes/eval/routes.py` | Expose new canonical names in responses; keep legacy aliases if needed (decision below). |
| `backend/app/routes/prompt_learning/algorithm/mipro/optimizer/optimizer.py` | Keep reading task app reward (rename references internally); keep legacy fallback behavior as needed for older task apps. |
| `backend/app/routes/prompt_learning/algorithm/gepa/optimizer.py` | Align field naming for rewards (`verifier_reward`, `local_api_reward` vs `outcome_reward`). |
| `backend/app/routes/clustered_training/core/algorithms/gspo/pipeline_rl/task_info.py` | Remove TaskInfo rubric payload builders (now `/info` only). |
| Backend unit/integration tests | Rewrite tests that assume `TaskInfo.rubric` exists; update fixtures/mocks to serve `/info.rubrics`. |

**Out of scope (but audited):** GraphGen rubric fields in `backend/app/routes/graphgen/*` and `backend/graphs/gepa_integration/*`
are not LocalAPI rubrics and are handled separately.

#### Integration tests / demo task apps

| Location | Change |
|----------|--------|
| `tests/integration/pipeline_rl/` | Ensure mock task apps serve rubrics from `/info`, not `TaskInfo.rubric`. |
| `tests/backend/integration/workflows/rl/math/rl/hendrycks_math_task_app.py` | Remove `rubric=base.rubric` from `TaskInfo` construction (line 391). |
| `agora_single_file.py` | Update `_blend_rubrics(base_info.rubric, ...)` logic (lines 1595, 1679-1681) to fetch from `/info`. |
| `agora_ex/task_app.py` | Remove `rubric=base.rubric` (line 211). |

---

### New backend rubric fetching logic

Replace all `task_info["rubric"]` reads with a single helper:

```python
# backend/app/routes/prompt_learning/core/rubric_fetcher.py (new file)

from typing import Optional, Dict, Any
import httpx

async def fetch_rubric_bundle(
    task_app_url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Fetch rubric bundle from task app's /info endpoint.
    
    Returns:
        {"outcome": {...}, "events": {...}} or None if not available.
    """
    url = f"{task_app_url.rstrip('/')}/info"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers or {})
        if response.status_code != 200:
            return None
        data = response.json()
        rubrics = data.get("rubrics")
        if not isinstance(rubrics, dict):
            return None
        return {
            "outcome": rubrics.get("outcome"),
            "events": rubrics.get("events"),
        }
    except Exception:
        return None
```

All backend code that currently reads `task_info["rubric"]` should call `fetch_rubric_bundle()` instead.

---

### Migration path (for external task apps)

Since this is a **breaking change**, external task apps that still use `TaskInfo.rubric` will break.

**No grace period. No fallback.** The deprecated field has been marked for long enough — time to rip off the bandaid.

**Migration guide** (to be published with release):

1. Move rubric definitions from `TaskInfo(rubric=...)` to `LocalAPIConfig(rubrics=RubricBundle(...))`.
2. Delete `rubric=` from all `TaskInfo` construction.
3. Verify `GET /info` returns `rubrics.outcome` and/or `rubrics.events`.

**SDK version gate**: Bump SDK major version (e.g. `synth-ai>=2.0.0` requires `/info` rubrics).

---

### Implementation order (single PR/branch)

All changes will be done in **one coordinated PR** across both repos:

**synth-ai SDK + demos** (breaking changes):
- Delete `RubricInfo` and delete `TaskInfo.rubric` from `synth_ai/sdk/task/contracts.py`.
- Remove exports in `synth_ai/sdk/task/__init__.py` (`RubricInfo`, `RubricCriterion`, `RubricSection`).
- Update all affected demos (`demos/gepa_ptcg`, `demos/gepa_crafter_vlm`, `demos/web-design`) to advertise rubrics via
  `LocalAPIConfig(rubrics=RubricBundle(...))` only.
- Update any SDK templates/docs that mention `TaskInfo.rubric`.
- Bump SDK major version.

**monorepo backend** (breaking changes):
- Add `rubric_fetcher.py` helper.
- Replace all `task_info["rubric"]` reads with `fetch_rubric_bundle()` (and remove any TaskInfo-rubric fallback).
- Update eval verifier evaluation to **not** pass task reward into verifier evaluation:
  - In `backend/app/routes/eval/scoring.py` line 50: Change `env_reward=env_rewards.get(seed)` to `env_reward=None` (prevents double counting).
  - **Note**: The parameter name may still be `env_reward` in the verifier API signature; the key is passing `None` for eval jobs.
- **Naming clarification**: The plan uses `local_api_reward` / `verifier_reward` as canonical names, but existing code uses `env_reward` / `outcome_reward` / `verifier_reward`. We can:
  - Keep existing field names in persisted data (`outcome_reward`, `verifier_reward`) for backward compatibility.
  - Optionally add API aliases (`local_api_reward` → `outcome_reward`, `verifier_reward` → `verifier_reward`) if desired.
  - Rename internal variables for clarity where it doesn't break compatibility.
- Update tests and fixtures to use `/info.rubrics`.

**Integration test pass** (same PR):
- Run full pipeline RL / GEPA / eval test suite.
- Fix any remaining `TaskInfo.rubric` assumptions.
- Validate that `local_api_reward` and `verifier_reward` are both present in outputs and fusion is correct.

---

### Validation / acceptance criteria (Part 2)

- [ ] `RubricInfo` does not exist in `synth-ai` SDK; `TaskInfo` has no `rubric` field.
- [ ] No `TaskInfo.rubric` usage exists anywhere in `synth-ai` demos (verified: `gepa_ptcg`, `gepa_crafter_vlm`, `web-design` all migrated).
- [ ] Backend does not read `task_info["rubric"]` anywhere.
- [ ] All rubric-enabled task apps expose rubrics via `GET /info` as `rubrics.outcome` and/or `rubrics.events`.
- [ ] `RubricBundle` is a Pydantic `BaseModel` (not `@dataclass`) for OpenAPI compatibility.
- [ ] `/info` endpoint has typed response model (`InfoResponse`) for OpenAPI schema generation.
- [ ] FastAPI generates complete OpenAPI schema for `/info` endpoint (verifiable via `/docs` or `/openapi.json`).
- [ ] Eval results contain **two distinct values**:
  - `outcome_reward` (task app reward, also known as `local_api_reward` in plan terminology)
  - `verifier_reward` (verifier reward, canonical naming)
- [ ] Fusion uses them exactly once (no double counting): verifier evaluation receives `env_reward=None` for eval jobs.
- [ ] All existing tests pass (with updates).

---

### Risk assessment

| Risk | Mitigation |
|------|------------|
| External task apps break | Clear migration guide; announce in release notes. **No fallback period.** |
| Backend regression | Comprehensive test coverage for rubric fetching. |
| Performance (extra /info call) | Cache `/info` response per task app per job run. |

---

### Timeline estimate

| Step | Effort |
|------|--------|
| Single PR (SDK + backend + tests) | 3-5 days |
| **Total** | ~1 week |

---

## Summary

This plan covers:

1. **Verifier fusion** for PTCG evals — task app computes win/loss, verifier evaluates gameplay quality, backend fuses.
2. **Reward semantics fix** — pass `local_api_reward=None` to verifier evaluation for eval jobs to prevent double-counting.
3. **Rubrics consolidation** — delete `TaskInfo.rubric`, single source via `GET /info` rubrics.

All three changes are related and should be done together as a coordinated SDK + monorepo refactor.

---

## Finalization notes (what is “done” when this plan is approved)

This plan is considered finalized when:
- The above inventory items are either updated or explicitly marked out-of-scope in code review.
- Single PR lands with:
  - SDK: `TaskInfo.rubric` fully removed and demos migrated
  - Backend: verifier evaluation independent of task reward in eval jobs
  - Backend: explicit result separation (`local_api_reward`, `verifier_reward`)
  - Backend: rubrics sourced from `/info` only
  - Tests: end-to-end validation passes

---

## Verification checklist (comprehensive sweep)

**✅ synth-ai SDK changes:**
- [x] `RubricInfo` deletion scoped (`contracts.py`)
- [x] `TaskInfo.rubric` deletion scoped (`contracts.py`)
- [x] Legacy exports removal scoped (`__init__.py`: `RubricInfo`, `RubricCriterion`, `RubricSection`)
- [x] All demos identified: `gepa_ptcg`, `gepa_crafter_vlm`, `web-design`

**✅ monorepo backend changes:**
- [x] Double-counting fix scoped (`eval/scoring.py` line 50: `env_reward=None`)
- [x] Rubric fetching refactor scoped (`rubric_pipeline.py`, `routes_online.py`, `task_info.py`)
- [x] All `task_info["rubric"]` reads identified and scoped for removal
- [x] GraphGen rubrics explicitly marked out-of-scope (with TODOs added)

**✅ Naming clarity:**
- [x] Plan terminology (`local_api_reward`, `verifier_reward`) documented
- [x] Existing field names (`outcome_reward`, `verifier_reward`) acknowledged
- [x] Migration path clarified (keep existing names for compatibility, add aliases if desired)

**✅ 0→1 approach confirmed:**
- [x] No partial deprecation — `TaskInfo.rubric` fully removed
- [x] No grace period — breaking change with migration guide
- [x] Single source of truth — `/info` rubrics only

**✅ Implementation order:**
- [x] Single PR/branch: All changes coordinated together (SDK + backend + tests)

**Plan status: ✅ COMPREHENSIVE AND FINAL**
