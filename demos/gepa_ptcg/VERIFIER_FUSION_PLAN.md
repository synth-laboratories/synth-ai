# Verifier Fusion Plan: PTCG Gameplay Eval + Rubrics Consolidation Refactor

This document covers **two related refactors**:

1. **Part 1**: Verifier fusion for PTCG gameplay evals (task reward + zero-shot verifier + fused score)
2. **Part 2**: Rubrics consolidation (delete `TaskInfo.rubric`, single source via `/info`)

Both changes tie together because they clean up how rubrics flow from task apps to the backend verifier.

---

## Part 1: Verifier Fusion for PTCG Gameplay Eval

### Goal

Run **headless gameplay evals** for `demos/gepa_ptcg` where:

- The **task app** computes the canonical task reward (win/loss or outcome reward).
- The Synth backend runs a **zero-shot verifier** that scores **gameplay quality** from the hydrated trace against a rubric.
- The backend **fuses** task reward + verifier score into a single per-seed `score`.

This is intended to work with the existing **eval job** flow (not prompt-learning/GEPA optimization yet).

---

### Goal / Non-goals

- **Goal**: In eval jobs, record *both*:
  - `outcome_reward` (task app, who won)
  - `verifier_score` (zero-shot rubric, gameplay quality)
  - `score` (fused)
- **Goal**: Reuse **interceptor hydration** for traces (avoid building v3 traces in task app).
- **Non-goal**: Modify `monorepo/specs` (explicitly forbidden).
- **Non-goal**: Embed LLM calls into Rust servers / UI harness. This is headless LocalAPI.

---

### Current state (what already exists)

#### 1) Eval job pipeline supports verifier scoring

The backend eval job service supports an optional `verifier_config` in the eval request.
If present and enabled, it computes a per-seed `verifier_score` and then fuses it into the final `score`.

#### 2) Verifier endpoint exists and is zero-shot capable

Backend route: `POST /api/graphs/verifiers/completions`

It supports built-in zero-shot verifier graph IDs:
- `zero_shot_verifier` (auto routing)
- `zero_shot_verifier_rubric_single`, `..._mapreduce`, `..._rlm`, etc.

#### 3) Trace hydration exists for eval jobs

Eval jobs already hydrate v3 traces from the interceptor store and normalize rollouts to v3 traces for scoring.
This means task apps do **not** need to build tracing-v3 `event_history` manually, as long as:
- LLM calls go through the interceptor (which we already do in `gepa_ptcg`).

---

### Proposed scoring semantics

We want:

- **Task reward** (from task app): \( r_{env} \in [0, 1] \)
  - Example: win=1, loss=0, draw=0.5, etc.
- **Verifier reward** (from verifier): \( r_{verifier} \in [0, 1] \)
  - A rubric-based "gameplay quality" score independent of win/loss
- **Fused score**:

\[
r_{final} = w_{env}\cdot r_{env} + w_{verifier}\cdot r_{verifier}
\]

Example weights:
- `w_env=0.5`, `w_verifier=0.5` (simple balanced)

---

### Scoring semantics: double-counting risk (CRITICAL)

#### The problem

The verifier pipeline can optionally accept `env_reward` as input:

```python
# backend/app/routes/eval/scoring.py (current)
verifier_result = await verifier.score_trajectory(
    ...,
    env_reward=env_rewards.get(seed),  # task app reward passed in
)
```

Depending on the verifier graph implementation, the verifier may:
- **Ignore** `env_reward` and score purely on trace + rubric (desired for independent fusion)
- **Incorporate** `env_reward` into its output (causes double-counting when fused again)

If the verifier incorporates `env_reward`, and then the eval job service fuses:

```
score = w_env * env_reward + w_verifier * verifier_score
```

…the task reward gets counted **twice**: once inside `verifier_score`, once in the outer fusion.

#### The fix (required monorepo change)

To guarantee no double-counting, we change the eval verifier scoring call to:

```python
# backend/app/routes/eval/scoring.py (proposed)
verifier_result = await verifier.score_trajectory(
    ...,
    env_reward=None,  # <-- do NOT pass task reward into verifier
)
```

This ensures the verifier scores only the trace + rubric, and fusion happens exactly once in the eval job service.

#### Code change

Single line change in `backend/app/routes/eval/scoring.py`:

```diff
- env_reward=env_rewards.get(seed),
+ env_reward=None,
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
- We fuse only outcome-level verifier score into final reward initially (keep it simple).
- If event-level scoring returns per-event totals, we can later add `weight_event > 0`.

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
  - `outcome_reward` (task app)
  - `verifier_score` (non-null for most seeds)
  - `score` differs from both and matches the configured fusion weights
- Backend logs show:
  - trace hydration succeeded
  - verifier endpoint calls succeeded (200)

Failure modes and what they mean:
- **verifier_score is null**: rubric missing/unparseable, verifier_graph_id wrong, trace hydration missing, or verifier endpoint errors.
- **verifier_score correlates too strongly with win**: likely env_reward leaked into verifier scoring or rubric is too outcome-focused.

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

@dataclass
class Criterion:
    id: str                    # unique criterion identifier
    description: str           # what this criterion evaluates
    weight: float = 1.0        # relative importance
    required: bool = False     # if True, failing this criterion fails the rubric

class Rubric(BaseModel):
    version: str = "1.0"
    goal_text: str             # high-level goal description
    criteria: list[Criterion]  # list of evaluation criteria
    aggregation: str = "weighted_mean"  # how to combine criterion scores

# synth_ai/sdk/task/server.py

@dataclass
class RubricBundle:
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

---

### Scope of changes

#### synth-ai SDK (breaking)

| File | Change |
|------|--------|
| `synth_ai/sdk/task/contracts.py` | **Delete** `TaskInfo.rubric` field (lines 287-290) |
| `synth_ai/sdk/task/server.py` | Already correct — `LocalAPIConfig.rubrics` served from `/info` |
| All demo task apps | Audit and remove any `rubric=...` in `TaskInfo` returns; ensure rubrics are in `LocalAPIConfig.rubrics` |

#### monorepo backend (breaking)

| File | Change |
|------|--------|
| `backend/app/routes/prompt_learning/core/rubric_pipeline.py` | **Delete** `_build_rubric_payload()` that reads `task_info["rubric"]`. Replace with direct fetch from `/info` rubrics. Remove fallback merge logic in `_TaskInfoFetcher.fetch()` (lines 131-169). |
| `backend/app/routes/clustered_training/core/algorithms/gspo/pipeline_rl/task_info.py` | **Delete** `build_rubric_payload()` (entire function, lines 142-167). |
| `backend/app/routes/prompt_learning/routes_online.py` | Update validation logic (line 623) to check `/info` rubrics, not `TaskInfo.rubric`. |
| `backend/graphs/gepa_integration/graph_evolve_job.py` | Update `task_rubric` extraction to use `/info` rubrics (lines 4628-4653). |
| `backend/app/routes/graphgen/dataset.py` | Update `task_rubric = task.rubric` (line 476) to source from `/info`. |
| `backend/app/routes/graphgen/routes.py` | Update `task.rubric.model_dump()` (line 3325) to source from `/info`. |
| `backend/app/routes/blog/demos/synth_ai_environments/examples/*.py` | Update all `"rubric": self.intent.rubric` usages to use `/info` pattern. |
| `backend/app/routes/eval/scoring.py` | Change `env_reward=env_rewards.get(seed)` to `env_reward=None` (Part 1 fix). |
| `backend/tests/unit/prompt_learning/test_rubric_pipeline_task_info.py` | **Delete or rewrite** tests that assume `TaskInfo.rubric` exists. |
| `backend/tests/unit/test_rubric_pipeline.py` | Update tests to use `/info` rubrics. |
| `backend/tests/unit/test_clustered_trainer_rubric.py` | Update tests. |
| `merge_review/stash_files/backend/routes_online.py` | Delete or update stash file (or ignore if not used). |

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

### Implementation order

1. **synth-ai SDK** (PR 1):
   - Delete `TaskInfo.rubric` field from `contracts.py`.
   - Audit all demo task apps; move rubrics to `LocalAPIConfig.rubrics`.
   - Bump SDK version to indicate breaking change.

2. **monorepo backend** (PR 2, depends on PR 1 merged):
   - Add `rubric_fetcher.py` helper.
   - Replace all `task_info["rubric"]` reads with `fetch_rubric_bundle()`.
   - Delete dead code: `_build_rubric_payload()`, `build_rubric_payload()`, fallback merge logic.
   - Change `env_reward=None` in eval scoring (Part 1 fix).
   - Update tests.

3. **Integration test pass** (PR 3):
   - Run full pipeline RL / GEPA / eval test suite.
   - Fix any remaining `TaskInfo.rubric` assumptions.

---

### Validation / acceptance criteria (Part 2)

- [ ] `TaskInfo` no longer has a `rubric` field in SDK.
- [ ] Backend does not read `task_info["rubric"]` anywhere.
- [ ] All demo task apps serve rubrics from `/info`.
- [ ] `GET /info` returns `{"rubrics": {"outcome": {...}, "events": {...}}}` for rubric-enabled task apps.
- [ ] Verifier scoring works end-to-end with rubrics sourced from `/info`.
- [ ] No double rubric fetch (backend calls `/info` once, caches result).
- [ ] All existing tests pass (with updates).
- [ ] Eval verifier scoring passes `env_reward=None` (no double-counting).

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
| SDK PR (delete field, update demos) | 1-2 days |
| Backend PR (refactor fetching, delete dead code, env_reward=None) | 2-3 days |
| Test pass / fix regressions | 1-2 days |
| **Total** | ~1 week |

---

## Summary

This plan covers:

1. **Verifier fusion** for PTCG evals — task app computes win/loss, verifier scores gameplay quality, backend fuses.
2. **Scoring semantics fix** — pass `env_reward=None` to verifier to prevent double-counting.
3. **Rubrics consolidation** — delete `TaskInfo.rubric`, single source via `GET /info` rubrics.

All three changes are related and should be done together as a coordinated SDK + monorepo refactor.
