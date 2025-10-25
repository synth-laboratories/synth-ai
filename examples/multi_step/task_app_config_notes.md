# Task App Config for Crafter RL: Dense Stepwise Rewards

Goal: Allow configuring the Crafter task app to enable/disable dense (decision-stepwise) event rewards and pass that choice from the RL config, through the backend, into the task app’s /rollout execution. This should be broader than just policy config – a `task_app_config` concept – but we can implement with the existing `env.config` today and optionally add a top-level alias later.

## Findings (current behaviour)

- Rollout request contract already supports two config payloads:
  - `env.config: dict` and `policy.config: dict`
- The hosted Crafter rollout implementation already supports decision-stepwise rewards, controlled via a `step_rewards` block in either `policy.config` or `env.config`.
- When active, it computes per-decision “unique achievement” deltas and attaches per-turn metadata; it also returns `decision_samples` when enabled.

Key locations and behaviour:

- Rollout schema (env/policy config):
```51:87:synth-ai/synth_ai/task/contracts.py
class RolloutEnvSpec(BaseModel):
    env_id: str | None = None
    env_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

class RolloutPolicySpec(BaseModel):
    policy_id: str | None = None
    policy_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
```

- Crafter hosted rollout reads step-reward config from policy, then env; gates on `enabled` and `mode == "decision_stepwise"`:
```1041:1067:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
# Stepwise reward configuration (Crafter shaping; gate on explicit enable)
step_rewards_cfg_raw: dict[str, Any] = {}
...
if not step_rewards_cfg_raw:
    if isinstance(request.env.config, dict):
        step_rewards_cfg_raw = dict(request.env.config.get("step_rewards") or {})

step_rewards_enabled = bool(step_rewards_cfg_raw.get("enabled", False))
step_rewards_mode = str(step_rewards_cfg_raw.get("mode") or "off").lower()
...
step_rewards_active = step_rewards_enabled and step_rewards_mode == "decision_stepwise"
```

- When active, it computes decision-level indicators and metadata, and adds to each step’s `info.meta.decision_rewards`; also accumulates `decision_samples`:
```1554:1596:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
if step_rewards_active:
    decision_actions = _summarize_tool_calls(pending_tool_calls)
    stepwise_info, decision_record, stats = compute_stepwise_reward(
        prev_achievements or {},
        new_achievement_state,
        decision_index,
        decision_actions,
        step_rewards_indicator_lambda,
    )
    ...
    # Compute decision-level rewards (absolute vs unique) and attach to metadata
    turned_true = set(stepwise_info.get("new_achievements") or [])
    seen_before = set(episode_seen_achievements)
    new_unique = sorted(turned_true - seen_before)
    ach_delta = int(len(turned_true))
    unique_delta = int(len(new_unique))
    meta_block = (_info.get("meta") if isinstance(_info.get("meta"), dict) else {})
    decision_rewards = {"turn": int(decision_index), "ach_delta": ach_delta, "unique_delta": unique_delta, "all": all_list, "unique": new_unique}
    decision_rewards_meta = decision_rewards
    meta_block["decision_rewards"] = decision_rewards
    _info["meta"] = meta_block
    episode_seen_achievements.update(turned_true)
    decision_samples.append(decision_record)
```

- The simpler published Crafter app (`examples/warming_up_to_rl/task_app/grpo_crafter.py`) sets sane defaults for `step_rewards` in both env and policy when it aliases math → crafter, but the hosted rollout above is the one actually used in production paths.
```479:490:synth-ai/examples/warming_up_to_rl/task_app/grpo_crafter.py
env_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
...
policy_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
```

- Backend RPC: The backend constructs the rollout HTTP payload with both env_config and policy_config; these are forwarded to the task app `/rollout`:
```456:470:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py
payload = {
    "run_id": run_id,
    "env": {"env_name": env_name, "config": env_config, "seed": seed},
    "policy": {"policy_name": policy_name, "config": policy_config},
    "ops": ops,
    "record": {"trajectories": True, "logprobs": False, "value": False},
    "on_done": on_done,
}
```

- RL config ingestion: The CLI forwards the full TOML in the job payload. The backend trainer flattens some rollout options and (optionally) picks up `rollout.env_config`:
```364:393:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py
# Rollout config
if "rollout" in config_dict:
    flat_config["env_name"] = config_dict["rollout"].get("env_name", "crafter")
    ...
    if "env_config" in config_dict["rollout"]:
        flat_config["env_config"] = config_dict["rollout"]["env_config"]
```

Implication: We can carry a broader "task app config" today via `rollout.env_config` without changing wire contracts. The task app already reads `env.config.step_rewards`.

## Proposed configuration shape (TOML)

Recommended to use `rollout.env_config.step_rewards` so the backend passes it through to the task app:

```toml
[rollout]
env_name = "crafter"
policy_name = "crafter-react"
max_turns = 10
ops = ["agent", "env"]

[rollout.env_config.step_rewards]
# Toggle dense per-decision rewards
enabled = true
# Supported: "off" | "decision_stepwise" | (future) "env_sparse"
mode = "decision_stepwise"
# Reward = indicator_lambda * I(unique_achievements_delta > 0)
indicator_lambda = 1.0
# Reserved for shaped/intermediate signals (currently unused)
step_beta = 0.0
```

Optional (policy sampling, still supported via `policy.config` in the task app runner):

```toml
[rollout.policy_config]
temperature = 0.2
top_p = 0.95
max_tokens = 512
```

Notes:
- The hosted Crafter rollout checks `policy.config.step_rewards` first, then falls back to `env.config.step_rewards`. Prefer `env_config` as the canonical place for app-level settings.
- If you want the app to disable stepwise rewards entirely, set `enabled=false` or `mode="off"`.

## Wire and data flow

1) CLI → Backend: CLI includes the entire TOML in the job payload (`build_rl_payload`).
2) Backend → Trainer: Trainer flattens rollout properties and can include `env_config`.
3) Trainer → Task App: Rollout HTTP payload embeds `env.config` and `policy.config`.
4) Task App: Hosted rollout computes decision-level metadata and returns:
   - `RolloutStep.info.meta.decision_rewards` with `{turn, ach_delta, unique_delta, ...}`
   - `trajectory.decision_samples` summarizing per-turn reward inputs

## Minimal code changes to adopt

- synth-ai (optional):
  - Add example configs under `examples/warming_up_to_rl/configs/*.toml` using `[rollout.env_config.step_rewards]`.
  - Document this block in docs and the multi_step walkthrough.

- monorepo backend:
  - Verify trainer always passes `rollout.env_config` (ClusteredTrainerConfig appears to support it; ensure it flows into the runtime’s rollout request builder in the trainer where the payload is assembled).
  - No contract changes needed: task app already reads from `env.config`.

- Task App:
  - Already supports the block; no changes needed for the hosted Crafter rollout.
  - If you want a first-class `task_app_config` top-level, we can add an alias resolver that copies `config["task_app_config"]` → `env.config` inside the rollout executor.

## Open questions / follow-ups

- Does the current trainer consume `decision_samples` or `step.info.meta.decision_rewards` for credit assignment? If not, wire this into the per-step reward/advantage pipeline.
- Decide whether to disable the default enabling of stepwise rewards in `grpo_crafter.py` aliases (`DEFAULT_ALIAS_STEP_REWARDS`) so the TOML fully drives behaviour.
- Standardize on `env_config.step_rewards` for app-level settings across environments.

## Reference: CRAfter RL LoRA example (expected first 10 rewards)
These are the first ten batch rewards printed at RL start:
```
- INFO - All batch rewards: [0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.3125, 0.375, 0.4375, 0.5, 0.9375]
```

---

## Enable stepwise during EVALS and compare vs final

We can enable stepwise shaping for evaluation-only runs and compare “stepwise” vs “final (outcome)” returns.

Two evaluation paths exist today:

- Backend evaluator endpoint (preferred for hosted):
```1114:1136:monorepo/backend/app/routes/clustered_training/core/routes.py
class RlEvaluateRequest(BaseModel):
    model: str
    seeds: list[int]
    rollouts_per_seed: int = 1
    env_name: str
    env_config: Dict[str, Any] = Field(default_factory=dict)
    policy_name: str
    thinking_mode: str
    thinking_budget: int | None = None
    max_steps_per_episode: int = 100
    max_concurrent_rollouts: int = 8
    on_done: str = "terminate"
    task_service_url: str | None = None
    vllm_url: str | None = None
    vllm_public_url: str | None = None
```

Pass `env_config.step_rewards` here to turn on stepwise shaping during evals (no trainer changes needed). The evaluator will forward `env_config` into each rollout:
```383:396:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py
payload = {
  "env": {"env_name": env_name, "config": env_config, ...},
  "policy": {"policy_name": policy_name, "config": policy_config},
  ...
}
```

Task app already computes and attaches:
- Per-decision metadata at `step.info.meta.decision_rewards`
- Aggregates we can expose (see below) for stepwise vs final

Recommended enhancement (small change in task app): include a summary under `response.metrics.details.stepwise` so eval clients don’t need to parse per-step:
```python
metrics.details["stepwise"] = {
  "indicator_sum": stepwise_indicator_sum,
  "reward_sum": stepwise_reward_sum,
  "new_achievements_total": stepwise_new_achievements_total,
}
```

For local SDK evals (without backend), call the `/rollout` endpoint directly with the same `env.config.step_rewards` block.

Example payload fragment:
```json
{
  "env": {
    "env_name": "crafter",
    "config": {
      "step_rewards": { "enabled": true, "mode": "decision_stepwise", "indicator_lambda": 1.0 }
    },
    "seed": 0
  },
  "policy": { "policy_name": "crafter-react", "config": {"temperature": 0.2} },
  "ops": ["agent", "env"]
}
```

---

## Simple vs Complex stepwise modes (proposal)

Add a `strategy` under the existing `step_rewards` block:

```toml
[rollout.env_config.step_rewards]
enabled = true
mode = "decision_stepwise"        # gate remains the same
strategy = "simple"                # "simple" | "complex"
indicator_lambda = 1.0

# Complex-only (optional)
weights = { collect_sapling = 0.1, craft_wood_pickaxe = 0.3, collect_diamond = 1.0 }
k_limits = { collect_sapling = 1, craft_wood_pickaxe = 2, collect_diamond = 3 }
```

Behaviour:
- strategy="simple": reward 1.0×indicator_lambda if any new achievement unlocked at that decision, else 0. (Current logic already does this; just make it explicit.)
- strategy="complex":
  - Maintain per-episode `achieve_count[name]`.
  - For each achievement newly unlocked at the decision, if `achieve_count[name] < k_limits.get(name, 1)`, add `weights.get(name, 1.0)` to the stepwise reward and increment the count.
  - The uniqueness baseline should be the “turned true this decision” set; combining with episode-level uniqueness is optional if we intend multiple rewards up to K.

Minimal code touch points:
- synth-ai task app (hosted Crafter rollout):
  - Extend `compute_stepwise_reward(prev_achievements, new_achievements, decision_index, actions_summary, indicator_lambda)` to optionally take `strategy`, `weights`, `k_limits`, and a `counts` dict.
  - Thread an `episode_ach_counts: Dict[str, int]` through the rollout loop (similar to `episode_seen_achievements`).
  - Build `reward_stepwise` as per strategy; keep existing `decision_rewards` metadata (ach/unique deltas) unchanged.
  - Add `metrics.details["stepwise"]` summary (indicator_sum, reward_sum, new_achievements_total).

- monorepo backend (evals):
  - No contract change: pass the same `env_config.step_rewards` in `RlEvaluateRequest.env_config`.
  - For convenience, surface stepwise summary in any eval aggregation/CSV if present under `metrics.details.stepwise`.

Open choice:
- Either keep `mode="decision_stepwise"` and add `strategy`, or introduce `mode` values `{ "simple_stepwise", "complex_stepwise" }`. The former is backward compatible and clearer.

Testing plan:
- Unit-test `compute_stepwise_reward` for both strategies with synthetic prev/new achievement maps.
- Smoke eval over a few seeds with `strategy=simple` and `strategy=complex` to compare `metrics.details.stepwise.reward_sum` vs `metrics.mean_return`.

---

## Eval script scope: Groq Qwen/Qwen3-32B stepwise vs outcome

Objective: run many Crafter rollouts against Groq `Qwen/Qwen3-32B` and compare distributions and correlations between stepwise rewards and final (outcome) rewards, for both simple and complex stepwise strategies.

Inputs/flags:
- `--task-url` Task app base URL (Modal deployment)
- `--env-key` ENVIRONMENT_API_KEY (or from `.env`)
- `--model` default `Qwen/Qwen3-32B`
- `--seeds` list or `--num-seeds` N (use 0..N-1)
- `--rollouts-per-seed` default 3
- `--max-turns` default 10
- `--strategy` `simple|complex|both` (default both)
- `--weights-json` optional JSON path for complex weighting
- `--k-limits-json` optional JSON path for complex K-limits
- `--out` output directory for CSV/plots

What it does:
1) Builds rollout payloads for each seed and strategy variant.
2) For each rollout, passes `env.config.step_rewards` with:
   - common: `{ enabled: true, mode: "decision_stepwise" }`
   - simple: `strategy: "simple", indicator_lambda: 1.0`
   - complex: `strategy: "complex", weights, k_limits`
3) Uses policy config to route inference to Groq with the requested model.
4) Collects per-rollout summary:
   - `final_return = response.metrics.mean_return`
   - `step_indicator_sum`, `step_reward_sum`, `new_achievements_total` from `metrics.details.stepwise` (or compute from steps if absent)
   - counts like `num_steps`, `tool_calls_total`
5) Writes a wide CSV with one row per rollout, including seed, strategy, and the above fields.
6) Visualizes:
   - Histogram of `step_reward_sum` by strategy
   - Scatter: `step_reward_sum` vs `final_return`, per strategy (with Pearson/Spearman r)
   - Optional ECDFs for indicator_sum

Data schema (CSV):
```
seed,int | rollout_idx,int | strategy,str | final_return,float | step_reward_sum,float |
step_indicator_sum,float | new_achievements_total,int | num_steps,int | tool_calls_total,int |
model,str | max_turns,int | timestamp,iso
```

Pseudocode (Python):
```python
import os, json, csv, time, math, statistics
import httpx

TASK_URL = os.environ.get("TASK_APP_URL")
ENV_KEY = os.environ.get("ENVIRONMENT_API_KEY")

def build_step_cfg(strategy, weights=None, k_limits=None):
    cfg = {"enabled": True, "mode": "decision_stepwise", "strategy": strategy, "indicator_lambda": 1.0}
    if strategy == "complex":
        if weights: cfg["weights"] = weights
        if k_limits: cfg["k_limits"] = k_limits
    return cfg

async def run_rollout(seed, strategy, model, max_turns, weights, k_limits):
    step_cfg = build_step_cfg(strategy, weights, k_limits)
    payload = {
        "run_id": f"eval-{seed}-{strategy}-{int(time.time())}",
        "env": {"env_name": "crafter", "seed": seed, "config": {"step_rewards": step_cfg, "env_params": {"max_steps_per_episode": max_turns}}},
        "policy": {"policy_name": "crafter-react", "config": {"inference_url": "https://groq.synth-ai.internal/proxy", "model": model, "temperature": 0.2, "top_p": 0.95, "max_tokens": 512}},
        "ops": ["agent", "env"] * max_turns,
        "record": {"trajectories": True},
        "on_done": "terminate",
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{TASK_URL}/rollout", headers={"X-API-Key": ENV_KEY}, json=payload)
        r.raise_for_status()
        resp = r.json()
    met = resp.get("metrics", {})
    details = met.get("details", {})
    step = details.get("stepwise", {})
    final_return = float(met.get("mean_return") or 0.0)
    step_reward_sum = float(step.get("reward_sum") or 0.0)
    step_indicator_sum = float(step.get("indicator_sum") or 0.0)
    new_ach_total = int(step.get("new_achievements_total") or 0)
    num_steps = int(met.get("num_steps") or 0)
    tool_calls_total = sum(len(s.get("tool_calls", [])) for s in (resp.get("trajectories", [{}])[0].get("steps", []))) if resp.get("trajectories") else 0
    return {
        "seed": seed, "strategy": strategy, "final_return": final_return,
        "step_reward_sum": step_reward_sum, "step_indicator_sum": step_indicator_sum,
        "new_achievements_total": new_ach_total, "num_steps": num_steps,
        "tool_calls_total": tool_calls_total,
    }
```

CLI example:
```bash
uv run python tools/eval_stepwise_vs_final.py \
  --task-url $TASK_APP_URL \
  --env-key $ENVIRONMENT_API_KEY \
  --model "Qwen/Qwen3-32B" \
  --num-seeds 100 --rollouts-per-seed 3 --max-turns 10 \
  --strategy both --out results/qwen32b
```

Notes:
- The correlation/plots can be produced with `matplotlib` or `plotly`; write PNG + HTML.
- If `metrics.details.stepwise` is not yet populated by the task app, compute `indicator_sum` and `reward_sum` on the client by scanning `steps[].info.meta.decision_rewards`.

### Output artifacts (JSON + Markdown)

Directory layout under `--out` (example: `results/qwen32b`):

- `runs/` — one JSONL file per strategy with one record per rollout
  - `runs/simple.jsonl`
  - `runs/complex.jsonl`
- `summary/`
  - `summary.json` — aggregates per strategy (mean/median/std, correlations, counts)
  - `stats_by_seed.json` — per-seed aggregates
  - `config_snapshot.json` — CLI args, weights, k-limits, timestamp, git SHA
- `plots/`
  - `hist_step_reward_simple.png`, `hist_step_reward_complex.png`
  - `scatter_step_reward_vs_final_simple.png`, `scatter_step_reward_vs_final_complex.png`
  - `ecdf_indicator_simple.png`, `ecdf_indicator_complex.png`
- `report.md` — human-friendly Markdown summary linking to plots

Record schema (per line in `runs/*.jsonl`):

```json
{
  "seed": 0,
  "rollout_idx": 1,
  "strategy": "simple",                  // or "complex"
  "final_return": 0.9375,
  "step_reward_sum": 1.2,
  "step_indicator_sum": 3.0,
  "new_achievements_total": 3,
  "num_steps": 10,
  "tool_calls_total": 12,
  "model": "Qwen/Qwen3-32B",
  "max_turns": 10,
  "timestamp": "2025-10-17T22:14:05Z",
  "meta": {
    "task_url": "...",
    "weights": {"collect_diamond": 1.0},        // only for complex
    "k_limits": {"collect_diamond": 3}          // only for complex
  }
}
```

Markdown report (`report.md`) outline:

- Title, timestamp, model, config summary
- Table of global aggregates per strategy (mean, median, std of `final_return`, `step_reward_sum`, correlations r/Pearson, ρ/Spearman)
- Small tables per-seed (top-N seeds by return)
- Inline images linking to `plots/*.png`
- Pointers to raw data files and `config_snapshot.json`


I'll scan both repos for Crafter RL, policy/task app config, rollout calls, and backend RL endpoints, then draft notes under `examples/multi_step/`.

[5 tools called]

I'll open the Crafter task app implementation to inspect rubrics and where stepwise reward configuration is applied, then check the RL evaluator to see how env/policy config is passed to the task app.

[11 tools called]

I scanned both repos to map how stepwise rewards already work in the Crafter task app and how rollout configs flow from the RL TOML into the task app. I’ll add a detailed notes document under `examples/multi_step/` with findings, code references, and a concrete config proposal.

[6 tools called]

- I researched Crafter RL and documented how to toggle dense stepwise rewards and flow the config end-to-end.
- Notes added at `synth-ai/examples/multi_step/task_app_config_notes.md` with exact code references and a TOML example.

Key points:
- Use `rollout.env_config.step_rewards` in the RL TOML; task app already reads `policy.config.step_rewards` first, then `env.config.step_rewards`.
- Enable with:
```toml
[rollout.env_config.step_rewards]
enabled = true
mode = "decision_stepwise"
indicator_lambda = 1.0
step_beta = 0.0
```
- Backend already forwards `env_config` and `policy_config` in rollout requests; no wire change needed. Ensure trainer populates `env_config` when present.
- Task app attaches per-turn `decision_rewards` and returns `decision_samples`; consider consuming these in credit assignment.

References:
- Task contracts and rollout schema:
```51:87:synth-ai/synth_ai/task/contracts.py
class RolloutEnvSpec(BaseModel):
    ...
    config: dict[str, Any] = Field(default_factory=dict)
...
class RolloutPolicySpec(BaseModel):
    ...
    config: dict[str, Any] = Field(default_factory=dict)
```
- Crafter stepwise config reading and reward metadata:
```1041:1067:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
# Stepwise reward configuration ...
```
```1554:1596:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
# Compute decision-level rewards ...
```
- Backend rollout payload shape:
```456:470:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py
payload = {
  "run_id": run_id,
  "env": {"env_name": env_name, "config": env_config, "seed": seed},
  "policy": {"policy_name": policy_name, "config": policy_config},
  ...
}
```

Status: Notes committed on branch `friday-cleanup` and pushed.

### Operational guardrails

- Treat avg_turns == 0 (or a high fraction of episodes with turns == 0) as a failure condition; exit non‑zero.
- Fail fast when the first policy step returns a 4xx/5xx from the inference target; include the HTTP status and URL in the error message.
- CI hint: a tiny smoke run (2 seeds × 1 rollout) should see turns > 0 in healthy setups.